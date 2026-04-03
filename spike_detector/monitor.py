"""
TrainingMonitor: hooks into a PyTorch training loop to track gradient norms,
detect anomalies via SPC, and save forensic snapshots on spike detection.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from .detectors import CUSUMDetector, ShewhartDetector, Alert


@dataclass
class ForensicSnapshot:
    step: int
    timestamp: float
    alert: dict
    loss: float
    gradient_norms: dict[str, float]
    parameter_stats: dict[str, dict]
    optimizer_state_summary: dict[str, dict]
    batch_sample: dict  # first few items from the batch
    learning_rate: float


class TrainingMonitor:
    """
    Attaches to a training loop to monitor gradient norms per layer,
    detect anomalies using CUSUM and Shewhart detectors, and save
    forensic snapshots when spikes are detected.

    Usage:
        monitor = TrainingMonitor(model, optimizer, log_dir="./spike_logs")

        for step, batch in enumerate(dataloader):
            loss = train_step(model, batch)
            loss.backward()

            # Call before optimizer.step() so gradients are still populated
            monitor.step(step, loss.item(), batch)

            optimizer.step()
            optimizer.zero_grad()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        log_dir: str = "./spike_logs",
        cusum_allowance: float = 0.5,
        cusum_threshold: float = 4.0,
        shewhart_sigma: float = 3.0,
        warmup_steps: int = 50,
    ):
        self.model = model
        self.optimizer = optimizer
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Collect layer names for per-layer gradient tracking (used in
        # forensic snapshots to identify *which* layers spiked, but NOT
        # for anomaly detection — per-layer norms are too noisy for SPC).
        self.layer_names: list[str] = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.layer_names.append(name)

        # Anomaly detection on aggregate signals only: total gradient norm + loss
        self.cusum_total = CUSUMDetector(
            allowance=cusum_allowance, threshold=cusum_threshold, warmup_steps=warmup_steps
        )
        self.shewhart_total = ShewhartDetector(
            n_sigma=shewhart_sigma, warmup_steps=warmup_steps
        )

        # Loss detectors
        self.cusum_loss = CUSUMDetector(
            allowance=cusum_allowance, threshold=cusum_threshold, warmup_steps=warmup_steps
        )
        self.shewhart_loss = ShewhartDetector(
            n_sigma=shewhart_sigma, warmup_steps=warmup_steps
        )

        # History for reporting
        self.history = {
            "steps": [],
            "loss": [],
            "total_grad_norm": [],
            "layer_grad_norms": {name: [] for name in self.layer_names},
            "learning_rates": [],
        }
        self.alerts: list[Alert] = []
        self.snapshots: list[ForensicSnapshot] = []

    def _compute_grad_norms(self) -> dict[str, float]:
        norms = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                norms[name] = param.grad.data.norm(2).item()
            elif param.requires_grad:
                norms[name] = 0.0
        return norms

    def _get_optimizer_state_summary(self) -> dict[str, dict]:
        summary = {}
        for i, group in enumerate(self.optimizer.param_groups):
            group_summary = {"lr": group["lr"]}
            # Sample first param's optimizer state
            if group["params"]:
                p = group["params"][0]
                state = self.optimizer.state.get(p, {})
                for key, val in state.items():
                    if isinstance(val, torch.Tensor):
                        group_summary[f"state_{key}_norm"] = val.norm().item()
                        group_summary[f"state_{key}_mean"] = val.mean().item()
                        group_summary[f"state_{key}_max"] = val.max().item()
                    elif isinstance(val, (int, float)):
                        group_summary[f"state_{key}"] = val
            summary[f"group_{i}"] = group_summary
        return summary

    def _get_param_stats(self) -> dict[str, dict]:
        stats = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                stats[name] = {
                    "mean": param.data.mean().item(),
                    "std": param.data.std().item(),
                    "max": param.data.max().item(),
                    "min": param.data.min().item(),
                    "norm": param.data.norm().item(),
                }
        return stats

    def _extract_batch_sample(self, batch) -> dict:
        """Extract a small sample from the batch for forensic purposes."""
        sample = {}
        if isinstance(batch, torch.Tensor):
            sample["input_ids"] = batch[:2, :32].tolist()  # first 2 sequences, first 32 tokens
            sample["shape"] = list(batch.shape)
        elif isinstance(batch, dict):
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    sample[key] = {
                        "shape": list(val.shape),
                        "sample": val[:2, :32].tolist() if val.dim() >= 2 else val[:2].tolist(),
                    }
        elif isinstance(batch, (list, tuple)) and len(batch) >= 1:
            for i, item in enumerate(batch[:2]):
                if isinstance(item, torch.Tensor):
                    sample[f"item_{i}"] = {
                        "shape": list(item.shape),
                        "sample": item[:2, :32].tolist() if item.dim() >= 2 else item[:2].tolist(),
                    }
        return sample

    def _save_snapshot(self, alert: Alert, loss: float, grad_norms: dict[str, float],
                       batch, lr: float):
        snapshot = ForensicSnapshot(
            step=alert.step,
            timestamp=time.time(),
            alert=asdict(alert),
            loss=loss,
            gradient_norms=grad_norms,
            parameter_stats=self._get_param_stats(),
            optimizer_state_summary=self._get_optimizer_state_summary(),
            batch_sample=self._extract_batch_sample(batch),
            learning_rate=lr,
        )
        self.snapshots.append(snapshot)

        # Save to disk
        snapshot_path = self.log_dir / f"snapshot_step_{alert.step}_{alert.detector_type}.json"
        with open(snapshot_path, "w") as f:
            json.dump(asdict(snapshot), f, indent=2, default=str)

        return snapshot

    def step(self, step: int, loss: float, batch=None) -> list[Alert]:
        """
        Call after loss.backward() but before optimizer.step().
        Returns list of alerts triggered at this step.
        """
        grad_norms = self._compute_grad_norms()
        total_norm = sum(v ** 2 for v in grad_norms.values()) ** 0.5

        lr = self.optimizer.param_groups[0]["lr"]

        # Record history
        self.history["steps"].append(step)
        self.history["loss"].append(loss)
        self.history["total_grad_norm"].append(total_norm)
        self.history["learning_rates"].append(lr)
        for name in self.layer_names:
            self.history["layer_grad_norms"][name].append(grad_norms.get(name, 0.0))

        # Run detectors
        step_alerts = []

        # Check total gradient norm (only upward spikes)
        for detector in [self.cusum_total, self.shewhart_total]:
            alert = detector.update(total_norm, step, "total_grad_norm")
            if alert and "LOW" not in alert.detector_type:
                step_alerts.append(alert)

        # Check loss (only flag upward spikes — loss naturally decreases)
        for detector in [self.cusum_loss, self.shewhart_loss]:
            alert = detector.update(loss, step, "loss")
            if alert and "LOW" not in alert.detector_type:
                step_alerts.append(alert)

        # Per-layer norms are recorded in history for forensic analysis
        # but not used for anomaly detection (too noisy for SPC)

        # Save forensic snapshots for any alerts
        if step_alerts:
            # Deduplicate: save one snapshot per step with all alerts
            primary_alert = step_alerts[0]
            primary_alert.details["all_alerts_this_step"] = [
                {"type": a.detector_type, "metric": a.metric_name} for a in step_alerts
            ]
            self._save_snapshot(primary_alert, loss, grad_norms, batch, lr)
            self.alerts.extend(step_alerts)

            print(f"\n{'='*60}")
            print(f"SPIKE DETECTED at step {step}")
            print(f"  Loss: {loss:.4f}")
            print(f"  Total grad norm: {total_norm:.4f}")
            print(f"  Alerts: {len(step_alerts)}")
            for a in step_alerts:
                print(f"    [{a.detector_type}] {a.metric_name}: {a.value:.4f}")
            print(f"  Snapshot saved to {self.log_dir}")
            print(f"{'='*60}\n")

        return step_alerts

    def get_history(self) -> dict:
        return self.history

    def get_alerts(self) -> list[Alert]:
        return self.alerts

    def get_snapshots(self) -> list[ForensicSnapshot]:
        return self.snapshots
