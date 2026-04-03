"""
Post-mortem report generator.

Produces a comprehensive visual report from TrainingMonitor data,
including control charts, per-layer gradient heatmaps, and
forensic snapshot analysis.
"""

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class PostMortemReport:
    """
    Generates a visual post-mortem report from training monitor data.

    Usage:
        report = PostMortemReport(monitor)
        report.generate("./reports/run_001")
    """

    def __init__(self, monitor):
        self.monitor = monitor
        self.history = monitor.get_history()
        self.alerts = monitor.get_alerts()
        self.snapshots = monitor.get_snapshots()
        # Extract actual detector parameters for accurate reconstruction
        self._cusum_k = monitor.cusum_total.k
        self._cusum_h = monitor.cusum_total.h
        self._warmup = monitor.cusum_total.warmup_steps

    def generate(self, output_dir: str = "./reports"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._plot_loss_with_control_chart(output_dir)
        self._plot_gradient_norms(output_dir)
        self._plot_layer_gradient_heatmap(output_dir)
        self._plot_cusum_accumulators(output_dir)
        self._write_text_report(output_dir)
        self._plot_spike_forensics(output_dir)

        print(f"\nPost-mortem report saved to {output_dir}/")

    def _plot_loss_with_control_chart(self, output_dir: Path):
        fig, ax = plt.subplots(figsize=(14, 5))
        steps = self.history["steps"]
        loss = self.history["loss"]

        ax.plot(steps, loss, "b-", alpha=0.7, linewidth=0.8, label="Loss")

        # Compute rolling statistics for control limits
        window = 50
        if len(loss) > window:
            rolling_mean = np.convolve(loss, np.ones(window) / window, mode="valid")
            rolling_std = np.array([
                np.std(loss[max(0, i - window):i]) for i in range(window, len(loss) + 1)
            ])
            x = steps[window - 1:]
            ax.plot(x, rolling_mean, "k-", linewidth=1.5, label="Rolling mean")
            ax.fill_between(
                x,
                rolling_mean - 3 * rolling_std,
                rolling_mean + 3 * rolling_std,
                alpha=0.15, color="blue", label="3-sigma control limits",
            )

        # Mark alerts on loss
        loss_alerts = [a for a in self.alerts if a.metric_name == "loss"]
        if loss_alerts:
            alert_steps = [a.step for a in loss_alerts]
            alert_vals = [a.value for a in loss_alerts]
            ax.scatter(alert_steps, alert_vals, c="red", s=80, zorder=5,
                       marker="x", linewidths=2, label="Anomaly detected")

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss — Shewhart Control Chart")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "loss_control_chart.png", dpi=150)
        plt.close(fig)

    def _plot_gradient_norms(self, output_dir: Path):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        steps = self.history["steps"]

        # Total gradient norm
        ax = axes[0]
        total = self.history["total_grad_norm"]
        ax.plot(steps, total, "g-", alpha=0.7, linewidth=0.8)
        ax.set_ylabel("Total Gradient Norm")
        ax.set_title("Gradient Norms Over Training")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Mark gradient alerts
        grad_alerts = [a for a in self.alerts if "grad_norm" in a.metric_name]
        if grad_alerts:
            alert_steps = [a.step for a in grad_alerts]
            alert_vals = [a.value for a in grad_alerts]
            ax.scatter(alert_steps, alert_vals, c="red", s=60, zorder=5,
                       marker="x", linewidths=2, label="Spike")
            ax.legend()

        # Learning rate
        ax2 = axes[1]
        ax2.plot(steps, self.history["learning_rates"], "m-", linewidth=1.2)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "gradient_norms.png", dpi=150)
        plt.close(fig)

    def _plot_layer_gradient_heatmap(self, output_dir: Path):
        layer_norms = self.history["layer_grad_norms"]

        # Select top N layers by max gradient norm for readability
        max_norms = {name: max(vals) if vals else 0 for name, vals in layer_norms.items()}
        top_layers = sorted(max_norms, key=max_norms.get, reverse=True)[:20]

        if not top_layers:
            return

        data = np.array([layer_norms[name] for name in top_layers])
        # Log-scale for visibility
        data = np.log10(data + 1e-10)

        fig, ax = plt.subplots(figsize=(16, max(4, len(top_layers) * 0.35)))
        im = ax.imshow(data, aspect="auto", cmap="hot", interpolation="nearest")
        ax.set_yticks(range(len(top_layers)))
        short_names = [n.split(".")[-2] + "." + n.split(".")[-1] if "." in n else n
                       for n in top_layers]
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_xlabel("Step")
        ax.set_title("Per-Layer Gradient Norms (log10 scale) — Top 20 Layers")
        fig.colorbar(im, ax=ax, label="log10(grad norm)")

        # Mark spike steps
        spike_steps = set(a.step for a in self.alerts)
        step_list = self.history["steps"]
        for ss in spike_steps:
            if ss in step_list:
                idx = step_list.index(ss)
                ax.axvline(idx, color="cyan", alpha=0.5, linewidth=0.8)

        fig.tight_layout()
        fig.savefig(output_dir / "layer_gradient_heatmap.png", dpi=150)
        plt.close(fig)

    def _plot_cusum_accumulators(self, output_dir: Path):
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        # Reconstruct CUSUM trace for total grad norm
        values = self.history["total_grad_norm"]
        steps = self.history["steps"]

        k = self._cusum_k
        h = self._cusum_h
        warmup_steps = self._warmup

        for ax, (vals, title) in zip(axes, [
            (values, "Total Gradient Norm — CUSUM Accumulator"),
            (self.history["loss"], "Loss — CUSUM Accumulator"),
        ]):
            if len(vals) < warmup_steps:
                continue

            # Match the detector's calibration: last 30% of warmup window
            cal_start = max(0, warmup_steps - warmup_steps // 3)
            warmup_data = np.array(vals[cal_start:warmup_steps])
            mean = float(np.mean(warmup_data))
            std = float(np.std(warmup_data)) + 1e-8

            S_high, S_low = [], []
            sh, sl = 0.0, 0.0
            for i, v in enumerate(vals):
                if i < warmup_steps:
                    S_high.append(0.0)
                    S_low.append(0.0)
                    continue
                z = (v - mean) / std
                sh = max(0, sh + z - k)
                sl = max(0, sl - z - k)
                S_high.append(sh)
                S_low.append(sl)
                # Reset after alarm, matching the real detector
                if sh > h:
                    sh = 0.0
                if sl > h:
                    sl = 0.0

            ax.plot(steps, S_high, "r-", alpha=0.8, label="S+ (upward shift)")
            ax.plot(steps, S_low, "b-", alpha=0.8, label="S- (downward shift)")
            ax.axhline(h, color="k", linestyle="--", alpha=0.5, label=f"Threshold h={h}")
            ax.set_ylabel("CUSUM Statistic")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Step")
        fig.tight_layout()
        fig.savefig(output_dir / "cusum_accumulators.png", dpi=150)
        plt.close(fig)

    def _plot_spike_forensics(self, output_dir: Path):
        if not self.snapshots:
            return

        for i, snapshot in enumerate(self.snapshots[:20]):  # limit to 20
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Gradient norms at spike
            grad_norms = snapshot.gradient_norms
            names = list(grad_norms.keys())
            vals = list(grad_norms.values())

            # Show top 15 layers by gradient norm
            sorted_pairs = sorted(zip(names, vals), key=lambda x: x[1], reverse=True)[:15]
            s_names, s_vals = zip(*sorted_pairs) if sorted_pairs else ([], [])

            ax = axes[0]
            short = [n.split(".")[-2] + "." + n.split(".")[-1] if "." in n else n for n in s_names]
            ax.barh(range(len(short)), s_vals, color="coral")
            ax.set_yticks(range(len(short)))
            ax.set_yticklabels(short, fontsize=8)
            ax.set_xlabel("Gradient Norm")
            ax.set_title(f"Step {snapshot.step}: Top Gradient Norms")
            ax.invert_yaxis()

            # Parameter stats at spike
            ax2 = axes[1]
            param_stats = snapshot.parameter_stats
            p_names = list(param_stats.keys())[:15]
            p_norms = [param_stats[n]["norm"] for n in p_names]
            short_p = [n.split(".")[-2] + "." + n.split(".")[-1] if "." in n else n for n in p_names]
            ax2.barh(range(len(short_p)), p_norms, color="steelblue")
            ax2.set_yticks(range(len(short_p)))
            ax2.set_yticklabels(short_p, fontsize=8)
            ax2.set_xlabel("Parameter Norm")
            ax2.set_title(f"Step {snapshot.step}: Parameter Norms")
            ax2.invert_yaxis()

            fig.suptitle(
                f"Forensic Snapshot — Step {snapshot.step} | "
                f"Loss={snapshot.loss:.4f} | LR={snapshot.learning_rate:.2e} | "
                f"Alert: {snapshot.alert['detector_type']}",
                fontsize=11, fontweight="bold",
            )
            fig.tight_layout()
            fig.savefig(output_dir / f"forensic_step_{snapshot.step}.png", dpi=150)
            plt.close(fig)

    def _write_text_report(self, output_dir: Path):
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("TRAINING POST-MORTEM REPORT")
        report_lines.append("=" * 70)

        total_steps = len(self.history["steps"])
        report_lines.append(f"\nTotal training steps: {total_steps}")
        report_lines.append(f"Total anomalies detected: {len(self.alerts)}")
        report_lines.append(f"Forensic snapshots saved: {len(self.snapshots)}")

        if self.history["loss"]:
            report_lines.append(f"Final loss: {self.history['loss'][-1]:.4f}")
            report_lines.append(f"Min loss: {min(self.history['loss']):.4f}")
            report_lines.append(f"Max loss: {max(self.history['loss']):.4f}")

        # Alert summary
        report_lines.append(f"\n{'='*70}")
        report_lines.append("ANOMALY SUMMARY")
        report_lines.append(f"{'='*70}")

        detector_counts = {}
        for a in self.alerts:
            key = a.detector_type
            detector_counts[key] = detector_counts.get(key, 0) + 1

        for dtype, count in sorted(detector_counts.items()):
            report_lines.append(f"  {dtype}: {count} alerts")

        # Per-spike details
        report_lines.append(f"\n{'='*70}")
        report_lines.append("SPIKE DETAILS")
        report_lines.append(f"{'='*70}")

        for snapshot in self.snapshots:
            report_lines.append(f"\n--- Step {snapshot.step} ---")
            report_lines.append(f"  Loss: {snapshot.loss:.6f}")
            report_lines.append(f"  Learning rate: {snapshot.learning_rate:.2e}")
            report_lines.append(f"  Alert type: {snapshot.alert['detector_type']}")
            report_lines.append(f"  Metric: {snapshot.alert['metric_name']}")
            report_lines.append(f"  Value: {snapshot.alert['value']:.6f}")

            # Top 5 gradient norms
            grad_norms = snapshot.gradient_norms
            top5 = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:5]
            report_lines.append("  Top 5 gradient norms:")
            for name, val in top5:
                report_lines.append(f"    {name}: {val:.6f}")

            # Optimizer state
            report_lines.append("  Optimizer state:")
            for group_name, state in snapshot.optimizer_state_summary.items():
                report_lines.append(f"    {group_name}: {state}")

        report_lines.append(f"\n{'='*70}")
        report_lines.append("RECOMMENDATIONS")
        report_lines.append(f"{'='*70}")

        if self.alerts:
            # Analyze patterns
            grad_alerts = [a for a in self.alerts if "grad_norm" in a.metric_name]
            loss_alerts = [a for a in self.alerts if a.metric_name == "loss"]
            cusum_alerts = [a for a in self.alerts if "CUSUM" in a.detector_type]

            if grad_alerts:
                report_lines.append(
                    "\n- GRADIENT SPIKES DETECTED: Consider gradient clipping "
                    "(torch.nn.utils.clip_grad_norm_) or reducing learning rate."
                )
                # Find which layers spiked most
                layer_spike_counts = {}
                for a in grad_alerts:
                    layer = a.metric_name
                    layer_spike_counts[layer] = layer_spike_counts.get(layer, 0) + 1
                worst = sorted(layer_spike_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                report_lines.append(f"  Most affected layers: {worst}")

            if loss_alerts:
                report_lines.append(
                    "\n- LOSS SPIKES DETECTED: Inspect the batch data at flagged steps "
                    "for corrupted inputs. Check learning rate schedule."
                )

            if cusum_alerts:
                report_lines.append(
                    "\n- SUSTAINED SHIFTS DETECTED (CUSUM): The training process "
                    "experienced a persistent change in gradient/loss statistics, "
                    "not just isolated outliers. This may indicate data distribution "
                    "shift or learning rate being too high for convergence."
                )
        else:
            report_lines.append("\nNo anomalies detected. Training appears stable.")

        report_text = "\n".join(report_lines)

        with open(output_dir / "post_mortem_report.txt", "w") as f:
            f.write(report_text)

        print(report_text)
