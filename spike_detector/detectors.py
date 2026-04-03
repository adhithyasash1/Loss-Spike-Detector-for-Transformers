"""
Statistical Process Control detectors for training anomalies.

Implements CUSUM (Cumulative Sum) and Shewhart control chart methods
for detecting shifts in gradient norms and loss values during training.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Alert:
    step: int
    metric_name: str
    value: float
    detector_type: str
    details: dict = field(default_factory=dict)


class CUSUMDetector:
    """
    Tabular CUSUM (Cumulative Sum) detector.

    Detects sustained shifts in a process mean. Maintains two accumulators:
    - S_high: detects upward shifts (gradient explosion)
    - S_low: detects downward shifts (gradient vanishing)

    Parameters:
        allowance (k): The slack value, typically half the shift size to detect.
                       Smaller k = more sensitive to small shifts.
        threshold (h): Decision interval. Alarm when cumulative sum exceeds h.
                       Larger h = fewer false alarms but slower detection.
        warmup_steps: Number of steps to collect before starting detection
                      (used to estimate the in-control mean and std).
    """

    def __init__(self, allowance: float = 0.5, threshold: float = 5.0,
                 warmup_steps: int = 50):
        self.k = allowance
        self.h = threshold
        self.warmup_steps = warmup_steps

        self.values: list[float] = []
        self.S_high: float = 0.0
        self.S_low: float = 0.0
        self.mean: float = 0.0
        self.std: float = 1.0
        self._warmed_up = False

    def reset(self):
        self.S_high = 0.0
        self.S_low = 0.0

    def update(self, value: float, step: int, metric_name: str = "grad_norm") -> Alert | None:
        self.values.append(value)

        if len(self.values) < self.warmup_steps:
            return None

        if not self._warmed_up:
            # Use the last 30% of warmup for calibration — the process
            # is more stable after initial transients settle
            cal_start = max(0, self.warmup_steps - self.warmup_steps // 3)
            warmup_data = np.array(self.values[cal_start:self.warmup_steps])
            self.mean = float(np.mean(warmup_data))
            self.std = float(np.std(warmup_data)) + 1e-8
            self._warmed_up = True

        # Standardize
        z = (value - self.mean) / self.std

        # Update cumulative sums
        self.S_high = max(0, self.S_high + z - self.k)
        self.S_low = max(0, self.S_low - z - self.k)

        if self.S_high > self.h:
            alert = Alert(
                step=step,
                metric_name=metric_name,
                value=value,
                detector_type="CUSUM_HIGH",
                details={
                    "S_high": self.S_high,
                    "z_score": z,
                    "in_control_mean": self.mean,
                    "in_control_std": self.std,
                },
            )
            self.S_high = 0.0  # reset after alarm
            return alert

        if self.S_low > self.h:
            alert = Alert(
                step=step,
                metric_name=metric_name,
                value=value,
                detector_type="CUSUM_LOW",
                details={
                    "S_low": self.S_low,
                    "z_score": z,
                    "in_control_mean": self.mean,
                    "in_control_std": self.std,
                },
            )
            self.S_low = 0.0
            return alert

        return None


class ShewhartDetector:
    """
    Shewhart (Western Electric) control chart detector.

    Flags individual points that fall outside control limits.
    Uses a rolling window to estimate process mean and standard deviation,
    making it adaptive to gradual drift.

    Rules implemented:
        1. Any single point > n_sigma standard deviations from mean
        2. 2 out of 3 consecutive points > 2 sigma (warning zone)

    Parameters:
        n_sigma: Number of standard deviations for the control limit.
        window_size: Rolling window for estimating mean/std.
        warmup_steps: Minimum observations before detection starts.
    """

    def __init__(self, n_sigma: float = 3.0, window_size: int = 100,
                 warmup_steps: int = 30):
        self.n_sigma = n_sigma
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.values: list[float] = []

    def update(self, value: float, step: int, metric_name: str = "grad_norm") -> Alert | None:
        self.values.append(value)

        if len(self.values) < self.warmup_steps:
            return None

        # Use rolling window (exclude the current point for fair estimation)
        window_start = max(0, len(self.values) - self.window_size - 1)
        window = np.array(self.values[window_start:-1])
        mean = float(np.mean(window))
        std = float(np.std(window)) + 1e-8

        z = (value - mean) / std

        # Rule 1: Single point beyond n_sigma (upward only — we care about
        # gradient explosions and loss spikes, not natural decreases)
        if z > self.n_sigma:
            return Alert(
                step=step,
                metric_name=metric_name,
                value=value,
                detector_type="SHEWHART",
                details={
                    "z_score": z,
                    "control_mean": mean,
                    "control_std": std,
                    "upper_limit": mean + self.n_sigma * std,
                    "lower_limit": mean - self.n_sigma * std,
                },
            )

        # Rule 2: 2 of 3 consecutive points > 2 sigma (upward only — catches spikes
        # without triggering during natural loss decrease)
        if len(self.values) >= 3:
            recent = self.values[-3:]
            z_recent = [(v - mean) / std for v in recent]
            violations = sum(1 for z_val in z_recent if z_val > 2.0)
            if violations >= 2:
                return Alert(
                    step=step,
                    metric_name=metric_name,
                    value=value,
                    detector_type="SHEWHART_2of3",
                    details={
                        "z_score": z,
                        "z_recent_3": z_recent,
                        "control_mean": mean,
                        "control_std": std,
                    },
                )

        return None
