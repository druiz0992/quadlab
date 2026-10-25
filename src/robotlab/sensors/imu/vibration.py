"""Accelerometer measurement stress, independent of vehicle dynamics and gyro."""

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...core.measurements import readonly_array


class AccelerometerVibration:
    """Sinusoidal acceleration plus correlated Gaussian noise, then saturation.

    Amplitudes are peaks in sensor-axis m/s²; random_std_mps2 is the stationary
    standard deviation of a first-order correlated process, not a noise density.
    Clipping applies to the complete specific force, including gravity response.
    """

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.enabled: bool = settings["enabled"]
        self._frequency: float = settings["frequency_hz"]
        self._amplitude = np.asarray(settings["accel_amplitude_mps2"], dtype=float)
        self._phase = np.deg2rad(settings["phase_deg"])
        self._std = np.asarray(settings["random_std_mps2"], dtype=float)
        self._tau: float = settings["random_correlation_s"]
        self._limit: float | None = settings["clip_limit_mps2"]
        self._rng = np.random.default_rng(settings["seed"])
        self._previous_time: int | None = None
        self._random = np.zeros(3)
        self._samples = 0
        self._clipped_samples = 0
        self._clipped_axes = np.zeros(3, dtype=int)
        self._peak = np.zeros(3)

    def apply(
        self, specific_force: ArrayLike, angular_rate: ArrayLike, sim_time_ns: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``ImuPerturbation``: vibrate the specific force; the rate is
        unchanged (vibration here is accelerometer-only)."""
        rate = readonly_array(angular_rate, (3,))
        return self.apply_force(specific_force, sim_time_ns), rate

    def apply_force(
        self, specific_force: ArrayLike, sim_time_ns: int
    ) -> NDArray[np.float64]:
        """Vibrated and clipped specific force (m/s², sensor axes)."""
        force = readonly_array(specific_force, (3,))
        if not self.enabled:
            return force
        if sim_time_ns < 0 or (
            self._previous_time is not None and sim_time_ns <= self._previous_time
        ):
            raise ValueError(
                "Vibration timestamps must be nonnegative and strictly increasing"
            )
        if self._previous_time is None:
            self._random = self._rng.normal(size=3) * self._std
        else:
            dt = (sim_time_ns - self._previous_time) * 1e-9
            alpha = math.exp(-dt / self._tau)
            self._random = alpha * self._random + math.sqrt(
                -math.expm1(-2 * dt / self._tau)
            ) * self._std * self._rng.normal(size=3)
        self._previous_time = sim_time_ns
        oscillation = self._amplitude * np.sin(
            2 * np.pi * self._frequency * sim_time_ns * 1e-9 + self._phase
        )
        measured = force + oscillation + self._random
        self._samples += 1
        self._peak = np.maximum(self._peak, np.abs(measured))
        if self._limit is not None:
            clipped = np.abs(measured) > self._limit
            self._clipped_samples += int(np.any(clipped))
            self._clipped_axes += clipped
            measured = np.clip(measured, -self._limit, self._limit)
        return measured

    def summary(self) -> dict[str, Any]:
        return dict(
            enabled=self.enabled,
            samples=self._samples,
            clipped_samples=self._clipped_samples,
            clipped_axes_xyz=self._clipped_axes.tolist(),
            peak_before_clip_mps2=self._peak.tolist(),
        )
