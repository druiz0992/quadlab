"""IMU imperfections beyond white noise and random-walk bias.

Three parts, applied at different stages of the IMU pipeline (see
experiments/common/sensor_pipeline.py: imperfections → Allan errors → temperature →
vibration → saturation → faults → timing):

``ImuImperfections`` (sensor-level, on the input signal), per instrument::

    y = (I + diag(s) + C) x + b_GM(t)        and, for the gyro, + G f

- ``*_scale_factor`` s: per-axis scale-factor error (fraction).
- ``*_cross_coupling`` C: off-diagonal axis misalignment/non-orthogonality
  (rad, small angle; the diagonal must be zero).
- ``gyro_g_sensitivity_rad_s_per_mps2`` G: gyro output per unit specific
  force (3×3); typical MEMS 0.01–0.1 °/s/g, i.e. 2e-5–2e-4 (rad/s)/(m/s²).
- ``*_bias_instability`` σ with ``bias_instability_correlation_s`` τ: a
  first-order Gauss–Markov bias, b ← e^{−Δt/τ} b + σ √(1 − e^{−2Δt/τ}) w,
  the common model for the flicker floor seen in Allan deviation plots.

``ImuSaturation``: each axis clipped to ±``accel_range_mps2`` /
±``gyro_range_rad_s`` (null: unlimited), as an ADC full scale.

``ImuTiming``: whole samples lost with ``sample_drop_probability`` or inside
``outages_s`` intervals, and the reported timestamp moved by
N(0, ``timestamp_jitter_s``²) clipped to ±0.45 of the sample period so time
never reverses. Jitter changes only the reported stamp; the physics and the
other models use the true sample time.

Seeded local RNGs; disabled parts are exact pass-throughs.
"""

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.measurements import readonly_array


def _matrix(values: Any) -> NDArray[np.float64]:
    return np.asarray(values, float).reshape(3, 3)


class ImuImperfections:
    """Scale factor, cross-coupling, g-sensitivity, and bias instability."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.enabled: bool = settings["enabled"]
        self._rng = np.random.default_rng(settings["seed"])
        self._accel_m = (
            np.eye(3)
            + np.diag(settings["accel_scale_factor"])
            + _matrix(settings["accel_cross_coupling"])
        )
        self._gyro_m = (
            np.eye(3)
            + np.diag(settings["gyro_scale_factor"])
            + _matrix(settings["gyro_cross_coupling"])
        )
        self._g_sens = _matrix(settings["gyro_g_sensitivity_rad_s_per_mps2"])
        self._sigma = (
            np.asarray(settings["accel_bias_instability_mps2"], float),
            np.asarray(settings["gyro_bias_instability_rad_s"], float),
        )
        self._tau = settings["bias_instability_correlation_s"]
        # Start in the stationary distribution.
        self._bias = [self._rng.normal(size=3) * s for s in self._sigma]
        self._previous_ns: int | None = None

    def apply(
        self,
        specific_force_mps2: ArrayLike,
        angular_rate_rad_s: ArrayLike,
        sim_time_ns: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        force = readonly_array(specific_force_mps2, (3,))
        rate = readonly_array(angular_rate_rad_s, (3,))
        if not self.enabled:
            return force, rate
        self._advance(sim_time_ns)
        accel = self._accel_m @ force + self._bias[0]
        gyro = self._gyro_m @ rate + self._g_sens @ force + self._bias[1]
        return accel, gyro

    def _advance(self, sim_time_ns: int) -> None:
        previous, self._previous_ns = self._previous_ns, sim_time_ns
        if previous is None or sim_time_ns <= previous:
            return
        decay = math.exp(-(sim_time_ns - previous) * 1e-9 / self._tau)
        spread = math.sqrt(1 - decay**2)
        self._bias = [
            decay * b + spread * s * self._rng.normal(size=3)
            for b, s in zip(self._bias, self._sigma)
        ]

    def current_biases(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """True Gauss–Markov (accel, gyro) bias now, sensor axes. Evaluation only."""
        if not self.enabled:
            return np.zeros(3), np.zeros(3)
        return self._bias[0].copy(), self._bias[1].copy()


class ImuSaturation:
    """Clip each axis to the instrument's full-scale range."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self._accel = settings["accel_range_mps2"]
        self._gyro = settings["gyro_range_rad_s"]
        self.enabled = bool(
            settings["enabled"] and (self._accel is not None or self._gyro is not None)
        )
        self.saturated_samples = 0

    def apply(
        self,
        specific_force_mps2: ArrayLike,
        angular_rate_rad_s: ArrayLike,
        sim_time_ns: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        force = np.asarray(specific_force_mps2, float)
        rate = np.asarray(angular_rate_rad_s, float)
        if not self.enabled:
            return force, rate
        clipped = False
        if self._accel is not None:
            clipped |= bool(np.any(np.abs(force) > self._accel))
            force = np.clip(force, -self._accel, self._accel)
        if self._gyro is not None:
            clipped |= bool(np.any(np.abs(rate) > self._gyro))
            rate = np.clip(rate, -self._gyro, self._gyro)
        self.saturated_samples += int(clipped)
        return force, rate


class ImuTiming:
    """Sample drops, outages, and timestamp jitter (see module docs)."""

    def __init__(self, settings: Mapping[str, Any], rate_hz: float) -> None:
        self._rng = np.random.default_rng(settings["seed"] + 1)
        self._p = settings["sample_drop_probability"]
        self._outages = [
            (round(a * 1e9), round(b * 1e9)) for a, b in settings["outages_s"]
        ]
        self._jitter = settings["timestamp_jitter_s"]
        self._limit = 0.45 / rate_hz
        self.enabled = bool(
            settings["enabled"] and (self._p or self._outages or self._jitter)
        )
        self.dropped = 0

    def drop(self, sim_time_ns: int) -> bool:
        """Whether the sample at ``sim_time_ns`` is lost."""
        if not self.enabled:
            return False
        lost = self._rng.random() < self._p or any(
            a <= sim_time_ns < b for a, b in self._outages
        )
        self.dropped += int(lost)
        return lost

    def stamp(self, sim_time_ns: int) -> int:
        """Reported timestamp with jitter (nanoseconds, never negative)."""
        if not self.enabled or not self._jitter:
            return sim_time_ns
        offset = float(
            np.clip(self._rng.normal() * self._jitter, -self._limit, self._limit)
        )
        return max(0, sim_time_ns + round(offset * 1e9))
