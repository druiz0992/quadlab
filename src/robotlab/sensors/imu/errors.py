"""Seeded IMU errors from Allan-variance-style parameters (accelerometer and gyro).

Each axis of each instrument receives, in sensor axes:

    measured = input + turn_on_bias + bias_walk(t) + white(t)

- ``turn_on_bias``: constant for the run, drawn once from N(0, turn_on_std²).
- ``bias_walk``: random walk; over an interval dt its increment has variance
  ``bias_random_walk² * dt``. Units are (m/s²)/√s or (rad/s)/√s, matching the
  EKF's ``accel_bias_walk``/``gyro_bias_walk`` so assumptions can match truth.
- ``white``: per-sample standard deviation ``noise_density * sqrt(rate_hz)``,
  i.e. the continuous density N in (unit)/√Hz discretized at the nominal IMU
  sample period. This is the usual Allan-variance (Kalibr-style) convention.

The model never touches timestamps, never sees vehicle truth beyond the
measurement it perturbs, and uses its own RNG so global NumPy state is
unchanged. Bias instability (flicker) and scale/misalignment are not modeled.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...core.measurements import readonly_array


@dataclass(frozen=True)
class InstrumentErrors:
    """Error parameters for one three-axis instrument, each a per-axis array."""

    noise_density: NDArray[np.float64]
    bias_random_walk: NDArray[np.float64]
    turn_on_bias_std: NDArray[np.float64]

    @classmethod
    def from_settings(
        cls, settings: Mapping[str, Any], prefix: str, unit: str
    ) -> "InstrumentErrors":
        """Read ``<prefix>_noise_density_<unit>_per_sqrt_hz`` style keys."""

        def axes(key: str) -> NDArray[np.float64]:
            return np.asarray(settings[key], dtype=float)

        return cls(
            noise_density=axes(f"{prefix}_noise_density_{unit}_per_sqrt_hz"),
            bias_random_walk=axes(f"{prefix}_bias_random_walk_{unit}_per_sqrt_s"),
            turn_on_bias_std=axes(f"{prefix}_turn_on_bias_std_{unit}"),
        )


class _InstrumentState:
    """Bias state and white-noise generator for one instrument."""

    def __init__(
        self, errors: InstrumentErrors, rate_hz: float, rng: np.random.Generator
    ) -> None:
        self._errors = errors
        self._rng = rng
        # Discretize the white-noise density at the nominal sample period.
        self._white_std = errors.noise_density * math.sqrt(rate_hz)
        self.turn_on_bias = rng.normal(size=3) * errors.turn_on_bias_std
        self.walk_bias = np.zeros(3)

    def step(self, dt_s: float) -> NDArray[np.float64]:
        """Advance the bias walk by dt_s and return the total additive error."""
        if dt_s > 0:
            increment_std = self._errors.bias_random_walk * math.sqrt(dt_s)
            self.walk_bias = self.walk_bias + increment_std * self._rng.normal(size=3)
        white = self._white_std * self._rng.normal(size=3)
        return self.turn_on_bias + self.walk_bias + white

    @property
    def total_bias(self) -> NDArray[np.float64]:
        """Current bias (turn-on plus walk), excluding white noise."""
        return self.turn_on_bias + self.walk_bias


class ImuErrorModel:
    """Apply seeded accelerometer and gyro errors to time-ordered IMU samples.

    ``settings`` is the validated ``sensors.imu.error_model`` section and
    ``rate_hz`` the configured IMU rate used to discretize noise densities.
    Draw order is fixed (accel turn-on, gyro turn-on, then per sample accel
    walk, accel white, gyro walk, gyro white) so a seed reproduces a run for
    identical sample timestamps.
    """

    def __init__(self, settings: Mapping[str, Any], rate_hz: float) -> None:
        self.enabled: bool = settings["enabled"]
        rng = np.random.default_rng(settings["seed"])
        self._accel = _InstrumentState(
            InstrumentErrors.from_settings(settings, "accel", "mps2"), rate_hz, rng
        )
        self._gyro = _InstrumentState(
            InstrumentErrors.from_settings(settings, "gyro", "rad_s"), rate_hz, rng
        )
        self._previous_time_ns: int | None = None
        self._samples = 0

    def apply(
        self,
        specific_force_mps2: ArrayLike,
        angular_rate_rad_s: ArrayLike,
        sim_time_ns: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return perturbed (specific force, angular rate) for one IMU sample.

        Timestamps must be nonnegative and strictly increasing. When disabled
        the inputs are returned unchanged and no random numbers are drawn.
        """
        force = readonly_array(specific_force_mps2, (3,))
        rate = readonly_array(angular_rate_rad_s, (3,))
        if not self.enabled:
            return force, rate
        dt_s = self._advance_clock(sim_time_ns)
        self._samples += 1
        return force + self._accel.step(dt_s), rate + self._gyro.step(dt_s)

    def _advance_clock(self, sim_time_ns: int) -> float:
        """Validate ordering and return elapsed seconds (0 for the first sample)."""
        previous = self._previous_time_ns
        if sim_time_ns < 0 or (previous is not None and sim_time_ns <= previous):
            raise ValueError(
                "IMU error timestamps must be nonnegative and strictly increasing"
            )
        self._previous_time_ns = sim_time_ns
        return 0.0 if previous is None else (sim_time_ns - previous) * 1e-9

    def current_biases(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """True (accel, gyro) bias now, sensor axes. Evaluation only, never an
        estimator input."""
        if not self.enabled:
            return np.zeros(3), np.zeros(3)
        return self._accel.total_bias, self._gyro.total_bias

    def summary(self) -> dict[str, Any]:
        """Diagnostics for summary.json: true injected biases, for evaluation only."""
        return dict(
            enabled=self.enabled,
            samples=self._samples,
            accel_turn_on_bias_mps2=self._accel.turn_on_bias.tolist(),
            gyro_turn_on_bias_rad_s=self._gyro.turn_on_bias.tolist(),
            final_accel_bias_mps2=self._accel.total_bias.tolist(),
            final_gyro_bias_rad_s=self._gyro.total_bias.tolist(),
        )
