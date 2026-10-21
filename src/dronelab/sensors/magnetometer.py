"""Seeded magnetometer errors: hard iron, soft iron, motor interference, noise.

Gazebo's magnetometer reports the world magnetic field rotated into the sensor
axes, in tesla. Real compasses on drones see that field distorted by the
airframe and the motors:

    measured = soft_iron @ field + hard_iron + interference * total_thrust
               + current_interference * battery_current + drift(t) + white

- ``hard_iron_ut``: constant offset from magnetized parts (µT, sensor axes).
- ``soft_iron``: 3×3 matrix for field-dependent distortion and scale (identity
  means none).
- ``motor_interference_ut_per_n``: offset per newton of total commanded rotor
  thrust, standing in for motor and power-wire currents (µT/N). Requires
  rotor-thrust mode; elsewhere total thrust is unknown and treated as zero.
- ``current_interference_ut_per_a``: offset per ampere of battery current
  (µT/A), from the power wiring; needs the battery model (else zero).
- ``drift_std_ut`` and ``drift_correlation_s``: slow time-varying offset, a
  first-order Gauss–Markov process per axis (magnetization and temperature
  changes), started in its stationary distribution.
- ``white_std_ut``: per-sample white noise std (µT).

All settings are in microtesla; messages stay in tesla. The RNG is local.
Fields from nearby structures are a separate physical model
(``magnetic_anomalies.py``) applied to the field before these errors.
"""

import math

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.measurements import readonly_array

TESLA_PER_MICROTESLA = 1e-6


class MagnetometerErrorModel:
    """Apply configured distortions to field vectors (tesla, sensor axes).

    ``settings`` is the validated ``sensors.magnetometer.error_model`` section.
    """

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.enabled: bool = settings["enabled"]
        self._rng = np.random.default_rng(settings["seed"])
        self._soft_iron = np.asarray(settings["soft_iron"], dtype=float)
        self._hard_iron_t = (
            np.asarray(settings["hard_iron_ut"], dtype=float) * TESLA_PER_MICROTESLA
        )
        self._interference_t_per_n = (
            np.asarray(settings["motor_interference_ut_per_n"], dtype=float)
            * TESLA_PER_MICROTESLA
        )
        self._white_std_t = (
            np.asarray(settings["white_std_ut"], dtype=float) * TESLA_PER_MICROTESLA
        )
        self._current_t_per_a = (
            np.asarray(settings["current_interference_ut_per_a"], dtype=float)
            * TESLA_PER_MICROTESLA
        )
        self._drift_std_t = (
            np.asarray(settings["drift_std_ut"], dtype=float) * TESLA_PER_MICROTESLA
        )
        self._drift_tau = settings["drift_correlation_s"]
        self._drift_t = self._drift_std_t * self._rng.normal(size=3)
        self._previous_ns: int | None = None
        self._samples = 0
        self._max_interference_ut = 0.0

    def apply(
        self,
        field_tesla: ArrayLike,
        total_thrust_n: float = 0.0,
        battery_current_a: float = 0.0,
        sim_time_ns: int | None = None,
    ) -> NDArray[np.float64]:
        """Return the distorted field (tesla); identity when disabled."""
        field = readonly_array(field_tesla, (3,))
        if not self.enabled:
            return field
        self._advance_drift(sim_time_ns)
        interference = (
            self._interference_t_per_n * total_thrust_n
            + self._current_t_per_a * battery_current_a
        )
        self._samples += 1
        self._max_interference_ut = max(
            self._max_interference_ut,
            float(np.linalg.norm(interference)) / TESLA_PER_MICROTESLA,
        )
        white = self._white_std_t * self._rng.normal(size=3)
        return (
            self._soft_iron @ field
            + self._hard_iron_t
            + interference
            + self._drift_t
            + white
        )

    def _advance_drift(self, sim_time_ns: int | None) -> None:
        previous, self._previous_ns = self._previous_ns, sim_time_ns
        if sim_time_ns is None or previous is None or sim_time_ns <= previous:
            return
        decay = math.exp(-(sim_time_ns - previous) * 1e-9 / self._drift_tau)
        self._drift_t = decay * self._drift_t + math.sqrt(
            1 - decay**2
        ) * self._drift_std_t * self._rng.normal(size=3)

    def summary(self) -> dict[str, Any]:
        """Diagnostics for summary.json, for evaluation only."""
        return dict(
            enabled=self.enabled,
            samples=self._samples,
            max_motor_interference_ut=self._max_interference_ut,
            final_drift_ut=(self._drift_t / TESLA_PER_MICROTESLA).tolist(),
        )
