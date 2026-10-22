"""IMU self-heating and temperature-dependent bias drift.

MEMS IMUs warm up after power-on, and their biases change with temperature,
so a filter calibrated at one temperature sees biases creep during flight:

    T(t)  = T_ambient + warmup_rise · (1 − exp(−t / τ))
    bias  = coefficient · (T − T_calibration)        (per axis, sensor axes)

with t from the first sample, τ = ``warmup_time_constant_s``, and separate
accelerometer (m/s²/°C) and gyro (rad/s/°C) coefficients. Typical MEMS
values are around 0.003–0.01 m/s²/°C and 1e-4–3e-4 rad/s/°C (0.005–0.02 °/s
per °C). Deterministic; ``temperature_c`` exposes the die temperature that
real IMUs report.
"""

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core.measurements import readonly_array


class ImuTemperatureModel:
    """Add temperature-dependent biases to (specific force, angular rate)."""

    def __init__(self, settings: Mapping[str, Any], ambient_c: float) -> None:
        self.enabled: bool = settings["enabled"]
        self._ambient = ambient_c
        self._rise = settings["warmup_rise_c"]
        self._tau = settings["warmup_time_constant_s"]
        self._calibration = settings["calibration_c"]
        self._accel = np.asarray(settings["accel_bias_per_c"], float)
        self._gyro = np.asarray(settings["gyro_bias_per_c"], float)
        self._start_ns: int | None = None
        self.temperature_c = ambient_c

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
        if self._start_ns is None:
            self._start_ns = sim_time_ns
        elapsed = (sim_time_ns - self._start_ns) * 1e-9
        self.temperature_c = self._ambient + self._rise * (
            1 - math.exp(-elapsed / self._tau)
        )
        offset = self.temperature_c - self._calibration
        return force + self._accel * offset, rate + self._gyro * offset

    def current_biases(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """True temperature-induced (accel, gyro) bias now, sensor axes.
        Evaluation only."""
        if not self.enabled:
            return np.zeros(3), np.zeros(3)
        offset = self.temperature_c - self._calibration
        return self._accel * offset, self._gyro * offset

    def set_ambient(self, ambient_c: float) -> None:
        """Change ambient temperature; the die follows at once (warm-up continues)."""
        self.temperature_c += ambient_c - self._ambient
        self._ambient = ambient_c

    def summary(self) -> dict[str, Any]:
        return dict(enabled=self.enabled, final_temperature_c=self.temperature_c)
