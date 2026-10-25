"""Applying an IMU calibration, as a driver does before data reach filters.

The parameter set is read from and written to ``sensors.imu.calibration``;
the fit that produces it is estimation (estimation/imu_calibration.py).

f̂ = A⁻¹ (a − b_a − k_a ΔT)
    ω̂ = ω − b_g − k_g ΔT − G f̂ − b_startup

with A scale and misalignment, b offsets, k temperature slopes
(ΔT = T − T_ref), G gyro g-sensitivity, and b_startup the optional startup
gyro bias: the mean corrected gyro reading over the first seconds after
power-up while the vehicle is still (turn-on bias changes at every
power-up, so it is not stored).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ImuCalibration:
    """Calibration result as used by drivers and filters (sensor axes, SI)."""

    accel_offset_mps2: NDArray[np.float64]
    accel_correction: NDArray[np.float64]  # A⁻¹
    accel_temperature_mps2_per_c: NDArray[np.float64]
    gyro_offset_rad_s: NDArray[np.float64]
    gyro_g_sensitivity: NDArray[np.float64]  # G, (rad/s)/(m/s²)
    gyro_temperature_rad_s_per_c: NDArray[np.float64]
    reference_temperature_c: float
    startup_gyro_calibration_s: float = 0.0
    fit: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "ImuCalibration":
        return cls(
            np.asarray(section["accel_offset_mps2"], float),
            np.asarray(section["accel_correction"], float),
            np.asarray(section["accel_temperature_mps2_per_c"], float),
            np.asarray(section["gyro_offset_rad_s"], float),
            np.asarray(section["gyro_g_sensitivity"], float),
            np.asarray(section["gyro_temperature_rad_s_per_c"], float),
            float(section["reference_temperature_c"]),
            float(section["startup_gyro_calibration_s"]),
        )

    def to_config(self) -> dict[str, Any]:
        return dict(
            enabled=True,
            accel_offset_mps2=self.accel_offset_mps2.tolist(),
            accel_correction=self.accel_correction.tolist(),
            accel_temperature_mps2_per_c=self.accel_temperature_mps2_per_c.tolist(),
            gyro_offset_rad_s=self.gyro_offset_rad_s.tolist(),
            gyro_g_sensitivity=self.gyro_g_sensitivity.tolist(),
            gyro_temperature_rad_s_per_c=self.gyro_temperature_rad_s_per_c.tolist(),
            reference_temperature_c=self.reference_temperature_c,
            startup_gyro_calibration_s=self.startup_gyro_calibration_s,
        )

    def with_startup(self, seconds: float) -> "ImuCalibration":
        return replace(self, startup_gyro_calibration_s=seconds)

    def apply(
        self,
        accel: ArrayLike,
        gyro: ArrayLike,
        temperature_c: float | None = None,
        startup_bias_rad_s: ArrayLike | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Corrected (specific force, angular rate)."""
        delta = (
            0.0
            if temperature_c is None
            else temperature_c - self.reference_temperature_c
        )
        force = self.accel_correction @ (
            np.asarray(accel, float)
            - self.accel_offset_mps2
            - self.accel_temperature_mps2_per_c * delta
        )
        rate = (
            np.asarray(gyro, float)
            - self.gyro_offset_rad_s
            - self.gyro_temperature_rad_s_per_c * delta
            - self.gyro_g_sensitivity @ force
        )
        if startup_bias_rad_s is not None:
            rate = rate - np.asarray(startup_bias_rad_s, float)
        return force, rate


class ImuCalibrationDriver:
    """Applies an ``ImuCalibration`` sample by sample, like an autopilot driver.

    With ``startup_gyro_calibration_s`` > 0, the corrected gyro readings of
    the first seconds (vehicle still on the ground) are averaged into a
    startup bias, subtracted from then on; ``ready`` turns true when done.
    """

    def __init__(self, calibration: ImuCalibration) -> None:
        self.calibration = calibration
        self._window_ns = round(calibration.startup_gyro_calibration_s * 1e9)
        self._first_ns: int | None = None
        self._sum = np.zeros(3)
        self._count = 0
        self.startup_bias_rad_s: NDArray[np.float64] | None = (
            None if self._window_ns > 0 else np.zeros(3)
        )

    @property
    def ready(self) -> bool:
        return self.startup_bias_rad_s is not None

    def apply(
        self,
        accel: ArrayLike,
        gyro: ArrayLike,
        temperature_c: float | None,
        sim_time_ns: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        force, rate = self.calibration.apply(accel, gyro, temperature_c)
        if self.startup_bias_rad_s is None:
            self._first_ns = (
                self._first_ns if self._first_ns is not None else sim_time_ns
            )
            self._sum += rate
            self._count += 1
            if sim_time_ns - self._first_ns >= self._window_ns:
                self.startup_bias_rad_s = self._sum / self._count
            return force, rate
        return force, rate - self.startup_bias_rad_s
