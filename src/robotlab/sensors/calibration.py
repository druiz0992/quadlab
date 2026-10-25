"""Applying sensor calibrations, as a driver does before data reach filters.

Parameter sets (sensor axes, SI except µT), read from and written to the
``sensors.imu.calibration`` / ``sensors.magnetometer.calibration`` config
sections; the fits that produce them are numerical estimation
(estimation/imu_calibration.py, estimation/magnetometer_calibration.py).

**IMU** (``ImuCalibration.apply``, ``ImuCalibrationDriver``):

    f̂ = A⁻¹ (a − b_a − k_a ΔT)
    ω̂ = ω − b_g − k_g ΔT − G f̂ − b_startup

with A scale and misalignment, b offsets, k temperature slopes
(ΔT = T − T_ref), G gyro g-sensitivity, and b_startup the optional startup
gyro bias: the mean corrected gyro reading over the first seconds after
power-up while the vehicle is still (turn-on bias changes at every
power-up, so it is not stored).

**Magnetometer** (``MagnetometerCalibration.apply``):

    b̂ = C (m − o − k_I I − k_T T)

with C the soft-iron correction, o the hard-iron offset, and k_I, k_T the
battery-current and thrust interference slopes; ``plausible`` checks |b̂|
against the local field strength.
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


@dataclass(frozen=True)
class MagnetometerCalibration:
    """Calibration result as used by drivers and filters (µT, sensor axes).

    ``field_strength_ut``, ``inclination_deg`` (positive down), and
    ``declination_deg`` (east of true north) describe the local Earth field;
    ``field_tolerance_ut`` is how far |b̂| may stray before a reading should
    be distrusted; ``heading_std_deg`` is the residual heading uncertainty.
    """

    hard_iron_offset_ut: NDArray[np.float64]
    soft_iron_correction: NDArray[np.float64]
    current_compensation_ut_per_a: NDArray[np.float64]
    thrust_compensation_ut_per_n: NDArray[np.float64]
    field_strength_ut: float
    inclination_deg: float
    declination_deg: float
    field_tolerance_ut: float
    heading_std_deg: float

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "MagnetometerCalibration":
        return cls(
            np.asarray(section["hard_iron_offset_ut"], float),
            np.asarray(section["soft_iron_correction"], float),
            np.asarray(section["current_compensation_ut_per_a"], float),
            np.asarray(section["thrust_compensation_ut_per_n"], float),
            float(section["field_strength_ut"]),
            float(section["inclination_deg"]),
            float(section["declination_deg"]),
            float(section["field_tolerance_ut"]),
            float(section["heading_std_deg"]),
        )

    def to_config(self) -> dict[str, Any]:
        return dict(
            enabled=True,
            hard_iron_offset_ut=self.hard_iron_offset_ut.tolist(),
            soft_iron_correction=self.soft_iron_correction.tolist(),
            current_compensation_ut_per_a=self.current_compensation_ut_per_a.tolist(),
            thrust_compensation_ut_per_n=self.thrust_compensation_ut_per_n.tolist(),
            field_strength_ut=self.field_strength_ut,
            inclination_deg=self.inclination_deg,
            declination_deg=self.declination_deg,
            field_tolerance_ut=self.field_tolerance_ut,
            heading_std_deg=self.heading_std_deg,
        )

    def apply(
        self, raw_ut: ArrayLike, current_a: float = 0.0, thrust_n: float = 0.0
    ) -> NDArray[np.float64]:
        """Corrected field b̂ (µT, sensor axes)."""
        compensated = (
            np.asarray(raw_ut, float)
            - self.hard_iron_offset_ut
            - self.current_compensation_ut_per_a * current_a
            - self.thrust_compensation_ut_per_n * thrust_n
        )
        return self.soft_iron_correction @ compensated

    def plausible(self, corrected_ut: ArrayLike) -> bool:
        """Whether |b̂| is within tolerance of the local field strength."""
        magnitude = float(np.linalg.norm(corrected_ut))
        return abs(magnitude - self.field_strength_ut) <= self.field_tolerance_ut
