"""IMU calibration: accelerometer scale/misalignment/offset, gyro offset and
g-sensitivity, temperature slopes, and startup gyro bias.

Static calibration (the six-position test, generalized): the IMU rests in
known orientations, so the true specific force in sensor axes is known,
f = R_mountᵀ R_bodyᵀ (0, 0, g), and the true rate is zero. With the die
temperature T reported by the IMU, each axis j of the readings satisfies

    accel:  a_j = A_j · f + b_j + k_j (T − T_ref)
    gyro:   ω_j = G_j · f + b_j + k_j (T − T_ref)

(A = scale and misalignment, G = g-sensitivity, b = offset, k = temperature
slope), a linear least-squares problem per axis. Six axis-aligned poses fix
A and b; extra tilted poses improve conditioning; poses at several
temperatures make k identifiable (otherwise it is reported as zero).

Correction (what a driver applies, ``ImuCalibration.apply``):

    f̂ = A⁻¹ (a − b_a − k_a ΔT)
    ω̂ = ω − b_g − k_g ΔT − G f̂ − b_startup

where b_startup is the optional startup gyro bias, the mean residual gyro
reading over the first seconds after power-up while the vehicle is still
(turn-on bias changes at every power-up, so it is not stored). Rate-table
quantities (gyro scale and misalignment) need a turntable and are not
calibrated here.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _regress(
    readings: NDArray[np.float64],
    forces: NDArray[np.float64],
    delta_t: NDArray[np.float64],
    with_temperature: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Per-axis fit readings = M f + b + k ΔT; returns (M, b, k, residual RMS)."""
    columns = [forces, np.ones((len(forces), 1))]
    if with_temperature:
        columns.append(delta_t[:, None])
    design = np.hstack(columns)
    solution, *_ = np.linalg.lstsq(design, readings, rcond=None)
    matrix, offset = solution[:3].T, solution[3]
    slope = solution[4] if with_temperature else np.zeros(3)
    residual = readings - design @ solution
    return matrix, offset, slope, float(np.sqrt(np.mean(residual**2)))


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
    def fit_static(
        cls,
        accel: ArrayLike,
        gyro: ArrayLike,
        forces: ArrayLike,
        temperatures_c: ArrayLike,
        reference_temperature_c: float = 25.0,
    ) -> "ImuCalibration":
        """Fit from static readings and the known specific force per sample."""
        a, w = np.asarray(accel, float), np.asarray(gyro, float)
        f = np.asarray(forces, float)
        delta = np.asarray(temperatures_c, float) - reference_temperature_c
        if len(f) < 6 or np.linalg.matrix_rank(np.c_[f, np.ones(len(f))]) < 4:
            raise ValueError("Static IMU calibration needs poses spanning all axes")
        thermal = bool(np.ptp(delta) > 1.0)
        scale, b_a, k_a, r_a = _regress(a, f, delta, thermal)
        g_sens, b_g, k_g, r_g = _regress(w, f, delta, thermal)
        return cls(
            b_a,
            np.linalg.inv(scale),
            k_a,
            b_g,
            g_sens,
            k_g,
            reference_temperature_c,
            fit=dict(
                accel_residual_rms_mps2=r_a,
                gyro_residual_rms_rad_s=r_g,
                temperature_span_c=float(np.ptp(delta)),
            ),
        )

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
