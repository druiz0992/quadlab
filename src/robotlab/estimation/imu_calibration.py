"""IMU calibration fit: accelerometer scale/misalignment/offset, gyro offset
and g-sensitivity, and temperature slopes.

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

``fit_static`` returns an ``ImuCalibration`` (sensors/calibration.py),
which also documents the correction a driver applies and the startup gyro
bias measured at power-up. Rate-table quantities (gyro scale and
misalignment) need a turntable and are not calibrated here. Raises
ValueError when the poses do not span all axes.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..sensors.calibration import ImuCalibration


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


def fit_static(
    accel: ArrayLike,
    gyro: ArrayLike,
    forces: ArrayLike,
    temperatures_c: ArrayLike,
    reference_temperature_c: float = 25.0,
) -> ImuCalibration:
    """Fit from static readings (n×3, sensor axes: m/s², rad/s), the known
    specific force per sample (n×3, m/s²), and die temperatures (°C)."""
    a, w = np.asarray(accel, float), np.asarray(gyro, float)
    f = np.asarray(forces, float)
    delta = np.asarray(temperatures_c, float) - reference_temperature_c
    if len(f) < 6 or np.linalg.matrix_rank(np.c_[f, np.ones(len(f))]) < 4:
        raise ValueError("Static IMU calibration needs poses spanning all axes")
    thermal = bool(np.ptp(delta) > 1.0)
    scale, b_a, k_a, r_a = _regress(a, f, delta, thermal)
    g_sens, b_g, k_g, r_g = _regress(w, f, delta, thermal)
    return ImuCalibration(
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
