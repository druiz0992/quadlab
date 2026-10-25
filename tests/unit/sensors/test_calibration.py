"""Applying sensor calibrations: IMU correction and startup gyro bias,
magnetometer compensation and plausibility, config round trips."""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from robotlab.sensors.calibration import (
    ImuCalibration,
    ImuCalibrationDriver,
    MagnetometerCalibration,
)

MS = 1_000_000
A = np.eye(3) + np.diag([0.005, -0.004, 0.006])
G = np.eye(3) * 5e-5
B_A, B_G = np.array([0.05, -0.03, 0.02]), np.array([0.003, -0.002, 0.001])
EARTH = np.array([0.0, 25.6, -37.2])  # µT, ENU
SOFT = np.array([[1.04, 0.02, -0.01], [0.02, 0.97, 0.015], [-0.01, 0.015, 1.02]])
HARD = np.array([6.0, -3.0, 2.0])


def imu_calibration(startup_s: float) -> ImuCalibration:
    """Exact parameters of the A, G, B_A, B_G error model, no temperature."""
    return ImuCalibration(
        B_A, np.linalg.inv(A), np.zeros(3), B_G, G, np.zeros(3), 25.0, startup_s
    )


def test_driver_learns_startup_gyro_bias_then_subtracts_it() -> None:
    driver = ImuCalibrationDriver(imu_calibration(0.5))
    still = np.array([0.0, 0.0, 9.81])
    turn_on = np.array([0.002, -0.001, 0.0005])  # changes at every power-up
    raw = (A @ still + B_A, G @ still + B_G + turn_on)
    for i in range(101):  # 1 s at 100 Hz, still and level
        driver.apply(*raw, 25.0, i * 10 * MS)
    assert driver.ready
    force, rate = driver.apply(*raw, 25.0, 2000 * MS)
    np.testing.assert_allclose(force, still, atol=1e-9)
    np.testing.assert_allclose(rate, 0, atol=1e-12)
    assert ImuCalibrationDriver(imu_calibration(0.0)).ready


def test_imu_calibration_config_round_trip() -> None:
    calibration = imu_calibration(1.5)
    section = json.loads(json.dumps(calibration.to_config()))
    again = ImuCalibration.from_config(section)
    assert section["enabled"] and again.startup_gyro_calibration_s == 1.5
    np.testing.assert_allclose(again.accel_correction, calibration.accel_correction)


def test_magnetometer_applies_compensation_and_checks_plausibility() -> None:
    cal = MagnetometerCalibration(
        HARD,
        np.linalg.inv(SOFT),
        np.array([0.4, 0, 0]),
        np.zeros(3),
        45.1,
        55.4,
        0.0,
        5.0,
        2.0,
    )
    true = Rotation.from_euler("z", 30, degrees=True).inv().apply(EARTH)
    raw = SOFT @ true + HARD + np.array([0.4, 0, 0]) * 20.0
    corrected = cal.apply(raw, current_a=20.0)
    np.testing.assert_allclose(corrected, true, atol=1e-9)
    assert cal.plausible(corrected) and not cal.plausible(corrected * 1.3)
    again = MagnetometerCalibration.from_config(json.loads(json.dumps(cal.to_config())))
    np.testing.assert_allclose(again.apply(raw, 20.0), corrected)
