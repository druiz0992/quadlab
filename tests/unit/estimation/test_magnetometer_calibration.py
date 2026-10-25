"""Magnetometer calibration: ellipsoid and interference fits, use by filters."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.core.measurements import GpsReading, MagnetometerReading
from robotlab.estimation.magnetometer_calibration import (
    fit_ellipsoid,
    fit_interference,
    heading_error_deg,
)
from robotlab.experiments.harness.algorithms import build_context
from robotlab.experiments.calibration.magnetometer_calibration import (
    calibration_config,
    write_world,
)
from robotlab.simulation.gazebo.config import load_drone_config

EARTH = np.array([0.0, 25.6, -37.2])  # µT, ENU: north and down at the default site
SOFT = np.array([[1.04, 0.02, -0.01], [0.02, 0.97, 0.015], [-0.01, 0.015, 1.02]])
HARD = np.array([6.0, -3.0, 2.0])


def distorted(count=400, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    rotations = Rotation.random(count, random_state=seed)
    true = rotations.inv().apply(EARTH)
    return true @ SOFT.T + HARD + rng.normal(size=(count, 3)) * noise, true, rotations


def test_ellipsoid_fit_recovers_hard_and_soft_iron():
    raw, true, _ = distorted()
    fit = fit_ellipsoid(raw, np.linalg.norm(EARTH))
    np.testing.assert_allclose(fit.offset_ut, HARD, atol=0.05)
    np.testing.assert_allclose(fit.correction @ SOFT, np.eye(3), atol=0.005)
    assert fit.residual_rms_ut < 0.15
    corrected = (raw - fit.offset_ut) @ fit.correction.T
    np.testing.assert_allclose(corrected, true, atol=0.6)


def test_ellipsoid_fit_needs_samples():
    with pytest.raises(ValueError, match="10 samples"):
        fit_ellipsoid(np.ones((5, 3)), 45.0)


def test_interference_regression():
    current = np.linspace(0, 30, 50)
    k = np.array([0.4, -0.2, 0.1])
    raw = np.outer(current, k) + [10, 20, 30]
    slope, intercept = fit_interference(raw, current)
    np.testing.assert_allclose(slope, k)
    np.testing.assert_allclose(intercept, [10, 20, 30])
    with pytest.raises(ValueError):
        fit_interference(raw, np.ones(50))


def test_heading_error_in_world_axes():
    to_world = Rotation.from_euler("z", 90, degrees=True)
    true = to_world.inv().apply(EARTH)
    rotated = Rotation.from_euler("z", 5, degrees=True).apply(EARTH)
    assert heading_error_deg(
        to_world.inv().apply(rotated), true, to_world
    ) == pytest.approx(5)


def test_calibration_layer_reaches_reading_and_filter_context(tmp_path):
    layer = {
        "sensors": {
            "magnetometer": {
                "enabled": True,
                "calibration": {"enabled": True, "heading_std_deg": 1.5},
            }
        }
    }
    path = tmp_path / "cal.json"
    path.write_text(json.dumps(layer))
    config = load_drone_config(path)
    context = build_context(
        config.data, config.properties, GpsReading(40, -3, 650), ("magnetometer",)
    )
    calibration = context.sensors["magnetometer"].calibration
    assert (
        calibration["heading_std_deg"] == 1.5
        and calibration["field_strength_ut"] == 45.1
    )
    reading = MagnetometerReading([1, 2, 3], calibrated_field_ut=[4, 5, 6])
    assert reading.calibrated_field_ut.tolist() == [4, 5, 6]
    plain = build_context(
        load_drone_config().data,
        config.properties,
        GpsReading(40, -3, 650),
        ("magnetometer",),
    )
    assert plain.sensors["magnetometer"].calibration is None


def test_calibration_world_has_no_gravity_and_raw_magnetometer(tmp_path):
    config = calibration_config(["configs/sensors/magnetometer.json"])
    sensors = config.data["sensors"]
    assert (
        sensors["magnetometer"]["enabled"]
        and not sensors["magnetometer"]["calibration"]["enabled"]
    )
    assert (
        not sensors["camera"]["enabled"] and not config.data["simulation"]["lockstep"]
    )
    world = ET.parse(write_world(config, tmp_path)).getroot().find("world")
    assert world.findtext("gravity") == "0 0 0"
