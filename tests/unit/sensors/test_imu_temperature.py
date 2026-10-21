import math

import numpy as np
import pytest

from dronelab.sensors.imu_temperature import ImuTemperatureModel
from dronelab.simulation.gazebo.config import load_drone_config

HOVER = np.array([0.0, 0.0, 9.81])
RATE = np.array([0.1, 0.2, 0.3])


def model(ambient_c=25.0, **overrides):
    settings = dict(load_drone_config().data["sensors"]["imu"]["temperature"])
    settings.update(enabled=True, **overrides)
    return ImuTemperatureModel(settings, ambient_c)


def test_disabled_by_default_is_passthrough():
    settings = load_drone_config().data["sensors"]["imu"]["temperature"]
    imu = ImuTemperatureModel(settings, 0.0)
    assert not imu.enabled
    force, rate = imu.apply(HOVER, RATE, 10**12)
    np.testing.assert_array_equal(force, HOVER)
    np.testing.assert_array_equal(rate, RATE)


def test_warmup_follows_first_order_rise_from_first_sample():
    imu = model(ambient_c=5.0, warmup_rise_c=10.0, warmup_time_constant_s=100.0)
    imu.apply(HOVER, RATE, 7 * 10**9)  # clock starts at the first sample
    assert imu.temperature_c == pytest.approx(5.0)
    imu.apply(HOVER, RATE, 107 * 10**9)
    assert imu.temperature_c == pytest.approx(5.0 + 10.0 * (1 - math.exp(-1)))


def test_bias_is_coefficient_times_offset_from_calibration():
    imu = model(
        ambient_c=5.0,
        warmup_rise_c=0.0,
        calibration_c=25.0,
        accel_bias_per_c=[0.01, 0.0, -0.005],
        gyro_bias_per_c=[0.0, 2e-4, 0.0],
    )
    force, rate = imu.apply(HOVER, RATE, 0)
    np.testing.assert_allclose(force, HOVER + [-0.2, 0.0, 0.1])
    np.testing.assert_allclose(rate, RATE + [0.0, -0.004, 0.0])


def test_ambient_step_moves_die_temperature_at_once():
    imu = model(ambient_c=25.0, warmup_rise_c=10.0, warmup_time_constant_s=100.0)
    imu.apply(HOVER, RATE, 0)
    imu.apply(HOVER, RATE, 100 * 10**9)
    warm = imu.temperature_c
    imu.set_ambient(5.0)
    assert imu.temperature_c == pytest.approx(warm - 20.0)
    imu.apply(HOVER, RATE, 100 * 10**9)
    assert imu.temperature_c == pytest.approx(warm - 20.0)
