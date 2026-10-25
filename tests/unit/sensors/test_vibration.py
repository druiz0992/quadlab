from types import SimpleNamespace as NS

import numpy as np
import pytest

from robotlab.sensors.gazebo import vibrate_imu_message
from robotlab.sensors.imu.vibration import AccelerometerVibration
from robotlab.simulation.gazebo.config import load_drone_config


def settings(**overrides):
    result = load_drone_config().data["sensors"]["imu"]["vibration"]
    result.update(overrides)
    return result


def test_disabled_is_passthrough_and_does_not_modify_input():
    model = AccelerometerVibration(settings())
    force = np.array([1.0, 2.0, 9.81])
    np.testing.assert_array_equal(model.apply(force, 0), force)
    assert model.summary()["samples"] == 0
    message = NS(linear_acceleration=NS(x=1, y=2, z=9.81))
    assert vibrate_imu_message(message, model, 0) is message


def test_periodic_amplitude_phase_and_specific_force_clipping():
    model = AccelerometerVibration(
        settings(
            enabled=True,
            frequency_hz=1,
            accel_amplitude_mps2=[2, 3, 15],
            phase_deg=[0, 90, 0],
            clip_limit_mps2=19.62,
        )
    )
    np.testing.assert_allclose(model.apply([0, 0, 9.81], 0), [0, 3, 9.81])
    np.testing.assert_allclose(
        model.apply([0, 0, 9.81], 250_000_000), [2, 0, 19.62], atol=1e-12
    )
    np.testing.assert_allclose(
        model.apply([0, 0, 9.81], 750_000_000), [-2, 0, -5.19], atol=1e-12
    )
    summary = model.summary()
    assert summary["clipped_samples"] == 1
    assert summary["clipped_axes_xyz"] == [0, 0, 1]
    assert summary["peak_before_clip_mps2"][2] == pytest.approx(24.81)


def test_clipping_can_bias_average_accelerometer_measurement():
    model = AccelerometerVibration(
        settings(
            enabled=True,
            frequency_hz=10,
            accel_amplitude_mps2=[0, 0, 15],
            clip_limit_mps2=19.62,
        )
    )
    readings = [model.apply([0, 0, 9.81], i * 1_000_000)[2] for i in range(1000)]
    assert np.mean(readings) < 9.0
    assert model.summary()["clipped_samples"] > 0


def test_seeded_correlated_noise_has_configured_stationary_scale():
    config = settings(
        enabled=True, random_std_mps2=[1, 2, 3], random_correlation_s=0.02, seed=4
    )
    first, second = AccelerometerVibration(config), AccelerometerVibration(config)
    before = np.random.get_state()
    readings = np.array([first.apply([0, 0, 0], i * 5_000_000) for i in range(10000)])
    for i in range(10):
        np.testing.assert_array_equal(
            second.apply([0, 0, 0], i * 5_000_000), readings[i]
        )
    np.testing.assert_allclose(readings.std(axis=0), [1, 2, 3], rtol=0.12)
    assert np.corrcoef(readings[:-1, 0], readings[1:, 0])[0, 1] == pytest.approx(
        np.exp(-0.005 / 0.02), abs=0.04
    )
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_message_adapter_preserves_gyro_header_and_source_message():
    model = AccelerometerVibration(
        settings(enabled=True, accel_amplitude_mps2=[2, 0, 0], phase_deg=[90, 0, 0])
    )
    source = NS(
        header=NS(stamp=NS(sec=3, nsec=42)),
        linear_acceleration=NS(x=0, y=0, z=9.81),
        angular_velocity=NS(x=0.1, y=0.2, z=0.3),
    )
    result = vibrate_imu_message(source, model, 0)
    assert source.linear_acceleration.x == 0
    assert result.linear_acceleration.x == 2
    assert result.header == source.header
    assert result.angular_velocity == source.angular_velocity
    with pytest.raises(ValueError, match="strictly increasing"):
        model.apply([0, 0, 0], 0)
