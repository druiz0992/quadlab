from types import SimpleNamespace as NS

import numpy as np
import pytest

from robotlab.sensors.gazebo import imu_errors_message
from robotlab.sensors.imu.errors import ImuErrorModel
from robotlab.simulation.gazebo.config import load_drone_config

RATE_HZ = 200.0
PERIOD_NS = 5_000_000


def settings(**overrides):
    result = load_drone_config().data["sensors"]["imu"]["error_model"]
    result.update({"enabled": True, **overrides})
    return result


def run(model, samples, force=(0, 0, 9.81), rate=(0, 0, 0)):
    """Return stacked (accel errors, gyro errors) for evenly spaced samples."""
    accel, gyro = [], []
    for i in range(samples):
        a, g = model.apply(force, rate, i * PERIOD_NS)
        accel.append(a - force)
        gyro.append(g - rate)
    return np.array(accel), np.array(gyro)


def test_disabled_is_passthrough_and_draws_nothing():
    model = ImuErrorModel(
        settings(enabled=False, accel_noise_density_mps2_per_sqrt_hz=[1, 1, 1]), RATE_HZ
    )
    force, rate = model.apply([1, 2, 3], [0.1, 0.2, 0.3], 0)
    np.testing.assert_array_equal(force, [1, 2, 3])
    np.testing.assert_array_equal(rate, [0.1, 0.2, 0.3])
    assert model.summary()["samples"] == 0
    message = NS(
        linear_acceleration=NS(x=1, y=2, z=3), angular_velocity=NS(x=0, y=0, z=0)
    )
    assert imu_errors_message(message, model, 0) is message


def test_white_noise_std_is_density_times_sqrt_rate():
    model = ImuErrorModel(
        settings(
            accel_noise_density_mps2_per_sqrt_hz=[2e-3, 4e-3, 6e-3],
            gyro_noise_density_rad_s_per_sqrt_hz=[1e-4, 2e-4, 3e-4],
        ),
        RATE_HZ,
    )
    accel, gyro = run(model, 20000)
    np.testing.assert_allclose(
        accel.std(axis=0), np.array([2e-3, 4e-3, 6e-3]) * np.sqrt(RATE_HZ), rtol=0.03
    )
    np.testing.assert_allclose(
        gyro.std(axis=0), np.array([1e-4, 2e-4, 3e-4]) * np.sqrt(RATE_HZ), rtol=0.03
    )
    # White noise has no sample-to-sample correlation and zero mean.
    assert abs(np.corrcoef(accel[:-1, 0], accel[1:, 0])[0, 1]) < 0.03
    np.testing.assert_allclose(accel.mean(axis=0), 0, atol=3e-3)


def test_bias_walk_variance_grows_linearly_with_time():
    walk = 3e-3
    duration_s, samples = 4.0, int(4.0 * RATE_HZ)
    finals = []
    for seed in range(400):
        model = ImuErrorModel(
            settings(seed=seed, accel_bias_random_walk_mps2_per_sqrt_s=[walk] * 3),
            RATE_HZ,
        )
        accel, _ = run(model, samples)
        finals.append(accel[-1])
    elapsed_s = (samples - 1) * PERIOD_NS * 1e-9
    # Matches the EKF convention: increment variance walk² · dt.
    np.testing.assert_allclose(
        np.std(finals, axis=0), walk * np.sqrt(elapsed_s), rtol=0.12
    )
    assert elapsed_s == pytest.approx(duration_s - 1 / RATE_HZ)


def test_turn_on_bias_is_constant_and_seeded():
    config = settings(
        seed=5,
        accel_turn_on_bias_std_mps2=[0.05] * 3,
        gyro_turn_on_bias_std_rad_s=[0.003] * 3,
    )
    first, second = ImuErrorModel(config, RATE_HZ), ImuErrorModel(config, RATE_HZ)
    accel, gyro = run(first, 50)
    np.testing.assert_allclose(accel - accel[0], 0, atol=1e-15)
    np.testing.assert_allclose(gyro - gyro[0], 0, atol=1e-15)
    np.testing.assert_allclose(
        first.summary()["accel_turn_on_bias_mps2"], accel[0], atol=1e-14
    )
    np.testing.assert_allclose(run(second, 1)[1][0], gyro[0], atol=1e-14)
    other = ImuErrorModel(
        settings(seed=6, accel_turn_on_bias_std_mps2=[0.05] * 3), RATE_HZ
    )
    assert not np.allclose(run(other, 1)[0][0], accel[0])


def test_local_rng_leaves_global_state_and_rejects_reordered_time():
    before = np.random.get_state()
    model = ImuErrorModel(
        settings(accel_noise_density_mps2_per_sqrt_hz=[1e-3] * 3), RATE_HZ
    )
    run(model, 10)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    with pytest.raises(ValueError, match="strictly increasing"):
        model.apply([0, 0, 0], [0, 0, 0], 0)


def test_message_adapter_changes_only_imu_vectors():
    model = ImuErrorModel(
        settings(gyro_turn_on_bias_std_rad_s=[0.01] * 3, seed=3), RATE_HZ
    )
    source = NS(
        header=NS(stamp=NS(sec=1, nsec=0)),
        orientation=NS(x=0, y=0, z=0, w=1),
        linear_acceleration=NS(x=0, y=0, z=9.81),
        angular_velocity=NS(x=0, y=0, z=0),
    )
    result = imu_errors_message(source, model, 1_000_000_000)
    assert source.angular_velocity.x == 0
    assert result.angular_velocity.x != 0
    assert result.linear_acceleration.z == 9.81
    assert result.header == source.header and result.orientation == source.orientation
