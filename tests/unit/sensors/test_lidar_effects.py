from types import SimpleNamespace as NS

import numpy as np
import pytest

from robotlab.sensors.imaging.lidar_effects import LidarEffects, sensor_frame_scan
from robotlab.simulation.gazebo.config import load_drone_config


def lidar(**effects):
    settings = load_drone_config().data["sensors"]["lidar"]
    settings["enabled"] = True
    settings["effects"].update(effects)
    return settings


def wall(n=4000, distance=10.0):
    """Points on a wall 10 m ahead, spread over the sweep."""
    y = np.linspace(-5, 5, n)
    return np.column_stack((np.full(n, distance), y, np.zeros(n))), np.linspace(
        0, 1, n, endpoint=False
    )


def test_disabled_without_effects():
    assert not LidarEffects(lidar()).enabled


def test_range_noise_grows_with_distance():
    effects = LidarEffects(lidar(noise_std_m=0.02, noise_per_m=0.001, seed=1))
    points, fraction = wall(20000, 10.0)
    out = effects.process(points, fraction, [0, 0, 0], [0, 0, 0])
    ranges = np.linalg.norm(points, axis=1)
    error = np.linalg.norm(out, axis=1) - ranges
    assert error.std() == pytest.approx(0.02 + 0.001 * ranges.mean(), rel=0.05)
    # Noise is along the ray: directions unchanged.
    np.testing.assert_allclose(
        out / np.linalg.norm(out, axis=1)[:, None], points / ranges[:, None], atol=1e-12
    )


def test_dropouts_uniform_and_far():
    uniform = LidarEffects(lidar(dropout_probability=0.1, seed=2))
    points, fraction = wall(20000)
    kept = len(uniform.process(points, fraction, [0, 0, 0], [0, 0, 0])) / 20000
    assert kept == pytest.approx(0.9, abs=0.01)
    far = LidarEffects(lidar(far_dropout_exponent=2.0, seed=3))  # max range 50 m
    near_points, near_fraction = wall(20000, 5.0)
    far_points, far_fraction = wall(20000, 45.0)
    near_kept = (
        len(far.process(near_points, near_fraction, [0, 0, 0], [0, 0, 0])) / 20000
    )
    far_kept = len(far.process(far_points, far_fraction, [0, 0, 0], [0, 0, 0])) / 20000
    assert near_kept > 0.98 and far_kept < 0.3


def test_motion_distortion_matches_rotation_and_translation_over_the_sweep():
    effects = LidarEffects(lidar(scan_time_s=0.1))
    points = np.array([[10.0, 0, 0], [10.0, 0, 0], [10.0, 0, 0]])
    fraction = np.array([0.0, 0.5, 0.999999])
    # Yawing at 1 rad/s: the first point (50 ms early) appears rotated +0.05 rad.
    out = effects.process(points, fraction, [0, 0, 0], [0, 0, 1.0])
    angles = np.arctan2(out[:, 1], out[:, 0])
    np.testing.assert_allclose(angles, [0.05, 0.0, -0.05], atol=1e-6)
    # Moving forward at 2 m/s: early points appear 0.1 m further away.
    out = effects.process(points, fraction, [2.0, 0, 0], [0, 0, 0])
    np.testing.assert_allclose(out[:, 0], [10.1, 10.0, 9.9], atol=1e-6)


def test_sensor_frame_scan_layout_and_sweep_fraction():
    message = NS(
        ranges=[2.0, 2.0, float("inf"), 2.0],
        angle_min=0.0,
        angle_step=np.pi / 2,
        count=2,
        vertical_count=2,
        vertical_angle_min=0.0,
        vertical_angle_step=0.1,
        range_min=0.1,
        range_max=30.0,
    )
    points, fraction = sensor_frame_scan(message)
    assert len(points) == 3  # the inf return is dropped
    np.testing.assert_allclose(fraction, [0.0, 0.5, 0.5])


def test_livox_example_loads():
    data = load_drone_config("configs/sensors/lidar_livox_mid360.json").data["sensors"][
        "lidar"
    ]
    assert (data["vertical_min_angle_deg"], data["vertical_max_angle_deg"]) == (-7, 52)
    assert LidarEffects(data).enabled
