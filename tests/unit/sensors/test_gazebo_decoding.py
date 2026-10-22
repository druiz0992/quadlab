from types import SimpleNamespace

import numpy as np

from robotlab.sensors.gazebo import lidar_points, rgb_image


def test_rgb_rows_exclude_padding():
    message = SimpleNamespace(
        height=2, width=1, step=4, data=bytes([255, 0, 0, 99, 0, 255, 0, 99])
    )
    np.testing.assert_array_equal(rgb_image(message), [[[255, 0, 0]], [[0, 255, 0]]])


def test_lidar_returns_valid_points_in_body_frame():
    message = SimpleNamespace(
        ranges=[2.0, 3.0, float("inf"), 0.05],
        angle_min=0.0,
        angle_step=np.pi / 2,
        range_min=0.15,
        range_max=50.0,
    )
    np.testing.assert_allclose(
        lidar_points(message), [[2.0, 0.0, 0.2], [0.0, 3.0, 0.2]], atol=1e-12
    )


def test_lidar_custom_mount_rotates_then_translates_points():
    message = SimpleNamespace(
        ranges=[2.0], angle_min=0.0, angle_step=1.0, range_min=0.15, range_max=50.0
    )
    np.testing.assert_allclose(
        lidar_points(
            message, position_m=[0.3, -0.1, 0.4], rotation_rad=[0, 0, np.pi / 2]
        ),
        [[0.3, 1.9, 0.4]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        lidar_points(message, position_m=[0, 0, 1], rotation_rad=[0, np.pi / 2, 0]),
        [[0, 0, -1]],
        atol=1e-12,
    )
    message.ranges = [float("inf")]
    assert lidar_points(message, rotation_rad=[1, 2, 3]).shape == (0, 3)
