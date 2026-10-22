import numpy as np
from scipy.spatial.transform import Rotation

from robotlab.simulation.gazebo.odometry import OdometryVelocityFix

MS = 1_000_000
LEVEL = [0, 0, 0, 1]


def test_startup_window_uses_pose_differences_then_passes_through():
    fix = OdometryVelocityFix()
    # Gazebo-like garbage: z / (t - 2 ms) for a drone falling from 1 m.
    samples = [
        (4, 0.9999, 499.94),
        (8, 0.9996, 166.6),
        (24, 0.9969, -0.15),
        (32, 0.9947, -0.23),
    ]
    out = [
        fix.correct(t * MS, [0, 0, z], LEVEL, [0, 0, vz], [0, 0, 0])
        for t, z, vz in samples
    ]
    np.testing.assert_array_equal(out[0][0], 0)  # first sample: at rest
    np.testing.assert_allclose(out[1][0], [0, 0, (0.9996 - 0.9999) / 0.004])
    np.testing.assert_allclose(
        out[2][0], [0, 0, (0.9969 - 0.9996) / 0.016]
    )  # still in window
    # 32 ms is more than 25 ms after the first sample (4 ms): reported value.
    np.testing.assert_allclose(out[3][0], [0, 0, -0.23])
    assert fix.corrected_samples == 3


def test_body_rates_from_relative_rotation():
    fix = OdometryVelocityFix()
    fix.correct(0, [0, 0, 1], LEVEL, [0, 0, 0], [9, 9, 9])
    yawed = Rotation.from_euler(
        "z", 0.004, degrees=False
    ).as_quat()  # 1 rad/s over 4 ms
    _, rates = fix.correct(4 * MS, [0, 0, 1], yawed, [0, 0, 0], [9, 9, 9])
    np.testing.assert_allclose(rates, [0, 0, 1.0], atol=1e-9)
