import numpy as np

from robotlab.control.waypoint import waypoint_velocity


def test_velocity_body_frame_and_speed_limit():
    from scipy.spatial.transform import Rotation

    q = Rotation.from_euler("z", 90, degrees=True).as_quat()
    v = waypoint_velocity(np.zeros(3), q, [10, 0, 0])
    np.testing.assert_allclose(v, [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(waypoint_velocity(np.ones(3), q, [1, 1, 1]), 0)
