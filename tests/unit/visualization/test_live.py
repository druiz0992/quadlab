import numpy as np

from dronelab.simulation.state import initial_state


def test_drone_geometry_rotates_with_attitude():
    from scipy.spatial.transform import Rotation
    from dronelab.visualization.live import drone_geometry

    state = initial_state()
    state["q"] = Rotation.from_euler("z", 90, degrees=True).as_quat()
    points = drone_geometry(state)
    np.testing.assert_allclose(points[5] - points[4], [0.0, 0.28, 0.0], atol=1e-12)
