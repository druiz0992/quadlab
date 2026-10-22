import numpy as np

from scipy.spatial.transform import Rotation

from robotlab.estimation.dead_reckoning import ImuDeadReckoning
from robotlab.simulation.state import initial_state


def test_ideal_hover_estimator():
    state = initial_state()
    state["q"] = Rotation.from_euler("xyz", [20, -10, 30], degrees=True).as_quat()
    estimator = ImuDeadReckoning(state)
    force = Rotation.from_quat(state["q"]).inv().apply([0, 0, 9.81])
    for _ in range(100):
        estimator.step(force, np.zeros(3), 0.01)
    np.testing.assert_allclose(estimator.position, state["x"], atol=1e-10)
    np.testing.assert_allclose(estimator.velocity, 0, atol=1e-10)
