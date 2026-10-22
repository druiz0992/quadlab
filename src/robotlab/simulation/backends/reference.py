"""Small rigid-body teaching model with ideal actuation."""

import numpy as np
from scipy.spatial.transform import Rotation

from ...core.frames import GRAVITY
from ..state import initial_state


class ReferenceBackend:
    """Simplified rigid body with ideal collective thrust and torque actuation.

    This is a teaching reference, not the same vehicle/controller as RotorPy.
    No contact dynamics or aerodynamic model is included.
    """

    def __init__(self):
        self.state = initial_state()
        self.mass = 0.5
        self.inertia = np.array([0.003, 0.003, 0.005])

    def step(self, t, dt, target):
        s = self.state
        rotation = Rotation.from_quat(s["q"])
        R = rotation.as_matrix()
        desired_acc = 3.0 * (target - s["x"]) - 3.0 * s["v"] - GRAVITY
        z = desired_acc / np.linalg.norm(desired_acc)
        y = np.cross(z, np.array([1.0, 0.0, 0.0]))
        y /= np.linalg.norm(y)
        desired_R = np.column_stack((np.cross(y, z), y, z))
        error_matrix = 0.5 * (desired_R.T @ R - R.T @ desired_R)
        error = np.array([error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0]])
        torque = -0.12 * error - 0.025 * s["w"]
        thrust = np.clip(self.mass * np.dot(desired_acc, R[:, 2]), 0.0, 15.0)
        acceleration = GRAVITY + thrust / self.mass * R[:, 2]
        angular_acc = (torque - np.cross(s["w"], self.inertia * s["w"])) / self.inertia
        self.state = dict(
            x=s["x"] + s["v"] * dt + 0.5 * acceleration * dt**2,
            v=s["v"] + acceleration * dt,
            q=(rotation * Rotation.from_rotvec(s["w"] * dt)).as_quat(),
            w=s["w"] + angular_acc * dt,
            wind=np.zeros(3),
            rotor_speeds=np.zeros(4),
        )
        return acceleration
