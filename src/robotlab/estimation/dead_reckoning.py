"""IMU dead reckoning baseline, initialized with the true initial state.

No bias correction or position updates: this intentionally exhibits drift.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from ..core.frames import GRAVITY


class ImuDeadReckoning:
    def __init__(self, state):
        self.position = state["x"].copy()
        self.velocity = state["v"].copy()
        self.rotation = Rotation.from_quat(state["q"])

    def step(self, accel, gyro, dt):
        world_accel = self.rotation.apply(accel) + GRAVITY
        self.position += self.velocity * dt + 0.5 * world_accel * dt**2
        self.velocity += world_accel * dt
        self.rotation = self.rotation * Rotation.from_rotvec(np.asarray(gyro) * dt)
