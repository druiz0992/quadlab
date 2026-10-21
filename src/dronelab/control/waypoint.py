"""Speed-limited position feedback expressed as body-frame velocity."""

import numpy as np
from scipy.spatial.transform import Rotation


def waypoint_velocity(position, quaternion, target, max_speed=1.0):
    velocity = 0.8 * (np.asarray(target) - position)
    velocity *= min(1.0, max_speed / max(np.linalg.norm(velocity), 1e-12))
    return Rotation.from_quat(quaternion).inv().apply(velocity)
