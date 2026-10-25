"""Initial rigid-body state dict (position, velocity, attitude, rates) used
by numerical tests of the sensor models and dead reckoning."""

import numpy as np


def initial_state():
    return dict(
        x=np.array([0.0, 0.0, 1.0]),
        v=np.zeros(3),
        q=np.array([0.0, 0.0, 0.0, 1.0]),
        w=np.zeros(3),
        wind=np.zeros(3),
        rotor_speeds=np.zeros(4),
    )
