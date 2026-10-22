"""Sample noise standard deviations are per sample, not noise densities."""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from ..core.frames import GRAVITY


@dataclass
class SensorConfig:
    accel_std: float = 0.08  # m/s² per sample
    gyro_std: float = 0.003  # rad/s per sample
    position_std: float = 0.03  # m per sample
    accel_bias: np.ndarray = field(
        default_factory=lambda: np.array([0.03, -0.02, 0.04])
    )
    gyro_bias: np.ndarray = field(
        default_factory=lambda: np.array([0.001, -0.001, 0.002])
    )


class Sensors:
    def __init__(self, seed=7, config=None):
        self.config = config or SensorConfig()
        self.rng = np.random.default_rng(seed)

    def imu(self, state, acceleration):
        c = self.config
        specific_force = (
            Rotation.from_quat(state["q"]).inv().apply(acceleration - GRAVITY)
        )
        accel = specific_force + c.accel_bias + self.rng.normal(0.0, c.accel_std, 3)
        gyro = state["w"] + c.gyro_bias + self.rng.normal(0.0, c.gyro_std, 3)
        return accel, gyro

    def position(self, state):
        return state["x"] + self.rng.normal(0.0, self.config.position_std, 3)
