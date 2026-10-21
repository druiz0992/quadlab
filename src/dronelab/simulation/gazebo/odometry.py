"""Correct Gazebo OdometryPublisher velocities during its start-up window.

Measured with gz-sim 8: for the first 20 ms of simulation time the publisher
reports velocity as if the vehicle had started at the world origin, e.g. a
drone spawned at rest 1 m up reports +500, +167, +100, +71, +55 m/s in its
first five 250 Hz samples (z / (t − 2 ms)), then correct values. Controllers
and drag models fed those numbers react violently (body drag reached 10⁷ N
and crashed Gazebo's physics).

During ``warmup_s`` after the first sample this adapter replaces the
reported linear and angular velocity with finite differences of consecutive
poses; the very first sample, which has no predecessor, uses zero (vehicles
spawn at rest). Afterwards the reported values pass through unchanged.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

WARMUP_S = 0.025  # Measured bad window is 20 ms; margin for timestamp grids.


class OdometryVelocityFix:
    """Stateful corrector for (world velocity, body rates) from odometry."""

    def __init__(self, warmup_s: float = WARMUP_S) -> None:
        self._warmup_ns = round(warmup_s * 1e9)
        self._first_ns: int | None = None
        self._previous: tuple[int, NDArray[np.float64], Rotation] | None = None
        self.corrected_samples = 0

    def correct(
        self,
        sim_time_ns: int,
        position_m: ArrayLike,
        quaternion_xyzw: ArrayLike,
        velocity_world_mps: ArrayLike,
        body_rates_rad_s: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (world velocity, body rates) to use for this sample."""
        position = np.asarray(position_m, dtype=float)
        attitude = Rotation.from_quat(quaternion_xyzw)
        previous, self._previous = self._previous, (sim_time_ns, position, attitude)
        if self._first_ns is None:
            self._first_ns = sim_time_ns
        if sim_time_ns - self._first_ns > self._warmup_ns:
            return np.asarray(velocity_world_mps, float), np.asarray(
                body_rates_rad_s, float
            )
        self.corrected_samples += 1
        if previous is None or sim_time_ns <= previous[0]:
            return np.zeros(3), np.zeros(3)
        dt = (sim_time_ns - previous[0]) * 1e-9
        velocity = (position - previous[1]) / dt
        # Body-axis rate from the relative rotation previous -> current.
        rates = (previous[2].inv() * attitude).as_rotvec() / dt
        return velocity, rates
