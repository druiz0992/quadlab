"""Example wheel-odometry filter: driven by wheel encoders, corrected by GPS.

Select it with::

    "algorithms": {"estimator": "filter",
                   "estimator_options": {
                       "filter": "robotlab.estimation.examples.wheel_odometry_filter:WheelOdometryFilter",
                       "filter_options": {"initial_yaw_deg": 90.0}}}

with ``sensors.wheel_encoders`` and ``sensors.gps`` enabled on a wheeled
robot. Like the IMU example (imu_gps_filter.py) it holds only the mathematics;
the node (node.py) does the timing and converts each encoder reading into
a ``WheelMotion`` (hub positions, rolling directions, rolling speeds in
body x/y, from the context's nominal wheel geometry). Deliberately simple:

- motion: planar rigid body, body velocity u = (v_x, v_y, ω_z). Wheel i at
  hub r_i = (x_i, y_i) moves at (v_x − ω y_i, v_y + ω x_i); along its
  rolling direction d_i that equals the measured rolling speed s_i, and
  across it (n_i ⊥ d_i) it would be zero if the tyre did not slip. Stacking
  the rows gives a linear least-squares problem for u. Lateral rows are
  weighted by ``lateral_weight``, 0 by default: real tyres run at slip
  angles (unsteered rear wheels slip sideways in every turn), so only the
  rolling rows are trusted, and the minimum-norm solution sets a v_y they
  cannot observe (no steered wheels, e.g. skid steering) to zero. Kinematic
  odometry, e.g. Siegwart et al., *Introduction to Autonomous Mobile
  Robots*, 2nd ed., §3.2;
- state x = [east, north, up, yaw] (m, rad), propagated with the midpoint
  yaw: p += R(ψ + ω dt/2) (v_x, v_y) dt, ψ += ω dt; up only random-walks
  (``height_walk_m``, per sample), since the robot is assumed on level
  ground (roll = pitch = 0);
- process noise: white body-velocity and yaw-rate errors per encoder sample
  (``speed_std_mps``, ``yaw_rate_std_rad_s``), Q = G diag(σ²) Gᵀ;
- ``update_gps``: antenna position z = p + R_z(ψ) l (lever arm l in body
  axes), gated at ``gps_gate_squared``; GPS makes yaw observable once the
  robot moves.

It reports position, velocity, and a level attitude at the estimated yaw,
without covariance (no NEES; NIS is meaningful). The encoder speed
averages the interval before each sample, so the odometry lags by one
encoder period. Units: m, m/s, rad, s.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ...core.measurements import EstimatedState
from ..interface import Innovation
from ..filter.kalman import kalman_update
from ..filter.measurements import GpsFix, WheelMotion
from ..filter.models import gps_position
from ..filter.state_layout import StateLayout

DEFAULTS: dict[str, Any] = dict(
    initial_position_m=[0.0, 0.0, 0.0],
    initial_yaw_deg=0.0,
    initial_position_std_m=1.0,
    initial_yaw_std_deg=5.0,
    speed_std_mps=0.05,
    yaw_rate_std_rad_s=0.02,
    height_walk_m=0.001,
    lateral_weight=0.0,
    gps_gate_squared=25.0,
)
# The state x = [east, north, up, yaw]; the shared GPS model
# (filter/models.py) finds the position and yaw columns by name.
LAYOUT = StateLayout.of("position", "yaw")


def body_velocity(motion: WheelMotion, lateral_weight: float) -> NDArray[np.float64]:
    """Least-squares planar body velocity (v_x, v_y, ω_z) from the wheels'
    rolling and no-lateral-slip constraints (module docs)."""
    x, y = motion.hub_positions_body_m.T
    d = motion.rolling_direction_body
    n = np.column_stack([-d[:, 1], d[:, 0]])
    # Row for direction e: e · (v_x − ω y, v_y + ω x) = e_x v_x + e_y v_y + (e_y x − e_x y) ω.
    rolling = np.column_stack([d[:, 0], d[:, 1], d[:, 1] * x - d[:, 0] * y])
    lateral = np.column_stack([n[:, 0], n[:, 1], n[:, 1] * x - n[:, 0] * y])
    A = np.vstack([rolling, lateral_weight * lateral])
    b = np.concatenate([motion.rolling_speed_mps, np.zeros(len(x))])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _yaw_matrix(yaw_rad: float) -> NDArray[np.float64]:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s], [s, c]])


class WheelOdometryFilter:
    """Wheel odometry with GPS corrections (module docs)."""

    def __init__(self, options: Mapping[str, Any]) -> None:
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"WheelOdometryFilter: unknown options {sorted(unknown)}")
        o = {**DEFAULTS, **options}
        self.x = np.zeros(LAYOUT.size)
        self.x[:3] = np.asarray(o["initial_position_m"], dtype=float)
        self.x[3] = np.deg2rad(float(o["initial_yaw_deg"]))
        self.P = np.diag(
            [float(o["initial_position_std_m"]) ** 2] * 3
            + [np.deg2rad(float(o["initial_yaw_std_deg"])) ** 2]
        )
        self.noise_std = np.array(
            [o["speed_std_mps"], o["speed_std_mps"], o["yaw_rate_std_rad_s"]], float
        )
        self.height_walk_m = float(o["height_walk_m"])
        self.lateral_weight = float(o["lateral_weight"])
        self.gps_gate_squared = float(o["gps_gate_squared"])
        if np.any(self.noise_std <= 0) or self.lateral_weight < 0:
            raise ValueError("Noise must be positive and lateral_weight nonnegative")
        self.velocity_enu_mps = np.zeros(3)

    def predict_wheel_encoders(self, motion: WheelMotion, dt_s: float) -> None:
        """Propagate ``dt_s`` seconds with one held encoder reading."""
        vx, vy, omega = body_velocity(motion, self.lateral_weight)
        yaw_mid = self.x[3] + 0.5 * omega * dt_s
        R = _yaw_matrix(yaw_mid)
        velocity = R @ (vx, vy)
        F = np.eye(4)
        # d(position)/d(yaw) of R(ψ) v dt: the 90°-rotated displacement.
        F[:2, 3] = np.array([-velocity[1], velocity[0]]) * dt_s
        G = np.zeros((4, 3))
        G[:2, :2] = R * dt_s
        G[:2, 2] = F[:2, 3] * 0.5 * dt_s  # yaw-rate error moves the midpoint yaw
        G[3, 2] = dt_s
        Q = (G * self.noise_std**2) @ G.T
        Q[2, 2] += self.height_walk_m**2
        self.x[:2] += velocity * dt_s
        self.x[3] += omega * dt_s
        self.P = F @ self.P @ F.T + Q
        self.velocity_enu_mps = np.array([velocity[0], velocity[1], 0.0])

    def update_gps(self, fix: GpsFix) -> Innovation:
        """Gated Kalman update with the antenna position (filter/kalman.py)."""
        rotation = np.eye(3)
        rotation[:2, :2] = _yaw_matrix(self.x[3])
        measurement = gps_position(
            LAYOUT, fix, position_m=self.x[:3], rotation_matrix=rotation
        )
        correction = kalman_update(self.P, measurement, self.gps_gate_squared)
        if correction.delta is not None:
            self.x = self.x + correction.delta
            self.P = correction.covariance
        return correction.innovation

    def state(self) -> EstimatedState:
        return EstimatedState(
            position_m=self.x[:3],
            velocity_mps=self.velocity_enu_mps,
            quaternion_xyzw=Rotation.from_euler("z", self.x[3]).as_quat(),
        )
