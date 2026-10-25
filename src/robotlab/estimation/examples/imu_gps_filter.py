"""Example IMU-driven filter (``RecursiveFilter``) with GPS and barometer updates.

Select it with::

    "algorithms": {"estimator": "filter",
                   "estimator_options": {
                       "filter": "robotlab.estimation.examples.imu_gps_filter:ImuGpsFilter",
                       "filter_options": {"accel_std_mps2": 0.2}}}

and test it with ``robotlab.testing.plugins.check_estimator(FilterEstimator,
options)``. The node (node.py) does all timing, frames, and conversion: this
class only holds the mathematics, deliberately simple so the structure of
a ``RecursiveFilter`` (protocol.py) stays visible:

- attitude R: integrated from the gyro, R ← R · Exp(ω dt), never corrected
  (a real filter estimates attitude error and gyro bias);
- position p and velocity v (local ENU): a linear Kalman filter driven by
  the world acceleration a = R f + g, with white acceleration noise of
  ``accel_std_mps2`` per IMU sample (process noise Q = G G^T σ², with
  G = [dt²/2, dt]);
- ``update_gps``: antenna position z = p + R l (lever arm l in body axes),
  gated at ``gps_gate_squared``;
- ``update_baro``: altitude z = p_up, gated at ``baro_gate_squared``.

It reports position, velocity, and attitude; the covariance is None
because attitude uncertainty is not tracked (no NEES; NIS is meaningful).
Units: m, m/s, s, rad.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ...core.frames import GRAVITY
from ...core.measurements import EstimatedState
from ..interface import Innovation
from ..filter.measurements import BaroAltitude, GpsFix, ImuInput

DEFAULTS: dict[str, Any] = dict(
    initial_position_m=[0.0, 0.0, 0.0],
    initial_position_std_m=5.0,
    initial_velocity_std_mps=1.0,
    accel_std_mps2=0.2,
    gps_gate_squared=25.0,
    baro_gate_squared=25.0,
)


class ImuGpsFilter:
    """Position/velocity KF with gyro-integrated attitude (module docs)."""

    def __init__(self, options: Mapping[str, Any]) -> None:
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"ImuGpsFilter: unknown options {sorted(unknown)}")
        o = {**DEFAULTS, **options}
        self.x = np.zeros(6)  # [p (3), v (3)], ENU
        self.x[:3] = np.asarray(o["initial_position_m"], dtype=float)
        self.P = np.diag(
            [o["initial_position_std_m"] ** 2] * 3
            + [o["initial_velocity_std_mps"] ** 2] * 3
        )
        self.rotation = Rotation.identity()
        self.accel_std = float(o["accel_std_mps2"])
        self.gates = dict(
            gps=float(o["gps_gate_squared"]), baro=float(o["baro_gate_squared"])
        )

    def predict_imu(self, imu: ImuInput, dt_s: float) -> None:
        """Propagate ``dt_s`` seconds with one held IMU reading (body axes)."""
        accel = self.rotation.apply(imu.specific_force_mps2) + GRAVITY
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt_s
        G = np.vstack([np.eye(3) * dt_s**2 / 2, np.eye(3) * dt_s])
        self.x = F @ self.x + G @ accel
        self.P = F @ self.P @ F.T + G @ G.T * self.accel_std**2
        self.rotation = self.rotation * Rotation.from_rotvec(
            imu.angular_rate_rad_s * dt_s
        )

    def update_gps(self, fix: GpsFix) -> Innovation:
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        predicted = self.x[:3] + self.rotation.apply(fix.lever_arm_body_m)
        return self._correct(
            "gps", fix.position_enu_m - predicted, H, fix.covariance_m2
        )

    def update_baro(self, baro: BaroAltitude) -> Innovation:
        H = np.zeros((1, 6))
        H[0, 2] = 1.0
        residual = np.array([baro.altitude_m - self.x[2]])
        return self._correct("baro", residual, H, np.array([[baro.variance_m2]]))

    def _correct(
        self,
        name: str,
        residual: NDArray[np.float64],
        H: NDArray[np.float64],
        R: NDArray[np.float64],
    ) -> Innovation:
        """Gated Kalman update (Joseph form); returns the innovation."""
        S = H @ self.P @ H.T + R
        nis = float(residual @ np.linalg.solve(S, residual))
        accepted = nis <= self.gates[name]
        if accepted:
            K = np.linalg.solve(S, H @ self.P).T
            self.x = self.x + K @ residual
            A = np.eye(6) - K @ H
            self.P = A @ self.P @ A.T + K @ R @ K.T
        return Innovation(name, nis, len(residual), accepted)

    def state(self) -> EstimatedState:
        return EstimatedState(
            position_m=self.x[:3],
            velocity_mps=self.x[3:],
            quaternion_xyzw=self.rotation.as_quat(),
        )
