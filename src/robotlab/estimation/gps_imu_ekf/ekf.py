"""Educational GPS/IMU error-state EKF, with explicit initialization.

Nominal state: body-origin position/velocity in local ENU, body-to-world
rotation, body-axis accelerometer/gyro biases. Right attitude error, 15x15
covariance. IMU values are held at interval starts. Bias random walks are
injected at interval ends. This is not a calibrated navigation system.

It is a ``RecursiveFilter`` (estimation/filter/protocol.py) driven by the
IMU: the reusable filter estimator does the sample timing, mounts, GPS
conversion, and publication. ``algorithms.estimator: gps_imu_ekf`` runs it there, built by
``GpsImuEkf.from_options`` from the ``ekf`` configuration section.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from ...core.frames import GRAVITY
from ...core.measurements import EstimatedState, readonly_array
from ..filter.measurements import GpsFix, ImuInput
from ..interface import Innovation


def skew(v: ArrayLike) -> NDArray[np.float64]:
    x, y, z = np.asarray(v, float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def right_jacobian(v: ArrayLike) -> NDArray[np.float64]:
    theta = np.linalg.norm(v)
    cross = skew(v)
    if theta < 1e-6:
        return np.eye(3) - 0.5 * cross + cross @ cross / 6
    return (
        np.eye(3)
        - (1 - np.cos(theta)) / theta**2 * cross
        + (theta - np.sin(theta)) / theta**3 * cross @ cross
    )


def covariance(value: ArrayLike, size: int) -> NDArray[np.float64]:
    result = readonly_array(value, (size, size)).copy()
    if (
        not np.allclose(result, result.T, atol=1e-12, rtol=0)
        or np.linalg.eigvalsh(result).min() <= 0
    ):
        raise ValueError("Covariance must be symmetric positive definite")
    return result


# ``ekf`` section keys the filter reads (the rest configure the converter).
OPTION_KEYS = (
    "initial_position_m",
    "initial_velocity_mps",
    "initial_rotation_deg",
    "initial_accel_bias_mps2",
    "initial_gyro_bias_rad_s",
    "initial_position_std_m",
    "initial_velocity_std_mps",
    "initial_attitude_std_deg",
    "initial_accel_bias_std_mps2",
    "initial_gyro_bias_std_rad_s",
    "accel_std_mps2",
    "gyro_std_rad_s",
    "accel_bias_walk",
    "gyro_bias_walk",
    "gps_gate_squared",
    "initialize_position_from_gps",
)


class GpsImuEkf:
    """15-state error-state EKF (module docs); build it explicitly with
    keyword arguments or from configuration with ``from_options``."""

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "GpsImuEkf":
        """The filter from declared priors (``OPTION_KEYS``, never simulator
        truth): diagonal initial covariance from the ``initial_*_std``
        values, attitude from roll/pitch/yaw degrees. Raises ValueError for
        missing or unknown keys."""
        missing = set(OPTION_KEYS) - set(options)
        unknown = set(options) - set(OPTION_KEYS)
        if missing or unknown:
            raise ValueError(
                f"GpsImuEkf options: missing {sorted(missing)}, "
                f"unknown {sorted(unknown)}"
            )
        e = options
        std = np.repeat(
            [
                e["initial_position_std_m"],
                e["initial_velocity_std_mps"],
                np.deg2rad(e["initial_attitude_std_deg"]),
                e["initial_accel_bias_std_mps2"],
                e["initial_gyro_bias_std_rad_s"],
            ],
            3,
        )
        return cls(
            position_m=e["initial_position_m"],
            velocity_mps=e["initial_velocity_mps"],
            quaternion_xyzw=Rotation.from_euler(
                "xyz", e["initial_rotation_deg"], degrees=True
            ).as_quat(),
            initial_covariance=np.diag(std**2),
            accel_bias_mps2=e["initial_accel_bias_mps2"],
            gyro_bias_rad_s=e["initial_gyro_bias_rad_s"],
            accel_std_mps2=e["accel_std_mps2"],
            gyro_std_rad_s=e["gyro_std_rad_s"],
            accel_bias_walk=e["accel_bias_walk"],
            gyro_bias_walk=e["gyro_bias_walk"],
            gps_gate_squared=e["gps_gate_squared"],
            initialize_position_from_gps=e["initialize_position_from_gps"],
        )

    def __init__(
        self,
        *,
        position_m: ArrayLike,
        velocity_mps: ArrayLike,
        quaternion_xyzw: ArrayLike,
        initial_covariance: ArrayLike,
        accel_bias_mps2: ArrayLike = (0, 0, 0),
        gyro_bias_rad_s: ArrayLike = (0, 0, 0),
        accel_std_mps2: float = 0.08,
        gyro_std_rad_s: float = 0.003,
        accel_bias_walk: float = 0.001,
        gyro_bias_walk: float = 0.0001,
        gps_gate_squared: float = 25.0,
        initialize_position_from_gps: bool = False,
    ) -> None:
        self.position = readonly_array(position_m, (3,)).copy()
        self.velocity = readonly_array(velocity_mps, (3,)).copy()
        quaternion = readonly_array(quaternion_xyzw, (4,))
        if not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-6):
            raise ValueError("Initial quaternion must be normalized")
        self.rotation = Rotation.from_quat(quaternion)
        self.accel_bias = readonly_array(accel_bias_mps2, (3,)).copy()
        self.gyro_bias = readonly_array(gyro_bias_rad_s, (3,)).copy()
        self.P = covariance(initial_covariance, 15)
        self.noise = np.array(
            [accel_std_mps2, gyro_std_rad_s, accel_bias_walk, gyro_bias_walk],
            dtype=float,
        )
        if not np.isfinite(self.noise).all() or np.any(self.noise < 0):
            raise ValueError("Noise standard deviations must be finite and nonnegative")
        if not np.isfinite(gps_gate_squared) or gps_gate_squared <= 0:
            raise ValueError("GPS gate must be finite and positive")
        self.gps_gate_squared = gps_gate_squared
        # Initial bias standard deviations, reported by summary().
        self.bias_priors = dict(
            accel_bias_std_mps2=float(np.sqrt(self.P[9, 9])),
            gyro_bias_std_rad_s=float(np.sqrt(self.P[12, 12])),
        )
        self.accepted_gps = self.rejected_gps = 0
        # Normalized innovation squared of the latest GPS update (3 dof),
        # gated or not, for consistency checks (NIS).
        self.last_gps_nis: float | None = None
        # With initialize_position_from_gps the first fix sets the position
        # (initialize_position) instead of correcting the declared prior.
        self.initialize_position_from_gps = initialize_position_from_gps
        self.position_initialized = False

    def predict(
        self,
        specific_force_mps2: ArrayLike,
        angular_rate_rad_s: ArrayLike,
        dt: float,
    ) -> None:
        if not np.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("IMU dt must be in (0, 0.1] seconds")
        force = readonly_array(specific_force_mps2, (3,)) - self.accel_bias
        omega = readonly_array(angular_rate_rad_s, (3,)) - self.gyro_bias
        R = self.rotation.as_matrix()
        increment = omega * dt
        Jr = right_jacobian(increment)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -0.5 * R @ skew(force) * dt**2
        F[3:6, 6:9] = -R @ skew(force) * dt
        F[0:3, 9:12] = -0.5 * R * dt**2
        F[3:6, 9:12] = -R * dt
        F[6:9, 6:9] = Rotation.from_rotvec(-increment).as_matrix()
        F[6:9, 12:15] = -Jr * dt
        G = np.zeros((15, 12))
        G[0:3, 0:3] = 0.5 * R * dt**2
        G[3:6, 0:3] = R * dt
        G[6:9, 3:6] = Jr * dt
        G[9:12, 6:9] = np.eye(3) * np.sqrt(dt)
        G[12:15, 9:12] = np.eye(3) * np.sqrt(dt)
        Q = (G * np.repeat(self.noise**2, 3)) @ G.T
        world_accel = R @ force + GRAVITY
        self.position += self.velocity * dt + 0.5 * world_accel * dt**2
        self.velocity += world_accel * dt
        self.rotation = self.rotation * Rotation.from_rotvec(increment)
        self.P = F @ self.P @ F.T + Q
        self.P = (self.P + self.P.T) / 2

    def predict_imu(self, imu: ImuInput, dt_s: float) -> None:
        """``RecursiveFilter`` prediction: ``predict`` with one held IMU
        reading in body axes."""
        self.predict(imu.specific_force_mps2, imu.angular_rate_rad_s, dt_s)

    def initialize_position(
        self,
        position_enu_m: ArrayLike,
        measurement_covariance: ArrayLike,
        *,
        receiver_position_body_m: ArrayLike = (0, 0, 0),
    ) -> None:
        """Set the position from one GPS fix instead of correcting a prior.

        Position becomes the antenna fix minus the lever arm (rotated by the
        current attitude); its covariance block becomes the measurement
        covariance and its cross-covariances are cleared, so the fix replaces
        the declared initial position and ``initial_position_std_m``.
        """
        observation = readonly_array(position_enu_m, (3,))
        lever = readonly_array(receiver_position_body_m, (3,))
        noise = covariance(measurement_covariance, 3)
        self.position = observation - self.rotation.as_matrix() @ lever
        self.P[:3, :] = 0.0
        self.P[:, :3] = 0.0
        self.P[:3, :3] = noise

    def correct_gps(
        self,
        position_enu_m: ArrayLike,
        measurement_covariance: ArrayLike,
        *,
        receiver_position_body_m: ArrayLike = (0, 0, 0),
    ) -> bool:
        """Correct with local ENU antenna position; return False for a gated outlier."""
        observation = readonly_array(position_enu_m, (3,))
        lever = readonly_array(receiver_position_body_m, (3,))
        noise = covariance(measurement_covariance, 3)
        R = self.rotation.as_matrix()
        H = np.zeros((3, 15))
        H[:, :3] = np.eye(3)
        H[:, 6:9] = -R @ skew(lever)
        innovation = observation - self.position - R @ lever
        S = H @ self.P @ H.T + noise
        distance = innovation @ np.linalg.solve(S, innovation)
        self.last_gps_nis = float(distance)
        if distance > self.gps_gate_squared:
            self.rejected_gps += 1
            return False
        K = np.linalg.solve(S, H @ self.P).T
        delta = K @ innovation
        A = np.eye(15) - K @ H
        posterior = A @ self.P @ A.T + K @ noise @ K.T
        self.position += delta[:3]
        self.velocity += delta[3:6]
        self.rotation = self.rotation * Rotation.from_rotvec(delta[6:9])
        self.accel_bias += delta[9:12]
        self.gyro_bias += delta[12:15]
        reset = np.eye(15)
        reset[6:9, 6:9] = right_jacobian(delta[6:9])
        self.P = reset @ posterior @ reset.T
        self.P = (self.P + self.P.T) / 2
        self.accepted_gps += 1
        return True

    def update_gps(self, fix: GpsFix) -> Innovation | None:
        """``RecursiveFilter`` GPS update: the first fix initializes the
        position when configured (no innovation, returns None), every other
        fix corrects (``correct_gps``) and returns its 3-dof innovation."""
        lever = fix.lever_arm_body_m
        if self.initialize_position_from_gps and not self.position_initialized:
            self.initialize_position(
                fix.position_enu_m, fix.covariance_m2, receiver_position_body_m=lever
            )
            self.position_initialized = True
            return None
        accepted = self.correct_gps(
            fix.position_enu_m, fix.covariance_m2, receiver_position_body_m=lever
        )
        assert self.last_gps_nis is not None
        return Innovation("gps", self.last_gps_nis, 3, accepted)

    def state(self) -> EstimatedState:
        return EstimatedState(
            self.position,
            self.velocity,
            self.rotation.as_quat(),
            self.accel_bias,
            self.gyro_bias,
            self.P,
        )

    def summary(self) -> dict[str, Any]:
        """Initialization and bias priors for ``summary.json``
        (``estimator_summary.filter_summary``); counts are the estimator's."""
        return dict(
            initialization=(
                "position from the first GPS fix; other states from declared "
                "ekf priors; no simulator truth"
                if self.initialize_position_from_gps
                else "declared ekf priors; no simulator truth"
            ),
            position_initialized=self.position_initialized,
            priors=self.bias_priors,
        )
