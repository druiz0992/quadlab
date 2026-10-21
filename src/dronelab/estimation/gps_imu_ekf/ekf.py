"""Educational GPS/IMU error-state EKF, with explicit initialization.

Nominal state: body-origin position/velocity in local ENU, body-to-world
rotation, body-axis accelerometer/gyro biases. Right attitude error, 15x15
covariance. IMU values are held at interval starts. Bias random walks are
injected at interval ends. This is not a calibrated navigation system.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from ...core.frames import GRAVITY
from ...core.measurements import EstimatedState, readonly_array


def skew(v: ArrayLike) -> NDArray[np.float64]:
    x, y, z = v
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


class GpsImuEkf:
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
        self.accepted_gps = self.rejected_gps = 0
        # Normalized innovation squared of the latest GPS update (3 dof),
        # gated or not, for consistency checks (NIS).
        self.last_gps_nis: float | None = None

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

    def state(self) -> EstimatedState:
        return EstimatedState(
            self.position,
            self.velocity,
            self.rotation.as_quat(),
            self.accel_bias,
            self.gyro_bias,
            self.P,
        )
