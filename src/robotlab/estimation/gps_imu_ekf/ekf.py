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
from ..filter.kalman import checked_covariance, kalman_update
from ..filter.measurements import GpsFix, ImuInput
from ..filter.models import gps_position
from ..filter.state_layout import StateLayout
from ..interface import Innovation
from ..lie import right_jacobian, skew

LAYOUT = StateLayout.of("position", "velocity", "attitude", "accel_bias", "gyro_bias")
POS, VEL, ATT, ACCEL_BIAS, GYRO_BIAS = (LAYOUT.index(name) for name in LAYOUT.names)
N = LAYOUT.size  # 15 error states


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
        self.P = checked_covariance(initial_covariance, N)
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
            accel_bias_std_mps2=float(
                np.sqrt(self.P[ACCEL_BIAS.start, ACCEL_BIAS.start])
            ),
            gyro_bias_std_rad_s=float(
                np.sqrt(self.P[GYRO_BIAS.start, GYRO_BIAS.start])
            ),
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
        F = np.eye(N)
        F[POS, VEL] = np.eye(3) * dt
        F[POS, ATT] = -0.5 * R @ skew(force) * dt**2
        F[VEL, ATT] = -R @ skew(force) * dt
        F[POS, ACCEL_BIAS] = -0.5 * R * dt**2
        F[VEL, ACCEL_BIAS] = -R * dt
        F[ATT, ATT] = Rotation.from_rotvec(-increment).as_matrix()
        F[ATT, GYRO_BIAS] = -Jr * dt
        # Noise inputs: accel, gyro, accel-bias walk, gyro-bias walk.
        G = np.zeros((N, 12))
        G[POS, 0:3] = 0.5 * R * dt**2
        G[VEL, 0:3] = R * dt
        G[ATT, 3:6] = Jr * dt
        G[ACCEL_BIAS, 6:9] = np.eye(3) * np.sqrt(dt)
        G[GYRO_BIAS, 9:12] = np.eye(3) * np.sqrt(dt)
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
        noise = checked_covariance(measurement_covariance, 3)
        self.position = observation - self.rotation.as_matrix() @ lever
        self.P[POS, :] = 0.0
        self.P[:, POS] = 0.0
        self.P[POS, POS] = noise

    def correct_gps(
        self,
        position_enu_m: ArrayLike,
        measurement_covariance: ArrayLike,
        *,
        receiver_position_body_m: ArrayLike = (0, 0, 0),
    ) -> bool:
        """Correct with local ENU antenna position; return False for a gated outlier."""
        fix = GpsFix(
            position_enu_m=readonly_array(position_enu_m, (3,)),
            covariance_m2=checked_covariance(measurement_covariance, 3),
            lever_arm_body_m=readonly_array(receiver_position_body_m, (3,)),
        )
        measurement = gps_position(
            LAYOUT,
            fix,
            position_m=self.position,
            rotation_matrix=self.rotation.as_matrix(),
        )
        correction = kalman_update(self.P, measurement, self.gps_gate_squared)
        self.last_gps_nis = correction.innovation.nis
        if correction.delta is None:
            self.rejected_gps += 1
            return False
        self._inject(correction.delta, correction.covariance)
        self.accepted_gps += 1
        return True

    def _inject(
        self, delta: NDArray[np.float64], posterior: NDArray[np.float64]
    ) -> None:
        """Apply an error-state correction and reset the covariance.

        The attitude error is folded in on the right, R ← R · Exp(δθ); the
        covariance is then expressed about the corrected attitude with the
        Jacobian of that reset (J_r(δθ) on the attitude block).
        """
        self.position += delta[POS]
        self.velocity += delta[VEL]
        self.rotation = self.rotation * Rotation.from_rotvec(delta[ATT])
        self.accel_bias += delta[ACCEL_BIAS]
        self.gyro_bias += delta[GYRO_BIAS]
        reset = np.eye(N)
        reset[ATT, ATT] = right_jacobian(delta[ATT])
        self.P = reset @ posterior @ reset.T
        self.P = (self.P + self.P.T) / 2

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
        return LAYOUT.estimated_state(
            self.P,
            position_m=self.position,
            velocity_mps=self.velocity,
            quaternion_xyzw=self.rotation.as_quat(),
            accel_bias_mps2=self.accel_bias,
            gyro_bias_rad_s=self.gyro_bias,
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
