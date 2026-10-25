"""The GPS/IMU EKF as an estimator plug-in (``algorithms.estimator: gps_imu_ekf``).

Options are the ``ekf`` configuration section (initial state and priors,
noise assumptions, GPS gate). Sensor mounts come from the context: the IMU
mount rotation and the GPS antenna lever arm. Samples are handled exactly as
delivered; a GPS fix older than the filter time (e.g. after GPS latency) is
counted in ``rejected_gps_timing``, since the filter does not re-propagate.

It is ``GpsImuEkf`` (ekf.py) run by the reusable inertial estimator
(estimation/inertial/), which does the sample timing, mounts, GPS
conversion, and publication for any filter; this module only builds the
filter from the ``ekf`` section and keeps its summary.

Debug channel (``context.log``, the reference example for plug-ins): at
each GPS update, ``ekf/gps`` = NIS and accepted (1/0), and ``ekf/bias`` =
the estimated accelerometer (m/s²) and gyro (rad/s) biases.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ...core.algorithms import AlgorithmContext
from ..inertial.estimator import InertialEstimator
from ..inertial.node import MeasurementOutcome
from .ekf import GpsImuEkf


def with_calibrated_priors(
    options: Mapping[str, Any], imu_calibration: Mapping[str, Any] | None
) -> dict[str, Any]:
    """EKF options with the initial bias uncertainty an enabled IMU
    calibration declares (``filter_*_bias_std``); unchanged without one."""
    result = dict(options)
    if imu_calibration:
        for source, target in (
            ("filter_accel_bias_std_mps2", "initial_accel_bias_std_mps2"),
            ("filter_gyro_bias_std_rad_s", "initial_gyro_bias_std_rad_s"),
        ):
            if imu_calibration.get(source) is not None:
                result[target] = imu_calibration[source]
    return result


def make_ekf(options: Mapping[str, Any]) -> GpsImuEkf:
    """Construct the filter from declared priors (never simulator truth)."""
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
    return GpsImuEkf(
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


class GpsImuEkfEstimator(InertialEstimator):
    """``StateEstimator`` for ``GpsImuEkf`` (see module docs)."""

    def __init__(self, context: AlgorithmContext, options: Mapping[str, Any]) -> None:
        for kind in ("imu", "gps"):
            if kind not in context.sensors:
                raise ValueError(f"gps_imu_ekf needs the {kind} sensor enabled")
        imu = context.sensors["imu"]
        # A calibrated IMU has smaller remaining biases: tighter priors.
        options = with_calibrated_priors(options, imu.calibration)
        self.priors = dict(
            accel_bias_std_mps2=options["initial_accel_bias_std_mps2"],
            gyro_bias_std_rad_s=options["initial_gyro_bias_std_rad_s"],
            calibrated_imu=imu.calibration is not None,
        )
        if any(imu.mount_position_m):
            raise ValueError(
                "gps_imu_ekf requires a body-origin IMU (rotation allowed)"
            )
        super().__init__(
            context,
            dict(
                gps_std_m=options["gps_std_m"],
                use_receiver_gps_accuracy=options["use_receiver_gps_accuracy"],
            ),
            make_ekf(options),
            name="gps_imu_ekf",
            log_prefix="ekf",
        )
        self.position_initialized_ns: int | None = None

    def _record(self, kind: str, sim_time_ns: int, outcome: MeasurementOutcome) -> None:
        if outcome.accepted and outcome.innovation is None:
            self.position_initialized_ns = sim_time_ns
        super()._record(kind, sim_time_ns, outcome)

    def summary(self) -> dict[str, Any]:
        ekf: GpsImuEkf = self.filter
        initialized = self.position_initialized_ns
        return dict(
            gps_updates=ekf.accepted_gps,
            gps_outliers=ekf.rejected_gps,
            rejected_gps_timing=self.node.rejected_timing["gps"],
            initialization=(
                "position from the first GPS fix; other states from declared "
                "ekf priors; no simulator truth"
                if ekf.initialize_position_from_gps
                else "declared ekf priors; no simulator truth"
            ),
            position_initialized_from_gps_s=(
                None if initialized is None else initialized * 1e-9
            ),
            receiver_covariance_fixes=self.node.converter.receiver_covariance_fixes,
            priors=self.priors,
        )
