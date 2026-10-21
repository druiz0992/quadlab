"""The GPS/IMU EKF as an estimator plug-in (``algorithms.estimator: gps_imu_ekf``).

Options are the ``ekf`` configuration section (initial state and priors,
noise assumptions, GPS gate). Sensor mounts come from the context: the IMU
mount rotation and the GPS antenna lever arm. Samples are handled exactly as
delivered; a GPS fix older than the filter time (e.g. after GPS latency) is
counted in ``rejected_gps_timing``, since the filter does not re-propagate.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ...core.algorithms import AlgorithmContext
from ...core.measurements import EstimatedState, Sample
from ..interface import Innovation
from .ekf import GpsImuEkf
from .node import GpsImuEkfNode


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
    )


class GpsImuEkfEstimator:
    """``StateEstimator`` wrapping ``GpsImuEkfNode`` (see module docs)."""

    inputs = ("imu", "gps")

    def __init__(self, context: AlgorithmContext, options: Mapping[str, Any]) -> None:
        for kind in self.inputs:
            if kind not in context.sensors:
                raise ValueError(f"gps_imu_ekf needs the {kind} sensor enabled")
        imu, gps = context.sensors["imu"], context.sensors["gps"]
        if any(imu.mount_position_m):
            raise ValueError(
                "gps_imu_ekf requires a body-origin IMU (rotation allowed)"
            )
        self.node = GpsImuEkfNode(
            make_ekf(options),
            gps_reference=context.gps_reference,
            gps_covariance_m2=np.diag(np.asarray(options["gps_std_m"]) ** 2),
            imu_quaternion_body_xyzw=imu.mount_quaternion_xyzw,
            gps_position_body_m=gps.mount_position_m,
            allow_intermediate_gps=True,
        )
        self._estimates = self.node.estimates.subscribe(
            capacity=1, overflow="drop_oldest"
        )
        self._latest: Sample[EstimatedState] | None = None
        self._innovations: list[Innovation] = []
        self._errors = 0

    def on_sample(self, kind: str, sample: Sample[Any]) -> None:
        if kind == "imu":
            self._latest = self.node.on_imu(sample)
            return
        before = self.node.ekf.accepted_gps + self.node.ekf.rejected_gps
        accepted = self.node.on_gps(sample)
        ekf = self.node.ekf
        if (
            ekf.accepted_gps + ekf.rejected_gps > before
            and ekf.last_gps_nis is not None
        ):
            self._innovations.append(Innovation("gps", ekf.last_gps_nis, 3, accepted))
        if accepted:
            self._latest = self._estimates.next(timeout_wall_s=0)

    def latest(self) -> Sample[EstimatedState] | None:
        return self._latest

    def drain_innovations(self) -> list[Innovation]:
        innovations, self._innovations = self._innovations, []
        return innovations

    def summary(self) -> dict[str, Any]:
        ekf = self.node.ekf
        return dict(
            gps_updates=ekf.accepted_gps,
            gps_outliers=ekf.rejected_gps,
            rejected_gps_timing=self.node.rejected_gps_timing,
            initialization="declared ekf priors; no simulator truth",
        )
