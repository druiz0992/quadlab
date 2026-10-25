"""Biped truth and IMU readings extracted from MuJoCo data.

Truth is evaluation and recording data only: it never enters the sensor
bus or an estimator (AGENTS.md §3). Frames and units: world z-up, metres;
quaternions (x, y, z, w) rotate trunk-body vectors into world; velocity is
the trunk origin's in world axes, angular velocity in trunk-body axes
(MuJoCo's free-joint convention); joint angles in rad (m for slides).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .scene import BipedScene


@dataclass(frozen=True)
class BipedTruth:
    """True biped state at ``sim_time_ns`` (see module docs)."""

    sim_time_ns: int
    position_m: NDArray[np.float64]
    quaternion_xyzw: NDArray[np.float64]
    velocity_mps: NDArray[np.float64]
    angular_velocity_rad_s: NDArray[np.float64]
    joint_position_rad: NDArray[np.float64]
    joint_velocity_rad_s: NDArray[np.float64]

    def record(self) -> dict[str, Any]:
        """One ``telemetry.jsonl`` line."""
        return dict(
            sim_time_ns=self.sim_time_ns,
            position_m=self.position_m.tolist(),
            quaternion_xyzw=self.quaternion_xyzw.tolist(),
            velocity_mps=self.velocity_mps.tolist(),
            angular_velocity_rad_s=self.angular_velocity_rad_s.tolist(),
            joint_position_rad=self.joint_position_rad.tolist(),
            joint_velocity_rad_s=self.joint_velocity_rad_s.tolist(),
        )


def read_truth(scene: BipedScene, data: Any, sim_time_ns: int) -> BipedTruth:
    """Copy the current state out of ``data`` (valid after ``mj_step1`` or
    ``mj_forward``, before integration)."""
    q, v = scene.free_qpos, scene.free_dof
    w, x, y, z = data.qpos[q + 3 : q + 7]
    return BipedTruth(
        sim_time_ns=sim_time_ns,
        position_m=np.array(data.qpos[q : q + 3], dtype=float),
        quaternion_xyzw=np.array([x, y, z, w], dtype=float),
        velocity_mps=np.array(data.qvel[v : v + 3], dtype=float),
        angular_velocity_rad_s=np.array(data.qvel[v + 3 : v + 6], dtype=float),
        joint_position_rad=np.array(data.qpos[scene.qpos_index], dtype=float),
        joint_velocity_rad_s=np.array(data.qvel[scene.dof_index], dtype=float),
    )


def imu_reading(
    scene: BipedScene, data: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Noise-free (specific force m/s², angular rate rad/s) in IMU-site axes.

    MuJoCo's accelerometer reports R^T (a - g) (a level body at rest reads
    about +9.81 on z), RobotLab's IMU convention. Valid after ``mj_step2``
    (or ``mj_forward``): both readings then describe the state at the start
    of that step.
    """
    a, g = scene.accel_address, scene.gyro_address
    return (
        np.array(data.sensordata[a : a + 3], dtype=float),
        np.array(data.sensordata[g : g + 3], dtype=float),
    )


def tilt_deg(quaternion_xyzw: NDArray[np.float64]) -> float:
    """Angle (deg) between the trunk's +z axis and world +z."""
    x, y, _, w = quaternion_xyzw / np.linalg.norm(quaternion_xyzw)
    # Body z in world is the third column of R; its world-z component is
    # 1 - 2(x² + y²).
    cos_tilt = float(np.clip(1 - 2 * (x * x + y * y), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_tilt)))
