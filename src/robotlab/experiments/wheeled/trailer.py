"""Towed trailer runtime: hitch kinematics, wheel forces, brakes, jackknife.

The trailer is part of the tractor's Gazebo model: link ``trailer`` hangs
from the hitch on a yaw (body z) and a pitch joint, so its roll follows the
tractor's (a simplification). Gazebo publishes the joint angles and rates
every physics step (``trailer`` messages, ordered before the tractor's
odometry of the same period). From them and the tractor's truth Python
computes the trailer's state by forward kinematics:

    R_tr = R_t · Rz(ψ) · Ry(θ),   p_tr = h + R_tr · (−d, 0, −c)
    ω_tr = ω_t + ψ̇ R_t ẑ + θ̇ R_t Rz(ψ) ŷ,   v_tr = v_h + ω_tr × (p_tr − h)

with the hitch h = p_t + R_t h_b (h_b: ``hitch_position_m``), drawbar d,
and centre of mass c below the hitch; v_h = v_t + ω_t × (h − p_t). A
wheeled model without drive or steering (``trailer_properties``) gives the
trailer's wheel wrench, applied as one instantaneous wrench on
``<model>::trailer``. The trailer brakes with the tractor: its brake
torque is the tractor's mean commanded brake fraction times
``max_brake_torque_nm`` (an overrun-brake stand-in).
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ...core.wheeled import WheelCommands
from ...core.wheeled_vehicle import WheeledProperties
from ...simulation.gazebo.messages import timestamp_ns
from ...simulation.wheeled.config import trailer_properties
from ...simulation.wheeled.ground import GroundModel
from ...simulation.wheeled.model import (
    ChassisState,
    WheeledVehicleModel,
    rotation_matrix,
)

# Fraction of the yaw joint limit that counts as a jackknife.
JACKKNIFE_FRACTION = 0.9


def _rz(angle: float) -> NDArray[np.float64]:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(angle: float) -> NDArray[np.float64]:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def trailer_state(
    tractor: ChassisState,
    settings: Mapping[str, Any],
    yaw: float,
    pitch: float,
    yaw_rate: float,
    pitch_rate: float,
) -> ChassisState:
    """Trailer centre-of-mass state from the tractor's (see module docs)."""
    r_t = rotation_matrix(tractor.quaternion_xyzw)
    r_yaw = r_t @ _rz(yaw)
    r_tr = r_yaw @ _ry(pitch)
    hitch = tractor.position_m + r_t @ np.asarray(settings["hitch_position_m"], float)
    offset = r_tr @ np.array(
        [-settings["drawbar_m"], 0.0, -settings["cg_below_hitch_m"]]
    )
    position = hitch + offset
    omega_t = r_t @ tractor.body_rates_rad_s
    omega = omega_t + yaw_rate * r_t[:, 2] + pitch_rate * r_yaw[:, 1]
    v_hitch = tractor.velocity_world_mps + np.cross(omega_t, hitch - tractor.position_m)
    velocity = v_hitch + np.cross(omega, offset)
    return ChassisState(
        position,
        Rotation.from_matrix(r_tr).as_quat(),
        velocity,
        r_tr.T @ omega,
    )


class Trailer:
    """One trailer's runtime state (see module docs)."""

    def __init__(
        self,
        settings: Mapping[str, Any],
        tractor: WheeledProperties,
        ground: GroundModel,
        wrench: Any,
    ) -> None:
        self.settings = settings
        self.properties = trailer_properties(settings, tractor)
        self.model = WheeledVehicleModel(self.properties, ground)
        self.wrench = wrench
        self._tractor_max_brake = tractor.drive.max_brake_torque_nm
        self.yaw = self.pitch = self.yaw_rate = self.pitch_rate = 0.0
        self.stamp_ns: int | None = None
        self.state: ChassisState | None = None
        self.max_yaw_deg = 0.0
        self.jackknife_s = 0.0
        self.stale_periods = 0

    def on_joint_state(self, message: Any) -> None:
        """Hitch joint angles and rates from Gazebo's joint-state message."""
        joints = {joint.name: joint.axis1 for joint in message.joint}
        if "trailer_yaw" not in joints or "trailer_pitch" not in joints:
            raise RuntimeError("trailer joint state lacks trailer_yaw/trailer_pitch")
        self.yaw, self.yaw_rate = (
            joints["trailer_yaw"].position,
            joints["trailer_yaw"].velocity,
        )
        self.pitch, self.pitch_rate = (
            joints["trailer_pitch"].position,
            joints["trailer_pitch"].velocity,
        )
        self.stamp_ns = timestamp_ns(message)

    def step(
        self, stamp_ns: int, tractor: ChassisState, brake_fraction: float, dt_s: float
    ) -> bool:
        """Advance the trailer one period; True when jackknifed."""
        if self.stamp_ns != stamp_ns:
            self.stale_periods += 1
        n = self.properties.wheel_count
        brake = tuple(
            np.where(
                self.properties.mask("braked"),
                brake_fraction * self.settings["max_brake_torque_nm"],
                0.0,
            )
        )
        self.model.actuators.accept(
            WheelCommands((0.0,) * n, brake, (0.0,) * n), stamp_ns
        )
        self.state = trailer_state(
            tractor, self.settings, self.yaw, self.pitch, self.yaw_rate, self.pitch_rate
        )
        wrench = self.model.step(self.state, dt_s)
        self.wrench.set(wrench.force_world_n, wrench.torque_world_nm)
        yaw_deg = abs(np.degrees(self.yaw))
        self.max_yaw_deg = max(self.max_yaw_deg, yaw_deg)
        jackknifed = yaw_deg > JACKKNIFE_FRACTION * self.settings["yaw_limit_deg"]
        if jackknifed:
            self.jackknife_s += dt_s
        return jackknifed

    def brake_fraction(self, tractor_brake_nm: tuple[float, ...]) -> float:
        """Mean tractor brake command as a fraction of its brake limit."""
        if not tractor_brake_nm or self._tractor_max_brake <= 0:
            return 0.0
        return float(
            np.clip(np.mean(tractor_brake_nm) / self._tractor_max_brake, 0.0, 1.0)
        )

    def summary(self) -> dict[str, Any]:
        return dict(
            mass_kg=self.properties.mass_kg,
            max_yaw_deg=self.max_yaw_deg,
            jackknife_s=self.jackknife_s,
            stale_periods=self.stale_periods,
        )
