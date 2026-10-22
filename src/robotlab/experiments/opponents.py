"""Opponent vehicles: other wheeled robots of the ego's kind, each with its
own controller, wheel model, and wrench (``ground_vehicle.opponents``).

Each opponent is a Gazebo model (``car_1``, ``car_2``, ...) on the scene's
spawn points after the ego, publishing truth odometry on
``/robotlab/opponents/<name>/odometry`` (no sensors). Its messages have the
kind ``opponent:<name>`` and are processed, in name order, just before the
ego's odometry of the same period (``LockstepOrderer``): each opponent
stores its state and runs its controller at the control rate, and when the
ego's odometry arrives one batched model step advances every vehicle
(vehicle 0 is the ego, opponent i is vehicle i). The ego controller sees
the opponents' states of the same period (``WheeledControlState.opponents``,
labelled ground truth). Opponent controllers drive on their own truth
(exact wheel speeds and steering angles, no encoder model) with no driver
request; the default is the centreline follower
(``control/track_follower.py``). This is ``mode: "dynamic"``
(``Opponent``).

``mode: "kinematic"`` (the default, ``KinematicOpponent``) makes opponents
cheap moving obstacles: no wheel model or wrench. Their odometry arrives
at ``update_rate_hz`` and each message yields one ``KinematicTrackDriver``
command, published as a body-frame twist to Gazebo's VelocityControl
system (gravity off): forward speed, yaw rate, a vertical term holding the
start-up height above the ground, and roll/pitch rates levelling the body
after a knock. Gazebo still resolves their collisions with the ego. The
ego controller sees their states up to one update period old.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..control.track_follower import KinematicCommand, KinematicTrackDriver
from ..control.wheeled import OpponentState, WheeledControlState, WheeledController
from ..core.algorithms import AlgorithmContext
from ..simulation.gazebo.messages import timestamp_ns, vector
from ..simulation.gazebo.odometry import OdometryVelocityFix
from ..simulation.wheeled.model import ChassisState
from .schedule import next_due_ns

# Kinematic opponents: gains (1/s) pulling the height above ground back to
# its start-up value and the body back to level, and the vertical speed cap.
HEIGHT_GAIN_PER_S = 5.0
LEVEL_GAIN_PER_S = 5.0
MAX_VERTICAL_SPEED_MPS = 2.0


@dataclass
class Opponent:
    """One dynamic opponent's runtime parts and latest state (vehicle
    ``index`` of the shared model)."""

    dynamic = True

    name: str
    index: int
    controller: Any
    wrench: Any
    laps: Any | None
    control_period_ns: int
    odometry_fix: OdometryVelocityFix
    chassis: ChassisState | None = None
    state: OpponentState | None = None
    stamp_ns: int | None = None
    next_control_ns: int | None = None
    last_control_ns: int | None = None
    max_speed_mps: float = 0.0
    control_ticks: int = 0

    def on_odometry(self, message: Any, model: Any) -> OpponentState:
        """Decode truth; run the controller at the control rate."""
        stamp = timestamp_ns(message)
        chassis = _chassis(message, stamp, self.odometry_fix)
        self.chassis, self.stamp_ns = chassis, stamp
        if self.next_control_ns is None or stamp >= self.next_control_ns:
            self._control(stamp, chassis, model)
        self.max_speed_mps = max(
            self.max_speed_mps, float(np.linalg.norm(chassis.velocity_world_mps))
        )
        self.state = OpponentState(
            self.name,
            stamp,
            chassis.position_m,
            chassis.velocity_world_mps,
            chassis.quaternion_xyzw,
            chassis.body_rates_rad_s,
        )
        return self.state

    def _control(self, stamp: int, chassis: ChassisState, model: Any) -> None:
        dt_s = (
            0.0
            if self.last_control_ns is None
            else (stamp - self.last_control_ns) * 1e-9
        )
        self.last_control_ns = stamp
        self.next_control_ns = next_due_ns(
            self.next_control_ns, stamp, self.control_period_ns
        )
        rows = model.rows(self.index)
        state = WheeledControlState(
            stamp,
            chassis.position_m,
            chassis.velocity_world_mps,
            chassis.quaternion_xyzw,
            chassis.body_rates_rad_s,
            model.wheel_speed_rad_s[rows].copy(),
            model.actuators.steer_rad[rows].copy(),
        )
        commands = self.controller.update(state, None, dt_s)
        model.actuators.accept(commands, stamp, vehicle=self.index)
        self.control_ticks += 1

    def after_step(self, wrench: Any, model: Any, evaluate: bool) -> None:
        """Apply this period's wrench; lap timing when ``evaluate``."""
        self.wrench.set(wrench.force_world_n, wrench.torque_world_nm)
        if evaluate and self.laps is not None and self.chassis is not None:
            telemetry = model.telemetry_of(self.index)
            self.laps.on_truth(
                (self.stamp_ns or 0) * 1e-9,
                self.chassis.position_m,
                telemetry.contact_point_m[:, :2],
            )

    def summary(self) -> dict[str, Any]:
        summary = (
            self.controller.summary() if hasattr(self.controller, "summary") else None
        )
        return dict(
            name=self.name,
            mode="dynamic",
            max_speed_mps=self.max_speed_mps,
            control_ticks=self.control_ticks,
            track=self.laps.summary() if self.laps is not None else None,
            controller=summary,
        )


@dataclass
class KinematicOpponent:
    """One kinematic opponent (see module docs): ``driver`` turns its truth
    into a ``KinematicCommand``, ``publish(twist_fields)`` sends the body
    twist, ``height_m(x, y)`` is the ground height, and ``hub_xy_m`` are the
    wheel hubs in body axes (lap timing contact points)."""

    dynamic = False

    name: str
    driver: KinematicTrackDriver
    publish: Any
    height_m: Any
    hub_xy_m: NDArray[np.float64]
    laps: Any | None
    odometry_fix: OdometryVelocityFix
    clearance_m: float | None = None
    state: OpponentState | None = None
    last_stamp_ns: int | None = None
    max_speed_mps: float = 0.0
    control_ticks: int = 0

    def on_odometry(self, message: Any, model: Any) -> OpponentState:
        """Decode truth, publish the next command, update lap timing."""
        stamp = timestamp_ns(message)
        chassis = _chassis(message, stamp, self.odometry_fix)
        dt_s = (
            0.0 if self.last_stamp_ns is None else (stamp - self.last_stamp_ns) * 1e-9
        )
        self.last_stamp_ns = stamp
        state = WheeledControlState(
            stamp,
            chassis.position_m,
            chassis.velocity_world_mps,
            chassis.quaternion_xyzw,
            chassis.body_rates_rad_s,
            np.zeros(len(self.hub_xy_m)),
            np.zeros(len(self.hub_xy_m)),
        )
        self.publish(self._twist(chassis, self.driver.update(state, dt_s)))
        self.control_ticks += 1
        speed = float(np.linalg.norm(chassis.velocity_world_mps))
        self.max_speed_mps = max(self.max_speed_mps, speed)
        if self.laps is not None:
            self.laps.on_truth(
                stamp * 1e-9, chassis.position_m, self._contacts_xy(chassis)
            )
        self.state = OpponentState(
            self.name,
            stamp,
            chassis.position_m,
            chassis.velocity_world_mps,
            chassis.quaternion_xyzw,
            chassis.body_rates_rad_s,
        )
        return self.state

    def _twist(
        self, chassis: ChassisState, command: KinematicCommand
    ) -> dict[str, tuple[float, float, float]]:
        """Body-frame linear and angular velocity: the command plus height
        and levelling corrections (proportional, small-angle)."""
        x, y, z = (float(v) for v in chassis.position_m)
        ground = float(self.height_m(np.array(x), np.array(y)))
        if self.clearance_m is None:
            self.clearance_m = z - ground
        vertical = HEIGHT_GAIN_PER_S * (ground + self.clearance_m - z)
        vertical = float(
            np.clip(vertical, -MAX_VERTICAL_SPEED_MPS, MAX_VERTICAL_SPEED_MPS)
        )
        roll, pitch, _ = Rotation.from_quat(chassis.quaternion_xyzw).as_euler("xyz")
        return dict(
            linear=(command.speed_mps, 0.0, vertical),
            angular=(
                -LEVEL_GAIN_PER_S * float(roll),
                -LEVEL_GAIN_PER_S * float(pitch),
                command.yaw_rate_rad_s,
            ),
        )

    def _contacts_xy(self, chassis: ChassisState) -> NDArray[np.float64]:
        x, y, z, w = chassis.quaternion_xyzw
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = np.cos(yaw), np.sin(yaw)
        hubs = self.hub_xy_m
        return np.column_stack(
            (
                chassis.position_m[0] + c * hubs[:, 0] - s * hubs[:, 1],
                chassis.position_m[1] + s * hubs[:, 0] + c * hubs[:, 1],
            )
        )

    def summary(self) -> dict[str, Any]:
        return dict(
            name=self.name,
            mode="kinematic",
            max_speed_mps=self.max_speed_mps,
            control_ticks=self.control_ticks,
            track=self.laps.summary() if self.laps is not None else None,
            controller=self.driver.summary(),
        )


def _chassis(message: Any, stamp: int, fix: OdometryVelocityFix) -> ChassisState:
    p = vector(message.pose.position)
    q = message.pose.orientation
    quaternion = np.array([q.x, q.y, q.z, q.w])
    velocity = Rotation.from_quat(quaternion).apply(vector(message.twist.linear))
    velocity, rates = fix.correct(
        stamp, p, quaternion, velocity, vector(message.twist.angular)
    )
    return ChassisState(p, quaternion, velocity, rates)


def opponent_options(settings: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Controller options of opponent ``index`` (0-based): common options,
    then that opponent's own entry."""
    opponents = settings["ground_vehicle"]["opponents"]
    options = dict(opponents["controller_options"])
    per: Sequence[Mapping[str, Any]] = opponents["per_opponent_options"]
    if index < len(per):
        options.update(per[index])
    return options


def build_opponent_controller(
    spec: str,
    builtins: Mapping[str, str],
    context: AlgorithmContext,
    options: Mapping[str, Any],
) -> Any:
    """Construct and check one opponent controller (ValueError if invalid)."""
    from .algorithms import load_class

    controller = load_class(spec, builtins)(context, dict(options))
    if not isinstance(controller, WheeledController):
        raise ValueError(f"{spec} does not implement WheeledController (update)")
    return controller


def latest_states(
    opponents: Sequence[Opponent | KinematicOpponent],
) -> tuple[OpponentState, ...]:
    """States of every opponent that has published, in name order."""
    return tuple(o.state for o in opponents if o.state is not None)


def positions(opponents: Sequence[Opponent | KinematicOpponent]) -> NDArray[np.float64]:
    """Latest opponent positions (n×3; empty when none has published)."""
    states = latest_states(opponents)
    return np.array([s.position_m for s in states]) if states else np.zeros((0, 3))
