"""Baseline quadplane autopilot: hover, transition, fixed-wing cruise, return.

A minimal, replaceable controller (``algorithms.controller:
dronelab.control.quadplane:QuadplaneController``) that exercises a quadplane
airframe: four vertical lift rotors, a pusher, and aileron/elevator/rudder
(``fixed_wing``). Not a tuned or certified autopilot.

Modes (``mode``), advanced from the controller's own state:

1. ``hover``: the multirotor controller (``ManualController``) climbs to
   ``cruise_altitude_m`` above the mission reference position (home).
2. ``transition``: once there and settled, the pusher runs at
   ``transition_thrust_n`` while the lift rotors hold altitude and a level
   attitude without braking horizontally (velocity reference = current
   velocity). Ends when airspeed reaches ``transition_airspeed_mps``.
3. ``cruise``: lift rotors fade out over ``lift_fade_s`` (holding altitude
   while the wing takes the load), then stay off. Lateral: L1 guidance (Park, Deyst & How
   2004) along the route, a_cmd = 2V²/L1 · sin η, φ_c = atan(a_cmd / g),
   with L1 = ``l1_period_s`` · V / π (damping 1/√2); waypoints switch within
   ``acceptance_radius_m``. Longitudinal: pitch command from altitude error,
   pusher thrust from airspeed error with a small integral. Inner loops on
   ailerons and elevator; a washed-out yaw-rate damper on the rudder.
4. ``back_transition``: after the last lap, pusher off; the multirotor
   controller brakes to a hover where it is.
5. ``return`` then ``land``: hover back over home at cruise altitude, then
   descend at ``land_speed_mps`` to the ground.

Tracking (evaluation only, ``tracking_reference``): the closest point on the
path segment of the current mode, so the error is the distance off the
path, not a lag behind a moving setpoint. Segments: the vertical line above
home (hover, land), home to the first waypoint at cruise altitude
(transition), the current route leg (cruise), and the straight line from
where the return began to home (return). Back-transition brakes wherever it
is, so only altitude is scored there. Reference velocity is the vehicle's
own velocity along the segment (only off-path velocity counts) and yaw is
not scored.

Angles: roll φ about body +x (right wing down positive); "nose-up pitch"
θ = −(FLU pitch). Surface signs follow simulation/fixed_wing.py (positive
elevator is nose down, positive aileron rolls right wing down, positive
rudder yaws nose left).
"""

from collections.abc import Mapping
import math
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .interface import ControlState
from .manual import FlightCommand, ManualController
from .references import ReferencePoint

DEFAULT_OPTIONS: dict[str, Any] = {
    "cruise_altitude_m": 40.0,
    "cruise_airspeed_mps": 22.0,
    # Optional [[seconds after cruise start, airspeed], ...], interpolated
    # linearly (flight tests such as finding the minimum level-flight speed).
    "airspeed_schedule": [],
    "transition_airspeed_mps": 19.0,
    "transition_thrust_n": 60.0,
    "max_pusher_thrust_n": 66.0,
    "route_m": [[300.0, 0.0], [300.0, 300.0], [0.0, 300.0], [0.0, 0.0]],
    "laps": 1,
    "acceptance_radius_m": 40.0,
    "l1_period_s": 20.0,
    "max_roll_deg": 35.0,
    "max_pitch_deg": 15.0,
    "trim_pitch_deg": 3.0,
    "trim_thrust_n": 18.0,
    "settle_s": 3.0,
    "lift_fade_s": 2.0,
    "land_speed_mps": 0.8,
    "gains": {
        "roll": 0.6,
        "roll_rate": 0.12,
        "pitch": 0.9,
        "pitch_rate": 0.25,
        "pitch_integral": 0.5,
        "altitude": 0.06,
        "climb_rate": 0.08,
        "airspeed": 6.0,
        "airspeed_integral": 1.0,
        "yaw_damper": 0.3,
        "washout_s": 1.0,
    },
    "hover": None,  # ManualController gains; None: the controller section
}


T = TypeVar("T")


def _started(value: T | None, name: str) -> T:
    """``value``; RuntimeError if a mode uses it before the autopilot has
    set it (a programming error, not a flight condition)."""
    if value is None:
        raise RuntimeError(f"QuadplaneController: {name} used before it was set")
    return value


class QuadplaneController:
    """See module docs. ``options`` override ``DEFAULT_OPTIONS``."""

    def __init__(self, context: Any, options: Mapping[str, Any] | None = None) -> None:
        o = dict(DEFAULT_OPTIONS, **(options or {}))
        o["gains"] = dict(DEFAULT_OPTIONS["gains"], **o.get("gains", {}))
        self.o = o
        self._g = o["gains"]
        self.vehicle = context.vehicle
        self._lift = list(self.vehicle.lift_rotors)
        self._pushers = [
            i for i in range(len(self.vehicle.rotors)) if i not in self._lift
        ]
        self.hover = ManualController(self.vehicle, o["hover"])
        self.mode = "hover"
        self.saturated = False
        self.mode_log: list[tuple[float, str]] = [(0.0, "hover")]
        self._home: NDArray[np.float64] | None = None
        self._settled_since: float | None = None
        self._route = [np.asarray(p, float) for p in o["route_m"]] * o["laps"]
        self._waypoint = 0
        self._previous_wp: NDArray[np.float64] | None = None
        self._thrust_integral = 0.0
        self._surfaces = {"aileron": 0.0, "elevator": 0.0, "rudder": 0.0}
        self._pusher = 0.0
        self._washout = 0.0
        self._time_s = 0.0
        self._hover_target: NDArray[np.float64] | None = None
        self._cruise_start: float | None = None
        self._pitch_integral = 0.0
        self._return_start: NDArray[np.float64] | None = None

    # -- state set once the autopilot starts ----------------------------------------

    @property
    def _home_m(self) -> NDArray[np.float64]:
        return _started(self._home, "home")

    @property
    def _cruise_start_s(self) -> float:
        return _started(self._cruise_start, "cruise start")

    @property
    def _previous_wp_m(self) -> NDArray[np.float64]:
        return _started(self._previous_wp, "previous waypoint")

    @property
    def _hover_target_m(self) -> NDArray[np.float64]:
        return _started(self._hover_target, "back-transition hover target")

    # -- RotorController contract -------------------------------------------------

    @property
    def setpoint_m(self):
        return self.hover.target

    def allow_landing(self) -> None:
        self.hover.allow_landing()

    def surface_commands(self) -> dict[str, float]:
        return dict(self._surfaces)

    def update_position(self, state: ControlState, request: Any, dt_s: float) -> None:
        """Outer loops: mode logic, guidance, and the multirotor position loop."""
        self._time_s = state.sim_time_ns * 1e-9
        if self._home is None:
            anchor = (
                request.position_m
                if isinstance(request, ReferencePoint)
                else state.position_m
            )
            self._home = np.array(anchor, float)
        self._advance_mode(state)
        if self.mode != "cruise" or self._fading():
            self.hover.update_position_reference(
                state.position_m,
                state.velocity_world_mps,
                state.quaternion_xyzw,
                self._hover_reference(state, dt_s),
            )
        if self.mode in ("transition", "cruise"):
            self._fixed_wing_outer(state, dt_s)

    def update_attitude(self, state: ControlState) -> NDArray[np.float64]:
        """Inner loops; rotor thrusts (N), lift rotors then pushers."""
        thrust = np.zeros(len(self.vehicle.rotors))
        if self.mode != "cruise" or self._fading():
            thrust = self.hover.update_attitude(
                state.quaternion_xyzw, state.body_rates_rad_s
            )
            self.saturated = self.hover.saturated
            if self.mode == "cruise":
                elapsed = self._time_s - self._cruise_start_s
                thrust *= max(0.0, 1.0 - elapsed / self.o["lift_fade_s"])
        else:
            self.saturated = False
        if self.mode in ("transition", "cruise"):
            self._fixed_wing_inner(state)
        else:
            self._surfaces = {"aileron": 0.0, "elevator": 0.0, "rudder": 0.0}
        for index in self._pushers:
            thrust[index] = (
                self._pusher if self.mode in ("transition", "cruise") else 0.0
            )
        return thrust

    # -- modes -------------------------------------------------------------------

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self.mode_log.append((self._time_s, mode))

    def _advance_mode(self, state: ControlState) -> None:
        o, p = self.o, state.position_m
        altitude = self._home_m[2] + o["cruise_altitude_m"]
        airspeed = state.airspeed_mps or 0.0
        if self.mode == "hover":
            close = (
                abs(p[2] - altitude) < 1.0
                and np.linalg.norm(state.velocity_world_mps) < 0.5
            )
            settled = (self._settled_since or self._time_s) if close else None
            self._settled_since = settled
            if settled is not None and self._time_s - settled >= o["settle_s"]:
                self._set_mode("transition")
        elif self.mode == "transition" and airspeed >= o["transition_airspeed_mps"]:
            self._previous_wp = p[:2].copy()
            self._cruise_start = self._time_s
            self._set_mode("cruise")
        elif self.mode == "cruise" and self._waypoint >= len(self._route):
            self._hover_target = p.copy()
            self._hover_target[2] = altitude
            self._set_mode("back_transition")
        elif (
            self.mode == "back_transition"
            and np.linalg.norm(state.velocity_world_mps) < 1.0
        ):
            self._return_start = p[:2].copy()
            self._set_mode("return")
        elif self.mode == "return" and np.linalg.norm(p[:2] - self._home_m[:2]) < 1.0:
            self.hover.allow_landing()
            self._set_mode("land")

    def _hover_reference(self, state: ControlState, dt_s: float) -> ReferencePoint:
        o, p, v = self.o, state.position_m, state.velocity_world_mps
        altitude = self._home_m[2] + o["cruise_altitude_m"]
        yaw = math.atan2(*(self._heading_vector()[::-1]))
        zero = np.zeros(3)
        if self.mode == "hover":
            target = np.array([self._home_m[0], self._home_m[1], altitude])
            return ReferencePoint(target, zero, zero, yaw)
        if self.mode in ("transition", "cruise"):
            # Hold altitude and attitude; no horizontal braking.
            return ReferencePoint(
                np.array([p[0], p[1], altitude]), np.array([v[0], v[1], 0.0]), zero, yaw
            )
        if self.mode == "back_transition":
            return ReferencePoint(
                self._hover_target_m,
                zero,
                zero,
                float(Rotation.from_quat(state.quaternion_xyzw).as_euler("ZYX")[0]),
            )
        if self.mode == "return":
            # Fly home at up to 5 m/s as a multicopter.
            offset = self._home_m[:2] - p[:2]
            step = (
                offset
                if np.linalg.norm(offset) < 2.0
                else 2.0 * offset / np.linalg.norm(offset)
            )
            target = np.array([p[0] + step[0], p[1] + step[1], altitude])
            return ReferencePoint(
                target,
                zero,
                zero,
                float(Rotation.from_quat(state.quaternion_xyzw).as_euler("ZYX")[0]),
            )
        target = np.array([self._home_m[0], self._home_m[1], self._home_m[2] - 1.0])
        descent = np.array([0.0, 0.0, -o["land_speed_mps"]])
        target[2] = max(target[2], p[2] - 1.0)
        return ReferencePoint(
            target,
            descent,
            zero,
            float(Rotation.from_quat(state.quaternion_xyzw).as_euler("ZYX")[0]),
        )

    def _fading(self) -> bool:
        return (
            self.mode == "cruise"
            and self._time_s - self._cruise_start_s < self.o["lift_fade_s"]
        )

    def _heading_vector(self) -> NDArray[np.float64]:
        """Direction of the first route leg (transition heading)."""
        first = self._route[0] - self._home_m[:2]
        norm = np.linalg.norm(first)
        return np.array([1.0, 0.0]) if norm < 1e-6 else first / norm

    # -- fixed-wing loops -----------------------------------------------------------

    def _fixed_wing_outer(self, state: ControlState, dt_s: float) -> None:
        """Guidance and energy: roll/pitch commands and pusher thrust."""
        o, g = self.o, self._g
        airspeed = max(state.airspeed_mps or 0.0, 1.0)
        altitude = self._home_m[2] + o["cruise_altitude_m"]
        if self.mode == "transition":
            self._roll_cmd, self._pitch_cmd = 0.0, math.radians(o["trim_pitch_deg"])
            self._pusher = o["transition_thrust_n"]
            return
        self._roll_cmd = self._l1_roll(state, airspeed)
        climb = float(state.velocity_world_mps[2])
        pitch = (
            math.radians(o["trim_pitch_deg"])
            + g["altitude"] * (altitude - state.position_m[2])
            - g["climb_rate"] * climb
        )
        limit = math.radians(o["max_pitch_deg"])
        self._pitch_cmd = float(np.clip(pitch, -limit, limit))
        error = self.commanded_airspeed() - airspeed
        self._thrust_integral = float(
            np.clip(self._thrust_integral + error * dt_s, -20, 20)
        )
        thrust = (
            o["trim_thrust_n"]
            + g["airspeed"] * error
            + g["airspeed_integral"] * self._thrust_integral
        )
        self._pusher = float(np.clip(thrust, 0.0, o["max_pusher_thrust_n"]))

    def commanded_airspeed(self) -> float:
        """Cruise airspeed now: the schedule if given, else the constant."""
        schedule = self.o["airspeed_schedule"]
        if not schedule or self._cruise_start is None:
            return self.o["cruise_airspeed_mps"]
        times, speeds = zip(*schedule)
        return float(np.interp(self._time_s - self._cruise_start_s, times, speeds))

    def _l1_roll(self, state: ControlState, airspeed: float) -> float:
        """L1 lateral guidance toward the current route leg."""
        o = self.o
        p, v = state.position_m[:2], state.velocity_world_mps[:2]
        target = self._route[self._waypoint]
        if np.linalg.norm(target - p) < o["acceptance_radius_m"]:
            self._previous_wp = target
            self._waypoint += 1
            if self._waypoint >= len(self._route):
                return 0.0
            target = self._route[self._waypoint]
        start = self._previous_wp_m
        leg = target - start
        length = np.linalg.norm(leg)
        unit = leg / length if length > 1e-6 else np.array([1.0, 0.0])
        l1 = o["l1_period_s"] * airspeed / math.pi
        # Point on the leg L1 ahead of the vehicle's projection.
        along = float((p - start) @ unit)
        lookahead = start + unit * min(along + l1, length + l1)
        to_point = lookahead - p
        speed = max(np.linalg.norm(v), 1.0)
        eta = math.atan2(v[0] * to_point[1] - v[1] * to_point[0], v @ to_point)
        # Target behind: sin η → 0 would command no turn, so clamp to ±90°
        # (a full turn toward the side the target is on; left if dead astern).
        if abs(eta) > math.pi / 2:
            eta = math.copysign(math.pi / 2, eta if eta != 0 else 1.0)
        accel = 2 * speed**2 / max(np.linalg.norm(to_point), 1.0) * math.sin(eta)
        # Left turn (η > 0) needs left wing down: negative roll.
        roll = -math.atan(accel / 9.81)
        limit = math.radians(o["max_roll_deg"])
        return float(np.clip(roll, -limit, limit))

    def _fixed_wing_inner(self, state: ControlState) -> None:
        """Attitude loops on the control surfaces."""
        g = self._g
        _, pitch_about_left, roll = Rotation.from_quat(state.quaternion_xyzw).as_euler(
            "ZYX"
        )
        pitch = -pitch_about_left  # nose-up positive (FLU +y points left)
        p, q_flu, r_flu = state.body_rates_rad_s
        aileron = g["roll"] * (self._roll_cmd - roll) - g["roll_rate"] * p
        # Positive elevator pitches nose down, so the nose-up law is negated.
        # Integral term: holding high α needs large elevator (nose-down Cm).
        error = self._pitch_cmd - pitch
        self._pitch_integral = float(
            np.clip(self._pitch_integral + error * 0.02, -0.5, 0.5)
        )
        elevator = -(
            g["pitch"] * error
            + g["pitch_integral"] * self._pitch_integral
            + g["pitch_rate"] * q_flu
        )
        # Washed-out yaw-rate damper: oppose yaw-rate changes, not steady
        # turns. Positive rudder yaws nose left (+z), so it opposes +r.
        self._washout += (r_flu - self._washout) * min(1.0, 0.02 / g["washout_s"])
        rudder = -g["yaw_damper"] * (r_flu - self._washout)
        self._surfaces = {
            "aileron": float(aileron),
            "elevator": float(elevator),
            "rudder": float(rudder),
        }

    # -- evaluation ---------------------------------------------------------------

    def tracking_reference(
        self,
        position_m: NDArray[np.float64],
        velocity_world_mps: NDArray[np.float64],
        quaternion_xyzw: NDArray[np.float64],
    ) -> ReferencePoint | None:
        """Closest point on the current path segment (see module docs);
        None before the first position update. Reads state only."""
        if self._home is None:
            return None
        p = np.asarray(position_m, float)
        v = np.asarray(velocity_world_mps, float)
        yaw = float(Rotation.from_quat(quaternion_xyzw).as_euler("ZYX")[0])
        segment = self._path_segment()
        if segment is None:  # altitude hold only
            altitude = self._home_m[2] + self.o["cruise_altitude_m"]
            return ReferencePoint(
                np.array([p[0], p[1], altitude]),
                np.array([v[0], v[1], 0.0]),
                np.zeros(3),
                yaw,
            )
        start, end = segment
        leg = end - start
        length = float(np.linalg.norm(leg))
        unit = leg / length if length > 1e-9 else np.zeros(3)
        along = float(np.clip((p - start) @ unit, 0.0, length))
        return ReferencePoint(start + unit * along, (v @ unit) * unit, np.zeros(3), yaw)

    def _path_segment(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """(start, end) of the segment flown now, or None (altitude hold)."""
        home = self._home_m
        altitude = home[2] + self.o["cruise_altitude_m"]

        def at_altitude(xy: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.array([xy[0], xy[1], altitude])

        if self.mode in ("hover", "land"):
            return home.copy(), at_altitude(home[:2])
        if self.mode == "transition":
            return at_altitude(home[:2]), at_altitude(self._route[0])
        if self.mode == "cruise" and self._waypoint < len(self._route):
            return (
                at_altitude(self._previous_wp_m),
                at_altitude(self._route[self._waypoint]),
            )
        if self.mode == "return" and self._return_start is not None:
            return at_altitude(self._return_start), at_altitude(home[:2])
        return None

    def summary(self) -> dict[str, Any]:
        return dict(
            mode=self.mode, mode_log=self.mode_log, waypoints_reached=self._waypoint
        )
