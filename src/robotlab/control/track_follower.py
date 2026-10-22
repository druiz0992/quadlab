"""Centreline follower: a simple traffic driver for a scene track.

Drives opponent cars by default (``ground_vehicle.opponents``) and can drive
the ego vehicle (``algorithms.controller: track_follower``). It is a
baseline for traffic, not a racing algorithm:

1. **Path**: the track centreline shifted by ``lateral_offset_m`` (positive
   left). The look-ahead point lies ``lookahead_s`` · v (at least
   ``min_lookahead_m``) ahead along the path from the vehicle's projection.
2. **Steering**: pure pursuit (Coulter, "Implementation of the Pure Pursuit
   Path Tracking Algorithm", CMU-RI-TR-92-01, 1992): curvature
   κ = 2 sin α / L_d with α the look-ahead bearing in body axes; Ackermann
   and four-wheel (front mode) layouts steer δ = atan(κ · wheelbase); skid
   steering turns the curvature into a yaw-rate target v κ held by a
   left/right torque difference.
3. **Speed**: the target is min(``max_speed_mps``, √(a_lat / |κ_max|)) with
   κ_max the largest centreline curvature in the next ``preview_m`` and
   a_lat = ``grip_fraction`` · μ_lat · g; drive torque follows the speed
   error (``speed_gain_per_s``, desired acceleration per m/s), braking when
   the error is negative. Acceleration and braking demands are capped at
   ``grip_fraction`` · μ_long · g, the acceleration additionally by the
   static load share of the driven wheels (so a rear-drive car does not
   spin its wheels and lose lateral grip).

Uses only ``state`` (own truth or estimate), ``context.wheeled``, and
``context.track``; raises ValueError without a track.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.algorithms import AlgorithmContext
from ..core.wheeled import WheelCommands
from .wheeled import DriverCommand, WheeledControlState
from .wheeled_keyboard import steering_angles

DEFAULTS = dict(
    lateral_offset_m=0.0,
    lookahead_s=0.6,
    min_lookahead_m=6.0,
    preview_m=60.0,
    grip_fraction=0.7,
    max_speed_mps=40.0,
    speed_gain_per_s=1.5,
)


class TrackFollower:
    """``WheeledController`` following the scene track (see module docs)."""

    def __init__(self, context: AlgorithmContext, options: Mapping[str, Any]) -> None:
        if context.wheeled is None:
            raise ValueError("TrackFollower needs a wheeled vehicle")
        if context.track is None:
            raise ValueError("TrackFollower needs a scene track (scene 'track')")
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"TrackFollower: unknown options {sorted(unknown)}")
        self.options = {**DEFAULTS, **options}
        self.vehicle = context.wheeled
        self.track = context.track
        self._path, self._s = self._offset_path(self.options["lateral_offset_m"])
        self._length = float(self._s[-1])
        self._curvature = _curvature(self._path)
        self._abs_curvature = np.abs(self._curvature)
        self._driven = self.vehicle.mask("driven")
        self._braked = self.vehicle.mask("braked")
        self._side = np.sign(self.vehicle.hub_positions_m()[:, 1])
        self.last_target_mps = 0.0
        self._segment: int | None = None

    def _offset_path(self, offset_m: float) -> tuple[NDArray, NDArray]:
        """Path vertices (closing vertex included) and their arc lengths."""
        points = self.track.points
        tangent = np.gradient(points, axis=0)
        if self.track.closed:
            # Wrap the end tangents for a smooth loop.
            tangent[0] = tangent[-1] = points[1] - points[-2]
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
        path = points + offset_m * normal
        s = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        )
        return path, s

    def _point_at(self, s: float) -> NDArray[np.float64]:
        if self.track.closed:
            s = s % self._length
        s = float(np.clip(s, 0.0, self._length))
        return np.array(
            [
                np.interp(s, self._s, self._path[:, 0]),
                np.interp(s, self._s, self._path[:, 1]),
            ]
        )

    def update(
        self,
        state: WheeledControlState,
        request: DriverCommand | None,
        dt_s: float,
    ) -> WheelCommands:
        o, v = self.options, self.vehicle
        x, y, z, w = state.quaternion_xyzw
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        heading = np.array([np.cos(yaw), np.sin(yaw)])
        forward = float(heading @ state.velocity_world_mps[:2])
        projection = self.track.project(state.position_m[:2], self._segment)
        self._segment = int(projection.segment[0])
        s_here = float(projection.s_m[0])
        lookahead = max(o["min_lookahead_m"], o["lookahead_s"] * abs(forward))
        target = self._point_at(s_here + lookahead) - state.position_m[:2]
        bearing = np.arctan2(target[1], target[0]) - yaw
        curvature = 2 * np.sin(bearing) / max(np.linalg.norm(target), 1e-3)
        speed = self._target_speed(s_here)
        self.last_target_mps = speed
        return self._commands(forward, curvature, speed, state)

    def _target_speed(self, s_here: float) -> float:
        """Speed limit from the largest curvature within ``preview_m`` ahead."""
        o = self.options
        s, curvature = self._s, self._abs_curvature
        if self.track.closed:
            begin = s_here % self._length
            end = begin + o["preview_m"]
            first = int(np.searchsorted(s, begin))
            if end <= self._length:
                values = curvature[first : int(np.searchsorted(s, end, "right"))]
            else:
                wrapped = int(np.searchsorted(s, end - self._length, "right"))
                values = np.concatenate([curvature[first:], curvature[:wrapped]])
        else:
            first = int(np.searchsorted(s, s_here))
            values = curvature[
                first : int(np.searchsorted(s, s_here + o["preview_m"], "right"))
            ]
        kappa = float(np.max(values, initial=1e-6))
        lateral = o["grip_fraction"] * self.vehicle.tyre.friction_lateral * 9.81
        return float(min(o["max_speed_mps"], np.sqrt(lateral / max(kappa, 1e-6))))

    def _commands(
        self, forward: float, curvature: float, speed: float, state: WheeledControlState
    ) -> WheelCommands:
        v, o = self.vehicle, self.options
        grip = o["grip_fraction"] * v.tyre.friction_longitudinal * 9.81
        driven = max(int(self._driven.sum()), 1)
        braked = max(int(self._braked.sum()), 1)
        acceleration = float(
            np.clip(
                o["speed_gain_per_s"] * (speed - forward),
                -grip,
                grip * driven / v.wheel_count,
            )
        )
        force = acceleration * v.mass_kg
        limit = v.drive.max_torque_nm
        per_wheel = float(np.clip(force * v.tyre.radius_m / driven, -limit, limit))
        torque = np.full(v.wheel_count, per_wheel)
        brake = np.zeros(v.wheel_count)
        if acceleration < 0 and forward > 0.5:
            per_brake = min(
                -force * v.tyre.radius_m / braked, v.drive.max_brake_torque_nm
            )
            brake = np.where(self._braked, per_brake, 0.0)
            torque[:] = 0.0
        angles = np.zeros(v.wheel_count)
        if v.steering_layout == "skid":
            yaw_error = forward * curvature - float(state.body_rates_rad_s[2])
            torque = np.clip(torque - self._side * yaw_error * limit, -limit, limit)
        else:
            delta = np.arctan(curvature * v.wheelbase_m)
            steer = float(np.clip(delta / v.drive.max_steer_rad, -1.0, 1.0))
            angles = steering_angles(v, steer, "front")
        return WheelCommands(
            tuple(np.where(self._driven, torque, 0.0)), tuple(brake), tuple(angles)
        )

    def summary(self) -> dict[str, Any]:
        return dict(options=self.options, path_length_m=self._length)


def _curvature(path: NDArray[np.float64]) -> NDArray[np.float64]:
    """Signed curvature (1/m) at each path vertex (finite differences)."""
    d1 = np.gradient(path, axis=0)
    d2 = np.gradient(d1, axis=0)
    speed = np.maximum(np.linalg.norm(d1, axis=1), 1e-9)
    return np.asarray((d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / speed**3)
