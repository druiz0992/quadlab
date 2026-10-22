"""Moving obstacles that patrol waypoint paths at constant speed.

Path math is transport-free: ``PatrolPath`` gives the desired position and
velocity at any simulation time. ``MovingObstacleCommands`` turns that into
Gazebo VelocityControl commands, adding feedback on the measured pose so
obstacles stay on their path over many laps (velocity commands alone would
accumulate the error of switching segments only at control ticks).

Scene JSON entries (``moving_obstacles``):
``{"name", "size": [x, y, z], "waypoints": [[x, y, z], …], "speed_mps",
"mode": "pingpong" | "loop"}``. ``pingpong`` goes back and forth along the
waypoints; ``loop`` returns from the last waypoint to the first.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

MODES = ("pingpong", "loop")
POSITION_GAIN_PER_S = 2.0  # Feedback on path error, 1/s.


def validate_moving_obstacle(entry: Mapping[str, Any]) -> None:
    keys = {"name", "size", "waypoints", "speed_mps", "mode"}
    if set(entry) != keys:
        raise ValueError(f"moving obstacle requires exactly {sorted(keys)}")
    size = np.asarray(entry["size"], dtype=float)
    points = np.asarray(entry["waypoints"], dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or size.min() <= 0:
        raise ValueError(f"{entry['name']}: size needs three positive values")
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 2
        or not np.isfinite(points).all()
    ):
        raise ValueError(
            f"{entry['name']}: waypoints need at least two [x, y, z] points"
        )
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) == 0):
        raise ValueError(f"{entry['name']}: consecutive waypoints must differ")
    if not entry["speed_mps"] > 0 or entry["mode"] not in MODES:
        raise ValueError(
            f"{entry['name']}: speed must be positive and mode one of {MODES}"
        )


class PatrolPath:
    """Constant-speed motion along a polyline of waypoints."""

    def __init__(
        self, waypoints: Sequence[Sequence[float]], speed_mps: float, mode: str
    ) -> None:
        points = np.asarray(waypoints, dtype=float)
        if mode == "loop":
            points = np.vstack((points, points[:1]))
        self._points = points
        self._lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        self._cumulative = np.concatenate(([0.0], np.cumsum(self._lengths)))
        self._total = float(self._cumulative[-1])
        self._speed = speed_mps
        self._mode = mode

    def state(self, time_s: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Desired (position, velocity) at ``time_s`` after the start."""
        distance = self._speed * time_s
        direction = 1.0
        if self._mode == "pingpong":
            period = 2 * self._total
            distance %= period
            if distance > self._total:
                distance, direction = period - distance, -1.0
        else:
            distance %= self._total
        index = min(
            int(np.searchsorted(self._cumulative, distance, side="right")) - 1,
            len(self._lengths) - 1,
        )
        along = distance - self._cumulative[index]
        unit = (self._points[index + 1] - self._points[index]) / self._lengths[index]
        return self._points[index] + along * unit, direction * self._speed * unit


class MovingObstacleCommands:
    """Publish VelocityControl commands for each obstacle (Gazebo adapter)."""

    def __init__(self, node: Any, entries: Sequence[Mapping[str, Any]]) -> None:
        from gz.msgs10.twist_pb2 import Twist

        self._twist = Twist
        self._node = node
        self._paths: dict[str, PatrolPath] = {}
        self._start_s: dict[str, float] = {}
        self._publishers: dict[str, Any] = {}
        self._poses: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        self.max_path_error_m = 0.0
        for entry in entries:
            self.add(entry, start_time_s=0.0)

    @property
    def names(self) -> set[str]:
        return set(self._paths)

    def add(self, entry: Mapping[str, Any], start_time_s: float) -> None:
        """Start commanding an obstacle; its path time starts at ``start_time_s``."""
        name = entry["name"]
        self._paths[name] = PatrolPath(
            entry["waypoints"], entry["speed_mps"], entry["mode"]
        )
        self._start_s[name] = start_time_s
        self._publishers[name] = self._node.advertise(
            f"/model/{name}/cmd_vel", self._twist
        )

    def remove(self, name: str) -> None:
        """Stop commanding an obstacle (its model is removed separately)."""
        for table in (self._paths, self._start_s, self._publishers, self._poses):
            table.pop(name, None)

    def on_pose(
        self,
        name: str,
        position: NDArray[np.float64],
        quaternion_xyzw: NDArray[np.float64],
    ) -> None:
        """Record a measured obstacle pose (from the world pose stream)."""
        if name in self._paths:
            self._poses[name] = (position, quaternion_xyzw)

    def update(self, time_s: float) -> None:
        """Command every obstacle towards its path state at ``time_s``."""
        from scipy.spatial.transform import Rotation

        for name, path in self._paths.items():
            desired, velocity = path.state(max(0.0, time_s - self._start_s[name]))
            command_world = velocity
            orientation = Rotation.identity()
            if name in self._poses:
                position, quaternion = self._poses[name]
                error = desired - position
                self.max_path_error_m = max(
                    self.max_path_error_m, float(np.linalg.norm(error))
                )
                command_world = velocity + POSITION_GAIN_PER_S * error
                orientation = Rotation.from_quat(quaternion)
            # VelocityControl takes the command in the model's own axes.
            linear = orientation.inv().apply(command_world)
            message = self._twist()
            message.linear.x, message.linear.y, message.linear.z = map(float, linear)
            self._publishers[name].publish(message)
