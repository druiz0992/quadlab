"""Where a wheeled vehicle starts: scene spawn points or the configured pose.

A scene may list ``spawn_points`` (``[{"position_m": [x, y], "yaw_deg":
ψ}, ...]``, e.g. a race grid); the ego vehicle then starts at
``ground_vehicle.spawn_index`` and other vehicles take the following
points. Without spawn points the vehicle starts at
``simulation.spawn_position_m`` (x, y; z is ignored) facing
``ground_vehicle.spawn_yaw_deg``. The height puts the lowest tyre
``SPAWN_DROP_M`` above the ground (wheels unloaded; the vehicle settles
by its static sag in the first fraction of a second).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ground import GroundModel
from ...core.wheeled_vehicle import WheeledProperties

SPAWN_DROP_M = 0.01


@dataclass(frozen=True)
class SpawnPose:
    """Model pose: position (m, world ENU) and yaw (rad, about +z)."""

    position_m: tuple[float, float, float]
    yaw_rad: float


def validate_spawn_points(points: Any) -> None:
    """Raise ValueError unless ``points`` is a list of valid spawn points."""
    if not isinstance(points, list):
        raise ValueError("spawn_points must be a list")
    for index, point in enumerate(points):
        name = f"spawn_points[{index}]"
        if not isinstance(point, dict) or not set(point) <= {"position_m", "yaw_deg"}:
            raise ValueError(f"{name} needs position_m and optional yaw_deg")
        position = np.asarray(point.get("position_m"), float)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError(f"{name}.position_m must be two finite numbers")
        yaw = point.get("yaw_deg", 0.0)
        if isinstance(yaw, bool) or not isinstance(yaw, (int, float)):
            raise ValueError(f"{name}.yaw_deg must be a number")


def spawn_xy_yaw(
    settings: Mapping[str, Any], scene: Mapping[str, Any], offset: int = 0
) -> tuple[float, float, float]:
    """Spawn x, y (m) and yaw (rad) for the vehicle ``offset`` places after
    the ego vehicle (0: the ego). Raises ValueError for a missing point."""
    points: Sequence[Mapping[str, Any]] = scene.get("spawn_points", [])
    index = settings["ground_vehicle"]["spawn_index"] + offset
    if points:
        if index >= len(points):
            raise ValueError(
                f"scene has {len(points)} spawn points; vehicle needs index {index}"
            )
        x, y = points[index]["position_m"]
        return float(x), float(y), float(np.radians(points[index].get("yaw_deg", 0.0)))
    if offset:
        raise ValueError("additional vehicles need scene spawn_points")
    x, y, _ = settings["simulation"]["spawn_position_m"]
    return (
        float(x),
        float(y),
        float(np.radians(settings["ground_vehicle"]["spawn_yaw_deg"])),
    )


def spawn_pose(
    properties: WheeledProperties,
    ground: GroundModel,
    x: float,
    y: float,
    yaw_rad: float,
) -> SpawnPose:
    """Pose with the lowest tyre ``SPAWN_DROP_M`` above the highest ground
    point under any wheel (terrain or box top)."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    hubs = properties.hub_positions_m()
    world_xy = np.column_stack(
        (x + c * hubs[:, 0] - s * hubs[:, 1], y + s * hubs[:, 0] + c * hubs[:, 1])
    )
    high = np.full(len(hubs), 1e4)
    origins = np.column_stack((world_xy, high))
    down = np.tile([0.0, 0.0, -1.0], (len(hubs), 1))
    hits = ground.cast(origins, down, np.full(len(hubs), 2e4))
    if not hits.hit.all():
        raise ValueError("spawn position leaves a wheel outside the ground area")
    ground_z = float(np.max(high - hits.distance_m))
    z = ground_z + properties.tyre.radius_m - float(hubs[:, 2].min()) + SPAWN_DROP_M
    return SpawnPose((x, y, z), yaw_rad)


def kinematic_spawn_pose(
    properties: WheeledProperties,
    ground: GroundModel,
    x: float,
    y: float,
    yaw_rad: float,
) -> SpawnPose:
    """Pose of a kinematic vehicle (no suspension): wheel visuals, drawn at
    the ride drop below their hubs, touch the highest ground point under
    any wheel."""
    pose = spawn_pose(properties, ground, x, y, yaw_rad)
    drop = min(properties.ride_drop_m, properties.suspension.travel_m)
    z = pose.position_m[2] - SPAWN_DROP_M - drop
    return SpawnPose((pose.position_m[0], pose.position_m[1], z), yaw_rad)
