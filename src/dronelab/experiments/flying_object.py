"""Keyboard-spawned flying object that circles the drone (effect key 7).

Orchestration only: builds a circular loop path around the drone's position
at the moment of the key press, spawns it in Gazebo as a gravity-free moving
obstacle, hands it to the moving-obstacle controller, and mirrors it in Rerun.
Pressing again removes it. Each spawn gets a new name so a quick
remove/re-add cannot collide with Gazebo's asynchronous deletion.
"""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

CIRCLE_POINTS = 8


def circle_entry(
    name: str, centre: np.ndarray, settings: Mapping[str, Any]
) -> dict[str, Any]:
    """Moving-obstacle entry looping a horizontal circle at the centre's altitude."""
    radius = settings["flying_object_radius_m"]
    angles = np.linspace(0, 2 * np.pi, CIRCLE_POINTS, endpoint=False)
    waypoints = [
        [
            float(centre[0] + radius * np.cos(a)),
            float(centre[1] + radius * np.sin(a)),
            float(centre[2]),
        ]
        for a in angles
    ]
    return dict(
        name=name,
        size=list(settings["flying_object_size_m"]),
        waypoints=waypoints,
        speed_mps=settings["flying_object_speed_mps"],
        mode="loop",
    )


class FlyingObjectSpawner:
    """Spawn/remove one circling flying object at a time."""

    def __init__(
        self,
        settings: Mapping[str, Any],
        *,
        entities: Any,
        moving: Any,
        recorder: Any,
        sdf_text: Callable[[Mapping[str, Any]], str],
        drone_state: Callable[[], tuple[np.ndarray, float]],
    ) -> None:
        self._settings = settings
        self._entities, self._moving, self._recorder = entities, moving, recorder
        self._sdf_text = sdf_text
        self._drone_state = drone_state  # () -> (position, sim time in s)
        self._count = 0
        self.active: str | None = None

    def set(self, on: bool) -> None:
        if on and self.active is None:
            self._spawn()
        elif not on and self.active is not None:
            self._remove()

    def _spawn(self) -> None:
        position, time_s = self._drone_state()
        self._count += 1
        name = f"flying_object_{self._count}"
        entry = circle_entry(name, np.asarray(position, float), self._settings)
        self._entities.spawn(self._sdf_text(entry))
        self._moving.add(entry, start_time_s=time_s)
        self._recorder.add_obstacle(name, entry["size"])
        self.active = name

    def _remove(self) -> None:
        name, self.active = self.active, None
        self._moving.remove(name)
        self._entities.remove(name)
        self._recorder.remove_obstacle(name)
