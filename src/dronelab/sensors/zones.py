"""Axis-aligned world boxes used to localize sensor faults (blockouts, multipath).

Zones are evaluated against the simulator's true vehicle position, which is
fault-model *state*, like simulation time for scheduled outages. The true
position is never passed to estimators through this module.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class BoxZone:
    """Closed box in world coordinates (metres, z-up), from min_m to max_m."""

    min_m: tuple[float, float, float]
    max_m: tuple[float, float, float]

    @classmethod
    def from_settings(cls, settings: Mapping[str, Sequence[float]]) -> "BoxZone":
        return cls(
            tuple(map(float, settings["min_m"])), tuple(map(float, settings["max_m"]))
        )

    def contains(self, position_m: ArrayLike) -> bool:
        position = np.asarray(position_m, dtype=float)
        return bool(np.all(position >= self.min_m) and np.all(position <= self.max_m))


def zones_from_settings(settings: Iterable[Mapping[str, Any]]) -> tuple[BoxZone, ...]:
    """Build immutable zones from validated configuration entries."""
    return tuple(BoxZone.from_settings(zone) for zone in settings)


def inside_any(zones: Sequence[BoxZone], position_m: ArrayLike | None) -> bool:
    """True if position is inside a zone; unknown position counts as outside."""
    return position_m is not None and any(zone.contains(position_m) for zone in zones)
