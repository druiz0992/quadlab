"""Mission clock: when a reference trajectory starts and what it asks for.

Mission logic, not a controller. The reference (control/references.py) is
built when the mission starts, from the vehicle position at that moment
(waypoint missions begin there). The start is the first control tick, or
the tick after take-off completes when the vehicle starts on the ground.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..control.references import Reference, ReferencePoint, build_reference


class MissionRunner:
    """Sample the configured ``mission`` reference on mission time."""

    def __init__(self, mission: Mapping[str, Any]) -> None:
        self._settings = mission
        self.reference: Reference | None = None
        self.start_s: float | None = None
        self.start_position_m: list[float] | None = None

    @property
    def enabled(self) -> bool:
        return self._settings["type"] != "none"

    def command(self, time_s: float, position_m: ArrayLike) -> ReferencePoint:
        """Reference at ``time_s`` (seconds since the run's first state).

        The first call starts the mission at ``time_s`` from ``position_m``.
        """
        if self.reference is None or self.start_s is None:
            reference = build_reference(self._settings, position_m)
            if reference is None:
                raise ValueError("mission type none has no reference to fly")
            self.start_s = time_s
            self.start_position_m = np.asarray(position_m, float).tolist()
            self.reference = reference
        return self.reference.sample(time_s - self.start_s)

    def sample(self, time_s: float) -> ReferencePoint | None:
        """Reference at ``time_s`` for evaluation; None before the start."""
        if self.reference is None or self.start_s is None or time_s < self.start_s:
            return None
        return self.reference.sample(time_s - self.start_s)

    def summary(self) -> dict[str, Any]:
        return dict(
            type=self._settings["type"],
            start_s=self.start_s,
            start_position_m=self.start_position_m,
            duration_s=None if self.reference is None else self.reference.duration_s,
        )
