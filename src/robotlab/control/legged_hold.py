"""Built-in legged controller ``hold``: keep the start keyframe's targets.

The default when ``algorithms.controller`` is empty. It ignores the state
and the request; a biped that cannot balance without feedback (the
Microduck) tips over with it.
"""

from collections.abc import Mapping
from typing import Any

from ..core.algorithms import AlgorithmContext
from ..core.legged import JointTargets
from .legged import LeggedControlState, WalkCommand


class HoldPose:
    """Return ``JointTargets.hold`` every period (``LeggedController``)."""

    def __init__(self, context: AlgorithmContext, options: Mapping[str, Any]) -> None:
        if context.legged is None:
            raise ValueError("HoldPose needs a legged robot (context.legged)")
        if options:
            raise ValueError(f"HoldPose takes no options, got {sorted(options)}")
        self._targets = JointTargets.hold(context.legged)

    def update(
        self, state: LeggedControlState, request: WalkCommand | None, dt_s: float
    ) -> JointTargets:
        return self._targets
