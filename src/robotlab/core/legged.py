"""Joint-level command contract and nominal properties for legged robots.

A ``JointTargets`` holds one position target per actuated joint, in the
robot's joint order (``LeggedProperties.joint_names``, the MJCF actuator
order): radians for hinge joints, metres for slide joints. The simulated
servos (MuJoCo position actuators) track the target with their own gains.

Structural errors (wrong length, non-finite values) raise ``ValueError``
at the actuator boundary. Targets outside a joint's command range are
clipped to it by the mission and counted per joint
(``summary.json.control.limited_commands``), never silently.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LeggedProperties:
    """Nominal legged robot, read from its model (no MuJoCo types).

    ``target_lower_rad``/``target_upper_rad`` bound the accepted targets;
    ``position_gain`` is each servo's proportional gain (N·m/rad, or N/m
    for slides); ``initial_target_rad`` the start keyframe's targets;
    ``mass_kg`` the total mass; ``control_rate_hz`` the controller rate.
    Arrays follow ``joint_names`` order.
    """

    name: str
    joint_names: tuple[str, ...]
    target_lower_rad: NDArray[np.float64]
    target_upper_rad: NDArray[np.float64]
    position_gain: NDArray[np.float64]
    initial_target_rad: NDArray[np.float64]
    mass_kg: float
    control_rate_hz: float

    def __post_init__(self) -> None:
        count = len(self.joint_names)
        for field in (
            "target_lower_rad",
            "target_upper_rad",
            "position_gain",
            "initial_target_rad",
        ):
            array = np.array(getattr(self, field), dtype=float)
            if array.shape != (count,):
                raise ValueError(f"LeggedProperties.{field} needs {count} values")
            array.flags.writeable = False
            object.__setattr__(self, field, array)
        if np.any(self.target_lower_rad > self.target_upper_rad):
            raise ValueError("LeggedProperties: a lower target bound exceeds its upper")

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    def index(self, joint: str) -> int:
        """Position of ``joint`` in the command order; raises ValueError."""
        try:
            return self.joint_names.index(joint)
        except ValueError:
            raise ValueError(f"{self.name} has no actuated joint {joint!r}") from None


@dataclass(frozen=True)
class JointTargets:
    """One position target per actuated joint (see module docs)."""

    position_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position_rad", tuple(float(v) for v in self.position_rad)
        )

    @classmethod
    def hold(cls, properties: LeggedProperties) -> "JointTargets":
        """The start keyframe's targets."""
        return cls(tuple(properties.initial_target_rad))

    def validate(self, joint_count: int) -> None:
        """Raise ValueError for a command the robot cannot accept at all."""
        if len(self.position_rad) != joint_count:
            raise ValueError(
                f"position_rad needs {joint_count} values (one per actuated joint)"
            )
        if not np.all(np.isfinite(self.position_rad)):
            raise ValueError("position_rad must be finite")
