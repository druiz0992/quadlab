"""Example legged-robot controller (``LeggedController``): pose hold.

Select it with
``algorithms.controller: robotlab.control.examples.pose_hold_legged_controller:PoseHoldLeggedController``
(or your copy's ``package.module:Class``), and test it with
``robotlab.testing.plugins.check_legged_controller`` before running it in
MuJoCo.

It holds the start keyframe's pose and adds a proportional correction for
forward lean, so the structure of a controller stays visible (it is not a
balance controller and does not walk; ``request`` is ignored):

1. lean θ (rad, positive leaning forward, i.e. nose down about body +y)
   from world up in body axes u = Rᵀ ẑ: leaning forward tips u backwards,
   so θ = atan2(−u_x, u_z). With trunk truth (control
   state source ``truth``) R comes from its quaternion; otherwise from the
   newest IMU sample, using specific force f ≈ 9.81 u when the trunk is not
   accelerating (static approximation);
2. target_j = initial_j + K_j θ for each joint named in ``lean_gains``
   (rad per rad; signs depend on the model's joint axes), clipped to the
   joint's target bounds by the mission (counted, not silent).

Everything comes from ``state`` and ``context``; it must not use
simulator truth itself.
"""

from collections.abc import Mapping
import math
from typing import Any

import numpy as np

from ...core.algorithms import AlgorithmContext
from ...core.legged import JointTargets
from ..legged import LeggedControlState, WalkCommand

DEFAULTS: dict[str, Any] = dict(
    # Joint name -> lean gain (rad/rad); empty: hold the pose only.
    lean_gains={},
)


class PoseHoldLeggedController:
    """Pose hold plus lean feedback following ``LeggedController``."""

    def __init__(self, context: AlgorithmContext, options: Mapping[str, Any]) -> None:
        if context.legged is None:
            raise ValueError("PoseHoldLeggedController needs a legged robot")
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise ValueError(
                f"PoseHoldLeggedController: unknown options {sorted(unknown)}"
            )
        self.options = {**DEFAULTS, **options}
        robot = context.legged
        self._initial = np.array(robot.initial_target_rad)
        self._gains = np.zeros(robot.joint_count)
        for joint, gain in dict(self.options["lean_gains"]).items():
            if not math.isfinite(float(gain)):
                raise ValueError(f"lean_gains[{joint!r}] must be finite")
            self._gains[robot.index(joint)] = float(gain)
        self.last_lean_rad = 0.0

    def update(
        self, state: LeggedControlState, request: WalkCommand | None, dt_s: float
    ) -> JointTargets:
        lean = self._lean_rad(state)
        if lean is not None:
            self.last_lean_rad = lean
        return JointTargets(tuple(self._initial + self._gains * self.last_lean_rad))

    @staticmethod
    def _lean_rad(state: LeggedControlState) -> float | None:
        """Forward lean (module docs, step 1), or None without data."""
        if state.trunk is not None:
            x, y, z, w = state.trunk.quaternion_xyzw
            # Third row of R (world z in body axes = Rᵀ ẑ).
            up = np.array(
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
            )
        elif state.imu is not None:
            up = np.asarray(state.imu.payload.best_specific_force_mps2)
        else:
            return None
        return math.atan2(-float(up[0]), float(up[2]))

    def summary(self) -> dict[str, Any]:
        return dict(options=self.options, last_lean_rad=self.last_lean_rad)
