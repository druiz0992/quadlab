"""Rear three-point-hitch implement (tractor): lift, working depth, soil draft.

Draft force follows the ASABE D497 form (ASABE D497.7, *Agricultural
Machinery Management Data*, 2015)::

    D = F_i (A + B S + C S²) W T      [N]

with S the forward speed (km/h), W the working width (m), T the working
depth (cm), machine coefficients A, B, C, and the soil-texture factor F_i
(fine, medium, coarse). The built-in ``moldboard_plough`` uses A = 652,
B = 0, C = 5.1 and F_i = 1.0 / 0.70 / 0.45; ``custom`` takes A, B, C and
F_i from the configuration. The force acts at the tool point (bottom
centre of the implement) against the tool's horizontal velocity, faded in
by tanh(v / ``STATIC_SPEED_MPS``) so that a stopped tractor is not pushed
backwards (no draft without motion). Vertical soil forces are not modelled.

Lift: the tool moves at ``lift_rate_mps`` between raised (zero lift) and a
lowered position found by depth control: it goes down until the tool is
``working_depth_m`` below the ground under it, or ``lift_travel_m`` below
raised, whichever comes first; on uneven ground the depth follows the
terrain at the lift rate. Python owns the lift kinematics; Gazebo's
implement link (mass, visuals) follows the published lift position.

Frames: body axes from the tractor's centre of mass; ``hitch_position_m``
is the implement box centre when raised. Units: m, s, N.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .ground import GroundModel
from .model import ChassisState, rotation_matrix

# Speed scale (m/s) over which draft fades in from standstill.
STATIC_SPEED_MPS = 0.05
TEXTURES = ("fine", "medium", "coarse")
IMPLEMENT_TYPES = {
    # A (N/(m·cm)), B (N·h/(km·m·cm)), C (N·h²/(km²·m·cm)), F_i by texture.
    "moldboard_plough": (652.0, 0.0, 5.1, (1.0, 0.70, 0.45)),
}


@dataclass(frozen=True)
class ImplementTelemetry:
    """One period: lift (m, ≤ 0 below raised), tool depth in the soil (m),
    draft magnitude (N), tool speed (m/s), and whether it is lowered."""

    lift_m: float
    depth_m: float
    draft_n: float
    tool_speed_mps: float
    lowered: bool


def draft_coefficients(
    settings: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    """A, B, C, and F_i of the configured implement type and soil texture."""
    texture = TEXTURES.index(settings["texture"])
    if settings["type"] == "custom":
        a, b, c = settings["draft_a"], settings["draft_b"], settings["draft_c"]
        return a, b, c, settings["texture_factor"]
    a, b, c, factors = IMPLEMENT_TYPES[settings["type"]]
    return a, b, c, factors[texture]


def draft_n(
    coefficients: tuple[float, float, float, float],
    speed_mps: float,
    width_m: float,
    depth_m: float,
) -> float:
    """ASABE D497 draft (N) for speed (m/s), width (m), and depth (m)."""
    a, b, c, texture = coefficients
    speed_kmh = 3.6 * abs(speed_mps)
    return texture * (a + b * speed_kmh + c * speed_kmh**2) * width_m * 100.0 * depth_m


class ImplementModel:
    """Lift state and draft of one hitch implement (see module docs)."""

    def __init__(self, settings: Mapping[str, Any], ground: GroundModel) -> None:
        self.settings = settings
        self.ground = ground
        self.coefficients = draft_coefficients(settings)
        self.lowered = bool(settings["start_lowered"])
        self.lift_m = 0.0
        size = np.asarray(settings["size_m"], float)
        hitch = np.asarray(settings["hitch_position_m"], float)
        self._tool_raised_body = hitch - np.array([0.0, 0.0, size[2] / 2])
        self.telemetry = ImplementTelemetry(0.0, 0.0, 0.0, 0.0, self.lowered)
        self.worked_area_m2 = 0.0
        self.draft_energy_j = 0.0
        self.max_draft_n = 0.0

    def command(self, lowered: bool) -> None:
        """Raise (False) or lower (True) the implement."""
        self.lowered = bool(lowered)

    def step(
        self, chassis: ChassisState, dt_s: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Advance the lift; return the draft force (N, world) and its torque
        about the centre of mass (N·m, world)."""
        s = self.settings
        rotation = rotation_matrix(chassis.quaternion_xyzw)
        tool = self._tool_world(chassis, rotation)
        ground_z = float(
            self.ground.height(np.array([tool[0]]), np.array([tool[1]]))[0]
        )
        depth = ground_z - tool[2]
        self._move_lift(depth, dt_s)
        tool = self._tool_world(chassis, rotation)
        depth = max(ground_z - tool[2], 0.0)
        omega = rotation @ chassis.body_rates_rad_s
        velocity = chassis.velocity_world_mps + np.cross(
            omega, tool - chassis.position_m
        )
        horizontal = velocity[:2]
        speed = float(np.linalg.norm(horizontal))
        draft = draft_n(
            self.coefficients, speed, s["width_m"], min(depth, 2 * s["working_depth_m"])
        )
        draft *= float(np.tanh(speed / STATIC_SPEED_MPS))
        force = np.zeros(3)
        if speed > 0 and draft > 0:
            force[:2] = -draft * horizontal / speed
        torque = np.cross(tool - chassis.position_m, force)
        self._account(draft, speed, depth, dt_s)
        self.telemetry = ImplementTelemetry(
            self.lift_m, depth, draft, speed, self.lowered
        )
        return force, torque

    def _tool_world(self, chassis: ChassisState, rotation: NDArray) -> NDArray:
        body = self._tool_raised_body + np.array([0.0, 0.0, self.lift_m])
        return chassis.position_m + rotation @ body

    def _move_lift(self, depth_m: float, dt_s: float) -> None:
        """Rate-limited lift toward raised, or down to the working depth."""
        s = self.settings
        step = s["lift_rate_mps"] * dt_s
        if not self.lowered:
            self.lift_m = min(self.lift_m + step, 0.0)
            return
        error = s["working_depth_m"] - depth_m  # positive: go deeper
        move = float(np.clip(error, -step, step))
        self.lift_m = float(np.clip(self.lift_m - move, -s["lift_travel_m"], 0.0))

    def _account(self, draft: float, speed: float, depth: float, dt_s: float) -> None:
        if depth >= 0.5 * self.settings["working_depth_m"]:
            self.worked_area_m2 += self.settings["width_m"] * speed * dt_s
        self.draft_energy_j += draft * speed * dt_s
        self.max_draft_n = max(self.max_draft_n, draft)

    def summary(self) -> dict[str, Any]:
        return dict(
            type=self.settings["type"],
            lowered_at_end=self.lowered,
            worked_area_m2=self.worked_area_m2,
            draft_energy_j=self.draft_energy_j,
            max_draft_n=self.max_draft_n,
        )
