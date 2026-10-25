"""Python-computed external forces on the vehicle: body drag and wings.

Gazebo simulates rotors, contacts, and its native wind; RobotLab adds the
still-air quadratic body drag (``aerodynamics.body_drag``) and the
fixed-wing aerodynamic wrench (``fixed_wing``) as one persistent world-axis
wrench on the base link, updated at every odometry sample. Units: N, N·m,
m/s; world ENU axes.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...simulation.aerodynamics import quadratic_body_drag_world
from ...simulation.fixed_wing import FixedWingAero, ServoBank


class ExternalForces:
    """Body drag plus wing wrench, applied through a ``BodyForce`` publisher.

    ``body_force`` is None when neither effect is enabled (nothing is
    published). ``recorder`` logs to Rerun (or a fake).
    """

    def __init__(
        self,
        settings: dict[str, Any],
        body_force: Any | None,
        recorder: Any,
    ) -> None:
        aero = settings["aerodynamics"]
        self._drag = aero["body_drag"]
        self._air_density = aero["air_density_kg_m3"]
        self.fixed_wing = FixedWingAero(settings["fixed_wing"], self._air_density)
        self.servos = ServoBank(settings["fixed_wing"]["servos"])
        self._body_force = body_force
        self._recorder = recorder
        self.max_drag_n = 0.0

    @staticmethod
    def needed(settings: dict[str, Any]) -> bool:
        """Whether the configuration needs the wrench publisher."""
        return (
            settings["aerodynamics"]["body_drag"]["enabled"]
            or settings["fixed_wing"]["enabled"]
        )

    def update(
        self,
        sim_time_ns: int,
        velocity_world_mps: NDArray[np.float64],
        wind: Callable[[], NDArray[np.float64]],
        quaternion_xyzw: NDArray[np.float64],
        body_rates_rad_s: NDArray[np.float64],
    ) -> None:
        """Recompute and apply the wrench for one true state sample."""
        if self._body_force is None:
            return
        drag = (
            quadratic_body_drag_world(
                velocity_world_mps,
                wind(),
                quaternion_xyzw,
                self._drag["drag_area_m2"],
                self._air_density,
            )
            if self._drag["enabled"]
            else np.zeros(3)
        )
        wing_force, wing_moment = self._wing_wrench(
            sim_time_ns, velocity_world_mps, wind(), quaternion_xyzw, body_rates_rad_s
        )
        self._body_force.set(drag + wing_force, wing_moment)
        self.max_drag_n = max(self.max_drag_n, float(np.linalg.norm(drag)))
        self._recorder.log_vector("aero/body_drag_n", drag)

    def _wing_wrench(
        self,
        sim_time_ns: int,
        velocity: NDArray[np.float64],
        wind: NDArray[np.float64],
        quaternion: NDArray[np.float64],
        rates: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Wing force and moment (world axes) with current servo positions."""
        if not self.fixed_wing.enabled:
            return np.zeros(3), np.zeros(3)
        deflections = self.servos.update(sim_time_ns)
        force, moment = self.fixed_wing.wrench_world(
            velocity, wind, quaternion, rates, deflections
        )
        state = self.fixed_wing.last
        if state is None:
            raise RuntimeError("fixed-wing wrench computed without an aero state")
        self._recorder.log_vector(
            "aero/fixed_wing",
            [
                state.airspeed_mps,
                np.degrees(state.alpha_rad),
                np.degrees(state.beta_rad),
            ],
            ["airspeed_mps", "alpha_deg", "beta_deg"],
        )
        self._recorder.log_vector(
            "aero/surfaces_deg",
            [np.degrees(v) for v in deflections.values()],
            list(deflections),
        )
        return force, moment

    def summary(self) -> dict[str, float] | None:
        """``summary.json.body_drag``: largest drag force, or None if unused."""
        return None if self._body_force is None else dict(max_force_n=self.max_drag_n)
