"""Plug-in contract for rotor-thrust flight controllers.

A controller is constructed as ``cls(context, options)`` with an
``AlgorithmContext`` and its ``algorithms.controller_options`` mapping. The
control loop (experiments/rotor_loop.py) calls ``update_position`` every
position-loop tick and ``update_attitude`` every control tick, with a
``ControlState`` that comes from simulator truth or from the estimator
(``algorithms.control_state_source``), and submits the returned rotor
thrusts. Requests are pilot sticks (``FlightCommand``) or a mission
reference (``ReferencePoint``).
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .manual import FlightCommand
from .references import ReferencePoint


@dataclass(frozen=True)
class ControlState:
    """State the controller flies on (world ENU, xyzw body-to-world,
    body-axis rates). ``source`` is "truth" or "estimate". ``airspeed_mps``:
    air-relative speed along body +x (truth), or the latest delivered pitot
    reading (estimate mode); None without one."""

    sim_time_ns: int
    position_m: NDArray[np.float64]
    velocity_world_mps: NDArray[np.float64]
    quaternion_xyzw: NDArray[np.float64]
    body_rates_rad_s: NDArray[np.float64]
    source: str = "truth"
    airspeed_mps: float | None = None


@runtime_checkable
class RotorController(Protocol):
    """See module docs."""

    saturated: bool  # last attitude update hit actuator limits

    @property
    def setpoint_m(self) -> NDArray[np.float64] | None:
        """Current position setpoint (take-off logic and logging), or None."""

    def update_position(
        self,
        state: ControlState,
        request: FlightCommand | ReferencePoint,
        dt_s: float,
    ) -> None:
        """Outer loop at the position rate; dt_s since the previous call."""

    def update_attitude(self, state: ControlState) -> NDArray[np.float64]:
        """Inner loop at the control rate; return rotor thrusts (N)."""

    def allow_landing(self) -> None:
        """Permit descent to the ground (landing sequences)."""

    # Optional: ``surface_commands() -> dict[str, float]`` returning aileron,
    # elevator, and rudder deflections (rad) for vehicles with ``fixed_wing``;
    # read after each attitude update and sent to the servos.
