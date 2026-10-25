"""Plug-in contract for wheeled-vehicle controllers (rover, car, tractor).

A controller is constructed as ``cls(context, options)`` with an
``AlgorithmContext`` (``context.wheeled`` holds the nominal vehicle:
wheel layout, radius, actuator limits) and its
``algorithms.controller_options`` mapping. The mission calls ``update``
every control tick (``control.control_rate_hz``) and sends the returned
``WheelCommands`` (``core/wheeled.py``) to the simulated actuators; the
command holds until the next tick.

``state`` comes from simulator truth or from the estimator
(``algorithms.control_state_source``) and includes the wheel encoders and
steering-angle sensors. ``request`` is the driver's input (``DriverCommand``
from the keyboard) or None when nobody drives. A controller must not use
simulator truth itself.

Write your own (e.g. starting from ``control/examples/speed_yaw_wheeled_controller.py``) and check it with
``robotlab.testing.plugins.check_wheeled_controller``; the built-in keyboard
driver is ``control/wheeled_keyboard.py``.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..core.wheeled import WheelCommands

STEERING_MODES = ("front", "coordinated", "crab")


@dataclass(frozen=True)
class OpponentState:
    """Ground-truth state of another vehicle (simulator truth, labelled as
    such: a stand-in for perception) at ``sim_time_ns``; world ENU, xyzw."""

    name: str
    sim_time_ns: int
    position_m: NDArray[np.float64]
    velocity_world_mps: NDArray[np.float64]
    quaternion_xyzw: NDArray[np.float64]
    body_rates_rad_s: NDArray[np.float64]


@dataclass(frozen=True)
class WheeledControlState:
    """State the controller drives on: chassis in world ENU (xyzw body-to-world,
    body-axis rates) plus per-wheel encoder speed (rad/s, positive rolling
    forward) and measured steering angle (rad, positive left), in the
    vehicle's wheel order. ``source`` is "truth" or "estimate".
    ``opponents`` holds the other vehicles' ground-truth states of the
    same period (empty without opponents)."""

    sim_time_ns: int
    position_m: NDArray[np.float64]
    velocity_world_mps: NDArray[np.float64]
    quaternion_xyzw: NDArray[np.float64]
    body_rates_rad_s: NDArray[np.float64]
    wheel_speed_rad_s: NDArray[np.float64]
    steer_angle_rad: NDArray[np.float64]
    source: str = "truth"
    opponents: tuple[OpponentState, ...] = ()


@dataclass(frozen=True)
class DriverCommand:
    """Driver input, normalized: ``throttle`` in [−1, 1] (negative: reverse),
    ``brake`` in [0, 1], ``steer`` in [−1, 1] (positive: left),
    ``steering_mode`` for four-wheel steering (``STEERING_MODES``), and
    ``implement_lowered`` (tractor implement; ignored without one).

    Raises ValueError outside these ranges.
    """

    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    steering_mode: str = "front"
    implement_lowered: bool = False

    def __post_init__(self) -> None:
        if not (-1 <= self.throttle <= 1 and 0 <= self.brake <= 1):
            raise ValueError("DriverCommand throttle in [-1, 1], brake in [0, 1]")
        if not -1 <= self.steer <= 1:
            raise ValueError("DriverCommand steer must be in [-1, 1]")
        if self.steering_mode not in STEERING_MODES:
            raise ValueError(f"steering_mode must be one of {STEERING_MODES}")


@runtime_checkable
class WheeledController(Protocol):
    """See module docs."""

    def update(
        self,
        state: WheeledControlState,
        request: DriverCommand | None,
        dt_s: float,
    ) -> WheelCommands:
        """One control tick; ``dt_s`` since the previous call (0 at first)."""
        ...

    # Optional: ``summary() -> dict`` for summary.json (algorithms section).
