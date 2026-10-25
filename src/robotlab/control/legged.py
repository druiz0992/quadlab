"""Plug-in contract for legged-robot controllers (MuJoCo bipeds).

A controller is constructed as ``cls(context, options)`` with an
``AlgorithmContext`` (``context.legged`` holds the nominal robot: joint
names and order, target bounds, servo gains, initial targets, control
rate; ``context.sensors`` the IMU mount) and its
``algorithms.controller_options`` mapping. The mission calls ``update``
every control period (``biped.control_rate_hz``) and applies the returned
``JointTargets`` (``core/legged.py``) to the servos; they hold until the
next period.

``state`` carries the newest sensor-bus samples (joint encoders, IMU) and,
only when ``algorithms.control_state_source`` is ``truth`` (the default),
the trunk's true pose and velocity, labelled as such: a stand-in for an
onboard attitude filter. With ``sensors`` the controller gets measurements
only. ``request`` is the operator's walking command, or None when nobody
commands. A controller must not use simulator truth itself.

Write your own (e.g. starting from ``control/examples/pose_hold_legged_controller.py``) and check it with
``robotlab.testing.plugins.check_legged_controller``.
"""

from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..core.legged import JointTargets
from ..core.measurements import ImuReading, Sample, JointStateReading

CONTROL_STATE_SOURCES = ("truth", "sensors")


@dataclass(frozen=True)
class TrunkState:
    """True trunk state (simulator truth, labelled as such): world z-up
    position and velocity, (x, y, z, w) body-to-world quaternion, and
    trunk-body angular rates."""

    position_m: NDArray[np.float64]
    velocity_world_mps: NDArray[np.float64]
    quaternion_xyzw: NDArray[np.float64]
    body_rates_rad_s: NDArray[np.float64]


@dataclass(frozen=True)
class LeggedControlState:
    """What the controller sees at ``sim_time_ns`` (see module docs).

    ``joints`` and ``imu`` are the newest delivered bus samples (None before
    the first one arrives; each carries its own measurement time).
    ``trunk`` is None unless the control state source is ``truth``.
    """

    sim_time_ns: int
    joints: Sample[JointStateReading] | None
    imu: Sample[ImuReading] | None
    trunk: TrunkState | None = None

    @property
    def source(self) -> str:
        """``truth`` when the trunk truth is included, else ``sensors``."""
        return "sensors" if self.trunk is None else "truth"


@dataclass(frozen=True)
class WalkCommand:
    """Operator walking request in the trunk's heading frame: forward and
    left speed (m/s) and yaw rate (rad/s, positive left). Raises ValueError
    for non-finite values."""

    forward_mps: float = 0.0
    left_mps: float = 0.0
    yaw_rate_rad_s: float = 0.0

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v)
            for v in (self.forward_mps, self.left_mps, self.yaw_rate_rad_s)
        ):
            raise ValueError("WalkCommand values must be finite")


@runtime_checkable
class LeggedController(Protocol):
    """See module docs."""

    def update(
        self,
        state: LeggedControlState,
        request: WalkCommand | None,
        dt_s: float,
    ) -> JointTargets:
        """One control period; ``dt_s`` since the previous call (0 at first)."""
        ...

    # Optional: ``summary() -> dict`` for summary.json (control section).
