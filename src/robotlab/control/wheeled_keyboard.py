"""Built-in keyboard driver: map ``DriverCommand`` to per-wheel commands.

The simplest controller for each steering layout, with no feedback beyond
the steering geometry (it is the "manual" driver, not an autopilot):

- **drive**: throttle × ``max_torque_nm`` on every driven wheel (an open
  differential: equal torque per wheel); optional ``max_speed_mps`` fades
  the throttle out above that forward speed;
- **brake**: brake × ``max_brake_torque_nm`` on every braked wheel;
- **steering** (``ground_vehicle.steering_layout``):

  - ``skid``: no steered wheels; steer adds torque on the right side and
    removes it on the left (left turn for positive steer), so the vehicle
    turns on the spot with zero throttle;
  - ``ackermann``: front wheels turn about a point on the rear axle line,
    inner wheel more (Ackermann geometry); the steer input sets the angle of
    a virtual wheel on the centreline at the front axle;
  - ``four_wheel``: ``front`` (rear wheels straight), ``coordinated``
    (front and rear turn opposite about the middle of the wheelbase,
    roughly halving the turning radius), or ``crab`` (all wheels parallel:
    the vehicle moves diagonally without yawing).

Wheel angles are δ_i = atan((x_i − x_c) / (R − y_i)) for a turning centre
(x_c, R) in body axes, clipped to the steering lock. Keyboard input is
turned into a ``DriverCommand`` by ``DriverInput`` (key ramps, release to
centre), which is independent of Tk so it can be unit tested.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.algorithms import AlgorithmContext
from ..core.wheeled import WheelCommands
from ..core.wheeled_vehicle import WheeledProperties
from .wheeled import DriverCommand, WheeledControlState

OPTIONS = {"max_speed_mps"}
# Forward speed (m/s) below which holding "down" reverses instead of braking.
REVERSE_BELOW_MPS = 0.5
# Speed band (m/s) over which max_speed_mps fades the throttle to zero.
SPEED_LIMIT_BAND_MPS = 1.0


def steering_angles(
    properties: WheeledProperties, steer: float, mode: str
) -> NDArray[np.float64]:
    """Per-wheel steering angles (rad) for a normalized steer input."""
    angles = np.zeros(properties.wheel_count)
    steered = properties.mask("steered")
    if properties.steering_layout == "skid" or steer == 0 or not steered.any():
        return angles
    limit = properties.drive.max_steer_rad
    hubs = properties.hub_positions_m()
    x, y = hubs[:, 0], hubs[:, 1]
    front_x, rear_x = float(x.max()), float(x.min())
    reference = steer * limit
    if properties.steering_layout == "four_wheel" and mode == "crab":
        return np.where(steered, reference, 0.0)
    coordinated = properties.steering_layout == "four_wheel" and mode == "coordinated"
    centre_x = 0.5 * (front_x + rear_x) if coordinated else rear_x
    radius = (front_x - centre_x) / np.tan(reference)
    angles = np.arctan((x - centre_x) / (radius - y))
    if not coordinated:
        angles = np.where(x > centre_x + 1e-9, angles, 0.0)
    return np.where(steered, np.clip(angles, -limit, limit), 0.0)


class KeyboardDriver:
    """``WheeledController`` for the keyboard (see module docs)."""

    def __init__(self, context: AlgorithmContext, options: Mapping[str, Any]) -> None:
        if context.wheeled is None:
            raise ValueError("KeyboardDriver needs a wheeled vehicle context")
        unknown = set(options) - OPTIONS
        if unknown:
            raise ValueError(f"KeyboardDriver: unknown options {sorted(unknown)}")
        self.vehicle = context.wheeled
        self.max_speed_mps = float(options.get("max_speed_mps", 0.0))
        self._driven = self.vehicle.mask("driven")
        self._braked = self.vehicle.mask("braked")
        hubs = self.vehicle.hub_positions_m()
        self._side = np.sign(hubs[:, 1])  # +1 left, −1 right

    def update(
        self,
        state: WheeledControlState,
        request: DriverCommand | None,
        dt_s: float,
    ) -> WheelCommands:
        if request is None:
            return WheelCommands.zeros(self.vehicle.wheel_count)
        drive = self.vehicle.drive
        throttle = request.throttle * self._speed_factor(state, request.throttle)
        torque = np.full(self.vehicle.wheel_count, throttle * drive.max_torque_nm)
        if self.vehicle.steering_layout == "skid":
            # Positive steer: right side forward, left side back (left turn).
            torque = drive.max_torque_nm * np.clip(
                throttle - self._side * request.steer, -1.0, 1.0
            )
        angles = steering_angles(self.vehicle, request.steer, request.steering_mode)
        return WheelCommands(
            tuple(np.where(self._driven, torque, 0.0)),
            tuple(
                np.where(self._braked, request.brake * drive.max_brake_torque_nm, 0.0)
            ),
            tuple(angles),
        )

    def _speed_factor(self, state: WheeledControlState, throttle: float) -> float:
        """1, fading to 0 within SPEED_LIMIT_BAND_MPS above max_speed_mps."""
        if self.max_speed_mps <= 0 or throttle <= 0:
            return 1.0
        forward = _forward_speed(state)
        excess = (forward - self.max_speed_mps) / SPEED_LIMIT_BAND_MPS
        return float(np.clip(1.0 - excess, 0.0, 1.0))

    def summary(self) -> dict[str, Any]:
        return dict(
            layout=self.vehicle.steering_layout, max_speed_mps=self.max_speed_mps
        )


def _forward_speed(state: WheeledControlState) -> float:
    """Speed along body +x (m/s) from the state's attitude and velocity."""
    x, y, z, w = state.quaternion_xyzw
    heading = np.array(
        [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)]
    )
    return float(heading @ state.velocity_world_mps)


@dataclass
class DriverInput:
    """Turn held keys into a ``DriverCommand`` with smooth ramps.

    Keys: ``Up`` throttle; ``Down`` brake while moving forward faster than
    ``REVERSE_BELOW_MPS``, else reverse throttle; ``Left``/``Right`` steer
    (returns to centre on release, twice as fast); ``space`` full brake;
    ``1``/``2``/``3`` steering mode front/coordinated/crab (four-wheel
    steering); ``h`` toggles the implement (lowered/raised). Rates are per
    second of key hold (``ground_vehicle.keyboard``).
    """

    throttle_rate_per_s: float = 2.0
    steer_rate_per_s: float = 1.5
    throttle: float = 0.0
    steer: float = 0.0
    steering_mode: str = "front"
    implement_lowered: bool = False

    def update(
        self, keys: Collection[str], forward_mps: float, dt_s: float
    ) -> DriverCommand:
        """Advance the ramps by ``dt_s`` (wall or sim seconds) and return input."""
        reverse = "Down" in keys and forward_mps < REVERSE_BELOW_MPS
        target = 1.0 if "Up" in keys else -1.0 if reverse else 0.0
        step = self.throttle_rate_per_s * dt_s
        if target == 0.0:
            step *= 2  # release returns faster than press
        self.throttle += float(np.clip(target - self.throttle, -step, step))
        steer_target = ("Left" in keys) - ("Right" in keys)
        step = self.steer_rate_per_s * dt_s * (2 if steer_target == 0 else 1)
        self.steer += float(np.clip(steer_target - self.steer, -step, step))
        brake = 1.0 if "space" in keys or ("Down" in keys and not reverse) else 0.0
        if brake:
            self.throttle = min(self.throttle, 0.0) if not reverse else self.throttle
        return DriverCommand(
            float(np.clip(self.throttle, -1, 1)),
            brake,
            float(np.clip(self.steer, -1, 1)),
            self.steering_mode,
            self.implement_lowered,
        )

    def press(self, key: str) -> None:
        """One-shot keys: steering mode and implement toggle."""
        modes = {"1": "front", "2": "coordinated", "3": "crab"}
        if key in modes:
            self.steering_mode = modes[key]
        elif key == "h":
            self.implement_lowered = not self.implement_lowered
