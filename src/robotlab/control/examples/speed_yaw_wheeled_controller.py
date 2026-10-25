"""Example wheeled-vehicle controller (``WheeledController``): speed and yaw rate.

Select it with
``algorithms.controller: robotlab.control.examples.speed_yaw_wheeled_controller:SpeedYawWheeledController``
(or your copy's ``package.module:Class``), and test it with
``robotlab.testing.plugins.check_wheeled_controller`` before driving it in
Gazebo.

It holds a target forward speed with a proportional torque law and steers
with a proportional yaw-rate law, so the structure of a controller stays
visible (it is not tuned for every vehicle):

1. forward speed v_x = R ẋ_body (from the state);
2. drive torque T = K_v (v_ref − v_x) · m · r / n_driven (N·m per driven
   wheel: a desired acceleration times mass, shared over the driven
   wheels), clipped to the nominal torque limit; braking when T < 0 and
   moving forward;
3. steering: for a steered layout the front angle is K_ψ (ψ̇_ref − ψ̇) / v
   (bounded); for skid steering the yaw-rate error becomes a left/right
   torque difference.

With a ``DriverCommand`` request, throttle and steer set v_ref (×
``cruise_speed_mps``) and ψ̇_ref (× ``max_yaw_rate_rad_s``); without one,
it holds ``cruise_speed_mps`` straight ahead. It must not use simulator
truth itself: everything comes from ``state`` and ``context``.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from ...core.algorithms import AlgorithmContext
from ...core.wheeled import WheelCommands
from ..wheeled import DriverCommand, WheeledControlState
from ..wheeled_keyboard import steering_angles

DEFAULTS = dict(
    cruise_speed_mps=2.0,
    speed_gain_per_s=1.0,  # desired acceleration per m/s of speed error
    max_yaw_rate_rad_s=0.5,
    yaw_rate_gain=0.5,
)


class SpeedYawWheeledController:
    """Speed and yaw-rate P controller following ``WheeledController``."""

    def __init__(self, context: AlgorithmContext, options: Mapping[str, Any]) -> None:
        if context.wheeled is None:
            raise ValueError("SpeedYawWheeledController needs a wheeled vehicle")
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise ValueError(
                f"SpeedYawWheeledController: unknown options {sorted(unknown)}"
            )
        self.options = {**DEFAULTS, **options}
        self.vehicle = context.wheeled
        self._driven = self.vehicle.mask("driven")
        self._braked = self.vehicle.mask("braked")
        self._side = np.sign(self.vehicle.hub_positions_m()[:, 1])

    def update(
        self,
        state: WheeledControlState,
        request: DriverCommand | None,
        dt_s: float,
    ) -> WheelCommands:
        o, v = self.options, self.vehicle
        speed_ref = o["cruise_speed_mps"] * (request.throttle if request else 1.0)
        yaw_rate_ref = o["max_yaw_rate_rad_s"] * (request.steer if request else 0.0)
        x, y, z, w = state.quaternion_xyzw
        heading = np.array(
            [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)]
        )
        forward = float(heading @ state.velocity_world_mps)
        yaw_rate = float(state.body_rates_rad_s[2])
        acceleration = o["speed_gain_per_s"] * (speed_ref - forward)
        per_wheel = acceleration * v.mass_kg * v.tyre.radius_m / self._driven.sum()
        limit = v.drive.max_torque_nm
        torque = np.full(v.wheel_count, float(np.clip(per_wheel, -limit, limit)))
        brake = np.zeros(v.wheel_count)
        if per_wheel < 0 and forward > 0.2:
            brake = np.where(
                self._braked, min(-per_wheel, v.drive.max_brake_torque_nm), 0.0
            )
            torque[:] = 0.0
        yaw_error = yaw_rate_ref - yaw_rate
        angles = np.zeros(v.wheel_count)
        if v.steering_layout == "skid":
            torque = np.clip(
                torque - self._side * o["yaw_rate_gain"] * yaw_error * limit,
                -limit,
                limit,
            )
        else:
            steer = o["yaw_rate_gain"] * yaw_error / max(abs(forward), 1.0)
            angles = steering_angles(v, float(np.clip(steer, -1.0, 1.0)), "front")
        return WheelCommands(
            tuple(np.where(self._driven, torque, 0.0)), tuple(brake), tuple(angles)
        )

    def summary(self) -> dict[str, Any]:
        return dict(options=self.options)
