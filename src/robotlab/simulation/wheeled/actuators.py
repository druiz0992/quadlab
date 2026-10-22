"""Simulated wheel actuators: drive envelope, brakes, rate-limited steering.

The drive motor at each driven wheel delivers at most
``min(max_torque_nm, max_power_w / |ω|)`` (constant torque, then constant
power: an electric-motor envelope; a combustion engine with a gearbox is
approximated by the same constant-power curve). Torque in the direction of
rotation tapers linearly to zero over the last ``TOP_SPEED_TAPER`` fraction
of ``max_wheel_speed_rad_s`` (a hard cut would chatter at top speed);
torque against the rotation (regenerative braking) is always allowed. Brakes are clipped to
``max_brake_torque_nm``. Steering targets are clipped to ±``max_steer_rad``
and the actual angle slews towards the target at ``max_steer_rate_rad_s``.

Every clipped value is reported per wheel (``WheelCommandReceipt.limited``);
structurally invalid commands raise ValueError (see ``core/wheeled.py``).
"""

import numpy as np
from numpy.typing import NDArray

from ...core.wheeled import WheelCommandReceipt, WheelCommands
from ...core.wheeled_vehicle import WheeledProperties

# Wheel speed below which the power limit is not applied (rad/s), avoiding
# the 1/ω singularity at standstill; the torque limit applies there anyway.
MIN_POWER_SPEED_RAD_S = 1e-3
# Fraction of the top wheel speed over which forward torque fades out.
TOP_SPEED_TAPER = 0.05


class WheelActuators:
    """Accepted commands and actual steering angles of ``vehicles`` identical
    vehicles (rows: vehicle-major, then wheel order; ``rows(i)`` selects one
    vehicle). A vehicle's command holds until its next one."""

    def __init__(self, properties: WheeledProperties, vehicles: int = 1) -> None:
        if vehicles < 1:
            raise ValueError("WheelActuators needs at least one vehicle")
        self.properties = properties
        self.vehicles = vehicles
        n = properties.wheel_count
        count = vehicles * n
        self._masks = tuple(
            tuple(bool(v) for v in properties.mask(name))
            for name in ("driven", "braked", "steered")
        )
        self.commands = [WheelCommands.zeros(n) for _ in range(vehicles)]
        self._drive_request = np.zeros(count)
        self._limited_request = np.zeros(count)
        self.steer_rad = np.zeros(count)
        self._steer_target = np.zeros(count)
        self.brake_nm = np.zeros(count)

    def rows(self, vehicle: int) -> slice:
        """Row slice of one vehicle's wheels."""
        n = self.properties.wheel_count
        return slice(vehicle * n, (vehicle + 1) * n)

    def accept(
        self, commands: WheelCommands, sim_time_ns: int, vehicle: int = 0
    ) -> WheelCommandReceipt:
        """Validate and limit a new command for ``vehicle``."""
        commands.validate(*self._masks)
        d = self.properties.drive
        rows = self.rows(vehicle)
        drive = np.asarray(commands.drive_torque_nm)
        brake = np.asarray(commands.brake_torque_nm)
        steer = np.asarray(commands.steer_angle_rad)
        limited_drive = np.clip(drive, -d.max_torque_nm, d.max_torque_nm)
        limited_brake = np.minimum(brake, d.max_brake_torque_nm)
        target = np.clip(steer, -d.max_steer_rad, d.max_steer_rad)
        self._drive_request[rows] = drive
        self._limited_request[rows] = limited_drive
        self.brake_nm[rows] = limited_brake
        self._steer_target[rows] = target
        accepted = WheelCommands(
            tuple(limited_drive), tuple(limited_brake), tuple(target)
        )
        self.commands[vehicle] = accepted
        return WheelCommandReceipt(
            sim_time_ns,
            commands,
            accepted,
            dict(
                drive=_changed(drive, limited_drive),
                brake=_changed(brake, limited_brake),
                steer=_changed(steer, target),
            ),
        )

    def drive_torque(self, wheel_speed_rad_s: NDArray[np.float64]) -> NDArray:
        """Delivered drive torque (N·m) per row at the current wheel speeds."""
        d = self.properties.drive
        request = self._limited_request
        speed = np.abs(wheel_speed_rad_s)
        power_limit = d.max_power_w / np.maximum(speed, MIN_POWER_SPEED_RAD_S)
        torque = np.minimum(np.maximum(request, -power_limit), power_limit)
        # Near top speed, fade out torque that would spin the wheel faster.
        taper = np.maximum(
            (d.max_wheel_speed_rad_s - speed)
            / (TOP_SPEED_TAPER * d.max_wheel_speed_rad_s),
            0.0,
        )
        accelerating = torque * wheel_speed_rad_s > 0
        return np.where(accelerating, torque * np.minimum(taper, 1.0), torque)

    def slew_steering(self, dt_s: float) -> None:
        """Move the actual angles towards their targets at the rate limit."""
        step = self.properties.drive.max_steer_rate_rad_s * dt_s
        self.steer_rad = self.steer_rad + np.clip(
            self._steer_target - self.steer_rad, -step, step
        )

    def limited_now(self, wheel_speed_rad_s: NDArray[np.float64]) -> tuple[int, ...]:
        """Rows whose drive torque the power or speed limit cuts right now."""
        return _changed(self._limited_request, self.drive_torque(wheel_speed_rad_s))


def _changed(requested: NDArray, accepted: NDArray) -> tuple[int, ...]:
    return tuple(int(i) for i in np.flatnonzero(~np.isclose(requested, accepted)))
