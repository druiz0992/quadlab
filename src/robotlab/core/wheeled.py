"""Actuator-level command contract for wheeled vehicles, in SI units.

A ``WheelCommands`` holds one entry per wheel, in the vehicle's wheel order
(``WheeledProperties.wheels``; the same order as the config's
``ground_vehicle.wheels``):

- ``drive_torque_nm``: motor torque at the wheel (N·m), positive drives
  forward; nonzero only on driven wheels;
- ``brake_torque_nm``: brake torque magnitude (N·m, >= 0), always opposing
  wheel spin; nonzero only on braked wheels;
- ``steer_angle_rad``: steering angle target (rad, positive turns the wheel
  left about body +z); nonzero only on steered wheels.

Structural errors (wrong length, non-finite values, a negative brake, or a
command on a wheel without that actuator) raise ``ValueError`` at the
actuator boundary. Physical limits (torque, power, wheel speed, brake,
steering angle and rate) are applied by the simulated actuators and
reported per wheel in ``WheelCommandReceipt.limited``.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WheelCommands:
    """One actuator command per wheel (see module docs)."""

    drive_torque_nm: tuple[float, ...]
    brake_torque_nm: tuple[float, ...]
    steer_angle_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("drive_torque_nm", "brake_torque_nm", "steer_angle_rad"):
            object.__setattr__(
                self, name, tuple(float(value) for value in getattr(self, name))
            )

    @classmethod
    def zeros(cls, wheel_count: int) -> "WheelCommands":
        """No torque, no brake, wheels straight."""
        zero = (0.0,) * wheel_count
        return cls(zero, zero, zero)

    def validate(
        self,
        driven: tuple[bool, ...],
        braked: tuple[bool, ...],
        steered: tuple[bool, ...],
    ) -> None:
        """Raise ValueError for a command the vehicle cannot accept at all."""
        count = len(driven)
        for name, values, allowed in (
            ("drive_torque_nm", self.drive_torque_nm, driven),
            ("brake_torque_nm", self.brake_torque_nm, braked),
            ("steer_angle_rad", self.steer_angle_rad, steered),
        ):
            if len(values) != count:
                raise ValueError(f"{name} needs {count} values (one per wheel)")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite")
            for index, (value, ok) in enumerate(zip(values, allowed)):
                if value and not ok:
                    raise ValueError(
                        f"{name}[{index}] = {value} on a wheel without that actuator"
                    )
        if min(self.brake_torque_nm, default=0.0) < 0:
            raise ValueError("brake_torque_nm must be >= 0")


@dataclass(frozen=True)
class WheelCommandReceipt:
    """What the actuators did with one command: ``accepted`` after limits,
    ``limited`` wheel indices per channel (``drive``, ``brake``, ``steer``),
    and the simulation time the command took effect (ns)."""

    sim_time_ns: int
    requested: WheelCommands
    accepted: WheelCommands
    limited: dict[str, tuple[int, ...]]
