"""Immutable per-rotor command contracts, in SI units."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RotorThrusts:
    thrust_n: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "thrust_n", tuple(self.thrust_n))


@dataclass(frozen=True)
class RotorSpec:
    id: str
    index: int
    position_body_m: tuple[float, float, float]
    axis_body: tuple[float, float, float]
    direction: str
    max_thrust_n: float
    thrust_coefficient_n_s2: float


@dataclass(frozen=True)
class CommandReceipt:
    command_id: int
    submitted_sim_time_ns: int
    expires_sim_time_ns: int
    requested: RotorThrusts
    accepted: RotorThrusts
    motor_speed_targets_rad_s: tuple[float, ...]
    limited_rotor_indices: tuple[int, ...]
