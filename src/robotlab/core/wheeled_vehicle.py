"""Nominal properties of a wheeled vehicle, built from ``ground_vehicle`` settings.

Value types only (no simulation): controllers and estimators receive these
through ``AlgorithmContext.wheeled``; the simulator's models live in
``simulation/wheeled/``.

Body frame: origin at the centre of mass (where Gazebo odometry is
reported and wrenches act), x forward, y left, z up. Each wheel's
``position_m`` is its hub centre with the suspension fully extended
(spring at free length); the strut mount sits ``travel_m`` above it and the
suspension acts along body −z. Wheel spin is positive when rolling forward
(about the wheel's +y axis).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

STEERING_LAYOUTS = ("ackermann", "skid", "four_wheel")


@dataclass(frozen=True)
class MagicFormulaCoefficients:
    """One direction's Magic Formula curve: stiffness B, shape C, curvature E
    (the peak D = μ F_z comes from the load); see simulation/wheeled/tyre.py.

    Raises ValueError for B <= 0, C outside (0, 2), or E > 1.
    """

    stiffness_b: float
    shape_c: float
    curvature_e: float

    def __post_init__(self) -> None:
        if not (self.stiffness_b > 0 and 0 < self.shape_c < 2):
            raise ValueError("Magic Formula needs B > 0 and 0 < C < 2")
        if not self.curvature_e <= 1:
            raise ValueError("Magic Formula curvature E must be <= 1")


@dataclass(frozen=True)
class TyreParameters:
    """Per-tyre constants (see module docs). Friction μ is the peak value on
    the reference surface; surface zones scale it per contact."""

    radius_m: float
    friction_longitudinal: float
    friction_lateral: float
    longitudinal: MagicFormulaCoefficients
    lateral: MagicFormulaCoefficients
    relaxation_longitudinal_m: float
    relaxation_lateral_m: float
    rolling_resistance: float
    low_speed_mps: float
    low_speed_damping_n_s_per_m: float

    @classmethod
    def from_settings(cls, tyre: Mapping[str, Any]) -> "TyreParameters":
        """Build from a validated ``ground_vehicle.tyre`` config section."""
        long, lat = tyre["longitudinal"], tyre["lateral"]
        return cls(
            radius_m=tyre["radius_m"],
            friction_longitudinal=long["friction"],
            friction_lateral=lat["friction"],
            longitudinal=MagicFormulaCoefficients(long["b"], long["c"], long["e"]),
            lateral=MagicFormulaCoefficients(lat["b"], lat["c"], lat["e"]),
            relaxation_longitudinal_m=long["relaxation_length_m"],
            relaxation_lateral_m=lat["relaxation_length_m"],
            rolling_resistance=tyre["rolling_resistance"],
            low_speed_mps=tyre["low_speed_mps"],
            low_speed_damping_n_s_per_m=tyre["low_speed_damping_n_s_per_m"],
        )


@dataclass(frozen=True)
class WheelProperties:
    """One wheel: hub position at full droop (body, m) and its actuators."""

    name: str
    position_m: tuple[float, float, float]
    steered: bool
    driven: bool
    braked: bool


@dataclass(frozen=True)
class SuspensionProperties:
    """Linear spring-damper per wheel, with a stiffer bump stop beyond
    ``travel_m`` of compression (N/m, N·s/m, m)."""

    stiffness_n_per_m: float
    damping_n_s_per_m: float
    travel_m: float
    bump_stiffness_n_per_m: float


@dataclass(frozen=True)
class DriveProperties:
    """Per-wheel actuator envelope: drive torque (N·m) limited by torque,
    power (W), and wheel speed (rad/s); brake torque (N·m); steering angle
    (rad) and slew rate (rad/s)."""

    max_torque_nm: float
    max_power_w: float
    max_wheel_speed_rad_s: float
    max_brake_torque_nm: float
    max_steer_rad: float
    max_steer_rate_rad_s: float


@dataclass(frozen=True)
class AeroProperties:
    """Quadratic aerodynamic drag and downforce areas (C·A, m²); the
    downforce acts at the front and rear axle lines by ``front_fraction``."""

    drag_area_m2: float
    downforce_area_m2: float
    downforce_front_fraction: float
    air_density_kg_m3: float


@dataclass(frozen=True)
class WheeledProperties:
    """Nominal vehicle (as designed); see module docs for frames."""

    kind: str
    model_name: str
    steering_layout: str
    mass_kg: float
    inertia_kg_m2: tuple[float, float, float]
    collision_size_m: tuple[float, float, float]
    collision_offset_m: tuple[float, float, float]
    wheels: tuple[WheelProperties, ...]
    wheel_width_m: float
    wheel_inertia_kg_m2: float
    tyre: TyreParameters
    suspension: SuspensionProperties
    drive: DriveProperties
    aero: AeroProperties
    substeps: int

    @property
    def wheel_count(self) -> int:
        return len(self.wheels)

    @property
    def wheel_names(self) -> tuple[str, ...]:
        return tuple(wheel.name for wheel in self.wheels)

    def hub_positions_m(self) -> NDArray[np.float64]:
        """Hub centres at full droop, body axes (n×3)."""
        return np.array([wheel.position_m for wheel in self.wheels], float)

    def mount_positions_m(self) -> NDArray[np.float64]:
        """Strut mounts (hub at full compression), body axes (n×3)."""
        mounts = self.hub_positions_m()
        mounts[:, 2] += self.suspension.travel_m
        return mounts

    def mask(self, attribute: str) -> NDArray[np.bool_]:
        """Boolean per wheel for ``steered``, ``driven``, or ``braked``."""
        return np.array([getattr(wheel, attribute) for wheel in self.wheels])

    @property
    def wheelbase_m(self) -> float:
        """Distance between the front-most and rear-most axle lines."""
        x = self.hub_positions_m()[:, 0]
        return float(x.max() - x.min())

    @property
    def track_m(self) -> float:
        """Distance between the left-most and right-most wheel centres."""
        y = self.hub_positions_m()[:, 1]
        return float(y.max() - y.min())

    @property
    def ride_drop_m(self) -> float:
        """Static sag (m): spring compression under an equal share of weight."""
        share = self.mass_kg * 9.81 / self.wheel_count
        return share / self.suspension.stiffness_n_per_m

    @classmethod
    def from_settings(
        cls, kind: str, settings: Mapping[str, Any], air_density_kg_m3: float
    ) -> "WheeledProperties":
        """Build from a validated ``ground_vehicle`` section (see config.py)."""
        g = settings
        chassis, drive, steering = g["chassis"], g["drive"], g["steering"]
        suspension, aero = g["suspension"], g["aero"]
        return cls(
            kind=kind,
            model_name=g["model_name"] or kind,
            steering_layout=g["steering_layout"],
            mass_kg=chassis["mass_kg"],
            inertia_kg_m2=tuple(chassis["inertia_kg_m2"]),
            collision_size_m=tuple(chassis["collision_size_m"]),
            collision_offset_m=tuple(chassis["collision_offset_m"]),
            wheels=tuple(
                WheelProperties(
                    name=w["name"],
                    position_m=tuple(w["position_m"]),
                    steered=w["steered"],
                    driven=w["driven"],
                    braked=w["braked"],
                )
                for w in g["wheels"]
            ),
            wheel_width_m=g["wheel"]["width_m"],
            wheel_inertia_kg_m2=g["wheel"]["inertia_kg_m2"],
            tyre=TyreParameters.from_settings(g["tyre"]),
            suspension=SuspensionProperties(
                suspension["stiffness_n_per_m"],
                suspension["damping_n_s_per_m"],
                suspension["travel_m"],
                suspension["bump_stiffness_n_per_m"],
            ),
            drive=DriveProperties(
                drive["max_torque_nm"],
                drive["max_power_w"],
                drive["max_wheel_speed_rad_s"],
                drive["max_brake_torque_nm"],
                steering["max_angle_rad"],
                steering["max_rate_rad_s"],
            ),
            aero=AeroProperties(
                aero["drag_area_m2"],
                aero["downforce_area_m2"],
                aero["downforce_front_fraction"],
                air_density_kg_m3,
            ),
            substeps=g["substeps"],
        )
