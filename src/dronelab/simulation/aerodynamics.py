"""Simplified aerodynamic effects: ground effect, turbulence, body drag,
translational lift, and vortex ring state.

All are transport-free numerical models; Gazebo adapters apply their output.
Blade flapping is not modelled here: Gazebo's MulticopterMotorModel already
applies a rotor drag force and rolling moment proportional to rotor speed and
edgewise airspeed (``motors.rotor_drag_coefficient``), which approximates it.
"""

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation


def ground_effect_factor(rotor_height_m: float, rotor_radius_m: float) -> float:
    """Thrust ratio in/out of ground effect at equal rotor speed.

    Cheeseman-Bennett model: ``1 / (1 - (R / (4 z))²)`` for rotor height z
    above flat ground and rotor radius R. It diverges as z -> R/4 and is
    unreliable below about z = R/2, so height is clamped at R/2 (factor 4/3).
    The effect is about 2 % at z = 1.8 R and under 1 % above z = 2.5 R.
    """
    height = max(rotor_height_m, 0.5 * rotor_radius_m)
    return 1.0 / (1.0 - (rotor_radius_m / (4.0 * height)) ** 2)


class DrydenTurbulence:
    """Seeded first-order turbulence velocities in along/cross/vertical axes.

    Each axis is a Gauss-Markov process with stationary standard deviation
    ``intensity_mps[i]`` and correlation time ``length_scale_m[i] / U``, where
    U is the speed at which turbulence is carried past the vehicle (Taylor's
    frozen-turbulence hypothesis): the mean wind speed, at least
    ``min_advection_speed_mps``. This matches the Dryden along-wind spectrum;
    Dryden's lateral and vertical components are second order and are
    approximated here by first order. Updates happen on a fixed simulation-time
    grid of ``update_period_s``, so results do not depend on the caller's rate.
    """

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.enabled: bool = settings["enabled"]
        self._rng = np.random.default_rng(settings["seed"])
        self._std = np.asarray(settings["intensity_mps"], dtype=float)
        self._length = np.asarray(settings["length_scale_m"], dtype=float)
        self._period_ns = max(1, round(settings["update_period_s"] * 1e9))
        self._min_speed: float = settings["min_advection_speed_mps"]
        self._tick = -1
        self._state = np.zeros(3)

    @property
    def period_ns(self) -> int:
        return self._period_ns

    def advance(self, sim_time_ns: int, mean_wind_enu_mps: NDArray[np.float64]) -> bool:
        """Advance to the tick containing sim_time_ns; True if the value changed."""
        tick = sim_time_ns // self._period_ns
        if not self.enabled or tick <= self._tick:
            return False
        if self._tick < 0:
            # Start stationary: turbulence exists from the first sample.
            self._state = self._std * self._rng.normal(size=3)
            self._tick = tick
            return True
        speed = max(float(np.linalg.norm(mean_wind_enu_mps[:2])), self._min_speed)
        tau = self._length / speed
        dt = self._period_ns * 1e-9
        decay = np.exp(-dt / tau)
        # Exact discretization keeps the stationary std for any tau.
        drive = self._std * np.sqrt(-np.expm1(-2 * dt / tau))
        # Step every grid tick, including ones without a caller sample.
        for _ in range(tick - self._tick):
            self._state = decay * self._state + drive * self._rng.normal(size=3)
        self._tick = tick
        return True

    def velocity_enu(
        self, mean_wind_enu_mps: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Current turbulence rotated from along/cross/vertical into ENU.

        The along-wind axis follows the mean horizontal wind direction (east
        when the mean horizontal wind is zero).
        """
        if not self.enabled:
            return np.zeros(3)
        heading = (
            math.atan2(mean_wind_enu_mps[1], mean_wind_enu_mps[0])
            if np.any(mean_wind_enu_mps[:2])
            else 0.0
        )
        along, cross, vertical = self._state
        cos, sin = math.cos(heading), math.sin(heading)
        return np.array(
            [along * cos - cross * sin, along * sin + cross * cos, vertical]
        )


# Specific gas constant of dry air, J/(kg·K) (R / M = 8314.32 / 28.9644, the
# values of the barometer's standard atmosphere).
DRY_AIR_GAS_CONSTANT = 8314.32 / 28.9644


def site_air_density_kg_m3(elevation_m: float, temperature_c: float) -> float:
    """Dry-air density at a site: ``ρ = p / (R_d T)`` with ``p`` the
    standard-atmosphere pressure at ``elevation_m`` (the model Gazebo's
    barometer uses) and ``T`` the ambient temperature.

    650 m and 25 °C give 1.095 kg/m³; sea level and 15 °C give the standard
    1.225 kg/m³. Raises ValueError for a temperature at or below 0 K.
    """
    from ..sensors.barometer import altitude_to_pressure

    temperature_k = temperature_c + 273.15
    if not temperature_k > 0:
        raise ValueError("temperature must be above absolute zero")
    return altitude_to_pressure(elevation_m) / (DRY_AIR_GAS_CONSTANT * temperature_k)


def quadratic_body_drag_world(
    velocity_world_mps: ArrayLike,
    wind_world_mps: ArrayLike,
    quaternion_xyzw: ArrayLike,
    drag_area_body_m2: ArrayLike,
    air_density_kg_m3: float,
) -> NDArray[np.float64]:
    """Body drag force in world axes, quadratic in airspeed per body axis.

    ``F_b,i = −½ ρ (Cd·A)_i v_i |v_i|`` with v the airspeed (velocity minus
    wind) in body axes and ``drag_area_body_m2`` the drag area Cd·A per body
    axis (m²). Rotated back to world axes for Gazebo.
    """
    body = Rotation.from_quat(quaternion_xyzw)
    airspeed = np.asarray(velocity_world_mps, float) - np.asarray(wind_world_mps, float)
    v = body.inv().apply(airspeed)
    force = (
        -0.5 * air_density_kg_m3 * np.asarray(drag_area_body_m2, float) * v * np.abs(v)
    )
    return body.apply(force)


def hover_induced_velocity(
    thrust_n: float, rotor_radius_m: float, air_density_kg_m3: float
) -> float:
    """Momentum-theory induced velocity ``sqrt(T / (2 ρ π R²))`` (m/s)."""
    return math.sqrt(thrust_n / (2 * air_density_kg_m3 * math.pi * rotor_radius_m**2))


def translational_lift_factor(
    edgewise_speed_mps: float, induced_mps: float, max_gain: float
) -> float:
    """Thrust ratio from edgewise airflow at equal rotor speed.

    Smooth empirical shape ``1 + max_gain · (1 − v_i / sqrt(v_i² + v_h²))``:
    1 in hover, rising towards ``1 + max_gain`` once edgewise speed v_h is a
    few times the hover induced velocity v_i (the rotor meets clean air).
    """
    return 1.0 + max_gain * (
        1.0 - induced_mps / math.hypot(induced_mps, edgewise_speed_mps)
    )


def vortex_ring_factor(
    descent_speed_mps: float, induced_mps: float, thrust_loss: float
) -> float:
    """Mean thrust ratio in vortex ring state, before fluctuation.

    Descending along the rotor axis at speeds comparable to the hover induced
    velocity makes the rotor re-ingest its own wake. Empirical bell shape
    centred at descent = v_i with half-width 0.4·v_i:
    ``1 − thrust_loss · exp(−((d/v_i − 1) / 0.4)²)``; 1 when climbing.
    """
    if descent_speed_mps <= 0:
        return 1.0
    x = (descent_speed_mps / induced_mps - 1.0) / 0.4
    return 1.0 - thrust_loss * math.exp(-x * x)


class RotorAerodynamics:
    """Per-rotor thrust factors from translational lift and vortex ring state.

    ``settings`` is the validated ``aerodynamics`` section. Airspeed is the
    body's velocity relative to the wind (rotor lever arms ignored).
    Fluctuation in vortex ring state is seeded, independent per rotor, and
    scaled by how deep the rotor is in the state.
    """

    def __init__(
        self,
        settings: Mapping[str, Any],
        hover_thrust_n: float,
        rotor_radius_m: float,
        rotors: int,
    ) -> None:
        self._lift = settings["translational_lift"]
        self._ring = settings["vortex_ring"]
        self.enabled = bool(self._lift["enabled"] or self._ring["enabled"])
        self._induced = hover_induced_velocity(
            hover_thrust_n, rotor_radius_m, settings["air_density_kg_m3"]
        )
        self._rng = np.random.default_rng(self._ring["seed"])
        self._rotors = rotors
        self.max_lift_factor = 1.0
        self.min_ring_factor = 1.0

    @property
    def induced_velocity_mps(self) -> float:
        return self._induced

    def factors(
        self, airspeed_world_mps: ArrayLike, quaternion_xyzw: ArrayLike
    ) -> NDArray[np.float64]:
        """Thrust factor per rotor (1 = no effect)."""
        factors = np.ones(self._rotors)
        if not self.enabled:
            return factors
        axis = Rotation.from_quat(quaternion_xyzw).apply([0.0, 0.0, 1.0])
        airspeed = np.asarray(airspeed_world_mps, float)
        axial = float(airspeed @ axis)  # positive when climbing along the thrust axis
        edgewise = float(np.linalg.norm(airspeed - axial * axis))
        if self._lift["enabled"]:
            lift = translational_lift_factor(
                edgewise, self._induced, self._lift["max_gain"]
            )
            self.max_lift_factor = max(self.max_lift_factor, lift)
            factors *= lift
        if self._ring["enabled"]:
            ring = vortex_ring_factor(-axial, self._induced, self._ring["thrust_loss"])
            depth = (
                (1.0 - ring) / self._ring["thrust_loss"]
                if self._ring["thrust_loss"]
                else 0.0
            )
            noise = 1.0 + self._ring["fluctuation_std"] * depth * self._rng.normal(
                size=self._rotors
            )
            self.min_ring_factor = min(self.min_ring_factor, ring)
            factors *= np.clip(ring * noise, 0.0, None)
        return factors
