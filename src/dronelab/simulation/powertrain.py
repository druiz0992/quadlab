"""Motor-speed effects Gazebo's motor plugin cannot express: battery and ground effect.

Gazebo's MulticopterMotorModel produces thrust ``k_t ω²`` for a commanded
speed ω, with a fixed maximum speed. ``Powertrain`` sits between the
controller's thrust requests and the published motor speeds:

1. The controller's model converts thrust to speed: ω = sqrt(T / k_t).
2. **Battery** (optional): the achievable speed scales with loaded pack
   voltage, ``ω_max · V / V_full``, the usual motor-constant (Kv)
   approximation. Faster requests are limited. An empty pack (SoC 0) gives
   zero speed: the vehicle loses power.
3. **Ground effect** (optional): near the ground a rotor produces more thrust
   at the same speed. Gazebo cannot change ``k_t``, so the published speed is
   raised by sqrt(factor) to make Gazebo produce ``factor · k_t ω²``.
4. **Motor faults** (optional, simulation/motor_faults.py): a speed cap per
   rotor, and a thrust fraction applied like ground effect (a stopped motor
   publishes zero speed).

Power drawn is the rotor shaft power ``Σ c_m k_t ω³`` (reaction torque times
speed) divided by the powertrain efficiency, plus constant avionics power.
Rotor heights come from the vehicle's true pose over flat ground at z = 0;
this is physics state, never estimator input. Thread-safe: the thrust worker
calls ``motor_speeds`` while the mission loop calls ``update``.
"""

from collections.abc import Mapping
import math
import threading
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from ..core.vehicle import DroneProperties
from .aerodynamics import RotorAerodynamics, ground_effect_factor
from .battery import Battery
from .motor_faults import MotorFaults


class Powertrain:
    """Battery-limited, ground-effect-adjusted motor speeds for the built-in quad.

    ``battery`` is the validated ``battery`` section and ``ground_effect`` the
    ``aerodynamics.ground_effect`` section; disabled parts have no effect.
    """

    def __init__(
        self,
        properties: DroneProperties,
        battery: Mapping[str, Any],
        ground_effect: Mapping[str, Any],
        motor_faults: Mapping[str, Any] | None = None,
        *,
        overridable: bool = False,
        aerodynamics: Mapping[str, Any] | None = None,
        terrain: Any = None,
        ambient_c: float = 25.0,
        forward_figure_of_merit: float = 0.8,
    ) -> None:
        # Optional simulation.terrain.Terrain for ground effect; flat without it.
        self._terrain = terrain
        self._properties = properties
        # Keyboard effect toggles need the powertrain in the loop from start.
        self._overridable = overridable
        self._aero = (
            RotorAerodynamics(
                aerodynamics,
                properties.mass * 9.81 / len(properties.lift_rotors),
                properties.propeller_diameter / 2,
                len(properties.rotors),
            )
            if aerodynamics is not None
            else None
        )
        self._aero_factors = np.ones(len(properties.rotors))
        self._faults = MotorFaults(
            motor_faults or {"events": []}, len(properties.rotors)
        )
        self._thrust_fraction = np.ones(len(properties.rotors))
        self._speed_cap = np.ones(len(properties.rotors))
        self._battery = Battery(battery, ambient_c) if battery["enabled"] else None
        self._temperature_range = [ambient_c, ambient_c]
        self._efficiency: float = battery["powertrain_efficiency"]
        self._avionics_w: float = battery["avionics_power_w"]
        self._ground_effect = ground_effect["enabled"]
        self._rotor_radius_m = properties.propeller_diameter / 2
        self._rotor_offsets = np.array(properties.positions, float)
        # Ground effect and rotor aerodynamics apply to vertical rotors only.
        self._lift_mask = np.zeros(len(properties.rotors), bool)
        self._lift_mask[list(properties.lift_rotors)] = True
        # Pusher power model (see _power_w): air-relative speed along each axis.
        self._axes = np.array(properties.axes, float)
        self._axial_airspeed = np.zeros(len(properties.rotors))
        self._air_density = (aerodynamics or {}).get("air_density_kg_m3", 1.225)
        self._disk_area = np.pi * (properties.propeller_diameter / 2) ** 2
        self._figure_of_merit = forward_figure_of_merit
        self._lock = threading.Lock()
        self._factors = np.ones(len(properties.rotors))
        self._speeds = np.zeros(
            len(properties.rotors)
        )  # Physical speeds last commanded
        self._previous_time_ns: int | None = None
        self._stats = dict(
            limited_commands=0, max_ground_effect_factor=1.0, depleted_sim_time_ns=None
        )
        self._min_voltage_v = math.inf

    @property
    def enabled(self) -> bool:
        return (
            self._overridable
            or (self._aero is not None and self._aero.enabled)
            or self._battery is not None
            or self._ground_effect
            or self._faults.enabled
        )

    def set_motor_override(self, name: str, event: Mapping[str, Any] | None) -> None:
        """Add or remove a manual motor fault (see MotorFaults.set_override)."""
        with self._lock:
            self._faults.set_override(name, event)

    def update(
        self,
        sim_time_ns: int,
        position_m: ArrayLike,
        quaternion_xyzw: ArrayLike,
        airspeed_world_mps: ArrayLike | None = None,
    ) -> None:
        """Advance the battery to sim_time_ns; refresh ground-effect and fault state."""
        with self._lock:
            if self._aero is not None and self._aero.enabled:
                airspeed = (
                    np.zeros(3) if airspeed_world_mps is None else airspeed_world_mps
                )
                self._aero_factors = np.where(
                    self._lift_mask,
                    self._aero.factors(airspeed, quaternion_xyzw),
                    1.0,
                )
            # With overrides possible, refresh even when none are active, so
            # removing the last one restores full health.
            if self._faults.enabled or self._overridable:
                self._thrust_fraction, self._speed_cap = self._faults.health(
                    sim_time_ns
                )
            if airspeed_world_mps is not None and not self._lift_mask.all():
                body = Rotation.from_quat(quaternion_xyzw).inv()
                self._axial_airspeed = self._axes @ body.apply(airspeed_world_mps)
            if self._ground_effect:
                self._factors = self._rotor_factors(position_m, quaternion_xyzw)
                self._stats["max_ground_effect_factor"] = max(
                    self._stats["max_ground_effect_factor"], float(self._factors.max())
                )
            previous, self._previous_time_ns = self._previous_time_ns, sim_time_ns
            if self._battery is None or previous is None:
                return
            if sim_time_ns <= previous:
                raise ValueError("Powertrain time must strictly increase")
            state = self._battery.step((sim_time_ns - previous) * 1e-9, self._power_w())
            self._min_voltage_v = min(self._min_voltage_v, state.terminal_v)
            low, high = self._temperature_range
            self._temperature_range = [
                min(low, state.temperature_c),
                max(high, state.temperature_c),
            ]
            if self._battery.depleted and self._stats["depleted_sim_time_ns"] is None:
                self._stats["depleted_sim_time_ns"] = sim_time_ns

    def _rotor_factors(
        self, position_m: ArrayLike, quaternion_xyzw: ArrayLike
    ) -> NDArray[np.float64]:
        rotor_world = Rotation.from_quat(quaternion_xyzw).apply(
            self._rotor_offsets
        ) + np.asarray(position_m)
        ground = (
            np.zeros(len(rotor_world))
            if self._terrain is None
            else self._terrain.height(rotor_world[:, 0], rotor_world[:, 1])
        )
        heights = rotor_world[:, 2] - ground
        factors = np.array(
            [ground_effect_factor(h, self._rotor_radius_m) for h in heights]
        )
        return np.where(self._lift_mask, factors, 1.0)

    def _power_w(self) -> float:
        """Electrical power: shaft power / efficiency + avionics.

        Lift rotors: reaction torque × speed, c_m·k_t·ω³ (static model).
        Other rotors (pushers) see axial inflow V along their axis, so they
        use actuator-disk momentum theory: v_i = −V/2 + √((V/2)² + T/(2ρA)),
        P = T (V + v_i) / FM with figure of merit FM.
        """
        p = self._properties
        static = p.moment_ratio * p.thrust_coefficient * self._speeds**3
        thrust = p.thrust_coefficient * self._speeds**2
        inflow = np.maximum(self._axial_airspeed, 0.0)
        induced = -inflow / 2 + np.sqrt(
            (inflow / 2) ** 2 + thrust / (2 * self._air_density * self._disk_area)
        )
        forward = thrust * (inflow + induced) / self._figure_of_merit
        shaft_w = float(np.sum(np.where(self._lift_mask, static, forward)))
        return shaft_w / self._efficiency + self._avionics_w

    def _max_speed_rad_s(self) -> float:
        """Battery-limited physical motor speed; call with the lock held."""
        nominal = self._properties.max_rotor_speed
        if self._battery is None:
            return nominal
        if self._battery.depleted:
            return 0.0
        return nominal * self._battery.state.terminal_v / self._battery.full_voltage_v

    def motor_speeds(self, thrust_n: NDArray[np.float64]) -> NDArray[np.float64]:
        """Speeds to publish for validated thrust requests (N per rotor)."""
        with self._lock:
            requested = np.sqrt(
                np.asarray(thrust_n, dtype=float) / self._properties.thrust_coefficient
            )
            # Battery limit, then any per-rotor fault speed cap.
            limit = np.minimum(
                self._max_speed_rad_s(),
                self._speed_cap * self._properties.max_rotor_speed,
            )
            physical = np.minimum(requested, limit)
            if np.any(requested > limit + 1e-9):
                self._stats["limited_commands"] += 1
            # A stopped motor neither spins nor draws power.
            physical = np.where(self._thrust_fraction > 0, physical, 0.0)
            self._speeds = physical
            # Gazebo produces k_t ω²; scale ω so thrust includes ground effect
            # and any fault thrust fraction.
            return physical * np.sqrt(
                self._factors * self._aero_factors * self._thrust_fraction
            )

    def summary(self) -> dict[str, Any]:
        """Diagnostics for summary.json."""
        with self._lock:
            result: dict[str, Any] = dict(
                self._stats,
                ground_effect=self._ground_effect,
                motor_faults=self._faults.summary(),
            )
            if self._aero is not None and self._aero.enabled:
                result["rotor_aerodynamics"] = dict(
                    hover_induced_velocity_mps=self._aero.induced_velocity_mps,
                    max_translational_lift_factor=self._aero.max_lift_factor,
                    min_vortex_ring_factor=self._aero.min_ring_factor,
                )
            battery = self._battery
            if battery is not None:
                state = battery.state
                result["battery"] = dict(
                    final_soc=state.soc,
                    final_terminal_v=state.terminal_v,
                    min_terminal_v=(
                        None if math.isinf(self._min_voltage_v) else self._min_voltage_v
                    ),
                    energy_used_wh=state.energy_used_wh,
                    temperature_range_c=self._temperature_range,
                )
            return result

    def battery_state(self):
        """Latest BatteryState, or None without a battery (for recording)."""
        with self._lock:
            return None if self._battery is None else self._battery.state

    @property
    def has_battery(self) -> bool:
        return self._battery is not None

    @property
    def has_thermal_battery(self) -> bool:
        return self._battery is not None and self._battery.thermal

    def set_ambient(self, ambient_c: float) -> None:
        """Change the battery's ambient temperature (see Battery.set_ambient)."""
        with self._lock:
            if self._battery is None:
                return
            self._battery.set_ambient(ambient_c)
            low, high = self._temperature_range
            temperature = self._battery.temperature_c
            self._temperature_range = [min(low, temperature), max(high, temperature)]

    def adjust_battery(
        self, *, delta_soc: float = 0.0, refill: bool = False
    ) -> float | None:
        """Refill to full or change charge by ``delta_soc``; returns the new SoC.

        Returns None without a battery. Recovers from depletion when refilled.
        """
        with self._lock:
            if self._battery is None:
                return None
            soc = 1.0 if refill else self._battery.state.soc + delta_soc
            self._battery.set_soc(soc)
            if not self._battery.depleted:
                self._stats["depleted_sim_time_ns"] = None
            return self._battery.state.soc

    def available_thrust_fraction(self) -> NDArray[np.float64]:
        """Per-rotor maximum thrust available relative to healthy (for display)."""
        with self._lock:
            return self._thrust_fraction * self._speed_cap**2

    def rotor_thrust_factors(self) -> NDArray[np.float64]:
        """Combined aerodynamic thrust factors per rotor (ground effect, lift, ring)."""
        with self._lock:
            return self._factors * self._aero_factors

    def ground_effect_factors(self) -> NDArray[np.float64]:
        with self._lock:
            return self._factors.copy()
