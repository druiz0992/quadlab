"""Barometric pressure/altitude conversion and a seeded barometer error model.

Conversion uses the International Standard Atmosphere troposphere model,
matching Gazebo's AirPressure sensor:

    T(h) = T0 - L * h
    p(h) = P0 * (T0 / T(h)) ** -(g * M / (R * L))

with sea-level pressure P0 = 101325 Pa, temperature T0 = 288.15 K, lapse
rate L = 0.0065 K/m. Valid below 11 km. Altitude is height above mean sea
level in metres (the world's 650 m elevation plus local z, for Gazebo).

Real barometers measure pressure; weather and temperature shift the
pressure at a given height, so absolute pressure altitude drifts while
height changes stay accurate. ``BaroErrorModel`` adds, in metres of
altitude: a constant offset, a slow Gauss-Markov drift, and white noise,
then converts back to pressure. Optional airflow terms model propeller
downwash and forward airspeed acting on the static port.
"""

from collections.abc import Mapping
import math
from typing import Any

import numpy as np

SEA_LEVEL_PRESSURE_PA = 101325.0
SEA_LEVEL_TEMPERATURE_K = 288.15
LAPSE_RATE_K_PER_M = 0.0065
# g * M / (R * L): gravity 9.80665 m/s², molar mass 28.9644 kg/kmol,
# gas constant 8314.32 J/(kmol·K). Dimensionless, about 5.256.
_EXPONENT = 9.80665 * 28.9644 / (8314.32 * LAPSE_RATE_K_PER_M)


def altitude_to_pressure(altitude_m: float) -> float:
    """Standard-atmosphere static pressure (Pa) at an altitude above sea level."""
    temperature = SEA_LEVEL_TEMPERATURE_K - LAPSE_RATE_K_PER_M * altitude_m
    return SEA_LEVEL_PRESSURE_PA * (SEA_LEVEL_TEMPERATURE_K / temperature) ** -_EXPONENT


def pressure_to_altitude(pressure_pa: float) -> float:
    """Inverse of ``altitude_to_pressure``: pressure altitude (m above sea level)."""
    if not pressure_pa > 0:
        raise ValueError("Pressure must be positive")
    ratio = (pressure_pa / SEA_LEVEL_PRESSURE_PA) ** (1 / _EXPONENT)
    return SEA_LEVEL_TEMPERATURE_K * (1 - ratio) / LAPSE_RATE_K_PER_M


class BaroErrorModel:
    """Seeded offset + Gauss-Markov drift + white noise, in altitude metres.

    ``settings`` is the validated ``sensors.baro.error_model`` section:
    ``offset_m`` (signed constant), ``drift_std_m`` (stationary std) with
    ``drift_correlation_time_s``, and ``white_std_m`` (per sample). The drift
    is updated exactly over the elapsed simulation time and starts in its
    stationary distribution. A local RNG leaves global NumPy state unchanged.
    """

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.enabled: bool = settings["enabled"]
        self._rng = np.random.default_rng(settings["seed"])
        self._offset_m: float = settings["offset_m"]
        self._drift_std_m: float = settings["drift_std_m"]
        self._tau_s: float = settings["drift_correlation_time_s"]
        self._white_std_m: float = settings["white_std_m"]
        # Airflow on the static port; see apply().
        self._wash: float = settings["prop_wash_coefficient"]
        self._airspeed: float = settings["airspeed_coefficient"]
        self._drift_m = 0.0
        self._previous_time_ns: int | None = None
        self._samples = 0
        self._peak_error_m = 0.0

    def apply(
        self,
        pressure_pa: float,
        sim_time_ns: int,
        induced_velocity_mps: float = 0.0,
        airspeed_mps: float = 0.0,
    ) -> float:
        """Return the perturbed pressure (Pa); identity when disabled.

        ``induced_velocity_mps`` is the rotors' downwash velocity
        sqrt(T / 2ρA) for the current thrust; ``airspeed_mps`` the horizontal
        speed through the air. Both act as dynamic pressure on the static
        port: error = coefficient · v² / 2g metres (positive reads high).
        """
        if not self.enabled:
            return pressure_pa
        self._advance(sim_time_ns)
        error_m = (
            self._offset_m + self._drift_m + self._white_std_m * self._rng.normal()
        )
        error_m += self._wash * induced_velocity_mps**2 / (2 * 9.80665)
        error_m += self._airspeed * airspeed_mps**2 / (2 * 9.80665)
        self._samples += 1
        self._peak_error_m = max(self._peak_error_m, abs(error_m))
        return altitude_to_pressure(pressure_to_altitude(pressure_pa) + error_m)

    def _advance(self, sim_time_ns: int) -> None:
        previous = self._previous_time_ns
        if sim_time_ns < 0 or (previous is not None and sim_time_ns <= previous):
            raise ValueError(
                "Barometer timestamps must be nonnegative and strictly increasing"
            )
        self._previous_time_ns = sim_time_ns
        if previous is None:
            self._drift_m = self._drift_std_m * self._rng.normal()
            return
        dt_s = (sim_time_ns - previous) * 1e-9
        decay = math.exp(-dt_s / self._tau_s)
        drive = self._drift_std_m * math.sqrt(-math.expm1(-2 * dt_s / self._tau_s))
        self._drift_m = decay * self._drift_m + drive * self._rng.normal()

    def summary(self) -> dict[str, Any]:
        """Diagnostics for summary.json, for evaluation only."""
        return dict(
            enabled=self.enabled,
            samples=self._samples,
            offset_m=self._offset_m,
            final_drift_m=self._drift_m,
            peak_error_m=self._peak_error_m,
        )
