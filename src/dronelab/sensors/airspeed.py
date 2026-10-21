"""Synthetic pitot-static airspeed sensor.

Gazebo's airspeed sensor ignores DroneLab's Python-side wind targets, so the
measurement is synthesized from truth, as for optical flow. The pitot tube
points along body +x: it senses the dynamic pressure of the air-relative
velocity component along that axis,

    Δp = scale · ½ ρ max(u_air, 0)² + offset + N(0, noise²)        (Pa)

and reports indicated airspeed IAS = √(2 max(Δp, 0) / ρ₀) with sea-level
density ρ₀ = 1.225 kg/m³ (as a real airspeed sensor does; true airspeed
differs with altitude and temperature through ρ). During a ``blockage``
interval [start, end] (iced or blocked pitot) Δp reads only the offset and
noise. Sampled at ``rate_hz`` on simulation time; seeded local RNG.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

SEA_LEVEL_DENSITY_KG_M3 = 1.225


@dataclass(frozen=True)
class AirspeedReading:
    """Differential pressure (Pa) and indicated airspeed (m/s)."""

    sim_time_ns: int
    differential_pressure_pa: float
    indicated_airspeed_mps: float


class AirspeedSensor:
    """Pitot measurement from air-relative body velocity (see module docs)."""

    def __init__(self, settings: Mapping[str, Any], air_density_kg_m3: float) -> None:
        self.enabled: bool = settings["enabled"]
        self._period_ns = round(1e9 / settings["rate_hz"])
        self._scale = settings["scale"]
        self._offset = settings["offset_pa"]
        self._noise = settings["noise_std_pa"]
        self._blockage = [
            (round(a * 1e9), round(b * 1e9)) for a, b in settings["outages_s"]
        ]
        self._rho = air_density_kg_m3
        self._rng = np.random.default_rng(settings["seed"])
        self._next_ns: int | None = None
        self.samples = 0

    def sample(
        self, sim_time_ns: int, airspeed_body_mps: ArrayLike
    ) -> AirspeedReading | None:
        """A reading when one is due at ``sim_time_ns``, else None."""
        if not self.enabled:
            return None
        if self._next_ns is not None and sim_time_ns < self._next_ns:
            return None
        self._next_ns = sim_time_ns + self._period_ns
        u = max(float(np.asarray(airspeed_body_mps, float)[0]), 0.0)
        blocked = any(a <= sim_time_ns < b for a, b in self._blockage)
        dynamic = 0.0 if blocked else self._scale * 0.5 * self._rho * u * u
        pressure = dynamic + self._offset + self._noise * self._rng.normal()
        ias = math.sqrt(2 * max(pressure, 0.0) / SEA_LEVEL_DENSITY_KG_M3)
        self.samples += 1
        return AirspeedReading(sim_time_ns, pressure, ias)

    def summary(self) -> dict[str, Any]:
        return dict(enabled=self.enabled, samples=self.samples)
