"""LiPo battery pack: state of charge, loaded terminal voltage, and current.

Simplified equivalent circuit: an open-circuit voltage that depends on state
of charge (SoC), in series with an internal resistance R. For an electrical
load power P the terminal voltage V and current I satisfy

    V = OCV(SoC) - I * R,   I = P / V   =>   V = (OCV + sqrt(OCV² - 4 P R)) / 2

If P exceeds the pack's maximum deliverable power OCV² / (4 R), the pack
operates at that maximum (V = OCV / 2). SoC decreases by I·dt / capacity.

Optional temperature model (``battery.thermal``): the pack heats from its
internal losses I²R and cools towards ambient,
``C dT/dt = I²R(T) − h (T − T_ambient)``; internal resistance grows in the
cold, ``R(T) = R₂₅ · exp(k_R (25 − T))`` (k_R ≈ 0.028/°C doubles it at
0 °C), and usable capacity shrinks below 25 °C by ``k_C`` per °C. The pack
starts at ambient temperature.

Not modelled: ageing, relaxation (RC) dynamics, and cell imbalance. These are plausible parameters for teaching, not a calibrated pack.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

# Typical LiPo open-circuit voltage per cell versus state of charge.
_LIPO_SOC = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
_LIPO_OCV_V = np.array(
    [3.27, 3.61, 3.69, 3.73, 3.77, 3.79, 3.82, 3.87, 3.92, 3.97, 4.10, 4.20]
)
FULL_CELL_VOLTAGE_V = 4.2


@dataclass(frozen=True)
class BatteryState:
    """Instantaneous pack state after the latest update."""

    soc: float  # State of charge, 0 (empty) to 1 (full)
    open_circuit_v: float
    terminal_v: float  # Loaded voltage seen by the motors
    current_a: float
    power_w: float  # Electrical power drawn
    energy_used_wh: float
    temperature_c: float = 25.0


class Battery:
    """Integrate pack discharge from electrical power over simulation time.

    ``settings`` is the validated ``battery`` configuration section.
    """

    def __init__(self, settings: Mapping[str, Any], ambient_c: float = 25.0) -> None:
        self._cells: int = settings["cells"]
        self._capacity_as = settings["capacity_mah"] * 3.6  # mAh -> ampere-seconds
        self._resistance_25: float = settings["internal_resistance_ohm"]
        self._thermal = settings["thermal"]
        self._ambient_c = ambient_c
        self.temperature_c = ambient_c
        self._soc: float = settings["initial_soc"]
        # Infinite: charge never drops, but voltage still sags under load.
        self._infinite: bool = settings["infinite"]
        self._energy_j = 0.0
        self.state = self._solve(0.0)

    @property
    def full_voltage_v(self) -> float:
        """Unloaded voltage of a fully charged pack."""
        return self._cells * FULL_CELL_VOLTAGE_V

    @property
    def depleted(self) -> bool:
        return self._soc <= 0.0

    def open_circuit_voltage(self) -> float:
        return self._cells * float(np.interp(self._soc, _LIPO_SOC, _LIPO_OCV_V))

    def step(self, dt_s: float, power_w: float) -> BatteryState:
        """Draw ``power_w`` electrical watts for ``dt_s`` seconds."""
        if dt_s < 0 or power_w < 0:
            raise ValueError("Battery step requires nonnegative time and power")
        state = self._solve(power_w)
        if not self._infinite:
            capacity = self._capacity_as * self._capacity_factor()
            self._soc = max(0.0, self._soc - state.current_a * dt_s / capacity)
        if self._thermal["enabled"]:
            t = self._thermal
            heating = state.current_a**2 * self._resistance
            cooling = t["cooling_w_per_k"] * (self.temperature_c - self._ambient_c)
            self.temperature_c += (
                (heating - cooling) * dt_s / t["heat_capacity_j_per_k"]
            )
        self._energy_j += state.power_w * dt_s
        self.state = self._solve(power_w)
        return self.state

    @property
    def thermal(self) -> bool:
        return bool(self._thermal["enabled"])

    def set_ambient(self, ambient_c: float) -> None:
        """Change ambient temperature, shifting the pack by the same amount.

        An instant jump for experiments (as if swapping in a pack soaked at
        the new temperature), not a physical transient.
        """
        self.temperature_c += ambient_c - self._ambient_c
        self._ambient_c = ambient_c
        self.state = self._solve(self.state.power_w)

    def set_soc(self, soc: float) -> None:
        """Set the state of charge directly (e.g. keyboard refill/drain), 0–1."""
        self._soc = min(1.0, max(0.0, soc))
        self.state = self._solve(self.state.power_w)

    @property
    def _resistance(self) -> float:
        """Internal resistance at the current pack temperature (Ω)."""
        if not self._thermal["enabled"]:
            return self._resistance_25
        k = self._thermal["resistance_coefficient_per_c"]
        return self._resistance_25 * math.exp(k * (25.0 - self.temperature_c))

    def _capacity_factor(self) -> float:
        """Usable fraction of rated capacity (1 at or above 25 °C, ≥ 0.3)."""
        if not self._thermal["enabled"]:
            return 1.0
        cold = max(0.0, 25.0 - self.temperature_c)
        return max(0.3, 1.0 - self._thermal["capacity_coefficient_per_c"] * cold)

    def _solve(self, power_w: float) -> BatteryState:
        """Terminal voltage and current for the current SoC and load."""
        ocv = self.open_circuit_voltage()
        # Clamp to the maximum power the internal resistance allows.
        power = min(power_w, ocv**2 / (4 * self._resistance))
        voltage = (ocv + math.sqrt(max(0.0, ocv**2 - 4 * power * self._resistance))) / 2
        return BatteryState(
            soc=self._soc,
            open_circuit_v=ocv,
            terminal_v=voltage,
            current_a=power / voltage,
            power_w=power,
            energy_used_wh=self._energy_j / 3600,
            temperature_c=self.temperature_c,
        )
