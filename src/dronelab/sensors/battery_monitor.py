"""Battery monitor (power module): measured voltage and current, coulomb counting.

Drones measure pack voltage and current with a power module; the flight
controller integrates the measured current into consumed capacity and
estimates remaining charge. This model samples the simulated pack
(simulation/battery.py) on a fixed simulation-time grid and adds:

- voltage: white noise ``voltage_noise_std_v`` (V);
- current: constant offset ``current_offset_a`` plus white noise
  ``current_noise_std_a`` (A), as from an uncalibrated current sensor.

``consumed_mah`` integrates measured current; ``remaining_pct`` is
``100 · (1 − consumed / capacity)``, assuming the pack was **full at start**,
as a simple flight controller does. Starting a partly charged pack therefore
overestimates remaining charge, a realistic failure. Truth (the pack state)
generates the measurement; nothing else reaches consumers.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..simulation.battery import BatteryState


@dataclass(frozen=True)
class BatteryReading:
    sim_time_ns: int
    voltage_v: float
    current_a: float
    consumed_mah: float
    remaining_pct: float


class BatteryMonitor:
    """Sample a BatteryState at ``rate_hz`` with seeded measurement errors.

    ``settings`` is the validated ``sensors.battery_monitor`` section and
    ``capacity_mah`` the configured pack capacity used for the estimate.
    """

    def __init__(self, settings: Mapping[str, Any], capacity_mah: float) -> None:
        self.enabled: bool = settings["enabled"]
        self._period_ns = round(1e9 / settings["rate_hz"])
        self._rng = np.random.default_rng(settings["seed"])
        self._voltage_std: float = settings["voltage_noise_std_v"]
        self._current_std: float = settings["current_noise_std_a"]
        self._current_offset: float = settings["current_offset_a"]
        self._capacity_mah = capacity_mah
        self._last_tick = -1
        self._last_time_ns: int | None = None
        self._consumed_mah = 0.0
        self.samples = 0

    def sample(self, sim_time_ns: int, state: BatteryState) -> BatteryReading | None:
        """Return a reading when a grid time is reached, else None."""
        tick = sim_time_ns // self._period_ns
        if not self.enabled or tick <= self._last_tick:
            return None
        self._last_tick = tick
        voltage = state.terminal_v + self._voltage_std * self._rng.normal()
        current = (
            state.current_a
            + self._current_offset
            + self._current_std * self._rng.normal()
        )
        if self._last_time_ns is not None:
            # Coulomb counting over the interval: A · s / 3.6 = mAh.
            self._consumed_mah += (
                current * (sim_time_ns - self._last_time_ns) * 1e-9 / 3.6
            )
        self._last_time_ns = sim_time_ns
        self.samples += 1
        remaining = 100.0 * (1.0 - self._consumed_mah / self._capacity_mah)
        return BatteryReading(
            sim_time_ns, voltage, current, self._consumed_mah, remaining
        )

    def summary(self) -> dict[str, Any]:
        return dict(
            enabled=self.enabled, samples=self.samples, consumed_mah=self._consumed_mah
        )
