"""Scheduled motor and propeller failures for the rotor-thrust powertrain.

Each event affects one rotor (index 0–3, actuator order) during
``[start_s, end_s)`` absolute simulation seconds (``end_s: null`` = until the
end of the run):

| type             | value                          | effect                                   |
| ---------------- | ------------------------------ | ---------------------------------------- |
| ``stop``         | (none)                         | Motor/ESC failure: zero speed and thrust |
| ``partial_thrust`` | remaining thrust fraction [0, 1) | Damaged propeller: same speed, less thrust |
| ``speed_limit``  | speed fraction (0, 1)          | Weak or overheated motor: speed capped at value × maximum |

Overlapping events on one rotor combine: thrust fractions multiply and the
lowest speed cap wins. The controller is not informed, as on a real vehicle.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

EVENT_TYPES = ("stop", "partial_thrust", "speed_limit")


class _MotorFault:
    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.type: str = settings["type"]
        self.rotor: int = settings["rotor"]
        self.value: float = settings.get("value", 0.0)
        self.start_ns = round(settings["start_s"] * 1e9)
        end_s = settings["end_s"]
        self.end_ns = None if end_s is None else round(end_s * 1e9)
        self.first_ns: int | None = None
        self.last_ns: int | None = None

    def active(self, sim_time_ns: int) -> bool:
        return self.start_ns <= sim_time_ns and (
            self.end_ns is None or sim_time_ns < self.end_ns
        )

    def summary(self) -> dict[str, Any]:
        return dict(
            type=self.type,
            rotor=self.rotor,
            value=self.value,
            first_active_sim_time_ns=self.first_ns,
            last_active_sim_time_ns=self.last_ns,
        )


class MotorFaults:
    """Per-rotor thrust fractions and speed caps at a simulation time.

    ``settings`` is the validated ``motors.faults`` section.
    """

    def __init__(self, settings: Mapping[str, Any], rotor_count: int) -> None:
        self._events = [_MotorFault(event) for event in settings["events"]]
        self._rotors = rotor_count
        # Manual overrides (e.g. keyboard toggles), active while present.
        self._overrides: dict[str, _MotorFault] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._events or self._overrides)

    def set_override(self, name: str, event: Mapping[str, Any] | None) -> None:
        """Add (event settings without start/end) or remove (None) an override."""
        if event is None:
            self._overrides.pop(name, None)
        else:
            self._overrides[name] = _MotorFault(dict(event, start_s=0.0, end_s=None))

    def health(
        self, sim_time_ns: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (thrust fraction, speed cap fraction) per rotor; 1 = healthy."""
        thrust = np.ones(self._rotors)
        speed_cap = np.ones(self._rotors)
        for event in [*self._events, *self._overrides.values()]:
            if not event.active(sim_time_ns):
                continue
            event.first_ns = sim_time_ns if event.first_ns is None else event.first_ns
            event.last_ns = sim_time_ns
            if event.type == "stop":
                thrust[event.rotor] = 0.0
                speed_cap[event.rotor] = 0.0
            elif event.type == "partial_thrust":
                thrust[event.rotor] *= event.value
            else:
                speed_cap[event.rotor] = min(speed_cap[event.rotor], event.value)
        return thrust, speed_cap

    def summary(self) -> dict[str, Any]:
        return dict(
            enabled=self.enabled, events=[event.summary() for event in self._events]
        )
