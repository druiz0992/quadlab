"""Live on/off switches for simulation effects during keyboard flight.

Each switch drives a manual override in an existing model, alongside any
scheduled behaviour from the configuration:

| Key | Effect | Model override |
| --- | --- | --- |
| 1 | Motor stop | Powertrain motor fault ``stop`` on ``motor_stop_rotor`` |
| 2 | Damaged propeller | Motor fault ``partial_thrust`` on ``damaged_rotor`` |
| 3 | Wind | Wind targets on/off (still air when off); needs ``wind.enabled`` |
| 4 | GPS blackout | Every GPS fix dropped |
| 5 | GPS multipath | Fixed ENU position jump on every fix |
| 6 | Gyro bias step | IMU ``bias_step`` on the gyro |
| 7 | Flying object | Spawn/remove an object circling the drone (needs a spawner) |
| 8 | Payload | Drop, or pick up when within reach (needs ``payload.enabled``) |

One-shot keys: 9 drains ``battery_drain_fraction`` of charge, 0 refills the
battery, and [ / ] lower / raise the ambient temperature by
``ambient_step_c`` (clamped to −40…60 °C). A temperature step moves the
battery pack and IMU die at once, as if starting with a pack soaked at the new
temperature; it needs ``battery.thermal`` or ``sensors.imu.temperature``.

Toggles happen in wall-clock time from a person, so keyboard runs are not
repeatable; scheduled faults remain the repeatable option. Every change is
returned as an event for logging.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Effect:
    name: str
    key: str
    label: str


EFFECTS = (
    Effect("motor_stop", "1", "Motor stop"),
    Effect("damaged_prop", "2", "Damaged propeller"),
    Effect("wind", "3", "Wind"),
    Effect("gps_blackout", "4", "GPS blackout"),
    Effect("gps_multipath", "5", "GPS multipath"),
    Effect("gyro_bias", "6", "Gyro bias step"),
    Effect("flying_object", "7", "Flying object"),
    Effect("payload", "8", "Payload attached"),
)


# One-shot actions (not toggles): key, name, label.
ACTIONS = (
    Effect("battery_drain", "9", "Battery −"),
    Effect("battery_refill", "0", "Battery full"),
    Effect("colder", "bracketleft", "Ambient −"),
    Effect("warmer", "bracketright", "Ambient +"),
)
AMBIENT_LIMITS_C = (-40.0, 60.0)


class EffectToggles:
    """Apply effect switches to the mission's model objects.

    ``settings`` is the validated ``keyboard_effects`` section. ``wind`` is
    the mission's WindCommands, or None when wind is not configured (the wind
    switch is then unavailable).
    """

    def __init__(
        self,
        settings: Mapping[str, Any],
        *,
        powertrain: Any,
        wind: Any,
        gps_dropout: Any,
        gps_multipath: Any,
        imu_faults: Any,
        flying_object: Any = None,
        payload: Any = None,
        imu_temperature: Any = None,
        ambient_c: float = 25.0,
    ) -> None:
        self._settings = settings
        self._powertrain = powertrain
        self._wind = wind
        self._gps_dropout = gps_dropout
        self._gps_multipath = gps_multipath
        self._imu_faults = imu_faults
        self._flying_object = flying_object
        # payload(on, sim_time_ns) -> event; on = attached. Pickup can fail.
        self._payload = payload
        self._imu_temperature = imu_temperature
        self.ambient_c = ambient_c
        # Configured wind starts on; every other effect starts off.
        # Configured wind and a carried payload start on; others start off.
        self.state = {effect.name: False for effect in EFFECTS}
        self.state["wind"] = wind is not None
        self.state["payload"] = payload is not None
        self.events: list[dict[str, Any]] = []

    def available(self, name: str) -> bool:
        if name == "wind":
            return self._wind is not None
        if name == "flying_object":
            return self._flying_object is not None
        if name == "payload":
            return self._payload is not None
        return True

    def set(self, name: str, on: bool, sim_time_ns: int) -> dict[str, Any] | None:
        """Switch an effect; returns the logged event, or None if unchanged/unavailable."""
        if not self.available(name) or self.state[name] == on:
            return None
        if name == "payload":
            result = self._payload(on, sim_time_ns)
            on = on if result and result.get("done") else self.state[name]
            event = dict(sim_time_ns=sim_time_ns, effect=name, on=on, detail=result)
            self.state[name] = on
            self.events.append(event)
            return event
        self._apply(name, on)
        self.state[name] = on
        event = dict(sim_time_ns=sim_time_ns, effect=name, on=on)
        self.events.append(event)
        return event

    def perform(self, name: str, sim_time_ns: int) -> dict[str, Any] | None:
        """Run a one-shot action; returns the logged event (None if unavailable)."""
        if name not in {a.name for a in ACTIONS} or not self.action_available(name):
            return None
        if name in ("colder", "warmer"):
            return self._step_ambient(name, sim_time_ns)
        if name == "battery_refill":
            soc = self._powertrain.adjust_battery(refill=True)
        else:
            soc = self._powertrain.adjust_battery(
                delta_soc=-self._settings["battery_drain_fraction"]
            )
        if soc is None:
            return None
        event = dict(sim_time_ns=sim_time_ns, effect=name, on=True, soc=soc)
        self.events.append(event)
        return event

    def action_available(self, name: str) -> bool:
        if name in ("colder", "warmer"):
            imu = self._imu_temperature
            return bool(
                (self._powertrain is not None and self._powertrain.has_thermal_battery)
                or (imu is not None and imu.enabled)
            )
        return self._powertrain is not None and self._powertrain.has_battery

    def _step_ambient(self, name: str, sim_time_ns: int) -> dict[str, Any] | None:
        step = self._settings["ambient_step_c"] * (1 if name == "warmer" else -1)
        ambient = float(np.clip(self.ambient_c + step, *AMBIENT_LIMITS_C))
        if ambient == self.ambient_c:
            return None
        self.ambient_c = ambient
        if self._powertrain is not None:
            self._powertrain.set_ambient(ambient)
        if self._imu_temperature is not None:
            self._imu_temperature.set_ambient(ambient)
        event = dict(sim_time_ns=sim_time_ns, effect=name, on=True, ambient_c=ambient)
        self.events.append(event)
        return event

    def _apply(self, name: str, on: bool) -> None:
        s = self._settings
        if name == "motor_stop":
            event = dict(type="stop", rotor=s["motor_stop_rotor"])
            self._powertrain.set_motor_override(name, event if on else None)
        elif name == "damaged_prop":
            event = dict(
                type="partial_thrust",
                rotor=s["damaged_rotor"],
                value=s["damaged_thrust_fraction"],
            )
            self._powertrain.set_motor_override(name, event if on else None)
        elif name == "wind":
            self._wind.set_enabled(on)
        elif name == "gps_blackout":
            self._gps_dropout.forced = on
        elif name == "flying_object":
            self._flying_object.set(on)
        elif name == "gps_multipath":
            offset = np.asarray(s["gps_multipath_offset_enu_m"], dtype=float)
            self._gps_multipath.forced_offset_enu_m = offset if on else None
        else:
            event = dict(
                type="bias_step", instrument="gyro", value=s["gyro_bias_step_rad_s"]
            )
            self._imu_faults.set_override(name, event if on else None)

    def summary(self) -> dict[str, Any]:
        return dict(
            final_state=dict(self.state), ambient_c=self.ambient_c, events=self.events
        )
