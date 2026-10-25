"""Low-battery failsafe: warn, then land or return home and land.

Mission logic (no Gazebo or viewer code). It reads the **measured** battery
state, as a real flight controller does: the battery monitor's remaining
charge and voltage when that sensor is enabled (so its counting errors
matter), otherwise ideal values from the battery model. Vehicle position and
vertical speed are the same truth feedback the controller already flies on.

States: ``normal`` → ``warned`` → (``returning`` →) ``landing`` → ``landed``.
Landing commands a constant descent; touchdown is declared when the vehicle
is within ``touchdown_height_m`` of the ground with vertical speed below
``touchdown_speed_mps`` for ``touchdown_hold_s``, after which the motors are
disarmed. Pilot and scripted commands are overridden from ``returning`` on.
"""

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ...control.manual import FlightCommand

ACTIONS = ("warn", "land", "return_and_land")


class BatteryFailsafe:
    """State machine producing override commands and events."""

    def __init__(
        self, settings: Mapping[str, Any], home_xy: ArrayLike, cells: int
    ) -> None:
        self._s = settings
        self._home = np.asarray(home_xy, float)
        self._critical_v = settings["critical_cell_voltage_v"] * cells
        self.state = "normal"
        self.events: list[dict[str, Any]] = []
        self._still_since_s: float | None = None

    def update(
        self,
        time_s: float,
        remaining_pct: float,
        voltage_v: float,
        position_m: ArrayLike,
        yaw_rad: float,
        vertical_speed_mps: float,
        height_above_ground_m: float,
    ) -> FlightCommand | None:
        """Advance the state machine; return an override command, or None."""
        if self.state == "normal" and remaining_pct <= self._s["warn_remaining_pct"]:
            self._enter("warned", time_s, remaining_pct, voltage_v)
        critical = voltage_v <= self._critical_v
        if (
            self.state in ("normal", "warned")
            and self._s["action"] != "warn"
            and (remaining_pct <= self._s["land_remaining_pct"] or critical)
        ):
            going_home = self._s["action"] == "return_and_land" and not critical
            self._enter(
                "returning" if going_home else "landing",
                time_s,
                remaining_pct,
                voltage_v,
            )
        if self.state == "returning":
            return self._return_home(
                time_s, np.asarray(position_m, float), yaw_rad, remaining_pct, voltage_v
            )
        if self.state == "landing":
            return self._land(
                time_s,
                vertical_speed_mps,
                height_above_ground_m,
                remaining_pct,
                voltage_v,
            )
        return None

    def _return_home(self, time_s, position, yaw, remaining_pct, voltage_v):
        offset = self._home - position[:2]
        distance = float(np.hypot(*offset))
        if distance < self._s["home_radius_m"]:
            self._enter("landing", time_s, remaining_pct, voltage_v)
            return FlightCommand()
        bearing = math.atan2(offset[1], offset[0])
        error = (bearing - yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_rate = float(np.clip(2.0 * error, -1.0, 1.0))
        # Fly forward only once roughly facing home; slow down near it.
        forward = (
            min(self._s["return_speed_mps"], 0.5 * distance)
            if abs(error) < 0.3
            else 0.0
        )
        return FlightCommand(forward_mps=forward, yaw_rate_rps=yaw_rate)

    def _land(self, time_s, vertical_speed, height, remaining_pct, voltage_v):
        still = abs(vertical_speed) < self._s["touchdown_speed_mps"]
        if height <= self._s["touchdown_height_m"] and still:
            self._still_since_s = (
                self._still_since_s if self._still_since_s is not None else time_s
            )
            if time_s - self._still_since_s >= self._s["touchdown_hold_s"]:
                self._enter("landed", time_s, remaining_pct, voltage_v)
                return FlightCommand()
        else:
            self._still_since_s = None
        return FlightCommand(climb_mps=-self._s["descent_mps"])

    def _enter(self, state, time_s, remaining_pct, voltage_v):
        self.state = state
        self.events.append(
            dict(
                time_s=time_s,
                state=state,
                remaining_pct=remaining_pct,
                voltage_v=voltage_v,
            )
        )

    @property
    def overriding(self) -> bool:
        return self.state in ("returning", "landing")

    def summary(self) -> dict[str, Any]:
        return dict(final_state=self.state, events=self.events)
