"""Automatic take-off for missions that start on the ground (rotor-thrust mode).

Mission logic, not a controller: until the controller's altitude setpoint
reaches the take-off altitude, pilot or scripted commands are replaced by a
pure climb request at a fixed rate; the existing controller does the flying.
After that, commands pass through unchanged.

The setpoint, not the vehicle's height, ends the climb: the controller's
minimum-altitude rule lifts its setpoint at start, so the setpoint leads the
vehicle, and stopping on vehicle height left the hover about 0.3 m high.
"""

from ..control.manual import FlightCommand


class TakeoffSequence:
    """Override commands with a climb until the setpoint first reaches ``altitude_m``."""

    def __init__(self, altitude_m: float, climb_mps: float) -> None:
        self._altitude_m = altitude_m
        self._climb = FlightCommand(climb_mps=climb_mps)
        self.complete = False
        self.complete_time_s: float | None = None

    def command(
        self, altitude_m: float, time_s: float, requested: FlightCommand
    ) -> FlightCommand:
        """Return the command at ``time_s`` given the controller's altitude setpoint."""
        if self.complete:
            return requested
        if altitude_m >= self._altitude_m:
            self.complete = True
            self.complete_time_s = time_s
            return requested
        return self._climb
