"""Fixed-rate rotor-thrust control loop with modeled state and command latency.

Connects explicit (truth) state samples, the numerical ``ManualController``,
the timing helpers, and the exclusive thrust handle. It has no transport or
viewer dependencies, so it can be driven by tests with fake handles.

Per state sample (simulation time t):

1. Store the sample; advance the thrust service clock to t.
2. If a grid tick is due, take the newest state at least ``state_latency_s``
   old, run the position loop (every ``position_every`` ticks) and the
   attitude loop, and queue the thrusts for ``command_latency_s``.
3. Submit the newest queued command whose release time has arrived.

Warm-up: the vehicle spawns airborne, so before latency has elapsed the
oldest state is used and the first computed command is repeated, as if the
pipeline were already full. Without latency there is no warm-up.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...control.interface import ControlState, RotorController
from ...control.manual import FlightCommand, ManualController, ManualRotorController
from ...control.references import ReferencePoint
from ...control.timing import DelayLine, FixedRateTicker, StateBuffer, Tick
from ...core.commands import RotorThrusts

# Truth-fed control state; kept under its original name for callers.
TruthState = ControlState


@dataclass
class _Pending:
    """A computed command waiting in the delay line, with its log fields."""

    thrust_n: NDArray[np.float64]
    record: dict[str, Any] = field(default_factory=dict)


def _ns(seconds: float) -> int:
    return round(seconds * 1e9)


class RotorControlLoop:
    """Run a ManualController on a fixed simulation-time grid.

    ``control`` is the RotorThrustControl service (for its clock) and
    ``handle`` its acquired exclusive handle. ``timing`` is the validated
    ``control`` configuration section. ``tolerance_ns`` absorbs timestamp
    quantization (use half a physics step).
    """

    def __init__(
        self,
        controller: RotorController | ManualController,
        control: Any,
        handle: Any,
        timing: Mapping[str, Any],
        *,
        tolerance_ns: int,
    ) -> None:
        # Plain ManualController instances are adapted to the plug-in contract.
        self._controller: RotorController = (
            ManualRotorController.wrap(controller)
            if isinstance(controller, ManualController)
            else controller
        )
        self._control = control
        self._handle = handle
        self._timing = timing
        self._tolerance_ns = tolerance_ns
        rate = timing["control_rate_hz"]
        position_rate = timing["position_rate_hz"] or rate
        self._position_every = round(rate / position_rate)
        self._ticker = FixedRateTicker(rate, tolerance_ns=tolerance_ns)
        self._states: StateBuffer[TruthState] = StateBuffer(
            _ns(timing["state_latency_s"])
        )
        self._commands: DelayLine[_Pending] = DelayLine(
            _ns(timing["command_latency_s"])
        )
        self._first_state: TruthState | None = None
        self._warmup: _Pending | None = None
        self._last_position_tick: Tick | None = None
        self.disarmed = False
        self.stats: dict[str, Any] = dict(
            position_updates=0,
            commands_released=0,
            superseded_before_release=0,
            warmup_commands=0,
            saturated_ticks=0,
            max_state_age_ms=0.0,
        )

    def on_state(
        self,
        state: TruthState,
        pilot: Callable[[float], FlightCommand | ReferencePoint],
    ) -> list[dict[str, Any]]:
        """Process one state sample; return log records of submitted commands.

        ``pilot(tick_time_s)`` returns the pilot request at a tick time: a
        ``FlightCommand`` (velocity/yaw-rate sticks) or a ``ReferencePoint``
        (mission trajectory) for the position loop.
        """
        now = state.sim_time_ns
        self._states.push(now, state)
        self._first_state = self._first_state or state
        self._control.update_clock(now)
        if self.disarmed:
            return []
        tick = self._ticker.offer(now)
        if tick is not None:
            self._run_tick(tick, pilot)
        return self._release(now)

    def _run_tick(
        self, tick: Tick, pilot: Callable[[float], FlightCommand | ReferencePoint]
    ) -> None:
        sample = self._states.available(tick.sim_time_ns)
        # Warm-up: no sample is old enough yet, so use the oldest one.
        state = self._first_state if sample is None else sample[1]
        if state is None:
            raise RuntimeError("control tick before the first state sample")
        self.stats["max_state_age_ms"] = max(
            self.stats["max_state_age_ms"],
            (tick.sim_time_ns - state.sim_time_ns) * 1e-6,
        )
        command = pilot(tick.sim_time_ns * 1e-9)
        if self._position_due(tick):
            self._update_position(tick, state, command)
        thrusts = np.asarray(self._controller.update_attitude(state), float)
        self.stats["saturated_ticks"] += int(self._controller.saturated)
        pending = _Pending(thrusts, self._record(tick, state, thrusts, command))
        self._commands.push(tick.sim_time_ns, pending)
        self._warmup = self._warmup or pending

    def _position_due(self, tick: Tick) -> bool:
        last = self._last_position_tick
        return last is None or tick.index - last.index >= self._position_every

    def _update_position(
        self, tick: Tick, state: TruthState, command: FlightCommand | ReferencePoint
    ) -> None:
        last = self._last_position_tick
        dt = tick.dt_s if last is None else (tick.sim_time_ns - last.sim_time_ns) * 1e-9
        self._controller.update_position(state, command, dt)
        self._last_position_tick = tick
        self.stats["position_updates"] += 1

    def _record(
        self,
        tick: Tick,
        state: TruthState,
        thrusts: NDArray[np.float64],
        command: FlightCommand | ReferencePoint,
    ) -> dict[str, Any]:
        controller = self._controller
        if isinstance(command, ReferencePoint):
            pilot = dict(
                reference_position_m=command.position_m.tolist(),
                reference_velocity_mps=command.velocity_mps.tolist(),
                reference_yaw_rad=command.yaw_rad,
            )
        else:
            pilot = dict(
                forward_mps=command.forward_mps,
                yaw_rate_rps=command.yaw_rate_rps,
                climb_mps=command.climb_mps,
            )
        return dict(
            time=tick.sim_time_ns * 1e-9,
            tick_sim_time_ns=tick.sim_time_ns,
            state_sim_time_ns=state.sim_time_ns,
            missed_ticks=tick.missed,
            thrust_n=thrusts.tolist(),
            target=(
                None
                if controller.setpoint_m is None
                else controller.setpoint_m.tolist()
            ),
            yaw_target=getattr(controller, "yaw", None),
            state_source=state.source,
            saturated=controller.saturated,
            pilot=pilot,
        )

    def _release(self, now: int) -> list[dict[str, Any]]:
        ready = self._commands.pop_ready(now + self._tolerance_ns)
        if not ready:
            return self._repeat_warmup(now)
        # Only the newest due command matters for the latest-command slot.
        self.stats["superseded_before_release"] += len(ready) - 1
        self._warmup = None
        return [self._submit(ready[-1][1], now)]

    def _repeat_warmup(self, now: int) -> list[dict[str, Any]]:
        """Before the first delayed release, keep the first command active."""
        if self._warmup is None or self.stats["commands_released"] > 0:
            return []
        if self._ticker.ticks == 0:
            return []
        self.stats["warmup_commands"] += 1
        return [self._submit(self._warmup, now, warmup=True)]

    def _submit(
        self, pending: _Pending, now: int, *, warmup: bool = False
    ) -> dict[str, Any]:
        receipt = self._handle.send(
            RotorThrusts(tuple(map(float, pending.thrust_n))),
            lifetime_sim_s=self._timing["command_lifetime_sim_s"],
        )
        if not warmup:
            self.stats["commands_released"] += 1
        return dict(
            pending.record,
            release_sim_time_ns=now,
            warmup=warmup,
            motor_speed_rad_s=list(receipt.motor_speed_targets_rad_s),
            command_id=receipt.command_id,
            expires_sim_time_ns=receipt.expires_sim_time_ns,
            accepted_thrust_n=list(receipt.accepted.thrust_n),
        )

    def disarm(self) -> None:
        """Stop the motors and send no further commands (e.g. after landing)."""
        if not self.disarmed:
            self.disarmed = True
            self._handle.stop()

    def summary(self) -> dict[str, Any]:
        """Timing configuration and counters for summary.json."""
        timing = self._timing
        return dict(
            self.stats,
            control_rate_hz=timing["control_rate_hz"],
            position_rate_hz=timing["position_rate_hz"] or timing["control_rate_hz"],
            state_latency_s=timing["state_latency_s"],
            command_latency_s=timing["command_latency_s"],
            publish_on_send=timing["publish_on_send"],
            ticks=self._ticker.ticks,
            missed_ticks=self._ticker.missed,
            pending_at_close=len(self._commands),
        )
