"""Exclusive thrust command lifecycle, independent of simulator transport.

The owner supplies simulation clock updates. A worker republishes the latest
command and checks both simulation expiry and monotonic client freshness.
With ``publish_on_send`` each accepted command is also published immediately.
"""

from collections import deque
import math
from numbers import Real
import threading
import time

from ..core.commands import CommandReceipt, RotorSpec, RotorThrusts
from ..core.vehicle import DEFAULT_DRONE


class ControlBusy(RuntimeError):
    pass


class ControlModeError(RuntimeError):
    pass


class SimulationPaused(RuntimeError):
    pass


class RotorThrustControl:
    def __init__(
        self,
        publish,
        *,
        properties=DEFAULT_DRONE,
        control_mode="rotor-thrust",
        publish_rate_hz=50.0,
        max_lifetime_sim_s=0.1,
        watchdog_wall_s=1.0,
        monotonic=time.monotonic,
        start_worker=True,
        publish_on_send=False,
    ):
        # watchdog_wall_s None disables the wall-clock submission watchdog, for
        # lockstep simulation where Gazebo cannot advance while Python stalls.
        timeouts = (publish_rate_hz, max_lifetime_sim_s) + (
            () if watchdog_wall_s is None else (watchdog_wall_s,)
        )
        for value in timeouts:
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Rates and timeouts must be finite and positive")
        self.rotors = tuple(
            RotorSpec(
                f"rotor_{i}",
                i,
                tuple(position),
                tuple(axis),
                "ccw" if direction == 1 else "cw",
                properties.max_rotor_thrust,
                properties.thrust_coefficient,
            )
            for i, ((_, _, direction), position, axis) in enumerate(
                zip(properties.rotors, properties.positions, properties.axes)
            )
        )
        self._publish = publish
        self._mode = control_mode
        self._period = 1 / publish_rate_hz
        self._max_lifetime = max_lifetime_sim_s
        self._watchdog = watchdog_wall_s
        # When True, send() publishes immediately, so actuation timing follows
        # the caller's simulation-time schedule instead of the worker's
        # wall-clock period. The worker still republishes and enforces expiry.
        self._publish_on_send = publish_on_send
        self._now = monotonic
        self._lock = threading.RLock()
        self._halt = threading.Event()
        self._owner = None
        self._pending = None
        self._clock = None
        self._paused = False
        self._closed = False
        self._fault = None
        self._sequence = 0
        self._events = deque(maxlen=4096)
        self.dropped_events = 0
        self._worker = None
        if start_worker:
            self._worker = threading.Thread(target=self._run, daemon=True)
            self._worker.start()

    def _check(self):
        if self._fault is not None:
            raise RuntimeError("Thrust control faulted") from self._fault
        if self._closed:
            raise RuntimeError("Thrust control closed")

    def acquire(self):
        with self._lock:
            self._check()
            if self._mode != "rotor-thrust":
                raise ControlModeError("Rotor-thrust mode required")
            if self._owner is not None:
                raise ControlBusy("Control already acquired")
            self._owner = RotorThrustHandle(self)
            return self._owner

    def update_clock(self, sim_time_ns, *, paused=False):
        if (
            isinstance(sim_time_ns, bool)
            or not isinstance(sim_time_ns, int)
            or sim_time_ns < 0
        ):
            raise ValueError("Clock requires nonnegative integer nanoseconds")
        with self._lock:
            self._check()
            if self._clock is not None and sim_time_ns < self._clock:
                error = RuntimeError("Simulation time reversed")
                self._zero("failed")
                self._fault = error
                raise error
            self._clock = sim_time_ns
            if paused and not self._paused:
                self._zero("stopped")
            self._paused = paused

    def _event(self, status, receipt=None, error=None):
        if len(self._events) == self._events.maxlen:
            self.dropped_events += 1
        self._events.append(
            dict(
                status=status,
                command_id=receipt.command_id if receipt else None,
                sim_time_ns=self._clock,
                wall_time=self._now(),
                error=error,
                thrust_n=(
                    list(receipt.accepted.thrust_n)
                    if status == "published" and receipt
                    else (
                        [0.0] * len(self.rotors)
                        if status in {"expired", "stopped"}
                        else None
                    )
                ),
            )
        )

    def drain_events(self):
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def _output(self, values, receipt=None):
        try:
            self._publish(values)
        except Exception as error:
            self._fault = error
            self._pending = None
            self._event("failed", receipt, str(error))
            raise

    def _zero(self, status):
        pending, self._pending = self._pending, None
        self._output((0.0,) * len(self.rotors), pending)
        self._event(status, pending)

    def tick(self):
        """Publish/check once; exposed for deterministic clock/transport tests."""
        with self._lock:
            self._check()
            if self._paused or self._pending is None:
                return
            receipt = self._pending
            stalled = (
                self._watchdog is not None
                and self._now() - self._submitted_wall >= self._watchdog
            )
            if self._clock >= receipt.expires_sim_time_ns or stalled:
                self._zero("expired")
            else:
                self._output(receipt.accepted.thrust_n, receipt)
                self._event("published", receipt)

    def _run(self):
        while not self._halt.wait(self._period):
            try:
                self.tick()
            except Exception:
                return  # Fault is retained and raised to the owner by check_health.

    def check_health(self):
        with self._lock:
            self._check()

    def close(self):
        self._halt.set()
        try:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                self._zero("stopped")
        finally:
            if self._worker is not None:
                self._worker.join()


class RotorThrustHandle:
    def __init__(self, control):
        self._control = control
        self._closed = False

    @property
    def rotors(self):
        return self._control.rotors

    def _check(self):
        self._control._check()
        if self._closed or self._control._owner is not self:
            raise RuntimeError("Control handle closed")

    def send(self, command, *, lifetime_sim_s, bounds="reject"):
        c = self._control
        with c._lock:
            self._check()
            if c._paused:
                raise SimulationPaused("Fresh command required after resume")
            if c._clock is None:
                raise RuntimeError("Simulation clock not ready")
            if (
                isinstance(lifetime_sim_s, bool)
                or not isinstance(lifetime_sim_s, Real)
                or not math.isfinite(lifetime_sim_s)
                or not 0 < lifetime_sim_s <= c._max_lifetime
                or int(lifetime_sim_s * 1e9) < 1
            ):
                raise ValueError("Invalid command lifetime")
            if bounds not in {"reject", "clip"}:
                raise ValueError("Unknown bounds policy")
            if not isinstance(command, RotorThrusts) or len(command.thrust_n) != len(
                c.rotors
            ):
                raise ValueError("One thrust per rotor required")
            requested = command.thrust_n
            if any(
                isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                for v in requested
            ):
                raise ValueError("Thrust values must be finite real numbers")
            accepted = tuple(
                min(max(float(v), 0), r.max_thrust_n)
                for v, r in zip(requested, c.rotors)
            )
            limited = tuple(
                i for i, (a, b) in enumerate(zip(requested, accepted)) if a != b
            )
            if limited and bounds == "reject":
                raise ValueError("Thrust outside rotor bounds")
            c._sequence += 1
            receipt = CommandReceipt(
                c._sequence,
                c._clock,
                c._clock + int(lifetime_sim_s * 1e9),
                command,
                RotorThrusts(accepted),
                tuple(
                    math.sqrt(v / r.thrust_coefficient_n_s2)
                    for v, r in zip(accepted, c.rotors)
                ),
                limited,
            )
            if c._pending is not None:
                c._event("superseded", c._pending)
            c._pending = receipt
            c._submitted_wall = c._now()
            if c._publish_on_send:
                c._output(accepted, receipt)
                c._event("published", receipt)
            return receipt

    def stop(self):
        with self._control._lock:
            self._check()
            self._control._zero("stopped")

    def close(self):
        c = self._control
        with c._lock:
            if self._closed:
                return
            try:
                if not c._closed:
                    c._zero("stopped")
            finally:
                self._closed = True
                c._owner = None

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc):
        self.close()
