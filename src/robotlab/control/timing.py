"""Simulation-time scheduling for controllers: fixed-rate ticks and latency.

These helpers model the timing of an embedded flight controller inside an
asynchronous simulation. They are transport-free and use integer simulation
nanoseconds throughout:

- ``FixedRateTicker`` decides when a controller runs on a fixed grid.
- ``StateBuffer`` returns the newest state at least a given latency old.
- ``DelayLine`` holds commands until their release time.

Ticks can only occur when a sample is processed, so tick timing resolution is
the sample period of the driving stream (see docs/4_control.md).
"""

from collections import deque
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Tick:
    """One controller execution on the fixed grid."""

    index: int  # Ticks since the first one; index 0 is the first tick.
    sim_time_ns: int  # Grid time this tick represents.
    dt_s: float  # Grid time since the previous executed tick.
    missed: int  # Grid ticks skipped since the previous executed tick.


class FixedRateTicker:
    """Fixed simulation-time grid anchored at the first offered sample.

    ``offer(t)`` returns a Tick when a grid time has been reached, else None.
    A sample within ``tolerance_ns`` before a grid time counts as reaching it,
    absorbing timestamp quantization. If samples skip grid times (a gap), only
    the most recent grid time runs and the skipped ones are counted as missed:
    re-running a controller on the same held state would add no information.
    """

    def __init__(self, rate_hz: float, *, tolerance_ns: int = 0) -> None:
        if not rate_hz > 0:
            raise ValueError("Tick rate must be positive")
        self.period_ns = round(1e9 / rate_hz)
        self._tolerance_ns = tolerance_ns
        self._anchor_ns: int | None = None
        self._last_index = -1
        self.ticks = 0
        self.missed = 0

    def offer(self, sim_time_ns: int) -> Tick | None:
        if self._anchor_ns is None:
            self._anchor_ns = sim_time_ns
        index = (sim_time_ns - self._anchor_ns + self._tolerance_ns) // self.period_ns
        if index <= self._last_index:
            return None
        missed = 0 if self._last_index < 0 else index - self._last_index - 1
        # The first tick uses one nominal period, matching the previous behaviour.
        steps = 1 if self._last_index < 0 else index - self._last_index
        self._last_index = index
        self.ticks += 1
        self.missed += missed
        return Tick(
            index,
            self._anchor_ns + index * self.period_ns,
            steps * self.period_ns * 1e-9,
            missed,
        )


class StateBuffer(Generic[T]):
    """Time-ordered samples; returns the newest one at least ``latency_ns`` old.

    Older samples that can never be selected again are discarded, so memory is
    bounded by the latency divided by the sample period.
    """

    def __init__(self, latency_ns: int) -> None:
        if latency_ns < 0:
            raise ValueError("State latency must be nonnegative")
        self._latency_ns = latency_ns
        self._samples: deque[tuple[int, T]] = deque()

    def push(self, sim_time_ns: int, sample: T) -> None:
        if self._samples and sim_time_ns <= self._samples[-1][0]:
            raise ValueError("State samples must have increasing timestamps")
        self._samples.append((sim_time_ns, sample))

    def available(self, now_ns: int) -> tuple[int, T] | None:
        """Newest (time, sample) with time <= now - latency, or None if none yet."""
        cutoff = now_ns - self._latency_ns
        # Drop samples superseded by a newer one that is also old enough.
        while len(self._samples) > 1 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        if not self._samples or self._samples[0][0] > cutoff:
            return None
        return self._samples[0]


class DelayLine(Generic[T]):
    """FIFO releasing items ``delay_ns`` after their submission time."""

    def __init__(self, delay_ns: int) -> None:
        if delay_ns < 0:
            raise ValueError("Delay must be nonnegative")
        self._delay_ns = delay_ns
        self._items: deque[tuple[int, T]] = deque()

    @property
    def delay_ns(self) -> int:
        return self._delay_ns

    def push(self, sim_time_ns: int, item: T) -> None:
        self._items.append((sim_time_ns + self._delay_ns, item))

    def pop_ready(self, now_ns: int) -> list[tuple[int, T]]:
        """Remove and return (release time, item) pairs due at or before now."""
        ready = []
        while self._items and self._items[0][0] <= now_ns:
            ready.append(self._items.popleft())
        return ready

    def __len__(self) -> int:
        return len(self._items)
