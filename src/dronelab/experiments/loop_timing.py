"""Wall-clock cost of the Gazebo mission loop, for ``summary.json``.

The mission loop is one Python thread that must keep up with Gazebo. In
real time it falls behind when it spends more wall time handling a
simulated second of messages than one second (``load`` > 1 at a real-time
factor of 1); the backlog then shows up as growing odometry queue age (wall
seconds from transport receipt to processing), which aborts rotor-thrust
missions at ``MAX_QUEUE_AGE_WALL_S``. In lockstep the world waits for
Python, so ``load`` is the wall cost per simulated second and queue age
stays small.

These are diagnostics only: nothing here feeds back into control, and the
clock is injectable so tests use a fake.
"""

from collections import Counter, defaultdict
from collections.abc import Callable
import time
from typing import Any

# Odometry older than this (wall s) when processed counts as late.
LATE_ODOMETRY_WALL_S = 0.05


class LoopTiming:
    """Accumulate per-message processing time and odometry queue age.

    Call ``begin`` before handling a message (or a non-message loop section,
    timed under its own name) and ``end`` after it (also when handling
    raised; the caller decides). Simulation times are seconds
    from message stamps; wall times come from ``clock`` (monotonic s).
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started_wall: float | None = None
        self._kind: str | None = None
        self.busy_wall_s: defaultdict[str, float] = defaultdict(float)
        self.messages: Counter[str] = Counter()
        self.slowest_wall_s: dict[str, float] = {}
        self.odometry_count = 0
        self.odometry_age_sum_s = 0.0
        self.odometry_age_max_s = 0.0
        self.late_odometry = 0
        self.first_sim_s: float | None = None
        self.last_sim_s: float | None = None

    def begin(self, kind: str, received_at_wall_s: float | None = None) -> None:
        """Start timing one message received at ``received_at_wall_s``, or a
        loop section (such as ``housekeeping``) when that is None."""
        now = self._clock()
        self._started_wall, self._kind = now, kind
        if kind == "odometry" and received_at_wall_s is not None:
            age = now - received_at_wall_s
            self.odometry_count += 1
            self.odometry_age_sum_s += age
            self.odometry_age_max_s = max(self.odometry_age_max_s, age)
            self.late_odometry += age > LATE_ODOMETRY_WALL_S

    def end(self, sim_time_s: float | None = None) -> None:
        """Finish timing the current message; ``sim_time_s`` extends the
        simulated span (odometry stamps). Raises if ``begin`` was not called."""
        if self._started_wall is None or self._kind is None:
            raise RuntimeError("LoopTiming.end without begin")
        elapsed = self._clock() - self._started_wall
        kind = self._kind
        self.busy_wall_s[kind] += elapsed
        self.messages[kind] += 1
        self.slowest_wall_s[kind] = max(self.slowest_wall_s.get(kind, 0.0), elapsed)
        if sim_time_s is not None:
            if self.first_sim_s is None:
                self.first_sim_s = sim_time_s
            self.last_sim_s = sim_time_s
        self._started_wall = self._kind = None

    def summary(self) -> dict[str, Any]:
        """``loop_timing`` entry: wall cost per simulated second and queue age.

        ``load`` is total busy wall seconds over the simulated span (None
        before two odometry samples); ``per_kind`` holds busy wall seconds,
        message counts, and the slowest single message (ms) for each kind.
        """
        busy = sum(self.busy_wall_s.values())
        span = (
            self.last_sim_s - self.first_sim_s
            if self.first_sim_s is not None and self.last_sim_s is not None
            else 0.0
        )
        count = self.odometry_count
        return dict(
            busy_wall_s=busy,
            sim_span_s=span,
            load=busy / span if span > 0 else None,
            odometry_queue_age_ms=dict(
                mean=1e3 * self.odometry_age_sum_s / count if count else None,
                max=1e3 * self.odometry_age_max_s if count else None,
                late=self.late_odometry,
                late_threshold_ms=1e3 * LATE_ODOMETRY_WALL_S,
            ),
            per_kind={
                kind: dict(
                    busy_wall_s=self.busy_wall_s[kind],
                    messages=self.messages[kind],
                    slowest_ms=1e3 * self.slowest_wall_s[kind],
                )
                for kind in sorted(self.messages)
            },
        )
