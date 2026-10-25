"""Rate schedules on the odometry tick grid.

A task running at period P (control, encoders) is due at t0 + k·P, with t0
the first tick it ran on. It runs on the first odometry tick at or after
each due time. Scheduling the next run from the tick that just ran instead
(tick + P) would drift whenever P is not a multiple of the tick: a 10 ms
encoder on 4 ms ticks would run every 12 ms (83 Hz instead of 100 Hz).
"""


def next_due_ns(due_ns: int | None, stamp_ns: int, period_ns: int) -> int:
    """Next due time (ns) after a run at tick ``stamp_ns``.

    ``due_ns`` is the due time this run served (None for the first run,
    which starts the grid at ``stamp_ns``). Due times that the tick already
    passed are skipped, never run twice. Raises ValueError for a
    non-positive period.
    """
    if period_ns <= 0:
        raise ValueError("next_due_ns: period_ns must be positive")
    due = stamp_ns if due_ns is None else due_ns
    if due <= stamp_ns:
        # Whole periods to the first due time after this tick.
        due += ((stamp_ns - due) // period_ns + 1) * period_ns
    return due
