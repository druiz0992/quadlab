"""Stop runs gracefully on SIGTERM, and keep their cleanup from being cut short.

Python's default SIGTERM action ends the process at once: no ``finally``
blocks, so no ``summary.json`` and an orphaned Gazebo. ``sigterm_as_interrupt``
makes SIGTERM behave like Ctrl-C (``KeyboardInterrupt``), which the mission
loop and calibration sessions already handle by saving results and
stopping Gazebo. ``shielded`` defers SIGINT/SIGTERM during that cleanup, so
a second signal (Ctrl-C pressed twice, or a batch terminating a child that
is already stopping) cannot abort it half-way; cleanup itself is bounded
(Gazebo gets ``mission_loop.GAZEBO_STOP_TIMEOUT_WALL_S`` before it is killed).

Signal handlers can only be installed from the main thread; elsewhere both
context managers do nothing.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import signal
import threading
from types import FrameType


@dataclass
class Termination:
    """Whether SIGTERM was received inside ``sigterm_as_interrupt``."""

    received: bool = False


def _main_thread() -> bool:
    return threading.current_thread() is threading.main_thread()


@contextmanager
def sigterm_as_interrupt() -> Iterator[Termination]:
    """Raise ``KeyboardInterrupt`` on SIGTERM within the block; the yielded
    ``Termination`` records it (CLIs exit with 128 + SIGTERM = 143)."""
    termination = Termination()
    if not _main_thread():
        yield termination
        return

    def interrupt(signum: int, frame: FrameType | None) -> None:
        termination.received = True
        raise KeyboardInterrupt(f"signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield termination
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextmanager
def shielded() -> Iterator[list[int]]:
    """Defer SIGINT and SIGTERM until the block ends; the yielded list
    collects the signals received meanwhile. They are not re-raised: the
    run is already stopping."""
    received: list[int] = []
    if not _main_thread():
        yield received
        return

    def record(signum: int, frame: FrameType | None) -> None:
        received.append(signum)

    previous = {
        number: signal.signal(number, record)
        for number in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield received
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
