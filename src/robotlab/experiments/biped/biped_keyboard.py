"""Keyboard walking for biped runs: Tk key window and real-time pacing.

``KeyboardOperator`` implements the biped loop's ``Operator``: each control
period it turns the held keys into a ``WalkCommand``
(control/legged_keyboard.py, ramped over wall time); after every physics
step it sleeps so simulation time does not run ahead of wall time (it may
fall behind when stepping is slower than real time, never ahead). Closing
the window (or Esc) ends the run. The robot is shown in Rerun
(visualization/biped_rerun.py), not MuJoCo's own viewer, whose passive
mode crashed or hung on shutdown in about half of the tries on the
development machine (MuJoCo 3.17, X11). Tk needs a desktop; its import is
deferred.
"""

from collections.abc import Callable
import time
from typing import Any

from ...control.legged import WalkCommand
from ...control.legged_keyboard import WALK_KEYS, WalkInput

HELP = (
    "↑/↓ Walk forward/back · ←/→ Turn\n"
    "A/D Step left/right · Esc Finish\nRelease keys to stand"
)


class KeyboardOperator:
    """See module docs. ``window`` provides ``poll_keys()``, ``closed``, and
    ``close()``; ``clock`` and ``sleep`` are wall-time seconds."""

    def __init__(
        self,
        window: Any,
        walk: WalkInput,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.window, self.walk = window, walk
        self._clock, self._sleep = clock, sleep
        self._start_wall: float | None = None
        self._last_poll: float | None = None

    def request(self, sim_time_ns: int) -> WalkCommand:
        now = self._clock()
        dt = 0.0 if self._last_poll is None else now - self._last_poll
        self._last_poll = now
        return self.walk.update(self.window.poll_keys(), dt)

    def running(self) -> bool:
        return not self.window.closed

    def after_step(self, data: Any, sim_time_ns: int) -> None:
        now = self._clock()
        if self._start_wall is None:
            self._start_wall = now
        ahead_s = sim_time_ns * 1e-9 - (now - self._start_wall)
        if ahead_s > 0:
            self._sleep(ahead_s)

    def close(self) -> None:
        self.window.close()


def open_keyboard_operator(walk: WalkInput) -> KeyboardOperator:
    """Open the key window; raises ValueError without a desktop."""
    from ...visualization.driving_keyboard import DrivingKeyboardWindow

    window = DrivingKeyboardWindow("biped walking", "biped", WALK_KEYS, HELP)
    return KeyboardOperator(window, walk)
