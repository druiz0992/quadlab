"""Electronic speed controller (ESC) command path: update rate and resolution.

Real ESCs take speed or throttle commands at a finite rate and resolution
(DShot: 2000 throttle steps). ``EscModel`` applies both to the motor-speed
targets before they reach Gazebo's motor model:

- ``resolution_steps`` N > 0 quantizes each target to N levels of the
  maximum speed: ω_q = round(ω / ω_max · N) · ω_max / N.
- ``update_rate_hz`` f limits when new targets take effect: a command is
  forwarded only if simulation time has reached the next ESC tick (period
  1/f); otherwise the motors keep the previous target (Gazebo holds the last
  command). Ticks are evaluated when commands are sent, so the hold is
  aligned to the control loop; it adds sample-and-hold delay when f is below
  the command rate. Unchanged targets (heartbeat republishing) pass without
  using a tick, so a later held change goes out with the next send after the
  tick; stop commands always pass.

Counts of held and quantized commands are reported in ``summary()``.
"""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray


class EscModel:
    """Quantize and rate-limit motor-speed targets (see module docs)."""

    def __init__(
        self,
        settings: Mapping[str, Any],
        max_speed_rad_s: float,
        clock_ns: Callable[[], int | None],
    ) -> None:
        rate = settings["update_rate_hz"]
        self._period_ns = None if rate is None else round(1e9 / rate)
        self._steps = settings["resolution_steps"]
        self._max = max_speed_rad_s
        self._clock = clock_ns
        self._next_ns: int | None = None
        self._last: NDArray[np.float64] | None = None
        self.held = 0
        self.forwarded = 0

    @property
    def enabled(self) -> bool:
        return self._period_ns is not None or self._steps > 0

    def quantize(self, speeds: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._steps <= 0:
            return speeds
        step = self._max / self._steps
        return np.clip(np.round(speeds / step) * step, 0.0, self._max)

    def accept(self, speeds: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """Speeds to publish now, or None to keep the previous target."""
        speeds = self.quantize(np.asarray(speeds, float))
        now = self._clock()
        changed = self._last is None or not np.array_equal(speeds, self._last)
        limited = self._period_ns is not None and now is not None and changed
        if limited and not np.all(speeds == 0):
            if self._next_ns is not None and now < self._next_ns:
                self.held += 1
                return None
            self._next_ns = now + self._period_ns
        self._last = speeds
        self.forwarded += 1
        return speeds

    def summary(self) -> dict[str, Any]:
        return dict(forwarded=self.forwarded, held=self.held)
