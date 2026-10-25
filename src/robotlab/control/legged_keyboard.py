"""Keyboard walking input for legged robots: held keys → ``WalkCommand``.

Keys: ``Up``/``Down`` walk forward/back, ``Left``/``Right`` turn (yaw left
/right), ``a``/``d`` step left/right. Each axis ramps towards ±its maximum
while a key is held and back to zero when released, at ``ramp_per_s``
(fraction of the maximum per second; release returns twice as fast), so
the policy sees smooth commands. Maxima come from ``biped.keyboard``
(m/s, rad/s). Pure logic: the window is visualization/driving_keyboard.py.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .legged import WalkCommand

WALK_KEYS = frozenset({"Up", "Down", "Left", "Right", "a", "d"})


@dataclass
class WalkInput:
    """Ramped walking command from held keys (see module docs)."""

    max_forward_mps: float
    max_left_mps: float
    max_yaw_rate_rad_s: float
    ramp_per_s: float
    _levels: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "WalkInput":
        """From the validated ``biped.keyboard`` section."""
        return cls(
            settings["max_forward_mps"],
            settings["max_left_mps"],
            settings["max_yaw_rate_rad_s"],
            settings["ramp_per_s"],
        )

    def update(self, keys: Collection[str], dt_s: float) -> WalkCommand:
        """Advance the ramps by ``dt_s`` seconds and return the command."""
        target = np.array(
            [
                ("Up" in keys) - ("Down" in keys),
                ("a" in keys) - ("d" in keys),
                ("Left" in keys) - ("Right" in keys),
            ],
            dtype=float,
        )
        step = np.where(target == 0, 2.0, 1.0) * self.ramp_per_s * dt_s
        self._levels += np.clip(target - self._levels, -step, step)
        forward, left, yaw = self._levels * [
            self.max_forward_mps,
            self.max_left_mps,
            self.max_yaw_rate_rad_s,
        ]
        return WalkCommand(float(forward), float(left), float(yaw))
