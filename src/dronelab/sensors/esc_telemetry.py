"""ESC telemetry: measured rotor speeds, as reported by bidirectional DShot.

Gazebo's motor model spins each rotor joint at the actual (lagged, limited)
rotor speed divided by ``rotorVelocitySlowdownSim``; a JointStatePublisher
reports those joint velocities. This model rescales them to rad/s, drops the
spin-direction sign, decimates to ``rate_hz``, and adds seeded white noise of
``noise_std_rad_s`` per sample. Measured speeds feed RPM-based filtering or
motor-fault detection in algorithms.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..core.measurements import EscTelemetry


class EscTelemetryModel:
    """Turn rotor joint velocities into ESC speed reports (see module docs)."""

    def __init__(self, settings: Mapping[str, Any], slowdown: float) -> None:
        self.enabled: bool = settings["enabled"]
        self._period_ns = round(1e9 / settings["rate_hz"])
        self._noise = settings["noise_std_rad_s"]
        self._slowdown = slowdown
        self._rng = np.random.default_rng(settings["seed"])
        self._next_ns: int | None = None
        self.samples = 0

    def sample(
        self, sim_time_ns: int, joint_velocities: Sequence[float]
    ) -> EscTelemetry | None:
        """A report at ``sim_time_ns`` when one is due, else None."""
        if not self.enabled:
            return None
        if self._next_ns is not None and sim_time_ns < self._next_ns:
            return None
        self._next_ns = (self._next_ns or sim_time_ns) + self._period_ns
        if self._next_ns <= sim_time_ns:  # resynchronize after a gap
            self._next_ns = sim_time_ns + self._period_ns
        speeds = np.abs(np.asarray(joint_velocities, float)) * self._slowdown
        noisy = speeds + self._rng.normal(size=speeds.shape) * self._noise
        self.samples += 1
        return EscTelemetry(np.maximum(noisy, 0.0))

    def summary(self) -> dict[str, Any]:
        return dict(enabled=self.enabled, samples=self.samples)
