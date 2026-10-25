"""Seeded GPS multipath: sudden position jumps that persist, then clear.

Reflections near structures make receivers report positions offset by metres
for seconds at a time. This model approximates that with discrete episodes:

- An episode starts with probability ``1 - exp(-onset_rate_hz * dt)`` per fix,
  where dt is the time since the previous fix (a Poisson onset process).
- Its ENU offset is drawn once from N(0, diag(h², h², v²)) using
  ``horizontal_std_m`` (h) and ``vertical_std_m`` (v), and is held for
  ``duration_s`` simulation seconds.
- With ``zones``, episodes start only inside a zone and end early on leaving
  all zones. Without zones they can happen anywhere.

This is a statistical stand-in, not ray-traced multipath from world geometry.
Velocity fields are not modified. The random draws depend only on the fix
timestamps and zone membership, so a seed reproduces a run.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...core.measurements import GpsReading
from .errors import offset_fix
from .zones import inside_any, zones_from_settings

_MAX_LOGGED_EPISODES = 1000


@dataclass
class _Episode:
    start_ns: int
    end_ns: int
    offset_enu_m: NDArray[np.float64]


class GpsMultipath:
    """Apply multipath episodes to time-ordered GPS fixes."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.enabled: bool = settings["enabled"]
        self._rng = np.random.default_rng(settings["seed"])
        self._zones = zones_from_settings(settings["zones"])
        self._rate_hz: float = settings["onset_rate_hz"]
        self._duration_ns = round(settings["duration_s"] * 1e9)
        h, v = settings["horizontal_std_m"], settings["vertical_std_m"]
        self._std = np.array([h, h, v], dtype=float)
        self._previous_time_ns: int | None = None
        self._active: _Episode | None = None
        self._episodes: list[dict[str, Any]] = []
        self._episode_count = 0
        self._affected_fixes = 0
        # Manual jump (e.g. keyboard toggle): ENU offset applied while set.
        self.forced_offset_enu_m: NDArray[np.float64] | None = None

    def apply(
        self, fix: GpsReading, sim_time_ns: int, position_m: ArrayLike | None = None
    ) -> GpsReading:
        """Return the fix with any active multipath offset added.

        position_m is the vehicle's world position, used only for zones.
        """
        if self.forced_offset_enu_m is not None:
            if self.enabled:
                # Keep the random sequence advancing while overridden.
                self._update_episode(
                    sim_time_ns, self._advance_clock(sim_time_ns), False
                )
            return offset_fix(fix, self.forced_offset_enu_m)
        if not self.enabled:
            return fix
        dt_s = self._advance_clock(sim_time_ns)
        in_zone = not self._zones or inside_any(self._zones, position_m)
        self._update_episode(sim_time_ns, dt_s, in_zone)
        if self._active is None:
            return fix
        self._affected_fixes += 1
        return offset_fix(fix, self._active.offset_enu_m)

    def _advance_clock(self, sim_time_ns: int) -> float:
        previous = self._previous_time_ns
        if sim_time_ns < 0 or (previous is not None and sim_time_ns <= previous):
            raise ValueError(
                "Multipath timestamps must be nonnegative and strictly increasing"
            )
        self._previous_time_ns = sim_time_ns
        return 0.0 if previous is None else (sim_time_ns - previous) * 1e-9

    def _update_episode(self, sim_time_ns: int, dt_s: float, in_zone: bool) -> None:
        """End expired or out-of-zone episodes, then maybe start a new one."""
        # Always draw so the RNG sequence does not depend on episode state.
        onset_draw = self._rng.random()
        offset_draw = self._rng.normal(size=3)
        active = self._active
        if active is not None and (sim_time_ns >= active.end_ns or not in_zone):
            self._active = None
        if self._active is not None or not in_zone:
            return
        if onset_draw < -math.expm1(-self._rate_hz * dt_s):
            self._start_episode(sim_time_ns, self._std * offset_draw)

    def _start_episode(self, sim_time_ns: int, offset: NDArray[np.float64]) -> None:
        self._active = _Episode(sim_time_ns, sim_time_ns + self._duration_ns, offset)
        self._episode_count += 1
        if len(self._episodes) < _MAX_LOGGED_EPISODES:
            self._episodes.append(
                dict(start_sim_time_ns=sim_time_ns, offset_enu_m=offset.tolist())
            )

    @property
    def active(self) -> bool:
        """Whether a multipath episode or manual jump affected the latest fix."""
        return self._active is not None or self.forced_offset_enu_m is not None

    @property
    def affects_fixes(self) -> bool:
        """Whether fixes can be modified (model enabled or a manual jump set)."""
        return self.enabled or self.forced_offset_enu_m is not None

    def summary(self) -> dict[str, Any]:
        """Diagnostics for summary.json (first 1000 episodes), for evaluation only."""
        return dict(
            enabled=self.enabled,
            episodes=self._episode_count,
            affected_fixes=self._affected_fixes,
            episode_log=self._episodes,
        )
