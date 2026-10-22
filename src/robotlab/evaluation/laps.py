"""Lap timing and track limits for wheeled robots on a scene ``track``.

Evaluation only: uses truth (vehicle position and the wheel contact points)
and never feeds the run.

- **Laps** (closed tracks): a lap ends when the vehicle crosses the
  start/finish line forwards, detected as the arc length s jumping from the
  last quarter of the lap to the first quarter while moving forward along
  the track. The first crossing starts timing; a vehicle that starts
  behind the line (a grid slot) therefore times full laps from its first
  crossing. Crossing backwards undoes the lap count (no lap is credited for
  reversing over the line).
- **Track limits**: the vehicle is off track while every wheel contact
  point lies more than ``width/2`` from the centreline (all four wheels
  over the line, as in motor racing). Episodes and time off track are
  counted.

Times are simulation seconds from the run's first truth sample.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..core.track import Track


class LapTimer:
    """Streaming lap and track-limit detector for one vehicle."""

    def __init__(self, track: Track) -> None:
        self.track = track
        self._length = track.length_m
        self._previous_s: float | None = None
        self._lap_start_s: float | None = None
        self._lap_count = 0
        self.lap_times_s: list[float] = []
        self.off_track_episodes = 0
        self.off_track_s = 0.0
        self._off_since: float | None = None
        self._last_t: float | None = None
        self.max_lateral_m = 0.0
        self.progress_m = 0.0
        # Previous nearest segment: a search hint (the car moves little
        # between samples).
        self._segment: int | None = None

    def on_truth(self, t: float, position_m: ArrayLike, contacts_xy: ArrayLike) -> None:
        """One truth sample: vehicle position (m) and wheel contact points (n×2)."""
        projection = self.track.project(
            np.asarray(position_m, float)[:2], self._segment
        )
        self._segment = int(projection.segment[0])
        s = float(projection.s_m[0])
        self.max_lateral_m = max(
            self.max_lateral_m, abs(float(projection.lateral_m[0]))
        )
        if self.track.closed:
            self._laps(t, s)
        self._limits(t, contacts_xy)
        self._last_t = t

    def _laps(self, t: float, s: float) -> None:
        previous, self._previous_s = self._previous_s, s
        if previous is None:
            return
        delta = s - previous
        quarter = self._length / 4
        if previous > 3 * quarter and s < quarter:  # forward over the line
            delta += self._length
            if self._lap_start_s is not None:
                self.lap_times_s.append(t - self._lap_start_s)
            self._lap_start_s = t
            self._lap_count += 1
        elif previous < quarter and s > 3 * quarter:  # backwards over the line
            delta -= self._length
            self._lap_count -= 1
            self._lap_start_s = None
        self.progress_m += delta

    def _limits(self, t: float, contacts_xy: ArrayLike) -> None:
        points = np.atleast_2d(np.asarray(contacts_xy, float))
        lateral = self.track.project(points, self._segment).lateral_m
        off = bool(np.all(np.abs(lateral) > self.track.width_m / 2))
        if off and self._off_since is not None and self._last_t is not None:
            # Time between consecutive off-track samples.
            self.off_track_s += t - self._last_t
        if off and self._off_since is None:
            self._off_since = t
            self.off_track_episodes += 1
        if not off:
            self._off_since = None

    @property
    def off_track_since_s(self) -> float | None:
        """Start of the current off-track episode, or None."""
        return self._off_since

    def summary(self) -> dict[str, Any]:
        return dict(
            length_m=self._length,
            laps_completed=len(self.lap_times_s),
            lap_times_s=self.lap_times_s,
            best_lap_s=min(self.lap_times_s) if self.lap_times_s else None,
            line_crossings=self._lap_count,
            progress_m=self.progress_m,
            off_track_episodes=self.off_track_episodes,
            off_track_s=self.off_track_s,
            max_lateral_offset_m=self.max_lateral_m,
        )
