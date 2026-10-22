"""Race-track geometry: a centreline polyline with a constant width.

A scene's ``track`` (``{"centreline_m": [[x, y], ...], "width_m": w,
"closed": true}``) is map knowledge, not vehicle truth: controllers receive
it as ``AlgorithmContext.track`` and evaluation uses it for lap timing and
track limits. ``project`` maps world points to the arc length s along the
centreline (m, from point 0, increasing in point order) and the signed
lateral offset (m, positive to the left of the driving direction). The
start/finish line is the normal to the centreline at point 0. Units: m.

A wheeled robot's reference route (``ground_vehicle.route``) is built with
``Track.from_route`` and takes the scene track's place.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Longest straight-leg sample spacing of a route (m).
ROUTE_SPACING_M = 2.0


@dataclass(frozen=True)
class TrackProjection:
    """Per point: arc length ``s_m``, lateral offset ``lateral_m``, and the
    index of the nearest centreline segment."""

    s_m: NDArray[np.float64]
    lateral_m: NDArray[np.float64]
    segment: NDArray[np.int64]


@dataclass(frozen=True)
class Track:
    """Centreline (n×2, m), full width (m), closed loop or open route."""

    centreline_m: tuple[tuple[float, float], ...]
    width_m: float
    closed: bool = True

    def __post_init__(self) -> None:
        points = np.asarray(self.centreline_m, float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("track.centreline_m needs at least three [x, y] points")
        if not np.isfinite(points).all():
            raise ValueError("track.centreline_m must be finite")
        if not (np.isfinite(self.width_m) and self.width_m > 0):
            raise ValueError("track.width_m must be positive")
        segments = np.diff(self._closed_points(points), axis=0)
        if np.any(np.linalg.norm(segments, axis=1) < 1e-6):
            raise ValueError("track.centreline_m has repeated points")

    @classmethod
    def from_scene(cls, scene: Mapping[str, Any]) -> "Track | None":
        """The scene's track, or None; raises ValueError when invalid."""
        track = scene.get("track")
        if track is None:
            return None
        if not isinstance(track, Mapping) or not set(track) <= {
            "centreline_m",
            "width_m",
            "closed",
        }:
            raise ValueError("track needs centreline_m, width_m, and optional closed")
        closed = track.get("closed", True)
        if not isinstance(closed, bool):
            raise ValueError("track.closed must be true or false")
        width = track.get("width_m")
        if isinstance(width, bool) or not isinstance(width, (int, float)):
            raise ValueError("track.width_m must be a number")
        points = np.asarray(track.get("centreline_m", []), float)
        return cls(tuple(map(tuple, points.tolist())), float(width), closed)

    @classmethod
    def from_route(
        cls,
        waypoints_m: Sequence[Sequence[float]],
        width_m: float,
        spacing_m: float = ROUTE_SPACING_M,
    ) -> "Track":
        """Route through ``[[x, y], ...]`` waypoints (m), straight legs
        resampled at most ``spacing_m`` apart (so a two-point route is a
        valid track and curvature estimates stay local). A last waypoint
        equal to the first closes the loop. Raises ValueError for fewer
        than two distinct points, repeated consecutive points, or
        non-finite values."""
        try:
            points = np.asarray(waypoints_m, float)
        except (TypeError, ValueError):
            raise ValueError("route waypoints_m must be [[x, y], ...] numbers")
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("route waypoints_m needs at least two [x, y] points")
        if not np.isfinite(points).all():
            raise ValueError("route waypoints_m must be finite")
        legs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(legs < 1e-6):
            raise ValueError("route waypoints_m has repeated consecutive points")
        closed = len(points) > 3 and bool(np.allclose(points[0], points[-1]))
        dense = [points[0]]
        for start, end, length in zip(points[:-1], points[1:], legs):
            count = max(1, int(np.ceil(length / spacing_m)))
            for k in range(1, count + 1):
                dense.append(start + (end - start) * k / count)
        if closed:
            dense.pop()  # the closing vertex is implied
        return cls(tuple((float(x), float(y)) for x, y in dense), width_m, closed)

    def _closed_points(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.vstack([points, points[:1]]) if self.closed else points

    @property
    def points(self) -> NDArray[np.float64]:
        """Polyline vertices including the closing vertex of a loop (m×2)."""
        return self._closed_points(np.asarray(self.centreline_m, float))

    @property
    def length_m(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1)))

    def _segments(self) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Cached segment starts, vectors, lengths, and start arc lengths."""
        cache = self.__dict__.get("_segment_cache")
        if cache is None:
            p = self.points
            a, ab = p[:-1], np.diff(p, axis=0)
            length = np.linalg.norm(ab, axis=1)
            start_s = np.concatenate([[0.0], np.cumsum(length)[:-1]])
            cache = (a, ab, length, start_s)
            object.__setattr__(self, "_segment_cache", cache)
        return cache

    def project(
        self, xy: ArrayLike, near: int | None = None, window: int = 30
    ) -> TrackProjection:
        """Arc length and lateral offset of world points (…×2) on the nearest
        segment. With ``near`` (a previous nearest segment) only segments
        within ``window`` of it are searched, unless the result is farther
        than the track width from the centreline (then all are searched)."""
        q = np.atleast_2d(np.asarray(xy, float))
        a, ab, length, start_s = self._segments()
        count = len(a)
        if near is not None and 2 * window + 1 < count:
            offsets = np.arange(-window, window + 1) + int(near)
            candidates = (
                offsets % count if self.closed else np.clip(offsets, 0, count - 1)
            )
            result = self._nearest(q, candidates, a, ab, length, start_s)
            if np.all(np.abs(result.lateral_m) <= self.width_m):
                return result
        return self._nearest(q, np.arange(count), a, ab, length, start_s)

    @staticmethod
    def _nearest(q, candidates, a, ab, length, start_s) -> TrackProjection:
        a, ab, length, start_s = (
            a[candidates],
            ab[candidates],
            length[candidates],
            start_s[candidates],
        )
        rel = q[:, None, :] - a[None, :, :]
        t = np.clip(np.sum(rel * ab, axis=2) / length**2, 0.0, 1.0)
        nearest = a + t[..., None] * ab
        distance = np.linalg.norm(q[:, None, :] - nearest, axis=2)
        best = np.argmin(distance, axis=1)
        rows = np.arange(len(q))
        direction = ab[best] / length[best, None]
        offset = q - nearest[rows, best]
        lateral = direction[:, 0] * offset[:, 1] - direction[:, 1] * offset[:, 0]
        s = start_s[best] + t[rows, best] * length[best]
        return TrackProjection(s, lateral, np.asarray(candidates)[best])

    def heading_rad(self, segment: ArrayLike) -> NDArray[np.float64]:
        """Driving direction (rad, from +x) of centreline segments."""
        ab = np.diff(self.points, axis=0)[np.asarray(segment)]
        return np.arctan2(ab[..., 1], ab[..., 0])
