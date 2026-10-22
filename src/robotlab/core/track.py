"""Race-track geometry: a centreline polyline with a constant width.

A scene's ``track`` (``{"centreline_m": [[x, y], ...], "width_m": w,
"closed": true}``) is map knowledge, not vehicle truth: controllers receive
it as ``AlgorithmContext.track`` and evaluation uses it for lap timing and
track limits. ``project`` maps world points to the arc length s along the
centreline (m, from point 0, increasing in point order) and the signed
lateral offset (m, positive to the left of the driving direction). The
start/finish line is the normal to the centreline at point 0. Units: m.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


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

    def _closed_points(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.vstack([points, points[:1]]) if self.closed else points

    @property
    def points(self) -> NDArray[np.float64]:
        """Polyline vertices including the closing vertex of a loop (m×2)."""
        return self._closed_points(np.asarray(self.centreline_m, float))

    @property
    def length_m(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1)))

    def project(self, xy: ArrayLike) -> TrackProjection:
        """Arc length and lateral offset of world points (…×2) on the nearest
        segment (brute force over segments; fine for a few hundred)."""
        q = np.atleast_2d(np.asarray(xy, float))
        p = self.points
        a, ab = p[:-1], np.diff(p, axis=0)
        length = np.linalg.norm(ab, axis=1)
        start_s = np.concatenate([[0.0], np.cumsum(length)[:-1]])
        rel = q[:, None, :] - a[None, :, :]
        t = np.clip(np.sum(rel * ab, axis=2) / length**2, 0.0, 1.0)
        nearest = a + t[..., None] * ab
        distance = np.linalg.norm(q[:, None, :] - nearest, axis=2)
        segment = np.argmin(distance, axis=1)
        rows = np.arange(len(q))
        direction = ab[segment] / length[segment, None]
        offset = q - nearest[rows, segment]
        lateral = direction[:, 0] * offset[:, 1] - direction[:, 1] * offset[:, 0]
        s = start_s[segment] + t[rows, segment] * length[segment]
        return TrackProjection(s, lateral, segment)

    def heading_rad(self, segment: ArrayLike) -> NDArray[np.float64]:
        """Driving direction (rad, from +x) of centreline segments."""
        ab = np.diff(self.points, axis=0)[np.asarray(segment)]
        return np.arctan2(ab[..., 1], ab[..., 0])
