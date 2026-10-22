"""Seeded rolling terrain: one height function for Gazebo geometry and Python models.

The terrain is a sum of Gaussian hills on top of the flat ground plane
(z = 0), multiplied by a smooth blend that keeps a disc around the origin
flat for spawning:

    h(x, y) = blend(r) · Σ_k height_k · exp(−|p − c_k|² / (2 σ_k²))

Hill centres, heights (up to ``max_height_m``), and widths σ (within
``hill_radius_m``) come from a local seeded RNG, so a seed reproduces the
terrain.

Ditches (``ditches``: ``[{"points_m": [[x, y], ...], "width_m": w,
"depth_m": d}, ...]``) are cut into that surface along polylines with a
smooth cosine trough: at lateral distance ρ from the polyline,

    depth(ρ) = d · ½ (1 + cos(π ρ / (w/2)))   for ρ < w/2, else 0,

so the floor is d deep and the edges meet the surface without a step
(steepest wall slope π d / w). Where ditches overlap the deepest one wins.
The flat spawn disc does not apply to ditches. The same function generates the triangle mesh Gazebo uses for
collision and visuals and answers height and ray queries for models that
previously assumed flat ground (optical flow, ground effect, ground start).
The mesh is a grid at ``resolution_m``; Gazebo's surface therefore differs
from the analytic height by the grid's linear-interpolation error.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULTS = {
    "seed": 0,
    "size_m": [60.0, 60.0],
    "resolution_m": 1.0,
    "hills": 12,
    "max_height_m": 3.0,
    "hill_radius_m": [3.0, 10.0],
    "flat_radius_m": 5.0,
    "ditches": [],
}
DITCH_KEYS = {"points_m", "width_m", "depth_m"}


def validate_terrain(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Merge defaults and check ranges; returns the resolved settings."""
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown terrain keys: {', '.join(sorted(unknown))}")
    data = {**DEFAULTS, **settings}
    size, radius = data["size_m"], data["hill_radius_m"]
    if (
        len(size) != 2
        or min(size) <= 0
        or len(radius) != 2
        or not 0 < radius[0] <= radius[1]
    ):
        raise ValueError(
            "terrain size_m needs two positive values; hill_radius_m needs 0 < min <= max"
        )
    if data["resolution_m"] <= 0 or data["resolution_m"] > min(size) / 2:
        raise ValueError(
            "terrain resolution_m must be positive and at most half the size"
        )
    if (
        type(data["hills"]) is not int
        or data["hills"] < 0
        or type(data["seed"]) is not int
    ):
        raise ValueError("terrain hills and seed must be nonnegative integers")
    if data["max_height_m"] < 0 or data["flat_radius_m"] < 0:
        raise ValueError("terrain max_height_m and flat_radius_m must be nonnegative")
    if not isinstance(data["ditches"], list):
        raise ValueError("terrain ditches must be a list")
    for index, ditch in enumerate(data["ditches"]):
        _validate_ditch(ditch, f"terrain.ditches[{index}]", size)
    return data


def _validate_ditch(ditch: Any, name: str, size_m: list[float]) -> None:
    """One ditch: keys, at least two distinct finite points, positive width
    and depth, and the whole trough inside the terrain square."""
    if not isinstance(ditch, Mapping) or set(ditch) != DITCH_KEYS:
        raise ValueError(f"{name} needs exactly {sorted(DITCH_KEYS)}")
    for key in ("width_m", "depth_m"):
        value = ditch[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}.{key} must be a number")
        if not (np.isfinite(value) and value > 0):
            raise ValueError(f"{name}.{key} must be positive")
    try:
        points = np.asarray(ditch["points_m"], float)
    except (TypeError, ValueError):
        raise ValueError(f"{name}.points_m must be [[x, y], ...] numbers")
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError(f"{name}.points_m needs at least two [x, y] points")
    if not np.isfinite(points).all():
        raise ValueError(f"{name}.points_m must be finite")
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) < 1e-6):
        raise ValueError(f"{name}.points_m has repeated consecutive points")
    reach = np.abs(points) + ditch["width_m"] / 2
    if np.any(reach > np.asarray(size_m, float) / 2):
        raise ValueError(f"{name} must lie inside the terrain square")


def _ditch_segments(
    ditches: list[Mapping[str, Any]],
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray] | None:
    """Per segment of every ditch: start (K×2), start-to-end vector (K×2),
    squared length, half width, and depth; None without ditches."""
    if not ditches:
        return None
    starts, alongs, halves, depths = [], [], [], []
    for ditch in ditches:
        points = np.asarray(ditch["points_m"], float)
        starts.append(points[:-1])
        alongs.append(np.diff(points, axis=0))
        halves += [ditch["width_m"] / 2] * (len(points) - 1)
        depths += [ditch["depth_m"]] * (len(points) - 1)
    along = np.vstack(alongs)
    return (
        np.vstack(starts),
        along,
        (along**2).sum(axis=1),
        np.asarray(halves, float),
        np.asarray(depths, float),
    )


class Terrain:
    """Analytic height field with mesh export and ray intersection."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = validate_terrain(settings)
        s = self.settings
        rng = np.random.default_rng(s["seed"])
        half = np.asarray(s["size_m"], float) / 2
        self._centres = rng.uniform(-half, half, size=(s["hills"], 2))
        self._heights = rng.uniform(0.3, 1.0, size=s["hills"]) * s["max_height_m"]
        self._sigmas = rng.uniform(*s["hill_radius_m"], size=s["hills"])
        self._half = half
        self._flat = s["flat_radius_m"]
        self._ditches = _ditch_segments(s["ditches"])

    def height(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        """Terrain height (m) at world x, y; 0 outside the terrain square."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        shape = np.broadcast(x, y).shape
        if len(self._heights):
            # All hills at once; the cumulative sum along the hill axis adds
            # the terms in the same order as a loop over hills, so heights
            # are bit-identical to the sequential sum (and ~50× faster for
            # the few points of a wheel ray cast).
            cx, cy = self._centres[:, 0], self._centres[:, 1]
            dx2 = (x[..., None] - cx) ** 2
            dy2 = (y[..., None] - cy) ** 2
            terms = self._heights * np.exp(-(dx2 + dy2) / (2 * self._sigmas**2))
            h = np.cumsum(
                np.broadcast_to(terms, (*shape, len(self._heights))), axis=-1
            )[..., -1]
        else:
            h = np.zeros(shape)
        r = np.hypot(x, y)
        # Smoothstep from 0 inside the flat disc to 1 at twice its radius.
        u = (
            np.clip((r - self._flat) / max(self._flat, 1e-9), 0.0, 1.0)
            if self._flat
            else 1.0
        )
        inside = (np.abs(x) <= self._half[0]) & (np.abs(y) <= self._half[1])
        surface = h * u * u * (3 - 2 * u)
        if self._ditches is not None:
            surface = surface - self._ditch_depth(x, y)
        return np.where(inside, surface, 0.0)

    def _ditch_depth(self, x: NDArray, y: NDArray) -> NDArray[np.float64]:
        """Deepest ditch cut (m, ≥ 0) at each point (module docs)."""
        assert self._ditches is not None
        start, along, length2, half_width, depth = self._ditches
        px = x[..., None] - start[:, 0]
        py = y[..., None] - start[:, 1]
        # Closest point on each segment: clamp the projection to [0, 1].
        t = np.clip((px * along[:, 0] + py * along[:, 1]) / length2, 0.0, 1.0)
        rho = np.hypot(px - t * along[:, 0], py - t * along[:, 1])
        ratio = np.minimum(rho / half_width, 1.0)
        cut = depth * 0.5 * (1.0 + np.cos(np.pi * ratio))
        return np.asarray(cut.max(axis=-1))

    def mesh(self) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Grid vertices (N×3) and triangle faces (M×3, zero-based)."""
        step = self.settings["resolution_m"]
        xs = np.arange(-self._half[0], self._half[0] + step / 2, step)
        ys = np.arange(-self._half[1], self._half[1] + step / 2, step)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        vertices = np.column_stack(
            (gx.ravel(), gy.ravel(), self.height(gx, gy).ravel())
        )
        nx = len(xs)
        faces = []
        for j in range(len(ys) - 1):
            for i in range(nx - 1):
                a, b = j * nx + i, j * nx + i + 1
                c, d = a + nx, b + nx
                faces += [(a, b, d), (a, d, c)]  # counter-clockwise from above
        return vertices, np.asarray(faces, dtype=np.int64)

    def normals(
        self, x: ArrayLike, y: ArrayLike, eps: float = 1e-3
    ) -> NDArray[np.float64]:
        """Unit surface normals (…×3) from central differences of the height."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        dx = (self.height(x + eps, y) - self.height(x - eps, y)) / (2 * eps)
        dy = (self.height(x, y + eps) - self.height(x, y - eps)) / (2 * eps)
        n = np.stack((-dx, -dy, np.ones_like(dx)), axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def write_obj(self, path: Path) -> Path:
        """Write the mesh as Wavefront OBJ with per-vertex normals.

        Gazebo's DART collision rejects meshes without normals (the submesh is
        dropped and the physics engine then crashes), so every vertex carries
        one and faces reference it (``f v//vn``, one-based).
        """
        vertices, faces = self.mesh()
        normals = self.normals(vertices[:, 0], vertices[:, 1])
        lines = [f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in vertices]
        lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in normals]
        lines += [
            f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}" for a, b, c in faces
        ]
        path.write_text("\n".join(lines) + "\n")
        return path

    def ray_distance(
        self,
        origin: ArrayLike,
        direction: ArrayLike,
        max_distance: float,
        step: float = 0.05,
    ) -> float:
        """Distance along a unit ray to the terrain surface (or the z = 0 plane).

        Marches in ``step`` increments, then bisects the crossing. Returns NaN
        when the ray does not reach the surface within ``max_distance``.
        """
        o, d = np.asarray(origin, float), np.asarray(direction, float)

        def above(t: float) -> float:
            p = o + t * d
            return p[2] - float(self.height(p[0], p[1]))

        if above(0.0) <= 0:
            return 0.0
        previous = 0.0
        t = step
        while t <= max_distance + step:
            if above(t) <= 0:
                low, high = previous, t
                for _ in range(30):
                    mid = 0.5 * (low + high)
                    low, high = (mid, high) if above(mid) > 0 else (low, mid)
                return high if high <= max_distance else float("nan")
            previous, t = t, t + step
        return float("nan")
