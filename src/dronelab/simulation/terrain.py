"""Seeded rolling terrain: one height function for Gazebo geometry and Python models.

The terrain is a sum of Gaussian hills on top of the flat ground plane
(z = 0), multiplied by a smooth blend that keeps a disc around the origin
flat for spawning:

    h(x, y) = blend(r) · Σ_k height_k · exp(−|p − c_k|² / (2 σ_k²))

Hill centres, heights (up to ``max_height_m``), and widths σ (within
``hill_radius_m``) come from a local seeded RNG, so a seed reproduces the
terrain. The same function generates the triangle mesh Gazebo uses for
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
}


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
    return data


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

    def height(self, x: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
        """Terrain height (m) at world x, y; 0 outside the terrain square."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        h = np.zeros(np.broadcast(x, y).shape)
        for (cx, cy), height, sigma in zip(self._centres, self._heights, self._sigmas):
            h = h + height * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        r = np.hypot(x, y)
        # Smoothstep from 0 inside the flat disc to 1 at twice its radius.
        u = (
            np.clip((r - self._flat) / max(self._flat, 1e-9), 0.0, 1.0)
            if self._flat
            else 1.0
        )
        inside = (np.abs(x) <= self._half[0]) & (np.abs(y) <= self._half[1])
        return np.where(inside, h * u * u * (3 - 2 * u), 0.0)

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
