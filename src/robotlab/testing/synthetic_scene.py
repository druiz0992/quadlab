"""Ray-cast scene for synthetic rangefinder, lidar, and camera data.

The scene is the ground plane z = 0 plus axis-aligned boxes (world ENU, m).
``cast`` intersects many rays at once (slab method for boxes; Kay & Kajiya,
"Ray tracing complex scenes", SIGGRAPH 1986) and returns, per ray, the
distance to the first surface, that surface's index (0 = ground, i + 1 =
box i, -1 = no hit within ``max_range_m``) and its outward unit normal.

``shade`` turns hits into RGB for a synthetic camera: a 1 m checkerboard
ground, a flat colour per box, Lambertian shading from a fixed sun, and a
sky colour where nothing is hit. It is a test fixture, not a renderer:
no shadows, textures beyond the checkerboard, or lens effects.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

SKY_RGB = np.array([170.0, 200.0, 235.0])
GROUND_RGB = (np.array([90.0, 110.0, 80.0]), np.array([150.0, 165.0, 130.0]))
BOX_RGB = np.array(
    [
        [200.0, 80.0, 60.0],
        [70.0, 120.0, 200.0],
        [220.0, 190.0, 70.0],
        [140.0, 90.0, 170.0],
    ]
)
SUN_DIRECTION = np.array([0.3, 0.4, 1.0]) / np.linalg.norm([0.3, 0.4, 1.0])
GROUND_REFLECTIVITY = 0.3
BOX_REFLECTIVITY = 0.6


@dataclass(frozen=True)
class SceneBox:
    """Axis-aligned box from ``min_m`` to ``max_m`` corners (world, m)."""

    min_m: tuple[float, float, float]
    max_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not all(a < b for a, b in zip(self.min_m, self.max_m)):
            raise ValueError(f"box corners must satisfy min < max: {self}")


@dataclass(frozen=True)
class RayHits:
    """Per-ray results of ``SyntheticScene.cast`` (N rays)."""

    distance_m: NDArray[np.float64]  # inf where nothing is hit
    surface: NDArray[np.int64]  # -1 none, 0 ground, i + 1 box i
    normal: NDArray[np.float64]  # N×3 outward normals (zero without a hit)


@dataclass(frozen=True)
class SyntheticScene:
    """Ground plane z = 0 plus ``boxes`` (see module docs)."""

    boxes: tuple[SceneBox, ...] = ()

    def cast(
        self,
        origins_m: NDArray[np.float64],
        directions: NDArray[np.float64],
        max_range_m: float,
    ) -> RayHits:
        """First hits of rays (N×3 origins, N×3 unit directions, world)."""
        origins = np.atleast_2d(np.asarray(origins_m, float))
        directions = np.atleast_2d(np.asarray(directions, float))
        origins = np.broadcast_to(origins, directions.shape)
        count = len(directions)
        surface = np.full(count, -1, dtype=np.int64)
        axis = np.full(count, 2, dtype=np.int64)  # entry axis (ground: z)
        # Ground: hit from above only (rays going down from z > 0).
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = np.where(
                (directions[:, 2] < 0) & (origins[:, 2] > 0),
                -origins[:, 2] / directions[:, 2],
                np.inf,
            )
        surface[np.isfinite(distance)] = 0
        with np.errstate(divide="ignore"):
            inverse = 1.0 / np.ascontiguousarray(directions.T)
        origins_t = np.ascontiguousarray(origins.T)
        for index, box in enumerate(self.boxes):
            t_box, entry_axis = _slab(origins_t, inverse, box)
            nearer = t_box < distance
            distance[nearer] = t_box[nearer]
            surface[nearer] = index + 1
            axis[nearer] = entry_axis[nearer]
        surface[distance > max_range_m] = -1
        distance[surface < 0] = np.inf
        # Outward normal: against the ray along the entry axis (ground: +z).
        normal = np.zeros((count, 3))
        rows = np.flatnonzero(surface >= 0)
        normal[rows, axis[rows]] = -np.sign(directions[rows, axis[rows]])
        return RayHits(distance, surface, normal)

    def shade(
        self,
        origins_m: NDArray[np.float64],
        directions: NDArray[np.float64],
        hits: RayHits,
    ) -> NDArray[np.float64]:
        """N×3 RGB in [0, 255] for camera rays (see module docs)."""
        rgb = np.tile(SKY_RGB, (len(hits.distance_m), 1))
        hit = hits.surface >= 0
        points = origins_m + directions * np.where(hit, hits.distance_m, 0.0)[:, None]
        ground = hits.surface == 0
        checker = (np.floor(points[:, 0]) + np.floor(points[:, 1])) % 2 == 0
        rgb[ground & checker] = GROUND_RGB[0]
        rgb[ground & ~checker] = GROUND_RGB[1]
        boxes = hits.surface > 0
        rgb[boxes] = BOX_RGB[(hits.surface[boxes] - 1) % len(BOX_RGB)]
        light = 0.35 + 0.65 * np.clip(hits.normal @ SUN_DIRECTION, 0.0, 1.0)
        rgb[hit] *= light[hit, None]
        return rgb

    def reflectivity(self, hits: RayHits) -> NDArray[np.float64]:
        """Per-ray lidar reflectivity (0 where nothing is hit)."""
        return np.select(
            [hits.surface == 0, hits.surface > 0],
            [GROUND_REFLECTIVITY, BOX_REFLECTIVITY],
            0.0,
        )


def _slab(
    origins: NDArray[np.float64], inverse: NDArray[np.float64], box: SceneBox
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Entry distance (inf if missed or starting inside) and entry axis.

    Arrays are 3×N (one row per axis: per-axis rows are contiguous, much
    faster than axis-1 reductions over N×3). ``inverse`` is 1 / direction;
    zero components give ±inf slab distances, and ``fmin``/``fmax`` skip
    the NaN of a ray lying exactly on a slab plane.
    """
    low, high = np.asarray(box.min_m)[:, None], np.asarray(box.max_m)[:, None]
    with np.errstate(invalid="ignore"):
        t1, t2 = (low - origins) * inverse, (high - origins) * inverse
    t_near, t_far = np.fmin(t1, t2), np.fmax(t1, t2)
    enter = np.maximum(np.maximum(t_near[0], t_near[1]), t_near[2])
    leave = np.minimum(np.minimum(t_far[0], t_far[1]), t_far[2])
    t = np.where((enter <= leave) & (enter > 0), enter, np.inf)
    return t, np.argmax(t_near, axis=0)


# A 40 m × 40 m walled yard with three pillars, all more than 14 m from the
# origin so the built-in circles (radius ≤ 12 m) stay clear: walls give the
# horizontal lidar and the camera something to see from any point inside.
DEFAULT_SCENE = SyntheticScene(
    (
        SceneBox((-20.0, 20.0, 0.0), (20.0, 20.5, 4.0)),
        SceneBox((-20.0, -20.5, 0.0), (20.0, -20.0, 4.0)),
        SceneBox((20.0, -20.0, 0.0), (20.5, 20.0, 4.0)),
        SceneBox((-20.5, -20.0, 0.0), (-20.0, 20.0, 4.0)),
        SceneBox((13.0, 13.0, 0.0), (14.0, 14.0, 3.0)),
        SceneBox((-14.0, 9.0, 0.0), (-13.0, 10.0, 3.0)),
        SceneBox((3.0, -15.0, 0.0), (4.0, -14.0, 3.0)),
    )
)
