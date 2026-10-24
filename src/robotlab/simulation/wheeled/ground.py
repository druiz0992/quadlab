"""Ground under the wheels: ray casts and surface properties for wheeled vehicles.

Wheels do not collide in Gazebo (see ``model.py``): each suspension casts a
ray along its strut against an analytic copy of the scene ground, which is

- the seeded terrain height field (``simulation/terrain.py``) inside its
  square, else the flat plane z = 0, both only within the ground area
  (``ground_size_m``; beyond it nothing supports a wheel);
- the tops of static scene boxes (``obstacles``), so a vehicle can drive
  onto a ramp-free platform only if its strut starts above the top; a box
  side is a wall that only the Gazebo chassis and wheel bump-stop
  collisions feel.

Surface zones (``scene["surfaces"]``) scale tyre friction and rolling
resistance by region: axis-aligned-or-rotated rectangles and polyline
strips (a road or race track: a centreline with a width). Later zones
override earlier ones; outside every zone ``scene["ground_surface"]``
applies (default: scale 1). For speed the zones are rasterized once into a
grid of surface indices (``RASTER_RESOLUTION_M``, cell centres tested
exactly), so a zone boundary is resolved to within half a cell diagonal
(0.18 m); a lookup is one array index per wheel. Units: m, world ENU.

Gazebo's terrain collision mesh is the same height field sampled at the
terrain resolution, so wheels and chassis see slightly different surfaces
(the mesh interpolation error); see docs/8_ground_vehicles.md.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..terrain import Terrain

# Fixed-point iterations for a ray against the height field (converges for
# struts steeper than the terrain slope; vehicles are upright).
TERRAIN_ITERATIONS = 4
# Surface raster cell size (m); see module docs.
RASTER_RESOLUTION_M = 0.25

SURFACE_KEYS = {"name", "friction_scale", "rolling_resistance_scale"}
ZONE_KEYS = SURFACE_KEYS | {
    "shape",
    "center_m",
    "size_m",
    "yaw_deg",
    "points_m",
    "width_m",
    "closed",
}
ZONE_SHAPES = {"rectangle", "strip"}


@dataclass(frozen=True)
class Surface:
    """Friction and rolling-resistance multipliers of one surface material."""

    name: str
    friction_scale: float = 1.0
    rolling_resistance_scale: float = 1.0


@dataclass(frozen=True)
class SurfaceZone:
    """A region with one surface. ``rectangle``: centre, size, yaw;
    ``strip``: polyline ``points`` and full ``width_m``."""

    surface: Surface
    shape: str
    center_m: tuple[float, float] = (0.0, 0.0)
    size_m: tuple[float, float] = (0.0, 0.0)
    yaw_rad: float = 0.0
    points_m: tuple[tuple[float, float], ...] = ()
    width_m: float = 0.0
    closed: bool = False

    def contains(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray:
        """Boolean mask of points inside the zone."""
        if self.shape == "rectangle":
            c, s = np.cos(self.yaw_rad), np.sin(self.yaw_rad)
            dx, dy = x - self.center_m[0], y - self.center_m[1]
            u, v = c * dx + s * dy, -s * dx + c * dy
            return (np.abs(u) <= self.size_m[0] / 2) & (np.abs(v) <= self.size_m[1] / 2)
        return strip_distance(self.points_m, self.closed, x, y) <= self.width_m / 2


def strip_distance(
    points: Sequence[Sequence[float]],
    closed: bool,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Distance (m) from each point to a polyline (closed: last joins first)."""
    p = np.asarray(points, float)
    if closed:
        p = np.vstack([p, p[:1]])
    a, b = p[:-1], p[1:]
    q = np.stack([np.asarray(x, float), np.asarray(y, float)], axis=-1)[..., None, :]
    ab = b - a
    length2 = np.maximum(np.sum(ab * ab, axis=1), 1e-12)
    t = np.clip(np.sum((q - a) * ab, axis=-1) / length2, 0.0, 1.0)
    nearest = a + t[..., None] * ab
    return np.min(np.linalg.norm(q - nearest, axis=-1), axis=-1)


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not np.isfinite(value) or (positive and value <= 0) or value < 0:
        raise ValueError(f"{name} must be {'positive' if positive else '>= 0'}")
    return float(value)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def parse_surface(
    entry: Mapping[str, Any], name: str, allowed: set[str] = SURFACE_KEYS
) -> Surface:
    """Validated ``Surface`` from a scene mapping; raises ValueError."""
    if not isinstance(entry, Mapping):
        raise ValueError(f"{name} must be an object")
    unknown = set(entry) - allowed
    if unknown:
        raise ValueError(f"{name}: unknown keys {sorted(unknown)}")
    return Surface(
        name=str(entry.get("name", name)),
        friction_scale=_number(
            entry.get("friction_scale", 1.0), f"{name}.friction_scale", positive=True
        ),
        rolling_resistance_scale=_number(
            entry.get("rolling_resistance_scale", 1.0),
            f"{name}.rolling_resistance_scale",
        ),
    )


def _pair(value: Any, name: str) -> tuple[float, float]:
    array = np.asarray(value, float)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be two finite numbers")
    return float(array[0]), float(array[1])


def parse_zone(entry: Mapping[str, Any], index: int) -> SurfaceZone:
    """Validated ``SurfaceZone`` from one ``scene["surfaces"]`` entry."""
    name = f"surfaces[{index}]"
    surface = parse_surface(entry, name, ZONE_KEYS)
    shape = entry.get("shape")
    if shape not in ZONE_SHAPES:
        raise ValueError(f"{name}.shape must be one of {sorted(ZONE_SHAPES)}")
    if shape == "rectangle":
        size = _pair(entry.get("size_m"), f"{name}.size_m")
        if min(size) <= 0:
            raise ValueError(f"{name}.size_m must be positive")
        return SurfaceZone(
            surface,
            shape,
            center_m=_pair(entry.get("center_m"), f"{name}.center_m"),
            size_m=size,
            yaw_rad=float(
                np.radians(_finite(entry.get("yaw_deg", 0.0), f"{name}.yaw_deg"))
            ),
        )
    points = tuple(
        _pair(p, f"{name}.points_m[{i}]")
        for i, p in enumerate(entry.get("points_m", []))
    )
    if len(points) < 2:
        raise ValueError(f"{name}.points_m needs at least two points")
    closed = entry.get("closed", False)
    if not isinstance(closed, bool):
        raise ValueError(f"{name}.closed must be true or false")
    return SurfaceZone(
        surface,
        shape,
        points_m=points,
        width_m=_number(entry.get("width_m"), f"{name}.width_m", positive=True),
        closed=closed,
    )


def validate_surfaces(scene: Mapping[str, Any]) -> None:
    """Raise ValueError for an invalid ``surfaces`` / ``ground_surface``."""
    surfaces_from_scene(scene)


def surfaces_from_scene(
    scene: Mapping[str, Any],
) -> tuple[Surface, tuple[SurfaceZone, ...]]:
    """Default surface and the zones of a loaded scene."""
    default = parse_surface(
        {"name": "ground", **scene.get("ground_surface", {})}, "ground_surface"
    )
    zones = scene.get("surfaces", [])
    if not isinstance(zones, list):
        raise ValueError("surfaces must be a list")
    return default, tuple(parse_zone(entry, i) for i, entry in enumerate(zones))


class SurfaceRaster:
    """Grid of surface indices (0: default, k: zone k−1) over the zones'
    bounding box; later zones overwrite earlier ones (see module docs)."""

    def __init__(
        self,
        default: Surface,
        zones: Sequence[SurfaceZone],
        resolution_m: float = RASTER_RESOLUTION_M,
    ) -> None:
        self.resolution_m = resolution_m
        surfaces = [default, *(zone.surface for zone in zones)]
        self.friction = np.array([s.friction_scale for s in surfaces])
        self.rolling = np.array([s.rolling_resistance_scale for s in surfaces])
        self.names = [s.name for s in surfaces]
        boxes = (
            np.array([_zone_bounds(zone) for zone in zones])
            if zones
            else np.zeros((0, 4))
        )
        if not len(boxes):
            self.origin = np.zeros(2)
            self.grid = np.zeros((0, 0), np.uint8)
            return
        low, high = boxes[:, :2].min(axis=0), boxes[:, 2:].max(axis=0)
        self.origin = low
        shape = np.ceil((high - low) / resolution_m).astype(int) + 1
        self.grid = np.zeros(tuple(shape), np.uint16 if len(zones) > 254 else np.uint8)
        for index, zone in enumerate(zones, start=1):
            self._paint(zone, index)

    def _paint(self, zone: SurfaceZone, index: int) -> None:
        """Mark the cells whose centres lie in ``zone`` (per strip segment,
        only within that segment's bounding box)."""
        if zone.shape == "rectangle":
            parts = [zone]
        else:
            points = list(zone.points_m) + ([zone.points_m[0]] if zone.closed else [])
            parts = [
                SurfaceZone(
                    zone.surface, "strip", points_m=(a, b), width_m=zone.width_m
                )
                for a, b in zip(points[:-1], points[1:])
            ]
        for part in parts:
            x0, y0, x1, y1 = _zone_bounds(part)
            i0, j0 = self._cell(x0, y0)
            i1, j1 = self._cell(x1, y1)
            ii, jj = np.meshgrid(
                np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij"
            )
            x = self.origin[0] + ii * self.resolution_m
            y = self.origin[1] + jj * self.resolution_m
            inside = part.contains(x.ravel(), y.ravel()).reshape(ii.shape)
            self.grid[ii[inside], jj[inside]] = index

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        i = int(
            np.clip(
                round((x - self.origin[0]) / self.resolution_m),
                0,
                self.grid.shape[0] - 1,
            )
        )
        j = int(
            np.clip(
                round((y - self.origin[1]) / self.resolution_m),
                0,
                self.grid.shape[1] - 1,
            )
        )
        return i, j

    def lookup(self, x: NDArray, y: NDArray) -> NDArray[np.int64]:
        """Surface index per point (0 outside the raster)."""
        if not self.grid.size:
            return np.zeros(np.shape(x), np.int64)
        i = np.rint((np.asarray(x) - self.origin[0]) / self.resolution_m).astype(
            np.int64
        )
        j = np.rint((np.asarray(y) - self.origin[1]) / self.resolution_m).astype(
            np.int64
        )
        inside = (
            (i >= 0) & (j >= 0) & (i < self.grid.shape[0]) & (j < self.grid.shape[1])
        )
        index = np.zeros(np.shape(i), np.int64)
        index[inside] = self.grid[i[inside], j[inside]]
        return index


def _zone_bounds(zone: SurfaceZone) -> tuple[float, float, float, float]:
    """Axis-aligned bounds (x0, y0, x1, y1) of a zone (m)."""
    if zone.shape == "rectangle":
        c, s = abs(np.cos(zone.yaw_rad)), abs(np.sin(zone.yaw_rad))
        half_x = 0.5 * (c * zone.size_m[0] + s * zone.size_m[1])
        half_y = 0.5 * (s * zone.size_m[0] + c * zone.size_m[1])
        x, y = zone.center_m
        return x - half_x, y - half_y, x + half_x, y + half_y
    points = np.asarray(zone.points_m, float)
    margin = zone.width_m / 2
    low, high = points.min(axis=0) - margin, points.max(axis=0) + margin
    return float(low[0]), float(low[1]), float(high[0]), float(high[1])


@dataclass(frozen=True)
class RayHits:
    """Per-ray results: ``hit`` mask, ``distance_m`` along the ray (NaN when
    missed), unit ``normal``, and surface multipliers at the hit point."""

    hit: NDArray[np.bool_]
    distance_m: NDArray[np.float64]
    normal: NDArray[np.float64]
    friction_scale: NDArray[np.float64]
    rolling_resistance_scale: NDArray[np.float64]


class GroundModel:
    """Analytic scene ground for wheel ray casts (see module docs)."""

    def __init__(
        self,
        terrain: Terrain | None,
        boxes: Sequence[Mapping[str, Any]],
        ground_half_m: float,
        default_surface: Surface = Surface("ground"),
        zones: Sequence[SurfaceZone] = (),
    ) -> None:
        self._terrain = terrain
        self._ground_half_m = ground_half_m
        self._default = default_surface
        self._zones = tuple(zones)
        self._raster = SurfaceRaster(default_surface, self._zones)
        if boxes:
            centre = np.array([b["position"] for b in boxes], float)
            size = np.array([b["size"] for b in boxes], float)
            self._box_low = centre[:, :2] - size[:, :2] / 2
            self._box_high = centre[:, :2] + size[:, :2] / 2
            self._box_top = centre[:, 2] + size[:, 2] / 2
        else:
            self._box_low = self._box_high = np.zeros((0, 2))
            self._box_top = np.zeros(0)

    @classmethod
    def from_scene(cls, scene: Mapping[str, Any], terrain: Terrain | None):
        """Ground model of a loaded scene (``world.load_scene``)."""
        default, zones = surfaces_from_scene(scene)
        solid = [o for o in scene["obstacles"] if not o.get("visual_only", False)]
        return cls(
            terrain,
            solid,
            scene.get("ground_size_m", 100.0) / 2,
            default,
            zones,
        )

    def height(self, x: NDArray, y: NDArray) -> NDArray[np.float64]:
        """Ground (terrain or plane) height (m), ignoring boxes."""
        if self._terrain is None:
            return np.zeros(np.broadcast(x, y).shape)
        return self._terrain.height(x, y)

    def cast(
        self, origin: NDArray, direction: NDArray, max_distance_m: NDArray
    ) -> RayHits:
        """Nearest hit of rays ``origin + t·direction`` (n×3, unit directions
        pointing down) for 0 ≤ t ≤ max_distance_m."""
        o, d = np.asarray(origin, float), np.asarray(direction, float)
        down = np.maximum(-d[:, 2], 1e-6)
        t = self._ground_distance(o, d, down)
        normal = self._ground_normal(o + t[:, None] * d)
        t_box, on_box = self._box_distance(o, d, down)
        use_box = on_box & (t_box < t)
        t = np.where(use_box, t_box, t)
        normal = np.where(use_box[:, None], np.array([0.0, 0.0, 1.0]), normal)
        point = o + t[:, None] * d
        inside = np.all(np.abs(point[:, :2]) <= self._ground_half_m, axis=1)
        hit = (t >= 0) & (t <= max_distance_m) & (inside | use_box)
        friction, rolling = self._surface(point[:, 0], point[:, 1])
        return RayHits(hit, np.where(hit, t, np.nan), normal, friction, rolling)

    def _ground_distance(self, o: NDArray, d: NDArray, down: NDArray) -> NDArray:
        # Fixed-point iteration t = (o_z − h(x(t), y(t))) / (−d_z).
        t = (o[:, 2] - self.height(o[:, 0], o[:, 1])) / down
        if self._terrain is not None:
            for _ in range(TERRAIN_ITERATIONS):
                p = o + t[:, None] * d
                t = (o[:, 2] - self.height(p[:, 0], p[:, 1])) / down
        return t

    def _ground_normal(self, point: NDArray) -> NDArray[np.float64]:
        if self._terrain is None:
            return np.tile([0.0, 0.0, 1.0], (len(point), 1))
        return self._terrain.normals(point[:, 0], point[:, 1])

    def _box_distance(
        self, o: NDArray, d: NDArray, down: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Distance to the highest box top below each ray origin."""
        if not len(self._box_top):
            return np.full(len(o), np.inf), np.zeros(len(o), bool)
        t = (o[:, 2:3] - self._box_top[None, :]) / down[:, None]
        p = o[:, None, :2] + t[..., None] * d[:, None, :2]
        inside = np.all((p >= self._box_low) & (p <= self._box_high), axis=2)
        valid = inside & (t >= 0)
        best = np.where(valid, t, np.inf).min(axis=1)
        return best, np.isfinite(best)

    def _surface(self, x: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
        index = self._raster.lookup(x, y)
        return self._raster.friction[index], self._raster.rolling[index]

    def surface_name(self, x: float, y: float) -> str:
        """Name of the surface at one point (telemetry and health)."""
        return self._raster.names[
            int(self._raster.lookup(np.array([x]), np.array([y]))[0])
        ]
