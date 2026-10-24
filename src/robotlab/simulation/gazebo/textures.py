"""Procedural surface textures for Gazebo worlds (visual odometry testing).

Feature-tracking estimators need visual texture; flat colours give them
nothing. A texture spec in a scene (``ground_texture``, an obstacle's
``texture``) becomes a seamless square tile, generated deterministically
into the world's output directory as a PNG, repeated over the surface:

    {"pattern": "speckle", "seed": 3, "tile_m": 8, "pixels_per_m": 64,
     "contrast": 1.0, "base": [0.55, 0.55, 0.52]}

Patterns (grey-level shading of the ``base`` RGB, 0..1):

- ``speckle``: multi-octave value noise plus random dark and light blobs
  of 5–30 cm, feature-rich at every scale (asphalt, gravel, concrete);
- ``bricks``: 0.4 m × 0.2 m blocks in running bond with 2 cm mortar and
  slight per-block shading, repetitive (a stress test for data
  association);
- ``checker``: 1 m squares.

Tiles wrap seamlessly (periodic noise grid and blobs; ``tile_m`` is
rounded to whole blocks or squares for the regular patterns), so the only
repetition is the tile period. Textured surfaces are drawn with a box
mesh (``write_box_obj``) whose texture coordinates are metres / ``tile_m``
on every face, so the density is the same everywhere; collision geometry
is the unchanged box. Defaults: ``DEFAULTS`` (an 8 m tile at 64 px/m,
512 × 512 px).
"""

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import struct
from typing import Any
import zlib

import numpy as np
from numpy.typing import NDArray

PATTERNS = ("speckle", "bricks", "checker")
MAX_PIXELS = 4096
DEFAULTS: dict[str, Any] = dict(
    seed=0, tile_m=8.0, pixels_per_m=64.0, contrast=1.0, base=[0.55, 0.55, 0.52]
)
# Period of the regular patterns (m): tiles are whole multiples of it.
PERIOD_M = {"bricks": 0.8, "checker": 2.0}


def validate_texture(spec: Any, name: str) -> dict[str, Any]:
    """Texture spec with defaults filled in; ``ValueError`` names ``name``."""
    if not isinstance(spec, Mapping):
        raise ValueError(f"{name}: texture must be an object")
    unknown = set(spec) - {"pattern", *DEFAULTS}
    if unknown:
        raise ValueError(f"{name}: unknown texture keys {sorted(unknown)}")
    result = {**DEFAULTS, **spec}
    if result.get("pattern") not in PATTERNS:
        raise ValueError(f"{name}: texture pattern must be one of {PATTERNS}")
    if not isinstance(result["seed"], int) or isinstance(result["seed"], bool):
        raise ValueError(f"{name}: texture seed must be an integer")
    if not 0.5 <= _number(result["tile_m"], name) <= 100:
        raise ValueError(f"{name}: tile_m must be in [0.5, 100]")
    if not 4 <= _number(result["pixels_per_m"], name) <= 512:
        raise ValueError(f"{name}: pixels_per_m must be in [4, 512]")
    if not 0 <= _number(result["contrast"], name) <= 1:
        raise ValueError(f"{name}: contrast must be in [0, 1]")
    period = PERIOD_M.get(result["pattern"])
    if period is not None:  # whole periods, so the tile wraps seamlessly
        result["tile_m"] = max(1, round(result["tile_m"] / period)) * period
    if round(result["tile_m"] * result["pixels_per_m"]) > MAX_PIXELS:
        raise ValueError(f"{name}: tile_m × pixels_per_m exceeds {MAX_PIXELS} px")
    result["base"] = validate_color(result["base"], f"{name}: base")
    return result


def validate_color(value: Any, name: str) -> list[float]:
    """RGB in [0, 1]."""
    rgb = np.asarray(value, dtype=float)
    if rgb.shape != (3,) or not np.all((rgb >= 0) & (rgb <= 1)):
        raise ValueError(f"{name} must be three values in [0, 1]")
    return [float(v) for v in rgb]


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: texture values must be numbers")
    return float(value)


def texture_image(spec: Mapping[str, Any]) -> NDArray[np.uint8]:
    """Square RGB tile (pixels × pixels × 3) of a validated spec."""
    pixels = round(spec["tile_m"] * spec["pixels_per_m"])
    metres_per_px = spec["tile_m"] / pixels
    rng = np.random.default_rng(spec["seed"])
    pattern = spec["pattern"]
    if pattern == "speckle":
        level = _speckle(pixels, metres_per_px, rng)
    elif pattern == "bricks":
        level = _bricks(pixels, metres_per_px, rng)
    else:
        level = _checker(pixels, metres_per_px)
    shade = 1 + spec["contrast"] * (level - 0.5)  # 0.5 .. 1.5 at full contrast
    rgb = np.clip(np.asarray(spec["base"])[None, None, :] * shade[..., None], 0, 1)
    return np.rint(rgb * 255).astype(np.uint8)


def _value_noise(pixels: int, cells: int, rng) -> NDArray[np.float64]:
    """Bilinear value noise on a periodic ``cells`` × ``cells`` grid."""
    grid = rng.random((cells, cells))
    coordinate = np.arange(pixels) * cells / pixels
    low = coordinate.astype(int)
    high = (low + 1) % cells  # wrap: the tile is seamless
    frac = coordinate - low
    fy, fx = frac[:, None], frac[None, :]
    a, b = grid[np.ix_(low, low)], grid[np.ix_(low, high)]
    c, d = grid[np.ix_(high, low)], grid[np.ix_(high, high)]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _speckle(pixels: int, metres_per_px: float, rng) -> NDArray[np.float64]:
    """Octaves from 2 m cells down to 2 px, plus wrapped blobs of 5–30 cm."""
    tile_m = pixels * metres_per_px
    level, total = np.zeros((pixels, pixels)), 0.0
    cell_m = 2.0
    while cell_m >= 2 * metres_per_px:
        cells = max(1, round(tile_m / cell_m))
        weight = cell_m**0.5
        level += weight * _value_noise(pixels, cells, rng)
        total += weight
        cell_m /= 2
    level = level / total if total else np.full((pixels, pixels), 0.5)
    index = np.arange(pixels)
    for _ in range(min(int(tile_m**2 * 2), 4000)):
        radius = rng.uniform(0.05, 0.3) / metres_per_px
        centre = rng.uniform(0, pixels, 2)
        # Wrapped distances, so blobs continue across the tile edges.
        dy = np.abs(index - centre[0])
        dx = np.abs(index - centre[1])
        dy, dx = np.minimum(dy, pixels - dy), np.minimum(dx, pixels - dx)
        rows, cols = np.flatnonzero(dy <= radius), np.flatnonzero(dx <= radius)
        inside = dy[rows, None] ** 2 + dx[None, cols] ** 2 <= radius**2
        patch = level[np.ix_(rows, cols)]
        patch[inside] = rng.choice([0.1, 0.9])
        level[np.ix_(rows, cols)] = patch
    return level


def _bricks(pixels: int, metres_per_px: float, rng) -> NDArray[np.float64]:
    """0.4 m × 0.2 m blocks in running bond, 2 cm mortar (tile: whole
    pairs of courses and blocks, so it wraps)."""
    metres = (np.arange(pixels) + 0.5) * metres_per_px
    course = np.floor(metres / 0.2).astype(int)
    shifted = np.mod(
        metres[None, :] + 0.2 * (course[:, None] % 2), pixels * metres_per_px
    )
    column = np.floor(shifted / 0.4).astype(int)
    shades = rng.uniform(0.35, 0.65, (course.max() + 1, column.max() + 1))
    level = shades[course[:, None], column]
    mortar = (np.mod(metres, 0.2)[:, None] < 0.02) | (np.mod(shifted, 0.4) < 0.02)
    return np.where(mortar, 0.9, level)


def _checker(pixels: int, metres_per_px: float) -> NDArray[np.float64]:
    metres = (np.arange(pixels) + 0.5) * metres_per_px
    even = (np.floor(metres)[None, :] + np.floor(metres)[:, None]) % 2 == 0
    return np.where(even, 0.25, 0.75)


def write_png(path: Path, image: NDArray[np.uint8]) -> None:
    """Minimal 8-bit RGB PNG writer (no dependency beyond zlib)."""
    rows, cols, _ = image.shape
    raw = b"".join(b"\x00" + image[r].tobytes() for r in range(rows))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", cols, rows, 8, 2, 0, 0, 0)
    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def write_box_obj(path: Path, size_m: Sequence[float], tile_m: float) -> None:
    """Wavefront OBJ of a box centred at the origin, two triangles per face,
    with outward normals and texture coordinates in metres / ``tile_m``
    along each face's axes (so a tile repeats at the same scale on every
    face), and a ``.mtl`` beside it using the ``.png`` of the same stem.
    No blank lines (the Rerun loader reads every line)."""
    half = np.asarray(size_m, float) / 2
    path = Path(path)
    # A material naming the tile next to the mesh (same stem): Gazebo warns
    # about OBJ shapes without one; the SDF material still applies.
    path.with_suffix(".mtl").write_text(
        f"newmtl surface\nKd 1 1 1\nmap_Kd {path.with_suffix('.png').name}\n"
    )
    lines: list[str] = [f"mtllib {path.with_suffix('.mtl').name}", "usemtl surface"]
    for face, (axis, sign) in enumerate((a, s) for a in range(3) for s in (1, -1)):
        u_axis, v_axis = [i for i in range(3) if i != axis]
        corners = []
        for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            point = np.zeros(3)
            point[axis] = sign * half[axis]
            point[u_axis], point[v_axis] = du * half[u_axis], dv * half[v_axis]
            corners.append(point)
        normal = np.zeros(3)
        normal[axis] = sign
        winding = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        if winding @ normal < 0:
            corners.reverse()  # counter-clockwise seen from outside
        lines += [f"v {p[0]:.6g} {p[1]:.6g} {p[2]:.6g}" for p in corners]
        lines += [
            f"vt {(p[u_axis] + half[u_axis]) / tile_m:.6g} "
            f"{(p[v_axis] + half[v_axis]) / tile_m:.6g}"
            for p in corners
        ]
        lines.append(f"vn {normal[0]:g} {normal[1]:g} {normal[2]:g}")
        k, n = 4 * face + 1, face + 1
        lines.append(f"f {k}/{k}/{n} {k + 1}/{k + 1}/{n} {k + 2}/{k + 2}/{n}")
        lines.append(f"f {k}/{k}/{n} {k + 2}/{k + 2}/{n} {k + 3}/{k + 3}/{n}")
    Path(path).write_text("\n".join(lines) + "\n")
