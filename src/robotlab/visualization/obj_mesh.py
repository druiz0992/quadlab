"""Read generated OBJ meshes for Rerun, with their tiled textures.

RobotLab writes two kinds of OBJ: terrain and surface meshes (positions,
faces, maybe normals) and textured box meshes
(``simulation/gazebo/textures.write_box_obj``), whose texture coordinates
are metres / tile, beyond 1, and whose ``.mtl`` names the tile PNG
(``map_Kd``). Gazebo repeats the tile; this module does not rely on the
viewer doing the same: ``textured_faces`` splits a textured mesh into its
planar faces (triangles sharing a normal) and gives each an image of the
tile repeated over that face's extent, with texture coordinates in [0, 1].

Texture coordinate v runs up the image in OBJ but down image rows, so
rows are flipped; images larger than ``MAX_FACE_PIXELS`` per side are
subsampled (the viewer only needs a likeness). Raises ``ValueError`` for
malformed files.
"""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

MAX_FACE_PIXELS = 2048


@dataclass(frozen=True)
class ObjMesh:
    """Positions (n×3), and per triangle the (position, texcoord, normal)
    indices of its corners (m×3×3, −1 where absent)."""

    positions: NDArray[np.float64]
    texcoords: NDArray[np.float64]  # k×2 (empty without vt lines)
    corners: NDArray[np.int64]
    texture: Path | None  # map_Kd of the mesh's material file


@dataclass(frozen=True)
class TexturedFace:
    """One planar face ready for ``rr.Mesh3D``."""

    positions: NDArray[np.float64]
    triangles: NDArray[np.int64]
    texcoords: NDArray[np.float64]  # in [0, 1]
    image: NDArray[np.uint8]  # rows × columns × 3


def read_obj(path: Path) -> ObjMesh:
    """Parse ``v``, ``vt``, ``f`` (``v``, ``v/vt``, ``v//vn``, ``v/vt/vn``,
    triangles or fans) and the ``mtllib`` texture."""
    path = Path(path)
    positions, texcoords, corners = [], [], []
    texture = None
    for line in path.read_text().splitlines():
        kind, *values = line.split() or [""]
        if kind == "v":
            positions.append([float(v) for v in values[:3]])
        elif kind == "vt":
            texcoords.append([float(v) for v in values[:2]])
        elif kind == "f":
            entries = [_corner(entry) for entry in values]
            for i in range(1, len(entries) - 1):  # fan for polygons
                corners.append([entries[0], entries[i], entries[i + 1]])
        elif kind == "mtllib":
            texture = _material_texture(path.parent / values[0])
    return ObjMesh(
        np.array(positions, float).reshape(-1, 3),
        np.array(texcoords, float).reshape(-1, 2),
        np.array(corners, np.int64).reshape(-1, 3, 3),
        texture,
    )


def _corner(entry: str) -> list[int]:
    parts = entry.split("/") + ["", ""]
    try:
        return [int(p) - 1 if p else -1 for p in parts[:3]]
    except ValueError:
        raise ValueError(f"malformed OBJ face entry {entry!r}") from None


def _material_texture(mtl: Path) -> Path | None:
    if not mtl.is_file():
        return None
    for line in mtl.read_text().splitlines():
        kind, *values = line.split() or [""]
        if kind == "map_Kd" and values:
            return mtl.parent / values[0]
    return None


def textured_faces(mesh: ObjMesh, tile: NDArray[np.uint8]) -> list[TexturedFace]:
    """Split a textured mesh into faces with pre-tiled images (module docs).

    Faces are the triangles sharing a normal index; ``tile`` is the
    texture image (rows × columns × 3, uint8).
    """
    if not len(mesh.texcoords) or np.any(mesh.corners[:, :, 1] < 0):
        raise ValueError("mesh has no texture coordinates")
    faces = []
    for normal in np.unique(mesh.corners[:, 0, 2]):
        triangles = mesh.corners[mesh.corners[:, 0, 2] == normal]
        faces.append(_face(mesh, triangles, tile))
    return faces


def _face(mesh: ObjMesh, triangles, tile) -> TexturedFace:
    flat = triangles[:, :, :2].reshape(-1, 2)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    uv = mesh.texcoords[unique[:, 1]]
    low = np.floor(uv.min(axis=0))
    extent = np.maximum(uv.max(axis=0) - low, 1e-9)  # in tiles
    image = _tiled(tile, extent)
    # v up in OBJ, rows down in the image.
    local = (uv - low) / extent
    texcoords = np.column_stack([local[:, 0], 1.0 - local[:, 1]])
    return TexturedFace(
        mesh.positions[unique[:, 0]],
        inverse.reshape(-1, 3).astype(np.int64),
        texcoords,
        image,
    )


def _tiled(tile: NDArray[np.uint8], extent: NDArray[np.float64]) -> NDArray[np.uint8]:
    """The tile repeated over ``extent`` (u, v tiles), v up, cropped exactly."""
    rows, cols = tile.shape[:2]
    width = max(1, round(extent[0] * cols))
    height = max(1, round(extent[1] * rows))
    repeats = (math.ceil(height / rows), math.ceil(width / cols), 1)
    # Image row 0 is the top of the face (largest v), so tile from the bottom.
    image = np.tile(tile, repeats)[-height:, :width]
    step = max(1, math.ceil(max(image.shape[:2]) / MAX_FACE_PIXELS))
    return np.ascontiguousarray(image[::step, ::step])
