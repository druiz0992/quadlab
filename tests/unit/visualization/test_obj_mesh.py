"""OBJ meshes and pre-tiled textures for Rerun (visualization/obj_mesh.py)."""

from pathlib import Path

import numpy as np
import pytest

from robotlab.simulation.gazebo.textures import write_box_obj, write_png
from robotlab.visualization.gazebo_rerun import GazeboRerunRecorder
from robotlab.visualization.obj_mesh import read_obj, textured_faces

TILE = np.random.default_rng(1).integers(0, 256, (8, 8, 3), dtype=np.uint8)


def textured_box(tmp_path: Path) -> Path:
    write_png(tmp_path / "box.png", TILE)
    write_box_obj(tmp_path / "box.obj", (4.0, 2.0, 1.0), tile_m=2.0)
    return tmp_path / "box.obj"


def gazebo_sample(uv: np.ndarray) -> np.ndarray:
    """What a repeating sampler reads at OBJ coordinates ``uv`` (v up)."""
    u, v = uv - np.floor(uv)
    row = min(int((1 - v) * TILE.shape[0]), TILE.shape[0] - 1)
    return TILE[row, min(int(u * TILE.shape[1]), TILE.shape[1] - 1)]


def test_read_obj_finds_corners_and_texture(tmp_path):
    mesh = read_obj(textured_box(tmp_path))
    assert mesh.positions.shape == (24, 3) and mesh.texcoords.shape == (24, 2)
    assert mesh.corners.shape == (12, 3, 3) and mesh.texture == tmp_path / "box.png"


def test_faces_carry_the_tile_repeated_over_their_extent(tmp_path):
    mesh = read_obj(textured_box(tmp_path))
    faces = textured_faces(mesh, TILE)
    assert len(faces) == 6
    shapes = sorted(face.image.shape[:2] for face in faces)
    # 4 × 2 m faces: 2 × 1 tiles; 4 × 1 m: 2 × 0.5; 2 × 1 m: 1 × 0.5.
    assert shapes == [(4, 8), (4, 8), (4, 16), (4, 16), (8, 16), (8, 16)]
    rng = np.random.default_rng(0)
    for face, triangles in zip(faces, _triangles_by_normal(mesh)):
        assert face.texcoords.min() >= 0 and face.texcoords.max() <= 1
        for _ in range(20):
            weights = rng.dirichlet([1, 1, 1])
            corner = rng.integers(len(face.triangles))
            new_uv = weights @ face.texcoords[face.triangles[corner]]
            old_uv = weights @ mesh.texcoords[triangles[corner][:, 1]]
            rows, cols = face.image.shape[:2]
            pixel = face.image[
                min(int(new_uv[1] * rows), rows - 1),
                min(int(new_uv[0] * cols), cols - 1),
            ]
            np.testing.assert_array_equal(pixel, gazebo_sample(old_uv))


def _triangles_by_normal(mesh):
    return [
        mesh.corners[mesh.corners[:, 0, 2] == normal]
        for normal in np.unique(mesh.corners[:, 0, 2])
    ]


def test_plain_meshes_have_no_texture(tmp_path):
    (tmp_path / "terrain.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//1 3//1\n"
    )
    mesh = read_obj(tmp_path / "terrain.obj")
    assert mesh.texture is None and mesh.corners[0, :, 0].tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="texture coordinates"):
        textured_faces(mesh, TILE)
    (tmp_path / "bad.obj").write_text("v 0 0 0\nf a b c\n")
    with pytest.raises(ValueError, match="malformed"):
        read_obj(tmp_path / "bad.obj")


class LoggingRerun:
    def __init__(self):
        self.logged = []

    def Mesh3D(self, **fields):
        return fields

    def log(self, path, entity, static=False):
        self.logged.append((path, entity))


def test_recorder_logs_textured_faces_and_plain_terrain(tmp_path):
    recorder = GazeboRerunRecorder.__new__(GazeboRerunRecorder)
    recorder.rr = LoggingRerun()
    recorder._log_obj_mesh(
        "world/obstacles/wall/mesh0", textured_box(tmp_path).as_uri()
    )
    logged = recorder.rr.logged
    assert [path for path, _ in logged] == [
        f"world/obstacles/wall/mesh0/face{i}" for i in range(6)
    ]
    assert all(entity["albedo_texture"].dtype == np.uint8 for _, entity in logged)
    (tmp_path / "terrain.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    recorder.rr.logged.clear()
    recorder._log_obj_mesh("world/terrain", (tmp_path / "terrain.obj").as_uri())
    [(path, entity)] = recorder.rr.logged
    assert path == "world/terrain" and "vertex_colors" in entity
