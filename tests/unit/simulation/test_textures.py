"""Procedural textures and the textured yard world."""

import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pytest

from robotlab.simulation.gazebo.textures import (
    texture_image,
    validate_texture,
    write_box_obj,
    write_png,
)
from robotlab.simulation.gazebo.world import build_world, load_scene
from robotlab.simulation.wheeled.ground import GroundModel
from robotlab.testing.synthetic_scene import SceneBox, SyntheticScene

YARD = Path(__file__).resolve().parents[3] / "worlds" / "textured_yard.json"


def test_spec_defaults_and_validation():
    spec = validate_texture({"pattern": "speckle"}, "wall")
    assert spec["tile_m"] == 8.0 and spec["pixels_per_m"] == 64.0
    # Regular patterns round the tile to whole periods, so it wraps.
    assert validate_texture({"pattern": "bricks", "tile_m": 3.0}, "w")[
        "tile_m"
    ] == pytest.approx(3.2)
    for bad, message in (
        ({"pattern": "marble"}, "pattern"),
        ({"pattern": "speckle", "seed": 1.5}, "seed"),
        ({"pattern": "speckle", "contrast": 2}, "contrast"),
        ({"pattern": "speckle", "tile_m": 64, "pixels_per_m": 128}, "exceeds"),
        ({"pattern": "speckle", "base": [1, 2, 3]}, "base"),
        ({"pattern": "speckle", "colour": [1, 1, 1]}, "unknown"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_texture(bad, "wall")


@pytest.mark.parametrize("pattern", ["speckle", "bricks", "checker"])
def test_tiles_are_deterministic_and_textured(pattern):
    spec = validate_texture({"pattern": pattern, "seed": 3, "tile_m": 4}, "t")
    image = texture_image(spec)
    np.testing.assert_array_equal(image, texture_image(spec))
    assert image.astype(float).mean(axis=2).std() > 5


def test_speckle_wraps_without_a_seam():
    spec = validate_texture({"pattern": "speckle", "seed": 3, "tile_m": 4}, "t")
    image = texture_image(spec).astype(float).mean(axis=2)
    for axis in (0, 1):
        inside = np.abs(np.diff(image, axis=axis)).mean()
        edge = np.take(image, 0, axis) - np.take(image, -1, axis)
        assert np.abs(edge).mean() <= 3 * inside + 1


def test_regular_patterns_are_periodic_within_the_tile():
    """A whole number of periods per tile: shifting by one period
    cyclically reproduces the layout, so repeated tiles join exactly."""
    checker = texture_image(validate_texture({"pattern": "checker", "tile_m": 4}, "t"))
    period = checker.shape[1] // 2  # 2 m of a 4 m tile
    np.testing.assert_array_equal(np.roll(checker, period, axis=1), checker)
    # 50 px/m: a period is a whole number of pixels (0.8 m = 40 px).
    bricks = texture_image(
        validate_texture({"pattern": "bricks", "tile_m": 3.2, "pixels_per_m": 50}, "t")
    )
    mortar = (
        bricks.astype(float).mean(axis=2) == bricks.astype(float).mean(axis=2).max()
    )
    pixels_per_m = bricks.shape[1] / 3.2
    np.testing.assert_array_equal(
        np.roll(mortar, round(0.8 * pixels_per_m), axis=1), mortar
    )
    np.testing.assert_array_equal(
        np.roll(mortar, round(0.4 * pixels_per_m), axis=0), mortar
    )


def test_png_round_trip(tmp_path):
    image = np.random.default_rng(0).integers(0, 256, (5, 7, 3), dtype=np.uint8)
    write_png(tmp_path / "t.png", image)
    data = (tmp_path / "t.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    idat_length = struct.unpack(">I", data[33:37])[0]
    raw = zlib.decompress(data[41 : 41 + idat_length])
    rows = np.frombuffer(raw, np.uint8).reshape(height, 1 + 3 * width)
    assert (width, height) == (7, 5) and not rows[:, 0].any()
    np.testing.assert_array_equal(rows[:, 1:].reshape(5, 7, 3), image)


def test_box_mesh_has_outward_faces_and_metric_uvs(tmp_path):
    write_box_obj(tmp_path / "box.obj", (4.0, 2.0, 1.0), tile_m=2.0)
    lines = (tmp_path / "box.obj").read_text().splitlines()
    assert all(line.strip() for line in lines)  # no blank lines
    v = np.array([list(map(float, l.split()[1:])) for l in lines if l.startswith("v ")])
    vt = np.array(
        [list(map(float, l.split()[1:])) for l in lines if l.startswith("vt ")]
    )
    vn = np.array(
        [list(map(float, l.split()[1:])) for l in lines if l.startswith("vn ")]
    )
    faces = [l.split()[1:] for l in lines if l.startswith("f ")]
    assert (
        v.shape == (24, 3)
        and vt.shape == (24, 2)
        and vn.shape == (6, 3)
        and len(faces) == 12
    )
    np.testing.assert_allclose(np.abs(v).max(axis=0), [2.0, 1.0, 0.5])
    assert vt.max() == pytest.approx(2.0)  # 4 m / 2 m tile
    for face in faces:
        index = [int(entry.split("/")[0]) - 1 for entry in face]
        normal = vn[int(face[0].split("/")[2]) - 1]
        a, b, c = v[index]
        assert np.cross(b - a, c - a) @ normal > 0
        assert normal @ a > 0  # outward
    assert "map_Kd box.png" in (tmp_path / "box.mtl").read_text()


def test_scene_appearance_keys_are_validated(tmp_path):
    for extra, message in (
        ({"color": [2, 0, 0]}, "color"),
        ({"visual_only": "yes"}, "visual_only"),
        ({"texture": {"pattern": "plaid"}}, "pattern"),
    ):
        obstacle = {"name": "wall", "position": [0, 0, 1], "size": [1, 1, 2], **extra}
        path = tmp_path / "scene.json"
        path.write_text(json.dumps({"obstacles": [obstacle]}))
        with pytest.raises(ValueError, match=message):
            load_scene(path)


def test_textured_yard_builds_textures_and_marks(tmp_path):
    scene = load_scene(YARD)
    world = ET.parse(build_world(scene, tmp_path / "world.sdf")).getroot()
    maps = [e.text for e in world.iter("albedo_map")]
    assert len(maps) == 11 and all(Path(m).is_file() for m in maps)
    floor = next(m for m in world.iter("model") if m.get("name") == "corridor_floor")
    assert floor.find(".//collision") is None and floor.find(".//visual") is not None
    corridor = next(m for m in world.iter("model") if m.get("name") == "corridor_north")
    assert corridor.find(".//albedo_map") is None  # plain, featureless
    # Markings are not ground for the wheels.
    ground = GroundModel.from_scene(scene, None)
    assert len(ground._box_top) == len(scene["obstacles"]) - 1


def test_corridor_middle_is_degenerate_for_a_horizontal_lidar():
    """From the corridor middle, no beam of a 50 m horizontal scan hits a
    surface facing along the corridor: position along it is unobservable."""
    scene = load_scene(YARD)
    boxes = tuple(
        SceneBox(
            tuple(np.subtract(o["position"], np.divide(o["size"], 2))),
            tuple(np.add(o["position"], np.divide(o["size"], 2))),
        )
        for o in scene["obstacles"]
        if not o.get("visual_only")
    )
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    beams = np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
    hits = SyntheticScene(boxes).cast(np.array([75.0, 0.0, 1.2]), beams, 50.0)
    facing = np.abs(hits.normal[hits.surface >= 0, 0]) > 0.5
    assert (hits.surface >= 0).sum() > 300 and not facing.any()
