"""Terrain ditches: profile, overlap, validation, and the ground frame."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from robotlab.simulation.gazebo.world import build_world
from robotlab.simulation.terrain import Terrain

DITCH = {"points_m": [[-10.0, 5.0], [10.0, 5.0]], "width_m": 2.0, "depth_m": 0.5}


def flat_with(*ditches):
    return Terrain({"hills": 0, "flat_radius_m": 0.0, "ditches": list(ditches)})


def test_cosine_trough_profile():
    terrain = flat_with(DITCH)
    y = np.array([5.0, 5.5, 6.0, 7.0, 4.5])
    np.testing.assert_allclose(
        terrain.height(np.zeros(5), y), [-0.5, -0.25, 0.0, 0.0, -0.25], atol=1e-12
    )
    # Past the end the trough rounds off around the end point.
    assert terrain.height(np.array(10.5), np.array(5.0)) == pytest.approx(-0.25)
    # The steepest wall slope is pi d / w.
    eps = 1e-5
    y0 = 5.5  # mid-wall
    slope = (
        terrain.height(np.array(0.0), np.array(y0 + eps))
        - terrain.height(np.array(0.0), np.array(y0 - eps))
    ) / (2 * eps)
    assert slope == pytest.approx(np.pi * 0.5 / 2.0, rel=1e-6)


def test_overlapping_ditches_take_the_deepest_and_corners_are_continuous():
    deep = {"points_m": [[0.0, -10.0], [0.0, 10.0]], "width_m": 2.0, "depth_m": 0.8}
    terrain = flat_with(DITCH, deep)
    assert terrain.height(np.array(0.0), np.array(5.0)) == pytest.approx(-0.8)
    bend = flat_with({**DITCH, "points_m": [[-10, 0], [0, 0], [0, 10]]})
    assert bend.height(np.array(0.0), np.array(0.0)) == pytest.approx(-0.5)
    assert bend.height(np.array(0.3), np.array(-0.3)) > -0.5


def test_hills_without_ditches_are_unchanged():
    settings = {"seed": 7, "hills": 14, "max_height_m": 3.0}
    x, y = np.meshgrid(np.linspace(-20, 20, 30), np.linspace(-20, 20, 30))
    a = Terrain(settings).height(x, y)
    b = Terrain({**settings, "ditches": []}).height(x, y)
    assert np.array_equal(a, b)


@pytest.mark.parametrize(
    "ditch, message",
    [
        ({"points_m": [[0, 0], [1, 0]], "width_m": 1.0}, "exactly"),
        ({**DITCH, "depth_m": 0.0}, "positive"),
        ({**DITCH, "width_m": True}, "number"),
        ({**DITCH, "points_m": [[0, 0]]}, "two"),
        ({**DITCH, "points_m": [[0, 0], [0, 0]]}, "repeated"),
        ({**DITCH, "points_m": [[0, 0], [29.5, 0]]}, "inside"),
        ({**DITCH, "points_m": [["a", 0], [1, 0]]}, "numbers"),
    ],
)
def test_invalid_ditches(ditch, message):
    with pytest.raises(ValueError, match=message):
        flat_with(ditch)


def test_ground_becomes_a_frame_around_terrain_with_ditches(tmp_path):
    scene = {
        "ground_size_m": 80.0,
        "obstacles": [],
        "terrain": {"size_m": [60.0, 40.0], "hills": 0, "ditches": [DITCH]},
    }
    root = ET.parse(build_world(scene, tmp_path / "w.sdf"))
    boxes = {
        c.get("name"): (
            [float(v) for v in c.findtext("pose").split()[:2]],
            [float(v) for v in c.findtext("geometry/box/size").split()[:2]],
        )
        for c in root.findall('./world/model[@name="ground"]/link/collision')
    }
    assert set(boxes) == {"ground_north", "ground_south", "ground_east", "ground_west"}
    assert boxes["ground_north"] == ([0.0, 30.0], [80.0, 20.0])
    assert boxes["ground_east"] == ([35.0, 0.0], [10.0, 40.0])
    # Without ditches the ground stays one box (worlds unchanged).
    del scene["terrain"]["ditches"]
    root = ET.parse(build_world(scene, tmp_path / "w2.sdf"))
    names = [
        c.get("name")
        for c in root.findall('./world/model[@name="ground"]/link/collision')
    ]
    assert names == ["ground_collision"]
