import numpy as np
import pytest

from robotlab.simulation.terrain import Terrain
from robotlab.simulation.wheeled.ground import (
    GroundModel,
    Surface,
    parse_zone,
    strip_distance,
    surfaces_from_scene,
)

DOWN = np.array([[0.0, 0.0, -1.0]])


def test_flat_ground_and_area_edge():
    ground = GroundModel(None, [], ground_half_m=10.0)
    hits = ground.cast(np.array([[1.0, 2.0, 0.5]]), DOWN, np.array([1.0]))
    assert hits.hit[0] and hits.distance_m[0] == pytest.approx(0.5)
    np.testing.assert_allclose(hits.normal[0], [0, 0, 1])
    outside = ground.cast(np.array([[11.0, 0.0, 0.5]]), DOWN, np.array([1.0]))
    assert not outside.hit[0] and np.isnan(outside.distance_m[0])


def test_ray_longer_than_limit_misses():
    ground = GroundModel(None, [], ground_half_m=10.0)
    assert not ground.cast(np.array([[0.0, 0.0, 2.0]]), DOWN, np.array([1.0])).hit[0]


def test_terrain_matches_its_own_ray_cast():
    terrain = Terrain(dict(seed=3, hills=6, max_height_m=2.0, flat_radius_m=0.0))
    ground = GroundModel(terrain, [], ground_half_m=50.0)
    origins = np.array([[x, -x / 2, 5.0] for x in (3.0, 7.0, -9.0)])
    hits = ground.cast(origins, np.tile(DOWN, (3, 1)), np.full(3, 10.0))
    for origin, distance in zip(origins, hits.distance_m):
        assert distance == pytest.approx(
            terrain.ray_distance(origin, DOWN[0], 10.0), abs=1e-6
        )


def test_box_top_only_when_ray_starts_above_it():
    box = dict(position=[0.0, 0.0, 0.5], size=[2.0, 2.0, 1.0])
    ground = GroundModel(None, [box], ground_half_m=10.0)
    above = ground.cast(np.array([[0.0, 0.0, 1.3]]), DOWN, np.array([2.0]))
    assert above.distance_m[0] == pytest.approx(0.3)
    inside = ground.cast(np.array([[0.0, 0.0, 0.8]]), DOWN, np.array([2.0]))
    assert inside.distance_m[0] == pytest.approx(0.8)  # strut below the top: floor


def test_surface_zones_later_override_earlier():
    scene = dict(
        obstacles=[],
        ground_surface=dict(name="grass", friction_scale=0.6),
        surfaces=[
            dict(
                shape="rectangle",
                name="gravel",
                center_m=[0, 0],
                size_m=[10, 10],
                friction_scale=0.7,
            ),
            dict(
                shape="strip",
                name="asphalt",
                points_m=[[-5, 0], [5, 0]],
                width_m=2,
                friction_scale=1.0,
            ),
        ],
    )
    ground = GroundModel.from_scene(scene, None)
    points = np.array([[0.0, 0.0, 1.0], [0.0, 3.0, 1.0], [20.0, 0.0, 1.0]])
    hits = ground.cast(points, np.tile(DOWN, (3, 1)), np.full(3, 2.0))
    np.testing.assert_allclose(hits.friction_scale, [1.0, 0.7, 0.6])
    assert ground.surface_name(0.0, 0.0) == "asphalt"


def test_rotated_rectangle():
    zone = parse_zone(
        dict(
            shape="rectangle", center_m=[0, 0], size_m=[4, 1], yaw_deg=90, name="kerb"
        ),
        0,
    )
    assert zone.contains(np.array([0.0]), np.array([1.9]))[0]
    assert not zone.contains(np.array([1.9]), np.array([0.0]))[0]


def test_strip_distance_closed_loop():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert strip_distance(square, True, np.array([-1.0]), np.array([5.0]))[
        0
    ] == pytest.approx(1.0)
    assert strip_distance(square, False, np.array([-1.0]), np.array([5.0]))[
        0
    ] == pytest.approx(np.sqrt(26))
    assert strip_distance(square, False, np.array([5.0]), np.array([11.0]))[
        0
    ] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "zone, message",
    [
        (dict(shape="circle"), "shape"),
        (dict(shape="rectangle", center_m=[0, 0], size_m=[0, 1]), "positive"),
        (dict(shape="strip", points_m=[[0, 0]], width_m=1), "two points"),
        (
            dict(shape="strip", points_m=[[0, 0], [1, 1]], width_m=1, colour="red"),
            "unknown",
        ),
        (
            dict(shape="rectangle", center_m=[0, 0], size_m=[1, 1], friction_scale=0),
            "positive",
        ),
    ],
)
def test_invalid_zones(zone, message):
    with pytest.raises(ValueError, match=message):
        surfaces_from_scene(dict(obstacles=[], surfaces=[zone]))


def test_default_surface():
    default, zones = surfaces_from_scene(dict(obstacles=[]))
    assert default == Surface("ground") and zones == ()


def test_raster_matches_exact_zones_away_from_edges():
    from robotlab.simulation.wheeled.ground import RASTER_RESOLUTION_M, SurfaceRaster

    zones = [
        parse_zone(
            dict(
                shape="rectangle",
                center_m=[0, 0],
                size_m=[20, 10],
                yaw_deg=30,
                name="a",
            ),
            0,
        ),
        parse_zone(
            dict(
                shape="strip",
                points_m=[[-15, -15], [0, 5], [15, -15]],
                width_m=4,
                name="b",
            ),
            1,
        ),
    ]
    raster = SurfaceRaster(Surface("ground"), zones)
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-25, 25, (2, 4000))
    exact = np.zeros(4000, int)
    for index, zone in enumerate(zones, start=1):
        exact = np.where(zone.contains(x, y), index, exact)
    looked_up = raster.lookup(x, y)
    # Disagreements only within a cell of a boundary.
    near = np.zeros(4000, bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            shifted = np.zeros(4000, int)
            for index, zone in enumerate(zones, start=1):
                inside = zone.contains(
                    x + dx * RASTER_RESOLUTION_M, y + dy * RASTER_RESOLUTION_M
                )
                shifted = np.where(inside, index, shifted)
            near |= shifted != exact
    assert np.all((looked_up == exact) | near)
    assert np.mean(looked_up == exact) > 0.97
