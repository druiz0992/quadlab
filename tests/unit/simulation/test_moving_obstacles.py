import json

import numpy as np
import pytest

from dronelab.simulation.gazebo.world import build_world, load_scene
from dronelab.simulation.moving_obstacles import PatrolPath


def test_pingpong_goes_back_and_forth_at_constant_speed():
    path = PatrolPath([[0, 0, 1], [4, 0, 1]], speed_mps=2.0, mode="pingpong")
    np.testing.assert_allclose(path.state(0.0)[0], [0, 0, 1])
    np.testing.assert_allclose(path.state(1.0)[0], [2, 0, 1])
    position, velocity = path.state(3.0)  # reached the end at 2 s, coming back
    np.testing.assert_allclose(position, [2, 0, 1])
    np.testing.assert_allclose(velocity, [-2, 0, 0])
    np.testing.assert_allclose(path.state(4.0)[0], [0, 0, 1])


def test_loop_returns_to_start_through_3d_waypoints():
    path = PatrolPath([[0, 0, 1], [3, 0, 1], [3, 0, 5]], speed_mps=1.0, mode="loop")
    total = 3 + 4 + 5  # closing leg from (3, 0, 5) back to (0, 0, 1) is 5 m
    np.testing.assert_allclose(path.state(5.0)[0], [3, 0, 3])  # climbing leg
    np.testing.assert_allclose(path.state(5.0)[1], [0, 0, 1])
    np.testing.assert_allclose(path.state(total)[0], [0, 0, 1], atol=1e-12)


def test_example_scene_builds_flying_and_ground_obstacles(tmp_path):
    import xml.etree.ElementTree as ET

    scene = load_scene("worlds/moving.json")
    root = ET.parse(build_world(scene, tmp_path / "w.sdf")).getroot()
    flyer = root.find(".//model[@name='flyer']")
    assert flyer.findtext("pose").startswith("-4 -4 2.0")
    assert flyer.findtext("./link/gravity") == "false"
    assert flyer.find("./plugin[@name='gz::sim::systems::VelocityControl']") is not None


@pytest.mark.parametrize(
    "entry",
    [
        {
            "name": "a",
            "size": [1, 1, 1],
            "waypoints": [[0, 0, 1]],
            "speed_mps": 1,
            "mode": "loop",
        },
        {
            "name": "a",
            "size": [1, 1, 1],
            "waypoints": [[0, 0, 1], [0, 0, 1]],
            "speed_mps": 1,
            "mode": "loop",
        },
        {
            "name": "a",
            "size": [1, 1, 1],
            "waypoints": [[0, 0, 1], [1, 0, 1]],
            "speed_mps": 0,
            "mode": "loop",
        },
        {
            "name": "a",
            "size": [1, 1, 1],
            "waypoints": [[0, 0, 1], [1, 0, 1]],
            "speed_mps": 1,
            "mode": "spin",
        },
        {
            "name": "drone",
            "size": [1, 1, 1],
            "waypoints": [[0, 0, 1], [1, 0, 1]],
            "speed_mps": 1,
            "mode": "loop",
        },
    ],
)
def test_invalid_moving_obstacles_rejected(tmp_path, entry):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"obstacles": [], "moving_obstacles": [entry]}))
    with pytest.raises(ValueError):
        load_scene(path)


def test_terrain_scene_spawn_and_mesh(tmp_path):
    from dronelab.simulation.gazebo.config import load_drone_config
    from dronelab.simulation.gazebo.vehicle import spawn_position
    from dronelab.simulation.gazebo.world import scene_terrain

    scene = load_scene("worlds/hills.json")
    terrain = scene_terrain(scene)
    assert float(terrain.height(0, 0)) == 0.0  # flat spawn disc
    settings = load_drone_config().data
    settings["simulation"].update(
        start_on_ground=True, spawn_position_m=[8.0, 0.0, 0.0]
    )
    assert spawn_position(settings, terrain)[2] == pytest.approx(
        float(terrain.height(8, 0)) + 0.062
    )
    build_world(scene, tmp_path / "w.sdf")
    text = (tmp_path / "terrain.obj").read_text()
    assert text.count("\nvn ") > 0 and "//" in text  # DART needs vertex normals
    # Ray straight down from 3 m above a point hits the analytic surface.
    distance = terrain.ray_distance([10.0, 0.0, 3.0], [0, 0, -1], 10.0)
    assert distance == pytest.approx(3.0 - float(terrain.height(10, 0)), abs=1e-6)
