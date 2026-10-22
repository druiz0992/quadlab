import xml.etree.ElementTree as ET

import numpy as np

import pytest

from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.simulation.gazebo.world import RESERVED_NAMES, build_world, load_scene


@pytest.mark.parametrize("kind", ["rover", "car", "tractor"])
def test_wheeled_world(tmp_path, kind):
    config = load_drone_config(robot=kind)
    scene = {
        "obstacles": [],
        "surfaces": [
            {
                "shape": "strip",
                "name": "asphalt",
                "points_m": [[0, 0], [10, 0], [10, 10]],
                "width_m": 4,
            },
            {"shape": "rectangle", "name": "mud", "center_m": [5, 5], "size_m": [2, 2]},
        ],
        "spawn_points": [{"position_m": [1, 0], "yaw_deg": 0}],
    }
    root = ET.parse(build_world(scene, tmp_path / "world.sdf", config=config))
    model = root.find(f'./world/model[@name="{kind}"]')
    assert model is not None and root.find('./world/model[@name="drone"]') is None
    link = model.find("link")
    collisions = [c.get("name") for c in link.findall("collision")]
    assert collisions[0] == "body"
    assert sorted(collisions[1:]) == sorted(
        f"wheel_{w}" for w in config.wheeled.wheel_names
    )
    assert not model.findall("joint")
    assert float(link.findtext("inertial/mass")) == config.wheeled.mass_kg
    assert link.findtext("sensor/contact/collision") == "body"
    plugins = {p.get("name") for p in root.findall("./world/plugin")}
    assert "gz::sim::systems::ApplyLinkWrench" in plugins
    x = float(model.findtext("pose").split()[0])
    assert x == 1.0
    surfaces = root.find('./world/model[@name="surfaces"]/link')
    assert len(surfaces.findall("visual")) == 2  # one mesh per zone
    assert not surfaces.findall("collision")
    assert (tmp_path / "surface_0.obj").exists()


def test_vehicle_names_are_reserved(tmp_path):
    assert {"car", "rover", "tractor", "drone"} <= RESERVED_NAMES
    path = tmp_path / "scene.json"
    path.write_text(
        '{"obstacles": [{"name": "car", "position": [0,0,1], "size": [1,1,1]}]}'
    )
    with pytest.raises(ValueError, match="obstacle name"):
        load_scene(path)


def test_scene_surface_validation(tmp_path):
    path = tmp_path / "scene.json"
    path.write_text('{"obstacles": [], "surfaces": [{"shape": "blob"}]}')
    with pytest.raises(ValueError, match="shape"):
        load_scene(path)


def test_opponents_get_models_on_the_next_spawn_points(tmp_path):
    layer = tmp_path / "layer.json"
    layer.write_text('{"ground_vehicle": {"opponents": {"count": 2}}}')
    config = load_drone_config(layer, robot="car")
    scene = {
        "obstacles": [],
        "spawn_points": [
            {"position_m": [0, 0]},
            {"position_m": [-8, 3]},
            {"position_m": [-16, -3]},
        ],
    }
    root = ET.parse(build_world(scene, tmp_path / "world.sdf", config=config))
    for name, x in (("car_1", -8.0), ("car_2", -16.0)):
        model = root.find(f'./world/model[@name="{name}"]')
        assert float(model.findtext("pose").split()[0]) == x
        assert not model.findall("link/sensor")
        topic = model.find("plugin/odom_topic").text
        assert topic == f"/robotlab/opponents/{name}/odometry"
    scene["spawn_points"].pop()
    with pytest.raises(ValueError, match="spawn points"):
        build_world(scene, tmp_path / "w2.sdf", config=config)
    scene = {
        "obstacles": [{"name": "car_1", "position": [50, 0, 1], "size": [1, 1, 1]}],
        "spawn_points": scene["spawn_points"] + [{"position_m": [-16, -3]}],
    }
    with pytest.raises(ValueError, match="opponent"):
        build_world(scene, tmp_path / "w3.sdf", config=config)


def test_zone_mesh_ribbon():
    from robotlab.simulation.gazebo.world import zone_mesh
    from robotlab.simulation.wheeled.ground import parse_zone

    zone = parse_zone(
        {"shape": "strip", "points_m": [[0, 0], [10, 0], [10, 10]], "width_m": 2}, 0
    )
    vertices, faces = zone_mesh(zone)
    assert vertices.shape == (6, 2) and faces.shape == (4, 3)
    np.testing.assert_allclose(vertices[0], [0, 1])
    np.testing.assert_allclose(vertices[1], [9, 1])  # mitred corner


def test_tractor_implement_and_trailer_models(tmp_path):
    implement = tmp_path / "implement.json"
    implement.write_text('{"ground_vehicle": {"implement": {"enabled": true}}}')
    root = ET.parse(
        build_world(
            {"obstacles": []},
            tmp_path / "a.sdf",
            config=load_drone_config(implement, robot="tractor"),
        )
    )
    tractor = root.find('./world/model[@name="tractor"]')
    assert tractor.find('joint[@name="implement_lift"]').get("type") == "prismatic"
    assert tractor.find('link[@name="implement"]/collision') is None
    servo = [
        p
        for p in tractor.findall("plugin")
        if p.get("name").endswith("JointPositionController")
    ]
    assert servo and servo[0].findtext("topic") == "/robotlab/implement/lift"
    trailer = tmp_path / "trailer.json"
    trailer.write_text('{"ground_vehicle": {"trailer": {"enabled": true}}}')
    root = ET.parse(
        build_world(
            {"obstacles": []},
            tmp_path / "b.sdf",
            config=load_drone_config(trailer, robot="tractor"),
        )
    )
    tractor = root.find('./world/model[@name="tractor"]')
    assert {j.get("name") for j in tractor.findall("joint")} == {
        "trailer_yaw",
        "trailer_pitch",
    }
    states = [
        p
        for p in tractor.findall("plugin")
        if p.get("name").endswith("JointStatePublisher")
    ]
    assert states[0].findtext("topic") == "/robotlab/trailer/joint_state"
    assert (
        tractor.find('link[@name="trailer"]/collision[@name="trailer_body"]')
        is not None
    )
    both = tmp_path / "both.json"
    both.write_text(
        '{"ground_vehicle": {"implement": {"enabled": true}, "trailer": {"enabled": true}}}'
    )
    with pytest.raises(ValueError, match="one rear hitch"):
        load_drone_config(both, robot="tractor")


@pytest.mark.parametrize(
    "scene, robot, configs",
    [
        ("worlds/race_circuit.json", "car", ["configs/scenarios/race_opponents.json"]),
        ("worlds/farm_field.json", "tractor", ["configs/scenarios/field_plough.json"]),
        (
            "worlds/farm_field.json",
            "tractor",
            ["configs/scenarios/tractor_trailer.json"],
        ),
        ("worlds/rover_terrain.json", "rover", []),
    ],
)
def test_ground_vehicle_worlds_generate(tmp_path, scene, robot, configs):
    config = load_drone_config(configs, robot=robot)
    root = ET.parse(
        build_world(load_scene(scene), tmp_path / "world.sdf", config=config)
    )
    assert root.find(f'./world/model[@name="{robot}"]') is not None
