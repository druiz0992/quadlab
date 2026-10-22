import json
import xml.etree.ElementTree as ET

import pytest

from robotlab.simulation.gazebo.world import build_world, load_scene


def test_world_geometry_and_sensors(tmp_path):
    scene = {
        "obstacles": [{"name": "wall", "position": [2, 0, 1], "size": [0.3, 4, 2]}]
    }
    path = build_world(scene, tmp_path / "world.sdf", render_sensors=True)
    root = ET.parse(path)
    obstacle = root.find('./world/model[@name="wall"]/link')
    assert obstacle.findtext("./collision/geometry/box/size") == obstacle.findtext(
        "./visual/geometry/box/size"
    )
    drone = root.find('./world/model[@name="drone"]')
    assert len(drone.findall("joint")) == 4
    assert {s.get("type") for s in drone.findall("./link/sensor")} == {
        "imu",
        "navsat",
        "contact",
        "camera",
        "gpu_lidar",
    }
    assert drone.findtext("./link/sensor/contact/topic") == "/robotlab/contacts"
    assert len(drone.findall("./plugin/rotorConfiguration/rotor")) == 4
    assert not root.findall(".//uri")  # No remote assets needed.


def test_invalid_obstacle(tmp_path):
    path = tmp_path / "scene.json"
    path.write_text(
        json.dumps(
            {"obstacles": [{"name": "wall", "position": [0, 0, 1], "size": [-1, 2, 2]}]}
        )
    )
    with pytest.raises(ValueError, match="positive"):
        load_scene(path)


def test_thrust_world_has_no_competing_velocity_controller(tmp_path):
    root = ET.parse(
        build_world(
            {"obstacles": []}, tmp_path / "thrust.sdf", control_mode="rotor-thrust"
        )
    )
    plugins = root.findall('./world/model[@name="drone"]/plugin')
    assert not any("VelocityControl" in p.get("name") for p in plugins)
    motors = [p for p in plugins if "MotorModel" in p.get("name")]
    assert len(motors) == 4
    assert [p.findtext("actuator_number") for p in motors] == ["0", "1", "2", "3"]
    assert all(p.findtext("commandSubTopic") == "command/motor_speed" for p in motors)


def test_scene_reflectivity_becomes_laser_retro(tmp_path):
    import json
    import xml.etree.ElementTree as ET

    scene = dict(
        ground_reflectivity=0.2,
        obstacles=[
            dict(name="sign", position=[5, 0, 1], size=[0.1, 1, 1], reflectivity=3.0),
            dict(name="plain", position=[5, 3, 1], size=[1, 1, 1]),
        ],
    )
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene))
    world = ET.parse(build_world(load_scene(path), tmp_path / "world.sdf")).getroot()
    retro = {
        m.get("name"): m.findtext(".//visual/laser_retro") for m in world.iter("model")
    }
    assert float(retro["sign"]) == 300.0 and float(retro["ground"]) == 20.0
    assert retro["plain"] is None
    scene["obstacles"][1]["reflectivity"] = 0
    path.write_text(json.dumps(scene))
    with pytest.raises(ValueError, match="reflectivity"):
        load_scene(path)


def test_physics_engine_settings_are_explicit(tmp_path):
    root = ET.parse(build_world({"obstacles": []}, tmp_path / "world.sdf"))
    physics = root.find("./world/physics")
    assert physics.get("type") == "dart"
    assert physics.findtext("./dart/collision_detector") == "ode"
    assert physics.findtext("./dart/solver/solver_type") == "dantzig"
    engine = root.find(
        './world/plugin[@name="gz::sim::systems::Physics"]/engine/filename'
    )
    assert engine.text == "gz-physics-dartsim-plugin"
    assert root.findtext("./world/gravity") == "0 0 -9.81"
