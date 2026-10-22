import xml.etree.ElementTree as ET

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
    assert len(surfaces.findall("visual")) == 2 * 2 + 1  # 2 segments + discs, 1 box
    assert not surfaces.findall("collision")


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
