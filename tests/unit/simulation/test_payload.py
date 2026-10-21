import json
import xml.etree.ElementTree as ET
from types import ModuleType
import sys

import numpy as np
import pytest

from dronelab.simulation.gazebo.config import load_drone_config
from dronelab.simulation.gazebo.vehicle import spawn_position
from dronelab.simulation.gazebo.world import build_world


class FakeNode:
    def __init__(self):
        self.published = []

    def advertise(self, topic, kind):
        node = self

        class Publisher:
            def publish(self, message):
                node.published.append(topic)
                return True

        return Publisher()


@pytest.fixture
def control(monkeypatch):
    # Stub gz.msgs10.empty_pb2 so the adapter runs without Gazebo bindings.
    for name in ("gz", "gz.msgs10"):
        monkeypatch.setitem(
            sys.modules, name, sys.modules.get(name) or ModuleType(name)
        )
    empty = ModuleType("gz.msgs10.empty_pb2")
    empty.Empty = object
    monkeypatch.setitem(sys.modules, "gz.msgs10.empty_pb2", empty)
    from dronelab.simulation.gazebo.payload import PayloadControl

    settings = dict(load_drone_config().data["payload"], enabled=True, drop_s=2.0)
    node = FakeNode()
    return PayloadControl(node, settings), node


def test_scheduled_drop_once(control):
    payload, node = control
    assert payload.update(1_000_000_000) is None
    assert payload.update(2_000_000_000)["action"] == "drop"
    assert payload.update(3_000_000_000) is None
    assert node.published == ["/dronelab/payload/detach"] and not payload.attached


def test_pickup_only_within_reach_of_hook(control):
    payload, node = control
    payload.drop(0)
    level = [0, 0, 0, 1]
    payload.on_pose("payload", [0.0, 0.0, 0.04])
    far = payload.pickup(1, [0.0, 0.0, 1.0], level)  # hook at 0.88 m: 0.84 m away
    assert not far["done"] and far["reason"] == "out of reach"
    near = payload.pickup(2, [0.1, 0.0, 0.3], level)  # hook at (0.1, 0, 0.18)
    assert near["done"] and near["distance_m"] == pytest.approx(np.hypot(0.1, 0.14))
    assert node.published[-1] == "/dronelab/payload/attach" and payload.attached


def test_payload_sdf_joint_and_ground_start(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(
        json.dumps(
            {"payload": {"enabled": True}, "simulation": {"start_on_ground": True}}
        )
    )
    config = load_drone_config(path)
    root = ET.parse(
        build_world({"obstacles": []}, tmp_path / "w.sdf", config=config)
    ).getroot()
    joint = root.find(
        ".//model[@name='drone']/plugin[@name='gz::sim::systems::DetachableJoint']"
    )
    assert joint.findtext("child_model") == "payload"
    pose = list(map(float, root.findtext(".//model[@name='payload']/pose").split()))
    # Drone stands on the payload: payload bottom 2 mm above the ground.
    assert pose[2] - 0.04 == pytest.approx(0.002)
    assert spawn_position(config.data)[2] == pytest.approx(0.002 + 0.04 + 0.12)


@pytest.mark.parametrize(
    "payload",
    [
        {"enabled": True, "offset_m": [0, 0, -0.05]},  # overlaps the body
        {"enabled": True, "mass_kg": 5.0},  # cannot hover
        {"enabled": True, "pickup_radius_m": 0},
        {"enabled": True, "size_m": [0.1, 0, 0.1]},
    ],
)
def test_invalid_payload(tmp_path, payload):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"payload": payload}))
    with pytest.raises(ValueError):
        load_drone_config(path)


@pytest.fixture
def winch(monkeypatch):
    for name in ("gz", "gz.msgs10"):
        monkeypatch.setitem(
            sys.modules, name, sys.modules.get(name) or ModuleType(name)
        )
    double = ModuleType("gz.msgs10.double_pb2")

    class Double:
        data = 0.0

    double.Double = Double
    monkeypatch.setitem(sys.modules, "gz.msgs10.double_pb2", double)
    from dronelab.simulation.gazebo.payload import Winch

    tether = dict(
        load_drone_config().data["payload"]["tether"],
        enabled=True,
        length_schedule=[{"time_s": 1.0, "length_m": 2.0}],
    )
    return Winch(FakeNode(), tether)


def test_winch_follows_schedule_at_speed_and_keys(winch):
    ms = 1_000_000
    assert winch.update(0) == 1.0
    assert winch.update(500 * ms) == 1.0  # before the scheduled change
    assert winch.update(1500 * ms) == pytest.approx(1.5)  # 0.5 m/s for 1 s
    assert winch.update(5000 * ms) == pytest.approx(2.0)  # reached and held
    assert winch.update(6000 * ms, key_rate=-1) == pytest.approx(1.5)  # S raises
    assert winch.update(30000 * ms, key_rate=+1) == pytest.approx(3.0)  # W, max length


def test_tether_sdf_joints_and_payload_under_hook(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps({"payload": {"enabled": True, "tether": {"enabled": True}}})
    )
    config = load_drone_config(path)
    root = ET.parse(
        build_world({"obstacles": []}, tmp_path / "w.sdf", config=config)
    ).getroot()
    drone = root.find(".//model[@name='drone']")
    assert drone.find("./joint[@name='tether_ball']").get("type") == "ball"
    winch = drone.find("./joint[@name='winch']")
    assert (
        winch.get("type") == "prismatic"
        and winch.findtext(".//limit/effort") == "1000000.0"
    )
    joint = drone.find("./plugin[@name='gz::sim::systems::DetachableJoint']")
    assert joint.findtext("parent_link") == "hook"
    payload_z = float(root.findtext(".//model[@name='payload']/pose").split()[2])
    assert payload_z == pytest.approx(1.0 - 0.07 - 1.0 - 0.03 - 0.04)


@pytest.mark.parametrize(
    "tether",
    [
        {"enabled": True, "initial_length_m": 4.0},  # above max length
        {"enabled": True, "anchor_m": [0, 0, 0]},  # inside the body
        {
            "enabled": True,
            "length_schedule": [
                {"time_s": 2, "length_m": 1},
                {"time_s": 1, "length_m": 1},
            ],
        },
        {"enabled": True, "length_schedule": [{"time_s": 1, "length_m": 0.05}]},
    ],
)
def test_invalid_tether(tmp_path, tether):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"payload": {"enabled": True, "tether": tether}}))
    with pytest.raises(ValueError):
        load_drone_config(path)


def test_tether_rejects_ground_start(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            {
                "payload": {"enabled": True, "tether": {"enabled": True}},
                "simulation": {"start_on_ground": True},
            }
        )
    )
    with pytest.raises(ValueError, match="airborne"):
        load_drone_config(path)
