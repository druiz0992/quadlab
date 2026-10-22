"""General rotor layouts: hexacopter allocation, tilted (pusher) rotors."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from robotlab.control.manual import FlightCommand, ManualController
from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.simulation.gazebo.world import build_world


def test_hexa_hovers_with_equal_shares_and_yaw_authority():
    config = load_drone_config("configs/vehicles/hexa_x.json")
    p = config.properties
    assert len(p.rotors) == 6 and p.lift_rotors == tuple(range(6))
    controller = ManualController(p, config.data["controller"])
    level = np.array([0, 0, 0, 1.0])
    controller.update_position([0, 0, 1], np.zeros(3), level, FlightCommand(), 0.02)
    thrust = controller.update_attitude(level, np.zeros(3))
    np.testing.assert_allclose(thrust, p.mass * 9.81 / 6, rtol=1e-6)
    # A yaw-rate demand changes ccw and cw rotors in opposite directions.
    controller.update_position(
        [0, 0, 1], np.zeros(3), level, FlightCommand(yaw_rate_rps=0.5), 0.02
    )
    turning = controller.update_attitude(level, np.zeros(3))
    ccw, cw = turning[0::2] - thrust[0::2], turning[1::2] - thrust[1::2]
    assert np.all(np.sign(ccw) == -np.sign(cw)) and np.abs(ccw).max() > 0


def quad_with_pusher(tmp_path):
    layer = dict(
        vehicle=dict(
            rotors=[
                dict(position_m=[0.2, 0.2, 0.08], direction="ccw", axis=[0, 0, 1]),
                dict(position_m=[-0.2, -0.2, 0.08], direction="ccw", axis=[0, 0, 1]),
                dict(position_m=[0.2, -0.2, 0.08], direction="cw", axis=[0, 0, 1]),
                dict(position_m=[-0.2, 0.2, 0.08], direction="cw", axis=[0, 0, 1]),
                dict(position_m=[-0.3, 0.0, 0.0], direction="ccw", axis=[1, 0, 0]),
            ]
        )
    )
    path = tmp_path / "pusher.json"
    path.write_text(json.dumps(layer))
    return load_drone_config(path)


def test_pusher_gets_no_hover_thrust_and_is_pitched_in_sdf(tmp_path):
    config = quad_with_pusher(tmp_path)
    p = config.properties
    assert p.lift_rotors == (0, 1, 2, 3) and p.axes[4] == (1.0, 0.0, 0.0)
    controller = ManualController(p, config.data["controller"])
    level = np.array([0, 0, 0, 1.0])
    controller.update_position([0, 0, 1], np.zeros(3), level, FlightCommand(), 0.02)
    thrust = controller.update_attitude(level, np.zeros(3))
    assert thrust.shape == (5,) and thrust[4] == 0
    np.testing.assert_allclose(thrust[:4].sum(), p.mass * 9.81, rtol=1e-6)
    world = build_world(
        {"obstacles": []},
        tmp_path / "w.sdf",
        config=config,
        control_mode="rotor-thrust",
    )
    pose = ET.parse(world).getroot().find(".//link[@name='rotor_4']/pose").text.split()
    np.testing.assert_allclose(
        [float(v) for v in pose], [-0.3, 0, 0, 0, np.pi / 2, 0], atol=1e-9
    )
    with pytest.raises(ValueError, match="vertical"):
        build_world(
            {"obstacles": []},
            tmp_path / "v.sdf",
            config=config,
            control_mode="velocity",
        )


def test_layout_validation(tmp_path):
    def load(layer):
        path = tmp_path / "x.json"
        path.write_text(json.dumps(layer))
        return load_drone_config(path)

    three = [dict(position_m=[0.2, 0, 0.08], direction="ccw", axis=[0, 0, 1])] * 3
    with pytest.raises(ValueError, match="four vertical"):
        load({"vehicle": {"rotors": three}})
    with pytest.raises(ValueError, match="unit vector"):
        load({"vehicle": {"rotors": [dict(three[0], axis=[0, 0, 2])] * 4}})
    with pytest.raises(ValueError, match="one value per rotor"):
        load({"motors": {"rotor_thrust_scale": [1, 1, 1]}})
