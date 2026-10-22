import math

import pytest

from robotlab.control.manual import FlightCommand, ManualController
from robotlab.experiments.failsafe import BatteryFailsafe
from robotlab.simulation.gazebo.config import load_drone_config


def failsafe(**overrides):
    settings = dict(load_drone_config().data["failsafe"]["low_battery"], enabled=True)
    settings.update(overrides)
    return BatteryFailsafe(settings, home_xy=[0.0, 0.0], cells=4)


def step(f, t, pct, *, volts=15.0, pos=(0, 0, 1), yaw=0.0, vz=0.0, height=1.0):
    return f.update(t, pct, volts, pos, yaw, vz, height)


def test_warn_then_land_then_touchdown_after_hold():
    f = failsafe()
    assert step(f, 0, 50) is None and f.state == "normal"
    assert step(f, 1, 30) is None and f.state == "warned"
    command = step(f, 2, 15)
    assert f.state == "landing" and command == FlightCommand(climb_mps=-0.5)
    step(f, 3, 14, height=0.03, vz=-0.02)  # near ground and still: start hold
    assert f.state == "landing"
    step(f, 3.6, 14, height=0.03, vz=-0.01)
    assert f.state == "landed"
    assert [e["state"] for e in f.events] == ["warned", "landing", "landed"]


def test_critical_voltage_lands_immediately_even_when_returning_home():
    f = failsafe(action="return_and_land")
    step(f, 0, 80, volts=3.2 * 4, pos=(10, 0, 2))
    assert f.state == "landing"  # 3.2 V/cell < 3.3: no time to fly home


def test_return_home_turns_then_flies_then_lands():
    f = failsafe(action="return_and_land")
    command = step(f, 0, 10, pos=(10, 0, 2), yaw=0.0)  # home is behind (bearing pi)
    assert (
        f.state == "returning"
        and command.forward_mps == 0
        and abs(command.yaw_rate_rps) == 1.0
    )
    command = step(f, 1, 10, pos=(10, 0, 2), yaw=math.pi)
    assert command.forward_mps == pytest.approx(
        3.0
    ) and command.yaw_rate_rps == pytest.approx(0, abs=1e-9)
    step(f, 2, 10, pos=(0.5, 0, 2), yaw=math.pi)
    assert f.state == "landing"


def test_warn_only_never_overrides():
    f = failsafe(action="warn")
    assert step(f, 0, 5) is None and f.state == "warned" and not f.overriding


def test_landing_lifts_controller_altitude_floor():
    controller = ManualController()
    controller.update_position(
        [0, 0, 0.4], [0, 0, 0], [0, 0, 0, 1], FlightCommand(climb_mps=-0.5), 0.02
    )
    assert controller.target[2] == pytest.approx(0.4)  # floor holds
    controller.allow_landing()
    controller.update_position(
        [0, 0, 0.4], [0, 0, 0], [0, 0, 0, 1], FlightCommand(climb_mps=-0.5), 0.02
    )
    assert controller.target[2] < 0.4


@pytest.mark.parametrize(
    "data",
    [
        {"failsafe": {"low_battery": {"enabled": True}}},  # no battery
        {
            "battery": {"enabled": True},
            "failsafe": {"low_battery": {"enabled": True, "action": "panic"}},
        },
        {
            "battery": {"enabled": True},
            "failsafe": {"low_battery": {"enabled": True, "land_remaining_pct": 40}},
        },
    ],
)
def test_invalid_failsafe(tmp_path, data):
    import json

    path = tmp_path / "f.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_drone_config(path)
