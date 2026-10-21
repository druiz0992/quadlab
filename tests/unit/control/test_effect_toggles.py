"""Keyboard effect switches drive the models' manual overrides."""

import numpy as np
import pytest

from dronelab.core.measurements import GpsReading
from dronelab.experiments.effect_toggles import EffectToggles
from dronelab.sensors.dropout import GpsDropout
from dronelab.sensors.imu_faults import ImuFaults
from dronelab.sensors.multipath import GpsMultipath
from dronelab.simulation.gazebo.config import load_drone_config
from dronelab.simulation.gazebo.wind import WindProfile
from dronelab.simulation.powertrain import Powertrain

HOVER_N = 1.54 * 9.81 / 4
FIX = GpsReading(40.4168, -3.7038, 650.0)


class FakeWind:
    def __init__(self):
        self.calls = []

    def set_enabled(self, on):
        self.calls.append(on)


def setup(wind=None):
    data = load_drone_config().data
    powertrain = Powertrain(
        load_drone_config().properties,
        data["battery"],
        data["aerodynamics"]["ground_effect"],
        data["motors"]["faults"],
        overridable=True,
    )
    models = dict(
        powertrain=powertrain,
        wind=wind,
        gps_dropout=GpsDropout(),
        gps_multipath=GpsMultipath(data["sensors"]["gps"]["multipath"]),
        imu_faults=ImuFaults(data["sensors"]["imu"]["faults"]),
    )
    return EffectToggles(data["keyboard_effects"], **models), models


def test_overridable_powertrain_is_plain_conversion_until_toggled():
    toggles, m = setup()
    p = m["powertrain"]
    assert p.enabled
    p.update(0, [0, 0, 1], [0, 0, 0, 1])
    np.testing.assert_allclose(
        p.motor_speeds(np.full(4, HOVER_N)), np.sqrt(HOVER_N / 8.55e-6)
    )
    assert toggles.set("motor_stop", True, 5)["effect"] == "motor_stop"
    p.update(1, [0, 0, 1], [0, 0, 0, 1])
    assert p.motor_speeds(np.full(4, HOVER_N))[0] == 0
    toggles.set("damaged_prop", True, 6)
    p.update(2, [0, 0, 1], [0, 0, 0, 1])
    np.testing.assert_allclose(p.available_thrust_fraction(), [0, 0.7, 1, 1])
    toggles.set("motor_stop", False, 7)
    toggles.set("damaged_prop", False, 8)
    p.update(3, [0, 0, 1], [0, 0, 0, 1])
    np.testing.assert_allclose(p.available_thrust_fraction(), 1)


def test_gps_and_imu_switches():
    toggles, m = setup()
    toggles.set("gps_blackout", True, 0)
    assert m["gps_dropout"].drop(0) and m["gps_dropout"].summary()["manual_drops"] == 1
    toggles.set("gps_blackout", False, 1)
    assert not m["gps_dropout"].drop(1)
    toggles.set("gps_multipath", True, 2)
    multipath = m["gps_multipath"]
    assert multipath.affects_fixes and multipath.active
    moved = multipath.apply(FIX, 2)
    assert moved != FIX
    toggles.set("gps_multipath", False, 3)
    assert multipath.apply(FIX, 3) is FIX
    toggles.set("gyro_bias", True, 4)
    imu = m["imu_faults"]
    _, gyro = imu.apply([0, 0, 9.81], [0, 0, 0], 4)
    np.testing.assert_allclose(gyro, [0, 0, 0.02])
    assert imu.active_count(4) == 1


def test_wind_switch_needs_configured_wind_and_events_are_logged():
    toggles, _ = setup(wind=None)
    assert not toggles.available("wind")
    assert toggles.set("wind", False, 0) is None
    wind = FakeWind()
    toggles, _ = setup(wind=wind)
    assert toggles.state["wind"]  # configured wind starts on
    assert toggles.set("wind", False, 10) == dict(
        sim_time_ns=10, effect="wind", on=False
    )
    assert toggles.set("wind", False, 11) is None  # unchanged
    assert wind.calls == [False]
    assert toggles.summary()["events"] == [
        dict(sim_time_ns=10, effect="wind", on=False)
    ]


def test_wind_profile_off_publishes_still_air_once_then_resumes():
    from dronelab.simulation.gazebo.wind import WindCommands

    commands = WindCommands.__new__(WindCommands)
    settings = load_drone_config().data["wind"]
    settings.update(enabled=True, velocity_enu_mps=[3, 0, 0])
    commands._profile = WindProfile(settings)
    commands._on = True
    commands._zero_sent = False
    published = []
    commands._message_type = lambda: type(
        "M", (), {"linear_velocity": type("V", (), {})(), "enable_wind": False}
    )()
    commands._publisher = type(
        "P", (), {"publish": lambda self, m: published.append(m) or True}
    )()
    assert commands.update(0)["velocity_enu_mps"] == [3.0, 0.0, 0.0]
    commands.set_enabled(False)
    assert commands.update(1_000_000_000)["velocity_enu_mps"] == [0.0, 0.0, 0.0]
    assert commands.update(2_000_000_000) is None  # still air sent once
    commands.set_enabled(True)
    assert commands.update(3_000_000_000)["velocity_enu_mps"] == [3.0, 0.0, 0.0]


def test_flying_object_spawns_circle_and_removes_with_fresh_names():
    from dronelab.experiments.flying_object import FlyingObjectSpawner, circle_entry

    calls = []

    class Recorder:
        def add_obstacle(self, name, size):
            calls.append(("show", name))

        def remove_obstacle(self, name):
            calls.append(("hide", name))

    class Moving:
        def add(self, entry, start_time_s):
            calls.append(("add", entry["name"], start_time_s))

        def remove(self, name):
            calls.append(("remove", name))

    class Entities:
        def spawn(self, sdf):
            calls.append(("spawn", "flying_object" in sdf))

        def remove(self, name):
            calls.append(("delete", name))

    from dronelab.simulation.gazebo.world import moving_obstacle_sdf

    settings = load_drone_config().data["keyboard_effects"]
    spawner = FlyingObjectSpawner(
        settings,
        entities=Entities(),
        moving=Moving(),
        recorder=Recorder(),
        sdf_text=moving_obstacle_sdf,
        drone_state=lambda: (np.array([1.0, 2.0, 1.5]), 4.0),
    )
    spawner.set(True)
    spawner.set(True)  # already active: no second object
    spawner.set(False)
    spawner.set(True)
    assert calls[:3] == [
        ("spawn", True),
        ("add", "flying_object_1", 4.0),
        ("show", "flying_object_1"),
    ]
    assert ("remove", "flying_object_1") in calls and (
        "delete",
        "flying_object_1",
    ) in calls
    assert calls[-2] == ("add", "flying_object_2", 4.0)
    entry = circle_entry("x", np.array([1.0, 2.0, 1.5]), settings)
    points = np.array(entry["waypoints"])
    np.testing.assert_allclose(np.hypot(points[:, 0] - 1, points[:, 1] - 2), 3.0)
    np.testing.assert_allclose(points[:, 2], 1.5)


def test_battery_actions_need_a_battery_and_log_charge():
    toggles, models = setup()
    assert not toggles.action_available("battery_refill")
    assert toggles.perform("battery_refill", 0) is None
    data = load_drone_config().data
    battery = dict(data["battery"], enabled=True, initial_soc=0.5)
    powertrain = Powertrain(
        load_drone_config().properties, battery, data["aerodynamics"]["ground_effect"]
    )
    toggles = EffectToggles(
        data["keyboard_effects"], **dict(models, powertrain=powertrain)
    )
    assert toggles.action_available("battery_drain")
    assert toggles.perform("battery_drain", 5)["soc"] == pytest.approx(0.4)
    assert toggles.perform("battery_refill", 6) == dict(
        sim_time_ns=6, effect="battery_refill", on=True, soc=1.0
    )


def test_ambient_keys_need_a_temperature_model_and_move_pack_and_imu():
    from dronelab.sensors.imu_temperature import ImuTemperatureModel

    toggles, models = setup()
    assert not toggles.action_available("colder")
    assert toggles.perform("colder", 0) is None
    data = load_drone_config().data
    battery = dict(data["battery"], enabled=True)
    battery["thermal"] = dict(battery["thermal"], enabled=True)
    powertrain = Powertrain(
        load_drone_config().properties, battery, data["aerodynamics"]["ground_effect"]
    )
    imu = ImuTemperatureModel(
        dict(data["sensors"]["imu"]["temperature"], enabled=True), 25.0
    )
    toggles = EffectToggles(
        data["keyboard_effects"],
        **dict(models, powertrain=powertrain),
        imu_temperature=imu,
        ambient_c=25.0,
    )
    assert toggles.action_available("colder")
    assert toggles.perform("colder", 5)["ambient_c"] == pytest.approx(15.0)
    assert powertrain.battery_state().temperature_c == pytest.approx(15.0)
    assert imu.temperature_c == pytest.approx(15.0)
    for _ in range(10):
        toggles.perform("colder", 6)
    assert toggles.ambient_c == -40.0  # clamped
    assert toggles.perform("colder", 7) is None
    assert toggles.perform("warmer", 8)["ambient_c"] == pytest.approx(-30.0)
