import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dronelab.simulation.gazebo.config import load_drone_config
from dronelab.simulation.gazebo.wind import WindCommands, WindProfile, dynamic_wind


def settings(**overrides):
    result = load_drone_config().data["wind"]
    result.update(overrides)
    return result


def test_schedule_boundaries_and_hold():
    profile = WindProfile(
        settings(
            velocity_enu_mps=[3, 0, 0],
            schedule=[
                {"time_s": 0.75, "velocity_enu_mps": [0, 4, 1]},
                {"time_s": 2, "velocity_enu_mps": [-2, 0, 0]},
            ],
        )
    )
    np.testing.assert_allclose(profile.target(0), [3, 0, 0])
    assert profile.target(100_000_000) is None
    np.testing.assert_allclose(profile.target(749_999_999), [3, 0, 0])
    np.testing.assert_allclose(profile.target(750_000_000), [0, 4, 1], atol=1e-12)
    assert profile.target(750_000_001) is None
    np.testing.assert_allclose(profile.target(2_000_000_000), [-2, 0, 0], atol=1e-12)
    with pytest.raises(ValueError, match="nondecreasing"):
        profile.target(0)


def test_time_zero_schedule_and_disabled_default():
    config = settings(schedule=[{"time_s": 0, "velocity_enu_mps": [0, -2, 0]}])
    np.testing.assert_allclose(WindProfile(config).target(0), [0, -2, 0], atol=1e-12)
    assert not dynamic_wind(config)
    config["enabled"] = True
    assert dynamic_wind(config)
    assert not dynamic_wind(settings(enabled=True))


def test_seeded_random_gusts_and_direction_are_independent_of_poll_frequency():
    config = settings(
        velocity_enu_mps=[3, 0, 1],
        random_speed_std_mps=0.5,
        random_direction_std_deg=20,
        random_seed=12,
    )
    before = np.random.get_state()
    fast, sparse = WindProfile(config), WindProfile(config)
    at_seconds = {}
    for tick in range(501):
        target = fast.target(tick * 10_000_000)
        if tick % 100 == 0:
            at_seconds[tick // 100] = target
    for second, expected in at_seconds.items():
        np.testing.assert_allclose(sparse.target(second * 1_000_000_000), expected)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    speeds = [math.hypot(*v[:2]) for v in at_seconds.values()]
    assert np.std(speeds) > 0.1
    assert any(abs(v[1]) > 0.1 for v in at_seconds.values())
    assert all(v[2] == 1 for v in at_seconds.values())


def test_random_speed_cannot_reverse_wind_and_direction_only_preserves_speed():
    gust = WindProfile(settings(velocity_enu_mps=[0, 0, 0], random_speed_std_mps=100))
    for tick in range(20):
        x, y, _ = gust.target(tick * 1_000_000_000)
        assert x >= 0 and y == 0
    direction = WindProfile(
        settings(velocity_enu_mps=[3, 4, 0], random_direction_std_deg=90)
    )
    for tick in range(20):
        x, y, _ = direction.target(tick * 1_000_000_000)
        assert math.hypot(x, y) == pytest.approx(5)


def test_wind_commands_publish_native_messages_and_report_failures(monkeypatch):
    class Wind:
        def __init__(self):
            self.linear_velocity = SimpleNamespace(x=0, y=0, z=0)

    monkeypatch.setitem(sys.modules, "gz.msgs10.wind_pb2", SimpleNamespace(Wind=Wind))
    messages = []

    class Publisher:
        success = True

        def publish(self, message):
            messages.append(message)
            return self.success

    publisher = Publisher()

    class Node:
        def advertise(self, topic, message_type):
            assert topic == "/world/dronelab/wind" and message_type is Wind
            return publisher

    commands = WindCommands(Node(), settings(velocity_enu_mps=[3, 0, 0]))
    assert commands.update(0) == dict(sim_time_ns=0, velocity_enu_mps=[3, 0, 0])
    assert messages[0].enable_wind
    assert messages[0].linear_velocity.x == 3
    assert commands.update(1) is None
    assert len(messages) == 1
    publisher.success = False
    with pytest.raises(RuntimeError, match="publish wind"):
        commands.update(1_000_000_000)
