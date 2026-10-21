"""Gazebo transport ingress: bounded queue, drop counting, topics (no Gazebo)."""

from pathlib import Path
from queue import Empty

import pytest

from dronelab.simulation.gazebo import ingress as ingress_module
from dronelab.simulation.gazebo.ingress import (
    MessageIngress,
    gz_sim_command,
    subscriptions,
)


def test_queue_stamps_receipt_and_counts_drops():
    ingress = MessageIngress(capacity=2, clock=lambda: 12.5)
    imu, gps = ingress.callback("imu"), ingress.callback("gps")
    imu("a")
    gps("b")
    imu("c")  # full: dropped and counted, never silently lost
    assert ingress.dropped == {"imu": 1}
    assert ingress.get(timeout_s=0) == ("imu", "a", 12.5)
    assert ingress.get(timeout_s=0) == ("gps", "b", 12.5)
    with pytest.raises(Empty):
        ingress.get(timeout_s=0)


def test_subscriptions_follow_enabled_sensors():
    sensors = {kind: {"enabled": False} for kind, _, _ in ingress_module.SENSOR_TOPICS}
    sensors["imu"]["enabled"] = sensors["contacts"]["enabled"] = True
    kinds = [kind for _, _, kind in subscriptions(sensors, esc_telemetry=False)]
    assert kinds == ["odometry", "poses", "imu", "contacts"]
    assert "esc" in [kind for _, _, kind in subscriptions(sensors, esc_telemetry=True)]


def test_subscribe_failure_is_an_error(monkeypatch):
    monkeypatch.setattr(ingress_module, "_message_class", lambda name: name)

    class Node:
        def subscribe(self, cls, topic, callback):
            return topic != "/dronelab/imu"

    topics = [("x", "/dronelab/odometry", "odometry"), ("y", "/dronelab/imu", "imu")]
    with pytest.raises(ValueError, match="/dronelab/imu"):
        MessageIngress().subscribe(Node(), topics)


def test_gz_sim_command():
    world = Path("/w/world.sdf")
    assert gz_sim_command(
        world, lockstep=False, headless=False, render_sensors=False
    ) == ["gz", "sim", "-r", "-v", "3", "/w/world.sdf"]
    assert gz_sim_command(world, lockstep=True, headless=True, render_sensors=True) == [
        "gz",
        "sim",
        "-v",
        "3",
        "-s",
        "--headless-rendering",
        "/w/world.sdf",
    ]
