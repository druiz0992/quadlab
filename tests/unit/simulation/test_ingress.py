"""Gazebo transport ingress: bounded queue, drop counting, topics (no Gazebo)."""

from pathlib import Path
from queue import Empty

import pytest

from robotlab.simulation.gazebo import ingress as ingress_module
from robotlab.simulation.gazebo.ingress import (
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
            return topic != "/robotlab/imu"

    topics = [("x", "/robotlab/odometry", "odometry"), ("y", "/robotlab/imu", "imu")]
    with pytest.raises(ValueError, match="/robotlab/imu"):
        MessageIngress().subscribe(Node(), topics)


def test_gz_sim_command():
    world = Path("/w/world.sdf")
    assert gz_sim_command(
        world, paused=False, headless=False, render_sensors=False
    ) == ["gz", "sim", "-r", "-v", "3", "/w/world.sdf"]
    assert gz_sim_command(world, paused=True, headless=True, render_sensors=True) == [
        "gz",
        "sim",
        "-v",
        "3",
        "-s",
        "--headless-rendering",
        "/w/world.sdf",
    ]


# Lockstep ordering -------------------------------------------------------------

from types import SimpleNamespace

from robotlab.simulation.gazebo.ingress import LockstepOrderer, scheduled_stamp_ns

STEP = 2_000_000


def msg(stamp_ns, tag=None):
    sec, nsec = divmod(stamp_ns, 1_000_000_000)
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nsec=nsec)), tag=tag
    )


def test_scheduled_stamps_match_gazebo():
    # Measured with gz-sim 8 on a 2 ms step (see scheduled_stamp_ns).
    imu = [scheduled_stamp_ns(5e6, STEP, t * 1_000_000) for t in (2, 4, 6, 10, 16, 20)]
    assert imu == [s * 1_000_000 for s in (2, 2, 6, 10, 16, 20)]
    camera = [
        scheduled_stamp_ns(1e9 / 30, STEP, t * 1_000_000) for t in (33, 34, 68, 100)
    ]
    assert camera == [s * 1_000_000 for s in (2, 34, 68, 100)]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.001
        return self.now


def orderer(periods, **kwargs):
    ingress = MessageIngress(clock=lambda: 0.0)
    return ingress, LockstepOrderer(ingress, periods, STEP, clock=FakeClock(), **kwargs)


def drain(source):
    out = []
    while True:
        try:
            kind, message, _ = source.get(timeout_s=0.05)
        except Empty:
            return out
        out.append((kind, message.tag))


def test_waits_for_scheduled_samples_then_releases_in_stamp_order():
    ingress, source = orderer({"imu": 5e6, "magnetometer": 20e6})
    # Each stream arrives in order; streams interleave arbitrarily.
    for kind, message in [
        ("imu", msg(2_000_000, "i2")),
        ("magnetometer", msg(2_000_000, "m2")),
        ("odometry", msg(20_000_000, "o20")),  # barrier arrives early
        ("magnetometer", msg(20_000_000, "m20")),
        ("imu", msg(6_000_000, "i6")),
        ("imu", msg(10_000_000, "i10")),
    ]:
        ingress.callback(kind)(message)
    # imu 16 and 20 ms still missing: nothing is released yet.
    assert drain(source) == []
    for stamp in (16, 20):
        ingress.callback("imu")(msg(stamp * 1_000_000, f"i{stamp}"))
    assert drain(source) == [
        ("imu", "i2"),
        ("magnetometer", "m2"),
        ("imu", "i6"),
        ("imu", "i10"),
        ("imu", "i16"),
        ("imu", "i20"),
        ("magnetometer", "m20"),
        ("odometry", "o20"),  # measurements stamped T come before truth at T
    ]
    assert source.barrier_timeouts == 0


def test_duplicates_are_dropped_and_counted():
    ingress, source = orderer({})
    for stamp in (2_000_000, 2_000_000, 4_000_000, 4_000_000, 4_000_000):
        ingress.callback("esc")(msg(stamp, stamp))
    ingress.callback("odometry")(msg(4_000_000, "o"))
    assert drain(source) == [("esc", 2_000_000), ("esc", 4_000_000), ("odometry", "o")]
    assert source.duplicates == {"esc": 3}


def test_missing_sample_times_out_and_is_counted():
    ingress, source = orderer({"gps": 100e6}, wait_timeout_s=0.01)
    ingress.callback("odometry")(msg(20_000_000, "o"))
    assert drain(source) == [("odometry", "o")]
    assert source.barrier_timeouts == 1
    assert source.summary()["timeout_streams"] == {"gps": 1}


# Start-up subscriber barrier -----------------------------------------------------

from robotlab.simulation.gazebo.stepping import TrackedNode, wait_for_subscribers


class FakePublisher:
    def __init__(self, connected_after):
        self.polls, self.connected_after = 0, connected_after

    def has_connections(self):
        self.polls += 1
        return self.polls > self.connected_after


def test_tracked_node_records_advertised_topics():
    inner = SimpleNamespace(advertise=lambda topic, cls: (topic, cls), request="r")
    node = TrackedNode(inner)
    node.advertise("/a", int)
    assert node.publishers == [("/a", ("/a", int))] and node.request == "r"


def test_wait_for_subscribers_blocks_until_connected_or_names_topics():
    clock = FakeClock()
    publishers = [("/motors", FakePublisher(3)), ("/wind", FakePublisher(0))]
    wait_for_subscribers(publishers, monotonic=clock, sleep=lambda s: None)
    assert publishers[0][1].polls == 4
    never = [("/motors", FakePublisher(10**9)), ("/wind", FakePublisher(0))]
    with pytest.raises(RuntimeError, match=r"\['/motors'\]"):
        wait_for_subscribers(
            never, timeout_s=0.01, monotonic=clock, sleep=lambda s: None
        )


def test_wait_for_subscribers_fails_fast_when_gazebo_exits():
    clock = FakeClock()
    never = [("/motors", FakePublisher(10**9))]
    checks = []

    def alive():
        checks.append(clock())
        return len(checks) < 3  # Gazebo dies on the third check

    with pytest.raises(RuntimeError, match="exited during start-up"):
        wait_for_subscribers(
            never, timeout_s=3600, monotonic=clock, sleep=lambda s: None, alive=alive
        )
    assert len(checks) == 3


def test_opponent_odometry_precedes_ego_odometry_in_name_order():
    from robotlab.simulation.gazebo.ingress import KIND_PRIORITY, kind_priority

    assert (
        kind_priority("contacts")
        < kind_priority("opponent:car_2")
        < kind_priority("odometry")
    )
    assert kind_priority("opponent:car_1") == kind_priority("opponent:car_2")
    assert kind_priority("unknown") == KIND_PRIORITY["odometry"]
