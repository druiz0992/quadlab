"""Sensor bus: stamping, delivery delay, ordering, out-of-order streams."""

import numpy as np
import pytest

from robotlab.core.measurements import GpsReading, ImuReading
from robotlab.sensors.bus import SensorBus
from robotlab.simulation.gazebo.config import load_drone_config

MS = 1_000_000


def timing(**streams):
    settings = load_drone_config().data["sensor_timing"]
    settings["reorder_window_s"] = 0.0
    for kind, values in streams.items():
        settings[kind].update(values)
    return settings


def imu():
    return ImuReading([0, 0, 9.81], [0, 0, 0])


def test_ideal_bus_delivers_immediately_in_time_order_imu_first():
    bus = SensorBus(timing(), ["imu", "gps"])
    bus.submit("gps", GpsReading(40, -3, 650), 10 * MS)
    bus.submit("imu", imu(), 10 * MS)
    released = bus.advance(10 * MS)
    assert [k for k, _ in released] == ["imu", "gps"]
    assert [s.sim_time_ns for _, s in released] == [10 * MS, 10 * MS]
    assert released[1][1].frame_id == "wgs84"


def test_latency_makes_gps_arrive_after_newer_imu():
    bus = SensorBus(timing(gps=dict(latency_s=0.1)), ["imu", "gps"])
    bus.submit("gps", GpsReading(40, -3, 650), 0)
    delivered = []
    for i in range(1, 30):
        bus.submit("imu", imu(), i * 5 * MS)
        delivered += bus.advance(i * 5 * MS)
    kinds = [k for k, _ in delivered]
    gps_index = kinds.index("gps")
    assert gps_index == 19  # after IMU at 5..95 ms, with the IMU at 100 ms
    assert delivered[gps_index][1].sim_time_ns == 0  # stamped at measurement
    assert bus.summary()["late_samples"] == 1
    assert bus.summary()["max_delay_ms"]["gps"] == pytest.approx(100)


def test_timestamp_offset_and_jitter_keep_streams_fifo():
    bus = SensorBus(timing(imu=dict(timestamp_offset_s=-0.002, jitter_s=0.01)), ["imu"])
    for i in range(200):
        bus.submit("imu", imu(), (i + 1) * 4 * MS)
    released = bus.advance(10_000 * MS)
    stamps = [s.sim_time_ns for _, s in released]
    assert stamps == sorted(stamps) and stamps[0] == 2 * MS
    assert [s.sequence for _, s in released] == list(range(1, 201))


def test_reorder_window_holds_samples():
    settings = timing()
    settings["reorder_window_s"] = 0.004
    bus = SensorBus(settings, ["imu"])
    bus.submit("imu", imu(), 10 * MS)
    assert bus.advance(12 * MS) == []
    assert len(bus.advance(14 * MS)) == 1


def test_disabled_streams_are_ignored_and_unknown_channels_fail():
    bus = SensorBus(timing(), ["imu"])
    bus.submit("gps", GpsReading(40, -3, 650), 0)
    assert bus.advance(MS) == []
    with pytest.raises(KeyError):
        bus.channel("gps")
    channel = bus.channel("imu")
    with channel.subscribe(capacity=4) as reader:
        bus.submit("imu", imu(), MS)
        bus.advance(MS)
        assert np.allclose(
            reader.next(timeout_wall_s=0).payload.specific_force_mps2, [0, 0, 9.81]
        )
