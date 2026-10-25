import threading

import numpy as np
import pytest

from robotlab.core.measurements import ImuReading
from robotlab.sensors.bus.streams import (
    Publisher,
    ReadTimeout,
    SessionClosed,
    StaleSample,
    StreamOverflow,
)


def reading():
    return ImuReading([0, 0, 9.81], [0, 0, 0])


def test_fanout_overflow_and_readonly_buffers():
    producer = Publisher("imu", ImuReading, frame_id="imu")
    fast = producer.channel.subscribe(capacity=4)
    slow = producer.channel.subscribe(capacity=1)
    latest_only = producer.channel.subscribe(capacity=1, overflow="drop_oldest")
    first = producer.publish(reading(), sim_time_ns=0)
    producer.publish(reading(), sim_time_ns=1)
    assert fast.next(timeout_wall_s=0) is first
    assert fast.next(timeout_wall_s=0).sequence == 2
    with pytest.raises(StreamOverflow):
        slow.next(timeout_wall_s=0)
    assert latest_only.next(timeout_wall_s=0).sequence == 2
    assert latest_only.dropped == 1
    with pytest.raises(ValueError):
        first.payload.specific_force_mps2.setflags(write=True)
    with pytest.raises(ReadTimeout):
        fast.next(timeout_wall_s=0)
    assert producer.channel.latest(now_sim_time_ns=1, max_age_sim_s=0).sequence == 2
    with pytest.raises(StaleSample):
        producer.channel.latest(now_sim_time_ns=10, max_age_sim_s=0)
    with pytest.raises(ValueError):
        producer.publish(reading(), sim_time_ns=0)
    producer.close()
    with pytest.raises(SessionClosed):
        producer.publish(reading(), sim_time_ns=2)


def test_close_unblocks_reader_and_producer_close_drains_queue():
    producer = Publisher("imu", ImuReading, frame_id="imu")
    reader = producer.channel.subscribe(capacity=2)
    received = []

    def consume():
        received.extend(reader)

    worker = threading.Thread(target=consume)
    worker.start()
    producer.publish(reading(), sim_time_ns=0)
    producer.close()
    worker.join(timeout=2)
    assert not worker.is_alive() and len(received) == 1
    reader.close()
