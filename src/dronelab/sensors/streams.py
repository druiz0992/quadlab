"""Bounded in-process channels. Gazebo session/worker wiring is separate work.

Only the producer holds a Publisher; consumers receive its read-only Channel.
Payload types must themselves be immutable (as the measurement contracts are).
"""

from collections import deque
import threading
import time
from typing import Generic, TypeVar

from ..core.measurements import Sample

T = TypeVar("T")


class StreamOverflow(RuntimeError):
    pass


class SessionClosed(RuntimeError):
    pass


class ReadTimeout(TimeoutError):
    pass


class StaleSample(RuntimeError):
    pass


class Subscription(Generic[T]):
    def __init__(self, channel, capacity, overflow):
        self._channel, self._capacity, self._overflow = channel, capacity, overflow
        self._queue = deque()
        self._closed = False
        self._failed = False
        self.dropped = 0

    def next(self, *, timeout_wall_s=None):
        if timeout_wall_s is not None and (
            not isinstance(timeout_wall_s, (int, float))
            or not 0 <= timeout_wall_s < float("inf")
        ):
            raise ValueError("Timeout must be finite and nonnegative")
        deadline = None if timeout_wall_s is None else time.monotonic() + timeout_wall_s
        with self._channel._condition:
            while True:
                if self._failed:
                    raise StreamOverflow("Subscription lost samples")
                if self._queue:
                    return self._queue.popleft()
                if self._closed or self._channel._closed:
                    raise SessionClosed("Channel or subscription closed")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise ReadTimeout("No sample before timeout")
                self._channel._condition.wait(remaining)

    def close(self):
        with self._channel._condition:
            self._closed = True
            self._queue.clear()
            self._channel._subscribers.discard(self)
            self._channel._condition.notify_all()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.next()
        except SessionClosed:
            raise StopIteration from None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Channel(Generic[T]):
    def __init__(self, stream_id, payload_type, frame_id):
        self.stream_id, self.payload_type, self.frame_id = (
            stream_id,
            payload_type,
            frame_id,
        )
        self._condition = threading.Condition()
        self._subscribers = set()
        self._latest = None
        self._closed = False

    def subscribe(self, *, capacity, overflow="error"):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
        if overflow not in {"error", "drop_oldest"}:
            raise ValueError("Unknown overflow policy")
        with self._condition:
            if self._closed:
                raise SessionClosed("Channel closed")
            subscription = Subscription(self, capacity, overflow)
            self._subscribers.add(subscription)
            return subscription

    def latest(self, *, now_sim_time_ns, max_age_sim_s):
        if not isinstance(now_sim_time_ns, int) or now_sim_time_ns < 0:
            raise ValueError("Simulation time must be nonnegative integer ns")
        if not 0 <= max_age_sim_s < float("inf"):
            raise ValueError("Maximum age must be finite and nonnegative")
        with self._condition:
            if self._closed:
                raise SessionClosed("Channel closed")
            if self._latest is None:
                return None
            age = now_sim_time_ns - self._latest.sim_time_ns
            if age < 0 or age > max_age_sim_s * 1e9:
                raise StaleSample("Latest sample is outside the requested time window")
            return self._latest


class Publisher(Generic[T]):
    def __init__(self, stream_id, payload_type, *, frame_id, epoch=0):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer")
        self.channel = Channel(stream_id, payload_type, frame_id)
        self._epoch, self._sequence = epoch, 0

    def publish(self, payload, *, sim_time_ns, parents=()):
        if (
            isinstance(sim_time_ns, bool)
            or not isinstance(sim_time_ns, int)
            or sim_time_ns < 0
        ):
            raise ValueError("Simulation time must be nonnegative integer ns")
        if not isinstance(payload, self.channel.payload_type):
            raise TypeError("Payload does not match channel type")
        with self.channel._condition:
            if self.channel._closed:
                raise SessionClosed("Publisher closed")
            latest = self.channel._latest
            if latest is not None and sim_time_ns < latest.sim_time_ns:
                raise ValueError("Output timestamps cannot decrease")
            self._sequence += 1
            sample = Sample(
                self.channel.stream_id,
                self._sequence,
                self._epoch,
                sim_time_ns,
                time.monotonic_ns(),
                self.channel.frame_id,
                payload,
                tuple(parents),
            )
            self.channel._latest = sample
            for reader in self.channel._subscribers:
                if reader._failed:
                    continue
                if len(reader._queue) == reader._capacity:
                    reader.dropped += 1
                    if reader._overflow == "error":
                        reader._failed = True
                        reader._queue.clear()
                        continue
                    reader._queue.popleft()
                reader._queue.append(sample)
            self.channel._condition.notify_all()
            return sample

    def close(self):
        with self.channel._condition:
            self.channel._closed = True
            self.channel._condition.notify_all()
