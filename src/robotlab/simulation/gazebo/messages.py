"""Small adapters for Gazebo message timestamps and vectors."""

import numpy as np


def vector(value):
    return np.array([value.x, value.y, value.z])


def timestamp(message):
    stamp = message.header.stamp
    return stamp.sec + stamp.nsec * 1e-9


def with_timestamp_ns(message, sim_time_ns):
    """Copy of a message with its header stamp set to ``sim_time_ns``."""
    from copy import deepcopy

    result = deepcopy(message)
    result.header.stamp.sec, result.header.stamp.nsec = divmod(
        int(sim_time_ns), 1_000_000_000
    )
    return result


def timestamp_ns(message):
    """Preserve integer source time for algorithms, without float round trips."""
    stamp = message.header.stamp
    if stamp.sec < 0 or not 0 <= stamp.nsec < 1_000_000_000:
        raise ValueError("Invalid Gazebo source timestamp")
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nsec)
