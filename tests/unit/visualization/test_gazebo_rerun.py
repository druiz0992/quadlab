import pytest

from dronelab.visualization.gazebo_rerun import GazeboRerunRecorder, ScalarBatch
from dronelab.visualization.gazebo_rerun import _display_path


def test_display_paths_bound_long_run_work_and_preserve_endpoints():
    for size in (0, 1, 1999, 2000, 2001, 8301, 100000):
        positions = list(range(size))
        displayed = _display_path(positions)
        assert len(displayed) <= 2000
        if size:
            assert displayed[0] == 0
            assert displayed[-1] == size - 1
            assert displayed == sorted(set(displayed))
        if size <= 2000:
            assert displayed == positions
        assert len(positions) == size


class RecordingRerun:
    def __init__(self):
        self.times = []

    def set_time(self, timeline, *, duration):
        self.times.append((timeline, duration))


def recorder_with_columns():
    """Recorder without Rerun; scalar columns land in ``columns``."""
    recorder = GazeboRerunRecorder.__new__(GazeboRerunRecorder)
    recorder.rr = RecordingRerun()
    recorder._time_s = 0.0
    columns = []
    recorder._scalars = ScalarBatch(
        lambda path, times, values: columns.append((path, times, values)),
        period_sim_s=1e9,  # flushed explicitly by the tests
    )
    return recorder, columns


def test_published_thrust_timeseries_and_zero_on_expiry():
    recorder, columns = recorder_with_columns()
    event = dict(status="published", sim_time_ns=2_000_000_000, thrust_n=[1, 2, 3, 4])
    recorder.log_actuator_event(event)
    assert recorder.rr.times == [("sim_time", 2)]
    recorder.log_actuator_event(dict(status="superseded", sim_time_ns=2, thrust_n=None))
    recorder.log_actuator_event(
        dict(status="expired", sim_time_ns=3_000_000_000, thrust_n=[0] * 4)
    )
    recorder._scalars.flush()
    assert columns == [
        (f"actuators/thrust_target_n/rotor_{i}", [2.0, 3.0], [float(value), 0.0])
        for i, value in enumerate([1, 2, 3, 4])
    ]


def test_scalars_use_the_latest_set_time():
    recorder, columns = recorder_with_columns()
    recorder._set_time(1.5)
    recorder.log_vector("algorithms/x", [1, 2], labels=("a", "b"))
    recorder._scalars.flush()
    assert columns == [
        ("algorithms/x/a", [1.5], [1.0]),
        ("algorithms/x/b", [1.5], [2.0]),
    ]


def test_scalar_batch_keeps_every_sample_and_flushes_by_sim_time():
    sent = []
    batch = ScalarBatch(lambda *column: sent.append(column), period_sim_s=0.1)
    for k in range(10):  # 0.00 .. 0.09 s: one batch still open
        batch.add("a", k * 0.01, k)
    assert sent == []
    batch.add("a", 0.1, 10)  # period reached: everything so far is sent
    batch.add("b", 0.11, -1)
    assert sent == [("a", [k * 0.01 for k in range(10)] + [0.1], list(range(11)))]
    batch.flush()
    assert sent[1:] == [("b", [0.11], [-1.0])]
    batch.flush()  # nothing buffered: nothing sent
    assert len(sent) == 2


def test_scalar_batch_rejects_negative_period():
    with pytest.raises(ValueError):
        ScalarBatch(lambda *column: None, period_sim_s=-1)
