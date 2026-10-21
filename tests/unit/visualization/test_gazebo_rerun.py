from dronelab.visualization.gazebo_rerun import GazeboRerunRecorder
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
        self.logs = []
        self.times = []

    def set_time(self, timeline, *, duration):
        self.times.append((timeline, duration))

    def log(self, path, data):
        self.logs.append((path, data))

    def Scalars(self, value):
        return value


def test_published_thrust_timeseries_and_zero_on_expiry():
    recorder = GazeboRerunRecorder.__new__(GazeboRerunRecorder)
    rr = recorder.rr = RecordingRerun()
    event = dict(status="published", sim_time_ns=2_000_000_000, thrust_n=[1, 2, 3, 4])
    recorder.log_actuator_event(event)
    assert rr.times == [("sim_time", 2)]
    assert rr.logs == [
        (f"actuators/thrust_target_n/rotor_{i}", value)
        for i, value in enumerate([1, 2, 3, 4])
    ]
    recorder.log_actuator_event(dict(status="superseded", sim_time_ns=2, thrust_n=None))
    assert len(rr.logs) == 4
    recorder.log_actuator_event(
        dict(status="expired", sim_time_ns=3_000_000_000, thrust_n=[0] * 4)
    )
    assert [data for _, data in rr.logs[4:]] == [0] * 4
