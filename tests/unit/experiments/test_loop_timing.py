import pytest

from dronelab.experiments.loop_timing import LATE_ODOMETRY_WALL_S, LoopTiming


class FakeClock:
    def __init__(self) -> None:
        self.now_s = 100.0

    def __call__(self) -> float:
        return self.now_s


def test_load_queue_age_and_per_kind_cost() -> None:
    clock = FakeClock()
    timing = LoopTiming(clock)
    # Odometry received 1 ms ago, handled in 2 ms, at sim 0.0 s.
    timing.begin("odometry", received_at_wall_s=clock.now_s - 0.001)
    clock.now_s += 0.002
    timing.end(sim_time_s=0.0)
    # Housekeeping section: no queue age.
    timing.begin("housekeeping")
    clock.now_s += 0.001
    timing.end(sim_time_s=0.0)
    # Late odometry (beyond the threshold), handled in 3 ms, at sim 0.01 s.
    timing.begin("odometry", received_at_wall_s=clock.now_s - 2 * LATE_ODOMETRY_WALL_S)
    clock.now_s += 0.003
    timing.end(sim_time_s=0.01)

    summary = timing.summary()
    assert summary["busy_wall_s"] == pytest.approx(0.006)
    assert summary["sim_span_s"] == pytest.approx(0.01)
    assert summary["load"] == pytest.approx(0.6)
    age = summary["odometry_queue_age_ms"]
    assert age["max"] == pytest.approx(100.0)
    assert age["mean"] == pytest.approx(50.5)
    assert age["late"] == 1
    odometry = summary["per_kind"]["odometry"]
    assert odometry["messages"] == 2
    assert odometry["slowest_ms"] == pytest.approx(3.0)
    assert summary["per_kind"]["housekeeping"]["busy_wall_s"] == pytest.approx(0.001)


def test_empty_run_has_no_load() -> None:
    summary = LoopTiming(FakeClock()).summary()
    assert summary["load"] is None
    assert summary["odometry_queue_age_ms"]["mean"] is None
    assert summary["per_kind"] == {}


def test_end_without_begin_raises() -> None:
    timing = LoopTiming(FakeClock())
    with pytest.raises(RuntimeError):
        timing.end()
