import pytest

from robotlab.control.timing import DelayLine, FixedRateTicker, StateBuffer

MS = 1_000_000


def test_ticker_runs_on_grid_anchored_at_first_sample():
    ticker = FixedRateTicker(100.0)  # 10 ms grid
    times = [2 * MS + i * 5 * MS for i in range(6)]  # 200 Hz samples
    ticks = [ticker.offer(t) for t in times]
    assert [t is not None for t in ticks] == [True, False, True, False, True, False]
    assert [t.sim_time_ns for t in ticks if t] == [2 * MS, 12 * MS, 22 * MS]
    assert all(t.dt_s == pytest.approx(0.01) for t in ticks if t)


def test_ticker_tolerance_absorbs_early_timestamps():
    strict, tolerant = FixedRateTicker(100.0), FixedRateTicker(100.0, tolerance_ns=MS)
    for ticker in (strict, tolerant):
        ticker.offer(0)
    assert strict.offer(10 * MS - MS // 2) is None
    assert tolerant.offer(10 * MS - MS // 2).sim_time_ns == 10 * MS


def test_ticker_gap_runs_latest_grid_time_and_counts_missed():
    ticker = FixedRateTicker(100.0)
    ticker.offer(0)
    tick = ticker.offer(35 * MS)
    assert (tick.index, tick.sim_time_ns, tick.missed) == (3, 30 * MS, 2)
    assert tick.dt_s == pytest.approx(0.03)
    assert (ticker.ticks, ticker.missed) == (2, 2)


def test_state_buffer_returns_newest_sample_old_enough():
    buffer = StateBuffer(latency_ns=20 * MS)
    for t in range(0, 60 * MS, 10 * MS):
        buffer.push(t, t // MS)
    assert buffer.available(10 * MS) is None  # nothing 20 ms old yet
    assert buffer.available(45 * MS) == (20 * MS, 20)
    assert buffer.available(50 * MS) == (30 * MS, 30)
    with pytest.raises(ValueError, match="increasing"):
        buffer.push(50 * MS, 0)


def test_state_buffer_zero_latency_returns_latest():
    buffer = StateBuffer(latency_ns=0)
    buffer.push(0, "a")
    buffer.push(5, "b")
    assert buffer.available(5) == (5, "b")


def test_delay_line_releases_in_order_after_delay():
    line = DelayLine(delay_ns=15 * MS)
    line.push(0, "a")
    line.push(10 * MS, "b")
    assert line.pop_ready(14 * MS) == []
    assert line.pop_ready(30 * MS) == [(15 * MS, "a"), (25 * MS, "b")]
    assert len(line) == 0
    with pytest.raises(ValueError):
        DelayLine(-1)
