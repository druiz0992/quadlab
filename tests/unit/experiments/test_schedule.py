import pytest

from robotlab.experiments.common.schedule import next_due_ns

MS = 1_000_000


def run_ticks(period_ms, tick_ms, until_ms):
    """Ticks on which a task of ``period_ms`` runs, on a ``tick_ms`` grid."""
    runs, due = [], None
    for stamp in range(tick_ms * MS, until_ms * MS + 1, tick_ms * MS):
        if due is None or stamp >= due:
            runs.append(stamp // MS)
            due = next_due_ns(due, stamp, period_ms * MS)
    return runs


def test_period_not_a_multiple_of_the_tick_keeps_its_rate():
    # 10 ms on 4 ms ticks: first tick at or after 4, 14, 24, 34, ... ms.
    assert run_ticks(10, 4, 44) == [4, 16, 24, 36, 44]
    assert len(run_ticks(10, 4, 1004)) == 101  # 100 Hz, not 83 Hz


def test_multiples_of_the_tick_are_unchanged():
    assert run_ticks(8, 4, 40) == [4, 12, 20, 28, 36]


def test_missed_due_times_are_skipped_not_repeated():
    assert next_due_ns(10 * MS, 35 * MS, 10 * MS) == 40 * MS
    assert next_due_ns(None, 7, 5) == 12
    with pytest.raises(ValueError, match="positive"):
        next_due_ns(None, 0, 0)
