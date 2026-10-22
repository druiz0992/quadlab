import numpy as np
import pytest

from robotlab.core.track import Track
from robotlab.evaluation.laps import LapTimer

SQUARE = Track(((0, 0), (100, 0), (100, 100), (0, 100)), 10.0)


def loop_positions(laps, start_s=-20.0, step_m=1.0):
    """Points along the square centreline from ``start_s`` (before the line)."""
    s = np.arange(start_s, laps * 400 + 1, step_m) % 400
    points = []
    for value in s:
        side, along = divmod(value, 100)
        corner = np.array([[0, 0], [100, 0], [100, 100], [0, 100]][int(side)], float)
        direction = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]][int(side)], float)
        points.append(corner + along * direction)
    return np.array(points)


def wheels(point, offset=0.0):
    return np.array([point + [0, offset + 1], point + [0, offset - 1]])


def test_laps_are_timed_from_the_first_crossing():
    timer = LapTimer(SQUARE)
    for index, point in enumerate(loop_positions(2)):
        timer.on_truth(index * 0.1, point, wheels(point))
    summary = timer.summary()
    # Crossings at s = 0 (timing starts), 400, and 800 m: two timed laps.
    assert summary["laps_completed"] == 2
    np.testing.assert_allclose(summary["lap_times_s"], [40.0, 40.0], atol=0.11)
    assert summary["progress_m"] == pytest.approx(820.0, abs=1.5)
    assert summary["off_track_episodes"] == 0


def test_reversing_over_the_line_undoes_the_crossing():
    timer = LapTimer(SQUARE)
    points = [np.array([x, 0.0]) for x in (-5.0, -1.0, 1.0, 5.0, 1.0, -1.0, -5.0)]
    for index, point in enumerate(points):
        timer.on_truth(index * 1.0, point, wheels(point))
    assert timer.summary()["line_crossings"] == 0
    assert timer.summary()["laps_completed"] == 0


def test_off_track_needs_every_wheel_outside():
    timer = LapTimer(SQUARE)
    point = np.array([50.0, 0.0])
    timer.on_truth(0.0, point, wheels(point, offset=4.5))  # one wheel inside
    timer.on_truth(0.1, point, wheels(point, offset=7.0))  # both outside
    timer.on_truth(0.2, point, wheels(point, offset=7.0))
    timer.on_truth(0.3, point, wheels(point))
    timer.on_truth(0.4, point, wheels(point, offset=-8.0))
    summary = timer.summary()
    assert summary["off_track_episodes"] == 2
    assert summary["off_track_s"] == pytest.approx(0.1)
    assert timer.off_track_since_s == pytest.approx(0.4)


def test_open_route_progress_finish_and_rms_offset():
    route = Track.from_route([[0, 0], [50, 0]], 4.0)
    timer = LapTimer(route)
    for index, x in enumerate(np.arange(0.0, 50.5, 0.5)):
        point = np.array([x, 1.0 if index % 2 else -1.0])
        timer.on_truth(index * 0.1, point, wheels(point))
    summary = timer.summary()
    assert summary["closed"] is False and summary["laps_completed"] == 0
    assert summary["progress_m"] == pytest.approx(50.0)
    # Finished at the first sample within 1 m of the end (x = 49 m).
    assert summary["finished_s"] == pytest.approx(9.8)
    assert summary["rms_lateral_offset_m"] == pytest.approx(1.0)
    assert summary["max_lateral_offset_m"] == pytest.approx(1.0)
    assert summary["off_track_episodes"] == 0


def test_closed_loop_reports_no_finish():
    timer = LapTimer(SQUARE)
    point = np.array([5.0, 0.0])
    timer.on_truth(0.0, point, wheels(point))
    assert timer.summary()["finished_s"] is None
    assert timer.summary()["closed"] is True
