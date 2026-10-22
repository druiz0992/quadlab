import numpy as np
import pytest

from robotlab.core.track import Track

SQUARE = Track(((0, 0), (100, 0), (100, 100), (0, 100)), 10.0, closed=True)


def test_projection_on_a_square_loop():
    projection = SQUARE.project([[50, 2], [100.5, 30], [10, 100], [-3, 50]])
    np.testing.assert_allclose(projection.s_m, [50, 130, 290, 350])
    # Driving down the last side (−y), +x is to the left.
    np.testing.assert_allclose(projection.lateral_m, [2, -0.5, 0, -3])
    assert SQUARE.length_m == 400
    np.testing.assert_allclose(
        SQUARE.heading_rad([0, 1, 2, 3]), [0, np.pi / 2, np.pi, -np.pi / 2]
    )


def test_open_route_has_no_closing_segment():
    route = Track(((0, 0), (10, 0), (20, 0)), 4.0, closed=False)
    assert route.length_m == 20
    assert route.project([[-5, 1]]).s_m[0] == 0


@pytest.mark.parametrize(
    "track, message",
    [
        ({"centreline_m": [[0, 0], [1, 0]], "width_m": 5}, "three"),
        ({"centreline_m": [[0, 0], [1, 0], [2, 0]], "width_m": 0}, "positive"),
        ({"centreline_m": [[0, 0], [0, 0], [2, 0]], "width_m": 3}, "repeated"),
        ({"centreline_m": [[0, 0], [1, 0], [2, 0]], "width_m": 3, "kerbs": 1}, "needs"),
        (
            {"centreline_m": [[0, 0], [1, 0], [2, 1]], "width_m": 3, "closed": 1},
            "closed",
        ),
    ],
)
def test_invalid_tracks(track, message):
    with pytest.raises(ValueError, match=message):
        Track.from_scene({"track": track})


def test_scene_without_track():
    assert Track.from_scene({}) is None
