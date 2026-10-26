"""Named state blocks: slices, validation, reported estimates."""

import numpy as np
import pytest

from robotlab.estimation.filter.state_layout import StateLayout


def test_blocks_get_consecutive_slices() -> None:
    layout = StateLayout.of("position", ("baro_offset", 1), "yaw")
    assert layout.size == 5 and layout.names == ("position", "baro_offset", "yaw")
    assert layout.index("baro_offset") == slice(3, 4)
    assert layout.index("yaw") == slice(4, 5)
    assert "yaw" in layout and "attitude" not in layout
    assert layout.jacobian(2).shape == (2, 5)


@pytest.mark.parametrize(
    "blocks, message",
    [
        (("position", "position"), "unique"),
        ((("position", 2),), "size 3"),
        (("attitude", "yaw"), "either"),
        ((("offset", 0),), "positive"),
    ],
)
def test_invalid_layouts_raise(blocks: tuple, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StateLayout.of(*blocks)


def test_unknown_names_need_a_size_and_unknown_blocks_raise() -> None:
    with pytest.raises(ValueError, match="standard block"):
        StateLayout.of("offset")
    with pytest.raises(ValueError, match="No state block"):
        StateLayout.of("position").index("velocity")


def test_estimated_state_marginalizes_unreported_blocks() -> None:
    layout = StateLayout.of("velocity", ("baro_offset", 1), "position")
    P = np.diag(np.arange(1.0, 8.0))
    estimate = layout.estimated_state(
        P, position_m=[1.0, 2.0, 3.0], velocity_mps=[0.0, 0.0, 0.0]
    )
    # EstimatedState order (position, velocity), without the offset.
    assert estimate.covariance is not None
    np.testing.assert_array_equal(
        np.diag(estimate.covariance), [5.0, 6.0, 7.0, 1.0, 2.0, 3.0]
    )


def test_estimated_state_needs_covariance_for_every_reported_block() -> None:
    layout = StateLayout.of("position", "yaw")
    with pytest.raises(ValueError, match="covariance=None"):
        layout.estimated_state(
            np.eye(4), position_m=[0, 0, 0], quaternion_xyzw=[0, 0, 0, 1]
        )
    estimate = layout.estimated_state(
        position_m=[0, 0, 0], quaternion_xyzw=[0, 0, 0, 1]
    )
    assert estimate.covariance is None
    with pytest.raises(ValueError, match="4x4"):
        layout.estimated_state(np.eye(3), position_m=[0, 0, 0])
