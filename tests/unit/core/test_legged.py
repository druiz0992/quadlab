"""Joint targets and nominal legged properties."""

import numpy as np
import pytest

from robotlab.core.legged import JointTargets, LeggedProperties
from robotlab.testing.plugins import nominal_legged


def test_joint_targets_validate_length_and_finiteness() -> None:
    JointTargets((0.0, 1.0)).validate(2)
    with pytest.raises(ValueError, match="needs 3 values"):
        JointTargets((0.0, 1.0)).validate(3)
    with pytest.raises(ValueError, match="finite"):
        JointTargets((0.0, float("nan"))).validate(2)
    assert JointTargets((1, 2)).position_rad == (1.0, 2.0)


def test_hold_uses_the_initial_targets() -> None:
    robot = nominal_legged()
    assert JointTargets.hold(robot).position_rad == (0.0,) * 6
    assert robot.index("right_knee") == 4
    with pytest.raises(ValueError, match="no actuated joint"):
        robot.index("tail")


def test_properties_reject_bad_shapes_and_bounds() -> None:
    robot = nominal_legged()
    assert not robot.initial_target_rad.flags.writeable
    fields = dict(
        name="r",
        joint_names=("a", "b"),
        target_lower_rad=np.array([-1.0, -1.0]),
        target_upper_rad=np.array([1.0, 1.0]),
        position_gain=np.ones(2),
        initial_target_rad=np.zeros(2),
        mass_kg=1.0,
        control_rate_hz=50.0,
    )
    with pytest.raises(ValueError, match="position_gain needs 2"):
        LeggedProperties(**{**fields, "position_gain": np.ones(3)})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lower target bound"):
        LeggedProperties(**{**fields, "target_lower_rad": np.array([2.0, -1.0])})  # type: ignore[arg-type]
