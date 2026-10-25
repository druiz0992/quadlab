"""Legged controller contract: built-ins, example, and the contract kit."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from robotlab.control.legged import LeggedControlState, WalkCommand
from robotlab.control.legged_hold import HoldPose
from robotlab.control.examples.pose_hold_legged_controller import (
    PoseHoldLeggedController,
)
from robotlab.core.algorithms import AlgorithmContext
from robotlab.core.legged import JointTargets
from robotlab.testing.plugins import (
    PluginContractError,
    check_legged_controller,
    legged_context,
    nominal_legged,
    synthetic_context,
)


def test_hold_pose_follows_the_contract() -> None:
    report = check_legged_controller(HoldPose)
    assert all(c.position_rad == (0.0,) * 6 for c in report.commands)
    assert report.out_of_bounds == ()
    with pytest.raises(ValueError, match="no options"):
        HoldPose(legged_context(nominal_legged()), {"gain": 1})
    with pytest.raises(ValueError, match="legged robot"):
        HoldPose(synthetic_context(), {})


def test_example_lean_feedback_agrees_for_truth_and_imu() -> None:
    gains = {"lean_gains": {"left_ankle_pitch": 1.0, "right_ankle_pitch": -2.0}}
    report = check_legged_controller(PoseHoldLeggedController, gains)
    # Probes: no data, then (truth, sensors-only) pairs at 0, +0.2, -0.3 rad,
    # each with and without a request.
    left = [round(c.position_rad[2], 9) for c in report.commands]
    assert left == [0, 0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, -0.3, -0.3, -0.3, -0.3]
    assert report.commands[-1].position_rad[5] == pytest.approx(0.6)
    assert report.summary["last_lean_rad"] == pytest.approx(-0.3)


def test_example_rejects_unknown_options_and_joints() -> None:
    context = legged_context(nominal_legged())
    with pytest.raises(ValueError, match="unknown options"):
        PoseHoldLeggedController(context, {"gain": 1})
    with pytest.raises(ValueError, match="no actuated joint"):
        PoseHoldLeggedController(context, {"lean_gains": {"tail": 1.0}})
    with pytest.raises(ValueError, match="finite"):
        PoseHoldLeggedController(context, {"lean_gains": {"left_knee": float("inf")}})


def test_walk_command_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        WalkCommand(float("nan"))


class _Bad:
    """Configurable misbehaving controller for the contract kit."""

    calls = 0

    def __init__(self, context: AlgorithmContext, options: Mapping[str, Any]) -> None:
        self.mode = options["mode"]

    def update(
        self, state: LeggedControlState, request: WalkCommand | None, dt_s: float
    ) -> Any:
        _Bad.calls += 1
        if self.mode == "type":
            return [0.0] * 6
        if self.mode == "length":
            return JointTargets((0.0,) * 5)
        if self.mode == "random":
            return JointTargets((float(_Bad.calls),) * 6)
        return JointTargets((2.0,) * 6)

    def summary(self) -> Any:
        return {"bad": object()} if self.mode == "summary" else {}


@pytest.mark.parametrize(
    "mode, message",
    [
        ("type", "not JointTargets"),
        ("length", "needs 6 values"),
        ("random", "different targets"),
        ("summary", "JSON"),
    ],
)
def test_contract_kit_catches_broken_controllers(mode: str, message: str) -> None:
    with pytest.raises(PluginContractError, match=message):
        check_legged_controller(_Bad, {"mode": mode})


def test_contract_kit_reports_out_of_bounds_targets() -> None:
    report = check_legged_controller(_Bad, {"mode": "outside"})
    assert report.out_of_bounds == tuple(range(len(report.commands)))
    assert np.all(np.array(report.commands[0].position_rad) == 2.0)
