"""ATE and RPE (robotlab.evaluation.trajectory)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.core.measurements import EstimatedState
from robotlab.evaluation.estimation import EstimationMetrics
from robotlab.evaluation.trajectory import (
    absolute_trajectory_error,
    align,
    relative_pose_error,
)
from robotlab.evaluation.truth import TruthHistory

TIMES = np.linspace(0.0, 20.0, 201)
# A wandering 3D path with changing heading.
TRUE_M = np.column_stack(
    [5 * np.sin(0.3 * TIMES), 3 * np.sin(0.5 * TIMES), 1 + 0.5 * np.sin(0.2 * TIMES)]
)
TRUE_Q = Rotation.from_euler(
    "ZYX", np.column_stack([0.4 * TIMES, 0.1 * np.sin(TIMES), 0.05 * np.cos(TIMES)])
).as_quat()


def rigidly_moved(rotation, translation, scale=1.0):
    return scale * rotation.apply(TRUE_M) + translation


def test_alignment_recovers_rigid_transforms():
    yaw = Rotation.from_euler("z", 0.7)
    estimated = rigidly_moved(yaw.inv(), [3.0, -2.0, 0.5])
    result = align(estimated, TRUE_M, "yaw")
    np.testing.assert_allclose(result.apply(estimated), TRUE_M, atol=1e-9)
    assert result.summary()["yaw_deg"] == pytest.approx(np.degrees(0.7))
    tilted = Rotation.from_euler("xyz", [0.2, -0.3, 1.0])
    estimated = rigidly_moved(tilted, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        align(estimated, TRUE_M, "se3").apply(estimated), TRUE_M, atol=1e-9
    )
    estimated = rigidly_moved(tilted, [1.0, 2.0, 3.0], scale=1.7)
    sim3 = align(estimated, TRUE_M, "sim3")
    assert sim3.scale == pytest.approx(1 / 1.7)
    np.testing.assert_allclose(sim3.apply(estimated), TRUE_M, atol=1e-9)


def test_yaw_alignment_cannot_remove_tilt():
    tilted = Rotation.from_euler("x", 0.2)
    estimated = rigidly_moved(tilted, [0.0, 0.0, 0.0])
    yaw_ate = absolute_trajectory_error(estimated, TRUE_M, "yaw")["position_m"]["rmse"]
    se3_ate = absolute_trajectory_error(estimated, TRUE_M, "se3")["position_m"]["rmse"]
    assert yaw_ate > 0.1 and se3_ate < 1e-9


def test_ate_rotation_uses_the_alignment():
    yaw = Rotation.from_euler("z", -0.5)
    estimated_q = (yaw * Rotation.from_quat(TRUE_Q)).as_quat()
    result = absolute_trajectory_error(
        rigidly_moved(yaw, [1.0, 0.0, 0.0]), TRUE_M, "yaw", estimated_q, TRUE_Q
    )
    assert result["position_m"]["rmse"] < 1e-9
    assert result["rotation_deg"]["max"] < 1e-6
    raw = absolute_trajectory_error(
        rigidly_moved(yaw, [1.0, 0.0, 0.0]), TRUE_M, "none", estimated_q, TRUE_Q
    )
    assert raw["rotation_deg"]["mean"] == pytest.approx(np.degrees(0.5))


def test_rpe_ignores_a_global_offset_and_measures_drift():
    yaw = Rotation.from_euler("z", 1.1)
    estimated_q = (yaw * Rotation.from_quat(TRUE_Q)).as_quat()
    offset = relative_pose_error(
        TIMES,
        rigidly_moved(yaw, [4.0, 4.0, 0.0]),
        estimated_q,
        TRUE_M,
        TRUE_Q,
        segment_s=2.0,
    )
    assert offset["count"] == 181  # starts with a sample 2 s later
    assert offset["translation_m"]["max"] < 1e-9
    assert offset["rotation_deg"]["max"] < 1e-6
    # 0.1 m/s drift along world x, attitude exact: 0.1 m per second.
    drifting = TRUE_M + np.outer(TIMES, [0.1, 0.0, 0.0])
    one_second = relative_pose_error(
        TIMES, drifting, TRUE_Q, TRUE_M, TRUE_Q, segment_s=1.0
    )
    assert one_second["translation_m"]["mean"] == pytest.approx(0.1)


def test_rpe_by_distance_reports_drift_percent():
    times = np.linspace(0.0, 30.0, 301)
    true = np.column_stack([times, np.zeros_like(times), np.ones_like(times)])  # 1 m/s
    level = np.tile([0.0, 0.0, 0.0, 1.0], (len(times), 1))
    estimated = true * [1.05, 1.0, 1.0]  # 5 % scale error along the path
    result = relative_pose_error(times, estimated, level, true, level, segment_m=10.0)
    assert result["drift_pct"]["mean"] == pytest.approx(5.0)
    too_long = relative_pose_error(
        times, estimated, level, true, level, segment_m=100.0
    )
    assert too_long == dict(segment=100.0, count=0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="alignment must be"):
        align(TRUE_M, TRUE_M, "affine")
    with pytest.raises(ValueError, match="match in shape"):
        align(TRUE_M[:5], TRUE_M, "yaw")
    with pytest.raises(ValueError, match="exactly one"):
        relative_pose_error(TIMES, TRUE_M, TRUE_Q, TRUE_M, TRUE_Q)
    with pytest.raises(ValueError, match="increase"):
        relative_pose_error(TIMES[::-1], TRUE_M, TRUE_Q, TRUE_M, TRUE_Q, segment_s=1.0)


def test_estimation_metrics_report_drift():
    truth = TruthHistory()
    metrics = EstimationMetrics(
        truth, rate_hz=1000.0, rpe_segments_s=(1.0,), rpe_segments_m=()
    )
    for index, t in enumerate(TIMES):
        stamp = round(t * 1e9)
        truth.add(stamp, TRUE_M[index], np.zeros(3), TRUE_Q[index])
        estimate = EstimatedState(
            position_m=TRUE_M[index] + [0.05 * t, 0.0, 0.0],
            quaternion_xyzw=TRUE_Q[index],
        )
        metrics.add_estimate(stamp, estimate)
    metrics.process()
    trajectory = metrics.summary()["trajectory"]
    assert set(trajectory["ate"]) == {"none", "yaw"}
    assert trajectory["rpe"]["1s"]["translation_m"]["mean"] == pytest.approx(0.05)
    # Attitude-free estimates: ATE without a rotation part, no RPE.
    metrics = EstimationMetrics(truth, rate_hz=1000.0)
    for index, t in enumerate(TIMES):
        metrics.add_estimate(round(t * 1e9), EstimatedState(position_m=TRUE_M[index]))
    metrics.process()
    trajectory = metrics.summary()["trajectory"]
    assert "rpe" not in trajectory and "rotation_deg" not in trajectory["ate"]["none"]
