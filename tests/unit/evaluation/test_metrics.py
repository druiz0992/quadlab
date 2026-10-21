"""Estimation and control scoring against truth."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronelab.control.references import ReferencePoint
from dronelab.core.measurements import EstimatedState
from dronelab.evaluation.control import ControlMetrics, step_response
from dronelab.evaluation.estimation import EstimationMetrics, chi2_consistency
from dronelab.evaluation.truth import TruthHistory

MS = 1_000_000


def estimate(position, quaternion=(0, 0, 0, 1), sigma=1.0, velocity=(0, 0, 0)):
    return EstimatedState(
        np.asarray(position, float),
        np.asarray(velocity, float),
        np.asarray(quaternion, float),
        np.zeros(3),
        np.zeros(3),
        np.eye(15) * sigma**2,
    )


def test_truth_history_interpolates_position_and_attitude():
    truth = TruthHistory()
    q1 = Rotation.from_euler("z", 90, degrees=True).as_quat()
    truth.add(0, [0, 0, 0], [1, 0, 0], [0, 0, 0, 1])
    truth.add(10 * MS, [1, 0, 0], [1, 0, 0], q1)
    mid = truth.at(5 * MS)
    np.testing.assert_allclose(mid.position_m, [0.5, 0, 0])
    assert Rotation.from_quat(mid.quaternion_xyzw).as_euler("ZYX", degrees=True)[
        0
    ] == pytest.approx(45)
    assert truth.at(11 * MS) is None


def test_estimation_errors_attitude_and_nees():
    truth = TruthHistory()
    for i in range(101):
        truth.add(i * 10 * MS, [i * 0.01, 0, 1], [1, 0, 0], [0, 0, 0, 1])
    metrics = EstimationMetrics(truth, rate_hz=1000.0)
    yaw = Rotation.from_euler("z", 2, degrees=True).as_quat()
    # Estimate 0.3 m east of truth, 2° yaw off, unit covariance.
    for i in range(0, 100, 10):
        metrics.add_estimate(
            i * 10 * MS + 5 * MS,
            estimate([i * 0.01 + 0.005 + 0.3, 0, 1], yaw, velocity=(1, 0, 0)),
        )
    scored = metrics.process()
    assert len(scored) == 10
    summary = metrics.summary()
    assert summary["errors"]["position_m"]["rmse"] == pytest.approx(0.3)
    assert summary["errors"]["yaw_deg"]["max"] == pytest.approx(2.0)
    # NEES = 0.3² + (2° in rad)² with unit covariance.
    assert summary["nees"]["mean"] == pytest.approx(0.09 + np.radians(2) ** 2)
    assert summary["nees"]["dof"] == 9


def test_estimates_wait_for_truth_and_decimate():
    truth = TruthHistory()
    metrics = EstimationMetrics(truth, rate_hz=10.0, start_s=0.05)
    for i in range(30):
        metrics.add_estimate(i * 10 * MS, estimate([0, 0, 1]))
    assert metrics.process() == []  # no truth yet
    for i in range(31):
        truth.add(i * 10 * MS, [0, 0, 1], [0, 0, 0], [0, 0, 0, 1])
    # Skip the first 50 ms, then one per 100 ms: 50, 150, 250 ms.
    assert [s["sim_time_s"] for s in metrics.process()] == pytest.approx(
        [0.05, 0.15, 0.25]
    )


def test_chi2_consistency_of_true_chi2_samples():
    rng = np.random.default_rng(0)
    values = list(np.sum(rng.normal(size=(20000, 3)) ** 2, axis=1))
    result = chi2_consistency(values, 3)
    assert result["mean"] == pytest.approx(3, rel=0.03)
    assert result["fraction_inside_95"] == pytest.approx(0.95, abs=0.01)


def test_step_response_of_second_order_system():
    zeta, wn = 0.5, 2.0
    t = np.linspace(0, 20, 20001)
    wd = wn * np.sqrt(1 - zeta**2)
    phi = np.arccos(zeta)
    y = 1 - np.exp(-zeta * wn * t) / np.sqrt(1 - zeta**2) * np.sin(wd * t + phi)
    result = step_response(t + 1.0, y, step_time_s=1.0, initial=0.0, final=1.0)
    assert result["overshoot_pct"] == pytest.approx(
        100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)), rel=1e-3
    )
    # 5 cm band (> 2 %): settles when the envelope e^{-ζωn t}/√(1−ζ²) < 0.05.
    assert (
        result["settling_time_s"]
        < -np.log(0.05 * np.sqrt(1 - zeta**2)) / (zeta * wn) + 0.01
    )
    assert result["steady_state_error_m"] == pytest.approx(0, abs=1e-3)
    assert 0.5 < result["rise_time_s"] < 1.0


def test_control_metrics_tracking_and_effort():
    metrics = ControlMetrics(weight_n=10.0)
    reference = ReferencePoint(np.array([1.0, 0, 1]), np.zeros(3), np.zeros(3), 0.0)
    for i in range(10):
        metrics.add_state(i * 0.1, [1.0, 0.1, 1.0], [0, 0, 0], [0, 0, 0, 1], reference)
        metrics.add_command(i * 0.1, [2.5, 2.5, 2.5, 2.5 + (i % 2)], False)
    summary = metrics.summary()
    assert summary["tracking"]["horizontal_m"]["rmse"] == pytest.approx(0.1)
    assert summary["effort"]["mean_thrust_to_weight"] == pytest.approx(1.05)
    assert summary["effort"]["mean_rotor_slew_n_per_s"] == pytest.approx(2.5)
    assert summary["saturated_fraction"] == 0
