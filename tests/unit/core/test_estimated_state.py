"""Partial estimates: EstimatedState and every consumer of it."""

import dataclasses
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.core.measurements import EstimatedState, ImuReading, Sample
from robotlab.evaluation.estimation import EstimationMetrics
from robotlab.evaluation.truth import TruthHistory
from robotlab.experiments.algorithms import EstimateStateSource
from robotlab.experiments.replay_log import estimate_record
from robotlab.testing.scenarios import SCENARIOS, run_scenario

LEVEL = [0.0, 0.0, 0.0, 1.0]


def test_blocks_and_covariance_size_follow_what_is_present():
    full = EstimatedState(
        np.zeros(3), np.zeros(3), LEVEL, np.zeros(3), np.zeros(3), np.eye(15)
    )
    assert full.blocks == (
        "position_m",
        "velocity_mps",
        "quaternion_xyzw",
        "accel_bias_mps2",
        "gyro_bias_rad_s",
    )
    attitude = EstimatedState(
        quaternion_xyzw=LEVEL, gyro_bias_rad_s=np.zeros(3), covariance=2 * np.eye(6)
    )
    assert attitude.blocks == ("quaternion_xyzw", "gyro_bias_rad_s")
    np.testing.assert_array_equal(
        attitude.covariance_block("gyro_bias_rad_s"), 2 * np.eye(3)
    )
    assert attitude.covariance_block("position_m") is None
    assert EstimatedState(position_m=np.zeros(3)).covariance is None


def test_invalid_partial_estimates_raise():
    with pytest.raises(ValueError, match="position or an attitude"):
        EstimatedState(velocity_mps=np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        EstimatedState(position_m=np.zeros(3), covariance=np.eye(15))


def test_metrics_score_only_present_blocks():
    truth = TruthHistory()
    yaw = Rotation.from_euler("z", 0.1)
    for step in range(3):
        truth.add(step * 10_000_000, [1.0, 2.0, 3.0], np.zeros(3), yaw.as_quat())
    metrics = EstimationMetrics(truth, rate_hz=1000.0)
    # Attitude only, 0.1 rad yaw error, 0.1 rad attitude sigma: NEES = 1.
    metrics.add_estimate(
        0, EstimatedState(quaternion_xyzw=LEVEL, covariance=0.01 * np.eye(3))
    )
    # Position only, no covariance: scored, no NEES.
    metrics.add_estimate(10_000_000, EstimatedState(position_m=[1.0, 2.0, 3.5]))
    scored = metrics.process()
    assert scored[0]["nees"] == pytest.approx(1.0)
    assert scored[0]["attitude_deg"] == pytest.approx(math.degrees(0.1))
    assert "position_m" not in scored[0] and "nees" not in scored[1]
    summary = metrics.summary()
    assert summary["samples"] == 2
    assert summary["errors"]["position_m"]["count"] == 1
    assert summary["errors"]["attitude_deg"]["count"] == 1
    assert "velocity_mps" not in summary["errors"]
    assert summary["nees"]["dof"] == 3 and summary["nees"]["count"] == 1
    assert summary["position_rmse_axes_m"] == pytest.approx([0.0, 0.0, 0.5])


def test_nees_is_split_by_dof_when_the_blocks_change():
    truth = TruthHistory()
    for step in range(3):
        truth.add(step * 10_000_000, np.zeros(3), np.zeros(3), LEVEL)
    metrics = EstimationMetrics(truth, rate_hz=1000.0)
    metrics.add_estimate(0, EstimatedState(quaternion_xyzw=LEVEL, covariance=np.eye(3)))
    metrics.add_estimate(
        10_000_000,
        EstimatedState(np.zeros(3), np.zeros(3), LEVEL, covariance=np.eye(9)),
    )
    metrics.process()
    by_dof = metrics.summary()["nees"]["by_dof"]
    assert set(by_dof) == {"3", "9"}


def test_replay_record_writes_null_for_absent_blocks():
    sample = Sample(
        "estimates/ahrs", 1, 0, 5, 0, "world", EstimatedState(quaternion_xyzw=LEVEL)
    )
    record = estimate_record(sample)
    assert record["quaternion_xyzw"] == LEVEL
    assert record["position_m"] is None and record["covariance"] is None


def test_estimate_driven_control_needs_a_full_pose():
    source = EstimateStateSource(LEVEL, max_age_s=0.1)
    imu = Sample(
        "sensors/imu", 1, 0, 0, 0, "drone/imu", ImuReading([0, 0, 9.81], [0, 0, 0.2])
    )
    source.on_delivered("imu", imu)
    pose = EstimatedState(np.zeros(3), np.zeros(3), LEVEL)
    state = source.state(0, Sample("e", 1, 0, 0, 0, "world", pose))
    np.testing.assert_allclose(state.body_rates_rad_s, [0, 0, 0.2])  # no bias estimate
    with pytest.raises(ValueError, match="needs position, velocity, and attitude"):
        source.state(
            0, Sample("e", 1, 0, 0, 0, "world", EstimatedState(quaternion_xyzw=LEVEL))
        )


class GyroAhrs:
    """Attitude-only filter: gyro integration with a slow accelerometer
    tilt correction; publishes quaternion and a fixed 1° attitude sigma."""

    inputs = ("imu",)

    def __init__(self, context, options):
        self._attitude = Rotation.identity()
        self._last_ns = None
        self._latest = None

    def on_sample(self, kind, sample):
        reading = sample.payload
        if self._last_ns is not None:
            dt = (sample.sim_time_ns - self._last_ns) * 1e-9
            self._attitude = self._attitude * Rotation.from_rotvec(
                reading.angular_rate_rad_s * dt
            )
            up = self._attitude.apply(reading.specific_force_mps2)
            axis = np.cross(up / np.linalg.norm(up), [0.0, 0.0, 1.0])
            self._attitude = Rotation.from_rotvec(0.5 * dt * axis) * self._attitude
        self._last_ns = sample.sim_time_ns
        state = EstimatedState(
            quaternion_xyzw=self._attitude.as_quat(),
            covariance=np.radians(1.0) ** 2 * np.eye(3),
        )
        self._latest = Sample(
            "estimates/ahrs", 1, 0, sample.sim_time_ns, 0, "world", state
        )

    def latest(self):
        return self._latest

    def drain_innovations(self):
        return []

    def summary(self):
        return {}


def test_attitude_only_estimator_runs_the_scenario_suite():
    scenario = dataclasses.replace(SCENARIOS["tumbling"], duration_s=10.0)
    report = run_scenario(GyroAhrs, {}, scenario, sensors=("imu",))
    assert report.position_rmse_m is None and report.velocity_rmse_mps is None
    assert report.final_accel_bias_error_mps2 is None
    assert report.tilt_rmse_deg < 1.0 and report.attitude_rmse_deg < 1.0
    assert report.violations({"position_rmse_m": 1.0}) == [
        "tumbling: position_rmse_m = None > 1.0"
    ]
