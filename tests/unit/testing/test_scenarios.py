"""Scenario generator and scoring (robotlab.testing.scenarios)."""

import dataclasses

import numpy as np
import pytest

from robotlab.core.measurements import EstimatedState, Sample
from robotlab.sensors.geodesy import gps_to_enu
from robotlab.testing.plugins import REFERENCE
from robotlab.testing.scenarios import (
    SCENARIOS,
    circling,
    run_scenario,
    scenario_samples,
)


def _quiet(name):
    """A scenario without noise, for exact checks."""
    return dataclasses.replace(SCENARIOS[name], imu_noise=(0.0, 0.0), gps_noise_m=0.0)


def test_hover_imu_reads_gravity_up_and_gps_the_position():
    samples = scenario_samples(_quiet("hover"))
    imu = [s for k, s in samples if k == "imu"]
    gps = [s for k, s in samples if k == "gps"]
    assert len(imu) == 4001 and len(gps) == 200  # 0..20 s; fixes from 0.1 s
    np.testing.assert_allclose(imu[7].payload.specific_force_mps2, [0, 0, 9.81])
    np.testing.assert_allclose(
        gps_to_enu(gps[3].payload, REFERENCE), [0, 0, 1], atol=1e-3
    )
    assert gps[0].sim_time_ns == 100_000_000


def test_circle_imu_is_constant_in_body_axes():
    """Centripetal acceleration v²/r points to body +y (left) on a CCW circle."""
    scenario = dataclasses.replace(_quiet("circle"), trajectory=circling(5.0, 2.0))
    imu = [s.payload for k, s in scenario_samples(scenario) if k == "imu"]
    for reading in (imu[0], imu[1234], imu[-1]):
        np.testing.assert_allclose(
            reading.specific_force_mps2, [0, 0.8, 9.81], atol=1e-9
        )
        np.testing.assert_allclose(reading.angular_rate_rad_s, [0, 0, 0.4])


def test_outage_removes_fixes_and_keeps_sequences_consecutive():
    samples = scenario_samples(SCENARIOS["gps_dropout"])
    gps = [s for k, s in samples if k == "gps"]
    times = np.array([s.sim_time_ns for s in gps]) * 1e-9
    assert not np.any((times >= 8.0) & (times < 13.0))
    assert [s.sequence for s in gps] == list(range(1, len(gps) + 1))


def test_biases_are_added_to_the_imu():
    scenario = dataclasses.replace(_quiet("biased_imu"))
    imu = next(s.payload for k, s in scenario_samples(scenario) if k == "imu")
    np.testing.assert_allclose(imu.specific_force_mps2, [0.15, -0.1, 10.01])
    np.testing.assert_allclose(imu.angular_rate_rad_s, [0.003, -0.002, 0.004])


class Perfect:
    """Reads the truth from the scenario (only possible in a test) and
    reports it at each IMU sample: scores must be zero."""

    inputs = ("imu",)

    def __init__(self, context, options):
        self.trajectory = options["trajectory"]
        self._latest = None

    def on_sample(self, kind, sample):
        truth = self.trajectory(sample.sim_time_ns * 1e-9)
        state = EstimatedState(
            truth.position_m,
            truth.velocity_mps,
            truth.quaternion_xyzw,
            np.zeros(3),
            np.zeros(3),
            np.eye(15),
        )
        self._latest = Sample(
            "estimates/perfect", 1, 0, sample.sim_time_ns, 0, "world", state
        )

    def latest(self):
        return self._latest

    def drain_innovations(self):
        return []

    def summary(self):
        return {}


def test_scores_are_zero_for_the_truth_and_violations_name_metrics():
    scenario = dataclasses.replace(SCENARIOS["circle"], duration_s=5.0)
    report = run_scenario(Perfect, {"trajectory": scenario.trajectory}, scenario)
    assert report.scored_estimates == 601  # 2 s .. 5 s at 200 Hz
    assert report.position_rmse_m == pytest.approx(0, abs=1e-9)
    assert report.attitude_rmse_deg == pytest.approx(0, abs=1e-5)
    assert report.mean_nis_per_dof is None
    assert report.violations({"position_rmse_m": 0.01}) == []
    assert report.violations({"mean_nis_per_dof": 1.0}) == [
        "circle: mean_nis_per_dof = None > 1.0"
    ]
    with pytest.raises(ValueError, match="not a ScenarioReport metric"):
        report.violations({"positon_rmse_m": 1.0})


def test_estimator_without_estimates_after_settling_fails():
    class Silent(Perfect):
        def latest(self):
            return None

    scenario = dataclasses.replace(SCENARIOS["hover"], duration_s=3.0)
    with pytest.raises(ValueError, match="no estimate after"):
        run_scenario(Silent, {"trajectory": scenario.trajectory}, scenario)
