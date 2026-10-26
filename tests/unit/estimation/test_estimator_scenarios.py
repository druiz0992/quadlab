"""Shared scenario suite: every estimator in CANDIDATES runs every scenario.

The candidates (estimator, options builder, sensors, limits) are in
``robotlab.testing.candidates``; to test your own filter add a
``Candidate`` there with limits for each scenario in
``robotlab.testing.scenarios.SCENARIOS`` (a new scenario without limits
fails until every candidate declares them), or score it without a test
via ``robotlab-eval-estimators --candidates``. Scenarios a candidate's
sensors cannot run on are skipped.
"""

from functools import cache

import pytest

from robotlab.testing.candidates import CANDIDATES
from robotlab.testing.scenarios import SCENARIOS, ScenarioReport, run_scenario


@cache
def report_for(candidate_name: str, scenario_name: str) -> ScenarioReport:
    """One run per (candidate, scenario), shared by the tests below."""
    candidate = CANDIDATES[candidate_name]
    scenario = SCENARIOS[scenario_name]
    return run_scenario(
        candidate.factory, candidate.options(scenario), scenario, candidate.sensors
    )


@pytest.mark.parametrize("scenario_name", sorted(SCENARIOS))
@pytest.mark.parametrize("candidate_name", sorted(CANDIDATES))
def test_estimator_meets_its_scenario_limits(candidate_name, scenario_name):
    candidate = CANDIDATES[candidate_name]
    scenario = SCENARIOS[scenario_name]
    if not candidate.runs_on(scenario):
        pytest.skip(f"{candidate_name} needs sensors {scenario.vehicle} lacks")
    assert (
        scenario_name in candidate.limits
    ), f"{candidate_name} declares no limits for scenario {scenario_name!r}"
    assert not candidate.violations(report_for(candidate_name, scenario_name))


def test_bias_learning_shows_in_the_outage():
    """The EKF learns the biases before the outage and coasts far better
    than the complementary estimator, which does not estimate them."""
    ekf = report_for("gps_imu_ekf", "biased_imu")
    complementary = report_for("complementary", "biased_imu")
    assert ekf.max_position_error_m < 0.1 * complementary.max_position_error_m
