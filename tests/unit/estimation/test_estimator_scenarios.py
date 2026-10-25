"""Shared scenario suite: every estimator in CANDIDATES runs every scenario.

To test your own filter, add a ``Candidate`` with its options builder, the
sensors it should get (any streams of ``sensors/bus.py``; the context holds
exactly these), and limits for each scenario in
``robotlab.testing.scenarios.SCENARIOS``; a new scenario without limits
fails until every candidate declares them. Limits are about 1.5-2x the
verified scores, so they catch regressions (frames, signs, timing, bias
handling), not small tuning changes.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

import pytest
from scipy.spatial.transform import Rotation

from robotlab.estimation.inertial.estimator import InertialEstimator
from robotlab.estimation.template import TemplateEstimator
from robotlab.experiments.harness.algorithms import gps_imu_ekf_options
from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.testing.scenarios import (
    SCENARIOS,
    EstimatorScenario,
    ScenarioReport,
    initial_truth,
    run_scenario,
)

# Mean NIS per degree of freedom of a consistent filter is about 1.
NIS_BAND = (0.7, 1.3)


@dataclass(frozen=True)
class Candidate:
    """An estimator under test.

    ``options`` builds its options from a scenario (declared priors and
    noise only, never the truth trajectory); ``limits`` maps scenario name
    to ``ScenarioReport`` metric maxima; ``sensors`` selects the synthetic
    streams (``wheel_encoders`` only exist on wheeled scenarios and ``esc``
    only on multirotor ones); ``consistent`` requires the mean NIS per dof
    inside ``NIS_BAND``.
    """

    factory: Callable[..., Any]
    options: Callable[[EstimatorScenario], Mapping[str, Any]]
    limits: dict[str, dict[str, float]]
    sensors: tuple[str, ...] = ("imu", "gps")
    consistent: bool = False


def ekf_options(scenario: EstimatorScenario) -> dict[str, Any]:
    """Default ``ekf`` section with the start pose and the scenario's GPS
    noise declared (the IMU noise assumptions stay the configured ones), as
    ``inertial`` estimator options (what ``gps_imu_ekf`` runs)."""
    start = initial_truth(scenario)
    return gps_imu_ekf_options(
        dict(
            load_drone_config().data["ekf"],
            initial_position_m=list(start.position_m),
            initial_velocity_mps=list(start.velocity_mps),
            initial_rotation_deg=list(
                Rotation.from_quat(start.quaternion_xyzw).as_euler("xyz", degrees=True)
            ),
            gps_std_m=[scenario.gps_noise_m] * 3,
        )
    )


def template_options(scenario: EstimatorScenario) -> dict[str, Any]:
    """The template only takes a start position (it starts level, facing east)."""
    return dict(initial_position_m=list(initial_truth(scenario).position_m))


STEADY = dict(position_rmse_m=0.07, velocity_rmse_mps=0.1, tilt_rmse_deg=1.0)
CANDIDATES = {
    "gps_imu_ekf": Candidate(
        InertialEstimator,
        ekf_options,
        {
            "hover": STEADY,
            "constant_velocity": STEADY,
            # Yaw is weakly observable from 0.8 m/s² centripetal acceleration.
            "circle": dict(STEADY, attitude_rmse_deg=7.0),
            "figure_eight": dict(STEADY, attitude_rmse_deg=4.0),
            "aggressive": dict(STEADY, attitude_rmse_deg=3.0),
            # Stationary: world yaw is unobservable without a magnetometer.
            "tumbling": dict(STEADY, attitude_rmse_deg=20.0),
            "gps_dropout": dict(
                STEADY,
                position_rmse_m=0.3,
                max_position_error_m=1.0,
                final_position_error_m=0.1,
            ),
            # Horizontal accelerometer bias is indistinguishable from tilt in
            # a hover (0.18 m/s² ≈ 1°), so the tilt limit is wider.
            "biased_imu": dict(
                STEADY,
                position_rmse_m=0.15,
                tilt_rmse_deg=2.0,
                max_position_error_m=0.7,
            ),
            "car_circle": dict(STEADY, attitude_rmse_deg=3.0),
        },
        consistent=True,
    ),
    # Velocity from GPS differences (noisy), no bias estimation, fixed
    # covariance and no innovations: looser limits, no NIS check.
    "template": Candidate(
        TemplateEstimator,
        template_options,
        {
            "hover": dict(position_rmse_m=0.15, velocity_rmse_mps=2.0),
            "constant_velocity": dict(position_rmse_m=0.15, velocity_rmse_mps=2.0),
            "circle": dict(position_rmse_m=0.15, tilt_rmse_deg=6.0),
            # Its tilt correction assumes the accelerometer measures gravity,
            # which banked flight violates.
            "figure_eight": dict(position_rmse_m=0.2, tilt_rmse_deg=8.0),
            "aggressive": dict(position_rmse_m=0.25, tilt_rmse_deg=13.0),
            "tumbling": dict(position_rmse_m=0.15, attitude_rmse_deg=0.3),
            # Coasts on a velocity from two noisy fixes: large outage drift.
            "gps_dropout": dict(max_position_error_m=12.0),
            "biased_imu": dict(max_position_error_m=8.0, tilt_rmse_deg=2.0),
            # Accelerometer tilt correction mistakes 2 m/s² centripetal
            # acceleration for tilt.
            "car_circle": dict(position_rmse_m=0.2, tilt_rmse_deg=15.0),
        },
    ),
}


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
    assert (
        scenario_name in candidate.limits
    ), f"{candidate_name} declares no limits for scenario {scenario_name!r}"
    report = report_for(candidate_name, scenario_name)
    assert not report.violations(candidate.limits[scenario_name])
    if candidate.consistent:
        assert report.mean_nis_per_dof is not None
        assert NIS_BAND[0] <= report.mean_nis_per_dof <= NIS_BAND[1]


def test_bias_learning_shows_in_the_outage():
    """The EKF learns the biases before the outage and coasts far better
    than the template, which does not estimate them."""
    ekf = report_for("gps_imu_ekf", "biased_imu")
    template = report_for("template", "biased_imu")
    assert ekf.max_position_error_m < 0.1 * template.max_position_error_m
