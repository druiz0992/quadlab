"""Registry of estimator candidates for the scenario suite and the benchmark.

A ``Candidate`` is an estimator plug-in plus how to configure it for a
scenario (declared priors and noise only, never the truth trajectory), the
synthetic sensors it gets, and regression limits per scenario. The
built-in ``CANDIDATES`` are scored by the unit tests
(tests/unit/estimation/test_estimator_scenarios.py) and by
``robotlab-eval-estimators`` (experiments/harness/estimator_benchmark.py).
To score your own filter, build a ``dict[str, Candidate]`` in your
package and pass it as ``--candidates mypkg.bench:CANDIDATES``; add it
here (with limits for every scenario) to make it a regression test.

Limits are about 1.5-2x the verified scores, so they catch regressions
(frames, signs, timing, bias handling), not small tuning changes.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from scipy.spatial.transform import Rotation

from ..estimation.examples.complementary_estimator import ComplementaryEstimator
from ..estimation.filter.estimator import FilterEstimator
from ..experiments.harness.algorithms import gps_imu_ekf_options
from ..simulation.gazebo.config import load_drone_config
from .scenarios import (
    WHEELED_KINDS,
    EstimatorScenario,
    ScenarioReport,
    initial_truth,
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
    limits: dict[str, dict[str, float]] = field(default_factory=dict)
    sensors: tuple[str, ...] = ("imu", "gps")
    consistent: bool = False

    def runs_on(self, scenario: EstimatorScenario) -> bool:
        """Whether the scenario's vehicle has the candidate's sensors
        (wheel encoders need wheels, ESC telemetry rotors)."""
        wheeled = scenario.vehicle in WHEELED_KINDS
        if "wheel_encoders" in self.sensors and not wheeled:
            return False
        return not ("esc" in self.sensors and wheeled)

    def violations(self, report: ScenarioReport) -> list[str] | None:
        """Limit violations of ``report`` (plus the NIS band when
        ``consistent``); None when no limits are declared for its scenario."""
        limits = self.limits.get(report.scenario)
        if limits is None:
            return None
        messages = report.violations(limits)
        nis = report.mean_nis_per_dof
        if self.consistent and (nis is None or not NIS_BAND[0] <= nis <= NIS_BAND[1]):
            messages.append(
                f"{report.scenario}: mean_nis_per_dof = {nis} outside {NIS_BAND}"
            )
        return messages


def ekf_options(scenario: EstimatorScenario) -> dict[str, Any]:
    """Default ``ekf`` section with the start pose and the scenario's GPS
    noise declared (the IMU noise assumptions stay the configured ones), as
    ``filter`` estimator options (what ``gps_imu_ekf`` runs)."""
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


def wheel_odometry_options(scenario: EstimatorScenario) -> dict[str, Any]:
    """Wheel-odometry example with the start position and yaw declared."""
    start = initial_truth(scenario)
    yaw_deg = Rotation.from_quat(start.quaternion_xyzw).as_euler("xyz", degrees=True)[2]
    return dict(
        filter="robotlab.estimation.examples.wheel_odometry_filter:WheelOdometryFilter",
        filter_options=dict(
            initial_position_m=list(start.position_m), initial_yaw_deg=float(yaw_deg)
        ),
        gps_std_m=[scenario.gps_noise_m] * 3,
    )


def complementary_options(scenario: EstimatorScenario) -> dict[str, Any]:
    """The complementary estimator only takes a start position (it starts level, facing east)."""
    return dict(initial_position_m=list(initial_truth(scenario).position_m))


STEADY = dict(position_rmse_m=0.07, velocity_rmse_mps=0.1, tilt_rmse_deg=1.0)
CANDIDATES = {
    "gps_imu_ekf": Candidate(
        FilterEstimator,
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
    "complementary": Candidate(
        ComplementaryEstimator,
        complementary_options,
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
    # Driven by wheel encoders, corrected by GPS. Its sideways speed is
    # observable only through the steering angle, so quantized encoder
    # speeds make the velocity noisy (about 0.6 m/s RMS).
    "wheel_odometry": Candidate(
        FilterEstimator,
        wheel_odometry_options,
        {
            "car_circle": dict(
                position_rmse_m=0.04,
                max_position_error_m=0.1,
                velocity_rmse_mps=1.0,
                attitude_rmse_deg=0.2,
            ),
        },
        sensors=("wheel_encoders", "gps"),
        consistent=True,
    ),
}
