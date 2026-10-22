"""Shared synthetic scenarios for scoring estimator plug-ins (no Gazebo).

Every estimator, built-in or yours, can run the same seeded scenarios and
be scored against the known truth::

    from robotlab.testing.scenarios import SCENARIOS, run_scenario

    report = run_scenario(MyFilter, options, SCENARIOS["circle"])
    assert not report.violations({"position_rmse_m": 0.1})

A scenario is a truth trajectory (position, velocity, acceleration,
attitude, body rates as functions of time) plus sensor settings. From it,
``scenario_samples`` synthesizes what the bus would deliver:

- IMU at 200 Hz from t = 0: specific force ``R^T (a - g)`` and body rates,
  in body axes (body-origin mount), plus a constant bias and white noise
  per sample (``imu_noise``: m/s², rad/s). Each sample is the value at its
  timestamp, which the filter holds until the next one; the built-in
  trajectories have constant body-frame IMU values, so holding is exact.
- GPS at 10 Hz from t = 0.1 s (antenna at the body origin): the true
  position plus white noise per axis (``gps_noise_m``), with the true ENU
  velocity plus 0.05 m/s noise; none inside ``gps_outages_s``.
- At equal timestamps the IMU sample comes first.

The estimator never sees the truth: ``run_scenario`` passes only the
samples (through ``plugins.run_estimator``, so the contract is checked on
the way), and scores each estimate against the truth at its timestamp.
Initial conditions are explicit: ``initial_truth(scenario)`` is the state at
t = 0, which a test may translate into the estimator's declared priors.

The data are idealized (no latency, no vibration, a level or yawing body),
so these scenarios catch frame, sign, timing, and observability mistakes,
not tuning for a real vehicle; judge that with Gazebo runs and batches.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..core.algorithms import AlgorithmContext
from ..core.frames import GRAVITY
from ..core.measurements import EstimatedState, GpsReading, ImuReading, Sample
from .plugins import _sample, enu_to_gps, run_estimator

IMU_RATE_HZ = 200
GPS_RATE_HZ = 10
GPS_VELOCITY_NOISE_MPS = 0.05


@dataclass(frozen=True)
class TruthPoint:
    """True state at one time (world ENU, body-to-world quaternion xyzw)."""

    position_m: NDArray[np.float64]
    velocity_mps: NDArray[np.float64]
    acceleration_mps2: NDArray[np.float64]
    quaternion_xyzw: NDArray[np.float64]
    body_rate_rad_s: NDArray[np.float64]


Trajectory = Callable[[float], TruthPoint]


@dataclass(frozen=True)
class EstimatorScenario:
    """One seeded scenario: a truth trajectory and its sensor settings.

    ``settle_s`` excludes the initial convergence from the scores;
    ``gps_outages_s`` lists [start, end) windows without GPS fixes.
    """

    name: str
    description: str
    trajectory: Trajectory
    duration_s: float = 20.0
    settle_s: float = 2.0
    accel_bias_mps2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_bias_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    imu_noise: tuple[float, float] = (0.02, 0.002)  # m/s², rad/s per sample
    gps_noise_m: float = 0.05  # per axis, per fix
    gps_outages_s: tuple[tuple[float, float], ...] = ()
    seed: int = 0  # each built-in scenario has its own


@dataclass(frozen=True)
class ScenarioReport:
    """Scores of one run (errors after ``settle_s``; NIS over the whole run).

    Attitude error is the angle of ``R_true^T R_est``; tilt error is the
    angle between the true and estimated body z axes (yaw-free). Bias
    errors are norms at the final estimate (true minus estimated bias).
    ``mean_nis_per_dof`` is None when the estimator reports no innovations.
    """

    scenario: str
    estimates: int
    scored_estimates: int
    position_rmse_m: float
    max_position_error_m: float
    final_position_error_m: float
    velocity_rmse_mps: float
    attitude_rmse_deg: float
    tilt_rmse_deg: float
    final_accel_bias_error_mps2: float
    final_gyro_bias_error_rad_s: float
    mean_nis_per_dof: float | None
    summary: dict[str, Any] = field(default_factory=dict)

    def violations(self, limits: Mapping[str, float]) -> list[str]:
        """Messages for every metric above its limit (empty if none).

        ``limits`` maps metric names of this report to maxima; an unknown
        name raises ``ValueError``, and a None metric violates its limit.
        """
        names = {f.name for f in fields(self)}
        messages = []
        for name, limit in limits.items():
            if name not in names or name in ("scenario", "summary"):
                raise ValueError(f"{name!r} is not a ScenarioReport metric")
            value = getattr(self, name)
            if value is None or value > limit:
                messages.append(f"{self.scenario}: {name} = {value} > {limit}")
        return messages


# Trajectories ---------------------------------------------------------------------


def _level(yaw_rad: float) -> NDArray[np.float64]:
    return Rotation.from_euler("z", yaw_rad).as_quat()


def hovering(position_m: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> Trajectory:
    """Level, still, facing east."""
    point = TruthPoint(
        np.array(position_m, float), np.zeros(3), np.zeros(3), _level(0.0), np.zeros(3)
    )
    return lambda t: point


def straight(
    velocity_mps: tuple[float, float, float],
    start_m: tuple[float, float, float] = (0.0, 0.0, 1.0),
    yaw_rad: float = 0.0,
) -> Trajectory:
    """Constant velocity, level, fixed heading ``yaw_rad`` (from east)."""
    velocity = np.array(velocity_mps, float)
    start = np.array(start_m, float)
    attitude = _level(yaw_rad)
    return lambda t: TruthPoint(
        start + velocity * t, velocity, np.zeros(3), attitude, np.zeros(3)
    )


def circling(radius_m: float, speed_mps: float, height_m: float = 1.0) -> Trajectory:
    """Counter-clockwise level circle about the origin, nose along the path.

    Starts at (radius, 0) heading north. In body axes the specific force
    (0, v²/r, g) and the yaw rate v/r are constant.
    """
    if radius_m <= 0 or speed_mps <= 0:
        raise ValueError("circle radius and speed must be positive")
    rate = speed_mps / radius_m

    def at(t: float) -> TruthPoint:
        angle = rate * t
        radial = np.array([np.cos(angle), np.sin(angle), 0.0])
        tangent = np.array([-np.sin(angle), np.cos(angle), 0.0])
        return TruthPoint(
            radius_m * radial + [0.0, 0.0, height_m],
            speed_mps * tangent,
            -rate * speed_mps * radial,
            _level(angle + np.pi / 2),
            np.array([0.0, 0.0, rate]),
        )

    return at


SCENARIOS: dict[str, EstimatorScenario] = {
    s.name: s
    for s in (
        EstimatorScenario("hover", "level hover at 1 m", hovering()),
        EstimatorScenario(
            "constant_velocity",
            "straight level flight at 1.58 m/s, heading east",
            straight((1.5, 0.5, 0.0)),
            seed=1,
        ),
        EstimatorScenario(
            "circle",
            "5 m radius level circle at 2 m/s, yawing with the path",
            circling(5.0, 2.0),
            duration_s=30.0,
            seed=2,
        ),
        EstimatorScenario(
            "gps_dropout",
            "constant velocity with no GPS from 8 s to 13 s",
            straight((1.5, 0.5, 0.0)),
            gps_outages_s=((8.0, 13.0),),
            seed=3,
        ),
        EstimatorScenario(
            "biased_imu",
            "hover with constant IMU biases, then no GPS from 30 s to 35 s",
            hovering(),
            duration_s=40.0,
            settle_s=10.0,
            gps_outages_s=((30.0, 35.0),),
            accel_bias_mps2=(0.15, -0.1, 0.2),
            gyro_bias_rad_s=(0.003, -0.002, 0.004),
            seed=4,
        ),
    )
}


def initial_truth(scenario: EstimatorScenario) -> TruthPoint:
    """The true state at t = 0, for explicit estimator initialization."""
    return scenario.trajectory(0.0)


# Samples --------------------------------------------------------------------------


def scenario_samples(scenario: EstimatorScenario) -> list[tuple[str, Sample]]:
    """Seeded IMU and GPS samples for ``scenario`` (see module docs)."""
    rng = np.random.default_rng(scenario.seed)
    accel_bias = np.array(scenario.accel_bias_mps2, float)
    gyro_bias = np.array(scenario.gyro_bias_rad_s, float)
    accel_noise, gyro_noise = scenario.imu_noise
    imu_period_ns = 1_000_000_000 // IMU_RATE_HZ
    per_fix = IMU_RATE_HZ // GPS_RATE_HZ
    samples: list[tuple[str, Sample]] = []
    fixes = 0
    for step in range(round(scenario.duration_s * IMU_RATE_HZ) + 1):
        t_ns = step * imu_period_ns
        truth = scenario.trajectory(t_ns * 1e-9)
        to_body = Rotation.from_quat(truth.quaternion_xyzw).inv()
        imu = ImuReading(
            to_body.apply(truth.acceleration_mps2 - GRAVITY)
            + accel_bias
            + rng.normal(0, accel_noise, 3),
            truth.body_rate_rad_s + gyro_bias + rng.normal(0, gyro_noise, 3),
        )
        samples.append(("imu", _sample("imu", step + 1, t_ns, imu)))
        if step == 0 or step % per_fix or _in_outage(scenario, t_ns * 1e-9):
            continue
        fixes += 1
        samples.append(("gps", _gps_sample(truth, rng, scenario, t_ns, fixes)))
    return samples


def _in_outage(scenario: EstimatorScenario, t_s: float) -> bool:
    return any(start <= t_s < end for start, end in scenario.gps_outages_s)


def _gps_sample(
    truth: TruthPoint,
    rng: np.random.Generator,
    scenario: EstimatorScenario,
    t_ns: int,
    sequence: int,
) -> Sample:
    noisy = truth.position_m + rng.normal(0, scenario.gps_noise_m, 3)
    fix = enu_to_gps((float(noisy[0]), float(noisy[1]), float(noisy[2])))
    velocity = truth.velocity_mps + rng.normal(0, GPS_VELOCITY_NOISE_MPS, 3)
    reading = GpsReading(fix.latitude_deg, fix.longitude_deg, fix.altitude_m, velocity)
    # Consecutive sequence numbers, as the bus assigns them: an outage is a
    # time gap, not lost samples.
    return _sample("gps", sequence, t_ns, reading)


# Scoring --------------------------------------------------------------------------


def run_scenario(
    factory: Callable[[AlgorithmContext, Mapping[str, Any]], Any],
    options: Mapping[str, Any],
    scenario: EstimatorScenario,
) -> ScenarioReport:
    """Run an estimator on ``scenario`` and score it against the truth.

    Raises ``PluginContractError`` if the estimator breaks its contract and
    ``ValueError`` if it publishes no estimate after ``settle_s``.
    """
    run = run_estimator(factory, options, scenario_samples(scenario))
    settle_ns = round(scenario.settle_s * 1e9)
    scored = [s for s in run.states if s.sim_time_ns >= settle_ns]
    if not scored:
        raise ValueError(
            f"{scenario.name}: no estimate after the {scenario.settle_s} s settle time"
        )
    errors = np.array([_errors(s, scenario.trajectory) for s in scored])
    position, velocity, attitude, tilt = errors.T
    final = scored[-1].payload
    nis = [i.nis / i.dof for i in run.innovations]
    return ScenarioReport(
        scenario.name,
        len(run.states),
        len(scored),
        _rms(position),
        float(position.max()),
        float(position[-1]),
        _rms(velocity),
        _rms(attitude),
        _rms(tilt),
        _norm_difference(scenario.accel_bias_mps2, final.accel_bias_mps2),
        _norm_difference(scenario.gyro_bias_rad_s, final.gyro_bias_rad_s),
        float(np.mean(nis)) if nis else None,
        run.summary,
    )


def _errors(estimate: Sample[EstimatedState], trajectory: Trajectory) -> list[float]:
    """Position (m), velocity (m/s), attitude and tilt (deg) error norms."""
    truth = trajectory(estimate.sim_time_ns * 1e-9)
    state = estimate.payload
    true_rotation = Rotation.from_quat(truth.quaternion_xyzw)
    estimated = Rotation.from_quat(state.quaternion_xyzw)
    up_true = true_rotation.apply([0.0, 0.0, 1.0])
    up_estimated = estimated.apply([0.0, 0.0, 1.0])
    return [
        float(np.linalg.norm(state.position_m - truth.position_m)),
        float(np.linalg.norm(state.velocity_mps - truth.velocity_mps)),
        float(np.degrees((true_rotation.inv() * estimated).magnitude())),
        float(np.degrees(np.arccos(np.clip(up_true @ up_estimated, -1.0, 1.0)))),
    ]


def _norm_difference(true: Any, estimated: Any) -> float:
    return float(np.linalg.norm(np.subtract(true, estimated)))


def _rms(values: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean(values**2)))
