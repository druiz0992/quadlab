"""Shared synthetic scenarios for scoring estimator plug-ins (no Gazebo).

Every estimator, built-in or yours, can run the same seeded scenarios with
the sensors it chooses and be scored against the known truth::

    from robotlab.testing.scenarios import SCENARIOS, run_scenario

    report = run_scenario(
        MyFilter, options, SCENARIOS["circle"],
        sensors=("imu", "gps", "magnetometer", "baro"),
    )
    assert not report.violations({"position_rmse_m": 0.1})

A scenario is a truth trajectory (``testing/trajectories.py``), a vehicle
(``multirotor`` or a wheeled ``rover``/``car``/``tractor``), a ray-cast
scene, and sensor settings. ``scenario_samples`` synthesizes bus samples for
any selection of the streams in ``sensors/bus.py`` (see
``testing/synthetic_sensors.py`` for how each is made):

- IMU at 200 Hz from t = 0 and GPS at 10 Hz from 0.1 s are set by the
  scenario's own fields (``imu_noise``, biases, ``gps_noise_m``,
  ``gps_outages_s``); IMU values are those at the sample time (as Gazebo
  reports them), held by the filter until the next one.
- Multirotor scenarios that turn are flown (``trajectories.flown``): the
  body banks so thrust supplies the acceleration, as a real multirotor
  must; ``car_circle`` is level (the tyres supply the centripetal force).
- Every other sensor follows the drone configuration (same keys and
  defaults as ``configs/``: mounts, rates, ranges, noise), with
  ``SENSOR_NOISE`` layered on so none is noise-free by default, then the
  scenario's ``settings`` overrides.
- Each kind samples at k · period on simulation time; equal timestamps are
  delivered in bus priority order (IMU first). Sequence numbers are
  consecutive per stream, as the bus assigns them.

The context the estimator receives is built as in a Gazebo run
(``experiments.algorithms.build_context``) from those settings, with only
the selected sensors. The estimator never sees the truth: ``run_scenario``
delivers samples through ``plugins.run_estimator`` (contract checks on the
way) and scores each estimate against the truth at its timestamp. Initial
conditions are explicit: ``initial_truth(scenario)`` is the state at t = 0,
which a test may translate into declared priors.

The data are idealized (no latency, vibration, banking, or wind), so these
scenarios catch frame, sign, timing, and sensor-fusion mistakes, not tuning
for a real vehicle; judge that with Gazebo runs and batches.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..core.algorithms import AlgorithmContext, DebugLog
from ..core.measurements import EstimatedState, Sample
from .plugins import REFERENCE, _sample, run_estimator
from .synthetic_scene import DEFAULT_SCENE, SyntheticScene
from .synthetic_sensors import (
    GPS_RATE_HZ,
    IMU_RATE_HZ,
    BUILDERS,
    SynthesisInputs,
    build_synthesizer,
)
from .trajectories import (
    Harmonic,
    HarmonicPath,
    Trajectory,
    TruthPoint,
    circling,
    figure_eight,
    flown,
    hovering,
    straight,
    tumbling,
)

__all__ = [
    "EstimatorScenario",
    "GPS_RATE_HZ",
    "IMU_RATE_HZ",
    "SCENARIOS",
    "SENSOR_KINDS",
    "SENSOR_NOISE",
    "ScenarioReport",
    "Trajectory",
    "TruthPoint",
    "circling",
    "figure_eight",
    "flown",
    "hovering",
    "initial_truth",
    "run_scenario",
    "scenario_context",
    "scenario_samples",
    "scenario_settings",
    "straight",
    "tumbling",
]

SENSOR_KINDS = tuple(BUILDERS)
DEFAULT_SENSORS = ("imu", "gps")
WHEELED_KINDS = ("rover", "car", "tractor")

# Modest noise for sensors whose configured default is ideal (config keys).
SENSOR_NOISE: dict[str, Any] = {
    "sensors": {
        "baro": {"pressure_pa": {"noise_stddev": 3.0}},
        "magnetometer": {"field_ut": {"noise_stddev": [0.3, 0.3, 0.3]}},
        "rangefinder": {"range_m": {"noise_stddev": 0.02}},
        "lidar": {"range_m": {"noise_stddev": 0.02}},
        "camera": {"intensity": {"noise_stddev": 0.01}},
        "optical_flow": {"noise_std_rad_s": 0.01},
        "airspeed": {"noise_std_pa": 1.0},
        "battery_monitor": {"voltage_noise_std_v": 0.02, "current_noise_std_a": 0.1},
        "wheel_encoders": {"steering_noise_std_rad": 0.002},
    },
    "motors": {"esc": {"telemetry": {"noise_std_rad_s": 5.0}}},
}


@dataclass(frozen=True)
class EstimatorScenario:
    """One seeded scenario: truth, vehicle, scene, and sensor settings.

    ``settle_s`` excludes the initial convergence from the scores;
    ``gps_outages_s`` lists [start, end) windows without GPS (and GPS
    heading) epochs; ``settings`` overrides configuration keys (as in a
    ``configs/`` layer) after ``SENSOR_NOISE``.
    """

    name: str
    description: str
    trajectory: Trajectory
    duration_s: float = 20.0
    settle_s: float = 2.0
    vehicle: str = "multirotor"
    scene: SyntheticScene = DEFAULT_SCENE
    accel_bias_mps2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_bias_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    imu_noise: tuple[float, float] = (0.02, 0.002)  # m/s², rad/s per sample
    gps_noise_m: float = 0.05  # per axis, per fix
    gps_outages_s: tuple[tuple[float, float], ...] = ()
    battery_current_a: float = 10.0
    settings: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0  # each built-in scenario has its own


@dataclass(frozen=True)
class ScenarioReport:
    """Scores of one run (errors after ``settle_s``; NIS over the whole run).

    Attitude error is the angle of ``R_true^T R_est``; tilt error is the
    angle between the true and estimated world-up direction in body axes
    (``R^T e_z``), which no world-yaw error changes, even on a tilted body. Bias
    errors are norms at the final estimate (true minus estimated bias).
    A metric is None when the estimates lack its block (a partial
    ``EstimatedState``), and ``mean_nis_per_dof`` when the estimator
    reports no innovations; a limit on a None metric is a violation.
    """

    scenario: str
    estimates: int
    scored_estimates: int
    position_rmse_m: float | None
    max_position_error_m: float | None
    final_position_error_m: float | None
    velocity_rmse_mps: float | None
    attitude_rmse_deg: float | None
    tilt_rmse_deg: float | None
    final_accel_bias_error_mps2: float | None
    final_gyro_bias_error_rad_s: float | None
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


# Counter-clockwise 5 m circle at 2 m/s (0.4 rad/s), starting at (5, 0, 1).
BANKED_CIRCLE = HarmonicPath(
    (0.0, 0.0, 1.0),
    harmonics=(Harmonic(0, 5.0, 0.4, np.pi / 2), Harmonic(1, 5.0, 0.4)),
)

# A car's body origin sits at tyre radius minus hub height above flat
# ground (0.30 + 0.08 m, ignoring suspension sag); a unit test checks it.
CAR_BODY_HEIGHT_M = 0.38

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
            "5 m radius circle at 2 m/s, banked 4.7°, nose along the path",
            flown(BANKED_CIRCLE, yaw_along_path=True),
            duration_s=30.0,
            seed=2,
        ),
        EstimatorScenario(
            "figure_eight",
            "12 m × 6 m figure eight in 16 s, climbing ±1 m, nose along the path",
            flown(figure_eight(6.0, 16.0, climb_m=1.0), yaw_along_path=True),
            duration_s=40.0,
            seed=6,
        ),
        EstimatorScenario(
            "aggressive",
            "8 m × 4 m figure eight in 7 s (up to 36° tilt), yawing at 0.8 rad/s",
            flown(figure_eight(4.0, 7.0, climb_m=0.5), yaw_rate_rad_s=0.8),
            duration_s=30.0,
            seed=7,
        ),
        EstimatorScenario(
            "tumbling",
            "stationary, rolling ±60°, pitching ±45°, yawing at 0.5 rad/s",
            tumbling(),
            duration_s=30.0,
            seed=8,
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
            accel_bias_mps2=(0.15, -0.1, 0.2),
            gyro_bias_rad_s=(0.003, -0.002, 0.004),
            gps_outages_s=((30.0, 35.0),),
            seed=4,
        ),
        EstimatorScenario(
            "car_circle",
            "car on a 12 m radius circle at 5 m/s on flat ground",
            circling(12.0, 5.0, CAR_BODY_HEIGHT_M),
            duration_s=30.0,
            vehicle="car",
            seed=5,
        ),
    )
}


def initial_truth(scenario: EstimatorScenario) -> TruthPoint:
    """The true state at t = 0, for explicit estimator initialization."""
    return scenario.trajectory(0.0)


# Settings and context -------------------------------------------------------------


@cache
def _base_config(vehicle: str) -> Any:
    from ..simulation.gazebo.config import load_drone_config

    if vehicle != "multirotor" and vehicle not in WHEELED_KINDS:
        raise ValueError(
            f"scenario vehicle must be multirotor or one of {WHEELED_KINDS}"
        )
    return load_drone_config(robot=None if vehicle == "multirotor" else vehicle)


def scenario_settings(scenario: EstimatorScenario) -> dict[str, Any]:
    """Merged configuration: vehicle defaults, ``SENSOR_NOISE``, then the
    scenario's ``settings`` (unknown keys raise ``ValueError``)."""
    from ..simulation.gazebo.config import merge

    data = deepcopy(_base_config(scenario.vehicle).data)
    return merge(merge(data, SENSOR_NOISE), dict(scenario.settings))


def _check_sensors(scenario: EstimatorScenario, sensors: tuple[str, ...]) -> None:
    unknown = set(sensors) - set(SENSOR_KINDS)
    if not sensors or unknown or len(set(sensors)) != len(sensors):
        raise ValueError(
            f"sensors must be distinct kinds of {list(SENSOR_KINDS)}, got {sensors}"
        )
    wheeled = scenario.vehicle in WHEELED_KINDS
    if "wheel_encoders" in sensors and not wheeled:
        raise ValueError(f"{scenario.name}: wheel_encoders need a wheeled vehicle")
    if "esc" in sensors and wheeled:
        raise ValueError(f"{scenario.name}: esc telemetry needs a multirotor")


def scenario_context(
    scenario: EstimatorScenario,
    sensors: tuple[str, ...] = DEFAULT_SENSORS,
    log: DebugLog | None = None,
) -> AlgorithmContext:
    """The context a Gazebo run with these settings and sensors would give."""
    from ..core.algorithms import ignore_debug
    from ..experiments.algorithms import build_context

    _check_sensors(scenario, sensors)
    config = _base_config(scenario.vehicle)
    return build_context(
        scenario_settings(scenario),
        config.properties,
        REFERENCE,
        sensors,
        log or ignore_debug,
        wheeled=config.wheeled,
    )


# Samples --------------------------------------------------------------------------


def scenario_samples(
    scenario: EstimatorScenario, sensors: tuple[str, ...] = DEFAULT_SENSORS
) -> list[tuple[str, Sample]]:
    """Seeded samples of the selected ``sensors`` (see module docs)."""
    from ..sensors.bus import PRIORITY

    _check_sensors(scenario, sensors)
    settings = scenario_settings(scenario)
    config = _base_config(scenario.vehicle)
    inputs = SynthesisInputs(
        REFERENCE,
        scenario.scene,
        config.properties,
        config.wheeled,
        scenario.seed,
        scenario.imu_noise,
        scenario.accel_bias_mps2,
        scenario.gyro_bias_rad_s,
        scenario.gps_noise_m,
        scenario.gps_outages_s,
        scenario.battery_current_a,
    )
    synthesizers = {k: build_synthesizer(k, inputs, settings) for k in sensors}
    end_ns = round(scenario.duration_s * 1e9)
    events = sorted(
        (t_ns, PRIORITY[kind], kind)
        for kind, synthesizer in synthesizers.items()
        for t_ns in range(0, end_ns + 1, synthesizer.period_ns)
    )
    samples: list[tuple[str, Sample]] = []
    sequence = dict.fromkeys(sensors, 0)
    truth: tuple[int, TruthPoint] | None = None
    for t_ns, _, kind in events:
        if truth is None or truth[0] != t_ns:
            truth = (t_ns, scenario.trajectory(t_ns * 1e-9))
        payload = synthesizers[kind].sample(t_ns, truth[1])
        if payload is not None:
            sequence[kind] += 1
            samples.append((kind, _sample(kind, sequence[kind], t_ns, payload)))
    return samples


# Scoring --------------------------------------------------------------------------


def run_scenario(
    factory: Callable[[AlgorithmContext, Mapping[str, Any]], Any],
    options: Mapping[str, Any],
    scenario: EstimatorScenario,
    sensors: tuple[str, ...] = DEFAULT_SENSORS,
) -> ScenarioReport:
    """Run an estimator on ``scenario`` with the selected ``sensors`` and
    score it against the truth.

    The context holds exactly ``sensors``; the estimator receives the
    samples of the kinds in its ``inputs``. Raises ``PluginContractError``
    if it breaks its contract and ``ValueError`` for an invalid selection
    or no estimate after ``settle_s``.
    """
    run = run_estimator(
        factory,
        options,
        scenario_samples(scenario, sensors),
        lambda log: scenario_context(scenario, sensors, log),
    )
    settle_ns = round(scenario.settle_s * 1e9)
    scored = [s for s in run.states if s.sim_time_ns >= settle_ns]
    if not scored:
        raise ValueError(
            f"{scenario.name}: no estimate after the {scenario.settle_s} s settle time"
        )
    errors = [_errors(s, scenario.trajectory) for s in scored]
    position, velocity, attitude, tilt = (
        _present([e[i] for e in errors]) for i in range(4)
    )
    final = scored[-1].payload
    nis = [i.nis / i.dof for i in run.innovations]
    return ScenarioReport(
        scenario.name,
        len(run.states),
        len(scored),
        _rms(position),
        None if position is None else float(position.max()),
        None if position is None else float(position[-1]),
        _rms(velocity),
        _rms(attitude),
        _rms(tilt),
        _bias_error(scenario.accel_bias_mps2, final.accel_bias_mps2),
        _bias_error(scenario.gyro_bias_rad_s, final.gyro_bias_rad_s),
        float(np.mean(nis)) if nis else None,
        run.summary,
    )


def _errors(
    estimate: Sample[EstimatedState], trajectory: Trajectory
) -> list[float | None]:
    """Position (m), velocity (m/s), attitude and tilt (deg) error norms;
    None for blocks the estimate lacks."""
    truth = trajectory(estimate.sim_time_ns * 1e-9)
    state = estimate.payload
    position = velocity = attitude = tilt = None
    if state.position_m is not None:
        position = float(np.linalg.norm(state.position_m - truth.position_m))
    if state.velocity_mps is not None:
        velocity = float(np.linalg.norm(state.velocity_mps - truth.velocity_mps))
    if state.quaternion_xyzw is not None:
        true_rotation = Rotation.from_quat(truth.quaternion_xyzw)
        estimated = Rotation.from_quat(state.quaternion_xyzw)
        attitude = float(np.degrees((true_rotation.inv() * estimated).magnitude()))
        # World up in body axes: a world-yaw error leaves it unchanged.
        up_true = true_rotation.inv().apply([0.0, 0.0, 1.0])
        up_estimated = estimated.inv().apply([0.0, 0.0, 1.0])
        tilt = float(np.degrees(np.arccos(np.clip(up_true @ up_estimated, -1.0, 1.0))))
    return [position, velocity, attitude, tilt]


def _present(values: list[float | None]) -> NDArray[np.float64] | None:
    """Values of estimates that have the block (None if none have it)."""
    kept = [v for v in values if v is not None]
    return np.array(kept) if kept else None


def _bias_error(true: Any, estimated: Any) -> float | None:
    if estimated is None:
        return None
    return float(np.linalg.norm(np.subtract(true, estimated)))


def _rms(values: NDArray[np.float64] | None) -> float | None:
    return None if values is None else float(np.sqrt(np.mean(values**2)))
