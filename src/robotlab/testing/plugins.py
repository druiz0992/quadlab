"""Contract test kit for estimator and controller plug-ins (no Gazebo).

Use in a plug-in's unit tests::

    from robotlab.testing.plugins import check_controller, check_estimator

    def test_my_filter_follows_the_contract():
        report = check_estimator(MyFilter, {"gain": 2.0})
        assert report.final_position_error_m < 0.5

``check_estimator`` feeds a synthetic, seeded hover (vehicle level at
``HOVER_POSITION_M``; IMU at 200 Hz and GPS at 10 Hz with small white
noise) and checks the ``StateEstimator`` contract (estimation/interface.py):

- ``inputs`` names known sensor streams;
- ``latest()`` is None or a ``Sample[EstimatedState]`` whose time never
  decreases and never runs ahead of the newest delivered sample;
- estimates (any subset of blocks, see ``EstimatedState``; values are
  finite by construction) have a unit quaternion and, when reported, a
  symmetric positive semi-definite covariance over the present blocks;
- innovations have finite, non-negative NIS and positive degrees of freedom;
- ``summary()`` is a JSON-serializable dict;
- ``context.log`` entries are valid (see experiments/algorithm_debug.py);
- two runs give identical estimates (no hidden global randomness).

``check_controller`` checks the ``RotorController`` contract
(control/interface.py) on the nominal vehicle at hover: thrusts are
finite, non-negative, one per rotor, within the rotor limit unless
``saturated`` is set, sum to the weight within ``weight_tolerance``, and
repeat exactly on a second run; ``setpoint_m`` is None or three finite
numbers; an optional ``tracking_reference`` returns None or a finite
``ReferencePoint`` without changing the controller's output; ``allow_landing``
can be called.

``check_wheeled_controller`` checks the ``WheeledController`` contract
(control/wheeled.py) on a nominal rover, car, or tractor: from rest and
while driving at a few speeds, with no request and with keyboard-like
``DriverCommand`` requests, ``update`` returns ``WheelCommands`` that the
vehicle's actuators accept (one finite value per wheel, torque only on
driven wheels, brakes >= 0 on braked wheels, angles only on steered
wheels), repeats exactly on a second run, and ``summary()`` (optional) is
a JSON-serializable dict.

``run_estimator`` is the checked delivery loop on its own, for any sample
list; ``testing/scenarios.py`` uses it to score estimators against truth.

All three raise ``PluginContractError`` (an AssertionError) with the reason,
and return a report for further assertions. The data are synthetic and
noise-light: passing says the plug-in fits the harness, not that it is a
good algorithm; judge that with Gazebo runs and ``robotlab-replay``.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
from typing import Any, cast

import numpy as np

from ..control.interface import ControlState
from ..control.references import ReferencePoint
from ..core.algorithms import (
    AlgorithmContext,
    DebugLog,
    SensorDescription,
    ignore_debug,
)
from ..core.measurements import EstimatedState, GpsReading, ImuReading, Sample
from ..core.vehicle import DroneProperties
from ..core.wheeled import WheelCommands
from ..core.wheeled_vehicle import WheeledProperties
from ..estimation.interface import Innovation
from ..experiments.algorithm_debug import validate_debug_entry
from ..sensors.bus import STREAMS
from ..sensors.geodesy import gps_to_enu

GRAVITY_MPS2 = 9.81
HOVER_POSITION_M = (0.0, 0.0, 1.0)
REFERENCE = GpsReading(40.4168, -3.7038, 650.0)
EARTH_RADIUS_M = 6378137.0


class PluginContractError(AssertionError):
    """A plug-in broke its contract (the message says how)."""


@dataclass(frozen=True)
class EstimatorReport:
    """What ``check_estimator`` saw (positions in world ENU, m)."""

    samples_delivered: int
    estimates: int
    innovations: int
    debug_entries: int
    final_position_m: tuple[float, float, float] | None
    final_position_error_m: float | None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ControllerReport:
    """What ``check_controller`` saw at hover (thrusts in N)."""

    thrust_n: tuple[float, ...]
    total_thrust_n: float
    weight_n: float
    saturated: bool


def synthetic_context(
    kinds: tuple[str, ...] = ("imu", "gps"),
    vehicle: DroneProperties | None = None,
    log: Callable[..., None] | None = None,
    wheeled: WheeledProperties | None = None,
) -> AlgorithmContext:
    """Context like a Gazebo run's: body-origin mounts, nominal rates.

    ``wheeled`` makes it a wheeled-robot context (``context.wheeled``).
    """
    rates = {"imu": 200.0, "gps": 10.0}
    sensors = {
        kind: SensorDescription(kind, STREAMS[kind][1], rates.get(kind))
        for kind in kinds
    }
    return AlgorithmContext(
        REFERENCE,
        vehicle or DroneProperties(),
        sensors,
        GRAVITY_MPS2,
        log=log or ignore_debug,
        wheeled=wheeled,
    )


def enu_to_gps(position_m: tuple[float, float, float]) -> GpsReading:
    """Fix at an ENU offset from ``REFERENCE``: the exact inverse of
    ``sensors.geodesy.gps_to_enu`` (Newton iterations from a spherical guess;
    converges to well below a millimetre for local offsets)."""
    target = np.asarray(position_m, float)
    east, north, up = target
    guess = np.array(
        [
            REFERENCE.latitude_deg + np.degrees(north / EARTH_RADIUS_M),
            REFERENCE.longitude_deg
            + np.degrees(
                east / (EARTH_RADIUS_M * np.cos(np.radians(REFERENCE.latitude_deg)))
            ),
            REFERENCE.altitude_m + up,
        ]
    )
    step = np.array([1e-7, 1e-7, 1e-2])  # finite-difference steps (deg, deg, m)
    for _ in range(5):
        here = gps_to_enu(GpsReading(*guess), REFERENCE)
        jacobian = np.column_stack(
            [
                (
                    gps_to_enu(GpsReading(*(guess + np.eye(3)[i] * step[i])), REFERENCE)
                    - here
                )
                / step[i]
                for i in range(3)
            ]
        )
        guess = guess + np.linalg.solve(jacobian, target - here)
    return GpsReading(float(guess[0]), float(guess[1]), float(guess[2]), np.zeros(3))


def hover_samples(duration_s: float = 10.0, seed: int = 0) -> list[tuple[str, Sample]]:
    """Seeded IMU (200 Hz) and GPS (10 Hz) samples of a level hover.

    Noise: 0.02 m/s² and 0.002 rad/s per IMU sample, 0.05 m per GPS axis.
    """
    rng = np.random.default_rng(seed)
    samples: list[tuple[str, Sample]] = []
    sequence = {"imu": 0, "gps": 0}
    for step in range(1, round(duration_s * 200) + 1):
        t_ns = step * 5_000_000
        sequence["imu"] += 1
        imu = ImuReading(
            np.array([0.0, 0.0, GRAVITY_MPS2]) + rng.normal(0, 0.02, 3),
            rng.normal(0, 0.002, 3),
        )
        samples.append(("imu", _sample("imu", sequence["imu"], t_ns, imu)))
        if step % 20 == 0:
            sequence["gps"] += 1
            noisy = np.array(HOVER_POSITION_M) + rng.normal(0, 0.05, 3)
            gps = enu_to_gps(tuple(noisy))
            samples.append(("gps", _sample("gps", sequence["gps"], t_ns, gps)))
    return samples


def _sample(kind: str, sequence: int, t_ns: int, payload: Any) -> Sample:
    return Sample(f"sensors/{kind}", sequence, 0, t_ns, 0, STREAMS[kind][1], payload)


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise PluginContractError(message)


def _json_dict(value: Any, name: str) -> None:
    _fail(isinstance(value, dict), f"{name}() must return a dict")
    try:
        json.dumps(value)
    except (TypeError, ValueError) as error:
        raise PluginContractError(f"{name}() is not JSON-serializable: {error}")


# Estimators -----------------------------------------------------------------------


def check_estimator(
    factory: Callable[[AlgorithmContext, Mapping[str, Any]], Any],
    options: Mapping[str, Any] | None = None,
    *,
    duration_s: float = 10.0,
) -> EstimatorReport:
    """Run the estimator contract checks (see module docs)."""
    samples = hover_samples(duration_s)
    first = run_estimator(factory, options or {}, samples)
    second = run_estimator(factory, options or {}, samples)
    _fail(
        len(first.states) == len(second.states)
        and all(_same_state(a, b) for a, b in zip(first.states, second.states)),
        "estimates differ between two identical runs (hidden randomness or state)",
    )
    last = first.states[-1] if first.states else None
    final = None if last is None else last.payload.position_m
    error = None if final is None else float(np.linalg.norm(final - HOVER_POSITION_M))
    return EstimatorReport(
        first.samples_delivered,
        len(first.states),
        len(first.innovations),
        first.debug_entries,
        None if final is None else (float(final[0]), float(final[1]), float(final[2])),
        error,
        first.summary,
    )


@dataclass(frozen=True)
class EstimatorRun:
    """Everything one contract-checked estimator run produced."""

    samples_delivered: int
    states: tuple[Sample[EstimatedState], ...]  # each new latest(), in order
    innovations: tuple[Innovation, ...]
    debug_entries: int
    summary: dict[str, Any]


def run_estimator(
    factory: Callable[[AlgorithmContext, Mapping[str, Any]], Any],
    options: Mapping[str, Any],
    samples: list[tuple[str, Sample]],
    context: Callable[[DebugLog], AlgorithmContext] | None = None,
) -> EstimatorRun:
    """Deliver ``samples`` (kind, sample) in order to a new estimator,
    checking the contract after every delivery.

    ``context`` builds the estimator's context around the checked debug
    log; default: ``synthetic_context`` (IMU and GPS).

    Samples of kinds the estimator does not declare in ``inputs`` are
    skipped. Raises ``PluginContractError`` on the first violation.
    """
    debug_entries = [0]

    def log(name, sim_time_ns, values, labels=None):
        try:
            validate_debug_entry(name, sim_time_ns, values, labels)
        except ValueError as error:
            raise PluginContractError(f"invalid context.log entry: {error}")
        debug_entries[0] += 1

    built = (context or (lambda debug: synthetic_context(log=debug)))(log)
    estimator = factory(built, dict(options))
    declared = getattr(estimator, "inputs", None)
    _fail(
        isinstance(declared, tuple)
        and len(declared) > 0
        and all(k in STREAMS for k in declared),
        f"inputs must be a non-empty tuple of {sorted(STREAMS)}, got {declared!r}",
    )
    inputs = cast(tuple[str, ...], declared)  # validated just above
    states: list[Sample[EstimatedState]] = []
    innovations: list[Innovation] = []
    newest_ns, last, delivered = -1, None, 0
    for kind, sample in samples:
        if kind not in inputs:
            continue
        delivered += 1
        newest_ns = max(newest_ns, sample.sim_time_ns)
        estimator.on_sample(kind, sample)
        latest = estimator.latest()
        if latest is not None and latest is not last:
            _check_estimate(latest, last, newest_ns)
            states.append(latest)
            last = latest
        innovations += _check_innovations(estimator.drain_innovations())
    summary = estimator.summary()
    _json_dict(summary, "summary")
    return EstimatorRun(
        delivered, tuple(states), tuple(innovations), debug_entries[0], summary
    )


def _check_estimate(latest: Any, previous: Any, newest_ns: int) -> None:
    _fail(isinstance(latest, Sample), "latest() must return a Sample or None")
    state = latest.payload
    _fail(isinstance(state, EstimatedState), "estimate payload must be EstimatedState")
    _fail(
        latest.sim_time_ns <= newest_ns,
        f"estimate at {latest.sim_time_ns} ns is ahead of the newest sample "
        f"({newest_ns} ns)",
    )
    _fail(
        previous is None or latest.sim_time_ns >= previous.sim_time_ns,
        "estimate times must not decrease",
    )
    _fail(
        state.quaternion_xyzw is None
        or abs(np.linalg.norm(state.quaternion_xyzw) - 1) < 1e-6,
        "estimate quaternion must be unit length",
    )
    covariance = state.covariance
    if covariance is None:
        return
    _fail(
        np.allclose(
            covariance, covariance.T, atol=1e-9 * max(1.0, np.abs(covariance).max())
        ),
        "covariance must be symmetric",
    )
    _fail(
        np.linalg.eigvalsh(0.5 * (covariance + covariance.T)).min()
        >= -1e-9 * max(1.0, np.abs(covariance).max()),
        "covariance must be positive semi-definite",
    )


def _check_innovations(innovations: Any) -> list[Innovation]:
    _fail(isinstance(innovations, list), "drain_innovations() must return a list")
    for item in innovations:
        _fail(isinstance(item, Innovation), "innovations must be Innovation objects")
        _fail(
            np.isfinite(item.nis) and item.nis >= 0 and item.dof > 0,
            f"innovation {item.name!r}: NIS must be finite and ≥ 0, dof > 0",
        )
    return innovations


def _same_state(a: Sample, b: Sample) -> bool:
    if a.sim_time_ns != b.sim_time_ns:
        return False
    x, y = a.payload, b.payload
    return all(
        _same_array(getattr(x, name), getattr(y, name))
        for name in (
            "position_m",
            "velocity_mps",
            "quaternion_xyzw",
            "accel_bias_mps2",
            "gyro_bias_rad_s",
            "covariance",
        )
    )


def _same_array(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return bool(np.array_equal(a, b))


# Controllers ----------------------------------------------------------------------


def check_controller(
    factory: Callable[[AlgorithmContext, Mapping[str, Any]], Any],
    options: Mapping[str, Any] | None = None,
    *,
    vehicle: DroneProperties | None = None,
    weight_tolerance: float = 0.1,
    seconds: float = 2.0,
) -> ControllerReport:
    """Run the controller contract checks at hover (see module docs)."""
    first = _run_controller(factory, options or {}, vehicle, seconds)
    second = _run_controller(
        factory, options or {}, vehicle, seconds, probe_reference=True
    )
    _fail(
        all(np.array_equal(a, b) for a, b in zip(first[1], second[1])),
        "thrusts differ between two identical runs (hidden randomness or "
        "state, or a tracking_reference that changes the controller)",
    )
    report = first[0]
    _fail(
        abs(report.total_thrust_n - report.weight_n)
        <= weight_tolerance * report.weight_n,
        f"hover thrust {report.total_thrust_n:.3f} N differs from the weight "
        f"{report.weight_n:.3f} N by more than {weight_tolerance:.0%}",
    )
    return report


def _run_controller(factory, options, vehicle, seconds, probe_reference=False):
    context = synthetic_context(vehicle=vehicle)
    controller = factory(context, dict(options))
    properties = context.vehicle
    hover = np.array(HOVER_POSITION_M)
    reference = ReferencePoint(hover, np.zeros(3), np.zeros(3), 0.0)
    history = []
    for tick in range(round(seconds * 50)):
        state = ControlState(
            (tick + 1) * 20_000_000,
            hover.copy(),
            np.zeros(3),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.zeros(3),
        )
        controller.update_position(state, reference, 0.02)
        if probe_reference:
            _check_tracking_reference(controller, state)
        thrust = np.asarray(controller.update_attitude(state), dtype=float)
        _check_thrust(thrust, properties, controller)
        history.append(thrust)
    setpoint = controller.setpoint_m
    _fail(
        setpoint is None
        or (np.shape(setpoint) == (3,) and np.all(np.isfinite(setpoint))),
        "setpoint_m must be None or three finite numbers",
    )
    controller.allow_landing()
    final = history[-1]
    report = ControllerReport(
        tuple(float(v) for v in final),
        float(final.sum()),
        properties.mass * GRAVITY_MPS2,
        bool(controller.saturated),
    )
    return report, history


def _check_tracking_reference(controller: Any, state: ControlState) -> None:
    """Optional evaluation hook: None or a finite ReferencePoint. (Called
    every tick of the second run only, so a hook that changes controller
    state fails the two-run comparison.)"""
    method = getattr(controller, "tracking_reference", None)
    if method is None:
        return
    reference = method(
        state.position_m, state.velocity_world_mps, state.quaternion_xyzw
    )
    if reference is None:
        return
    _fail(
        isinstance(reference, ReferencePoint),
        "tracking_reference must return a ReferencePoint or None",
    )
    values = [
        np.shape(reference.position_m) == (3,),
        np.shape(reference.velocity_mps) == (3,),
        bool(np.all(np.isfinite(reference.position_m))),
        bool(np.all(np.isfinite(reference.velocity_mps))),
        bool(np.isfinite(reference.yaw_rad)),
    ]
    _fail(all(values), "tracking_reference must give finite 3-vectors and yaw")


def _check_thrust(thrust: np.ndarray, properties: DroneProperties, controller) -> None:
    rotors = len(properties.rotors)
    _fail(
        thrust.shape == (rotors,),
        f"expected {rotors} rotor thrusts, got {thrust.shape}",
    )
    _fail(bool(np.all(np.isfinite(thrust))), "thrusts contain NaN or infinity")
    _fail(bool(np.all(thrust >= 0)), "thrusts must be non-negative")
    _fail(
        bool(controller.saturated)
        or bool(np.all(thrust <= properties.max_rotor_thrust + 1e-9)),
        "thrust above the rotor limit without setting saturated",
    )


@dataclass(frozen=True)
class WheeledControllerReport:
    """Commands ``check_wheeled_controller`` saw, in probe order."""

    commands: tuple[WheelCommands, ...]
    summary: dict[str, Any] = field(default_factory=dict)


def nominal_wheeled(kind: str = "car") -> WheeledProperties:
    """The built-in preset vehicle of ``kind`` (rover, car, tractor)."""
    from ..simulation.gazebo.config import load_drone_config

    properties = load_drone_config(robot=kind).wheeled
    if properties is None:
        raise ValueError(f"{kind!r} is not a wheeled robot kind")
    return properties


def _wheeled_probes(properties: WheeledProperties) -> list[tuple[Any, Any]]:
    """(state, request) pairs: at rest and moving, with and without input."""
    from ..control.wheeled import DriverCommand, WheeledControlState

    n = properties.wheel_count
    probes: list[tuple[Any, Any]] = []
    for speed in (0.0, 1.0, 5.0):
        for yaw_rate in (0.0, 0.3):
            state = WheeledControlState(
                sim_time_ns=len(probes) * 10_000_000,
                position_m=np.zeros(3),
                velocity_world_mps=np.array([speed, 0.0, 0.0]),
                quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                body_rates_rad_s=np.array([0.0, 0.0, yaw_rate]),
                wheel_speed_rad_s=np.full(n, speed / properties.tyre.radius_m),
                steer_angle_rad=np.zeros(n),
            )
            for request in (
                None,
                DriverCommand(throttle=0.5, steer=0.3),
                DriverCommand(throttle=-0.5, brake=0.0, steer=-1.0),
                DriverCommand(brake=1.0),
            ):
                probes.append((state, request))
    return probes


def _run_wheeled(factory, options, properties) -> tuple[list[WheelCommands], Any]:
    controller = factory(synthetic_context(wheeled=properties), dict(options))
    commands = []
    for index, (state, request) in enumerate(_wheeled_probes(properties)):
        command = controller.update(state, request, 0.0 if index == 0 else 0.01)
        _fail(
            isinstance(command, WheelCommands),
            f"update returned {type(command).__name__}, not WheelCommands",
        )
        try:
            command.validate(
                tuple(properties.mask("driven")),
                tuple(properties.mask("braked")),
                tuple(properties.mask("steered")),
            )
        except ValueError as error:
            raise PluginContractError(f"invalid WheelCommands: {error}") from error
        commands.append(command)
    return commands, controller


def check_wheeled_controller(
    factory: Callable[[AlgorithmContext, Mapping[str, Any]], Any],
    options: Mapping[str, Any] | None = None,
    kind: str = "car",
) -> WheeledControllerReport:
    """Check a ``WheeledController`` class (see module docs)."""
    from ..control.wheeled import WheeledController

    properties = nominal_wheeled(kind)
    options = options or {}
    first, controller = _run_wheeled(factory, options, properties)
    _fail(
        isinstance(controller, WheeledController),
        "controller does not implement WheeledController (update)",
    )
    second, _ = _run_wheeled(factory, options, properties)
    _fail(first == second, "two runs gave different commands (hidden state or RNG)")
    summary: dict[str, Any] = {}
    if hasattr(controller, "summary"):
        summary = controller.summary()
        _json_dict(summary, "summary()")
    return WheeledControllerReport(tuple(first), summary)
