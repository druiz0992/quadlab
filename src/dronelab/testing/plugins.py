"""Contract test kit for estimator and controller plug-ins (no Gazebo).

Use in a plug-in's unit tests::

    from dronelab.testing.plugins import check_controller, check_estimator

    def test_my_filter_follows_the_contract():
        report = check_estimator(MyFilter, {"gain": 2.0})
        assert report.final_position_error_m < 0.5

``check_estimator`` feeds a synthetic, seeded hover (vehicle level at
``HOVER_POSITION_M``; IMU at 200 Hz and GPS at 10 Hz with small white
noise) and checks the ``StateEstimator`` contract (estimation/interface.py):

- ``inputs`` names known sensor streams;
- ``latest()`` is None or a ``Sample[EstimatedState]`` whose time never
  decreases and never runs ahead of the newest delivered sample;
- estimates are finite, with a symmetric positive semi-definite 15×15
  covariance;
- innovations have finite, non-negative NIS and positive degrees of freedom;
- ``summary()`` is a JSON-serializable dict;
- ``context.log`` entries are valid (see experiments/algorithm_debug.py);
- two runs give identical estimates (no hidden global randomness).

``check_controller`` checks the ``RotorController`` contract
(control/interface.py) on the nominal vehicle at hover: thrusts are
finite, non-negative, one per rotor, within the rotor limit unless
``saturated`` is set, sum to the weight within ``weight_tolerance``, and
repeat exactly on a second run; ``setpoint_m`` is None or three finite
numbers; ``allow_landing`` can be called.

Both raise ``PluginContractError`` (an AssertionError) with the reason,
and return a report for further assertions. The data are synthetic and
noise-light: passing says the plug-in fits the harness, not that it is a
good algorithm; judge that with Gazebo runs and ``dronelab-replay``.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
from typing import Any

import numpy as np

from ..control.interface import ControlState
from ..control.references import ReferencePoint
from ..core.algorithms import AlgorithmContext, SensorDescription
from ..core.measurements import EstimatedState, GpsReading, ImuReading, Sample
from ..core.vehicle import DroneProperties
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
) -> AlgorithmContext:
    """Context like a Gazebo run's: body-origin mounts, nominal rates."""
    rates = {"imu": 200.0, "gps": 10.0}
    sensors = {
        kind: SensorDescription(kind, STREAMS[kind][1], rates.get(kind))
        for kind in kinds
    }
    extra = {} if log is None else {"log": log}
    return AlgorithmContext(
        REFERENCE, vehicle or DroneProperties(), sensors, GRAVITY_MPS2, **extra
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
    return GpsReading(*guess, np.zeros(3))


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
    first = _run_estimator(factory, options or {}, duration_s)
    second = _run_estimator(factory, options or {}, duration_s)
    _fail(
        len(first["states"]) == len(second["states"])
        and all(_same_state(a, b) for a, b in zip(first["states"], second["states"])),
        "estimates differ between two identical runs (hidden randomness or state)",
    )
    return first["report"]


def _run_estimator(factory, options, duration_s) -> dict[str, Any]:
    debug_entries = [0]

    def log(name, sim_time_ns, values, labels=None):
        try:
            validate_debug_entry(name, sim_time_ns, values, labels)
        except ValueError as error:
            raise PluginContractError(f"invalid context.log entry: {error}")
        debug_entries[0] += 1

    context = synthetic_context(log=log)
    estimator = factory(context, dict(options))
    inputs = getattr(estimator, "inputs", None)
    _fail(
        isinstance(inputs, tuple) and inputs and all(k in STREAMS for k in inputs),
        f"inputs must be a non-empty tuple of {sorted(STREAMS)}, got {inputs!r}",
    )
    states, innovations, newest_ns, last = [], 0, -1, None
    delivered = 0
    for kind, sample in hover_samples(duration_s):
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
    final = None if last is None else last.payload.position_m
    error = None if final is None else float(np.linalg.norm(final - HOVER_POSITION_M))
    report = EstimatorReport(
        delivered,
        len(states),
        innovations,
        debug_entries[0],
        None if final is None else tuple(float(v) for v in final),
        error,
        summary,
    )
    return dict(report=report, states=states)


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
    values = np.concatenate(
        [
            state.position_m,
            state.velocity_mps,
            state.quaternion_xyzw,
            state.accel_bias_mps2,
            state.gyro_bias_rad_s,
            state.covariance.ravel(),
        ]
    )
    _fail(bool(np.all(np.isfinite(values))), "estimate contains NaN or infinity")
    _fail(
        abs(np.linalg.norm(state.quaternion_xyzw) - 1) < 1e-6,
        "estimate quaternion must be unit length",
    )
    covariance = state.covariance
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


def _check_innovations(innovations: Any) -> int:
    _fail(isinstance(innovations, list), "drain_innovations() must return a list")
    for item in innovations:
        _fail(isinstance(item, Innovation), "innovations must be Innovation objects")
        _fail(
            np.isfinite(item.nis) and item.nis >= 0 and item.dof > 0,
            f"innovation {item.name!r}: NIS must be finite and ≥ 0, dof > 0",
        )
    return len(innovations)


def _same_state(a: Sample, b: Sample) -> bool:
    if a.sim_time_ns != b.sim_time_ns:
        return False
    x, y = a.payload, b.payload
    return all(
        np.array_equal(getattr(x, name), getattr(y, name))
        for name in (
            "position_m",
            "velocity_mps",
            "quaternion_xyzw",
            "accel_bias_mps2",
            "gyro_bias_rad_s",
            "covariance",
        )
    )


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
    second = _run_controller(factory, options or {}, vehicle, seconds)
    _fail(
        all(np.array_equal(a, b) for a, b in zip(first[1], second[1])),
        "thrusts differ between two identical runs (hidden randomness or state)",
    )
    report = first[0]
    _fail(
        abs(report.total_thrust_n - report.weight_n)
        <= weight_tolerance * report.weight_n,
        f"hover thrust {report.total_thrust_n:.3f} N differs from the weight "
        f"{report.weight_n:.3f} N by more than {weight_tolerance:.0%}",
    )
    return report


def _run_controller(factory, options, vehicle, seconds):
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
