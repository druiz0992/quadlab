"""Load estimator/controller plug-ins and connect them to the sensor bus.

``algorithms.estimator`` and ``algorithms.controller`` name a class as
``"package.module:Class"`` or a built-in alias (``BUILTIN_ESTIMATORS``,
``BUILTIN_CONTROLLERS``); "" means no estimator and the built-in manual
controller. Plug-ins are constructed as ``cls(context, options)`` (see
estimation/interface.py and control/interface.py). Built-ins receive their
existing configuration sections as options (``ekf``, ``controller``) unless
options are given.

The runner dispatches bus samples to the estimator serially on the mission
thread, in delivery order. ``EstimateStateSource`` turns the latest estimate
into the controller's state when ``control_state_source`` is "estimate":
position, velocity, and attitude from the estimate; body rates from the
latest delivered gyro sample (rotated by the declared IMU mount) minus the
estimated gyro bias. If the estimate is older than ``max_estimate_age_s`` at
a control tick, ``StaleEstimate`` is raised and the mission stops thrust.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..control.interface import ControlState, RotorController
from ..core.algorithms import AlgorithmContext, SensorDescription
from ..core.measurements import EstimatedState, GpsReading, Sample
from ..core.vehicle import DroneProperties
from ..estimation.interface import Innovation, StateEstimator
from ..sensors.bus import STREAMS, SensorBus

BUILTIN_ESTIMATORS = {
    "gps_imu_ekf": "dronelab.estimation.gps_imu_ekf.plugin:GpsImuEkfEstimator",
}
BUILTIN_CONTROLLERS = {
    "manual": "dronelab.control.manual:ManualRotorController",
}
# Sensors whose configuration has a mount (others are body-frame products).
MOUNTED = ("imu", "gps", "baro", "magnetometer", "rangefinder", "camera")


class StaleEstimate(RuntimeError):
    """The controller's estimated state is older than allowed."""


def load_class(spec: str, builtins: Mapping[str, str]) -> type:
    """Import ``package.module:Class`` (or a built-in alias); explicit errors."""
    path = builtins.get(spec, spec)
    module_name, _, class_name = path.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"Algorithm {spec!r} must be 'package.module:Class'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(f"Cannot import algorithm module {module_name!r}: {error}")
    try:
        return getattr(module, class_name)
    except AttributeError:
        raise ValueError(f"{module_name} has no class {class_name!r}") from None


def build_context(
    settings: Mapping[str, Any],
    vehicle: DroneProperties,
    gps_reference: GpsReading,
    kinds: tuple[str, ...],
) -> AlgorithmContext:
    """Context from configuration only: sensor frames, rates, and mounts."""
    sensors = settings["sensors"]
    descriptions = {}
    for kind in kinds:
        section = sensors.get(kind, {})
        mount = section.get("mount") if kind in MOUNTED else None
        quaternion, position = (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0)
        if mount is not None:
            quaternion = tuple(
                Rotation.from_euler(
                    "xyz", mount["rotation_deg"], degrees=True
                ).as_quat()
            )
            position = tuple(mount["position_m"])
        calibration = section.get("calibration")
        descriptions[kind] = SensorDescription(
            kind,
            STREAMS[kind][1],
            section.get("rate_hz"),
            position,
            quaternion,
            calibration if calibration and calibration.get("enabled") else None,
        )
    return AlgorithmContext(gps_reference, vehicle, descriptions)


def build_estimator(
    settings: Mapping[str, Any], context: AlgorithmContext
) -> StateEstimator | None:
    algorithms = settings["algorithms"]
    spec = algorithms["estimator"]
    if not spec:
        return None
    options = algorithms["estimator_options"]
    if spec == "gps_imu_ekf" and not options:
        options = settings["ekf"]
    estimator = load_class(spec, BUILTIN_ESTIMATORS)(context, options)
    if not isinstance(estimator, StateEstimator):
        raise ValueError(f"{spec} does not implement StateEstimator")
    missing = set(estimator.inputs) - set(context.sensors)
    if missing:
        raise ValueError(f"{spec} needs disabled sensors: {sorted(missing)}")
    return estimator


def build_controller(
    settings: Mapping[str, Any], context: AlgorithmContext
) -> RotorController:
    algorithms = settings["algorithms"]
    spec = algorithms["controller"] or "manual"
    options = algorithms["controller_options"]
    if spec == "manual" and not options:
        options = settings["controller"]
    controller = load_class(spec, BUILTIN_CONTROLLERS)(context, options)
    if not isinstance(controller, RotorController):
        raise ValueError(f"{spec} does not implement RotorController")
    return controller


class AlgorithmRunner:
    """Deliver bus samples to the estimator; collect estimates and NIS."""

    def __init__(self, bus: SensorBus, estimator: StateEstimator | None) -> None:
        self.bus = bus
        self.estimator = estimator
        self._inputs = set(estimator.inputs) if estimator is not None else set()
        self._last_estimate: Sample[EstimatedState] | None = None
        self.handler_errors = 0

    def advance(
        self, now_ns: int
    ) -> tuple[
        list[tuple[str, Sample]], list[Sample[EstimatedState]], list[Innovation]
    ]:
        """Release due samples; return (delivered, new estimates, innovations)."""
        delivered = self.bus.advance(now_ns)
        estimates: list[Sample[EstimatedState]] = []
        if self.estimator is None:
            return delivered, estimates, []
        for kind, sample in delivered:
            if kind not in self._inputs:
                continue
            self.estimator.on_sample(kind, sample)
            latest = self.estimator.latest()
            if latest is not None and latest is not self._last_estimate:
                estimates.append(latest)
                self._last_estimate = latest
        return delivered, estimates, self.estimator.drain_innovations()

    @property
    def latest_estimate(self) -> Sample[EstimatedState] | None:
        return self._last_estimate


class EstimateStateSource:
    """Controller state from the estimator (see module docs)."""

    def __init__(self, imu_mount_xyzw: tuple[float, ...], max_age_s: float) -> None:
        self._imu_rotation = Rotation.from_quat(imu_mount_xyzw)
        self._max_age_ns = round(max_age_s * 1e9)
        self._gyro_body: NDArray[np.float64] | None = None
        self._airspeed: float | None = None
        self.max_age_ms = 0.0

    def on_delivered(self, kind: str, sample: Sample) -> None:
        if kind == "imu":
            self._gyro_body = self._imu_rotation.apply(
                sample.payload.best_angular_rate_rad_s
            )
        elif kind == "airspeed":
            self._airspeed = sample.payload.indicated_airspeed_mps

    def state(
        self, now_ns: int, estimate: Sample[EstimatedState] | None
    ) -> ControlState | None:
        """State at ``now_ns``; None before the first estimate (start-up)."""
        if estimate is None or self._gyro_body is None:
            return None
        age = now_ns - estimate.sim_time_ns
        self.max_age_ms = max(self.max_age_ms, age * 1e-6)
        if age > self._max_age_ns:
            raise StaleEstimate(
                f"Estimate is {age * 1e-6:.1f} ms old at a control tick "
                f"(limit {self._max_age_ns * 1e-6:.1f} ms)"
            )
        x = estimate.payload
        return ControlState(
            now_ns,
            np.array(x.position_m),
            np.array(x.velocity_mps),
            np.array(x.quaternion_xyzw),
            self._gyro_body - x.gyro_bias_rad_s,
            source="estimate",
            airspeed_mps=self._airspeed,
        )


@dataclass(frozen=True)
class AlgorithmStack:
    """The plug-ins of one run and how they are wired.

    ``bus`` and ``runner`` exist only with an estimator; ``estimate_source``
    only when the controller flies on the estimate; ``controller`` only in
    rotor-thrust mode.
    """

    context: AlgorithmContext
    estimator: StateEstimator | None
    bus: SensorBus | None
    runner: AlgorithmRunner | None
    estimate_source: EstimateStateSource | None
    controller: RotorController | None


def build_algorithm_stack(
    settings: Mapping[str, Any],
    vehicle: DroneProperties,
    reference: GpsReading,
    bus_kinds: tuple[str, ...],
    *,
    rotor_mode: bool,
) -> AlgorithmStack:
    """Context, estimator, sensor bus, runner, state source, and controller.

    Algorithms see configuration-derived context only, never truth.
    """
    algorithms = settings["algorithms"]
    context = build_context(settings, vehicle, reference, bus_kinds)
    estimator = build_estimator(settings, context)
    bus = SensorBus(settings["sensor_timing"], bus_kinds) if estimator else None
    runner = AlgorithmRunner(bus, estimator) if bus is not None else None
    estimate_source = (
        EstimateStateSource(
            context.sensors["imu"].mount_quaternion_xyzw,
            algorithms["max_estimate_age_s"],
        )
        if algorithms["control_state_source"] == "estimate"
        else None
    )
    controller = build_controller(settings, context) if rotor_mode else None
    return AlgorithmStack(context, estimator, bus, runner, estimate_source, controller)
