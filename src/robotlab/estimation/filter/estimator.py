"""``StateEstimator`` plug-in for any ``RecursiveFilter`` (``filter``).

Select a filter class with::

    "algorithms": {"estimator": "filter",
                   "estimator_options": {"filter": "mypkg.my_ekf:MyEkf",
                                         "filter_options": {...},
                                         "gps_std_m": [0.5, 0.5, 0.8]}}

The filter is built as ``cls(filter_options)`` (``filter`` may also name a
factory such as ``"pkg.mod:Class.from_options"``) and must implement
``RecursiveFilter`` (protocol.py). The estimator subscribes to the filter's
driving stream (its ``predict_<kind>``) plus every **enabled** stream it has
an ``update_<kind>`` for (listed in ``summary()["inputs"]``); it raises
ValueError when the driving stream is disabled or no enabled stream can
correct it. Mounts, wheel geometry, and the ENU origin come from the
context. The remaining options are the measurement converter's
(``CONVERTER_DEFAULTS``, measurements.py), ``max_driver_interval_s``
(longest gap between driving samples, default 0.1 s), and ``name`` (output
stream and debug prefix, default ``filter``).

Debug channel (``context.log``): ``<name>/<kind>`` = NIS and accepted (1/0)
for every innovation, and ``<name>/bias`` = accelerometer (m/s²) and gyro
(rad/s) biases after every accepted update when the state holds them.
"""

from collections.abc import Mapping
from typing import Any

from ...core.algorithms import AlgorithmContext
from ...core.measurements import EstimatedState, Sample
from ..interface import Innovation
from .measurements import CONVERTER_DEFAULTS
from .node import DEFAULT_MAX_DRIVER_INTERVAL_S, FilterNode, MeasurementOutcome
from .protocol import RecursiveFilter, predict_kind, update_kinds

ESTIMATOR_KEYS = {"filter", "filter_options", "name", "max_driver_interval_s"}


class FilterEstimator:
    """Bus plug-in around ``FilterNode`` (see module docs)."""

    def __init__(self, context: AlgorithmContext, options: Mapping[str, Any]) -> None:
        filter_ = _build_filter(options)
        self.name = str(options.get("name", "filter"))
        driver = predict_kind(filter_)
        supported = update_kinds(filter_)
        kinds = tuple(kind for kind in supported if kind in context.sensors)
        if driver not in context.sensors:
            raise ValueError(f"{self.name} needs the {driver} sensor enabled")
        if not kinds:
            raise ValueError(
                f"{self.name} needs one of the {', '.join(supported)} sensors enabled"
            )
        self.inputs: tuple[str, ...] = (driver, *kinds)
        self.node = FilterNode(
            filter_,
            name=self.name,
            gps_reference=context.gps_reference,
            sensors=context.sensors,
            wheeled=context.wheeled,
            converter_options=_converter_options(options),
            max_driver_interval_s=float(
                options.get("max_driver_interval_s", DEFAULT_MAX_DRIVER_INTERVAL_S)
            ),
            updates=kinds,
        )
        self._latest: Sample[EstimatedState] | None = None
        self._innovations: list[Innovation] = []
        self._log = context.log
        self.updates = {kind: 0 for kind in kinds}
        self.outliers = {kind: 0 for kind in kinds}
        # Simulation time (s) of each stream's initialization (an accepted
        # measurement without innovation), e.g. position from the first fix.
        self.initialized_s: dict[str, float] = {}

    @property
    def filter(self) -> Any:
        return self.node.filter

    def on_sample(self, kind: str, sample: Sample[Any]) -> None:
        if kind == self.node.driver:
            self._latest = self.node.on_driver(sample)
            return
        outcome = self.node.on_measurement(kind, sample)
        self._record(kind, sample.sim_time_ns, outcome)

    def _record(self, kind: str, sim_time_ns: int, outcome: MeasurementOutcome) -> None:
        """Innovations, counters, debug series, and the newest estimate."""
        innovation = outcome.innovation
        if outcome.accepted and innovation is None:
            self.initialized_s.setdefault(kind, sim_time_ns * 1e-9)
        if innovation is not None:
            self._innovations.append(innovation)
            (self.updates if innovation.accepted else self.outliers)[kind] += 1
            self._log(
                f"{self.name}/{innovation.name}",
                sim_time_ns,
                [innovation.nis, float(innovation.accepted)],
                ["nis", "accepted"],
            )
        if outcome.estimate is None:
            return
        self._latest = outcome.estimate
        state = outcome.estimate.payload
        if state.accel_bias_mps2 is not None and state.gyro_bias_rad_s is not None:
            self._log(
                f"{self.name}/bias",
                sim_time_ns,
                [*state.accel_bias_mps2, *state.gyro_bias_rad_s],
                ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
            )

    def latest(self) -> Sample[EstimatedState] | None:
        return self._latest

    def drain_innovations(self) -> list[Innovation]:
        innovations, self._innovations = self._innovations, []
        return innovations

    def summary(self) -> dict[str, Any]:
        filter_summary = getattr(self.filter, "summary", None)
        return dict(
            filter=type(self.filter).__name__,
            inputs=list(self.inputs),
            updates=self.updates,
            outliers=self.outliers,
            rejected_timing={k: self.node.rejected_timing[k] for k in self.inputs[1:]},
            initialized_s=self.initialized_s,
            receiver_covariance_fixes=self.node.converter.receiver_covariance_fixes,
            filter_summary=filter_summary() if callable(filter_summary) else None,
            initialization="declared filter options; no simulator truth",
        )


def _build_filter(options: Mapping[str, Any]) -> Any:
    """``cls(filter_options)`` for ``options["filter"]``; raises ValueError
    for a missing or invalid class."""
    unknown = set(options) - ESTIMATOR_KEYS - set(CONVERTER_DEFAULTS)
    if unknown:
        raise ValueError(f"filter estimator: unknown options {sorted(unknown)}")
    spec = options.get("filter")
    if not spec:
        raise ValueError(
            'filter estimator needs option "filter": "package.module:Class"'
        )
    filter_options = options.get("filter_options", {})
    if not isinstance(filter_options, Mapping):
        raise ValueError("filter_options must be an object")
    filter_ = _load_class(str(spec))(dict(filter_options))
    if not isinstance(filter_, RecursiveFilter):
        raise ValueError(f"{spec} does not implement RecursiveFilter (state)")
    predict_kind(filter_)  # exactly one driving stream, or ValueError
    if not update_kinds(filter_):
        raise ValueError(f"{spec} has no update_<kind> method: nothing corrects it")
    return filter_


def _load_class(spec: str) -> Any:
    """Import ``package.module:Class`` (or ``Class.factory``); raises
    ValueError with the reason."""
    import importlib

    module_name, _, class_name = spec.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"filter {spec!r} must be 'package.module:Class'")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(
            f"Cannot import filter module {module_name!r}: {error}"
        ) from error
    target: Any = module
    try:
        for attribute in class_name.split("."):
            target = getattr(target, attribute)
    except AttributeError:
        raise ValueError(f"{module_name} has no class {class_name!r}") from None
    return target


def _converter_options(options: Mapping[str, Any]) -> dict[str, Any]:
    return {k: options[k] for k in CONVERTER_DEFAULTS if k in options}
