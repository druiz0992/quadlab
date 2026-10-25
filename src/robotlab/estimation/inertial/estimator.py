"""``StateEstimator`` plug-in for any ``InertialFilter`` (``inertial``).

Select a filter class with::

    "algorithms": {"estimator": "inertial",
                   "estimator_options": {"filter": "mypkg.my_ekf:MyEkf",
                                         "filter_options": {...},
                                         "gps_std_m": [0.5, 0.5, 0.8]}}

The filter is built as ``cls(filter_options)`` and must implement
``InertialFilter`` (filter.py). The estimator subscribes to ``imu`` plus
every **enabled** stream the filter has an ``update_<kind>`` for (listed in
``summary()["inputs"]``); it raises ValueError without the IMU or without
any enabled stream to correct with. Mounts and the ENU origin come from the context. The remaining
options are the measurement converter's (``CONVERTER_DEFAULTS``,
measurements.py) and ``name`` (output stream and debug prefix, default
``inertial``).

Debug channel (``context.log``): ``<name>/<kind>`` = NIS and accepted (1/0)
for every innovation, and ``<name>/bias`` = accelerometer (m/s²) and gyro
(rad/s) biases after every accepted update when the state holds them.
"""

from collections.abc import Mapping
from typing import Any

from ...core.algorithms import AlgorithmContext
from ...core.measurements import EstimatedState, Sample
from ..interface import Innovation
from .filter import InertialFilter, update_kinds
from .measurements import CONVERTER_DEFAULTS
from .node import InertialNode, MeasurementOutcome

ESTIMATOR_KEYS = {"filter", "filter_options", "name"}


class InertialEstimator:
    """Bus plug-in around ``InertialNode`` (see module docs).

    Subclasses (e.g. the GPS/IMU EKF) pass a ready ``filter_`` and their
    converter options instead of ``filter``/``filter_options``.
    """

    def __init__(
        self,
        context: AlgorithmContext,
        options: Mapping[str, Any],
        filter_: Any | None = None,
        *,
        name: str | None = None,
        log_prefix: str | None = None,
    ) -> None:
        if filter_ is None:
            filter_ = _build_filter(options)
            converter_options = _converter_options(options)
        else:
            converter_options = {
                k: options[k] for k in CONVERTER_DEFAULTS if k in options
            }
        self.name = name or str(options.get("name", "inertial"))
        self.log_prefix = log_prefix or self.name
        supported = update_kinds(filter_)
        kinds = tuple(kind for kind in supported if kind in context.sensors)
        if "imu" not in context.sensors:
            raise ValueError(f"{self.name} needs the imu sensor enabled")
        if not kinds:
            raise ValueError(
                f"{self.name} needs one of the {', '.join(supported)} sensors enabled"
            )
        self.inputs: tuple[str, ...] = ("imu", *kinds)
        self.node = InertialNode(
            filter_,
            name=self.name,
            gps_reference=context.gps_reference,
            sensors=context.sensors,
            converter_options=converter_options,
        )
        self._latest: Sample[EstimatedState] | None = None
        self._innovations: list[Innovation] = []
        self._log = context.log
        self.updates = {kind: 0 for kind in kinds}
        self.outliers = {kind: 0 for kind in kinds}

    @property
    def filter(self) -> Any:
        return self.node.filter

    def on_sample(self, kind: str, sample: Sample[Any]) -> None:
        if kind == "imu":
            self._latest = self.node.on_imu(sample)
            return
        outcome = self.node.on_measurement(kind, sample)
        self._record(kind, sample.sim_time_ns, outcome)

    def _record(self, kind: str, sim_time_ns: int, outcome: MeasurementOutcome) -> None:
        """Innovations, counters, debug series, and the newest estimate."""
        innovation = outcome.innovation
        if innovation is not None:
            self._innovations.append(innovation)
            (self.updates if innovation.accepted else self.outliers)[kind] += 1
            self._log(
                f"{self.log_prefix}/{innovation.name}",
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
                f"{self.log_prefix}/bias",
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
            receiver_covariance_fixes=self.node.converter.receiver_covariance_fixes,
            filter_summary=filter_summary() if callable(filter_summary) else None,
            initialization="declared filter options; no simulator truth",
        )


def _build_filter(options: Mapping[str, Any]) -> Any:
    """``cls(filter_options)`` for ``options["filter"]``; raises ValueError
    for a missing or invalid class."""
    unknown = set(options) - ESTIMATOR_KEYS - set(CONVERTER_DEFAULTS)
    if unknown:
        raise ValueError(f"inertial estimator: unknown options {sorted(unknown)}")
    spec = options.get("filter")
    if not spec:
        raise ValueError(
            'inertial estimator needs option "filter": "package.module:Class"'
        )
    filter_options = options.get("filter_options", {})
    if not isinstance(filter_options, Mapping):
        raise ValueError("filter_options must be an object")
    filter_ = _load_class(str(spec))(dict(filter_options))
    if not isinstance(filter_, InertialFilter):
        raise ValueError(f"{spec} does not implement InertialFilter (predict, state)")
    if not update_kinds(filter_):
        raise ValueError(f"{spec} has no update_<kind> method: nothing corrects it")
    return filter_


def _load_class(spec: str) -> type:
    """Import ``package.module:Class``; raises ValueError with the reason."""
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
    try:
        return getattr(module, class_name)
    except AttributeError:
        raise ValueError(f"{module_name} has no class {class_name!r}") from None


def _converter_options(options: Mapping[str, Any]) -> dict[str, Any]:
    return {k: options[k] for k in CONVERTER_DEFAULTS if k in options}
