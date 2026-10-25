"""The interface a recursive filter implements to run in ``FilterNode``.

A filter owns only its mathematics; the node calls it with explicit,
converted values (estimation/filter/measurements.py), never streams,
frames, or timestamps:

- exactly one ``predict_<kind>(measurement, dt_s)``: the **driving**
  stream, e.g. ``predict_imu(ImuInput, dt_s)`` for an inertial filter or
  ``predict_wheel_encoders(WheelMotion, dt_s)`` for wheel odometry. Each
  reading is held from its sample time, and the filter propagates by
  ``dt_s`` seconds with it, up to the next reading or measurement;
- ``update_<kind>(measurement)`` for each fused stream, e.g.
  ``update_gps(GpsFix)``, ``update_magnetometer(MagneticField)``,
  ``update_baro(BaroAltitude)``. Return the ``Innovation`` (NIS, degrees of
  freedom, accepted) of the correction, or None when the measurement was
  used to initialize rather than correct the state (no innovation);
- ``state()``: the current ``EstimatedState`` (ENU position/velocity, xyzw
  body-to-world quaternion, body-axis biases, covariance over the present
  blocks with RobotLab's error convention: right rotation-vector attitude
  error, see core/measurements.py; convert if the filter uses another);
- optionally ``summary()``: a dict for ``summary.json``.

Which methods exist decides which streams the node accepts and the
estimator subscribes to; the driving stream cannot also be an update. Raise
ValueError for invalid inputs; never use simulator truth.
"""

from typing import Any, Protocol, runtime_checkable

from ...core.measurements import EstimatedState

PREDICT_PREFIX = "predict_"
UPDATE_PREFIX = "update_"


@runtime_checkable
class RecursiveFilter(Protocol):
    """See module docs; ``predict_<kind>``/``update_<kind>`` are found by
    name."""

    def state(self) -> EstimatedState: ...


def _kinds(filter_: Any, prefix: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            name[len(prefix) :]
            for name in dir(filter_)
            if name.startswith(prefix) and callable(getattr(filter_, name))
        )
    )


def update_kinds(filter_: Any) -> tuple[str, ...]:
    """Stream kinds ``filter_`` fuses, from its ``update_<kind>`` methods."""
    return _kinds(filter_, UPDATE_PREFIX)


def predict_kind(filter_: Any) -> str:
    """The driving stream, from ``filter_``'s single ``predict_<kind>``.

    Raises ValueError for none (naming the old IMU signature when the filter
    has a bare ``predict``), several, or one that is also an update kind.
    """
    name = type(filter_).__name__
    kinds = _kinds(filter_, PREDICT_PREFIX)
    if not kinds:
        hint = (
            ": rename predict(specific_force, angular_rate, dt_s) to "
            "predict_imu(imu: ImuInput, dt_s)"
            if callable(getattr(filter_, "predict", None))
            else ""
        )
        raise ValueError(f"{name} has no {PREDICT_PREFIX}<kind> method{hint}")
    if len(kinds) > 1:
        raise ValueError(f"{name} has several driving streams: {list(kinds)}")
    if kinds[0] in update_kinds(filter_):
        raise ValueError(f"{name} both predicts and updates with {kinds[0]!r}")
    return kinds[0]
