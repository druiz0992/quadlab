"""The interface an inertial filter implements to run in ``InertialNode``.

A filter owns only its mathematics; the node calls it with explicit,
converted values (estimation/inertial/measurements.py):

- ``predict(specific_force_mps2, angular_rate_rad_s, dt_s)``: propagate by
  ``dt_s`` seconds with one IMU reading in **body** axes (specific force
  m/s², angular rate rad/s), held constant over the interval;
- ``state()``: the current ``EstimatedState`` (ENU position/velocity, xyzw
  body-to-world quaternion, body-axis biases, covariance over the present
  blocks with RobotLab's error convention: right rotation-vector attitude
  error, see core/measurements.py; convert if the filter uses another);
- ``update_<kind>(measurement)`` for each fused stream, e.g.
  ``update_gps(GpsFix)``, ``update_magnetometer(MagneticField)``,
  ``update_baro(BaroAltitude)``. Return the ``Innovation`` (NIS, degrees of
  freedom, accepted) of the correction, or None when the measurement was
  used to initialize rather than correct the state (no innovation).

Which ``update_<kind>`` methods exist decides which streams the node
accepts and the estimator subscribes to. Raise ValueError for invalid
inputs; never use simulator truth.
"""

from typing import Any, Protocol, runtime_checkable

from numpy.typing import ArrayLike

from ...core.measurements import EstimatedState

UPDATE_PREFIX = "update_"


@runtime_checkable
class InertialFilter(Protocol):
    """See module docs; ``update_<kind>`` methods are found by name."""

    def predict(
        self, specific_force_mps2: ArrayLike, angular_rate_rad_s: ArrayLike, dt_s: float
    ) -> None: ...

    def state(self) -> EstimatedState: ...


def update_kinds(filter_: Any) -> tuple[str, ...]:
    """Stream kinds ``filter_`` fuses, from its ``update_<kind>`` methods."""
    return tuple(
        sorted(
            name[len(UPDATE_PREFIX) :]
            for name in dir(filter_)
            if name.startswith(UPDATE_PREFIX) and callable(getattr(filter_, name))
        )
    )
