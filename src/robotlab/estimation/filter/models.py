"""Reusable measurement models: residual, Jacobian, and noise by state block.

Each function linearizes one converted measurement (measurements.py) about
a filter's nominal state and returns a ``LinearizedMeasurement`` for
``kalman_update`` (kalman.py). Jacobian columns are placed by name through
the filter's ``StateLayout`` (state_layout.py), so one model serves any
filter that has the blocks it needs, whatever else the state holds. The
nominal values are passed explicitly (``rotation_matrix`` is
body-to-world); a block the measurement depends on but the layout lacks is
treated as known (zero columns), except the required ones named in each
function, whose absence raises ValueError.

Rotation columns follow the layout's error convention: ``attitude`` is a
right body-axes rotation vector, R_true = R · Exp(δθ), so for a body
vector b, R_true b ≈ R b − R [b]× δθ; ``yaw`` is a heading error about
world up, R_true = R_z(δψ) · R, so R_true b ≈ R b + (e_z × R b) δψ
(Solà et al. 2018, see estimation/lie.py).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..lie import skew
from .measurements import BaroAltitude, GpsFix, MagneticField
from .state_layout import StateLayout

UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class LinearizedMeasurement:
    """Residual r = z − h(x̂) (m entries), Jacobian H = ∂h/∂δx (m × n),
    and measurement covariance R (m × m), in the measurement's units.
    Raises ValueError for inconsistent shapes."""

    name: str
    residual: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    covariance: NDArray[np.float64]

    def __post_init__(self) -> None:
        rows = self.residual.shape
        if (
            len(rows) != 1
            or self.jacobian.ndim != 2
            or self.jacobian.shape[0] != rows[0]
            or self.covariance.shape != (rows[0], rows[0])
        ):
            raise ValueError(f"{self.name}: inconsistent measurement shapes")


def gps_position(
    layout: StateLayout,
    fix: GpsFix,
    *,
    position_m: ArrayLike,
    rotation_matrix: NDArray[np.float64],
) -> LinearizedMeasurement:
    """Antenna position z = p + R l (ENU, m) with lever arm l (body, m).

    Needs a ``position`` block; uses ``attitude`` or ``yaw`` when present.
    """
    H = layout.jacobian(3)
    H[:, layout.index("position")] = np.eye(3)
    lever_world = rotation_matrix @ fix.lever_arm_body_m
    _rotate_world_vector(layout, H, rotation_matrix, fix.lever_arm_body_m)
    residual = fix.position_enu_m - np.asarray(position_m, float) - lever_world
    return LinearizedMeasurement("gps", residual, H, fix.covariance_m2)


def gps_velocity(
    layout: StateLayout,
    fix: GpsFix,
    *,
    velocity_mps: ArrayLike,
    rotation_matrix: NDArray[np.float64],
    angular_rate_body_rad_s: ArrayLike = (0.0, 0.0, 0.0),
) -> LinearizedMeasurement:
    """Antenna velocity z = v + R (ω × l) (ENU, m/s), with the body rate ω
    (rad/s, bias-corrected by the caller) treated as known.

    Needs a ``velocity`` block and a fix with velocity and its variance
    (ValueError otherwise); uses ``attitude`` or ``yaw`` when present.
    """
    if fix.velocity_enu_mps is None or fix.velocity_variance_m2_s2 is None:
        raise ValueError("gps_velocity needs a fix with velocity and its accuracy")
    H = layout.jacobian(3)
    H[:, layout.index("velocity")] = np.eye(3)
    rate = np.asarray(angular_rate_body_rad_s, float)
    lever_rate_body = np.cross(rate, fix.lever_arm_body_m)
    _rotate_world_vector(layout, H, rotation_matrix, lever_rate_body)
    residual = (
        fix.velocity_enu_mps
        - np.asarray(velocity_mps, float)
        - rotation_matrix @ lever_rate_body
    )
    noise = np.eye(3) * fix.velocity_variance_m2_s2
    return LinearizedMeasurement("gps_velocity", residual, H, noise)


def baro_altitude(
    layout: StateLayout, baro: BaroAltitude, *, position_m: ArrayLike
) -> LinearizedMeasurement:
    """Altitude above the reference z = p_up (m). Needs a ``position`` block.

    Datum and drift offsets are not modelled; a filter that estimates one
    adds its own column (z = p_up + b) to the returned Jacobian.
    """
    H = layout.jacobian(1)
    H[0, layout.index("position").start + 2] = 1.0
    residual = np.array([baro.altitude_m - float(np.asarray(position_m, float)[2])])
    return LinearizedMeasurement("baro", residual, H, np.array([[baro.variance_m2]]))


def magnetometer_field(
    layout: StateLayout,
    field: MagneticField,
    *,
    rotation_matrix: NDArray[np.float64],
    world_field_ut: ArrayLike,
) -> LinearizedMeasurement:
    """Body-axes field z = Rᵀ m (µT) for the expected world field m (µT,
    ENU, the filter's own knowledge). Needs an ``attitude`` or ``yaw``
    block. Sensor biases are not modelled (calibrate first, or add a bias
    column)."""
    world_field = np.asarray(world_field_ut, float)
    predicted = rotation_matrix.T @ world_field
    H = layout.jacobian(3)
    if "attitude" in layout:
        # Exp(δθ)ᵀ Rᵀ m ≈ Rᵀ m + [Rᵀ m]× δθ.
        H[:, layout.index("attitude")] = skew(predicted)
    elif "yaw" in layout:
        # Rᵀ R_z(δψ)ᵀ m ≈ Rᵀ m − Rᵀ (e_z × m) δψ.
        column = -rotation_matrix.T @ np.cross(UP, world_field)
        H[:, layout.index("yaw").start] = column
    else:
        raise ValueError("magnetometer_field needs an attitude or yaw block")
    residual = field.field_body_ut - predicted
    return LinearizedMeasurement("magnetometer", residual, H, field.covariance_ut2)


def _rotate_world_vector(
    layout: StateLayout,
    H: NDArray[np.float64],
    rotation_matrix: NDArray[np.float64],
    body_vector: NDArray[np.float64],
) -> None:
    """Add the columns of R b (b in body axes) for the layout's rotation
    block, if any (module docs)."""
    if "attitude" in layout:
        H[:, layout.index("attitude")] = -rotation_matrix @ skew(body_vector)
    elif "yaw" in layout:
        H[:, layout.index("yaw").start] = np.cross(UP, rotation_matrix @ body_vector)
