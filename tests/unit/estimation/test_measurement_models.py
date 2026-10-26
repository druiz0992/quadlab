"""Shared measurement models: residuals and Jacobians against finite
differences of the measurement function under each error convention."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from robotlab.estimation.filter.measurements import BaroAltitude, GpsFix, MagneticField
from robotlab.estimation.filter.models import (
    LinearizedMeasurement,
    baro_altitude,
    gps_position,
    gps_velocity,
    magnetometer_field,
)
from robotlab.estimation.filter.state_layout import StateLayout

LEVER = np.array([0.2, -0.1, 0.3])
RATE = np.array([0.4, -0.2, 0.7])
WORLD_FIELD = np.array([0.0, 22.0, -42.0])
FULL = StateLayout.of("position", "velocity", "attitude", "gyro_bias")
PLANAR = StateLayout.of("position", "yaw")


@dataclass(frozen=True)
class Nominal:
    """Nominal state the models linearize about."""

    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    rotation: NDArray[np.float64]


NOMINAL = Nominal(
    np.array([1.0, 2.0, 3.0]),
    np.array([0.5, -1.0, 0.2]),
    Rotation.from_euler("xyz", [0.2, -0.3, 1.0]).as_matrix(),
)


def perturbed(layout: StateLayout, dx: NDArray[np.float64]) -> Nominal:
    """NOMINAL ⊕ dx with the layout's conventions (state_layout.py)."""
    x = NOMINAL
    position = x.position + (
        dx[layout.index("position")] if "position" in layout else 0
    )
    velocity = x.velocity + (
        dx[layout.index("velocity")] if "velocity" in layout else 0
    )
    rotation = x.rotation
    if "attitude" in layout:
        rotation = (
            rotation @ Rotation.from_rotvec(dx[layout.index("attitude")]).as_matrix()
        )
    if "yaw" in layout:
        yaw = Rotation.from_euler("z", dx[layout.index("yaw")][0]).as_matrix()
        rotation = yaw @ rotation
    return Nominal(position, velocity, rotation)


def numeric_jacobian(
    layout: StateLayout, h: Callable[[Nominal], NDArray[np.float64]]
) -> NDArray[np.float64]:
    """Central differences of h(NOMINAL ⊕ dx) at dx = 0."""
    step = 1e-6
    columns = []
    for i in range(layout.size):
        dx = np.zeros(layout.size)
        dx[i] = step
        columns.append(
            (h(perturbed(layout, dx)) - h(perturbed(layout, -dx))) / (2 * step)
        )
    return np.column_stack(columns)


def check(
    layout: StateLayout,
    z: LinearizedMeasurement,
    observed: NDArray[np.float64],
    h: Callable[[Nominal], NDArray[np.float64]],
) -> None:
    np.testing.assert_allclose(z.residual, observed - h(NOMINAL), atol=1e-12)
    np.testing.assert_allclose(z.jacobian, numeric_jacobian(layout, h), atol=1e-8)


def gps_fix() -> GpsFix:
    return GpsFix(
        position_enu_m=np.array([1.5, 2.5, 2.0]),
        covariance_m2=np.eye(3) * 0.25,
        lever_arm_body_m=LEVER,
        velocity_enu_mps=np.array([0.4, -0.8, 0.0]),
        velocity_variance_m2_s2=0.01,
    )


@pytest.mark.parametrize("layout", [FULL, PLANAR, StateLayout.of("position")])
def test_gps_position(layout: StateLayout) -> None:
    fix = gps_fix()
    z = gps_position(
        layout, fix, position_m=NOMINAL.position, rotation_matrix=NOMINAL.rotation
    )
    check(layout, z, fix.position_enu_m, lambda x: x.position + x.rotation @ LEVER)
    assert z.name == "gps" and z.covariance is fix.covariance_m2


@pytest.mark.parametrize("layout", [FULL, StateLayout.of("velocity", "yaw")])
def test_gps_velocity(layout: StateLayout) -> None:
    fix = gps_fix()
    z = gps_velocity(
        layout,
        fix,
        velocity_mps=NOMINAL.velocity,
        rotation_matrix=NOMINAL.rotation,
        angular_rate_body_rad_s=RATE,
    )
    assert fix.velocity_enu_mps is not None
    check(
        layout,
        z,
        fix.velocity_enu_mps,
        lambda x: x.velocity + x.rotation @ np.cross(RATE, LEVER),
    )
    np.testing.assert_allclose(z.covariance, np.eye(3) * 0.01)


def test_baro_altitude() -> None:
    z = baro_altitude(FULL, BaroAltitude(2.5, 0.09), position_m=NOMINAL.position)
    check(FULL, z, np.array([2.5]), lambda x: x.position[2:])
    assert z.covariance[0, 0] == 0.09


@pytest.mark.parametrize("layout", [FULL, PLANAR])
def test_magnetometer_field(layout: StateLayout) -> None:
    field = MagneticField(np.array([10.0, 20.0, -40.0]), np.eye(3))
    z = magnetometer_field(
        layout, field, rotation_matrix=NOMINAL.rotation, world_field_ut=WORLD_FIELD
    )
    check(layout, z, field.field_body_ut, lambda x: x.rotation.T @ WORLD_FIELD)


def test_models_raise_for_missing_blocks_or_data() -> None:
    fix = gps_fix()
    with pytest.raises(ValueError, match="No state block 'position'"):
        gps_position(
            StateLayout.of("velocity"),
            fix,
            position_m=NOMINAL.position,
            rotation_matrix=NOMINAL.rotation,
        )
    no_velocity = GpsFix(fix.position_enu_m, fix.covariance_m2, LEVER)
    with pytest.raises(ValueError, match="velocity and its accuracy"):
        gps_velocity(
            FULL, no_velocity, velocity_mps=NOMINAL.velocity, rotation_matrix=np.eye(3)
        )
    with pytest.raises(ValueError, match="attitude or yaw"):
        magnetometer_field(
            StateLayout.of("position"),
            MagneticField(np.zeros(3), np.eye(3)),
            rotation_matrix=np.eye(3),
            world_field_ut=WORLD_FIELD,
        )
