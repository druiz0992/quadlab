"""Wheel-odometry template: kinematic solve, propagation, GPS, estimator wiring."""

import numpy as np
import pytest

from robotlab.estimation.filter.estimator import FilterEstimator
from robotlab.estimation.filter.measurements import GpsFix, WheelMotion
from robotlab.estimation.filter.wheel_odometry import (
    TemplateWheelOdometry,
    body_velocity,
)
from robotlab.testing.scenarios import SCENARIOS, scenario_context

HUBS = np.array([[1.5, 0.8], [1.5, -0.8], [-1.3, 0.8], [-1.3, -0.8]])
FILTER = "robotlab.estimation.filter.wheel_odometry:TemplateWheelOdometry"


def rolling(
    velocity: tuple[float, float, float], steered: tuple[bool, ...]
) -> WheelMotion:
    """Encoder view of a rigid planar motion (v_x, v_y, ω): steered wheels
    point along their hub velocity, unsteered wheels straight ahead."""
    vx, vy, omega = velocity
    hub_velocity = np.column_stack([vx - omega * HUBS[:, 1], vy + omega * HUBS[:, 0]])
    angle = np.where(steered, np.arctan2(hub_velocity[:, 1], hub_velocity[:, 0]), 0.0)
    direction = np.column_stack([np.cos(angle), np.sin(angle)])
    return WheelMotion(HUBS, direction, np.sum(hub_velocity * direction, axis=1))


def test_body_velocity_recovers_rigid_motion_with_steering() -> None:
    truth = (5.0, 0.3, 0.4)
    motion = rolling(truth, (True, True, False, False))
    np.testing.assert_allclose(body_velocity(motion, 0.0), truth, atol=1e-9)


def test_skid_steering_leaves_unobservable_side_speed_at_zero() -> None:
    motion = rolling((2.0, 0.5, 0.3), (False,) * 4)
    vx, vy, omega = body_velocity(motion, 0.0)
    assert (vx, omega) == pytest.approx((2.0, 0.3)) and vy == pytest.approx(0.0)


def test_straight_drive_integrates_along_the_heading() -> None:
    odometry = TemplateWheelOdometry({"initial_yaw_deg": 90.0})
    for _ in range(100):
        odometry.predict_wheel_encoders(rolling((2.0, 0.0, 0.0), (True,) * 4), 0.01)
    state = odometry.state()
    assert state.position_m is not None and state.velocity_mps is not None
    np.testing.assert_allclose(state.position_m, [0, 2, 0], atol=1e-9)
    np.testing.assert_allclose(state.velocity_mps, [0, 2, 0], atol=1e-9)


def test_gps_correction_pulls_position_and_reports_innovation() -> None:
    odometry = TemplateWheelOdometry({"initial_position_std_m": 2.0})
    fix = GpsFix(np.array([1.0, 0.0, 0.0]), np.eye(3) * 0.01, np.zeros(3))
    innovation = odometry.update_gps(fix)
    assert innovation.accepted and innovation.dof == 3
    position = odometry.state().position_m
    assert position is not None and position[0] == pytest.approx(1.0, abs=0.01)


def test_options_are_validated() -> None:
    with pytest.raises(ValueError, match="unknown options"):
        TemplateWheelOdometry({"gain": 1})
    with pytest.raises(ValueError, match="positive"):
        TemplateWheelOdometry({"speed_std_mps": 0})


def test_estimator_subscribes_to_encoders_and_needs_them() -> None:
    scenario = SCENARIOS["car_circle"]
    context = scenario_context(scenario, ("wheel_encoders", "gps"))
    estimator = FilterEstimator(context, {"filter": FILTER})
    assert estimator.inputs == ("wheel_encoders", "gps")
    with pytest.raises(ValueError, match="needs the wheel_encoders sensor"):
        FilterEstimator(scenario_context(scenario, ("gps",)), {"filter": FILTER})
    with pytest.raises(ValueError, match="max_driver_interval_s"):
        FilterEstimator(context, {"filter": FILTER, "max_driver_interval_s": 0})
