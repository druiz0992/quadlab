"""Baseline quadplane autopilot: mode logic, guidance and surface signs."""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.control.interface import ControlState
from robotlab.control.quadplane import QuadplaneController
from robotlab.control.references import ReferencePoint
from robotlab.core.algorithms import AlgorithmContext
from robotlab.core.measurements import GpsReading
from robotlab.simulation.gazebo.config import load_drone_config

HOME = ReferencePoint(np.zeros(3), np.zeros(3), np.zeros(3), 0.0)


def controller(**options):
    config = load_drone_config("configs/vehicles/kite_quadplane.json")
    context = AlgorithmContext(GpsReading(40, -3, 650), config.properties)
    return QuadplaneController(
        context, dict(dict(hover=config.data["controller"]), **options)
    )


def state(
    t_s, position, velocity=(0, 0, 0), airspeed=0.0, yaw=0.0, roll=0.0, pitch_up=0.0
):
    q = Rotation.from_euler("ZYX", [yaw, -pitch_up, roll]).as_quat()
    return ControlState(
        round(t_s * 1e9),
        np.array(position, float),
        np.array(velocity, float),
        q,
        np.zeros(3),
        airspeed_mps=airspeed,
    )


def test_hover_then_transition_after_settling_at_altitude():
    c = controller(cruise_altitude_m=40.0, settle_s=2.0)
    c.update_position(state(0, [0, 0, 2]), HOME, 0.02)
    assert c.mode == "hover" and c.setpoint_m[2] > 2
    for t in np.arange(10, 13, 0.1):
        c.update_position(state(t, [0, 0, 40]), HOME, 0.1)
    assert c.mode == "transition"
    thrust = c.update_attitude(state(13, [0, 0, 40]))
    assert (
        thrust[4] == pytest.approx(c.o["transition_thrust_n"])
        and thrust[:4].sum() > 150
    )
    c.update_position(state(20, [100, 0, 40], [19.5, 0, 0], airspeed=19.5), HOME, 0.02)
    assert c.mode == "cruise"
    # Lift rotors fade out over lift_fade_s, then stay off.
    c.update_position(state(23, [150, 0, 40], [20, 0, 0], airspeed=20), HOME, 0.02)
    assert np.all(
        c.update_attitude(state(23, [150, 0, 40], [20, 0, 0], airspeed=20))[:4] == 0
    )


def cruising(**options):
    c = controller(**options)
    c.mode, c._cruise_start, c._home = "cruise", -100.0, np.zeros(3)
    c._previous_wp = np.zeros(2)
    return c


def test_l1_turns_toward_a_target_on_the_left():
    c = cruising(route_m=[[0, 300]])  # flying east, target due north
    c.update_position(state(0, [0, 0, 40], [22, 0, 0], airspeed=22), HOME, 0.02)
    assert c._roll_cmd < 0  # left turn: left wing down
    c = cruising(route_m=[[0, -300]])
    c.update_position(state(0, [0, 0, 40], [22, 0, 0], airspeed=22), HOME, 0.02)
    assert c._roll_cmd > 0


def test_surface_signs_correct_attitude_and_speed_errors():
    c = cruising(route_m=[[3000, 0]])
    c.update_position(state(0, [0, 0, 30], [22, 0, 0], airspeed=18), HOME, 0.02)
    assert c._pitch_cmd > math.radians(c.o["trim_pitch_deg"])  # below altitude: climb
    assert c._pusher > c.o["trim_thrust_n"]  # slow: more thrust
    c.update_attitude(state(0, [0, 0, 30], [22, 0, 0], airspeed=18))
    assert c.surface_commands()["elevator"] < 0  # nose up (trailing edge up)
    c.update_attitude(state(0, [0, 0, 40], [22, 0, 0], airspeed=22, roll=0.3))
    assert c.surface_commands()["aileron"] < 0  # right wing down: roll back left


def test_l1_turns_around_for_a_target_behind():
    c = cruising(route_m=[[-500, 1]])  # flying east, target (almost) dead astern
    c._previous_wp = np.array([0.0, 0.0])
    c.update_position(state(0, [0, 0, 40], [22, 0, 0], airspeed=22), HOME, 0.02)
    assert c._roll_cmd == pytest.approx(-math.radians(c.o["max_roll_deg"]))


def test_tracking_reference_is_the_closest_point_on_the_current_segment():
    q = Rotation.identity().as_quat()
    c = controller(cruise_altitude_m=40.0, route_m=[[300, 0], [300, 300]])
    assert c.tracking_reference(np.zeros(3), np.zeros(3), q) is None  # no home yet
    c.update_position(state(0, [0, 0, 2]), HOME, 0.02)
    # Hover climb: vertical line above home; only horizontal offset counts.
    ref = c.tracking_reference(np.array([0.5, 0, 10]), np.array([0, 0, 2.0]), q)
    np.testing.assert_allclose(ref.position_m, [0, 0, 10])
    np.testing.assert_allclose(ref.velocity_mps, [0, 0, 2])
    # Cruise on the first leg, 3 m north of it and 1 m low: projection.
    c.mode, c._previous_wp, c._waypoint = "cruise", np.array([100.0, 0]), 0
    ref = c.tracking_reference(np.array([150, 3, 39]), np.array([22, 1, 0.0]), q)
    np.testing.assert_allclose(ref.position_m, [150, 0, 40])
    np.testing.assert_allclose(ref.velocity_mps, [22, 0, 0])  # along-leg only
    # Past the end of the leg the reference stays at the waypoint.
    ref = c.tracking_reference(np.array([320, 0, 40]), np.zeros(3), q)
    np.testing.assert_allclose(ref.position_m, [300, 0, 40])
    # Back-transition: altitude only.
    c.mode = "back_transition"
    ref = c.tracking_reference(np.array([5, 6, 38]), np.array([3, 4, 1.0]), q)
    np.testing.assert_allclose(ref.position_m, [5, 6, 40])
    np.testing.assert_allclose(ref.velocity_mps, [3, 4, 0])
    # Return: straight line from where it began to home.
    c.mode, c._return_start = "return", np.array([0.0, 300])
    ref = c.tracking_reference(np.array([2, 150, 40]), np.zeros(3), q)
    np.testing.assert_allclose(ref.position_m, [0, 150, 40])
    assert c.mode == "return" and c._waypoint == 0  # state untouched
