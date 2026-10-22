"""Wheeled model on the reference rigid-body integrator (no Gazebo)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.core.wheeled import WheelCommands
from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.simulation.wheeled.ground import GroundModel
from robotlab.simulation.wheeled.model import (
    ChassisState,
    WheeledVehicleModel,
    cross,
    rotation_matrix,
)
from robotlab.simulation.wheeled.rigid_body import RigidBody
from robotlab.simulation.wheeled.spawn import spawn_pose

DT = 0.002


class Slope:
    """Duck-typed terrain: a plane rising along +x (height = grade·x)."""

    def __init__(self, grade):
        self.grade = grade

    def height(self, x, y):
        return self.grade * np.asarray(x, float) + 0 * np.asarray(y, float)

    def normals(self, x, y):
        n = np.array([-self.grade, 0.0, 1.0]) / np.hypot(self.grade, 1.0)
        return np.tile(n, (np.size(x), 1))


def make(kind, terrain=None, yaw=0.0):
    vehicle = load_drone_config(robot=kind).wheeled
    ground = GroundModel(terrain, [], ground_half_m=5000.0)
    pose = spawn_pose(vehicle, ground, 0.0, 0.0, yaw)
    body = RigidBody.at_rest(
        vehicle.mass_kg, vehicle.inertia_kg_m2, pose.position_m, yaw
    )
    return vehicle, WheeledVehicleModel(vehicle, ground), body


def run(model, body, seconds, commands=None):
    if commands is not None:
        model.actuators.accept(commands, 0)
    for _ in range(int(round(seconds / DT))):
        body.step(model.step(body.state(), DT), DT)


def drive(vehicle, torque=0.0, brake=0.0, steer=0.0):
    return WheelCommands(
        tuple(vehicle.mask("driven") * torque),
        tuple(vehicle.mask("braked") * brake),
        tuple(vehicle.mask("steered") * steer),
    )


def test_rotation_helpers_match_scipy():
    q = Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_quat()
    np.testing.assert_allclose(
        rotation_matrix(q), Rotation.from_quat(q).as_matrix(), atol=1e-12
    )
    a, b = np.random.default_rng(0).normal(size=(2, 5, 3))
    np.testing.assert_allclose(cross(a, b), np.cross(a, b))


@pytest.mark.parametrize("kind", ["rover", "car", "tractor"])
def test_settles_to_static_sag(kind):
    vehicle, model, body = make(kind)
    run(model, body, 2.0)
    telemetry = model.telemetry
    assert telemetry.in_contact.all()
    assert telemetry.normal_load_n.sum() == pytest.approx(
        vehicle.mass_kg * 9.81, rel=1e-3
    )
    assert np.linalg.norm(body.velocity_world_mps) < 1e-3
    assert telemetry.compression_m.mean() == pytest.approx(
        vehicle.ride_drop_m, rel=0.02
    )


def rolling(model, body, speed_mps):
    """Settled, then rolling straight ahead at ``speed_mps``."""
    run(model, body, 0.5)
    body.velocity_world_mps = np.array([speed_mps, 0.0, 0.0])
    model.wheel_speed_rad_s[:] = speed_mps / model.properties.tyre.radius_m


@pytest.mark.parametrize("kind", ["rover", "car", "tractor"])
def test_full_throttle_holds_top_speed_and_brakes_stop(kind):
    vehicle, model, body = make(kind)
    top = vehicle.drive.max_wheel_speed_rad_s * vehicle.tyre.radius_m
    rolling(model, body, 0.9 * top)
    run(model, body, 2.0, drive(vehicle, vehicle.drive.max_torque_nm))
    speed = body.velocity_world_mps[0]
    # The car is drag-limited (~72 m/s) below its 81 m/s wheel-speed limit.
    assert 0.85 * top < speed <= top * 1.001
    assert abs(body.velocity_world_mps[1]) < 1e-6  # straight line
    run(model, body, 5.0, drive(vehicle, brake=vehicle.drive.max_brake_torque_nm))
    assert np.linalg.norm(body.velocity_world_mps) < 0.01


def test_braking_deceleration_bounded_by_friction():
    vehicle, model, body = make("rover")
    rolling(model, body, 1.0)
    v0 = body.velocity_world_mps[0]
    run(model, body, 0.1, drive(vehicle, brake=vehicle.drive.max_brake_torque_nm))
    deceleration = (v0 - body.velocity_world_mps[0]) / 0.1
    assert 0 < deceleration <= vehicle.tyre.friction_longitudinal * 9.81 * 1.05


def test_parked_on_slope_holds_with_brakes():
    vehicle, model, body = make("tractor", Slope(0.15))
    run(model, body, 2.0, drive(vehicle, brake=vehicle.drive.max_brake_torque_nm))
    start = body.position_m.copy()
    run(model, body, 2.0)
    assert np.linalg.norm(body.position_m[:2] - start[:2]) < 2e-3


def test_unbraked_rolls_down_slope():
    vehicle, model, body = make("car", Slope(0.1))
    run(model, body, 3.0)
    assert body.velocity_world_mps[0] < -0.3


def test_steady_cornering_within_friction_limit():
    vehicle, model, body = make("car")
    rolling(model, body, 20.0)
    run(model, body, 4.0, drive(vehicle, 150.0, steer=0.12))
    v = np.linalg.norm(body.velocity_world_mps[:2])
    yaw_rate = body.body_rates_rad_s[2]
    lateral = v * yaw_rate
    assert yaw_rate > 0  # positive steer turns left
    friction = vehicle.tyre.friction_lateral * 9.81
    assert 0 < lateral < friction * 2.0  # downforce adds grip at speed


def test_skid_steer_turns_on_the_spot():
    vehicle, model, body = make("rover")
    run(model, body, 1.0)
    side = np.sign(vehicle.hub_positions_m()[:, 1])
    torque = -side * vehicle.drive.max_torque_nm  # left back, right forward
    run(model, body, 3.0, WheelCommands(tuple(torque), (0,) * 4, (0,) * 4))
    assert body.body_rates_rad_s[2] > 0.2
    assert np.linalg.norm(body.position_m[:2]) < 0.1


def test_model_rejects_bad_step():
    _, model, body = make("car")
    with pytest.raises(ValueError):
        model.step(body.state(), 0.0)


def test_airborne_wheels_carry_no_load():
    vehicle, model, _ = make("car")
    state = ChassisState(
        np.array([0.0, 0.0, 5.0]), np.array([0.0, 0, 0, 1]), np.zeros(3), np.zeros(3)
    )
    wrench = model.step(state, DT)
    assert not model.telemetry.in_contact.any()
    np.testing.assert_allclose(wrench.force_world_n, 0.0)


def test_batched_vehicles_match_separate_models():
    vehicle, single_a, body_a = make("car")
    _, single_b, body_b = make("rover" if False else "car")
    body_b.position_m = body_b.position_m + np.array([10.0, 3.0, 0.0])
    ground = single_a.ground
    batched = WheeledVehicleModel(vehicle, ground, vehicles=2)
    a, b = drive(vehicle, 300.0, steer=0.1), drive(vehicle, 100.0)
    single_a.actuators.accept(a, 0)
    single_b.actuators.accept(b, 0)
    batched.actuators.accept(a, 0, vehicle=0)
    batched.actuators.accept(b, 0, vehicle=1)
    pair = [body_a.state(), body_b.state()]
    for _ in range(300):
        wa = single_a.step(pair[0], DT)
        wb = single_b.step(pair[1], DT)
        both = batched.step_all(pair, DT)
        np.testing.assert_allclose(
            both[0].force_world_n, wa.force_world_n, rtol=1e-9, atol=1e-9
        )
        np.testing.assert_allclose(
            both[1].torque_world_nm, wb.torque_world_nm, rtol=1e-9, atol=1e-9
        )
        body_a.step(wa, DT)
        body_b.step(wb, DT)
        pair = [body_a.state(), body_b.state()]
    with pytest.raises(ValueError, match="step_all"):
        batched.step(pair[0], DT)
