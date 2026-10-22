import dataclasses

import numpy as np
import pytest

from robotlab.control.track_follower import KinematicTrackDriver, TrackFollower
from robotlab.control.wheeled import WheeledControlState
from robotlab.core.track import Track
from robotlab.evaluation.laps import LapTimer
from robotlab.simulation.wheeled.ground import GroundModel
from robotlab.simulation.wheeled.model import WheeledVehicleModel
from robotlab.simulation.wheeled.rigid_body import RigidBody
from robotlab.simulation.wheeled.spawn import spawn_pose
from robotlab.testing.plugins import (
    check_wheeled_controller,
    nominal_wheeled,
    synthetic_context,
)


def oval(radius=15.0, straight=30.0, step=2.0):
    """Closed stadium-shaped centreline starting mid-straight, heading +x."""
    points = []
    for x in np.arange(0, straight / 2, step):
        points.append((x, -radius))
    for a in np.arange(-np.pi / 2, np.pi / 2, step / radius):
        points.append((straight / 2 + radius * np.cos(a), radius * np.sin(a)))
    for x in np.arange(straight / 2, -straight / 2, -step):
        points.append((x, radius))
    for a in np.arange(np.pi / 2, 3 * np.pi / 2, step / radius):
        points.append((-straight / 2 + radius * np.cos(a), radius * np.sin(a)))
    for x in np.arange(-straight / 2, 0, step):
        points.append((x, -radius))
    return Track(tuple(points), 10.0)


def context(kind, track):
    return dataclasses.replace(
        synthetic_context(wheeled=nominal_wheeled(kind)), track=track
    )


def test_follower_needs_a_track_and_known_options():
    with pytest.raises(ValueError, match="track"):
        TrackFollower(synthetic_context(wheeled=nominal_wheeled("car")), {})
    with pytest.raises(ValueError, match="unknown"):
        TrackFollower(context("car", oval()), {"speed": 3})


@pytest.mark.parametrize("kind", ["rover", "car", "tractor"])
def test_follower_follows_the_contract(kind):
    track = oval()

    def factory(ctx, options):
        return TrackFollower(dataclasses.replace(ctx, track=track), options)

    check_wheeled_controller(factory, kind=kind)


def test_car_laps_an_oval_on_the_reference_integrator():
    track = oval()
    vehicle = nominal_wheeled("car")
    ground = GroundModel(None, [], 500.0)
    pose = spawn_pose(vehicle, ground, -10.0, -15.0, 0.0)  # behind the line
    body = RigidBody.at_rest(vehicle.mass_kg, vehicle.inertia_kg_m2, pose.position_m)
    model = WheeledVehicleModel(vehicle, ground)
    follower = TrackFollower(context("car", track), {"max_speed_mps": 20.0})
    timer = LapTimer(track)
    # 4 ms periods (two 2 ms rigid-body steps) keep the test short; the
    # model is stable there too.
    for period in range(int(16.0 / 0.004)):
        state = body.state()
        if period % 3 == 0:
            control = WheeledControlState(
                period * 4_000_000,
                state.position_m,
                state.velocity_world_mps,
                state.quaternion_xyzw,
                state.body_rates_rad_s,
                model.wheel_speed_rad_s.copy(),
                model.actuators.steer_rad.copy(),
            )
            model.actuators.accept(follower.update(control, None, 0.012), 0)
            timer.on_truth(
                period * 0.002,
                state.position_m,
                (
                    model.telemetry.contact_point_m[:, :2]
                    if model.telemetry
                    else state.position_m[None, :2]
                ),
            )
        wrench = model.step(state, 0.004)
        for _ in range(2):
            body.step(wrench, 0.002)
    summary = timer.summary()
    assert summary["laps_completed"] >= 1
    assert summary["off_track_episodes"] == 0
    assert summary["max_lateral_offset_m"] < 2.0


def state_at(x, y, yaw, speed=0.0):
    return WheeledControlState(
        0,
        np.array([x, y, 0.4]),
        np.array([speed * np.cos(yaw), speed * np.sin(yaw), 0.0]),
        np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]),
        np.zeros(3),
        np.zeros(4),
        np.zeros(4),
    )


def test_kinematic_driver_ramps_speed_at_the_grip_limit():
    driver = KinematicTrackDriver(context("car", oval()), {"max_speed_mps": 10.0})
    braking, accelerating = driver.follower.acceleration_limits_mps2()
    vehicle = nominal_wheeled("car")
    grip = 0.7 * vehicle.tyre.friction_longitudinal * 9.81
    assert braking == pytest.approx(grip)
    driven = sum(w.driven for w in vehicle.wheels)
    assert accelerating == pytest.approx(grip * driven / vehicle.wheel_count)
    first = driver.update(state_at(0.0, -15.0, 0.0), 0.0)
    assert first.speed_mps == 0.0  # no time elapsed
    command = driver.update(state_at(0.0, -15.0, 0.0), 0.1)
    assert command.speed_mps == pytest.approx(accelerating * 0.1)
    for _ in range(100):
        command = driver.update(state_at(0.0, -15.0, 0.0), 0.1)
    assert command.speed_mps == pytest.approx(10.0)
    with pytest.raises(ValueError, match="dt_s"):
        driver.update(state_at(0.0, -15.0, 0.0), -0.1)


def test_kinematic_driver_turns_toward_the_path():
    driver = KinematicTrackDriver(context("car", oval()), {})
    driver.speed_mps = 5.0
    # On the bottom straight heading +x but 2 m right of the line (y < -15):
    # the look-ahead point is to the left, so the yaw rate is positive and
    # equals v κ.
    command = driver.update(state_at(0.0, -17.0, 0.0, 5.0), 0.0)
    guidance = driver.follower.guidance(state_at(0.0, -17.0, 0.0, 5.0))
    assert command.yaw_rate_rad_s > 0
    assert command.yaw_rate_rad_s == pytest.approx(5.0 * guidance.curvature_per_m)


def test_kinematic_driver_laps_an_oval():
    """Integrate the commands as an ideal unicycle: the path stays on the
    track and the speed respects the curvature limit."""
    track = oval()
    driver = KinematicTrackDriver(context("car", track), {"max_speed_mps": 20.0})
    timer = LapTimer(track)
    x, y, yaw, dt = -10.0, -15.0, 0.0, 0.02
    for step in range(int(30.0 / dt)):
        command = driver.update(state_at(x, y, yaw, driver.speed_mps), dt)
        x += command.speed_mps * np.cos(yaw) * dt
        y += command.speed_mps * np.sin(yaw) * dt
        yaw += command.yaw_rate_rad_s * dt
        timer.on_truth(step * dt, np.array([x, y, 0.0]), np.array([[x, y]]))
    summary = timer.summary()
    assert summary["laps_completed"] >= 2
    assert summary["off_track_episodes"] == 0
    assert summary["max_lateral_offset_m"] < 1.0
