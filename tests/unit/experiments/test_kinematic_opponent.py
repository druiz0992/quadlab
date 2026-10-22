"""Kinematic opponents: commands, ground following, levelling, laps."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.control.track_follower import KinematicTrackDriver
from robotlab.core.track import Track
from robotlab.evaluation.laps import LapTimer
from robotlab.experiments.opponents import (
    HEIGHT_GAIN_PER_S,
    LEVEL_GAIN_PER_S,
    MAX_VERTICAL_SPEED_MPS,
    KinematicOpponent,
    latest_states,
)
from robotlab.simulation.gazebo.odometry import OdometryVelocityFix
from robotlab.simulation.wheeled.model import ChassisState
from robotlab.testing.plugins import nominal_wheeled, synthetic_context


def xyz(values):
    return SimpleNamespace(x=values[0], y=values[1], z=values[2])


def odometry(stamp_ns, position, quaternion=(0.0, 0.0, 0.0, 1.0)):
    sec, nsec = divmod(stamp_ns, 1_000_000_000)
    x, y, z, w = quaternion
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nsec=nsec)),
        pose=SimpleNamespace(
            position=xyz(position), orientation=SimpleNamespace(x=x, y=y, z=z, w=w)
        ),
        twist=SimpleNamespace(linear=xyz((0, 0, 0)), angular=xyz((0, 0, 0))),
    )


def opponent(height=lambda x, y: 0.0 * x):
    vehicle = nominal_wheeled("car")
    track = Track(tuple((float(x), 0.0) for x in range(-50, 200, 5)), 10.0, False)
    context = dataclasses.replace(synthetic_context(wheeled=vehicle), track=track)
    sent: list = []
    car = KinematicOpponent(
        name="car_1",
        driver=KinematicTrackDriver(context, {}),
        publish=sent.append,
        height_m=height,
        hub_xy_m=vehicle.hub_positions_m()[:, :2],
        laps=LapTimer(track),
        odometry_fix=OdometryVelocityFix(),
    )
    return car, sent


def test_each_odometry_message_publishes_one_command():
    car, sent = opponent()
    state = car.on_odometry(odometry(20_000_000, (0.0, 0.0, 0.4)), model=None)
    assert state.name == "car_1" and state.sim_time_ns == 20_000_000
    assert sent == [dict(linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0))]
    car.on_odometry(odometry(40_000_000, (0.0, 0.0, 0.4)), model=None)
    accelerating = car.driver.follower.acceleration_limits_mps2()[1]
    assert sent[-1]["linear"][0] == pytest.approx(accelerating * 0.02)
    assert car.control_ticks == 2
    assert latest_states([car]) == (car.state,)
    assert car.summary()["mode"] == "kinematic"


def test_height_above_ground_is_held_and_capped():
    car, sent = opponent(height=lambda x, y: 0.1 * x)
    car.on_odometry(odometry(20_000_000, (0.0, 0.0, 0.4)), model=None)
    assert car.clearance_m == pytest.approx(0.4)
    # 1 m further on, the ground is 0.1 m higher: climb at the gain.
    car.on_odometry(odometry(40_000_000, (1.0, 0.0, 0.4)), model=None)
    assert sent[-1]["linear"][2] == pytest.approx(HEIGHT_GAIN_PER_S * 0.1)
    car.on_odometry(odometry(60_000_000, (10.0, 0.0, 0.4)), model=None)
    assert sent[-1]["linear"][2] == pytest.approx(MAX_VERTICAL_SPEED_MPS)


def test_a_knocked_body_is_levelled():
    car, sent = opponent()
    tilted = Rotation.from_euler("xyz", [0.1, -0.05, 0.0]).as_quat()
    car.on_odometry(odometry(20_000_000, (0.0, 0.0, 0.4), tilted), model=None)
    roll_rate, pitch_rate, _ = sent[-1]["angular"]
    assert roll_rate == pytest.approx(-LEVEL_GAIN_PER_S * 0.1, rel=1e-6)
    assert pitch_rate == pytest.approx(LEVEL_GAIN_PER_S * 0.05, rel=1e-6)


def test_contact_points_follow_the_heading():
    car, _ = opponent()
    yaw = np.pi / 2
    chassis = ChassisState(
        np.array([5.0, 1.0, 0.4]),
        np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]),
        np.zeros(3),
        np.zeros(3),
    )
    points = car._contacts_xy(chassis)
    hubs = car.hub_xy_m
    np.testing.assert_allclose(points[:, 0], 5.0 - hubs[:, 1], atol=1e-12)
    np.testing.assert_allclose(points[:, 1], 1.0 + hubs[:, 0], atol=1e-12)
