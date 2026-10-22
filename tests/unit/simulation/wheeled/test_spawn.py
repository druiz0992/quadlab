import numpy as np
import pytest

from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.simulation.wheeled.ground import GroundModel
from robotlab.simulation.wheeled.spawn import (
    SPAWN_DROP_M,
    spawn_pose,
    spawn_xy_yaw,
    validate_spawn_points,
)


def test_spawn_points_and_fallback():
    settings = load_drone_config(robot="car").data
    scene = {
        "spawn_points": [{"position_m": [1, 2], "yaw_deg": 90}, {"position_m": [3, 4]}]
    }
    assert spawn_xy_yaw(settings, scene) == pytest.approx((1, 2, np.pi / 2))
    assert spawn_xy_yaw(settings, scene, offset=1) == pytest.approx((3, 4, 0))
    with pytest.raises(ValueError, match="spawn points"):
        spawn_xy_yaw(settings, scene, offset=2)
    assert spawn_xy_yaw(settings, {}) == pytest.approx((0, 0, 0))


def test_spawn_height_puts_lowest_tyre_above_box_top():
    car = load_drone_config(robot="car").wheeled
    box = dict(position=[0, 0, 0.25], size=[10, 10, 0.5])
    pose = spawn_pose(car, GroundModel(None, [box], 100.0), 0.0, 0.0, 0.0)
    lowest = pose.position_m[2] + car.hub_positions_m()[:, 2].min() - car.tyre.radius_m
    assert lowest == pytest.approx(0.5 + SPAWN_DROP_M)


def test_spawn_outside_ground_rejected():
    car = load_drone_config(robot="car").wheeled
    with pytest.raises(ValueError, match="outside"):
        spawn_pose(car, GroundModel(None, [], 1.0), 0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "points",
    [
        {},
        [{"position_m": [0, 0, 0]}],
        [{"position_m": [0, 0], "yaw": 1}],
        [{"position_m": [0, 0], "yaw_deg": "n"}],
    ],
)
def test_invalid_spawn_points(points):
    with pytest.raises(ValueError):
        validate_spawn_points(points)
