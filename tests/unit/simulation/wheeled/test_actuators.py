import numpy as np
import pytest

from robotlab.core.wheeled import WheelCommands
from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.simulation.wheeled.actuators import TOP_SPEED_TAPER, WheelActuators


@pytest.fixture
def car():
    return load_drone_config(robot="car").wheeled


def test_limits_are_applied_and_reported(car):
    actuators = WheelActuators(car)
    receipt = actuators.accept(
        WheelCommands((0, 0, 1e4, 100.0), (1e5, 0, 0, 0), (1.0, -0.1, 0, 0)), 7
    )
    d = car.drive
    assert receipt.accepted.drive_torque_nm[2] == d.max_torque_nm
    assert receipt.accepted.brake_torque_nm[0] == d.max_brake_torque_nm
    assert receipt.accepted.steer_angle_rad[0] == d.max_steer_rad
    assert receipt.limited == dict(drive=(2,), brake=(0,), steer=(0,))
    assert receipt.sim_time_ns == 7


@pytest.mark.parametrize(
    "commands, message",
    [
        (WheelCommands((100.0, 0, 0, 0), (0,) * 4, (0,) * 4), "without that actuator"),
        (WheelCommands((0,) * 4, (0,) * 4, (0, 0, 0.1, 0)), "without that actuator"),
        (WheelCommands((0,) * 3, (0,) * 3, (0,) * 3), "one per wheel"),
        (WheelCommands((0, 0, float("nan"), 0), (0,) * 4, (0,) * 4), "finite"),
        (WheelCommands((0,) * 4, (-1.0, 0, 0, 0), (0,) * 4), ">= 0"),
    ],
)
def test_structurally_invalid_commands_raise(car, commands, message):
    with pytest.raises(ValueError, match=message):
        WheelActuators(car).accept(commands, 0)


def test_power_and_top_speed_envelope(car):
    actuators = WheelActuators(car)
    actuators.accept(WheelCommands((0, 0, 1500, 1500), (0,) * 4, (0,) * 4), 0)
    d = car.drive
    low, mid, top = (np.full(4, w) for w in (1.0, 150.0, d.max_wheel_speed_rad_s))
    assert actuators.drive_torque(low)[2] == d.max_torque_nm
    assert actuators.drive_torque(mid)[2] == pytest.approx(d.max_power_w / 150.0)
    assert actuators.drive_torque(top)[2] == 0.0
    near = np.full(4, d.max_wheel_speed_rad_s * (1 - TOP_SPEED_TAPER / 2))
    full = d.max_power_w / near[2]
    assert actuators.drive_torque(near)[2] == pytest.approx(0.5 * full)
    # Torque against the rotation (regeneration) is not cut at top speed.
    actuators.accept(WheelCommands((0, 0, -500, -500), (0,) * 4, (0,) * 4), 0)
    assert actuators.drive_torque(top)[2] < 0
    assert actuators.limited_now(top) == (2, 3)


def test_steering_slews_at_rate_limit(car):
    actuators = WheelActuators(car)
    actuators.accept(WheelCommands((0,) * 4, (0,) * 4, (0.3, 0.3, 0, 0)), 0)
    actuators.slew_steering(0.05)
    assert actuators.steer_rad[0] == pytest.approx(
        car.drive.max_steer_rate_rad_s * 0.05
    )
    for _ in range(100):
        actuators.slew_steering(0.05)
    assert actuators.steer_rad[0] == pytest.approx(0.3)
