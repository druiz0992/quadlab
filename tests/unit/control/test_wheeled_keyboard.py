import numpy as np
import pytest

from robotlab.control.wheeled import DriverCommand, WheeledControlState
from robotlab.control.wheeled_keyboard import (
    REVERSE_BELOW_MPS,
    DriverInput,
    KeyboardDriver,
    steering_angles,
)
from robotlab.control.wheeled_template import TemplateWheeledController
from robotlab.testing.plugins import (
    PluginContractError,
    check_wheeled_controller,
    nominal_wheeled,
    synthetic_context,
)


def state(vehicle, speed=0.0):
    n = vehicle.wheel_count
    return WheeledControlState(
        0,
        np.zeros(3),
        np.array([speed, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.zeros(3),
        np.zeros(n),
        np.zeros(n),
    )


def test_ackermann_inner_wheel_turns_more_about_rear_axle():
    car = nominal_wheeled("car")
    angles = steering_angles(car, 0.5, "front")
    hubs = car.hub_positions_m()
    left, right = angles[0], angles[1]
    assert left > right > 0  # left turn: left wheel is inner
    # Both front wheel normals pass through one point on the rear axle line.
    rear_x = hubs[:, 0].min()
    centres = [hubs[i, 1] + (hubs[i, 0] - rear_x) / np.tan(angles[i]) for i in (0, 1)]
    assert centres[0] == pytest.approx(centres[1])
    np.testing.assert_array_equal(angles[2:], 0.0)


def test_four_wheel_modes():
    tractor = nominal_wheeled("tractor")
    front = steering_angles(tractor, 0.5, "front")
    coordinated = steering_angles(tractor, 0.5, "coordinated")
    crab = steering_angles(tractor, 0.5, "crab")
    np.testing.assert_array_equal(front[2:], 0.0)
    assert coordinated[2] == pytest.approx(-coordinated[0])
    np.testing.assert_allclose(crab, 0.5 * tractor.drive.max_steer_rad)
    assert np.all(np.abs(coordinated) <= tractor.drive.max_steer_rad)


def test_skid_steer_mixes_left_and_right():
    rover = nominal_wheeled("rover")
    driver = KeyboardDriver(synthetic_context(wheeled=rover), {})
    commands = driver.update(state(rover), DriverCommand(steer=1.0), 0.01)
    torque = np.array(commands.drive_torque_nm)
    side = np.sign(rover.hub_positions_m()[:, 1])
    np.testing.assert_allclose(torque, -side * rover.drive.max_torque_nm)
    assert commands.steer_angle_rad == (0.0,) * 4


def test_keyboard_driver_throttle_brake_and_speed_limit():
    car = nominal_wheeled("car")
    driver = KeyboardDriver(synthetic_context(wheeled=car), {"max_speed_mps": 10.0})
    slow = driver.update(state(car, 5.0), DriverCommand(throttle=1.0), 0.01)
    fast = driver.update(state(car, 12.0), DriverCommand(throttle=1.0), 0.01)
    assert slow.drive_torque_nm[2] == car.drive.max_torque_nm
    assert fast.drive_torque_nm[2] == 0.0
    brake = driver.update(state(car, 5.0), DriverCommand(brake=1.0), 0.01)
    assert set(brake.brake_torque_nm) == {car.drive.max_brake_torque_nm}
    parked = driver.update(state(car), None, 0.0)
    assert parked == parked.zeros(4)
    with pytest.raises(ValueError, match="unknown"):
        KeyboardDriver(synthetic_context(wheeled=car), {"gain": 1})
    with pytest.raises(ValueError, match="wheeled"):
        KeyboardDriver(synthetic_context(), {})


def test_driver_input_ramps_reverse_and_modes():
    keys = DriverInput(throttle_rate_per_s=2.0, steer_rate_per_s=1.0)
    command = keys.update({"Up"}, 0.0, 0.25)
    assert command.throttle == pytest.approx(0.5)
    command = keys.update(set(), 0.0, 0.1)
    assert command.throttle == pytest.approx(0.1)  # releases twice as fast
    moving = keys.update({"Down"}, 5.0, 0.1)
    assert moving.brake == 1.0 and moving.throttle <= 0
    reverse = DriverInput().update({"Down"}, REVERSE_BELOW_MPS / 2, 0.25)
    assert reverse.throttle < 0 and reverse.brake == 0.0
    steer = DriverInput(steer_rate_per_s=1.0)
    assert steer.update({"Left"}, 0.0, 0.5).steer == pytest.approx(0.5)
    assert steer.update(set(), 0.0, 0.1).steer == pytest.approx(0.3)
    keys.press("2")
    keys.press("h")
    command = keys.update(set(), 0.0, 0.01)
    assert command.steering_mode == "coordinated" and command.implement_lowered


@pytest.mark.parametrize(
    "kwargs",
    [dict(throttle=2.0), dict(brake=-0.1), dict(steer=1.5), dict(steering_mode="tank")],
)
def test_driver_command_validation(kwargs):
    with pytest.raises(ValueError):
        DriverCommand(**kwargs)


@pytest.mark.parametrize("kind", ["rover", "car", "tractor"])
@pytest.mark.parametrize("controller", [KeyboardDriver, TemplateWheeledController])
def test_built_in_controllers_follow_the_contract(kind, controller):
    report = check_wheeled_controller(controller, kind=kind)
    assert len(report.commands) == 24


def test_contract_kit_rejects_bad_commands():
    class TorqueOnSteeredWheel:
        def __init__(self, context, options):
            self.n = context.wheeled.wheel_count

        def update(self, state, request, dt_s):
            from robotlab.core.wheeled import WheelCommands

            return WheelCommands((1.0,) * self.n, (0.0,) * self.n, (0.0,) * self.n)

    with pytest.raises(PluginContractError, match="without that actuator"):
        check_wheeled_controller(TorqueOnSteeredWheel, kind="car")

    class Random:
        def __init__(self, context, options):
            self.n = context.wheeled.wheel_count
            self.rng = np.random.default_rng()

        def update(self, state, request, dt_s):
            from robotlab.core.wheeled import WheelCommands

            brake = tuple(self.rng.uniform(0, 1, self.n))
            return WheelCommands((0.0,) * self.n, brake, (0.0,) * self.n)

    with pytest.raises(PluginContractError, match="different"):
        check_wheeled_controller(Random, kind="tractor")
