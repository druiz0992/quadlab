import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dronelab.control.manual import FlightCommand, ManualController
from dronelab.simulation.gazebo.actuators import thrust_to_motor_speeds
from dronelab.simulation.gazebo.properties import DEFAULT_DRONE


def update(
    controller, command=FlightCommand(), quaternion=(0, 0, 0, 1), rates=(0, 0, 0)
):
    return controller.update(
        np.array([0.0, 0.0, 1.0]), np.zeros(3), quaternion, rates, command, 0.02
    )


def test_hover_and_actuator_units():
    forces = update(ManualController())
    np.testing.assert_allclose(forces, DEFAULT_DRONE.mass * 9.81 / 4)
    speeds = thrust_to_motor_speeds(forces)
    np.testing.assert_allclose(DEFAULT_DRONE.thrust_coefficient * speeds**2, forces)
    np.testing.assert_array_equal(thrust_to_motor_speeds([0, 0, 0, 0]), 0)


def test_control_axis_and_yaw_signs():
    forward = ManualController()
    forces = update(forward, FlightCommand(forward_mps=1))
    wrench = forward.allocation @ forces
    assert wrench[2] > 0  # positive pitch tilts +body-z thrust forward
    turn = ManualController()
    forces = update(turn, FlightCommand(yaw_rate_rps=0.7))
    assert (turn.allocation @ forces)[3] > 0
    assert forces[2] > forces[0]  # CW rotors generate positive yaw reaction
    climb = ManualController()
    assert sum(update(climb, FlightCommand(climb_mps=0.6))) > DEFAULT_DRONE.mass * 9.81
    descend = ManualController()
    assert (
        sum(update(descend, FlightCommand(climb_mps=-0.6))) < DEFAULT_DRONE.mass * 9.81
    )


def test_heading_and_attitude_recovery():
    controller = ManualController()
    update(
        controller,
        FlightCommand(forward_mps=1),
        Rotation.from_euler("z", 90, degrees=True).as_quat(),
    )
    assert controller.target[1] > 0
    assert abs(controller.target[0]) < 1e-8
    controller = ManualController()
    forces = update(controller, quaternion=Rotation.from_euler("x", 0.1).as_quat())
    assert (controller.allocation @ forces)[1] < 0


def test_limits_and_key_release():
    controller = ManualController()
    forces = update(controller, rates=(100, -100, 100))
    assert controller.saturated
    assert np.all(forces >= 0)
    assert np.all(forces <= DEFAULT_DRONE.max_rotor_thrust)
    assert (
        FlightCommand.from_keys({"Up", "Down", "q", "a", "Left", "Right"})
        == FlightCommand()
    )
    assert FlightCommand.from_keys({"Up", "Left", "q"}) == FlightCommand(1.0, 0.7, 0.6)
    assert FlightCommand.from_keys(set()) == FlightCommand()


@pytest.mark.parametrize(
    "forces", [[-1, 0, 0, 0], [0, 0, 0], [np.nan] * 4, [np.inf] * 4, [10] * 4]
)
def test_invalid_actuator_commands(forces):
    with pytest.raises(ValueError):
        thrust_to_motor_speeds(forces)


def test_split_loops_match_combined_update_and_hold_setpoint():
    from dronelab.control.manual import FlightCommand, ManualController

    combined, split = ManualController(), ManualController()
    args = ([0, 0, 1.0], [0.2, 0, 0], [0, 0.05, 0, 0.99875], [0.1, 0, 0])
    expected = combined.update(*args, FlightCommand(forward_mps=1.0), 0.02)
    split.update_position(*args[:3], FlightCommand(forward_mps=1.0), 0.02)
    np.testing.assert_allclose(split.update_attitude(args[2], args[3]), expected)
    # A second attitude update reuses the held setpoint with new rates.
    assert not np.allclose(split.update_attitude(args[2], [0, 0, 0]), expected)


def test_attitude_loop_requires_position_setpoint():
    from dronelab.control.manual import ManualController

    with pytest.raises(RuntimeError, match="update_position"):
        ManualController().update_attitude([0, 0, 0, 1], [0, 0, 0])
