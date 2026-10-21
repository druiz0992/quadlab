from dronelab.control.manual import FlightCommand
from dronelab.experiments.takeoff import TakeoffSequence
from dronelab.simulation.gazebo.config import load_drone_config
from dronelab.simulation.gazebo.vehicle import spawn_position


def test_climbs_until_setpoint_reaches_altitude_then_passes_commands():
    takeoff = TakeoffSequence(altitude_m=1.0, climb_mps=0.5)
    pilot = FlightCommand(forward_mps=2.0)
    assert takeoff.command(0.4, 0.0, pilot) == FlightCommand(climb_mps=0.5)
    assert takeoff.command(0.99, 1.0, pilot).climb_mps == 0.5
    assert takeoff.command(1.0, 1.2, pilot) == pilot  # reached: hand over
    assert takeoff.complete and takeoff.complete_time_s == 1.2
    # Once complete, it never takes over again, even below the altitude.
    assert takeoff.command(0.2, 5.0, pilot) == pilot


def test_ground_spawn_rests_body_on_ground():
    settings = load_drone_config().data
    assert spawn_position(settings) == [0.0, 0.0, 1.0]
    settings["simulation"]["start_on_ground"] = True
    settings["simulation"]["spawn_position_m"] = [2.0, -1.0, 5.0]
    assert spawn_position(settings) == [2.0, -1.0, 0.06 + 0.002]
