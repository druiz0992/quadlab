import numpy as np
import pytest

from robotlab.control.drone import RotorThrustControl
from robotlab.control.manual import FlightCommand, ManualController
from robotlab.experiments.rotor_loop import RotorControlLoop, TruthState
from robotlab.simulation.gazebo.config import load_drone_config

MS = 1_000_000


def hover_state(t_ns, z=1.0):
    return TruthState(
        t_ns, np.array([0, 0, z]), np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3)
    )


def make_loop(**timing):
    config = load_drone_config()
    settings = dict(config.data["control"], **timing)
    outputs = []
    control = RotorThrustControl(
        outputs.append,
        properties=config.properties,
        start_worker=False,
        publish_on_send=True,
    )
    handle = control.acquire()
    controller = ManualController(config.properties, config.data["controller"])
    loop = RotorControlLoop(controller, control, handle, settings, tolerance_ns=MS)
    return loop, controller, outputs


def drive(loop, times, pilot=lambda t: FlightCommand(), state=hover_state):
    return [record for t in times for record in loop.on_state(state(t), pilot)]


def test_default_timing_matches_single_loop_controller():
    loop, _, _ = make_loop()
    reference = ManualController(
        load_drone_config().properties, load_drone_config().data["controller"]
    )
    records = drive(
        loop,
        [i * 20 * MS for i in range(5)],
        state=lambda t: hover_state(t, 1.0 + t * 1e-9),
    )
    for i, record in enumerate(records):
        state = hover_state(i * 20 * MS, 1.0 + i * 0.02)
        expected = reference.update(
            state.position_m,
            state.velocity_world_mps,
            state.quaternion_xyzw,
            state.body_rates_rad_s,
            FlightCommand(),
            0.02,
        )
        np.testing.assert_allclose(record["thrust_n"], expected)
        assert (
            record["tick_sim_time_ns"]
            == record["state_sim_time_ns"]
            == record["release_sim_time_ns"]
        )
    assert loop.summary()["ticks"] == 5 and loop.summary()["warmup_commands"] == 0


def test_control_rate_below_sample_rate_ticks_on_grid():
    loop, _, outputs = make_loop(odometry_rate_hz=200.0, control_rate_hz=100.0)
    records = drive(loop, [i * 5 * MS for i in range(8)])
    assert [r["tick_sim_time_ns"] for r in records] == [0, 10 * MS, 20 * MS, 30 * MS]
    assert len(outputs) == 4  # publish_on_send: one publication per command


def test_position_loop_runs_every_nth_tick():
    loop, _, _ = make_loop(control_rate_hz=50.0, position_rate_hz=25.0)
    drive(loop, [i * 20 * MS for i in range(6)])
    assert loop.summary()["position_updates"] == 3 and loop.summary()["ticks"] == 6


def test_state_latency_uses_older_state_after_warmup():
    loop, _, _ = make_loop(state_latency_s=0.04)
    records = drive(loop, [i * 20 * MS for i in range(5)])
    # Warm-up: the first sample stands in until a sample is 40 ms old.
    assert [r["state_sim_time_ns"] for r in records] == [0, 0, 0, 20 * MS, 40 * MS]
    assert loop.summary()["max_state_age_ms"] == pytest.approx(40.0)


def test_command_latency_delays_release_and_repeats_first_command_during_warmup():
    loop, _, _ = make_loop(command_latency_s=0.04)
    records = drive(loop, [i * 20 * MS for i in range(5)])
    assert [r["warmup"] for r in records] == [True, True, False, False, False]
    released = [r for r in records if not r["warmup"]]
    assert [r["tick_sim_time_ns"] for r in released] == [0, 20 * MS, 40 * MS]
    assert [r["release_sim_time_ns"] for r in released] == [40 * MS, 60 * MS, 80 * MS]
    summary = loop.summary()
    assert summary["pending_at_close"] == 2 and summary["commands_released"] == 3


def test_pilot_is_sampled_at_tick_time():
    loop, controller, _ = make_loop()
    seen = []
    drive(
        loop,
        [0, 20 * MS],
        pilot=lambda t: seen.append(t) or FlightCommand(climb_mps=0.5),
    )
    assert seen == [0.0, 0.02]
    assert controller.target[2] > 1.0


def test_disarm_stops_motors_and_commands():
    loop, _, outputs = make_loop()
    drive(loop, [0])
    loop.disarm()
    assert outputs[-1] == (0.0, 0.0, 0.0, 0.0)
    assert drive(loop, [20 * MS, 40 * MS]) == []  # nothing more is sent
