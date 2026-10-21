import json

import numpy as np
import pytest

from dronelab.simulation.gazebo.config import load_drone_config
from dronelab.simulation.motor_faults import MotorFaults
from dronelab.simulation.powertrain import Powertrain

MS = 1_000_000
HOVER_N = 1.54 * 9.81 / 4


def event(kind, rotor, start_s=0.0, end_s=None, **extra):
    return dict(type=kind, rotor=rotor, start_s=start_s, end_s=end_s, **extra)


def test_health_windows_and_combination():
    faults = MotorFaults(
        {
            "events": [
                event("partial_thrust", 1, 1.0, 3.0, value=0.5),
                event("partial_thrust", 1, 2.0, None, value=0.8),
                event("speed_limit", 2, 0.0, None, value=0.6),
                event("speed_limit", 2, 0.0, None, value=0.9),
                event("stop", 3, 5.0, None),
            ]
        },
        4,
    )
    thrust, cap = faults.health(0)
    np.testing.assert_allclose(thrust, [1, 1, 1, 1])
    np.testing.assert_allclose(cap, [1, 1, 0.6, 1])  # lowest cap wins
    thrust, _ = faults.health(2500 * MS)
    assert thrust[1] == pytest.approx(0.4)  # 0.5 × 0.8 while both active
    thrust, cap = faults.health(5000 * MS)
    assert thrust[1] == pytest.approx(0.8) and thrust[3] == 0 and cap[3] == 0
    summary = faults.summary()["events"]
    assert summary[0]["first_active_sim_time_ns"] == 2500 * MS
    assert summary[4]["first_active_sim_time_ns"] == 5000 * MS


def powertrain(events):
    config = load_drone_config()
    return Powertrain(
        config.properties,
        config.data["battery"],
        config.data["aerodynamics"]["ground_effect"],
        {"events": events},
    )


def test_published_speeds_implement_fault_physics():
    p = powertrain(
        [
            event("stop", 0),
            event("partial_thrust", 1, value=0.49),
            event("speed_limit", 2, value=0.5),
        ]
    )
    assert p.enabled
    p.update(0, [0, 0, 1], [0, 0, 0, 1])
    hover_speed = np.sqrt(HOVER_N / 8.55e-6)
    speeds = p.motor_speeds(np.full(4, HOVER_N))
    assert speeds[0] == 0  # stopped
    assert speeds[1] == pytest.approx(hover_speed * 0.7)  # k_t ω² scaled by 0.49
    assert speeds[2] == pytest.approx(500.0)  # capped at half of 1000 rad/s
    assert speeds[3] == pytest.approx(hover_speed)
    np.testing.assert_allclose(p.available_thrust_fraction(), [0, 0.49, 0.25, 1])
    assert p.summary()["limited_commands"] == 1


def test_faults_start_on_schedule():
    p = powertrain([event("stop", 2, start_s=1.0)])
    p.update(999 * MS, [0, 0, 1], [0, 0, 0, 1])
    assert p.motor_speeds(np.full(4, HOVER_N))[2] > 0
    p.update(1000 * MS, [0, 0, 1], [0, 0, 0, 1])
    assert p.motor_speeds(np.full(4, HOVER_N))[2] == 0


@pytest.mark.parametrize(
    "bad",
    [
        event("explode", 0),
        event("stop", 4),
        event("stop", 0, value=0.5),  # stop takes no value
        event("partial_thrust", 1),  # missing value
        event("partial_thrust", 1, value=1.0),
        event("speed_limit", 1, value=0.0),
        event("speed_limit", 1, 2.0, 1.0, value=0.5),
    ],
)
def test_invalid_motor_fault_configuration(tmp_path, bad):
    path = tmp_path / "f.json"
    path.write_text(json.dumps({"motors": {"faults": {"events": [bad]}}}))
    with pytest.raises(ValueError):
        load_drone_config(path)
