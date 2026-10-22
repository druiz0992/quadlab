import numpy as np
import pytest

from robotlab.sensors.battery_monitor import BatteryMonitor
from robotlab.simulation.battery import BatteryState
from robotlab.simulation.gazebo.config import load_drone_config

MS = 1_000_000


def state(current=10.0, voltage=16.0):
    return BatteryState(0.9, 16.4, voltage, current, current * voltage, 0.0)


def monitor(capacity_mah=5000.0, **overrides):
    settings = load_drone_config().data["sensors"]["battery_monitor"]
    settings.update({"enabled": True, **overrides})
    return BatteryMonitor(settings, capacity_mah)


def test_fixed_rate_and_exact_coulomb_counting_without_noise():
    m = monitor(rate_hz=10.0)
    readings = [
        m.sample(i * 20 * MS, state()) for i in range(501)
    ]  # 50 Hz offers, 10 s
    kept = [r for r in readings if r is not None]
    assert len(kept) == 101
    # 10 A for 10 s = 100 A·s = 27.78 mAh.
    assert kept[-1].consumed_mah == pytest.approx(100 / 3.6)
    assert kept[-1].remaining_pct == pytest.approx(100 * (1 - 100 / 3.6 / 5000))
    assert kept[-1].voltage_v == 16.0 and kept[-1].current_a == 10.0


def test_current_offset_biases_consumed_capacity_and_noise_is_seeded():
    biased = monitor(current_offset_a=1.0)
    for i in range(101):
        reading = biased.sample(i * 100 * MS, state())
    assert reading.consumed_mah == pytest.approx(11 * 10 / 3.6)  # 11 A for 10 s
    noisy = [monitor(voltage_noise_std_v=0.05, seed=3) for _ in range(2)]
    values = [
        [m.sample(i * 100 * MS, state()).voltage_v for i in range(2000)] for m in noisy
    ]
    assert values[0] == values[1]
    assert np.std(values[0]) == pytest.approx(0.05, rel=0.1)


def test_disabled_and_requires_battery(tmp_path):
    assert monitor(enabled=False).sample(0, state()) is None
    import json

    path = tmp_path / "m.json"
    path.write_text(json.dumps({"sensors": {"battery_monitor": {"enabled": True}}}))
    with pytest.raises(ValueError, match="battery.enabled"):
        load_drone_config(path)
    assert load_drone_config(
        ["configs/vehicles/racing_5in.json", "configs/sensors/battery_monitor.json"]
    )
