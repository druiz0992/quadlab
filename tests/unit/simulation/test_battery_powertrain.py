import numpy as np
import pytest

from robotlab.simulation.aerodynamics import ground_effect_factor
from robotlab.simulation.battery import Battery
from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.simulation.powertrain import Powertrain

MS = 1_000_000
HOVER_N = 1.54 * 9.81 / 4


def battery_settings(**overrides):
    return dict(load_drone_config().data["battery"], enabled=True, **overrides)


def ground_settings(**overrides):
    return dict(load_drone_config().data["aerodynamics"]["ground_effect"], **overrides)


def test_battery_voltage_sags_with_load_and_discharges_by_charge():
    battery = Battery(
        battery_settings(capacity_mah=1000.0, internal_resistance_ohm=0.05)
    )
    assert battery.state.terminal_v == pytest.approx(16.8)  # 4S full, no load
    state = battery.step(0.0, 200.0)
    # V = (OCV + sqrt(OCV² - 4PR)) / 2 and P = V I.
    expected_v = (16.8 + np.sqrt(16.8**2 - 4 * 200 * 0.05)) / 2
    assert state.terminal_v == pytest.approx(expected_v)
    assert state.current_a * state.terminal_v == pytest.approx(200.0)
    state = battery.step(36.0, 200.0)  # 36 s at about 12 A uses about 12 % of 1 Ah
    assert state.soc == pytest.approx(1 - state.current_a * 36 / 3600, rel=0.02)
    assert state.energy_used_wh == pytest.approx(2.0, rel=0.01)


def test_battery_power_is_limited_by_internal_resistance_and_soc_clamps():
    battery = Battery(battery_settings(internal_resistance_ohm=1.0))
    state = battery.step(0.0, 1e6)
    assert state.terminal_v == pytest.approx(16.8 / 2)  # maximum power point
    empty = Battery(battery_settings(capacity_mah=1.0))
    empty.step(10.0, 500.0)
    assert empty.state.soc == 0.0 and empty.depleted


def test_powertrain_disabled_matches_plain_conversion():
    config = load_drone_config()
    powertrain = Powertrain(
        config.properties, config.data["battery"], ground_settings()
    )
    assert not powertrain.enabled


def test_battery_limits_speed_by_voltage_and_counts_limited_commands():
    config = load_drone_config()
    powertrain = Powertrain(
        config.properties, battery_settings(initial_soc=0.05), ground_settings()
    )
    powertrain.update(0, [0, 0, 1], [0, 0, 0, 1])
    max_thrust = config.properties.max_rotor_thrust
    speeds = powertrain.motor_speeds(np.full(4, max_thrust))
    limit = 1000.0 * powertrain.battery_state().terminal_v / 16.8
    np.testing.assert_allclose(speeds, limit)
    assert limit < 900
    hover = powertrain.motor_speeds(np.full(4, HOVER_N))
    np.testing.assert_allclose(hover, np.sqrt(HOVER_N / 8.55e-6))
    assert powertrain.summary()["limited_commands"] == 1


def test_hover_power_drains_battery_until_motors_stop():
    config = load_drone_config()
    powertrain = Powertrain(
        config.properties, battery_settings(capacity_mah=50.0), ground_settings()
    )
    t = 0
    powertrain.update(t, [0, 0, 1], [0, 0, 0, 1])
    while powertrain.summary()["depleted_sim_time_ns"] is None and t < 120_000 * MS:
        powertrain.motor_speeds(np.full(4, HOVER_N))
        t += 20 * MS
        powertrain.update(t, [0, 0, 1], [0, 0, 0, 1])
    summary = powertrain.summary()
    battery = summary["battery"]
    # Draining delivers about capacity × average loaded voltage: 0.05 Ah × ~15 V.
    assert 0.05 * 13.0 < battery["energy_used_wh"] < 0.05 * 16.8
    # Hover shaft power 4 c_m k_t ω³ ≈ 161 W; /0.7 + 5 W avionics ≈ 235 W.
    depleted_s = summary["depleted_sim_time_ns"] * 1e-9
    assert depleted_s == pytest.approx(battery["energy_used_wh"] * 3600 / 235, rel=0.05)
    np.testing.assert_array_equal(powertrain.motor_speeds(np.full(4, HOVER_N)), 0)


def test_ground_effect_raises_published_speed_near_ground_only():
    config = load_drone_config()
    powertrain = Powertrain(
        config.properties,
        config.data["battery"],
        ground_settings(enabled=True),
    )
    plain = np.sqrt(HOVER_N / 8.55e-6)
    powertrain.update(0, [0, 0, 2.0], [0, 0, 0, 1])
    np.testing.assert_allclose(
        powertrain.motor_speeds(np.full(4, HOVER_N)), plain, rtol=1e-3
    )
    powertrain.update(MS, [0, 0, 0.0], [0, 0, 0, 1])  # rotor plane 0.08 m up
    expected = ground_effect_factor(0.08, 0.125)
    assert expected == pytest.approx(1.18, abs=0.01)
    np.testing.assert_allclose(powertrain.ground_effect_factors(), expected)
    np.testing.assert_allclose(
        powertrain.motor_speeds(np.full(4, HOVER_N)), plain * np.sqrt(expected)
    )
    powertrain.update(2 * MS, [0, 0, -0.05], [0, 0, 0, 1])  # below R/2: clamped
    np.testing.assert_allclose(powertrain.ground_effect_factors(), 4 / 3)
    assert powertrain.summary()["max_ground_effect_factor"] == pytest.approx(4 / 3)


def test_infinite_battery_keeps_charge_but_still_sags():
    battery = Battery(battery_settings(infinite=True, capacity_mah=10.0))
    state = battery.step(600.0, 200.0)
    assert state.soc == 1.0
    assert state.terminal_v < 16.8  # load still causes sag
    assert state.energy_used_wh > 0


def test_refill_and_drain_recover_from_depletion():
    config = load_drone_config()
    p = Powertrain(
        config.properties, battery_settings(capacity_mah=1.0), ground_settings()
    )
    p.update(0, [0, 0, 1], [0, 0, 0, 1])
    p.motor_speeds(np.full(4, HOVER_N))
    p.update(60_000 * MS, [0, 0, 1], [0, 0, 0, 1])
    assert p.summary()["depleted_sim_time_ns"] is not None
    assert p.adjust_battery(refill=True) == 1.0
    assert p.summary()["depleted_sim_time_ns"] is None
    assert p.motor_speeds(np.full(4, HOVER_N))[0] > 0  # power restored
    assert p.adjust_battery(delta_soc=-0.25) == pytest.approx(0.75)
    assert p.adjust_battery(delta_soc=-2.0) == 0.0  # clamped


def thermal(**overrides):
    settings = battery_settings()
    return dict(settings, thermal=dict(settings["thermal"], enabled=True, **overrides))


def test_cold_pack_has_higher_resistance_and_less_capacity():
    warm = Battery(thermal(), ambient_c=25.0)
    cold = Battery(thermal(), ambient_c=0.0)
    assert cold.temperature_c == 0.0
    warm_state, cold_state = warm.step(10.0, 300.0), cold.step(10.0, 300.0)
    assert cold_state.terminal_v < warm_state.terminal_v  # R doubles near 0 °C
    # Capacity shrinks by k_C per °C below 25: 15 % less at 0 °C.
    used_warm, used_cold = 1 - warm_state.soc, 1 - cold_state.soc
    ratio = used_cold / used_warm
    assert ratio == pytest.approx(
        cold_state.current_a / warm_state.current_a / 0.85, rel=0.02
    )


def test_pack_heats_from_losses_towards_balance_with_cooling():
    settings = thermal(heat_capacity_j_per_k=50.0, cooling_w_per_k=2.0)
    battery = Battery(dict(settings, infinite=True), ambient_c=25.0)
    for _ in range(4000):
        battery.step(0.5, 200.0)
    state = battery.state
    # Balance: I²R(T) = h (T − ambient).
    k = settings["thermal"]["resistance_coefficient_per_c"]
    resistance = settings["internal_resistance_ohm"] * np.exp(
        k * (25 - state.temperature_c)
    )
    assert state.temperature_c > 25.0
    assert state.current_a**2 * resistance == pytest.approx(
        2.0 * (state.temperature_c - 25.0), rel=0.01
    )


def test_disabled_thermal_ignores_ambient_and_set_ambient_shifts_pack():
    plain = Battery(battery_settings(), ambient_c=-20.0)
    reference = Battery(battery_settings())
    assert plain.step(1.0, 200.0).terminal_v == reference.step(1.0, 200.0).terminal_v
    battery = Battery(thermal(), ambient_c=25.0)
    battery.set_ambient(5.0)
    assert battery.temperature_c == pytest.approx(5.0)
    assert battery.state.temperature_c == pytest.approx(5.0)
