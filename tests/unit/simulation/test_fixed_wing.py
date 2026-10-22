"""Fixed-wing aerodynamics (FLU sign conventions), servos, pitot sensor."""

import math

import numpy as np
import pytest

from robotlab.sensors.airspeed import AirspeedSensor
from robotlab.simulation.fixed_wing import (
    FixedWingAero,
    ServoBank,
    lift_coefficient,
    DEFAULT_COEFFICIENTS,
)
from robotlab.simulation.gazebo.config import load_drone_config

RHO = 1.225


def wing(**overrides):
    settings = dict(load_drone_config().data["fixed_wing"], enabled=True)
    settings.update(overrides)
    return FixedWingAero(settings, RHO)


def air_at(speed, alpha=0.0, beta=0.0):
    """FLU body air velocity for angle of attack α and sideslip β."""
    u = speed * math.cos(alpha) * math.cos(beta)
    return np.array(
        [u, -speed * math.sin(beta), -speed * math.cos(beta) * math.sin(alpha)]
    )


def test_no_wrench_below_minimum_airspeed():
    force, moment = wing().wrench_body([0.5, 0, 0], [0, 0, 0], {})
    assert not force.any() and not moment.any()


def test_positive_alpha_lifts_up_and_symmetric_flight_has_no_lateral_terms():
    model = wing()
    force, moment = model.wrench_body(air_at(20, math.radians(4)), [0, 0, 0], {})
    qbar_s = 0.5 * RHO * 400 * 0.55
    cl = lift_coefficient(math.radians(4), DEFAULT_COEFFICIENTS, 50, 0.47)
    assert force[2] == pytest.approx(qbar_s * cl * math.cos(math.radians(4)), rel=0.05)
    assert (
        force[1] == pytest.approx(0)
        and moment[0] == pytest.approx(0)
        and moment[2] == pytest.approx(0)
    )
    assert model.last.alpha_rad == pytest.approx(math.radians(4))


def test_lift_falls_beyond_stall():
    values = [
        lift_coefficient(math.radians(a), DEFAULT_COEFFICIENTS, 50, 0.47)
        for a in range(0, 45)
    ]
    peak = int(np.argmax(values))
    assert 20 <= peak <= 30 and values[40] < values[peak] * 0.9


def test_elevator_and_damping_signs_in_flu():
    model = wing()
    _, base = model.wrench_body(air_at(20, 0.05), [0, 0, 0], {})
    _, down = model.wrench_body(air_at(20, 0.05), [0, 0, 0], {"elevator": 0.1})
    assert down[1] > base[1]  # trailing edge down: nose down = +y moment in FLU
    _, rolling = model.wrench_body(air_at(20, 0.05), [1.0, 0, 0], {})
    assert rolling[0] < 0  # roll damping opposes +x rate
    _, yawing = model.wrench_body(air_at(20, 0.05, beta=0.1), [0, 0, 0], {})
    # Wind from the right (β > 0 in FRD): weathervane yaws the nose right,
    # i.e. negative about FLU +z.
    assert yawing[2] < 0


def test_servos_limit_travel_rate_and_lag():
    servos = ServoBank(
        dict(max_deflection_rad=0.4, rate_limit_rad_s=2.0, time_constant_s=0.0)
    )
    servos.update(0)
    servos.set_command({"elevator": 1.0})
    assert servos.command["elevator"] == 0.4
    position = servos.update(100_000_000)["elevator"]
    assert position == pytest.approx(0.2)  # 2 rad/s for 0.1 s
    assert servos.update(1_000_000_000)["elevator"] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        servos.set_command({"flap": 0.1})


def test_pitot_ias_offset_and_blockage():
    settings = dict(
        load_drone_config().data["sensors"]["airspeed"],
        enabled=True,
        outages_s=[[1.0, 2.0]],
    )
    sensor = AirspeedSensor(settings, air_density_kg_m3=1.0)
    reading = sensor.sample(0, [20.0, 0, 0])
    # Thinner air: IAS = TAS·√(ρ/ρ₀).
    assert reading.indicated_airspeed_mps == pytest.approx(20 * math.sqrt(1.0 / 1.225))
    assert sensor.sample(10_000_000, [20.0, 0, 0]) is None  # 50 Hz
    assert sensor.sample(1_500_000_000, [20.0, 0, 0]).indicated_airspeed_mps == 0.0
    assert sensor.sample(3_000_000_000, [-5.0, 0, 0]).indicated_airspeed_mps == 0.0
