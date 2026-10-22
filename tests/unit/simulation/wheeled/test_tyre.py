import numpy as np
import pytest

from robotlab.core.wheeled_vehicle import MagicFormulaCoefficients, TyreParameters
from robotlab.simulation.wheeled.tyre import (
    MAX_SLIP,
    TyreModel,
    magic_formula,
    peak_slip,
)

DRY = MagicFormulaCoefficients(10.0, 1.9, 0.97)


def parameters(**overrides):
    values = dict(
        radius_m=0.3,
        friction_longitudinal=1.0,
        friction_lateral=1.0,
        longitudinal=DRY,
        lateral=DRY,
        relaxation_longitudinal_m=0.1,
        relaxation_lateral_m=0.2,
        rolling_resistance=0.01,
        low_speed_mps=2.0,
        low_speed_damping_n_s_per_m=1000.0,
    )
    values.update(overrides)
    return TyreParameters(**values)


def test_magic_formula_is_odd_and_peaks_at_one():
    slips = np.linspace(-1, 1, 201)
    np.testing.assert_allclose(magic_formula(DRY, slips), -magic_formula(DRY, -slips))
    peak = peak_slip(DRY)
    assert magic_formula(DRY, peak) == pytest.approx(1.0, abs=1e-6)
    assert 0.05 < peak < 0.3


@pytest.mark.parametrize("b, c, e", [(0.0, 1.5, 0.0), (5.0, 2.0, 0.0), (5.0, 1.5, 1.5)])
def test_invalid_coefficients(b, c, e):
    with pytest.raises(ValueError):
        MagicFormulaCoefficients(b, c, e)


def test_steady_state_slip_ratio_matches_definition():
    # Wheel spinning 5 % faster than the ground: κ = (ωR − v)/|v| = 0.05.
    tyre = TyreModel(parameters(), 1)
    v, omega = np.array([10.0]), np.array([10.5 / 0.3])
    for _ in range(2000):
        tyre.advance(0.001, v, np.zeros(1), omega, np.array([True]))
    kappa, tan_alpha = tyre.slips()
    assert kappa[0] == pytest.approx(0.05, rel=1e-6)
    assert tan_alpha[0] == pytest.approx(0.0)
    forces = tyre.forces(np.array([3000.0]), np.ones(1), v, np.zeros(1), omega)
    expected = 3000.0 * magic_formula(DRY, 0.05)
    assert forces.longitudinal_n[0] == pytest.approx(expected, rel=1e-6)


def test_lateral_force_opposes_sideslip_and_respects_friction():
    tyre = TyreModel(parameters(), 1)
    v, vy = np.array([10.0]), np.array([3.0])  # sliding left
    for _ in range(3000):
        tyre.advance(0.001, v, vy, v / 0.3, np.array([True]))
    forces = tyre.forces(np.array([2000.0]), np.ones(1), v, vy, v / 0.3)
    assert forces.lateral_n[0] < 0
    assert abs(forces.lateral_n[0]) <= 2000.0 + 1e-9


def test_combined_slip_stays_inside_friction_ellipse():
    tyre = TyreModel(parameters(friction_lateral=1.2), 3)
    tyre.u_m = np.array([0.02, 0.05, 0.2]) * 0.1
    tyre.v_m = np.array([0.05, 0.2, 0.3]) * 0.2
    load = np.full(3, 1000.0)
    forces = tyre.forces(
        load, np.ones(3), np.full(3, 10.0), np.zeros(3), np.full(3, 10.0 / 0.3)
    )
    ellipse = (forces.longitudinal_n / 1000.0) ** 2 + (forces.lateral_n / 1200.0) ** 2
    assert np.all(ellipse <= 1 + 1e-9)


def test_parked_tyre_acts_as_spring_and_damper():
    # At standstill the deflection holds a force (no creep on a slope).
    tyre = TyreModel(parameters(), 1)
    tyre.advance(0.01, np.array([-0.01]), np.zeros(1), np.zeros(1), np.array([True]))
    assert tyre.u_m[0] == pytest.approx(1e-4, rel=1e-2)
    forces = tyre.forces(
        np.array([1000.0]), np.ones(1), np.zeros(1), np.zeros(1), np.zeros(1)
    )
    assert forces.longitudinal_n[0] > 0


def test_deflection_is_clamped_and_released_without_contact():
    tyre = TyreModel(parameters(), 1)
    for _ in range(1000):
        tyre.advance(
            0.01, np.zeros(1), np.zeros(1), np.array([100.0]), np.array([True])
        )
    assert tyre.slips()[0][0] == pytest.approx(MAX_SLIP)
    tyre.advance(0.001, np.zeros(1), np.zeros(1), np.zeros(1), np.array([False]))
    assert tyre.u_m[0] == 0.0 and tyre.v_m[0] == 0.0


def test_no_load_no_force():
    tyre = TyreModel(parameters(), 2)
    tyre.u_m[:] = 0.01
    forces = tyre.forces(
        np.array([0.0, -5.0]), np.ones(2), np.zeros(2), np.zeros(2), np.zeros(2)
    )
    np.testing.assert_array_equal(forces.longitudinal_n, 0.0)
    np.testing.assert_array_equal(forces.rolling_resistance_nm, 0.0)


def test_low_speed_damping_never_exceeds_friction():
    # Locked wheel sliding at 1 m/s (inside the damping band).
    tyre = TyreModel(parameters(low_speed_damping_n_s_per_m=1e5), 1)
    for _ in range(100):
        tyre.advance(0.001, np.array([1.0]), np.zeros(1), np.zeros(1), np.array([True]))
    forces = tyre.forces(
        np.array([100.0]), np.ones(1), np.array([1.0]), np.zeros(1), np.zeros(1)
    )
    assert forces.longitudinal_n[0] == pytest.approx(-100.0)
