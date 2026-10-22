import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.simulation.aerodynamics import (
    RotorAerodynamics,
    hover_induced_velocity,
    quadratic_body_drag_world,
    translational_lift_factor,
    vortex_ring_factor,
)
from robotlab.simulation.gazebo.config import load_drone_config

LEVEL = [0, 0, 0, 1]


def test_quadratic_drag_magnitude_direction_and_body_axes():
    force = quadratic_body_drag_world([2, 0, 0], [0, 0, 0], LEVEL, [0.1, 0.2, 0.3], 1.2)
    np.testing.assert_allclose(force, [-0.5 * 1.2 * 0.1 * 4, 0, 0])
    # Wind cancels airspeed.
    np.testing.assert_allclose(
        quadratic_body_drag_world([3, 0, 0], [3, 0, 0], LEVEL, [1, 1, 1], 1.2), 0
    )
    # Yawed 90°: world-x motion is body −y, so the body-y drag area applies.
    yaw = Rotation.from_euler("z", 90, degrees=True).as_quat()
    force = quadratic_body_drag_world([2, 0, 0], [0, 0, 0], yaw, [0.1, 0.2, 0.3], 1.2)
    np.testing.assert_allclose(force, [-0.5 * 1.2 * 0.2 * 4, 0, 0], atol=1e-12)


def test_induced_velocity_lift_and_ring_shapes():
    vi = hover_induced_velocity(3.777, 0.125, 1.225)
    assert vi == pytest.approx(math.sqrt(3.777 / (2 * 1.225 * math.pi * 0.125**2)))
    assert translational_lift_factor(0.0, vi, 0.15) == 1.0
    assert translational_lift_factor(vi, vi, 0.15) == pytest.approx(
        1 + 0.15 * (1 - 1 / math.sqrt(2))
    )
    assert translational_lift_factor(100 * vi, vi, 0.15) == pytest.approx(
        1.15, abs=2e-3
    )
    assert vortex_ring_factor(-1.0, vi, 0.3) == 1.0  # climbing
    assert vortex_ring_factor(vi, vi, 0.3) == pytest.approx(0.7)  # deepest at d = v_i
    assert vortex_ring_factor(3 * vi, vi, 0.3) > 0.999  # fast descent leaves the ring


def aero(**overrides):
    settings = load_drone_config().data["aerodynamics"]
    for key, value in overrides.items():
        settings[key].update(value)
    return RotorAerodynamics(settings, 3.777, 0.125, 4)


def test_rotor_aerodynamics_uses_axial_and_edgewise_airspeed():
    model = aero(
        translational_lift={"enabled": True},
        vortex_ring={"enabled": True, "fluctuation_std": 0.0},
    )
    vi = model.induced_velocity_mps
    np.testing.assert_allclose(model.factors([0, 0, 0], LEVEL), 1.0)
    edgewise = model.factors([3 * vi, 0, 0], LEVEL)
    assert np.all(edgewise > 1.05)
    descending = model.factors([0, 0, -vi], LEVEL)
    np.testing.assert_allclose(descending, 0.7)
    # Pitched 90°: world −z motion is edgewise for the rotor, not a descent.
    pitched = Rotation.from_euler("y", 90, degrees=True).as_quat()
    assert np.all(model.factors([0, 0, -vi], pitched) > 1.0)
    assert model.min_ring_factor == pytest.approx(0.7)


def test_ring_fluctuation_is_seeded_and_only_inside_the_ring():
    a = aero(vortex_ring={"enabled": True, "fluctuation_std": 0.2, "seed": 5})
    b = aero(vortex_ring={"enabled": True, "fluctuation_std": 0.2, "seed": 5})
    vi = a.induced_velocity_mps
    samples = np.array([a.factors([0, 0, -vi], LEVEL) for _ in range(5000)])
    np.testing.assert_array_equal(b.factors([0, 0, -vi], LEVEL), samples[0])
    assert samples.std() == pytest.approx(0.7 * 0.2, rel=0.1)
    np.testing.assert_allclose(a.factors([0, 0, 0], LEVEL), 1.0)  # hover: no noise


@pytest.mark.parametrize(
    "data",
    [
        {"aerodynamics": {"body_drag": {"drag_area_m2": [-0.1, 0, 0]}}},
        {"aerodynamics": {"translational_lift": {"max_gain": 1.5}}},
        {"aerodynamics": {"vortex_ring": {"thrust_loss": 1.0}}},
        {"aerodynamics": {"air_density_kg_m3": 0}},
    ],
)
def test_invalid_aerodynamics_configuration(tmp_path, data):
    import json

    path = tmp_path / "a.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_drone_config(path)
