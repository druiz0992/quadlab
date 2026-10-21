"""Reference trajectories: continuity, analytic derivatives, timing."""

import math

import numpy as np
import pytest

from dronelab.control.references import (
    Chirp,
    Circle,
    Doublet,
    FigureEight,
    Step,
    Waypoints,
    build_reference,
    minimum_jerk,
    validate_mission,
)
from dronelab.simulation.gazebo.config import load_drone_config


def numeric_derivative(reference, t, h=1e-5):
    a, b = reference.sample(t - h), reference.sample(t + h)
    return (b.position_m - a.position_m) / (2 * h), (
        b.velocity_mps - a.velocity_mps
    ) / (2 * h)


def mission(**overrides):
    return dict(load_drone_config().data["mission"], **overrides)


def test_minimum_jerk_boundaries_and_peak_speed():
    assert minimum_jerk(0) == (0, 0, 0) and minimum_jerk(1)[0] == pytest.approx(1)
    assert minimum_jerk(0.5)[1] == pytest.approx(1.875)


@pytest.mark.parametrize(
    "reference",
    [
        Circle([0, 0, 1], 2.0, 1.5, "path", 0.0),
        FigureEight([0, 0, 1], 3.0, 2.0, 0.0),
        Chirp([0, 0, 1], "y", 0.5, 0.1, 1.0, 20.0, 0.0),
        Waypoints(
            [0, 0, 1],
            [[3, 0, 1], [3, 4, 2]],
            speed_mps=1.0,
            dwell_s=0.5,
            laps=1,
            yaw_mode="path",
            yaw_rad=0.0,
            yaw_rate_rad_s=1.0,
        ),
    ],
)
def test_velocity_and_acceleration_are_analytic_derivatives(reference):
    for t in np.linspace(0.3, reference.duration_s - 0.3, 17):
        point = reference.sample(t)
        dp, dv = numeric_derivative(reference, t)
        np.testing.assert_allclose(point.velocity_mps, dp, atol=1e-4)
        np.testing.assert_allclose(point.acceleration_mps2, dv, atol=1e-3)


def test_waypoints_respect_speed_turn_first_and_end_at_last_point():
    ref = Waypoints(
        [0, 0, 1],
        [[4, 0, 1], [4, 4, 1]],
        speed_mps=2.0,
        dwell_s=1.0,
        laps=1,
        yaw_mode="path",
        yaw_rad=0.0,
        yaw_rate_rad_s=math.radians(90),
    )
    kinds = [p.kind for p in ref.phases]
    assert kinds == ["fly", "dwell", "turn", "fly", "dwell"]
    speeds = [
        np.linalg.norm(ref.sample(t).velocity_mps)
        for t in np.linspace(0, ref.duration_s, 2000)
    ]
    assert max(speeds) == pytest.approx(2.0, rel=1e-3)
    # 4 m at 2 m/s peak takes 1.875·4/2 = 3.75 s; the 90° turn 1.875·(π/2)/(π/2) s.
    assert ref.duration_s == pytest.approx(3.75 + 1 + 1.875 + 3.75 + 1)
    end = ref.sample(ref.duration_s + 5)
    np.testing.assert_allclose(end.position_m, [4, 4, 1])
    assert end.yaw_rad == pytest.approx(math.pi / 2)


def test_laps_repeat_waypoints_from_mission_start():
    ref = build_reference(
        mission(
            type="waypoints", waypoints_m=[[2, 0, 1], [0, 0, 1]], laps=2, dwell_s=0.0
        ),
        [0, 0, 1],
    )
    assert len(ref.phases) == 4


def test_step_and_doublet_are_discontinuous_on_purpose():
    step = Step([0, 0, 1], [1, 0, 0], 2.0, 0.0)
    assert step.sample(1.999).position_m[0] == 0 and step.sample(2.0).position_m[0] == 1
    doublet = Doublet([0, 0, 1], "z", 0.3, 1.5, 1.0, 0.0)
    assert [
        doublet.sample(t).position_m[2] for t in (0.5, 1.5, 3.0, 5.0)
    ] == pytest.approx([1.0, 1.3, 0.7, 1.0])


def test_chirp_frequency_sweeps_linearly():
    chirp = Chirp([0, 0, 1], "x", 0.5, 0.2, 1.2, 10.0, 0.0)
    assert chirp.frequency_hz(0) == 0.2 and chirp.frequency_hz(5) == pytest.approx(0.7)
    assert chirp.sample(11).position_m[0] == 0


def test_mission_validation():
    with pytest.raises(ValueError, match="waypoints_m"):
        validate_mission(mission(type="waypoints"))
    with pytest.raises(ValueError, match="type"):
        validate_mission(mission(type="loop"))
    assert build_reference(mission(), [0, 0, 1]) is None
