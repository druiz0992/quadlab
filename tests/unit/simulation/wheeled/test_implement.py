import numpy as np
import pytest

from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.simulation.wheeled.ground import GroundModel
from robotlab.simulation.wheeled.implement import (
    STATIC_SPEED_MPS,
    ImplementModel,
    draft_coefficients,
    draft_n,
)
from robotlab.simulation.wheeled.model import ChassisState

SETTINGS = load_drone_config(robot="tractor").data["ground_vehicle"]["implement"]


def chassis(speed=2.0, z=1.023):
    return ChassisState(
        np.array([0.0, 0.0, z]),
        np.array([0, 0, 0, 1.0]),
        np.array([speed, 0, 0]),
        np.zeros(3),
    )


def test_moldboard_draft_matches_asabe_form():
    coefficients = draft_coefficients(dict(SETTINGS, texture="medium"))
    assert coefficients == (652.0, 0.0, 5.1, 0.70)
    # 7 km/h, 1.6 m wide, 20 cm deep, medium soil.
    expected = 0.70 * (652 + 5.1 * 49) * 1.6 * 20
    assert draft_n(coefficients, 7 / 3.6, 1.6, 0.2) == pytest.approx(expected)
    custom = dict(
        SETTINGS,
        type="custom",
        draft_a=100.0,
        draft_b=10.0,
        draft_c=1.0,
        texture_factor=0.5,
    )
    assert draft_n(draft_coefficients(custom), 10 / 3.6, 2.0, 0.1) == pytest.approx(
        0.5 * 300 * 2 * 10
    )


def test_lowering_reaches_working_depth_and_draft_opposes_motion():
    implement = ImplementModel(
        dict(SETTINGS, start_lowered=True), GroundModel(None, [], 100.0)
    )
    for _ in range(4000):
        force, torque = implement.step(chassis(), 0.002)
    t = implement.telemetry
    assert t.depth_m == pytest.approx(SETTINGS["working_depth_m"], abs=1e-3)
    assert force[0] < 0 and force[1] == 0 and force[2] == 0
    assert abs(force[0]) == pytest.approx(t.draft_n)
    assert torque[2] == pytest.approx(0.0)  # on the centreline
    # r × F with r = (x < 0, 0, z < 0) and F = (−D, 0, 0): +y torque
    # (nose down about the centre of mass, as a low drawbar pull does).
    assert torque[1] > 0
    implement.command(False)
    for _ in range(4000):
        implement.step(chassis(), 0.002)
    assert implement.lift_m == 0.0 and implement.telemetry.draft_n == 0.0
    assert implement.summary()["worked_area_m2"] > 0


def test_no_draft_when_stopped():
    implement = ImplementModel(
        dict(SETTINGS, start_lowered=True), GroundModel(None, [], 100.0)
    )
    for _ in range(4000):
        implement.step(chassis(speed=0.0), 0.002)
    assert implement.telemetry.draft_n == 0.0
    slow = ImplementModel(
        dict(SETTINGS, start_lowered=True), GroundModel(None, [], 100.0)
    )
    for _ in range(4000):
        slow.step(chassis(speed=STATIC_SPEED_MPS), 0.002)
    full = draft_n(
        slow.coefficients,
        STATIC_SPEED_MPS,
        SETTINGS["width_m"],
        SETTINGS["working_depth_m"],
    )
    assert slow.telemetry.draft_n == pytest.approx(np.tanh(1.0) * full, rel=1e-3)


def test_lift_travel_limits_depth_when_vehicle_is_high():
    implement = ImplementModel(
        dict(SETTINGS, start_lowered=True), GroundModel(None, [], 100.0)
    )
    for _ in range(4000):
        implement.step(chassis(z=3.0), 0.002)
    assert implement.lift_m == -SETTINGS["lift_travel_m"]
    assert implement.telemetry.depth_m == 0.0
