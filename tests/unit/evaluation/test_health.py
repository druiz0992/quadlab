"""Run health detector: findings, severities, and verdicts (no Gazebo)."""

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.evaluation.health import (
    HealthThresholds,
    RunHealth,
    contact_model,
    tilt_deg,
)

LEVEL = [0.0, 0.0, 0.0, 1.0]
STILL = [0.0, 0.0, 0.0]
S = 1_000_000_000  # nanoseconds per second


def roll(deg):
    return Rotation.from_euler("x", deg, degrees=True).as_quat()


def summary(health, completed=True, exit_reason="duration", dropped=None):
    return health.summary(
        completed=completed, exit_reason=exit_reason, dropped=dropped or {}
    )


def kinds(result):
    return [f["kind"] for f in result["findings"]]


def command(thrust_n, saturated=False, t_s=0.0):
    return dict(
        accepted_thrust_n=thrust_n,
        saturated=saturated,
        release_sim_time_ns=round(t_s * S),
    )


def hover(health, start_s=0.0, end_s=1.0, z=1.0, rate_hz=100):
    for t in np.arange(start_s, end_s, 1 / rate_hz):
        health.on_truth(round(t * S), [0, 0, z], STILL, LEVEL)


def test_quiet_flight_is_ok():
    health = RunHealth(HealthThresholds(), ground_half_m=50)
    hover(health)
    health.on_command(command([4.0] * 4))
    result = summary(health, dropped={"imu": 0})
    assert result["verdict"] == "ok" and result["findings"] == []
    assert result["thresholds"]["flip_tilt_deg"] == 90


def test_helpers():
    assert contact_model("drone::base_link::body") == "drone"
    assert contact_model("robotlab::wall::link::box") == "wall"
    assert contact_model("ground") == "ground"
    assert tilt_deg(LEVEL) == pytest.approx(0)
    assert tilt_deg(roll(120)) == pytest.approx(120)
    assert tilt_deg(2 * np.asarray(roll(30))) == pytest.approx(30)  # unnormalized


def test_flip_needs_sustained_tilt_and_reports_episode_start():
    health = RunHealth(HealthThresholds(flip_tilt_deg=90, flip_duration_s=0.5))
    health.on_truth(0, [0, 0, 1], STILL, LEVEL)
    for t in np.arange(1.0, 1.3, 0.01):  # brief: not a flip
        health.on_truth(round(t * S), [0, 0, 1], STILL, roll(150))
    health.on_truth(round(1.4 * S), [0, 0, 1], STILL, LEVEL)
    assert summary(health)["verdict"] == "ok"
    health = RunHealth(HealthThresholds(flip_tilt_deg=90, flip_duration_s=0.5))
    for i, t in enumerate(np.arange(0.0, 2.0, 0.01)):
        angle = 0 if t < 1 else min(180, 100 + 100 * (t - 1))
        health.on_truth(round(t * S), [0, 0, 1], STILL, roll(angle))
    (finding,) = summary(health)["findings"]
    assert finding["kind"] == "flipped" and finding["severity"] == "failed"
    assert finding["time_s"] == pytest.approx(1.0)
    assert finding["value"] == pytest.approx(180, abs=0.5)
    assert finding["count"] == 1


def test_hard_impact_uses_speed_just_before_contact():
    health = RunHealth(HealthThresholds(hard_impact_speed_mps=2.0))
    for t in np.arange(0.0, 1.0, 0.01):
        health.on_truth(round(t * S), [0, 0, 2 - t], [0, 0, -3.0], LEVEL)
    # Post-impact odometry arrives before the contact message.
    health.on_truth(round(1.0 * S), [0, 0, 0.05], STILL, LEVEL)
    health.on_contact(round(0.995 * S), "drone::base_link::body", "ground::link::c")
    result = summary(health)
    assert result["verdict"] == "failed"
    impact = next(f for f in result["findings"] if f["kind"] == "hard_impact")
    assert impact["value"] == pytest.approx(3.0)
    assert impact["time_s"] == pytest.approx(0.995)
    assert "ground_contact" in kinds(result)  # a touchdown after flight


def test_gentle_landing_warns_and_resting_start_is_silent():
    health = RunHealth(HealthThresholds())
    for t in np.arange(0.0, 3.0, 0.01):
        health.on_truth(round(t * S), [0, 0, 0.05], [0, 0, -0.3], LEVEL)
        if t < 0.5 or t > 2.5:  # resting at start, landing at the end
            health.on_contact(round(t * S), "ground::link::c", "drone::base_link::b")
    result = summary(health)
    assert result["verdict"] == "warning"
    (touchdown,) = result["findings"]
    assert touchdown["kind"] == "ground_contact"
    assert touchdown["time_s"] == pytest.approx(2.51, abs=0.011)
    assert touchdown["count"] == 1


def test_obstacle_contact_fails_per_obstacle_and_payload_is_ignored():
    health = RunHealth(HealthThresholds())
    hover(health)
    health.on_contact(round(0.2 * S), "drone::base_link::body", "payload::link::box")
    for t in (0.3, 0.31, 0.32):
        health.on_contact(round(t * S), "drone::base_link::body", "wall::link::box")
    health.on_contact(round(0.5 * S), "post::link::box", "drone::base_link::body")
    result = summary(health)
    contacts = [f for f in result["findings"] if f["kind"] == "obstacle_contact"]
    assert [(f["subject"], f["count"]) for f in contacts] == [("wall", 3), ("post", 1)]
    assert contacts[0]["time_s"] == pytest.approx(0.3)
    assert result["verdict"] == "failed"


def test_below_ground_follows_terrain_height():
    health = RunHealth(
        HealthThresholds(below_ground_margin_m=0.1),
        ground_height_m=lambda x, y: 0.5 * x,
    )
    health.on_truth(0, [0, 0, -0.05], STILL, LEVEL)  # within the margin
    health.on_truth(S, [2, 0, 0.8], STILL, LEVEL)  # 0.2 m under a slope
    (finding,) = summary(health)["findings"]
    assert finding["kind"] == "below_ground" and finding["time_s"] == 1.0
    assert finding["value"] == pytest.approx(0.2)


def test_left_ground_area_warns_once_with_first_time():
    health = RunHealth(HealthThresholds(), ground_half_m=5)
    for i, x in enumerate([0, 4, 6, 7, 3]):
        health.on_truth(i * S, [x, 0, 1], STILL, LEVEL)
    assert health.left_ground_area_s == 2.0
    (finding,) = summary(health)["findings"]
    assert finding["kind"] == "left_ground_area" and finding["severity"] == "warning"
    assert finding["time_s"] == 2.0 and finding["count"] == 2


def test_non_finite_truth_commands_and_estimates():
    health = RunHealth(HealthThresholds())
    health.on_truth(0, [0, 0, 1], STILL, LEVEL)
    health.on_truth(S, [math.nan, 0, 1], STILL, LEVEL)
    health.on_command(
        dict(
            accepted_thrust_n=[1, math.inf, 1, 1],
            saturated=False,
            release_sim_time_ns=2 * S,
        )
    )
    health.on_estimate(3 * S, [0, 0, 1], [math.nan] * 3, LEVEL)
    health.on_estimate(4 * S, [0, 0, 1], STILL, LEVEL)
    result = summary(health)
    found = {(f["subject"], f["time_s"]) for f in result["findings"]}
    assert found == {("truth", 1.0), ("command", 2.0), ("estimate", 3.0)}
    assert result["verdict"] == "failed"


def test_run_level_saturation_drops_and_exit_reasons():
    health = RunHealth(HealthThresholds(saturated_fraction=0.5, max_dropped_messages=2))
    hover(health)
    health.on_command(dict(command([0] * 4, saturated=True), warmup=True))
    for saturated in (True, True, False):
        health.on_command(command([9.0] * 4, saturated=saturated))
    result = summary(health, dropped={"imu": 2, "gps": 1, "baro": 0})
    by_kind = {f["kind"]: f for f in result["findings"]}
    assert by_kind["saturated"]["value"] == pytest.approx(2 / 3)
    assert by_kind["dropped_messages"]["value"] == 3
    assert "baro" not in by_kind["dropped_messages"]["message"]
    assert all(f["time_s"] is None for f in result["findings"])
    assert result["verdict"] == "warning"
    for reason, expected in (
        ("error", "failed"),
        ("interrupted", "warning"),
        ("keyboard_closed", "ok"),
    ):
        health = RunHealth(HealthThresholds())
        assert summary(health, False, reason)["verdict"] == expected


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"flip_tilt_deg": 0}, "flip_tilt_deg"),
        ({"flip_tilt_deg": 200}, "flip_tilt_deg"),
        ({"saturated_fraction": 1.5}, "saturated_fraction"),
        ({"max_dropped_messages": 1.5}, "integer"),
        ({"touchdown_gap_s": -1}, "touchdown_gap_s"),
        ({"bogus": 1}, "Unknown health keys"),
        ({"no_liftoff_thrust_ratio": 1.0}, "exceed 1"),
    ],
)
def test_threshold_validation(settings, message):
    with pytest.raises(ValueError, match=message):
        HealthThresholds.from_settings(settings)


def test_no_liftoff_when_thrust_moves_nothing():
    weight = 14.7
    health = RunHealth(HealthThresholds(), weight_n=weight)
    for t in np.arange(0.0, 2.0, 0.02):  # resting on the ground, motors ignored
        health.on_truth(round(t * S), [0, 0, 0.06], STILL, LEVEL)
        health.on_command(command([1.3 * weight / 4] * 4, t_s=t))
    result = summary(health)
    (finding,) = result["findings"]
    assert finding["kind"] == "no_liftoff" and result["verdict"] == "failed"
    assert finding["time_s"] == pytest.approx(0.0)
    assert finding["value"] == pytest.approx(1.3)
    assert finding["count"] == 1


def test_climbing_falling_or_hovering_is_not_no_liftoff():
    weight = 14.7
    climb = RunHealth(HealthThresholds(), weight_n=weight)
    for t in np.arange(0.0, 2.0, 0.02):
        climb.on_truth(round(t * S), [0, 0, 0.06 + 0.5 * t * t], [0, 0, t], LEVEL)
        climb.on_command(command([1.3 * weight / 4] * 4, t_s=t))
    fall = RunHealth(HealthThresholds(), weight_n=weight)
    for t in np.arange(0.0, 2.0, 0.02):  # full thrust while falling
        fall.on_truth(round(t * S), [0, 0, 5 - t], [0, 0, -1.0], LEVEL)
        fall.on_command(command([2 * weight / 4] * 4, t_s=t))
    hover_ = RunHealth(HealthThresholds(), weight_n=weight)
    for t in np.arange(0.0, 2.0, 0.02):
        hover_.on_truth(round(t * S), [0, 0, 1], STILL, LEVEL)
        hover_.on_command(command([weight / 4] * 4, t_s=t))
    assert all(summary(h)["verdict"] == "ok" for h in (climb, fall, hover_))


def test_tracking_lost_is_a_sustained_large_error_warning():
    health = RunHealth(HealthThresholds(tracking_lost_m=2.0, tracking_lost_s=3.0))
    for t in np.arange(0.0, 10.0, 0.1):
        health.on_truth(round(t * S), [0, 0, 1], STILL, LEVEL)
        error = 2.5 if 1.0 <= t < 3.0 or t >= 5.0 else 0.5  # first spell too short
        error += 0.1 if t > 9 else 0.0
        health.on_tracking_error(round(t * S), error)
    result = summary(health)
    (finding,) = result["findings"]
    assert finding["kind"] == "tracking_lost" and result["verdict"] == "warning"
    assert finding["time_s"] == pytest.approx(5.0)
    assert finding["value"] == pytest.approx(2.6)
