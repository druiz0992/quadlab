from robotlab.evaluation.health import HealthThresholds, RunHealth

SETTINGS = dict(
    flip_tilt_deg=90.0,
    flip_duration_s=0.5,
    hard_impact_speed_mps=2.0,
    impact_window_s=0.1,
    below_ground_margin_m=0.1,
)


def detector():
    from robotlab.simulation.gazebo.config import load_drone_config

    thresholds = HealthThresholds.from_settings(load_drone_config().data["health"])
    return RunHealth(
        thresholds,
        own_models=frozenset({"car"}),
        wheeled=True,
        vehicles=frozenset({"car_1"}),
    )


def kinds(health):
    summary = health.summary(completed=True, exit_reason="duration", dropped={})
    return {f["kind"] for f in summary["findings"]}


def test_own_body_is_not_an_obstacle_and_chassis_contact_warns():
    health = detector()
    health.on_truth(0, [0, 0, 0.3], [0, 0, 0], [0, 0, 0, 1])
    health.on_contact(
        10, "robotlab::car::base_link::body", "robotlab::car::base_link::wheel_x"
    )
    assert kinds(health) == set()
    health.on_contact(
        20, "robotlab::car::base_link::body", "robotlab::ground::link::ground_collision"
    )
    assert kinds(health) == {"chassis_ground_contact"}


def test_resting_on_wheels_is_not_a_touchdown():
    health = detector()
    for step in range(100):
        health.on_truth(step * 10_000_000, [0, 0, 0.3], [0, 0, 0], [0, 0, 0, 1])
    assert "ground_contact" not in kinds(health)


def test_obstacle_contact_fails():
    health = detector()
    health.on_truth(0, [0, 0, 0.3], [5, 0, 0], [0, 0, 0, 1])
    health.on_contact(1, "robotlab::car::base_link::body", "robotlab::wall::link::box")
    assert {"obstacle_contact", "hard_impact"} <= kinds(health)


def test_opponent_contact_is_a_warning():
    health = detector()
    health.on_truth(0, [0, 0, 0.3], [0, 0, 0], [0, 0, 0, 1])
    health.on_contact(
        1, "robotlab::car::base_link::body", "robotlab::car_1::base_link::body"
    )
    assert kinds(health) == {"vehicle_contact"}
    health.on_off_track(2)
    summary = health.summary(completed=True, exit_reason="duration", dropped={})
    assert summary["verdict"] == "warning"
