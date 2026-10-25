"""Biped health verdict from synthetic truth (no MuJoCo needed)."""

import numpy as np

from robotlab.evaluation.biped_health import BipedHealth
from robotlab.simulation.mujoco.state import BipedTruth, tilt_deg

SETTINGS = {"fall_tilt_deg": 60.0, "fall_height_fraction": 0.5}


def truth(
    t_ns: int, z: float = 0.2, tilt_deg_y: float = 0.0, nan: bool = False
) -> BipedTruth:
    half = np.radians(tilt_deg_y) / 2
    return BipedTruth(
        sim_time_ns=t_ns,
        position_m=np.array([0.0, 0.0, np.nan if nan else z]),
        quaternion_xyzw=np.array([0.0, np.sin(half), 0.0, np.cos(half)]),
        velocity_mps=np.zeros(3),
        angular_velocity_rad_s=np.zeros(3),
        joint_position_rad=np.zeros(2),
        joint_velocity_rad_s=np.zeros(2),
    )


def summary(health: BipedHealth, **kwargs: object) -> dict:
    args: dict = dict(completed=True, exit_reason="duration", warnings={})
    args.update(kwargs)
    return health.summary(**args)


def test_tilt_from_quaternion() -> None:
    assert abs(tilt_deg(truth(0, tilt_deg_y=30.0).quaternion_xyzw) - 30.0) < 1e-9
    assert tilt_deg(truth(0).quaternion_xyzw) == 0.0


def test_standing_run_is_ok() -> None:
    health = BipedHealth(SETTINGS)
    for k in range(10):
        health.on_truth(truth(k * 20_000_000, tilt_deg_y=5.0))
    result = summary(health)
    assert result["verdict"] == "ok" and result["findings"] == []


def test_tilt_and_low_trunk_are_falls() -> None:
    health = BipedHealth(SETTINGS)
    health.on_truth(truth(1_000_000_000))
    health.on_truth(truth(1_500_000_000, tilt_deg_y=70.0))
    health.on_truth(truth(2_000_000_000, z=0.05, tilt_deg_y=10.0))
    result = summary(health)
    assert result["verdict"] == "failed"
    (fall,) = result["findings"]
    assert fall["kind"] == "fallen" and fall["time_s"] == 0.5 and fall["count"] == 2
    assert abs(fall["value"] - 70.0) < 1e-6


def test_non_finite_truth_command_and_warnings() -> None:
    health = BipedHealth(SETTINGS)
    health.on_truth(truth(0))
    health.on_truth(truth(10_000_000, nan=True))
    health.on_command(20_000_000, [0.0, np.inf])
    result = summary(health, warnings={"mjWARN_BADQACC": 3})
    kinds = {f["kind"]: f for f in result["findings"]}
    assert kinds["non_finite"]["count"] == 2
    assert kinds["simulator_warning"]["count"] == 3
    assert result["verdict"] == "failed"


def test_interrupted_warns_and_errors_fail() -> None:
    interrupted = summary(
        BipedHealth(SETTINGS), completed=False, exit_reason="interrupted"
    )
    assert interrupted["verdict"] == "warning"
    failed = summary(BipedHealth(SETTINGS), completed=False, exit_reason="error")
    assert failed["findings"][0]["kind"] == "incomplete"
