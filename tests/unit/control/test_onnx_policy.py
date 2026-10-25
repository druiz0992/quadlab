"""ONNX policy controller with fake sessions (no onnxruntime needed)."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from robotlab.control.legged import LeggedControlState, TrunkState, WalkCommand
from robotlab.control.onnx_policy import (
    OnnxPolicyController,
    observation_width,
    projected_gravity,
)
from robotlab.core.algorithms import AlgorithmContext, SensorDescription
from robotlab.core.measurements import ImuReading, Sample
from robotlab.sensors.bus import STREAMS
from robotlab.sensors.joint_encoders import JointStateReading
from robotlab.testing.plugins import (
    check_legged_controller,
    legged_context,
    nominal_legged,
)

ROBOT = nominal_legged()
N = ROBOT.joint_count
WIDTH = observation_width(N)


@dataclass
class Tensor:
    name: str
    shape: list[Any]


class FakeSession:
    """Feed-forward (or LSTM) session returning ``action`` and recording feeds."""

    def __init__(
        self,
        metadata: dict[str, str] | None = None,
        action: Any = 0.1,
        lstm: tuple[int, int] | None = None,
        width: int = WIDTH,
    ) -> None:
        self.metadata = metadata or {}
        self.action = action
        self.lstm = lstm
        self.width = width
        self.feeds: list[dict[str, np.ndarray]] = []

    def get_inputs(self) -> list[Tensor]:
        inputs = [Tensor("obs", [1, self.width])]
        if self.lstm:
            layers, hidden = self.lstm
            inputs += [
                Tensor("h_in", [layers, 1, hidden]),
                Tensor("c_in", [layers, 1, hidden]),
            ]
        return inputs

    def get_outputs(self) -> list[Tensor]:
        names = ["actions", "h_out", "c_out"] if self.lstm else ["actions"]
        return [Tensor(name, []) for name in names]

    def get_modelmeta(self) -> Any:
        return type("Meta", (), {"custom_metadata_map": self.metadata})()

    def run(self, _: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append({k: v.copy() for k, v in feeds.items()})
        action = np.full((1, N), self.action, dtype=np.float32)
        if not self.lstm:
            return [action]
        return [action, feeds["h_in"] + 1, feeds["c_in"] + 2]


def controller(
    session: FakeSession, context: AlgorithmContext | None = None, **options: Any
) -> OnnxPolicyController:
    return OnnxPolicyController(
        context or legged_context(ROBOT), options, session=session
    )


def state(
    t_ns: int = 20_000_000, *, trunk_pitch_rad: float | None = 0.0, imu: bool = True
) -> LeggedControlState:
    joints = Sample(
        "sensors/joint_encoders",
        1,
        0,
        t_ns,
        0,
        STREAMS["joint_encoders"][1],
        JointStateReading(
            t_ns,
            ROBOT.joint_names,
            (0,) * N,
            np.arange(N, dtype=np.float64) * 0.1,
            np.full(N, 0.5),
        ),
    )
    reading = Sample(
        "sensors/imu",
        1,
        0,
        t_ns,
        0,
        STREAMS["imu"][1],
        ImuReading(np.array([0.0, 0.0, 9.81]), np.array([0.1, 0.2, 0.3])),
    )
    trunk = None
    if trunk_pitch_rad is not None:
        half = trunk_pitch_rad / 2
        trunk = TrunkState(
            np.zeros(3),
            np.zeros(3),
            np.array([0, np.sin(half), 0, np.cos(half)]),
            np.zeros(3),
        )
    return LeggedControlState(t_ns, joints, reading if imu else None, trunk)


def test_projected_gravity_of_a_forward_lean() -> None:
    np.testing.assert_allclose(projected_gravity(np.array([0, 0, 0, 1.0])), [0, 0, -1])
    half = 0.3 / 2
    g = projected_gravity(np.array([0, np.sin(half), 0, np.cos(half)]))
    np.testing.assert_allclose(g, [np.sin(0.3), 0, -np.cos(0.3)], atol=1e-12)


def test_observation_layout_and_targets() -> None:
    session = FakeSession({"action_scale": "0.5"})
    policy = controller(session, twist=[0.2, 0.0, 0.0])
    obs = policy.observation(state(), None)
    assert obs.dtype == np.float32 and obs.shape == (WIDTH,)
    np.testing.assert_allclose(obs[0:3], [0.1, 0.2, 0.3], atol=1e-7)  # gyro
    np.testing.assert_allclose(obs[3:6], [0, 0, -1])  # gravity from trunk truth
    np.testing.assert_allclose(
        obs[6 : 6 + N], np.arange(N) * 0.1, atol=1e-6
    )  # minus 0 default
    np.testing.assert_allclose(obs[6 + N : 6 + 2 * N], 0.5)
    np.testing.assert_allclose(obs[6 + 2 * N : 6 + 3 * N], 0.0)  # no previous action
    np.testing.assert_allclose(obs[6 + 3 * N :], [0.2] + [0.0] * 12, atol=1e-7)
    targets = policy.update(state(), None, 0.0)
    assert targets.position_rad == pytest.approx((0.05,) * N)
    second = policy.observation(state(), None)
    np.testing.assert_allclose(
        second[6 + 2 * N : 6 + 3 * N], 0.1, atol=1e-7
    )  # last action


def test_request_overrides_twist_and_twist_waits() -> None:
    policy = controller(FakeSession(), twist=[0.3, 0.0, 0.0], twist_after_s=1.0)
    assert policy.observation(state(500_000_000), None)[6 + 3 * N] == 0.0
    assert policy.observation(state(1_000_000_000), None)[6 + 3 * N] == pytest.approx(
        0.3
    )
    command = policy.observation(state(), WalkCommand(-0.1, 0.05, 0.7))[
        6 + 3 * N : 6 + 3 * N + 3
    ]
    np.testing.assert_allclose(command, [-0.1, 0.05, 0.7], atol=1e-7)


def test_imu_mount_and_sensor_only_gravity() -> None:
    # IMU yawed 90 deg: sensor x is body y.
    s = np.sqrt(0.5)
    sensors = {
        "imu": SensorDescription("imu", "biped/imu", 250.0, (0, 0, 0), (0, 0, s, s)),
        "joint_encoders": SensorDescription("joint_encoders", "biped/joints", 50.0),
    }
    context = AlgorithmContext(
        legged_context(ROBOT).gps_reference,
        legged_context(ROBOT).vehicle,
        sensors,
        legged=ROBOT,
    )
    policy = controller(FakeSession(), context)
    obs = policy.observation(state(trunk_pitch_rad=None), None)
    np.testing.assert_allclose(obs[0:3], [-0.2, 0.1, 0.3], atol=1e-6)
    np.testing.assert_allclose(obs[3:6], [0, 0, -1], atol=1e-7)  # -f/|f|
    first = controller(FakeSession()).observation(
        state(trunk_pitch_rad=None, imu=False), None
    )
    np.testing.assert_allclose(first[0:6], [0, 0, 0, 0, 0, -1])


def test_metadata_default_pose_and_checks() -> None:
    names = ",".join(ROBOT.joint_names)
    meta = {"joint_names": names, "default_joint_pos": ",".join(["0.1"] * N)}
    policy = controller(FakeSession(meta))
    assert policy.update(state(), None, 0.0).position_rad == pytest.approx((0.2,) * N)
    with pytest.raises(ValueError, match="names and order"):
        controller(FakeSession({"joint_names": ",".join(reversed(ROBOT.joint_names))}))
    with pytest.raises(ValueError, match="observation layout"):
        controller(FakeSession({"observation_names": "base_ang_vel,joint_pos"}))
    with pytest.raises(ValueError, match="width 37"):
        controller(FakeSession(width=61))
    with pytest.raises(ValueError, match="action_scale"):
        controller(FakeSession(), action_scale=0.0)
    with pytest.raises(ValueError, match="unknown options"):
        controller(FakeSession(), gain=1)
    with pytest.raises(ValueError, match="twist"):
        controller(FakeSession(), twist=[0.1])


def test_lstm_state_is_carried_between_periods() -> None:
    session = FakeSession(lstm=(1, 4))
    policy = controller(session)
    policy.update(state(), None, 0.0)
    policy.update(state(), None, 0.02)
    np.testing.assert_array_equal(session.feeds[0]["h_in"], np.zeros((1, 1, 4)))
    np.testing.assert_array_equal(session.feeds[1]["h_in"], np.ones((1, 1, 4)))
    np.testing.assert_array_equal(session.feeds[1]["c_in"], np.full((1, 1, 4), 2.0))
    assert policy.summary()["recurrent"] is True


def test_non_finite_action_and_missing_inputs_raise() -> None:
    with pytest.raises(RuntimeError, match="non-finite"):
        controller(FakeSession(action=np.nan)).update(state(), None, 0.0)
    no_joints = LeggedControlState(0, None, None)
    with pytest.raises(RuntimeError, match="no joint encoder"):
        controller(FakeSession()).observation(no_joints, None)
    holding = controller(FakeSession())
    assert holding.update(no_joints, None, 0.0).position_rad == (0.0,) * N
    assert holding.summary()["periods_without_joints"] == 1
    with pytest.raises(ValueError, match="policy_path"):
        OnnxPolicyController(legged_context(ROBOT), {})
    with pytest.raises(ValueError, match="not found"):
        OnnxPolicyController(legged_context(ROBOT), {"policy_path": "/no/policy.onnx"})


def test_contract_kit_accepts_the_policy_controller() -> None:
    def factory(context: AlgorithmContext, options: Any) -> OnnxPolicyController:
        return OnnxPolicyController(context, options, session=FakeSession())

    report = check_legged_controller(factory)
    # First probes: no samples yet (default pose held), then 0.1 rad actions.
    assert report.commands[0].position_rad == (0.0,) * N
    assert report.commands[-1].position_rad == pytest.approx((0.1,) * N)
    assert report.summary["periods_without_joints"] == 2
