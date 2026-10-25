"""Legged controller running an exported RL policy (ONNX), e.g. the Microduck's.

Follows Pollen Robotics' Microduck policy contract (microduck
``docs/policy-manifest.md`` and ``docs/recurrent-policies.md``): every
control period (50 Hz for the Microduck) the policy maps one observation to
one action per joint, and the joint target is ``default + scale · action``.
The observation is, in order (body axes, SI units):

1. ``base_ang_vel`` (3): gyro rate, from the newest IMU sample rotated by
   the IMU mount into trunk axes;
2. ``projected_gravity`` (3): the unit gravity direction in trunk axes,
   Rᵀ (0, 0, −1). With trunk truth (control state source ``truth``) R is
   the true attitude, standing in for the robot's onboard IMU fusion; with
   ``sensors`` it is −f/|f| of the newest specific force f, which is exact
   only when the trunk is not accelerating;
3. ``joint_pos`` (n): encoder angles minus the default pose;
4. ``joint_vel`` (n): encoder velocities;
5. ``actions`` (n): the previous raw action (zeros at first);
6. ``command`` (13): twist (forward m/s, left m/s, yaw rate rad/s) from the
   ``WalkCommand`` request, else option ``twist`` once ``twist_after_s`` has
   passed; then head pose (4) and body pose (6), held at zero.

Before the first IMU sample (the first period) the gyro reads zero and
gravity points straight down. Before the first joint encoder sample the
policy is not run: the default pose is held, counted in ``summary()``
(``periods_without_joints``). The model's metadata, when present, must name
the robot's joints in the same order (``joint_names``) and this observation
layout; ``default_joint_pos`` and ``action_scale`` come from it unless the
options override them. Feed-forward models take ``obs`` [1, W]; LSTM
exports also take ``h_in``/``c_in`` [layers, 1, hidden] and return
``h_out``/``c_out``, which are kept from period to period (zero at start).
A non-finite action raises RuntimeError. ``onnxruntime`` (the ``policies``
extra) is imported only when a policy file is loaded.
"""

from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.algorithms import AlgorithmContext
from ..core.legged import JointTargets
from .legged import LeggedControlState, WalkCommand

OBSERVATION_NAMES = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
    "head_command",
    "body_command",
)
COMMAND_WIDTH = 13  # twist 3, head pose 4, body pose 6
DEFAULTS: dict[str, Any] = dict(
    policy_path="",
    action_scale=None,  # None: the model's metadata, else 1.0
    twist=[0.0, 0.0, 0.0],
    twist_after_s=0.0,
)


def observation_width(joint_count: int) -> int:
    """Observation length for ``joint_count`` joints (61 for 14)."""
    return 6 + 3 * joint_count + COMMAND_WIDTH


def projected_gravity(quaternion_xyzw: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rᵀ (0, 0, −1) for the body-to-world quaternion (x, y, z, w)."""
    x, y, z, w = quaternion_xyzw / np.linalg.norm(quaternion_xyzw)
    # Minus the third row of R (world z expressed in body axes).
    return -np.array(
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    )


class OnnxPolicyController:
    """``LeggedController`` running an ONNX policy (see module docs).

    ``session`` replaces ``onnxruntime.InferenceSession(policy_path)`` (tests
    pass a fake); it needs ``get_inputs``, ``get_outputs``, ``run``, and
    ``get_modelmeta``.
    """

    def __init__(
        self,
        context: AlgorithmContext,
        options: Mapping[str, Any],
        session: Any | None = None,
    ) -> None:
        if context.legged is None:
            raise ValueError("OnnxPolicyController needs a legged robot")
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"OnnxPolicyController: unknown options {sorted(unknown)}")
        self.options = {**DEFAULTS, **options}
        self._check_sensors(context)
        self.robot = context.legged
        self._imu_to_body = context.sensors["imu"].to_body()[0]
        self._sha256: str | None = None
        if session is None:
            session, self._sha256 = _load_session(str(self.options["policy_path"]))
        self.session = session
        metadata = dict(session.get_modelmeta().custom_metadata_map)
        self.default = self._default_pose(metadata)
        self.scale = self._action_scale(metadata)
        self.twist = _twist(self.options)
        self._inputs = self._check_io(session)
        self._state = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in self._inputs.items()
            if name != "obs"
        }
        self.last_action = np.zeros(self.robot.joint_count, dtype=np.float32)
        self._gravity = np.array([0.0, 0.0, -1.0])
        self.inferences = 0
        self.periods_without_joints = 0

    @staticmethod
    def _check_sensors(context: AlgorithmContext) -> None:
        missing = {"imu", "joint_encoders"} - set(context.sensors)
        if missing:
            raise ValueError(
                f"OnnxPolicyController needs sensors {sorted(missing)} enabled"
            )

    def _default_pose(self, metadata: Mapping[str, str]) -> NDArray[np.float64]:
        """Default pose from the metadata (joint order checked), else the
        start keyframe's targets."""
        names = metadata.get("joint_names")
        if names is not None and tuple(names.split(",")) != self.robot.joint_names:
            raise ValueError(
                f"policy joints {names} differ from the robot's "
                f"{','.join(self.robot.joint_names)} (names and order must match)"
            )
        layout = metadata.get("observation_names")
        if layout is not None and tuple(layout.split(",")) != OBSERVATION_NAMES:
            raise ValueError(
                f"policy observation layout {layout} is not {','.join(OBSERVATION_NAMES)}"
            )
        if "default_joint_pos" not in metadata:
            return np.array(self.robot.initial_target_rad)
        default = np.array([float(v) for v in metadata["default_joint_pos"].split(",")])
        if default.shape != (self.robot.joint_count,):
            raise ValueError("policy default_joint_pos has the wrong joint count")
        return default

    def _action_scale(self, metadata: Mapping[str, str]) -> float:
        value = self.options["action_scale"]
        if value is None:
            value = float(metadata.get("action_scale", 1.0))
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("action_scale must be a positive number")
        return float(value)

    def _check_io(self, session: Any) -> dict[str, tuple[int, ...]]:
        """Input name -> shape; raises ValueError for an unsupported signature."""
        inputs = {i.name: i.shape for i in session.get_inputs()}
        outputs = [o.name for o in session.get_outputs()]
        width = observation_width(self.robot.joint_count)
        if "obs" not in inputs or list(inputs["obs"])[-1] != width:
            raise ValueError(f"policy needs an input 'obs' of width {width}")
        if set(inputs) == {"obs"}:
            if len(outputs) != 1:
                raise ValueError("a feed-forward policy needs exactly one output")
            return {"obs": (1, width)}
        if set(inputs) != {"obs", "h_in", "c_in"} or not {
            "actions",
            "h_out",
            "c_out",
        } <= set(outputs):
            raise ValueError(
                "unsupported policy signature: inputs obs (+ h_in, c_in for LSTM), "
                "outputs actions (+ h_out, c_out)"
            )
        shapes: dict[str, tuple[int, ...]] = {"obs": (1, width)}
        for name in ("h_in", "c_in"):
            shape = tuple(inputs[name])
            if len(shape) != 3 or not all(isinstance(d, int) and d > 0 for d in shape):
                raise ValueError(
                    f"LSTM state {name} needs a static [layers, 1, hidden] shape"
                )
            shapes[name] = shape
        if shapes["h_in"] != shapes["c_in"]:
            raise ValueError("LSTM h_in and c_in shapes differ")
        return shapes

    def observation(
        self, state: LeggedControlState, request: WalkCommand | None
    ) -> NDArray[np.float32]:
        """The policy input for ``state`` (module docs), float32."""
        if state.joints is None:
            raise RuntimeError("OnnxPolicyController: no joint encoder sample yet")
        gyro = np.zeros(3)
        if state.imu is not None:
            gyro = self._imu_to_body @ np.asarray(
                state.imu.payload.best_angular_rate_rad_s
            )
        if state.trunk is not None:
            self._gravity = projected_gravity(state.trunk.quaternion_xyzw)
        elif state.imu is not None:
            force = self._imu_to_body @ np.asarray(
                state.imu.payload.best_specific_force_mps2
            )
            if np.linalg.norm(force) > 0.1:  # keep the last direction in free fall
                self._gravity = -force / np.linalg.norm(force)
        command = np.zeros(COMMAND_WIDTH)
        command[:3] = self._twist(state.sim_time_ns, request)
        joints = state.joints.payload
        return np.concatenate(
            [
                gyro,
                self._gravity,
                np.asarray(joints.position_rad) - self.default,
                joints.velocity_rad_s,
                self.last_action,
                command,
            ]
        ).astype(np.float32)

    def _twist(
        self, sim_time_ns: int, request: WalkCommand | None
    ) -> NDArray[np.float64]:
        if request is not None:
            return np.array(
                [request.forward_mps, request.left_mps, request.yaw_rate_rad_s]
            )
        if sim_time_ns * 1e-9 >= self.options["twist_after_s"]:
            return self.twist
        return np.zeros(3)

    def update(
        self, state: LeggedControlState, request: WalkCommand | None, dt_s: float
    ) -> JointTargets:
        if state.joints is None:
            self.periods_without_joints += 1
            return JointTargets(tuple(self.default))
        feeds = {"obs": self.observation(state, request)[None, :], **self._state}
        names = [o.name for o in self.session.get_outputs()]
        results = dict(zip(names, self.session.run(None, feeds)))
        action_name = "actions" if "actions" in results else names[0]
        action = np.asarray(results[action_name], dtype=np.float32).reshape(-1)
        if action.shape != (self.robot.joint_count,) or not np.all(np.isfinite(action)):
            raise RuntimeError(
                f"policy returned {action.shape[0]} actions or non-finite values "
                f"(needs {self.robot.joint_count} finite)"
            )
        for name, output in (("h_in", "h_out"), ("c_in", "c_out")):
            if name in self._state:
                self._state[name] = np.asarray(results[output], dtype=np.float32)
        self.last_action = action
        self.inferences += 1
        return JointTargets(tuple(self.default + self.scale * action.astype(float)))

    def summary(self) -> dict[str, Any]:
        return dict(
            policy_path=self.options["policy_path"],
            policy_sha256=self._sha256,
            recurrent="h_in" in self._state,
            action_scale=self.scale,
            default_pose_rad=self.default.tolist(),
            inferences=self.inferences,
            periods_without_joints=self.periods_without_joints,
        )


def _twist(options: Mapping[str, Any]) -> NDArray[np.float64]:
    twist = np.asarray(options["twist"], dtype=float)
    if twist.shape != (3,) or not np.all(np.isfinite(twist)):
        raise ValueError("twist needs three finite numbers (forward, left, yaw rate)")
    after = float(options["twist_after_s"])
    if not math.isfinite(after) or after < 0:
        raise ValueError("twist_after_s must be a nonnegative number")
    return twist


def _load_session(path: str) -> tuple[Any, str]:
    """``onnxruntime`` session for ``path`` and the file's SHA-256; raises
    ValueError when the file or the package is missing."""
    if not path:
        raise ValueError("OnnxPolicyController needs option policy_path (.onnx file)")
    file = Path(path).expanduser()
    if not file.is_file():
        raise ValueError(
            f"policy {file} not found (download one from "
            "https://huggingface.co/pollen-robotics/microduck-policies)"
        )
    try:
        import onnxruntime
    except ImportError as error:
        raise ValueError(
            f"{error}. Install the policies extra: pip install -e .[policies]"
        ) from error
    session = onnxruntime.InferenceSession(
        str(file), providers=["CPUExecutionProvider"]
    )
    return session, hashlib.sha256(file.read_bytes()).hexdigest()
