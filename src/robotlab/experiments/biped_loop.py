"""Step a MuJoCo biped mission: sensors, control, recording, health.

The loop owns the clock: physics step k covers [k·Δ, (k+1)·Δ) with Δ the
integer-nanosecond step, so every timestamp is exact simulation time (never
wall-clock). Each task runs every N steps from t = 0 (rates are validated
as whole numbers of steps, simulation/mujoco/config.py). Within step k at
t = k·Δ:

1. ``mj_step1``: positions, velocities, and their sensors at t;
2. joint encoders due at t read the true joints at t (bus ``joint_encoders``);
3. the sensor bus releases samples due by t; the newest of each stream is
   kept for the controller;
4. control due at t calls the controller (control/legged.py) with those
   samples (and the trunk truth when ``algorithms.control_state_source``
   is ``truth``), clips its targets to the joints' command ranges
   (counted per joint), and applies them (``commands.jsonl``);
5. ``mj_step2``: accelerations at t, then integration to t + Δ;
6. an IMU sample due at t reads the accelerometer and gyro at t (they
   depend on the command applied at t, so the controller sees it no
   earlier than the next step) through the IMU error model (bus ``imu``);
7. truth due at t is recorded (``telemetry.jsonl``) and checked for health.

With an ``operator`` (the keyboard, experiments/biped_keyboard.py) each
control period also asks it for the walking request, and after every step
it may pace the run to real time; closing it ends the run (exit reason
``keyboard_closed``). A ``recorder`` (Rerun) logs every recorded state;
body poses come from MuJoCo's kinematics of step 1 (time t).

The final state at the run's end is always recorded, as is the initial one.
Results (``summary.json``) are written also after errors and Ctrl-C.
"""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, Protocol, TextIO

import numpy as np
from numpy.typing import NDArray

from ..control.legged import LeggedControlState, TrunkState, WalkCommand
from ..core.legged import JointTargets, LeggedProperties
from ..core.measurements import ImuReading, Sample
from ..evaluation.biped_health import BipedHealth
from ..sensors.bus import SensorBus
from ..sensors.imu_errors import ImuErrorModel
from ..sensors.joint_encoders import JointEncoderModel
from ..simulation.mujoco.config import BipedConfig, period_steps
from ..simulation.mujoco.scene import BipedScene
from ..simulation.mujoco.state import BipedTruth, imu_reading, read_truth
from .interrupts import shielded
from .run_summary import final_exit_reason


class Operator(Protocol):
    """Interactive input and display for a run (see module docs)."""

    def request(self, sim_time_ns: int) -> WalkCommand | None:
        """The walking request for the control period at ``sim_time_ns``."""
        ...

    def running(self) -> bool:
        """False once the user closed the window."""
        ...

    def after_step(self, data: Any, sim_time_ns: int) -> None:
        """Called after each physics step with the time it reached."""
        ...

    def close(self) -> None:
        """Close its windows (called once after the run)."""
        ...


class StateRecorder(Protocol):
    """Viewer/recording of recorded states (visualization/biped_rerun.py)."""

    def log_state(
        self, data: Any, truth: BipedTruth, request: WalkCommand | None
    ) -> None:
        """Log one recorded state; ``data`` holds MuJoCo's body poses."""
        ...

    def close(self) -> None:
        """Flush and disconnect (called once after the run)."""
        ...


class _UserExit(Exception):
    """The operator closed: end the run normally."""


@dataclass
class BipedParts:
    """Everything one biped mission needs, built before it starts.

    ``wall_clock`` (seconds, monotonic) only measures the real-time factor.
    """

    config: BipedConfig
    scene: BipedScene
    properties: LeggedProperties
    controller: Any
    controller_spec: str
    bus: SensorBus
    imu_errors: ImuErrorModel
    encoders: JointEncoderModel
    health: BipedHealth
    output: Path
    duration_s: float
    operator: Operator | None = None
    recorder: StateRecorder | None = None
    wall_clock: Callable[[], float] = field(default=time.perf_counter)


class BipedLoop:
    """Run one mission (see module docs); ``run()`` returns ``summary.json``."""

    def __init__(self, parts: BipedParts) -> None:
        self.parts = parts
        settings = parts.config.data
        step_ns = parts.scene.step_ns
        self.step_ns = step_ns
        self.total_steps = round(parts.duration_s * 1e9 / step_ns)
        if self.total_steps < 1:
            raise ValueError("Duration must cover at least one physics step")
        biped, sensors = settings["biped"], settings["sensors"]
        self.every = dict(
            control=period_steps(biped["control_rate_hz"], step_ns, "control"),
            record=period_steps(biped["record_rate_hz"], step_ns, "record"),
            imu=period_steps(sensors["imu"]["rate_hz"], step_ns, "imu"),
            joint_encoders=period_steps(
                sensors["joint_encoders"]["rate_hz"], step_ns, "joint_encoders"
            ),
        )
        self.enabled = dict(
            imu=bool(sensors["imu"]["enabled"]),
            joint_encoders=bool(sensors["joint_encoders"]["enabled"]),
        )
        self.truth_for_control = (
            settings["algorithms"]["control_state_source"] == "truth"
        )
        self.latest: dict[str, Sample] = {}
        self.limited = {name: 0 for name in parts.properties.joint_names}
        self.control_ticks = 0
        self.last_request: WalkCommand | None = None
        self._previous_control_ns: int | None = None
        self.steps_done = 0
        self.completed = False
        self.exit_reason = "duration"
        self.delivered = 0
        self.summary: dict[str, Any] | None = None
        self._files: dict[str, TextIO] = {}
        # Created here so an operator's viewer can show it before the run.
        self.data = parts.scene.new_data()

    def run(self) -> dict[str, Any]:
        """Step the mission; return ``summary.json`` (also written on failure)."""
        import mujoco

        data = self.data
        start_wall = self.parts.wall_clock()
        with ExitStack() as stack:
            for name in ("telemetry", "commands"):
                path = self.parts.output / f"{name}.jsonl"
                self._files[name] = stack.enter_context(path.open("w"))
            try:
                for step in range(self.total_steps):
                    self._step(mujoco, data, step)
                    self.steps_done = step + 1
                mujoco.mj_forward(self.parts.scene.model, data)
                self._record(read_truth(self.parts.scene, data, self._now_ns()))
                self.completed = True
            except KeyboardInterrupt:
                self.exit_reason = "interrupted"
                print("Interrupted; saving recorded data.")
            except _UserExit:
                self.exit_reason = "keyboard_closed"
            finally:
                with shielded():
                    self._write_summary(data, self.parts.wall_clock() - start_wall)
        assert self.summary is not None
        return self.summary

    def _now_ns(self) -> int:
        return self.steps_done * self.step_ns

    def _due(self, task: str, step: int) -> bool:
        return step % self.every[task] == 0

    def _step(self, mujoco: Any, data: Any, step: int) -> None:
        """One physics step with the tasks due at its start (module docs)."""
        scene, bus = self.parts.scene, self.parts.bus
        t_ns = step * self.step_ns
        mujoco.mj_step1(scene.model, data)
        truth = read_truth(scene, data, t_ns)
        if self.enabled["joint_encoders"] and self._due("joint_encoders", step):
            reading = self.parts.encoders.read(
                t_ns, truth.joint_position_rad, truth.joint_velocity_rad_s
            )
            bus.submit("joint_encoders", reading, t_ns)
        for kind, sample in bus.advance(t_ns):
            self.latest[kind] = sample
            self.delivered += 1
        if self._due("control", step):
            self._control(data, truth)
        mujoco.mj_step2(scene.model, data)
        if self.enabled["imu"] and self._due("imu", step):
            force, rate = self.parts.imu_errors.apply(*imu_reading(scene, data), t_ns)
            bus.submit("imu", ImuReading(force, rate), t_ns)
        if self._due("record", step):
            self._record(truth)
            if self.parts.recorder is not None:
                self.parts.recorder.log_state(data, truth, self.last_request)
        if self.parts.operator is not None:
            self.parts.operator.after_step(data, t_ns + self.step_ns)

    def _control(self, data: Any, truth: BipedTruth) -> None:
        """Run the controller and apply its clipped targets (module docs)."""
        t_ns = truth.sim_time_ns
        dt_s = 0.0
        if self._previous_control_ns is not None:
            dt_s = (t_ns - self._previous_control_ns) * 1e-9
        self._previous_control_ns = t_ns
        state = LeggedControlState(
            sim_time_ns=t_ns,
            joints=self.latest.get("joint_encoders"),
            imu=self.latest.get("imu"),
            trunk=_trunk(truth) if self.truth_for_control else None,
        )
        request = None
        operator = self.parts.operator
        if operator is not None:
            if not operator.running():
                raise _UserExit
            request = operator.request(t_ns)
        self.last_request = request
        requested = self.parts.controller.update(state, request, dt_s)
        applied, limited = self._accept(requested)
        self.control_ticks += 1
        self.parts.health.on_command(t_ns, applied)
        data.ctrl[:] = applied
        line: dict[str, Any] = dict(
            sim_time_ns=t_ns,
            joint_target_rad=applied.tolist(),
            requested_rad=list(requested.position_rad),
            limited=limited,
        )
        if request is not None:
            line["walk_command"] = [
                request.forward_mps,
                request.left_mps,
                request.yaw_rate_rad_s,
            ]
        self._files["commands"].write(json.dumps(line) + "\n")

    def _accept(self, requested: Any) -> tuple[NDArray[np.float64], list[int]]:
        """Validate and clip one command; raises RuntimeError when the
        controller returned something the servos cannot accept at all."""
        robot = self.parts.properties
        if not isinstance(requested, JointTargets):
            raise RuntimeError(
                f"{self.parts.controller_spec}: update returned "
                f"{type(requested).__name__}, not JointTargets"
            )
        try:
            requested.validate(robot.joint_count)
        except ValueError as error:
            raise RuntimeError(
                f"{self.parts.controller_spec}: invalid JointTargets: {error}"
            ) from error
        target = np.array(requested.position_rad)
        applied = np.clip(target, robot.target_lower_rad, robot.target_upper_rad)
        limited = [int(i) for i in np.flatnonzero(applied != target)]
        for index in limited:
            self.limited[robot.joint_names[index]] += 1
        return applied, limited

    def _record(self, truth: BipedTruth) -> None:
        self.parts.health.on_truth(truth)
        self._files["telemetry"].write(json.dumps(truth.record()) + "\n")

    def _write_summary(self, data: Any, wall_s: float) -> None:
        self.summary = biped_summary(self, data, wall_s)
        path = self.parts.output / "summary.json"
        path.write_text(json.dumps(self.summary, indent=2) + "\n")


def _trunk(truth: BipedTruth) -> TrunkState:
    return TrunkState(
        position_m=truth.position_m,
        velocity_world_mps=truth.velocity_mps,
        quaternion_xyzw=truth.quaternion_xyzw,
        body_rates_rad_s=truth.angular_velocity_rad_s,
    )


def mujoco_warnings(data: Any) -> dict[str, int]:
    """Nonzero MuJoCo warning counters by name (``mjtWarning``)."""
    import mujoco

    counts = {
        name: int(data.warning[int(value)].number)
        for name, value in mujoco.mjtWarning.__members__.items()
        if name != "mjNWARNING"
    }
    return {name: count for name, count in counts.items() if count > 0}


def biped_summary(loop: BipedLoop, data: Any, wall_s: float) -> dict[str, Any]:
    """``summary.json`` of a finished (or failed) ``BipedLoop``."""
    parts, scene = loop.parts, loop.parts.scene
    sim_s = loop.steps_done * loop.step_ns * 1e-9
    exit_reason = final_exit_reason(loop.completed, loop.exit_reason)
    return dict(
        completed=loop.completed,
        health=parts.health.summary(
            completed=loop.completed,
            exit_reason=exit_reason,
            warnings=mujoco_warnings(data),
        ),
        exit_reason=exit_reason,
        robot=dict(
            kind=parts.config.kind,
            simulator="mujoco",
            model=dict(
                name=scene.name,
                path=scene.source.path,
                commit=scene.source.commit,
                dirty=scene.source.dirty,
                overrides=scene.overrides,
            ),
            joint_names=list(scene.joint_names),
        ),
        control=_control_summary(loop),
        config=dict(
            sources=list(parts.config.sources),
            content_hash=parts.config.content_hash,
        ),
        timing=dict(
            step_ns=loop.step_ns,
            steps=loop.steps_done,
            sim_time_s=sim_s,
            wall_time_s=wall_s,
            realtime_factor=sim_s / wall_s if wall_s > 0 else None,
            periods_steps=loop.every,
        ),
        sensors=dict(
            imu=dict(
                enabled=loop.enabled["imu"],
                error_model=parts.imu_errors.summary(),
            ),
            joint_encoders=parts.encoders.summary(),
        ),
        bus=dict(parts.bus.summary(), delivered_total=loop.delivered),
    )


def _control_summary(loop: BipedLoop) -> dict[str, Any]:
    """``summary.json.control``: controller, state source, ticks, limits."""
    controller = loop.parts.controller
    summary = controller.summary() if hasattr(controller, "summary") else None
    return dict(
        controller=loop.parts.controller_spec,
        state_source="truth" if loop.truth_for_control else "sensors",
        ticks=loop.control_ticks,
        limited_commands=loop.limited,
        controller_summary=summary,
    )
