"""Step a MuJoCo biped mission: sensors, control, recording, health.

The loop owns the clock: physics step k covers [k·Δ, (k+1)·Δ) with Δ the
integer-nanosecond step, so every timestamp is exact simulation time (never
wall-clock). Each task runs every N steps from t = 0 (rates are validated
as whole numbers of steps, simulation/mujoco/config.py). Within step k at
t = k·Δ:

1. ``mj_step1``: positions, velocities, and their sensors at t;
2. joint encoders due at t read the true joints at t (bus ``joint_encoders``);
3. the sensor bus releases samples due by t (an estimator would read them here);
4. control due at t writes the joint targets (``commands.jsonl``); while no
   controller is configured the initial keyframe's targets are held;
5. ``mj_step2``: accelerations at t, then integration to t + Δ;
6. an IMU sample due at t reads the accelerometer and gyro at t (they
   depend on the command applied at t, so the controller sees it no
   earlier than the next step) through the IMU error model (bus ``imu``);
7. truth due at t is recorded (``telemetry.jsonl``) and checked for health.

The final state at the run's end is always recorded, as is the initial one.
Results (``summary.json``) are written also after errors and Ctrl-C.
"""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any, TextIO

import numpy as np
from numpy.typing import NDArray

from ..core.measurements import ImuReading
from ..evaluation.biped_health import BipedHealth
from ..sensors.bus import SensorBus
from ..sensors.imu_errors import ImuErrorModel
from ..sensors.joint_encoders import JointEncoderModel
from ..simulation.mujoco.config import BipedConfig, period_steps
from ..simulation.mujoco.scene import BipedScene
from ..simulation.mujoco.state import BipedTruth, imu_reading, read_truth
from .interrupts import shielded
from .run_summary import final_exit_reason


@dataclass
class BipedParts:
    """Everything one biped mission needs, built before it starts.

    ``wall_clock`` (seconds, monotonic) only measures the real-time factor.
    """

    config: BipedConfig
    scene: BipedScene
    bus: SensorBus
    imu_errors: ImuErrorModel
    encoders: JointEncoderModel
    health: BipedHealth
    output: Path
    duration_s: float
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
        self.target: NDArray[np.float64] = parts.scene.initial_ctrl.copy()
        self.steps_done = 0
        self.completed = False
        self.exit_reason = "duration"
        self.delivered = 0
        self.summary: dict[str, Any] | None = None
        self._files: dict[str, TextIO] = {}

    def run(self) -> dict[str, Any]:
        """Step the mission; return ``summary.json`` (also written on failure)."""
        import mujoco

        data = self.parts.scene.new_data()
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
        self.delivered += len(bus.advance(t_ns))
        if self._due("control", step):
            self._control(data, t_ns)
        mujoco.mj_step2(scene.model, data)
        if self.enabled["imu"] and self._due("imu", step):
            force, rate = self.parts.imu_errors.apply(*imu_reading(scene, data), t_ns)
            bus.submit("imu", ImuReading(force, rate), t_ns)
        if self._due("record", step):
            self._record(truth)

    def _control(self, data: Any, t_ns: int) -> None:
        """Apply the joint targets (rad) for this control period."""
        self.parts.health.on_command(t_ns, self.target)
        data.ctrl[:] = self.target
        line = dict(sim_time_ns=t_ns, joint_target_rad=self.target.tolist())
        self._files["commands"].write(json.dumps(line) + "\n")

    def _record(self, truth: BipedTruth) -> None:
        self.parts.health.on_truth(truth)
        self._files["telemetry"].write(json.dumps(truth.record()) + "\n")

    def _write_summary(self, data: Any, wall_s: float) -> None:
        self.summary = biped_summary(self, data, wall_s)
        path = self.parts.output / "summary.json"
        path.write_text(json.dumps(self.summary, indent=2) + "\n")


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
