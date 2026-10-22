"""The running mission of a wheeled robot: one serial lockstep loop.

``WheeledMissionLoop.run`` starts Gazebo (paused, lockstep) and repeats
until the duration elapses in simulation time, the keyboard window closes,
or something fails:

1. housekeeping: keyboard input, Gazebo liveness;
2. take the next message in lockstep order (``LockstepOrderer``);
3. measurement pipeline: error models, recording, sensor bus, estimator
   (shared with multirotors, ``sensor_pipeline.py``);
4. on odometry (the truth step, once per physics period):

   a. decode truth (body twist to world axes, start-up correction);
   b. health, metrics, ``telemetry.jsonl``;
   c. encoders at their rate; at the control rate the controller gets a
      ``WheeledControlState`` (truth or estimate) and the driver request
      and returns ``WheelCommands`` (``commands.jsonl``);
   d. the wheeled model (``simulation/wheeled/model.py``) advances wheels,
      suspension, and tyres one period from the true chassis state and the
      resulting wrench is applied through ApplyLinkWrench for the next
      period (``wheels.jsonl`` at the recording rate);
   e. advance the world by one burst.

On exit it clears the wrench, terminates Gazebo, and writes
``metrics.json`` and ``summary.json`` (also after errors and Ctrl-C),
then re-raises any error. Processing is deterministic in lockstep.
"""

from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path
from queue import Empty
import time
from typing import Any, TextIO

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..control.wheeled import DriverCommand, WheeledControlState
from ..core.wheeled import WheelCommandReceipt, WheelCommands
from ..simulation.gazebo.environment import gazebo_environment
from ..simulation.gazebo.ingress import OPPONENT_KIND_PREFIX
from ..simulation.gazebo.messages import timestamp, timestamp_ns, vector
from ..simulation.gazebo.process import GazeboProcess
from ..simulation.gazebo.stepping import wait_for_subscribers
from ..simulation.wheeled.model import ChassisState
from .interrupts import shielded
from .loop_timing import LoopTiming
from .opponents import latest_states
from .mission_loop import (
    GAZEBO_STOP_TIMEOUT_WALL_S,
    KEYBOARD_POLL_PERIOD_WALL_S,
    NO_TELEMETRY_TIMEOUT_WALL_S,
    STARTUP_TIMEOUT_WALL_S,
    lockstep_stepper,
)
from .replay_log import estimate_record
from .run_summary import final_exit_reason
from .schedule import next_due_ns

# Wheel telemetry is written to wheels.jsonl at most at this rate (Hz).
WHEEL_LOG_RATE_HZ = 50.0
# Lap timing and track limits are evaluated at this rate (Hz).
LAP_RATE_HZ = 100.0
# Falling this far behind the real-time schedule (wall s) restarts it.
PACE_RESET_S = 0.1
# Blocking wait for the world-control reply per burst (ms). The reply is
# ignored (stepping.py); at 500 bursts/s the default 10 ms cap cost ~1.5 ms
# per burst on average.
STEP_REPLY_TIMEOUT_MS = 1


@dataclass(frozen=True)
class WheeledParts:
    """Everything the running mission uses, built before Gazebo starts
    (see wheeled_run.py). Optional parts are None when not configured."""

    args: Any
    config: Any
    output: Path
    command: list[str]
    node: Any
    ingress: Any
    orderer: Any
    barrier: str
    recorder: Any
    pipeline: Any
    stack: Any
    controller: Any | None
    model: Any
    wrench: Any
    encoders: Any
    health: Any
    evaluation: Any
    truth: Any
    odometry_fix: Any
    debug: Any
    keyboard: Any | None
    laps: Any | None
    driver_input: Any | None
    opponents: tuple[Any, ...] = ()
    implement: Any | None = None
    lift_publisher: Any | None = None
    trailer: Any | None = None


class WheeledMissionLoop:
    """Run one wheeled-robot mission (see module docs)."""

    def __init__(self, parts: WheeledParts) -> None:
        self.parts = parts
        self.settings = parts.config.data
        control = self.settings["control"]
        self.period_s = 1.0 / control["odometry_rate_hz"]
        self.control_period_ns = round(1e9 / control["control_rate_hz"])
        self.encoder_period_ns = round(
            1e9 / self.settings["sensors"]["wheel_encoders"]["rate_hz"]
        )
        self.counts: Counter = Counter()
        self.timing = LoopTiming()
        self.positions: list[NDArray[np.float64]] = []
        self.collisions: list[dict[str, Any]] = []
        self.first_time: float | None = None
        self.last_time: float | None = None
        self.started = self.last_received = time.monotonic()
        self.completed = False
        self.exit_reason = "duration"
        self.left_ground_s: float | None = None
        self.stepper: Any | None = None
        self.process: GazeboProcess | None = None
        self.summary: dict[str, Any] | None = None
        self._files: dict[str, TextIO] = {}
        self._bus_now_ns = 0
        self._next_control_ns: int | None = None
        self._last_control_ns: int | None = None
        self._next_encoder_ns: int | None = None
        self._next_wheel_log_ns = 0
        self._next_lap_ns = 0
        self._published_lift_m: float | None = None
        self._pace_origin: tuple[float, float] | None = None
        self._last_pace: tuple[float, float] | None = None
        self._pace_first: tuple[float, float] | None = None
        self._encoder = None
        self._driver = DriverCommand()
        self._receipt: WheelCommandReceipt | None = None
        self._limited: Counter = Counter()
        self._next_keyboard_poll_wall_s = 0.0
        self._last_keyboard_wall_s: float | None = None
        self._keys: frozenset[str] = frozenset()
        self.max_speed_mps = 0.0
        self._opponents = {OPPONENT_KIND_PREFIX + o.name: o for o in parts.opponents}

    # Lifecycle --------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Drive the mission; return ``summary.json`` (also written on failure)."""
        with ExitStack() as stack:
            self._open_files(stack)
            try:
                self.process = GazeboProcess(
                    self.parts.command, self._files["gazebo"], gazebo_environment()
                )
                self._start_stepping()
                while self._step():
                    pass
            except KeyboardInterrupt:
                self.exit_reason = "interrupted"
                print("Interrupted; saving received data.")
            finally:
                with shielded():
                    self._shutdown()
                    self._write_results()
        if self.summary is None:
            raise RuntimeError("WheeledMissionLoop: summary missing after run")
        return self.summary

    def _open_files(self, stack: ExitStack) -> None:
        output = self.parts.output
        names = dict(
            gazebo="gazebo.log",
            telemetry="telemetry.jsonl",
            commands="commands.jsonl",
            wheels="wheels.jsonl",
        )
        if self.parts.stack.estimator is not None:
            names["estimates"] = "estimates.jsonl"
        for key, name in names.items():
            self._files[key] = stack.enter_context((output / name).open("w"))

    def _start_stepping(self) -> None:
        parts = self.parts
        self.stepper = lockstep_stepper(
            parts.node, self.settings, parts.barrier, STEP_REPLY_TIMEOUT_MS
        )
        wait_for_subscribers(
            parts.node.publishers,
            timeout_s=STARTUP_TIMEOUT_WALL_S,
            alive=self._gazebo_alive,
        )
        self.stepper.start()

    def _gazebo_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _shutdown(self) -> None:
        """Clear the wrench, release the barrier, stop Gazebo (never raises)."""
        try:
            self.parts.wrench.clear()
        except Exception:
            pass
        if self.stepper is not None:
            try:
                self.stepper.release()
            except Exception:
                pass
        if self.process is not None:
            self.process.stop(GAZEBO_STOP_TIMEOUT_WALL_S)
        if self.parts.stack.bus is not None:
            self.parts.stack.bus.close()

    def _write_results(self) -> None:
        from .wheeled_summary import build_wheeled_summary

        output = self.parts.output
        self.parts.debug.close()
        metrics = self.parts.evaluation.summary()
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        self.summary = build_wheeled_summary(self, metrics)
        (output / "summary.json").write_text(json.dumps(self.summary, indent=2) + "\n")
        self.parts.recorder.close()

    @property
    def final_exit_reason(self) -> str:
        return final_exit_reason(self.completed, self.exit_reason)

    # One iteration ------------------------------------------------------------

    def _step(self) -> bool:
        """Process at most one message; False when the mission is over."""
        if not self._poll_keyboard():
            self.exit_reason = "keyboard_closed"
            return False
        process = self.process
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Gazebo exited ({process.returncode}); "
                f'see {self.parts.output / "gazebo.log"}'
            )
        try:
            kind, message, received_at = self.parts.orderer.get(
                timeout_s=0.01 if self.parts.keyboard is not None else 0.2
            )
        except Empty:
            self._idle()
            return True
        self.timing.begin(kind, received_at)
        try:
            return self._handle(kind, message)
        finally:
            self.timing.end(self.last_time)

    def _poll_keyboard(self) -> bool:
        """Read held keys at most every KEYBOARD_POLL_PERIOD_WALL_S; False
        once the window has closed."""
        keyboard = self.parts.keyboard
        if keyboard is None:
            return True
        now = time.monotonic()
        if now < self._next_keyboard_poll_wall_s:
            return True
        self._next_keyboard_poll_wall_s = now + KEYBOARD_POLL_PERIOD_WALL_S
        self._keys = keyboard.poll_keys()
        driver_input = self.parts.driver_input
        if driver_input is None:
            raise RuntimeError("WheeledMissionLoop: keyboard without driver input")
        for key in keyboard.drain_presses():
            driver_input.press(key)
        return not keyboard.closed

    def _idle(self) -> None:
        if self.stepper is not None:
            self.stepper.check_stalled()
        if time.monotonic() - self.last_received > NO_TELEMETRY_TIMEOUT_WALL_S:
            raise RuntimeError(
                "No telemetry for 30 seconds; "
                f'see {self.parts.output / "gazebo.log"}'
            )

    def _handle(self, kind: str, message: Any) -> bool:
        parts = self.parts
        self.counts[kind] += 1
        if kind.startswith(OPPONENT_KIND_PREFIX):
            # Truth of another vehicle: no sensor pipeline or estimator.
            state = self._opponents[kind].on_odometry(message, parts.model)
            parts.recorder.log_opponent(state)
            return True
        message = parts.pipeline.apply_errors(kind, message)
        if message is None:
            return True
        t = timestamp(message)
        parts.pipeline.deliver(kind, message, t, self.last_time, parts.stack.bus)
        self._advance_estimator(kind, message)
        if kind == "odometry":
            return self._on_odometry(message, t)
        if kind == "trailer" and parts.trailer is not None:
            parts.trailer.on_joint_state(message)
            return True
        if kind == "contacts":
            for contact in message.contact:
                a, b = contact.collision1.name, contact.collision2.name
                parts.health.on_contact(timestamp_ns(message), a, b)
                self.collisions.append(dict(time=t, a=a, b=b))
        return True

    def _advance_estimator(self, kind: str, message: Any) -> None:
        stack = self.parts.stack
        if stack.bus is None:
            return
        if kind not in ("poses", "contacts"):
            self._bus_now_ns = max(self._bus_now_ns, timestamp_ns(message))
        delivered, estimates, innovations = stack.runner.advance(self._bus_now_ns)
        for delivered_kind, sample in delivered:
            if stack.estimate_source is not None:
                stack.estimate_source.on_delivered(delivered_kind, sample)
        for sample in estimates:
            state = sample.payload
            self._files["estimates"].write(json.dumps(estimate_record(sample)) + "\n")
            self.parts.recorder.log_estimate(sample)
            self.parts.evaluation.on_estimate(sample)
            self.parts.health.on_estimate(
                sample.sim_time_ns,
                state.position_m,
                state.velocity_mps,
                state.quaternion_xyzw,
            )
        self.parts.evaluation.on_innovations(innovations)

    # Truth step ---------------------------------------------------------------

    def _on_odometry(self, message: Any, t: float) -> bool:
        parts = self.parts
        self.last_received = time.monotonic()
        if self.first_time is None:
            self.first_time = t
        self.last_time = parts.truth.sim_time_s = t
        stamp = timestamp_ns(message)
        chassis = self._truth_state(message, stamp)
        rotation = parts.truth.attitude
        parts.pipeline.on_truth(
            stamp,
            chassis.position_m,
            chassis.velocity_world_mps,
            chassis.quaternion_xyzw,
            chassis.body_rates_rad_s,
            rotation.inv().apply(chassis.velocity_world_mps),
            self.counts,
            parts.stack.bus,
        )
        self._evaluate_truth(stamp, t, chassis)
        self._sample_encoders(stamp)
        if self._next_control_ns is None or stamp >= self._next_control_ns:
            self._control(stamp, chassis)
        self._apply_physics(stamp, t, chassis)
        if t - self.first_time >= parts.args.duration:
            self.completed = True
            return False
        self._pace(t)
        if self.stepper is not None:
            self.stepper.on_odometry(stamp)
        return True

    def _pace(self, t: float) -> None:
        """Hold the next burst until wall time reaches the simulated time
        divided by ``simulation.real_time_factor`` (0: unlimited). Never
        catches up after falling behind (no bursts of speed)."""
        now = time.monotonic()
        if self._pace_origin is None:
            self._pace_origin = self._pace_first = (now, t)
        self._last_pace = (now, t)
        factor = self.settings["simulation"]["real_time_factor"]
        if factor <= 0:
            return
        wall0, sim0 = self._pace_origin
        target = wall0 + (t - sim0) / factor
        if target > now:
            time.sleep(target - now)
        elif now - target > PACE_RESET_S:
            # Behind schedule: restart the schedule here instead of rushing.
            self._pace_origin = (now, t)

    @property
    def achieved_real_time_factor(self) -> float | None:
        """Simulated seconds per wall second since the first period."""
        if self._pace_first is None or self._last_pace is None:
            return None
        wall = self._last_pace[0] - self._pace_first[0]
        return (self._last_pace[1] - self._pace_first[1]) / wall if wall > 0 else None

    def _truth_state(self, message: Any, stamp: int) -> ChassisState:
        truth = self.parts.truth
        p = vector(message.pose.position)
        q = message.pose.orientation
        quaternion = np.array([q.x, q.y, q.z, q.w])
        # Gazebo publishes odometry twist in body axes.
        velocity_world = Rotation.from_quat(quaternion).apply(
            vector(message.twist.linear)
        )
        velocity_world, body_rates = self.parts.odometry_fix.correct(
            stamp, p, quaternion, velocity_world, vector(message.twist.angular)
        )
        truth.position_m = p
        truth.velocity_world_mps = velocity_world
        truth.body_rates_rad_s = body_rates
        truth.attitude = Rotation.from_quat(quaternion)
        return ChassisState(p, quaternion, velocity_world, body_rates)

    def _evaluate_truth(self, stamp: int, t: float, chassis: ChassisState) -> None:
        parts = self.parts
        p, v, q = (
            chassis.position_m,
            chassis.velocity_world_mps,
            chassis.quaternion_xyzw,
        )
        parts.evaluation.on_truth(stamp, p, v, q, controlled=False)
        parts.health.on_truth(stamp, p, v, q)
        self.positions.append(p)
        self.max_speed_mps = max(self.max_speed_mps, float(np.linalg.norm(v)))
        left = parts.health.left_ground_area_s
        if self.left_ground_s is None and left is not None:
            self.left_ground_s = left
            print(f"Warning: vehicle left the ground area at {left:.1f} s")
        self._files["telemetry"].write(
            json.dumps(dict(time=t, position=p.tolist(), quaternion=q.tolist())) + "\n"
        )

    def _sample_encoders(self, stamp: int) -> None:
        if self._next_encoder_ns is not None and stamp < self._next_encoder_ns:
            return
        model = self.parts.model
        rows = model.rows(0)  # the ego's wheels (opponents follow)
        self._encoder = self.parts.encoders.read(
            stamp,
            model.wheel_angle_rad[rows],
            model.wheel_speed_rad_s[rows],
            model.actuators.steer_rad[rows],
        )
        self._next_encoder_ns = next_due_ns(
            self._next_encoder_ns, stamp, self.encoder_period_ns
        )
        bus = self.parts.stack.bus
        if bus is not None and self.parts.encoders.enabled:
            bus.submit("wheel_encoders", self._encoder, stamp)

    def _control(self, stamp: int, chassis: ChassisState) -> None:
        """One controller tick: state, driver request, commands to actuators."""
        parts = self.parts
        dt_s = (
            0.0
            if self._last_control_ns is None
            else (stamp - self._last_control_ns) * 1e-9
        )
        self._last_control_ns = stamp
        self._next_control_ns = next_due_ns(
            self._next_control_ns, stamp, self.control_period_ns
        )
        request = self._driver_request(chassis, dt_s)
        state = self._control_state(stamp, chassis)
        if parts.controller is None or state is None:
            commands = WheelCommands.zeros(parts.model.properties.wheel_count)
        else:
            commands = parts.controller.update(state, request, dt_s)
        receipt = parts.model.actuators.accept(commands, stamp)
        self._command_implement(request)
        self._receipt = receipt
        for channel, wheels in receipt.limited.items():
            if wheels:
                self._limited[channel] += 1
        self.counts["control_ticks"] += 1
        record = dict(
            time=stamp * 1e-9,
            drive_torque_nm=receipt.accepted.drive_torque_nm,
            brake_torque_nm=receipt.accepted.brake_torque_nm,
            steer_angle_rad=receipt.accepted.steer_angle_rad,
            limited={k: list(v) for k, v in receipt.limited.items() if v},
            requested=(
                None
                if receipt.requested == receipt.accepted
                else dict(
                    drive_torque_nm=receipt.requested.drive_torque_nm,
                    brake_torque_nm=receipt.requested.brake_torque_nm,
                    steer_angle_rad=receipt.requested.steer_angle_rad,
                )
            ),
        )
        if request is not None:
            record["driver"] = dict(
                throttle=request.throttle,
                brake=request.brake,
                steer=request.steer,
                steering_mode=request.steering_mode,
            )
            parts.recorder.log_driver(stamp * 1e-9, request)
        self._files["commands"].write(json.dumps(record) + "\n")
        parts.recorder.log_receipt(receipt)

    def _step_trailer(self, stamp: int, chassis: ChassisState) -> None:
        trailer = self.parts.trailer
        if trailer is None:
            return
        brake = self.parts.model.actuators.commands[0].brake_torque_nm
        if trailer.step(stamp, chassis, trailer.brake_fraction(brake), self.period_s):
            self.parts.health.on_jackknife(stamp, abs(float(np.degrees(trailer.yaw))))
        if trailer.state is not None:
            self.parts.recorder.log_trailer(stamp * 1e-9, trailer)

    def _command_implement(self, request: DriverCommand | None) -> None:
        """Lower or raise the implement: the controller's optional
        ``implement_lowered()`` decides, else the driver's request."""
        implement, controller = self.parts.implement, self.parts.controller
        if implement is None:
            return
        if controller is not None and hasattr(controller, "implement_lowered"):
            implement.command(bool(controller.implement_lowered()))
        elif request is not None:
            implement.command(request.implement_lowered)

    def _driver_request(
        self, chassis: ChassisState, dt_s: float
    ) -> DriverCommand | None:
        """Keyboard input as a ``DriverCommand`` (None without a keyboard).

        The ramps advance by wall time between keyboard polls, so they feel
        the same whatever the simulation speed.
        """
        driver_input = self.parts.driver_input
        if driver_input is None:
            return None
        now = time.monotonic()
        wall_dt = (
            0.0
            if self._last_keyboard_wall_s is None
            else now - self._last_keyboard_wall_s
        )
        self._last_keyboard_wall_s = now
        forward = float(
            self.parts.truth.attitude.apply([1.0, 0.0, 0.0])
            @ chassis.velocity_world_mps
        )
        self._driver = driver_input.update(self._keys, forward, min(wall_dt, 0.1))
        return self._driver

    def _control_state(
        self, stamp: int, chassis: ChassisState
    ) -> WheeledControlState | None:
        """Truth or estimate, with the latest encoder reading; None while the
        estimate-driven state is not available yet (start-up)."""
        encoder = self._encoder
        if encoder is None:
            raise RuntimeError("encoders not sampled before the first control tick")
        source = self.parts.stack.estimate_source
        if source is None:
            return WheeledControlState(
                stamp,
                chassis.position_m,
                chassis.velocity_world_mps,
                chassis.quaternion_xyzw,
                chassis.body_rates_rad_s,
                encoder.wheel_speed_rad_s,
                encoder.steer_angle_rad,
                opponents=latest_states(self.parts.opponents),
            )
        estimated = source.state(stamp, self.parts.stack.runner.latest_estimate)
        if estimated is None:
            return None
        return WheeledControlState(
            stamp,
            estimated.position_m,
            estimated.velocity_world_mps,
            estimated.quaternion_xyzw,
            estimated.body_rates_rad_s,
            encoder.wheel_speed_rad_s,
            encoder.steer_angle_rad,
            source="estimate",
            opponents=latest_states(self.parts.opponents),
        )

    def _apply_physics(self, stamp: int, t: float, chassis: ChassisState) -> None:
        """Advance every vehicle's wheels in one batched step; apply each
        wrench for the next period. An opponent whose odometry of this
        period is missing steps from its previous state (counted)."""
        parts = self.parts
        states = [chassis]
        dynamic = [o for o in parts.opponents if o.dynamic]
        for opponent in dynamic:
            if opponent.chassis is None or opponent.stamp_ns != stamp:
                self.counts["stale_opponent_states"] += 1
            states.append(opponent.chassis if opponent.chassis is not None else chassis)
        wrenches = parts.model.step_all(states, self.period_s)
        wrench = wrenches[0]
        force, torque = wrench.force_world_n, wrench.torque_world_nm
        if parts.implement is not None:
            draft_force, draft_torque = parts.implement.step(chassis, self.period_s)
            force, torque = force + draft_force, torque + draft_torque
            self._publish_lift(parts.implement.lift_m)
        parts.wrench.set(force, torque)
        if parts.trailer is not None:
            self._step_trailer(stamp, chassis)
        evaluate_laps = stamp >= self._next_lap_ns
        for opponent, opponent_wrench in zip(dynamic, wrenches[1:]):
            if opponent.chassis is not None:
                opponent.after_step(opponent_wrench, parts.model, evaluate_laps)
        telemetry = parts.model.telemetry
        parts.recorder.log_wheels(t, telemetry, self._receipt)
        rotation = parts.truth.attitude
        body_velocity = rotation.inv().apply(chassis.velocity_world_mps)
        body_rates = chassis.body_rates_rad_s
        parts.recorder.log_vehicle(
            t,
            float(body_velocity[0]),
            float(body_rates[2]),
            float(body_velocity[0] * body_rates[2]),
        )
        if parts.laps is not None and stamp >= self._next_lap_ns:
            self._next_lap_ns = stamp + round(1e9 / LAP_RATE_HZ)
            self._update_laps(stamp, t, chassis, telemetry)
        if stamp >= self._next_wheel_log_ns:
            self._next_wheel_log_ns = stamp + round(1e9 / WHEEL_LOG_RATE_HZ)
            record = _wheel_record(t, telemetry, wrench)
            if parts.trailer is not None and parts.trailer.model.telemetry is not None:
                trailer = parts.trailer
                record["trailer"] = dict(
                    yaw_rad=trailer.yaw,
                    pitch_rad=trailer.pitch,
                    normal_load_n=np.round(
                        trailer.model.telemetry.normal_load_n, 3
                    ).tolist(),
                    position_m=np.round(trailer.state.position_m, 4).tolist(),
                )
            if parts.implement is not None:
                record["implement"] = parts.implement.telemetry.__dict__
                parts.recorder.log_implement(t, parts.implement.telemetry)
            self._files["wheels"].write(json.dumps(record) + "\n")

    def _publish_lift(self, lift_m: float) -> None:
        """Gazebo's implement link follows the Python lift position."""
        if lift_m == self._published_lift_m or self.parts.lift_publisher is None:
            return
        from gz.msgs10.double_pb2 import Double

        if not self.parts.lift_publisher.publish(Double(data=float(lift_m))):
            raise RuntimeError("Unable to publish implement lift")
        self._published_lift_m = lift_m

    def _update_laps(
        self, stamp: int, t: float, chassis: ChassisState, telemetry: Any
    ) -> None:
        laps = self.parts.laps
        if laps is None:
            return
        episodes = laps.off_track_episodes
        laps.on_truth(t, chassis.position_m, telemetry.contact_point_m[:, :2])
        if laps.off_track_episodes > episodes:
            self.parts.health.on_off_track(stamp)
        self.parts.recorder.log_track(t, laps)


def _wheel_record(t: float, telemetry: Any, wrench: Any) -> dict[str, Any]:
    """One ``wheels.jsonl`` line: per-wheel arrays (wheel order) and wrench."""
    return dict(
        time=t,
        in_contact=telemetry.in_contact.tolist(),
        compression_m=np.round(telemetry.compression_m, 6).tolist(),
        normal_load_n=np.round(telemetry.normal_load_n, 3).tolist(),
        longitudinal_n=np.round(telemetry.longitudinal_n, 3).tolist(),
        lateral_n=np.round(telemetry.lateral_n, 3).tolist(),
        slip_ratio=np.round(telemetry.slip_ratio, 6).tolist(),
        slip_angle_rad=np.round(telemetry.slip_angle_rad, 6).tolist(),
        wheel_speed_rad_s=np.round(telemetry.wheel_speed_rad_s, 5).tolist(),
        steer_rad=np.round(telemetry.steer_rad, 6).tolist(),
        drive_torque_nm=np.round(telemetry.drive_torque_nm, 3).tolist(),
        friction_scale=telemetry.friction_scale.tolist(),
        force_world_n=np.round(wrench.force_world_n, 3).tolist(),
        torque_world_nm=np.round(wrench.torque_world_nm, 3).tolist(),
    )
