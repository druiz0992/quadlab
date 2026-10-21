"""The running Gazebo mission: one serial loop over transported messages.

``MissionLoop.run`` starts Gazebo, then repeats until the duration elapses
(in simulation time), the keyboard window closes, or something fails:

1. housekeeping: live-EKF and thrust-worker health, keyboard input and
   effects, real-time watchdogs, Gazebo process liveness;
2. take the next message from the ingress queue (lockstep stall checks
   while waiting);
3. measurement pipeline: error models, recording, sensor bus
   (``sensor_pipeline.py``), then the estimator runner and the live EKF;
4. per kind: odometry is the truth step (payload, moving obstacles,
   external forces, synthetic sensors, evaluation, health, telemetry, then
   rotor control or velocity commands and the lockstep advance); poses
   feed payload and moving obstacles; contacts feed the health verdict.

On exit it always stops the motors, terminates Gazebo, and writes
``metrics.json`` and ``summary.json`` (also after errors and Ctrl-C), then
re-raises any error. The order of these steps matches the recorded,
deterministic lockstep outputs; change it only deliberately.
"""

from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path
from queue import Empty
import time
from typing import Any, TextIO, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..control.manual import FlightCommand
from ..control.waypoint import waypoint_velocity
from ..simulation.gazebo.environment import gazebo_environment
from ..simulation.gazebo.messages import timestamp, timestamp_ns, vector
from ..simulation.gazebo.process import GazeboProcess
from ..simulation.gazebo.stepping import wait_for_subscribers
from .interrupts import shielded
from .keyboard_flight import apply_keyboard_effects
from .loop_timing import LoopTiming
from .rotor_loop import RotorControlLoop
from .rotor_mission import RotorMission, RotorMissionParts
from .replay_log import estimate_record
from .run_summary import build_summary, final_exit_reason

# Real-time limits (wall seconds) for a silent simulator.
NO_TELEMETRY_TIMEOUT_WALL_S = 30.0
# Gazebo start-up allowance for command-topic subscriptions (wall s).
STARTUP_TIMEOUT_WALL_S = 60.0
# Time Gazebo gets to exit after SIGTERM before it is killed (wall s).
GAZEBO_STOP_TIMEOUT_WALL_S = 10.0
# Native velocity-mode command period (s, simulation time).
VELOCITY_COMMAND_PERIOD_S = 0.04
# Keyboard window service period (wall s). A Tk update with redraw costs
# ~0.5 ms (spikes to ~20 ms); doing it per message (~500/s at 250 Hz
# odometry) kept a 250 Hz real-time loop near full load. 50 Hz is still
# far below human reaction time.
KEYBOARD_POLL_PERIOD_WALL_S = 0.02


T = TypeVar("T")


def _started(value: T | None, name: str) -> T:
    """``value``; RuntimeError if the loop uses it before creating it (a
    programming error: these exist once ``run`` has started the mission)."""
    if value is None:
        raise RuntimeError(f"MissionLoop: {name} used before it was started")
    return value


@dataclass(frozen=True)
class MissionParts:
    """Everything the running mission uses, built before Gazebo starts.

    Optional parts are None when not configured. ``command`` is the
    ``gz sim`` command line; ``current_wind()`` the wind the vehicle sees;
    ``orderer`` (lockstep only) fixes the processing order of messages.
    """

    args: Any
    config: Any
    true_config: Any
    randomization_draws: dict[str, Any]
    output: Path
    rotor_mode: bool
    lockstep: bool
    barrier: str | None
    keyboard: Any | None
    command_source: Any | None
    recorder: Any
    node: Any
    ingress: Any
    orderer: Any | None
    replay: Any | None
    debug: Any
    command: list[str]
    wind_control: Any | None
    current_wind: Any
    powertrain: Any
    esc: Any
    actuator: Any | None
    publisher: Any | None
    odometry_fix: Any
    payload: Any | None
    winch: Any | None
    moving: Any | None
    forces: Any
    pipeline: Any
    failsafe: Any | None
    effects: Any | None
    stack: Any
    live_ekf: Any | None
    estimate_reader: Any | None
    evaluation: Any
    health: Any
    mission: Any
    takeoff: Any | None
    truth: Any
    terrain: Any | None
    ground_half_m: float


@dataclass(frozen=True)
class RunFiles:
    """Open JSONL outputs of a run (None when that output is not produced)."""

    gazebo_log: TextIO
    telemetry: TextIO
    commands: TextIO
    command_events: TextIO
    estimates: TextIO | None
    effects: TextIO | None
    wind: TextIO | None


def lockstep_stepper(node: Any, settings: dict[str, Any], barrier: str) -> Any:
    """Stepper advancing the paused world one odometry period per burst.

    Bursts are multi-step requests to the world control service. With the
    ``plugin`` barrier each request is preceded by a grant on the
    DronelabLockstep topic (advertised here, so the start-up barrier also
    waits for the plugin's subscription).
    """
    from types import SimpleNamespace

    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.world_control_pb2 import WorldControl

    from ..simulation.gazebo.stepping import (
        GRANT_TOPIC,
        LockstepStepper,
        steps_per_period,
    )

    grant = None
    if barrier == "plugin":
        from gz.msgs10.int64_pb2 import Int64

        publisher = node.advertise(GRANT_TOPIC, Int64)

        def publish_grant(sim_time_ns: int) -> None:
            if not publisher.publish(Int64(data=sim_time_ns)):
                raise RuntimeError("Unable to publish lockstep grant")

        grant = publish_grant

    period_s = 1 / settings["control"]["odometry_rate_hz"]
    return LockstepStepper(
        node.request,
        SimpleNamespace(WorldControl=WorldControl, Boolean=Boolean),
        steps=steps_per_period(period_s, settings["simulation"]["step_s"]),
        period_ns=round(period_s * 1e9),
        grant=grant,
    )


class MissionLoop:
    """Run one Gazebo mission (see module docs)."""

    def __init__(self, parts: MissionParts) -> None:
        self.parts = parts
        self.settings = parts.config.data
        self.counts: Counter = Counter()
        self.timing = LoopTiming()
        self.positions: list[NDArray[np.float64]] = []
        self.collisions: list[dict[str, Any]] = []
        self.first_time: float | None = None
        self.last_time: float | None = None
        self.last_received = time.monotonic()
        self.started = self.last_received
        self.completed = False
        self.exit_reason = "duration"
        self.left_ground_s: float | None = None
        self._last_command_time = -float("inf")
        self._bus_now_ns = 0
        self._pilot_command = FlightCommand()
        self._next_keyboard_poll_wall_s = 0.0
        self.thrust_control: Any | None = None
        self.control_loop: RotorControlLoop | None = None
        self.stepper: Any | None = None
        self.rotor: RotorMission | None = None
        self.process: GazeboProcess | None = None
        self.files: RunFiles | None = None
        self.summary: dict[str, Any] | None = None

    # Lifecycle --------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Fly the mission; return ``summary.json`` (also written on failure)."""
        with ExitStack() as stack:
            self.files = self._open_files(stack)
            try:
                self.process = GazeboProcess(
                    self.parts.command, self._files.gazebo_log, gazebo_environment()
                )
                self._start_control()
                while self._step():
                    pass
            except KeyboardInterrupt:  # Ctrl-C, or SIGTERM via interrupts.py
                self.exit_reason = "interrupted"
                print("Interrupted; saving received data.")
            finally:
                # A second Ctrl-C/SIGTERM must not leave Gazebo or results behind.
                with shielded():
                    self._shutdown()
                    self._write_results()
        return _started(self.summary, "summary")

    def _open_files(self, stack: ExitStack) -> RunFiles:
        output, parts = self.parts.output, self.parts

        def open_file(name: str) -> TextIO:
            return stack.enter_context((output / name).open("w"))

        def open_if(condition: bool, name: str) -> TextIO | None:
            return open_file(name) if condition else None

        return RunFiles(
            estimates=open_if(
                parts.args.ekf or parts.stack.estimator is not None, "estimates.jsonl"
            ),
            gazebo_log=open_file("gazebo.log"),
            telemetry=open_file("telemetry.jsonl"),
            commands=open_file("commands.jsonl"),
            command_events=open_file("command-events.jsonl"),
            effects=open_if(parts.keyboard is not None, "effect-toggles.jsonl"),
            wind=open_if(bool(parts.wind_control), "wind-commands.jsonl"),
        )

    def _start_control(self) -> None:
        """Thrust worker, control loop, and rotor mission; lockstep stepper."""
        parts, settings = self.parts, self.settings
        timing = settings["control"]
        if parts.rotor_mode:
            from ..control.drone import RotorThrustControl as ThrustCommands

            self.thrust_control = ThrustCommands(
                _started(parts.actuator, "actuator").send,
                properties=parts.config.properties,
                publish_rate_hz=timing["publish_rate_hz"],
                max_lifetime_sim_s=timing["command_lifetime_sim_s"],
                # Gazebo cannot advance while Python stalls in lockstep.
                watchdog_wall_s=None if parts.lockstep else timing["watchdog_wall_s"],
                publish_on_send=timing["publish_on_send"],
                # Lockstep: no wall-clock republishing; the loop calls tick()
                # once per odometry period instead (see _heartbeat).
                start_worker=not parts.lockstep,
            )
            handle = self.thrust_control.acquire()
            self.control_loop = RotorControlLoop(
                parts.stack.controller,
                self.thrust_control,
                handle,
                timing,
                # Half a physics step absorbs timestamp quantization.
                tolerance_ns=round(settings["simulation"]["step_s"] * 0.5e9),
            )
            self.rotor = RotorMission(self._rotor_parts())
        if parts.lockstep:
            self.stepper = lockstep_stepper(
                parts.node, settings, _started(parts.barrier, "lockstep barrier")
            )
            # Commands published before Gazebo subscribes are lost: connect
            # every command topic (and the plugin's grant topic) first.
            wait_for_subscribers(
                parts.node.publishers,
                timeout_s=STARTUP_TIMEOUT_WALL_S,
                alive=self._gazebo_alive,
            )
            self.stepper.start()

    def _gazebo_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _rotor_parts(self) -> RotorMissionParts:
        parts = self.parts
        return RotorMissionParts(
            settings=self.settings,
            rotor_count=len(parts.config.properties.rotors),
            controller=parts.stack.controller,
            control_loop=_started(self.control_loop, "control loop"),
            powertrain=parts.powertrain,
            pipeline=parts.pipeline,
            forces=parts.forces,
            recorder=parts.recorder,
            evaluation=parts.evaluation,
            health=parts.health,
            mission=parts.mission,
            takeoff=parts.takeoff,
            failsafe=parts.failsafe,
            estimate_source=parts.stack.estimate_source,
            runner=parts.stack.runner,
            bus=parts.stack.bus,
            keyboard=parts.keyboard,
            command_source=parts.command_source,
            terrain=parts.terrain,
            truth=parts.truth,
            counts=self.counts,
            command_log=self._files.commands,
            lockstep=parts.lockstep,
        )

    def _shutdown(self) -> None:
        """Stop the motors, end Gazebo, and flush workers (never raises on stop)."""
        parts = self.parts
        # Publication failure must not prevent owned-process cleanup.
        try:
            if self.thrust_control is not None:
                self.thrust_control.close()
            elif parts.actuator is not None:
                parts.actuator.stop()
            else:
                from gz.msgs10.twist_pb2 import Twist

                _started(parts.publisher, "velocity publisher").publish(Twist())
        except Exception:
            pass
        if self.stepper is not None:
            try:
                self.stepper.release()  # a held world must be free to exit
            except Exception:
                pass
        if self.process is not None:
            self.process.stop(GAZEBO_STOP_TIMEOUT_WALL_S)
        if self.thrust_control is not None:
            self._drain_command_events()
        if parts.live_ekf is not None:
            parts.live_ekf.close()
            _started(parts.estimate_reader, "estimate reader").close()
        if parts.stack.bus is not None:
            parts.stack.bus.close()

    def _write_results(self) -> None:
        output = self.parts.output
        if self.parts.replay is not None:
            self.parts.replay.close()
        self.parts.debug.close()
        metrics = self.parts.evaluation.summary()
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        self.summary = build_summary(self, metrics)
        (output / "summary.json").write_text(json.dumps(self.summary, indent=2) + "\n")
        self.parts.recorder.close()

    @property
    def _files(self) -> RunFiles:
        return _started(self.files, "run files")

    @property
    def final_exit_reason(self) -> str:
        return final_exit_reason(self.completed, self.exit_reason)

    # One iteration ------------------------------------------------------------

    def _step(self) -> bool:
        """Process at most one message; False when the mission is over."""
        self.timing.begin("housekeeping")
        try:
            running = self._housekeeping()
        finally:
            self.timing.end()
        if not running:
            return False
        try:
            source = self.parts.orderer or self.parts.ingress
            kind, message, received_at = source.get(
                timeout_s=0.01 if self.parts.keyboard is not None else 0.2
            )
        except Empty:
            self._idle()
            return True
        self.timing.begin(kind, received_at)
        try:
            return self._handle(kind, message, received_at)
        finally:
            # last_time is the newest odometry stamp: the simulated span.
            self.timing.end(self.last_time)

    def _housekeeping(self) -> bool:
        """Health checks and keyboard input; False when the keyboard closed."""
        parts = self.parts
        if parts.live_ekf is not None:
            parts.live_ekf.check_health()
            dropped = parts.ingress.dropped
            if dropped["imu"] or dropped["gps"]:
                raise RuntimeError("Sensor ingress loss during live EKF mission")
        if self.thrust_control is not None:
            self.thrust_control.check_health()
            self._drain_command_events()
        if not self._poll_keyboard():
            self.exit_reason = "keyboard_closed"
            return False
        self._check_watchdogs()
        return True

    def _poll_keyboard(self) -> bool:
        """Service the keyboard window at most every
        ``KEYBOARD_POLL_PERIOD_WALL_S``: pilot command, then effect keys
        (stamped with the latest odometry time). Between polls the last
        command holds. False once the window has closed."""
        parts, keyboard = self.parts, self.parts.keyboard
        if keyboard is None:
            return True
        now = time.monotonic()
        if now < self._next_keyboard_poll_wall_s:
            return True
        self._next_keyboard_poll_wall_s = now + KEYBOARD_POLL_PERIOD_WALL_S
        self._pilot_command = keyboard.poll()
        if parts.effects is not None:
            apply_keyboard_effects(
                keyboard,
                parts.effects,
                round((self.last_time or 0) * 1e9),
                _started(self._files.effects, "effect log"),
                parts.recorder,
            )
        return not keyboard.closed

    def _check_watchdogs(self) -> None:
        parts = self.parts
        if (
            parts.rotor_mode
            and not parts.lockstep
            and self.rotor is not None
            and self.rotor.previous_control_time is not None
            and time.monotonic() - self.last_received
            > self.settings["control"]["watchdog_wall_s"]
        ):
            raise RuntimeError(
                "Odometry watchdog expired; ending rotor-control mission"
            )
        process = _started(self.process, "Gazebo process")
        if process.poll() is not None:
            raise RuntimeError(
                f"Gazebo exited ({process.returncode}); "
                f'see {parts.output / "gazebo.log"}'
            )

    def _idle(self) -> None:
        """No message: check for a stalled lockstep world or a silent simulator."""
        if self.stepper is not None:
            self.stepper.check_stalled()
        if time.monotonic() - self.last_received > NO_TELEMETRY_TIMEOUT_WALL_S:
            raise RuntimeError(
                "No telemetry for 30 seconds; "
                f'see {self.parts.output / "gazebo.log"}'
            )

    def _drain_command_events(self) -> None:
        thrust = _started(self.thrust_control, "thrust control")
        for event in thrust.drain_events():
            self._files.command_events.write(json.dumps(event) + "\n")
            self.parts.recorder.log_actuator_event(event)

    def _handle(self, kind: str, message: Any, received_at: float) -> bool:
        """Measurement pipeline, algorithms, then the per-kind step."""
        parts = self.parts
        self.counts[kind] += 1
        # Error and fault models, then recording and the sensor bus
        # (see sensor_pipeline.py); a lost IMU sample goes nowhere.
        message = parts.pipeline.apply_errors(kind, message)
        if message is None:
            return True
        t = timestamp(message)
        if kind == "odometry" and parts.wind_control is not None:
            wind_event = parts.wind_control.update(timestamp_ns(message))
            if wind_event is not None:
                wind_log = _started(self._files.wind, "wind log")
                wind_log.write(json.dumps(wind_event) + "\n")
        gps_dropped = parts.pipeline.deliver(
            kind, message, t, self.last_time, parts.stack.bus
        )
        self._advance_algorithms(kind, message, received_at, gps_dropped)
        if kind == "odometry":
            return self._on_odometry(message, t, received_at)
        if kind == "poses" and (parts.moving is not None or parts.payload is not None):
            self._on_poses(message)
        elif kind == "contacts":
            self._on_contacts(message, t)
        return True

    # Algorithms -------------------------------------------------------------

    def _advance_algorithms(
        self, kind: str, message: Any, received_at: float, gps_dropped: bool
    ) -> None:
        """Deliver due bus samples to the estimator; feed the live EKF."""
        parts = self.parts
        stack = parts.stack
        if stack.bus is not None:
            if kind not in ("poses", "contacts"):
                self._bus_now_ns = max(self._bus_now_ns, timestamp_ns(message))
            delivered, estimates, innovations = stack.runner.advance(self._bus_now_ns)
            if parts.replay is not None:
                for delivered_kind, sample in delivered:
                    parts.replay.on_sample(self._bus_now_ns, delivered_kind, sample)
            for delivered_kind, sample in delivered:
                if stack.estimate_source is not None:
                    stack.estimate_source.on_delivered(delivered_kind, sample)
            for sample in estimates:
                self._write_estimate(sample)
            parts.evaluation.on_innovations(innovations)
        if parts.live_ekf is not None:
            calibrated = parts.pipeline.calibrated_imu
            parts.live_ekf.ingest(
                kind,
                message,
                received_at=received_at,
                gps_dropped=gps_dropped,
                calibrated=(
                    calibrated[:2] if kind == "imu" and calibrated is not None else None
                ),
            )
            self._drain_live_estimates()

    def _write_estimate(self, sample: Any) -> None:
        """Record one estimate (estimates.jsonl, Rerun, metrics, health)."""
        parts, state = self.parts, sample.payload
        estimates = _started(self._files.estimates, "estimates log")
        estimates.write(json.dumps(estimate_record(sample)) + "\n")
        parts.recorder.log_estimate(sample)
        parts.evaluation.on_estimate(sample)
        parts.health.on_estimate(
            sample.sim_time_ns,
            state.position_m,
            state.velocity_mps,
            state.quaternion_xyzw,
        )

    def _drain_live_estimates(self) -> None:
        reader = self.parts.estimate_reader
        if reader is None:
            return
        from ..sensors.streams import ReadTimeout

        while True:
            try:
                sample = reader.next(timeout_wall_s=0)
            except ReadTimeout:
                break
            self._write_estimate(sample)

    # Truth step ---------------------------------------------------------------

    def _on_odometry(self, message: Any, t: float, received_at: float) -> bool:
        """One true state sample; False when the mission duration is over."""
        parts = self.parts
        self.last_received = time.monotonic()
        if self.first_time is None:
            self.first_time = t
        self.last_time = parts.truth.sim_time_s = t
        stamp = timestamp_ns(message)
        p, quaternion, velocity_world, body_rates = self._truth_state(message, t)
        airspeed = velocity_world - parts.current_wind()
        air_body = parts.truth.attitude.inv().apply(airspeed)
        parts.forces.update(
            stamp, velocity_world, parts.current_wind, quaternion, body_rates
        )
        parts.pipeline.on_truth(
            stamp,
            p,
            velocity_world,
            quaternion,
            body_rates,
            air_body,
            self.counts,
            parts.stack.bus,
        )
        self._evaluate_truth(stamp, t, p, velocity_world, quaternion)
        if parts.rotor_mode:
            if not _started(self.rotor, "rotor mission").tick(
                stamp,
                t,
                self.first_time,
                received_at,
                p,
                velocity_world,
                quaternion,
                body_rates,
                air_body,
                airspeed,
                self._pilot_command,
            ):
                # No command yet: still advance the lockstep world.
                self._heartbeat()
                self._advance_world(stamp)
                return True
            self._heartbeat()
        elif t - self._last_command_time >= VELOCITY_COMMAND_PERIOD_S:
            self._velocity_command(p, quaternion)
            self._last_command_time = t
        if t - self.first_time >= parts.args.duration:
            live_ekf = parts.live_ekf
            if live_ekf is not None and live_ekf.node.ekf.accepted_gps == 0:
                raise RuntimeError(
                    "Live EKF has no accepted GPS correction; inspect summary"
                )
            self.completed = True
            return False
        # Everything for this period is done: advance one burst.
        self._advance_world(stamp)
        return True

    def _truth_state(self, message: Any, t: float) -> tuple[NDArray, ...]:
        """Decode odometry; update payload, obstacles, and the truth snapshot.

        Returns position, xyzw quaternion, world velocity, and body rates
        (after Gazebo's start-up velocity correction).
        """
        parts, truth = self.parts, self.parts.truth
        stamp = timestamp_ns(message)
        p = vector(message.pose.position)
        truth.position_m = p
        q = message.pose.orientation
        quaternion = np.array([q.x, q.y, q.z, q.w])
        # Gazebo publishes linear and angular odometry twist in body axes.
        velocity_world = Rotation.from_quat(quaternion).apply(
            vector(message.twist.linear)
        )
        self._update_payload_and_obstacles(stamp, t)
        # Gazebo reports bogus start-up velocities (see odometry.py).
        velocity_world, body_rates = parts.odometry_fix.correct(
            stamp, p, quaternion, velocity_world, vector(message.twist.angular)
        )
        # Latest motion, for camera and lidar effects.
        truth.body_rates_rad_s = body_rates
        truth.velocity_world_mps = velocity_world
        truth.attitude = Rotation.from_quat(quaternion)
        return p, quaternion, velocity_world, body_rates

    def _update_payload_and_obstacles(self, stamp: int, t: float) -> None:
        parts = self.parts
        if parts.winch is not None:
            keyboard = parts.keyboard
            rate = keyboard.winch_rate() if keyboard is not None else 0
            cable_m = parts.winch.update(stamp, rate)
            parts.recorder.log_vector(
                "payload/cable_length_m", [cable_m], labels=["value"]
            )
        if parts.payload is not None:
            payload_event = parts.payload.update(stamp)
            if payload_event is not None and parts.effects is not None:
                parts.effects.state["payload"] = False
        if parts.moving is not None:
            # Path time is absolute simulation time (start at waypoint 0).
            parts.moving.update(t)

    def _evaluate_truth(
        self,
        stamp: int,
        t: float,
        p: NDArray[np.float64],
        velocity_world: NDArray[np.float64],
        quaternion: NDArray[np.float64],
    ) -> None:
        """Metrics, health, ground-area warning, and telemetry.jsonl."""
        parts = self.parts
        if parts.replay is not None:
            parts.replay.on_truth(
                stamp, p, velocity_world, quaternion, parts.pipeline.imu_bias.current()
            )
        errors = parts.evaluation.on_truth(
            stamp, p, velocity_world, quaternion, controlled=parts.rotor_mode
        )
        parts.health.on_truth(stamp, p, velocity_world, quaternion)
        if errors is not None:
            parts.health.on_tracking_error(stamp, errors["position_m"])
        self.positions.append(p)
        left = parts.health.left_ground_area_s
        if self.left_ground_s is None and left is not None:
            self.left_ground_s = left
            print(
                f"Warning: vehicle left the {2 * parts.ground_half_m:.0f} m ground "
                f"area at {left:.1f} s; nothing below it collides"
            )
        self._files.telemetry.write(
            json.dumps(
                dict(time=t, position=p.tolist(), quaternion=quaternion.tolist())
            )
            + "\n"
        )

    def _velocity_command(self, p: NDArray, quaternion: NDArray) -> None:
        """Native velocity mode: body-velocity command towards ``--target``."""
        from gz.msgs10.twist_pb2 import Twist

        twist = Twist()
        velocity = waypoint_velocity(p, quaternion, self.parts.args.target)
        twist.linear.x, twist.linear.y, twist.linear.z = map(float, velocity)
        _started(self.parts.publisher, "velocity publisher").publish(twist)

    def _heartbeat(self) -> None:
        """Lockstep stand-in for the thrust worker: republish the current
        command (battery and ESC models see it) and expire stale ones, once
        per odometry period in simulation time instead of wall time."""
        if self.parts.lockstep and self.thrust_control is not None:
            self.thrust_control.tick()

    def _advance_world(self, stamp: int) -> None:
        if self.stepper is not None:
            self.stepper.on_odometry(stamp)

    # Other kinds ------------------------------------------------------------

    def _on_poses(self, message: Any) -> None:
        payload, moving = self.parts.payload, self.parts.moving
        for pose in message.pose:
            if payload is not None:
                payload.on_pose(pose.name, vector(pose.position))
            if moving is None:
                continue
            q = pose.orientation
            moving.on_pose(
                pose.name, vector(pose.position), np.array([q.x, q.y, q.z, q.w])
            )

    def _on_contacts(self, message: Any, t: float) -> None:
        for contact in message.contact:
            a, b = contact.collision1.name, contact.collision2.name
            self.parts.health.on_contact(timestamp_ns(message), a, b)
            self.collisions.append(dict(time=t, a=a, b=b))
