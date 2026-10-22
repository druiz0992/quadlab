"""One rotor-thrust control tick per odometry sample of a Gazebo mission.

At each true-state sample in rotor-thrust (or keyboard) mode,
``RotorMission.tick``:

1. checks the odometry interval (and, in real time, queue staleness),
2. advances the powertrain and samples the battery monitor,
3. picks the control state: truth, or the estimate when
   ``algorithms.control_state_source`` is ``estimate`` (see algorithms.py),
4. holds the motors off while the IMU's start-up gyro calibration runs,
5. runs the low-battery failsafe on measured battery values,
6. builds the pilot request (failsafe > mission/take-off > keyboard or a
   programmatic command source) and runs the control loop,
7. records commands and sends control-surface commands.

Truth enters only as the true state handed to the controller when it flies
on truth, the powertrain physics, and evaluation.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import json
import time
from typing import Any, TextIO

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..control.manual import FlightCommand
from ..sensors.bus import SensorBus
from ..simulation.powertrain import Powertrain
from ..simulation.terrain import Terrain
from .algorithms import AlgorithmRunner, EstimateStateSource, StaleEstimate
from .failsafe import BatteryFailsafe
from .mission import MissionRunner
from .rotor_loop import RotorControlLoop, TruthState
from .sensor_pipeline import SensorPipeline, TruthSnapshot
from .takeoff import TakeoffSequence
from .vehicle_forces import ExternalForces

# Longest accepted odometry interval in rotor-thrust mode (s).
MAX_ODOMETRY_INTERVAL_S = 0.1
# Oldest odometry the real-time loop may still act on (wall s).
MAX_QUEUE_AGE_WALL_S = 0.25


@dataclass(frozen=True)
class RotorMissionParts:
    """Everything one rotor-thrust mission tick needs, built by the mission.

    ``estimate_source`` and ``runner`` are None when the controller flies on
    truth or there is no estimator; ``command_source(elapsed_s)`` is an
    optional programmatic pilot; ``command_log`` receives commands.jsonl.
    """

    settings: dict[str, Any]
    rotor_count: int
    controller: Any
    control_loop: RotorControlLoop
    powertrain: Powertrain
    pipeline: SensorPipeline
    forces: ExternalForces
    recorder: Any
    evaluation: Any
    health: Any
    mission: MissionRunner
    takeoff: TakeoffSequence | None
    failsafe: BatteryFailsafe | None
    estimate_source: EstimateStateSource | None
    runner: AlgorithmRunner | None
    bus: SensorBus | None
    keyboard: Any | None
    command_source: Callable[[float], FlightCommand] | None
    terrain: Terrain | None
    truth: TruthSnapshot
    counts: Counter
    command_log: TextIO
    lockstep: bool


def control_state(
    estimate_source: EstimateStateSource | None,
    runner: AlgorithmRunner | None,
    now_ns: int,
    elapsed_s: float,
    max_age_s: float,
    truth: TruthState,
) -> Any | None:
    """Controller state from truth or the estimator (see experiments/algorithms.py).

    Returns None during start-up before the first estimate; raises
    StaleEstimate if there is still none after ``max_age_s``.
    """
    if estimate_source is None:
        return truth
    if runner is None:
        raise ValueError("estimate-driven control needs the algorithm runner")
    state = estimate_source.state(now_ns, runner.latest_estimate)
    if state is None and elapsed_s > max_age_s:
        raise StaleEstimate("No state estimate available for control")
    return state


class RotorMission:
    """Rotor-thrust control at odometry rate (see module docs)."""

    def __init__(self, parts: RotorMissionParts) -> None:
        self._parts = parts
        self._landing_requested = False
        self.previous_control_time: float | None = None

    def tick(
        self,
        sim_time_ns: int,
        t: float,
        first_time: float,
        received_at: float,
        position_m: NDArray[np.float64],
        velocity_world_mps: NDArray[np.float64],
        quaternion_xyzw: NDArray[np.float64],
        body_rates_rad_s: NDArray[np.float64],
        air_body_mps: NDArray[np.float64],
        airspeed_world_mps: NDArray[np.float64],
        pilot_command: FlightCommand,
    ) -> bool:
        """Run one tick; False when no command was due (start-up).

        Raises RuntimeError on stale or irregular odometry and
        StaleEstimate when estimate-driven control loses its estimate.
        """
        parts = self._parts
        self._check_interval(t, received_at)
        self._advance_powertrain(
            sim_time_ns, position_m, quaternion_xyzw, airspeed_world_mps
        )
        state = control_state(
            parts.estimate_source,
            parts.runner,
            sim_time_ns,
            t - first_time,
            parts.settings["algorithms"]["max_estimate_age_s"],
            TruthState(
                sim_time_ns,
                position_m,
                velocity_world_mps,
                quaternion_xyzw,
                body_rates_rad_s,
                airspeed_mps=float(air_body_mps[0]),
            ),
        )
        driver = parts.pipeline.imu_driver
        if driver is not None and not driver.ready:
            state = None  # startup gyro calibration: motors off
        if state is None:
            # Start-up: no estimate yet, or the IMU is still calibrating.
            return False
        override = self._failsafe_step(
            t - first_time,
            state.position_m,
            state.quaternion_xyzw,
            state.velocity_world_mps,
        )
        pilot = self._pilot(state, first_time, pilot_command, override)
        self._control(state, pilot, position_m)
        return True

    def _check_interval(self, t: float, received_at: float) -> None:
        if (
            not self._parts.lockstep
            and time.monotonic() - received_at > MAX_QUEUE_AGE_WALL_S
        ):
            raise RuntimeError("Stale odometry in rotor-control queue")
        dt = (
            1 / self._parts.settings["control"]["odometry_rate_hz"]
            if self.previous_control_time is None
            else t - self.previous_control_time
        )
        if dt <= 0 or dt > MAX_ODOMETRY_INTERVAL_S:
            raise RuntimeError("Invalid odometry interval in rotor-control mode")
        self.previous_control_time = t

    def _advance_powertrain(
        self,
        sim_time_ns: int,
        position_m: NDArray[np.float64],
        quaternion_xyzw: NDArray[np.float64],
        airspeed_world_mps: NDArray[np.float64],
    ) -> None:
        """Battery, ground effect, and rotor limits; then the battery monitor."""
        parts = self._parts
        powertrain = parts.powertrain
        if not powertrain.enabled:
            return
        powertrain.update(sim_time_ns, position_m, quaternion_xyzw, airspeed_world_mps)
        parts.recorder.log_vector(
            "aero/rotor_thrust_factor",
            powertrain.rotor_thrust_factors(),
            labels=[f"rotor_{i}" for i in range(parts.rotor_count)],
        )
        parts.recorder.log_powertrain(
            powertrain.battery_state(),
            powertrain.ground_effect_factors(),
            powertrain.available_thrust_fraction(),
        )
        parts.pipeline.on_battery(
            sim_time_ns, powertrain.battery_state(), parts.counts, parts.bus
        )

    def _failsafe_step(
        self,
        time_s: float,
        position: NDArray[np.float64],
        quaternion: NDArray[np.float64],
        velocity_world: NDArray[np.float64],
    ) -> FlightCommand | None:
        """Run the failsafe on measured battery values; apply landing/disarm."""
        parts = self._parts
        failsafe, loop = parts.failsafe, parts.control_loop
        if failsafe is None or loop.disarmed:
            return None
        reading = parts.pipeline.last_battery_reading
        if reading is not None:  # battery monitor (measured)
            remaining, voltage = reading.remaining_pct, reading.voltage_v
        else:  # ideal measurement from the battery model
            battery = parts.powertrain.battery_state()
            remaining, voltage = 100.0 * battery.soc, battery.terminal_v
        terrain = parts.terrain
        ground = 0.0 if terrain is None else float(terrain.height(*position[:2]))
        body_height_m = parts.settings["vehicle"]["body_size_m"][2]
        clearance = position[2] - ground - body_height_m / 2
        yaw = Rotation.from_quat(quaternion).as_euler("xyz")[2]
        before = len(failsafe.events)
        command = failsafe.update(
            time_s, remaining, voltage, position, yaw, velocity_world[2], clearance
        )
        for event in failsafe.events[before:]:
            parts.recorder.log_failsafe(event)
        if failsafe.state == "landing" and not self._landing_requested:
            parts.controller.allow_landing()
            self._landing_requested = True
        if failsafe.state == "landed":
            loop.disarm()
        return command

    def _pilot(
        self,
        state: Any,
        first_time: float,
        held: FlightCommand,
        override: FlightCommand | None,
    ) -> Callable[[float], Any]:
        """Request source for the control loop's ticks in this period."""
        parts = self._parts
        # Take-off ends on the controller's altitude setpoint (the vehicle's
        # height before the first update).
        setpoint = parts.controller.setpoint_m
        altitude = state.position_m[2] if setpoint is None else setpoint[2]
        position = state.position_m
        mission, takeoff, source = parts.mission, parts.takeoff, parts.command_source

        def pilot(tick_time_s: float) -> Any:
            if override is not None:
                return override  # failsafe takes precedence
            elapsed = tick_time_s - first_time
            if mission.enabled:
                if takeoff is not None and not takeoff.complete:
                    return takeoff.command(altitude, elapsed, FlightCommand())
                return mission.command(elapsed, position)
            command = held if source is None else source(elapsed)
            if takeoff is None:
                return command
            return takeoff.command(altitude, elapsed, command)

        return pilot

    def _control(
        self, state: Any, pilot: Callable[[float], Any], true_position_m: NDArray
    ) -> None:
        """Run the control loop; record commands; drive servos and the keyboard view."""
        parts = self._parts
        for record in parts.control_loop.on_state(state, pilot):
            parts.command_log.write(json.dumps(record) + "\n")
            parts.truth.total_thrust_n = sum(record["accepted_thrust_n"])
            parts.evaluation.on_command(record)
            parts.health.on_command(record)
        surface_commands = getattr(parts.controller, "surface_commands", None)
        if surface_commands is not None and parts.forces.fixed_wing.enabled:
            parts.forces.servos.set_command(surface_commands())
        if parts.keyboard is not None:
            parts.keyboard.show_position(true_position_m)
