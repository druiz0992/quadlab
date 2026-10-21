"""Automated magnetometer calibration run (``dronelab-magcal``).

The simulated counterpart of a ground-station compass calibration:

1. **Orientation dance.** Gazebo runs the configured vehicle with gravity
   off; the drone is turned through ``orientations`` seeded random
   orientations via the UserCommands ``set_pose`` service (standing in for
   hand rotation). At each one, ``samples`` fresh magnetometer readings are
   taken once odometry confirms the pose. Motors stay off.
2. **Throttle sweep** ("compassmot"). Held level, the total thrust command
   is stepped from 0 to the lift rotors' maximum. Interference and battery
   current come from the same Python models a mission uses (the motors are
   not spun in Gazebo, which only supplies the Earth field). The regressor
   is the battery monitor's measured current when the vehicle has a battery
   and a battery monitor, else the commanded thrust.

Raw readings pass through the mission's magnetometer error model (hard and
soft iron, interference, drift, noise). Every fifth orientation is held out
of the ellipsoid fit and used to evaluate the result against the true field
(Gazebo's field before the error model, evaluation only): heading and
magnitude errors before and after calibration.

Outputs in the output directory: ``magnetometer-calibration.json`` (a config
layer with ``sensors.magnetometer.calibration``, to pass with ``--drone``)
and ``calibration-report.json`` (fit quality, evaluation, and the injected
truth for comparison).
"""

from collections.abc import Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import time
from typing import Any
import uuid
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from ..estimation.magnetometer_calibration import (
    MagnetometerCalibration,
    fit_ellipsoid,
    fit_interference,
    heading_error_deg,
)
from ..sensors.battery_monitor import BatteryMonitor
from ..sensors.magnetometer import MagnetometerErrorModel
from ..simulation.gazebo.config import DroneConfig, load_drone_config
from ..simulation.gazebo.environment import gazebo_environment
from ..simulation.gazebo.messages import timestamp_ns, vector
from ..simulation.gazebo.world import build_world
from ..simulation.powertrain import Powertrain

HOLD_POSITION_M = (0.0, 0.0, 10.0)
SET_POSE_SERVICE = "/world/dronelab/set_pose"


def calibration_config(paths: Sequence[str] | None) -> DroneConfig:
    """The vehicle's configuration, prepared for a calibration run."""
    config = load_drone_config(paths)
    data = deepcopy(config.data)
    sensors = data["sensors"]
    sensors["magnetometer"]["enabled"] = True
    sensors["magnetometer"]["calibration"]["enabled"] = False  # calibrate raw
    for name in ("camera", "lidar", "rangefinder"):
        sensors[name]["enabled"] = False  # no rendering needed
    data["simulation"]["lockstep"] = False
    data["wind"]["enabled"] = False
    return DroneConfig(data, config.properties, config.sources, config.provenance)


def write_world(config: DroneConfig, output: Path) -> Path:
    """World without gravity, so the drone stays where set_pose puts it."""
    path = build_world(
        {"obstacles": []},
        output / "world.sdf",
        control_mode="rotor-thrust",
        config=config,
    )
    tree = ET.parse(path)
    world = tree.getroot().find("world")
    gravity = ET.Element("gravity")
    gravity.text = "0 0 0"
    world.insert(0, gravity)
    tree.write(path, encoding="unicode", xml_declaration=True)
    return path


class _Session:
    """Gazebo process with magnetometer and odometry subscriptions."""

    def __init__(self, world: Path, log: Path) -> None:
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.magnetometer_pb2 import Magnetometer
        from gz.msgs10.odometry_pb2 import Odometry
        from gz.msgs10.pose_pb2 import Pose
        from gz.transport13 import Node

        os.environ["GZ_PARTITION"] = "dronelab_magcal_" + uuid.uuid4().hex
        os.environ.setdefault("GZ_IP", "127.0.0.1")
        self._pose_type, self._boolean = Pose, Boolean
        self.node = Node()
        self.fields: Queue = Queue()
        self.attitude = None
        self.node.subscribe(
            Magnetometer, "/dronelab/magnetometer", lambda m: self.fields.put(m)
        )
        self.node.subscribe(Odometry, "/dronelab/odometry", self._on_odometry)
        self._log = log.open("w")
        self.process = subprocess.Popen(
            ["gz", "sim", "-s", "-r", "-v", "2", str(world)],
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=gazebo_environment(),
        )

    def _on_odometry(self, message) -> None:
        q = message.pose.orientation
        self.attitude = np.array([q.x, q.y, q.z, q.w])

    def set_attitude(self, rotation: Rotation, timeout_s: float = 10.0) -> None:
        """Teleport the drone and wait until odometry reports the attitude."""
        request = self._pose_type()
        request.name = "drone"
        request.position.x, request.position.y, request.position.z = HOLD_POSITION_M
        x, y, z, w = rotation.as_quat()
        request.orientation.x, request.orientation.y = x, y
        request.orientation.z, request.orientation.w = z, w
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            # Replies from Python are unreliable; odometry is the evidence.
            self.node.request(
                SET_POSE_SERVICE, request, self._pose_type, self._boolean, 100
            )
            time.sleep(0.05)
            if (
                self.attitude is not None
                and abs(np.dot(self.attitude, rotation.as_quat())) > 0.99999
            ):
                return
        raise RuntimeError("Gazebo did not apply the calibration pose; see gazebo log")

    def fresh_fields(self, count: int, timeout_s: float = 10.0) -> list:
        """``count`` magnetometer messages published after this call."""
        while not self.fields.empty():
            self.fields.get_nowait()
        self.fields.get(timeout=timeout_s)  # may predate the pose: discard
        messages = []
        while len(messages) < count:
            try:
                messages.append(self.fields.get(timeout=timeout_s))
            except Empty:
                raise RuntimeError("No magnetometer data; see gazebo log") from None
        return messages

    def close(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self._log.close()


class _Instruments:
    """Mission models that turn Gazebo's field into what the autopilot sees."""

    def __init__(self, config: DroneConfig) -> None:
        data = config.data
        self.errors = MagnetometerErrorModel(
            data["sensors"]["magnetometer"]["error_model"]
        )
        self.powertrain = Powertrain(
            config.properties, data["battery"], data["aerodynamics"]["ground_effect"]
        )
        self.monitor = BatteryMonitor(
            data["sensors"]["battery_monitor"], data["battery"]["capacity_mah"]
        )
        self.uses_current = bool(data["battery"]["enabled"] and self.monitor.enabled)
        self._properties = config.properties
        self._measured_current = 0.0

    def read(self, message, thrust_n: float) -> tuple[np.ndarray, np.ndarray, float]:
        """(raw µT, true field µT, regressor) for one magnetometer message."""
        stamp = timestamp_ns(message)
        lift = list(self._properties.lift_rotors)
        thrusts = np.zeros(len(self._properties.rotors))
        thrusts[lift] = thrust_n / len(lift)
        current = 0.0
        if self.powertrain.has_battery:
            self.powertrain.motor_speeds(thrusts)
            self.powertrain.update(stamp, HOLD_POSITION_M, [0, 0, 0, 1])
            state = self.powertrain.battery_state()
            current = state.current_a
            reading = self.monitor.sample(stamp, state)
            if reading is not None:
                self._measured_current = reading.current_a
        true = np.asarray(vector(message.field_tesla))
        raw = self.errors.apply(true, thrust_n, current, stamp)
        regressor = self._measured_current if self.uses_current else thrust_n
        return raw * 1e6, true * 1e6, regressor


def run_calibration(
    paths: Sequence[str] | None,
    output: str | Path,
    *,
    orientations: int = 120,
    samples: int = 3,
    sweep_steps: int = 11,
    seed: int = 0,
) -> dict[str, Any]:
    """Run both calibration steps; write the layer and the report."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    config = calibration_config(paths)
    settings = config.data["sensors"]["magnetometer"]
    field_strength = settings["calibration"]["field_strength_ut"]
    world = write_world(config, output)
    instruments = _Instruments(config)
    rotations = Rotation.random(orientations, random_state=seed)
    mount = Rotation.from_euler("xyz", settings["mount"]["rotation_deg"], degrees=True)
    session = _Session(world, output / "gazebo.log")
    try:
        dance = _dance(session, instruments, rotations, samples)
        sweep = _sweep(session, instruments, config, sweep_steps)
    finally:
        session.close()
    calibration, quality = _fit(
        dance, sweep, settings, field_strength, instruments.uses_current
    )
    report = dict(
        quality,
        evaluation=_evaluate(dance, calibration, rotations, mount),
        regressor=(
            "measured current (A)"
            if instruments.uses_current
            else "commanded thrust (N)"
        ),
        injected_truth=settings["error_model"],
    )
    calibration = MagnetometerCalibration(
        **dict(
            calibration.__dict__,
            heading_std_deg=max(
                1.0, round(report["evaluation"]["after"]["heading_rms_deg"], 2)
            ),
        )
    )
    layer = {"sensors": {"magnetometer": {"calibration": calibration.to_config()}}}
    (output / "magnetometer-calibration.json").write_text(
        json.dumps(layer, indent=2) + "\n"
    )
    (output / "calibration-report.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )
    return report


def _dance(session, instruments, rotations, samples) -> dict[str, list]:
    """Readings at each orientation (motors off)."""
    raw, true, regressor, index = [], [], [], []
    for i, rotation in enumerate(rotations):
        session.set_attitude(rotation)
        for message in session.fresh_fields(samples):
            r, t, u = instruments.read(message, 0.0)
            raw.append(r), true.append(t), regressor.append(u), index.append(i)
    return dict(
        raw=np.array(raw),
        true=np.array(true),
        regressor=np.array(regressor),
        index=np.array(index),
    )


def _sweep(session, instruments, config, steps) -> dict[str, np.ndarray]:
    """Readings held level while the total thrust command steps up."""
    session.set_attitude(Rotation.identity())
    lift = len(config.properties.lift_rotors)
    raw, regressor = [], []
    for thrust in np.linspace(0.0, lift * config.properties.max_rotor_thrust, steps):
        for message in session.fresh_fields(10):
            r, _, u = instruments.read(message, float(thrust))
            raw.append(r), regressor.append(u)
    return dict(raw=np.array(raw), regressor=np.array(regressor))


def _fit(dance, sweep, settings, field_strength, uses_current):
    """Ellipsoid on the fit orientations, interference on the sweep."""
    fit_mask = dance["index"] % 5 != 0  # every fifth orientation held out
    slope, _ = fit_interference(sweep["raw"], sweep["regressor"])
    # Remove the interference present during the dance before the fit.
    dance_raw = dance["raw"] - np.outer(dance["regressor"], slope)
    ellipsoid = fit_ellipsoid(dance_raw[fit_mask], field_strength)
    zero = np.zeros(3)
    calibration = MagnetometerCalibration(
        ellipsoid.offset_ut,
        ellipsoid.correction,
        slope if uses_current else zero,
        zero if uses_current else slope,
        field_strength,
        settings["calibration"]["inclination_deg"],
        settings["calibration"]["declination_deg"],
        settings["calibration"]["field_tolerance_ut"],
        settings["calibration"]["heading_std_deg"],
    )
    quality = dict(
        fit_samples=int(fit_mask.sum()),
        ellipsoid_residual_rms_ut=ellipsoid.residual_rms_ut,
        ellipsoid_condition=ellipsoid.condition,
        interference_slope=slope.tolist(),
    )
    return calibration, quality


def _evaluate(dance, calibration, rotations, mount) -> dict[str, Any]:
    """Held-out heading and magnitude errors, raw versus calibrated."""
    held = dance["index"] % 5 == 0
    result = {}
    for name, fields in (
        ("before", dance["raw"][held]),
        (
            "after",
            np.array(
                [
                    calibration.apply(r, *_regressors(calibration, u))
                    for r, u in zip(dance["raw"][held], dance["regressor"][held])
                ]
            ),
        ),
    ):
        headings, magnitudes = [], []
        for field, true, i in zip(fields, dance["true"][held], dance["index"][held]):
            to_world = rotations[i] * mount
            headings.append(heading_error_deg(field, true, to_world))
            magnitudes.append(np.linalg.norm(field) - np.linalg.norm(true))
        h, m = np.array(headings), np.array(magnitudes)
        result[name] = dict(
            heading_rms_deg=float(np.sqrt(np.mean(h**2))),
            heading_max_deg=float(np.abs(h).max()),
            magnitude_rms_ut=float(np.sqrt(np.mean(m**2))),
        )
    result["held_out_samples"] = int(held.sum())
    return result


def _regressors(calibration, value):
    """(current, thrust) arguments for apply() from the dance regressor."""
    if np.any(calibration.current_compensation_ut_per_a):
        return value, 0.0
    return 0.0, value
