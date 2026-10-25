"""Automated IMU calibration run (``robotlab-imucal``).

The simulated counterpart of an autopilot's accelerometer calibration
(the six-position test) plus a thermal calibration:

1. The configured vehicle is made static (held as on a calibration fixture;
   gravity stays on) and set into the six axis-aligned orientations plus
   ``extra_poses`` seeded tilted ones with the ``set_pose`` service.
2. With the IMU temperature model enabled, the whole pose set is repeated at
   ambient temperatures of 5, 25, and 45 °C (a fridge-to-warm-room sweep),
   and the die temperature the IMU reports is recorded with each sample.
3. Readings pass through the mission's IMU models: imperfections, Allan
   errors, temperature drift, and saturation (motors are off, so vibration
   and scheduled faults are left out).
4. ``ImuCalibration.fit_static`` fits accelerometer scale/misalignment,
   offsets, gyro g-sensitivity, and temperature slopes from the known
   specific force per pose (nominal mount, as an operator knows it).

Outputs: ``imu-calibration.json`` (a config layer with
``sensors.imu.calibration``, the settings fingerprint, and filter priors) and
``calibration-report.json`` (fit residuals, before/after static errors, and
the injected truth for comparison). Stored offsets include this run's
turn-on biases; enable ``startup_gyro_calibration_s`` to refresh the gyro
bias at every mission start.

**Filter priors.** The remaining bias uncertainty a filter should start
with is, per instrument and axis, √(r² + t² + i²): r the calibration's
static residual (evaluation, per axis), t the turn-on bias repeatability
and i the bias instability from the IMU's datasheet (here the configured
error model; the procedure plays the operator, the filter only sees the
result). For the gyro, t is left out when the startup calibration is on,
which removes the turn-on bias at every power-up. Written as
``filter_accel_bias_std_mps2`` and ``filter_gyro_bias_std_rad_s``.
"""

from collections.abc import Sequence
from copy import deepcopy
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from ...estimation.imu_calibration import ImuCalibration
from ...sensors.imu_errors import ImuErrorModel
from ...sensors.imu_imperfections import ImuImperfections, ImuSaturation
from ...sensors.imu_temperature import ImuTemperatureModel
from ...simulation.gazebo.config import (
    DroneConfig,
    load_drone_config,
    sensor_fingerprint,
)
from ...simulation.gazebo.messages import timestamp_ns, vector
from ...simulation.gazebo.world import build_world
from .calibration_session import CalibrationSession

GRAVITY_MPS2 = 9.81
AMBIENTS_C = (5.0, 25.0, 45.0)
AXIS_POSES = (
    Rotation.identity(),  # +z up
    Rotation.from_euler("x", 180, degrees=True),  # upside down
    Rotation.from_euler("y", 90, degrees=True),  # nose down
    Rotation.from_euler("y", -90, degrees=True),  # nose up
    Rotation.from_euler("x", 90, degrees=True),  # left side down
    Rotation.from_euler("x", -90, degrees=True),  # right side down
)


def imu_calibration_config(paths: Sequence[str] | None) -> DroneConfig:
    """The vehicle's configuration, prepared for a calibration run."""
    config = load_drone_config(paths)
    data = deepcopy(config.data)
    sensors = data["sensors"]
    sensors["imu"]["calibration"]["enabled"] = False  # calibrate raw readings
    for name in ("camera", "lidar", "rangefinder"):
        sensors[name]["enabled"] = False
    data["simulation"]["lockstep"] = False
    data["simulation"]["start_on_ground"] = False
    data["wind"]["enabled"] = False
    return DroneConfig(data, config.properties, config.sources, config.provenance)


def write_static_world(config: DroneConfig, output: Path) -> Path:
    """World with the drone static (a fixture holds it; gravity stays on)."""
    path = build_world(
        {"obstacles": []},
        output / "world.sdf",
        control_mode="rotor-thrust",
        config=config,
    )
    tree = ET.parse(path)
    drone = tree.getroot().find(".//model[@name='drone']")
    if drone is None:
        raise ValueError(f"{path} has no drone model")
    static = ET.Element("static")
    static.text = "true"
    drone.insert(0, static)
    tree.write(path, encoding="unicode", xml_declaration=True)
    return path


class _ImuPipeline:
    """The mission's static-relevant IMU models (see module docs)."""

    def __init__(self, sensors: dict, ambient_c: float) -> None:
        imu = sensors["imu"]
        self.imperfections = ImuImperfections(imu["imperfections"])
        self.errors = ImuErrorModel(imu["error_model"], rate_hz=imu["rate_hz"])
        self.temperature = ImuTemperatureModel(imu["temperature"], ambient_c)
        self.saturation = ImuSaturation(imu["imperfections"])

    def read(self, message) -> tuple[np.ndarray, np.ndarray, float | None]:
        stamp = timestamp_ns(message)
        force, rate = vector(message.linear_acceleration), vector(
            message.angular_velocity
        )
        for model in (
            self.imperfections,
            self.errors,
            self.temperature,
            self.saturation,
        ):
            force, rate = model.apply(force, rate, stamp)
        temperature = (
            self.temperature.temperature_c if self.temperature.enabled else None
        )
        return np.asarray(force), np.asarray(rate), temperature


def run_imu_calibration(
    paths: Sequence[str] | None,
    output: str | Path,
    *,
    extra_poses: int = 12,
    samples: int = 40,
    startup_gyro_s: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the static calibration; write the layer and the report."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    config = imu_calibration_config(paths)
    sensors = config.data["sensors"]
    pipeline = _ImuPipeline(
        sensors, config.data["environment"]["ambient_temperature_c"]
    )
    ambients = AMBIENTS_C if pipeline.temperature.enabled else (None,)
    poses = list(AXIS_POSES) + list(Rotation.random(extra_poses, random_state=seed))
    mount = Rotation.from_euler(
        "xyz", sensors["imu"]["mount"]["rotation_deg"], degrees=True
    )
    from gz.msgs10.imu_pb2 import IMU

    session = CalibrationSession(
        write_static_world(config, output), output / "gazebo.log", "/robotlab/imu", IMU
    )
    try:
        data = _collect(session, pipeline, poses, ambients, mount, samples)
    finally:
        session.close()
    calibration = ImuCalibration.fit_static(
        data["accel"], data["gyro"], data["force"], data["temperature"]
    ).with_startup(startup_gyro_s)
    evaluation = _evaluate(data, calibration)
    section = calibration.to_config()
    section["fingerprint"] = sensor_fingerprint(sensors, "imu")
    section.update(_filter_priors(sensors["imu"], evaluation["after"], startup_gyro_s))
    (output / "imu-calibration.json").write_text(
        json.dumps({"sensors": {"imu": {"calibration": section}}}, indent=2) + "\n"
    )
    report = dict(
        fit=calibration.fit,
        poses=len(poses),
        ambient_temperatures_c=[a for a in ambients if a is not None],
        evaluation=evaluation,
        filter_priors={
            k: section[k]
            for k in ("filter_accel_bias_std_mps2", "filter_gyro_bias_std_rad_s")
        },
        injected_truth=_truth(sensors, pipeline),
    )
    (output / "calibration-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _collect(
    session, pipeline, poses, ambients, mount, samples
) -> dict[str, np.ndarray]:
    """Mean readings per pose and ambient, with the known specific force."""
    rows: dict[str, list[Any]] = dict(accel=[], gyro=[], force=[], temperature=[])
    for ambient in ambients:
        if ambient is not None:
            pipeline.temperature.set_ambient(ambient)
        for pose in poses:
            session.set_attitude(pose)
            readings = [pipeline.read(m) for m in session.fresh(samples)]
            rows["accel"].append(np.mean([r[0] for r in readings], axis=0))
            rows["gyro"].append(np.mean([r[1] for r in readings], axis=0))
            temps = [r[2] for r in readings if r[2] is not None]
            rows["temperature"].append(float(np.mean(temps)) if temps else 25.0)
            # Known specific force in sensor axes (fixture orientation, nominal mount).
            rows["force"].append((pose * mount).inv().apply([0.0, 0.0, GRAVITY_MPS2]))
    return {k: np.asarray(v, float) for k, v in rows.items()}


def _filter_priors(
    imu: dict, residual: dict, startup_gyro_s: float
) -> dict[str, float]:
    """Initial bias uncertainty after calibration (see module docs)."""
    errors, imperfections = imu["error_model"], imu["imperfections"]

    def datasheet(key: str, section: dict) -> float:
        return float(np.max(section[key])) if section["enabled"] else 0.0

    accel = np.sqrt(
        (residual["accel_error_rms_mps2"] / np.sqrt(3)) ** 2
        + datasheet("accel_turn_on_bias_std_mps2", errors) ** 2
        + datasheet("accel_bias_instability_mps2", imperfections) ** 2
    )
    turn_on = (
        0.0 if startup_gyro_s > 0 else datasheet("gyro_turn_on_bias_std_rad_s", errors)
    )
    gyro = np.sqrt(
        (residual["gyro_error_rms_rad_s"] / np.sqrt(3)) ** 2
        + turn_on**2
        + datasheet("gyro_bias_instability_rad_s", imperfections) ** 2
    )
    return dict(
        filter_accel_bias_std_mps2=float(max(accel, 1e-4)),
        filter_gyro_bias_std_rad_s=float(max(gyro, 1e-5)),
    )


def _evaluate(data, calibration) -> dict[str, Any]:
    """Static errors against the known force (and zero rate), before/after."""
    corrected = [
        calibration.apply(a, w, t)
        for a, w, t in zip(data["accel"], data["gyro"], data["temperature"])
    ]
    accel_after = np.array([c[0] for c in corrected]) - data["force"]
    gyro_after = np.array([c[1] for c in corrected])
    rms = lambda x: float(np.sqrt(np.mean(np.sum(x**2, axis=1))))  # noqa: E731
    return dict(
        before=dict(
            accel_error_rms_mps2=rms(data["accel"] - data["force"]),
            gyro_error_rms_rad_s=rms(data["gyro"]),
        ),
        after=dict(
            accel_error_rms_mps2=rms(accel_after), gyro_error_rms_rad_s=rms(gyro_after)
        ),
    )


def _truth(sensors, pipeline) -> dict[str, Any]:
    """Injected parameters, for comparison only (never used by the fit)."""
    imperfections = sensors["imu"]["imperfections"]
    scale = np.eye(3)
    if imperfections["enabled"]:
        scale = (
            scale
            + np.diag(imperfections["accel_scale_factor"])
            + np.asarray(imperfections["accel_cross_coupling"])
        )
    accel_bias, gyro_bias = pipeline.errors.current_biases()
    return dict(
        accel_scale_and_misalignment=scale.tolist(),
        accel_temperature_mps2_per_c=sensors["imu"]["temperature"]["accel_bias_per_c"],
        gyro_temperature_rad_s_per_c=sensors["imu"]["temperature"]["gyro_bias_per_c"],
        gyro_g_sensitivity=imperfections["gyro_g_sensitivity_rad_s_per_mps2"],
        accel_bias_at_end_mps2=(
            np.asarray(sensors["imu"]["acceleration_mps2"]["bias"]) + accel_bias
        ).tolist(),
        gyro_bias_at_end_rad_s=(
            np.asarray(sensors["imu"]["angular_rate_rad_s"]["bias"]) + gyro_bias
        ).tolist(),
        temperature_calibration_c=sensors["imu"]["temperature"]["calibration_c"],
    )
