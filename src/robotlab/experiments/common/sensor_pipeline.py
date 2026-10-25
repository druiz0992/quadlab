"""Measurement pipeline of a Gazebo mission: error models to recording and bus.

Every transported Gazebo measurement passes through ``SensorPipeline``:

1. ``apply_errors``: sensor-error and fault models, before recording and
   any estimator. IMU: imperfections, Allan errors, temperature,
   vibration/clipping, saturation, then digital faults (a lost sample
   returns None). GPS: errors, RTK, multipath, and velocity errors see every
   fix, before dropout, so dropouts do not shift their random sequences.
   Magnetometer: structure anomalies, then the vehicle's own errors.
   Barometer: datum, downwash, and airspeed errors.
2. ``deliver``: GPS dropout and heading, camera/lidar effects, ESC
   telemetry, recording, sensor status, and submission to the sensor bus.

Synthetic sensors computed from truth (optical flow, pitot, battery
monitor) enter through ``on_truth`` and ``on_battery``. Truth is used only
to synthesize measurements (``TruthSnapshot``), never passed on to
algorithms. The order of model calls is part of the contract: seeded
models draw random numbers per call.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ...control.timing import DelayLine
from ...core.measurements import CameraFrame, LidarPoints, BatteryReading
from ...sensors.imu.calibration import ImuCalibration, ImuCalibrationDriver
from ...sensors.magnetometer.calibration import MagnetometerCalibration
from ...evaluation.imu_bias import EffectiveImuBias
from ...sensors.airspeed import AirspeedSensor
from ...sensors.barometer import BaroErrorModel
from ...sensors.battery_monitor import BatteryMonitor
from ...sensors.bus import STREAMS, SensorBus
from ...sensors.imaging.camera_effects import CameraEffects
from ...sensors.gnss.dropout import GpsDropout
from ...sensors.esc_telemetry import EscTelemetryModel
from ...sensors.gazebo import (
    baro_datum_message,
    baro_errors_message,
    bus_payload,
    gps_errors_message,
    gps_multipath_message,
    gps_rtk_message,
    gps_velocity_message,
    imu_errors_message,
    magnetometer_anomaly_message,
    magnetometer_errors_message,
    rgb_image,
)
from ...sensors.gnss.errors import GpsErrorModel
from ...sensors.gnss.receiver import (
    GpsHeading,
    GpsStatusModel,
    configured_position_sigma_m,
    GpsVelocityErrors,
    RtkErrorModel,
)
from ...sensors.imu import ImuPerturbation
from ...sensors.imu.errors import ImuErrorModel
from ...sensors.imu.faults import ImuFaults
from ...sensors.imu.imperfections import ImuImperfections, ImuSaturation, ImuTiming
from ...sensors.imu.temperature import ImuTemperatureModel
from ...sensors.imaging.lidar_effects import LidarEffects
from ...sensors.magnetometer.anomalies import MagneticAnomalies, scene_dipoles
from ...sensors.magnetometer.errors import MagnetometerErrorModel
from ...sensors.gnss.multipath import GpsMultipath
from ...sensors.optical_flow import OpticalFlowModel
from ...sensors.imu.vibration import AccelerometerVibration
from ...simulation.aerodynamics import hover_induced_velocity
from ...simulation.battery import BatteryState
from ...simulation.gazebo.config import sensor_fingerprint
from ...simulation.gazebo.messages import timestamp_ns, vector, with_timestamp_ns
from ...simulation.gazebo.vehicle import ROTOR_VELOCITY_SLOWDOWN
from ...simulation.powertrain import Powertrain
from ...simulation.terrain import Terrain


@dataclass
class TruthSnapshot:
    """Latest simulator truth, for measurement synthesis and fault models only.

    Updated by the mission's odometry step: ``position_m`` first (GPS zones,
    anomalies), the motion after the start-up velocity correction (camera
    and lidar motion effects, GPS heading). ``total_thrust_n`` is the latest
    commanded total thrust (motor interference, downwash), not truth.
    ``sim_time_s`` is the latest odometry time (s), the clock for models
    stamped by the mission (ESC ticks, keyboard effects).
    Mutable by design: one per run, owned by the mission loop.
    """

    sim_time_s: float | None = None

    position_m: NDArray[np.float64] | None = None
    velocity_world_mps: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    attitude: Rotation = field(default_factory=Rotation.identity)
    body_rates_rad_s: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    total_thrust_n: float = 0.0


def calibration_warnings(sensors: dict[str, Any]) -> list[str]:
    """Calibrations made for different sensor settings (stale)."""
    warnings = []
    for kind in ("imu", "magnetometer"):
        calibration = sensors[kind]["calibration"]
        stored = calibration["fingerprint"]
        if (
            calibration["enabled"]
            and stored
            and stored != sensor_fingerprint(sensors, kind)
        ):
            warnings.append(
                f"{kind} calibration was made for different {kind} settings; "
                "recalibrate or check the layers"
            )
    return warnings


def rotor_velocities(message: Any) -> list[float]:
    """Rotor joint velocities (rad/s, joint frame) in rotor index order."""
    by_name = {joint.name: joint.axis1.velocity for joint in message.joint}
    return [by_name[f"rotor_{i}_joint"] for i in range(len(by_name))]


def _mount_rotation(mount: dict[str, Any]) -> Rotation:
    return Rotation.from_euler("xyz", mount["rotation_deg"], degrees=True)


class SensorPipeline:
    """All measurement models of one run (see module docs).

    ``recorder`` logs to Rerun (or a fake); ``powertrain`` supplies battery
    current for magnetometer interference; ``current_wind()`` is the wind
    the vehicle sees (m/s, ENU); ``elevation_m`` is the world's sea-level
    elevation for the barometer datum; ``nominal_imu_mount`` and
    ``true_imu_mount`` rotations feed the effective-bias evaluation.
    """

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        scene: dict[str, Any],
        terrain: Terrain | None,
        elevation_m: float,
        propeller_radius_m: float,
        recorder: Any,
        powertrain: Powertrain,
        current_wind: Callable[[], NDArray[np.float64]],
        truth: TruthSnapshot,
        true_imu_mount: Rotation,
    ) -> None:
        sensors = settings["sensors"]
        self._sensors = sensors
        self._recorder = recorder
        self._powertrain = powertrain
        self._current_wind = current_wind
        self._truth = truth
        self._elevation_m = elevation_m
        self._propeller_radius_m = propeller_radius_m
        self._air_density = settings["aerodynamics"]["air_density_kg_m3"]
        self._build_imu(sensors, settings["environment"]["ambient_temperature_c"])
        self._build_gps(sensors["gps"])
        self.baro_errors = BaroErrorModel(sensors["baro"]["error_model"])
        self._build_magnetometer(sensors["magnetometer"], scene)
        self.optical_flow = OpticalFlowModel(sensors["optical_flow"], terrain)
        self.battery_monitor = BatteryMonitor(
            sensors["battery_monitor"], settings["battery"]["capacity_mah"]
        )
        self.airspeed = AirspeedSensor(sensors["airspeed"], self._air_density)
        self.esc_telemetry = EscTelemetryModel(
            settings["motors"]["esc"]["telemetry"], ROTOR_VELOCITY_SLOWDOWN
        )
        self._build_imaging(sensors, settings["environment"]["weather"])
        # Effective bias of the delivered IMU readings (evaluation only): true
        # mount (randomization may misalign it) versus the nominal one.
        self.imu_bias = EffectiveImuBias(
            _mount_rotation(sensors["imu"]["mount"]), true_imu_mount
        )
        self.calibration_warnings = calibration_warnings(sensors)
        self.calibrated_imu: tuple | None = None
        self.calibrated_mag: NDArray[np.float64] | None = None
        self.last_battery_reading: BatteryReading | None = None

    def _build_imu(self, sensors: dict[str, Any], ambient_c: float) -> None:
        imu = sensors["imu"]
        self.vibration = AccelerometerVibration(imu["vibration"])
        self.imu_errors = ImuErrorModel(imu["error_model"], rate_hz=imu["rate_hz"])
        self.imu_faults = ImuFaults(imu["faults"])
        self.imu_imperfections = ImuImperfections(imu["imperfections"])
        self.imu_saturation = ImuSaturation(imu["imperfections"])
        self.imu_timing = ImuTiming(imu["imperfections"], imu["rate_hz"])
        self.imu_temperature = ImuTemperatureModel(imu["temperature"], ambient_c)
        # Physical order: device imperfections, noise and bias, temperature
        # drift, mechanical vibration, range limits, then injected faults.
        self.imu_chain: tuple[ImuPerturbation, ...] = (
            self.imu_imperfections,
            self.imu_errors,
            self.imu_temperature,
            self.vibration,
            self.imu_saturation,
            self.imu_faults,
        )
        calibration = imu["calibration"]
        self.imu_driver = (
            ImuCalibrationDriver(ImuCalibration.from_config(calibration))
            if calibration["enabled"]
            else None
        )

    def _build_gps(self, gps: dict[str, Any]) -> None:
        self.gps_dropout = GpsDropout(**gps["dropout"])
        self.gps_errors = GpsErrorModel(gps["error_model"])
        self.gps_multipath = GpsMultipath(gps["multipath"])
        self.gps_velocity = GpsVelocityErrors(gps["velocity_error"])
        self.gps_rtk = RtkErrorModel(gps["rtk"])
        self.gps_status = GpsStatusModel(
            gps["receiver"], configured_position_sigma_m(gps)
        )
        self.gps_heading = GpsHeading(gps["heading"])

    def _build_magnetometer(self, magnetometer: dict[str, Any], scene: dict) -> None:
        self.magnetometer_errors = MagnetometerErrorModel(magnetometer["error_model"])
        calibration = magnetometer["calibration"]
        self.mag_calibration = (
            MagnetometerCalibration.from_config(calibration)
            if calibration["enabled"]
            else None
        )
        self.anomalies = MagneticAnomalies(
            list(magnetometer["dipoles"]) + scene_dipoles(scene)
        )

    def _build_imaging(self, sensors: dict[str, Any], weather: dict) -> None:
        camera, lidar = sensors["camera"], sensors["lidar"]
        self.camera_effects = CameraEffects(camera)
        self.camera_delay: DelayLine[Any] = DelayLine(round(camera["latency_s"] * 1e9))
        self._camera_mount = _mount_rotation(camera["mount"])
        self.lidar_effects = LidarEffects(lidar, weather)
        self._lidar_offset = np.asarray(lidar["mount"]["position_m"], float)
        self._lidar_mount = _mount_rotation(lidar["mount"])

    def bus_kinds(self) -> tuple[str, ...]:
        """Streams the sensor bus carries for this configuration."""
        sensors = self._sensors
        return tuple(
            kind
            for kind in STREAMS
            if (kind in sensors and sensors[kind]["enabled"])
            or (kind == "esc" and self.esc_telemetry.enabled)
            or (kind == "gps_heading" and self.gps_heading.enabled)
        )

    # 1. Error and fault models -------------------------------------------

    def apply_errors(self, kind: str, message: Any) -> Any | None:
        """Measured message after error/fault models; None if the sample is lost."""
        if kind == "imu":
            return self._imu_errors(message)
        if kind == "magnetometer":
            return self._magnetometer_errors(message)
        if kind == "baro":
            return self._baro_errors(message)
        if kind == "gps":
            return self._gps_errors(message)
        return message

    def _imu_errors(self, message: Any) -> Any | None:
        stamp = timestamp_ns(message)
        if self.imu_timing.drop(stamp):
            return None  # sample lost: never recorded or delivered
        ideal = message  # error-free reading, evaluation only
        for model in self.imu_chain:
            message = imu_errors_message(message, model, stamp)
        if self.imu_timing.enabled:
            message = with_timestamp_ns(message, self.imu_timing.stamp(stamp))
        self.calibrated_imu = self._calibrate_imu(message)
        delivered = (
            self.calibrated_imu[:2]
            if self.calibrated_imu is not None
            else (vector(message.linear_acceleration), vector(message.angular_velocity))
        )
        self.imu_bias.update(
            stamp,
            vector(ideal.linear_acceleration),
            vector(ideal.angular_velocity),
            *delivered,
        )
        return message

    def _calibrate_imu(self, message: Any) -> tuple | None:
        """Driver-side IMU correction, or None without a calibration.

        Uses the die temperature the IMU reports (the temperature model's
        value when it is enabled; unknown otherwise).
        """
        if self.imu_driver is None:
            return None
        model = self.imu_temperature
        temperature = model.temperature_c if model.enabled else None
        force, rate = self.imu_driver.apply(
            vector(message.linear_acceleration),
            vector(message.angular_velocity),
            temperature,
            timestamp_ns(message),
        )
        self._recorder.log_vector("imu/calibrated/accel", force)
        self._recorder.log_vector("imu/calibrated/gyro", rate)
        return force, rate, temperature

    def _magnetometer_errors(self, message: Any) -> Any:
        truth = self._truth
        if self.anomalies.enabled and truth.position_m is not None:
            message = magnetometer_anomaly_message(message, self._anomaly_in_sensor())
        powertrain = self._powertrain
        battery = powertrain.battery_state() if powertrain.enabled else None
        message = magnetometer_errors_message(
            message,
            self.magnetometer_errors,
            truth.total_thrust_n,
            battery.current_a if battery is not None else 0.0,
        )
        self.calibrated_mag = self._calibrate_magnetometer(message)
        return message

    def _anomaly_in_sensor(self) -> NDArray[np.float64]:
        """Anomaly field at the magnetometer, in sensor axes (truth synthesis)."""
        mount = self._sensors["magnetometer"]["mount"]
        attitude = self._truth.attitude
        sensor_position = np.asarray(self._truth.position_m) + attitude.apply(
            mount["position_m"]
        )
        world_to_sensor = (attitude * _mount_rotation(mount)).inv()
        return world_to_sensor.apply(self.anomalies.field_world(sensor_position))

    def _calibrate_magnetometer(self, message: Any) -> NDArray[np.float64] | None:
        """Driver-side correction with the configured calibration (µT), or None.

        Uses only what an autopilot knows: the measured battery current
        (battery monitor; 0 without one) and its own total thrust command.
        """
        if self.mag_calibration is None:
            return None
        raw = np.asarray(vector(message.field_tesla)) * 1e6
        reading = self.last_battery_reading
        current = reading.current_a if reading is not None else 0.0
        corrected = self.mag_calibration.apply(raw, current, self._truth.total_thrust_n)
        log = self._recorder.log_vector
        log("magnetometer/calibrated_ut", corrected, ["x", "y", "z"])
        log(
            "magnetometer/calibrated_level_heading_deg",
            [float(np.degrees(np.arctan2(corrected[0], corrected[1])))],
            ["value"],
        )
        return corrected

    def _baro_errors(self, message: Any) -> Any:
        message = baro_datum_message(message, self._elevation_m)
        # Downwash from the latest total commanded thrust (momentum theory)
        # and horizontal airspeed act on the static port.
        horizontal_air = self._truth.velocity_world_mps - self._current_wind()
        return baro_errors_message(
            message,
            self.baro_errors,
            timestamp_ns(message),
            induced_velocity_mps=hover_induced_velocity(
                self._truth.total_thrust_n / 4,
                self._propeller_radius_m,
                self._air_density,
            ),
            airspeed_mps=float(np.hypot(*horizontal_air[:2])),
        )

    def _gps_errors(self, message: Any) -> Any:
        stamp = timestamp_ns(message)
        message = gps_errors_message(message, self.gps_errors, stamp)
        message = gps_rtk_message(
            message, self.gps_rtk, stamp, self.gps_multipath.active
        )
        message = gps_multipath_message(
            message, self.gps_multipath, stamp, self._truth.position_m
        )
        message = gps_velocity_message(message, self.gps_velocity, stamp)
        status = self.gps_status.update(self.gps_multipath.active, self.gps_rtk)
        self._recorder.log_vector(
            "gps/status",
            [status.satellites, status.hdop, status.horizontal_accuracy_m],
            ["satellites", "hdop", "horizontal_accuracy_m"],
        )
        return message

    # 2. Delivery ------------------------------------------------------------

    def deliver(
        self,
        kind: str,
        message: Any,
        t: float,
        last_odometry_s: float | None,
        bus: SensorBus | None,
    ) -> bool:
        """Record and submit one measured message; return whether GPS dropped it.

        ``t`` is the message time (s); ``last_odometry_s`` the latest
        odometry time, which releases delayed camera frames.
        """
        gps_dropped = kind == "gps" and self.gps_dropout.drop(
            timestamp_ns(message), self._truth.position_m
        )
        if kind == "gps" and not gps_dropped and self.gps_heading.enabled:
            self._gps_heading(message, bus)
        self._record(kind, message, t, gps_dropped, bus)
        if self.camera_effects.enabled:
            self._release_camera_frames(t, last_odometry_s, bus)
        if kind == "imu" and self.imu_temperature.enabled:
            self._recorder.log_vector(
                "imu/temperature_c", [self.imu_temperature.temperature_c], ["value"]
            )
        self._log_status(kind, message, t, gps_dropped)
        if bus is not None:
            self._submit(bus, kind, message, gps_dropped)
        return gps_dropped

    def _gps_heading(self, message: Any, bus: SensorBus | None) -> None:
        heading = self.gps_heading.sample(
            float(self._truth.attitude.as_euler("ZYX")[0])
        )
        self._recorder.log_vector("gps/heading_rad", [heading.heading_rad], ["value"])
        if bus is not None:
            bus.submit("gps_heading", heading, timestamp_ns(message))

    def _record(
        self,
        kind: str,
        message: Any,
        t: float,
        gps_dropped: bool,
        bus: SensorBus | None,
    ) -> None:
        """Camera/lidar effects and ESC telemetry, or plain recording."""
        if kind == "camera" and self.camera_effects.enabled:
            # Effects need the camera rotation rate at capture.
            rates = self._camera_mount.inv().apply(self._truth.body_rates_rad_s)
            image = self.camera_effects.process(rgb_image(message), rates)
            self.camera_delay.push(timestamp_ns(message), image)
        elif kind == "lidar" and self.lidar_effects.enabled:
            points = self._lidar_with_effects(message)
            reflectivity = self.lidar_effects.last_reflectivity
            self._recorder.log_lidar_points(points, t, reflectivity)
            if bus is not None:
                bus.submit(
                    "lidar", LidarPoints(points, reflectivity), timestamp_ns(message)
                )
        elif kind == "esc":
            self._esc_telemetry(message, bus)
        elif not gps_dropped:
            self._recorder.log(kind, message, t)

    def _lidar_with_effects(self, message: Any) -> NDArray[np.float64]:
        """Scan with return physics, noise, dropouts, and motion distortion (body axes)."""
        truth, offset, mount = self._truth, self._lidar_offset, self._lidar_mount
        # Sensor velocity includes the mount lever arm: v + ω × r.
        velocity_body = truth.attitude.inv().apply(truth.velocity_world_mps) + np.cross(
            truth.body_rates_rad_s, offset
        )
        points = self.lidar_effects.process_scan(
            message,
            mount.inv().apply(velocity_body),
            mount.inv().apply(truth.body_rates_rad_s),
        )
        return mount.apply(points) + offset if len(points) else points

    def _esc_telemetry(self, message: Any, bus: SensorBus | None) -> None:
        report = self.esc_telemetry.sample(
            timestamp_ns(message), rotor_velocities(message)
        )
        if report is None:
            return
        speeds = report.rotor_speed_rad_s
        self._recorder.log_vector(
            "esc/rotor_speed_rad_s",
            speeds,
            [f"rotor_{i}" for i in range(len(speeds))],
        )
        if bus is not None:
            bus.submit("esc", report, timestamp_ns(message))

    def _release_camera_frames(
        self, t: float, last_odometry_s: float | None, bus: SensorBus | None
    ) -> None:
        """Deliver frames whose latency has elapsed, at delivery time."""
        newest = max(round(t * 1e9), round((last_odometry_s or 0) * 1e9))
        for ready_ns, frame in self.camera_delay.pop_ready(newest):
            self._recorder.log_camera_image(frame, ready_ns * 1e-9)
            if bus is not None:
                # Stamped at capture, delivered at release.
                bus.submit(
                    "camera", CameraFrame(frame), ready_ns - self.camera_delay.delay_ns
                )

    def _log_status(self, kind: str, message: Any, t: float, gps_dropped: bool) -> None:
        """Sensor health indicators for Rerun's Sensor status tab.

        1 = healthy/delivered, 0 = dropped/invalid; fault values count active
        faults. Uses only the mission's own fault-model state.
        """
        log = self._recorder.log_status
        if kind == "gps":
            log("gps_delivered", t, 0.0 if gps_dropped else 1.0)
            log("gps_multipath", t, float(self.gps_multipath.active))
        elif kind == "imu":
            stamp = timestamp_ns(message)
            log("imu_faults_active", t, self.imu_faults.active_count(stamp))
        elif kind == "rangefinder":
            from ...sensors.gazebo import rangefinder_distance

            log(
                "rangefinder_valid",
                t,
                float(np.isfinite(rangefinder_distance(message))),
            )

    def _submit(
        self, bus: SensorBus, kind: str, message: Any, gps_dropped: bool
    ) -> None:
        """Submit a processed transported measurement to the sensor bus.

        Lidar (with effects), delayed camera frames, and synthetic sensors
        are submitted where they are produced. Dropped GPS fixes never are.
        """
        if kind == "gps" and gps_dropped:
            return
        if kind == "camera" and not self.camera_effects.enabled:
            bus.submit("camera", CameraFrame(rgb_image(message)), timestamp_ns(message))
            return
        if kind == "lidar":
            return
        payload = bus_payload(kind, message)
        if kind == "gps":
            payload = replace(payload, status=self.gps_status.status)
        if kind == "magnetometer" and self.calibrated_mag is not None:
            payload = replace(payload, calibrated_field_ut=self.calibrated_mag)
        if kind == "imu" and self.calibrated_imu is not None:
            force, rate, temperature = self.calibrated_imu
            payload = replace(
                payload,
                calibrated_specific_force_mps2=force,
                calibrated_angular_rate_rad_s=rate,
                temperature_c=temperature,
            )
        if payload is not None:
            bus.submit(kind, payload, timestamp_ns(message))

    # Synthetic sensors ---------------------------------------------------------

    def on_truth(
        self,
        sim_time_ns: int,
        position_m: NDArray[np.float64],
        velocity_world_mps: NDArray[np.float64],
        quaternion_xyzw: NDArray[np.float64],
        body_rates_rad_s: NDArray[np.float64],
        air_body_mps: NDArray[np.float64],
        counts: Counter,
        bus: SensorBus | None,
    ) -> None:
        """Optical flow and pitot readings synthesized from one truth sample."""
        flow = self.optical_flow.sample(
            sim_time_ns,
            position_m,
            velocity_world_mps,
            quaternion_xyzw,
            body_rates_rad_s,
        )
        pitot = self.airspeed.sample(sim_time_ns, air_body_mps)
        if pitot is not None:
            counts["airspeed"] += 1
            self._recorder.log_vector(
                "airspeed/measured",
                [pitot.indicated_airspeed_mps, pitot.differential_pressure_pa],
                ["indicated_mps", "differential_pa"],
            )
            if bus is not None:
                bus.submit("airspeed", pitot, pitot.sim_time_ns)
        if flow is not None:
            counts["optical_flow"] += 1
            self._recorder.log_optical_flow(flow)
            if bus is not None:
                bus.submit("optical_flow", flow, flow.sim_time_ns)

    def on_battery(
        self,
        sim_time_ns: int,
        state: BatteryState,
        counts: Counter,
        bus: SensorBus | None,
    ) -> None:
        """Battery-monitor reading from the battery model's state."""
        reading = self.battery_monitor.sample(sim_time_ns, state)
        if reading is None:
            return
        counts["battery_monitor"] += 1
        self._recorder.log_battery_monitor(reading)
        self.last_battery_reading = reading
        if bus is not None:
            bus.submit("battery_monitor", reading, reading.sim_time_ns)

    def summary(self) -> dict[str, Any]:
        """Sensor-model entries of ``summary.json``."""
        status = self.gps_status.status
        driver = self.imu_driver
        return dict(
            gps_dropout=self.gps_dropout.summary(),
            accelerometer_vibration=self.vibration.summary(),
            imu_errors=self.imu_errors.summary(),
            gps_errors=self.gps_errors.summary(),
            gps_multipath=self.gps_multipath.summary(),
            imu_faults=self.imu_faults.summary(),
            baro_errors=self.baro_errors.summary(),
            magnetometer_errors=self.magnetometer_errors.summary(),
            magnetic_anomalies=(
                dict(max_field_ut=self.anomalies.max_field_ut)
                if self.anomalies.enabled
                else None
            ),
            optical_flow=self.optical_flow.summary(),
            battery_monitor=self.battery_monitor.summary(),
            lidar_effects=(
                self.lidar_effects.summary() if self.lidar_effects.enabled else None
            ),
            imu_temperature=self.imu_temperature.summary(),
            airspeed=self.airspeed.summary() if self.airspeed.enabled else None,
            gps_receiver=dict(
                last_status=None if status is None else status.__dict__,
                rtk_fixed_fixes=(
                    self.gps_rtk.fixed_fixes if self.gps_rtk.enabled else None
                ),
                heading_samples=self.gps_heading.samples,
            ),
            imu_imperfections=dict(
                enabled=self.imu_imperfections.enabled,
                saturated_samples=self.imu_saturation.saturated_samples,
                dropped_samples=self.imu_timing.dropped,
            ),
            camera_effects=(
                dict(max_blur_px=self.camera_effects.max_blur_px)
                if self.camera_effects.enabled
                else None
            ),
            calibration_warnings=self.calibration_warnings,
            imu_startup_gyro_bias_rad_s=(
                driver.startup_bias_rad_s.tolist()
                if driver is not None and driver.startup_bias_rad_s is not None
                else None
            ),
        )
