"""Synthetic bus payloads from truth, one synthesizer per sensor kind.

Each synthesizer has a ``kind`` (a ``sensors/bus.py`` stream), a sampling
``period_ns`` (samples at k · period from t = 0), and ``sample(t_ns,
truth)``, which returns the payload measured at that time or None (no
sample: an outage, or the first GPS epoch). ``build_synthesizer`` makes one
from the merged drone configuration (same keys as ``configs/``; mounts,
rates, ranges, and noise come from there) and ``SynthesisInputs``.

Where RobotLab already synthesizes a sensor in Python, the same model is
used, so semantics match Gazebo runs exactly: optical flow
(``OpticalFlowModel``, flat ground), airspeed (``AirspeedSensor``, still
air), GPS heading (``GpsHeading``), battery monitor (``BatteryMonitor``),
ESC telemetry (``EscTelemetryModel``), and wheel encoders
(``WheelEncoderModel``). Sensors Gazebo produces are synthesized here with
the configured Gaussian noise (``noise_stddev``/``bias`` of their noise
section): baro (ISA pressure at the mount's altitude), magnetometer (the
site field of ``magnetometer.calibration``: strength, inclination,
declination, rotated into sensor axes), and the ray sensors (rangefinder,
lidar, pinhole camera) against a ``SyntheticScene``.

Truth that the trajectory does not carry is derived and documented per
kind: rotor speeds share the thrust ``m · (Rᵀ(a − g))_z`` equally between
lift rotors (``ω = √(T / k)``); the battery pack draws a constant current;
wheels roll without longitudinal slip (speed = contact-point velocity along
the wheel heading / radius), steered wheels point along their contact
velocity, and unsteered wheels may slip sideways.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..core.frames import GRAVITY
from ..core.measurements import (
    BaroReading,
    CameraFrame,
    GpsReading,
    ImuReading,
    LidarPoints,
    MagnetometerReading,
    RangeReading,
)
from ..core.vehicle import DroneProperties
from ..core.wheeled_vehicle import WheeledProperties
from ..sensors.barometer import altitude_to_pressure, pressure_to_altitude
from .synthetic_scene import SyntheticScene
from .trajectories import TruthPoint

IMU_RATE_HZ = 200
GPS_RATE_HZ = 10
GPS_VELOCITY_NOISE_MPS = 0.05
CELL_EMPTY_V, CELL_FULL_V = 3.5, 4.2  # linear open-circuit voltage model


@dataclass(frozen=True)
class SynthesisInputs:
    """Everything besides the configuration that synthesizers need.

    ``reference`` is the GPS origin of world ENU; IMU and GPS noise and
    biases are per sample (m/s², rad/s, m); ``battery_current_a`` is the
    constant pack current.
    """

    reference: GpsReading
    scene: SyntheticScene
    vehicle: DroneProperties
    wheeled: WheeledProperties | None
    seed: int
    imu_noise: tuple[float, float]
    accel_bias_mps2: tuple[float, float, float]
    gyro_bias_rad_s: tuple[float, float, float]
    gps_noise_m: float
    gps_outages_s: tuple[tuple[float, float], ...]
    battery_current_a: float = 10.0


class SensorSynthesizer(Protocol):
    """See module docs."""

    kind: str
    period_ns: int

    def sample(self, sim_time_ns: int, truth: TruthPoint) -> Any | None:
        """Payload measured at ``sim_time_ns``, or None for no sample."""


def _period_ns(rate_hz: float) -> int:
    if not rate_hz > 0:
        raise ValueError(f"sensor rate must be positive, got {rate_hz}")
    return round(1e9 / rate_hz)


def _mount(section: Mapping[str, Any]) -> tuple[NDArray[np.float64], Rotation]:
    """Mount position (body, m) and sensor-to-body rotation (fixed-axis rpy)."""
    mount = section["mount"]
    rotation = Rotation.from_euler("xyz", mount["rotation_deg"], degrees=True)
    return np.asarray(mount["position_m"], float), rotation


def _sensor_pose(
    truth: TruthPoint, position: NDArray[np.float64], rotation: Rotation
) -> tuple[NDArray[np.float64], Rotation]:
    """Sensor origin (world, m) and sensor-to-world rotation."""
    body = Rotation.from_quat(truth.quaternion_xyzw)
    return truth.position_m + body.apply(position), body * rotation


def _gaussian(section: Mapping[str, Any], rng: np.random.Generator, size: int = 0):
    """Bias plus white noise from a Gazebo-style noise section."""
    shape = () if size == 0 else (size,)
    bias = np.asarray(section["bias"], float)
    return bias + np.asarray(section["noise_stddev"], float) * rng.normal(size=shape)


# Inertial and GNSS ----------------------------------------------------------------


class ImuSynthesizer:
    """Specific force ``Rᵀ(a − g)`` and body rates, plus bias and noise."""

    kind = "imu"
    period_ns = _period_ns(IMU_RATE_HZ)

    def __init__(self, inputs: SynthesisInputs, rng: np.random.Generator) -> None:
        self._inputs, self._rng = inputs, rng

    def sample(self, sim_time_ns: int, truth: TruthPoint) -> ImuReading:
        accel_noise, gyro_noise = self._inputs.imu_noise
        to_body = Rotation.from_quat(truth.quaternion_xyzw).inv()
        return ImuReading(
            to_body.apply(truth.acceleration_mps2 - GRAVITY)
            + self._inputs.accel_bias_mps2
            + self._rng.normal(0, accel_noise, 3),
            truth.body_rate_rad_s
            + self._inputs.gyro_bias_rad_s
            + self._rng.normal(0, gyro_noise, 3),
        )


def _gps_epoch(inputs: SynthesisInputs, sim_time_ns: int) -> bool:
    """GPS-receiver epochs: every period after t = 0, none in outages."""
    t_s = sim_time_ns * 1e-9
    outage = any(start <= t_s < end for start, end in inputs.gps_outages_s)
    return sim_time_ns > 0 and not outage


class GpsSynthesizer:
    """Fix at the true position plus noise; ENU velocity plus 0.05 m/s noise."""

    kind = "gps"
    period_ns = _period_ns(GPS_RATE_HZ)

    def __init__(self, inputs: SynthesisInputs, rng: np.random.Generator) -> None:
        self._inputs, self._rng = inputs, rng

    def sample(self, sim_time_ns: int, truth: TruthPoint) -> GpsReading | None:
        if not _gps_epoch(self._inputs, sim_time_ns):
            return None
        from .plugins import enu_to_gps

        noisy = truth.position_m + self._rng.normal(0, self._inputs.gps_noise_m, 3)
        fix = enu_to_gps((float(noisy[0]), float(noisy[1]), float(noisy[2])))
        velocity = truth.velocity_mps + self._rng.normal(0, GPS_VELOCITY_NOISE_MPS, 3)
        return GpsReading(fix.latitude_deg, fix.longitude_deg, fix.altitude_m, velocity)


class GpsHeadingSynthesizer:
    """Dual-antenna heading (``GpsHeading``) at the GPS epochs."""

    kind = "gps_heading"
    period_ns = _period_ns(GPS_RATE_HZ)

    def __init__(self, inputs: SynthesisInputs, settings: Mapping[str, Any]) -> None:
        from ..sensors.gps_receiver import GpsHeading

        self._inputs = inputs
        self._model = GpsHeading(settings)

    def sample(self, sim_time_ns: int, truth: TruthPoint) -> Any | None:
        if not _gps_epoch(self._inputs, sim_time_ns):
            return None
        nose = Rotation.from_quat(truth.quaternion_xyzw).apply([1.0, 0.0, 0.0])
        return self._model.sample(math.atan2(nose[1], nose[0]))


# Air data and magnetics -----------------------------------------------------------


class BaroSynthesizer:
    """ISA static pressure at the mount's altitude, plus noise (Pa)."""

    kind = "baro"

    def __init__(
        self, inputs: SynthesisInputs, section: Mapping[str, Any], rng
    ) -> None:
        self.period_ns = _period_ns(section["rate_hz"])
        self._mount = _mount(section)
        self._noise = section["pressure_pa"]
        self._reference_m = inputs.reference.altitude_m
        self._rng = rng

    def sample(self, sim_time_ns: int, truth: TruthPoint) -> BaroReading:
        origin, _ = _sensor_pose(truth, *self._mount)
        pressure = altitude_to_pressure(self._reference_m + origin[2])
        pressure += float(_gaussian(self._noise, self._rng))
        return BaroReading(pressure, pressure_to_altitude(pressure))


def site_field_ut(calibration: Mapping[str, Any]) -> NDArray[np.float64]:
    """World ENU field (µT) from strength, inclination (down positive), and
    declination (east of north): (F cos I sin D, F cos I cos D, −F sin I)."""
    strength = calibration["field_strength_ut"]
    inclination = math.radians(calibration["inclination_deg"])
    declination = math.radians(calibration["declination_deg"])
    horizontal = strength * math.cos(inclination)
    return np.array(
        [
            horizontal * math.sin(declination),
            horizontal * math.cos(declination),
            -strength * math.sin(inclination),
        ]
    )


class MagnetometerSynthesizer:
    """Site field in sensor axes plus noise (µT); no distortions."""

    kind = "magnetometer"

    def __init__(
        self, inputs: SynthesisInputs, section: Mapping[str, Any], rng
    ) -> None:
        self.period_ns = _period_ns(section["rate_hz"])
        self._mount = _mount(section)
        self._noise = section["field_ut"]
        self._field = site_field_ut(section["calibration"])
        self._rng = rng

    def sample(self, sim_time_ns: int, truth: TruthPoint) -> MagnetometerReading:
        _, sensor = _sensor_pose(truth, *self._mount)
        field = sensor.inv().apply(self._field)
        return MagnetometerReading(field + _gaussian(self._noise, self._rng, 3))


class AirspeedSynthesizer:
    """``AirspeedSensor`` on the body velocity (still air)."""

    kind = "airspeed"

    def __init__(self, section: Mapping[str, Any], air_density_kg_m3: float) -> None:
        from ..sensors.airspeed import AirspeedSensor

        self.period_ns = _period_ns(section["rate_hz"])
        self._model = AirspeedSensor(dict(section, enabled=True), air_density_kg_m3)

    def sample(self, sim_time_ns: int, truth: TruthPoint) -> Any | None:
        body = Rotation.from_quat(truth.quaternion_xyzw).inv()
        return self._model.sample(sim_time_ns, body.apply(truth.velocity_mps))


# Ray sensors ----------------------------------------------------------------------


class RangefinderSynthesizer:
    """Distance along the sensor +x axis to the scene, plus noise; NaN
    outside [min_range_m, max_range_m)."""

    kind = "rangefinder"

    def __init__(
        self, inputs: SynthesisInputs, section: Mapping[str, Any], rng
    ) -> None:
        self.period_ns = _period_ns(section["rate_hz"])
        self._mount = _mount(section)
        self._range = (section["min_range_m"], section["max_range_m"])
        self._noise = section["range_m"]
        self._scene, self._rng = inputs.scene, rng

    def sample(self, sim_time_ns: int, truth: TruthPoint) -> RangeReading:
        origin, sensor = _sensor_pose(truth, *self._mount)
        hits = self._scene.cast(origin, sensor.apply([[1.0, 0.0, 0.0]]), math.inf)
        distance = float(hits.distance_m[0]) + float(_gaussian(self._noise, self._rng))
        low, high = self._range
        return RangeReading(distance if low <= distance < high else math.nan)


class LidarSynthesizer:
    """Scan (azimuth 0..2π × configured vertical angles) against the scene.

    Points are body-axis (m), in range [min, max] after per-beam noise.
    """

    kind = "lidar"

    def __init__(
        self, inputs: SynthesisInputs, section: Mapping[str, Any], rng
    ) -> None:
        self.period_ns = _period_ns(section["rate_hz"])
        self._mount = _mount(section)
        self._range = (section["min_range_m"], section["max_range_m"])
        self._noise = section["range_m"]
        self._scene, self._rng = inputs.scene, rng
        azimuth = np.linspace(0, 2 * np.pi, section["samples"], endpoint=False)
        elevation = np.radians(
            np.linspace(
                section["vertical_min_angle_deg"],
                section["vertical_max_angle_deg"],
                section["vertical_samples"],
            )
        )
        a, e = np.meshgrid(azimuth, elevation)
        self._directions = np.column_stack(
            [
                (np.cos(e) * np.cos(a)).ravel(),
                (np.cos(e) * np.sin(a)).ravel(),
                np.sin(e).ravel(),
            ]
        )

    def sample(self, sim_time_ns: int, truth: TruthPoint) -> LidarPoints:
        origin, sensor = _sensor_pose(truth, *self._mount)
        world = sensor.apply(self._directions)
        hits = self._scene.cast(origin, world, self._range[1])
        ranges = hits.distance_m + _gaussian(self._noise, self._rng, len(world))
        keep = np.isfinite(hits.distance_m) & (ranges >= self._range[0])
        keep &= ranges <= self._range[1]
        position, rotation = self._mount
        points = position + rotation.apply(self._directions[keep] * ranges[keep, None])
        return LidarPoints(points, self._scene.reflectivity(hits)[keep])


class CameraSynthesizer:
    """Pinhole RGB image of the scene (optical axis +x, image right = −y,
    image down = −z, as Gazebo cameras), with intensity noise (0..1 scale)."""

    kind = "camera"

    def __init__(
        self, inputs: SynthesisInputs, section: Mapping[str, Any], rng
    ) -> None:
        if section["lens"]["type"] != "pinhole":
            raise ValueError("synthetic camera supports only the pinhole lens")
        self.period_ns = _period_ns(section["rate_hz"])
        self._mount = _mount(section)
        self._shape = (section["height"], section["width"])
        self._far_m = section["far_m"]
        self._noise = section["intensity"]
        self._scene, self._rng = inputs.scene, rng
        width, height = section["width"], section["height"]
        focal = width / 2 / math.tan(section["horizontal_fov_rad"] / 2)
        u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
        rays = np.column_stack(
            [
                np.full(u.size, focal),
                -(u.ravel() - width / 2),
                -(v.ravel() - height / 2),
            ]
        )
        self._rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def sample(self, sim_time_ns: int, truth: TruthPoint) -> CameraFrame:
        origin, sensor = _sensor_pose(truth, *self._mount)
        world = sensor.apply(self._rays)
        rgb = self._scene.shade(
            origin, world, self._scene.cast(origin, world, self._far_m)
        )
        rgb += 255 * _gaussian(self._noise, self._rng, rgb.size).reshape(rgb.shape)
        image = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        return CameraFrame(image.reshape(*self._shape, 3))


class OpticalFlowSynthesizer:
    """``OpticalFlowModel`` over flat ground (scene boxes are ignored)."""

    kind = "optical_flow"

    def __init__(self, section: Mapping[str, Any]) -> None:
        from ..sensors.optical_flow import OpticalFlowModel

        self.period_ns = _period_ns(section["rate_hz"])
        self._model = OpticalFlowModel(dict(section, enabled=True))

    def sample(self, sim_time_ns: int, truth: TruthPoint) -> Any | None:
        return self._model.sample(
            sim_time_ns,
            truth.position_m,
            truth.velocity_mps,
            truth.quaternion_xyzw,
            truth.body_rate_rad_s,
        )


# Vehicle-internal sensors ---------------------------------------------------------


class BatterySynthesizer:
    """``BatteryMonitor`` on a pack drawing a constant current.

    Open-circuit voltage is linear in state of charge between 3.5 and
    4.2 V per cell; terminal voltage subtracts I · R.
    """

    kind = "battery_monitor"

    def __init__(
        self,
        inputs: SynthesisInputs,
        section: Mapping[str, Any],
        battery: Mapping[str, Any],
    ) -> None:
        from ..sensors.battery_monitor import BatteryMonitor

        self.period_ns = _period_ns(section["rate_hz"])
        self._model = BatteryMonitor(
            dict(section, enabled=True), battery["capacity_mah"]
        )
        self._battery = battery
        self._current_a = inputs.battery_current_a

    def sample(self, sim_time_ns: int, truth: TruthPoint) -> Any | None:
        from ..simulation.battery import BatteryState

        b, current = self._battery, self._current_a
        used_ah = current * sim_time_ns * 1e-9 / 3600
        soc = max(b["initial_soc"] - used_ah * 1000 / b["capacity_mah"], 0.0)
        ocv = b["cells"] * (CELL_EMPTY_V + (CELL_FULL_V - CELL_EMPTY_V) * soc)
        terminal = ocv - current * b["internal_resistance_ohm"]
        state = BatteryState(
            soc, ocv, terminal, current, terminal * current, used_ah * ocv
        )
        return self._model.sample(sim_time_ns, state)


class EscSynthesizer:
    """``EscTelemetryModel`` on hover-share rotor speeds (see module docs)."""

    kind = "esc"

    def __init__(self, inputs: SynthesisInputs, section: Mapping[str, Any]) -> None:
        from ..sensors.esc_telemetry import EscTelemetryModel

        self.period_ns = _period_ns(section["rate_hz"])
        self._model = EscTelemetryModel(dict(section, enabled=True), slowdown=1.0)
        self._vehicle = inputs.vehicle

    def sample(self, sim_time_ns: int, truth: TruthPoint) -> Any | None:
        vehicle = self._vehicle
        body = Rotation.from_quat(truth.quaternion_xyzw).inv()
        thrust_n = vehicle.mass * body.apply(truth.acceleration_mps2 - GRAVITY)[2]
        lift = vehicle.lift_rotors
        speeds = np.zeros(len(vehicle.rotors))
        per_rotor = max(thrust_n, 0.0) / len(lift)
        speeds[list(lift)] = math.sqrt(per_rotor / vehicle.thrust_coefficient)
        return self._model.sample(sim_time_ns, speeds.tolist())


class WheelEncoderSynthesizer:
    """``WheelEncoderModel`` on rolling-without-slip wheels (module docs)."""

    kind = "wheel_encoders"

    def __init__(self, inputs: SynthesisInputs, section: Mapping[str, Any]) -> None:
        from ..sensors.wheel_encoders import WheelEncoderModel

        if inputs.wheeled is None:
            raise ValueError("wheel_encoders need a wheeled scenario vehicle")
        self.period_ns = _period_ns(section["rate_hz"])
        self._wheeled = inputs.wheeled
        self._model = WheelEncoderModel(section, inputs.wheeled.wheel_count)
        self._angle = np.zeros(inputs.wheeled.wheel_count)
        self._last_ns: int | None = None

    def sample(self, sim_time_ns: int, truth: TruthPoint) -> Any:
        wheeled = self._wheeled
        body = Rotation.from_quat(truth.quaternion_xyzw).inv()
        hubs = wheeled.hub_positions_m()
        # Contact-point velocity in body axes: v + ω × r (planar part).
        velocity = body.apply(truth.velocity_mps) + np.cross(
            truth.body_rate_rad_s, hubs
        )
        steered = wheeled.mask("steered")
        steer = np.where(
            steered & (np.hypot(velocity[:, 0], velocity[:, 1]) > 1e-6),
            np.arctan2(velocity[:, 1], velocity[:, 0]),
            0.0,
        )
        rolling = velocity[:, 0] * np.cos(steer) + velocity[:, 1] * np.sin(steer)
        speed = rolling / wheeled.tyre.radius_m
        if self._last_ns is not None:
            self._angle = self._angle + speed * (sim_time_ns - self._last_ns) * 1e-9
        self._last_ns = sim_time_ns
        return self._model.read(sim_time_ns, self._angle.copy(), speed, steer)


# Construction ---------------------------------------------------------------------


SynthesizerBuilder = Callable[
    [SynthesisInputs, Mapping[str, Any], np.random.Generator], SensorSynthesizer
]


def _seeded(section: Mapping[str, Any], inputs: SynthesisInputs, kind: str):
    """Section with a model seed distinct per scenario seed and kind."""
    from ..sensors.bus import PRIORITY

    return dict(section, seed=inputs.seed * 100 + PRIORITY[kind])


BUILDERS: dict[str, SynthesizerBuilder] = {
    "imu": lambda i, s, r: ImuSynthesizer(i, r),
    "gps": lambda i, s, r: GpsSynthesizer(i, r),
    "gps_heading": lambda i, s, r: GpsHeadingSynthesizer(
        i, _seeded(s["sensors"]["gps"]["heading"], i, "gps_heading")
    ),
    "baro": lambda i, s, r: BaroSynthesizer(i, s["sensors"]["baro"], r),
    "magnetometer": lambda i, s, r: MagnetometerSynthesizer(
        i, s["sensors"]["magnetometer"], r
    ),
    "rangefinder": lambda i, s, r: RangefinderSynthesizer(
        i, s["sensors"]["rangefinder"], r
    ),
    "optical_flow": lambda i, s, r: OpticalFlowSynthesizer(
        _seeded(s["sensors"]["optical_flow"], i, "optical_flow")
    ),
    "battery_monitor": lambda i, s, r: BatterySynthesizer(
        i, _seeded(s["sensors"]["battery_monitor"], i, "battery_monitor"), s["battery"]
    ),
    "airspeed": lambda i, s, r: AirspeedSynthesizer(
        _seeded(s["sensors"]["airspeed"], i, "airspeed"),
        s["aerodynamics"]["air_density_kg_m3"],
    ),
    "esc": lambda i, s, r: EscSynthesizer(
        i, _seeded(s["motors"]["esc"]["telemetry"], i, "esc")
    ),
    "lidar": lambda i, s, r: LidarSynthesizer(i, s["sensors"]["lidar"], r),
    "camera": lambda i, s, r: CameraSynthesizer(i, s["sensors"]["camera"], r),
    "wheel_encoders": lambda i, s, r: WheelEncoderSynthesizer(
        i, _seeded(s["sensors"]["wheel_encoders"], i, "wheel_encoders")
    ),
}


def build_synthesizer(
    kind: str, inputs: SynthesisInputs, settings: Mapping[str, Any]
) -> SensorSynthesizer:
    """Synthesizer for ``kind`` from the merged configuration ``settings``.

    Each kind gets its own RNG (seeded from ``inputs.seed`` and the kind),
    so selecting more sensors never changes another sensor's samples.
    Raises ``ValueError`` for an unknown kind.
    """
    if kind not in BUILDERS:
        raise ValueError(
            f"no synthetic {kind!r} sensor; choose from {sorted(BUILDERS)}"
        )
    from ..sensors.bus import PRIORITY

    rng = np.random.default_rng([inputs.seed, PRIORITY[kind]])
    return BUILDERS[kind](inputs, settings, rng)
