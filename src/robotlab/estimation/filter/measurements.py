"""Measurements converted for recursive filters: frames, units, covariance.

Bus samples carry raw sensor payloads in sensor axes or geodetic
coordinates. ``MeasurementConverter`` turns each into the form a filter's
``predict_<kind>``/``update_<kind>`` needs, so every filter receives the
same conventions:

- ``ImuInput`` (stream ``imu``): specific force (m/s²) and angular rate
  (rad/s) in **body** axes, the driver-calibrated values when present,
  rotated by the IMU mount. The IMU must sit at the body origin (no
  lever-arm compensation; ValueError at construction otherwise).
- ``WheelMotion`` (``wheel_encoders``): per wheel (wheel order), the hub
  position in body x/y (m, nominal geometry from the context), the rolling
  direction in body x/y (unit vector at the reported steering angle,
  positive about +z), and the rolling speed (encoder rad/s × nominal tyre
  radius, m/s). The encoder speed averages the interval ending at the
  sample time (sensors/wheel_encoders.py). Needs the context's wheeled
  vehicle (ValueError at construction otherwise).
- ``GpsFix`` (stream ``gps``): antenna position in local ENU (m) around the
  context's WGS84 reference, its 3×3 covariance (m²), the antenna lever arm
  in body axes (m), and, when the receiver reports it, ENU velocity (m/s)
  with an isotropic variance from the reported speed accuracy. The position
  covariance is the receiver's own 1-σ horizontal/vertical accuracy (assumed
  uncorrelated) with ``use_receiver_gps_accuracy`` and valid values, else
  ``diag(gps_std_m²)``.
- ``MagneticField`` (``magnetometer``): field in body axes (µT), the
  driver-calibrated reading when a calibration is enabled, rotated by the
  sensor mount; covariance ``magnetometer_std_ut² · I``. The expected world
  field is the filter's own knowledge (an option), not part of the data.
- ``BaroAltitude`` (``baro``): ISA pressure altitude minus the reference's
  altitude (m), i.e. local up for a standard atmosphere; variance
  ``baro_std_m²``. Datum and drift errors remain in the value: a filter
  that needs them estimates a baro offset itself.

Any other stream reaches the filter as its bus payload, unchanged (sensor
frame, as documented in core/measurements.py). Units are SI except the
magnetic field (µT, as the bus carries it). Invalid options raise
ValueError when the converter is built.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from ...core.algorithms import SensorDescription
from ...core.measurements import (
    BaroReading,
    GpsReading,
    ImuReading,
    MagnetometerReading,
    readonly_array,
    EncoderReading,
)
from ...core.wheeled_vehicle import WheeledProperties
from ...core.geodesy import gps_to_enu

CONVERTER_DEFAULTS: dict[str, Any] = dict(
    gps_std_m=[0.5, 0.5, 0.8],
    use_receiver_gps_accuracy=True,
    magnetometer_std_ut=1.0,
    baro_std_m=0.5,
)


@dataclass(frozen=True)
class ImuInput:
    """One IMU reading in body axes: specific force (m/s²), rate (rad/s)."""

    specific_force_mps2: NDArray[np.float64]
    angular_rate_rad_s: NDArray[np.float64]


@dataclass(frozen=True)
class WheelMotion:
    """Wheel rolling in body x/y (see module docs); arrays in wheel order:
    ``hub_positions_body_m`` and ``rolling_direction_body`` are n×2,
    ``rolling_speed_mps`` has n entries."""

    hub_positions_body_m: NDArray[np.float64]
    rolling_direction_body: NDArray[np.float64]
    rolling_speed_mps: NDArray[np.float64]


@dataclass(frozen=True)
class GpsFix:
    """GPS fix in local ENU (see module docs)."""

    position_enu_m: NDArray[np.float64]
    covariance_m2: NDArray[np.float64]
    lever_arm_body_m: NDArray[np.float64]
    velocity_enu_mps: NDArray[np.float64] | None = None
    velocity_variance_m2_s2: float | None = None


@dataclass(frozen=True)
class MagneticField:
    """Magnetic field in body axes (µT) and its 3×3 covariance (µT²)."""

    field_body_ut: NDArray[np.float64]
    covariance_ut2: NDArray[np.float64]


@dataclass(frozen=True)
class BaroAltitude:
    """Barometric altitude above the reference (m) and its variance (m²)."""

    altitude_m: float
    variance_m2: float


class MeasurementConverter:
    """Convert bus payloads for filters (see module docs).

    ``sensors`` are the context's sensor descriptions (mounts),
    ``reference`` the local ENU origin, ``options`` any of
    ``CONVERTER_DEFAULTS`` (unknown keys are ignored here; the estimator
    validates its own options). ``kinds`` are the streams that will be
    converted, checked now: a body-origin IMU, a wheeled vehicle for wheel
    encoders. ``wheeled`` is the context's nominal wheeled vehicle;
    ``imu_quaternion_body_xyzw`` overrides the IMU mount rotation.
    """

    def __init__(
        self,
        sensors: Mapping[str, SensorDescription],
        reference: GpsReading,
        options: Mapping[str, Any],
        *,
        kinds: tuple[str, ...] = (),
        wheeled: WheeledProperties | None = None,
        imu_quaternion_body_xyzw: ArrayLike | None = None,
    ) -> None:
        settings = {**CONVERTER_DEFAULTS, **options}
        gps_std = np.asarray(settings["gps_std_m"], dtype=float)
        if gps_std.shape != (3,) or not np.all(np.isfinite(gps_std) & (gps_std > 0)):
            raise ValueError("gps_std_m needs three positive numbers (east, north, up)")
        for key in ("magnetometer_std_ut", "baro_std_m"):
            value = float(settings[key])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be a positive number")
        self.reference = reference
        self.gps_covariance_m2 = np.diag(gps_std**2)
        self.use_receiver_gps_accuracy = bool(settings["use_receiver_gps_accuracy"])
        self.magnetometer_std_ut = float(settings["magnetometer_std_ut"])
        self.baro_std_m = float(settings["baro_std_m"])
        self._mounts = sensors
        self.receiver_covariance_fixes = 0
        self.imu_rotation = (
            _imu_rotation(sensors.get("imu"), imu_quaternion_body_xyzw)
            if "imu" in kinds
            else Rotation.identity()
        )
        if "wheel_encoders" in kinds and wheeled is None:
            raise ValueError("wheel_encoders need the context's wheeled vehicle")
        self._wheeled = wheeled

    def convert(self, kind: str, payload: Any) -> Any:
        """The measurement for ``kind``'s payload (module docs); the payload
        itself for a kind without a converter."""
        converters: dict[str, Callable[[Any], Any]] = {
            "imu": self.imu,
            "wheel_encoders": self.wheel_encoders,
            "gps": self.gps,
            "magnetometer": self.magnetometer,
            "baro": self.baro,
        }
        converter = converters.get(kind)
        return payload if converter is None else converter(payload)

    def imu(self, reading: ImuReading) -> ImuInput:
        return ImuInput(
            specific_force_mps2=self.imu_rotation.apply(
                reading.best_specific_force_mps2
            ),
            angular_rate_rad_s=self.imu_rotation.apply(reading.best_angular_rate_rad_s),
        )

    def wheel_encoders(self, reading: EncoderReading) -> WheelMotion:
        wheeled = self._wheeled
        if wheeled is None:
            raise ValueError("wheel_encoders need the context's wheeled vehicle")
        steer = np.asarray(reading.steer_angle_rad, float)
        speed = np.asarray(reading.wheel_speed_rad_s, float)
        if steer.shape != speed.shape or speed.shape != (wheeled.wheel_count,):
            raise ValueError(
                f"wheel_encoders reading needs {wheeled.wheel_count} wheels"
            )
        return WheelMotion(
            hub_positions_body_m=wheeled.hub_positions_m()[:, :2],
            rolling_direction_body=np.column_stack([np.cos(steer), np.sin(steer)]),
            rolling_speed_mps=speed * wheeled.tyre.radius_m,
        )

    def gps(self, fix: GpsReading) -> GpsFix:
        status = fix.status
        velocity_variance = None
        if fix.velocity_enu_mps is not None and status is not None:
            speed = status.speed_accuracy_mps
            if np.isfinite(speed) and speed > 0:
                velocity_variance = speed * speed
        return GpsFix(
            position_enu_m=gps_to_enu(fix, self.reference),
            covariance_m2=self._gps_covariance(fix),
            lever_arm_body_m=self._lever_arm("gps"),
            velocity_enu_mps=fix.velocity_enu_mps,
            velocity_variance_m2_s2=velocity_variance,
        )

    def _gps_covariance(self, fix: GpsReading) -> NDArray[np.float64]:
        """The receiver's covariance when enabled and valid, else the
        configured one."""
        status = fix.status
        if self.use_receiver_gps_accuracy and status is not None:
            h, v = status.horizontal_accuracy_m, status.vertical_accuracy_m
            if np.isfinite([h, v]).all() and h > 0 and v > 0:
                self.receiver_covariance_fixes += 1
                return np.diag([h * h, h * h, v * v])
        return self.gps_covariance_m2

    def magnetometer(self, reading: MagnetometerReading) -> MagneticField:
        field = (
            reading.field_ut
            if reading.calibrated_field_ut is None
            else reading.calibrated_field_ut
        )
        return MagneticField(
            field_body_ut=self._mount_rotation("magnetometer").apply(field),
            covariance_ut2=np.eye(3) * self.magnetometer_std_ut**2,
        )

    def baro(self, reading: BaroReading) -> BaroAltitude:
        return BaroAltitude(
            altitude_m=float(reading.altitude_m - self.reference.altitude_m),
            variance_m2=self.baro_std_m**2,
        )

    def _lever_arm(self, kind: str) -> NDArray[np.float64]:
        sensor = self._mounts.get(kind)
        position: ArrayLike = (
            (0.0, 0.0, 0.0) if sensor is None else sensor.mount_position_m
        )
        return np.array(position, dtype=float)

    def _mount_rotation(self, kind: str) -> Rotation:
        sensor = self._mounts.get(kind)
        return (
            Rotation.identity()
            if sensor is None
            else Rotation.from_quat(sensor.mount_quaternion_xyzw)
        )


def _imu_rotation(
    sensor: SensorDescription | None, quaternion: ArrayLike | None
) -> Rotation:
    """Sensor-to-body rotation of the IMU: the explicit ``quaternion``, else
    the sensor description's mount; raises ValueError for a mount position
    off the body origin or an unnormalized quaternion."""
    if sensor is not None and any(sensor.mount_position_m):
        raise ValueError(
            "IMU-driven filters require a body-origin IMU; lever-arm "
            "acceleration correction is not implemented"
        )
    if quaternion is None:
        quaternion = (0, 0, 0, 1) if sensor is None else sensor.mount_quaternion_xyzw
    q = readonly_array(quaternion, (4,))
    if not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1):
        raise ValueError("IMU mount quaternion must be normalized")
    return Rotation.from_quat(q)
