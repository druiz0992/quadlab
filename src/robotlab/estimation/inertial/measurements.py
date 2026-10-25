"""Measurements converted for inertial filters: frames, units, covariance.

Bus samples carry raw sensor payloads in sensor axes or geodetic
coordinates. ``MeasurementConverter`` turns each into the form a filter
update needs, so every filter receives the same conventions:

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

Units are SI except the magnetic field (µT, as the bus carries it).
Invalid options raise ValueError when the converter is built.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from ...core.algorithms import SensorDescription
from ...core.measurements import BaroReading, GpsReading, MagnetometerReading
from ...sensors.geodesy import gps_to_enu

CONVERTER_DEFAULTS: dict[str, Any] = dict(
    gps_std_m=[0.5, 0.5, 0.8],
    use_receiver_gps_accuracy=True,
    magnetometer_std_ut=1.0,
    baro_std_m=0.5,
)


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
    validates its own options).
    """

    def __init__(
        self,
        sensors: Mapping[str, SensorDescription],
        reference: GpsReading,
        options: Mapping[str, Any],
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

    def convert(self, kind: str, payload: Any) -> Any:
        """The measurement for ``kind``'s payload; raises KeyError for a kind
        without a converter."""
        if kind == "gps":
            return self.gps(payload)
        if kind == "magnetometer":
            return self.magnetometer(payload)
        if kind == "baro":
            return self.baro(payload)
        raise KeyError(f"No measurement converter for {kind!r}")

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
