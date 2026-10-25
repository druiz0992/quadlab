"""Decode Gazebo camera/lidar data and adapt IMU/GPS measurements."""

from typing import Protocol

import numpy as np
from copy import deepcopy
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation
from ..core.measurements import GpsReading
from .barometer import BaroErrorModel, altitude_to_pressure, pressure_to_altitude
from .gnss.errors import GpsErrorModel
from .magnetometer.errors import MagnetometerErrorModel
from .gnss.multipath import GpsMultipath
from .imu.vibration import AccelerometerVibration


class ImuPerturbation(Protocol):
    """Any IMU model mapping (specific force, angular rate, time) to new values,
    e.g. ImuErrorModel or ImuFaults."""

    @property
    def enabled(self) -> bool: ...

    def apply(
        self,
        specific_force_mps2: ArrayLike,
        angular_rate_rad_s: ArrayLike,
        sim_time_ns: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


def _xyz(vector) -> list[float]:
    return [vector.x, vector.y, vector.z]


def _set_xyz(vector, values) -> None:
    vector.x, vector.y, vector.z = map(float, values)


def _replace_fix(message, fix: GpsReading):
    """Copy a NavSat message with a new position; velocity fields unchanged."""
    result = deepcopy(message)
    result.latitude_deg = fix.latitude_deg
    result.longitude_deg = fix.longitude_deg
    result.altitude = fix.altitude_m
    return result


def _fix(message) -> GpsReading:
    return GpsReading(message.latitude_deg, message.longitude_deg, message.altitude)


def imu_errors_message(message, model: ImuPerturbation, sim_time_ns: int):
    """Copy an IMU message with a model applied to acceleration and gyro.

    Used for both sensor errors and faults. Header, orientation, and the
    source message are unchanged. A disabled model returns the original
    message object.
    """
    if not model.enabled:
        return message
    force, rate = model.apply(
        _xyz(message.linear_acceleration), _xyz(message.angular_velocity), sim_time_ns
    )
    result = deepcopy(message)
    _set_xyz(result.linear_acceleration, force)
    _set_xyz(result.angular_velocity, rate)
    return result


def gps_errors_message(message, model: GpsErrorModel, sim_time_ns: int):
    """Copy a NavSat message with position error added; velocity is unchanged."""
    if not model.enabled:
        return message
    return _replace_fix(message, model.apply(_fix(message), sim_time_ns))


def gps_velocity_message(message, model, sim_time_ns: int):
    """Copy a NavSat message with velocity errors added to its ENU velocity."""
    if not model.enabled:
        return message
    result = deepcopy(message)
    velocity = model.apply(
        [message.velocity_east, message.velocity_north, message.velocity_up],
        sim_time_ns,
    )
    result.velocity_east, result.velocity_north, result.velocity_up = map(
        float, velocity
    )
    return result


def gps_rtk_message(message, model, sim_time_ns: int, degraded: bool):
    """Copy a NavSat message with RTK float/fixed position error."""
    if not model.enabled:
        return message
    return _replace_fix(message, model.apply(_fix(message), sim_time_ns, degraded))


def baro_datum_message(message, elevation_m: float):
    """Copy a FluidPressure message, shifting pressure up by elevation_m.

    Gazebo reports pressure for world z as height above sea level; this makes
    it correspond to elevation_m + z, the same datum as GPS altitude.
    """
    result = deepcopy(message)
    altitude = pressure_to_altitude(message.pressure) + elevation_m
    result.pressure = altitude_to_pressure(altitude)
    return result


def baro_errors_message(
    message,
    model: BaroErrorModel,
    sim_time_ns: int,
    induced_velocity_mps: float = 0.0,
    airspeed_mps: float = 0.0,
):
    """Copy a FluidPressure message with barometer errors applied to pressure.

    Variance and header are unchanged. A disabled model returns the original.
    """
    if not model.enabled:
        return message
    result = deepcopy(message)
    result.pressure = model.apply(
        message.pressure, sim_time_ns, induced_velocity_mps, airspeed_mps
    )
    return result


def gps_multipath_message(message, model: GpsMultipath, sim_time_ns: int, position_m):
    """Copy a NavSat message with any active multipath jump added.

    position_m is the true world position, used only for multipath zones.
    """
    if not model.affects_fixes:
        return message
    return _replace_fix(message, model.apply(_fix(message), sim_time_ns, position_m))


def vibrate_imu_message(message, model: AccelerometerVibration, sim_time_ns: int):
    """Copy an IMU message, modifying only acceleration in sensor axes."""
    if not model.enabled:
        return message
    acceleration = message.linear_acceleration
    measured = model.apply(
        [acceleration.x, acceleration.y, acceleration.z], sim_time_ns
    )
    result = deepcopy(message)
    (
        result.linear_acceleration.x,
        result.linear_acceleration.y,
        result.linear_acceleration.z,
    ) = map(float, measured)
    return result


def rgb_image(message):
    """Remove RGB8 row padding while retaining image height and width."""
    pixels = np.frombuffer(message.data, dtype=np.uint8).reshape(
        message.height, message.step
    )
    return pixels[:, : message.width * 3].reshape(message.height, message.width, 3)


def lidar_points(message, *, position_m=(0, 0, 0.2), rotation_rad=(0, 0, 0)):
    """Return body-frame points using the explicit sensor-to-body mount.

    Handles 2D and multi-channel 3D scans. Gazebo orders ranges row by row:
    index = vertical_index * count + horizontal_index, with elevation
    ``vertical_angle_min + vertical_index * vertical_angle_step`` (0 for a
    single channel). Rotation is fixed-axis roll/pitch/yaw in radians.
    Defaults preserve the original horizontal scanner 0.2 m above the body.
    """
    ranges = np.asarray(message.ranges, dtype=float)
    rows = max(1, getattr(message, "vertical_count", 1) or 1)
    columns = len(ranges) // rows
    azimuth = message.angle_min + np.arange(columns) * message.angle_step
    elevation = getattr(message, "vertical_angle_min", 0.0) + np.arange(rows) * (
        getattr(message, "vertical_angle_step", 0.0) if rows > 1 else 0.0
    )
    azimuth, elevation = np.tile(azimuth, rows), np.repeat(elevation, columns)
    valid = (
        np.isfinite(ranges)
        & (ranges >= message.range_min)
        & (ranges < message.range_max)
    )
    r, az, el = ranges[valid], azimuth[valid], elevation[valid]
    points = np.column_stack(
        (r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el))
    )
    if len(points) == 0:
        return points
    return Rotation.from_euler("xyz", rotation_rad).apply(points) + np.asarray(
        position_m
    )


def rangefinder_distance(message) -> float:
    """Distance (m) from a single-ray rangefinder scan; NaN when out of range.

    Gazebo reports +inf when nothing is hit within range_max.
    """
    distance = float(message.ranges[0]) if len(message.ranges) else float("nan")
    if not (
        np.isfinite(distance) and message.range_min <= distance < message.range_max
    ):
        return float("nan")
    return distance


def magnetometer_errors_message(
    message,
    model: MagnetometerErrorModel,
    total_thrust_n: float,
    battery_current_a: float = 0.0,
):
    """Copy a Magnetometer message with distortions applied to field_tesla."""
    if not model.enabled:
        return message
    result = deepcopy(message)
    field = model.apply(
        _xyz(message.field_tesla),
        total_thrust_n,
        battery_current_a,
        message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nsec,
    )
    _set_xyz(result.field_tesla, field)
    return result


def magnetometer_anomaly_message(message, field_sensor_tesla: ArrayLike):
    """Copy a Magnetometer message with an external field added (sensor axes)."""
    result = deepcopy(message)
    _set_xyz(
        result.field_tesla,
        np.asarray(_xyz(message.field_tesla)) + np.asarray(field_sensor_tesla),
    )
    return result


def bus_payload(kind: str, message):
    """Typed bus payload for a processed Gazebo message, or None.

    Covers the transported sensors (IMU, GPS, barometer, magnetometer,
    rangefinder); synthetic sensors and processed lidar/camera data are
    submitted by the mission directly. Values are the measured ones, after
    the error and fault models.
    """
    from ..core.measurements import (
        BaroReading,
        ImuReading,
        MagnetometerReading,
        RangeReading,
    )

    if kind == "imu":
        return ImuReading(
            np.array(_xyz(message.linear_acceleration)),
            np.array(_xyz(message.angular_velocity)),
        )
    if kind == "gps":
        return GpsReading(
            message.latitude_deg,
            message.longitude_deg,
            message.altitude,
            velocity_enu_mps=np.array(
                [
                    message.velocity_east,
                    message.velocity_north,
                    message.velocity_up,
                ]
            ),
        )
    if kind == "baro":
        return BaroReading(message.pressure, pressure_to_altitude(message.pressure))
    if kind == "magnetometer":
        return MagnetometerReading(np.asarray(_xyz(message.field_tesla)) * 1e6)
    if kind == "rangefinder":
        return RangeReading(rangefinder_distance(message))
    return None
