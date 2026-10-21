"""Small immutable contracts for in-process sensor/algorithm examples."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


def readonly_array(values, shape):
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Expected finite array of shape {shape}")
    # Bytes-backed arrays cannot be made writable again by a subscriber.
    return np.frombuffer(array.tobytes(), dtype=float).reshape(shape)


@dataclass(frozen=True)
class ImuReading:
    specific_force_mps2: np.ndarray
    angular_rate_rad_s: np.ndarray

    def __post_init__(self):
        for field in ("specific_force_mps2", "angular_rate_rad_s"):
            object.__setattr__(self, field, readonly_array(getattr(self, field), (3,)))


@dataclass(frozen=True)
class GpsStatus:
    """Receiver-reported fix quality (as from a UBX-NAV-PVT-like message).

    ``fix_type``: "none", "2d", "3d", "rtk_float", or "rtk_fixed".
    Accuracies are the receiver's own 1-σ estimates, not the true error.
    """

    fix_type: str
    satellites: int
    hdop: float
    horizontal_accuracy_m: float
    vertical_accuracy_m: float
    speed_accuracy_mps: float


@dataclass(frozen=True)
class GpsReading:
    """Geodetic fix; optional ENU velocity (m/s) and receiver status."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    velocity_enu_mps: np.ndarray | None = None
    status: GpsStatus | None = None

    def __post_init__(self):
        if self.velocity_enu_mps is not None:
            object.__setattr__(
                self, "velocity_enu_mps", readonly_array(self.velocity_enu_mps, (3,))
            )
        if (
            not np.isfinite(
                [self.latitude_deg, self.longitude_deg, self.altitude_m]
            ).all()
            or not -90 <= self.latitude_deg <= 90
            or not -180 <= self.longitude_deg <= 180
        ):
            raise ValueError("Invalid geodetic GPS fix")


@dataclass(frozen=True)
class EstimatedState:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    quaternion_xyzw: np.ndarray
    accel_bias_mps2: np.ndarray
    gyro_bias_rad_s: np.ndarray
    covariance: (
        np.ndarray
    )  # error order: position, velocity, attitude, accel bias, gyro bias

    def __post_init__(self):
        for name in (
            "position_m",
            "velocity_mps",
            "accel_bias_mps2",
            "gyro_bias_rad_s",
        ):
            object.__setattr__(self, name, readonly_array(getattr(self, name), (3,)))
        object.__setattr__(
            self, "quaternion_xyzw", readonly_array(self.quaternion_xyzw, (4,))
        )
        object.__setattr__(
            self, "covariance", readonly_array(self.covariance, (15, 15))
        )


@dataclass(frozen=True)
class Sample(Generic[T]):
    stream_id: str
    sequence: int
    epoch: int
    sim_time_ns: int
    received_monotonic_ns: int
    frame_id: str
    payload: T
    parents: tuple[tuple[str, int], ...] = ()

    def __post_init__(self):
        for name, minimum in (
            ("sequence", 1),
            ("epoch", 0),
            ("sim_time_ns", 0),
            ("received_monotonic_ns", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Invalid sample {name}")
        if not self.stream_id or not self.frame_id:
            raise ValueError("Stream and frame IDs are required")
        object.__setattr__(
            self, "parents", tuple(tuple(parent) for parent in self.parents)
        )


@dataclass(frozen=True)
class BaroReading:
    """Static pressure (Pa) and its ISA pressure altitude above sea level (m)."""

    pressure_pa: float
    altitude_m: float


@dataclass(frozen=True)
class MagnetometerReading:
    """Magnetic field in sensor axes, µT."""

    field_ut: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "field_ut", readonly_array(self.field_ut, (3,)))


@dataclass(frozen=True)
class RangeReading:
    """Distance along the rangefinder ray (m); NaN when nothing is in range."""

    distance_m: float


@dataclass(frozen=True)
class LidarPoints:
    """One processed lidar scan in body axes (N×3, m) with per-point
    reflectivity (range-normalized return strength; 1 = white Lambertian)."""

    points_m: np.ndarray
    reflectivity: np.ndarray

    def __post_init__(self):
        points = np.asarray(self.points_m, float).reshape(-1, 3)
        values = np.asarray(self.reflectivity, float).reshape(-1)
        if len(values) != len(points):
            raise ValueError("Lidar reflectivity must match the number of points")
        object.__setattr__(self, "points_m", readonly_array(points, points.shape))
        object.__setattr__(self, "reflectivity", readonly_array(values, values.shape))


@dataclass(frozen=True)
class CameraFrame:
    """Processed RGB image (H×W×3 uint8, read-only) from the camera."""

    image: np.ndarray

    def __post_init__(self):
        image = np.ascontiguousarray(self.image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Camera frames must be H×W×3")
        copy = image.copy()
        copy.flags.writeable = False
        object.__setattr__(self, "image", copy)


@dataclass(frozen=True)
class GpsHeadingReading:
    """Dual-antenna GNSS heading: yaw of the antenna baseline (rad, ENU,
    counter-clockwise from east) with the receiver's 1-σ accuracy."""

    heading_rad: float
    accuracy_rad: float


@dataclass(frozen=True)
class EscTelemetry:
    """Measured rotor speeds (rad/s) reported by the ESCs, one per rotor."""

    rotor_speed_rad_s: np.ndarray

    def __post_init__(self):
        values = np.asarray(self.rotor_speed_rad_s, float).reshape(-1)
        object.__setattr__(
            self, "rotor_speed_rad_s", readonly_array(values, values.shape)
        )
