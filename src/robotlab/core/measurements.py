"""Small immutable contracts for in-process sensor/algorithm examples."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")


def readonly_array(values, shape):
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Expected finite array of shape {shape}")
    # Bytes-backed arrays cannot be made writable again by a subscriber.
    return np.frombuffer(array.tobytes(), dtype=float).reshape(shape)


@dataclass(frozen=True)
class ImuReading:
    """Raw IMU sample (sensor axes); with an enabled calibration also the
    driver-corrected values, and the die temperature the IMU reports."""

    specific_force_mps2: np.ndarray
    angular_rate_rad_s: np.ndarray
    calibrated_specific_force_mps2: np.ndarray | None = None
    calibrated_angular_rate_rad_s: np.ndarray | None = None
    temperature_c: float | None = None

    def __post_init__(self):
        for field in (
            "specific_force_mps2",
            "angular_rate_rad_s",
            "calibrated_specific_force_mps2",
            "calibrated_angular_rate_rad_s",
        ):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, readonly_array(value, (3,)))

    @property
    def best_specific_force_mps2(self) -> np.ndarray:
        """Calibrated specific force when available, else raw."""
        if self.calibrated_specific_force_mps2 is None:
            return self.specific_force_mps2
        return self.calibrated_specific_force_mps2

    @property
    def best_angular_rate_rad_s(self) -> np.ndarray:
        """Calibrated angular rate when available, else raw."""
        if self.calibrated_angular_rate_rad_s is None:
            return self.angular_rate_rad_s
        return self.calibrated_angular_rate_rad_s


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


# Error-state blocks of an estimate, in covariance order, with their sizes.
ESTIMATE_BLOCKS: tuple[tuple[str, int], ...] = (
    ("position_m", 3),
    ("velocity_mps", 3),
    ("quaternion_xyzw", 4),  # 3 attitude error states (rotation vector)
    ("accel_bias_mps2", 3),
    ("gyro_bias_rad_s", 3),
)


@dataclass(frozen=True)
class EstimatedState:
    """A pose estimate: any subset of the blocks in ``ESTIMATE_BLOCKS``.

    World ENU position (m) and velocity (m/s), body-to-world quaternion
    (xyzw), and body-axis accelerometer (m/s²) and gyro (rad/s) biases.
    A block an estimator does not estimate is None; at least position or
    attitude must be present. ``covariance`` covers exactly the present
    blocks, in that order, with 3 error states each (attitude as a right
    rotation-vector error, truth = estimate ⊕ error), so the full state is
    15×15; None means the estimator reports no uncertainty (no NEES).
    """

    position_m: np.ndarray | None = None
    velocity_mps: np.ndarray | None = None
    quaternion_xyzw: np.ndarray | None = None
    accel_bias_mps2: np.ndarray | None = None
    gyro_bias_rad_s: np.ndarray | None = None
    covariance: np.ndarray | None = None

    def __post_init__(self):
        for name, size in ESTIMATE_BLOCKS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, readonly_array(value, (size,)))
        if self.position_m is None and self.quaternion_xyzw is None:
            raise ValueError("An estimate needs a position or an attitude")
        if self.covariance is not None:
            size = 3 * len(self.blocks)
            object.__setattr__(
                self, "covariance", readonly_array(self.covariance, (size, size))
            )

    @property
    def blocks(self) -> tuple[str, ...]:
        """Names of the present blocks, in covariance order."""
        return tuple(n for n, _ in ESTIMATE_BLOCKS if getattr(self, n) is not None)

    def covariance_block(self, name: str) -> np.ndarray | None:
        """3×3 covariance of one present block, or None if unknown."""
        if self.covariance is None or name not in self.blocks:
            return None
        start = 3 * self.blocks.index(name)
        return self.covariance[start : start + 3, start : start + 3]


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
    """Magnetic field in sensor axes, µT: raw, and corrected with the
    configured calibration (as an autopilot driver applies it) when one is
    enabled."""

    field_ut: np.ndarray
    calibrated_field_ut: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "field_ut", readonly_array(self.field_ut, (3,)))
        if self.calibrated_field_ut is not None:
            object.__setattr__(
                self,
                "calibrated_field_ut",
                readonly_array(self.calibrated_field_ut, (3,)),
            )


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


# Readings of sensors simulated in Python (models in sensors/).


@dataclass(frozen=True)
class AirspeedReading:
    """Differential pressure (Pa) and indicated airspeed (m/s)."""

    sim_time_ns: int
    differential_pressure_pa: float
    indicated_airspeed_mps: float


@dataclass(frozen=True)
class BatteryReading:
    sim_time_ns: int
    voltage_v: float
    current_a: float
    consumed_mah: float
    remaining_pct: float


@dataclass(frozen=True)
class JointStateReading:
    """One reading per actuated joint (``names`` order) at ``sim_time_ns``."""

    sim_time_ns: int
    names: tuple[str, ...]
    counts: tuple[int, ...]
    position_rad: NDArray[np.float64]
    velocity_rad_s: NDArray[np.float64]


@dataclass(frozen=True)
class OpticalFlowReading:
    """One flow sample; flow is NaN when invalid."""

    sim_time_ns: int
    flow_rad_s: NDArray[np.float64]  # Image-axis angular rates (sensor y, z)
    distance_m: float  # Along the optical axis to the ground (NaN if none)
    valid: bool


@dataclass(frozen=True)
class EncoderReading:
    """One reading per wheel (wheel order), at ``sim_time_ns``."""

    sim_time_ns: int
    counts: tuple[int, ...]
    wheel_speed_rad_s: NDArray[np.float64]
    steer_angle_rad: NDArray[np.float64]
