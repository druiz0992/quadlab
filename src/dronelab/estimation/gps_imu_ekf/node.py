"""Timestamped GPS/IMU stream adaptation and state-estimate publication."""

import threading
from types import MappingProxyType
from typing import Mapping, TypeAlias, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from ...core.measurements import (
    EstimatedState,
    GpsReading,
    ImuReading,
    Sample,
    readonly_array,
)
from .ekf import GpsImuEkf, covariance
from ...sensors.geodesy import gps_to_enu
from ...sensors.streams import Channel, Publisher

StreamReference: TypeAlias = tuple[str, int]
StreamIdentity: TypeAlias = tuple[str, int, int]
ImuValues: TypeAlias = tuple[NDArray[np.float64], NDArray[np.float64]]
SensorReading = TypeVar("SensorReading", ImuReading, GpsReading)


class GpsImuEkfNode:
    """Sensor-stream adapter. Numerical EKF has no knowledge of channels/transport.

    IMU must be at the body origin; rotated mounts are supported explicitly.
    By default, GPS corrections must coincide with the current IMU state time.
    With allow_intermediate_gps, a newer GPS timestamp advances prediction
    using the previous IMU sample. Other timing mismatches are counted/rejected;
    timestamps are never changed. There is no rewind/repropagation.
    """

    def __init__(
        self,
        ekf: GpsImuEkf,
        *,
        gps_reference: GpsReading,
        gps_covariance_m2: ArrayLike,
        imu_position_body_m: ArrayLike = (0, 0, 0),
        imu_quaternion_body_xyzw: ArrayLike = (0, 0, 0, 1),
        gps_position_body_m: ArrayLike = (0, 0, 0),
        epoch: int = 0,
        allow_intermediate_gps: bool = False,
    ) -> None:
        if np.any(readonly_array(imu_position_body_m, (3,)) != 0):
            raise ValueError(
                "Example requires a body-origin IMU; lever-arm acceleration correction is not implemented"
            )
        q = np.asarray(imu_quaternion_body_xyzw, dtype=float)
        if (
            q.shape != (4,)
            or not np.isfinite(q).all()
            or not np.isclose(np.linalg.norm(q), 1)
        ):
            raise ValueError("IMU mount quaternion must be normalized")
        if not isinstance(gps_reference, GpsReading):
            raise TypeError("Explicit WGS84 GPS reference required")
        self.ekf, self.reference = ekf, gps_reference
        self.gps_covariance = covariance(gps_covariance_m2, 3)
        self.gps_position = readonly_array(gps_position_body_m, (3,))
        self.imu_rotation = Rotation.from_quat(q)
        self._publisher: Publisher[EstimatedState] = Publisher(
            "algorithms/gps_imu_ekf/state",
            EstimatedState,
            frame_id="local_enu",
            epoch=epoch,
        )
        self.estimates: Channel[EstimatedState] = self._publisher.channel
        self.outputs: Mapping[str, Channel[EstimatedState]] = MappingProxyType(
            {"state": self.estimates}
        )
        self._lock = threading.RLock()
        self._epoch = epoch
        self.allow_intermediate_gps = allow_intermediate_gps
        # A non-None state time implies that a previous IMU sample exists.
        self._time: int | None = None
        self._previous_imu: ImuValues | None = None
        self._previous_imu_ref: StreamReference | None = None
        self._output_ref: StreamReference | None = None
        self._last: dict[type[ImuReading] | type[GpsReading], StreamIdentity] = {}
        self.rejected_gps_timing = 0
        self._closed = False

    def _check_sample(
        self,
        sample: Sample[SensorReading],
        payload_type: type[SensorReading],
        frame: str,
    ) -> StreamIdentity:
        if self._closed:
            raise RuntimeError("EKF node closed")
        if (
            sample.epoch != self._epoch
            or sample.frame_id != frame
            or not isinstance(sample.payload, payload_type)
        ):
            raise ValueError("Input payload, epoch, or frame mismatch")
        previous = self._last.get(payload_type)
        if previous is not None:
            stream, sequence, timestamp = previous
            if (
                sample.stream_id != stream
                or sample.sequence != sequence + 1
                or sample.sim_time_ns <= timestamp
            ):
                raise ValueError(
                    "Input stream changed, lost samples, or reversed/repeated time"
                )
        return (sample.stream_id, sample.sequence, sample.sim_time_ns)

    def on_imu(self, sample: Sample[ImuReading]) -> Sample[EstimatedState]:
        with self._lock:
            identity = self._check_sample(sample, ImuReading, "drone/imu")
            previous = self._last.get(ImuReading)
            if previous is not None and sample.sim_time_ns - previous[2] > 100_000_000:
                raise ValueError("IMU interval exceeds 0.1 seconds")
            current = (
                self.imu_rotation.apply(sample.payload.specific_force_mps2),
                self.imu_rotation.apply(sample.payload.angular_rate_rad_s),
            )
            if self._time is not None:
                dt = (sample.sim_time_ns - self._time) * 1e-9
                if dt > 0:
                    self.ekf.predict(*cast(ImuValues, self._previous_imu), dt)
                elif dt < 0:
                    raise ValueError("IMU predates current estimator time")
            previous_ref = self._previous_imu_ref
            self._time = sample.sim_time_ns
            self._previous_imu = current
            self._previous_imu_ref = (sample.stream_id, sample.sequence)
            self._last[ImuReading] = identity
            return self._publish(
                sample, extra=(() if previous_ref is None else (previous_ref,))
            )

    def on_gps(self, sample: Sample[GpsReading]) -> bool:
        with self._lock:
            identity = self._check_sample(sample, GpsReading, "wgs84")
            self._last[GpsReading] = identity
            if (
                self.allow_intermediate_gps
                and self._time is not None
                and sample.sim_time_ns > self._time
            ):
                self.ekf.predict(
                    *cast(ImuValues, self._previous_imu),
                    (sample.sim_time_ns - self._time) * 1e-9,
                )
                self._time = sample.sim_time_ns
            if self._time is None or sample.sim_time_ns != self._time:
                self.rejected_gps_timing += 1
                return False
            accepted = self.ekf.correct_gps(
                gps_to_enu(sample.payload, self.reference),
                self.gps_covariance,
                receiver_position_body_m=self.gps_position,
            )
            if accepted:
                self._publish(
                    sample,
                    extra=(
                        ()
                        if self._previous_imu_ref is None
                        else (self._previous_imu_ref,)
                    ),
                )
            return accepted

    def _publish(
        self,
        parent: Sample[ImuReading] | Sample[GpsReading],
        extra: tuple[StreamReference, ...] = (),
    ) -> Sample[EstimatedState]:
        parents = ((parent.stream_id, parent.sequence),) + extra
        if self._output_ref is not None:
            parents += (self._output_ref,)
        result = self._publisher.publish(
            self.ekf.state(),
            sim_time_ns=cast(int, self._time),
            parents=tuple(dict.fromkeys(parents)),
        )
        self._output_ref = (result.stream_id, result.sequence)
        return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._publisher.close()
