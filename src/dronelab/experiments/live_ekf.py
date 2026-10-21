"""Live GPS/IMU composition with bounded timestamp reordering and no truth input."""

from collections import Counter
import heapq
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from ..core.measurements import GpsReading, ImuReading
from ..estimation.gps_imu_ekf import GpsImuEkfNode
from ..estimation.gps_imu_ekf.plugin import make_ekf
from ..sensors.streams import Publisher
from ..simulation.gazebo.messages import timestamp_ns, vector


def reference_from_world(world):
    coordinates = ET.parse(world).find("./world/spherical_coordinates")
    if (
        coordinates is None
        or coordinates.findtext("surface_model") != "EARTH_WGS84"
        or coordinates.findtext("world_frame_orientation") != "ENU"
        or float(coordinates.findtext("heading_deg", "0")) != 0
    ):
        raise ValueError("Live EKF requires a WGS84, ENU world with zero heading")
    return GpsReading(
        float(coordinates.findtext("latitude_deg")),
        float(coordinates.findtext("longitude_deg")),
        float(coordinates.findtext("elevation")),
    )


def validate_live_settings(settings):
    sensors = settings["sensors"]
    if not sensors["imu"]["enabled"] or not sensors["gps"]["enabled"]:
        raise ValueError("--ekf requires enabled IMU and GPS sensors")
    if any(sensors["imu"]["mount"]["position_m"]):
        raise ValueError(
            "--ekf requires a body-origin IMU; rotated IMU mounts are supported"
        )
    if sensors["imu"]["rate_hz"] < 20:
        raise ValueError("--ekf requires IMU rate >= 20 Hz")


class LiveEkf:
    """Sensor-only adapter, driven on the mission thread, never a transport callback.

    Each source must arrive in order. Samples from different sources may arrive
    in any interleaving. Dispatch waits for both source watermarks, then orders
    by time with IMU before GPS at equal time. Lag/fill/timeouts fail visibly.
    """

    def __init__(self, settings, reference, *, monotonic=time.monotonic):
        validate_live_settings(settings)
        self.settings = settings["ekf"]
        self._now = monotonic
        imu, gps = settings["sensors"]["imu"], settings["sensors"]["gps"]
        ekf = make_ekf(self.settings)
        self.node = GpsImuEkfNode(
            ekf,
            gps_reference=reference,
            gps_covariance_m2=np.diag(np.asarray(self.settings["gps_std_m"]) ** 2),
            imu_quaternion_body_xyzw=Rotation.from_euler(
                "xyz", imu["mount"]["rotation_deg"], degrees=True
            ).as_quat(),
            gps_position_body_m=gps["mount"]["position_m"],
            allow_intermediate_gps=True,
        )
        self.estimates = self.node.estimates
        self.sensors = {
            "imu": Publisher("sensors/imu", ImuReading, frame_id="drone/imu"),
            "gps": Publisher("sensors/gps", GpsReading, frame_id="wgs84"),
        }
        self._inputs = {
            name: producer.channel.subscribe(capacity=1)
            for name, producer in self.sensors.items()
        }
        self._pending = []
        self._latest = {}
        self._received = {}
        self._started = self._now()
        self._closed = False
        self.stats = Counter()

    def ingest(self, kind, message, *, received_at=None, gps_dropped=False):
        """Accept raw arrival timing; a modeled lost GPS fix supplies no payload.

        Dropped-fix timing advances the reorder watermark and transport health
        only. The estimator continues IMU prediction without GPS correction.
        """
        if self._closed:
            raise RuntimeError("Live EKF closed")
        if kind not in self.sensors:
            return
        stamp = timestamp_ns(message)
        if kind in self._latest and stamp <= self._latest[kind]:
            raise RuntimeError(f"Non-increasing live {kind} timestamps")
        if kind == "gps" and gps_dropped:
            sample = None
        elif kind == "imu":
            payload = ImuReading(
                vector(message.linear_acceleration), vector(message.angular_velocity)
            )
        else:
            payload = GpsReading(
                message.latitude_deg, message.longitude_deg, message.altitude
            )
        if not (kind == "gps" and gps_dropped):
            self.sensors[kind].publish(payload, sim_time_ns=stamp)
            sample = self._inputs[kind].next(timeout_wall_s=0)
        self._latest[kind] = stamp
        self._received[kind] = self._now() if received_at is None else received_at
        if sample is not None:
            heapq.heappush(
                self._pending,
                (stamp, 0 if kind == "imu" else 1, sample.sequence, kind, sample),
            )
        else:
            self.stats["gps_modeled_drops"] += 1
        self.stats[f"{kind}_received"] += 1
        self.stats["queue_high_water"] = max(
            self.stats["queue_high_water"], len(self._pending)
        )
        if len(self._pending) > self.settings["reorder_capacity"]:
            raise RuntimeError("EKF reorder buffer overflow")
        if len(self._latest) == 2:
            watermark = min(self._latest.values())
            if (
                max(self._latest.values()) - watermark
                > self.settings["max_sensor_lag_sim_s"] * 1e9
            ):
                raise RuntimeError("EKF sensor simulation-time lag exceeded")
            self._dispatch(watermark)
        self.check_health()

    def _dispatch(self, watermark):
        while self._pending and self._pending[0][0] <= watermark:
            _, _, _, kind, sample = heapq.heappop(self._pending)
            if kind == "imu":
                self.node.on_imu(sample)
                self.stats["imu_dispatched"] += 1
            else:
                self.node.on_gps(sample)
                self.stats["gps_dispatched"] += 1

    def check_health(self):
        now = self._now()
        for kind in ("imu", "gps"):
            if (
                now - self._received.get(kind, self._started)
                > self.settings["sensor_watchdog_wall_s"]
            ):
                raise RuntimeError(f"Live EKF {kind} watchdog expired")

    def summary(self):
        return dict(
            self.stats,
            gps_updates=self.node.ekf.accepted_gps,
            gps_outliers=self.node.ekf.rejected_gps,
            rejected_gps_timing=self.node.rejected_gps_timing,
            pending_at_close=len(self._pending),
            initialization="declared ekf priors; no simulator truth",
            control_feedback="simulator ground truth",
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        # Do not extrapolate past the last common source watermark on shutdown.
        for subscription in self._inputs.values():
            subscription.close()
        for producer in self.sensors.values():
            producer.close()
        self.node.close()
