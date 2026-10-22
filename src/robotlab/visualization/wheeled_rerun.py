"""Rerun recording for wheeled robots: live wheels, surfaces, and driving plots.

Extends ``GazeboRerunRecorder`` (world geometry, sensors, estimates) with

- the body box at its collision offset and the surface zones of the scene
  (flat coloured patches; strips as wide lines);
- the wheels as cylinders under the vehicle entity, at their live hub
  position and steering angle (Gazebo's wheel visuals do not move);
- time series per wheel (``wheels/<name>/…``: speed, slip ratio, slip
  angle, normal load, longitudinal and lateral force), the vehicle
  (``vehicle/…``: speed, yaw rate, lateral acceleration), the driver input
  (``driver/…``), and the accepted commands (``commands/…``).

Wheel geometry and plots are logged every ``decimation`` odometry periods
to keep the recording small at 500 Hz.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.wheeled import WheelCommandReceipt
from ..core.wheeled_vehicle import WheeledProperties
from ..simulation.gazebo.world import SURFACE_COLORS
from ..simulation.wheeled.ground import surfaces_from_scene
from .gazebo_rerun import GazeboRerunRecorder

WHEELED_TABS = (
    ("vehicle", "Vehicle"),
    ("wheels", "Wheels"),
    ("driver", "Driver input"),
    ("commands", "Wheel commands"),
    ("implement", "Implement"),
    ("estimation", "Estimate vs truth"),
    ("imu", "IMU"),
    ("gps", "GPS"),
    ("magnetometer", "Magnetometer"),
    ("status", "Sensor status"),
)
KIND_COLORS = {"rover": (220, 140, 30), "car": (205, 30, 30), "tractor": (40, 140, 65)}
TYRE_COLOR = (25, 25, 25)
OPPONENT_COLOR = (30, 80, 215)
TRAILER_COLOR = (140, 100, 50)


def _rgba(text: str) -> list[int]:
    return [round(255 * float(v)) for v in text.split()]


class WheeledRerunRecorder(GazeboRerunRecorder):
    """Rerun recording of one wheeled-robot run (see module docs)."""

    def __init__(
        self,
        world: Any,
        output: Any,
        headless: bool,
        properties: WheeledProperties,
        scene: Mapping[str, Any],
        decimation: int = 5,
    ) -> None:
        super().__init__(
            world,
            output,
            None,
            headless,
            body_size=properties.collision_size_m,
            vehicle_model=properties.model_name,
            body_color=KIND_COLORS[properties.kind],
            time_series_tabs=WHEELED_TABS,
            title="World and vehicle",
        )
        self.wheeled = properties
        self.decimation = decimation
        self._ticks = 0
        self._odometry = 0
        self._opponent_ticks: dict[str, int] = {}
        self._trailer_logged = False
        rr = self.rr
        rr.log(
            f"world/{self.vehicle}/body",
            rr.Boxes3D(
                half_sizes=[np.asarray(properties.collision_size_m) / 2],
                centers=[properties.collision_offset_m],
                colors=list(KIND_COLORS[properties.kind]),
            ),
            static=True,
        )
        self._log_surfaces(scene)

    def _log_surfaces(self, scene: Mapping[str, Any]) -> None:
        rr = self.rr
        _, zones = surfaces_from_scene(scene)
        for index, zone in enumerate(zones):
            color = _rgba(SURFACE_COLORS.get(zone.surface.name, ".55 .55 .5 1"))
            path = f"world/surfaces/{index}_{zone.surface.name}"
            if zone.shape == "rectangle":
                c, s = np.cos(zone.yaw_rad / 2), np.sin(zone.yaw_rad / 2)
                rr.log(
                    path,
                    rr.Boxes3D(
                        half_sizes=[[zone.size_m[0] / 2, zone.size_m[1] / 2, 0.005]],
                        centers=[[*zone.center_m, 0.005]],
                        quaternions=[[0.0, 0.0, s, c]],
                        colors=[color],
                        fill_mode="solid",
                    ),
                    static=True,
                )
                continue
            points = list(zone.points_m) + ([zone.points_m[0]] if zone.closed else [])
            rr.log(
                path,
                rr.LineStrips3D(
                    [[[x, y, 0.005] for x, y in points]],
                    radii=zone.width_m / 2,
                    colors=[color],
                ),
                static=True,
            )

    def log(self, kind, message, t):
        """As the base recorder, with odometry (500 Hz) decimated."""
        if kind == "odometry":
            self._odometry += 1
            if self._odometry % self.decimation != 1 and self.decimation > 1:
                self.counts[kind] += 1
                return
        super().log(kind, message, t)

    def log_wheels(self, sim_time_s: float, telemetry: Any, receipt: Any) -> None:
        """Wheel geometry and per-wheel series (every ``decimation`` calls)."""
        self._ticks += 1
        if self._ticks % self.decimation:
            return
        self._set_time(sim_time_s)
        self._log_wheel_geometry(telemetry)
        for index, name in enumerate(self.wheeled.wheel_names):
            path = f"wheels/{name}"
            self._scalar(
                f"{path}/speed_rad_s", float(telemetry.wheel_speed_rad_s[index])
            )
            self._scalar(f"{path}/slip_ratio", float(telemetry.slip_ratio[index]))
            self._scalar(
                f"{path}/slip_angle_deg",
                float(np.degrees(telemetry.slip_angle_rad[index])),
            )
            self._scalar(f"{path}/normal_load_n", float(telemetry.normal_load_n[index]))
            self._scalar(
                f"{path}/longitudinal_n", float(telemetry.longitudinal_n[index])
            )
            self._scalar(f"{path}/lateral_n", float(telemetry.lateral_n[index]))
            self._scalar(
                f"commands/drive_torque_nm/{name}",
                float(telemetry.drive_torque_nm[index]),
            )
            self._scalar(
                f"commands/steer_deg/{name}",
                float(np.degrees(telemetry.steer_rad[index])),
            )
        if receipt is not None:
            for index, name in enumerate(self.wheeled.wheel_names):
                self._scalar(
                    f"commands/brake_nm/{name}", receipt.accepted.brake_torque_nm[index]
                )

    def _log_wheel_geometry(self, telemetry: Any) -> None:
        """Cylinders at the live hub (body axes) turned by the steering angle."""
        rr = self.rr
        p = self.wheeled
        hubs = p.hub_positions_m()
        hubs[:, 2] += np.clip(telemetry.compression_m, 0.0, p.suspension.travel_m)
        quaternions = []
        for steer in telemetry.steer_rad:
            # Cylinder axis is z: roll 90° to the wheel's y axis, then yaw.
            yaw = _quat_z(float(steer))
            quaternions.append(
                _quat_multiply(yaw, [np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)])
            )
        rr.log(
            f"world/{self.vehicle}/wheels",
            rr.Cylinders3D(
                lengths=[p.wheel_width_m] * p.wheel_count,
                radii=[p.tyre.radius_m] * p.wheel_count,
                centers=hubs,
                quaternions=quaternions,
                colors=[TYRE_COLOR] * p.wheel_count,
            ),
        )

    def log_vehicle(
        self,
        sim_time_s: float,
        forward_mps: float,
        yaw_rate_rad_s: float,
        lateral_mps2: float,
    ) -> None:
        if self._ticks % self.decimation:
            return
        self._set_time(sim_time_s)
        self._scalar("vehicle/speed_mps", forward_mps)
        self._scalar("vehicle/speed_kmh", forward_mps * 3.6)
        self._scalar("vehicle/yaw_rate_deg_s", float(np.degrees(yaw_rate_rad_s)))
        self._scalar("vehicle/lateral_accel_mps2", lateral_mps2)

    def log_opponent(self, state: Any) -> None:
        """Opponent body at its pose (every ``decimation`` periods per car)."""
        count = self._opponent_ticks.get(state.name, 0) + 1
        self._opponent_ticks[state.name] = count
        if count % self.decimation != 1 and self.decimation > 1:
            return
        rr = self.rr
        self._set_time(state.sim_time_ns * 1e-9)
        path = f"world/opponents/{state.name}"
        if count == 1:
            rr.log(
                f"{path}/body",
                rr.Boxes3D(
                    half_sizes=[np.asarray(self.wheeled.collision_size_m) / 2],
                    centers=[self.wheeled.collision_offset_m],
                    colors=[OPPONENT_COLOR],
                    labels=[state.name],
                ),
                static=True,
            )
        rr.log(
            path,
            rr.Transform3D(
                translation=state.position_m, quaternion=state.quaternion_xyzw
            ),
        )

    def log_trailer(self, sim_time_s: float, trailer: Any) -> None:
        """Trailer body at its computed pose and the hitch angle (decimated)."""
        if self._ticks % self.decimation:
            return
        rr = self.rr
        self._set_time(sim_time_s)
        state = trailer.state
        if not self._trailer_logged:
            self._trailer_logged = True
            rr.log(
                "world/trailer/body",
                rr.Boxes3D(
                    half_sizes=[np.asarray(trailer.properties.collision_size_m) / 2],
                    colors=[TRAILER_COLOR],
                ),
                static=True,
            )
        rr.log(
            "world/trailer",
            rr.Transform3D(
                translation=state.position_m, quaternion=state.quaternion_xyzw
            ),
        )
        self._scalar("vehicle/trailer_yaw_deg", float(np.degrees(trailer.yaw)))

    def log_implement(self, sim_time_s: float, telemetry: Any) -> None:
        """Implement lift, depth, and draft (at the wheel-log rate)."""
        self._set_time(sim_time_s)
        self._scalar("implement/lift_m", telemetry.lift_m)
        self._scalar("implement/depth_m", telemetry.depth_m)
        self._scalar("implement/draft_n", telemetry.draft_n)
        self._scalar("implement/lowered", float(telemetry.lowered))

    def log_track(self, sim_time_s: float, laps: Any) -> None:
        """Lap progress (m) and completed laps on the scene track."""
        self._set_time(sim_time_s)
        self._scalar("vehicle/track_progress_m", laps.progress_m)
        self._scalar("vehicle/laps_completed", float(len(laps.lap_times_s)))

    def log_driver(self, sim_time_s: float, command: Any) -> None:
        """Driver input (throttle, brake, steer) at the control rate."""
        self._set_time(sim_time_s)
        self._scalar("driver/throttle", command.throttle)
        self._scalar("driver/brake", command.brake)
        self._scalar("driver/steer", command.steer)

    def log_receipt(self, receipt: WheelCommandReceipt) -> None:
        """Mark commands the actuators limited (count of limited wheels)."""
        self._set_time(receipt.sim_time_ns * 1e-9)
        for channel, wheels in receipt.limited.items():
            self._scalar(f"commands/limited/{channel}", float(len(wheels)))


def _quat_z(angle_rad: float) -> list[float]:
    return [0.0, 0.0, float(np.sin(angle_rad / 2)), float(np.cos(angle_rad / 2))]


def _quat_multiply(a: list[float] | NDArray, b: list[float] | NDArray) -> list[float]:
    """Hamilton product a ⊗ b of xyzw quaternions (apply b, then a)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
