"""Rerun visualization adapter for Gazebo messages; no flight-control logic."""

from collections import Counter
from collections.abc import Callable
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from ..simulation.gazebo.messages import vector
from ..sensors.barometer import pressure_to_altitude
from ..sensors.gazebo import lidar_points, rangefinder_distance, rgb_image
from ..core.vehicle import DEFAULT_DRONE


def _display_path(positions, max_vertices=2000):
    """Bound viewer serialization work while retaining the start and latest point.

    Full-resolution positions remain in the recorder and telemetry files.
    """
    stride = max(1, (len(positions) - 1 + max_vertices - 2) // (max_vertices - 1))
    vertices = positions[::stride]
    if positions and (len(positions) - 1) % stride:
        vertices.append(positions[-1])
    return vertices


# Scalar series reach Rerun in batches spanning this much simulated time.
SCALAR_BATCH_PERIOD_SIM_S = 0.1


class ScalarBatch:
    """Buffer scalar samples per entity path and send them in columns.

    One ``rr.log`` per scalar cost ~30 µs (archetype construction); a
    250 Hz flight logs ~30 scalars per odometry period, enough to push the
    real-time mission loop behind Gazebo. Sending each path's buffered
    samples as one column keeps every sample and timestamp at a fraction of
    that cost; the viewer sees them up to ``period_sim_s`` later.

    ``send(path, times_s, values)`` receives samples in arrival order.
    A batch is flushed once a sample arrives ``period_sim_s`` or more after
    the batch's first sample, and by ``flush()`` (the recorder's ``close``).
    """

    def __init__(
        self,
        send: Callable[[str, list[float], list[float]], None],
        period_sim_s: float = SCALAR_BATCH_PERIOD_SIM_S,
    ) -> None:
        if not period_sim_s >= 0:
            raise ValueError("period_sim_s must be non-negative")
        self._send = send
        self._period_sim_s = period_sim_s
        self._series: dict[str, tuple[list[float], list[float]]] = {}
        self._batch_start_s: float | None = None

    def add(self, path: str, time_s: float, value: float) -> None:
        """Buffer one sample of ``path`` at simulation time ``time_s``."""
        times, values = self._series.setdefault(path, ([], []))
        times.append(time_s)
        values.append(float(value))
        if self._batch_start_s is None:
            self._batch_start_s = time_s
        elif time_s - self._batch_start_s >= self._period_sim_s:
            self.flush()

    def flush(self) -> None:
        """Send every buffered series and start a new batch."""
        series, self._series = self._series, {}
        self._batch_start_s = None
        for path, (times, values) in series.items():
            self._send(path, times, values)


class GazeboRerunRecorder:
    def __init__(
        self,
        world,
        output,
        target,
        headless,
        *,
        properties=DEFAULT_DRONE,
        body_size=(0.48, 0.48, 0.12),
        optical_flow_mount=None,
    ):
        import rerun as rr

        self.rr = rr
        self.properties = properties
        self.output = output
        self.target = target
        self.positions = []
        self.estimate_positions = []
        self.truth_positions = {}
        self.counts = Counter()
        self._time_s = 0.0
        self._scalars = ScalarBatch(self._send_scalars)
        self._obstacle_poses: dict[str, tuple[float, ...]] = {}
        rr.init("dronelab_gazebo")
        sinks = [rr.FileSink(str(output / "flight.rrd"))]
        if not headless:
            rr.spawn(connect=False)
            sinks.append(
                rr.GrpcSink("rerun+http://127.0.0.1:9876/proxy", flush_timeout_sec=2)
            )
        import rerun.blueprint as blueprint

        layout = blueprint.Blueprint(
            blueprint.Horizontal(
                # The ground is the view's line grid, not logged geometry:
                # Rerun frames the initial camera on the scene's bounds, and a
                # 100 m ground box made it start zoomed far out.
                blueprint.Spatial3DView(
                    origin="world",
                    name="World and drone",
                    line_grid=blueprint.LineGrid3D(
                        visible=True, spacing=1.0, color=[100, 120, 100, 160]
                    ),
                ),
                blueprint.Vertical(
                    blueprint.Spatial2DView(origin="camera", name="Drone camera"),
                    blueprint.Tabs(
                        blueprint.TimeSeriesView(
                            origin="estimation", name="EKF vs truth"
                        ),
                        blueprint.TimeSeriesView(
                            origin="actuators/thrust_target_n",
                            name="Rotor thrust targets (N)",
                        ),
                        blueprint.TimeSeriesView(origin="imu", name="IMU"),
                        blueprint.TimeSeriesView(origin="gps", name="GPS"),
                        blueprint.TimeSeriesView(origin="baro", name="Barometer"),
                        blueprint.TimeSeriesView(
                            origin="magnetometer", name="Magnetometer"
                        ),
                        blueprint.TimeSeriesView(
                            origin="downward", name="Rangefinder and flow"
                        ),
                        blueprint.TimeSeriesView(origin="battery", name="Battery"),
                        blueprint.TimeSeriesView(origin="status", name="Sensor status"),
                        blueprint.TimeSeriesView(origin="aero", name="Aerodynamics"),
                        blueprint.TimeSeriesView(
                            origin="tracking", name="Waypoint error"
                        ),
                        blueprint.TextLogView(origin="events", name="Contacts"),
                    ),
                ),
            )
        )
        rr.set_sinks(*sinks, default_blueprint=layout)
        # A default alone does not replace a layout cached by the viewer.
        rr.send_blueprint(layout, make_active=True, make_default=True)
        self._set_time(0.0)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        if target is not None:
            rr.log(
                "world/target",
                rr.Points3D([target], radii=0.12, colors=[255, 160, 0]),
                static=True,
            )
        rr.log(
            "world/drone/body",
            rr.Boxes3D(half_sizes=[np.asarray(body_size) / 2], colors=[40, 100, 220]),
            static=True,
        )
        # Child geometry inherits each vehicle's body-to-world pose, so +X
        # indicates its forward direction even while it pitches or rolls.
        for entity, color in (
            ("drone", [40, 100, 220]),
            ("ekf", [230, 80, 180]),
        ):
            rr.log(
                f"world/{entity}/heading",
                rr.Arrows3D(
                    origins=[[0, 0, body_size[2] / 2 + 0.08]],
                    vectors=[[max(body_size[:2]) * 1.8, 0, 0]],
                    radii=0.025,
                    colors=color,
                ),
                static=True,
            )
        # Read geometry from the actual SDF, including saved edits. World pose
        # messages update these entities if they are moved while the run is live.
        sdf = ET.parse(world)
        self.lidar_pose = [0, 0, 0.2, 0, 0, 0]
        anchor = sdf.findtext('./world/model[@name="drone"]/link[@name="tether"]/pose')
        self.tether_anchor = (
            None if anchor is None else list(map(float, anchor.split()[:3]))
        )
        for sensor in sdf.findall(
            './world/model[@name="drone"]/link[@name="base_link"]/sensor'
        ):
            name = sensor.get("name")
            if name not in {
                "imu",
                "gps",
                "baro",
                "magnetometer",
                "rangefinder",
                "camera",
                "lidar",
            }:
                continue
            pose = list(map(float, sensor.findtext("pose", "0 0 0 0 0 0").split()))
            if name == "lidar":
                self.lidar_pose = pose
            self._log_sensor_marker(name, pose)
        if optical_flow_mount is not None:
            self._log_sensor_marker(
                "optical_flow",
                [
                    *optical_flow_mount["position_m"],
                    *np.radians(optical_flow_mount["rotation_deg"]),
                ],
            )
        obstacle_names = set()
        for model in sdf.findall("./world/model"):
            name = model.get("name")
            if name in {"drone", "ground"}:
                continue
            pose = list(map(float, model.findtext("pose", "0 0 0 0 0 0").split()))
            obstacle_names.add(name)
            rr.log(
                f"world/obstacles/{name}",
                rr.Transform3D(
                    translation=pose[:3],
                    quaternion=Rotation.from_euler("xyz", pose[3:]).as_quat(),
                ),
            )
            for index, visual in enumerate(model.findall("./link/visual")):
                mesh_uri = visual.findtext("./geometry/mesh/uri")
                if mesh_uri and mesh_uri.endswith(".obj"):
                    self._log_obj_mesh(f"world/obstacles/{name}/mesh{index}", mesh_uri)
                    continue
                size = visual.findtext("./geometry/box/size")
                if size:
                    local_pose = list(
                        map(float, visual.findtext("pose", "0 0 0 0 0 0").split())
                    )
                    rr.log(
                        f"world/obstacles/{name}/box{index}",
                        rr.Boxes3D(
                            half_sizes=[np.array(list(map(float, size.split()))) / 2],
                            centers=[local_pose[:3]],
                            colors=[170, 115, 75],
                        ),
                        static=True,
                    )
        self.obstacle_names = obstacle_names

    def _set_time(self, t: float) -> None:
        """Set the ``sim_time`` (s) of following logs, scalars included."""
        self._time_s = t
        self.rr.set_time("sim_time", duration=t)

    def _scalar(self, path: str, value: float) -> None:
        """One scalar sample at the current ``sim_time`` (batched)."""
        self._scalars.add(path, self._time_s, value)

    def _send_scalars(self, path: str, times_s: list[float], values: list[float]):
        rr = self.rr
        rr.send_columns(
            path,
            indexes=[rr.TimeColumn("sim_time", duration=times_s)],
            columns=rr.Scalars.columns(scalars=values),
        )

    def _log_obj_mesh(self, path, uri):
        """Show a generated OBJ mesh (e.g. terrain) from its file:// URI."""
        from urllib.parse import unquote, urlparse

        vertices, faces = [], []
        for line in open(unquote(urlparse(uri).path)):
            kind, *values = line.split()
            if kind == "v":
                vertices.append(list(map(float, values)))
            elif kind == "f":
                # Accept "v", "v/vt", and "v//vn" face entries.
                faces.append([int(v.split("/")[0]) - 1 for v in values])
        self.rr.log(
            path,
            self.rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
                vertex_colors=[[105, 125, 90]] * len(vertices),
            ),
            static=True,
        )

    # Marker colours and whether the sensor has a pointing direction (+x).
    SENSOR_STYLE = {
        "imu": ([255, 200, 0], False),
        "gps": ([0, 200, 255], False),
        "baro": ([150, 150, 255], False),
        "magnetometer": ([255, 80, 80], False),
        "rangefinder": ([0, 220, 120], True),
        "optical_flow": ([220, 120, 255], True),
        "camera": ([255, 255, 255], True),
        "lidar": ([255, 140, 0], False),
    }

    def _log_sensor_marker(self, name, pose):
        """Marker at a sensor mount; an arrow shows the sensor +x axis."""
        rr = self.rr
        color, directional = self.SENSOR_STYLE[name]
        path = f"world/drone/sensors/{name}"
        rr.log(
            path,
            rr.Transform3D(
                translation=pose[:3],
                quaternion=Rotation.from_euler("xyz", pose[3:]).as_quat(),
            ),
            static=True,
        )
        rr.log(
            f"{path}/marker",
            rr.Points3D([[0, 0, 0]], radii=0.02, colors=[color]),
            static=True,
        )
        if directional:
            rr.log(
                f"{path}/axis",
                rr.Arrows3D(
                    origins=[[0, 0, 0]], vectors=[[0.15, 0, 0]], colors=[color]
                ),
                static=True,
            )

    def add_obstacle(self, name, size):
        """Show an obstacle spawned at run time; its poses then update it."""
        self.obstacle_names.add(name)
        self.rr.log(
            f"world/obstacles/{name}/box0",
            self.rr.Boxes3D(half_sizes=[np.asarray(size) / 2], colors=[190, 60, 50]),
            static=True,
        )

    def remove_obstacle(self, name):
        self.obstacle_names.discard(name)
        self._obstacle_poses.pop(name, None)
        self.rr.log(f"world/obstacles/{name}", self.rr.Clear(recursive=True))

    def log_lidar_points(self, points_body, t, reflectivity=None):
        """Log processed lidar points (body axes) under the moving drone.

        With ``reflectivity`` the points are coloured dark (weak) to bright
        (strong, 1 = white Lambertian at no loss), like intensity views of
        real lidar drivers.
        """
        rr = self.rr
        self.counts["lidar"] += 1
        self._set_time(t)
        colors = None
        if reflectivity is not None and len(reflectivity) == len(points_body):
            level = np.clip(np.asarray(reflectivity, float), 0.0, 1.0)[:, None]
            colors = (40 + level * np.array([215, 175, 15])).astype(np.uint8)
        rr.log(
            "world/drone/lidar", rr.Points3D(points_body, radii=0.025, colors=colors)
        )

    def log_camera_image(self, image, t):
        """Log a processed camera frame at its delivery time (see camera_effects)."""
        rr = self.rr
        self.counts["camera"] += 1
        self._set_time(t)
        rr.log("camera/rgb", rr.Image(image))
        if self.counts["camera"] == 1:
            from PIL import Image as PillowImage

            PillowImage.fromarray(image).save(self.output / "camera-first.png")

    def log_effect(self, event):
        """Effect switch from keyboard flight: status series plus an event line."""
        rr = self.rr
        self._set_time(event["sim_time_ns"] * 1e-9)
        self._scalar(f"status/effect_{event['effect']}", float(event["on"]))
        if "soc" in event:  # one-shot battery action
            text = f"{event['effect']}: charge {event['soc']:.0%}"
        else:
            text = f"{event['effect']} {'ON' if event['on'] else 'off'}"
        rr.log("events/effects", rr.TextLog(text))

    def log_vector(self, path, values, labels=("x", "y", "z")):
        """Log each component of a vector as its own time series."""
        for label, value in zip(labels, values):
            self._scalar(f"{path}/{label}", float(value))

    FAILSAFE_LEVELS = {
        "normal": 0,
        "warned": 1,
        "returning": 2,
        "landing": 3,
        "landed": 4,
    }

    def log_failsafe(self, event):
        """Failsafe state change: status level plus an event line."""
        rr = self.rr
        self._scalar("status/failsafe_state", self.FAILSAFE_LEVELS[event["state"]])
        rr.log(
            "events/failsafe",
            rr.TextLog(
                f"{event['state']} at {event['remaining_pct']:.0f}% / {event['voltage_v']:.2f} V"
            ),
        )

    def log_status(self, name, t, value):
        """Sensor health indicator time series (see experiments/gazebo.py)."""
        self._set_time(t)
        self._scalar(f"status/{name}", float(value))

    def log_battery_monitor(self, reading):
        """Measured battery values from the power-module model."""
        self._set_time(reading.sim_time_ns * 1e-9)
        self._scalar("battery/measured/voltage_v", reading.voltage_v)
        self._scalar("battery/measured/current_a", reading.current_a)
        self._scalar("battery/measured/remaining_pct", reading.remaining_pct)

    def log_optical_flow(self, reading):
        """Optical-flow rates and distance; invalid samples log validity only."""
        self._set_time(reading.sim_time_ns * 1e-9)
        self._scalar("downward/flow_valid", float(reading.valid))
        self._scalar("status/optical_flow_valid", float(reading.valid))
        if reading.valid:
            for axis, value in zip(("y", "z"), reading.flow_rad_s):
                self._scalar(f"downward/flow_rad_s/{axis}", value)
            self._scalar("downward/flow_distance_m", reading.distance_m)

    def log_powertrain(self, battery, ground_effect_factors, thrust_available=()):
        """Battery voltage/SoC/current and per-rotor ground-effect factors.

        Logged at the current simulation time (set by the latest odometry).
        """
        if battery is not None:
            self._scalar("battery/terminal_v", battery.terminal_v)
            self._scalar("battery/soc", battery.soc)
            self._scalar("battery/current_a", battery.current_a)
            self._scalar("battery/temperature_c", battery.temperature_c)
        for index, factor in enumerate(ground_effect_factors):
            self._scalar(f"battery/ground_effect/rotor_{index}", factor)
        # Motor health: maximum thrust available vs healthy (1 = healthy).
        for index, fraction in enumerate(thrust_available):
            self._scalar(f"status/motor_{index}_thrust_available", fraction)

    def log_actuator_event(self, event):
        """Show published motor thrust targets, never inferred applied forces."""
        if event["status"] not in {"published", "expired", "stopped"}:
            return
        if event["sim_time_ns"] is None or event.get("thrust_n") is None:
            return
        self.log_thrust_targets(event["thrust_n"], event["sim_time_ns"] * 1e-9)

    def log_thrust_targets(self, thrust_n, t):
        self._set_time(t)
        for index, thrust in enumerate(thrust_n):
            self._scalar(f"actuators/thrust_target_n/rotor_{index}", thrust)

    def log(self, kind, message, t):
        rr = self.rr
        self.counts[kind] += 1
        self._set_time(t)
        if kind == "poses":
            for pose in message.pose:
                if pose.name == "hook" and self.tether_anchor is not None:
                    # Link poses are relative to the drone model: draw in its frame.
                    rr.log(
                        "world/drone/tether",
                        rr.LineStrips3D(
                            [[self.tether_anchor, vector(pose.position).tolist()]],
                            colors=[60, 60, 60],
                            radii=0.006,
                        ),
                    )
                    rr.log(
                        "world/drone/hook",
                        rr.Points3D(
                            [vector(pose.position)], radii=0.03, colors=[90, 90, 90]
                        ),
                    )
                if pose.name in self.obstacle_names:
                    q = pose.orientation
                    key = (
                        pose.position.x,
                        pose.position.y,
                        pose.position.z,
                        q.x,
                        q.y,
                        q.z,
                        q.w,
                    )
                    # Log a transform only when the obstacle moved.
                    if self._obstacle_poses.get(pose.name) != key:
                        self._obstacle_poses[pose.name] = key
                        rr.log(
                            f"world/obstacles/{pose.name}",
                            rr.Transform3D(
                                translation=key[:3], quaternion=list(key[3:])
                            ),
                        )
        elif kind == "odometry":
            p = vector(message.pose.position)
            q = message.pose.orientation
            rr.log(
                "world/drone",
                rr.Transform3D(translation=p, quaternion=[q.x, q.y, q.z, q.w]),
            )
            self.truth_positions[round(t * 1e9)] = p.copy()
            if len(self.truth_positions) > 4096:
                del self.truth_positions[next(iter(self.truth_positions))]
            self.positions.append(p)
            if len(self.positions) % 5 == 0:
                rr.log(
                    "world/path",
                    rr.LineStrips3D(
                        [_display_path(self.positions)], colors=[40, 100, 220]
                    ),
                )
            if self.target is not None:
                self._scalar("tracking/error_m", np.linalg.norm(p - self.target))
        elif kind == "imu":
            for label, values in [
                ("accel", message.linear_acceleration),
                ("gyro", message.angular_velocity),
            ]:
                for axis, value in zip("xyz", vector(values)):
                    self._scalar(f"imu/{label}/{axis}", value)
        elif kind == "gps":
            for label in ["latitude_deg", "longitude_deg", "altitude"]:
                self._scalar(f"gps/{label}", getattr(message, label))
        elif kind == "magnetometer":
            field_ut = vector(message.field_tesla) * 1e6
            for axis, value in zip("xyz", field_ut):
                self._scalar(f"magnetometer/field_ut/{axis}", value)
            # Compass yaw (ENU, from east) assuming a level sensor and
            # ignoring declination: m_x = B_n sin(yaw), m_y = B_n cos(yaw).
            yaw = np.degrees(np.arctan2(field_ut[0], field_ut[1]))
            self._scalar("magnetometer/level_yaw_deg", yaw)
        elif kind == "rangefinder":
            distance = rangefinder_distance(message)
            ray = "world/drone/sensors/rangefinder/ray"
            if np.isfinite(distance):
                self._scalar("downward/range_m", distance)
                # Ray from the sensor along its +x axis to the measured hit.
                rr.log(
                    ray,
                    rr.LineStrips3D(
                        [[[0, 0, 0], [distance, 0, 0]]], colors=[0, 220, 120]
                    ),
                )
            else:
                rr.log(ray, rr.Clear(recursive=False))
        elif kind == "baro":
            self._scalar("baro/pressure_pa", message.pressure)
            # Pressure altitude above sea level, comparable with GPS altitude.
            self._scalar("baro/altitude_m", pressure_to_altitude(message.pressure))
        elif kind == "contacts" and message.contact:
            for contact in message.contact:
                event = dict(
                    time=t, a=contact.collision1.name, b=contact.collision2.name
                )
                rr.log(
                    "events/contact", rr.TextLog(f"{event['a']} touches {event['b']}")
                )
        elif kind == "camera":
            image = rgb_image(message)
            rr.log("camera/rgb", rr.Image(image))
            if self.counts[kind] == 1:
                from PIL import Image as PillowImage

                PillowImage.fromarray(image).save(self.output / "camera-first.png")
        elif kind == "lidar":
            points = lidar_points(
                message,
                position_m=self.lidar_pose[:3],
                rotation_rad=self.lidar_pose[3:],
            )
            rr.log("world/drone/lidar", rr.Points3D(points, radii=0.025))

    def log_algorithm_debug(self, name, sim_time_ns, values, labels):
        """A plug-in's debug series under ``algorithms/<name>`` at its own time."""
        self._set_time(sim_time_ns * 1e-9)
        self.log_vector(f"algorithms/{name}", values, labels)

    def log_estimate(self, sample):
        """Observer only: truth is used for matched-time error plots, never filtering."""
        rr = self.rr
        state = sample.payload
        self._set_time(sample.sim_time_ns * 1e-9)
        rr.log(
            "world/ekf",
            rr.Transform3D(
                translation=state.position_m, quaternion=state.quaternion_xyzw
            ),
        )
        rr.log(
            "world/ekf/body",
            rr.Boxes3D(half_sizes=[[0.20, 0.20, 0.05]], colors=[230, 80, 180, 160]),
        )
        self.estimate_positions.append(state.position_m.copy())
        if len(self.estimate_positions) % 5 == 0:
            rr.log(
                "world/ekf_path",
                rr.LineStrips3D(
                    [_display_path(self.estimate_positions)], colors=[230, 80, 180]
                ),
            )
        truth = self.truth_positions.get(sample.sim_time_ns)
        for index, axis in enumerate("xyz"):
            self._scalar(f"estimation/position/{axis}", state.position_m[index])
            if truth is not None:
                self._scalar(f"estimation/truth_position/{axis}", truth[index])
        if truth is not None:
            self._scalar(
                "estimation/position_error_m",
                np.linalg.norm(state.position_m - truth),
            )
        self._scalar("estimation/speed_mps", np.linalg.norm(state.velocity_mps))

    def close(self):
        self._scalars.flush()
        self.rr.disconnect()
