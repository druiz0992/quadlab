"""Rerun visualization adapter for Gazebo messages; no flight-control logic."""

from collections import Counter
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
                blueprint.Spatial3DView(origin="world", name="World and drone"),
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
        rr.set_time("sim_time", duration=0.0)
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
        rr.log(
            "world/ground",
            rr.Boxes3D(
                centers=[[0, 0, -0.05]],
                half_sizes=[[50, 50, 0.05]],
                colors=[100, 120, 100],
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
        """Labelled marker at a sensor mount; an arrow shows the sensor +x axis."""
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
            rr.Points3D([[0, 0, 0]], radii=0.02, colors=[color], labels=[name]),
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
        self.rr.log(f"world/obstacles/{name}", self.rr.Clear(recursive=True))

    def log_lidar_points(self, points_body, t, reflectivity=None):
        """Log processed lidar points (body axes) under the moving drone.

        With ``reflectivity`` the points are coloured dark (weak) to bright
        (strong, 1 = white Lambertian at no loss), like intensity views of
        real lidar drivers.
        """
        rr = self.rr
        self.counts["lidar"] += 1
        rr.set_time("sim_time", duration=t)
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
        rr.set_time("sim_time", duration=t)
        rr.log("camera/rgb", rr.Image(image))
        if self.counts["camera"] == 1:
            from PIL import Image as PillowImage

            PillowImage.fromarray(image).save(self.output / "camera-first.png")

    def log_effect(self, event):
        """Effect switch from keyboard flight: status series plus an event line."""
        rr = self.rr
        rr.set_time("sim_time", duration=event["sim_time_ns"] * 1e-9)
        rr.log(f"status/effect_{event['effect']}", rr.Scalars(float(event["on"])))
        if "soc" in event:  # one-shot battery action
            text = f"{event['effect']}: charge {event['soc']:.0%}"
        else:
            text = f"{event['effect']} {'ON' if event['on'] else 'off'}"
        rr.log("events/effects", rr.TextLog(text))

    def log_vector(self, path, values, labels=("x", "y", "z")):
        """Log each component of a vector as its own time series."""
        rr = self.rr
        for label, value in zip(labels, values):
            rr.log(f"{path}/{label}", rr.Scalars(float(value)))

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
        rr.log(
            "status/failsafe_state", rr.Scalars(self.FAILSAFE_LEVELS[event["state"]])
        )
        rr.log(
            "events/failsafe",
            rr.TextLog(
                f"{event['state']} at {event['remaining_pct']:.0f}% / {event['voltage_v']:.2f} V"
            ),
        )

    def log_status(self, name, t, value):
        """Sensor health indicator time series (see experiments/gazebo.py)."""
        rr = self.rr
        rr.set_time("sim_time", duration=t)
        rr.log(f"status/{name}", rr.Scalars(float(value)))

    def log_battery_monitor(self, reading):
        """Measured battery values from the power-module model."""
        rr = self.rr
        rr.set_time("sim_time", duration=reading.sim_time_ns * 1e-9)
        rr.log("battery/measured/voltage_v", rr.Scalars(reading.voltage_v))
        rr.log("battery/measured/current_a", rr.Scalars(reading.current_a))
        rr.log("battery/measured/remaining_pct", rr.Scalars(reading.remaining_pct))

    def log_optical_flow(self, reading):
        """Optical-flow rates and distance; invalid samples log validity only."""
        rr = self.rr
        rr.set_time("sim_time", duration=reading.sim_time_ns * 1e-9)
        rr.log("downward/flow_valid", rr.Scalars(float(reading.valid)))
        rr.log("status/optical_flow_valid", rr.Scalars(float(reading.valid)))
        if reading.valid:
            for axis, value in zip(("y", "z"), reading.flow_rad_s):
                rr.log(f"downward/flow_rad_s/{axis}", rr.Scalars(value))
            rr.log("downward/flow_distance_m", rr.Scalars(reading.distance_m))

    def log_powertrain(self, battery, ground_effect_factors, thrust_available=()):
        """Battery voltage/SoC/current and per-rotor ground-effect factors.

        Logged at the current simulation time (set by the latest odometry).
        """
        rr = self.rr
        if battery is not None:
            rr.log("battery/terminal_v", rr.Scalars(battery.terminal_v))
            rr.log("battery/soc", rr.Scalars(battery.soc))
            rr.log("battery/current_a", rr.Scalars(battery.current_a))
            rr.log("battery/temperature_c", rr.Scalars(battery.temperature_c))
        for index, factor in enumerate(ground_effect_factors):
            rr.log(f"battery/ground_effect/rotor_{index}", rr.Scalars(factor))
        # Motor health: maximum thrust available vs healthy (1 = healthy).
        for index, fraction in enumerate(thrust_available):
            rr.log(f"status/motor_{index}_thrust_available", rr.Scalars(fraction))

    def log_actuator_event(self, event):
        """Show published motor thrust targets, never inferred applied forces."""
        if event["status"] not in {"published", "expired", "stopped"}:
            return
        if event["sim_time_ns"] is None or event.get("thrust_n") is None:
            return
        self.log_thrust_targets(event["thrust_n"], event["sim_time_ns"] * 1e-9)

    def log_thrust_targets(self, thrust_n, t):
        rr = self.rr
        rr.set_time("sim_time", duration=t)
        for index, thrust in enumerate(thrust_n):
            rr.log(f"actuators/thrust_target_n/rotor_{index}", rr.Scalars(thrust))

    def log(self, kind, message, t):
        rr = self.rr
        self.counts[kind] += 1
        rr.set_time("sim_time", duration=t)
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
                    rr.log(
                        f"world/obstacles/{pose.name}",
                        rr.Transform3D(
                            translation=vector(pose.position),
                            quaternion=[q.x, q.y, q.z, q.w],
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
                rr.log("tracking/error_m", rr.Scalars(np.linalg.norm(p - self.target)))
        elif kind == "imu":
            for label, values in [
                ("accel", message.linear_acceleration),
                ("gyro", message.angular_velocity),
            ]:
                for axis, value in zip("xyz", vector(values)):
                    rr.log(f"imu/{label}/{axis}", rr.Scalars(value))
        elif kind == "gps":
            for label in ["latitude_deg", "longitude_deg", "altitude"]:
                rr.log(f"gps/{label}", rr.Scalars(getattr(message, label)))
        elif kind == "magnetometer":
            field_ut = vector(message.field_tesla) * 1e6
            for axis, value in zip("xyz", field_ut):
                rr.log(f"magnetometer/field_ut/{axis}", rr.Scalars(value))
            # Compass yaw (ENU, from east) assuming a level sensor and
            # ignoring declination: m_x = B_n sin(yaw), m_y = B_n cos(yaw).
            yaw = np.degrees(np.arctan2(field_ut[0], field_ut[1]))
            rr.log("magnetometer/level_yaw_deg", rr.Scalars(yaw))
        elif kind == "rangefinder":
            distance = rangefinder_distance(message)
            ray = "world/drone/sensors/rangefinder/ray"
            if np.isfinite(distance):
                rr.log("downward/range_m", rr.Scalars(distance))
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
            rr.log("baro/pressure_pa", rr.Scalars(message.pressure))
            # Pressure altitude above sea level, comparable with GPS altitude.
            rr.log(
                "baro/altitude_m", rr.Scalars(pressure_to_altitude(message.pressure))
            )
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

    def log_estimate(self, sample):
        """Observer only: truth is used for matched-time error plots, never filtering."""
        rr = self.rr
        state = sample.payload
        rr.set_time("sim_time", duration=sample.sim_time_ns * 1e-9)
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
            rr.log(f"estimation/position/{axis}", rr.Scalars(state.position_m[index]))
            if truth is not None:
                rr.log(f"estimation/truth_position/{axis}", rr.Scalars(truth[index]))
        if truth is not None:
            rr.log(
                "estimation/position_error_m",
                rr.Scalars(np.linalg.norm(state.position_m - truth)),
            )
        rr.log("estimation/speed_mps", rr.Scalars(np.linalg.norm(state.velocity_mps)))

    def close(self):
        self.rr.disconnect()
