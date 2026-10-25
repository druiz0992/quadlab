"""Rerun recording of MuJoCo biped runs: the robot's bodies and a few plots.

At start, every visible geom (MuJoCo groups 0–2 with nonzero alpha, the
groups MuJoCo's own viewer shows; collision-only groups are hidden) is
logged once, static, under its body's entity ``world/robot/<body>``: meshes
from the model's vertices and faces, boxes, spheres, capsules, cylinders,
and ellipsoids as Rerun primitives; planes are left to the view's grid.
Each recorded state then moves every body (``log_state``, world pose from
MuJoCo's ``xpos``/``xquat``) and logs ``trunk/height_m``,
``trunk/tilt_deg``, ``trunk/forward_mps`` (speed along the trunk's heading),
and the walking command (``command/forward_mps``, ``left_mps``,
``yaw_rate_rad_s``) when there is one. Observes only; never decides
anything. Writes ``biped.rrd`` and, unless ``headless``, streams to a
spawned viewer. ``rerun`` (the ``gazebo`` extra) is imported on
construction.
"""

from pathlib import Path
from typing import Any

import numpy as np

from ..control.legged import WalkCommand
from ..simulation.mujoco.state import BipedTruth, tilt_deg

VISIBLE_GROUPS = (0, 1, 2)
# MuJoCo geom types (mjtGeom): plane, hfield, sphere, capsule, ellipsoid,
# cylinder, box, mesh.
PLANE, HFIELD, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = range(8)


def _xyzw(wxyz: Any) -> list[float]:
    w, x, y, z = (float(v) for v in wxyz)
    return [x, y, z, w]


def body_entity(model: Any, body: int) -> str:
    """Entity path of ``body`` (its name, or ``body_<index>``)."""
    name = model.body(body).name or f"body_{body}"
    return f"world/robot/{name}"


class BipedRerunRecorder:
    """See module docs. ``model`` is the compiled ``MjModel``."""

    def __init__(self, model: Any, output: Path, headless: bool, title: str) -> None:
        import rerun as rr
        import rerun.blueprint as blueprint

        self.rr, self.model = rr, model
        rr.init("robotlab_biped")
        sinks: list[Any] = [rr.FileSink(str(output / "biped.rrd"))]
        if not headless:
            rr.spawn(connect=False)
            sinks.append(
                rr.GrpcSink("rerun+http://127.0.0.1:9876/proxy", flush_timeout_sec=2)
            )
        layout = blueprint.Blueprint(
            blueprint.Horizontal(
                blueprint.Spatial3DView(
                    origin="world",
                    name=title,
                    line_grid=blueprint.LineGrid3D(
                        visible=True, spacing=0.1, color=[100, 120, 100, 160]
                    ),
                ),
                blueprint.Tabs(
                    blueprint.TimeSeriesView(origin="trunk", name="Trunk"),
                    blueprint.TimeSeriesView(origin="command", name="Walk command"),
                ),
                column_shares=[3, 2],
            )
        )
        rr.set_sinks(*sinks, default_blueprint=layout)
        rr.send_blueprint(layout, make_active=True, make_default=True)
        rr.set_time("sim_time", duration=0.0)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._bodies = sorted({int(model.geom_bodyid[g]) for g in self._visible()})
        for geom in self._visible():
            self._log_geom(geom)

    def _visible(self) -> list[int]:
        model = self.model
        return [
            g
            for g in range(model.ngeom)
            if model.geom_group[g] in VISIBLE_GROUPS
            and model.geom_rgba[g, 3] > 0
            and model.geom_type[g] not in (PLANE, HFIELD)
            and model.geom_bodyid[g] != 0  # world geoms: the grid stands in
        ]

    def _log_geom(self, geom: int) -> None:
        """One geom, static, in its body's frame."""
        rr, model = self.rr, self.model
        path = f"{body_entity(model, int(model.geom_bodyid[geom]))}/geom_{geom}"
        rr.log(
            path,
            rr.Transform3D(
                translation=model.geom_pos[geom].tolist(),
                quaternion=_xyzw(model.geom_quat[geom]),
            ),
            static=True,
        )
        color = (np.clip(model.geom_rgba[geom], 0, 1) * 255).astype(np.uint8).tolist()
        size = model.geom_size[geom]
        kind = int(model.geom_type[geom])
        if kind == MESH:
            mesh = int(model.geom_dataid[geom])
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            face, faces = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            shape: Any = rr.Mesh3D(
                vertex_positions=model.mesh_vert[start : start + count],
                triangle_indices=model.mesh_face[face : face + faces],
                albedo_factor=color,
            )
        elif kind == BOX:
            shape = rr.Boxes3D(
                half_sizes=[size.tolist()], colors=[color], fill_mode="solid"
            )
        elif kind in (SPHERE, ELLIPSOID):
            radii = [size[0]] * 3 if kind == SPHERE else size.tolist()
            shape = rr.Ellipsoids3D(
                half_sizes=[radii], colors=[color], fill_mode="solid"
            )
        elif kind == CAPSULE:
            # MuJoCo capsules are centred on the geom origin along its z axis.
            shape = rr.Capsules3D(
                lengths=[2 * size[1]],
                radii=[size[0]],
                translations=[[0, 0, -size[1]]],
                colors=[color],
            )
        else:  # CYLINDER
            shape = rr.Cylinders3D(
                lengths=[2 * size[1]], radii=[size[0]], colors=[color]
            )
        rr.log(f"{path}/shape", shape, static=True)

    def log_state(
        self, data: Any, truth: BipedTruth, request: WalkCommand | None
    ) -> None:
        """Body poses and plots at ``truth.sim_time_ns`` (see module docs)."""
        rr = self.rr
        rr.set_time("sim_time", duration=truth.sim_time_ns * 1e-9)
        for body in self._bodies:
            rr.log(
                body_entity(self.model, body),
                rr.Transform3D(
                    translation=data.xpos[body].tolist(),
                    quaternion=_xyzw(data.xquat[body]),
                ),
            )
        x, y, z, w = truth.quaternion_xyzw
        heading = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 0.0])
        norm = np.linalg.norm(heading)
        forward = float(heading @ truth.velocity_mps / norm) if norm > 1e-6 else 0.0
        for path, value in (
            ("trunk/height_m", truth.position_m[2]),
            ("trunk/tilt_deg", tilt_deg(truth.quaternion_xyzw)),
            ("trunk/forward_mps", forward),
        ):
            rr.log(path, rr.Scalars(float(value)))
        if request is not None:
            for name in ("forward_mps", "left_mps", "yaw_rate_rad_s"):
                rr.log(f"command/{name}", rr.Scalars(float(getattr(request, name))))

    def close(self) -> None:
        self.rr.disconnect()
