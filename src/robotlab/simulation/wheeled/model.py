"""Ray-cast wheeled vehicle: suspension, tyres, and wheel spin in Python.

Gazebo simulates one rigid chassis link (mass and inertia of the whole
vehicle) with gravity and collision boxes; it has no wheel joints. Each
odometry period Python computes, for every wheel,

1. **suspension** (a ray cast along the strut, as in game-engine ray-cast
   vehicles): the strut ray from the mount (``properties.py``) hits the
   ground (``ground.py``) at distance s; the hub sits at L = s − R/cos θ
   (θ between strut and ground normal), compression x = travel − L, and the
   normal load is N = max(0, k x + c ẋ + k_bump max(0, x − travel));
2. **tyre forces** from the transient Magic Formula (``tyre.py``) in the
   contact frame (heading e_x in the ground plane, e_y = n × e_x);
3. **wheel spin** ω from I ω̇ = T_drive − R F_x, with brakes and rolling
   resistance as Coulomb torques that stop the wheel without reversing it.

Tyre states and wheel spin are integrated in ``substeps`` per period with
the chassis velocity held (the wheel modes are much faster than the
chassis). The chassis receives, per wheel, the ground force F = N n + F_x
e_x + F_y e_y at the hub, the part of its moment about the hub that the
axle bearing transmits (everything but the spin-axis component), and the
reaction of the drive and brake torques about the spin axis; this is exact
for wheels with spin inertia but no translational mass. Aerodynamic drag
(at the centre of mass) and downforce (at the front and rear axle lines)
are added. The result is one world-axis wrench about the centre of mass,
applied by Gazebo for the next period (explicit coupling: the default
500 Hz period resolves the suspension and tyre carcass modes).

Inputs and outputs are SI, world ENU, body x forward / y left / z up;
quaternions are xyzw body-to-world. This is a textbook vehicle model with
constant coefficients, not a calibrated vehicle.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .actuators import WheelActuators
from .ground import GroundModel
from ...core.wheeled_vehicle import WheeledProperties
from .tyre import TyreModel

# Lower bound on cos θ (strut versus ground normal) for the hub offset.
MIN_COS_STRUT = 0.5


def rotation_matrix(quaternion_xyzw: NDArray[np.float64]) -> NDArray[np.float64]:
    """Body-to-world rotation matrix of a unit xyzw quaternion.

    Written out (instead of scipy's Rotation) because the model evaluates it
    every odometry period and per-call overhead dominates for 3×3 work.
    """
    x, y, z, w = np.asarray(quaternion_xyzw, float) / np.linalg.norm(quaternion_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def cross(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise cross product of (…×3) arrays (cheaper than np.cross here)."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return np.stack((ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), -1)


def _dot(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Row-wise dot product of (n×3) arrays."""
    return np.asarray((a * b).sum(axis=1))


@dataclass(frozen=True)
class ChassisState:
    """Truth chassis state at the start of a period (centre of mass)."""

    position_m: NDArray[np.float64]
    quaternion_xyzw: NDArray[np.float64]
    velocity_world_mps: NDArray[np.float64]
    body_rates_rad_s: NDArray[np.float64]


@dataclass(frozen=True)
class WheelTelemetry:
    """Per-wheel results of one period (arrays in wheel order)."""

    in_contact: NDArray[np.bool_]
    compression_m: NDArray[np.float64]
    normal_load_n: NDArray[np.float64]
    longitudinal_n: NDArray[np.float64]
    lateral_n: NDArray[np.float64]
    slip_ratio: NDArray[np.float64]
    slip_angle_rad: NDArray[np.float64]
    wheel_speed_rad_s: NDArray[np.float64]
    steer_rad: NDArray[np.float64]
    drive_torque_nm: NDArray[np.float64]
    friction_scale: NDArray[np.float64]
    contact_point_m: NDArray[np.float64]
    hub_world_m: NDArray[np.float64]


@dataclass(frozen=True)
class ChassisWrench:
    """Total force (N) and torque about the centre of mass (N·m), world axes."""

    force_world_n: NDArray[np.float64]
    torque_world_nm: NDArray[np.float64]


@dataclass(frozen=True)
class _Contacts:
    """Suspension geometry and loads of one period (internal)."""

    in_contact: NDArray[np.bool_]
    compression_m: NDArray[np.float64]
    normal_n: NDArray[np.float64]
    normal: NDArray[np.float64]
    hub_m: NDArray[np.float64]
    contact_m: NDArray[np.float64]
    friction: NDArray[np.float64]
    rolling_scale: NDArray[np.float64]


class WheeledVehicleModel:
    """Python half of a wheeled vehicle (see module docs).

    ``actuators`` holds the accepted command; ``step`` advances the wheels
    over one period ``dt_s`` from the chassis state and returns the wrench
    for Gazebo plus telemetry. Raises ValueError for a non-positive step.
    """

    def __init__(
        self,
        properties: WheeledProperties,
        ground: GroundModel,
        actuators: WheelActuators | None = None,
    ) -> None:
        self.properties = properties
        self.ground = ground
        self.actuators = actuators or WheelActuators(properties)
        count = properties.wheel_count
        self.tyres = TyreModel(properties.tyre, count)
        self.wheel_speed_rad_s = np.zeros(count)
        # Integrated wheel rotation (rad), for the encoders.
        self.wheel_angle_rad = np.zeros(count)
        self._mounts_body = properties.mount_positions_m()
        self._ray_length_m = np.full(
            count,
            properties.suspension.travel_m + properties.tyre.radius_m / MIN_COS_STRUT,
        )
        x = properties.hub_positions_m()[:, 0]
        # Front and rear axle lines (body), where downforce acts.
        self._axle_points_body = np.array([[x.max(), 0.0, 0.0], [x.min(), 0.0, 0.0]])
        front = properties.aero.downforce_front_fraction
        self._downforce_split = np.array([front, 1 - front])
        self.telemetry: WheelTelemetry | None = None

    def step(self, chassis: ChassisState, dt_s: float) -> ChassisWrench:
        """Advance wheels and tyres by ``dt_s``; return the chassis wrench."""
        if not dt_s > 0:
            raise ValueError("WheeledVehicleModel.step needs dt_s > 0")
        rotation = rotation_matrix(chassis.quaternion_xyzw)
        omega_world = rotation @ chassis.body_rates_rad_s
        self.actuators.slew_steering(dt_s)
        contacts = self._suspension(chassis, rotation, omega_world)
        e_x, e_y, axle = self._contact_axes(rotation, contacts.normal)
        v_contact = chassis.velocity_world_mps + cross(
            omega_world, contacts.contact_m - chassis.position_m
        )
        vx = _dot(v_contact, e_x)
        vy = _dot(v_contact, e_y)
        fx, fy, reaction, drive = self._spin(contacts, vx, vy, dt_s)
        wrench = self._assemble(chassis, rotation, contacts, e_x, e_y, axle, fx, fy)
        torque = wrench.torque_world_nm - np.sum(reaction[:, None] * axle, axis=0)
        kappa, tan_alpha = self.tyres.slips()
        self.telemetry = WheelTelemetry(
            contacts.in_contact,
            contacts.compression_m,
            contacts.normal_n,
            fx,
            fy,
            kappa,
            np.arctan(tan_alpha),
            self.wheel_speed_rad_s.copy(),
            self.actuators.steer_rad.copy(),
            drive,
            contacts.friction,
            contacts.contact_m,
            contacts.hub_m,
        )
        return ChassisWrench(wrench.force_world_n, torque)

    def _suspension(
        self, chassis: ChassisState, rotation: NDArray, omega_world: NDArray
    ) -> _Contacts:
        p, s = self.properties, self.properties.suspension
        radius = p.tyre.radius_m
        mounts = chassis.position_m + self._mounts_body @ rotation.T
        down = np.tile(-rotation[:, 2], (p.wheel_count, 1))
        hits = self.ground.cast(mounts, down, self._ray_length_m)
        cos_strut = np.maximum(-_dot(down, hits.normal), MIN_COS_STRUT)
        length = np.where(hits.hit, hits.distance_m - radius / cos_strut, s.travel_m)
        compression = s.travel_m - length
        in_contact = hits.hit & (compression > 0)
        v_mount = chassis.velocity_world_mps + cross(
            omega_world, mounts - chassis.position_m
        )
        # ẋ = −dL/dt; the ground is static, so L changes with the mount's
        # velocity along the normal (scaled to the strut direction).
        rate = -_dot(v_mount, hits.normal) / cos_strut
        force = (
            s.stiffness_n_per_m * compression
            + s.damping_n_s_per_m * rate
            + s.bump_stiffness_n_per_m * np.maximum(compression - s.travel_m, 0.0)
        )
        normal_n = np.where(in_contact, np.maximum(force, 0.0), 0.0)
        hub = mounts + np.minimum(np.maximum(length, 0.0), s.travel_m)[:, None] * down
        contact = hub - radius * hits.normal
        return _Contacts(
            in_contact,
            np.where(in_contact, compression, 0.0),
            normal_n,
            hits.normal,
            hub,
            contact,
            hits.friction_scale,
            hits.rolling_resistance_scale,
        )

    def _contact_axes(
        self, rotation: NDArray, normal: NDArray
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Heading e_x (ground plane), lateral e_y = n × e_x, and spin axis."""
        steer = self.actuators.steer_rad
        c, s = np.cos(steer)[:, None], np.sin(steer)[:, None]
        # Columns of R: body x and y in world axes.
        heading = c * rotation[:, 0] + s * rotation[:, 1]
        axle = c * rotation[:, 1] - s * rotation[:, 0]
        e_x = heading - _dot(heading, normal)[:, None] * normal
        e_x /= np.maximum(np.linalg.norm(e_x, axis=1, keepdims=True), 1e-9)
        return e_x, cross(normal, e_x), axle

    def _spin(
        self, contacts: _Contacts, vx: NDArray, vy: NDArray, dt_s: float
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Substep tyres and wheel spin; return mean F_x, F_y, the mean axle
        torque on the wheel (drive plus applied brake/rolling), and the mean
        delivered drive torque."""
        p = self.properties
        n = p.substeps
        h = dt_s / n
        inertia, radius = p.wheel_inertia_kg_m2, p.tyre.radius_m
        brake = self.actuators.brake_nm
        sums = np.zeros((4, p.wheel_count))
        omega = self.wheel_speed_rad_s
        for _ in range(n):
            forces = self.tyres.forces(
                contacts.normal_n, contacts.friction, vx, vy, omega
            )
            drive = self.actuators.drive_torque(omega)
            free = omega + h * (drive - radius * forces.longitudinal_n) / inertia
            resist = brake + forces.rolling_resistance_nm * contacts.rolling_scale
            # Coulomb torques stop the wheel but never reverse it.
            stopped = np.maximum(np.abs(free) - h * resist / inertia, 0.0)
            new = np.sign(free) * stopped
            applied = (new - free) * inertia / h
            omega = new
            self.wheel_angle_rad = self.wheel_angle_rad + h * omega
            self.tyres.advance(h, vx, vy, omega, contacts.in_contact)
            sums += [forces.longitudinal_n, forces.lateral_n, drive + applied, drive]
        self.wheel_speed_rad_s = omega
        mean = sums / n
        return mean[0], mean[1], mean[2], mean[3]

    def _assemble(
        self,
        chassis: ChassisState,
        rotation: NDArray,
        contacts: _Contacts,
        e_x: NDArray,
        e_y: NDArray,
        axle: NDArray,
        fx: NDArray,
        fy: NDArray,
    ) -> ChassisWrench:
        """Ground forces and aerodynamics as one wrench about the centre of mass."""
        ground = (
            contacts.normal_n[:, None] * contacts.normal
            + fx[:, None] * e_x
            + fy[:, None] * e_y
        ) * contacts.in_contact[:, None]
        # The bearing transmits the moment of the contact force about the
        # hub except its spin-axis part (that spins the wheel instead).
        moment_hub = cross(contacts.contact_m - contacts.hub_m, ground)
        moment_hub -= _dot(moment_hub, axle)[:, None] * axle
        arm = contacts.hub_m - chassis.position_m
        force = ground.sum(axis=0)
        torque = (cross(arm, ground) + moment_hub).sum(axis=0)
        aero_force, aero_torque = self._aero(chassis, rotation)
        return ChassisWrench(force + aero_force, torque + aero_torque)

    def _aero(
        self, chassis: ChassisState, rotation: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Still-air drag ½ρC_dA|v|v at the centre of mass and downforce
        ½ρC_LA v_x² split over the front and rear axle lines."""
        a = self.properties.aero
        v = chassis.velocity_world_mps
        q = 0.5 * a.air_density_kg_m3
        drag = -q * a.drag_area_m2 * np.linalg.norm(v) * v
        forward = float(rotation[:, 0] @ v)
        down = q * a.downforce_area_m2 * forward * forward
        if down == 0.0:
            return drag, np.zeros(3)
        points = self._axle_points_body @ rotation.T
        forces = (down * self._downforce_split)[:, None] * -rotation[:, 2]
        torque = cross(points, forces).sum(axis=0)
        return drag + forces.sum(axis=0), torque
