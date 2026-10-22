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

One model can step several identical vehicles at once (``vehicles``; the
ego and its opponents): every per-wheel array has one row per wheel of
every vehicle (vehicle-major), so the per-call overhead of the small numpy
operations is paid once per period instead of once per vehicle.

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
    """Python half of ``vehicles`` identical wheeled vehicles (module docs).

    ``actuators`` holds the accepted commands; ``step`` advances one
    vehicle (``step_all`` every vehicle) over one period ``dt_s`` from the
    chassis state(s) and returns the wrench(es) for Gazebo; ``telemetry``
    is vehicle 0's (``telemetry_of(i)`` any vehicle's). Raises ValueError
    for a non-positive step or a wrong number of states.
    """

    def __init__(
        self,
        properties: WheeledProperties,
        ground: GroundModel,
        actuators: WheelActuators | None = None,
        vehicles: int = 1,
    ) -> None:
        self.properties = properties
        self.ground = ground
        self.vehicles = vehicles
        self.actuators = actuators or WheelActuators(properties, vehicles)
        if self.actuators.vehicles != vehicles:
            raise ValueError("actuators and model need the same vehicle count")
        n = properties.wheel_count
        count = vehicles * n
        self.tyres = TyreModel(properties.tyre, count)
        self.wheel_speed_rad_s = np.zeros(count)
        # Integrated wheel rotation (rad), for the encoders.
        self.wheel_angle_rad = np.zeros(count)
        self._vehicle_of_row = np.repeat(np.arange(vehicles), n)
        self._mounts_body = np.tile(properties.mount_positions_m(), (vehicles, 1))
        self._ray_length_m = np.full(
            count,
            properties.suspension.travel_m + properties.tyre.radius_m / MIN_COS_STRUT,
        )
        x = properties.hub_positions_m()[:, 0]
        # Front and rear axle lines (body), where downforce acts.
        self._axle_points_body = np.array([[x.max(), 0.0, 0.0], [x.min(), 0.0, 0.0]])
        front = properties.aero.downforce_front_fraction
        self._downforce_split = np.array([front, 1 - front])
        self._telemetry: list[WheelTelemetry] = []

    @property
    def telemetry(self) -> WheelTelemetry | None:
        """Vehicle 0's telemetry of the last step (None before the first)."""
        return self._telemetry[0] if self._telemetry else None

    def telemetry_of(self, vehicle: int) -> WheelTelemetry:
        return self._telemetry[vehicle]

    def rows(self, vehicle: int) -> slice:
        return self.actuators.rows(vehicle)

    def step(self, chassis: ChassisState, dt_s: float) -> ChassisWrench:
        """Advance a single-vehicle model by ``dt_s``; return its wrench."""
        if self.vehicles != 1:
            raise ValueError("step() is for one vehicle; use step_all()")
        return self.step_all([chassis], dt_s)[0]

    def step_all(self, chassis: list[ChassisState], dt_s: float) -> list[ChassisWrench]:
        """Advance every vehicle by ``dt_s``; return their wrenches in order."""
        if not dt_s > 0:
            raise ValueError("WheeledVehicleModel.step needs dt_s > 0")
        if len(chassis) != self.vehicles:
            raise ValueError(f"step_all needs {self.vehicles} chassis states")
        p = np.array([c.position_m for c in chassis], float)
        v = np.array([c.velocity_world_mps for c in chassis], float)
        rotation = np.array([rotation_matrix(c.quaternion_xyzw) for c in chassis])
        rates = np.array([c.body_rates_rad_s for c in chassis], float)
        omega = np.einsum("kij,kj->ki", rotation, rates)
        rows = self._vehicle_of_row
        state = _Rows(p[rows], v[rows], rotation[rows], omega[rows])
        self.actuators.slew_steering(dt_s)
        contacts = self._suspension(state)
        e_x, e_y, axle = self._contact_axes(state.rotation, contacts.normal)
        v_contact = state.velocity + cross(
            state.omega, contacts.contact_m - state.position
        )
        vx, vy = _dot(v_contact, e_x), _dot(v_contact, e_y)
        fx, fy, reaction, drive = self._spin(contacts, vx, vy, dt_s)
        force, torque = self._assemble(state, contacts, e_x, e_y, axle, fx, fy)
        torque = torque - reaction[:, None] * axle
        kappa, tan_alpha = self.tyres.slips()
        wrenches, telemetry = [], []
        for vehicle in range(self.vehicles):
            r = self.rows(vehicle)
            aero_force, aero_torque = self._aero(v[vehicle], rotation[vehicle])
            wrenches.append(
                ChassisWrench(
                    force[r].sum(axis=0) + aero_force,
                    torque[r].sum(axis=0) + aero_torque,
                )
            )
            telemetry.append(
                WheelTelemetry(
                    contacts.in_contact[r],
                    contacts.compression_m[r],
                    contacts.normal_n[r],
                    fx[r],
                    fy[r],
                    kappa[r],
                    np.arctan(tan_alpha[r]),
                    self.wheel_speed_rad_s[r].copy(),
                    self.actuators.steer_rad[r].copy(),
                    drive[r],
                    contacts.friction[r],
                    contacts.contact_m[r],
                    contacts.hub_m[r],
                )
            )
        self._telemetry = telemetry
        return wrenches

    def _suspension(self, state: "_Rows") -> _Contacts:
        p, s = self.properties, self.properties.suspension
        radius = p.tyre.radius_m
        mounts = state.position + np.einsum(
            "kij,kj->ki", state.rotation, self._mounts_body
        )
        down = -state.rotation[:, :, 2]
        hits = self.ground.cast(mounts, down, self._ray_length_m)
        cos_strut = np.maximum(-_dot(down, hits.normal), MIN_COS_STRUT)
        length = np.where(hits.hit, hits.distance_m - radius / cos_strut, s.travel_m)
        compression = s.travel_m - length
        in_contact = hits.hit & (compression > 0)
        v_mount = state.velocity + cross(state.omega, mounts - state.position)
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
        heading = c * rotation[:, :, 0] + s * rotation[:, :, 1]
        axle = c * rotation[:, :, 1] - s * rotation[:, :, 0]
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
        sums = np.zeros((4, len(vx)))
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
        state: "_Rows",
        contacts: _Contacts,
        e_x: NDArray,
        e_y: NDArray,
        axle: NDArray,
        fx: NDArray,
        fy: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Per-row ground force and its torque about the vehicle's centre of mass."""
        ground = (
            contacts.normal_n[:, None] * contacts.normal
            + fx[:, None] * e_x
            + fy[:, None] * e_y
        ) * contacts.in_contact[:, None]
        # The bearing transmits the moment of the contact force about the
        # hub except its spin-axis part (that spins the wheel instead).
        moment_hub = cross(contacts.contact_m - contacts.hub_m, ground)
        moment_hub -= _dot(moment_hub, axle)[:, None] * axle
        arm = contacts.hub_m - state.position
        return ground, cross(arm, ground) + moment_hub

    def _aero(self, velocity: NDArray, rotation: NDArray) -> tuple[NDArray, NDArray]:
        """Still-air drag ½ρC_dA|v|v at the centre of mass and downforce
        ½ρC_LA v_x² split over the front and rear axle lines."""
        a = self.properties.aero
        q = 0.5 * a.air_density_kg_m3
        drag = -q * a.drag_area_m2 * np.linalg.norm(velocity) * velocity
        forward = float(rotation[:, 0] @ velocity)
        down = q * a.downforce_area_m2 * forward * forward
        if down == 0.0:
            return drag, np.zeros(3)
        points = self._axle_points_body @ rotation.T
        forces = (down * self._downforce_split)[:, None] * -rotation[:, 2]
        torque = cross(points, forces).sum(axis=0)
        return drag + forces.sum(axis=0), torque


@dataclass(frozen=True)
class _Rows:
    """Chassis state repeated per wheel row (internal)."""

    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    rotation: NDArray[np.float64]
    omega: NDArray[np.float64]
