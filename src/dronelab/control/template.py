"""Template controller plug-in: copy this file to start your own.

Select it with ``algorithms.controller: dronelab.control.template:TemplateController``
(or your copy's ``package.module:Class``), and test it with
``dronelab.testing.plugins.check_controller`` before flying it in Gazebo.

A textbook cascaded multirotor controller, kept short so the structure
stays visible (it is not tuned for every airframe):

1. position loop (``update_position``, position-loop rate): desired
   acceleration a = Kp (p_ref − p) + Kv (v_ref − v) + a_ref, horizontal part
   limited to the tilt limit; the specific force f = a + g ẑ gives the
   desired body z axis, and the yaw reference completes the attitude R_d;
2. attitude loop (``update_attitude``, control rate): attitude error
   e_R = ½ vee(R_dᵀR − RᵀR_d) (Lee et al., "Geometric tracking control of a
   quadrotor UAV on SE(3)", 2010), torque τ = J (−K_R e_R − K_ω ω), and
   collective thrust T = m f·(R ẑ);
3. allocation: the wrench [T, τ] is split over the vertical lift rotors
   with the pseudo-inverse of [1; y_i; −x_i; −s_i k_m] (thrust, roll and
   pitch arms, spin-direction yaw moment), clipped to [0, max thrust];
   clipping sets ``saturated``. Other rotors (a pusher) get zero.

Requests are either a mission ``ReferencePoint`` or pilot/take-off
``FlightCommand`` sticks, which move a position setpoint (forward along
the current yaw, climb, yaw rate). The controller receives a
``ControlState`` built from truth or from the estimate, never decides
which, and must not use simulator truth itself.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..core.algorithms import AlgorithmContext
from .interface import ControlState
from .manual import FlightCommand
from .references import ReferencePoint

DEFAULTS = dict(
    position_gain=1.0,  # 1/s²
    velocity_gain=2.0,  # 1/s
    attitude_gain=36.0,  # 1/s² (natural frequency 6 rad/s)
    rate_gain=12.0,  # 1/s (critically damped with attitude_gain)
    yaw_gain_fraction=0.3,  # yaw is slower than roll and pitch
    max_tilt_deg=25.0,
)


class TemplateController:
    """Cascaded PD controller following the ``RotorController`` contract."""

    def __init__(self, context: AlgorithmContext, options: Mapping[str, Any]) -> None:
        unknown = set(options) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"TemplateController: unknown options {sorted(unknown)}")
        self.options = {**DEFAULTS, **options}
        vehicle = context.vehicle
        self._mass_kg = vehicle.mass
        self._inertia = np.diag(vehicle.body_inertia)
        self._gravity = context.gravity_mps2
        self._max_thrust_n = vehicle.max_rotor_thrust
        self._rotor_count = len(vehicle.rotors)
        self._lift = np.array(vehicle.lift_rotors)
        x, y, spin = np.asarray(vehicle.rotors, float)[self._lift].T
        # Rows: thrust, roll moment, pitch moment, yaw moment (reaction
        # torque opposes the spin direction, as in Gazebo's motor model).
        self._allocation = np.array(
            [np.ones(len(x)), y, -x, -spin * vehicle.moment_ratio]
        )
        self._setpoint: NDArray[np.float64] | None = None
        self._yaw_rad = 0.0
        self._specific_force = np.array([0.0, 0.0, self._gravity])
        self._desired = Rotation.identity()
        self.saturated = False
        self.landing = False

    @property
    def setpoint_m(self) -> NDArray[np.float64] | None:
        return self._setpoint

    def update_position(
        self,
        state: ControlState,
        request: FlightCommand | ReferencePoint,
        dt_s: float,
    ) -> None:
        if self._setpoint is None:
            self._setpoint = np.array(state.position_m, float)
            self._yaw_rad = Rotation.from_quat(state.quaternion_xyzw).as_euler("ZYX")[0]
        if isinstance(request, ReferencePoint):
            p_ref, v_ref = request.position_m, request.velocity_mps
            a_ref, self._yaw_rad = request.acceleration_mps2, request.yaw_rad
            self._setpoint = np.array(p_ref, float)
        else:
            v_ref = self._stick_velocity(request)
            self._setpoint = self._setpoint + v_ref * dt_s
            self._yaw_rad += request.yaw_rate_rps * dt_s
            p_ref, a_ref = self._setpoint, np.zeros(3)
        o = self.options
        acceleration = (
            o["position_gain"] * (p_ref - state.position_m)
            + o["velocity_gain"] * (v_ref - state.velocity_world_mps)
            + a_ref
        )
        self._set_attitude_target(acceleration)

    def _stick_velocity(self, command: FlightCommand) -> NDArray[np.float64]:
        """World velocity from sticks: forward along the yaw heading, climb."""
        heading = np.array([np.cos(self._yaw_rad), np.sin(self._yaw_rad), 0.0])
        return command.forward_mps * heading + np.array([0.0, 0.0, command.climb_mps])

    def _set_attitude_target(self, acceleration: NDArray[np.float64]) -> None:
        """Tilt-limited specific force and the attitude that produces it."""
        limit = self._gravity * np.tan(np.radians(self.options["max_tilt_deg"]))
        horizontal = acceleration[:2]
        norm = np.linalg.norm(horizontal)
        if norm > limit:
            horizontal = horizontal * limit / norm
        force = np.array([*horizontal, max(acceleration[2] + self._gravity, 0.1)])
        z_body = force / np.linalg.norm(force)
        x_heading = np.array([np.cos(self._yaw_rad), np.sin(self._yaw_rad), 0.0])
        y_body = np.cross(z_body, x_heading)
        y_body /= np.linalg.norm(y_body)
        x_body = np.cross(y_body, z_body)
        self._desired = Rotation.from_matrix(np.column_stack([x_body, y_body, z_body]))
        self._specific_force = force

    def update_attitude(self, state: ControlState) -> NDArray[np.float64]:
        o = self.options
        R = Rotation.from_quat(state.quaternion_xyzw).as_matrix()
        Rd = self._desired.as_matrix()
        error_matrix = 0.5 * (Rd.T @ R - R.T @ Rd)
        error = np.array([error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0]])
        gains = np.array([1.0, 1.0, o["yaw_gain_fraction"]])
        angular = -gains * (
            o["attitude_gain"] * error + o["rate_gain"] * state.body_rates_rad_s
        )
        torque = self._inertia @ angular
        thrust = self._mass_kg * float(self._specific_force @ R[:, 2])
        forces = np.linalg.pinv(self._allocation) @ np.r_[max(thrust, 0.0), torque]
        self.saturated = bool(np.any(forces < 0) or np.any(forces > self._max_thrust_n))
        thrusts = np.zeros(self._rotor_count)
        thrusts[self._lift] = np.clip(forces, 0.0, self._max_thrust_n)
        return thrusts

    def allow_landing(self) -> None:
        """No altitude floor here, so nothing to unlock; kept for the contract."""
        self.landing = True

    def summary(self) -> dict[str, Any]:
        return dict(options=self.options)
