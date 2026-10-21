"""Ground-truth position/attitude stabilization with explicit per-rotor thrust output."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from ..core.vehicle import DEFAULT_DRONE


@dataclass(frozen=True)
class FlightCommand:
    forward_mps: float = 0.0
    yaw_rate_rps: float = 0.0
    climb_mps: float = 0.0

    @classmethod
    def from_keys(cls, keys, rates=None):
        rates = rates or dict(forward_mps=1.0, yaw_rate_rad_s=0.7, climb_mps=0.6)
        return cls(
            rates["forward_mps"] * (("Up" in keys) - ("Down" in keys)),
            rates["yaw_rate_rad_s"] * (("Left" in keys) - ("Right" in keys)),
            rates["climb_mps"] * (("q" in keys) - ("a" in keys)),
        )


class ManualController:
    """Keys move position/yaw setpoints; feedback stabilizes the built-in quad.

    Inputs are explicit world position/velocity, xyzw attitude, and body rates.
    This initial demo uses simulator truth, not a sensor-driven estimator.
    """

    def __init__(self, properties=DEFAULT_DRONE, settings=None):
        self.settings = settings or dict(
            position_gain=1.2,
            velocity_gain=2.2,
            attitude_gain=[1.8, 1.8, 0.8],
            angular_rate_gain=[0.35, 0.35, 0.25],
            max_acceleration_mps2=2.0,
            max_position_error_m=1.5,
            min_altitude_m=0.4,
        )
        self.properties = properties
        # Hover allocation over the vertical (lift) rotors only; other rotors
        # (e.g. a pusher) receive zero thrust from this controller.
        self._lift = np.array(properties.lift_rotors)
        self._rotor_count = len(properties.rotors)
        x, y, directions = np.asarray(properties.rotors)[self._lift].T
        # Gazebo reaction torque opposes the configured spin direction.
        self.allocation = np.array(
            [np.ones(len(x)), y, -x, -directions * properties.moment_ratio]
        )
        self.target = None
        self.yaw = 0.0
        self.saturated = False
        self.landing = False
        # Attitude setpoint held between position-loop updates.
        self._specific_force = None
        self._desired_R = None
        self._yaw_rate = 0.0

    def update(self, position, velocity, quaternion, body_rates, command, dt):
        """Run the position loop then the attitude loop; return rotor thrusts (N).

        Equivalent to calling ``update_position`` and ``update_attitude`` at
        the same rate, which is the original single-loop behaviour.
        """
        self.update_position(position, velocity, quaternion, command, dt)
        return self.update_attitude(quaternion, body_rates)

    def allow_landing(self):
        """Let descent continue below ``min_altitude_m`` (landing sequences)."""
        self.landing = True

    def update_position(self, position, velocity, quaternion, command, dt):
        """Outer loop: integrate pilot setpoints and compute the attitude setpoint.

        Stores the desired specific force, attitude, and yaw rate used by
        ``update_attitude`` until the next position update. dt is the time
        since the previous position update, in (0, 0.1] seconds.
        """
        values = np.concatenate(
            (
                position,
                velocity,
                quaternion,
                [command.forward_mps, command.yaw_rate_rps, command.climb_mps, dt],
            )
        )
        if not np.isfinite(values).all() or not 0 < dt <= 0.1:
            raise ValueError(
                "Controller requires finite state/commands and dt in (0, 0.1]"
            )
        settings = self.settings
        position, velocity = map(np.asarray, (position, velocity))
        R = Rotation.from_quat(quaternion).as_matrix()
        if self.target is None:
            self.target = np.array(position, dtype=float)
            self.yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        self.yaw = float(
            (self.yaw + command.yaw_rate_rps * dt + np.pi) % (2 * np.pi) - np.pi
        )
        desired_velocity = np.array(
            [
                command.forward_mps * np.cos(self.yaw),
                command.forward_mps * np.sin(self.yaw),
                command.climb_mps,
            ]
        )
        self.target += desired_velocity * dt
        # Avoid accumulating a distant setpoint while blocked by an obstacle.
        self.target = position + np.clip(
            self.target - position,
            -settings["max_position_error_m"],
            settings["max_position_error_m"],
        )
        # The minimum-altitude floor is lifted only for a landing sequence.
        if not self.landing:
            self.target[2] = max(settings["min_altitude_m"], self.target[2])
            if position[2] <= settings["min_altitude_m"] and desired_velocity[2] < 0:
                desired_velocity[2] = 0
        acceleration = np.clip(
            settings["position_gain"] * (self.target - position)
            + settings["velocity_gain"] * (desired_velocity - velocity),
            -settings["max_acceleration_mps2"],
            settings["max_acceleration_mps2"],
        )
        self._set_attitude_setpoint(acceleration, command.yaw_rate_rps)

    def update_position_reference(self, position, velocity, quaternion, reference):
        """Outer loop tracking a ``ReferencePoint`` (control/references.py).

        Same gains as pilot mode, plus the reference acceleration as
        feed-forward: a = Kp (p_ref − p) + Kv (v_ref − v) + a_ref, clipped to
        ``max_acceleration_mps2``. The position error is limited to
        ``max_position_error_m`` (so large steps saturate) and the reference
        altitude to at least ``min_altitude_m`` unless landing.
        """
        values = np.concatenate(
            (
                position,
                velocity,
                quaternion,
                reference.position_m,
                reference.velocity_mps,
                reference.acceleration_mps2,
                [reference.yaw_rad, reference.yaw_rate_rad_s],
            )
        )
        if not np.isfinite(values).all():
            raise ValueError("Controller requires finite state and reference")
        settings = self.settings
        position, velocity = map(np.asarray, (position, velocity))
        target = np.array(reference.position_m, float)
        if not self.landing:
            target[2] = max(settings["min_altitude_m"], target[2])
        limit = settings["max_position_error_m"]
        self.target = position + np.clip(target - position, -limit, limit)
        self.yaw = float(reference.yaw_rad)
        acceleration = np.clip(
            settings["position_gain"] * (self.target - position)
            + settings["velocity_gain"] * (reference.velocity_mps - velocity)
            + reference.acceleration_mps2,
            -settings["max_acceleration_mps2"],
            settings["max_acceleration_mps2"],
        )
        self._set_attitude_setpoint(acceleration, reference.yaw_rate_rad_s)

    def _set_attitude_setpoint(self, acceleration, yaw_rate_rps):
        """Desired specific force and attitude for an acceleration (world)."""
        specific_force = acceleration + [0, 0, 9.81]
        z_axis = specific_force / np.linalg.norm(specific_force)
        heading = np.array([np.cos(self.yaw), np.sin(self.yaw), 0])
        y_axis = np.cross(z_axis, heading)
        y_axis /= np.linalg.norm(y_axis)
        self._specific_force = specific_force
        self._desired_R = np.column_stack((np.cross(y_axis, z_axis), y_axis, z_axis))
        self._yaw_rate = yaw_rate_rps

    def update_attitude(self, quaternion, body_rates):
        """Inner loop: track the held attitude setpoint; return rotor thrusts (N).

        Requires at least one prior ``update_position`` call.
        """
        if self._desired_R is None:
            raise RuntimeError("update_position must run before update_attitude")
        if not np.isfinite(np.concatenate((quaternion, body_rates))).all():
            raise ValueError("Controller requires finite attitude and body rates")
        settings = self.settings
        rates = np.asarray(body_rates)
        R = Rotation.from_quat(quaternion).as_matrix()
        desired_R = self._desired_R
        error_matrix = 0.5 * (desired_R.T @ R - R.T @ desired_R)
        error = np.array([error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0]])
        desired_rates = R.T @ desired_R @ np.array([0, 0, self._yaw_rate])
        torque = -np.array(settings["attitude_gain"]) * error - np.array(
            settings["angular_rate_gain"]
        ) * (rates - desired_rates)
        thrust = max(0.0, self.properties.mass * np.dot(self._specific_force, R[:, 2]))
        wrench = np.r_[thrust, torque]
        # Minimum-norm solution: exact for four lift rotors, spreads load for more.
        forces = np.linalg.pinv(self.allocation) @ wrench
        maximum = self.properties.max_rotor_thrust
        self.saturated = bool(np.any(forces < 0) or np.any(forces > maximum))
        if self.saturated:
            weights = np.array([1, 4, 4, 1])
            forces = lsq_linear(
                weights[:, None] * self.allocation,
                weights * wrench,
                bounds=(0, maximum),
            ).x
        thrusts = np.zeros(self._rotor_count)
        thrusts[self._lift] = np.clip(forces, 0, maximum)
        return thrusts


class ManualRotorController:
    """``ManualController`` behind the ``RotorController`` plug-in contract.

    Built-in default controller (``algorithms.controller`` ""), constructed
    from the context's nominal vehicle and the ``controller`` gain section.
    ``wrap`` adapts an existing ManualController instance.
    """

    def __init__(self, context, options=None) -> None:
        self.inner = ManualController(
            context.vehicle, dict(options) if options else None
        )

    @classmethod
    def wrap(cls, controller: ManualController) -> "ManualRotorController":
        adapter = cls.__new__(cls)
        adapter.inner = controller
        return adapter

    @property
    def saturated(self) -> bool:
        return self.inner.saturated

    @property
    def setpoint_m(self):
        return self.inner.target

    @property
    def yaw(self) -> float:
        return self.inner.yaw

    @property
    def landing(self) -> bool:
        return self.inner.landing

    def update_position(self, state, request, dt_s: float) -> None:
        from .references import ReferencePoint

        args = (state.position_m, state.velocity_world_mps, state.quaternion_xyzw)
        if isinstance(request, ReferencePoint):
            self.inner.update_position_reference(*args, request)
        else:
            self.inner.update_position(*args, request, dt_s)

    def update_attitude(self, state):
        return self.inner.update_attitude(state.quaternion_xyzw, state.body_rates_rad_s)

    def allow_landing(self) -> None:
        self.inner.allow_landing()
