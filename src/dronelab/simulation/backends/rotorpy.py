"""RotorPy vehicle and baseline-controller adapter."""

from copy import deepcopy

import numpy as np

from ..state import initial_state


class RotorPyBackend:
    """Crazyflie with RotorPy dynamics and its SE(3) controller."""

    def __init__(self):
        from rotorpy.controllers.quadrotor_control import SE3Control
        from rotorpy.vehicles.crazyflie_params import quad_params
        from rotorpy.vehicles.multirotor import Multirotor

        params = deepcopy(quad_params)
        # Sensor randomness is handled by our own seeded generator. Avoid
        # RotorPy's process-global motor noise for repeatable experiments.
        params["motor_noise_std"] = 0.0
        self.vehicle = Multirotor(params, aero=True)
        self.controller = SE3Control(params)
        self.state = initial_state()
        hover = np.sqrt(params["mass"] * 9.81 / (4 * params["k_eta"]))
        self.state["rotor_speeds"][:] = hover

    def step(self, t, dt, target):
        flat = dict(
            x=target,
            x_dot=np.zeros(3),
            x_ddot=np.zeros(3),
            x_dddot=np.zeros(3),
            x_ddddot=np.zeros(3),
            yaw=0.0,
            yaw_dot=0.0,
            yaw_ddot=0.0,
        )
        control = self.controller.update(t, self.state, flat)
        # Specific force is sampled at the interval start, along with gyro.
        acceleration = self.vehicle.statedot(self.state, control, dt)["vdot"]
        self.state = self.vehicle.step(self.state, control, dt)
        return acceleration
