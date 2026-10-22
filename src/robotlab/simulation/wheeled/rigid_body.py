"""Reference rigid-body integrator for the wheeled model without Gazebo.

Semi-implicit (symplectic) Euler for one rigid body under gravity
(0, 0, −9.81) m/s² and a world-axis wrench about its centre of mass:
v ← v + dt (F/m + g), ω_b ← ω_b + dt I⁻¹(τ_b − ω_b × I ω_b), then the
pose from the new velocities (rotation by the exponential map). It stands
in for Gazebo's chassis link in unit tests and quick numerical checks;
Gazebo remains the simulator for missions (collisions, sensors).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

from .model import ChassisState, ChassisWrench

GRAVITY_MPS2 = np.array([0.0, 0.0, -9.81])


@dataclass
class RigidBody:
    """Mutable rigid-body state: mass (kg), principal inertia (kg·m²)."""

    mass_kg: float
    inertia_kg_m2: NDArray[np.float64]
    position_m: NDArray[np.float64]
    quaternion_xyzw: NDArray[np.float64]
    velocity_world_mps: NDArray[np.float64]
    body_rates_rad_s: NDArray[np.float64]

    @classmethod
    def at_rest(
        cls,
        mass_kg: float,
        inertia_kg_m2: ArrayLike,
        position_m: ArrayLike,
        yaw_rad: float = 0.0,
    ) -> "RigidBody":
        return cls(
            mass_kg,
            np.asarray(inertia_kg_m2, float),
            np.asarray(position_m, float),
            Rotation.from_euler("z", yaw_rad).as_quat(),
            np.zeros(3),
            np.zeros(3),
        )

    def state(self) -> ChassisState:
        return ChassisState(
            self.position_m.copy(),
            self.quaternion_xyzw.copy(),
            self.velocity_world_mps.copy(),
            self.body_rates_rad_s.copy(),
        )

    def step(self, wrench: ChassisWrench, dt_s: float) -> None:
        """Advance by ``dt_s`` under gravity and ``wrench`` (see module docs)."""
        rotation = Rotation.from_quat(self.quaternion_xyzw)
        self.velocity_world_mps = self.velocity_world_mps + dt_s * (
            wrench.force_world_n / self.mass_kg + GRAVITY_MPS2
        )
        torque_body = rotation.inv().apply(wrench.torque_world_nm)
        w, inertia = self.body_rates_rad_s, self.inertia_kg_m2
        self.body_rates_rad_s = w + dt_s * (torque_body - np.cross(w, inertia * w)) / (
            inertia
        )
        self.position_m = self.position_m + dt_s * self.velocity_world_mps
        delta = Rotation.from_rotvec(self.body_rates_rad_s * dt_s)
        self.quaternion_xyzw = (rotation * delta).as_quat()
