"""SO(3) helpers for filters that estimate attitude.

Conventions (docs/1_overview.md): rotation matrices R rotate body vectors
into world coordinates, and RobotLab's attitude error is a **right**
rotation vector δθ (body axes): R_true = R · Exp(δθ). Exp/Log are the
rotation-vector exponential and logarithm (``scipy`` ``Rotation``
``from_rotvec``/``as_rotvec``). Jacobians follow Solà, Deray, and
Atchuthan, *A micro Lie theory for state estimation in robotics* (2018),
eqs. (143)-(146). Angles in rad.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Below this angle (rad) the closed forms lose precision; their Taylor
# series to second order are accurate to float64 there.
SMALL_ANGLE_RAD = 1e-6


def skew(v: ArrayLike) -> NDArray[np.float64]:
    """The cross-product matrix [v]×, so that ``skew(a) @ b == cross(a, b)``."""
    x, y, z = np.asarray(v, float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def right_jacobian(v: ArrayLike) -> NDArray[np.float64]:
    """Right Jacobian J_r(θ) of SO(3): Exp(θ + δ) ≈ Exp(θ) · Exp(J_r(θ) δ).

    θ is a rotation vector (rad). Solà et al. eq. (143).
    """
    theta = np.linalg.norm(v)
    cross = skew(v)
    if theta < SMALL_ANGLE_RAD:
        return np.eye(3) - 0.5 * cross + cross @ cross / 6
    return (
        np.eye(3)
        - (1 - np.cos(theta)) / theta**2 * cross
        + (theta - np.sin(theta)) / theta**3 * cross @ cross
    )


def right_jacobian_inverse(v: ArrayLike) -> NDArray[np.float64]:
    """Inverse of ``right_jacobian``: Log(Exp(θ) · Exp(δ)) ≈ θ + J_r⁻¹(θ) δ.

    θ is a rotation vector (rad) with |θ| < π. Solà et al. eq. (144).
    Raises ValueError at or beyond π, where J_r is singular.
    """
    theta = np.linalg.norm(v)
    cross = skew(v)
    if theta < SMALL_ANGLE_RAD:
        return np.eye(3) + 0.5 * cross + cross @ cross / 12
    if theta >= np.pi:
        raise ValueError("right_jacobian_inverse needs a rotation angle below pi")
    coefficient = 1 / theta**2 - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))
    return np.eye(3) + 0.5 * cross + coefficient * cross @ cross
