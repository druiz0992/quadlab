"""Tyre forces: simplified Magic Formula with transient slip, per wheel.

Steady-state force (Pacejka, *Tire and Vehicle Dynamics*, 3rd ed., 2012,
§4.3.2, constant coefficients, no shifts)::

    F(s) = D sin(C atan(B s − E (B s − atan(B s)))),   D = μ F_z

with s the longitudinal slip ratio κ or the lateral slip tan α. Combined
slip uses normalized slips (the "similarity" approach, Pacejka §4.3.3):
each slip is divided by the slip at its own force peak, the resultant
ρ = |(κ/κ_pk, tan α/α_pk)| is evaluated in each direction's curve, and
the force is shared in the direction of the normalized slip vector, so the
combined force stays inside the friction ellipse (μ_x F_z, μ_y F_z).

Transient behaviour uses the single-contact-point relaxation model
(Pacejka §7.2.1): carcass deflections u (longitudinal) and v (lateral)
obey ::

    du/dt + |V_x| u / σ_κ = −V_sx,      dv/dt + |V_x| v / σ_α = −V_sy

with V_sx = V_x − ω R the longitudinal slip velocity and V_sy = V_y the
lateral contact-point velocity. The transient slips are κ' = u/σ_κ and
tan α' = v/σ_α; in steady state κ' = (ωR − V_x)/|V_x|, the usual slip
ratio, and positive slip gives positive (forward or leftward) force, so
force always opposes slip. At standstill the deflections act as springs:
a parked vehicle holds on a slope instead of creeping, and the slip
singularity at V_x = 0 never appears. A speed-blended damping term
(Pacejka §7.2.1, low-speed damping) removes the carcass oscillation below
``low_speed_mps``; the total force is then capped to the friction ellipse,
so damping never exceeds friction while a wheel slides. The decay term is
integrated implicitly, so the update is stable for any step at high speed.

Units: N, m, s, rad. Arrays are per wheel (shape (n,)). This is a
textbook model with constant coefficients, not a calibrated tyre.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...core.wheeled_vehicle import MagicFormulaCoefficients, TyreParameters

# Clamp on transient slips (κ' and tan α') so a saturated tyre does not wind
# its deflection up without bound (a reversal would then lag for seconds).
MAX_SLIP = 2.0


def magic_formula(
    coefficients: MagicFormulaCoefficients, slip: ArrayLike
) -> NDArray[np.float64]:
    """F / D for slip ``s`` (dimensionless, odd in s)."""
    b, c, e = coefficients.stiffness_b, coefficients.shape_c, coefficients.curvature_e
    bs = b * np.asarray(slip, float)
    return np.asarray(np.sin(c * np.arctan(bs - e * (bs - np.arctan(bs)))))


def peak_slip(coefficients: MagicFormulaCoefficients) -> float:
    """Slip at the force peak (grid search, then golden-section refinement)."""
    grid = np.linspace(1e-4, 5 * np.pi / coefficients.stiffness_b, 4001)
    index = int(np.argmax(magic_formula(coefficients, grid)))
    low = float(grid[max(index - 1, 0)])
    high = float(grid[min(index + 1, len(grid) - 1)])
    ratio = (np.sqrt(5) - 1) / 2
    for _ in range(60):
        a = high - ratio * (high - low)
        b = low + ratio * (high - low)
        if magic_formula(coefficients, a) > magic_formula(coefficients, b):
            high = b
        else:
            low = a
    return 0.5 * (low + high)


@dataclass(frozen=True)
class TyreForces:
    """Contact-frame forces per wheel: longitudinal (along the wheel heading
    in the ground plane), lateral (to the wheel's left), and the rolling
    resistance torque magnitude (N·m, opposing spin)."""

    longitudinal_n: NDArray[np.float64]
    lateral_n: NDArray[np.float64]
    rolling_resistance_nm: NDArray[np.float64]


class TyreModel:
    """Transient Magic-Formula tyres for ``count`` identical wheels.

    State: carcass deflections ``u_m`` and ``v_m`` (one per wheel).
    ``advance`` integrates them over ``dt_s`` for the given contact-point
    velocities and wheel speeds; ``forces`` evaluates the current forces.
    """

    def __init__(self, parameters: TyreParameters, count: int) -> None:
        if count < 1:
            raise ValueError("TyreModel needs at least one wheel")
        self.parameters = parameters
        self._peak_long = peak_slip(parameters.longitudinal)
        self._peak_lat = peak_slip(parameters.lateral)
        self._u_limit = MAX_SLIP * parameters.relaxation_longitudinal_m
        self._v_limit = MAX_SLIP * parameters.relaxation_lateral_m
        self.u_m = np.zeros(count)
        self.v_m = np.zeros(count)

    def advance(
        self,
        dt_s: float,
        vx_mps: NDArray[np.float64],
        vy_mps: NDArray[np.float64],
        wheel_speed_rad_s: NDArray[np.float64],
        in_contact: NDArray[np.bool_],
    ) -> None:
        """Integrate the deflections over ``dt_s`` (implicit decay, explicit
        slip input). Wheels out of contact relax to zero deflection."""
        p = self.parameters
        slip_x = vx_mps - wheel_speed_rad_s * p.radius_m
        speed = np.abs(vx_mps)
        u = (self.u_m - dt_s * slip_x) / (
            1 + dt_s * speed / p.relaxation_longitudinal_m
        )
        v = (self.v_m - dt_s * vy_mps) / (1 + dt_s * speed / p.relaxation_lateral_m)
        # minimum/maximum: np.clip's overhead dominates on 4-element arrays.
        u = np.minimum(np.maximum(u, -self._u_limit), self._u_limit)
        v = np.minimum(np.maximum(v, -self._v_limit), self._v_limit)
        self.u_m = np.where(in_contact, u, 0.0)
        self.v_m = np.where(in_contact, v, 0.0)

    def slips(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Transient slip ratio κ' and lateral slip tan α' per wheel."""
        p = self.parameters
        return (
            self.u_m / p.relaxation_longitudinal_m,
            self.v_m / p.relaxation_lateral_m,
        )

    def forces(
        self,
        normal_n: NDArray[np.float64],
        friction_scale: NDArray[np.float64],
        vx_mps: NDArray[np.float64],
        vy_mps: NDArray[np.float64],
        wheel_speed_rad_s: NDArray[np.float64],
    ) -> TyreForces:
        """Forces for the current deflections; negative loads count as zero."""
        p = self.parameters
        kappa, tan_alpha = self.slips()
        sx, sy = kappa / self._peak_long, tan_alpha / self._peak_lat
        rho = np.hypot(sx, sy)
        safe = np.where(rho > 1e-12, rho, 1.0)
        # Each direction's curve at the resultant normalized slip.
        fx_unit = magic_formula(p.longitudinal, rho * self._peak_long) * sx / safe
        fy_unit = magic_formula(p.lateral, rho * self._peak_lat) * sy / safe
        load = np.maximum(normal_n, 0.0)
        fx = load * friction_scale * p.friction_longitudinal * fx_unit
        fy = load * friction_scale * p.friction_lateral * fy_unit
        # Low-speed damping fades out linearly by low_speed_mps.
        blend = np.maximum(1 - np.abs(vx_mps) / p.low_speed_mps, 0.0)
        damping = p.low_speed_damping_n_s_per_m * blend * (load > 0)
        fx = fx - damping * (vx_mps - wheel_speed_rad_s * p.radius_m)
        fy = fy - damping * vy_mps
        # Damping is for the carcass mode near zero slip; while sliding it
        # must not push the force past friction: cap to the ellipse.
        limit_x = load * friction_scale * p.friction_longitudinal
        limit_y = load * friction_scale * p.friction_lateral
        ratio = np.hypot(
            fx / np.maximum(limit_x, 1e-12), fy / np.maximum(limit_y, 1e-12)
        )
        scale = np.where(ratio > 1.0, 1.0 / np.maximum(ratio, 1e-12), 1.0)
        return TyreForces(
            fx * scale, fy * scale, p.rolling_resistance * load * p.radius_m
        )
