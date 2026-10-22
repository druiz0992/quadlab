"""Fixed-wing aerodynamics and control-surface servos (transport-free).

Coefficient model of Beard & McLain, *Small Unmanned Aircraft: Theory and
Practice* (Princeton, 2012), ch. 4, which uses forward-right-down (FRD) body
axes. RobotLab is forward-left-up (FLU), so inputs are converted
(u, v, w) = (vₓ, −v_y, −v_z), (p, q, r) = (ωₓ, −ω_y, −ω_z), and outputs back
(fₓ, f_y, f_z) → (fₓ, −f_y, −f_z), likewise moments. With the airspeed
V = |v_air|, α = atan2(w, u), β = asin(v / V), q̄ = ½ρV²:

    C_L(α) = (1 − σ)(C_L0 + C_Lα α) + σ · 2 sign(α) sin²α cos α
    σ(α)   = (1 + e^{−M(α−α₀)} + e^{M(α+α₀)}) / ((1 + e^{−M(α−α₀)})(1 + e^{M(α+α₀)}))
    C_D(α) = C_Dp + (C_L0 + C_Lα α)² / (π e AR)
    lift   = q̄S (C_L(α) + C_Lq c q / 2V + C_Lδe δe)
    drag   = q̄S (C_D(α) + C_Dq c q / 2V + C_Dδe |δe|)
    [f_x, f_z] = R(α) [−drag, −lift]
    f_y = q̄S (C_Y0 + C_Yβ β + C_Yp b p / 2V + C_Yr b r / 2V + C_Yδa δa + C_Yδr δr)
    l = q̄Sb (C_l0 + C_lβ β + C_lp b p / 2V + C_lr b r / 2V + C_lδa δa + C_lδr δr)
    m = q̄Sc (C_m0 + C_mα α + C_mq c q / 2V + C_mδe δe)
    n = q̄Sb (C_n0 + C_nβ β + C_np b p / 2V + C_nr b r / 2V + C_nδa δa + C_nδr δr)

σ blends attached flow into flat-plate flow around the stall angle α₀
(sharpness M). Deflections follow the book: positive elevator is trailing
edge down (nose-down moment), positive aileron rolls right wing down in FRD,
positive rudder yaws nose left in FRD. Moments act about ``aero_center_m``
(body axes, metres from the body origin) and are returned about the body
origin, where Gazebo's ApplyLinkWrench applies them (verified with an
offset centre of mass). Below ``MIN_AIRSPEED_MPS`` the wing produces nothing.

``DEFAULT_COEFFICIENTS`` are the Aerosonde values of the book's companion
simulator (mavsim ``aerosonde_parameters.py``); a vehicle preset scales the
geometry. These are plausible teaching values, not a calibrated aircraft.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

MIN_AIRSPEED_MPS = 1.0
SURFACES = ("aileron", "elevator", "rudder")

DEFAULT_COEFFICIENTS = {
    "CL0": 0.23,
    "CLalpha": 5.61,
    "CLq": 7.95,
    "CLde": 0.13,
    "CD0": 0.043,
    "CDp": 0.0,
    "CDq": 0.0,
    "CDde": 0.0135,
    "Cm0": 0.0135,
    "Cmalpha": -2.74,
    "Cmq": -38.21,
    "Cmde": -0.99,
    "CY0": 0.0,
    "CYbeta": -0.98,
    "CYp": 0.0,
    "CYr": 0.0,
    "CYda": 0.075,
    "CYdr": 0.19,
    "Cl0": 0.0,
    "Clbeta": -0.13,
    "Clp": -0.51,
    "Clr": 0.25,
    "Clda": 0.17,
    "Cldr": 0.0024,
    "Cn0": 0.0,
    "Cnbeta": 0.073,
    "Cnp": -0.069,
    "Cnr": -0.095,
    "Cnda": -0.011,
    "Cndr": -0.069,
}


@dataclass(frozen=True)
class AeroState:
    """Air-relative state for one evaluation (diagnostics and logging)."""

    airspeed_mps: float
    alpha_rad: float
    beta_rad: float


def stall_blend(alpha: float, sharpness: float, alpha0: float) -> float:
    """σ(α): 0 in attached flow, 1 when fully stalled (Beard & McLain 4.10)."""
    a = math.exp(-sharpness * (alpha - alpha0))
    b = math.exp(sharpness * (alpha + alpha0))
    return (1 + a + b) / ((1 + a) * (1 + b))


def lift_coefficient(
    alpha: float, c: Mapping[str, float], sharpness: float, alpha0: float
) -> float:
    sigma = stall_blend(alpha, sharpness, alpha0)
    linear = c["CL0"] + c["CLalpha"] * alpha
    plate = 2 * math.copysign(1.0, alpha) * math.sin(alpha) ** 2 * math.cos(alpha)
    return (1 - sigma) * linear + sigma * plate


class FixedWingAero:
    """Wing, tail, and control-surface forces (see module docs)."""

    def __init__(self, settings: Mapping[str, Any], air_density_kg_m3: float) -> None:
        self.enabled: bool = settings["enabled"]
        self._s = settings["wing_area_m2"]
        self._b = settings["wingspan_m"]
        self._c = settings["chord_m"]
        self._e = settings["oswald_efficiency"]
        self._ar = self._b**2 / self._s
        self._ac = np.asarray(settings["aero_center_m"], float)
        self._k = dict(DEFAULT_COEFFICIENTS, **settings["coefficients"])
        self._m = settings["stall_sharpness"]
        self._alpha0 = settings["stall_alpha_rad"]
        self._rho = air_density_kg_m3
        self.last: AeroState | None = None

    def wrench_body(
        self,
        airspeed_body_mps: ArrayLike,
        body_rates_rad_s: ArrayLike,
        deflections_rad: Mapping[str, float],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(force, moment about the body origin), FLU body axes."""
        vx, vy, vz = np.asarray(airspeed_body_mps, float)
        u, v, w = vx, -vy, -vz
        p, q, r = np.asarray(body_rates_rad_s, float) * np.array([1.0, -1.0, -1.0])
        speed = math.sqrt(u * u + v * v + w * w)
        if speed < MIN_AIRSPEED_MPS:
            self.last = AeroState(speed, 0.0, 0.0)
            return np.zeros(3), np.zeros(3)
        alpha, beta = math.atan2(w, u), math.asin(max(-1.0, min(1.0, v / speed)))
        self.last = AeroState(speed, alpha, beta)
        k, s, b, c = self._k, self._s, self._b, self._c
        da, de, dr = (deflections_rad.get(name, 0.0) for name in SURFACES)
        qbar = 0.5 * self._rho * speed**2
        qn, pn, rn = c * q / (2 * speed), b * p / (2 * speed), b * r / (2 * speed)
        linear = k["CL0"] + k["CLalpha"] * alpha
        cd = k["CD0"] + k["CDp"] + linear**2 / (math.pi * self._e * self._ar)
        lift = (
            qbar
            * s
            * (
                lift_coefficient(alpha, k, self._m, self._alpha0)
                + k["CLq"] * qn
                + k["CLde"] * de
            )
        )
        drag = qbar * s * (cd + k["CDq"] * qn + k["CDde"] * abs(de))
        ca, sa = math.cos(alpha), math.sin(alpha)
        fx, fz = ca * -drag - sa * -lift, sa * -drag + ca * -lift
        fy = (
            qbar
            * s
            * (
                k["CY0"]
                + k["CYbeta"] * beta
                + k["CYp"] * pn
                + k["CYr"] * rn
                + k["CYda"] * da
                + k["CYdr"] * dr
            )
        )
        roll = (
            qbar
            * s
            * b
            * (
                k["Cl0"]
                + k["Clbeta"] * beta
                + k["Clp"] * pn
                + k["Clr"] * rn
                + k["Clda"] * da
                + k["Cldr"] * dr
            )
        )
        pitch = (
            qbar
            * s
            * c
            * (k["Cm0"] + k["Cmalpha"] * alpha + k["Cmq"] * qn + k["Cmde"] * de)
        )
        yaw = (
            qbar
            * s
            * b
            * (
                k["Cn0"]
                + k["Cnbeta"] * beta
                + k["Cnp"] * pn
                + k["Cnr"] * rn
                + k["Cnda"] * da
                + k["Cndr"] * dr
            )
        )
        flip = np.array([1.0, -1.0, -1.0])  # FRD → FLU
        force = np.array([fx, fy, fz]) * flip
        moment = np.array([roll, pitch, yaw]) * flip + np.cross(self._ac, force)
        return force, moment

    def wrench_world(
        self,
        velocity_world_mps: ArrayLike,
        wind_world_mps: ArrayLike,
        quaternion_xyzw: ArrayLike,
        body_rates_rad_s: ArrayLike,
        deflections_rad: Mapping[str, float],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(force, moment) in world axes for Gazebo (air = velocity − wind)."""
        body = Rotation.from_quat(quaternion_xyzw)
        air = body.inv().apply(
            np.asarray(velocity_world_mps, float) - np.asarray(wind_world_mps, float)
        )
        force, moment = self.wrench_body(air, body_rates_rad_s, deflections_rad)
        return body.apply(force), body.apply(moment)


class ServoBank:
    """Control-surface servos: travel limit, rate limit, first-order lag."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self._limit = settings["max_deflection_rad"]
        self._rate = settings["rate_limit_rad_s"]
        self._tau = settings["time_constant_s"]
        self.command = {name: 0.0 for name in SURFACES}
        self.position = {name: 0.0 for name in SURFACES}
        self._previous_ns: int | None = None

    def set_command(self, commands: Mapping[str, float]) -> None:
        """Deflection commands (rad), clipped to the travel limit."""
        for name, value in commands.items():
            if name not in self.command:
                raise ValueError(f"Unknown control surface {name!r}")
            self.command[name] = float(np.clip(value, -self._limit, self._limit))

    def update(self, sim_time_ns: int) -> dict[str, float]:
        """Advance the servos to ``sim_time_ns``; return positions (rad)."""
        if self._previous_ns is not None and sim_time_ns > self._previous_ns:
            dt = (sim_time_ns - self._previous_ns) * 1e-9
            alpha = 1 - math.exp(-dt / self._tau) if self._tau > 0 else 1.0
            for name, target in self.command.items():
                step = alpha * (target - self.position[name])
                step = float(np.clip(step, -self._rate * dt, self._rate * dt))
                self.position[name] += step
        self._previous_ns = sim_time_ns
        return dict(self.position)
