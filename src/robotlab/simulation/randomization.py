"""Per-run model mismatch between the simulated and the nominal vehicle.

``randomize_config`` returns the "true" vehicle that Gazebo simulates; the
nominal configuration stays what algorithms know (``AlgorithmContext``), what
the built-in controller allocates with, and what converts thrust commands to
motor speeds. With ``randomization.enabled`` false it returns the nominal
configuration unchanged.

Draws (seeded, fixed order, each N(0, σ²) truncated at ±3σ):

| Quantity | True value |
| --- | --- |
| body mass | nominal × (1 + δ), δ ~ ``mass_fraction_std`` |
| principal inertias | each × (1 + δ), ``inertia_fraction_std`` |
| centre of mass | nominal offset + δ per axis, ``center_of_mass_std_m`` (m) |
| per-rotor thrust coefficient | nominal × scale × (1 + δ), ``rotor_thrust_fraction_std`` |
| per-rotor reaction moment ratio | nominal × scale × (1 + δ), ``rotor_moment_fraction_std`` |
| motor time constants (up, down) | × (1 + δ), ``motor_time_constant_fraction_std`` |
| IMU mount rotation | nominal + δ per axis (deg), ``imu_mount_rotation_std_deg`` |

The draws are returned for ``drone-resolved.json`` so runs can be analysed.
"""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np

from .gazebo.config import rotor_scale


def _normal(rng: np.random.Generator, std: float, size: int | None = None):
    return np.clip(rng.standard_normal(size), -3.0, 3.0) * std


def randomize_data(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """True configuration data and the draws (empty when disabled)."""
    r = data["randomization"]
    if not r["enabled"]:
        return data, {}
    rng = np.random.default_rng(r["seed"])
    true = deepcopy(data)
    v, m = true["vehicle"], true["motors"]
    mass = 1 + _normal(rng, r["mass_fraction_std"])
    inertia = 1 + _normal(rng, r["inertia_fraction_std"], 3)
    com = _normal(rng, 1.0, 3) * np.asarray(r["center_of_mass_std_m"], float)
    count = len(v["rotors"]) or 4
    thrust = 1 + _normal(rng, r["rotor_thrust_fraction_std"], count)
    moment = 1 + _normal(rng, r["rotor_moment_fraction_std"], count)
    tau = 1 + _normal(rng, r["motor_time_constant_fraction_std"], 2)
    imu = _normal(rng, r["imu_mount_rotation_std_deg"], 3)
    v["body_mass_kg"] *= float(mass)
    v["body_inertia_kg_m2"] = (np.asarray(v["body_inertia_kg_m2"]) * inertia).tolist()
    v["center_of_mass_offset_m"] = (
        np.asarray(v["center_of_mass_offset_m"]) + com
    ).tolist()
    m["rotor_thrust_scale"] = (
        rotor_scale(m, "rotor_thrust_scale", count) * thrust
    ).tolist()
    m["rotor_moment_scale"] = (
        rotor_scale(m, "rotor_moment_scale", count) * moment
    ).tolist()
    m["time_constant_up_s"] *= float(tau[0])
    m["time_constant_down_s"] *= float(tau[1])
    mount = true["sensors"]["imu"]["mount"]
    mount["rotation_deg"] = (np.asarray(mount["rotation_deg"]) + imu).tolist()
    draws = dict(
        mass_factor=float(mass),
        inertia_factors=inertia.tolist(),
        center_of_mass_shift_m=com.tolist(),
        rotor_thrust_factors=thrust.tolist(),
        rotor_moment_factors=moment.tolist(),
        motor_time_constant_factors=tau.tolist(),
        imu_mount_rotation_shift_deg=imu.tolist(),
    )
    return true, draws


def randomize_config(config: Any) -> tuple[Any, dict[str, Any]]:
    """(true DroneConfig, draws). Properties follow the true mass and inertia."""
    true_data, draws = randomize_data(config.data)
    if not draws:
        return config, {}
    v = true_data["vehicle"]
    properties = replace(
        config.properties,
        body_mass=v["body_mass_kg"],
        body_inertia=tuple(v["body_inertia_kg_m2"]),
    )
    return replace(config, data=true_data, properties=properties), draws
