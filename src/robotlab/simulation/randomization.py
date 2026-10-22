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

Wheeled robots (rover, car, tractor) use ``ground_vehicle.randomization``
instead (the multirotor section stays disabled); the true
``WheeledProperties`` drive Gazebo and the Python wheel model, the nominal
ones ``context.wheeled``:

| Quantity | True value |
| --- | --- |
| chassis mass | nominal × (1 + δ), ``mass_fraction_std`` |
| principal inertias | from the principal second moments of the mass distribution, S_x = ∫x² dm = (I_yy + I_zz − I_xx)/2 and cyclic, each × (1 + δ), ``inertia_fraction_std``; I_xx = S_y + S_z and cyclic. Scaling the inertias themselves can break the triangle inequality (a flat car body has I_zz ≈ I_xx + I_yy; Gazebo then refuses the world) |
| centre of mass | moved by δ per axis (m), ``center_of_mass_std_m``: the wheels, collision box, and hitch points move by −δ in body axes (the body origin is the centre of mass) |
| tyre friction | longitudinal and lateral μ × (1 + δ), ``tyre_friction_fraction_std`` |
| drive torque and power limits | × (1 + δ), ``drive_torque_fraction_std`` |
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


def randomize_wheeled_data(
    data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """True wheeled configuration data and the draws (empty when disabled)."""
    r = data["ground_vehicle"]["randomization"]
    if not r["enabled"]:
        return data, {}
    rng = np.random.default_rng(r["seed"])
    true = deepcopy(data)
    g = true["ground_vehicle"]
    mass = 1 + _normal(rng, r["mass_fraction_std"])
    inertia = 1 + _normal(rng, r["inertia_fraction_std"], 3)
    com = _normal(rng, 1.0, 3) * np.asarray(r["center_of_mass_std_m"], float)
    friction = 1 + _normal(rng, r["tyre_friction_fraction_std"])
    torque = 1 + _normal(rng, r["drive_torque_fraction_std"])
    chassis = g["chassis"]
    chassis["mass_kg"] *= float(mass)
    chassis["inertia_kg_m2"] = _scaled_inertia(chassis["inertia_kg_m2"], inertia)

    def shifted(point: list[float]) -> list[float]:
        return (np.asarray(point, float) - com).tolist()

    chassis["collision_offset_m"] = shifted(chassis["collision_offset_m"])
    for wheel in g["wheels"]:
        wheel["position_m"] = shifted(wheel["position_m"])
    for section in ("implement", "trailer"):
        g[section]["hitch_position_m"] = shifted(g[section]["hitch_position_m"])
    for direction in ("longitudinal", "lateral"):
        g["tyre"][direction]["friction"] *= float(friction)
    g["drive"]["max_torque_nm"] *= float(torque)
    g["drive"]["max_power_w"] *= float(torque)
    draws = dict(
        mass_factor=float(mass),
        second_moment_factors=inertia.tolist(),
        center_of_mass_shift_m=com.tolist(),
        tyre_friction_factor=float(friction),
        drive_torque_factor=float(torque),
    )
    return true, draws


def _scaled_inertia(inertia_kg_m2: list[float], factors: np.ndarray) -> list[float]:
    """Principal inertias with the second moments S scaled by ``factors``
    (see module docs); raises ValueError for a nominal inertia that
    violates the triangle inequality."""
    i = np.asarray(inertia_kg_m2, float)
    second = (i.sum() - 2 * i) / 2  # S_x = (I_yy + I_zz - I_xx) / 2, cyclic
    if np.any(second < 0):
        raise ValueError(
            "ground_vehicle.chassis.inertia_kg_m2 violates the triangle inequality"
        )
    scaled = second * factors
    return (scaled.sum() - scaled).tolist()  # I_xx = S_y + S_z, cyclic


def randomize_wheeled_config(config: Any) -> tuple[Any, dict[str, Any]]:
    """(true DroneConfig with true ``wheeled`` properties, draws)."""
    from ..core.wheeled_vehicle import WheeledProperties

    true_data, draws = randomize_wheeled_data(config.data)
    if not draws:
        return config, {}
    wheeled = WheeledProperties.from_settings(
        config.wheeled.kind,
        true_data["ground_vehicle"],
        config.wheeled.aero.air_density_kg_m3,
    )
    return replace(config, data=true_data, wheeled=wheeled), draws


def randomize_config(config: Any) -> tuple[Any, dict[str, Any]]:
    """(true DroneConfig, draws). Properties follow the true mass and inertia.
    Wheeled robots: ``randomize_wheeled_config``."""
    if config.wheeled is not None:
        return randomize_wheeled_config(config)
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
