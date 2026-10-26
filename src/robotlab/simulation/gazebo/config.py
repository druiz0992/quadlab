"""Strict, versioned JSON settings for generated Gazebo vehicles.

Noise bias is a deterministic Gaussian mean; stddev is per measurement,
not a continuous-time noise density. No simulator imports are needed here.
"""

from copy import deepcopy
from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any

import numpy as np

from ...core.vehicle import DroneProperties
from ..wheeled.config import (
    GROUND_VEHICLE_DEFAULTS,
    MUJOCO_KINDS,
    ROBOT_KINDS,
    WHEELED_KINDS,
    preset_layer,
    validate_ground_vehicle,
)
from ...core.wheeled_vehicle import WheeledProperties

# Height of the generated world's origin above sea level
# (spherical_coordinates). GPS altitude, barometric altitude, and the "site"
# air density all use this sea-level datum.
WORLD_ELEVATION_M = 650.0


def noise(vector=False):
    return {
        "bias": [0.0] * 3 if vector else 0.0,
        "noise_stddev": [0.0] * 3 if vector else 0.0,
    }


# Settings each calibration depends on (seeds excluded: turn-on biases change
# at every power-up anyway, which the startup gyro calibration handles).
FINGERPRINT_KEYS = {
    "imu": (
        "mount",
        "acceleration_mps2",
        "angular_rate_rad_s",
        "error_model",
        "imperfections",
        "temperature",
    ),
    "magnetometer": ("mount", "field_ut", "error_model"),
}


def sensor_fingerprint(sensors, kind):
    """Short hash of the sensor settings a calibration was made for."""
    section = sensors[kind]
    chosen = {}
    for key in FINGERPRINT_KEYS[kind]:
        value = deepcopy(section[key])
        if isinstance(value, dict):
            value.pop("seed", None)
        chosen[key] = value
    text = json.dumps(chosen, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def stream_timing():
    """Per-stream delivery timing (sensors/bus/__init__.py), ideal by default."""
    return {"latency_s": 0.0, "jitter_s": 0.0, "timestamp_offset_s": 0.0}


def mount(position=(0.0, 0.0, 0.0)):
    return {"position_m": list(position), "rotation_deg": [0.0, 0.0, 0.0]}


def down_mount():
    """Just below the body (half height 0.06 m), +x optical axis pointing down."""
    return {"position_m": [0.0, 0.0, -0.07], "rotation_deg": [0.0, 90.0, 0.0]}


def mount_pose(sensor):
    """Return SDF x/y/z and fixed-axis roll/pitch/yaw in radians."""
    pose = sensor["mount"]
    return [*pose["position_m"], *map(math.radians, pose["rotation_deg"])]


DEFAULTS: dict[str, Any] = {
    "schema_version": 1,
    # Vehicle family: multirotor (the sections below) or a wheeled robot
    # (rover, car, tractor: ``ground_vehicle``); see simulation/wheeled/.
    "robot": {"kind": "multirotor"},
    "ground_vehicle": GROUND_VEHICLE_DEFAULTS,
    "vehicle": {
        "body_mass_kg": 1.5,
        "rotor_mass_kg": 0.01,
        "body_inertia_kg_m2": [0.03, 0.03, 0.05],
        "rotor_inertia_kg_m2": 0.00001,
        "body_size_m": [0.48, 0.48, 0.12],
        "arm_offset_m": 0.2,
        "rotor_height_m": 0.08,  # rotor plane above the body origin
        "propeller_diameter_m": 0.25,  # visuals and ground effect
        # General rotor layout: [{"position_m", "direction" (ccw/cw), "axis"}].
        # Empty: four vertical rotors in an X at ±arm_offset_m, rotor_height_m.
        "rotors": [],
        # Centre of mass relative to the body origin (geometric centre, where
        # odometry is reported and the controller assumes the mass to be).
        "center_of_mass_offset_m": [0.0, 0.0, 0.0],
    },
    "motors": {
        "thrust_coefficient_n_s2": 8.55e-6,
        "reaction_moment_ratio_m": 0.016,
        "max_speed_rad_s": 1000.0,
        "max_thrust_n": None,
        "time_constant_up_s": 0.02,
        "time_constant_down_s": 0.03,
        "rotor_drag_coefficient": 8e-5,
        "rolling_moment_coefficient": 1e-6,
        # Per-rotor manufacturing spread: true thrust coefficient and reaction
        # moment ratio are the nominal values times these (the controller and
        # thrust-to-speed conversion keep using the nominal values).
        "rotor_thrust_scale": [],
        "rotor_moment_scale": [],
        # Figure of merit of non-vertical (pusher) propellers for the
        # momentum-theory power model; see simulation/powertrain.py.
        "forward_figure_of_merit": 0.8,
        # ESC command path and telemetry; see simulation/esc.py and
        # sensors/esc_telemetry.py. null rate / 0 steps: ideal.
        "esc": {
            "update_rate_hz": None,
            "resolution_steps": 0,
            "telemetry": {
                "enabled": False,
                "rate_hz": 100.0,
                "noise_std_rad_s": 0.0,
                "seed": 0,
            },
        },
        # Scheduled motor/propeller failures; see simulation/motor_faults.py.
        "faults": {"events": []},
    },
    "sensors": {
        "imu": {
            "mount": mount(),
            "enabled": True,
            "rate_hz": 200.0,
            "acceleration_mps2": noise(True),
            "angular_rate_rad_s": noise(True),
            "vibration": {
                "enabled": False,
                "frequency_hz": 60.0,
                "accel_amplitude_mps2": [0.0, 0.0, 0.0],
                "phase_deg": [0.0, 0.0, 0.0],
                "random_std_mps2": [0.0, 0.0, 0.0],
                "random_correlation_s": 0.01,
                "clip_limit_mps2": None,
                "seed": 0,
            },
            # Seeded Allan-variance-style errors; see sensors/imu/errors.py.
            "error_model": {
                "enabled": False,
                "seed": 0,
                "accel_noise_density_mps2_per_sqrt_hz": [0.0, 0.0, 0.0],
                "accel_bias_random_walk_mps2_per_sqrt_s": [0.0, 0.0, 0.0],
                "accel_turn_on_bias_std_mps2": [0.0, 0.0, 0.0],
                "gyro_noise_density_rad_s_per_sqrt_hz": [0.0, 0.0, 0.0],
                "gyro_bias_random_walk_rad_s_per_sqrt_s": [0.0, 0.0, 0.0],
                "gyro_turn_on_bias_std_rad_s": [0.0, 0.0, 0.0],
            },
            # Self-heating and bias drift with temperature; see
            # sensors/imu/temperature.py. Ambient comes from environment.
            "temperature": {
                "enabled": False,
                "warmup_rise_c": 10.0,
                "warmup_time_constant_s": 120.0,
                "calibration_c": 25.0,
                "accel_bias_per_c": [0.0, 0.0, 0.0],
                "gyro_bias_per_c": [0.0, 0.0, 0.0],
            },
            # Scale factor, cross-coupling, g-sensitivity, bias instability,
            # saturation, sample drops/outages, timestamp jitter; see
            # sensors/imu/imperfections.py.
            "imperfections": {
                "enabled": False,
                "seed": 0,
                "accel_scale_factor": [0.0, 0.0, 0.0],
                "gyro_scale_factor": [0.0, 0.0, 0.0],
                "accel_cross_coupling": [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                "gyro_cross_coupling": [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                "gyro_g_sensitivity_rad_s_per_mps2": [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                "accel_bias_instability_mps2": [0.0, 0.0, 0.0],
                "gyro_bias_instability_rad_s": [0.0, 0.0, 0.0],
                "bias_instability_correlation_s": 100.0,
                "accel_range_mps2": None,
                "gyro_range_rad_s": None,
                "sample_drop_probability": 0.0,
                "outages_s": [],
                "timestamp_jitter_s": 0.0,
            },
            # Calibration the autopilot applies (not truth); produced by
            # `robotlab-imucal`. See estimation/imu_calibration.py.
            "calibration": {
                "enabled": False,
                "accel_offset_mps2": [0.0, 0.0, 0.0],
                "accel_correction": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "accel_temperature_mps2_per_c": [0.0, 0.0, 0.0],
                "gyro_offset_rad_s": [0.0, 0.0, 0.0],
                "gyro_g_sensitivity": [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                "gyro_temperature_rad_s_per_c": [0.0, 0.0, 0.0],
                "reference_temperature_c": 25.0,
                # Average the gyro this long after power-up (vehicle still on
                # the ground; take-off waits). 0: off.
                "startup_gyro_calibration_s": 0.0,
                # Remaining bias uncertainty after this calibration, for
                # filters' initial bias covariance (1-σ); null: filter default.
                "filter_accel_bias_std_mps2": None,
                "filter_gyro_bias_std_rad_s": None,
                # Settings the calibration was made for (sensor_fingerprint).
                "fingerprint": "",
            },
            # Scheduled malfunctions; see sensors/imu/faults.py.
            "faults": {"seed": 0, "events": []},
        },
        "gps": {
            "mount": mount(),
            "enabled": True,
            "rate_hz": 10.0,
            # zones: world boxes where every fix is lost (GPS blockout).
            "dropout": {"probability": 0.0, "seed": 0, "outages_s": [], "zones": []},
            # Position-jump episodes; see sensors/gnss/multipath.py.
            "multipath": {
                "enabled": False,
                "seed": 0,
                "zones": [],
                "onset_rate_hz": 0.0,
                "duration_s": 3.0,
                "horizontal_std_m": 0.0,
                "vertical_std_m": 0.0,
            },
            # Seeded time-correlated ENU position error; see sensors/gnss/errors.py.
            "error_model": {
                "enabled": False,
                "seed": 0,
                "horizontal_std_m": 0.0,
                "vertical_std_m": 0.0,
                "correlation_time_s": 60.0,
                "white_horizontal_std_m": 0.0,
                "white_vertical_std_m": 0.0,
            },
            "horizontal_position_m": noise(),
            "vertical_position_m": noise(),
            "horizontal_velocity_mps": noise(),
            "vertical_velocity_mps": noise(),
            # Receiver models; see sensors/gnss/receiver.py.
            "velocity_error": {
                "enabled": False,
                "seed": 0,
                "white_horizontal_std_mps": 0.0,
                "white_vertical_std_mps": 0.0,
                "correlated_std_mps": 0.0,
                "correlation_time_s": 10.0,
            },
            "receiver": {
                "seed": 0,
                "satellites": 14,
                "satellite_variation": 2,
                "multipath_satellite_loss": 5,
                "hdop": 0.9,
                # "auto": reported accuracy follows the configured GPS error.
                "uere_m": "auto",
                "min_accuracy_m": 0.05,
                "speed_accuracy_mps": 0.1,
            },
            "rtk": {
                "enabled": False,
                "seed": 0,
                "convergence_s": 30.0,
                "float_std_m": 0.3,
                "fixed_std_m": 0.015,
                "fixed_vertical_std_m": 0.03,
            },
            "heading": {
                "enabled": False,
                "seed": 0,
                "noise_std_deg": 0.3,
                "mount_yaw_deg": 0.0,
            },
        },
        "contacts": {"enabled": True, "rate_hz": 100.0},
        # Barometer (Gazebo air-pressure sensor); see sensors/barometer.py.
        "baro": {
            "mount": mount(),
            "enabled": False,
            "rate_hz": 50.0,
            "pressure_pa": noise(),  # Gazebo native Gaussian noise, in Pa
            "error_model": {
                "enabled": False,
                "seed": 0,
                "offset_m": 0.0,
                "drift_std_m": 0.0,
                "drift_correlation_time_s": 300.0,
                "white_std_m": 0.0,
                # Static-port airflow errors: coefficient · v² / 2g metres for
                # rotor downwash (rotor-thrust mode) and horizontal airspeed.
                "prop_wash_coefficient": 0.0,
                "airspeed_coefficient": 0.0,
            },
        },
        "camera": {
            "mount": mount((0.3, 0.0, 0.1)),
            "enabled": False,
            "rate_hz": 10.0,
            "width": 320,
            "height": 240,
            "horizontal_fov_rad": 1.047,
            "near_m": 0.1,
            "far_m": 100.0,
            "intensity": noise(),
            # pinhole, or a Gazebo wide-angle projection (fisheye, FOV up to 2π);
            # Brown–Conrady coefficients are applied in Python (camera_effects.py).
            "lens": {
                "type": "pinhole",
                "k1": 0.0,
                "k2": 0.0,
                "k3": 0.0,
                "p1": 0.0,
                "p2": 0.0,
            },
            "exposure_time_s": 0.0,  # motion blur
            "rolling_shutter_readout_s": 0.0,
            "latency_s": 0.0,  # frames delivered late (simulation time)
        },
        "lidar": {
            "mount": mount((0.0, 0.0, 0.2)),
            "enabled": False,
            "rate_hz": 10.0,
            "samples": 360,
            # More than one vertical sample makes a multi-channel 3D lidar.
            "vertical_samples": 1,
            "vertical_min_angle_deg": 0.0,
            "vertical_max_angle_deg": 0.0,
            "min_range_m": 0.15,
            "max_range_m": 50.0,
            "range_m": noise(),
            # Range noise, dropouts, motion distortion; see sensors/imaging/lidar_effects.py.
            "effects": {
                "seed": 0,
                "noise_std_m": 0.0,
                "noise_per_m": 0.0,
                "dropout_probability": 0.0,
                "far_dropout_exponent": 0.0,
                "scan_time_s": 0.0,
                # Mixed-pixel ghosts at depth edges; see sensors/imaging/lidar_returns.py.
                "beam_divergence_rad": 0.0,
                # Surfaces closer than this merge into one (ghost) return;
                # null: always merge. Farther ones become separate returns.
                "range_resolution_m": None,
                # Which returns per beam: first, last, strongest, or dual
                # (first + strongest).
                "return_mode": "first",
                "edge_min_jump_m": 0.3,
                # Drop returns too weak to detect (reflectivity, range, weather):
                # detection limit is reference_range_m at reference_reflectivity.
                "detection": {
                    "enabled": False,
                    "reference_range_m": 40.0,
                    "reference_reflectivity": 0.1,
                    # Surfaces without a scene reflectivity (Gazebo retro 0).
                    "default_reflectivity": 0.5,
                },
            },
        },
        # Gazebo magnetometer; see sensors/magnetometer/errors.py. Field in µT.
        "magnetometer": {
            "mount": mount(),
            "enabled": False,
            "rate_hz": 50.0,
            "field_ut": noise(True),  # Gazebo native Gaussian noise, µT
            "error_model": {
                "enabled": False,
                "seed": 0,
                "hard_iron_ut": [0.0, 0.0, 0.0],
                "soft_iron": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "motor_interference_ut_per_n": [0.0, 0.0, 0.0],
                "current_interference_ut_per_a": [0.0, 0.0, 0.0],
                "drift_std_ut": [0.0, 0.0, 0.0],
                "drift_correlation_s": 300.0,
                "white_std_ut": [0.0, 0.0, 0.0],
            },
            # Point-dipole anomalies (structures, pipes); see
            # sensors/magnetometer/anomalies.py. Scene obstacles may add their own.
            "dipoles": [],
            # Calibration the autopilot applies and filters use (not truth);
            # produced by `robotlab-magcal`. See
            # estimation/magnetometer_calibration.py. Local field: Gazebo's
            # table for the default site (as a geomagnetic model would give).
            "calibration": {
                "enabled": False,
                "hard_iron_offset_ut": [0.0, 0.0, 0.0],
                "soft_iron_correction": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "current_compensation_ut_per_a": [0.0, 0.0, 0.0],
                "thrust_compensation_ut_per_n": [0.0, 0.0, 0.0],
                "field_strength_ut": 45.1,
                "inclination_deg": 55.4,
                "declination_deg": 0.0,
                "field_tolerance_ut": 5.0,
                "heading_std_deg": 3.0,
                "fingerprint": "",
            },
        },
        # Downward single-ray Gazebo lidar (requires rendering).
        # Synthetic pitot-static airspeed along body +x; see sensors/airspeed.py.
        "airspeed": {
            "enabled": False,
            "rate_hz": 50.0,
            "scale": 1.0,
            "offset_pa": 0.0,
            "noise_std_pa": 0.0,
            "seed": 0,
            "outages_s": [],  # blockage intervals [start, end]
        },
        "rangefinder": {
            "mount": down_mount(),
            "enabled": False,
            "rate_hz": 20.0,
            "min_range_m": 0.1,
            "max_range_m": 12.0,
            "range_m": noise(),
        },
        # Power module on the simulated pack; see sensors/battery_monitor.py.
        "battery_monitor": {
            "enabled": False,
            "rate_hz": 10.0,
            "seed": 0,
            "voltage_noise_std_v": 0.0,
            "current_noise_std_a": 0.0,
            "current_offset_a": 0.0,
        },
        # Wheeled robots: incremental wheel encoders and steering-angle
        # sensors; see sensors/wheel_encoders.py (multirotors: unused).
        "wheel_encoders": {
            "enabled": False,
            "rate_hz": 100.0,
            "counts_per_revolution": 1024,
            "steering_noise_std_rad": 0.0,
            "seed": 0,
        },
        # Synthetic flow over flat ground; see sensors/optical_flow.py.
        "optical_flow": {
            "mount": down_mount(),
            "enabled": False,
            "rate_hz": 50.0,
            "seed": 0,
            "min_distance_m": 0.08,
            "max_distance_m": 8.0,
            "max_rate_rad_s": 7.0,
            "noise_std_rad_s": 0.0,
        },
    },
    "wind": {
        "enabled": False,
        "velocity_enu_mps": [0.0, 0.0, 0.0],
        "force_scale_per_s": 0.1,
        "rise_time_s": 1.0,
        "gust_amplitude_fraction": 0.0,
        "gust_period_s": 10.0,
        "random_seed": 0,
        "random_update_period_s": 0.5,
        "random_speed_std_mps": 0.0,
        "random_direction_std_deg": 0.0,
        "schedule": [],
        # Apply WindEffects linear drag relative to still air when wind is
        # disabled (with wind enabled, drag relative to the air already applies).
        "still_air_drag": False,
        # Dryden-style turbulence; see simulation/aerodynamics.py.
        "turbulence": {
            "enabled": False,
            "seed": 0,
            "intensity_mps": [0.0, 0.0, 0.0],
            "length_scale_m": [20.0, 20.0, 5.0],
            "update_period_s": 0.05,
            "min_advection_speed_mps": 1.0,
        },
    },
    # Payload hanging below the body, droppable and re-attachable; see
    # simulation/gazebo/payload.py. The controller does not know its mass.
    "payload": {
        "enabled": False,
        "mass_kg": 0.3,
        "size_m": [0.12, 0.12, 0.08],
        "offset_m": [0.0, 0.0, -0.12],  # payload centre from body origin
        "drop_s": None,  # scheduled drop time (s), or null
        "pickup_radius_m": 0.3,
        # Swinging payload on a winch cable (spherical pendulum of variable
        # length); offset_m is then unused. See simulation/gazebo/payload.py.
        "tether": {
            "enabled": False,
            "anchor_m": [0.0, 0.0, -0.07],  # winch point below the body
            "initial_length_m": 1.0,
            "max_length_m": 3.0,
            "winch_speed_mps": 0.5,
            "swing_damping": 0.0005,  # ball-joint damping, N·m·s/rad
            "length_schedule": [],  # [{"time_s", "length_m"}, ...]
        },
    },
    # Low-battery failsafe (rotor-thrust mode); see experiments/multirotor/failsafe.py.
    "failsafe": {
        "low_battery": {
            "enabled": False,
            "action": "land",  # warn | land | return_and_land
            "warn_remaining_pct": 30.0,
            "land_remaining_pct": 15.0,
            "critical_cell_voltage_v": 3.3,  # land immediately below this
            "descent_mps": 0.5,
            "return_speed_mps": 3.0,
            "home_radius_m": 1.0,
            "touchdown_height_m": 0.05,  # body bottom above ground
            "touchdown_speed_mps": 0.1,
            "touchdown_hold_s": 0.5,
        }
    },
    # Shared environment: ambient temperature for battery and IMU models.
    "environment": {
        "ambient_temperature_c": 25.0,
        # Rain, dust, fog for the lidar; see sensors/imaging/lidar_returns.py.
        "weather": {
            "visibility_m": None,  # null: clear air
            "lidar_clutter_per_m": 0.0,  # false returns per beam per metre
            "lidar_clutter_max_range_m": 10.0,
            "lidar_clutter_reflectivity": 0.05,
        },
    },
    # LiPo pack limiting motor speed (rotor-thrust mode); see simulation/battery.py.
    "battery": {
        "enabled": False,
        "cells": 4,
        "capacity_mah": 5000.0,
        "internal_resistance_ohm": 0.02,
        "initial_soc": 1.0,
        "infinite": False,  # charge never drops; voltage still sags under load
        # Pack temperature: I²R heating, cooling to ambient, cold raises
        # resistance and lowers capacity; see simulation/battery.py.
        "thermal": {
            "enabled": False,
            "heat_capacity_j_per_k": 300.0,
            "cooling_w_per_k": 1.5,
            "resistance_coefficient_per_c": 0.028,
            "capacity_coefficient_per_c": 0.006,
        },
        "powertrain_efficiency": 0.7,
        "avionics_power_w": 5.0,
    },
    # Effects applied to motor speeds (rotor-thrust mode); see simulation/powertrain.py.
    # Wing, tail, and control surfaces (rotor-thrust mode); see
    # simulation/fixed_wing.py. Defaults: Aerosonde geometry; `coefficients`
    # overrides individual DEFAULT_COEFFICIENTS entries.
    "fixed_wing": {
        "enabled": False,
        "wing_area_m2": 0.55,
        "wingspan_m": 2.8956,
        "chord_m": 0.18994,
        "oswald_efficiency": 0.9,
        "aero_center_m": [0.0, 0.0, 0.0],
        "stall_sharpness": 50.0,
        "stall_alpha_rad": 0.47,
        "coefficients": {},
        "servos": {
            "max_deflection_rad": 0.44,
            "rate_limit_rad_s": 6.0,
            "time_constant_s": 0.03,
        },
    },
    "aerodynamics": {
        "air_density_kg_m3": "site",
        # Uses vehicle.propeller_diameter_m and vehicle.rotor_height_m.
        "ground_effect": {"enabled": False},
        # Quadratic drag via Gazebo ApplyLinkWrench; Cd·A per body axis (m²).
        "body_drag": {"enabled": False, "drag_area_m2": [0.012, 0.012, 0.03]},
        # Rotor thrust effects (rotor-thrust mode); see simulation/aerodynamics.py.
        "translational_lift": {"enabled": False, "max_gain": 0.15},
        "vortex_ring": {
            "enabled": False,
            "thrust_loss": 0.3,
            "fluctuation_std": 0.1,
            "seed": 0,
        },
    },
    "ekf": {
        "initial_position_m": [0.0, 0.0, 0.0],
        "initial_velocity_mps": [0.0, 0.0, 0.0],
        "initial_rotation_deg": [0.0, 0.0, 0.0],
        "initial_accel_bias_mps2": [0.0, 0.0, 0.0],
        "initial_gyro_bias_rad_s": [0.0, 0.0, 0.0],
        "initial_position_std_m": 5.0,
        "initial_velocity_std_mps": 1.0,
        "initial_attitude_std_deg": 10.0,
        "initial_accel_bias_std_mps2": 0.2,
        "initial_gyro_bias_std_rad_s": 0.02,
        "accel_std_mps2": 0.08,
        "gyro_std_rad_s": 0.003,
        "accel_bias_walk": 0.001,
        "gyro_bias_walk": 0.0001,
        "gps_std_m": [0.5, 0.5, 0.8],
        "gps_gate_squared": 25.0,
        # First time-aligned GPS fix sets the position (and its covariance).
        "initialize_position_from_gps": True,
        # GPS covariance from the fix's own GpsStatus accuracies when present.
        "use_receiver_gps_accuracy": True,
    },
    "control": {
        "publish_rate_hz": 50.0,
        "command_lifetime_sim_s": 0.1,
        "watchdog_wall_s": 1.0,
        "odometry_rate_hz": 50.0,
        # Rotor-thrust controller timing; see control/timing.py.
        "control_rate_hz": 50.0,
        "position_rate_hz": None,  # null: run the position loop every tick
        "state_latency_s": 0.0,
        "command_latency_s": 0.0,
        "publish_on_send": False,
    },
    "controller": {
        "position_gain": 1.2,
        "velocity_gain": 2.2,
        "attitude_gain": [1.8, 1.8, 0.8],
        "angular_rate_gain": [0.35, 0.35, 0.25],
        "max_acceleration_mps2": 2.0,
        "max_position_error_m": 1.5,
        "min_altitude_m": 0.4,
    },
    "velocity_controller": {
        "velocity_gain": [3.0, 3.0, 4.0],
        "attitude_gain": [2.0, 2.0, 0.15],
        "angular_rate_gain": [0.4, 0.4, 0.18],
        "max_acceleration_mps2": [2.0, 2.0, 2.0],
    },
    # What keys 1-9 and 0 do during keyboard flight; see experiments/multirotor/effect_toggles.py.
    "keyboard_effects": {
        "motor_stop_rotor": 0,
        "damaged_rotor": 1,
        "damaged_thrust_fraction": 0.7,
        "gps_multipath_offset_enu_m": [4.0, -3.0, 2.0],
        "gyro_bias_step_rad_s": [0.0, 0.0, 0.02],
        # Key 7: object circling the drone's position at the key press.
        "flying_object_size_m": [0.5, 0.5, 0.3],
        "flying_object_radius_m": 3.0,
        "flying_object_speed_mps": 2.0,
        # Keys 9 (drain) and 0 (refill to full); state-of-charge fraction per press.
        "battery_drain_fraction": 0.1,
        # Keys [ (colder) and ] (warmer): ambient change per press, °C. The
        # battery pack and IMU die jump by the same amount.
        "ambient_step_c": 10.0,
    },
    # Per-run model mismatch: the simulated ("true") vehicle differs from the
    # nominal one the algorithms know; see simulation/randomization.py.
    # Fractions and angles are 1-σ; draws are truncated at 3σ.
    "randomization": {
        "enabled": False,
        "seed": 0,
        "mass_fraction_std": 0.0,
        "inertia_fraction_std": 0.0,
        "center_of_mass_std_m": [0.0, 0.0, 0.0],
        "rotor_thrust_fraction_std": 0.0,
        "rotor_moment_fraction_std": 0.0,
        "motor_time_constant_fraction_std": 0.0,
        "imu_mount_rotation_std_deg": 0.0,
    },
    # Estimator/controller plug-ins and what the controller flies on; see
    # experiments/harness/algorithms.py. Names: "" (none / built-in controller), a
    # built-in name, or "package.module:Class".
    "algorithms": {
        "estimator": "",
        "estimator_options": {},
        "controller": "",
        "controller_options": {},
        "control_state_source": "truth",  # truth | estimate
        "max_estimate_age_s": 0.1,
        # Extra estimators run beside "estimator" on the same deliveries and
        # scored the same way, never driving control: a list of {"name",
        # "estimator", "estimator_options"}; see experiments/harness/observers.py.
        "observers": [],
        # Scoring against truth (evaluation/), after initial convergence.
        "evaluation_rate_hz": 50.0,
        "evaluation_start_s": 2.0,
        # Record bus deliveries and truth for robotlab-replay:
        # off | light (all but camera/lidar) | all; see experiments/harness/replay_log.py.
        "record_bus": "off",
    },
    # Run health verdict thresholds (summary.json.health); see
    # evaluation/health.py. Evaluation only: never affects the flight.
    "health": {
        "flip_tilt_deg": 90.0,
        "flip_duration_s": 0.5,
        "hard_impact_speed_mps": 2.0,
        "impact_window_s": 0.1,
        "below_ground_margin_m": 0.1,
        "touchdown_gap_s": 0.5,
        "saturated_fraction": 0.5,
        "max_dropped_messages": 0,
        "no_liftoff_s": 1.0,
        "no_liftoff_thrust_ratio": 1.2,
        "tracking_lost_m": 2.0,
        "tracking_lost_s": 3.0,
    },
    # Delivery of processed measurements to algorithms; see sensors/bus/__init__.py.
    "sensor_timing": {
        "seed": 0,
        "reorder_window_s": 0.004,
        "imu": stream_timing(),
        "gps": stream_timing(),
        "gps_heading": stream_timing(),
        "baro": stream_timing(),
        "magnetometer": stream_timing(),
        "rangefinder": stream_timing(),
        "optical_flow": stream_timing(),
        "battery_monitor": stream_timing(),
        "esc": stream_timing(),
        "lidar": stream_timing(),
        "camera": stream_timing(),
        "wheel_encoders": stream_timing(),
    },
    # Reference trajectory for rotor-thrust missions; see control/references.py.
    # Each type uses a subset of the keys. Mission time starts at the first
    # state, or when take-off completes after a ground start.
    "mission": {
        "type": "none",
        "center_m": [0.0, 0.0, 1.0],
        "yaw_deg": 0.0,
        "yaw_mode": "fixed",
        "step_m": [1.0, 0.0, 0.0],
        "step_time_s": 2.0,
        "waypoints_m": [],
        "speed_mps": 1.0,
        "dwell_s": 1.0,
        "laps": 1,
        "yaw_rate_dps": 45.0,
        "radius_m": 2.0,
        "axis": "x",
        "amplitude_m": 0.5,
        "frequency_start_hz": 0.1,
        "frequency_end_hz": 1.0,
        "duration_s": 30.0,
        "pulse_s": 2.0,
    },
    "pilot": {
        "forward_mps": 1.0,
        "yaw_rate_rad_s": 0.7,
        "climb_mps": 0.6,
        # Automatic take-off when simulation.start_on_ground (rotor-thrust mode).
        "takeoff_altitude_m": 1.0,
        "takeoff_climb_mps": 0.5,
    },
    "simulation": {
        "step_s": 0.002,
        # Real-time pacing; in lockstep a speed cap, where 0 = as fast as possible.
        "real_time_factor": 1.0,
        # Python advances a paused Gazebo per odometry period; see
        # simulation/gazebo/stepping.py.
        "lockstep": False,
        # How Gazebo waits for Python in lockstep: auto | plugin | service;
        # see simulation/gazebo/plugins.py.
        "lockstep_barrier": "auto",
        "spawn_position_m": [0.0, 0.0, 1.0],
        # Rest on the ground at spawn x, y (z ignored) and take off automatically.
        "start_on_ground": False,
    },
}


# Keys allowing zero (zero disables that error term or delay).
IMU_ERROR_KEYS = {
    "accel_noise_density_mps2_per_sqrt_hz",
    "accel_bias_random_walk_mps2_per_sqrt_s",
    "accel_turn_on_bias_std_mps2",
    "gyro_noise_density_rad_s_per_sqrt_hz",
    "gyro_bias_random_walk_rad_s_per_sqrt_s",
    "gyro_turn_on_bias_std_rad_s",
}
# Gazebo projections: pinhole uses the camera sensor; others the wideanglecamera.
LENS_TYPES = (
    "pinhole",
    "equidistant",
    "stereographic",
    "equisolid_angle",
    "orthographic",
)
# Keys allowing any sign.
SIGNED_SCALAR_KEYS = {
    "reference_temperature_c",
    "inclination_deg",
    "declination_deg",
    "offset_pa",
    "mount_yaw_deg",
    "timestamp_offset_s",
    "yaw_deg",
    "ambient_temperature_c",
    "calibration_c",
    "prop_wash_coefficient",
    "airspeed_coefficient",
    "k1",
    "k2",
    "k3",
    "p1",
    "p2",
    "current_offset_a",
    "offset_m",
    "vertical_min_angle_deg",
    "vertical_max_angle_deg",
}
NONNEGATIVE_SCALAR_KEYS = {
    "steering_noise_std_rad",
    "flip_duration_s",
    "impact_window_s",
    "below_ground_margin_m",
    "touchdown_gap_s",
    "max_dropped_messages",
    "startup_gyro_calibration_s",
    "noise_std_pa",
    "time_constant_s",
    "white_horizontal_std_mps",
    "white_vertical_std_mps",
    "correlated_std_mps",
    "satellite_variation",
    "multipath_satellite_loss",
    "sample_drop_probability",
    "timestamp_jitter_s",
    "resolution_steps",
    "noise_std_rad_s",
    "mass_fraction_std",
    "inertia_fraction_std",
    "rotor_thrust_fraction_std",
    "rotor_moment_fraction_std",
    "motor_time_constant_fraction_std",
    "imu_mount_rotation_std_deg",
    "jitter_s",
    "reorder_window_s",
    "evaluation_start_s",
    "step_time_s",
    "dwell_s",
    "beam_divergence_rad",
    "lidar_clutter_per_m",
    "resistance_coefficient_per_c",
    "capacity_coefficient_per_c",
    "warmup_rise_c",
    "swing_damping",
    "battery_drain_fraction",
    "noise_std_m",
    "noise_per_m",
    "dropout_probability",
    "far_dropout_exponent",
    "scan_time_s",
    "exposure_time_s",
    "rolling_shutter_readout_s",
    "latency_s",
    "max_gain",
    "thrust_loss",
    "fluctuation_std",
    "motor_stop_rotor",
    "damaged_rotor",
    "damaged_thrust_fraction",
    "voltage_noise_std_v",
    "current_noise_std_a",
    "real_time_factor",
    "noise_std_rad_s",
    "avionics_power_w",
    "drift_std_m",
    "white_std_m",
    "state_latency_s",
    "command_latency_s",
    "horizontal_std_m",
    "vertical_std_m",
    "white_horizontal_std_m",
    "white_vertical_std_m",
    "onset_rate_hz",
}


def validate_length_schedule(events, name):
    """Winch targets: strictly increasing times, positive lengths."""
    if not isinstance(events, list):
        raise ValueError(f"{name} must be a list")
    previous = -1.0
    for event in events:
        if not isinstance(event, dict) or set(event) != {"time_s", "length_m"}:
            raise ValueError(f"{name} entries require time_s and length_m")
        number(event["time_s"], name, nonnegative=True)
        number(event["length_m"], name, positive=True)
        if event["time_s"] <= previous:
            raise ValueError(f"{name} times must strictly increase")
        previous = event["time_s"]


MATRIX_KEYS = {
    "accel_correction",
    "gyro_g_sensitivity",
    "soft_iron",
    "soft_iron_correction",
    "accel_cross_coupling",
    "gyro_cross_coupling",
    "gyro_g_sensitivity_rad_s_per_mps2",
}


def validate_matrix3(matrix, name):
    """A 3×3 matrix of finite numbers, given as three rows."""
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise ValueError(f"{name} must be a 3x3 matrix (three rows)")
    for row in matrix:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"{name} must be a 3x3 matrix (three rows)")
        for value in row:
            number(value, name)


def validate_new_sensors(data):
    """Range, angle, and rate checks for lidar channels, rangefinder, and flow."""
    s = data["sensors"]
    lidar = s["lidar"]
    low, high = lidar["vertical_min_angle_deg"], lidar["vertical_max_angle_deg"]
    if not -90 <= low <= high <= 90:
        raise ValueError("Lidar vertical angles must satisfy -90 <= min <= max <= 90")
    if lidar["effects"]["dropout_probability"] > 1:
        raise ValueError("lidar effects dropout_probability must be in [0, 1]")
    if lidar["effects"]["scan_time_s"] > 1:
        raise ValueError("lidar effects scan_time_s must be at most 1 s")
    if lidar["vertical_samples"] > 1 and low == high:
        raise ValueError("Multi-channel lidar requires vertical_min < vertical_max")
    if s["rangefinder"]["min_range_m"] >= s["rangefinder"]["max_range_m"]:
        raise ValueError("rangefinder: minimum range must be less than maximum")
    flow = s["optical_flow"]
    if flow["min_distance_m"] >= flow["max_distance_m"]:
        raise ValueError("optical_flow: minimum distance must be less than maximum")
    if flow["rate_hz"] > data["control"]["odometry_rate_hz"]:
        raise ValueError("optical_flow rate_hz cannot exceed odometry_rate_hz")
    monitor = s["battery_monitor"]
    if monitor["enabled"] and not data["battery"]["enabled"]:
        raise ValueError("sensors.battery_monitor requires battery.enabled")
    if monitor["rate_hz"] > data["control"]["odometry_rate_hz"]:
        raise ValueError("battery_monitor rate_hz cannot exceed odometry_rate_hz")


ROTOR_KEYS = {"position_m", "direction", "axis"}


def validate_rotors(rotors, name):
    """General rotor layout entries (see the vehicle section defaults)."""
    if not isinstance(rotors, list):
        raise ValueError(f"{name} must be a list of rotors")
    for rotor in rotors:
        if not isinstance(rotor, dict) or set(rotor) != ROTOR_KEYS:
            raise ValueError(f"{name} entries need exactly {sorted(ROTOR_KEYS)}")
        if rotor["direction"] not in ("ccw", "cw"):
            raise ValueError(f"{name} direction must be ccw or cw")
        validate_points([rotor["position_m"], rotor["axis"]], name)
        if not 0.99 < math.hypot(*rotor["axis"]) < 1.01:
            raise ValueError(f"{name} axis must be a unit vector")


def rotor_layout(vehicle):
    """(rotors as (x, y, dir), positions, unit axes) for the vehicle section."""
    if not vehicle["rotors"]:
        arm, height = vehicle["arm_offset_m"], vehicle["rotor_height_m"]
        rotors = ((arm, arm, 1), (-arm, -arm, 1), (arm, -arm, -1), (-arm, arm, -1))
        return (
            rotors,
            tuple((x, y, height) for x, y, _ in rotors),
            ((0.0, 0.0, 1.0),) * 4,
        )
    rotors, positions, axes = [], [], []
    for rotor in vehicle["rotors"]:
        x, y, z = map(float, rotor["position_m"])
        axis = np.asarray(rotor["axis"], float)
        rotors.append((x, y, 1 if rotor["direction"] == "ccw" else -1))
        positions.append((x, y, z))
        axes.append(tuple((axis / np.linalg.norm(axis)).tolist()))
    return tuple(rotors), tuple(positions), tuple(axes)


def rotor_scale(motors, key, count):
    """Per-rotor scale factors; an empty list means 1 for every rotor."""
    values = motors[key]
    return np.ones(count) if not values else np.asarray(values, float)


def validate_dipoles(dipoles, name):
    """A list of {position_m: [x, y, z], moment_am2: [mx, my, mz]}."""
    if not isinstance(dipoles, list):
        raise ValueError(f"{name} must be a list of dipoles")
    for dipole in dipoles:
        if not isinstance(dipole, dict) or set(dipole) != {"position_m", "moment_am2"}:
            raise ValueError(f"{name} entries need exactly position_m and moment_am2")
        validate_points([dipole["position_m"], dipole["moment_am2"]], name)


OBSERVER_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")


def validate_observers(observers, name):
    """``algorithms.observers``: objects with a unique directory-safe
    ``name``, a nonempty ``estimator`` spec, and optional
    ``estimator_options`` (an object, validated by the estimator)."""
    if not isinstance(observers, list):
        raise ValueError(f"{name} must be a list of observer estimators")
    names = set()
    for observer in observers:
        if not isinstance(observer, dict) or not {"name", "estimator"} <= set(observer):
            raise ValueError(f"{name} entries need name and estimator")
        unknown = set(observer) - {"name", "estimator", "estimator_options"}
        if unknown:
            raise ValueError(f"Unknown {name} keys: {', '.join(sorted(unknown))}")
        label = observer["name"]
        if not isinstance(label, str) or not OBSERVER_NAME.match(label):
            raise ValueError(f"{name} name {label!r}: use letters, digits, _ -")
        if label in names or label == "primary":
            raise ValueError(f"{name} names must be unique, not primary: {label!r}")
        names.add(label)
        if not isinstance(observer["estimator"], str) or not observer["estimator"]:
            raise ValueError(f"{name} {label}: estimator must be a nonempty string")
        if not isinstance(observer.get("estimator_options", {}), dict):
            raise ValueError(f"{name} {label}: estimator_options must be an object")


def validate_points(points, name):
    """A list of finite [x, y, z] points (may be empty)."""
    if not isinstance(points, list):
        raise ValueError(f"{name} must be a list of [x, y, z] points")
    for point in points:
        if not isinstance(point, list) or len(point) != 3:
            raise ValueError(f"{name} entries must be [x, y, z]")
        for value in point:
            number(value, name)


def validate_zones(zones, name):
    """Zones are lists of {min_m, max_m} world boxes with min < max per axis."""
    if not isinstance(zones, list):
        raise ValueError(f"{name} must be a list of zones")
    for zone in zones:
        if not isinstance(zone, dict) or set(zone) != {"min_m", "max_m"}:
            raise ValueError(f"{name} entries require exactly min_m and max_m")
        validate_tree(zone, {"min_m": [0.0] * 3, "max_m": [0.0] * 3}, name)
        if any(low >= high for low, high in zip(zone["min_m"], zone["max_m"])):
            raise ValueError(f"{name} requires min_m < max_m on every axis")


def validate_motor_fault_events(events, name):
    """Type, rotor index, window, and value range for motor fault events."""
    from ..motor_faults import EVENT_TYPES

    if not isinstance(events, list):
        raise ValueError(f"{name} must be a list of motor fault events")
    for event in events:
        if not isinstance(event, dict) or event.get("type") not in EVENT_TYPES:
            raise ValueError(f"{name} type must be one of {', '.join(EVENT_TYPES)}")
        expected = {"type", "rotor", "start_s", "end_s"}
        if event["type"] != "stop":
            expected.add("value")
        if set(event) != expected:
            raise ValueError(
                f"{name} {event['type']} events require {sorted(expected)}"
            )
        # Upper bound (rotor count) is checked once the layout is known.
        if type(event["rotor"]) is not int or event["rotor"] < 0:
            raise ValueError(f"{name} rotor must be a nonnegative integer")
        number(event["start_s"], name, nonnegative=True)
        if event["end_s"] is not None:
            number(event["end_s"], name, positive=True)
            if event["end_s"] <= event["start_s"]:
                raise ValueError(f"{name} requires end_s > start_s")
        if "value" in event:
            number(event["value"], name, nonnegative=True)
            upper_ok = event["value"] < 1
            lower_ok = event["value"] > 0 or event["type"] == "partial_thrust"
            if not (upper_ok and lower_ok):
                raise ValueError(
                    f"{name} value must be in [0, 1) for partial_thrust, (0, 1) for speed_limit"
                )


def validate_imu_fault_events(events, name):
    """Check type-specific keys, instrument, window, and parameter ranges."""
    from ...sensors.imu.faults import EVENT_TYPES, INSTRUMENTS

    if not isinstance(events, list):
        raise ValueError(f"{name} must be a list of fault events")
    for event in events:
        if not isinstance(event, dict) or event.get("type") not in EVENT_TYPES:
            raise ValueError(
                f"{name} event type must be one of {', '.join(EVENT_TYPES)}"
            )
        kind = event["type"]
        parameter = {"stuck": {"axes"}, "spikes": {"value", "probability"}}.get(
            kind, {"value"}
        )
        expected = {"type", "instrument", "start_s", "end_s"} | parameter
        if set(event) != expected:
            raise ValueError(f"{name} {kind} events require exactly {sorted(expected)}")
        if event["instrument"] not in INSTRUMENTS:
            raise ValueError(f"{name} instrument must be accel or gyro")
        number(event["start_s"], name, nonnegative=True)
        if event["end_s"] is not None:
            number(event["end_s"], name, positive=True)
            if event["end_s"] <= event["start_s"]:
                raise ValueError(f"{name} requires end_s > start_s")
        _validate_fault_parameter(event, kind, name)


def _validate_fault_parameter(event, kind, name):
    if kind == "stuck":
        axes = event["axes"]
        if (
            not isinstance(axes, list)
            or len(axes) != 3
            or any(type(a) is not bool for a in axes)
        ):
            raise ValueError(f"{name} stuck axes require three booleans")
        return
    values = event["value"]
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"{name} value requires three numbers")
    for value in values:
        number(value, name, nonnegative=kind in {"noise", "spikes"})
    if kind == "spikes":
        number(event["probability"], name, nonnegative=True)
        if event["probability"] > 1:
            raise ValueError(f"{name} spike probability must be in [0, 1]")


MAX_LATENCY_S = 0.5


# Sensors whose samples Gazebo schedules on physics steps (the others are
# sampled by Python on odometry ticks).
GAZEBO_SCHEDULED_SENSORS = (
    "imu",
    "gps",
    "contacts",
    "baro",
    "magnetometer",
    "rangefinder",
    "camera",
    "lidar",
)


def sample_timing_warnings(data: dict) -> list[str]:
    """Gazebo-scheduled streams whose period is not a whole number of
    physics steps.

    Gazebo publishes on the first step at or after each period, so such a
    stream alternates between the neighbouring whole-step intervals: a
    200 Hz IMU on 2 ms steps gives 4 and 6 ms (5 ms mean). Timestamps stay
    exact and the mean rate is kept, but per-sample intervals jitter (see
    docs/2_simulation.md#simulator-environment-audit). Returns one message
    per affected stream (odometry included); [] when all are aligned.
    """
    step_s = data["simulation"]["step_s"]
    streams = {"control.odometry_rate_hz": data["control"]["odometry_rate_hz"]}
    for name in GAZEBO_SCHEDULED_SENSORS:
        sensor = data["sensors"][name]
        if sensor.get("enabled", True):
            streams[f"sensors.{name}.rate_hz"] = sensor["rate_hz"]
    warnings = []
    for key, rate_hz in streams.items():
        steps = 1 / (rate_hz * step_s)
        if abs(steps - round(steps)) < 1e-6:
            continue
        low_ms, high_ms = (
            1e3 * step_s * n for n in (math.floor(steps), math.ceil(steps))
        )
        warnings.append(
            f"{key} = {rate_hz:g} Hz is not a multiple of the {1e3 * step_s:g} ms "
            f"physics step: sample intervals alternate between {low_ms:g} and "
            f"{high_ms:g} ms (mean rate kept)"
        )
    return warnings


def wind_effects_enabled(wind):
    """Whether generated worlds need Gazebo WindEffects (wind or still-air drag)."""
    return wind["enabled"] or wind["still_air_drag"]


def validate_lockstep(data):
    """Lockstep needs whole physics steps per odometry period; RTF 0 needs lockstep."""
    simulation = data["simulation"]
    if simulation["lockstep"]:
        from .stepping import steps_per_period

        steps = steps_per_period(
            1 / data["control"]["odometry_rate_hz"], simulation["step_s"]
        )
        if steps < 2:
            # Measured with gz-sim 8: one-step bursts never deliver their
            # odometry barrier, so the first burst stalls (multirotor and
            # wheeled alike); two or more steps per period work.
            raise ValueError(
                "Lockstep needs at least two physics steps per odometry period; "
                "lower control.odometry_rate_hz or simulation.step_s"
            )
    elif simulation["real_time_factor"] == 0:
        raise ValueError("real_time_factor 0 (unlimited) requires simulation.lockstep")


def validate_power_and_aero(data):
    """Cross-field checks for battery, turbulence, and drag settings."""
    battery, wind = data["battery"], data["wind"]
    for key in ("initial_soc", "powertrain_efficiency"):
        if battery[key] > 1:
            raise ValueError(f"battery.{key} must be in (0, 1]")
    turbulence = wind["turbulence"]
    if turbulence["enabled"] and not wind["enabled"]:
        raise ValueError("wind.turbulence requires wind.enabled")
    if turbulence["update_period_s"] < data["simulation"]["step_s"]:
        raise ValueError("Turbulence update_period_s must be at least the physics step")


def validate_control_timing(c):
    """Check the rotor-thrust controller rate, loop divisor, and latencies."""
    rate = c["control_rate_hz"]
    if not 10 <= rate <= c["odometry_rate_hz"]:
        raise ValueError("control_rate_hz must be between 10 Hz and odometry_rate_hz")
    if 1 / rate >= c["command_lifetime_sim_s"]:
        raise ValueError("Control period must be shorter than command lifetime")
    position = c["position_rate_hz"]
    if position is not None:
        ratio = rate / position
        if position < 10 or position > rate or abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                "position_rate_hz must be >= 10 Hz and divide control_rate_hz"
            )
    for key in ("state_latency_s", "command_latency_s"):
        if c[key] > MAX_LATENCY_S:
            raise ValueError(f"{key} must be at most {MAX_LATENCY_S} s")
    if rate > c["publish_rate_hz"] and not c["publish_on_send"]:
        raise ValueError(
            "control_rate_hz above publish_rate_hz requires publish_on_send: "
            "otherwise most commands are superseded before publication"
        )


# Objects passed through unvalidated to user plug-ins (replaced whole).
FREE_FORM_KEYS = {"estimator_options", "controller_options", "coefficients"}


def merge(defaults, supplied, path="config"):
    if not isinstance(supplied, dict):
        raise ValueError(f"{path} must be an object")
    unknown = supplied.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown {path} keys: {', '.join(sorted(unknown))}")
    result = deepcopy(defaults)
    for key, value in supplied.items():
        if key in FREE_FORM_KEYS:
            if not isinstance(value, dict):
                raise ValueError(f"{path}.{key} must be an object")
            result[key] = deepcopy(value)
            continue
        result[key] = (
            merge(defaults[key], value, f"{path}.{key}")
            if isinstance(defaults[key], dict)
            else value
        )
    return result


def number(value, path, *, positive=False, nonnegative=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{path} must be a finite number")
    if (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"{path} must be {'positive' if positive else 'nonnegative'}")


# Sections with their own validator (called from load_drone_config).
SELF_VALIDATED_KEYS = {"ground_vehicle"}


def validate_tree(value, template, path="config"):
    for key, default in template.items():
        item, name = value[key], f"{path}.{key}"
        if key in FREE_FORM_KEYS:
            continue  # validated by the plug-in that receives it
        if key in SELF_VALIDATED_KEYS:
            continue
        if key == "air_density_kg_m3":
            if item != "site":
                number(item, name, positive=True)
            continue
        if key == "uere_m":
            if item != "auto":
                number(item, name, positive=True)
            continue
        if isinstance(default, dict):
            validate_tree(item, default, name)
        elif isinstance(default, str):
            if not isinstance(item, str):
                raise ValueError(f"{name} must be a string")
        elif isinstance(default, bool):
            if type(item) is not bool:
                raise ValueError(f"{name} must be boolean")
        elif default is None:
            if item is not None:
                number(item, name, positive=True)
        elif isinstance(default, list):
            if key == "length_schedule":
                validate_length_schedule(item, name)
                continue
            if key == "schedule":
                if not isinstance(item, list):
                    raise ValueError(f"{name} must be a list of wind events")
                previous = -1.0
                for event in item:
                    if not isinstance(event, dict) or set(event) != {
                        "time_s",
                        "velocity_enu_mps",
                    }:
                        raise ValueError(
                            f"{name} events require time_s and velocity_enu_mps"
                        )
                    number(event["time_s"], name, nonnegative=True)
                    if event["time_s"] <= previous:
                        raise ValueError(f"{name} times must strictly increase")
                    previous = event["time_s"]
                    validate_tree(
                        event, {"time_s": 0.0, "velocity_enu_mps": [0.0] * 3}, name
                    )
                continue
            if key in MATRIX_KEYS:
                validate_matrix3(item, name)
                continue
            if key in ("rotor_thrust_scale", "rotor_moment_scale"):
                if not isinstance(item, list):
                    raise ValueError(f"{name} must be a list (one value per rotor)")
                for v in item:
                    number(v, name, positive=True)
                continue
            if key == "rotors":
                validate_rotors(item, name)
                continue
            if key == "dipoles":
                validate_dipoles(item, name)
                continue
            if key == "observers":
                validate_observers(item, name)
                continue
            if key == "waypoints_m":
                validate_points(item, name)
                continue
            if key == "zones":
                validate_zones(item, name)
                continue
            if key == "events" and name.startswith("config.motors"):
                validate_motor_fault_events(item, name)
                continue
            if key == "events":
                validate_imu_fault_events(item, name)
                continue
            if key == "outages_s":
                if not isinstance(item, list):
                    raise ValueError(f"{name} must be a list of [start, end] intervals")
                for interval in item:
                    if not isinstance(interval, list) or len(interval) != 2:
                        raise ValueError(f"{name} requires [start, end] intervals")
                    for endpoint in interval:
                        number(endpoint, name, nonnegative=True)
                    if interval[0] >= interval[1]:
                        raise ValueError(f"{name} requires start < end")
                continue
            if not isinstance(item, list) or len(item) != 3:
                raise ValueError(f"{name} requires three numbers")
            for v in item:
                number(
                    v,
                    name,
                    nonnegative=key
                    in {
                        "noise_stddev",
                        "accel_amplitude_mps2",
                        "random_std_mps2",
                        "intensity_mps",
                        "white_std_ut",
                        "drag_area_m2",
                        "center_of_mass_std_m",
                        "accel_bias_instability_mps2",
                        "drift_std_ut",
                        "gyro_bias_instability_rad_s",
                    }
                    | IMU_ERROR_KEYS,
                    positive=key
                    in {
                        "body_size_m",
                        "body_inertia_kg_m2",
                        "attitude_gain",
                        "angular_rate_gain",
                        "velocity_gain",
                        "max_acceleration_mps2",
                        "gps_std_m",
                        "length_scale_m",
                        "size_m",
                    },
                )
        else:
            number(
                item,
                name,
                positive=key
                not in {
                    "bias",
                    "noise_stddev",
                    "rotor_drag_coefficient",
                    "rolling_moment_coefficient",
                    "accel_std_mps2",
                    "gyro_std_rad_s",
                    "accel_bias_walk",
                    "gyro_bias_walk",
                    "probability",
                    "seed",
                    "gust_amplitude_fraction",
                    "random_seed",
                    "random_speed_std_mps",
                    "random_direction_std_deg",
                    "time_s",
                }
                | NONNEGATIVE_SCALAR_KEYS
                | SIGNED_SCALAR_KEYS,
                nonnegative=key
                in {
                    "noise_stddev",
                    "rotor_drag_coefficient",
                    "rolling_moment_coefficient",
                    "accel_std_mps2",
                    "gyro_std_rad_s",
                    "accel_bias_walk",
                    "gyro_bias_walk",
                    "probability",
                    "seed",
                    "gust_amplitude_fraction",
                    "random_seed",
                    "random_speed_std_mps",
                    "random_direction_std_deg",
                    "time_s",
                }
                | NONNEGATIVE_SCALAR_KEYS,
            )
            if (
                key
                in {
                    "schema_version",
                    "width",
                    "height",
                    "samples",
                    "reorder_capacity",
                    "seed",
                    "random_seed",
                    "cells",
                    "vertical_samples",
                    "motor_stop_rotor",
                    "damaged_rotor",
                    "laps",
                    "resolution_steps",
                    "satellites",
                    "satellite_variation",
                    "multipath_satellite_loss",
                    "counts_per_revolution",
                }
                and type(item) is not int
            ):
                raise ValueError(f"{name} must be an integer")


@dataclass(frozen=True)
class DroneConfig:
    data: dict
    properties: DroneProperties
    # Files in merge order (extends parents included) and, per dotted setting
    # key, the file that last set it. Defaults are not listed.
    sources: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict)
    # Nominal wheeled vehicle when robot.kind is rover, car, or tractor.
    wheeled: WheeledProperties | None = None

    @property
    def kind(self) -> str:
        """Robot family: ``multirotor``, ``rover``, ``car``, or ``tractor``."""
        return str(self.data["robot"]["kind"])

    @property
    def render_sensors(self):
        return any(
            self.data["sensors"][name]["enabled"]
            for name in ("camera", "lidar", "rangefinder")
        )

    @property
    def content_hash(self):
        return hashlib.sha256(
            json.dumps(self.data, sort_keys=True).encode()
        ).hexdigest()


ConfigPaths = str | Path | Sequence[str | Path] | None


def _as_paths(paths: ConfigPaths) -> list[Path]:
    """Accept no file, one path, or an ordered sequence of paths."""
    if not paths:
        return []
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _read_layer(path: Path) -> dict:
    try:
        layer = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load drone config: {error}") from error
    if not isinstance(layer, dict):
        raise ValueError(f"{path}: config must be a JSON object")
    return layer


def _expand_layers(path: Path, chain: tuple[Path, ...] = ()) -> list[tuple[str, dict]]:
    """Return (source, settings) layers for one file in merge order.

    A file's ``extends`` parents come first (recursively, in listed order,
    paths relative to that file), then the file itself. Cycles are rejected.
    """
    resolved = path.resolve()
    if resolved in chain:
        cycle = " -> ".join(str(p) for p in (*chain, resolved))
        raise ValueError(f"Config extends cycle: {cycle}")
    layer = _read_layer(path)
    parents = layer.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents):
        raise ValueError(f"{path}: extends must be a path or a list of paths")
    layers = []
    for parent in parents:
        layers += _expand_layers(resolved.parent / parent, (*chain, resolved))
    layers.append((str(path), layer))
    return layers


def _record_provenance(
    layer: dict, source: str, provenance: dict, prefix: str = ""
) -> None:
    """Map each setting a layer supplies (dotted key) to that layer's file."""
    for key, value in layer.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and isinstance(_lookup(DEFAULTS, name), dict):
            _record_provenance(value, source, provenance, name)
        else:
            provenance[name] = source


def _lookup(tree: dict, dotted: str):
    for part in dotted.split("."):
        if not isinstance(tree, dict) or part not in tree:
            return None
        tree = tree[part]
    return tree


def _robot_kind(
    robot: str | None,
    layers: list[tuple[str, dict]],
    allowed: tuple[str, ...] = ROBOT_KINDS,
) -> str:
    """Robot kind from ``--robot`` and the layers' ``robot.kind`` (the last
    layer that sets it); they must agree and be ``allowed``. Default:
    multirotor."""
    kinds = [
        (source, layer["robot"]["kind"])
        for source, layer in layers
        if isinstance(layer.get("robot"), dict) and "kind" in layer["robot"]
    ]
    kind = kinds[-1][1] if kinds else (robot or "multirotor")
    if kind in MUJOCO_KINDS and allowed == ROBOT_KINDS:
        raise ValueError(
            f"robot.kind {kind!r} runs in MuJoCo: use robotlab --robot {kind} "
            "(settings: simulation/mujoco/config.py), not a Gazebo/batch config"
        )
    if kind not in allowed:
        raise ValueError(f"robot.kind must be one of {', '.join(allowed)}")
    if robot is not None and kind != robot:
        raise ValueError(
            f"--robot {robot} conflicts with robot.kind {kind!r} in {kinds[-1][0]}"
        )
    return kind


def requested_robot_kind(robot: str | None, paths: ConfigPaths = None) -> str:
    """Robot kind a run asks for (``--robot`` and the layers' ``robot.kind``),
    including the MuJoCo kinds, without validating anything else; lets the
    CLI pick the simulator before loading settings. Raises ValueError when
    they conflict or name an unknown kind."""
    layers = [layer for path in _as_paths(paths) for layer in _expand_layers(path)]
    return _robot_kind(robot, layers, ROBOT_KINDS + MUJOCO_KINDS)


def load_drone_config(
    paths: ConfigPaths = None,
    *,
    render_sensors: bool = False,
    site_elevation_m: float = WORLD_ELEVATION_M,
    robot: str | None = None,
):
    """Load built-in defaults overridden by zero or more layered JSON files.

    Files apply in order, later ones overriding earlier ones; each file's
    ``extends`` parents apply just before it. Objects merge key by key, while
    lists and scalars replace the earlier value whole. The merged result is
    validated once, so a layer need not be valid on its own.

    ``aerodynamics.air_density_kg_m3: "site"`` (the default) is replaced by
    the dry-air density at ``site_elevation_m`` (m above sea level; the
    generated world's elevation unless a saved world says otherwise) and
    ``environment.ambient_temperature_c``, so the returned settings always
    hold a number.

    ``robot`` (the ``--robot`` option) or a layer's ``robot.kind`` selects
    the robot family; a wheeled kind's preset (simulation/wheeled/config.py)
    applies before every file layer.
    """
    layers = [layer for path in _as_paths(paths) for layer in _expand_layers(path)]
    if robot is not None and robot not in ROBOT_KINDS + MUJOCO_KINDS:
        raise ValueError(f"--robot must be one of {', '.join(ROBOT_KINDS)}")
    kind = _robot_kind(robot, layers)
    if kind != "multirotor":
        layers.insert(0, (f"robot preset {kind}", preset_layer(kind)))
    # Copy: callers may modify the returned settings.
    data = deepcopy(DEFAULTS)
    provenance: dict[str, str] = {}  # dotted key -> layer file
    for source, layer in layers:
        data = merge(data, layer, f"{source}: config")
        _record_provenance(layer, source, provenance)
    validate_tree(data, DEFAULTS)
    if data["sensors"]["gps"]["dropout"]["probability"] > 1:
        raise ValueError("GPS dropout probability must be in [0, 1]")
    if data["wind"]["gust_amplitude_fraction"] > 1:
        raise ValueError("Wind gust amplitude fraction must be in [0, 1]")
    if data["wind"]["rise_time_s"] < 1e-6:
        raise ValueError("Wind rise_time_s must be at least 1e-6")
    if data["wind"]["random_update_period_s"] < data["simulation"]["step_s"]:
        raise ValueError(
            "Wind random_update_period_s must be at least the physics step"
        )
    validate_power_and_aero(data)
    validate_new_sensors(data)
    validate_lockstep(data)
    aero = data["aerodynamics"]
    if aero["translational_lift"]["max_gain"] > 1:
        raise ValueError("aerodynamics.translational_lift.max_gain must be <= 1")
    if aero["vortex_ring"]["thrust_loss"] >= 1:
        raise ValueError("aerodynamics.vortex_ring.thrust_loss must be in [0, 1)")
    if not -40 <= data["environment"]["ambient_temperature_c"] <= 60:
        raise ValueError("environment.ambient_temperature_c must be in [-40, 60] °C")
    if aero["air_density_kg_m3"] == "site":
        from ..aerodynamics import site_air_density_kg_m3

        aero["air_density_kg_m3"] = site_air_density_kg_m3(
            site_elevation_m, data["environment"]["ambient_temperature_c"]
        )
    from ..fixed_wing import DEFAULT_COEFFICIENTS

    coefficients = data["fixed_wing"]["coefficients"]
    unknown = set(coefficients) - set(DEFAULT_COEFFICIENTS)
    if unknown:
        raise ValueError(f"Unknown fixed_wing coefficients: {sorted(unknown)}")
    for key, value in coefficients.items():
        number(value, f"fixed_wing.coefficients.{key}")
    if data["motors"]["forward_figure_of_merit"] > 1:
        raise ValueError("motors.forward_figure_of_merit must be in (0, 1]")
    gps = data["sensors"]["gps"]
    if gps["rtk"]["enabled"] and gps["error_model"]["enabled"]:
        raise ValueError("Choose sensors.gps.rtk or sensors.gps.error_model, not both")
    if gps["receiver"]["satellites"] < 4 + gps["receiver"]["satellite_variation"]:
        raise ValueError("gps receiver satellites must leave at least 4 in view")
    if (
        data["sensors"]["imu"]["calibration"]["startup_gyro_calibration_s"] > 0
        and not data["simulation"]["start_on_ground"]
    ):
        raise ValueError(
            "imu startup_gyro_calibration_s needs simulation.start_on_ground "
            "(the vehicle must be still)"
        )
    imperfections = data["sensors"]["imu"]["imperfections"]
    if imperfections["sample_drop_probability"] > 1:
        raise ValueError("imu imperfections sample_drop_probability must be <= 1")
    for key in ("accel_cross_coupling", "gyro_cross_coupling"):
        if any(imperfections[key][i][i] for i in range(3)):
            raise ValueError(f"imu imperfections {key} diagonal must be zero")
    lidar = data["sensors"]["lidar"]
    from ...sensors.imaging.lidar_returns import RETURN_MODES

    if lidar["effects"]["return_mode"] not in RETURN_MODES:
        raise ValueError(
            f"sensors.lidar.effects.return_mode must be one of {RETURN_MODES}"
        )
    if (
        lidar["enabled"]
        and data["environment"]["weather"]["visibility_m"] is not None
        and not lidar["effects"]["detection"]["enabled"]
    ):
        raise ValueError(
            "environment.weather.visibility_m attenuates lidar returns only "
            "through sensors.lidar.effects.detection; enable it"
        )
    from ...control.references import validate_mission

    validate_mission(data["mission"])
    from ...evaluation.health import HealthThresholds

    HealthThresholds.from_settings(data["health"])
    algorithms = data["algorithms"]
    from ...experiments.harness.replay_log import RECORD_MODES
    from .plugins import BARRIERS

    if data["simulation"]["lockstep_barrier"] not in BARRIERS:
        raise ValueError(f"simulation.lockstep_barrier must be one of {BARRIERS}")

    if algorithms["record_bus"] not in RECORD_MODES:
        raise ValueError(f"algorithms.record_bus must be one of {RECORD_MODES}")
    if algorithms["control_state_source"] not in ("truth", "estimate"):
        raise ValueError("algorithms.control_state_source must be truth or estimate")
    if algorithms["control_state_source"] == "estimate" and not algorithms["estimator"]:
        raise ValueError("Estimate-driven control needs algorithms.estimator")
    failsafe = data["failsafe"]["low_battery"]
    if failsafe["enabled"]:
        from ...experiments.multirotor.failsafe import ACTIONS as FAILSAFE_ACTIONS

        if failsafe["action"] not in FAILSAFE_ACTIONS:
            raise ValueError(f"failsafe action must be one of {FAILSAFE_ACTIONS}")
        if not data["battery"]["enabled"]:
            raise ValueError("failsafe.low_battery requires battery.enabled")
        if failsafe["land_remaining_pct"] > failsafe["warn_remaining_pct"]:
            raise ValueError(
                "failsafe land_remaining_pct must not exceed warn_remaining_pct"
            )
    effects = data["keyboard_effects"]
    rotor_count = len(data["vehicle"]["rotors"]) or 4
    if max(effects["motor_stop_rotor"], effects["damaged_rotor"]) >= rotor_count:
        raise ValueError(f"keyboard_effects rotors must be below {rotor_count}")
    if effects["battery_drain_fraction"] > 1:
        raise ValueError("keyboard_effects.battery_drain_fraction must be in [0, 1]")
    if effects["damaged_thrust_fraction"] >= 1:
        raise ValueError("keyboard_effects.damaged_thrust_fraction must be in [0, 1)")
    if data["schema_version"] != 1:
        raise ValueError("Unsupported drone schema_version")
    if render_sensors:
        for name in ("camera", "lidar"):
            data["sensors"][name]["enabled"] = True
    v, m, s, c = (data[k] for k in ("vehicle", "motors", "sensors", "control"))
    inertia = v["body_inertia_kg_m2"]
    if max(abs(x) for x in v["center_of_mass_offset_m"]) > v["arm_offset_m"] / 2:
        raise ValueError("center_of_mass_offset_m must stay within half the arm offset")
    for key in (
        "mass_fraction_std",
        "inertia_fraction_std",
        "rotor_thrust_fraction_std",
        "rotor_moment_fraction_std",
        "motor_time_constant_fraction_std",
    ):
        if data["randomization"][key] >= 1 / 3:
            raise ValueError(
                f"randomization.{key} must be below 1/3 (draws truncate at 3σ)"
            )
    if 2 * max(inertia) > sum(inertia) + 1e-12:
        raise ValueError("Body inertia violates principal-moment triangle inequalities")
    for name in ("camera", "lidar"):
        low, high = (
            ("near_m", "far_m") if name == "camera" else ("min_range_m", "max_range_m")
        )
        if s[name][low] >= s[name][high]:
            raise ValueError(f"{name}: minimum range must be less than maximum")
    lens = s["camera"]["lens"]["type"]
    if lens not in LENS_TYPES:
        raise ValueError(f"camera lens type must be one of {', '.join(LENS_TYPES)}")
    fov_limit = math.pi if lens == "pinhole" else 2 * math.pi
    if (
        not 0
        < s["camera"]["horizontal_fov_rad"]
        < fov_limit + (lens != "pinhole") * 1e-4
    ):
        raise ValueError(
            "Camera horizontal FOV must be in (0, pi) for pinhole, (0, 2*pi] for wide-angle"
        )
    if s["lidar"]["samples"] < 2:
        raise ValueError("Lidar requires at least two samples")
    step = data["simulation"]["step_s"]
    if step > 0.01:
        raise ValueError("Simulation step_s must be <= 0.01")
    for sensor in s.values():
        if sensor["rate_hz"] > 1 / step:
            raise ValueError("Sensor rate exceeds physics rate")
    if not 20 <= c["odometry_rate_hz"] <= 1 / step:
        raise ValueError("Odometry rate must be between 20 Hz and physics rate")
    if not 1 / c["odometry_rate_hz"] < c["command_lifetime_sim_s"] <= 1:
        raise ValueError("Command lifetime must exceed odometry interval and be <= 1 s")
    if 1 / c["publish_rate_hz"] >= c["command_lifetime_sim_s"]:
        raise ValueError("Publication interval must be shorter than command lifetime")
    validate_control_timing(c)
    rotors, positions, axes = rotor_layout(v)
    count = len(rotors)
    for key in ("rotor_thrust_scale", "rotor_moment_scale"):
        if m[key] and len(m[key]) != count:
            raise ValueError(f"motors.{key} needs one value per rotor ({count}) or []")
    if sum(1 for axis in axes if axis[2] > 0.99) < 4:
        raise ValueError("vehicle needs at least four vertical (lift) rotors")
    for event in m["faults"]["events"]:
        if event["rotor"] >= count:
            raise ValueError(f"motor fault rotor index must be below {count}")
    props = DroneProperties(
        body_mass=v["body_mass_kg"],
        rotor_mass=v["rotor_mass_kg"],
        body_inertia=tuple(inertia),
        rotor_inertia=v["rotor_inertia_kg_m2"],
        rotors=rotors,
        rotor_positions=None if not v["rotors"] else positions,
        rotor_axes=None if not v["rotors"] else axes,
        thrust_coefficient=m["thrust_coefficient_n_s2"],
        moment_ratio=m["reaction_moment_ratio_m"],
        max_rotor_speed=m["max_speed_rad_s"],
        thrust_limit_n=m["max_thrust_n"],
        rotor_height=v["rotor_height_m"],
        propeller_diameter=v["propeller_diameter_m"],
    )
    try:
        physical_max = props.thrust_coefficient * props.max_rotor_speed**2
    except OverflowError as error:
        raise ValueError("Motor parameters overflow numerical limits") from error
    if not math.isfinite(physical_max) or not math.isfinite(props.mass * 9.81):
        raise ValueError("Vehicle parameters overflow numerical limits")
    if m["max_thrust_n"] is not None and m["max_thrust_n"] > physical_max + 1e-12:
        raise ValueError("max_thrust_n exceeds motor speed/coefficient capability")
    payload = data["payload"]
    carried = payload["mass_kg"] if payload["enabled"] else 0.0
    tether = payload["tether"]
    if payload["enabled"] and tether["enabled"]:
        if data["simulation"]["start_on_ground"]:
            raise ValueError("A tethered payload needs an airborne start (rigid cable)")
        lengths = [tether["initial_length_m"]] + [
            e["length_m"] for e in tether["length_schedule"]
        ]
        if max(lengths) > tether["max_length_m"] or min(lengths) < 0.1:
            raise ValueError("tether lengths must be between 0.1 m and max_length_m")
        if tether["anchor_m"][2] > -v["body_size_m"][2] / 2:
            raise ValueError("tether anchor must be below the body")
    elif payload["enabled"] and (
        payload["offset_m"][2] + payload["size_m"][2] / 2 > -v["body_size_m"][2] / 2
    ):
        raise ValueError(
            "payload must hang below the body (offset_m z + size/2 below it)"
        )
    lift_count = len(props.lift_rotors)
    if (props.mass + carried) * 9.81 >= lift_count * props.max_rotor_thrust:
        raise ValueError(
            "Insufficient thrust for hover; increase limits or reduce mass"
        )
    wheeled = None
    if kind in WHEELED_KINDS:
        validate_ground_vehicle(data)
        wheeled = WheeledProperties.from_settings(
            kind, data["ground_vehicle"], aero["air_density_kg_m3"]
        )
    return DroneConfig(
        data, props, tuple(source for source, _ in layers), provenance, wheeled
    )
