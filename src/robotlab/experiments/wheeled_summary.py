"""``summary.json`` of a wheeled-robot mission, assembled from the finished loop.

Pure data assembly in a fixed key order; the health verdict is computed
once here (see evaluation/health.py). Shares the sensor-model entries
with the multirotor summary (``run_summary.py``).
"""

import time
from typing import Any

from ..simulation.gazebo.config import sample_timing_warnings
from .run_summary import _algorithms, _state_source


def build_wheeled_summary(loop: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    """The run summary of a ``WheeledMissionLoop`` after shutdown."""
    parts = loop.parts
    settings, config = loop.settings, parts.config
    vehicle = parts.model.properties
    sensors = parts.pipeline.summary()
    exit_reason = loop.final_exit_reason
    return dict(
        completed=loop.completed,
        health=parts.health.summary(
            completed=loop.completed,
            exit_reason=exit_reason,
            dropped=dict(parts.ingress.dropped),
            ordering_timeouts=parts.orderer.barrier_timeouts,
        ),
        exit_reason=exit_reason,
        robot=dict(
            kind=vehicle.kind,
            model=vehicle.model_name,
            steering_layout=vehicle.steering_layout,
            mass_kg=vehicle.mass_kg,
            wheels=list(vehicle.wheel_names),
        ),
        control_mode="wheel-commands",
        state_source=_state_source(settings, parts.stack),
        algorithms=_algorithms_entry(settings, parts),
        sensor_bus=parts.stack.bus.summary() if parts.stack.bus is not None else None,
        algorithm_debug_entries=parts.debug.entries,
        metrics=metrics,
        drone_config_sha256=config.content_hash,
        control_ticks=loop.counts["control_ticks"],
        limited_commands=dict(loop._limited),
        wrench_messages=parts.wrench.messages,
        max_speed_mps=loop.max_speed_mps,
        wheel_encoders=parts.encoders.summary(),
        messages={k: v for k, v in loop.counts.items() if k != "control_ticks"},
        dropped=dict(parts.ingress.dropped),
        message_ordering=parts.orderer.summary(),
        gps_dropout=sensors["gps_dropout"],
        imu_errors=sensors["imu_errors"],
        gps_errors=sensors["gps_errors"],
        gps_multipath=sensors["gps_multipath"],
        imu_faults=sensors["imu_faults"],
        baro_errors=sensors["baro_errors"],
        magnetometer_errors=sensors["magnetometer_errors"],
        magnetic_anomalies=sensors["magnetic_anomalies"],
        lidar_effects=sensors["lidar_effects"],
        imu_temperature=sensors["imu_temperature"],
        gps_receiver=sensors["gps_receiver"],
        imu_imperfections=sensors["imu_imperfections"],
        camera_effects=sensors["camera_effects"],
        odometry_startup_corrections=parts.odometry_fix.corrected_samples,
        sim_start=loop.first_time,
        sim_end=loop.last_time,
        collisions=loop.collisions,
        wall_seconds=time.monotonic() - loop.started,
        loop_timing=loop.timing.summary(),
        lockstep=loop.stepper.summary() if loop.stepper is not None else None,
        final_position=loop.positions[-1].tolist() if loop.positions else None,
        left_ground_area_s=loop.left_ground_s,
        calibration_warnings=sensors["calibration_warnings"],
        sample_timing_warnings=sample_timing_warnings(settings),
        imu_startup_gyro_bias_rad_s=sensors["imu_startup_gyro_bias_rad_s"],
    )


def _algorithms_entry(settings: dict[str, Any], parts: Any) -> dict[str, Any]:
    """Estimator entries as for multirotors; the controller is the wheeled one."""
    entry = _algorithms(settings, parts.stack)
    controller = parts.controller
    entry["controller"] = settings["algorithms"]["controller"] or (
        "keyboard" if controller is not None else "none"
    )
    entry["controller_summary"] = (
        controller.summary()
        if controller is not None and hasattr(controller, "summary")
        else None
    )
    return entry
