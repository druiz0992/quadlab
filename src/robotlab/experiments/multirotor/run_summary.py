"""``summary.json`` of a Gazebo mission, assembled from the finished loop.

Pure data assembly: every entry comes from a component's own ``summary()``
or from loop state, in a fixed key order. The health verdict is computed
here, once, at the end of the run (see evaluation/health.py).
"""

import time
from typing import Any

from ...simulation.gazebo.config import sample_timing_warnings
from ..common.exit_reason import final_exit_reason


def build_summary(loop: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    """The run summary of a ``MissionLoop`` after shutdown."""
    parts = loop.parts
    args, settings, config = parts.args, loop.settings, parts.config
    timing, rotor_mode = settings["control"], parts.rotor_mode
    thrust, control_loop = loop.thrust_control, loop.control_loop
    sensors = parts.pipeline.summary()
    exit_reason = loop.final_exit_reason
    return dict(
        completed=loop.completed,
        health=parts.health.summary(
            completed=loop.completed,
            exit_reason=exit_reason,
            dropped=dict(
                parts.ingress.dropped,
                command_events=thrust.dropped_events if thrust else 0,
            ),
            ordering_timeouts=(
                parts.orderer.barrier_timeouts if parts.orderer is not None else 0
            ),
        ),
        exit_reason=exit_reason,
        control_mode="rotor-thrust" if rotor_mode else "velocity",
        state_source=_state_source(settings, parts.stack),
        algorithms=_algorithms(settings, parts.stack),
        sensor_bus=parts.stack.bus.summary() if parts.stack.bus is not None else None,
        replay=parts.replay.summary() if parts.replay is not None else None,
        algorithm_debug_entries=parts.debug.entries,
        randomization=parts.randomization_draws or None,
        esc=_esc(parts),
        metrics=metrics,
        drone_config_sha256=config.content_hash if not args.world else None,
        saturated_commands=(
            control_loop.stats["saturated_ticks"] if control_loop else 0
        ),
        control_timing=control_loop.summary() if control_loop else None,
        dropped_command_events=thrust.dropped_events if thrust else 0,
        thrust_publish_rate_hz=timing["publish_rate_hz"] if rotor_mode else None,
        command_lifetime_sim_s=(
            timing["command_lifetime_sim_s"] if rotor_mode else None
        ),
        command_watchdog_wall_s=timing["watchdog_wall_s"] if rotor_mode else None,
        messages=dict(loop.counts),
        dropped=dict(parts.ingress.dropped),
        message_ordering=(
            parts.orderer.summary() if parts.orderer is not None else None
        ),
        gps_dropout=sensors["gps_dropout"],
        accelerometer_vibration=sensors["accelerometer_vibration"],
        imu_errors=sensors["imu_errors"],
        gps_errors=sensors["gps_errors"],
        gps_multipath=sensors["gps_multipath"],
        imu_faults=sensors["imu_faults"],
        baro_errors=sensors["baro_errors"],
        magnetometer_errors=sensors["magnetometer_errors"],
        magnetic_anomalies=sensors["magnetic_anomalies"],
        optical_flow=sensors["optical_flow"],
        battery_monitor=sensors["battery_monitor"],
        powertrain=parts.powertrain.summary() if parts.powertrain.enabled else None,
        effect_toggles=parts.effects.summary() if parts.effects is not None else None,
        odometry_startup_corrections=parts.odometry_fix.corrected_samples,
        lidar_effects=sensors["lidar_effects"],
        payload=parts.payload.summary() if parts.payload is not None else None,
        mission=parts.mission.summary() if parts.mission.enabled else None,
        imu_temperature=sensors["imu_temperature"],
        airspeed=sensors["airspeed"],
        gps_receiver=sensors["gps_receiver"],
        imu_imperfections=sensors["imu_imperfections"],
        failsafe=parts.failsafe.summary() if parts.failsafe is not None else None,
        camera_effects=sensors["camera_effects"],
        moving_obstacles=(
            dict(max_path_error_m=parts.moving.max_path_error_m)
            if parts.moving is not None
            else None
        ),
        body_drag=parts.forces.summary(),
        takeoff=(
            dict(
                complete=parts.takeoff.complete,
                complete_time_s=parts.takeoff.complete_time_s,
            )
            if parts.takeoff is not None
            else None
        ),
        sim_start=loop.first_time,
        sim_end=loop.last_time,
        collisions=loop.collisions,
        target=None if rotor_mode else args.target,
        wall_seconds=time.monotonic() - loop.started,
        loop_timing=loop.timing.summary(),
        lockstep=loop.stepper.summary() if loop.stepper is not None else None,
        final_position=loop.positions[-1].tolist() if loop.positions else None,
        left_ground_area_s=loop.left_ground_s,
        calibration_warnings=sensors["calibration_warnings"],
        sample_timing_warnings=sample_timing_warnings(settings),
        imu_startup_gyro_bias_rad_s=sensors["imu_startup_gyro_bias_rad_s"],
    )


def _state_source(settings: dict[str, Any], stack: Any) -> str:
    if stack.estimate_source is None:
        return "simulator ground truth"
    return f"estimate ({settings['algorithms']['estimator']})"


def _algorithms(settings: dict[str, Any], stack: Any) -> dict[str, Any]:
    algorithms = settings["algorithms"]
    estimator, controller = stack.estimator, stack.controller
    return dict(
        estimator=algorithms["estimator"] or None,
        controller=algorithms["controller"] or "manual",
        estimator_summary=estimator.summary() if estimator is not None else None,
        max_estimate_age_ms=(
            stack.estimate_source.max_age_ms if stack.estimate_source else None
        ),
        controller_summary=(
            controller.summary()
            if controller is not None and hasattr(controller, "summary")
            else None
        ),
    )


def _esc(parts: Any) -> dict[str, Any] | None:
    telemetry = parts.pipeline.esc_telemetry
    if not (parts.esc.enabled or telemetry.enabled):
        return None
    return dict(command=parts.esc.summary(), telemetry=telemetry.summary())
