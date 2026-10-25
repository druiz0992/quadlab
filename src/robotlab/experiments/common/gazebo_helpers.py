"""Gazebo mission helpers shared by drone and wheeled-robot missions.

Physics step, scheduled sample periods for lockstep ordering, the ``gz``
availability check, paused world editing, and the printed health summary
(also used by MuJoCo bipeds).
"""

import shutil
from typing import TYPE_CHECKING, Any

from ...simulation.gazebo.environment import gazebo_environment
from ...simulation.gazebo.process import GazeboProcess

if TYPE_CHECKING:  # avoids importing the sensor pipeline at module load
    from .sensor_pipeline import SensorPipeline


def step_ns(settings: dict) -> int:
    return round(settings["simulation"]["step_s"] * 1e9)


def scheduled_periods_ns(
    settings: dict, pipeline: "SensorPipeline"
) -> dict[str, float]:
    """Sample period (ns) of every transported stream with a fixed schedule.

    Poses (throttled by wall time), contacts (only while touching), and
    rendered sensors (frames shift with rendering speed) have none. Gazebo
    publishes rotor joint states every physics step.
    """
    sensors = settings["sensors"]
    periods = {
        kind: 1e9 / sensors[kind]["rate_hz"]
        for kind in ("imu", "gps", "baro", "magnetometer")
        if sensors[kind]["enabled"]
    }
    if pipeline.esc_telemetry.enabled:
        periods["esc"] = float(step_ns(settings))
    return periods


def require_gz() -> None:
    """Raise ValueError when the ``gz`` command is not on PATH."""
    if shutil.which("gz") is None:
        raise ValueError(
            "gz not found. Install Gazebo Harmonic and source "
            "/opt/ros/jazzy/setup.bash if using ROS Jazzy."
        )


def edit_world(world) -> None:
    """Gazebo with the GUI on ``world`` until its window closes; stopped
    completely on Ctrl-C or SIGTERM. Raises if Gazebo fails."""
    process = GazeboProcess(["gz", "sim", str(world)], None, gazebo_environment())
    try:
        code = process.wait()
    finally:
        process.stop()
    if code != 0:
        raise RuntimeError(f"Gazebo exited with status {code}")


def print_health(health: dict) -> None:
    """One line per health finding after the run (see evaluation/health.py)."""
    print(f"Health: {health['verdict']}")
    for finding in health["findings"]:
        at = "" if finding["time_s"] is None else f" at {finding['time_s']:.2f} s"
        count = f" (x{finding['count']})" if finding["count"] > 1 else ""
        print(f"  {finding['severity']}: {finding['message']}{at}{count}")
