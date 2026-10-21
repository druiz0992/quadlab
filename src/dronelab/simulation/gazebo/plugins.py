"""DroneLab's own Gazebo system plugins: location, build, and selection.

The plugins live in ``gazebo_plugins/`` (C++, CMake) and are built into
``build/gazebo_plugins`` by ``dronelab-build-plugins``;
``DRONELAB_PLUGIN_DIR`` overrides the location. Gazebo finds them through
``GZ_SIM_SYSTEM_PLUGIN_PATH`` (see ``environment.py``).

``DronelabLockstep`` is the lockstep barrier: Gazebo holds each step until
Python grants simulation time over a topic that shares the commands'
ordered transport connection (see the plugin source and
docs/2_simulation.md). ``simulation.lockstep_barrier`` selects it:
``plugin`` requires it, ``service`` uses the world control service
(commands can then take effect one period late under heavy load), and
``auto`` uses the plugin when it is built.
"""

import os
from pathlib import Path
import shutil
import subprocess

REPOSITORY = Path(__file__).resolve().parents[4]
SOURCE_DIR = REPOSITORY / "gazebo_plugins"
LOCKSTEP_LIBRARY = "libdronelab-lockstep.so"
LOCKSTEP_FILENAME = "dronelab-lockstep"
LOCKSTEP_NAME = "dronelab::Lockstep"
BARRIERS = ("auto", "plugin", "service")


def plugin_dir() -> Path:
    """Directory holding the built plugins."""
    override = os.environ.get("DRONELAB_PLUGIN_DIR")
    return Path(override) if override else REPOSITORY / "build" / "gazebo_plugins"


def lockstep_plugin_available() -> bool:
    return (plugin_dir() / LOCKSTEP_LIBRARY).is_file()


def lockstep_barrier(settings: dict) -> str | None:
    """``plugin`` or ``service`` for lockstep runs, None otherwise.

    Raises ValueError when the plugin is required but not built.
    """
    simulation = settings["simulation"]
    if not simulation["lockstep"]:
        return None
    choice = simulation["lockstep_barrier"]
    if choice not in BARRIERS:
        raise ValueError(f"simulation.lockstep_barrier must be one of {BARRIERS}")
    available = lockstep_plugin_available()
    if choice == "plugin" and not available:
        raise ValueError(
            f"simulation.lockstep_barrier is plugin but {LOCKSTEP_LIBRARY} is not "
            f"built in {plugin_dir()}; run dronelab-build-plugins"
        )
    if choice == "service" or not available:
        return "service"
    return "plugin"


def build_plugins(build_dir: Path | None = None) -> Path:
    """Configure and build the plugins with CMake; return the build directory.

    Needs cmake, a C++17 compiler, and the Gazebo Harmonic development files
    (source the ROS/Gazebo environment first); raises RuntimeError otherwise.
    """
    build_dir = build_dir or plugin_dir()
    if shutil.which("cmake") is None:
        raise RuntimeError("cmake not found; install cmake and a C++ compiler")
    for command in (
        [
            "cmake",
            "-S",
            str(SOURCE_DIR),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        ["cmake", "--build", str(build_dir), "-j"],
    ):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"{' '.join(command)} failed:\n{result.stdout[-2000:]}{result.stderr[-2000:]}"
            )
    return build_dir
