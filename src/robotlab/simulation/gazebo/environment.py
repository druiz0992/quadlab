"""Prepare child-process library paths without modifying the parent environment."""

import os


def gazebo_environment(environ=None):
    """Keep Snap IDE libraries out of the system Gazebo process."""
    env = dict(os.environ if environ is None else environ)
    for key in [
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "GTK_PATH",
        "GTK_EXE_PREFIX",
        "GIO_MODULE_DIR",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
    ]:
        if key in env:
            separator = " " if key == "LD_PRELOAD" else ":"
            paths = [p for p in env[key].split(separator) if p and "/snap/" not in p]
            if paths:
                env[key] = separator.join(paths)
            else:
                env.pop(key)
    # RobotLab's own system plugins (lockstep barrier).
    from .plugins import plugin_dir

    search = [p for p in env.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "").split(":") if p]
    env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = ":".join(search + [str(plugin_dir())])
    return env
