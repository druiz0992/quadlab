from dronelab.simulation.gazebo.environment import gazebo_environment


def test_snap_environment_cleanup_preserves_ros():
    original = {
        "LD_LIBRARY_PATH": "/opt/ros/jazzy/lib:/snap/core20/current/lib",
        "GTK_PATH": "/snap/code/271/usr/lib/gtk",
        "GIO_MODULE_DIR": "/home/david/snap/code/common/.cache/gio-modules",
        "DISPLAY": ":0",
        "GZ_PARTITION": "test",
    }
    clean = gazebo_environment(original)
    assert clean["LD_LIBRARY_PATH"] == "/opt/ros/jazzy/lib"
    assert "GTK_PATH" not in clean
    assert "GIO_MODULE_DIR" not in clean
    assert clean["DISPLAY"] == ":0"
    assert clean["GZ_PARTITION"] == "test"
    assert "GTK_PATH" in original
