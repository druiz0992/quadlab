from dronelab.cli import gazebo as cli
from dronelab.experiments import gazebo
from dronelab.visualization import keyboard


def test_headless_keyboard_allowed(monkeypatch):
    # Use a minimal window lifecycle without creating a GUI.
    class Window:
        closed = False

        def close(self):
            self.closed = True

    window = Window()
    monkeypatch.setattr(keyboard, "KeyboardWindow", lambda: window)
    seen = []
    monkeypatch.setattr(
        gazebo, "_run", lambda args, keys, source: seen.append((args, keys))
    )
    gazebo.run(gazebo.GazeboConfig(keyboard=True, headless=True))
    assert seen[0][0].headless and seen[0][1] is window
    assert window.closed


def test_cli_accepts_rerun_with_headless_keyboard(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["dronelab-gazebo", "--headless", "--keyboard", "--rerun"]
    )
    configs = []
    monkeypatch.setattr(cli, "run", configs.append)
    cli.main()
    assert configs[0].keyboard and configs[0].headless and configs[0].rerun
