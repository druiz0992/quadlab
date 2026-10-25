from robotlab.cli import gazebo as cli
from robotlab.experiments import run as mission_entry
from robotlab.visualization import keyboard


def test_headless_keyboard_allowed(monkeypatch):
    # Use a minimal window lifecycle without creating a GUI.
    class Window:
        closed = False

        def close(self):
            self.closed = True

    window = Window()
    monkeypatch.setattr(keyboard, "KeyboardWindow", lambda: window)
    seen = []
    # The mission opens the window itself, after validation, through the opener.
    monkeypatch.setattr(
        mission_entry,
        "_dispatch",
        lambda args, opener, source: seen.append((args, opener())),
    )
    mission_entry.run(mission_entry.GazeboConfig(keyboard=True, headless=True))
    assert seen[0][0].headless and seen[0][1] is window
    assert window.closed


def test_cli_accepts_rerun_with_headless_keyboard(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["robotlab-gazebo", "--headless", "--keyboard", "--rerun"]
    )
    configs = []
    monkeypatch.setattr(cli, "run", configs.append)
    cli.main()
    assert configs[0].keyboard and configs[0].headless and configs[0].rerun
