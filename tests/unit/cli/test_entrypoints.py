import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_python_module_runs_gazebo_generation_without_gui(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "robotlab",
            "--generate-only",
            "--control-mode",
            "rotor-thrust",
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert str(tmp_path / "world.sdf") in result.stdout
    assert (tmp_path / "world.sdf").exists()
    assert (tmp_path / "drone-resolved.json").exists()


def test_retired_backend_option_is_rejected():
    result = subprocess.run(
        [sys.executable, "-m", "robotlab", "--backend", "reference"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_live_ekf_option_is_forwarded(monkeypatch):
    from robotlab.cli import gazebo

    configurations = []
    monkeypatch.setattr(sys, "argv", ["robotlab", "--ekf", "--headless", "--rerun"])
    monkeypatch.setattr(gazebo, "run", configurations.append)
    gazebo.main()
    assert (
        configurations[0].ekf and configurations[0].headless and configurations[0].rerun
    )


def test_repeated_drone_option_layers_files_and_records_sources(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "robotlab",
            "--generate-only",
            "--drone",
            "configs/vehicles/10in_4s.json",
            "--config",
            "configs/sensors/baro.json",
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    import json

    resolved = json.loads((tmp_path / "drone-resolved.json").read_text())
    assert resolved["sources"] == [
        "configs/vehicles/10in_4s.json",
        "configs/sensors/baro.json",
    ]
    assert resolved["provenance"]["battery.enabled"] == "configs/vehicles/10in_4s.json"
    assert resolved["provenance"]["sensors.baro.enabled"] == "configs/sensors/baro.json"
    assert resolved["battery"]["enabled"] and resolved["sensors"]["baro"]["enabled"]


def test_runtime_error_exits_without_usage(monkeypatch, capsys):
    from robotlab.cli import gazebo

    def fail(config):
        raise RuntimeError("Stale odometry in rotor-control queue")

    monkeypatch.setattr(sys, "argv", ["robotlab"])
    monkeypatch.setattr(gazebo, "run", fail)
    with pytest.raises(SystemExit) as exit_info:
        gazebo.main()
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert "Stale odometry" in err and "usage:" not in err


def test_value_error_shows_usage(monkeypatch, capsys):
    from robotlab.cli import gazebo

    def invalid(config):
        raise ValueError("bad option")

    monkeypatch.setattr(sys, "argv", ["robotlab"])
    monkeypatch.setattr(gazebo, "run", invalid)
    with pytest.raises(SystemExit) as exit_info:
        gazebo.main()
    assert exit_info.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_interrupt_and_absorbed_sigterm_exit_statuses(capsys):
    import argparse
    import os
    import signal

    from robotlab.cli.common import run_reporting_errors

    parser = argparse.ArgumentParser(prog="robotlab")

    def interrupted():
        raise KeyboardInterrupt

    with pytest.raises(SystemExit) as exit_info:
        run_reporting_errors(parser, interrupted)
    assert exit_info.value.code == 130
    assert "interrupted" in capsys.readouterr().err

    def mission_absorbs_sigterm():
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            signal.pause()  # interrupted by the handler
        except KeyboardInterrupt:
            return "saved"  # like MissionLoop.run after an interruption

    with pytest.raises(SystemExit) as exit_info:
        run_reporting_errors(parser, mission_absorbs_sigterm)
    assert exit_info.value.code == 143
