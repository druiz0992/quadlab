import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_python_module_runs_gazebo_generation_without_gui(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dronelab",
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
        [sys.executable, "-m", "dronelab", "--backend", "reference"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_live_ekf_option_is_forwarded(monkeypatch):
    from dronelab.cli import gazebo

    configurations = []
    monkeypatch.setattr(sys, "argv", ["dronelab", "--ekf", "--headless", "--rerun"])
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
            "dronelab",
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
