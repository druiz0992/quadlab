"""GPS velocity errors, receiver status, RTK, dual-antenna heading."""

import math

import numpy as np
import pytest

from robotlab.core.measurements import GpsReading
from robotlab.sensors.geodesy import gps_to_enu
from robotlab.sensors.gps_receiver import (
    GpsHeading,
    GpsStatusModel,
    GpsVelocityErrors,
    RtkErrorModel,
)
from robotlab.simulation.gazebo.config import load_drone_config

MS = 1_000_000
ORIGIN = GpsReading(40.4168, -3.7038, 650.0)


def gps(section, **overrides):
    return dict(load_drone_config().data["sensors"]["gps"][section], **overrides)


def test_velocity_errors_white_and_correlated():
    white = GpsVelocityErrors(
        gps(
            "velocity_error",
            enabled=True,
            white_horizontal_std_mps=0.05,
            white_vertical_std_mps=0.1,
        )
    )
    errors = np.array(
        [white.apply([1, 2, 3], i * 100 * MS) - [1, 2, 3] for i in range(20000)]
    )
    np.testing.assert_allclose(errors.std(axis=0), [0.05, 0.05, 0.1], rtol=0.05)
    correlated = GpsVelocityErrors(
        gps(
            "velocity_error",
            enabled=True,
            correlated_std_mps=0.2,
            correlation_time_s=5.0,
            seed=2,
        )
    )
    e = np.array([correlated.apply([0, 0, 0], i * 100 * MS)[0] for i in range(50000)])
    assert e.std() == pytest.approx(0.2, rel=0.1)
    assert np.corrcoef(e[:-50], e[50:])[0, 1] == pytest.approx(math.exp(-1), abs=0.1)
    disabled = GpsVelocityErrors(gps("velocity_error"))
    np.testing.assert_array_equal(disabled.apply([1, 2, 3], 0), [1, 2, 3])


def test_rtk_converges_from_float_to_fixed_and_falls_back_in_multipath():
    rtk = RtkErrorModel(gps("rtk", enabled=True, convergence_s=10.0))
    states, errors = [], []
    for i in range(300):  # 30 s at 10 Hz, multipath 15–17 s
        t = i * 100 * MS
        fix = rtk.apply(ORIGIN, t, degraded=150 <= i < 170)
        states.append(rtk.state)
        errors.append(np.linalg.norm(gps_to_enu(fix, ORIGIN)[:2]))
    assert states[0] == "rtk_float" and states[99] == "rtk_float"
    assert states[100] == "rtk_fixed"
    assert states[160] == "rtk_float"
    # Re-fix convergence_s/3 after the last degraded fix (16.9 s → ~20.2 s).
    assert states[205] == "rtk_fixed" and states[200] == "rtk_float"
    fixed = np.array(errors[110:150])
    assert fixed.std() < 0.03 and np.mean(errors[:90]) > 0.05


def test_status_reports_receiver_estimates():
    model = GpsStatusModel(gps("receiver", uere_m=1.5))  # explicit UERE mode
    statuses = [model.update(False) for _ in range(500)]
    satellites = [s.satellites for s in statuses]
    assert min(satellites) >= 12 and max(satellites) <= 16
    nominal = next(s for s in statuses if s.satellites == 14)
    assert nominal.hdop == pytest.approx(0.9) and nominal.fix_type == "3d"
    assert nominal.horizontal_accuracy_m == pytest.approx(0.9 * 1.5)
    degraded = model.update(True)
    assert degraded.satellites <= 11 and degraded.hdop > 0.9
    rtk = RtkErrorModel(gps("rtk", enabled=True))
    rtk.apply(ORIGIN, 0, False)
    assert model.update(False, rtk).fix_type == "rtk_float"


def test_dual_antenna_heading_noise_mount_and_wrap():
    heading = GpsHeading(
        gps("heading", enabled=True, noise_std_deg=0.5, mount_yaw_deg=90.0)
    )
    values = np.array(
        [heading.sample(math.radians(170)).heading_rad for _ in range(5000)]
    )
    # 170° + 90° = 260° → −100°.
    assert np.degrees(np.mean(values)) == pytest.approx(-100, abs=0.05)
    assert np.degrees(values.std()) == pytest.approx(0.5, rel=0.05)


def test_rtk_and_standard_error_model_are_exclusive(tmp_path):
    import json

    path = tmp_path / "x.json"
    path.write_text(
        json.dumps(
            {
                "sensors": {
                    "gps": {"rtk": {"enabled": True}, "error_model": {"enabled": True}}
                }
            }
        )
    )
    with pytest.raises(ValueError, match="not both"):
        load_drone_config(path)


def test_auto_accuracy_follows_configured_error():
    from robotlab.sensors.gps_receiver import configured_position_sigma_m

    section = load_drone_config().data["sensors"]["gps"]
    # Error-free default GPS: the receiver reports its floor, not 1.35 m.
    assert configured_position_sigma_m(section) == (0.0, 0.0, 0.0, 0.0)
    model = GpsStatusModel(gps("receiver"), configured_position_sigma_m(section))
    status = model.update(False)
    assert status.horizontal_accuracy_m == status.vertical_accuracy_m == 0.05
    # Native noise and bias plus the error model add in quadrature.
    section["horizontal_position_m"] = {"bias": 0.3, "noise_stddev": 0.4}
    section["vertical_position_m"] = {"bias": 0.0, "noise_stddev": 0.6}
    section["error_model"].update(
        enabled=True,
        horizontal_std_m=1.2,
        white_horizontal_std_m=0.0,
        vertical_std_m=0.8,
        white_vertical_std_m=0.0,
    )
    section["multipath"].update(enabled=True, horizontal_std_m=2.0, vertical_std_m=3.0)
    sigma = configured_position_sigma_m(section)
    assert sigma == pytest.approx((1.3, 1.0, 2.0, 3.0))
    model = GpsStatusModel(gps("receiver", satellite_variation=0), sigma)
    nominal = model.update(False)  # 14 satellites: HDOP at its nominal 0.9
    assert nominal.horizontal_accuracy_m == pytest.approx(1.3)
    assert nominal.vertical_accuracy_m == pytest.approx(1.0)
    degraded = model.update(True)  # multipath: extra σ and fewer satellites
    scale = degraded.hdop / 0.9
    assert degraded.horizontal_accuracy_m == pytest.approx(math.hypot(1.3, 2.0) * scale)
    assert degraded.vertical_accuracy_m == pytest.approx(math.hypot(1.0, 3.0) * scale)
