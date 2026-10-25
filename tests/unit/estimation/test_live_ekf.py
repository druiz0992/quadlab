from types import SimpleNamespace as NS

import numpy as np
import pytest

from robotlab.core.measurements import GpsReading
from robotlab.experiments.harness.live_ekf import LiveEkf
from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.sensors.streams import ReadTimeout


def message(kind, stamp):
    header = NS(stamp=NS(sec=stamp // 1_000_000_000, nsec=stamp % 1_000_000_000))
    if kind == "imu":
        return NS(
            header=header,
            linear_acceleration=NS(x=0, y=0, z=9.81),
            angular_velocity=NS(x=0, y=0, z=0),
        )
    return NS(header=header, latitude_deg=40, longitude_deg=-3, altitude=650)


def test_reordering_and_intermediate_gps_without_truth():
    live = LiveEkf(load_drone_config().data, GpsReading(40, -3, 650))
    reader = live.estimates.subscribe(capacity=64)
    try:
        # Both sources stay in order, but their arrivals interleave arbitrarily.
        for kind, t in [
            ("imu", 0),
            ("imu", 10_000_000),
            ("imu", 20_000_000),
            ("gps", 0),
            ("gps", 15_000_000),
            ("imu", 30_000_000),
            ("gps", 30_000_000),
        ]:
            live.ingest(kind, message(kind, t))
        samples = []
        while True:
            try:
                samples.append(reader.next(timeout_wall_s=0))
            except ReadTimeout:
                break
        assert [sample.sim_time_ns for sample in samples] == sorted(
            sample.sim_time_ns for sample in samples
        )
        # The first fix initializes the position; the other three correct.
        assert live.ekf.position_initialized
        assert live.ekf.accepted_gps == 3 - 1
        assert live.node.rejected_timing["gps"] == 0
        np.testing.assert_allclose(samples[-1].payload.position_m, 0, atol=1e-10)
        assert "no simulator truth" in live.summary()["initialization"]
    finally:
        reader.close()
        live.close()


def test_vibrating_acceleration_reaches_ekf_without_modifying_gyro_or_timestamps():
    from robotlab.sensors.gazebo import vibrate_imu_message
    from robotlab.sensors.vibration import AccelerometerVibration

    settings = load_drone_config().data
    config = settings["sensors"]["imu"]["vibration"]
    config.update(enabled=True, accel_amplitude_mps2=[2, 0, 0], phase_deg=[90, 0, 0])
    vibration = AccelerometerVibration(config)
    live = LiveEkf(settings, GpsReading(40, -3, 650))
    reader = live.estimates.subscribe(capacity=16)
    try:
        for stamp in (0, 10_000_000):
            raw = message("imu", stamp)
            measured = vibrate_imu_message(raw, vibration, stamp)
            assert raw.linear_acceleration.x == 0
            assert measured.angular_velocity == raw.angular_velocity
            live.ingest("imu", measured)
            live.ingest("gps", message("gps", stamp), gps_dropped=stamp > 0)
        np.testing.assert_allclose(live.ekf.velocity, [0.02, 0, 0], atol=1e-12)
        np.testing.assert_allclose(
            live.ekf.rotation.as_quat(), [0, 0, 0, 1], atol=1e-12
        )
        assert (
            live.estimates.latest(
                now_sim_time_ns=10_000_000, max_age_sim_s=0
            ).sim_time_ns
            == 10_000_000
        )
    finally:
        reader.close()
        live.close()


def test_missing_stream_watchdog_and_overflow_fail_visibly():
    now = [0.0]
    settings = load_drone_config().data
    settings["ekf"]["reorder_capacity"] = 1
    live = LiveEkf(settings, GpsReading(40, -3, 650), monotonic=lambda: now[0])
    try:
        live.ingest("imu", message("imu", 0))
        with pytest.raises(RuntimeError, match="overflow"):
            live.ingest("imu", message("imu", 10_000_000))
        now[0] = 6
        with pytest.raises(RuntimeError, match="watchdog"):
            live.check_health()
    finally:
        live.close()


def test_nonorigin_imu_is_rejected_before_run():
    settings = load_drone_config().data
    settings["sensors"]["imu"]["mount"]["position_m"] = [0.1, 0, 0]
    with pytest.raises(ValueError, match="body-origin"):
        LiveEkf(settings, GpsReading(40, -3, 650))


def test_stale_queue_receipt_is_not_treated_as_fresh():
    live = LiveEkf(
        load_drone_config().data, GpsReading(40, -3, 650), monotonic=lambda: 10
    )
    try:
        with pytest.raises(RuntimeError, match="watchdog"):
            live.ingest("imu", message("imu", 0), received_at=0)
    finally:
        live.close()


def test_modeled_gps_outage_keeps_imu_prediction_and_recovers_without_sequence_gap():
    now = [0.0]
    settings = load_drone_config().data
    # Compares covariance growth with the declared prior: no fix replaces it.
    settings["ekf"]["initialize_position_from_gps"] = False
    live = LiveEkf(settings, GpsReading(40, -3, 650), monotonic=lambda: now[0])
    reader = live.estimates.subscribe(capacity=4096)
    initial_covariance = None
    try:
        for tick in range(701):
            stamp = tick * 10_000_000
            now[0] = tick * 0.01
            live.ingest("imu", message("imu", stamp))
            if tick % 10 == 0:
                outage = 0 < tick < 600
                # No position payload is needed or inspected for a lost fix.
                gps = message("gps", stamp)
                if outage:
                    gps = NS(header=gps.header)
                live.ingest("gps", gps, gps_dropped=outage)
            if tick == 0:
                initial_covariance = live.ekf.P.copy()
            if tick == 590:
                assert live.ekf.accepted_gps == 1
                assert live.ekf.P[0, 0] > initial_covariance[0, 0]
                assert (
                    live.estimates.latest(
                        now_sim_time_ns=stamp, max_age_sim_s=0.11
                    ).sim_time_ns
                    >= stamp - 100_000_000
                )
        assert live.summary()["gps_modeled_drops"] == 59
        assert live.ekf.accepted_gps == 12
        assert live.node.rejected_timing["gps"] == 0
        assert live.node._last["gps"][1] == 12
        # Real transport disappearance still trips health checks.
        now[0] = 13
        with pytest.raises(RuntimeError, match="watchdog"):
            live.check_health()
    finally:
        reader.close()
        live.close()
