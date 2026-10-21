"""Per-beam lidar return physics: detection, weather, beam spread."""

import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from dronelab.sensors.lidar_effects import LidarEffects
from dronelab.sensors.lidar_returns import (
    decode_scan,
    detected,
    extinction_per_m,
    mix_edge_returns,
    weather_clutter,
)
from dronelab.simulation.gazebo.config import load_drone_config

DETECTION = dict(
    enabled=True,
    reference_range_m=40.0,
    reference_reflectivity=0.1,
    default_reflectivity=0.5,
)


def scan(ranges, intensities=None, rows=1, step=0.01):
    return NS(
        ranges=list(ranges),
        intensities=list(intensities) if intensities is not None else [],
        angle_min=0.0,
        angle_step=step,
        vertical_count=rows,
        vertical_angle_min=0.0,
        vertical_angle_step=0.02,
        range_min=0.1,
        range_max=100.0,
    )


def test_decode_maps_retro_to_reflectivity_with_default_for_unset():
    grid = decode_scan(scan([5, 5, math.inf, 5], [200, 0, 0, 20], rows=2), 0.5)
    assert grid.ranges_m.shape == (2, 2)
    np.testing.assert_allclose(grid.reflectivity, [[2.0, 0.5], [0.5, 0.2]])
    assert grid.elevation_step_rad == pytest.approx(0.02)


def test_detection_range_scales_with_square_root_of_reflectivity():
    ranges = np.array([39.9, 40.1, 89.0, 90.0])
    rho = np.array([0.1, 0.1, 0.5, 0.5])  # 0.5 reaches 40·sqrt(5) = 89.4 m
    np.testing.assert_array_equal(
        detected(ranges, rho, 0.0, DETECTION), [True, False, True, False]
    )
    assert not detected(np.array([math.inf]), np.array([1.0]), 0.0, DETECTION)[0]


def test_extinction_follows_koschmieder_and_kim_wavelength_scaling():
    assert extinction_per_m(None) == 0.0
    # Fog below 500 m visibility: wavelength independent, 3.912 / V.
    assert extinction_per_m(100.0) == pytest.approx(0.03912)
    clear = extinction_per_m(20_000.0)
    assert clear == pytest.approx(3.912 / 20_000 * (905 / 550) ** -1.3)
    # Fog shortens the range for a 50 % target from 89 m to under 30 m.
    ranges = np.array([25.0, 30.0])
    alpha = extinction_per_m(100.0)
    np.testing.assert_array_equal(
        detected(ranges, np.full(2, 0.5), alpha, DETECTION), [True, False]
    )


def test_clutter_first_return_statistics_and_range_gate():
    rng = np.random.default_rng(0)
    ranges = np.full(200_000, 20.0)
    out, clutter = weather_clutter(ranges, 0.05, 10.0, 0.1, rng)
    # P(first event in [0.1, 10)) = exp(-0.005) - exp(-0.5).
    expected = math.exp(-0.005) - math.exp(-0.5)
    assert clutter.mean() == pytest.approx(expected, abs=0.005)
    assert out[clutter].max() < 10.0 and out[clutter].min() >= 0.1
    np.testing.assert_array_equal(out[~clutter], 20.0)
    # A target nearer than the clutter keeps its return; no-hit beams can clutter.
    near, c = weather_clutter(np.full(50_000, 1.0), 0.05, 10.0, 0.1, rng)
    assert c.mean() < 0.05 and np.all(near[c] < 1.0)
    sky, c = weather_clutter(np.full(1000, math.inf), 0.05, 10.0, 0.1, rng)
    assert c.any()


def test_edge_mixing_places_ghosts_between_surfaces():
    # Foreground at 5 m in columns 0-49, background at 10 m in 50-99.
    ranges = np.where(np.arange(100) < 50, 5.0, 10.0)[None, :].repeat(4000, 0)
    rho = np.ones_like(ranges)
    out, eff, mixed = mix_edge_returns(
        ranges, rho, 0.01, 0.006, 0.3, np.random.default_rng(1)
    )
    assert mixed[:, [49, 50]].any() and not mixed[:, :49].any()
    assert not mixed[:, 51:].any()
    ghosts = out[mixed]
    assert np.all((ghosts > 5.0) & (ghosts < 10.0))
    # Each beam overlaps the edge with probability (δ/2)/Δ = 0.3.
    assert mixed[:, 49].mean() == pytest.approx(0.3, abs=0.03)
    # Received power is conserved: share-weighted sum of ρ/r².
    row = np.nonzero(mixed[:, 49])[0][0]
    assert eff[row, 49] / out[row, 49] ** 2 < 1 / 25 and eff[row, 49] > 0


def test_edge_against_sky_loses_energy_without_moving_the_point():
    ranges = np.array([[5.0, math.inf]]).repeat(2000, 0)
    out, eff, mixed = mix_edge_returns(
        ranges, np.ones_like(ranges), 0.01, 0.02, 0.3, np.random.default_rng(2)
    )
    hit = mixed[:, 0]
    np.testing.assert_allclose(out[hit, 0], 5.0)
    assert np.all(eff[hit, 0] < 1.0) and np.all(eff[hit, 0] >= 0.5)
    # The sky beam grazing the object sees it, weakly (objects look wider).
    sky = mixed[:, 1]
    np.testing.assert_allclose(out[sky, 1], 5.0)
    assert np.all(eff[sky, 1] <= 0.5)
    assert np.all(np.isinf(out[~sky, 1]))


def test_effects_pipeline_counts_and_disabled_passthrough():
    settings = load_drone_config().data["sensors"]["lidar"]
    settings["enabled"] = True
    message = scan([5.0] * 50 + [math.inf] * 50, [10] * 100)
    effects = LidarEffects(settings)
    assert not effects.enabled
    settings["effects"]["detection"] = DETECTION
    weather = dict(
        visibility_m=None,
        lidar_clutter_per_m=0.0,
        lidar_clutter_max_range_m=10.0,
        lidar_clutter_reflectivity=0.05,
    )
    effects = LidarEffects(settings, weather)
    # Retro 10 → ρ 0.1, detectable to 40 m: all 50 hits survive.
    assert len(effects.process_scan(message, [0, 0, 0], [0, 0, 0])) == 50
    dark = scan([5.0] * 50, [1] * 50)  # ρ 0.01: limit 12.6 m, still seen
    assert len(effects.process_scan(dark, [0, 0, 0], [0, 0, 0])) == 50
    far_dark = scan([20.0] * 50, [1] * 50)
    assert len(effects.process_scan(far_dark, [0, 0, 0], [0, 0, 0])) == 0
    assert effects.summary()["undetected_returns"] == 50


def test_weather_requires_detection_when_lidar_enabled(tmp_path):
    import json

    layer = tmp_path / "fog.json"
    layer.write_text(
        json.dumps(
            {
                "environment": {"weather": {"visibility_m": 100}},
                "sensors": {"lidar": {"enabled": True}},
            }
        )
    )
    with pytest.raises(ValueError, match="detection"):
        load_drone_config(layer, render_sensors=True)


def test_example_weather_layers_load():
    for name in ("fog", "rain", "dust"):
        data = load_drone_config(
            [
                "configs/sensors/lidar_livox_mid360.json",
                f"configs/environment/{name}.json",
            ],
            render_sensors=True,
        ).data
        assert data["sensors"]["lidar"]["effects"]["detection"]["enabled"]
        assert LidarEffects(
            data["sensors"]["lidar"], data["environment"]["weather"]
        ).enabled


def effects(mode="first", resolution=None, divergence=0.0, weather=None, **extra):
    settings = load_drone_config().data["sensors"]["lidar"]
    settings["enabled"] = True
    settings["effects"].update(
        return_mode=mode,
        range_resolution_m=resolution,
        beam_divergence_rad=divergence,
        detection=DETECTION,
        **extra,
    )
    return LidarEffects(settings, weather)


def edge_scan(rows=2000):
    """Foreground at 5 m (ρ 0.5) in columns 0–4, background at 12 m (ρ 0.8)."""
    ranges = ([5.0] * 5 + [12.0] * 5) * rows
    retro = ([50] * 5 + [80] * 5) * rows
    return scan(ranges, retro, rows=rows, step=0.01)


def test_resolved_edges_give_two_returns_in_dual_mode():
    dual = effects("dual", resolution=0.5, divergence=0.008)
    points = dual.process_scan(edge_scan(), [0, 0, 0], [0, 0, 0])
    ranges = np.linalg.norm(points, axis=1)
    # Separated surfaces: no ghosts between them, and some second returns.
    assert not np.any((ranges > 5.01) & (ranges < 11.99))
    assert dual.summary()["second_returns"] > 0
    assert len(points) > 20000
    merged = effects("first", resolution=None, divergence=0.008)
    ghost = np.linalg.norm(
        merged.process_scan(edge_scan(), [0, 0, 0], [0, 0, 0]), axis=1
    )
    assert np.any((ghost > 5.01) & (ghost < 11.99))


def test_last_return_sees_through_rain_clutter():
    rain = dict(
        visibility_m=None,
        lidar_clutter_per_m=0.2,
        lidar_clutter_max_range_m=10.0,
        lidar_clutter_reflectivity=0.05,
    )
    wall = scan([8.0] * 5000, [50] * 5000)
    first = effects("first", weather=rain)
    first_ranges = np.linalg.norm(
        first.process_scan(wall, [0, 0, 0], [0, 0, 0]), axis=1
    )
    assert np.mean(first_ranges < 7.9) > 0.7  # most beams hit a drop first
    last = effects("last", weather=rain)
    last_ranges = np.linalg.norm(last.process_scan(wall, [0, 0, 0], [0, 0, 0]), axis=1)
    np.testing.assert_allclose(last_ranges, 8.0)
    assert last.summary()["weather_clutter_returns"] == 0


def test_reported_reflectivity_is_range_normalized_strength():
    fog = dict(
        visibility_m=100.0,
        lidar_clutter_per_m=0.0,
        lidar_clutter_max_range_m=10.0,
        lidar_clutter_reflectivity=0.05,
    )
    model = effects(weather=fog)
    model.process_scan(scan([10.0] * 10, [50] * 10), [0, 0, 0], [0, 0, 0])
    expected = 0.5 * math.exp(-2 * extinction_per_m(100.0) * 10.0)
    np.testing.assert_allclose(model.last_reflectivity, expected)
