import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.experiments.trailer import Trailer, trailer_state
from robotlab.simulation.gazebo.config import load_drone_config
from robotlab.simulation.wheeled.ground import GroundModel
from robotlab.simulation.wheeled.model import ChassisState


@pytest.fixture
def setup(tmp_path):
    layer = tmp_path / "layer.json"
    layer.write_text('{"ground_vehicle": {"trailer": {"enabled": true}}}')
    config = load_drone_config(layer, robot="tractor")
    return config.data["ground_vehicle"]["trailer"], config.wheeled


def tractor(yaw=0.0, yaw_rate=0.0, speed=0.0):
    q = Rotation.from_euler("z", yaw).as_quat()
    v = Rotation.from_euler("z", yaw).apply([speed, 0, 0])
    return ChassisState(np.array([1.0, 2.0, 1.0]), q, v, np.array([0, 0, yaw_rate]))


def test_straight_hitch_geometry(setup):
    settings, _ = setup
    state = trailer_state(tractor(), settings, 0.0, 0.0, 0.0, 0.0)
    hx, hy, hz = settings["hitch_position_m"]
    expected = np.array(
        [1 + hx - settings["drawbar_m"], 2 + hy, 1 + hz - settings["cg_below_hitch_m"]]
    )
    np.testing.assert_allclose(state.position_m, expected)
    np.testing.assert_allclose(state.quaternion_xyzw, [0, 0, 0, 1], atol=1e-12)


def test_velocity_matches_finite_difference(setup):
    settings, _ = setup
    dt = 1e-6
    yaw, yaw_rate, art, art_rate, pitch, pitch_rate = 0.3, 0.2, -0.4, 0.5, 0.05, -0.1
    state = trailer_state(
        tractor(yaw, yaw_rate, 3.0), settings, art, pitch, art_rate, pitch_rate
    )
    moved = tractor(yaw + yaw_rate * dt, yaw_rate, 3.0)
    moved = ChassisState(
        moved.position_m + moved.velocity_world_mps * dt,
        moved.quaternion_xyzw,
        moved.velocity_world_mps,
        moved.body_rates_rad_s,
    )
    later = trailer_state(
        moved,
        settings,
        art + art_rate * dt,
        pitch + pitch_rate * dt,
        art_rate,
        pitch_rate,
    )
    np.testing.assert_allclose(
        (later.position_m - state.position_m) / dt, state.velocity_world_mps, atol=1e-4
    )
    relative = (
        Rotation.from_quat(later.quaternion_xyzw)
        * Rotation.from_quat(state.quaternion_xyzw).inv()
    )
    np.testing.assert_allclose(
        relative.as_rotvec() / dt,
        Rotation.from_quat(state.quaternion_xyzw).apply(state.body_rates_rad_s),
        atol=1e-4,
    )


def test_trailer_properties_share_the_tractor_sag(setup):
    settings, tractor_props = setup

    class Wrench:
        def set(self, force, torque):
            self.force = force

    trailer = Trailer(settings, tractor_props, GroundModel(None, [], 100.0), Wrench())
    assert trailer.properties.ride_drop_m == pytest.approx(tractor_props.ride_drop_m)
    assert not trailer.properties.mask("driven").any()
    assert trailer.brake_fraction((0.0, 3000.0, 0.0, 3000.0)) == pytest.approx(0.25)
    joint = lambda name, position, velocity: types.SimpleNamespace(
        name=name, axis1=types.SimpleNamespace(position=position, velocity=velocity)
    )
    message = types.SimpleNamespace(
        joint=[joint("trailer_yaw", 1.3, 0.0), joint("trailer_pitch", 0.0, 0.0)],
        header=types.SimpleNamespace(
            stamp=types.SimpleNamespace(sec=0, nsec=2_000_000)
        ),
    )
    trailer.on_joint_state(message)
    assert trailer.step(2_000_000, tractor(), 0.0, 0.002)  # 74° > 0.9 × 80°: jackknifed
    assert trailer.stale_periods == 0
    with pytest.raises(RuntimeError, match="trailer_yaw"):
        trailer.on_joint_state(types.SimpleNamespace(joint=[], header=message.header))
