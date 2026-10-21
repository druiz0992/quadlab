"""Contract test kit: templates and built-ins pass, broken plug-ins fail."""

import numpy as np
import pytest

from dronelab.control.manual import ManualRotorController
from dronelab.control.template import TemplateController
from dronelab.core.measurements import EstimatedState, Sample
from dronelab.estimation.gps_imu_ekf.plugin import GpsImuEkfEstimator
from dronelab.estimation.template import TemplateEstimator
from dronelab.simulation.gazebo.config import load_drone_config
from dronelab.testing.plugins import (
    HOVER_POSITION_M,
    PluginContractError,
    check_controller,
    check_estimator,
    enu_to_gps,
)
from dronelab.sensors.geodesy import gps_to_enu
from dronelab.testing.plugins import REFERENCE


def test_templates_and_builtins_pass():
    template = check_estimator(TemplateEstimator, {"initial_position_m": [0, 0, 1]})
    assert template.estimates == 2000 and template.final_position_error_m < 0.1
    assert template.debug_entries == 100  # one gps_error entry per fix
    ekf_options = dict(load_drone_config().data["ekf"], initial_position_m=[0, 0, 1])
    ekf = check_estimator(GpsImuEkfEstimator, ekf_options)
    assert ekf.innovations == 100 and ekf.final_position_error_m < 0.1
    for factory in (TemplateController, ManualRotorController):
        report = check_controller(factory)
        assert report.total_thrust_n == pytest.approx(report.weight_n, rel=0.01)


def test_synthetic_gps_is_at_the_requested_offset():
    np.testing.assert_allclose(
        gps_to_enu(enu_to_gps((12.0, -7.0, 3.0)), REFERENCE), [12, -7, 3], atol=1e-3
    )


def test_templates_reject_unknown_options():
    with pytest.raises(ValueError, match="unknown options"):
        check_estimator(TemplateEstimator, {"gian": 1})
    with pytest.raises(ValueError, match="unknown options"):
        check_controller(TemplateController, {"gian": 1})


class GoodEstimator:
    inputs = ("imu",)

    def __init__(self, context, options):
        self.latest_sample, self.n = None, 0
        self.bad = options.get("bad")

    def state(self):
        covariance = np.eye(15)
        position = np.array(HOVER_POSITION_M)
        if self.bad == "nan":
            position = np.array([np.nan, 0, 1])
        if self.bad == "asymmetric":
            covariance[0, 1] = 1.0
        if self.bad == "negative":
            covariance[0, 0] = -1.0
        if self.bad == "random":
            position = position + np.random.default_rng().normal(0, 1e-3, 3)
        return EstimatedState(
            position,
            np.zeros(3),
            np.array([0, 0, 0, 1.0]),
            np.zeros(3),
            np.zeros(3),
            covariance,
        )

    def on_sample(self, kind, sample):
        self.n += 1
        t = sample.sim_time_ns
        if self.bad == "backwards" and self.n == 5:
            t -= 10_000_000
        if self.bad == "future":
            t += 1_000_000_000
        self.latest_sample = Sample("e", self.n, 0, t, 0, "world", self.state())

    def latest(self):
        return self.latest_sample

    def drain_innovations(self):
        return []

    def summary(self):
        return {"bad": object()} if self.bad == "summary" else {}


@pytest.mark.parametrize(
    "bad, message",
    [
        ("asymmetric", "symmetric"),
        ("negative", "semi-definite"),
        ("random", "differ between two identical runs"),
        ("backwards", "must not decrease"),
        ("future", "ahead of the newest sample"),
        ("summary", "JSON"),
    ],
)
def test_broken_estimators_are_caught(bad, message):
    check_estimator(GoodEstimator, {}, duration_s=1)
    with pytest.raises(PluginContractError, match=message):
        check_estimator(GoodEstimator, {"bad": bad}, duration_s=1)


class BadInputs(GoodEstimator):
    inputs = ("teleport",)


def test_unknown_inputs_are_caught():
    with pytest.raises(PluginContractError, match="inputs"):
        check_estimator(BadInputs, {}, duration_s=1)


class BrokenController(TemplateController):
    def __init__(self, context, options):
        self.bad = options.pop("bad")
        super().__init__(context, options)

    def update_attitude(self, state):
        thrust = super().update_attitude(state)
        if self.bad == "length":
            return thrust[:3]
        if self.bad == "negative":
            return thrust - 10
        if self.bad == "limit":
            self.saturated = False
            return thrust + 100
        if self.bad == "weight":
            return thrust * 0.5
        if self.bad == "random":
            return thrust + np.random.default_rng().uniform(0, 1e-6, len(thrust))
        return thrust


@pytest.mark.parametrize(
    "bad, message",
    [
        ("length", "rotor thrusts"),
        ("negative", "non-negative"),
        ("limit", "rotor limit"),
        ("weight", "differs from the weight"),
        ("random", "differ between two identical runs"),
    ],
)
def test_broken_controllers_are_caught(bad, message):
    with pytest.raises(PluginContractError, match=message):
        check_controller(BrokenController, {"bad": bad})


def test_plugin_errors_propagate_unchanged():
    # EstimatedState itself refuses NaN, so the plug-in's own call fails.
    with pytest.raises(ValueError, match="finite"):
        check_estimator(GoodEstimator, {"bad": "nan"}, duration_s=1)
