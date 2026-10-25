"""Keyboard-flight wiring: effect toggles, actions, and their event log.

Only used with ``--keyboard``. Connects the Tk window's effect keys
(``effect_toggles.py``) to the run's models (powertrain, wind, sensor
faults, payload, flying-object spawner) and records every applied effect
in ``effect-toggles.jsonl`` and Rerun, stamped with simulation time.
"""

from collections.abc import Callable
import json
from typing import Any, TextIO

import numpy as np

from ...simulation.gazebo.entities import EntityService
from ...simulation.gazebo.world import moving_obstacle_sdf
from .effect_toggles import ACTIONS, EFFECTS, EffectToggles
from .flying_object import FlyingObjectSpawner
from ..common.sensor_pipeline import SensorPipeline, TruthSnapshot


def keyboard_effects(
    keyboard: Any,
    settings: dict[str, Any],
    *,
    node: Any,
    recorder: Any,
    moving: Any,
    payload: Any | None,
    powertrain: Any,
    wind_control: Any | None,
    pipeline: SensorPipeline,
    truth: TruthSnapshot,
    sim_time_s: Callable[[], float],
) -> EffectToggles:
    """Build the effect toggles and show their keys in the keyboard window.

    ``sim_time_s()`` is the latest odometry time (0 before the first).
    """
    flying_object = FlyingObjectSpawner(
        settings["keyboard_effects"],
        entities=EntityService(node),
        moving=moving,
        recorder=recorder,
        sdf_text=moving_obstacle_sdf,
        drone_state=lambda: (
            truth.position_m if truth.position_m is not None else np.zeros(3),
            sim_time_s(),
        ),
    )

    def payload_toggle(on: bool, stamp: int) -> Any:
        """Payload switch: off drops, on tries a pickup."""
        if payload is None:
            raise RuntimeError("payload effect toggled without a payload")
        if not on:
            return payload.drop(stamp)
        return payload.pickup(stamp, truth.position_m, truth.attitude.as_quat())

    effects = EffectToggles(
        settings["keyboard_effects"],
        flying_object=flying_object,
        payload=payload_toggle if payload is not None else None,
        powertrain=powertrain,
        wind=wind_control,
        gps_dropout=pipeline.gps_dropout,
        gps_multipath=pipeline.gps_multipath,
        imu_faults=pipeline.imu_faults,
        imu_temperature=pipeline.imu_temperature,
        ambient_c=settings["environment"]["ambient_temperature_c"],
    )
    keyboard.configure_actions(
        [(a.key, a.name, a.label, effects.action_available(a.name)) for a in ACTIONS]
    )
    keyboard.configure_effects(
        [(e.key, e.name, e.label, effects.available(e.name)) for e in EFFECTS],
        effects.state,
    )
    return effects


def apply_keyboard_effects(
    keyboard: Any,
    effects: EffectToggles,
    sim_time_ns: int,
    effect_log: TextIO,
    recorder: Any,
) -> None:
    """Perform pressed actions and switched toggles; log what took effect."""
    events = [effects.perform(name, sim_time_ns) for name in keyboard.drain_actions()]
    for name, on in keyboard.drain_toggles():
        events.append(effects.set(name, on, sim_time_ns))
    for event in events:
        if event is not None:
            effect_log.write(json.dumps(event) + "\n")
            recorder.log_effect(event)
