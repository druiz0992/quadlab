"""Scheduled IMU malfunctions: bias step, stuck axes, spikes, scale, extra noise.

Each event targets one instrument (``accel`` in m/s² or ``gyro`` in rad/s),
in IMU sensor axes, during ``[start_s, end_s)`` absolute simulation seconds
(``end_s: null`` means until the end of the run). Active events are applied
in configuration order to the reading produced by earlier stages:

| type        | parameter                    | effect per active sample                  |
| ----------- | ---------------------------- | ----------------------------------------- |
| bias_step   | value [3] (unit)             | reading + value                           |
| scale       | value [3] (unitless)         | reading * value                           |
| noise       | value [3] (unit, per sample) | reading + N(0, value²)                    |
| spikes      | value [3], probability       | with probability p: reading + N(0, value²) |
| stuck       | axes [3] (booleans)          | selected axes hold their first faulted value |

These are digital output faults applied after vibration/clipping, so spikes
can exceed the configured clip limit. Samples are never removed, so the EKF's
IMU-interval checks are unaffected. Random draws use one local seeded RNG.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...core.measurements import readonly_array

INSTRUMENTS = ("accel", "gyro")
EVENT_TYPES = ("bias_step", "scale", "noise", "spikes", "stuck")


class _FaultEvent:
    """One configured fault with its own activity window and diagnostics."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.type: str = settings["type"]
        self.instrument: str = settings["instrument"]
        self.start_ns = round(settings["start_s"] * 1e9)
        end_s = settings["end_s"]
        self.end_ns = None if end_s is None else round(end_s * 1e9)
        self.value = np.asarray(settings.get("value", [0.0] * 3), dtype=float)
        self.axes = np.asarray(settings.get("axes", [False] * 3), dtype=bool)
        self.probability: float = settings.get("probability", 0.0)
        self.held: NDArray[np.float64] | None = None
        self.samples = 0
        self.spikes = 0
        self.first_ns: int | None = None
        self.last_ns: int | None = None

    def active(self, sim_time_ns: int) -> bool:
        return self.start_ns <= sim_time_ns and (
            self.end_ns is None or sim_time_ns < self.end_ns
        )

    def apply(
        self, reading: NDArray[np.float64], sim_time_ns: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Return the faulted reading for one active sample."""
        self.samples += 1
        self.first_ns = sim_time_ns if self.first_ns is None else self.first_ns
        self.last_ns = sim_time_ns
        if self.type == "bias_step":
            return reading + self.value
        if self.type == "scale":
            return reading * self.value
        if self.type == "noise":
            return reading + self.value * rng.normal(size=3)
        if self.type == "spikes":
            return self._spike(reading, rng)
        return self._stuck(reading)

    def _spike(
        self, reading: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        # Draw both values every sample so the sequence depends only on time.
        occurs = rng.random() < self.probability
        magnitude = self.value * rng.normal(size=3)
        if not occurs:
            return reading
        self.spikes += 1
        return reading + magnitude

    def _stuck(self, reading: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.held is None:
            self.held = reading.copy()
        return np.where(self.axes, self.held, reading)

    def summary(self) -> dict[str, Any]:
        return dict(
            type=self.type,
            instrument=self.instrument,
            samples=self.samples,
            spikes=self.spikes,
            first_sim_time_ns=self.first_ns,
            last_sim_time_ns=self.last_ns,
        )


class ImuFaults:
    """Apply scheduled faults to time-ordered (specific force, angular rate) samples.

    ``settings`` is the validated ``sensors.imu.faults`` section. Enabled when
    at least one event is configured.
    """

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self._events = [_FaultEvent(event) for event in settings["events"]]
        # Manual overrides (e.g. keyboard toggles), active while present.
        self._overrides: dict[str, _FaultEvent] = {}
        self._rng = np.random.default_rng(settings["seed"])
        self._previous_time_ns: int | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._events or self._overrides)

    def set_override(self, name: str, event: Mapping[str, Any] | None) -> None:
        """Add (event settings without start/end) or remove (None) an override."""
        if event is None:
            self._overrides.pop(name, None)
        else:
            self._overrides[name] = _FaultEvent(dict(event, start_s=0.0, end_s=None))

    def apply(
        self,
        specific_force_mps2: ArrayLike,
        angular_rate_rad_s: ArrayLike,
        sim_time_ns: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return faulted (specific force, angular rate); identity when no events."""
        readings = {
            "accel": readonly_array(specific_force_mps2, (3,)),
            "gyro": readonly_array(angular_rate_rad_s, (3,)),
        }
        if not self.enabled:
            return readings["accel"], readings["gyro"]
        self._check_order(sim_time_ns)
        for event in [*self._events, *self._overrides.values()]:
            if event.active(sim_time_ns):
                readings[event.instrument] = event.apply(
                    readings[event.instrument], sim_time_ns, self._rng
                )
        return readings["accel"], readings["gyro"]

    def _check_order(self, sim_time_ns: int) -> None:
        previous = self._previous_time_ns
        if sim_time_ns < 0 or (previous is not None and sim_time_ns <= previous):
            raise ValueError(
                "IMU fault timestamps must be nonnegative and strictly increasing"
            )
        self._previous_time_ns = sim_time_ns

    def active_count(self, sim_time_ns: int) -> int:
        """Number of fault events active at sim_time_ns (for status display)."""
        events = [*self._events, *self._overrides.values()]
        return sum(event.active(sim_time_ns) for event in events)

    def summary(self) -> dict[str, Any]:
        """Per-event diagnostics for summary.json, for evaluation only."""
        return dict(
            enabled=self.enabled, events=[event.summary() for event in self._events]
        )
