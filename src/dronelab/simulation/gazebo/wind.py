"""Seeded wind targets; native Gazebo WindEffects applies smoothing and forces."""

import bisect
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..aerodynamics import DrydenTurbulence


class WindProfile:
    """Piecewise mean velocity with held Gaussian speed/direction perturbations,
    plus optional Dryden-style turbulence.

    Updates use absolute simulation time. Fixed RNG ticks make results independent
    of odometry frequency. Native rise_time_s smooths these targets, while the
    existing native sinusoidal gust is applied on top of the smoothed mean.
    Turbulence has its own RNG and tick grid, so enabling it does not change
    the random speed/direction sequence.
    """

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self._base = settings["velocity_enu_mps"]
        self._events = settings["schedule"]
        self._times = [round(e["time_s"] * 1e9) for e in self._events]
        self._period_ns = max(1, round(settings["random_update_period_s"] * 1e9))
        self._speed_std = settings["random_speed_std_mps"]
        self._direction_std = math.radians(settings["random_direction_std_deg"])
        self._rng = np.random.default_rng(settings["random_seed"])
        self._tick = -1
        self._last_key: tuple[int, int] | None = None
        self._noise = np.zeros(2)
        self._last_time = -1
        self._turbulence = DrydenTurbulence(settings["turbulence"])

    def force_next(self) -> None:
        """Make the next ``target`` call return the current target."""
        self._last_key = None

    def target(self, sim_time_ns: int) -> tuple[float, float, float] | None:
        """Return a new target at tick/event changes, otherwise None."""
        if sim_time_ns < 0 or sim_time_ns < self._last_time:
            raise ValueError("Wind time must be nonnegative and nondecreasing")
        self._last_time = sim_time_ns
        tick = sim_time_ns // self._period_ns
        for _ in range(self._tick + 1, tick + 1):
            self._noise = self._rng.normal(size=2)
        self._tick = tick
        event = bisect.bisect_right(self._times, sim_time_ns) - 1
        mean = np.asarray(
            self._base if event < 0 else self._events[event]["velocity_enu_mps"],
            dtype=float,
        )
        turbulence_changed = self._turbulence.advance(sim_time_ns, mean)
        key = (tick, event)
        if key == self._last_key and not turbulence_changed:
            return None
        self._last_key = key
        x, y, z = mean
        speed = max(0.0, math.hypot(x, y) + self._noise[0] * self._speed_std)
        direction = math.atan2(y, x) + self._noise[1] * self._direction_std
        gusty = np.array([speed * math.cos(direction), speed * math.sin(direction), z])
        vx, vy, vz = gusty + self._turbulence.velocity_enu(mean)
        return (float(vx), float(vy), float(vz))


class WindCommands:
    """Publish wind seed velocities to the native world's WindEffects plugin."""

    def __init__(self, node: Any, settings: Mapping[str, Any]) -> None:
        from gz.msgs10.wind_pb2 import Wind

        self._message_type = Wind
        self._publisher = node.advertise("/world/dronelab/wind", Wind)
        self._profile = WindProfile(settings)
        self._on = True
        self._zero_sent = False
        # Latest published wind target (ENU m/s), e.g. for body airspeed.
        self.velocity_enu_mps = (0.0, 0.0, 0.0)

    def set_enabled(self, on: bool) -> None:
        """Manual wind switch: off publishes still air; on resumes the profile."""
        self._on = on
        self._zero_sent = False
        if on:
            self._profile.force_next()

    def update(self, sim_time_ns: int) -> dict[str, Any] | None:
        velocity = self._profile.target(sim_time_ns)
        if not self._on:
            # Keep the profile's random sequence advancing while switched off.
            if self._zero_sent:
                return None
            self._zero_sent = True
            velocity = (0.0, 0.0, 0.0)
        if velocity is None:
            return None
        message = self._message_type()
        message.enable_wind = True
        (
            message.linear_velocity.x,
            message.linear_velocity.y,
            message.linear_velocity.z,
        ) = velocity
        if not self._publisher.publish(message):
            raise RuntimeError("Unable to publish wind target")
        vx, vy, vz = velocity
        self.velocity_enu_mps = (float(vx), float(vy), float(vz))
        return dict(sim_time_ns=sim_time_ns, velocity_enu_mps=list(velocity))


def dynamic_wind(settings: Mapping[str, Any]) -> bool:
    return bool(
        settings["enabled"]
        and (
            settings["schedule"]
            or settings["random_speed_std_mps"]
            or settings["random_direction_std_deg"]
            or settings["turbulence"]["enabled"]
        )
    )
