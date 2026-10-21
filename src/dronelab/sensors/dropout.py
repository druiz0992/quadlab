"""Seeded loss of simulated GPS fixes; never substitutes synthetic readings."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .zones import inside_any, zones_from_settings


class GpsDropout:
    """Consume ordered fixes, applying random loss, timed outages, and zones.

    Times are absolute simulation seconds; outages are [start, end). Zones are
    world boxes where every fix is lost (e.g. under a structure). Settings are
    validated by the drone configuration loader. The RNG is local and never
    changes global noise state. Precedence in counters: scheduled, zone, random.
    """

    def __init__(
        self,
        *,
        probability: float = 0.0,
        seed: int = 0,
        outages_s: Sequence[Sequence[float]] = (),
        zones: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self._probability = probability
        self._rng = np.random.default_rng(seed)
        self._outages_ns = tuple(
            (round(start * 1e9), round(end * 1e9)) for start, end in outages_s
        )
        self._zones = zones_from_settings(zones)
        self.received = 0
        self.random_drops = 0
        self.scheduled_drops = 0
        self.zone_drops = 0
        # Manual blackout (e.g. keyboard toggle): drop every fix while True.
        self.forced = False
        self.manual_drops = 0

    def drop(self, sim_time_ns: int, position_m: ArrayLike | None = None) -> bool:
        """Return True if this fix is lost.

        position_m is the vehicle's world position at the fix time, used only
        to evaluate blockout zones; None means zones cannot apply.
        """
        self.received += 1
        # Draw independently of outages/zones for repeatable combinations.
        random_drop = self._rng.random() < self._probability
        if self.forced:
            self.manual_drops += 1
            return True
        if any(start <= sim_time_ns < end for start, end in self._outages_ns):
            self.scheduled_drops += 1
            return True
        if inside_any(self._zones, position_m):
            self.zone_drops += 1
            return True
        if random_drop:
            self.random_drops += 1
            return True
        return False

    def summary(self) -> dict[str, int]:
        dropped = (
            self.random_drops
            + self.scheduled_drops
            + self.zone_drops
            + self.manual_drops
        )
        return dict(
            received=self.received,
            delivered=self.received - dropped,
            random_drops=self.random_drops,
            scheduled_drops=self.scheduled_drops,
            zone_drops=self.zone_drops,
            manual_drops=self.manual_drops,
        )
