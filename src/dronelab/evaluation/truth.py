"""Time-indexed simulator truth with interpolation to arbitrary sample times."""

from bisect import bisect_left
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class TruthSample:
    """Vehicle truth at one instant (world ENU; xyzw body-to-world)."""

    sim_time_ns: int
    position_m: NDArray[np.float64]
    velocity_mps: NDArray[np.float64]
    quaternion_xyzw: NDArray[np.float64]


class TruthHistory:
    """Bounded history of truth samples, interpolated between neighbours.

    Position and velocity interpolate linearly; attitude by slerp. Samples
    must arrive in increasing time. ``max_samples`` bounds memory; queries
    older than the retained window return None.
    """

    def __init__(self, max_samples: int = 20_000) -> None:
        self._times: list[int] = []
        self._samples: list[TruthSample] = []
        self._max = max_samples

    def add(
        self,
        sim_time_ns: int,
        position_m: ArrayLike,
        velocity_mps: ArrayLike,
        quaternion_xyzw: ArrayLike,
    ) -> None:
        if self._times and sim_time_ns <= self._times[-1]:
            return  # repeated stamp: keep the first
        self._times.append(sim_time_ns)
        self._samples.append(
            TruthSample(
                sim_time_ns,
                np.asarray(position_m, float),
                np.asarray(velocity_mps, float),
                np.asarray(quaternion_xyzw, float),
            )
        )
        if len(self._times) > self._max:
            del self._times[: len(self._times) // 2]
            del self._samples[: len(self._samples) // 2]

    @property
    def latest_ns(self) -> int | None:
        return self._times[-1] if self._times else None

    def at(self, sim_time_ns: int) -> TruthSample | None:
        """Truth at ``sim_time_ns``, or None outside the retained window."""
        times = self._times
        if not times or not times[0] <= sim_time_ns <= times[-1]:
            return None
        i = bisect_left(times, sim_time_ns)
        if times[i] == sim_time_ns:
            return self._samples[i]
        a, b = self._samples[i - 1], self._samples[i]
        w = (sim_time_ns - a.sim_time_ns) / (b.sim_time_ns - a.sim_time_ns)
        slerp = Slerp(
            [0.0, 1.0], Rotation.from_quat([a.quaternion_xyzw, b.quaternion_xyzw])
        )
        return TruthSample(
            sim_time_ns,
            (1 - w) * a.position_m + w * b.position_m,
            (1 - w) * a.velocity_mps + w * b.velocity_mps,
            slerp(w).as_quat(),
        )
