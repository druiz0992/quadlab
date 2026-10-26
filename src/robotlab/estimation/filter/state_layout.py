"""Named blocks of a filter's (error-)state vector and covariance.

A ``StateLayout`` gives each block of the state a name and a size, so
measurement models (models.py) find their Jacobian columns by name and a
filter can add a block (a baro offset, a magnetometer bias) without
renumbering every index. Standard names and their meaning:

- ``position`` (3): local ENU position error (m);
- ``velocity`` (3): ENU velocity error (m/s);
- ``attitude`` (3): right rotation-vector error in body axes (rad),
  R_true = R · Exp(δθ), RobotLab's convention (estimation/lie.py);
- ``yaw`` (1): heading error about world up (rad), R_true = R_z(δψ) · R,
  for planar filters that estimate yaw only (not together with
  ``attitude``);
- ``accel_bias`` (3, m/s²) and ``gyro_bias`` (3, rad/s), body axes.

Any other name is a filter-specific block of any size. Raise ValueError for
duplicate names, wrong sizes of standard blocks, or unknown blocks.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ...core.measurements import EstimatedState

STANDARD_SIZES = dict(
    position=3, velocity=3, attitude=3, yaw=1, accel_bias=3, gyro_bias=3
)
# Layout block -> EstimatedState field, in EstimatedState covariance order.
REPORTED_BLOCKS = (
    ("position", "position_m"),
    ("velocity", "velocity_mps"),
    ("attitude", "quaternion_xyzw"),
    ("accel_bias", "accel_bias_mps2"),
    ("gyro_bias", "gyro_bias_rad_s"),
)


@dataclass(frozen=True)
class StateLayout:
    """Ordered ``(name, size)`` blocks of a state vector (module docs)."""

    blocks: tuple[tuple[str, int], ...]

    @classmethod
    def of(cls, *blocks: str | tuple[str, int]) -> "StateLayout":
        """Layout from standard names and ``(name, size)`` pairs, in order,
        e.g. ``StateLayout.of("position", "velocity", ("baro_offset", 1))``."""
        return cls(tuple(_block(item) for item in blocks))

    def __post_init__(self) -> None:
        names = [name for name, _ in self.blocks]
        if not names or len(set(names)) != len(names):
            raise ValueError(f"State blocks must be unique and nonempty: {names}")
        for name, size in self.blocks:
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(f"State block {name!r} needs a positive int size")
            if name in STANDARD_SIZES and size != STANDARD_SIZES[name]:
                raise ValueError(
                    f"State block {name!r} has size {STANDARD_SIZES[name]}, not {size}"
                )
        if "attitude" in names and "yaw" in names:
            raise ValueError("A state has either an attitude or a yaw block")
        starts = np.cumsum([0] + [size for _, size in self.blocks])
        slices = {
            name: slice(int(start), int(start) + size)
            for (name, size), start in zip(self.blocks, starts)
        }
        object.__setattr__(self, "_slices", slices)

    @property
    def size(self) -> int:
        """Length of the state vector (rows of the covariance)."""
        return sum(size for _, size in self.blocks)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.blocks)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def index(self, name: str) -> slice:
        """The slice of block ``name``; raises ValueError for an unknown block."""
        slices: dict[str, slice] = getattr(self, "_slices")
        if name not in slices:
            raise ValueError(f"No state block {name!r} in {list(self.names)}")
        return slices[name]

    def jacobian(self, rows: int) -> NDArray[np.float64]:
        """A zero ``rows`` × ``size`` measurement Jacobian to fill by block."""
        return np.zeros((rows, self.size))

    def estimated_state(
        self,
        covariance: ArrayLike | None = None,
        *,
        position_m: ArrayLike | None = None,
        velocity_mps: ArrayLike | None = None,
        quaternion_xyzw: ArrayLike | None = None,
        accel_bias_mps2: ArrayLike | None = None,
        gyro_bias_rad_s: ArrayLike | None = None,
    ) -> EstimatedState:
        """The ``EstimatedState`` reporting the given values.

        With ``covariance`` (this layout's ``size`` × ``size``) the estimate
        carries its marginal over the reported blocks, in EstimatedState
        order; blocks it does not report (yaw, filter-specific ones) are
        marginalized out. Raises ValueError when a reported value has no
        3-element block here: its uncertainty is unknown, so pass
        ``covariance=None`` (no NEES).
        """
        given = (
            position_m,
            velocity_mps,
            quaternion_xyzw,
            accel_bias_mps2,
            gyro_bias_rad_s,
        )
        values = [None if v is None else np.asarray(v, dtype=float) for v in given]
        reported = [b for (b, _), v in zip(REPORTED_BLOCKS, values) if v is not None]
        marginal = None
        if covariance is not None and reported:
            untracked = [b for b in reported if b not in self]
            if untracked:
                raise ValueError(
                    f"No covariance for reported blocks {untracked}: "
                    "pass covariance=None"
                )
            P = np.asarray(covariance, dtype=float)
            if P.shape != (self.size, self.size):
                raise ValueError(f"Covariance must be {self.size}x{self.size}")
            rows = np.concatenate(
                [np.arange(self.size)[self.index(b)] for b in reported]
            )
            marginal = P[np.ix_(rows, rows)]
        position, velocity, quaternion, accel_bias, gyro_bias = values
        return EstimatedState(
            position, velocity, quaternion, accel_bias, gyro_bias, marginal
        )


def _block(item: str | tuple[str, int]) -> tuple[str, int]:
    """``(name, size)`` for a standard name or an explicit pair."""
    if isinstance(item, str):
        if item not in STANDARD_SIZES:
            raise ValueError(
                f"{item!r} is not a standard block {sorted(STANDARD_SIZES)}; "
                "give (name, size)"
            )
        return item, STANDARD_SIZES[item]
    name, size = item
    return str(name), size
