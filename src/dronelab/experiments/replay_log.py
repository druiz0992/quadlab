"""Recording of a run's sensor-bus deliveries and truth for offline replay.

With ``algorithms.record_bus`` set to ``light`` (every stream except camera
and lidar) or ``all``, a Gazebo run writes ``replay.jsonl``:

- line 1, the header: format and version, the resolved settings, the true
  vehicle weight, and the ``AlgorithmContext`` the plug-ins were given;
- then events in the exact order the mission processed them:
  ``{"event": "sample", "kind", "now_ns", "sample"}`` for each sample the
  sensor bus delivered (``now_ns`` = bus time at delivery), and
  ``{"event": "truth", "sim_time_ns", "position_m", "velocity_mps",
  "quaternion_xyzw", "accel_bias_mps2", "gyro_bias_rad_s"}`` for each truth
  sample, with the effective IMU bias the metrics score against at that
  moment (evaluation only, see evaluation/imu_bias.py).

``dronelab-replay`` (experiments/replay.py) feeds the samples to any
estimator and scores it against the truth exactly as the live run would.
Truth lines are never given to an estimator.
"""

import json
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from numpy.typing import ArrayLike

from ..core.algorithms import AlgorithmContext
from ..core.measurements import EstimatedState, Sample
from ..core.serialization import encode

FORMAT = "dronelab-replay"
VERSION = 1
RECORD_MODES = ("off", "light", "all")
HEAVY_STREAMS = frozenset({"camera", "lidar"})


def estimate_record(sample: Sample[EstimatedState]) -> dict[str, Any]:
    """One ``estimates.jsonl`` line (shared by live runs and replay)."""
    state = sample.payload
    return dict(
        sim_time_ns=sample.sim_time_ns,
        sequence=sample.sequence,
        stream_id=sample.stream_id,
        epoch=sample.epoch,
        frame_id=sample.frame_id,
        position_m=state.position_m.tolist(),
        velocity_mps=state.velocity_mps.tolist(),
        quaternion_xyzw=state.quaternion_xyzw.tolist(),
        accel_bias_mps2=state.accel_bias_mps2.tolist(),
        gyro_bias_rad_s=state.gyro_bias_rad_s.tolist(),
        covariance=state.covariance.tolist(),
        parents=sample.parents,
    )


class ReplayRecorder:
    """Write ``replay.jsonl`` (see module docs); ``mode`` is light or all."""

    def __init__(
        self,
        path: Path,
        mode: str,
        settings: dict[str, Any],
        context: AlgorithmContext,
        weight_n: float,
    ) -> None:
        if mode not in RECORD_MODES[1:]:
            raise ValueError(f"record_bus mode must be one of {RECORD_MODES[1:]}")
        self._mode = mode
        self._stream: TextIO = path.open("w")
        self.samples = 0
        self.truth_samples = 0
        self.skipped: dict[str, int] = {}
        self._write(
            dict(
                format=FORMAT,
                version=VERSION,
                settings=settings,
                weight_n=weight_n,
                context=encode(context),
            )
        )

    def _write(self, record: dict[str, Any]) -> None:
        self._stream.write(json.dumps(record) + "\n")

    def on_sample(self, now_ns: int, kind: str, sample: Sample) -> None:
        """One sample as the bus delivered it."""
        if self._mode == "light" and kind in HEAVY_STREAMS:
            self.skipped[kind] = self.skipped.get(kind, 0) + 1
            return
        self.samples += 1
        self._write(
            dict(event="sample", kind=kind, now_ns=now_ns, sample=encode(sample))
        )

    def on_truth(
        self,
        sim_time_ns: int,
        position_m: ArrayLike,
        velocity_mps: ArrayLike,
        quaternion_xyzw: ArrayLike,
        imu_bias: tuple[ArrayLike, ArrayLike],
    ) -> None:
        """One truth sample with the effective IMU bias at that moment."""
        self.truth_samples += 1
        accel, gyro = imu_bias
        self._write(
            dict(
                event="truth",
                sim_time_ns=sim_time_ns,
                position_m=np.asarray(position_m, float).tolist(),
                velocity_mps=np.asarray(velocity_mps, float).tolist(),
                quaternion_xyzw=np.asarray(quaternion_xyzw, float).tolist(),
                accel_bias_mps2=np.asarray(accel, float).tolist(),
                gyro_bias_rad_s=np.asarray(gyro, float).tolist(),
            )
        )

    def close(self) -> None:
        self._stream.close()

    def summary(self) -> dict[str, Any]:
        return dict(
            mode=self._mode,
            samples=self.samples,
            truth_samples=self.truth_samples,
            skipped=self.skipped,
        )
