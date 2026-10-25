"""Per-run scoring: feeds truth, estimates, and commands to the metrics.

Owns the truth history (simulator odometry, used here and nowhere near the
algorithms), estimation and control metrics, and the mission's step-response
analysis, logs errors to Rerun under ``evaluation/``, and returns the
``metrics.json`` content.
"""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ...core.measurements import EstimatedState, Sample
from ...estimation.interface import Innovation
from ...evaluation.control import ControlMetrics
from ...evaluation.estimation import BiasSource, EstimationMetrics
from ...evaluation.truth import TruthHistory
from ...control.references import ReferencePoint
from ..common.mission import MissionRunner

# A controller's tracking_reference(position, velocity, quaternion) method.
TrackingReference = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    ReferencePoint | None,
]

AXIS = {"x": 0, "y": 1, "z": 2}


class RunEvaluation:
    """Score one mission (see module docs).

    ``log_vector(name, values, labels)`` records to Rerun (or a fake);
    ``true_biases`` gives the simulator's current IMU biases in body axes.
    ``tracking_reference(position, velocity, quaternion)``, when given (a
    controller's optional method, see control/interface.py), replaces the
    mission reference for tracking errors; ``metrics.json`` records which
    one was used under ``control.tracking_reference``.
    """

    def __init__(
        self,
        settings: Mapping[str, Any],
        weight_n: float,
        mission: MissionRunner,
        log_vector: Callable[..., None],
        true_biases: BiasSource | None = None,
        tracking_reference: TrackingReference | None = None,
    ) -> None:
        algorithms = settings["algorithms"]
        self.truth = TruthHistory()
        self.estimation = EstimationMetrics(
            self.truth,
            rate_hz=algorithms["evaluation_rate_hz"],
            start_s=algorithms["evaluation_start_s"],
            true_biases=true_biases,
        )
        self.control = ControlMetrics(weight_n)
        self._mission = mission
        self._tracking_reference = tracking_reference
        self._settings = settings
        self._log = log_vector
        self._first_ns: int | None = None
        self.estimates = 0

    def on_truth(
        self,
        sim_time_ns: int,
        position_m: NDArray[np.float64],
        velocity_mps: NDArray[np.float64],
        quaternion_xyzw: NDArray[np.float64],
        *,
        controlled: bool,
    ) -> dict[str, float] | None:
        """One odometry sample; ``controlled`` when a rotor controller flies.

        Returns the tracking errors at this sample, or None without a
        reference.
        """
        if self._first_ns is None:
            self._first_ns = sim_time_ns
        self.truth.add(sim_time_ns, position_m, velocity_mps, quaternion_xyzw)
        errors = None
        if controlled:
            elapsed = (sim_time_ns - self._first_ns) * 1e-9
            errors = self.control.add_state(
                elapsed,
                position_m,
                velocity_mps,
                quaternion_xyzw,
                self._reference(elapsed, position_m, velocity_mps, quaternion_xyzw),
            )
            if errors is not None:
                self._log("evaluation/tracking", list(errors.values()), list(errors))
        for scored in self.estimation.process():
            values = {k: v for k, v in scored.items() if k != "sim_time_s"}
            self._log("evaluation/estimation", list(values.values()), list(values))
        return errors

    def _reference(
        self,
        elapsed_s: float,
        position_m: NDArray[np.float64],
        velocity_mps: NDArray[np.float64],
        quaternion_xyzw: NDArray[np.float64],
    ) -> ReferencePoint | None:
        """The controller's own tracking reference if it has one, else the
        mission reference (None before the mission starts)."""
        if self._tracking_reference is not None:
            return self._tracking_reference(position_m, velocity_mps, quaternion_xyzw)
        return self._mission.sample(elapsed_s)

    def on_estimate(self, sample: Sample[EstimatedState]) -> None:
        self.estimates += 1
        self.estimation.add_estimate(sample.sim_time_ns, sample.payload)

    def on_innovations(self, innovations: list[Innovation]) -> None:
        for innovation in innovations:
            self.estimation.add_nis(innovation.name, innovation.nis, innovation.dof)

    def on_command(self, record: Mapping[str, Any]) -> None:
        if self._first_ns is None or record.get("warmup"):
            return
        elapsed = (record["release_sim_time_ns"] - self._first_ns) * 1e-9
        self.control.add_command(
            elapsed, record["accepted_thrust_n"], bool(record["saturated"])
        )

    def summary(self) -> dict[str, Any]:
        return dict(
            estimation=self.estimation.summary() if self.estimates else None,
            control=dict(
                self.control.summary(),
                tracking_reference=(
                    "mission" if self._tracking_reference is None else "controller"
                ),
            ),
            step_response=self._step_response(),
        )

    def _step_response(self) -> dict[str, Any] | None:
        """Step figures for step and doublet missions (first pulse)."""
        mission, runner = self._settings["mission"], self._mission
        if mission["type"] not in ("step", "doublet") or runner.start_s is None:
            return None
        center = np.asarray(mission["center_m"], float)
        start = runner.start_s + mission["step_time_s"]
        if mission["type"] == "step":
            step = np.asarray(mission["step_m"], float)
            axis = int(np.argmax(np.abs(step)))
            final, end = center[axis] + step[axis], None
        else:
            axis = AXIS[mission["axis"]]
            final = center[axis] + mission["amplitude_m"]
            end = start + mission["pulse_s"]
        return dict(
            axis="xyz"[axis],
            **self.control.step(axis, start, center[axis], final, end),
        )
