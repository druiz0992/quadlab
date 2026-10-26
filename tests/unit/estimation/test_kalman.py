"""Shared Kalman update: closed form, gating, Joseph form, validation."""

import numpy as np
import pytest

from robotlab.estimation.filter.kalman import checked_covariance, kalman_update
from robotlab.estimation.filter.models import LinearizedMeasurement


def measurement(residual: float, variance: float) -> LinearizedMeasurement:
    """Observe the first of two states."""
    return LinearizedMeasurement(
        "z", np.array([residual]), np.array([[1.0, 0.0]]), np.array([[variance]])
    )


def test_scalar_update_matches_the_closed_form() -> None:
    P = np.diag([4.0, 9.0])
    correction = kalman_update(P, measurement(2.0, 1.0))
    # S = 5, K = (0.8, 0), NIS = 4/5, posterior variance 4·1/5.
    assert correction.innovation.nis == pytest.approx(0.8)
    assert correction.innovation.dof == 1 and correction.innovation.accepted
    assert correction.delta is not None
    np.testing.assert_allclose(correction.delta, [1.6, 0.0])
    np.testing.assert_allclose(correction.covariance, np.diag([0.8, 9.0]))
    np.testing.assert_array_equal(P, np.diag([4.0, 9.0]))  # input untouched


def test_joseph_form_equals_the_standard_form_for_the_optimal_gain() -> None:
    P = np.array([[2.0, 0.5], [0.5, 1.0]])
    z = measurement(0.3, 0.5)
    correction = kalman_update(P, z)
    K = P @ z.jacobian.T / (P[0, 0] + 0.5)
    np.testing.assert_allclose(
        correction.covariance, (np.eye(2) - K @ z.jacobian) @ P, atol=1e-12
    )


def test_gate_rejects_without_changing_the_covariance() -> None:
    P = np.eye(2)
    correction = kalman_update(P, measurement(10.0, 1.0), gate_squared=25.0)
    assert not correction.innovation.accepted
    assert correction.innovation.nis == pytest.approx(50.0)
    assert correction.delta is None and correction.covariance is P


@pytest.mark.parametrize(
    "covariance, z, gate, message",
    [
        (np.eye(3), measurement(1.0, 1.0), 25.0, "columns"),
        (np.eye(2), measurement(1.0, 1.0), 0.0, "Gate"),
        (np.eye(2), measurement(np.nan, 1.0), 25.0, "non-finite"),
    ],
)
def test_invalid_updates_raise(
    covariance: np.ndarray, z: LinearizedMeasurement, gate: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        kalman_update(covariance, z, gate)


def test_inconsistent_measurement_shapes_raise() -> None:
    with pytest.raises(ValueError, match="shapes"):
        LinearizedMeasurement("z", np.zeros(2), np.zeros((1, 3)), np.eye(2))


@pytest.mark.parametrize(
    "value", [[[1.0, 0.1], [0.0, 1.0]], [[1.0, 0.0], [0.0, -1.0]], np.eye(3)]
)
def test_checked_covariance_rejects_invalid_matrices(
    value: list[list[float]] | np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        checked_covariance(value, 2)
