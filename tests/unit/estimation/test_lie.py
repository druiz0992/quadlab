"""SO(3) helpers: cross-product matrix and right Jacobians."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotlab.estimation.lie import right_jacobian, right_jacobian_inverse, skew

ANGLES = [np.array([0.3, -0.7, 1.1]), np.array([1e-8, 2e-8, -1e-8]), np.zeros(3)]


def test_skew_is_the_cross_product() -> None:
    a, b = np.array([1.0, -2.0, 0.5]), np.array([0.3, 0.4, -1.2])
    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b))
    np.testing.assert_allclose(skew(a).T, -skew(a))


@pytest.mark.parametrize("theta", ANGLES)
def test_right_jacobian_maps_small_increments(theta: np.ndarray) -> None:
    delta = np.array([1e-6, -2e-6, 1.5e-6])
    exact = Rotation.from_rotvec(theta + delta)
    approx = Rotation.from_rotvec(theta) * Rotation.from_rotvec(
        right_jacobian(theta) @ delta
    )
    assert (exact.inv() * approx).magnitude() < 1e-11


@pytest.mark.parametrize("theta", ANGLES)
def test_right_jacobian_inverse_inverts(theta: np.ndarray) -> None:
    np.testing.assert_allclose(
        right_jacobian_inverse(theta) @ right_jacobian(theta), np.eye(3), atol=1e-12
    )


def test_right_jacobian_inverse_needs_angle_below_pi() -> None:
    with pytest.raises(ValueError, match="below pi"):
        right_jacobian_inverse([np.pi, 0.0, 0.0])
