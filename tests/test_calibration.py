"""Искусственная траектория проверяет подбор, но не точность решателя."""

import numpy as np
import pytest

from euler_lab.calibration import fit_trajectory_gamma
from euler_lab.solver import EulerSolver


def test_trajectory_gamma_recovers_parameter_of_own_solver():
    initial = np.zeros((12, 8, 1, 5))
    initial[..., 0] = 0.3
    initial[..., 4] = 0.25
    initial[:6, ..., 0] = 1.0
    initial[:6, ..., 4] = 1.0
    initial[:, :4, ..., 2] = 0.15
    times = np.array([0.0, 0.04, 0.1])
    spacing = (1 / 12, 1 / 8, 1.0)
    reference = EulerSolver(gamma=1.4).solve(initial, times, spacing)

    result = fit_trajectory_gamma(initial, times, reference, spacing)

    assert result.success
    assert result.gamma == pytest.approx(1.4, abs=0.006)
    assert result.loss < 1e-6
    assert 3 <= result.evaluations <= 20


def test_initial_reference_frame_does_not_enter_trajectory_loss():
    initial = np.zeros((8, 1, 1, 5))
    initial[..., 0] = 0.3
    initial[..., 4] = 0.25
    initial[:4, ..., 0] = 1.0
    initial[:4, ..., 4] = 1.0
    times = np.array([0.0, 0.1])
    spacing = (1 / 8, 1.0, 1.0)
    reference = EulerSolver(gamma=1.4).solve(initial, times, spacing)
    result = fit_trajectory_gamma(initial, times, reference, spacing)
    reference[0, ..., [0, 4]] *= 10

    changed_frame_zero = fit_trajectory_gamma(initial, times, reference, spacing)

    assert changed_frame_zero.gamma == result.gamma
    assert changed_frame_zero.loss == result.loss


def test_trajectory_fit_requires_noninitial_training_frame():
    initial = np.zeros((2, 2, 1, 5))
    initial[..., 0] = initial[..., 4] = 1.0
    with pytest.raises(ValueError, match="хотя бы один"):
        fit_trajectory_gamma(initial, [0.0], initial[None], (0.5, 0.5, 1.0))
