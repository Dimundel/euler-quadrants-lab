"""Проверки фиксированной Huber-цели и совместимости исходного L2."""

import numpy as np
import pytest
from scipy.optimize import least_squares as scipy_least_squares
from scipy.special import huber

from navier_lab.model import fit_blocks, fit_parameters, least_squares, residual_scale, taylor_green


def regression(seed=17, n=200):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)) * [1.0, 25.0]
    truth = np.array([1.0, 0.01])
    y = x @ truth + rng.normal(0, 0.05, n)
    return x, y, truth


def test_l2_api_and_large_huber_cutoff_preserve_original_fit_exactly():
    x, y, _ = regression()
    expected = least_squares(x, y)
    np.testing.assert_array_equal(fit_parameters(x, y), expected)
    np.testing.assert_array_equal(fit_parameters(x, y, loss='l2'), expected)
    actual = fit_parameters(x, y, loss='huber', huber_scale=1.0, huber_delta=1e6)
    np.testing.assert_array_equal(actual, expected)


def test_huber_recovers_coefficients_with_response_outliers():
    x, y, truth = regression()
    y[:20] += 20 * x[:20, 0]
    robust = fit_parameters(x, y, loss='huber', huber_scale=0.05)
    ordinary = least_squares(x, y)
    assert np.linalg.norm(robust - truth) < 0.03
    assert np.linalg.norm(robust - truth) < 0.05 * np.linalg.norm(ordinary - truth)


def test_huber_objective_and_stationarity_match_scipy_reference():
    x, y, _ = regression(seed=29, n=45)
    y[[1, 7, 18, 32]] += [0.8, -1.2, 0.5, -0.9]
    sigma, delta = 0.07, 1.345
    actual = fit_parameters(x, y, loss='huber', huber_scale=sigma, huber_delta=delta)
    reference = scipy_least_squares(
        lambda theta: (x @ theta - y) / sigma, least_squares(x, y),
        jac=lambda theta: x / sigma, loss='huber', f_scale=delta,
        xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=1000,
    )
    assert reference.success
    np.testing.assert_allclose(actual, reference.x, atol=2e-8, rtol=1e-8)
    objective = np.sum(huber(delta, (x @ actual - y) / sigma))
    assert objective == pytest.approx(reference.cost, rel=1e-10, abs=1e-10)
    gradient = x.T @ np.clip((x @ actual - y) / sigma, -delta, delta) / sigma
    assert np.linalg.norm(gradient, ord=np.inf) < 1e-4


def test_huber_is_invariant_to_column_units():
    x, y, _ = regression()
    y[::13] += 3.0
    expected = fit_parameters(x, y, loss='huber', huber_scale=0.05)
    units = np.array([1e-5, 1e5])
    changed = fit_parameters(x * units, y, loss='huber', huber_scale=0.05)
    np.testing.assert_allclose(changed * units, expected, atol=1e-10)


@pytest.mark.parametrize('scale', [None, -0.1, 0.0, np.nan, np.inf])
def test_huber_requires_explicit_positive_finite_scale(scale):
    x, y, _ = regression()
    with pytest.raises(ValueError, match='huber_scale'):
        fit_parameters(x, y, loss='huber', huber_scale=scale)


@pytest.mark.parametrize('delta', [-1, 0, np.nan, np.inf])
def test_huber_requires_positive_finite_delta(delta):
    x, y, _ = regression()
    with pytest.raises(ValueError, match='huber_delta'):
        fit_parameters(x, y, loss='huber', huber_scale=0.05, huber_delta=delta)


def test_residual_scale_uses_original_l2_mad():
    x, y, _ = regression()
    residuals = y - x @ least_squares(x, y)
    expected = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
    assert residual_scale(x, y) == pytest.approx(expected)


@pytest.mark.parametrize('theta', [[0, 0], [1, 0.01]])
def test_exact_fit_has_no_data_driven_noise_scale(theta):
    x, _, _ = regression()
    with pytest.raises(ValueError, match='Нулевой MAD'):
        residual_scale(x, x @ theta)


def test_huber_rejects_unidentifiable_design():
    with pytest.raises(ValueError, match='неидентифицируемы'):
        fit_parameters(np.ones((20, 2)), np.arange(20), loss='huber', huber_scale=1.0)


def test_block_fits_use_supplied_window_scales_and_keep_l2_compatible():
    data, _ = taylor_green(n=8, nt=30)
    blocks = [np.arange(10), np.arange(10, 20)]
    scales = [residual_scale(*data.design(block)) for block in blocks]
    expected_l2 = np.array([least_squares(*data.design(block)) for block in blocks])
    np.testing.assert_array_equal(fit_blocks(data, blocks), expected_l2)
    expected = np.array([fit_parameters(*data.design(block), loss='huber', huber_scale=scale,
                                        huber_delta=2.0) for block, scale in zip(blocks, scales)])
    np.testing.assert_array_equal(fit_blocks(data, blocks, loss='huber', huber_scales=scales,
                                           huber_delta=2.0), expected)
    with pytest.raises(ValueError, match='huber_scales'):
        fit_blocks(data, blocks, loss='huber', huber_scales=scales[:1])
