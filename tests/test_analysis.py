"""Проверки формул и преобразований; искусственные данные не заменяют эталон."""

import networkx as nx
import numpy as np
import pytest

from euler_lab.analysis import (
    coarsen_conservative,
    equation_graph,
    error_metrics,
    extrude_planar,
    fit_gamma,
)


@pytest.mark.parametrize("dimensions,gamma", [(2, 1.4), (3, 1.67)])
def test_gamma_recovers_planted_value(dimensions, gamma):
    rng = np.random.default_rng(41)
    rho = rng.uniform(0.2, 2.0, (6, 7))
    velocity = rng.normal(0, 1.2, (*rho.shape, dimensions))
    pressure = rng.uniform(0.1, 3.0, rho.shape)
    momentum = rho[..., None] * velocity
    energy = pressure / (gamma - 1) + 0.5 * rho * np.sum(velocity**2, axis=-1)

    result = fit_gamma(rho, momentum, energy, pressure)

    assert result.success
    assert result.n_samples == 42
    assert result.gamma == pytest.approx(gamma, abs=1e-10)
    assert result.rmse < 1e-10


def test_gamma_with_noise_matches_linear_least_squares():
    rng = np.random.default_rng(42)
    rho = rng.uniform(0.5, 2, 30)
    momentum = rng.normal(size=(30, 3))
    internal_energy = rng.uniform(1, 8, 30)
    energy = internal_energy + np.sum(momentum**2, axis=-1) / (2 * rho)
    pressure = 0.4 * internal_energy + rng.normal(0, 0.03, 30)
    analytical_gamma = 1 + np.dot(internal_energy, pressure) / np.dot(internal_energy, internal_energy)

    result = fit_gamma(rho, momentum, energy, pressure)

    assert result.gamma == pytest.approx(analytical_gamma, abs=1e-10)
    residual = (result.gamma - 1) * internal_energy - pressure
    assert result.rmse == pytest.approx(np.sqrt(np.mean(residual**2)))


def test_gamma_respects_bounds_when_initial_guess_is_outside():
    result = fit_gamma([1, 2], [[0, 0], [0, 0]], [2, 4], [1.2, 2.4], bounds=(1.5, 1.8))
    assert result.success
    assert result.gamma == pytest.approx(1.6, abs=1e-9)


@pytest.mark.parametrize("bounds", [(1.0, 1.9), (1.4, 1.2), (1.4, 1.4), (1.05, np.inf), (1.1,)])
def test_gamma_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        fit_gamma([1], [[0, 0]], [2], [1], bounds=bounds)


@pytest.mark.parametrize(
    "rho,momentum,energy,pressure",
    [
        ([0], [[0, 0]], [2], [1]),
        ([1], [[0, 0]], [2], [0]),
        ([1], [[3, 4]], [2], [1]),
        ([1], [[0, 0]], [np.nan], [1]),
        ([1], [[0]], [2], [1]),
        ([1], [[0, 0]], [2, 3], [1]),
        ([1, 1], [[0, 0]], [2, 2], [1, 1]),
    ],
)
def test_gamma_rejects_nonphysical_or_mismatched_fields(rho, momentum, energy, pressure):
    with pytest.raises(ValueError):
        fit_gamma(rho, momentum, energy, pressure)


def test_equation_graph_has_physical_dependencies_and_shared_nodes():
    graph = equation_graph()

    assert nx.is_directed_acyclic_graph(graph)
    for source in ["rho", "mx", "my", "mz", "E", "gamma"]:
        assert nx.has_path(graph, source, "dU_dt")
    assert set(graph.predecessors("pressure")) == {"gamma", "E", "kinetic"}
    assert set(graph.successors("pressure")) == {"Fx", "Fy", "Fz"}
    assert set(graph.predecessors("kinetic")) == {"rho", "u", "v", "w"}
    for axis, velocity in [("x", "u"), ("y", "v"), ("z", "w")]:
        assert graph.has_edge(velocity, f"F{axis}")
        assert graph.has_edge(f"F{axis}", f"dF{axis}_d{axis}")
        assert graph.has_edge(f"dF{axis}_d{axis}", "dU_dt")
    assert [node for node, data in graph.nodes(data=True) if data["kind"] == "output"] == ["dU_dt"]
    assert graph.nodes["gamma"]["kind"] == "parameter"
    for node, data in graph.nodes(data=True):
        assert data["label"].startswith("$") and data["label"].endswith("$")
        assert isinstance(data["layer"], int)
    for source, target in graph.edges:
        assert graph.nodes[source]["layer"] < graph.nodes[target]["layer"]


def test_error_metrics_match_hand_calculation():
    metrics = error_metrics([[2, 4], [3, 6]], [[1, 2], [3, 4]])
    assert metrics["mae"] == pytest.approx(1.25)
    assert metrics["rmse"] == pytest.approx(1.5)
    assert metrics["relative_l2"] == pytest.approx(3 / np.sqrt(30))


def test_error_metrics_report_zero_reference_explicitly():
    assert error_metrics([0, 0], [0, 0]) == {"mae": 0.0, "rmse": 0.0, "relative_l2": 0.0}
    assert np.isinf(error_metrics([1, 0], [0, 0])["relative_l2"])


def test_small_reference_is_not_misclassified_as_zero():
    metrics = error_metrics([2e-200, 0], [1e-200, 0])
    assert metrics["relative_l2"] == pytest.approx(1.0)


@pytest.mark.parametrize("prediction,reference", [([1, 2], [[1, 2]]), ([np.inf], [1]), ([1], [np.nan]), ([], [])])
def test_error_metrics_reject_invalid_inputs(prediction, reference):
    with pytest.raises(ValueError):
        error_metrics(prediction, reference)


@pytest.mark.parametrize("leading_shape", [(), (3,)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_coarsening_preserves_component_averages_and_dtype(leading_shape, dtype):
    rng = np.random.default_rng(9)
    fine = rng.normal(size=(*leading_shape, 8, 12, 5)).astype(dtype)
    coarse = coarsen_conservative(fine, 4)

    assert coarse.shape == (*leading_shape, 2, 3, 5)
    assert coarse.dtype == fine.dtype
    np.testing.assert_allclose(coarse.mean(axis=(-3, -2)), fine.mean(axis=(-3, -2)), atol=5e-8)
    for ix in range(2):
        for iy in range(3):
            block_average = fine[..., 4 * ix : 4 * (ix + 1), 4 * iy : 4 * (iy + 1), :].mean(axis=(-3, -2))
            np.testing.assert_allclose(coarse[..., ix, iy, :], block_average, atol=5e-8)


def test_integer_coarsening_keeps_fractional_averages():
    fine = np.arange(20).reshape(2, 2, 5)
    coarse = coarsen_conservative(fine, 2)
    np.testing.assert_array_equal(coarse[0, 0], [7.5, 8.5, 9.5, 10.5, 11.5])
    assert coarse.dtype == np.float64


def test_unit_coarsening_returns_independent_copy():
    fine = np.ones((2, 3, 5), dtype=np.float32)
    coarse = coarsen_conservative(fine, 1)
    coarse[0, 0, 0] = 9
    assert fine[0, 0, 0] == 1


@pytest.mark.parametrize("factor", [0, -1, 1.5, True, 3])
def test_coarsening_rejects_invalid_factor_or_nondivisible_shape(factor):
    with pytest.raises(ValueError):
        coarsen_conservative(np.ones((4, 8, 5)), factor)


def test_extrusion_repeats_each_layer_and_preserves_dtype():
    W = np.zeros((2, 4, 6, 5), dtype=np.float32)
    W[..., 0] = 1
    W[..., 1] = np.arange(6)
    W[..., 2] = -0.5
    W[..., 4] = 2

    volume = extrude_planar(W, 3)

    assert volume.shape == (2, 4, 6, 3, 5)
    assert volume.dtype == W.dtype
    for iz in range(3):
        np.testing.assert_array_equal(volume[..., iz, :], W)
    volume[0, 0, 0, 0, 0] = 5
    assert W[0, 0, 0, 0] == 1
    assert volume[0, 0, 0, 1, 0] == 1


@pytest.mark.parametrize("nz", [0, -1, 2.5, True])
def test_extrusion_rejects_invalid_layer_count(nz):
    W = np.zeros((2, 2, 5))
    W[..., 0] = W[..., 4] = 1
    with pytest.raises(ValueError):
        extrude_planar(W, nz)


def test_extrusion_rejects_nonplanar_velocity():
    W = np.ones((2, 2, 5))
    with pytest.raises(ValueError, match="w=0"):
        extrude_planar(W, 2)
