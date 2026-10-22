"""Проверки интервенции, идентифицируемости и геометрии параметров ЛР №2."""

from pathlib import Path

import numpy as np
import pytest

from navier_lab.data import load_cylinder
from navier_lab.experiment import run_experiment
from navier_lab.geometry import Gaussian, estimate_gaussian, gaussian_kl, spd_distance, spd_geodesic
from navier_lab.model import derivative, least_squares, momentum_from_fields, taylor_green


def test_five_point_derivative_on_polynomial():
    x = np.linspace(-1, 1, 51)
    np.testing.assert_allclose(derivative(x**4, x[1]-x[0], 0)[2:-2], 4*x[2:-2]**3, atol=1e-12)
    np.testing.assert_allclose(derivative(x**4, x[1]-x[0], 0, 2)[2:-2], 12*x[2:-2]**2, atol=1e-10)


def test_skipped_initial_frame_does_not_enter_time_derivatives():
    time = np.arange(12) * 0.1
    x, y = np.linspace(-1, 1, 8), np.linspace(-1, 1, 7)
    tt, yy, xx = np.meshgrid(time, y, x, indexing='ij')
    fields = dict(time=time, x=x, y=y, u=tt**4+xx**2+yy**2,
                  v=2*tt**3+xx*yy, p=tt*xx+yy**2)
    clean = momentum_from_fields(**fields, skip_initial_frames=1)
    damaged = {key: value.copy() for key, value in fields.items()}
    for key in ('u', 'v', 'p'):
        damaged[key][0] = np.nan
    changed = momentum_from_fields(**damaged, skip_initial_frames=1)
    for key in ('time', 'convection', 'laplacian', 'pressure_gradient', 'acceleration'):
        np.testing.assert_array_equal(getattr(clean, key), getattr(changed, key))
    np.testing.assert_array_equal(clean.time, time[3:-2])
    expected = np.stack((4*tt**3, 6*tt**2), axis=-1)[3:-2, 2:-2, 2:-2]
    np.testing.assert_allclose(clean.acceleration, expected, atol=1e-13)


@pytest.mark.parametrize('skip', [-1, 0.5, True, 16])
def test_invalid_initial_frame_skip_is_rejected(skip):
    _, fields = taylor_green(n=8, nt=20)
    observed = {key: fields[key] for key in ('u', 'v', 'p', 'x', 'y', 'time')}
    with pytest.raises(ValueError):
        momentum_from_fields(**observed, skip_initial_frames=skip)


def test_taylor_green_satisfies_momentum_equation():
    d, _ = taylor_green(noise_fraction=0)
    expected = -d.convection + 0.01*d.laplacian - d.pressure_gradient
    np.testing.assert_allclose(expected, d.acceleration, atol=1e-16)
    np.testing.assert_allclose(least_squares(*d.design(slice(None))), [1, 0.01], atol=1e-12)


def test_zero_pressure_changes_only_pressure_branch():
    d, _ = taylor_green()
    saved = d.pressure_gradient.copy()
    changed = d.pressure_intervention()
    assert np.count_nonzero(changed.pressure_gradient) == 0
    np.testing.assert_array_equal(d.pressure_gradient, saved)
    for name in ('acceleration', 'laplacian', 'convection'):
        np.testing.assert_array_equal(getattr(d, name), getattr(changed, name))


def test_rank_deficient_fit_is_reported():
    with pytest.raises(ValueError, match='неидентифицируемы'):
        least_squares(np.ones((20, 2)), np.ones(20))


def test_gaussian_kl_known_case_and_direction():
    p = Gaussian(np.zeros(2), np.eye(2))
    q = Gaussian(np.array([2., 0]), np.eye(2))
    assert gaussian_kl(p, p) == 0
    assert gaussian_kl(p, q) == pytest.approx(2)
    q = Gaussian(np.zeros(2), 2*np.eye(2))
    assert gaussian_kl(p, q) == pytest.approx(np.log(2)-0.5)
    assert gaussian_kl(p, q) != pytest.approx(gaussian_kl(q, p))


def test_degenerate_covariance_is_not_hidden_by_epsilon():
    with pytest.raises(ValueError, match='вырождена'):
        estimate_gaussian(np.ones((12, 2)))


def test_geometry_is_invariant_to_common_change_of_parameter_units():
    p = Gaussian([0.5, 0.01], np.array([[2, 0.02], [0.02, 0.003]]))
    q = Gaussian([1.2, 0.04], np.array([[1.2, -0.01], [-0.01, 0.005]]))
    transform = np.array([[2., 0.5], [0., 100.]])
    shift = np.array([3., -2.])
    pp = Gaussian(transform@p.mean+shift, transform@p.covariance@transform.T)
    qq = Gaussian(transform@q.mean+shift, transform@q.covariance@transform.T)
    assert gaussian_kl(p, q) == pytest.approx(gaussian_kl(pp, qq), rel=1e-10)
    assert spd_distance(p.covariance, q.covariance) == pytest.approx(spd_distance(pp.covariance, qq.covariance))


def test_spd_geodesic_endpoints_and_midpoint():
    a, b = np.diag([1., 4.]), np.diag([9., 16.])
    np.testing.assert_allclose(spd_geodesic(a, b, 0), a)
    np.testing.assert_allclose(spd_geodesic(a, b, 1), b)
    mid = spd_geodesic(a, b, 0.5)
    np.testing.assert_allclose(mid, np.diag([3., 8.]))
    assert spd_distance(a, mid) == pytest.approx(spd_distance(a, b)/2)


def test_error_alone_misses_pressure_compensation():
    d, _ = taylor_green(n=16)
    r = run_experiment(d)
    before, after = r['rows'][0], r['rows'][-1]
    assert after['test_error'] == pytest.approx(before['test_error'], abs=1e-10)
    assert after['kl'] > 100
    assert r['decision'] == 'retain_pressure_branch'


def test_test_frames_do_not_enter_parameter_fits():
    d, _ = taylor_green(n=8)
    before = run_experiment(d)
    d.acceleration[before['test_frames']] += 1
    after = run_experiment(d)
    np.testing.assert_array_equal(before['samples'], after['samples'])
    assert before['rows'][-1]['kl'] == after['rows'][-1]['kl']
    assert before['rows'][0]['test_error'] != after['rows'][0]['test_error']


@pytest.fixture(scope='module')
def cylinder():
    root = Path(__file__).resolve().parents[1]
    return load_cylinder(root/'data'/'processed')[0]


def test_external_reference_has_correct_axes_and_expected_coefficients(cylinder):
    d = momentum_from_fields(**cylinder, skip_initial_frames=1)
    theta = least_squares(*d.design(slice(0, 120)))
    assert theta[0] == pytest.approx(1, abs=0.0001)
    assert theta[1] == pytest.approx(0.01, abs=0.00001)


def test_constant_pressure_shift_does_not_change_fit(cylinder):
    d = momentum_from_fields(**cylinder, skip_initial_frames=1)
    changed = dict(cylinder, p=cylinder['p'].astype(float)+3)
    shifted = momentum_from_fields(**changed, skip_initial_frames=1)
    np.testing.assert_allclose(least_squares(*d.design(slice(0,120))),
                               least_squares(*shifted.design(slice(0,120))), atol=1e-12)


def test_borrowed_graph_is_acyclic_and_has_pressure_paths():
    import networkx as nx
    from navier_lab.ivan_graph import build_navier_stokes_2d
    built = build_navier_stokes_2d()
    graph = nx.DiGraph()
    for node in built['tree'].nodes:
        graph.add_node(node.id)
        graph.add_edges_from((p.id, node.id) for p in node.parents)
    assert nx.is_directed_acyclic_graph(graph)
    assert nx.has_path(graph, built['p'].id, built['rhs_u'].id)
    assert nx.has_path(graph, built['p'].id, built['rhs_v'].id)
