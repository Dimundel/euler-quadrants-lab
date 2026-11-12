"""Checks for the new graph optimizer and training-only intervention criterion."""
import numpy as np
import pytest
from scipy.optimize import check_grad

from graph_lab.selection import (DiagnosticEngine, _data_moments, _profile_from_gram, exact_oracle, fit_weights, nmse,
                                 optimize_graphs, profile_objective, score_candidate)
from graph_lab.vendor.ivan_interventions import RegressionData, PARAM_PARENT, PARAMS


def dataset(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 9))
    X[:, 2] = 0  # a_z is completely unobserved in a planar trajectory.
    y = X @ np.array([1., .8, 0., .4, .3, 0., .2, .1, 0.]) + rng.normal(0, .05, n)
    return RegressionData(X, y, np.arange(n) // 10, np.zeros(n, dtype=int), np.ones(5))


def test_profile_gradient_matches_finite_differences():
    d = dataset()
    gates = np.linspace(.1, .9, 9)
    keep = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0])
    def fun(z):
        return profile_objective(z, d.X, d.y, keep, .03, .1, .01)[0]
    def jac(z):
        return profile_objective(z, d.X, d.y, keep, .03, .1, .01)[1]
    assert check_grad(fun, jac, gates) < 1e-6


def test_inactive_coefficients_are_exactly_zero():
    d = dataset()
    mask = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=bool)
    w = fit_weights(d.X, d.y, mask)
    np.testing.assert_array_equal(w[~mask], 0.)
    assert nmse(d.y, d.X @ (mask*w)) < 1
    with pytest.raises(ValueError, match='ridge'):
        fit_weights(d.X, d.y, mask, ridge=0)


def test_null_column_retains_prior_and_is_not_identifiable():
    d = dataset()
    perturbed = {parent: d for parent in set(PARAM_PARENT.values())}
    engine = DiagnosticEngine(d, perturbed, n_boot=20, n_splits=4, seed=42)
    result = engine.evaluate(np.ones(9, dtype=bool))
    assert result['info'][2] == pytest.approx(0., abs=1e-8)
    assert not result['keep'][2]
    assert result['shift'][2] == pytest.approx(0.)


class FixedEngine:
    def evaluate(self, mask):
        return dict(info=[10.]*9, shift=[10.]*9, null_q=[1.]*9,
                    keep=[True, True, False, True, True, False, True, True, False])


def test_archive_sampling_reproducible_and_scores_comparable():
    train, validation = dataset(1), dataset(2)
    kwargs = dict(seed=17, rounds=2, steps=8, samples_per_round=5)
    first = optimize_graphs(train, validation, FixedEngine(), **kwargs)
    second = optimize_graphs(train, validation, FixedEngine(), **kwargs)
    assert first == second
    reference = first['reference_diagnostics']['keep']
    for item in first['archive']:
        expected = score_candidate(train, validation, item['mask'], reference)
        assert item['score'] == pytest.approx(expected['score'])
    generated = {tuple(item['mask']) for item in first['archive']}
    assert tuple(first['selected']['mask']) in generated
    assert first['selected']['score'] == min(item['score'] for item in first['archive'])
    oracle = exact_oracle(train, validation, reference)
    assert oracle['candidate_count'] == 512
    assert oracle['score'] <= first['selected']['score'] + 1e-12


def test_gram_profile_matches_row_objective():
    d = dataset(8)
    gates = np.linspace(.07, .93, 9)
    keep = np.array([True, False, False, True, True, False, True, True, False])
    direct_loss, direct_gradient = profile_objective(gates, d.X, d.y, keep, .03, .12, .002)
    fast_loss, fast_gradient = _profile_from_gram(gates, *_data_moments(d), keep.astype(float), .03, .12, .002)
    assert fast_loss == pytest.approx(direct_loss, abs=1e-12)
    np.testing.assert_allclose(fast_gradient, direct_gradient, atol=1e-12)


@pytest.mark.parametrize('zero_part', ['train', 'validation'])
def test_zero_target_rejected_before_graph_scoring(zero_part):
    train, validation = dataset(1), dataset(2)
    selected = train if zero_part == 'train' else validation
    selected.y[:] = 0
    with pytest.raises(ValueError, match='nonzero target'):
        optimize_graphs(train, validation, FixedEngine(), rounds=1, steps=2)
    with pytest.raises(ValueError, match='nonzero target'):
        exact_oracle(train, validation, [True] * 9)


def test_validation_changes_scores_without_refitting_coefficients():
    train, validation = dataset(11), dataset(12)
    mask = np.array([True, True, False, True, True, False, True, True, False])
    before = score_candidate(train, validation, mask, mask)
    validation.y = validation.y + 2.0
    after = score_candidate(train, validation, mask, mask)
    assert before['weights'] == after['weights']
    assert before['coefficients'] == after['coefficients']
    assert before['validation_nmse'] != pytest.approx(after['validation_nmse'])


def test_contextual_intervention_feedback_changes_the_next_search_round():
    class ContextEngine(FixedEngine):
        def __init__(self, change_context):
            self.change_context = change_context
            self.calls = []

        def evaluate(self, mask):
            self.calls.append(np.asarray(mask, dtype=bool).tolist())
            diagnostic = super().evaluate(mask)
            if self.change_context and not np.all(mask):
                diagnostic['keep'] = [not value for value in diagnostic['keep']]
            return diagnostic

    train, validation = dataset(1), dataset(2)
    fixed, contextual = ContextEngine(False), ContextEngine(True)
    options = dict(seed=17, rounds=2, steps=40, samples_per_round=5)
    baseline = optimize_graphs(train, validation, fixed, **options)
    changed = optimize_graphs(train, validation, contextual, **options)
    assert baseline['rounds'][0] == changed['rounds'][0]
    first_mask = changed['rounds'][0]['selected_mask']
    assert not all(first_mask)
    assert contextual.calls[-1] == first_mask
    assert changed['rounds'][1]['mask_before'] == first_mask
    assert not np.allclose(baseline['rounds'][1]['final_gates'],
                           changed['rounds'][1]['final_gates'])
    # Adaptive proposals still use the original common criterion when ranked.
    assert baseline['reference_diagnostics'] == changed['reference_diagnostics']
    for row in changed['archive']:
        expected = score_candidate(train, validation, row['mask'],
                                   changed['reference_diagnostics']['keep'])
        assert row['score'] == pytest.approx(expected['score'])


def test_absent_branch_diagnostics_are_reconsidered_and_order_independent():
    data = dataset(5)
    perturbed = {parent: data for parent in set(PARAM_PARENT.values())}
    options = dict(n_boot=20, n_splits=4, seed=21)
    engine = DiagnosticEngine(data, perturbed, **options)
    empty = np.zeros(9, dtype=bool)
    before = engine.evaluate(empty)
    # Every absent branch is fitted in a one-branch context rather than ignored.
    assert len(engine._cache) == 9
    assert all(sum(key) == 1 for key in engine._cache)
    assert before['info'][0] > 0
    engine.evaluate(np.ones(9, dtype=bool))
    assert engine.evaluate(empty) == before
    other = DiagnosticEngine(data, perturbed, **options)
    other.evaluate(np.ones(9, dtype=bool))
    assert other.evaluate(empty) == before


def test_well_unseen_test_endpoint_cannot_change_training_or_validation(tmp_path):
    from pathlib import Path
    from graph_lab.experiments import prepare_dataset

    root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / 'data' / 'processed'
    data_dir.mkdir(parents=True)
    path = data_dir / 'well_sample.npz'
    with np.load(root / 'data' / 'processed' / 'well_sample.npz') as stored:
        fields = {key: stored[key] for key in stored.files}
    np.savez_compressed(path, **fields)
    before = prepare_dataset(tmp_path, 'well')
    # Frame 6 enters the test derivative at center 5 only. Frame 4 is shared
    # with validation's stencil, so this test intentionally does not change it.
    fields['conserved'][-1] *= 1.03
    np.savez_compressed(path, **fields)
    after = prepare_dataset(tmp_path, 'well')
    for name in ('train', 'validation'):
        old, new = getattr(before, name), getattr(after, name)
        np.testing.assert_array_equal(old.X, new.X)
        np.testing.assert_array_equal(old.y, new.y)
        np.testing.assert_array_equal(old.scales, new.scales)
    for parent in before.perturbed:
        np.testing.assert_array_equal(before.perturbed[parent].X,
                                      after.perturbed[parent].X)
    assert not np.array_equal(before.test.y, after.test.y)
