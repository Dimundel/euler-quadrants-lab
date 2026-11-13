"""Continuous selection of a fixed library of Euler graph branches.

The dictionary and parent perturbations come from Ivan Papay's lab 2. Gates
select groups of edges, not arbitrary edges between observed state variables.
All diagnostics use training data only. Validation ranks the generated masks.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import minimize

from .vendor.ivan_interventions import PARAM_PARENT, PARAMS, PRIOR_PRECISION, RegressionData


def nmse(y, prediction):
    """Squared relative L2 error, with a defined value for an all-zero target."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    denominator = max(float(y @ y), np.finfo(float).tiny)
    return float(np.sum((y - prediction) ** 2) / denominator)


def fit_weights(X, y, gates, ridge=1e-5):
    """Fit raw weights w for X @ (gates*w), minimizing NMSE+ridge*||w||²."""
    X, y, gates = np.asarray(X), np.asarray(y), np.asarray(gates, dtype=float)
    if ridge <= 0:
        raise ValueError("ridge must be positive to control gate/weight scaling")
    scale = max(float(y @ y), np.finfo(float).tiny)
    gram = (X.T @ X) / scale
    rhs = (X.T @ y) / scale
    return np.linalg.solve(gates[:, None] * gram * gates[None, :] + ridge * np.eye(X.shape[1]), gates * rhs)


def _fit_from_gram(gram, rhs, gates, ridge):
    return np.linalg.solve(gates[:, None] * gram * gates[None, :]
                           + ridge * np.eye(len(gates)), gates * rhs)


def _profile_from_gram(gates, gram, rhs, keep, lambda_complexity,
                       lambda_intervention, ridge):
    weights = _fit_from_gram(gram, rhs, gates, ridge)
    coefficients = gates * weights
    residual_correlation = gram @ coefficients - rhs
    error = max(float(1 - 2 * coefficients @ rhs + coefficients @ gram @ coefficients), 0.)
    disagreement = keep * (1 - gates) + (1 - keep) * gates
    loss = float(error + ridge * (weights @ weights) + lambda_complexity * gates.mean()
                 + lambda_intervention * disagreement.mean())
    gradient = 2 * weights * residual_correlation + lambda_complexity / len(gates)
    gradient += lambda_intervention * (1 - 2 * keep) / len(gates)
    return loss, gradient


def profile_objective(gates, X, y, keep, lambda_complexity=.02,
                      lambda_intervention=.1, ridge=1e-5):
    """Profiled smooth objective and exact envelope gradient on the box [0,1]^p."""
    X, y, gates = np.asarray(X), np.asarray(y), np.asarray(gates, dtype=float)
    keep = np.asarray(keep, dtype=float)
    weights = fit_weights(X, y, gates, ridge)
    residual = X @ (gates * weights) - y
    scale = max(float(y @ y), np.finfo(float).tiny)
    p = len(gates)
    disagreement = keep * (1 - gates) + (1 - keep) * gates
    loss = float(residual @ residual / scale + ridge * (weights @ weights)
                 + lambda_complexity * gates.mean() + lambda_intervention * disagreement.mean())
    gradient = (2 * weights * (X.T @ residual) / scale + lambda_complexity / p
                + lambda_intervention * (1 - 2 * keep) / p)
    return loss, gradient


def subset_data(data, mask):
    """Restrict columns while retaining response, row blocks and equation labels."""
    return RegressionData(data.X[:, np.asarray(mask, dtype=bool)], data.y,
                          data.block, data.equation, data.scales, dict(data.meta))


@dataclass
class _BlockStats:
    xx: np.ndarray
    xy: np.ndarray
    yy: np.ndarray
    n: np.ndarray

    @classmethod
    def from_data(cls, data):
        _, inverse = np.unique(data.block, return_inverse=True)
        nblocks = int(inverse.max()) + 1
        p = data.X.shape[1]
        xx, xy = np.zeros((nblocks, p, p)), np.zeros((nblocks, p))
        yy, n = np.zeros(nblocks), np.zeros(nblocks)
        np.add.at(xy, inverse, data.X * data.y[:, None])
        np.add.at(yy, inverse, data.y ** 2)
        np.add.at(n, inverse, 1)
        for j in range(p):
            np.add.at(xx[:, :, j], inverse, data.X * data.X[:, j, None])
        return cls(xx, xy, yy, n)

    def total(self, support, blocks=None):
        if blocks is None:
            blocks = np.arange(len(self.n))
        xx = self.xx[blocks].sum(axis=0)[np.ix_(support, support)]
        return xx, self.xy[blocks].sum(axis=0)[support], float(self.yy[blocks].sum()), float(self.n[blocks].sum())


def _laplace(stats, support, blocks=None):
    xx, xy, yy, n = stats.total(support, blocks)
    p = len(support)
    ols = np.linalg.lstsq(xx + 1e-12 * np.eye(p), xy, rcond=None)[0]
    residual_ss = max(float(yy - 2 * ols @ xy + ols @ xx @ ols), 0.)
    noise = max(residual_ss / max(n - p, 1), 1e-12)
    covariance = np.linalg.inv(xx / noise + PRIOR_PRECISION * np.eye(p))
    mean = covariance @ xy / noise
    return mean, covariance, noise


def _bootstrap(stats, support, n_boot, rng, blocks=None):
    if blocks is None:
        blocks = np.arange(len(stats.n))
    _, _, noise = _laplace(stats, support, blocks)
    xx = stats.xx[blocks][:, support][:, :, support]
    xy = stats.xy[blocks][:, support]
    nb, p = len(blocks), len(support)
    counts = rng.multinomial(nb, np.full(nb, 1 / nb), size=n_boot)
    matrices = np.einsum('bi,ijk->bjk', counts, xx) + PRIOR_PRECISION * noise * np.eye(p)
    rhs = counts @ xy
    samples = np.linalg.solve(matrices, rhs[..., None])[..., 0]
    # This describes the sampling distribution of MAP estimates, not a posterior.
    return samples.mean(axis=0), samples.var(axis=0, ddof=1) + 1e-10


def _jeffreys(mean0, var0, mean1, var1):
    return .25 * (var0 / var1 + var1 / var0 - 2
                  + (mean0 - mean1) ** 2 * (1 / var0 + 1 / var1))


class DiagnosticEngine:
    """Training-only Ivan diagnostics, cached for each contextual graph support.

    To reconsider an absent branch j, evaluate the current graph augmented by j.
    Information gain is based on the Laplace posterior. Bootstrap variances are
    used only for the perturbation and split-half distances.
    """
    def __init__(self, base, perturbed, n_boot=100, n_splits=16, seed=0):
        if n_boot < 2 or n_splits < 2:
            raise ValueError("at least two bootstrap draws and null splits are required")
        self.base = _BlockStats.from_data(base)
        self.perturbed = {name: _BlockStats.from_data(data) for name, data in perturbed.items()}
        self.p = base.X.shape[1]
        if self.p != len(PARAMS):
            raise ValueError("Ivan diagnostics require the nine-column Euler library")
        self.n_boot, self.n_splits, self.seed = n_boot, n_splits, seed
        self._cache = {}

    def _context(self, mask):
        key = tuple(bool(v) for v in mask)
        if key in self._cache:
            return self._cache[key]
        support = np.flatnonzero(mask)
        code = sum(int(v) << i for i, v in enumerate(mask))
        rng = np.random.default_rng(self.seed + 1009 * code)
        mean, covariance, _ = _laplace(self.base, support)
        variance = np.diag(covariance)
        prior_variance = 1 / PRIOR_PRECISION
        info = .5 * ((variance + mean ** 2) / prior_variance - 1 + np.log(prior_variance / variance))
        bm, bv = _bootstrap(self.base, support, self.n_boot, rng)
        shift = np.zeros(len(support))
        for parent in sorted({PARAM_PARENT[PARAMS[j]] for j in support}):
            pm, pv = _bootstrap(self.perturbed[parent], support, self.n_boot, rng)
            distances = _jeffreys(bm, bv, pm, pv)
            for k, j in enumerate(support):
                if PARAM_PARENT[PARAMS[j]] == parent:
                    shift[k] = distances[k]
        nb = len(self.base.n)
        if nb < 4:
            raise ValueError("at least four spatial-temporal blocks are required")
        null = []
        for _ in range(self.n_splits):
            indices = rng.permutation(nb)
            half = nb // 2
            m0, v0 = _bootstrap(self.base, support, self.n_boot, rng, indices[:half])
            m1, v1 = _bootstrap(self.base, support, self.n_boot, rng, indices[half:])
            null.append(_jeffreys(m0, v0, m1, v1))
        threshold = np.quantile(np.asarray(null), .95, axis=0)
        result = dict(support=support, info=info, shift=shift, null_q=threshold,
                      keep=(info >= 1) & (shift > threshold))
        self._cache[key] = result
        return result

    def evaluate(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.p,):
            raise ValueError("mask has the wrong shape")
        arrays = {key: np.zeros(self.p, dtype=(bool if key == 'keep' else float))
                  for key in ('info', 'shift', 'null_q', 'keep')}
        for j in range(self.p):
            context = mask.copy()
            context[j] = True
            diagnostic = self._context(context)
            k = int(np.flatnonzero(diagnostic['support'] == j)[0])
            for key in arrays:
                arrays[key][j] = diagnostic[key][k]
        return {key: value.tolist() for key, value in arrays.items()}


def diagnostics(base, perturbed, mask, n_boot=100, n_splits=16, seed=0):
    return DiagnosticEngine(base, perturbed, n_boot, n_splits, seed).evaluate(mask)


def _data_moments(data):
    scale = float(data.y @ data.y)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("relative graph scores require a finite nonzero target")
    return data.X.T @ data.X / scale, data.X.T @ data.y / scale


def _candidate_from_moments(train_moments, validation_moments, mask, reference_keep,
                             lambda_complexity, lambda_intervention, ridge):
    mask = np.asarray(mask, dtype=bool)
    gram, rhs = train_moments
    weights = _fit_from_gram(gram, rhs, mask.astype(float), ridge)
    coefficients = mask * weights
    val_gram, val_rhs = validation_moments
    error = max(float(1 - 2 * coefficients @ val_rhs + coefficients @ val_gram @ coefficients), 0.)
    complexity = float(mask.mean())
    penalty = float(np.mean(mask != np.asarray(reference_keep, dtype=bool)))
    return dict(mask=mask.tolist(), weights=weights.tolist(), coefficients=coefficients.tolist(),
                validation_nmse=error, complexity=complexity, intervention_penalty=penalty,
                score=float(error + lambda_complexity * complexity + lambda_intervention * penalty))


def score_candidate(train, validation, mask, reference_keep,
                    lambda_complexity=.02, lambda_intervention=.1, ridge=1e-5):
    return _candidate_from_moments(_data_moments(train), _data_moments(validation),
                                  mask, reference_keep, lambda_complexity,
                                  lambda_intervention, ridge)


def optimize_graphs(train, validation, engine, lambda_complexity=.02,
                    lambda_intervention=.1, ridge=1e-5, seed=0, rounds=4,
                    steps=40, samples_per_round=24, initial_gates=None):
    """Alternate intervention diagnostics and continuous gate optimization.

    Each round's intervention result guides the next continuous update. All
    discrete candidates are refitted and ranked against ONE fixed reference
    diagnostic from the full training model, so their scores are comparable.
    The exhaustive oracle is deliberately separate from this algorithm.
    """
    rng = np.random.default_rng(seed)
    p = train.X.shape[1]
    reference = engine.evaluate(np.ones(p, dtype=bool))
    gates = (rng.uniform(.35, .85, p) if initial_gates is None
             else np.asarray(initial_gates, dtype=float).copy())
    gates = np.clip(gates, 0, 1)
    archive, trajectory, round_log = [], [], []
    train_moments, validation_moments = _data_moments(train), _data_moments(validation)
    candidate_cache = {}

    def record(mask, round_number, source):
        key = tuple(bool(v) for v in mask)
        if key not in candidate_cache:
            candidate_cache[key] = _candidate_from_moments(
                train_moments, validation_moments, mask, reference['keep'],
                lambda_complexity, lambda_intervention, ridge)
        item = dict(candidate_cache[key])
        item.update(round=int(round_number), source=source)
        archive.append(item)
        return item

    record(np.zeros(p, dtype=bool), -1, 'empty_baseline')
    record(np.ones(p, dtype=bool), -1, 'full_baseline')
    current_mask = np.ones(p, dtype=bool)
    thresholds = (.01, .05, .1, .25, .5, .75)
    for r in range(rounds):
        diagnostic = engine.evaluate(current_mask)
        keep = np.asarray(diagnostic['keep'], dtype=float)
        # Small interior restart lets removed gates re-enter when context changes.
        gates = np.clip(gates, .02, .98)
        iteration = 0
        start = len(archive)

        def objective(z):
            return _profile_from_gram(z, *train_moments, keep,
                                      lambda_complexity, lambda_intervention, ridge)

        def log(z):
            nonlocal iteration
            loss, _ = objective(z)
            trajectory.append(dict(round=r, iteration=iteration, gates=z.tolist(), loss=loss))
            record(z >= .5, r, f'iterate_{iteration}_threshold_0.5')
            iteration += 1

        log(gates)
        result = minimize(objective, gates, jac=True, method='L-BFGS-B', bounds=[(0., 1.)] * p,
                          callback=log, options=dict(maxiter=steps, ftol=1e-11, gtol=1e-7))
        gates = np.clip(result.x, 0, 1)
        for threshold in thresholds:
            record(gates >= threshold, r, f'threshold_{threshold:g}')
        order = np.argsort(-gates, kind='stable')
        for k in range(1, p + 1):
            mask = np.zeros(p, dtype=bool)
            mask[order[:k]] = True
            record(mask, r, f'top_{k}')
        for k in range(samples_per_round):
            record(rng.random(p) < gates, r, f'bernoulli_{k}')
        best_round = min(archive[start:], key=lambda item: (item['score'], sum(item['mask'])))
        round_log.append(dict(round=r, mask_before=current_mask.tolist(), diagnostics=diagnostic,
                              optimizer_success=bool(result.success), optimizer_message=str(result.message),
                              optimizer_iterations=int(result.nit), final_gates=gates.tolist(),
                              selected_mask=best_round['mask']))
        current_mask = np.asarray(best_round['mask'], dtype=bool)
    selected = min(archive, key=lambda item: (item['score'], sum(item['mask'])))
    return dict(selected=selected, archive=archive, trajectory=trajectory, rounds=round_log,
                reference_diagnostics=reference, seed=int(seed),
                settings=dict(lambda_complexity=lambda_complexity,
                              lambda_intervention=lambda_intervention, ridge=ridge,
                              rounds=rounds, steps=steps, samples_per_round=samples_per_round,
                              thresholds=list(thresholds)))


def exact_oracle(train, validation, reference_keep, lambda_complexity=.02,
                 lambda_intervention=.1, ridge=1e-5):
    """Independent enumeration for checking the finite nine-branch experiment."""
    p = train.X.shape[1]
    train_moments, validation_moments = _data_moments(train), _data_moments(validation)
    candidates = [_candidate_from_moments(train_moments, validation_moments, mask, reference_keep,
                                         lambda_complexity, lambda_intervention, ridge)
                  for mask in product((False, True), repeat=p)]
    result = min(candidates, key=lambda item: (item['score'], sum(item['mask'])))
    return {**result, 'candidate_count': len(candidates)}
