"""Гауссово описание оценок параметров, KL и геометрия ковариаций."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


def _spd(matrix):
    a = np.asarray(matrix, float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)):
        raise ValueError("Нужна конечная квадратная матрица")
    if not np.allclose(a, a.T, rtol=1e-10, atol=1e-15):
        raise ValueError("Матрица должна быть симметричной")
    if np.min(np.linalg.eigvalsh(a)) <= 0:
        raise ValueError("Ковариация вырождена: KL и SPD-геометрия не определены")
    return a


@dataclass
class Gaussian:
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        self.mean = np.asarray(self.mean, float)
        self.covariance = _spd(self.covariance)
        if self.mean.shape != (len(self.covariance),) or not np.all(np.isfinite(self.mean)):
            raise ValueError("Неверный вектор среднего")


def estimate_gaussian(samples):
    samples = np.asarray(samples, float)
    if samples.ndim != 2 or len(samples) <= samples.shape[1] or not np.all(np.isfinite(samples)):
        raise ValueError("Недостаточно конечных векторов для ковариации")
    return Gaussian(samples.mean(axis=0), np.cov(samples, rowvar=False, ddof=1))


def gaussian_kl(p, q):
    """D_KL(P || Q), натуральный логарифм. Это направленная дивергенция."""
    if p.mean.shape != q.mean.shape:
        raise ValueError("Размерности распределений различаются")
    delta = q.mean - p.mean
    trace = np.trace(np.linalg.solve(q.covariance, p.covariance))
    shift = delta @ np.linalg.solve(q.covariance, delta)
    logdet = np.linalg.slogdet(q.covariance)[1] - np.linalg.slogdet(p.covariance)[1]
    result = 0.5 * (trace + shift - len(delta) + logdet)
    if result < -1e-8:
        raise ArithmeticError("Отрицательная KL: проверьте обусловленность")
    return float(max(result, 0.0))


def spd_distance(a, b):
    """Аффинно-инвариантное расстояние на SPD: ||log(A^-1/2 B A^-1/2)||F."""
    eigenvalues = eigh(_spd(b), _spd(a), eigvals_only=True)
    return float(np.linalg.norm(np.log(eigenvalues)))


def spd_geodesic(a, b, t):
    """Геодезическая только ковариационной части, не всей пары (mu, Sigma)."""
    if not 0 <= t <= 1:
        raise ValueError("t должна быть от 0 до 1")
    a, b = _spd(a), _spd(b)
    values, vectors = np.linalg.eigh(a)
    root = (vectors * np.sqrt(values)) @ vectors.T
    inverse = (vectors * (1 / np.sqrt(values))) @ vectors.T
    values, vectors = np.linalg.eigh(inverse @ b @ inverse)
    result = root @ ((vectors * values**t) @ vectors.T) @ root
    return (result + result.T) / 2


def metric_length(delta, covariance):
    """Локальная длина изменения параметров при g=Sigma^-1."""
    covariance = _spd(covariance)
    delta = np.asarray(delta, float)
    return float(np.sqrt(delta @ np.linalg.solve(covariance, delta)))
