"""Парные оценки параметров до и после зануления давления."""

import numpy as np

from .geometry import estimate_gaussian, gaussian_kl, metric_length, spd_distance
from .model import fit_blocks, least_squares, prediction_error


def run_experiment(data, block_size=10, n_blocks=12, gap=4):
    """Первые 120 кадров для оценки, после временного зазора — проверка.

    Это распределение оценок по временным окнам, не байесовский posterior
    и не доверительный интервал истинного коэффициента.
    """
    end = block_size * n_blocks
    if n_blocks < 4 or block_size < 1 or gap < 4 or len(data.time) <= end + gap:
        raise ValueError("Недостаточно кадров или слишком короткий зазор")
    blocks = [np.arange(i * block_size, (i + 1) * block_size) for i in range(n_blocks)]
    train = np.arange(end)
    test = np.arange(end + gap, len(data.time))
    base_samples = fit_blocks(data, blocks)
    base = estimate_gaussian(base_samples)
    global_theta = least_squares(*data.design(train))
    base_error = prediction_error(data, test, global_theta)
    rows, samples = [], []
    # alpha=1 — назначенное зануление. Промежуточные alpha показывают путь к нему.
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        changed = data.pressure_intervention(alpha)
        estimates = fit_blocks(changed, blocks)
        distribution = estimate_gaussian(estimates)
        theta = least_squares(*changed.design(train))
        error = prediction_error(changed, test, theta)
        rows.append(dict(
            alpha=alpha, mean=distribution.mean.tolist(),
            covariance=distribution.covariance.tolist(),
            global_theta=theta.tolist(),
            kl=gaussian_kl(base, distribution),
            reverse_kl=gaussian_kl(distribution, base),
            spd_distance=spd_distance(base.covariance, distribution.covariance),
            mean_metric_length=metric_length(distribution.mean - base.mean, base.covariance),
            test_error=error,
            error_with_frozen_theta=prediction_error(changed, test, global_theta),
        ))
        samples.append(estimates)
    final = rows[-1]
    # Пороги заданы как учебное правило до расчёта результатов, не как p-value.
    eligible = (base_error <= 0.05 and final["kl"] <= 0.1 and final["test_error"] <= base_error + 0.01
                and final["global_theta"][0] > 0 and final["global_theta"][1] > 0)
    return dict(
        rows=rows, samples=np.asarray(samples),
        train_frames=train, test_frames=test,
        train_time=[float(data.time[train[0]]), float(data.time[train[-1]])],
        test_time=[float(data.time[test[0]]), float(data.time[test[-1]])],
        block_count=n_blocks, block_size=block_size,
        decision="candidate_for_removal" if eligible else "retain_pressure_branch",
        rule={"max_baseline_relative_l2": 0.05,
              "max_kl": 0.1, "max_absolute_relative_l2_increase": 0.01,
              "positive_coefficients_required": True},
    )


def json_result(result):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in result.items()}
