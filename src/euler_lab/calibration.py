"""Подбор γ по наблюдаемому изменению плотности и давления."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize_scalar

from .solver import EulerSolver


@dataclass(frozen=True)
class TrajectoryFit:
    """Результат численного подбора; loss относится к обучающим кадрам."""

    gamma: float
    loss: float
    success: bool
    evaluations: int


def fit_trajectory_gamma(
    initial_primitive: ArrayLike,
    times: ArrayLike,
    reference_primitive: ArrayLike,
    spacing: ArrayLike,
    bounds: tuple[float, float] = (1.05, 1.9),
    cfl: float = 0.4,
    boundary: tuple[str, str, str] = ("outflow", "outflow", "periodic"),
) -> TrajectoryFit:
    """Минимизировать ошибку прямого расчёта по обучающей траектории.

    Начальное состояние (rho,u,v,w,p) задано при times[0]. Для каждого γ
    решатель вычисляет соответствующую ему начальную полную энергию.
    Нулевой кадр исключён из функции потерь. Плотность и давление делятся
    каждое на один RMS эталона по всем остальным обучающим кадрам и ячейкам;
    loss — средний квадрат двух нормированных ошибок.

    Передавать следует только обучающие кадры. Успех оптимизатора сам по себе
    не доказывает определимость γ, например в однородном неподвижном газе.
    """
    initial = np.asarray(initial_primitive, dtype=np.float64)
    reference = np.asarray(reference_primitive, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    limits = np.asarray(bounds, dtype=np.float64)

    if initial.ndim != 4 or initial.shape[-1] != 5 or initial.size == 0:
        raise ValueError("initial_primitive должен иметь форму (nx, ny, nz, 5)")
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError("Нужны начальный момент и хотя бы один конечный обучающий момент")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Моменты времени должны строго возрастать")
    if reference.shape != (times.size, *initial.shape):
        raise ValueError("reference_primitive должен иметь форму (nt, nx, ny, nz, 5)")
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(reference)):
        raise ValueError("Начальное состояние и эталон должны содержать конечные числа")
    if any(np.any(field[..., [0, 4]] <= 0) for field in (initial, reference)):
        raise ValueError("Плотность и давление должны быть положительными")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing должен содержать три положительных шага сетки")
    if limits.shape != (2,) or not np.all(np.isfinite(limits)) or not 1 < limits[0] < limits[1]:
        raise ValueError("Для границ показателя адиабаты требуется 1 < lower < upper")
    if not np.isfinite(cfl) or not 0 < cfl <= 1:
        raise ValueError("cfl должен лежать в интервале (0, 1]")
    if len(boundary) != 3 or any(value not in ("periodic", "outflow") for value in boundary):
        raise ValueError("Нужны три граничных условия: periodic или outflow")

    target = reference[1:][..., [0, 4]]
    maxima = np.max(target, axis=(0, 1, 2, 3))
    scale = maxima * np.sqrt(np.mean((target / maxima) ** 2, axis=(0, 1, 2, 3)))

    def objective(gamma: float) -> float:
        predicted = EulerSolver(gamma=gamma, cfl=cfl, boundary=boundary).solve(
            initial, times, spacing
        )
        residual = (predicted[1:][..., [0, 4]] - target) / scale
        return float(np.mean(residual**2))

    result = minimize_scalar(
        objective,
        bounds=tuple(limits),
        method="bounded",
        options={"xatol": 0.005, "maxiter": 20},
    )
    return TrajectoryFit(
        gamma=float(result.x),
        loss=float(result.fun),
        success=bool(result.success and np.isfinite(result.fun)),
        evaluations=int(result.nfev),
    )
