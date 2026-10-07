"""Вспомогательные расчёты для задач Римана."""

from dataclasses import dataclass
from operator import index

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares


@dataclass(frozen=True)
class GammaFit:
    """Результат подбора показателя адиабаты по заданным ячейкам."""

    gamma: float
    rmse: float
    n_samples: int
    success: bool


def equation_graph() -> nx.DiGraph:
    """Граф вычисления правой части трёхмерных уравнений Эйлера."""
    graph = nx.DiGraph()
    nodes = [
        ("rho", r"$\rho$", "input", 0),
        ("mx", r"$m_x$", "input", 0),
        ("my", r"$m_y$", "input", 0),
        ("mz", r"$m_z$", "input", 0),
        ("E", r"$E$", "input", 0),
        ("gamma", r"$\gamma$", "parameter", 0),
        ("u", r"$u=m_x/\rho$", "operation", 1),
        ("v", r"$v=m_y/\rho$", "operation", 1),
        ("w", r"$w=m_z/\rho$", "operation", 1),
        ("kinetic", r"$K=\rho(u^2+v^2+w^2)/2$", "operation", 2),
        ("pressure", r"$p=(\gamma-1)(E-K)$", "operation", 3),
        ("Fx", r"$\mathbf{F}_x(\mathbf{U})$", "operation", 4),
        ("Fy", r"$\mathbf{F}_y(\mathbf{U})$", "operation", 4),
        ("Fz", r"$\mathbf{F}_z(\mathbf{U})$", "operation", 4),
        ("dFx_dx", r"$\partial_x\mathbf{F}_x$", "operation", 5),
        ("dFy_dy", r"$\partial_y\mathbf{F}_y$", "operation", 5),
        ("dFz_dz", r"$\partial_z\mathbf{F}_z$", "operation", 5),
        ("dU_dt", r"$\partial_t\mathbf{U}=-\nabla\cdot\mathbf{F}$", "output", 6),
    ]
    for name, label, kind, layer in nodes:
        graph.add_node(name, label=label, kind=kind, layer=layer)

    dependencies = {
        "u": ("mx", "rho"),
        "v": ("my", "rho"),
        "w": ("mz", "rho"),
        "kinetic": ("rho", "u", "v", "w"),
        "pressure": ("gamma", "E", "kinetic"),
        # Fx = (mx, mx*u+p, my*u, mz*u, (E+p)*u); Fy и Fz аналогично.
        "Fx": ("mx", "my", "mz", "u", "pressure", "E"),
        "Fy": ("mx", "my", "mz", "v", "pressure", "E"),
        "Fz": ("mx", "my", "mz", "w", "pressure", "E"),
        "dFx_dx": ("Fx",),
        "dFy_dy": ("Fy",),
        "dFz_dz": ("Fz",),
        "dU_dt": ("dFx_dx", "dFy_dy", "dFz_dz"),
    }
    for target, sources in dependencies.items():
        graph.add_edges_from((source, target) for source in sources)
    return graph


def _finite_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name}: ожидается вещественный числовой массив")
    array = np.asarray(array, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name}: массив должен быть непустым и конечным")
    return array


def fit_gamma(
    rho: ArrayLike,
    momentum: ArrayLike,
    energy: ArrayLike,
    pressure: ArrayLike,
    bounds: tuple[float, float] = (1.05, 1.9),
) -> GammaFit:
    """Подобрать γ из p=(γ−1)(E−|m|²/(2ρ)) методом наименьших квадратов.

    E — полная энергия на единицу объёма, а m — плотность импульса.
    Последняя ось m содержит две или три компоненты. Все остальные формы
    совпадают. Передавать нужно только обучающую выборку; rmse относится к ней.
    Начальное приближение 1.3 ограничивается переданными границами.
    """
    density = _finite_array(rho, "rho")
    momenta = _finite_array(momentum, "momentum")
    total_energy = _finite_array(energy, "energy")
    observed_pressure = _finite_array(pressure, "pressure")
    if density.shape != total_energy.shape or density.shape != observed_pressure.shape:
        raise ValueError("rho, energy и pressure должны иметь одинаковую форму")
    if (
        momenta.ndim != density.ndim + 1
        or momenta.shape[:-1] != density.shape
        or momenta.shape[-1] not in (2, 3)
    ):
        raise ValueError("momentum должен иметь форму rho.shape + (2,) или (3,)")
    limits = np.asarray(bounds, dtype=np.float64)
    if limits.shape != (2,) or not np.all(np.isfinite(limits)):
        raise ValueError("bounds должны содержать две конечные границы")
    lower, upper = limits
    if not 1.0 < lower < upper:
        raise ValueError("Для показателя адиабаты требуется 1 < lower < upper")
    if np.any(density <= 0) or np.any(observed_pressure <= 0):
        raise ValueError("Плотность и давление должны быть положительными")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        internal_energy = total_energy - np.sum(momenta**2, axis=-1) / (2 * density)
    if not np.all(np.isfinite(internal_energy)) or np.any(internal_energy <= 0):
        raise ValueError("Внутренняя энергия E−|m|²/(2ρ) должна быть конечной и положительной")

    internal_flat = internal_energy.ravel()
    pressure_flat = observed_pressure.ravel()
    # Масштабирование не меняет минимум, но убирает зависимость допуска от единиц.
    scale = np.max(pressure_flat)
    scaled_internal = internal_flat / scale
    scaled_pressure = pressure_flat / scale
    fit = least_squares(
        lambda gamma: (gamma[0] - 1.0) * scaled_internal - scaled_pressure,
        x0=[np.clip(1.3, lower, upper)],
        jac=lambda gamma: scaled_internal[:, None],
        bounds=([lower], [upper]),
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    gamma = float(fit.x[0])
    residual = (gamma - 1.0) * internal_flat - pressure_flat
    return GammaFit(
        gamma=gamma,
        rmse=_rms(residual),
        n_samples=int(density.size),
        success=bool(fit.success),
    )


def _rms(values: NDArray[np.float64]) -> float:
    """Среднеквадратичное значение без возведения больших чисел в квадрат."""
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    return float(scale * np.sqrt(np.mean((values / scale) ** 2)))


def error_metrics(prediction: ArrayLike, reference: ArrayLike) -> dict[str, float]:
    """MAE, RMSE и относительная L2-ошибка по всем переданным элементам.

    Для нулевого эталона relative_l2 равна 0 при точном совпадении и inf иначе.
    Массивы должны иметь одинаковую форму, автоматического broadcasting нет.
    """
    predicted = _finite_array(prediction, "prediction")
    expected = _finite_array(reference, "reference")
    if predicted.shape != expected.shape:
        raise ValueError("prediction и reference должны иметь одинаковую форму")
    difference = predicted - expected
    rmse = _rms(difference)
    reference_rms = _rms(expected)
    relative_l2 = rmse / reference_rms if reference_rms > 0 else (0.0 if rmse == 0 else float("inf"))
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "relative_l2": relative_l2,
    }


def _positive_integer(value: int, name: str) -> int:
    try:
        integer = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} должен быть положительным целым числом") from exc
    if isinstance(value, (bool, np.bool_)) or integer < 1:
        raise ValueError(f"{name} должен быть положительным целым числом")
    return integer


def _field_array(value: ArrayLike, name: str) -> NDArray:
    array = np.asarray(value)
    if array.ndim < 3 or array.shape[-1] != 5:
        raise ValueError(f"{name}: требуется форма (..., nx, ny, 5)")
    if (
        array.size == 0
        or not np.issubdtype(array.dtype, np.number)
        or np.iscomplexobj(array)
        or not np.all(np.isfinite(array))
    ):
        raise ValueError(f"{name}: требуется непустой конечный вещественный массив")
    return array


def coarsen_conservative(U: ArrayLike, factor: int) -> NDArray:
    """Усреднить консервативные переменные по квадратным блокам factor×factor.

    Пространственные оси — предпоследние перед осью компонент: (..., nx, ny, 5).
    Сетка считается равномерной. Дробные типы сохраняются; целые переходят в
    float64, чтобы не округлять средние. Для factor=1 возвращается копия.
    """
    values = _field_array(U, "U")
    factor = _positive_integer(factor, "factor")
    nx, ny = values.shape[-3:-1]
    if nx % factor or ny % factor:
        raise ValueError("Размеры nx и ny должны делиться на factor без остатка")
    if factor == 1:
        return values.copy()
    blocks = values.reshape(*values.shape[:-3], nx // factor, factor, ny // factor, factor, 5)
    return blocks.mean(axis=(-4, -2))


def extrude_planar(W: ArrayLike, nz: int) -> NDArray:
    """Продлить плоское состояние (ρ,u,v,w=0,p) одинаковыми слоями по z.

    Форма (..., nx, ny, 5) переходит в (..., nx, ny, nz, 5).
    Возвращается отдельный массив, а не разделяющее память представление.
    """
    values = _field_array(W, "W")
    nz = _positive_integer(nz, "nz")
    if np.any(values[..., 0] <= 0) or np.any(values[..., 4] <= 0):
        raise ValueError("Плотность и давление должны быть положительными")
    if np.any(values[..., 3] != 0):
        raise ValueError("Для плоского состояния требуется w=0")
    return np.repeat(values[..., None, :], nz, axis=-2)
