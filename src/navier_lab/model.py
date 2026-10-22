"""Оценка коэффициентов уравнения по наблюдаемым полям скорости и давления."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MomentumData:
    # Массивы имеют форму (nt, ny, nx, 2), последняя ось — уравнения u и v.
    time: np.ndarray
    convection: np.ndarray
    laplacian: np.ndarray
    pressure_gradient: np.ndarray
    acceleration: np.ndarray

    def pressure_intervention(self, fraction=1.0):
        """p -> (1-fraction)*p. Основной опыт использует полное зануление."""
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("fraction должна быть от 0 до 1")
        return MomentumData(self.time.copy(), self.convection.copy(),
                            self.laplacian.copy(),
                            (1 - fraction) * self.pressure_gradient,
                            self.acceleration.copy())

    def design(self, indices):
        """a = -beta*C + nu*L - grad(p), rho=1, внешняя сила равна нулю."""
        x = np.column_stack((-self.convection[indices].ravel(),
                             self.laplacian[indices].ravel()))
        y = (self.acceleration[indices] + self.pressure_gradient[indices]).ravel()
        return x, y


def derivative(values, step, axis, order=1):
    """Центральный пятиузловой шаблон. Края нужно исключить после расчёта."""
    if not np.isfinite(step) or step <= 0 or values.shape[axis] < 5:
        raise ValueError("Нужны положительный шаг и минимум пять узлов")
    if order == 1:
        weights = (1, -8, 0, 8, -1)
        denominator = 12 * step
    elif order == 2:
        weights = (-1, 16, -30, 16, -1)
        denominator = 12 * step**2
    else:
        raise ValueError("Поддерживаются производные порядка 1 и 2")
    return sum(w * np.roll(values, -offset, axis=axis)
               for offset, w in zip(range(-2, 3), weights)) / denominator


def _uniform_step(coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    differences = np.diff(coordinates)
    if coordinates.ndim != 1 or len(coordinates) < 5 or not np.all(np.isfinite(coordinates)):
        raise ValueError("Нужна конечная одномерная сетка из пяти или более узлов")
    if differences[0] <= 0 or not np.allclose(differences, differences[0], rtol=1e-6):
        raise ValueError("Сетка должна быть равномерной и возрастающей")
    return float(differences[0])


def momentum_from_fields(u, v, p, x, y, time, *, skip_initial_frames=0):
    """Производные внутренних узлов, без периодического склеивания границ.

    skip_initial_frames убирает начальные кадры до расчёта производных.
    Так исключённый кадр не попадёт в пятиузловой временной шаблон.
    """
    if (isinstance(skip_initial_frames, (bool, np.bool_))
            or not isinstance(skip_initial_frames, (int, np.integer))
            or skip_initial_frames < 0):
        raise ValueError("skip_initial_frames должен быть целым неотрицательным числом")
    u, v, p = (np.asarray(a, dtype=float) for a in (u, v, p))
    time = np.asarray(time, dtype=float)
    shape = (len(time), len(y), len(x))
    if any(a.shape != shape for a in (u, v, p)):
        raise ValueError("Поля должны иметь форму (nt, ny, nx)")
    if len(time) - skip_initial_frames < 5:
        raise ValueError("После исключения начальных кадров должно остаться минимум пять")
    time = time[skip_initial_frames:]
    u, v, p = (a[skip_initial_frames:] for a in (u, v, p))
    if any(not np.all(np.isfinite(a)) for a in (u, v, p)):
        raise ValueError("Поля должны иметь конечные значения и форму (nt, ny, nx)")
    dt, dy, dx = (_uniform_step(c) for c in (time, y, x))
    ux, uy = derivative(u, dx, 2), derivative(u, dy, 1)
    vx, vy = derivative(v, dx, 2), derivative(v, dy, 1)
    cut = (slice(2, -2), slice(2, -2), slice(2, -2))
    def pair(a, b):
        return np.stack((a[cut], b[cut]), axis=-1)
    return MomentumData(
        time[2:-2],
        pair(u * ux + v * uy, u * vx + v * vy),
        pair(derivative(u, dx, 2, 2) + derivative(u, dy, 1, 2),
             derivative(v, dx, 2, 2) + derivative(v, dy, 1, 2)),
        pair(derivative(p, dx, 2), derivative(p, dy, 1)),
        pair(derivative(u, dt, 0), derivative(v, dt, 0)),
    )


def taylor_green(n=32, nt=200, nu=0.01, noise_fraction=0.01, seed=11):
    """Точное течение и наблюдаемые ускорения с заданной погрешностью.

    Шум добавлен только к ускорению, не к скорости или давлению. Это модель
    погрешности синтетических наблюдений, а интервенция в работе — p=0.
    """
    if n < 8 or nt < 20 or nu <= 0 or not 0 <= noise_fraction < 1:
        raise ValueError("Некорректные параметры синтетического опыта")
    coordinate = np.arange(n) * 2 * np.pi / n
    x, y = np.meshgrid(coordinate, coordinate)
    time = np.arange(nt) * 0.1
    decay = np.exp(-2 * nu * time)[:, None, None]
    u = np.sin(x) * np.cos(y) * decay
    v = -np.cos(x) * np.sin(y) * decay
    p = 0.25 * (np.cos(2 * x) + np.cos(2 * y)) * decay**2
    convection = np.stack((np.sin(x) * np.cos(x) * decay**2,
                           np.sin(y) * np.cos(y) * decay**2), axis=-1)
    laplacian = -2 * np.stack((u, v), axis=-1)
    acceleration = nu * laplacian
    sigma = noise_fraction * np.sqrt(np.mean(acceleration**2))
    observed = acceleration + np.random.default_rng(seed).normal(0, sigma, acceleration.shape)
    data = MomentumData(time, convection, laplacian, -convection, observed)
    fields = dict(x=coordinate, y=coordinate, time=time, u=u, v=v, p=p,
                  sigma_acceleration=float(sigma), nu=nu)
    return data, fields


def least_squares(x, y):
    """МНК через SVD с масштабированием столбцов, без ridge и ограничений."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim != 2 or x.shape[1] != 2 or y.shape != (len(x),) or len(x) < 3:
        raise ValueError("Нужны два признака и не менее трёх наблюдений")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Наблюдения должны быть конечными")
    scale = np.sqrt(np.mean(x**2, axis=0))
    if np.any(scale == 0):
        raise ValueError("Параметры неидентифицируемы: нулевой столбец")
    normalized = x / scale
    fitted, _, rank, singular = np.linalg.lstsq(normalized, y, rcond=None)
    if rank != 2 or singular[0] / singular[-1] > 1e8:
        raise ValueError("Параметры неидентифицируемы: вырожденный дизайн")
    return fitted / scale


def fit_blocks(data, blocks):
    """Один вектор (beta, nu) для каждого непересекающегося временного блока."""
    return np.asarray([least_squares(*data.design(block)) for block in blocks])


def prediction_error(data, indices, theta):
    """Относительная L2-ошибка ускорения, не ошибка будущей траектории."""
    x, _ = data.design(indices)
    predicted = x @ theta - data.pressure_gradient[indices].ravel()
    expected = data.acceleration[indices].ravel()
    scale = np.linalg.norm(expected)
    if scale == 0:
        raise ValueError("Нулевая норма ускорения")
    return float(np.linalg.norm(predicted - expected) / scale)
