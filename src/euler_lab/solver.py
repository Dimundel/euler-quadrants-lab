"""Конечные объёмы для трёхмерных уравнений Эйлера идеального газа."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _check_gamma(gamma):
    if not np.isfinite(gamma) or gamma <= 1:
        raise ValueError("gamma должен быть конечным числом больше 1")


def _check_axis(axis):
    if not isinstance(axis, Integral) or isinstance(axis, bool) or axis not in (0, 1, 2):
        raise ValueError("axis должен быть равен 0, 1 или 2")


def _as_state(state):
    state = np.asarray(state, dtype=np.float64)
    if state.ndim < 1 or state.shape[-1] != 5 or state.size == 0:
        raise ValueError("Последняя ось состояния должна иметь длину 5")
    if not np.all(np.isfinite(state)):
        raise ValueError("Состояние содержит NaN или бесконечность")
    return state


def _check_positive(primitive):
    if np.any(primitive[..., 0] <= 0):
        raise ValueError("Плотность должна быть положительной во всех ячейках")
    if np.any(primitive[..., 4] <= 0):
        raise ValueError("Давление должно быть положительным во всех ячейках")


def primitive_to_conservative(primitive, gamma=1.4):
    """Перевести (rho, u, v, w, p) в (rho, rho*u, rho*v, rho*w, E)."""
    _check_gamma(gamma)
    primitive = _as_state(primitive)
    _check_positive(primitive)
    rho = primitive[..., 0]
    velocity = primitive[..., 1:4]
    conservative = np.empty_like(primitive)
    conservative[..., 0] = rho
    with np.errstate(over="ignore", invalid="ignore"):
        conservative[..., 1:4] = rho[..., None] * velocity
        conservative[..., 4] = (
            primitive[..., 4] / (gamma - 1)
            + 0.5 * rho * np.sum(velocity**2, axis=-1)
        )
    return _as_state(conservative)


def conservative_to_primitive(conservative, gamma=1.4):
    """Восстановить примитивные переменные; нефизическое состояние — ошибка."""
    _check_gamma(gamma)
    conservative = _as_state(conservative)
    rho = conservative[..., 0]
    if np.any(rho <= 0):
        raise ValueError("Плотность должна быть положительной во всех ячейках")
    primitive = np.empty_like(conservative)
    primitive[..., 0] = rho
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        primitive[..., 1:4] = conservative[..., 1:4] / rho[..., None]
        kinetic = 0.5 * np.sum(
            conservative[..., 1:4] * primitive[..., 1:4], axis=-1
        )
        primitive[..., 4] = (gamma - 1) * (conservative[..., 4] - kinetic)
    _as_state(primitive)
    _check_positive(primitive)
    return primitive


def _physical_flux(conservative, primitive, axis):
    normal_velocity = primitive[..., axis + 1]
    pressure = primitive[..., 4]
    result = conservative * normal_velocity[..., None]
    result[..., axis + 1] += pressure
    result[..., 4] += pressure * normal_velocity
    return result


def flux(conservative, axis, gamma=1.4):
    """Физический поток через грань с нормалью вдоль x, y или z."""
    _check_axis(axis)
    primitive = conservative_to_primitive(conservative, gamma)
    return _physical_flux(np.asarray(conservative, dtype=np.float64), primitive, axis)


def rusanov_flux(left, right, axis, gamma=1.4):
    """Поток Русанова с максимальной скоростью |v_n| + c по обе стороны."""
    _check_axis(axis)
    left = _as_state(left)
    right = _as_state(right)
    if left.shape != right.shape:
        raise ValueError("Состояния слева и справа должны иметь одинаковую форму")
    wl = conservative_to_primitive(left, gamma)
    wr = conservative_to_primitive(right, gamma)
    speed_left = np.abs(wl[..., axis + 1]) + np.sqrt(gamma * wl[..., 4] / wl[..., 0])
    speed_right = np.abs(wr[..., axis + 1]) + np.sqrt(gamma * wr[..., 4] / wr[..., 0])
    speed = np.maximum(speed_left, speed_right)
    return (
        0.5 * (_physical_flux(left, wl, axis) + _physical_flux(right, wr, axis))
        - 0.5 * speed[..., None] * (right - left)
    )


@dataclass
class EulerSolver:
    """Явная схема первого порядка на равномерной декартовой сетке."""

    gamma: float = 1.4
    cfl: float = 0.4
    boundary: tuple = ("outflow", "outflow", "periodic")

    def _validate(self):
        _check_gamma(self.gamma)
        if not np.isfinite(self.cfl) or not 0 < self.cfl <= 1:
            raise ValueError("cfl должен лежать в интервале (0, 1]")
        if len(self.boundary) != 3 or any(
            boundary not in ("periodic", "outflow") for boundary in self.boundary
        ):
            raise ValueError("Нужны три граничных условия: periodic или outflow")

    def _rhs(self, conservative, spacing):
        result = np.zeros_like(conservative)
        for axis in range(3):
            if conservative.shape[axis] == 1:
                continue
            # По одной фиктивной ячейке с каждой стороны достаточно для первого порядка.
            first = np.take(conservative, [0], axis=axis)
            last = np.take(conservative, [-1], axis=axis)
            if self.boundary[axis] == "periodic":
                padded = np.concatenate((last, conservative, first), axis=axis)
            else:
                padded = np.concatenate((first, conservative, last), axis=axis)
            left = [slice(None)] * 4
            right = [slice(None)] * 4
            left[axis] = slice(None, -1)
            right[axis] = slice(1, None)
            face_flux = rusanov_flux(
                padded[tuple(left)], padded[tuple(right)], axis, self.gamma
            )
            result -= np.diff(face_flux, axis=axis) / spacing[axis]
        return result

    def solve(self, initial_primitive, times, spacing):
        """Вернуть (nt, nx, ny, nz, 5); начальное состояние задано при times[0]."""
        self._validate()
        initial = _as_state(initial_primitive)
        if initial.ndim != 4:
            raise ValueError("Начальное состояние должно иметь форму (nx, ny, nz, 5)")
        times = np.asarray(times, dtype=np.float64)
        if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
            raise ValueError("times должен быть непустым одномерным массивом конечных чисел")
        if np.any(np.diff(times) <= 0):
            raise ValueError("Моменты времени должны строго возрастать")
        spacing = np.asarray(spacing, dtype=np.float64)
        if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            raise ValueError("spacing должен содержать три положительных шага сетки")

        conservative = primitive_to_conservative(initial, self.gamma)
        primitive = conservative_to_primitive(conservative, self.gamma)
        snapshots = np.empty((times.size, *initial.shape), dtype=np.float64)
        snapshots[0] = primitive
        current_time = float(times[0])
        active_axes = [axis for axis in range(3) if initial.shape[axis] > 1]

        for frame, target in enumerate(times[1:], start=1):
            while current_time < target:
                sound_speed = np.sqrt(self.gamma * primitive[..., 4] / primitive[..., 0])
                # В нерасщеплённой схеме складываются ограничения всех направлений.
                rate = sum(
                    np.max(np.abs(primitive[..., axis + 1]) + sound_speed) / spacing[axis]
                    for axis in active_axes
                )
                dt = min(self.cfl / rate, target - current_time) if rate > 0 else target - current_time
                if not np.isfinite(dt) or dt <= 0 or current_time + dt <= current_time:
                    raise RuntimeError("Шаг времени слишком мал; проверьте масштаб входных данных")
                conservative = conservative + dt * self._rhs(conservative, spacing)
                try:
                    primitive = conservative_to_primitive(conservative, self.gamma)
                except ValueError as error:
                    raise RuntimeError(
                        f"Нефизическое состояние при t={current_time + dt:.8g}: {error}. "
                        "Уменьшите cfl и проверьте начальные данные."
                    ) from error
                current_time += dt
            snapshots[frame] = primitive
        return snapshots
