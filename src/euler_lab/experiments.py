"""Начальные условия и точное гладкое решение для проверки солвера."""

import numpy as np


def quadrant_state(quadrants, n=64, nz=4):
    """Квадранты UL/UR/LL/LR заданы в порядке rho, u, v, w, p."""
    if n < 2 or n % 2 or nz < 1:
        raise ValueError("n должно быть положительным чётным, nz >= 1")
    state = np.empty((n, n, nz, 5), dtype=float)
    half = n // 2
    for key, xpart, ypart in (
        ("UL", slice(0, half), slice(half, n)),
        ("UR", slice(half, n), slice(half, n)),
        ("LL", slice(0, half), slice(0, half)),
        ("LR", slice(half, n), slice(0, half)),
    ):
        values = np.asarray(quadrants[key], dtype=float)
        if values.shape != (5,) or not np.all(np.isfinite(values)):
            raise ValueError("В каждом квадранте нужны пять конечных значений")
        if values[0] <= 0 or values[4] <= 0:
            raise ValueError("Плотность и давление должны быть положительными")
        state[xpart, ypart] = values
    return state


def entropy_wave(n=24, time=0.0, velocity=(0.3, -0.2, 0.1), amplitude=0.2):
    """Точные средние по ячейкам 3D-волны плотности при постоянных p и скорости.

    rho = 1 + a sin(2 pi (x-u t)) sin(2 pi (y-v t)) sin(2 pi (z-w t)).
    Множитель sinc(1/n)^3 переводит значения в центрах в средние по ячейкам.
    """
    if not isinstance(n, int) or n < 2 or not 0 <= amplitude < 1:
        raise ValueError("Нужны целое n >= 2 и 0 <= amplitude < 1")
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != (3,) or not np.all(np.isfinite(velocity)) or not np.isfinite(time):
        raise ValueError("Скорость и время должны быть конечными")
    axis = (np.arange(n) + 0.5) / n
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    wave = np.ones((n, n, n))
    for coordinate, speed in zip((x, y, z), velocity):
        wave *= np.sin(2 * np.pi * (coordinate - speed * time))
    state = np.empty((n, n, n, 5))
    state[..., 0] = 1 + amplitude * np.sinc(1 / n) ** 3 * wave
    state[..., 1:4] = velocity
    state[..., 4] = 1.0
    return state

