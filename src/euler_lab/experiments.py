"""Начальные условия двумерных задач Римана."""

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
