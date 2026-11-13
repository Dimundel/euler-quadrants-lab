"""Rusanov solver for the control mechanism p = (gamma - 1) E.

Ivan Papay's control changes the pressure law but retains the ideal-gas
Euler wave speed. Here the flux is unchanged from that control, while the
numerical dissipation and CFL step use its own exact characteristic speeds.
The original vendored implementation remains available without modification.
"""
from __future__ import annotations

import numpy as np


def control_flux(conservative, axis, gamma=1.4):
    """Flux of the modified conservation law, along a coordinate axis."""
    conservative = np.asarray(conservative)
    normal_velocity = conservative[..., axis + 1] / conservative[..., 0]
    pressure = (gamma - 1) * conservative[..., 4]
    result = conservative * normal_velocity[..., None]
    result[..., axis + 1] += pressure
    result[..., 4] += pressure * normal_velocity
    return result


def control_wave_speed(conservative, axis, gamma=1.4):
    """Spectral radius of the modified flux Jacobian.

    The eigenvalues are u_d (multiplicity three) and
    (gamma+1)*u_d/2 +/- sqrt((gamma-1)^2*u_d^2/4 + gamma*p/rho).
    They are real for rho>0, p>0 and gamma>1. In particular the usual
    Euler expression abs(u_d)+sqrt(gamma*p/rho) is too small when u_d != 0.
    """
    conservative = np.asarray(conservative)
    rho = conservative[..., 0]
    velocity = conservative[..., axis + 1] / rho
    pressure = (gamma - 1) * conservative[..., 4]
    return ((gamma + 1) * np.abs(velocity) / 2
            + np.sqrt(((gamma - 1) * velocity / 2) ** 2 + gamma * pressure / rho))


def solve_pressure_control(initial_primitive, times, spacing, gamma=1.4, cfl=.4):
    """Solve the planar control PDE with outflow boundaries.

    Input has shape (nx, ny, 1, 5), ordered (rho,u,v,w,p); output has
    shape (nt,nx,ny,1,5), ordered (rho,mx,my,mz,E). For this control
    E=p/(gamma-1), with no kinetic contribution. This is an artificial
    conservation law for checking graph selection, not ideal-gas Euler.
    """
    initial = np.asarray(initial_primitive, dtype=float)
    times = np.asarray(times, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    if (initial.ndim != 4 or initial.shape[2:] != (1, 5)
            or min(initial.shape[:2]) < 1 or not np.all(np.isfinite(initial))):
        raise ValueError("initial_primitive must be a finite planar (nx,ny,1,5) field")
    if (not np.isfinite(gamma) or gamma <= 1 or not np.isfinite(cfl)
            or not 0 < cfl <= 1):
        raise ValueError("gamma must exceed 1 and cfl must lie in (0,1]")
    if (times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("times must be finite and strictly increasing")
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing must contain three positive finite values")
    if np.any(initial[..., 0] <= 0) or np.any(initial[..., 4] <= 0):
        raise ValueError("density and pressure must be positive")

    state = np.empty_like(initial)
    state[..., 0] = initial[..., 0]
    state[..., 1:4] = initial[..., :1] * initial[..., 1:4]
    state[..., 4] = initial[..., 4] / (gamma - 1)

    def rhs(conservative):
        derivative = np.zeros_like(conservative)
        for axis in (0, 1):
            padded = np.concatenate((np.take(conservative, [0], axis=axis),
                                     conservative,
                                     np.take(conservative, [-1], axis=axis)), axis=axis)
            flux = control_flux(padded, axis, gamma)
            speed = control_wave_speed(padded, axis, gamma)
            left, right = [slice(None)] * 4, [slice(None)] * 4
            left[axis], right[axis] = slice(None, -1), slice(1, None)
            left, right = tuple(left), tuple(right)
            interface_speed = np.maximum(speed[left[:3]], speed[right[:3]])[..., None]
            face = (.5 * (flux[left] + flux[right])
                    - .5 * interface_speed * (padded[right] - padded[left]))
            derivative -= np.diff(face, axis=axis) / spacing[axis]
        return derivative

    snapshots = np.empty((len(times), *state.shape))
    snapshots[0] = state
    time = float(times[0])
    for i, target in enumerate(times[1:], start=1):
        while time < target:
            rate = sum(np.max(control_wave_speed(state, axis, gamma)) / spacing[axis]
                       for axis in (0, 1))
            step = min(cfl / rate, target - time)
            state = state + step * rhs(state)
            if (not np.all(np.isfinite(state)) or np.any(state[..., 0] <= 0)
                    or np.any(state[..., 4] <= 0)):
                raise RuntimeError("nonpositive or nonfinite state in pressure control")
            time += step
        snapshots[i] = state
    return snapshots
