"""Independent Jacobian checks for the changed pressure law."""
import numpy as np
import pytest

from graph_lab.control_solver import control_flux, control_wave_speed, solve_pressure_control


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("gamma", [1.2, 1.4, 1.67])
def test_wave_speed_matches_flux_jacobian_eigenvalues(axis, gamma):
    # Complex-step derivatives independently check the analytical spectrum.
    for velocity in ([2., -.7, .2], [-3., 1.5, -.4], [0., 0., 0.]):
        rho, pressure = .8, 1.3
        state = np.array([rho, *(rho * np.asarray(velocity)), pressure / (gamma - 1)])
        jacobian = np.column_stack([
            np.imag(control_flux(state.astype(complex) + 1e-25j * np.eye(5)[j], axis, gamma)) / 1e-25
            for j in range(5)
        ])
        eigenvalues = np.linalg.eigvals(jacobian)
        assert np.max(np.abs(eigenvalues.imag)) < 1e-10
        assert control_wave_speed(state, axis, gamma) == pytest.approx(
            np.max(np.abs(eigenvalues)), rel=1e-12)


def test_euler_speed_is_not_valid_for_modified_pressure():
    state = np.array([1., 2., 0., 0., 1. / .4])
    euler_speed = 2. + np.sqrt(1.4)
    assert control_wave_speed(state, 0) > euler_speed


def test_constant_control_state_is_preserved_with_modified_energy():
    initial = np.broadcast_to([1.2, 1., -.5, .3, .8], (6, 7, 1, 5)).copy()
    result = solve_pressure_control(initial, [0., .03, .1], [1/6, 1/7, 1.])
    expected = np.broadcast_to([1.2, 1.2, -.6, .36, 2.], result.shape)
    np.testing.assert_allclose(result, expected, atol=1e-14)


def test_planar_control_keeps_zero_transverse_momentum():
    initial = np.broadcast_to([1., .2, .1, 0., 1.], (12, 12, 1, 5)).copy()
    initial[:6, :, :, 0] = .7
    initial[:, :6, :, 4] = .8
    result = solve_pressure_control(initial, [0., .02, .04], [1/12, 1/12, 1.])
    assert np.all(np.isfinite(result))
    assert np.min(result[..., 0]) > 0
    assert np.min(result[..., 4]) > 0
    np.testing.assert_array_equal(result[..., 3], 0.)
