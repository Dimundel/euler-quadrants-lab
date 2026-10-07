"""Проверки физических свойств схемы, без эталонных массивов из самого солвера."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from euler_lab.solver import (
    EulerSolver,
    conservative_to_primitive,
    flux,
    primitive_to_conservative,
    rusanov_flux,
)


def smooth_state(shape):
    coordinates = [(np.arange(n) + 0.5) / n for n in shape]
    x, y, z = np.meshgrid(*coordinates, indexing="ij")
    state = np.empty((*shape, 5))
    state[..., 0] = 1 + 0.15 * np.sin(2 * np.pi * (x + y + z))
    state[..., 1] = 0.2 + 0.05 * np.cos(2 * np.pi * y)
    state[..., 2] = -0.1 + 0.05 * np.sin(2 * np.pi * z)
    state[..., 3] = 0.1 + 0.05 * np.cos(2 * np.pi * x)
    state[..., 4] = 1 + 0.1 * np.cos(2 * np.pi * (x - y + z))
    return state


class TestEulerSolver(unittest.TestCase):
    def test_primitive_roundtrip_and_kinetic_energy(self):
        primitive = np.array([2.0, 3.0, -4.0, 2.0, 1.5])
        conservative = primitive_to_conservative(primitive)
        assert_allclose(conservative, [2, 6, -8, 4, 32.75])
        assert_allclose(conservative_to_primitive(conservative), primitive)

    def test_physical_flux_and_equal_state_interface(self):
        state = primitive_to_conservative([2, 3, -4, 2, 1.5])
        expected = (
            [6, 19.5, -24, 12, 102.75],
            [-8, -24, 33.5, -16, -137],
            [4, 12, -16, 9.5, 68.5],
        )
        for axis in range(3):
            assert_allclose(flux(state, axis), expected[axis])
            assert_allclose(rusanov_flux(state, state, axis), expected[axis])

    def test_uniform_state_is_preserved_in_three_dimensions(self):
        state = np.broadcast_to([1.2, 0.4, -0.2, 0.3, 1.1], (7, 6, 5, 5)).copy()
        solver = EulerSolver(boundary=("outflow", "periodic", "outflow"))
        result = solver.solve(state, [0.7, 0.71, 0.75], [0.1, 0.2, 0.15])
        self.assertEqual(result.shape, (3, 7, 6, 5, 5))
        for snapshot in result:
            assert_allclose(snapshot, state, rtol=1e-13, atol=1e-13)

    def test_periodic_conservation_in_three_dimensions(self):
        shape = (12, 10, 8)
        state = smooth_state(shape)
        solver = EulerSolver(boundary=("periodic",) * 3)
        result = solver.solve(state, [0, 0.04, 0.1], [1 / n for n in shape])
        totals = primitive_to_conservative(result).sum(axis=(1, 2, 3))
        for total in totals:
            assert_allclose(total, totals[0], rtol=5e-13, atol=1e-11)
        self.assertGreater(np.max(np.abs(result[-1] - state)), 1e-3)

    def test_planar_data_remains_identical_along_z(self):
        state = smooth_state((10, 8, 1))
        state[..., 3] = 0
        extruded = np.repeat(state, 5, axis=2)
        result = EulerSolver().solve(extruded, [0, 0.05], [0.1, 0.125, 0.2])
        for z_index in range(1, 5):
            assert_allclose(result[:, :, :, z_index], result[:, :, :, 0], atol=1e-14)
        assert_allclose(result[..., 3], 0, atol=1e-14)
        # При Nz=1 ось z исключается из ограничения шага; это допустимый плоский режим.
        planar = EulerSolver().solve(state, [0, 0.05], [0.1, 0.125, 0.2])
        self.assertTrue(np.all(planar[..., (0, 4)] > 0))

    def test_permuting_x_y_and_velocities_preserves_solution(self):
        state = smooth_state((9, 7, 5))
        swapped = state.transpose(1, 0, 2, 3)[..., [0, 2, 1, 3, 4]]
        solver = EulerSolver(boundary=("periodic",) * 3)
        original = solver.solve(state, [0, 0.05], [1 / 9, 1 / 7, 1 / 5])
        changed = solver.solve(swapped, [0, 0.05], [1 / 7, 1 / 9, 1 / 5])
        restored = changed.transpose(0, 2, 1, 3, 4)[..., [0, 2, 1, 3, 4]]
        assert_allclose(restored, original, rtol=1e-12, atol=1e-12)

    def test_density_wave_is_transported_along_z(self):
        z = (np.arange(64) + 0.5) / 64
        state = np.broadcast_to([1, 0, 0, 0.5, 1], (1, 1, 64, 5)).copy()
        state[0, 0, :, 0] = 1 + 0.1 * np.sin(2 * np.pi * z)
        numerical = EulerSolver().solve(state, [0, 0.1], [1, 1, 1 / 64])[-1, 0, 0]
        exact_density = 1 + 0.1 * np.sin(2 * np.pi * (z - 0.5 * 0.1))
        initial_error = np.linalg.norm(state[0, 0, :, 0] - exact_density)
        final_error = np.linalg.norm(numerical[:, 0] - exact_density)
        self.assertLess(final_error, 0.5 * initial_error)
        assert_allclose(numerical[:, 1:4], state[0, 0, :, 1:4], atol=1e-13)
        assert_allclose(numerical[:, 4], 1, atol=1e-13)

    def test_sod_shock_stays_physical_and_moves(self):
        state = np.empty((80, 1, 1, 5))
        state[:40] = [1, 0, 0, 0, 1]
        state[40:] = [0.125, 0, 0, 0, 0.1]
        result = EulerSolver().solve(state, [0, 0.12], [1 / 80, 1, 1])[-1]
        self.assertTrue(np.all(np.isfinite(result)))
        self.assertTrue(np.all(result[..., (0, 4)] > 0))
        self.assertGreater(result[40, 0, 0, 1], 0.1)
        self.assertGreater(result[45, 0, 0, 4], 0.1)
        assert_allclose(result[..., 2:4], 0, atol=1e-14)

    def test_invalid_inputs_fail_explicitly(self):
        state = np.broadcast_to([1, 0, 0, 0, 1], (2, 2, 2, 5)).copy()
        with self.assertRaisesRegex(ValueError, "Давление"):
            primitive_to_conservative([1, 0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "Плотность"):
            conservative_to_primitive([0, 0, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, "Давление"):
            conservative_to_primitive([1, 3, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, "строго возрастать"):
            EulerSolver().solve(state, [0, 0], [1, 1, 1])
        with self.assertRaisesRegex(ValueError, "положительных"):
            EulerSolver().solve(state, [0, 1], [1, 0, 1])
        with self.assertRaisesRegex(ValueError, "cfl"):
            EulerSolver(cfl=1.1).solve(state, [0, 1], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
