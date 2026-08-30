import math
import unittest

import numpy as np

from flexsense.fem2d import (
    Element,
    Frame,
    deformed_nodes,
    internal_force,
    rectangular_section,
    solve_static,
    tangent_stiffness,
)


def straight_cantilever(length=100.0, n_elements=20, youngs=2000.0,
                        thickness=2.0, depth=10.0, angle=0.0):
    area, second_moment = rectangular_section(thickness, depth)
    stations = np.linspace(0.0, length, n_elements + 1)
    nodes = np.column_stack([stations * math.cos(angle), stations * math.sin(angle)])
    elements = [Element(i, i + 1, youngs, area, second_moment) for i in range(n_elements)]
    return Frame(nodes, elements), second_moment * youngs


class TangentTests(unittest.TestCase):
    def test_tangent_matches_finite_difference(self):
        frame, _ = straight_cantilever(n_elements=4)
        rng = np.random.default_rng(0)
        u = rng.normal(scale=0.4, size=frame.n_dof)
        analytic = tangent_stiffness(frame, u)
        step = 1e-6
        numeric = np.zeros_like(analytic)
        for dof in range(frame.n_dof):
            plus, minus = u.copy(), u.copy()
            plus[dof] += step
            minus[dof] -= step
            numeric[:, dof] = (internal_force(frame, plus) - internal_force(frame, minus)) / (2 * step)
        np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-4)

    def test_rigid_body_motion_is_stress_free(self):
        frame, _ = straight_cantilever(n_elements=6)
        angle = 1.1
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        shift = np.array([37.0, -12.0])
        moved = frame.nodes @ rotation.T + shift
        u = np.zeros(frame.n_dof)
        u.reshape(-1, 3)[:, :2] = moved - frame.nodes
        u.reshape(-1, 3)[:, 2] = angle
        self.assertLess(float(np.abs(internal_force(frame, u)).max()), 1e-9)


class CantileverTests(unittest.TestCase):
    def test_small_deflection_matches_pl3_over_3ei(self):
        length = 100.0
        frame, flexural = straight_cantilever(length=length, n_elements=20)
        load = 0.05
        force = np.zeros(frame.n_dof)
        force[3 * (frame.n_nodes - 1) + 1] = load
        result = solve_static(frame, fixed_dof=np.array([0, 1, 2]), external_force=force)
        self.assertTrue(result.converged)
        tip = deformed_nodes(frame, result.displacement)[-1, 1]
        self.assertAlmostEqual(tip / (load * length ** 3 / (3.0 * flexural)), 1.0, places=3)

    def test_end_moment_rolls_beam_into_a_closed_circle(self):
        """M = 2*pi*EI/L bends a cantilever into exactly one full circle.

        An exact large-rotation benchmark: the tip must come back to the root.
        """
        length = 100.0
        frame, flexural = straight_cantilever(length=length, n_elements=60)
        target = 2.0 * math.pi * flexural / length
        u = None
        for fraction in np.linspace(0.05, 1.0, 20):
            force = np.zeros(frame.n_dof)
            force[3 * (frame.n_nodes - 1) + 2] = fraction * target
            result = solve_static(frame, np.array([0, 1, 2]), force, u0=u)
            self.assertTrue(result.converged)
            u = result.displacement
        tip = deformed_nodes(frame, u)[-1]
        self.assertLess(float(np.hypot(*tip)), length * 2e-3)
        self.assertAlmostEqual(u[-1] / (2.0 * math.pi), 1.0, places=3)

    def test_large_tip_load_matches_elastica(self):
        """Nondimensional tip load PL^2/EI = 10 against the elastica solution.

        Reference values come from integrating the exact elastica for a tip
        point load: vertical tip travel 0.8156 L, horizontal draw-in 0.5537 L.
        """
        length = 100.0
        frame, flexural = straight_cantilever(length=length, n_elements=80)
        total = 10.0 * flexural / length ** 2
        u = None
        for fraction in np.linspace(0.05, 1.0, 30):
            force = np.zeros(frame.n_dof)
            force[3 * (frame.n_nodes - 1) + 1] = fraction * total
            result = solve_static(frame, np.array([0, 1, 2]), force, u0=u)
            self.assertTrue(result.converged)
            u = result.displacement
        tip = deformed_nodes(frame, u)[-1]
        self.assertAlmostEqual(tip[1] / length, 0.8156, places=2)
        self.assertAlmostEqual((length - tip[0]) / length, 0.5537, places=2)

    def test_orientation_does_not_change_the_answer(self):
        load = 0.05
        length = 100.0
        tips = []
        for angle in (0.0, 0.9):
            frame, _ = straight_cantilever(length=length, n_elements=20, angle=angle)
            direction = np.array([-math.sin(angle), math.cos(angle)])
            force = np.zeros(frame.n_dof)
            force[3 * (frame.n_nodes - 1):3 * (frame.n_nodes - 1) + 2] = load * direction
            result = solve_static(frame, np.array([0, 1, 2]), force)
            tip = deformed_nodes(frame, result.displacement)[-1] - frame.nodes[-1]
            tips.append(float(tip @ direction))
        self.assertAlmostEqual(tips[0], tips[1], places=6)


if __name__ == "__main__":
    unittest.main()
