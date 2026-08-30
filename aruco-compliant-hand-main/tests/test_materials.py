import math
import unittest

import numpy as np

from flexsense.fem2d import (
    Element,
    Frame,
    _fibre_local_state,
    _section_response,
    deformed_nodes,
    internal_force,
    peak_fibre_strain,
    rectangular_section,
    solve_static,
    tangent_stiffness,
)
from flexsense.materials import (
    TPU95A,
    TPU95A_REFERENCE,
    LinearMaterial,
    Yeoh,
    fit_yeoh,
    neo_hookean,
)


def strip(length=100.0, n_elements=8, youngs=2000.0, thickness=2.0, depth=10.0,
          material=None):
    area, second_moment = rectangular_section(thickness, depth)
    nodes = np.column_stack([np.linspace(0.0, length, n_elements + 1),
                             np.zeros(n_elements + 1)])
    elements = [Element(i, i + 1, youngs, area, second_moment, material)
                for i in range(n_elements)]
    return Frame(nodes, elements)


class MaterialLawTests(unittest.TestCase):
    def test_tangent_is_the_derivative_of_the_stress(self):
        for material in (TPU95A, neo_hookean(30.0), LinearMaterial(2000.0)):
            grid = np.linspace(-0.3, 1.0, 40)
            step = 1e-6
            plus, _ = material.response(grid + step)
            minus, _ = material.response(grid - step)
            _, tangent = material.response(grid)
            np.testing.assert_allclose(tangent, (plus - minus) / (2 * step),
                                       rtol=2e-5, atol=2e-5)

    def test_zero_strain_is_stress_free(self):
        stress, _ = TPU95A.response(np.array([0.0]))
        self.assertAlmostEqual(float(stress[0]), 0.0, places=12)

    def test_neo_hookean_initial_modulus_is_six_c10(self):
        self.assertAlmostEqual(neo_hookean(30.0).initial_modulus, 30.0, places=9)
        self.assertAlmostEqual(Yeoh(c10=5.0, c20=-2.0, c30=1.0).initial_modulus,
                               30.0, places=9)

    def test_tpu_curve_stays_stable_out_to_one_hundred_percent(self):
        """A negative tangent modulus is not a material, it is a bad fit, and
        Newton will happily chase it into nonsense."""
        _, tangent = TPU95A.response(np.linspace(-0.3, 1.0, 400))
        self.assertGreater(float(tangent.min()), 0.0)

    def test_tpu_curve_matches_the_samples_it_was_fitted_to(self):
        strain = np.array([s for s, _ in TPU95A_REFERENCE])
        stress = np.array([v for _, v in TPU95A_REFERENCE])
        predicted, _ = TPU95A.response(strain)
        np.testing.assert_allclose(predicted, stress, atol=0.2)

    def test_tpu_softens_in_tension_and_stiffens_in_compression(self):
        """Both halves matter. The softening is why a Hookean fit is wrong; the
        stiffening is why the error mostly cancels once the load is bending."""
        stress, tangent = TPU95A.response(np.array([0.20]))
        self.assertLess(float(stress[0]), 0.8 * TPU95A.initial_modulus * 0.20)
        self.assertLess(float(tangent[0]), 0.6 * TPU95A.initial_modulus)

        stress, tangent = TPU95A.response(np.array([-0.20]))
        self.assertGreater(abs(float(stress[0])), TPU95A.initial_modulus * 0.20)
        self.assertGreater(float(tangent[0]), TPU95A.initial_modulus)

    def test_fit_recovers_coefficients_it_generated(self):
        truth = Yeoh(c10=4.0, c20=-1.5, c30=0.9)
        strain = np.linspace(0.02, 0.6, 25)
        stress, _ = truth.response(strain)
        fitted = fit_yeoh(strain, stress)
        self.assertAlmostEqual(fitted.c10, truth.c10, places=6)
        self.assertAlmostEqual(fitted.c20, truth.c20, places=6)
        self.assertAlmostEqual(fitted.c30, truth.c30, places=6)

    def test_fit_rejects_unusable_input(self):
        with self.assertRaises(ValueError):
            fit_yeoh([0.1, 0.2], [1.0, 2.0])
        with self.assertRaises(ValueError):
            fit_yeoh([-1.5, 0.2, 0.3], [1.0, 2.0, 3.0])


class FibreIntegrationTests(unittest.TestCase):
    """A linear material must make the fibre element identical to the closed
    form. If it does not, every hyperelastic number downstream is guesswork."""

    def test_internal_force_matches_the_closed_form(self):
        closed = strip()
        fibre = strip(material=LinearMaterial(2000.0))
        rng = np.random.default_rng(4)
        for _ in range(5):
            u = rng.normal(scale=0.3, size=closed.n_dof)
            np.testing.assert_allclose(internal_force(fibre, u),
                                       internal_force(closed, u), rtol=1e-9, atol=1e-9)

    def test_tangent_matches_the_closed_form(self):
        closed = strip()
        fibre = strip(material=LinearMaterial(2000.0))
        rng = np.random.default_rng(5)
        u = rng.normal(scale=0.3, size=closed.n_dof)
        np.testing.assert_allclose(tangent_stiffness(fibre, u),
                                   tangent_stiffness(closed, u), rtol=1e-8, atol=1e-8)

    def test_fibre_element_reproduces_pl3_over_3ei(self):
        length, youngs, thickness, depth = 100.0, 2000.0, 2.0, 10.0
        _, second_moment = rectangular_section(thickness, depth)
        frame = strip(material=LinearMaterial(youngs))
        load = 0.05
        force = np.zeros(frame.n_dof)
        force[3 * (frame.n_nodes - 1) + 1] = load
        result = solve_static(frame, np.array([0, 1, 2]), force)
        self.assertTrue(result.converged)
        tip = deformed_nodes(frame, result.displacement)[-1, 1]
        analytic = load * length ** 3 / (3.0 * youngs * second_moment)
        self.assertAlmostEqual(tip / analytic, 1.0, places=3)

    def test_tangent_is_still_the_derivative_with_a_hyperelastic_law(self):
        frame = strip(n_elements=3, material=TPU95A, thickness=1.5)
        rng = np.random.default_rng(6)
        u = rng.normal(scale=0.05, size=frame.n_dof)
        analytic = tangent_stiffness(frame, u)
        step = 1e-7
        numeric = np.zeros_like(analytic)
        for dof in range(frame.n_dof):
            plus, minus = u.copy(), u.copy()
            plus[dof] += step
            minus[dof] -= step
            numeric[:, dof] = (internal_force(frame, plus)
                               - internal_force(frame, minus)) / (2 * step)
        np.testing.assert_allclose(analytic, numeric, atol=2e-3, rtol=2e-3)

    def _sections(self):
        area, second_moment = rectangular_section(1.5, 15.0)
        hooke = Element(0, 1, 1.0, area, second_moment,
                        LinearMaterial(TPU95A.initial_modulus))
        yeoh = Element(0, 1, 1.0, area, second_moment, TPU95A)
        return hooke, yeoh

    def test_pure_tension_shows_the_full_softening(self):
        hooke, yeoh = self._sections()
        length = 9.0
        for target, ceiling in ((0.10, 0.90), (0.20, 0.78), (0.30, 0.67)):
            linear, _ = _fibre_local_state(hooke, target * length, 0.0, 0.0, length)
            actual, _ = _fibre_local_state(yeoh, target * length, 0.0, 0.0, length)
            self.assertLess(actual[0] / linear[0], ceiling)

    def test_pure_bending_barely_softens_because_compression_stiffens(self):
        """Worth pinning down: a section in bending has fibres on both sides of
        the neutral axis, and for an incompressible law the tension softening
        and the compression stiffening very nearly cancel. The tensile curve
        alone badly overstates how much the finger's response should change."""
        hooke, yeoh = self._sections()
        length = 9.0
        for target, floor in ((0.10, 0.97), (0.22, 0.93)):
            curvature = target / 0.75
            rotation = curvature * length / 2.0
            linear, _ = _fibre_local_state(hooke, 0.0, rotation, -rotation, length)
            actual, _ = _fibre_local_state(yeoh, 0.0, rotation, -rotation, length)
            ratio = actual[1] / linear[1]
            self.assertLess(ratio, 1.0)
            self.assertGreater(ratio, floor)

    def test_bending_shifts_the_neutral_axis_off_mid_thickness(self):
        """A Hookean section keeps S1 at zero by symmetry. A hyperelastic one
        does not, and that coupling is exactly what the closed form cannot
        represent."""
        hooke, yeoh = self._sections()

        def offset(element, curvature):
            _axial, _moment, s0, s1, _s2 = _section_response(element, 0.0, curvature)
            return abs(s1) / s0

        # A Hookean section is symmetric, so the neutral axis never moves.
        self.assertAlmostEqual(offset(hooke, 0.333), 0.0, delta=1e-9)
        # A hyperelastic one walks it towards the stiffer compression side -
        # here to 15% of the half-thickness, which no EI can represent.
        self.assertLess(offset(yeoh, 0.0133), 0.01)
        self.assertGreater(offset(yeoh, 0.333), 0.10)

    def test_rigid_body_motion_is_stress_free_with_fibres(self):
        frame = strip(n_elements=6, material=TPU95A)
        angle = 0.9
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        moved = frame.nodes @ rotation.T + np.array([11.0, -4.0])
        u = np.zeros(frame.n_dof)
        u.reshape(-1, 3)[:, :2] = moved - frame.nodes
        u.reshape(-1, 3)[:, 2] = angle
        self.assertLess(float(np.abs(internal_force(frame, u)).max()), 1e-9)


if __name__ == "__main__":
    unittest.main()
