import math
import unittest

import numpy as np

from flexsense.fem2d import deformed_nodes, solve_static
from flexsense.finray_geometry import FingerSpec, build_finger, outline
from flexsense.finray_sim import (
    CylinderObstacle,
    FlatObstacle,
    contact_patch,
    press,
    summarise,
    tip_pose,
)


class ReconstructionTests(unittest.TestCase):
    """The drawing gives more dimensions than the model needs. The spare ones
    are the check that the reconstruction is right rather than merely fitted."""

    def setUp(self):
        self.spec = FingerSpec()

    def test_base_clear_height_reproduces_the_dimension_it_was_solved_from(self):
        self.assertAlmostEqual(self.spec.clear_height(self.spec.front_length), 26.4, places=6)

    def test_back_member_length_falls_out_at_91mm(self):
        """Not an input. The taper is solved from the 26.4 mm base height and
        the 86 mm front member; the 91 mm back member then has to agree."""
        corner = self.spec.base_outer_corner
        self.assertAlmostEqual(float(np.hypot(*corner)), 91.0, places=6)

    def test_base_face_is_square_to_the_contact_face(self):
        corner = self.spec.base_outer_corner
        angle = math.degrees(math.atan2(corner[1], corner[0] - self.spec.front_length))
        self.assertAlmostEqual(angle, 90.0, delta=0.5)

    def test_taper_angle_is_about_nineteen_degrees(self):
        self.assertAlmostEqual(math.degrees(self.spec.taper_angle), 18.93, delta=0.05)

    def test_rib_pitch_is_regular(self):
        stations = np.asarray(self.spec.rib_stations)
        pitch = np.diff(stations)[:4]
        self.assertLess(float(pitch.std()), 0.1)
        self.assertAlmostEqual(float(pitch.mean()), 9.1, delta=0.2)

    def test_every_dimensioned_height_round_trips(self):
        for height in self.spec.window_heights:
            station = self.spec.station_of_height(height)
            self.assertAlmostEqual(self.spec.clear_height(station), height, places=8)

    def test_cavity_starts_where_the_walls_separate(self):
        self.assertAlmostEqual(self.spec.clear_height(self.spec.cavity_start), 0.0, places=8)
        self.assertGreater(self.spec.rib_stations[0], self.spec.cavity_start)

    def test_outline_closes_on_the_dimensioned_corners(self):
        points = outline(self.spec)
        np.testing.assert_allclose(points[0], points[-1], atol=1e-12)
        self.assertAlmostEqual(float(np.hypot(*(points[1] - points[0]))), 86.0, places=6)


class MeshTests(unittest.TestCase):
    def test_members_share_one_node_at_the_tip(self):
        model = build_finger()
        self.assertEqual(model.front_nodes[0], model.back_nodes[0])
        self.assertEqual(model.tip_node, model.front_nodes[0])

    def test_base_block_is_clamped_and_the_rest_is_free(self):
        model = build_finger()
        clamped = np.unique(np.asarray(model.fixed_dof) // 3)
        stations = model.frame.nodes[clamped, 0]
        self.assertTrue((stations >= model.spec.rib_stations[-1] - 1e-9).all())
        self.assertLess(len(clamped), model.frame.n_nodes // 4)

    def test_unloaded_finger_does_not_move(self):
        model = build_finger()
        result = solve_static(model.frame, model.fixed_dof)
        self.assertTrue(result.converged)
        self.assertLess(float(np.abs(result.displacement).max()), 1e-12)

    def test_stiffness_scales_linearly_with_modulus_and_depth(self):
        def probe(**kwargs):
            model = build_finger(FingerSpec(**kwargs))
            node = model.node_at_front_station(50.0)
            force = np.zeros(model.frame.n_dof)
            force[3 * node + 1] = 0.2
            result = solve_static(model.frame, model.fixed_dof, force)
            return result.displacement[3 * node + 1]

        base = probe(youngs_modulus=1000.0, depth=15.0)
        self.assertAlmostEqual(probe(youngs_modulus=2000.0, depth=15.0) / base, 0.5, places=3)
        self.assertAlmostEqual(probe(youngs_modulus=1000.0, depth=30.0) / base, 0.5, places=3)


class ContactTests(unittest.TestCase):
    def _soft(self):
        return build_finger(FingerSpec(youngs_modulus=30.0, depth=15.0))

    def test_no_contact_means_no_force(self):
        model = self._soft()
        obstacle = CylinderObstacle(radius=10.0, station=45.0, clearance=5.0)
        result = press(model, obstacle, max_advance=1.0, n_steps=4)
        self.assertAlmostEqual(float(np.linalg.norm(result.steps[-1].total_force)), 0.0)

    def test_force_is_proportional_to_modulus_at_fixed_indentation(self):
        obstacle = CylinderObstacle(radius=15.0, station=55.0)
        forces = []
        for modulus in (30.0, 120.0):
            model = build_finger(FingerSpec(youngs_modulus=modulus, depth=15.0))
            result = press(model, obstacle, max_advance=4.0, n_steps=10)
            self.assertTrue(result.steps[-1].converged)
            forces.append(result.steps[-1].total_force[1])
        self.assertAlmostEqual(forces[1] / forces[0], 4.0, delta=0.05)

    def test_deep_indentation_still_converges(self):
        model = self._soft()
        result = press(model, CylinderObstacle(15.0, 55.0), max_advance=16.0, n_steps=20)
        self.assertTrue(result.steps[-1].converged)
        self.assertGreater(result.steps[-1].advance, 15.9)

    def test_the_face_wraps_a_cylinder_instead_of_touching_one_point(self):
        """The whole reason for the rib geometry: contact has to spread."""
        model = self._soft()
        result = press(model, CylinderObstacle(15.0, 55.0), max_advance=16.0, n_steps=20)
        shallow = contact_patch(model, result.steps[1])[0]
        deep = contact_patch(model, result.steps[-1])[0]
        self.assertLess(shallow, 3.0)
        self.assertGreater(deep, 8.0)

    def test_a_flat_object_flattens_the_contact_face(self):
        model = self._soft()
        result = press(model, FlatObstacle(span=(8.0, 58.0)), max_advance=8.0, n_steps=16)
        self.assertTrue(result.completed)
        self.assertGreater(contact_patch(model, result.steps[-1])[0], 30.0)

    def test_an_incomplete_sweep_reports_only_states_that_solved(self):
        """Flat contact against a rigid plate stops converging somewhere past
        10 mm of indentation - far beyond any useful grip. What matters is that
        the result says so instead of handing back a diverged step."""
        model = self._soft()
        result = press(model, FlatObstacle(span=(8.0, 58.0)), max_advance=30.0, n_steps=20)
        self.assertFalse(result.completed)
        self.assertGreater(result.reached_mm, 8.0)
        self.assertTrue(all(step.converged for step in result.steps))

    def test_clamped_nodes_never_carry_contact_force(self):
        model = self._soft()
        clamped = set(np.asarray(model.fixed_dof) // 3)
        result = press(model, FlatObstacle(span=(8.0, 120.0)), max_advance=6.0, n_steps=8)
        for step in result.steps:
            self.assertFalse(clamped & set(step.contact_nodes.tolist()))

    def test_summary_rows_line_up_with_the_steps(self):
        model = self._soft()
        result = press(model, CylinderObstacle(15.0, 55.0), max_advance=4.0, n_steps=6)
        rows = summarise(result)
        self.assertEqual(len(rows), len(result.steps))
        self.assertGreater(rows[-1]["normal_force_n"], rows[0]["normal_force_n"])

    def test_tip_pose_matches_the_displacement_vector(self):
        model = self._soft()
        result = press(model, CylinderObstacle(15.0, 55.0), max_advance=6.0, n_steps=8)
        u = result.steps[-1].displacement
        position, rotation = tip_pose(model, u)
        expected = deformed_nodes(model.frame, u)[model.tip_node]
        np.testing.assert_allclose(position, expected)
        self.assertAlmostEqual(rotation, u[3 * model.tip_node + 2])


if __name__ == "__main__":
    unittest.main()
