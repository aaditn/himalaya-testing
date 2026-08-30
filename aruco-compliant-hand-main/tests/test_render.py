"""Tests for the CAD mesh, the spine, and the live display.

These deliberately run the code rather than inspect it. The bugs this project
has actually hit - a shadowed variable, a half-applied patch, a CLI argument
that was never added - all survived reading and died the first time something
executed the path end to end.
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from flexsense.fingermesh import FingerMesh, _extrusion_axis, load_stl, split_shells
from flexsense.grip import (
    GripThresholds,
    bend_deg_for_tip_travel,
    load_finger_mesh,
    run_grip_live,
    tip_travel_for_bend_deg,
)
from flexsense.handconfig import load_hand, to_grip_thresholds, to_sim_rig
from flexsense.handtools import _intrinsics
from flexsense.hud import THEME, TextLayer, draw_gauge, draw_trace
from flexsense.render3d import OrthoProjector, draw_mesh, draw_polyline_3d
from flexsense.simrig import render as render_rig
from flexsense.smoothing import MarkerPoseTracker, OneEuroFilter
from flexsense.spine import FingerSpine, Station, frame_from_marker

HAND = Path(__file__).resolve().parents[1] / "config" / "hand.yaml"
ASSET = Path(__file__).resolve().parents[1] / "assets" / "finger_v2.stl"


def _prism(length=100.0, width=30.0, thickness=8.0, taper=0.15):
    """A tapered slab: same family as the real finger, small enough to reason about."""
    quads = []
    steps = 12
    for i in range(steps):
        y0, y1 = length * i / steps, length * (i + 1) / steps
        w0 = width * (1 - (1 - taper) * y0 / length) / 2
        w1 = width * (1 - (1 - taper) * y1 / length) / 2
        for z in (-thickness / 2, thickness / 2):
            quads.append(([-w0, y0, z], [w0, y0, z], [w1, y1, z], [-w1, y1, z]))
        quads.append(([w0, y0, -thickness / 2], [w0, y0, thickness / 2],
                      [w1, y1, thickness / 2], [w1, y1, -thickness / 2]))
        quads.append(([-w0, y0, -thickness / 2], [-w0, y0, thickness / 2],
                      [-w1, y1, thickness / 2], [-w1, y1, -thickness / 2]))
    triangles = []
    for a, b, c, d in quads:
        triangles.append([a, b, c])
        triangles.append([a, c, d])
    return np.asarray(triangles, float)


def _write_binary_stl(path, triangles):
    with open(path, "wb") as handle:
        handle.write(b"\0" * 80)
        handle.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            handle.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for vertex in triangle:
                handle.write(struct.pack("<3f", *vertex))
            handle.write(b"\0\0")


def _write_ascii_stl(path, triangles):
    lines = ["solid test"]
    for triangle in triangles:
        lines.append(" facet normal 0 0 0")
        lines.append("  outer loop")
        for vertex in triangle:
            lines.append("   vertex {:.6f} {:.6f} {:.6f}".format(*vertex))
        lines.append("  endloop")
        lines.append(" endfacet")
    lines.append("endsolid test")
    Path(path).write_text("\n".join(lines))


class StlLoading(unittest.TestCase):
    def test_binary_and_ascii_agree(self):
        triangles = _prism()
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / "a.stl"
            ascii_path = Path(folder) / "b.stl"
            _write_binary_stl(binary, triangles)
            _write_ascii_stl(ascii_path, triangles)
            first = load_stl(binary)
            second = load_stl(ascii_path)
        self.assertEqual(first.shape, triangles.shape)
        np.testing.assert_allclose(first, second, atol=1e-3)
        np.testing.assert_allclose(first, triangles, atol=1e-3)

    def test_rejects_rubbish(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "junk.stl"
            path.write_bytes(b"not an stl at all")
            with self.assertRaises(ValueError):
                load_stl(path)

    def test_splits_shells(self):
        triangles = np.vstack([_prism(), _prism() + np.array([500.0, 0.0, 0.0])])
        shells = split_shells(triangles)
        self.assertEqual(len(shells), 2)
        self.assertEqual(sum(len(s) for s in shells), len(triangles))


class Canonicalisation(unittest.TestCase):
    def test_finds_the_extrusion_axis(self):
        axis = _extrusion_axis(_prism())
        self.assertIsNotNone(axis)
        # The slab is 8 mm thick along z; the axis may point either way.
        self.assertAlmostEqual(abs(float(axis[2])), 1.0, places=6)

    def test_no_axis_for_a_non_prism(self):
        # A tetrahedron has no pair of parallel faces holding every vertex.
        triangles = np.array([
            [[0, 0, 0], [10, 0, 0], [0, 10, 0]],
            [[0, 0, 0], [10, 0, 0], [0, 0, 10]],
            [[0, 0, 0], [0, 10, 0], [0, 0, 10]],
            [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        ], float)
        self.assertIsNone(_extrusion_axis(triangles))

    def test_canonical_frame_from_an_arbitrary_pose(self):
        """A rotated, translated export must land in the same canonical frame."""
        base = _prism(length=100.0, width=30.0, thickness=8.0)
        angle = 0.7
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1.0]])
        tilted = np.array([
            [1.0, 0, 0],
            [0, np.cos(0.4), -np.sin(0.4)],
            [0, np.sin(0.4), np.cos(0.4)]])
        moved = base @ (tilted @ rotation).T + np.array([37.0, -12.0, 90.0])
        with tempfile.TemporaryDirectory() as folder:
            for name, triangles in (("plain.stl", base), ("moved.stl", moved)):
                _write_binary_stl(Path(folder) / name, triangles)
            plain = FingerMesh.from_stl(Path(folder) / "plain.stl", units="mm")
            shifted = FingerMesh.from_stl(Path(folder) / "moved.stl", units="mm")

        for mesh in (plain, shifted):
            self.assertAlmostEqual(mesh.length_mm, 100.0, delta=0.6)
            self.assertAlmostEqual(mesh.width_mm, 8.0, delta=0.3)
            self.assertAlmostEqual(float(mesh.vertices[:, 1].min()), 0.0, places=5)
        self.assertAlmostEqual(plain.length_mm, shifted.length_mm, delta=0.6)

    def test_root_is_the_thick_end(self):
        mesh = FingerMesh._canonicalise(_prism(), "test", "", 1)
        depth = mesh.vertices[:, 0]
        along = mesh.vertices[:, 1]
        near_root = np.ptp(depth[along < 0.2 * mesh.length_mm])
        near_tip = np.ptp(depth[along > 0.8 * mesh.length_mm])
        self.assertGreater(near_root, near_tip * 2)

    def test_units_auto_detection(self):
        inches = _prism() / 25.4
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "inch.stl"
            _write_binary_stl(path, inches)
            mesh = FingerMesh.from_stl(path)
            self.assertIn("inch", mesh.scale_note)
            self.assertAlmostEqual(mesh.length_mm, 100.0, delta=0.6)
            forced = FingerMesh.from_stl(path, units="mm")
            self.assertAlmostEqual(forced.length_mm, 100.0 / 25.4, delta=0.05)

    def test_rescale(self):
        mesh = FingerMesh._canonicalise(_prism(), "test", "", 1)
        rescaled = mesh.scaled_to_length(50.0)
        self.assertAlmostEqual(rescaled.length_mm, 50.0, places=6)
        self.assertAlmostEqual(rescaled.width_mm, mesh.width_mm / 2, delta=0.01)

    def test_roll_swaps_the_cross_axes(self):
        """A quarter turn about the long axis moves the tags to the side wall."""
        flat = FingerMesh._canonicalise(_prism(width=30.0, thickness=8.0), "t", "", 1)
        turned = flat.rolled(-90.0)
        self.assertAlmostEqual(float(np.ptp(turned.vertices[:, 0])), 8.0, delta=0.2)
        self.assertAlmostEqual(float(np.ptp(turned.vertices[:, 2])), 30.0, delta=0.2)
        self.assertAlmostEqual(turned.length_mm, flat.length_mm, places=6)
        # A quarter turn each way must not be the same mesh.
        self.assertGreater(float(np.abs(
            flat.rolled(90.0).vertices - turned.vertices).max()), 1.0)

    def test_roll_is_reversible(self):
        flat = FingerMesh._canonicalise(_prism(), "t", "", 1)
        back = flat.rolled(-90.0).rolled(90.0)
        np.testing.assert_allclose(back.vertices, flat.vertices, atol=1e-9)

    def test_face_on_spine_levels_and_lands_the_tagged_surface(self):
        """The tags are on the outside of the finger, so the surface - not the
        mid-plane - is what the spine has to follow."""
        mesh = FingerMesh._canonicalise(_prism(), "t", "", 1).rolled(-90.0)
        placed = mesh.with_face_on_spine()
        heights = []
        for y in np.linspace(2.0, placed.length_mm - 2.0, 12):
            band = placed.vertices[np.abs(placed.vertices[:, 1] - y) < 10.0]
            if len(band):
                heights.append(float(band[:, 2].max()))
        self.assertLess(float(np.max(np.abs(heights))), 2.5)
        # And the body must sit behind the tags, not straddle them.
        self.assertLess(float(placed.vertices[:, 2].mean()), 0.0)

    @unittest.skipUnless(ASSET.exists(), "shipped finger CAD is missing")
    def test_shipped_finger(self):
        mesh = FingerMesh.from_stl(ASSET)
        self.assertGreater(mesh.length_mm, 60.0)
        self.assertLess(mesh.length_mm, 200.0)
        self.assertAlmostEqual(mesh.width_mm, 20.0, delta=0.5)


def _arc_station(y, total_deg, length, measured=True):
    k = np.radians(total_deg) / length
    if abs(k) < 1e-9:
        position, angle = np.array([0.0, y, 0.0]), 0.0
    else:
        angle = k * y
        position = np.array([0.0, np.sin(angle) / k, (1 - np.cos(angle)) / k])
    rotation = np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    return Station(y, position, rotation, measured=measured)


class Spine(unittest.TestCase):
    LENGTH = 105.0

    def _spine(self, total_deg, stations=(0.0, 24.0, 58.0)):
        return FingerSpine.from_stations(
            [_arc_station(y, total_deg, self.LENGTH, measured=y > 0) for y in stations],
            self.LENGTH)

    def test_cubic_is_exact_between_the_tags(self):
        """A cantilever's deflection curve is a cubic, so the interpolation is
        not an approximation inside the measured span."""
        for total in (5.0, 20.0, 40.0, -30.0):
            spine = self._spine(total)
            for y in np.linspace(0.0, 58.0, 40):
                expected = _arc_station(y, total, self.LENGTH).position
                got = spine.frames_at(np.array([y]))[0][0]
                self.assertLess(float(np.linalg.norm(got - expected)), 0.05,
                                f"{total} deg at {y:.0f} mm")

    def test_extrapolation_is_straight_not_curled(self):
        spine = self._spine(35.0)
        tail = spine.sample(200)[-40:]
        steps = np.diff(tail, axis=0)
        directions = steps / np.linalg.norm(steps, axis=1, keepdims=True)
        spread = float(np.max(np.linalg.norm(directions - directions[0], axis=1)))
        self.assertLess(spread, 1e-6)

    def test_further_tags_shrink_tip_error(self):
        """Moving the outer tag toward the tip is the direct fix for tip error."""
        def tip_error(stations):
            spine = self._spine(40.0, stations)
            expected = _arc_station(self.LENGTH, 40.0, self.LENGTH).position
            return float(np.linalg.norm(spine.sample(200)[-1] - expected))

        near = tip_error((0.0, 24.0, 58.0))
        far = tip_error((0.0, 30.0, 85.0))
        self.assertGreater(near, far * 2.0)

    def test_frames_match_the_measured_stations(self):
        spine = self._spine(25.0)
        for y in (24.0, 58.0):
            expected = _arc_station(y, 25.0, self.LENGTH).rotation
            _origin, rotation = spine.frames_at(np.array([y]))
            for column in range(3):
                self.assertGreater(float(rotation[0][:, column] @ expected[:, column]),
                                   0.995, f"axis {column} at {y} mm")

    def test_rotations_stay_orthonormal(self):
        spine = self._spine(30.0)
        _origins, rotations = spine.frames_at(np.linspace(0, self.LENGTH, 25))
        for rotation in rotations:
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            self.assertGreater(float(np.linalg.det(rotation)), 0.99)

    def test_two_stations_is_enough(self):
        spine = FingerSpine.from_stations(
            [_arc_station(0.0, 10.0, self.LENGTH, measured=False),
             _arc_station(24.0, 10.0, self.LENGTH)], self.LENGTH)
        self.assertEqual(len(spine.sample(10)), 10)

    def test_one_station_is_refused(self):
        with self.assertRaises(ValueError):
            FingerSpine.from_stations([_arc_station(24.0, 0.0, self.LENGTH)],
                                      self.LENGTH)


class MarkerFrames(unittest.TestCase):
    def test_along_axis_is_found_whatever_way_up_the_tag_is(self):
        """A tag stuck on rotated by any multiple of 90 degrees must still give
        the same finger frame, because its own axes carry no such meaning."""
        along = np.array([0.0, 1.0, 0.0])
        base = np.eye(3)
        frames = []
        for quarter in range(4):
            angle = quarter * np.pi / 2
            spin = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0],
                             [0, 0, 1.0]])
            frames.append(frame_from_marker(base @ spin, along))
        for frame in frames:
            np.testing.assert_allclose(frame[:, 1], along, atol=1e-9)
            np.testing.assert_allclose(frame[:, 2], base[:, 2], atol=1e-9)
            np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-9)


class Filtering(unittest.TestCase):
    def test_one_euro_settles_on_a_constant(self):
        filt = OneEuroFilter()
        rng = np.random.default_rng(3)
        target = np.array([10.0, -4.0, 90.0])
        raw, smoothed = [], []
        for _ in range(300):
            sample = target + rng.normal(0, 0.4, 3)
            raw.append(sample)
            smoothed.append(filt(sample, 1 / 30).copy())
        raw_spread = float(np.std(np.asarray(raw[100:]), axis=0).mean())
        smooth_spread = float(np.std(np.asarray(smoothed[100:]), axis=0).mean())
        self.assertLess(smooth_spread, raw_spread * 0.6)
        self.assertLess(float(np.linalg.norm(smoothed[-1] - target)), 0.5)

    def test_one_euro_tracks_a_ramp(self):
        filt = OneEuroFilter()
        for step in range(120):
            value = filt(np.array([step * 2.0]), 1 / 30)
        self.assertLess(abs(float(value[0]) - 119 * 2.0), 8.0)

    def test_tracker_rejects_a_flipped_pose(self):
        """The ambiguous solution must lose to continuity, not to reprojection."""
        from flexsense.pose3d import square_object_points
        import cv2

        camera_matrix = np.array([[730.0, 0, 640], [0, 726.0, 360], [0, 0, 1.0]])
        dist = np.zeros(5)
        tracker = MarkerPoseTracker(18.0, camera_matrix, dist, rotation_gain=1.0)

        angle = np.radians(9.0)
        rotation = np.array([[1, 0, 0],
                             [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]])
        translation = np.array([0.0, 0.0, 150.0])
        rvec, _ = cv2.Rodrigues(rotation)
        corners, _ = cv2.projectPoints(square_object_points(18.0), rvec, translation,
                                       camera_matrix, dist)
        corners = corners.reshape(4, 2)

        seen = [tracker.solve(1, corners, 1 / 30) for _ in range(12)]
        self.assertTrue(all(pose is not None for pose in seen))
        normals = np.array([pose.rotation[:, 2] for pose in seen])
        # Every frame must agree on which way the tag faces.
        self.assertGreater(float(np.min(normals @ normals[0])), 0.99)


class Rendering(unittest.TestCase):
    def test_ortho_framing_fits_the_panel(self):
        points = np.random.default_rng(0).normal(0, 20, (60, 3))
        projector = OrthoProjector.framing(points, (1, 0, 0), (0, -1, 0), (160, 200),
                                           margin=10)
        pixels, _depth = projector.project(points)
        self.assertGreaterEqual(float(pixels[:, 0].min()), 5.0)
        self.assertLessEqual(float(pixels[:, 0].max()), 155.0)
        self.assertGreaterEqual(float(pixels[:, 1].min()), 5.0)
        self.assertLessEqual(float(pixels[:, 1].max()), 195.0)

    def test_draw_mesh_paints_something(self):
        mesh = FingerMesh._canonicalise(_prism(), "test", "", 1)
        canvas = np.zeros((200, 160, 3), np.uint8)
        projector = OrthoProjector.framing(mesh.vertices, (0, 0, 1), (0, -1, 0),
                                           (160, 200))
        draw_mesh(canvas, mesh.triangles, projector, (60, 200, 60))
        self.assertGreater(int((canvas > 0).sum()), 2000)

    def test_draw_mesh_survives_degenerate_input(self):
        canvas = np.zeros((60, 60, 3), np.uint8)
        projector = OrthoProjector((0, 0, 1), (0, -1, 0), np.zeros(3), 1.0,
                                   np.array([30.0, 30.0]))
        draw_mesh(canvas, np.zeros((0, 3, 3)), projector, (10, 10, 10))
        draw_mesh(canvas, np.full((2, 3, 3), np.nan), projector, (10, 10, 10))
        draw_polyline_3d(canvas, np.full((4, 3), np.nan), projector, (10, 10, 10))
        self.assertEqual(int(canvas.sum()), 0)

    def test_deformed_mesh_keeps_its_length(self):
        """Skinning must bend the finger, not stretch it."""
        mesh = FingerMesh._canonicalise(_prism(length=105.0), "test", "", 1)
        straight = FingerSpine.from_stations(
            [_arc_station(y, 0.0, 105.0, measured=y > 0) for y in (0, 24, 58)], 105.0)
        bent = FingerSpine.from_stations(
            [_arc_station(y, 30.0, 105.0, measured=y > 0) for y in (0, 24, 58)], 105.0)
        for spine in (straight, bent):
            points = mesh.deform(spine).reshape(-1, 3)
            self.assertAlmostEqual(float(np.ptp(np.linalg.norm(
                points - points.mean(0), axis=1))), float(np.ptp(np.linalg.norm(
                    mesh.vertices - mesh.vertices.mean(0), axis=1))), delta=6.0)


class Hud(unittest.TestCase):
    def test_gauge_and_trace_handle_awkward_values(self):
        canvas = np.zeros((80, 300, 3), np.uint8)
        for value in (float("nan"), 0.0, 4.0, -4.0, 1e6, -1e6):
            draw_gauge(canvas, (10, 10, 280, 8), value, 6.0, 4.0, 15.0)
        draw_trace(canvas, (10, 30, 280, 26), [], 15.0, THEME.good)
        draw_trace(canvas, (10, 30, 280, 26), [None, None], 15.0, THEME.good)
        draw_trace(canvas, (10, 30, 280, 26), [1.0, None, -400.0, 3.0], 15.0, THEME.good)

    def test_text_layer_writes_pixels(self):
        for available in (True, False):
            layer = TextLayer.create()
            layer.available = layer.available and available
            canvas = np.zeros((60, 300, 3), np.uint8)
            layer.add("wrapping 12.3", (10, 10), 16, (255, 255, 255))
            layer.flush(canvas)
            self.assertGreater(int((canvas > 0).sum()), 20,
                               f"nothing drawn with available={available}")

    def test_measure_is_positive(self):
        layer = TextLayer.create()
        width, height = layer.measure("backbending", 14)
        self.assertGreater(width, 10)
        self.assertGreater(height, 4)


class LiveDisplay(unittest.TestCase):
    """Run the real live loop over simulated frames, headlessly."""

    @classmethod
    def setUpClass(cls):
        cls.hand = load_hand(HAND)
        cls.camera_matrix, cls.dist, _note = _intrinsics(cls.hand)
        cls.rig = to_sim_rig(cls.hand)
        cls.pose = cls.hand.camera.pose()
        cls.size = (cls.hand.camera.width, cls.hand.camera.height)

    def _script(self):
        frames = [{} for _ in range(12)]
        for i in range(14):
            frames.append({"left": 6.0 * i / 13, "middle": 5.0 * i / 13,
                           "right": -6.0 * i / 13})
        return frames

    def _run(self, **kwargs):
        import cv2

        import flexsense.vision as vision

        script = self._script()
        rendered = []

        class FakeCapture:
            def __init__(inner):
                inner.index = 0

            def read(inner):
                if inner.index >= len(script):
                    return False, None
                frame, _ = render_rig(self.rig, script[inner.index],
                                      self.camera_matrix, self.dist, self.size,
                                      self.pose)
                inner.index += 1
                return True, frame

            def release(inner):
                pass

        original_open = vision.open_capture
        original_show = cv2.imshow
        original_wait = cv2.waitKey
        original_destroy = cv2.destroyAllWindows
        vision.open_capture = lambda *a, **k: FakeCapture()
        cv2.imshow = lambda name, image: rendered.append(image.copy())
        cv2.waitKey = lambda delay: -1
        cv2.destroyAllWindows = lambda: None
        try:
            code = run_grip_live(self.hand, 0, display=True,
                                 thresholds=to_grip_thresholds(self.hand),
                                 zero_frames=12, **kwargs)
        finally:
            vision.open_capture = original_open
            cv2.imshow = original_show
            cv2.waitKey = original_wait
            cv2.destroyAllWindows = original_destroy
        return code, rendered

    def test_full_loop_renders_every_frame(self):
        code, rendered = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(len(rendered), len(self._script()))
        from flexsense.gripview import SIDEBAR_WIDTH
        from flexsense.hud import CONTROL_BAR_HEIGHT
        for image in rendered:
            self.assertEqual(image.shape,
                             (self.size[1] + CONTROL_BAR_HEIGHT,
                              self.size[0] + SIDEBAR_WIDTH, 3))
        # The sidebar must actually be drawn on, not left as flat panel colour.
        sidebar = rendered[-1][:self.size[1], self.size[0]:]
        self.assertGreater(len(np.unique(sidebar.reshape(-1, 3), axis=0)), 12)
        # Keyboard help lives in its own high-contrast row below the calibrated
        # camera image, so it remains visible without covering marker pixels.
        control_bar = rendered[-1][self.size[1]:]
        self.assertGreater(len(np.unique(control_bar.reshape(-1, 3), axis=0)), 6)

    def test_plain_mode_still_runs(self):
        code, rendered = self._run(use_mesh=False)
        self.assertEqual(code, 0)
        for image in rendered:
            self.assertEqual(image.shape[1], self.size[0])

    def test_shipped_config_loads_its_mesh(self):
        mesh = load_finger_mesh(self.hand)
        self.assertIsNotNone(mesh)
        self.assertGreater(mesh.length_mm, 60.0)

    def test_shipped_mesh_is_rolled_onto_its_side_wall(self):
        """Tags are on the narrow 20 mm face, so that is what the camera sees
        across the finger; the wedge depth points away from it."""
        mesh = load_finger_mesh(self.hand)
        lateral = float(np.ptp(mesh.vertices[:, 0]))
        depth = float(np.ptp(mesh.vertices[:, 2]))
        self.assertAlmostEqual(lateral, 20.0, delta=0.5)
        self.assertGreater(depth, lateral)
        # Constant across the finger: the sweep runs this way, so it cannot taper.
        for y in (10.0, 40.0, 80.0):
            band = mesh.vertices[np.abs(mesh.vertices[:, 1] - y) < 12.0]
            if len(band):
                self.assertAlmostEqual(float(np.ptp(band[:, 0])), 20.0, delta=0.5)

    def test_missing_mesh_degrades_instead_of_raising(self):
        from dataclasses import replace
        broken = replace(self.hand,
                         mesh=replace(self.hand.mesh, path="does/not/exist.stl"))
        self.assertIsNone(load_finger_mesh(broken))
        with self.assertRaises(OSError):
            load_finger_mesh(broken, strict=True)


class Thresholds(unittest.TestCase):
    def test_shipped_config_keeps_the_calibrated_sign(self):
        thresholds = to_grip_thresholds(load_hand(HAND))
        self.assertIsInstance(thresholds, GripThresholds)
        self.assertEqual(thresholds.wrap_sign, -1.0)

    def test_bend_and_tip_travel_invert_each_other(self):
        for travel in (2.0, 8.0, 14.0, 30.0, 60.0):
            degrees = bend_deg_for_tip_travel(travel, 24.0, 58.0, 104.8)
            self.assertAlmostEqual(
                tip_travel_for_bend_deg(degrees, 24.0, 58.0, 104.8), travel, places=3)

    def test_inverse_saturates_rather_than_taking_the_wrong_branch(self):
        """Past a finger length the forward curve turns over; the search must
        stop at the bound instead of converging on the far side of the peak."""
        unreachable = bend_deg_for_tip_travel(104.8, 24.0, 58.0, 104.8) + 5.0
        self.assertAlmostEqual(
            tip_travel_for_bend_deg(unreachable, 24.0, 58.0, 104.8), 104.8, places=3)

    def test_bend_grows_with_travel_and_vanishes_at_rest(self):
        self.assertAlmostEqual(bend_deg_for_tip_travel(0.0, 24.0, 58.0, 104.8), 0.0)
        previous = 0.0
        for travel in (1.0, 5.0, 10.0, 20.0, 40.0):
            value = bend_deg_for_tip_travel(travel, 24.0, 58.0, 104.8)
            self.assertGreater(value, previous)
            previous = value

    def test_wrap_threshold_is_the_intended_14mm_of_tip_travel(self):
        """The threshold is chosen in millimetres of travel; degrees are just how
        it gets stored. This is the assertion that catches an edit to the wrong
        one of the two."""
        hand = load_hand(HAND)
        finger = hand.fingers[0]
        travel = tip_travel_for_bend_deg(
            to_grip_thresholds(hand).wrap_deg,
            finger.stations_mm[0], finger.stations_mm[-1], finger.length_mm)
        self.assertAlmostEqual(travel, 14.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
