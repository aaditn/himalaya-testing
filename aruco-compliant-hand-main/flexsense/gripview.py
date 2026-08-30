"""Compose the live grip display: video, rendered fingers, and readouts.

Layout is a camera pane at native resolution with a fixed sidebar beside it,
rather than panels floating over the video. Overlays that sit on the image have
to fight it for contrast; a sidebar never does, and the video keeps every pixel
the calibration was measured at.

The inset side view is the part that earns its space. From the wrist camera the
fingers point nearly down the lens, so bending is foreshortened to almost
nothing and the display has to *tell* you an angle. Once the poses are solved in
the reference tag's frame, re-rendering them from a viewpoint square to the
bend costs nothing and lets you simply see it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .grip import GripAssessment, GripState, GripThresholds
from .hud import THEME, TextLayer, Theme, blend_rect, draw_gauge, draw_trace, outline_rect
from .render3d import OrthoProjector, PinholeProjector, draw_mesh, draw_polyline_3d
from .vision import require_cv2

SIDEBAR_WIDTH = 330
INSET = (252, 264)

STATE_COLOUR = {
    GripState.WRAPPING.value: THEME.good,
    GripState.BACKBENDING.value: THEME.bad,
    GripState.NEUTRAL.value: THEME.neutral,
    GripState.UNKNOWN.value: THEME.faint,
}
VERDICT_COLOUR = {
    "good": THEME.good,
    "marginal": THEME.warn,
    "bad": THEME.bad,
    "no-contact": THEME.muted,
    "unknown": THEME.faint,
}


@dataclass
class FrameState:
    """Everything the view needs for one frame."""

    assessment: GripAssessment | None = None
    spines: dict = field(default_factory=dict)
    reference: object = None
    tag_pixels: dict = field(default_factory=dict)
    expected_tags: int = 0
    fps: float = 0.0
    zeroing: tuple[int, int] | None = None
    zeroed_seconds: float | None = None
    message: str | None = None
    flips_rejected: int = 0
    calibration: str = ""


class GripView:
    """Renders frames for the live grip tool."""

    def __init__(self, hand_config, mesh, thresholds: GripThresholds,
                 camera_matrix, dist_coeffs, theme: Theme = THEME):
        self.hand = hand_config
        self.mesh = mesh
        self.thresholds = thresholds
        self.camera_matrix = np.asarray(camera_matrix, float)
        self.dist_coeffs = np.asarray(dist_coeffs, float)
        self.theme = theme
        self.full_scale = max(thresholds.wrap_deg * 2.5,
                              thresholds.backbend_deg * 2.5, 12.0)
        self.history: dict[str, deque] = {
            finger.name: deque(maxlen=180) for finger in hand_config.fingers}
        self._inset_view = None
        # Built once: recreating it per frame would reload the system font
        # every frame, which costs more than everything else drawn here.
        self._text = TextLayer.create()

    def reframe(self) -> None:
        """Re-fit the inset view to what the fingers are doing now."""
        self._inset_view = None

    def record(self, assessment: GripAssessment | None) -> None:
        for name, history in self.history.items():
            finger = assessment.fingers.get(name) if assessment else None
            oriented = None
            if finger is not None and finger.signed_bend_deg is not None:
                oriented = finger.signed_bend_deg * self.thresholds.wrap_sign
            history.append(oriented)

    def render(self, frame: np.ndarray, state: FrameState) -> np.ndarray:
        cv2 = require_cv2()
        height, width = frame.shape[:2]
        canvas = np.empty((height, width + SIDEBAR_WIDTH, 3), np.uint8)
        canvas[:, :width] = frame
        canvas[:, width:] = self.theme.panel

        text = self._text
        pane = canvas[:, :width]

        if state.reference is not None and state.spines:
            self._draw_overlay(pane, state)
            self._draw_inset(pane, state, text)
        self._draw_tag_marks(pane, state)
        self._draw_status_strip(pane, state, text)
        self._draw_sidebar(canvas, width, state, text)
        text.flush(canvas)
        return canvas

    # -- camera pane ----------------------------------------------------

    def _draw_overlay(self, pane: np.ndarray, state: FrameState) -> None:
        projector = PinholeProjector(self.camera_matrix, self.dist_coeffs,
                                     state.reference)
        for finger in self.hand.fingers:
            spine = state.spines.get(finger.name)
            if spine is None:
                continue
            colour = self._finger_colour(state, finger.name)
            # A dim outline keeps the silhouette readable whether the video
            # behind it is a dark bench or a bright wall.
            draw_mesh(pane, self.mesh.deform(spine), projector, colour,
                      alpha=0.42, edge_colour=_dim(colour, 0.7))
            draw_polyline_3d(pane, spine.sample(56), projector, colour, 2,
                             dashed_from=spine.measured_fraction())

    def _draw_inset(self, pane: np.ndarray, state: FrameState,
                    text: TextLayer) -> None:
        """Each finger side-on, in its own lane, on a shared scale.

        A single true side view would stack the fingers on top of each other -
        they are separated along the very axis you have to look down to see the
        bend. Splitting them into lanes keeps every finger legible, and one
        shared scale keeps the lanes honestly comparable.
        """
        cv2 = require_cv2()
        width, height = INSET
        x = 18
        y = pane.shape[0] - height - 18
        blend_rect(pane, (x, y, width, height), self.theme.panel_deep, 0.9, radius=8)
        outline_rect(pane, (x, y, width, height), self.theme.hairline, 1, radius=8)
        panel = pane[y:y + height, x:x + width]
        text.add("side view", (x + 12, y + 10), 11, self.theme.faint)

        names = [f.name for f in self.hand.fingers]
        lane = width // max(len(names), 1)
        top, body = 26, height - 46

        if self._inset_view is None:
            rooted = [_rooted(s.sample(24)) for s in state.spines.values()]
            if not rooted:
                return
            self._inset_view = OrthoProjector.framing(
                _padded(np.vstack(rooted)), forward=(1, 0, 0), up=(0, -1, 0),
                size=(lane, body), margin=10)
        projector = self._inset_view

        for index, name in enumerate(names):
            spine = state.spines.get(name)
            left = index * lane
            if index:
                cv2.line(panel, (left, top), (left, top + body), self.theme.hairline, 1)
            if spine is None:
                continue
            lane_view = panel[top:top + body, left:left + lane]
            colour = self._finger_colour(state, name)
            origin = spine.points[0]
            draw_mesh(lane_view, self.mesh.deform(spine) - origin, projector,
                      colour, alpha=0.92, edge_colour=_dim(colour))
            draw_polyline_3d(lane_view, spine.sample(56) - origin, projector,
                             self.theme.text, 1,
                             dashed_from=spine.measured_fraction())
            text.add(name, (x + left + lane // 2, y + height - 17), 11,
                     self._finger_colour(state, name), anchor="ma")

        cv2.line(panel, (10, top + body + 4), (width - 10, top + body + 4),
                 self.theme.hairline, 1)

    def _draw_tag_marks(self, pane: np.ndarray, state: FrameState) -> None:
        """Corner brackets instead of aruco's filled outlines and ID blobs."""
        cv2 = require_cv2()
        reference_ids = set(self.hand.reference.tag_ids)
        for marker_id, quad in state.tag_pixels.items():
            colour = (self.theme.reference if marker_id in reference_ids
                      else self.theme.faint)
            corners = np.asarray(quad, float).reshape(4, 2)
            for index in range(4):
                start = corners[index]
                for other in (corners[(index + 1) % 4], corners[(index - 1) % 4]):
                    end = start + (other - start) * 0.28
                    cv2.line(pane, tuple(start.astype(int)), tuple(end.astype(int)),
                             colour, 1, cv2.LINE_AA)

    def _draw_status_strip(self, pane: np.ndarray, state: FrameState,
                           text: TextLayer) -> None:
        blend_rect(pane, (0, 0, pane.shape[1], 44), self.theme.backdrop, 0.72)
        if state.message:
            text.add(state.message, (18, 13), 17, self.theme.warn, bold=True)
        elif state.zeroing:
            done, total = state.zeroing
            text.add(f"zeroing {done}/{total} — keep the fingers unloaded",
                     (18, 13), 17, self.theme.warn, bold=True)
        else:
            text.add("tracking", (18, 14), 15, self.theme.muted)
        text.add(f"{state.fps:4.1f} fps", (pane.shape[1] - 18, 14), 14,
                 self.theme.muted, mono=True, anchor="ra")

    # -- sidebar --------------------------------------------------------

    def _draw_sidebar(self, canvas: np.ndarray, left: int, state: FrameState,
                      text: TextLayer) -> None:
        cv2 = require_cv2()
        x = left + 22
        inner = SIDEBAR_WIDTH - 44
        assessment = state.assessment

        verdict = assessment.verdict if assessment else "—"
        colour = VERDICT_COLOUR.get(verdict, self.theme.faint)
        chip_width = max(74, text.measure(verdict, 17)[0] + 34)
        blend_rect(canvas, (x, 26, chip_width, 30), _tint(colour), 1.0, radius=15)
        text.add(verdict, (x + 17, 33), 17, colour)
        if assessment:
            text.add(assessment.reason, (x + chip_width + 12, 37), 12, self.theme.muted)

        y = 84
        for finger in self.hand.fingers:
            reading = assessment.fingers.get(finger.name) if assessment else None
            state_name = reading.state if reading else GripState.UNKNOWN.value
            colour = STATE_COLOUR.get(state_name, self.theme.faint)
            oriented = None
            if reading is not None and reading.signed_bend_deg is not None:
                oriented = reading.signed_bend_deg * self.thresholds.wrap_sign

            text.add(finger.name, (x, y), 14, self.theme.text)
            label = "—" if oriented is None else f"{oriented:+.1f}°"
            text.add(label, (x + inner, y), 15, colour, mono=True, anchor="ra")

            draw_gauge(canvas, (x, y + 24, inner, 8),
                       oriented if oriented is not None else float("nan"),
                       self.thresholds.wrap_deg, self.thresholds.backbend_deg,
                       self.full_scale, self.theme)
            draw_trace(canvas, (x, y + 42, inner, 26), self.history[finger.name],
                       self.full_scale, colour, self.theme)
            text.add(state_name, (x, y + 72), 11, self.theme.faint)
            y += 100

        cv2.line(canvas, (x, y + 4), (x + inner, y + 4), self.theme.hairline, 1)
        y += 24
        for label, value in self._footer(state):
            text.add(label, (x, y), 12, self.theme.muted)
            text.add(value, (x + inner, y), 12, self.theme.text, mono=True, anchor="ra")
            y += 22

    def _footer(self, state: FrameState) -> list[tuple[str, str]]:
        seen = len(state.tag_pixels)
        sizes = [_edge_length(q) for q in state.tag_pixels.values()]
        rows = [("tags", f"{seen}/{state.expected_tags}"
                 + (f"  ·  {min(sizes):.0f} px" if sizes else ""))]
        if state.zeroed_seconds is not None:
            rows.append(("zeroed", f"{state.zeroed_seconds:.0f} s ago"))
        if state.calibration:
            rows.append(("calibration", state.calibration))
        if state.flips_rejected:
            rows.append(("pose flips caught", str(state.flips_rejected)))
        rows.append(("measured span", f"{self._measured_span():.0f}%"))
        return rows

    def _measured_span(self) -> float:
        spans = [max(f.stations_mm) / self.mesh.length_mm
                 for f in self.hand.fingers if f.stations_mm]
        return 100.0 * float(np.mean(spans)) if spans else 100.0

    def _finger_colour(self, state: FrameState, name: str):
        reading = state.assessment.fingers.get(name) if state.assessment else None
        if reading is None:
            return self.theme.faint
        return STATE_COLOUR.get(reading.state, self.theme.faint)


def _tint(colour, amount: float = 0.26):
    return tuple(int(c * amount + b * (1 - amount))
                 for c, b in zip(colour, THEME.panel_deep))


def _rooted(points: np.ndarray) -> np.ndarray:
    """Move a finger's samples so its root sits at the origin."""
    points = np.asarray(points, float)
    return points - points[0]


def _dim(colour, amount: float = 0.45):
    return tuple(int(c * amount) for c in colour)


def _padded(points: np.ndarray) -> np.ndarray:
    """Grow a point cloud so a locked framing leaves room for real bending."""
    centre = points.mean(axis=0)
    return np.vstack([centre + (points - centre) * 1.18, points])


def _edge_length(quad) -> float:
    pts = np.asarray(quad, float).reshape(4, 2)
    return float(np.mean([np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]))
