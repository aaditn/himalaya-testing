"""Drawing primitives for the live grip display.

OpenCV can only draw its own stroked Hershey fonts, which is why every computer
vision demo looks like a computer vision demo. Text here goes through PIL and a
real system typeface instead, batched so the whole frame costs one conversion
to PIL and back rather than one per string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .vision import require_cv2

# Ordered by preference; the first that loads wins.
SANS_CANDIDATES = (
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
MONO_CANDIDATES = (
    "C:/Windows/Fonts/consola.ttf",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)


@dataclass(frozen=True)
class Theme:
    """Colours in BGR, because that is what OpenCV canvases hold."""

    backdrop: tuple = (26, 22, 20)
    panel: tuple = (35, 29, 26)
    panel_deep: tuple = (26, 20, 17)
    hairline: tuple = (58, 48, 43)
    text: tuple = (219, 206, 198)
    muted: tuple = (143, 130, 119)
    neutral: tuple = (122, 112, 104)
    faint: tuple = (110, 100, 92)
    good: tuple = (128, 222, 74)
    good_dim: tuple = (99, 156, 63)
    bad: tuple = (113, 113, 248)
    bad_dim: tuple = (63, 63, 163)
    warn: tuple = (65, 179, 227)
    accent: tuple = (168, 226, 126)
    reference: tuple = (65, 179, 227)


THEME = Theme()

CONTROL_BAR_HEIGHT = 52


def _first_font(candidates):
    from PIL import ImageFont
    for path in candidates:
        if not Path(path).exists():
            continue
        try:
            ImageFont.truetype(path, 16)
            return path
        except OSError:
            continue
    return None


@dataclass
class TextLayer:
    """Accumulates text for a frame and rasterises it in one pass."""

    sans: str | None = None
    mono: str | None = None
    _queue: list = field(default_factory=list)
    _fonts: dict = field(default_factory=dict)
    available: bool = True

    @classmethod
    def create(cls) -> "TextLayer":
        try:
            from PIL import ImageFont  # noqa: F401
        except ImportError:
            return cls(available=False)
        return cls(sans=_first_font(SANS_CANDIDATES), mono=_first_font(MONO_CANDIDATES))

    def _font(self, size: int, mono: bool, bold: bool):
        from PIL import ImageFont
        key = (size, mono, bold)
        if key not in self._fonts:
            path = (self.mono if mono else self.sans) or self.sans or self.mono
            font = (ImageFont.truetype(path, size) if path
                    else ImageFont.load_default(size=size))
            if bold:
                # SF's variable weight axis is not exposed through PIL, so bold
                # is faked at draw time with a stroke instead.
                pass
            self._fonts[key] = font
        return self._fonts[key]

    def add(self, text: str, xy, size: int = 14, colour=THEME.text,
            mono: bool = False, bold: bool = False, anchor: str = "la") -> None:
        self._queue.append((str(text), (float(xy[0]), float(xy[1])), size,
                            tuple(colour), mono, bold, anchor))

    def measure(self, text: str, size: int = 14, mono: bool = False) -> tuple[int, int]:
        if not self.available:
            cv2 = require_cv2()
            (w, h), _ = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX,
                                        size / 30.0, 1)
            return w, h
        from PIL import ImageDraw, Image
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        box = draw.textbbox((0, 0), str(text), font=self._font(size, mono, False))
        return int(box[2] - box[0]), int(box[3] - box[1])

    def flush(self, canvas: np.ndarray) -> None:
        if not self._queue:
            return
        if not self.available:
            self._flush_opencv(canvas)
            return
        from PIL import Image, ImageDraw
        # The canvas stays in BGR and the fills are passed in BGR too. PIL
        # mislabels both identically, so the two errors cancel and two
        # full-frame channel reversals per frame disappear.
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        for text, xy, size, colour, mono, bold, anchor in self._queue:
            draw.text(xy, text, font=self._font(size, mono, bold),
                      fill=tuple(colour), anchor=anchor,
                      stroke_width=1 if bold else 0,
                      stroke_fill=tuple(colour))
        canvas[:, :, :] = np.asarray(image)
        self._queue.clear()

    def _flush_opencv(self, canvas: np.ndarray) -> None:
        cv2 = require_cv2()
        for text, xy, size, colour, _mono, bold, anchor in self._queue:
            scale = size / 30.0
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            x, y = xy
            if anchor[0] == "r":
                x -= w
            elif anchor[0] == "m":
                x -= w / 2
            y += h if anchor[1] in ("a", "t") else 0
            cv2.putText(canvas, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, colour, 2 if bold else 1, cv2.LINE_AA)
        self._queue.clear()


def append_control_bar(canvas: np.ndarray, controls: tuple[tuple[str, str], ...],
                       status: str = "", theme: Theme = THEME) -> np.ndarray:
    """Append a dedicated, always-visible keyboard help bar below a view.

    The camera pixels keep their calibrated size and coordinates; the help is
    added outside the image instead of obscuring tags or being clipped by a
    busy frame.  Keeping the key/description pairs next to the event loop also
    makes it straightforward for callers to display the controls they really
    implement.
    """
    height, width = canvas.shape[:2]
    output = np.empty((height + CONTROL_BAR_HEIGHT, width, 3), np.uint8)
    output[:height] = canvas
    output[height:] = theme.panel_deep

    cv2 = require_cv2()
    cv2.line(output, (0, height), (width, height), theme.hairline, 1, cv2.LINE_AA)
    text = TextLayer.create()
    x = 18
    baseline = height + 17
    for key, label in controls:
        key = str(key).upper()
        key_width = max(30, text.measure(key, 13, mono=True)[0] + 16)
        blend_rect(output, (x, height + 10, key_width, 31), theme.hairline,
                   1.0, radius=6)
        outline_rect(output, (x, height + 10, key_width, 31), theme.muted,
                     1, radius=6)
        text.add(key, (x + key_width / 2, baseline), 13, theme.text,
                 mono=True, bold=True, anchor="ma")
        x += key_width + 8
        text.add(label, (x, baseline + 1), 13, theme.muted)
        x += text.measure(label, 13)[0] + 24

    if status:
        # A long dataset path or status is allowed to yield to the controls.
        # It remains right-aligned and is clipped naturally at the frame edge.
        text.add(status, (width - 18, baseline + 1), 13, theme.accent,
                 mono=True, anchor="ra")
    text.flush(output)
    return output

def blend_rect(canvas: np.ndarray, rect, colour, alpha: float = 1.0,
               radius: int = 0) -> None:
    """Fill a rectangle, optionally rounded and translucent."""
    cv2 = require_cv2()
    x, y, w, h = (int(v) for v in rect)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, canvas.shape[1]), min(y + h, canvas.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    region = canvas[y0:y1, x0:x1]
    patch = np.empty_like(region)
    patch[:, :] = colour
    if radius > 0:
        mask = np.zeros(region.shape[:2], np.uint8)
        r = int(min(radius, (x1 - x0) // 2, (y1 - y0) // 2))
        cv2.rectangle(mask, (r, 0), (x1 - x0 - r, y1 - y0), 255, -1)
        cv2.rectangle(mask, (0, r), (x1 - x0, y1 - y0 - r), 255, -1)
        for cx, cy in ((r, r), (x1 - x0 - r, r), (r, y1 - y0 - r),
                       (x1 - x0 - r, y1 - y0 - r)):
            cv2.circle(mask, (cx, cy), r, 255, -1)
        keep = mask.astype(bool)
        blended = (patch * alpha + region * (1 - alpha)).astype(region.dtype)
        region[keep] = blended[keep]
    else:
        region[:, :] = (patch * alpha + region * (1 - alpha)).astype(region.dtype)


def outline_rect(canvas: np.ndarray, rect, colour, thickness: int = 1,
                 radius: int = 0) -> None:
    cv2 = require_cv2()
    x, y, w, h = (int(v) for v in rect)
    if radius <= 0:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, thickness, cv2.LINE_AA)
        return
    r = int(min(radius, w // 2, h // 2))
    cv2.line(canvas, (x + r, y), (x + w - r, y), colour, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x + r, y + h), (x + w - r, y + h), colour, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x, y + r), (x, y + h - r), colour, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x + w, y + r), (x + w, y + h - r), colour, thickness, cv2.LINE_AA)
    for (cx, cy), start in (((x + r, y + r), 180), ((x + w - r, y + r), 270),
                            ((x + w - r, y + h - r), 0), ((x + r, y + h - r), 90)):
        cv2.ellipse(canvas, (cx, cy), (r, r), 0, start, start + 90, colour,
                    thickness, cv2.LINE_AA)


def draw_gauge(canvas: np.ndarray, rect, value: float, wrap_deg: float,
               backbend_deg: float, full_scale: float,
               theme: Theme = THEME) -> None:
    """A signed bend gauge: zero at the centre, wrapping right, back-bending left.

    The thresholds are drawn as ticks so you can see how close a reading sits to
    a class boundary - the thing a bare number cannot show, and exactly what
    matters when the thresholds still need tuning.
    """
    cv2 = require_cv2()
    x, y, w, h = (int(v) for v in rect)
    blend_rect(canvas, rect, theme.panel_deep, 1.0, radius=h // 2)

    centre = x + w / 2.0
    per_deg = (w / 2.0) / max(full_scale, 1e-6)

    def at(degrees: float) -> int:
        return int(np.clip(centre + degrees * per_deg, x, x + w))

    # Neutral band: the zone that will be reported as "not loaded enough".
    lo, hi = at(-backbend_deg), at(wrap_deg)
    blend_rect(canvas, (lo, y, hi - lo, h), theme.hairline, 1.0)

    if np.isfinite(value):
        end = at(value)
        colour = (theme.good_dim if value >= wrap_deg
                  else theme.bad_dim if value <= -backbend_deg else theme.faint)
        left, right = (centre, end) if end >= centre else (end, centre)
        blend_rect(canvas, (int(left), y, max(int(right - left), 2), h), colour, 1.0,
                   radius=h // 2)
        tip = (theme.good if value >= wrap_deg
               else theme.bad if value <= -backbend_deg else theme.text)
        cv2.line(canvas, (end, y - 3), (end, y + h + 3), tip, 2, cv2.LINE_AA)

    for degrees, colour in ((wrap_deg, theme.good), (-backbend_deg, theme.bad)):
        tick = at(degrees)
        cv2.line(canvas, (tick, y), (tick, y + h), colour, 1, cv2.LINE_AA)
    cv2.line(canvas, (int(centre), y), (int(centre), y + h), theme.faint, 1, cv2.LINE_AA)


def draw_trace(canvas: np.ndarray, rect, history, full_scale: float, colour,
               theme: Theme = THEME) -> None:
    """A rolling history strip; a slip shows up as a step, not a number blur."""
    cv2 = require_cv2()
    x, y, w, h = (int(v) for v in rect)
    values = [v for v in history if v is not None and np.isfinite(v)]
    cv2.line(canvas, (x, y + h // 2), (x + w, y + h // 2), theme.hairline, 1, cv2.LINE_AA)
    if len(values) < 2:
        return
    xs = np.linspace(x, x + w, len(values))
    ys = y + h / 2.0 - np.clip(np.array(values) / max(full_scale, 1e-6), -1, 1) * (h / 2.0 - 1)
    points = np.column_stack([xs, ys]).astype(np.int32)
    cv2.polylines(canvas, [points], False, colour, 1, cv2.LINE_AA)
