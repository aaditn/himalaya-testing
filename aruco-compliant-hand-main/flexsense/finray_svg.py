"""Draw the finger and its deformed shapes as SVG.

Plain string building, no matplotlib. The rest of this project ships on numpy,
OpenCV and PyYAML, and a plotting stack is a lot of dependency to add for a few
polylines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .fem2d import deformed_nodes
from .finray_geometry import outline
from .finray_sim import SimulationResult, summarise

_STEP_COLOURS = ("#2f6fb0", "#5aa0d8", "#e08a2e", "#c8452d")


def _polyline(points: np.ndarray, colour: str, width: float, opacity: float = 1.0) -> str:
    coords = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
    return (f'<polyline points="{coords}" fill="none" stroke="{colour}" '
            f'stroke-width="{width:.2f}" stroke-opacity="{opacity:.2f}" '
            f'stroke-linecap="round"/>')


def render(result: SimulationResult, output_path: str | Path,
           frames: int = 4, margin: float = 12.0) -> Path:
    model = result.model
    frame = model.frame
    spec = model.spec

    picked = []
    if result.steps:
        indices = np.unique(np.linspace(0, len(result.steps) - 1, frames).astype(int))
        picked = [int(i) for i in indices]

    shapes: list[np.ndarray] = [outline(spec)]
    for index in picked:
        shapes.append(deformed_nodes(frame, result.steps[index].displacement))
        shapes.append(result.obstacle.surface(result.steps[index].advance))
    stacked = np.vstack(shapes)
    lo = stacked.min(axis=0) - margin
    hi = stacked.max(axis=0) + margin
    width, height = hi - lo

    # SVG y grows downward; flip so the drawing matches the CAD side view.
    def to_svg(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        return np.column_stack([points[:, 0] - lo[0], hi[1] - points[:, 1]])

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width * 4:.0f}" '
        f'height="{height * 4:.0f}" viewBox="0 0 {width:.3f} {height:.3f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _polyline(to_svg(outline(spec)), "#b8b8b8", 0.5),
    ]

    for order, index in enumerate(picked):
        step = result.steps[index]
        colour = _STEP_COLOURS[min(order, len(_STEP_COLOURS) - 1)]
        nodes = deformed_nodes(frame, step.displacement)
        for element in frame.elements:
            segment = nodes[[element.node_i, element.node_j]]
            parts.append(_polyline(to_svg(segment), colour, 0.42))
        parts.append(_polyline(to_svg(result.obstacle.surface(step.advance)),
                               colour, 0.42, opacity=0.55))

    rows = summarise(result)
    label = (f"{spec.youngs_modulus:.0f} MPa, {spec.depth:.0f} mm deep, "
             f"walls {spec.wall} mm, ribs {spec.rib_thickness} mm")
    parts.append(f'<text x="1.5" y="4.2" font-family="sans-serif" '
                 f'font-size="3">{label}</text>')
    if rows:
        last = rows[-1]
        detail = (f"{last['advance_mm']:.2f} mm indentation -> "
                  f"{last['normal_force_n']:.2f} N, contact patch "
                  f"{last['contact_patch_mm']:.1f} mm, tip rotation "
                  f"{last['tip_rotation_deg']:+.1f} deg")
        parts.append(f'<text x="1.5" y="8.2" font-family="sans-serif" '
                     f'font-size="3">{detail}</text>')
    if not result.completed:
        parts.append(f'<text x="1.5" y="12.2" font-family="sans-serif" '
                     f'font-size="3" fill="#c8452d">stopped converging at '
                     f'{result.reached_mm:.2f} mm</text>')
    parts.append("</svg>")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
