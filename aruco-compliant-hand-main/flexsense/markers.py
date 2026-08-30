from __future__ import annotations

import base64
from pathlib import Path

from .config import AppConfig
from .vision import MarkerDetector, require_cv2


def generate_marker_sheet(config: AppConfig, output_path: str | Path) -> Path:
    """The old two-finger planar layout, driven by `config/so101.yaml`."""
    return _sheet(config.markers.dictionary, sorted(config.required_marker_ids),
                  config.markers.size_mm,
                  lambda i: "REFERENCE" if i == config.markers.reference_id else "FLEXURE",
                  output_path)


def generate_hand_marker_sheet(hand, output_path: str | Path) -> Path:
    """The sheet the built hand actually needs, driven by `config/hand.yaml`.

    Roles come from the same declaration the tracker reads, so a tag can never
    be printed under one id here and expected under another there.
    """
    return _sheet(hand.dictionary, sorted(hand.all_tag_ids), hand.tag_mm,
                  lambda i: hand.owner_of(i).upper(), output_path)


def _sheet(dictionary: str, marker_ids: list[int], marker_size: float,
           role_of, output_path: str | Path) -> Path:
    cv2 = require_cv2()
    detector = MarkerDetector(dictionary)
    quiet = 3.0
    label_height = 7.0
    cell_width = marker_size + 2 * quiet + 12.0
    cell_height = marker_size + 2 * quiet + label_height + 6.0
    columns = 4
    rows = (len(marker_ids) + columns - 1) // columns
    page_width = 210.0
    page_height = max(60.0, 15.0 + rows * cell_height)
    start_x = (page_width - columns * cell_width) / 2.0

    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{page_width}mm" '
            f'height="{page_height}mm" viewBox="0 0 {page_width} {page_height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="10" y="8" font-family="sans-serif" font-size="4">'
        f'FlexSense {dictionary}, print at 100%, tags {marker_size:.1f} mm</text>',
    ]

    for index, marker_id in enumerate(marker_ids):
        image = cv2.aruco.generateImageMarker(detector.dictionary, marker_id, 600)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError(f"Could not encode marker {marker_id}")
        data = base64.b64encode(encoded.tobytes()).decode("ascii")
        column = index % columns
        row = index // columns
        cell_x = start_x + column * cell_width
        cell_y = 12.0 + row * cell_height
        image_x = cell_x + (cell_width - marker_size) / 2.0
        image_y = cell_y + quiet
        role = role_of(marker_id)
        svg.extend(
            [
                (
                    f'<rect x="{image_x - quiet}" y="{image_y - quiet}" '
                    f'width="{marker_size + 2 * quiet}" height="{marker_size + 2 * quiet}" '
                    'fill="white" stroke="#cccccc" stroke-width="0.2"/>'
                ),
                (
                    f'<image x="{image_x}" y="{image_y}" width="{marker_size}" '
                    f'height="{marker_size}" style="image-rendering:pixelated" '
                    f'xlink:href="data:image/png;base64,{data}"/>'
                ),
                (
                    f'<text x="{cell_x + cell_width / 2}" '
                    f'y="{image_y + marker_size + 5}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="3.2">ID {marker_id} {role}</text>'
                ),
            ]
        )
    svg.append("</svg>")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")
    return output

