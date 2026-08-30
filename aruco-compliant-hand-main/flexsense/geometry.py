from __future__ import annotations

import math

import numpy as np


def homography_from_four_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return H such that dst ~ H @ src for four 2D point pairs."""
    src = np.asarray(src, dtype=float).reshape(4, 2)
    dst = np.asarray(dst, dtype=float).reshape(4, 2)
    rows: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    h = np.linalg.solve(np.asarray(rows), np.asarray(values))
    return np.append(h, 1.0).reshape(3, 3)


def apply_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    mapped = (np.asarray(homography, dtype=float) @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def reference_homography(reference_corners_px: np.ndarray, marker_size_mm: float) -> np.ndarray:
    """Map image pixels into the rigid reference marker's millimetre frame.

    OpenCV orders corners top-left, top-right, bottom-right, bottom-left.
    The returned frame has +x right and +y up on the printed marker.
    """
    half = marker_size_mm / 2.0
    destination = np.asarray(
        [[-half, half], [half, half], [half, -half], [-half, -half]],
        dtype=float,
    )
    return homography_from_four_points(reference_corners_px, destination)


def marker_center_and_angle(corners_px: np.ndarray, homography: np.ndarray) -> tuple[np.ndarray, float]:
    mapped = apply_homography(np.asarray(corners_px).reshape(4, 2), homography)
    center = mapped.mean(axis=0)
    top_edge = mapped[1] - mapped[0]
    angle_deg = math.degrees(math.atan2(top_edge[1], top_edge[0]))
    return center, angle_deg


def wrapped_angle_delta_deg(current: float, baseline: float) -> float:
    return (current - baseline + 180.0) % 360.0 - 180.0


def signed_tip_curvature(points: np.ndarray) -> float | None:
    """Fit x(s), y(s) quadratics and return signed curvature at the tip.

    This is only a compact bend-shape indicator. It is not a material strain
    calculation and requires at least three visible markers.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(points) < 3:
        return None
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(steps)])
    if s[-1] < 1e-6:
        return None
    x_coeff = np.polyfit(s, points[:, 0], deg=2)
    y_coeff = np.polyfit(s, points[:, 1], deg=2)
    tip = s[-1]
    dx = 2.0 * x_coeff[0] * tip + x_coeff[1]
    dy = 2.0 * y_coeff[0] * tip + y_coeff[1]
    ddx = 2.0 * x_coeff[0]
    ddy = 2.0 * y_coeff[0]
    denominator = (dx * dx + dy * dy) ** 1.5
    if denominator < 1e-9:
        return None
    return float((dx * ddy - dy * ddx) / denominator)


def axis_component(vector_xy: np.ndarray, axis: str, sign: float = 1.0) -> float:
    index = {"x": 0, "y": 1}[axis]
    return float(sign * np.asarray(vector_xy)[index])

