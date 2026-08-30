"""Front end for the finger simulation: build, press, report."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .fem2d import peak_fibre_strain, peak_fibre_stress
from .finray_geometry import MATERIALS_MPA, FingerSpec, build_finger
from .finray_sim import CylinderObstacle, FlatObstacle, press, summarise
from .materials import TPU95A, Material, Yeoh, neo_hookean

# Elastomers get a hyperelastic law by default. A single modulus describes them
# only over the first few percent of strain, and this finger runs past twenty.
HYPERELASTIC: dict[str, Material] = {
    "tpu95a": TPU95A,
    "tpu85a": neo_hookean(MATERIALS_MPA["tpu85a"]),
}


def resolve_material(material: str, hookean: bool = False) -> tuple[float, Material | None]:
    """Return (initial modulus in MPa, constitutive law or None for Hookean)."""
    key = material.strip().lower()
    if key in HYPERELASTIC and not hookean:
        law = HYPERELASTIC[key]
        return law.initial_modulus, law
    if key in HYPERELASTIC:
        return HYPERELASTIC[key].initial_modulus, None
    if key in MATERIALS_MPA:
        return MATERIALS_MPA[key], None
    try:
        return float(material), None
    except ValueError as error:
        known = ", ".join(sorted(MATERIALS_MPA))
        raise SystemExit(f"unknown material {material!r}; use one of {known} "
                         f"or a modulus in MPa") from error


def _element_thickness(frame) -> np.ndarray:
    # A rectangle has I = b*t^3/12 and A = b*t, so t = sqrt(12 I / A) whatever
    # the depth is. Saves carrying a parallel array around.
    return np.array([math.sqrt(12.0 * e.second_moment / e.area) for e in frame.elements])


def run(material: str = "petg", depth: float = 15.0, rib_thickness: float = 1.5,
        shape: str = "cylinder", radius: float = 15.0, station: float = 55.0,
        span: tuple[float, float] = (8.0, 58.0), advance: float = 8.0,
        steps: int = 20, refine: int = 8, hookean: bool = False) -> dict:
    modulus, law = resolve_material(material, hookean)
    spec = FingerSpec(youngs_modulus=modulus, depth=depth, material=law,
                      rib_thickness=rib_thickness, elements_per_bay=refine)
    model = build_finger(spec)
    obstacle = (CylinderObstacle(radius=radius, station=station) if shape == "cylinder"
                else FlatObstacle(span=span))
    result = press(model, obstacle, advance, steps)

    thickness = _element_thickness(model.frame)
    rows = summarise(result)
    for row, step in zip(rows, result.steps):
        stress, _ = peak_fibre_stress(model.frame, step.displacement, thickness)
        strain, _ = peak_fibre_strain(model.frame, step.displacement)
        row["peak_stress_mpa"] = float(stress)
        row["peak_strain_pct"] = 100.0 * float(strain)

    report = {
        "geometry": {
            "front_length_mm": spec.front_length,
            "back_length_mm_derived": float(np.hypot(*spec.base_outer_corner)),
            "taper_angle_deg": math.degrees(spec.taper_angle),
            "base_clear_height_mm": spec.clear_height(spec.front_length),
            "rib_stations_mm": [round(v, 3) for v in spec.rib_stations],
            "wall_mm": spec.wall,
            "rib_thickness_mm": spec.rib_thickness,
            "depth_mm": spec.depth,
            "initial_modulus_mpa": spec.youngs_modulus,
            "constitutive_law": (type(law).__name__ if law is not None else "Hookean"),
            "yeoh_coefficients": ([law.c10, law.c20, law.c30]
                                  if isinstance(law, Yeoh) else None),
        },
        "obstacle": {"shape": shape, "radius_mm": radius, "station_mm": station,
                     "span_mm": list(span)},
        "solver": {"nodes": model.frame.n_nodes, "elements": len(model.frame.elements),
                   "penalty_n_per_mm": result.penalty,
                   "completed": result.completed,
                   "reached_mm": result.reached_mm},
        "steps": rows,
    }
    return {"result": result, "report": report}


def write_report(report: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return target
