from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PolynomialForceModel:
    coefficients: tuple[float, ...]
    clamp_min_n: float | None = None
    clamp_max_n: float | None = None

    def evaluate(self, deflection_mm: float) -> float:
        value = float(np.polyval(self.coefficients, deflection_mm))
        if self.clamp_min_n is not None:
            value = max(self.clamp_min_n, value)
        if self.clamp_max_n is not None:
            value = min(self.clamp_max_n, value)
        return value


class ForceModelSet:
    def __init__(self, models: dict[str, dict[str, PolynomialForceModel]] | None = None):
        self.models = models or {}

    @classmethod
    def load(cls, path: str | Path) -> "ForceModelSet":
        model_path = Path(path)
        if not model_path.exists():
            return cls()
        with model_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        models: dict[str, dict[str, PolynomialForceModel]] = {}
        for sensor_name, axes in raw.get("models", {}).items():
            models[sensor_name] = {}
            for axis_name, item in axes.items():
                models[sensor_name][axis_name] = PolynomialForceModel(
                    coefficients=tuple(float(x) for x in item["coefficients"]),
                    clamp_min_n=(
                        float(item["clamp_min_n"])
                        if item.get("clamp_min_n") is not None
                        else None
                    ),
                    clamp_max_n=(
                        float(item["clamp_max_n"])
                        if item.get("clamp_max_n") is not None
                        else None
                    ),
                )
        return cls(models)

    def evaluate(self, sensor_name: str, axis_name: str, deflection_mm: float) -> float | None:
        model = self.models.get(sensor_name, {}).get(axis_name)
        return None if model is None else model.evaluate(deflection_mm)

    @staticmethod
    def update_file(
        path: str | Path,
        sensor_name: str,
        axis_name: str,
        coefficients: list[float],
        metadata: dict[str, Any],
        clamp_min_n: float | None,
    ) -> None:
        model_path = Path(path)
        if model_path.exists():
            with model_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        else:
            raw = {"schema_version": 1, "models": {}}
        raw.setdefault("models", {}).setdefault(sensor_name, {})[axis_name] = {
            "coefficients": [float(x) for x in coefficients],
            "clamp_min_n": clamp_min_n,
            "clamp_max_n": None,
            "metadata": metadata,
        }
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("w", encoding="utf-8") as handle:
            json.dump(raw, handle, indent=2, sort_keys=True)
            handle.write("\n")

