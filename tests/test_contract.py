"""Dependency-light checks for the fixed Stage-I experiment contract."""

from __future__ import annotations

import ast
from pathlib import Path

from himalaya.tasks.g1_cfg import (
    CURRICULUM_SLOPES_DEG,
    FOUR_CONTACT_ACTOR_OBSERVATION_SIZE,
    FOUR_CONTACT_CURRICULUM_SLOPES_DEG,
    FOUR_CONTACT_PRIVILEGED_OBSERVATION_SIZE,
    G1_ACTION_SIZE,
    G1_ACTOR_OBSERVATION_SIZE,
    G1_PRIVILEGED_OBSERVATION_SIZE,
    validate_slope,
)


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_curriculum_and_spaces() -> None:
    assert CURRICULUM_SLOPES_DEG == (0.0, 5.0, 10.0, 15.0, 20.0)
    assert G1_ACTION_SIZE == 29
    assert G1_ACTOR_OBSERVATION_SIZE == 103
    assert G1_PRIVILEGED_OBSERVATION_SIZE == 221
    assert FOUR_CONTACT_CURRICULUM_SLOPES_DEG == (30.0, 35.0)
    assert FOUR_CONTACT_ACTOR_OBSERVATION_SIZE == 103
    assert FOUR_CONTACT_PRIVILEGED_OBSERVATION_SIZE == 241


def test_unknown_slope_is_rejected() -> None:
    try:
        validate_slope(12.5)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported grade was accepted")


def test_package_has_no_disallowed_simulator_imports() -> None:
    disallowed_prefix = "i" + "saac"
    for path in (ROOT / "himalaya").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert all(
            not name.lower().startswith(disallowed_prefix) for name in imported
        )
