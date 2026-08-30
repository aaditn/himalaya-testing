"""Configuration helpers for the MuJoCo Playground Unitree G1.

The robot model, joint order, position actuators, PD gains, and limits come
from DeepMind's G1 joystick environment. Himalaya intentionally does not
duplicate or replace that locomotion stack.
"""

from __future__ import annotations

from dataclasses import dataclass


CURRICULUM_SLOPES_DEG = (0.0, 5.0, 10.0, 15.0, 20.0)
# MODIFIED: crawl acquisition is intentionally restricted to the two reviewed
# grades.  The failed 5-degree checkpoint is diagnostic only and cannot be
# selected by the training CLI.
FOUR_CONTACT_CURRICULUM_SLOPES_DEG = (30.0, 35.0)
G1_ACTION_SIZE = 29
G1_ACTOR_OBSERVATION_SIZE = 103
G1_BASE_PRIVILEGED_OBSERVATION_SIZE = 216
SLOPE_DESCRIPTOR_SIZE = 5
G1_PRIVILEGED_OBSERVATION_SIZE = (
    G1_BASE_PRIVILEGED_OBSERVATION_SIZE + SLOPE_DESCRIPTOR_SIZE
)
FOUR_CONTACT_ACTOR_OBSERVATION_SIZE = 103
FOUR_CONTACT_PRIVILEGED_OBSERVATION_SIZE = 241


@dataclass(frozen=True)
class ValidationTargets:
    """Promotion criteria for each uniform-ramp curriculum stage."""

    success_distance_m: float = 6.0
    min_success_rate: float = 0.90
    commanded_speed_mps: float = 0.50
    trials: int = 64


def validate_slope(slope_degrees: float) -> float:
    """Return a supported curriculum slope, raising on accidental roughness."""

    slope = float(slope_degrees)
    if slope not in CURRICULUM_SLOPES_DEG:
        allowed = ", ".join(f"{x:g}" for x in CURRICULUM_SLOPES_DEG)
        raise ValueError(
            f"slope_degrees must be one of [{allowed}] for Stage I; got {slope:g}"
        )
    return slope


def validate_four_contact_slope(slope_degrees: float) -> float:
    """Return a supported four-contact curriculum grade."""

    slope = float(slope_degrees)
    if slope not in FOUR_CONTACT_CURRICULUM_SLOPES_DEG:
        allowed = ", ".join(
            f"{x:g}" for x in FOUR_CONTACT_CURRICULUM_SLOPES_DEG
        )
        raise ValueError(
            f"four-contact slope must be one of [{allowed}]; got {slope:g}"
        )
    return slope
