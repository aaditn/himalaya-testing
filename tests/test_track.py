"""Dependency-light tests for the configurable slope track."""

from __future__ import annotations

import math

import numpy as np
import pytest

from himalaya.track import TrackConfig, generate_heightfield, ideal_track_height


def test_defaults_are_a_30_degree_climb() -> None:
    config = TrackConfig()
    assert config.slope_degrees == 30.0
    assert config.flat_start_m == 2.0
    assert config.rise_m == pytest.approx(3.0)
    assert config.minimum_static_friction == pytest.approx(math.tan(math.pi / 6))


def test_flat_start_can_be_removed_without_changing_saved_length() -> None:
    config = TrackConfig(flat_start_enabled=False, flat_start_length_m=3.5)
    assert config.flat_start_m == 0.0
    expected = config.ramp_run_m + config.summit_length_m
    assert config.total_length_m == pytest.approx(expected)


@pytest.mark.parametrize("angle", [4.99, 45.01])
def test_angle_slider_limits_are_enforced(angle: float) -> None:
    with pytest.raises(ValueError, match="slope_degrees"):
        TrackConfig(slope_degrees=angle)


def test_heightfield_is_deterministic_and_bounded() -> None:
    config = TrackConfig(roughness_m=0.035, seed=42)
    first = generate_heightfield(config, nrow=33, ncol=65)
    second = generate_heightfield(config, nrow=33, ncol=65)
    np.testing.assert_array_equal(first.data, second.data)
    assert first.data.shape == (33, 65)
    assert float(first.data.min()) >= 0.0
    assert float(first.data.max()) <= 1.0


def test_flat_start_is_level_and_ramp_reaches_expected_rise() -> None:
    config = TrackConfig(roughness_m=0.0)
    spec = generate_heightfield(config, nrow=17, ncol=33)
    start_height = ideal_track_height(config, spec.track_start_x_m + 0.25)
    ramp_height = ideal_track_height(config, spec.ramp_end_x_m)
    assert float(start_height) == pytest.approx(0.0)
    assert float(ramp_height) == pytest.approx(config.rise_m)


def test_unknown_option_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown track option"):
        TrackConfig.from_mapping({"slope_degrees": 20, "surprise": True})
