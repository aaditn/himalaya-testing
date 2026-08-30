#!/usr/bin/env python3
"""Load ``configs/track.json`` into MuJoCo and preview the generated track."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from himalaya.env import base as g1_base
from himalaya.env import g1_constants
from himalaya.track import apply_track_to_model, ideal_track_height, load_track_config


def build_model(config_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData, object]:
    config = load_track_config(config_path)
    xml_path = g1_constants.task_to_xml("custom_track")
    model = mujoco.MjModel.from_xml_string(
        xml_path.read_text(), assets=g1_base.get_assets()
    )
    spec = apply_track_to_model(model, config)
    data = mujoco.MjData(model)
    key_id = model.key("knees_bent").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    if config.flat_start_enabled:
        surface_x = spec.track_start_x_m + 0.5 * config.flat_start_length_m
        angle = 0.0
    else:
        surface_x = spec.ramp_start_x_m + min(0.75, 0.25 * config.ramp_run_m)
        angle = config.slope_radians
    surface_z = float(ideal_track_height(config, surface_x))
    nominal_height = float(model.key("knees_bent").qpos[2])
    normal = np.array([-math.sin(angle), 0.0, math.cos(angle)])
    surface_point = np.array([surface_x, 0.0, surface_z])
    data.qpos[:3] = surface_point + nominal_height * normal
    data.qpos[3:7] = (math.cos(angle / 2.0), 0.0, -math.sin(angle / 2.0), 0.0)
    data.ctrl[:] = data.qpos[7:]
    mujoco.mj_forward(model, data)
    return model, data, spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "track.json")
    parser.add_argument("--seconds", type=float, default=0.0, help="0 runs until closed")
    parser.add_argument("--headless", action="store_true", help="validate and print only")
    parser.add_argument("--image", type=Path, help="render one PNG and exit")
    return parser


def render_image(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: object,
    output: Path,
) -> Path:
    """Render a verification frame without opening the interactive viewer."""

    from PIL import Image

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, max(0.8, 0.45 * config.rise_m))
    camera.distance = max(7.0, config.total_length_m * 1.05)
    camera.azimuth = 90.0
    camera.elevation = -12.0
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    with mujoco.Renderer(model, height=720, width=1280) as renderer:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
    Image.fromarray(pixels).save(output)
    return output


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    model, data, spec = build_model(config_path)
    config = load_track_config(config_path)
    print(json_summary(config, model, spec))
    if args.image:
        print(f"rendered {render_image(model, data, config, args.image)}")
        return
    if args.headless:
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, max(0.7, 0.5 * config.rise_m))
        viewer.cam.distance = max(7.0, config.total_length_m * 0.9)
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -18
        started = time.monotonic()
        while viewer.is_running():
            if args.seconds and time.monotonic() - started >= args.seconds:
                break
            before = time.monotonic()
            mujoco.mj_step(model, data)
            viewer.sync()
            remaining = model.opt.timestep - (time.monotonic() - before)
            if remaining > 0:
                time.sleep(remaining)


def json_summary(config: object, model: mujoco.MjModel, spec: object) -> str:
    return (
        f"track angle={config.slope_degrees:g}deg friction={config.friction:.2f} "
        f"roughness={config.roughness_m * 1000:.0f}mm "
        f"flat_start={'on' if config.flat_start_enabled else 'off'} "
        f"rise={config.rise_m:.2f}m footprint={config.total_length_m:.2f}m "
        f"hfield={model.hfield_nrow[0]}x{model.hfield_ncol[0]} "
        f"extent={2 * spec.half_x_m:.2f}x{2 * spec.half_y_m:.2f}m"
    )


if __name__ == "__main__":
    sys.exit(main())
