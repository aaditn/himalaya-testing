"""Single source of truth for the SCENE: geometry, slope, spawn, assets.

Everything here was previously duplicated between himalaya/env (which trains)
and scripts/view.py (which you look at). That split is how a spawn change
landed in training while the viewer kept showing the old position -- the two
copies drifted the moment either was edited.

Deliberately NUMPY ONLY. No jax, no mujoco_playground. That is what lets the
laptop viewer import the same code the pod trains with, instead of keeping its
own copy to avoid installing the training stack. joystick.py wraps these values
in jax where it needs to trace through them.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
XMLS = ROOT / "xmls"
ASSETS = XMLS / "assets"
# G1 meshes. The pod resolves these through mujoco_playground's MENAGERIE_PATH;
# locally we keep a copy so the viewer needs nothing but mujoco.
LOCAL_MENAGERIE = Path.home() / ".cache" / "himalaya" / "unitree_g1"

SCENES = {
    "flat_terrain": "scene_mjx_feetonly_flat_terrain.xml",
    "rough_terrain": "scene_mjx_feetonly_rough_terrain.xml",
    "slope_terrain": "scene_mjx_feetonly_slope_terrain.xml",
    "ramp_terrain": "scene_mjx_feetonly_ramp_terrain.xml",
    "mountain_terrain": "scene_mjx_feetonly_mountain_terrain.xml",
}

# --- spawn -----------------------------------------------------------------
# Where the robot starts on the mountain, in metres. +x is UPHILL on the tilted
# floor, so a smaller x is further down the slope.
SPAWN = (4.0, 5.5)
# Half-width of the grid sampled to find the rock under the spawn. The grid
# takes the HIGHEST cell in the span, so a wide span lifts the robot onto a
# boulder it is standing beside rather than on: 0.25 m overshot the measured
# 0.17-0.22 m touchdown height badly.
SPAWN_PROBE_SPAN = 0.12
# Clearance left above the rock. Bilinear sampling smooths sharp ledges, so the
# true surface under a foot can sit slightly above the interpolated value.
SPAWN_CLEARANCE = 0.05

# --- physics ---------------------------------------------------------------
NJMAX = 160
NACONMAX = 131072
# Contact pairs 0-3 are the floor pairs, in this order. Pairs 4+ are
# self-collision (hand-thigh, foot-foot) and must keep their own friction.
# MuJoCo REORDERS <pair> elements, so re-verify with mj_id2name after any XML
# pair change -- adding the platform pairs once silently reshuffled these.
FLOOR_PAIRS = slice(0, 4)
# Network shape. Training and rendering must agree or the checkpoint will not
# load into the policy.
HIDDEN_LAYERS = (512, 256, 128)


def scene_xml(task):
    return XMLS / SCENES[task]


def slope_quat(slope_rad):
    """Floor rotation about +Y, as a (w,x,y,z) quaternion."""
    half = 0.5 * slope_rad
    return np.array([np.cos(half), 0.0, np.sin(half), 0.0])


def tilted_xml(task, slope_rad):
    """Scene XML text with the floor tilted, ready to compile.

    The substitution has to happen BEFORE compile: MuJoCo bakes worldbody geom
    orientation into geom_xmat at compile time, so writing geom_quat afterwards
    does nothing at all. That failed silently once -- contact sensors fired,
    the logs said 45 degrees, and the robot walked on level ground.
    """
    xml = scene_xml(task).read_text()
    if slope_rad != 0.0:
        q = slope_quat(slope_rad)
        xml = xml.replace('quat="1 0 0 0"',
                          f'quat="{q[0]} 0 {q[2]} 0"', 1)
    return xml


def slope_normal(slope_rad):
    """Surface normal of the tilted floor, in world coordinates.

    Equals the floor geom's local z axis (verified against geom_xmat), which is
    why terrain relief stacks along it rather than along world z.
    """
    return np.array([np.sin(slope_rad), 0.0, np.cos(slope_rad)])


def uphill(slope_rad):
    """Unit vector pointing up the slope, in world coordinates."""
    return np.array([-np.cos(slope_rad), 0.0, np.sin(slope_rad)])


def surface_z(x, slope_rad):
    """World z of the tilted mean plane at up-slope position x."""
    return -x * np.tan(slope_rad)


def lane_mouths():
    """Lateral offset of each route's entrance, or None if the map has no routes."""
    p = ASSETS / "mountain_centre.npy"
    return np.load(p.as_posix()) if p.exists() else None


def probe_offsets(span=None):
    """3x3 grid of x,y offsets used to find the rock under a spawn point."""
    s = SPAWN_PROBE_SPAN if span is None else span
    return np.array([[dx * s, dy * s, 0.0]
                     for dx in (0, 1, -1) for dy in (0, 1, -1)])


def local_assets():
    """Every asset by NAME, for a viewer that has no mujoco_playground.

    The XMLs reference meshes as "../../../mujoco_menagerie/..." which never
    resolves on disk -- MuJoCo looks the name up in this dict instead.
    """
    assets = {}
    for d in (XMLS, ASSETS, LOCAL_MENAGERIE, LOCAL_MENAGERIE / "assets"):
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.is_file():
                assets[f.name] = f.read_bytes()
    return assets
