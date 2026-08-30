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
SPAWN = (4.41, 1.54)
# Heading on a slope, radians about world +z. pi faces the robot up the hill:
# the body tilt below rotates it about +Y to stand perpendicular to the surface,
# which pitches its forward axis DOWN the hill, so without this a forward
# velocity command is an instruction to descend.
SPAWN_YAW = 3.910
SPAWN_YAW_JITTER = 0.3
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


def route_lines(slope_rad=0.0):
    """(n_routes, steps, 2) WORLD x,y centreline of every corridor, or None.

    make_route.py traces these in heightfield GRID space, which is the floor
    geom's local frame. On a tilted floor that is not world space: the geom is
    rotated about +Y, so local x maps to world x/cos(slope). Skipping that
    conversion put every route -- and every relief measurement taken against
    one -- at the wrong place on the hill, by more than a metre at 15 degrees.
    """
    p = ASSETS / "mountain_lines.npy"
    if not p.exists():
        return None
    # Already WORLD space: scripts/trace_routes.py ray-casts the compiled
    # geometry, so no frame conversion belongs here. An earlier version read
    # mountain.png directly and indexed it in the heightfield's LOCAL frame,
    # which on a tilted floor is off by more than a metre at 15 degrees -- and
    # because the x and y errors differ, it ROTATES the line. That is what put
    # the reward 90 degrees off the visible corridor.
    return np.load(p.as_posix())


def route_tangent(lines, xy):
    """Unit heading of the nearest corridor at a world point, and the distance.

    Returns (tangent_xy, distance_m). The tangent always points UP the slope
    (towards +x, since the floor is tilted about +Y and descends with x).
    """
    pts = lines.reshape(-1, 2)
    d = np.linalg.norm(pts - np.asarray(xy)[:2], axis=1)
    k = int(np.argmin(d))
    ri, si = k // lines.shape[1], k % lines.shape[1]
    nxt = min(si + 8, lines.shape[1] - 1)
    prv = max(si - 8, 0)
    t = lines[ri, nxt] - lines[ri, prv]
    nrm = np.linalg.norm(t)
    if nrm < 1e-9:
        return np.array([-1.0, 0.0]), float(d[k])
    t = t / nrm
    # ORIENT IT UP-SLOPE. A traced line is the same path in either direction,
    # and which way the tracer happened to walk is arbitrary -- as written it
    # came out pointing downhill, so the reward paid for descending the
    # corridor. The floor descends with world x (surface_z = -x*tan), so
    # up-slope is -x, and any tangent with a positive x component is reversed.
    if t[0] > 0.0:
        t = -t
    return t, float(d[k])


def lane_mouths():
    """Lateral offset of each route's entrance, or None if the map has no routes."""
    p = ASSETS / "mountain_centre.npy"
    return np.load(p.as_posix()) if p.exists() else None


def spawn_pose(qpos, slope_rad, init_height, terrain_height, yaw_jitter=0.0,
               relief_span=None):
    """Place a robot on the slope: position, heading, body tilt, height.

    THE single definition of where the robot starts. Both joystick.py reset()
    and scripts/view.py call this, so the viewer cannot drift from the trainer
    -- which it did: the viewer applied the body tilt but no yaw at all, so it
    showed the robot facing a different way than training used, and none of the
    height corrections.

    qpos is a length-7+ array (position, quaternion, joints); only [0:7] is
    touched. terrain_height is a callable taking an xyz and returning relief
    above the mean plane, so this module needs no heightfield of its own.
    Returns a new qpos. numpy in, numpy out -- joystick.py wraps it in jax.
    """
    q = np.array(qpos, dtype=float).copy()
    if slope_rad == 0.0:
        return q
    n = slope_normal(slope_rad)
    q[0], q[1] = SPAWN

    # Heading first, about world +z, then the body tilt about +Y.
    yaw = SPAWN_YAW + yaw_jitter
    yq = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])
    q[3:7] = _quat_mul(q[3:7], yq)
    half = 0.5 * slope_rad
    tilt = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
    q[3:7] = _quat_mul(tilt, q[3:7])

    # Surface of the tilted mean plane, then the pelvis rise that keeps the
    # feet on it (rotating about the pelvis swings them down).
    q[2] += surface_z(q[0], slope_rad)
    q[2] += init_height * (1.0 - np.cos(slope_rad))
    # The terrain lift is applied by the CALLER, because under jax.jit the
    # heightfield lookup returns a tracer that cannot be turned into a float
    # here. spawn_lift() below is the shared definition of that step; both
    # callers use it, one in numpy and one in jax.
    if terrain_height is not None:
        q[0:3] += spawn_lift_relief(q[0:3], terrain_height, relief_span) * n
        q[2] += SPAWN_CLEARANCE
    return q


def spawn_lift_relief(xyz, terrain_height, relief_span=None):
    """Highest terrain relief in the probe grid around a point.

    Separate from spawn_pose so a jax caller can supply a traced
    terrain_height and combine the result itself. Keep the grid and the
    max-reduction here so the two callers cannot disagree about them.
    """
    vals = [terrain_height(np.asarray(xyz) + o)
            for o in probe_offsets(relief_span)]
    return max(float(v) for v in vals)


def _quat_mul(a, b):
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])


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
