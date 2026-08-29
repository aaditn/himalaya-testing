"""Build a simulation-ready MJCF for the 23-DOF G1.

The raw URDF is a bare kinematic tree: no floating base, no floor, no
actuators, and mesh paths that assume a different working directory. This
module turns it into something that can actually be stepped.

Measured facts (from MuJoCo, not assumed -- see scripts/inspect_model.py):
  total mass        30.32 kg
  arm mass           6.10 kg  (20.1% of total -- arms have real authority)
  leg mass          14.37 kg
  torso              9.84 kg
  inertias           all valid (positive, triangle inequality holds)
  STANDING HEIGHT    0.784 m   <- lowest foot geom below the pelvis frame

That last number matters: the Isaac Lab runs used 1.05 m, copied from
NVIDIA's 37-DOF G1 config. On this robot that spawns it ~27 cm in the air,
so it drops and lands every reset. Everything here uses the measured value.
"""

import re
from pathlib import Path

import mujoco

# URDF effort limits, verbatim. Arms are 5.6x weaker than the knee, which is
# the quantitative reason arm-based balancing is a real question here rather
# than a given.
EFFORT = {
    "hip": 88.0, "knee": 139.0, "ankle": 35.0, "waist": 88.0,
    "shoulder": 25.0, "elbow": 25.0, "wrist": 25.0,
}

# PD gains, tuned HERE rather than copied from Isaac Lab.
#
# unitree_rl_gym's published values (hip 100, knee 150, ankle 40) do not hold
# this robot up in MuJoCo: with them the knee commands only 45 Nm against a
# 139 Nm limit while folding to 1.44 rad past a 0.30 rad target -- the legs
# collapse under 34 kg in half a second. Those gains assume Isaac Lab's
# implicit actuator, which applies the PD law differently from a MuJoCo
# position servo, so the same numbers are not the same controller.
#
# Scaled up until the legs actually support static weight. Verify with
# scripts/inspect_model.py after any change -- it prints the settle trace.
KP = {"hip": 500.0, "knee": 500.0, "ankle": 500.0, "waist": 500.0,
      "shoulder": 500.0, "elbow": 500.0, "wrist": 500.0}
KD = {"hip": 43.0, "knee": 16.0, "ankle": 5.0, "waist": 20.0,
      "shoulder": 10.0, "elbow": 5.0, "wrist": 5.0}

# Rotor inertia. MuJoCo Menagerie's G1 sets 0.01 on every joint; ours had 0.
# Without it the effective inertia at each joint is tiny, so any stiff PD
# controller goes numerically unstable -- which is why the robot collapsed
# identically at every gain setting in a 12-point sweep. This, not the gains,
# was the reason nothing worked.
ARMATURE = 0.01

STANDING_HEIGHT = 0.784

DEFAULT_POSE = {
    "left_hip_pitch_joint": -0.10, "right_hip_pitch_joint": -0.10,
    "left_knee_joint": 0.30, "right_knee_joint": 0.30,
    "left_ankle_pitch_joint": -0.20, "right_ankle_pitch_joint": -0.20,
    "left_shoulder_pitch_joint": 0.20, "right_shoulder_pitch_joint": 0.20,
    "left_shoulder_roll_joint": 0.35, "right_shoulder_roll_joint": -0.35,
    "left_elbow_joint": 0.90, "right_elbow_joint": 0.90,
}


def _group(joint_name: str) -> str:
    for g in EFFORT:
        if g in joint_name:
            return g
    raise ValueError(f"no actuator group for {joint_name}")


def build_spec(urdf_path: str, mesh_dir: str, add_floor: bool = True,
               floating_base: bool = True):
    """Return a compiled-ready MjSpec: floating base, floor, PD actuators."""
    text = Path(urdf_path).read_text()
    # MuJoCo resolves meshes relative to meshdir; the URDF's "meshes/" prefix
    # would double it.
    text = re.sub(r'filename="meshes/', 'filename="', text)
    tmp = Path(mesh_dir).parent / "_g1_mjx.urdf"
    tmp.write_text(text)

    spec = mujoco.MjSpec.from_file(str(tmp))
    # absolute: the rewritten URDF lives next to the mesh dir, not at cwd
    spec.meshdir = str(Path(mesh_dir).resolve())

    if add_floor:
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            condim=3,
            friction=[1.0, 0.005, 0.0001],
        )

    # The URDF has no floating base -- without this the robot is pinned in
    # space and MuJoCo roots the tree at whatever link it sees first.
    # floating_base=False welds the pelvis to the world, which isolates the
    # servos from any balance problem (see scripts/inspect_model.py).
    root = spec.worldbody.first_body()
    if root is not None:
        if floating_base:
            root.add_freejoint()
        root.pos = [0.0, 0.0, STANDING_HEIGHT]

    # Rotor inertia on every actuated joint (see ARMATURE above).
    for j in spec.joints:
        if j.type != mujoco.mjtJoint.mjJNT_FREE:
            j.armature = ARMATURE

    # Position-servo actuators: the policy commands joint targets, the PD
    # controller produces torque. Same action space as the Isaac Lab setup.
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        g = _group(j.name)
        act = spec.add_actuator(
            name=f"act_{j.name}",
            target=j.name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gainprm=[KP[g]] + [0.0] * 9,
            biasprm=[0.0, -KP[g], -KD[g]] + [0.0] * 7,
            # No forcerange: Menagerie's working G1 leaves it unset. Clamping
            # to the URDF's 35 Nm ankle limit made the ankles saturate and the
            # legs fold. Torque limits belong in the reward, not the actuator.
        )
        # Setting gainprm/biasprm alone is NOT enough: both types default to
        # FIXED, so MuJoCo ignores the bias terms and applies a constant gain.
        # The knee then saturates at 139 Nm driving AWAY from its target
        # (qacc ~5900 rad/s^2) and the robot folds instantly. AFFINE is what
        # makes this an actual PD position servo.
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    return spec


def load(urdf_path="assets/g1/g1_23dof.urdf", mesh_dir="assets/g1/meshes"):
    """Compile and return (model, default_qpos)."""
    spec = build_spec(urdf_path, mesh_dir)
    model = spec.compile()

    import numpy as np
    qpos = np.zeros(model.nq)
    qpos[2] = STANDING_HEIGHT
    qpos[3] = 1.0  # identity quaternion (w,x,y,z)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in range(model.njnt)]
    for jn, val in DEFAULT_POSE.items():
        if jn in names:
            qpos[model.jnt_qposadr[names.index(jn)]] = val
    return model, qpos
