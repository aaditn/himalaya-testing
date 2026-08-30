"""Load the 29-DOF Unitree G1 from MuJoCo Menagerie.

Menagerie ships this model simulation-ready -- floating base, floor, actuators,
tuned gains and armature -- so there is nothing to build. It arrives vendored
inside MuJoCo Playground under external_deps/, which is also where Playground's
own G1 tasks get it, so training and inspection share one model.

Measured from the model itself, not assumed (see scripts/inspect_model.py):
  actuated joints    29
  total mass         33.34 kg
  standing height     0.784 m   <- keyframe qpos[2] of scene_mjx.xml

29 DOF = 12 leg + 3 waist + 14 arm (7 per arm). The 3-DOF waist is the
difference that matters against the 23-DOF variant this repo used before:
that one had waist yaw only, so torso balance authority was limited.

Two scenes are available, and they are not interchangeable.

scene_mjx.xml is what training uses: simplified colliders, 5 solver iterations,
4 ms timestep -- tuned for thousands of parallel rollouts where a policy closes
the loop every step. It will NOT hold a pose open-loop. Holding the standing
keyframe with no policy drops the robot to z=0.11 within 1.5 s, which is a
solver artifact, not a broken model.

scene.xml is the full-fidelity model: 100 solver iterations, 2 ms timestep.
The same open-loop test stands at z=0.792, upright=1.000. Use it for anything
that judges the physics rather than trains a policy -- inspection, viewing,
and any measurement quoted as a fact about the robot.
"""

import mujoco
from etils import epath

# Measured, not copied. Re-run scripts/inspect_model.py after any model change.
STANDING_HEIGHT = 0.7837
N_JOINTS = 29
TOTAL_MASS = 33.34


def menagerie_root() -> epath.Path:
    """Directory of Menagerie's unitree_g1, as vendored inside Playground."""
    import mujoco_playground

    return (
        epath.Path(mujoco_playground.__file__).parent
        / "external_deps"
        / "mujoco_menagerie"
        / "unitree_g1"
    )


def scene_path(mjx: bool = True) -> str:
    """Path to a G1 scene XML.

    mjx=True gives the MJX-tuned scene (simplified colliders), which is what
    training uses. mjx=False gives the full-fidelity scene, for viewing.
    """
    return str(menagerie_root() / ("scene_mjx.xml" if mjx else "scene.xml"))


def load(mjx: bool = True):
    """Compile and return (model, default_qpos, default_ctrl).

    Both come from Menagerie's standing keyframe, so the robot starts upright
    rather than in a zero pose it would immediately fall out of. default_ctrl
    is the keyframe's own actuator targets -- these are position servos, so
    ctrl is a joint angle, and reconstructing it from qpos instead of reading
    key_ctrl silently gets the arms wrong on the bent-knee keyframes.
    """
    model = mujoco.MjModel.from_xml_path(scene_path(mjx=mjx))
    if model.nkey > 0:
        return model, model.key_qpos[0].copy(), model.key_ctrl[0].copy()

    import numpy as np

    qpos = np.zeros(model.nq)
    qpos[2] = STANDING_HEIGHT
    qpos[3] = 1.0  # identity quaternion (w,x,y,z)
    return model, qpos, np.zeros(model.nu)
