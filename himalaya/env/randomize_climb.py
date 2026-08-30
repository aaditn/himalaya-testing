# ==============================================================================
# Domain randomisation for the climbing task. Ours, not vendored.
#
# Differs from randomize.py (walking) in two ways that matter:
#
#  1. IT RANDOMISES THE TERRAIN ITSELF. The slope angle is drawn per environment
#     by rotating the floor geom, so one policy sees the whole 30-45 deg band.
#
#  2. IT TAKES PAIR INDICES BY ARGUMENT, NEVER BY HARDCODED SLICE.
#     randomize.py writes pair_friction[0:2], which happens to be the two
#     foot-floor pairs in the walking scenes. In the climb scenes MuJoCo sorts
#     the pair table differently -- the feet land at 2 and 5, and 0:2 is
#     (left_thigh_floor, left_shin_floor). Reusing that slice would randomise
#     the thigh and shin while the feet kept a fixed friction, and nothing would
#     error: training curves would look entirely normal. The indices are
#     therefore resolved from the model BY NAME in train_climb.py and bound with
#     functools.partial.
# ==============================================================================
"""Domain randomisation for quadrupedal slope climbing."""

import jax
import jax.numpy as jp
from mujoco import mjx

TORSO_BODY_ID = 16


def domain_randomize(
    model: mjx.Model,
    rng: jax.Array,
    *,
    floor_geom_id: int,
    ground_pair_ids: jax.Array,
    slope_deg=(30.0, 45.0),
    friction=(2.0, 3.0),
):
  """Randomise slope angle, slipperiness, and the usual dynamics.

  FRICTION IS BOUNDED BELOW BY THE PHYSICS OF THE SLOPE, not by taste.
  Holding station needs mu >= tan(theta): 0.58 at 30 deg, 1.00 at 45 deg.

  The first version drew mu in [0.6, 1.4] deliberately straddling that line, on
  the theory that it made a difficulty gradient. It does not. It makes a large
  subset with NO gradient: at 45 deg half those draws are unclimbable by any
  policy, about a quarter across the whole band. PPO optimises the average, so
  the best available behaviour was "descend slowly without falling" -- and that
  is exactly what 11.5M steps produced. Measured on the resulting policy at
  37.5 deg and mu=1.2: -2.70 m net travel DOWN the slope in 6 s, zero falls,
  while episode length rose 92 -> 450 and reward/step improved throughout.

  [2.0, 3.0] is the IDEAL-CONDITIONS band: rubber-on-rock grip, double
  tan(45 deg) at the bottom of the range, so no drawn environment is ever
  friction-limited and the policy's whole problem is posture and propulsion.
  The earlier [1.1, 2.0] band cleared feasibility but left almost no margin at
  the steep end; if a trained policy should later tolerate slick rock, tighten
  this band back down as a curriculum, not at first training.
  """

  @jax.vmap
  def rand(rng):
    # --- slope angle: rotate the floor geom about +y by -theta ---
    rng, key = jax.random.split(rng)
    theta = jp.deg2rad(
        jax.random.uniform(key, minval=slope_deg[0], maxval=slope_deg[1])
    )
    # quat for R_y(-theta), MuJoCo order (w, x, y, z)
    quat = jp.array([jp.cos(theta / 2), 0.0, -jp.sin(theta / 2), 0.0])
    geom_quat = model.geom_quat.at[floor_geom_id].set(quat)

    # --- slipperiness, on every limb-vs-floor pair, by index ---
    rng, key = jax.random.split(rng)
    mu = jax.random.uniform(key, minval=friction[0], maxval=friction[1])
    pair_friction = model.pair_friction.at[ground_pair_ids, 0:2].set(mu)

    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
        key, shape=(29,), minval=0.5, maxval=2.0
    )
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

    rng, key = jax.random.split(rng)
    armature = model.dof_armature[6:] * jax.random.uniform(
        key, shape=(29,), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[6:].set(armature)

    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, shape=(model.nbody,), minval=0.9, maxval=1.1)
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

    rng, key = jax.random.split(rng)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + jax.random.uniform(key, minval=-1.0, maxval=1.0)
    )

    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0.at[7:].set(
        model.qpos0[7:]
        + jax.random.uniform(key, shape=(29,), minval=-0.05, maxval=0.05)
    )
    return geom_quat, pair_friction, dof_frictionloss, dof_armature, body_mass, qpos0

  geom_quat, pair_friction, frictionloss, armature, body_mass, qpos0 = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_quat": 0,
      "pair_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
  })
  model = model.tree_replace({
      "geom_quat": geom_quat,
      "pair_friction": pair_friction,
      "dof_frictionloss": frictionloss,
      "dof_armature": armature,
      "body_mass": body_mass,
      "qpos0": qpos0,
  })
  return model, in_axes


def ground_pair_ids(mj_model) -> jax.Array:
  """Indices of every limb-vs-floor pair, resolved BY NAME.

  Do not replace this with a slice. Adding the leg pairs moved the feet from
  (0,1) to (2,5); a slice would silently randomise the wrong geoms.
  """
  names = [
      "left_foot_floor", "right_foot_floor",
      "left_hand_floor", "right_hand_floor",
      "left_thigh_floor", "right_thigh_floor",
      "left_shin_floor", "right_shin_floor",
  ]
  ids = []
  for n in names:
    pid = mj_model.pair(n).id
    ids.append(pid)
  return jp.array(sorted(ids))
