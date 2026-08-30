# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# ==============================================================================
# Vendored from MuJoCo Playground (google-deepmind/mujoco_playground),
# _src/locomotion/g1/randomize.py, Apache-2.0. See LICENSE.playground.
#
# Modified: imports rewired to himalaya.env so this file is OURS to edit.
# The reward terms, observations, and termination in here are the things this
# project changes; owning the file means editing them directly instead of
# subclassing or monkeypatching a library class. Monkeypatching in particular
# does not survive jax.jit tracing -- it fails silently during training.
# ==============================================================================
"""Utilities for randomization."""
import jax
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 16


def domain_randomize(model: mjx.Model, rng: jax.Array):
  @jax.vmap
  def rand_dynamics(rng):
    # Floor contact friction: =U(0.4, 1.0).
    #
    # MODIFIED: slice widened 0:2 -> 0:4 to cover the two hand-floor pairs
    # added alongside the feet. It is the same rock under both, so they share
    # one sampled friction rather than the hands staying pinned at the XML
    # default while the feet vary. Indices are POSITIONAL over the <pair>
    # elements: 0,1 are left/right foot-floor and 2,3 are left/right
    # hand-floor, so new pairs must be appended after these, never inserted.
    rng, key = jax.random.split(rng)
    friction = jax.random.uniform(key, minval=0.4, maxval=1.0)
    # 0:4 covers both feet and both hands against the floor. MuJoCo SORTS
    # <pair> elements, so these indices do not follow XML order -- verify
    # against mj_id2name after any pair change.
    pair_friction = model.pair_friction.at[0:4, 0:2].set(friction)

    # Scale static friction: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
        key, shape=(29,), minval=0.5, maxval=2.0
    )
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

    # Scale armature: *U(1.0, 1.05).
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[6:] * jax.random.uniform(
        key, shape=(29,), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[6:].set(armature)

    # Scale all link masses: *U(0.9, 1.1).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(model.nbody,), minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

    # Add mass to torso: +U(-1.0, 1.0).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + dmass
    )

    # Jitter qpos0: +U(-0.05, 0.05).
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[7:].set(
        qpos0[7:]
        + jax.random.uniform(key, shape=(29,), minval=-0.05, maxval=0.05)
    )

    return (
        pair_friction,
        dof_frictionloss,
        dof_armature,
        body_mass,
        qpos0,
    )

  (
      pair_friction,
      frictionloss,
      armature,
      body_mass,
      qpos0,
  ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "pair_friction": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "body_mass": 0,
      "qpos0": 0,
  })

  model = model.tree_replace({
      "pair_friction": pair_friction,
      "dof_frictionloss": frictionloss,
      "dof_armature": armature,
      "body_mass": body_mass,
      "qpos0": qpos0,
  })

  return model, in_axes
