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
# _src/locomotion/g1/g1_constants.py, Apache-2.0. See LICENSE.playground.
#
# Modified: imports rewired to himalaya.env so this file is OURS to edit.
# The reward terms, observations, and termination in here are the things this
# project changes; owning the file means editing them directly instead of
# subclassing or monkeypatching a library class. Monkeypatching in particular
# does not survive jax.jit tracing -- it fails silently during training.
# ==============================================================================
"""Constants for G1."""

from etils import epath

from mujoco_playground._src import mjx_env

# MODIFIED: was mjx_env.ROOT_PATH / "locomotion" / "g1", i.e. the copy inside
# site-packages. Points at our own xmls/ so terrain, sensors, and collision
# geometry are editable here rather than in the library.
#
# The mesh paths inside those XMLs still read "../../../mujoco_menagerie/...",
# which looks broken but is not: base.py loads every asset by NAME into a dict
# (see update_assets), pulling the STLs from MENAGERIE_PATH separately, so the
# relative strings are never resolved against the filesystem.
ROOT_PATH = epath.Path(__file__).parent
FEET_ONLY_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"
)
FEET_ONLY_ROUGH_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
)
# Same rough heightfield, but the floor geom carries a quat that gets overwritten
# at load time from --slope-deg. Rough rather than a smooth plane on purpose: on a
# smooth slope four limbs buy nothing (friction is mu*W*cos(a) however many
# contacts share it, so the requirement stays mu >= tan(a) for two limbs or four).
# Hands only help where there is something to brace against, and the hfield's 5 cm
# bumps are that.
FEET_ONLY_SLOPE_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_slope_terrain.xml"
)


def task_to_xml(task_name: str) -> epath.Path:
  return {
      "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
      "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
      "slope_terrain": FEET_ONLY_SLOPE_TERRAIN_XML,
  }[task_name]


FEET_SITES = [
    "left_foot",
    "right_foot",
]

HAND_SITES = [
    "left_palm",
    "right_palm",
]

LEFT_FEET_GEOMS = ["left_foot"]
RIGHT_FEET_GEOMS = ["right_foot"]
FEET_GEOMS = LEFT_FEET_GEOMS + RIGHT_FEET_GEOMS

ROOT_BODY = "torso_link"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"

RESTRICTED_JOINT_RANGE = (
    # Left leg.
    (-1.57, 1.57),
    (-0.5, 0.5),
    (-0.7, 0.7),
    (0, 1.57),
    (-0.4, 0.4),
    (-0.2, 0.2),
    # Right leg.
    (-1.57, 1.57),
    (-0.5, 0.5),
    (-0.7, 0.7),
    (0, 1.57),
    (-0.4, 0.4),
    (-0.2, 0.2),
    # Waist.
    (-2.618, 2.618),
    (-0.52, 0.52),
    (-0.52, 0.52),
    # Left shoulder.
    (-3.0892, 2.6704),
    (-1.5882, 2.2515),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
    (-1.61443, 1.61443),
    (-1.61443, 1.61443),
    # Right shoulder.
    (-3.0892, 2.6704),
    (-2.2515, 1.5882),
    (-2.618, 2.618),
    (-1.0472, 2.0944),
    (-1.97222, 1.97222),
    (-1.61443, 1.61443),
    (-1.61443, 1.61443),
)
