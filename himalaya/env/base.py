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
# _src/locomotion/g1/base.py, Apache-2.0. See LICENSE.playground.
#
# Modified: imports rewired to himalaya.env so this file is OURS to edit.
# The reward terms, observations, and termination in here are the things this
# project changes; owning the file means editing them directly instead of
# subclassing or monkeypatching a library class. Monkeypatching in particular
# does not survive jax.jit tracing -- it fails silently during training.
# ==============================================================================
"""Base classes for G1."""

import math
from pathlib import Path
from typing import Any

import jax
import mujoco
import numpy as np
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from himalaya.env import g1_constants as consts


def get_assets() -> dict[str, bytes]:
  assets = {}
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "assets")
  path = mjx_env.MENAGERIE_PATH / "unitree_g1"
  if path.exists():
    mjx_env.update_assets(assets, path, "*.xml")
    mjx_env.update_assets(assets, path / "assets")
  else:
    # MODIFIED: Playground wheels do not always bundle external_deps. Fall
    # back to the repository's licensed Menagerie copy instead of requiring a
    # source checkout of Playground.
    candidates = [
        Path(__file__).resolve().parents[2] / "assets" / "unitree_g1",
        Path(__file__).resolve().parents[1] / "assets" / "unitree_g1",
    ]
    vendored = next(
        (candidate for candidate in candidates if (candidate / "assets").is_dir()),
        candidates[0],
    )
    if not (vendored / "assets").is_dir():
      raise FileNotFoundError(
          "Unitree G1 assets are missing from Playground and assets/unitree_g1"
      )
    for asset in (vendored / "assets").iterdir():
      if asset.is_file():
        assets[asset.name] = asset.read_bytes()
    for xml in vendored.glob("*.xml"):
      assets[xml.name] = xml.read_bytes()
  return assets


class G1Env(mjx_env.MjxEnv):
  """Base class for G1 environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: dict[str, str | int | list[Any]] | None = None,
  ) -> None:
    super().__init__(config, config_overrides)

    self._model_assets = get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt

    if self._config.get("climb", {}).get("enabled", False):
      # MODIFIED: configure the climb scene before converting it to MJX. This
      # keeps grade, roughness, and all four spike contact pairs inside the
      # model that Brax actually traces and steps.
      angle = math.radians(float(self._config.climb.slope_degrees))
      floor_id = self._mj_model.geom("floor").id
      floor_pos = self._mj_model.geom_pos[floor_id].copy()
      reference_floor_quat = self._mj_model.geom_quat[floor_id].copy()
      reference_angle = -2.0 * math.atan2(
          reference_floor_quat[2], reference_floor_quat[0]
      )
      self._mj_model.geom_quat[floor_id] = [
          math.cos(-angle / 2), 0.0, math.sin(-angle / 2), 0.0
      ]
      # MODIFIED: the crouch was solved on the XML's reference ramp. Rotate
      # and translate its floating root with the configured ramp about the
      # floor origin, preserving four-point reset distances at every grade.
      delta = angle - reference_angle
      alignment_quat = np.array([
          math.cos(-delta / 2), 0.0, math.sin(-delta / 2), 0.0
      ])
      alignment_rotation = np.empty(9)
      mujoco.mju_quat2Mat(alignment_rotation, alignment_quat)
      alignment_rotation = alignment_rotation.reshape(3, 3)
      for key_id in range(self._mj_model.nkey):
        root_pos = self._mj_model.key_qpos[key_id, :3].copy()
        self._mj_model.key_qpos[key_id, :3] = (
            floor_pos + alignment_rotation @ (root_pos - floor_pos)
        )
        root_quat = self._mj_model.key_qpos[key_id, 3:7].copy()
        aligned_quat = np.empty(4)
        mujoco.mju_mulQuat(aligned_quat, alignment_quat, root_quat)
        self._mj_model.key_qpos[key_id, 3:7] = aligned_quat
      self._mj_model.hfield_size[0, 2] = float(self._config.climb.roughness_m)
      spike_pair_ids = [
          pair_id
          for pair_id in range(self._mj_model.npair)
          if self._mj_model.pair(pair_id).name in {
              "left_foot_floor",
              "right_foot_floor",
              "left_hand_floor",
              "right_hand_floor",
          }
          or "_boulder_" in self._mj_model.pair(pair_id).name
      ]
      for pair_id in spike_pair_ids:
        self._mj_model.pair_dim[pair_id] = 6
        pair_name = self._mj_model.pair(pair_id).name
        is_foot_pair = pair_name.startswith(("left_foot_", "right_foot_"))
        friction = float(
            self._config.climb.foot_spike_friction
            if is_foot_pair
            else self._config.climb.spike_friction
        )
        self._mj_model.pair_friction[pair_id] = [
            friction, friction, .08, .03, .03
        ]
        self._mj_model.pair_solref[pair_id] = [.008, 1.0]
        self._mj_model.pair_solimp[pair_id, :3] = [.95, .99, .001]

      # MODIFIED: place each sphere on the actual sampled height field using
      # its compiled radius, so rock-size edits cannot leave it buried/floating.
      floor_quat = self._mj_model.geom_quat[floor_id]
      rotation = np.empty(9)
      mujoco.mju_quat2Mat(rotation, floor_quat)
      rotation = rotation.reshape(3, 3)
      half_x, half_y, height_scale = self._mj_model.hfield_size[0, :3]
      rows = self._mj_model.hfield_nrow[0]
      cols = self._mj_model.hfield_ncol[0]
      heights = self._mj_model.hfield_data.reshape(rows, cols)
      for index in range(10):
        boulder_id = self._mj_model.geom(f"boulder_{index:02d}").id
        if not self._config.climb.boulders_enabled:
          # MODIFIED: the crawl bootstrap removes discrete obstacles while
          # retaining the exact same model/pair topology for checkpoint reuse.
          self._mj_model.geom_pos[boulder_id, 2] = -10.0
          continue
        local_x, local_y = self._mj_model.geom_pos[boulder_id, :2]
        col = int(np.clip(round((local_x / half_x + 1) * .5 * (cols - 1)), 0, cols - 1))
        row = int(np.clip(round((local_y / half_y + 1) * .5 * (rows - 1)), 0, rows - 1))
        radius = self._mj_model.geom_size[boulder_id, 0]
        local_z = heights[row, col] * height_scale + radius
        self._mj_model.geom_pos[boulder_id] = (
            floor_pos + rotation @ np.array([local_x, local_y, local_z])
        )

    if self._config.restricted_joint_range:
      self._mj_model.jnt_range[1:] = consts.RESTRICTED_JOINT_RANGE
      self._mj_model.actuator_ctrlrange[:] = consts.RESTRICTED_JOINT_RANGE

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path

  # Sensor readings.

  def get_gravity(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the gravity vector in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GRAVITY_SENSOR}_{frame}"
    )

  def get_global_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the linear velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GLOBAL_LINVEL_SENSOR}_{frame}"
    )

  def get_global_angvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the angular velocity of the robot in the world frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GLOBAL_ANGVEL_SENSOR}_{frame}"
    )

  def get_local_linvel(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the linear velocity of the robot in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.LOCAL_LINVEL_SENSOR}_{frame}"
    )

  def get_accelerometer(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the accelerometer readings in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.ACCELEROMETER_SENSOR}_{frame}"
    )

  def get_gyro(self, data: mjx.Data, frame: str) -> jax.Array:
    """Return the gyroscope readings in the local frame."""
    return mjx_env.get_sensor_data(
        self.mj_model, data, f"{consts.GYRO_SENSOR}_{frame}"
    )

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return self._mjx_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
