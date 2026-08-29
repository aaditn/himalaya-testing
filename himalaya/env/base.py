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

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from himalaya.env import g1_constants as consts


def get_assets() -> Dict[str, bytes]:
  assets = {}
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "assets")
  path = mjx_env.MENAGERIE_PATH / "unitree_g1"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  return assets


class G1Env(mjx_env.MjxEnv):
  """Base class for G1 environments."""

  def __init__(
      self,
      xml_path: str,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)

    self._model_assets = get_assets()
    self._mj_model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt

    if self._config.restricted_joint_range:
      self._mj_model.jnt_range[1:] = consts.RESTRICTED_JOINT_RANGE
      self._mj_model.actuator_ctrlrange[:] = consts.RESTRICTED_JOINT_RANGE

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    # MODIFIED: tilt the floor for the climbing task.
    #
    # One slope per run rather than per environment. geom_quat IS vmappable --
    # it is (ngeom, 4), the same per-geom class of field as body_mass that
    # domain_randomize already batches -- but a per-env angle would have to be
    # sampled in BOTH randomize.py (for the physics) and reset (for the reward's
    # uphill direction), and those two RNGs do not share state. They would
    # silently disagree. Fixing the slope per run makes the normal a
    # compile-time constant, free under jit, and every metric in a run
    # attributable to one angle.
    self._slope_rad = float(np.deg2rad(self._config.slope_deg))
    floor_gid = self._mj_model.geom("floor").id
    # Rotate about +Y so the slope rises along +X: uphill is +X.
    half = 0.5 * self._slope_rad
    self._mj_model.geom_quat[floor_gid] = [np.cos(half), 0.0, np.sin(half), 0.0]
    # Surface normal and up-gradient direction, both in world coords. Derived
    # from the same angle as the quat above, so they cannot drift apart.
    self._slope_normal = np.array(
        [np.sin(self._slope_rad), 0.0, np.cos(self._slope_rad)]
    )
    # Rotating about +Y drops the surface along +X, so up-gradient is -X.
    # (Verified: uphill.z must be POSITIVE -- a downhill vector here would make
    # progress_uphill pay the policy for sliding to the bottom.)
    self._uphill = np.array(
        [-np.cos(self._slope_rad), 0.0, np.sin(self._slope_rad)]
    )

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
