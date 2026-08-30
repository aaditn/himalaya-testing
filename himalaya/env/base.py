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
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from himalaya.env import g1_constants as consts
from himalaya.env import scene as scene_mod


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
    xml_text = epath.Path(xml_path).read_text()

    # MODIFIED: tilt the floor for the climbing task, BEFORE compiling.
    #
    # This has to be a text substitution rather than a post-load write to
    # mj_model.geom_quat. MuJoCo bakes a worldbody geom's orientation into
    # geom_xmat at compile time, so assigning geom_quat afterwards updates the
    # stored field and nothing else: geom_xmat -- which is what the collider
    # actually reads -- stays identity. Measured, 45 deg about +Y:
    #     XML-baked     geom_xmat normal [0.707 0 0.707]   (a real slope)
    #     post-compile  geom_xmat normal [0     0 1    ]   (flat)
    # The failure is silent. Contact sensors still fire, episodes still run,
    # and the robot simply walks on level ground while every log says 45 deg.
    self._slope_rad = float(np.deg2rad(self._config.slope_deg))
    if self._slope_rad != 0.0:
        if 'quat="1 0 0 0"' not in xml_text:
            raise ValueError(
                f"{xml_path} has no floor quat placeholder to substitute; "
                "slope_deg requires a scene with quat=\"1 0 0 0\" on its floor geom"
            )
        # scene.py owns the substitution so the local viewer tilts the floor
        # exactly the way training does.
        q = scene_mod.slope_quat(self._slope_rad)
        xml_text = xml_text.replace(
            'quat="1 0 0 0"', f'quat="{q[0]} 0 {q[2]} 0"', 1)

    self._mj_model = mujoco.MjModel.from_xml_string(
        xml_text, assets=self._model_assets
    )
    self._mj_model.opt.timestep = self.sim_dt

    if self._config.restricted_joint_range:
      self._mj_model.jnt_range[1:] = consts.RESTRICTED_JOINT_RANGE
      self._mj_model.actuator_ctrlrange[:] = consts.RESTRICTED_JOINT_RANGE

    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    # Surface normal and up-gradient direction, both in world coords. Derived
    # from the same angle as the quat substituted above, so they cannot drift.
    # One slope per run: the normal is a compile-time constant, free under jit,
    # and every metric in a run is attributable to one angle.
    self._slope_normal = scene_mod.slope_normal(self._slope_rad)
    # Rotating about +Y drops the surface along +X, so up-gradient is -X.
    # (Verified: uphill.z must be POSITIVE -- a downhill vector here would make
    # progress_uphill pay the policy for sliding to the bottom.)
    self._uphill = scene_mod.uphill(self._slope_rad)

    # Highest point of the terrain above its own mean plane. reset() lifts the
    # spawn by this so the robot starts above the rock rather than inside it.
    # A smooth-plane correction is not enough once the heightfield carries real
    # relief: measured on the 2.0 m mountain terrain, every foot spawned an
    # average of 1.55 m underground.
    # Uphill edge of the starting platform, in world x. Read from the geom so
    # it cannot drift from the XML. None when the scene has no platform.
    self._platform_x = None
    self._platform_top = 0.0
    try:
      pid = self._mj_model.geom("start_platform").id
      self._platform_x = float(
          self._mj_model.geom_pos[pid][0] - self._mj_model.geom_size[pid][0]
      )
      self._platform_top = float(
          self._mj_model.geom_pos[pid][2] + self._mj_model.geom_size[pid][2]
      )
    except KeyError:
      pass

    # Lane centreline, if the terrain ships one. Used to reward progress ALONG
    # the route rather than straight uphill: the lane wanders 4.4 m laterally
    # at a median 38 degrees off the uphill axis, so a plain uphill projection
    # pays only cos(38) for following the corridor and makes charging straight
    # up the wall the better-paid option.
    # Lateral position of each route's mouth. reset() picks one at random, so
    # the policy meets a different corridor every episode. The routes differ
    # from each other, so one that memorised a single path fails on the next.
    self._lane = None
    if self._mj_model.nhfield > 0:
      import numpy as _np
      self._lane = scene_mod.lane_mouths()

    # Route tangent field, precomputed on a coarse grid.
    #
    # progress_uphill rewards dot(velocity, world_up), i.e. height gained. On
    # this map that is the WRONG objective: the corridors run roughly across
    # the fall line (measured at the spawn, low ground bears 90-120 degrees
    # while every other heading meets a 0.4-0.8 m bank), so the fastest way to
    # gain height is to charge straight at a bank and the reward pays most for
    # leaving the route. Two runs drifted downhill under it.
    #
    # This field lets the reward measure progress ALONG the corridor instead.
    # Sampling a precomputed grid keeps it a constant under jit -- the route
    # never moves, so there is nothing to recompute per step.
    self._route_dir = None
    lines = scene_mod.route_lines()
    if lines is not None and self._slope_rad != 0.0:
      gn = 48
      half = 6.0
      xs = np.linspace(-half, half, gn)
      ys = np.linspace(-half, half, gn)
      field = np.zeros((gn, gn, 2))
      for i, yy in enumerate(ys):
        for j, xx in enumerate(xs):
          t, _ = scene_mod.route_tangent(lines, np.array([xx, yy]))
          field[i, j] = t
      self._route_dir = jp.array(field)
      self._route_grid = (float(half), gn)

    # World "up". Height gain along this is the climbing objective -- see
    # _reward_progress_uphill. Kept here so the reward needs to know nothing
    # about where the route goes.
    self._slope_normal_up = np.array([0.0, 0.0, 1.0])

    # Heightfield relief, kept as a grid so reset() can look up the height at
    # the SPAWN POINT instead of assuming the worst case.
    #
    # This used to be a single number, hf.max(), added to every spawn. That is
    # 1.10 m on this terrain while the mean relief is 0.34 m, so the average
    # episode began 0.76 m in the air and the flattest ones a full 1.10 m up --
    # about the robot's own standing height, dropped onto a 35 degree slope
    # before the policy had taken an action. It was written when the terrain had
    # 2.0 m of relief and the alternative was spawning 1.55 m underground; the
    # blanket maximum solved that at the cost of turning every reset into a fall.
    self._terrain_peak = 0.0
    self._hfield_grid = None
    if self._mj_model.nhfield > 0:
      nr = int(self._mj_model.hfield_nrow[0])
      nc = int(self._mj_model.hfield_ncol[0])
      z_top = float(self._mj_model.hfield_size[0][2])
      hf = np.array(self._mj_model.hfield_data[: nr * nc])
      self._terrain_peak = float(hf.max()) * z_top
      # (nrow, ncol) in metres. MuJoCo stores hfield_data row-major with rows
      # along local y and columns along local x, spanning [-size_x, +size_x]
      # and [-size_y, +size_y] about the geom origin.
      self._hfield_grid = jp.array(hf.reshape(nr, nc) * z_top)
      self._hfield_z_top = z_top
      self._hfield_half = (
          float(self._mj_model.hfield_size[0][0]),
          float(self._mj_model.hfield_size[0][1]),
      )
      # Rotation from world into the floor geom's local frame. The geom sits at
      # the origin and its local z axis IS the slope normal (verified against
      # _slope_normal), so relief stacks along the normal, not along world z.
      gid = self._mj_model.geom("floor").id
      d = mujoco.MjData(self._mj_model)
      mujoco.mj_forward(self._mj_model, d)
      self._floor_xmat = jp.array(d.geom_xmat[gid].reshape(3, 3))

    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._xml_path = xml_path

  def route_dir_at(self, world_xyz: jax.Array) -> jax.Array:
    """Unit xy heading of the corridor at a world point. Zero if no routes.

    Nearest-cell lookup into a precomputed field, so it costs one gather and
    traces cleanly under jit.
    """
    if self._route_dir is None:
      return jp.zeros(2)
    half, gn = self._route_grid
    fx = jp.clip((world_xyz[0] + half) / (2 * half) * (gn - 1), 0, gn - 1)
    fy = jp.clip((world_xyz[1] + half) / (2 * half) * (gn - 1), 0, gn - 1)
    return self._route_dir[jp.round(fy).astype(jp.int32),
                           jp.round(fx).astype(jp.int32)]

  def terrain_height_at(self, world_xyz: jax.Array,
                        hfield_data: jax.Array = None) -> jax.Array:
    """Relief above the mean plane at a world point, measured along the normal.

    Returns 0 when the scene has no heightfield. Bilinear, so a spawn between
    grid cells does not snap to a neighbouring rock's height.
    """
    if self._hfield_grid is None:
      return jp.zeros(())
    local = self._floor_xmat.T @ world_xyz
    hx, hy = self._hfield_half
    nr, nc = self._hfield_grid.shape
    # Map local x,y in [-half, +half] onto fractional grid indices.
    fx = (local[0] + hx) / (2.0 * hx) * (nc - 1)
    fy = (local[1] + hy) / (2.0 * hy) * (nr - 1)
    fx = jp.clip(fx, 0.0, nc - 1.0)
    fy = jp.clip(fy, 0.0, nr - 1.0)
    x0 = jp.floor(fx).astype(jp.int32)
    y0 = jp.floor(fy).astype(jp.int32)
    x1 = jp.minimum(x0 + 1, nc - 1)
    y1 = jp.minimum(y0 + 1, nr - 1)
    tx, ty = fx - x0, fy - y0
    # Read the LIVE heightfield when one is supplied.
    #
    # self._hfield_grid is captured from mj_model at __init__. Under per-env
    # randomization the physics steps mjx_model.hfield_data, which differs per
    # environment, while this cached copy never changes -- so the robot felt
    # one terrain and observed another. Measured: four different terrain
    # variants all produced an identical heightmap.
    if hfield_data is not None:
      g = hfield_data.reshape(self._hfield_grid.shape) * self._hfield_z_top
    else:
      g = self._hfield_grid
    top = g[y0, x0] * (1 - tx) + g[y0, x1] * tx
    bot = g[y1, x0] * (1 - tx) + g[y1, x1] * tx
    return top * (1 - ty) + bot * ty


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
