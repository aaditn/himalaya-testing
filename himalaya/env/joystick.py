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
# _src/locomotion/g1/joystick.py, Apache-2.0. See LICENSE.playground.
#
# Modified: imports rewired to himalaya.env so this file is OURS to edit.
# The reward terms, observations, and termination in here are the things this
# project changes; owning the file means editing them directly instead of
# subclassing or monkeypatching a library class. Monkeypatching in particular
# does not survive jax.jit tracing -- it fails silently during training.
# ==============================================================================
"""Joystick task for Unitree G1."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from himalaya.env import gait
from mujoco_playground._src import mjx_env
from himalaya.env import base as g1_base
from himalaya.env import g1_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      action_scale=0.5,
      history_len=1,
      restricted_joint_range=False,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Tracking related rewards.
              # MODIFIED: the climbing objective. Deliberately posture-blind --
              # it says "go up" and nothing about how. Zero on flat ground, so
              # every existing task is unchanged.
              progress_uphill=0.0,
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.75,
              # Base related rewards.
              lin_vel_z=0.0,
              ang_vel_xy=-0.15,
              orientation=-2.0,
              base_height=0.0,
              # Energy related rewards.
              torques=0.0,
              action_rate=0.0,
              energy=0.0,
              dof_acc=0.0,
              # Feet related rewards.
              feet_clearance=0.0,
              feet_air_time=2.0,
              feet_slip=-0.25,
              feet_height=0.0,
              feet_phase=1.0,
              # Other rewards.
              alive=0.0,
              stand_still=-1.0,
              termination=-100.0,
              collision=-0.1,
              contact_force=-0.01,
              # Pose related rewards.
              joint_deviation_knee=-0.1,
              joint_deviation_hip=-0.25,
              dof_pos_limits=-1.0,
              pose=-0.1,
          ),
          # Target climb rate for progress_uphill, m/s along the surface.
          max_uphill_speed=0.8,
          # Per-step cost of sitting on the starting platform, in the same
          # units as progress_uphill. Set below max_uphill_speed so climbing
          # is always the better option, but high enough that waiting is not
          # free.
          platform_loiter=0.5,
          tracking_sigma=0.25,
          max_foot_height=0.15,
          base_height_target=0.5,
          max_contact_force=500.0,
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 2.0],
      ),
      command_config=config_dict.create(
          # Uniform distribution for command amplitude.
          a=[1.0, 0.8, 1.0],
          # Probability of not zeroing out new command.
          b=[0.9, 0.25, 0.5],
      ),
      lin_vel_x=[-1.0, 1.0],
      lin_vel_y=[-0.5, 0.5],
      ang_vel_yaw=[-1.0, 1.0],
      # Floor tilt in degrees, applied to the floor geom at load time (see
      # base.py). 0.0 leaves every existing scene exactly as it was.
      slope_deg=0.0,
      # Half-width of the spawn xy jitter, metres. Wide on the climbing task
      # so the policy cannot memorise one patch of rock.
      spawn_jitter=0.5,
      impl="warp",
      naconmax=8 * 8192,
      njmax=29 * 2 + 8 * 4,
  )


class Joystick(g1_base.G1Env):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("knees_bent").qpos)
    self._default_pose = jp.array(
        self._mj_model.keyframe("knees_bent").qpos[7:]
    )

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    waist_indices = []
    waist_joint_names = [
        "waist_yaw",
        "waist_roll",
        "waist_pitch",
    ]
    for joint_name in waist_joint_names:
      waist_indices.append(
          self._mj_model.joint(f"{joint_name}_joint").qposadr - 7
      )
    self._waist_indices = jp.array(waist_indices)

    arm_indices = []
    arm_joint_names = [
        "shoulder_roll",
        "shoulder_yaw",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    ]
    for side in ["left", "right"]:
      for joint_name in arm_joint_names:
        arm_indices.append(
            self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7
        )
    self._arm_indices = jp.array(arm_indices)

    hip_indices = []
    hip_joint_names = [
        "hip_roll",
        "hip_yaw",
    ]
    for side in ["left", "right"]:
      for joint_name in hip_joint_names:
        hip_indices.append(
            self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7
        )
    self._hip_indices = jp.array(hip_indices)

    knee_indices = []
    knee_joint_names = ["knee"]
    for side in ["left", "right"]:
      for joint_name in knee_joint_names:
        knee_indices.append(
            self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7
        )
    self._knee_indices = jp.array(knee_indices)

    # fmt: off
    self._weights = jp.array([
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # left leg.
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # right leg.
        1.0, 1.0, 1.0,  # waist.
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left arm.
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right arm.
    ])
    # fmt: on

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._torso_imu_site_id = self._mj_model.site("imu_in_torso").id
    self._pelvis_imu_site_id = self._mj_model.site("imu_in_pelvis").id

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._hands_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.HAND_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._cmd_a = jp.array(self._config.command_config.a)
    self._cmd_b = jp.array(self._config.command_config.b)

    self._left_hand_geom_id = self._mj_model.geom("left_hand_collision").id
    self._right_hand_geom_id = self._mj_model.geom("right_hand_collision").id
    self._left_foot_geom_id = self._mj_model.geom("left_foot").id
    self._right_foot_geom_id = self._mj_model.geom("right_foot").id
    self._left_shin_geom_id = self._mj_model.geom("left_shin").id
    self._right_shin_geom_id = self._mj_model.geom("right_shin").id
    self._left_thigh_geom_id = self._mj_model.geom("left_thigh").id
    self._right_thigh_geom_id = self._mj_model.geom("right_thigh").id

    self._feet_floor_found_sensor = [
        self._mj_model.sensor(foot_geom + "_floor_found").id
        for foot_geom in ["left_foot", "right_foot"]
    ]
    self._right_foot_left_foot_found_sensor = self._mj_model.sensor(
        "right_foot_left_foot_found"
    ).id
    self._left_foot_right_shin_found_sensor = self._mj_model.sensor(
        "left_foot_right_shin_found"
    ).id
    self._right_foot_left_shin_found_sensor = self._mj_model.sensor(
        "right_foot_left_shin_found"
    ).id
    self._left_hand_left_thigh_found_sensor = self._mj_model.sensor(
        "left_hand_left_thigh_found"
    ).id
    self._right_hand_right_thigh_found_sensor = self._mj_model.sensor(
        "right_hand_right_thigh_found"
    ).id

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-s, s), y=+U(-s, s), yaw=U(-3.14, 3.14).
    #
    # MODIFIED: spawn_jitter is configurable and much wider on the climbing
    # task. At +/-0.5 m the robot always started within 8% of a 12 m patch, so
    # it saw the same handful of bumps every episode and could learn those
    # rather than how to climb. Wide spawning forces it to handle whatever
    # rock it lands on.
    rng, key = jax.random.split(rng)
    jitter = self._config.spawn_jitter
    dxy = jax.random.uniform(key, (2,), minval=-jitter, maxval=jitter)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)

    # MODIFIED: on the climbing task, start ON the flat platform rather than
    # scattered over the slope. Spawning mid-slope meant the robot was already
    # sliding before its first action, so it never had a stable state to act
    # from. x is placed across the platform's width; y keeps the jitter above,
    # so it still sees different rock each episode and cannot memorise one
    # line up the hill.
    if self._platform_x is not None:
      rng, key = jax.random.split(rng)
      px = jax.random.uniform(
          key, minval=self._platform_x + 0.4, maxval=self._platform_x + 3.0
      )
      qpos = qpos.at[0].set(px)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # MODIFIED: place the robot ON the slope, standing perpendicular to it.
    #
    # The keyframe height is a fixed world z, which is only correct on level
    # ground. The floor is rotated about +Y, so its surface sits at
    # z = -x*tan(slope), and at 45 degrees the xy jitter above alone spans
    # 1.0 m of surface height: the robot was being dropped from up to 1.25 m
    # in the air and landing hard, or spawned underneath the surface. Every
    # episode began in a failed state, which reads as "the policy cannot
    # learn" rather than "the spawn is wrong".
    #
    # Tilting the body by the same angle matters too: spawned world-vertical
    # on a 45 degree slope the robot is already 45 degrees off the surface
    # normal, i.e. mid-fall before it has acted.
    if self._slope_rad != 0.0:
        # On the platform the ground is LEVEL at a fixed height, not the
        # tilted surface -- using the slope formula here would drop the robot
        # from metres up, or bury it.
        if self._platform_x is not None:
            surface_z = self._platform_top
        else:
            surface_z = -qpos[0] * jp.tan(self._slope_rad)
        qpos = qpos.at[2].set(qpos[2] + surface_z)
        if self._platform_x is None:
            # Align the body to the slope. On the platform the ground is
            # level, so the robot should stand upright instead.
            tilt = math.axis_angle_to_quat(
                jp.array([0.0, 1.0, 0.0]), jp.array([self._slope_rad])
            )
            qpos = qpos.at[3:7].set(math.quat_mul(tilt, qpos[3:7]))
        # Rotating the body about the pelvis swings the feet downward, so the
        # pelvis has to rise by the standing height times (1 - cos(slope)) to
        # keep the feet on the surface rather than buried in it. Measured
        # without this: feet sit 0.196 m under a 45 degree slope.
        if self._platform_x is None:
            qpos = qpos.at[2].add(
                self._init_q[2] * (1.0 - jp.cos(self._slope_rad))
            )
        # Clear the terrain's own relief. Spawning against the mean plane puts
        # the robot inside any rock above it; on the 2.0 m mountain terrain
        # that was every foot, ~1.55 m under. Starting at the peak costs a
        # short drop onto the surface, which the policy handles, and is far
        # cheaper than an episode that begins underground.
        if self._platform_x is None:
            qpos = qpos.at[2].add(self._terrain_peak)

    # qpos[7:]=*U(0.5, 1.5)
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
        qpos[7:] * jax.random.uniform(key, (29,), minval=0.5, maxval=1.5)
    )

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Phase, freq=U(1.0, 1.5)
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    phase = jp.array([0, jp.pi])

    rng, cmd_rng = jax.random.split(rng)
    cmd = self.sample_command(cmd_rng)

    # Sample push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        # Phase related.
        "phase_dt": phase_dt,
        "phase": phase,
        # Push related.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push *= (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
        == 0
    )
    push *= self._config.push_config.enable
    qvel = state.data.qvel
    qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
    data = state.data.replace(qvel=qvel)
    state = state.replace(data=data)

    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt

    state.info["push"] = push
    state.info["step"] += 1
    state.info["push_step"] += 1
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    # NOTE(kevin): Enable this to make the policy stand still at 0 command.
    # state.info["phase"] = jp.where(
    #     jp.linalg.norm(state.info["command"]) > 0.01,
    #     state.info["phase"],
    #     jp.ones(2) * jp.pi,
    # )
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    state.info["command"] = jp.where(
        state.info["step"] > 500,
        self.sample_command(cmd_rng),
        state.info["command"],
    )
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  # MODIFIED: stricter fall termination.
  #
  # Stock ends an episode only at gravity_z < 0.0, i.e. once the torso is past
  # horizontal. A robot tipped to 89 degrees, or settled on its back, never
  # trips that -- it lies there collecting reward, and MJX's low-iteration
  # solver lets a prone 33 kg body sink partway through the floor. The policy
  # then learns a stable fallen pose instead of walking, which makes every
  # episode-length and reward number from such a run meaningless.
  MAX_TILT = 0.5           # gravity-z; 1.0 = upright, 0.0 = horizontal (~60 deg)
  MIN_TORSO_HEIGHT = 0.4   # metres; nominal standing pelvis is ~0.78

  # Slope-relative fall limits, used when slope_deg != 0.
  #
  # A climber on all fours is legitimately pitched 40-70 degrees off the
  # surface normal (cos(70) = 0.34), so TILT_TOL has to sit well below the
  # bipedal MAX_TILT or correct behaviour terminates. With the tilt check that
  # permissive, the clearance check is what actually catches a fall.
  TILT_TOL = 0.0            # dot(torso_up, slope_normal); 0.0 = past horizontal
  MIN_SLOPE_CLEARANCE = 0.25  # metres measured ALONG the surface normal

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    if self._slope_rad != 0.0:
      # MODIFIED: on a slope both stock checks measure against world vertical,
      # which condemns the target behaviour. A robot aligned to a 45 degree
      # slope reads gravity_z = 0.707 and its pelvis height is meaningless
      # once the ground rises with distance. Measure both against the surface.
      n = jp.array(self._slope_normal)
      fall_termination = jp.dot(self.get_gravity(data, "torso"), n) < self.TILT_TOL
      fall_termination |= jp.dot(data.qpos[0:3], n) < self.MIN_SLOPE_CLEARANCE
    else:
      # MODIFIED: was `< 0.0`. Fire at ~60 degrees rather than waiting for 90,
      # and treat a pelvis on the ground as down whatever the orientation --
      # the case the stock height-free check misses entirely.
      fall_termination = self.get_gravity(data, "torso")[-1] < self.MAX_TILT
      fall_termination |= data.qpos[2] < self.MIN_TORSO_HEIGHT
    # MODIFIED: foot-to-foot contact does not end a climbing episode.
    #
    # Inherited from the walking task, where the feet touching means the legs
    # have crossed. On a slope a narrow stance is CORRECT -- it keeps the
    # centre of mass over the feet -- and this check fired within ~6 steps in
    # 60 of 60 episodes, killing every attempt before the policy could do
    # anything. Neither the tilt nor the clearance check ever fired; this was
    # the entire reason climbing episodes were so short.
    #
    # The shin checks below stay: a foot against the opposite SHIN is a real
    # tangle at any slope.
    if self._slope_rad != 0.0:
      contact_termination = jp.zeros((), dtype=bool)
    else:
      contact_termination = data.sensordata[
          self._mj_model.sensor_adr[self._right_foot_left_foot_found_sensor]
      ] > 0
    contact_termination |= data.sensordata[
        self._mj_model.sensor_adr[self._left_foot_right_shin_found_sensor]
    ] > 0
    contact_termination |= data.sensordata[
        self._mj_model.sensor_adr[self._right_foot_left_shin_found_sensor]
    ] > 0
    return (
        fall_termination
        | contact_termination
        | jp.isnan(data.qpos).any()
        | jp.isnan(data.qvel).any()
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    gyro = self.get_gyro(data, "pelvis")
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    linvel = self.get_local_linvel(data, "pelvis")
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        info["command"],  # 3
        noisy_joint_angles - self._default_pose,  # 29
        noisy_joint_vel,  # 29
        info["last_act"],  # 29
        phase,
    ])

    accelerometer = self.get_accelerometer(data, "pelvis")
    global_angvel = self.get_global_angvel(data, "pelvis")
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = data.qpos[2]

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        global_angvel,  # 3
        joint_angles - self._default_pose,
        joint_vel,
        root_height,  # 1
        data.actuator_force,  # 29
        contact,  # 2
        feet_vel,  # 4*3
        info["feet_air_time"],  # 2
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        # The climbing objective.
        "progress_uphill": self._reward_progress_uphill(
            self.get_global_linvel(data, "torso"), data.qpos[0:3]
        ),
        # Tracking rewards.
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data, "pelvis")
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data, "pelvis")
        ),
        # Base-related rewards.
        "lin_vel_z": self._cost_lin_vel_z(
            self.get_global_linvel(data, "pelvis"),
            self.get_global_linvel(data, "torso"),
        ),
        "ang_vel_xy": self._cost_ang_vel_xy(
            self.get_global_angvel(data, "torso")
        ),
        "orientation": self._cost_orientation(self.get_gravity(data, "torso")),
        "base_height": self._cost_base_height(data.qpos[2]),
        # Energy related rewards.
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "dof_acc": self._cost_dof_acc(data.qacc[6:]),
        # Feet related rewards.
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_clearance": self._cost_feet_clearance(data, info),
        "feet_height": self._cost_feet_height(
            info["swing_peak"], first_contact, info
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "feet_phase": self._reward_feet_phase(
            data,
            info["phase"],
            self._config.reward_config.max_foot_height,
            info["command"],
        ),
        # Other rewards.
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "collision": self._cost_collision(data),
        "contact_force": self._cost_contact_force(data),
        # Pose related rewards.
        "joint_deviation_hip": self._cost_joint_deviation_hip(
            data.qpos[7:], info["command"]
        ),
        "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "pose": self._cost_pose(data.qpos[7:]),
    }

  def _cost_contact_force(self, data: mjx.Data) -> jax.Array:
    l_contact_force = mjx_env.get_sensor_data(
        self.mj_model, data, "left_foot_force"
    )
    r_contact_force = mjx_env.get_sensor_data(
        self.mj_model, data, "right_foot_force"
    )
    cost = jp.clip(
        jp.abs(l_contact_force[2])
        - self._config.reward_config.max_contact_force,
        min=0.0,
    )
    cost += jp.clip(
        jp.abs(r_contact_force[2])
        - self._config.reward_config.max_contact_force,
        min=0.0,
    )
    return cost

  def _cost_collision(self, data: mjx.Data) -> jax.Array:
    c = (
        data.sensordata[
            self._mj_model.sensor_adr[self._left_hand_left_thigh_found_sensor]
        ]
        > 0
    )
    c |= (
        data.sensordata[
            self._mj_model.sensor_adr[self._right_hand_right_thigh_found_sensor]
        ]
        > 0
    )
    return jp.any(c)

  # Tracking rewards.

  def _cost_joint_deviation_hip(
      self, qpos: jax.Array, cmd: jax.Array
  ) -> jax.Array:
    error = qpos[self._hip_indices] - self._default_pose[self._hip_indices]
    # Allow roll deviation when lateral velocity is high.
    weight = jp.where(
        cmd[1] > 0.1,
        jp.array([0.0, 1.0, 0.0, 1.0]),
        jp.array([1.0, 1.0, 1.0, 1.0]),
    )
    cost = jp.sum(jp.abs(error) * weight)
    return cost

  def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
    error = qpos[self._knee_indices] - self._default_pose[self._knee_indices]
    return jp.sum(jp.abs(error))

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos - self._default_pose))

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _reward_progress_uphill(self, global_linvel: jax.Array,
                              data_pos: jax.Array) -> jax.Array:
    """Speed up the slope, clipped at a target pace.

    Deliberately says nothing about posture, limb count, or gait -- if
    quadrupedal climbing is the only way to make progress on a steep slope,
    that is what the policy should discover, not what this term should
    prescribe. The clip stops it paying unboundedly for a downhill-then-launch
    exploit and keeps the scale comparable to the tracking terms.

    Zero on flat ground: uphill is undefined there and the scale is 0 anyway.
    """
    if self._slope_rad == 0.0:
      return jp.zeros(())

    # Reward HEIGHT GAINED, not velocity along some chosen direction.
    #
    # A projection needs a direction, and both options were wrong. World-uphill
    # punishes the corridor: the lane wanders 4.4 m laterally at a median 38
    # degrees off the uphill axis, so following it pays only cos(38) and
    # charging straight up the wall pays more. The lane tangent fixes that but
    # scripts the route -- the env looks up where the lane goes and pays for
    # alignment with it, which hands the policy the answer and needs a
    # centreline a real robot would not have.
    #
    # Height gained has no direction in it at all. Rounding a bend still gains
    # height; going backwards loses it. The policy is never told where the lane
    # is -- it has to discover that the corridor is the cheapest way up. And it
    # is optimising something a real robot could actually measure, since its own
    # altitude is implied by the gravity vector and proprioception it already
    # senses.
    speed = jp.dot(global_linvel, jp.array(self._slope_normal_up))
    reward = jp.clip(speed, -1.0, self._config.reward_config.max_uphill_speed)
    # The platform is a starting position, not part of the task. Movement
    # there earns nothing -- otherwise it is just a flat-ground walking
    # reward to farm instead of climbing.
    #
    # But zero is not neutral, it is a SAFE HARBOUR: with termination at -100
    # and the slope costing reward to slide down, standing still on flat
    # ground beats attempting the climb. Measured, episode length went 54 ->
    # 584 steps the moment the platform existed, with reward drifting DOWN --
    # the policy had found somewhere safe to wait.
    #
    # So loitering costs a fixed rate per step. The robot has to leave, and
    # once on the slope the penalty stops and real climbing reward starts.
    if self._platform_x is not None:
      on_platform = data_pos[0] > self._platform_x
      reward = jp.where(
          on_platform, -self._config.reward_config.platform_loiter, reward
      )
    return reward

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base-related rewards.

  def _cost_lin_vel_z(
      self,
      global_linvel_torso: jax.Array,
      global_linvel_pelvis: jax.Array,
  ) -> jax.Array:
    torso_cost = jp.square(global_linvel_torso[2])
    pelvis_cost = jp.square(global_linvel_pelvis[2])
    return torso_cost + pelvis_cost

  def _cost_ang_vel_xy(self, global_angvel_torso: jax.Array) -> jax.Array:
    return jp.sum(jp.square(global_angvel_torso[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torso_zaxis - jp.array([0.073, 0.0, 1.0])))

  def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
    return jp.square(
        base_height - self._config.reward_config.base_height_target
    )

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))

  # Other rewards.

  def _cost_stand_still(
      self, commands: jax.Array, qpos: jax.Array
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    cost = jp.sum(jp.abs(qpos - self._default_pose))
    cost *= cmd_norm < 0.01
    return cost

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _reward_alive(self) -> jax.Array:
    return jp.array(1.0)

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    del info  # Unused.
    body_vel = self.get_global_linvel(data, "pelvis")[:2]
    reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
    return reward

  def _cost_feet_clearance(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    del info  # Unused.
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
    del info  # Unused.
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact)

  def _reward_feet_air_time(
      self,
      air_time: jax.Array,
      first_contact: jax.Array,
      commands: jax.Array,
      threshold_min: float = 0.2,
      threshold_max: float = 0.5,
  ) -> jax.Array:
    del commands  # Unused.
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    return reward

  def _reward_feet_phase(
      self,
      data: mjx.Data,
      phase: jax.Array,
      foot_height: jax.Array,
      command: jax.Array,
  ) -> jax.Array:
    # Reward for tracking the desired foot height.
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    body_linvel = self.get_global_linvel(data, "pelvis")[:2]
    body_angvel = self.get_global_angvel(data, "pelvis")[2]
    linvel_mask = jp.logical_or(
        jp.linalg.norm(body_linvel) > 0.1,
        jp.abs(body_angvel) > 0.1,
    )
    mask = jp.logical_or(linvel_mask, jp.linalg.norm(command) > 0.01)
    reward *= mask
    return reward

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    lin_vel_x = jax.random.uniform(
        rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=self._config.ang_vel_yaw[0],
        maxval=self._config.ang_vel_yaw[1],
    )

    # With 10% chance, set everything to zero.
    return jp.where(
        jax.random.bernoulli(rng4, p=0.1),
        jp.zeros(3),
        jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
    )
