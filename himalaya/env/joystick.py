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

import math as python_math
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env

from himalaya.env import base as g1_base
from himalaya.env import g1_constants as consts
from himalaya.env import gait


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
              # MODIFIED: four-limb climb terms; enabled by climb_terrain.
              uphill_progress=0.0,
              assisted_uphill_progress=0.0,
              mountain_progress=0.0,
              new_high_progress=0.0,
              waypoint_bonus=0.0,
              limb_touchdown_advance=0.0,
              large_foot_step=0.0,
              failed_ascent=0.0,
              crawl_height=0.0,
              mixed_support=0.0,
              foot_uphill_drive=0.0,
              climb_time=0.0,
              hand_proximity=0.0,
              continuous_hand_support=0.0,
              hand_contact_schedule=0.0,
              hand_phase=0.0,
              hand_lift_height=0.0,
              hand_lift_target=0.0,
              diagonal_swing_sync=0.0,
              diagonal_support=0.0,
              hand_load_share=0.0,
              hand_slip=0.0,
              knee_clearance=0.0,
              knee_contact=0.0,
          ),
          tracking_sigma=0.25,
          max_foot_height=0.15,
          max_hand_height=0.24384,  # 0.8 ft terrain-normal palm lift.
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
      climb=config_dict.create(
          enabled=False,
          slope_degrees=12.0,
          roughness_m=0.060,
          spike_friction=0.95,
          target_uphill_speed=0.30,
          target_hand_load_share=0.28,
          hand_load_sigma=0.025,
          # MODIFIED: opposed 60% hand duty cycles retain 20% two-hand
          # overlap while leaving enough time for each ball to lift/reach.
          hand_contact_duty_factor=0.60,
          waypoint_interval=0.25,
          progress_epsilon=0.005,
          min_limb_air_time=0.08,
          # MODIFIED: larger foot plants for steep scrambling. Keep the
          # touchdown target aligned with the longer hip/knee reference.
          target_stride_length=0.20,
          boulders_enabled=True,
          target_pelvis_clearance=0.49,
          min_pelvis_clearance=0.36,
          fall_pelvis_clearance=0.25,
          # MODIFIED: surface clearance around each knee joint. The safety
          # radius approximates the knee housing around the body origin.
          target_knee_clearance=0.05,
          knee_safety_radius=0.06,
          # MODIFIED: amplitude of a continuous fore-aft hip sweep. Its full
          # peak-to-peak travel is twice this value; the stance half actively
          # retracts the planted foot and propels the pelvis uphill.
          reference_hip_swing=0.30,
          reference_knee_lift=0.38,
          reference_shoulder_swing=0.30,
          reference_shoulder_lift=0.45,
          reference_elbow_lift=0.22,
          reference_support_knee_extension=0.16,
          reference_support_elbow_push=0.14,
          gait_frequency_range=[0.70, 0.95],
          stall_seconds=3.0,
          max_regression_distance=0.35,
          max_downhill_distance=1.5,
          max_lateral_distance=2.1,
      ),
      impl="warp",
      naconmax=8 * 8192,
      njmax=256,
  )


class Joystick(g1_base.G1Env):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict | None = None,
      config_overrides: dict[str, str | int | list[Any]] | None = None,
  ):
    config = config or default_config()
    self._climb = task == "climb_terrain"
    if self._climb:
      # MODIFIED: a climb is always a slow forward four-limb crawl, not a
      # joystick task that sometimes asks for standing, strafing, or turning.
      config.climb.enabled = True
      config.push_config.enable = False
      config.lin_vel_x = [
          config.climb.target_uphill_speed,
          config.climb.target_uphill_speed,
      ]
      config.lin_vel_y = [0.0, 0.0]
      config.ang_vel_yaw = [0.0, 0.0]
      scales = config.reward_config.scales
      scales.tracking_lin_vel = 0.0
      scales.tracking_ang_vel = 0.1
      scales.orientation = -1.0
      # A crawl policy initially falls after only a few exchanges.  The stock
      # -100 terminal event, combined with a full ascent clawback, dominated
      # every forward sample and taught PPO to minimize motion.  Keep a clear
      # fall cost, but leave enough return for a partially successful stride
      # to be better than shuffling in place.
      scales.termination = -20.0
      # MODIFIED: reward a completed airborne step. The stock phase-height
      # curve stays positive through nearly the whole cycle and conflicts
      # with planted crawl support, so it is disabled for climbing.
      scales.feet_air_time = 0.4
      scales.feet_phase = 0.0
      # MODIFIED: static crouch-deviation costs directly oppose the large,
      # cyclic leg and arm motion required for climbing. Physical/soft joint
      # limits, action-rate cost, and collision cost remain active.
      scales.pose = 0.0
      scales.joint_deviation_hip = 0.0
      scales.joint_deviation_knee = 0.0
      scales.action_rate = -0.01
      scales.energy = -2e-5
      scales.collision = -0.25
      # MODIFIED: forward ascent is the primary objective. Contact rewards are
      # supporting constraints and cannot outweigh sustained uphill progress.
      # MODIFIED: dense forward-velocity shaping is secondary to the signed
      # displacement potential (10.0). It guides cadence without making a
      # high-speed fall more valuable than controlled physical progress.
      scales.uphill_progress = 8.0
      scales.assisted_uphill_progress = 1.5
      scales.mountain_progress = 10.0
      scales.new_high_progress = 3.0
      scales.waypoint_bonus = 2.0
      scales.limb_touchdown_advance = 1.0
      # MODIFIED: one-time, foot-only step-length bonus. Squaring normalized
      # advance makes a full 20 cm step worth four 10 cm steps, while the
      # episode-best plant record prevents repeated stamping from farming it.
      scales.large_foot_step = 1.0
      scales.failed_ascent = -2.0
      scales.crawl_height = 0.5
      scales.mixed_support = 0.5
      scales.foot_uphill_drive = 0.8
      scales.climb_time = -0.05
      scales.hand_proximity = 0.05
      scales.continuous_hand_support = 0.1
      scales.hand_contact_schedule = 0.5
      # MODIFIED: the generic hand phase-height curve contradicted the contact
      # schedule by requesting clearance during support. Keep the explicitly
      # supported 0.8 ft lift objective, at shaping scales that cannot compete
      # with forward progress.
      scales.hand_phase = 0.0
      scales.hand_lift_height = 0.5
      scales.hand_lift_target = 0.2
      scales.diagonal_swing_sync = 0.5
      scales.diagonal_support = 0.3
      scales.hand_load_share = 0.3
      scales.hand_slip = -0.35
      # MODIFIED: positive, progress-weighted clearance shaping plus an
      # explicit cost once the knee safety envelope intersects terrain.
      scales.knee_clearance = 0.2
      scales.knee_contact = -1.0
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()
    angle = python_math.radians(float(self._config.climb.slope_degrees))
    self._slope_tangent = jp.array(
        [python_math.cos(angle), 0.0, python_math.sin(angle)]
    )
    self._slope_normal = jp.array(
        [-python_math.sin(angle), 0.0, python_math.cos(angle)]
    )
    self._slope_encoding = jp.array(
        [python_math.sin(angle), python_math.cos(angle)]
    )
    self._terrain_plane_offset = 0.20

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
    self._knee_body_ids = jp.array([
        self._mj_model.body("left_knee_link").id,
        self._mj_model.body("right_knee_link").id,
    ])
    if self._climb:
      self._boulder_geom_ids = jp.array([
          self._mj_model.geom(f"boulder_{index:02d}").id
          for index in range(10)
      ])
      self._boulder_radius = jp.array(
          self._mj_model.geom_size[int(self._boulder_geom_ids[0]), 0]
      )

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
    self._hand_contact_sensor_ids = np.array([
        self._mj_model.sensor("left_hand_floor_found").id,
        self._mj_model.sensor("right_hand_floor_found").id,
    ])
    hand_vel_sensor_adr = []
    for site in consts.HAND_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      start = self._mj_model.sensor_adr[sensor_id]
      hand_vel_sensor_adr.append(
          list(range(start, start + self._mj_model.sensor_dim[sensor_id]))
      )
    self._hand_vel_sensor_adr = jp.array(hand_vel_sensor_adr)

  def _hand_contact(self, data: mjx.Data) -> jax.Array:
    return jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._hand_contact_sensor_ids
    ])

  def _limb_force_norms(
      self, data: mjx.Data
  ) -> tuple[jax.Array, jax.Array]:
    foot = jp.array([
        jp.linalg.norm(mjx_env.get_sensor_data(self.mj_model, data, name))
        for name in ("left_foot_force", "right_foot_force")
    ])
    hand = jp.array([
        jp.linalg.norm(mjx_env.get_sensor_data(self.mj_model, data, name))
        for name in ("left_hand_force", "right_hand_force")
    ])
    return foot, hand

  def _knee_clearance(self, data: mjx.Data) -> jax.Array:
    """Conservative knee-housing clearance from ramp and boulders."""
    # MODIFIED: the simplified MJX model does not collide knees with terrain.
    # Use the measured knee body origins plus a housing radius so the reward
    # remains dense/differentiable instead of waiting for penetration.
    knee_pos = data.xpos[self._knee_body_ids]
    floor_clearance = (
        knee_pos @ self._slope_normal
        - self._terrain_plane_offset
        - float(self._config.climb.roughness_m)
        - self._config.climb.knee_safety_radius
    )
    if not self._config.climb.boulders_enabled:
      return floor_clearance
    boulder_pos = data.geom_xpos[self._boulder_geom_ids]
    center_distance = jp.linalg.norm(
        knee_pos[:, None, :] - boulder_pos[None, :, :], axis=-1
    )
    boulder_clearance = (
        center_distance
        - self._boulder_radius
        - self._config.climb.knee_safety_radius
    )
    return jp.minimum(floor_clearance, jp.min(boulder_clearance, axis=1))

  def _crawl_reference(self, phase: jax.Array) -> jax.Array:
    """Small diagonal crawl reference; policy actions remain residuals."""
    # MODIFIED: hip pitch uses the full cosine for reach plus active stance
    # retraction. Knee/elbow lift keeps the stable squared half-wave timing.
    foot_swing = jp.square(jp.maximum(jp.cos(phase), 0.0))
    hand_swing = foot_swing[::-1]
    foot_sweep = jp.cos(phase)
    reference = jp.zeros_like(self._default_pose)
    reference = reference.at[0].set(
        -self._config.climb.reference_hip_swing * foot_sweep[0]
    )
    reference = reference.at[3].set(
        self._config.climb.reference_knee_lift * foot_swing[0]
        - self._config.climb.reference_support_knee_extension * foot_swing[1]
    )
    reference = reference.at[6].set(
        -self._config.climb.reference_hip_swing * foot_sweep[1]
    )
    reference = reference.at[9].set(
        self._config.climb.reference_knee_lift * foot_swing[1]
        - self._config.climb.reference_support_knee_extension * foot_swing[0]
    )
    reference = reference.at[15].set(
        -self._config.climb.reference_shoulder_swing * hand_swing[0]
    )
    reference = reference.at[18].set(
        -self._config.climb.reference_elbow_lift * hand_swing[0]
        + self._config.climb.reference_support_elbow_push * foot_swing[0]
    )
    reference = reference.at[22].set(
        -self._config.climb.reference_shoulder_swing * hand_swing[1]
    )
    reference = reference.at[16].set(
        self._config.climb.reference_shoulder_lift * hand_swing[0]
    )
    reference = reference.at[23].set(
        -self._config.climb.reference_shoulder_lift * hand_swing[1]
    )
    reference = reference.at[25].set(
        -self._config.climb.reference_elbow_lift * hand_swing[1]
        + self._config.climb.reference_support_elbow_push * foot_swing[1]
    )
    return reference

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # MODIFIED: climb resets face uphill near the IK reference; flat/rough
    # joystick tasks retain the stock broad position and heading randomization.
    rng, key = jax.random.split(rng)
    dxy_limit = 0.05 if self._climb else 0.5
    dxy = jax.random.uniform(
        key, (2,), minval=-dxy_limit, maxval=dxy_limit
    )
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw_limit = 0.05 if self._climb else 3.14
    yaw = jax.random.uniform(
        key, (1,), minval=-yaw_limit, maxval=yaw_limit
    )
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # MODIFIED: multiplicative stock noise destroys the IK crawl stance.
    rng, key = jax.random.split(rng)
    if self._climb:
      qpos = qpos.at[7:].add(
          jax.random.uniform(key, (29,), minval=-0.025, maxval=0.025)
      )
    else:
      qpos = qpos.at[7:].set(
          qpos[7:] * jax.random.uniform(
              key, (29,), minval=0.5, maxval=1.5
          )
      )

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel_limit = 0.05 if self._climb else 0.5
    qvel = qvel.at[0:6].set(jax.random.uniform(
        key, (6,), minval=-qvel_limit, maxval=qvel_limit
    ))

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

    # Phase. MODIFIED: a four-limb crawl needs enough time to unload, reach,
    # and replant a palm; the stock 1.25-1.5 Hz biped cadence truncates it.
    rng, key = jax.random.split(rng)
    if self._climb:
      gait_freq = jax.random.uniform(
          key,
          (1,),
          minval=self._config.climb.gait_frequency_range[0],
          maxval=self._config.climb.gait_frequency_range[1],
      )
    else:
      gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    # MODIFIED: begin at the all-planted exchange boundary, then ramp into the
    # first swing. Starting at [0, pi] commanded maximum reach against four
    # planted high-friction contacts and produced a startup drag instead.
    phase = jp.array([-0.5 * jp.pi, 0.5 * jp.pi])

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

    initial_uphill_position = jp.dot(qpos[:3], self._slope_tangent)
    initial_foot_plant = (
        data.site_xpos[self._feet_site_id] @ self._slope_tangent
    )
    initial_hand_plant = (
        data.site_xpos[self._hands_site_id] @ self._slope_tangent
    )
    initial_hand_plant_normal = (
        data.site_xpos[self._hands_site_id] @ self._slope_normal
    )
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
        # MODIFIED: previous hand state lets the actor anticipate support
        # exchange; air time is retained for future crawl-gait diagnostics.
        "last_hand_contact": jp.zeros(2, dtype=bool),
        "hand_air_time": jp.zeros(2),
        "last_uphill_position": initial_uphill_position,
        # MODIFIED: episode-relative progress state makes height records and
        # waypoint bonuses monotonic, so revisiting ground cannot be farmed.
        "start_uphill_position": initial_uphill_position,
        "max_uphill_position": initial_uphill_position,
        "progress_checkpoint_position": initial_uphill_position,
        "steps_without_progress": jp.array(0, dtype=jp.int32),
        "last_waypoint": jp.array(0, dtype=jp.int32),
        # MODIFIED: per-limb uphill records provide learnable stepping credit
        # before a coordinated stride is large enough to move the pelvis.
        "best_foot_plant_uphill": initial_foot_plant,
        "best_hand_plant_uphill": initial_hand_plant,
        "last_hand_plant_normal": initial_hand_plant_normal,
        "hand_lift_achieved": jp.zeros(2, dtype=bool),
        # Phase related.
        "phase_dt": phase_dt,
        "phase": phase,
        # Push related.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
    }

    metrics = {}
    for k in self._config.reward_config.scales:
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())
    metrics["climb/uphill_velocity"] = jp.zeros(())
    metrics["climb/mountain_progress"] = jp.zeros(())
    metrics["climb/distance_from_start"] = jp.zeros(())
    metrics["climb/distance_from_high"] = jp.zeros(())
    metrics["climb/waypoint"] = jp.zeros(())
    metrics["climb/limb_touchdown_advance"] = jp.zeros(())
    metrics["climb/large_foot_step_bonus"] = jp.zeros(())
    metrics["climb/pelvis_clearance"] = jp.zeros(())
    metrics["climb/posture_gate"] = jp.zeros(())
    metrics["climb/steps_without_progress"] = jp.zeros(())
    metrics["climb/foot_uphill_force"] = jp.zeros(())
    metrics["climb/hand_contact_fraction"] = jp.zeros(())
    metrics["climb/hand_phase"] = jp.zeros(())
    metrics["climb/hand_lift_height"] = jp.zeros(())
    metrics["climb/hand_lift_target"] = jp.zeros(())
    metrics["climb/diagonal_swing_sync"] = jp.zeros(())
    metrics["climb/hand_load_share"] = jp.zeros(())
    metrics["climb/diagonal_support"] = jp.zeros(())
    metrics["climb/hand_slip_speed"] = jp.zeros(())
    metrics["climb/knee_clearance_min"] = jp.zeros(())
    metrics["climb/knee_contact_fraction"] = jp.zeros(())

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
    info["last_hand_contact"] = self._hand_contact(data)
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

    crawl_reference = (
        self._crawl_reference(state.info["phase"])
        if self._climb
        else jp.zeros_like(self._default_pose)
    )
    motor_targets = (
        self._default_pose
        + crawl_reference
        + action * self._config.action_scale
    )
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])
    if self._climb:
      first_contact = (
          contact
          & ~state.info["last_contact"]
          & (
              state.info["feet_air_time"]
              >= self._config.climb.min_limb_air_time
          )
      )
    else:
      contact_filt = contact | state.info["last_contact"]
      first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    hand_contact = self._hand_contact(data)
    first_hand_contact = (
        hand_contact
        & ~state.info["last_hand_contact"]
        & (
            state.info["hand_air_time"]
            >= self._config.climb.min_limb_air_time
        )
    )
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    if self._climb:
      foot_clearance = jp.maximum(
          p_f @ self._slope_normal - self._terrain_plane_offset, 0.0
      )
    else:
      foot_clearance = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(
        state.info["swing_peak"], foot_clearance
    )

    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data, state.info)

    rewards = self._get_reward(
        data,
        action,
        state.info,
        state.metrics,
        done,
        first_contact,
        first_hand_contact,
        contact,
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
    state.info["hand_air_time"] += self.dt
    state.info["hand_air_time"] *= ~hand_contact
    state.info["last_hand_contact"] = hand_contact
    foot_plant = data.site_xpos[self._feet_site_id] @ self._slope_tangent
    hand_plant = data.site_xpos[self._hands_site_id] @ self._slope_tangent
    state.info["best_foot_plant_uphill"] = jp.where(
        first_contact,
        jp.maximum(state.info["best_foot_plant_uphill"], foot_plant),
        state.info["best_foot_plant_uphill"],
    )
    state.info["best_hand_plant_uphill"] = jp.where(
        first_hand_contact,
        jp.maximum(state.info["best_hand_plant_uphill"], hand_plant),
        state.info["best_hand_plant_uphill"],
    )
    hand_plant_normal = (
        data.site_xpos[self._hands_site_id] @ self._slope_normal
    )
    hand_lift = jp.maximum(
        hand_plant_normal - state.info["last_hand_plant_normal"], 0.0
    )
    state.info["last_hand_plant_normal"] = jp.where(
        first_hand_contact,
        hand_plant_normal,
        state.info["last_hand_plant_normal"],
    )
    state.info["hand_lift_achieved"] = jp.where(
        hand_contact,
        False,
        state.info["hand_lift_achieved"]
        | (hand_lift >= self._config.reward_config.max_hand_height),
    )
    state.info["last_uphill_position"] = jp.dot(
        data.qpos[:3], self._slope_tangent
    )
    uphill_position = state.info["last_uphill_position"]
    made_progress = uphill_position >= (
        state.info["progress_checkpoint_position"]
        + self._config.climb.progress_epsilon
    )
    state.info["progress_checkpoint_position"] = jp.where(
        made_progress,
        uphill_position,
        state.info["progress_checkpoint_position"],
    )
    state.info["steps_without_progress"] = jp.where(
        made_progress,
        0,
        state.info["steps_without_progress"] + 1,
    )
    state.info["max_uphill_position"] = jp.maximum(
        state.info["max_uphill_position"], uphill_position
    )
    state.info["last_waypoint"] = jp.floor(
        jp.maximum(
            state.info["max_uphill_position"]
            - state.info["start_uphill_position"],
            0.0,
        ) / self._config.climb.waypoint_interval
    ).astype(jp.int32)
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

  def _get_termination(
      self, data: mjx.Data, info: dict[str, Any] | None = None
  ) -> jax.Array:
    # MODIFIED: was `< 0.0`. Fire at ~60 degrees rather than waiting for 90,
    # and treat a pelvis on the ground as down whatever the orientation --
    # the case the stock height-free check misses entirely.
    torso_up = self.get_gravity(data, "torso")
    if self._climb:
      # MODIFIED: a four-point crouch intentionally rotates the torso z-axis
      # toward uphill. Falling means losing that slope alignment, not ceasing
      # to be vertically upright like the stock biped task.
      fall_termination = jp.dot(torso_up, self._slope_tangent) < 0.35
    else:
      fall_termination = torso_up[-1] < self.MAX_TILT
    if self._climb:
      pelvis_clearance = (
          jp.dot(data.qpos[:3], self._slope_normal)
          - self._terrain_plane_offset
      )
      fall_termination |= (
          pelvis_clearance < self._config.climb.fall_pelvis_clearance
      )
    else:
      fall_termination |= data.qpos[2] < self.MIN_TORSO_HEIGHT
    contact_termination = data.sensordata[
        self._mj_model.sensor_adr[self._right_foot_left_foot_found_sensor]
    ] > 0
    contact_termination |= data.sensordata[
        self._mj_model.sensor_adr[self._left_foot_right_shin_found_sensor]
    ] > 0
    contact_termination |= data.sensordata[
        self._mj_model.sensor_adr[self._right_foot_left_shin_found_sensor]
    ] > 0
    done = (
        fall_termination
        | contact_termination
        | jp.isnan(data.qpos).any()
        | jp.isnan(data.qvel).any()
    )
    if self._climb:
      uphill = jp.dot(data.qpos[:3], self._slope_tangent)
      done |= uphill < -self._config.climb.max_downhill_distance
      done |= jp.abs(data.qpos[1]) > self._config.climb.max_lateral_distance
      if info is not None:
        # MODIFIED: end policies that settle in place or give back substantial
        # earned height. A 5 mm checkpoint filters solver/contact jitter.
        made_progress = uphill >= (
            info["progress_checkpoint_position"]
            + self._config.climb.progress_epsilon
        )
        stalled_steps = jp.where(
            made_progress, 0, info["steps_without_progress"] + 1
        )
        stall_limit = int(
            self._config.climb.stall_seconds / self.dt
        )
        done |= stalled_steps >= stall_limit
        done |= (
            info["max_uphill_position"] - uphill
            > self._config.climb.max_regression_distance
        )
    return done

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

    if self._climb:
      # MODIFIED: the actor receives only quantities available from IMU/joint
      # state plus contact/load sensors; terrain direction is critic-only.
      hand_contact = self._hand_contact(data).astype(data.qpos.dtype)
      foot_force, hand_force = self._limb_force_norms(data)
      total_force = jp.sum(foot_force) + jp.sum(hand_force) + 1e-6
      hand_load_share = jp.sum(hand_force) / total_force
      climb_state = jp.hstack([
          self._slope_encoding,
          hand_contact,
          info["last_hand_contact"].astype(data.qpos.dtype),
          jp.clip(foot_force / 500.0, 0.0, 2.0),
          jp.clip(hand_force / 300.0, 0.0, 2.0),
          hand_load_share,
      ])
      state = jp.hstack([state, climb_state])
      privileged_state = jp.hstack([
          privileged_state, climb_state, self._slope_tangent
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
      first_hand_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    rewards = {
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
    if not self._climb:
      return rewards

    # MODIFIED: continuous crawl reward. At least one hand should normally be
    # loaded; opposite hand/foot contacts receive a separate stability reward.
    hand_contact = self._hand_contact(data)
    foot_force, hand_force = self._limb_force_norms(data)
    total_force = jp.sum(foot_force) + jp.sum(hand_force) + 1e-6
    hand_share = jp.sum(hand_force) / total_force
    share_error = jp.square(
        hand_share - self._config.climb.target_hand_load_share
    )
    uphill_velocity = jp.dot(
        self.get_global_linvel(data, "pelvis"), self._slope_tangent
    )
    uphill_position = jp.dot(data.qpos[:3], self._slope_tangent)
    mountain_progress = (
        uphill_position - info["last_uphill_position"]
    ) / self.dt
    normalized_progress = jp.clip(
        mountain_progress / self._config.climb.target_uphill_speed,
        -1.5,
        2.0,
    )
    positive_progress = jp.maximum(normalized_progress, 0.0)
    palm_pos = data.site_xpos[self._hands_site_id]
    plane_dist = jp.abs(
        palm_pos @ self._slope_normal - self._terrain_plane_offset
    )
    palm_vel = data.sensordata[self._hand_vel_sensor_adr]
    tangent_vel = palm_vel @ self._slope_tangent
    any_hand = jp.any(hand_contact)
    any_foot = jp.any(contact)
    mixed_support = any_hand & any_foot
    pelvis_clearance = (
        jp.dot(data.qpos[:3], self._slope_normal)
        - self._terrain_plane_offset
    )
    torso_alignment = jp.dot(
        self.get_gravity(data, "torso"), self._slope_tangent
    )
    clearance_gate = jp.clip(
        (
            pelvis_clearance - self._config.climb.min_pelvis_clearance
        ) / (
            self._config.climb.target_pelvis_clearance
            - self._config.climb.min_pelvis_clearance
        ),
        0.0,
        1.0,
    )
    alignment_gate = jp.clip((torso_alignment - 0.35) / 0.55, 0.0, 1.0)
    posture_gate = clearance_gate * alignment_gate
    support_multiplier = 0.3 + 0.7 * any_hand.astype(data.qpos.dtype)
    # MODIFIED: uphill displacement is the potential difference. Supported
    # ascent earns its full value; unsupported ascent keeps 30% so hand
    # exchange remains possible. Regression is penalized 1.75x regardless of
    # contacts, closing the "drop hands and slide" loophole.
    potential_progress = jp.where(
        normalized_progress >= 0.0,
        normalized_progress * support_multiplier * posture_gate,
        1.75 * normalized_progress,
    )
    new_high_delta = jp.maximum(
        uphill_position - info["max_uphill_position"], 0.0
    )
    new_high_progress = jp.clip(
        new_high_delta
        / (self._config.climb.target_uphill_speed * self.dt),
        0.0,
        2.0,
    ) * posture_gate
    new_max = jp.maximum(info["max_uphill_position"], uphill_position)
    current_waypoint = jp.floor(
        jp.maximum(new_max - info["start_uphill_position"], 0.0)
        / self._config.climb.waypoint_interval
    ).astype(jp.int32)
    waypoints_crossed = jp.maximum(
        current_waypoint - info["last_waypoint"], 0
    )
    # Rewards are multiplied by dt by step(); divide here so each newly
    # crossed waypoint is an actual one-time bonus of its configured scale.
    waypoint_bonus = (
        waypoints_crossed.astype(data.qpos.dtype) / self.dt * posture_gate
    )
    progress_gate = jp.clip(positive_progress, 0.0, 1.0) * posture_gate
    knee_clearance = self._knee_clearance(data)
    knee_clearance_fraction = jp.clip(
        knee_clearance / self._config.climb.target_knee_clearance, 0.0, 1.0
    )
    knee_clearance_shape = jp.square(knee_clearance_fraction) * (
        3.0 - 2.0 * knee_clearance_fraction
    )
    # A small stationary component teaches clearance before locomotion starts,
    # but remains below the climb time cost; full credit requires progress.
    knee_motion_gate = posture_gate * (
        0.2 + 0.8 * jp.clip(positive_progress, 0.0, 1.0)
    )
    knee_clearance_reward = jp.mean(knee_clearance_shape) * knee_motion_gate
    knee_contact = knee_clearance <= 0.0
    foot_plant = data.site_xpos[self._feet_site_id] @ self._slope_tangent
    hand_plant = palm_pos @ self._slope_tangent
    foot_advance = jp.clip(
        (
            foot_plant - info["best_foot_plant_uphill"]
        ) / self._config.climb.target_stride_length,
        0.0,
        1.5,
    ) * first_contact
    hand_advance = jp.clip(
        (
            hand_plant - info["best_hand_plant_uphill"]
        ) / self._config.climb.target_stride_length,
        0.0,
        1.5,
    ) * first_hand_contact
    limb_touchdown_advance = (
        jp.sum(foot_advance) + jp.sum(hand_advance)
    ) * posture_gate
    # step() multiplies all rewards by dt. Divide this single-timestep event
    # here so a normalized target-length step receives its configured bonus.
    large_foot_step = (
        jp.sum(jp.square(foot_advance)) / self.dt * posture_gate
    )
    episode_high = jp.maximum(
        new_max - info["start_uphill_position"], 0.0
    )
    # Return 20% of accumulated ascent credit when an episode ends in a fall,
    # stall, or regression.  A full clawback made every early forward attempt
    # worthless because an untrained 29-DoF policy nearly always falls; this
    # partial clawback still makes a completed stride preferable to a lunge.
    failed_ascent = (
        done
        * episode_high
        / (self._config.climb.target_uphill_speed * self.dt)
    )
    phase = info["phase"][::-1]
    duty = self._config.climb.hand_contact_duty_factor
    # MODIFIED: each arm swings with the opposite leg. get_rz peaks at phase
    # zero, so contact must occupy the complementary interval around +/-pi.
    # The previous `cos(phase) > cos(pi*duty)` did the reverse: it requested a
    # planted palm at maximum swing height and prevented diagonal limb motion.
    desired_hand_contact = jp.cos(phase) < jp.cos(jp.pi * (1.0 - duty))
    schedule_match = jp.mean(hand_contact == desired_hand_contact)
    hand_clearance = jp.maximum(
        palm_pos @ self._slope_normal - self._terrain_plane_offset, 0.0
    )
    target_hand_clearance = gait.get_rz(
        phase, swing_height=self._config.reward_config.max_hand_height
    )
    hand_phase_reward = jp.exp(
        -jp.mean(jp.square(hand_clearance - target_hand_clearance)) / 0.005
    )
    # Actual (not merely scheduled) diagonal swing synchronization. Index
    # reversal pairs left hand/right foot and right hand/left foot.
    diagonal_swing_sync = jp.mean(
        ((~hand_contact) == (~contact[::-1])).astype(data.qpos.dtype)
    )
    # MODIFIED: measure lift from each palm's last meaningful plant height,
    # not the nominal plane, so the 0.8 ft target remains valid on boulders.
    hand_lift = jp.maximum(
        palm_pos @ self._slope_normal - info["last_hand_plant_normal"], 0.0
    )
    lift_fraction = jp.clip(
        hand_lift / self._config.reward_config.max_hand_height, 0.0, 1.0
    )
    smooth_lift = jp.square(lift_fraction) * (3.0 - 2.0 * lift_fraction)
    supported_hand_swing = (
        (~hand_contact)
        & hand_contact[::-1]
        & any_foot
        & (~desired_hand_contact)
    )
    hand_lift_height = (
        jp.sum(smooth_lift * supported_hand_swing) * posture_gate
    )
    new_lift_target = (
        (hand_lift >= self._config.reward_config.max_hand_height)
        & (~info["hand_lift_achieved"])
        & supported_hand_swing
    )
    # step() multiplies by dt; divide so crossing 0.8 ft earns one configured
    # event bonus rather than a tiny one-timestep pulse.
    hand_lift_target = (
        jp.sum(new_lift_target.astype(data.qpos.dtype))
        / self.dt
        * posture_gate
    )
    diagonal = (hand_contact[0] & contact[1]) | (
        hand_contact[1] & contact[0]
    )
    hand_slip_speed = jp.sqrt(
        jp.sum(jp.square(tangent_vel) * hand_contact) + 1e-8
    )
    foot_force_local = jp.array([
        mjx_env.get_sensor_data(self.mj_model, data, name)
        for name in ("left_foot_force", "right_foot_force")
    ])
    foot_force_world = jp.einsum(
        "nij,nj->ni", data.site_xmat[self._feet_site_id], foot_force_local
    )
    # MuJoCo's force sensors report force transmitted into the parent body;
    # the terrain reaction on each foot has the opposite sign.
    foot_uphill_force = jp.clip(
        -(foot_force_world @ self._slope_tangent)
        / (self._torso_mass * 9.81),
        0.0,
        3.0,
    )
    foot_uphill_drive = jp.mean(
        foot_uphill_force * contact.astype(data.qpos.dtype)
    ) * positive_progress * posture_gate
    rewards.update({
        # Linear credit makes every increment of forward speed useful during
        # early learning; saturation at the command removes an overspeed
        # incentive. Posture and mixed support reject fall-forward exploits.
        "uphill_progress": self._reward_uphill_velocity(
            uphill_velocity, posture_gate, mixed_support
        ),
        "assisted_uphill_progress": (
            positive_progress * any_hand * posture_gate
        ),
        "mountain_progress": potential_progress,
        "new_high_progress": new_high_progress,
        "waypoint_bonus": waypoint_bonus,
        "limb_touchdown_advance": limb_touchdown_advance,
        "large_foot_step": large_foot_step,
        "failed_ascent": failed_ascent,
        "crawl_height": jp.exp(
            -jp.square(
                pelvis_clearance
                - self._config.climb.target_pelvis_clearance
            ) / 0.01
        ),
        "mixed_support": mixed_support.astype(data.qpos.dtype),
        "foot_uphill_drive": foot_uphill_drive,
        "climb_time": jp.ones(()),
        "hand_proximity": (
            jp.mean(jp.exp(-plane_dist / 0.18)) * progress_gate
        ),
        # These two phase scaffolds are intentionally not progress-gated: the
        # actor must first learn to unload and exchange support before a full
        # body stride is discoverable. Their small scales cannot dominate
        # the signed pelvis potential.
        "continuous_hand_support": any_hand.astype(data.qpos.dtype),
        "hand_contact_schedule": schedule_match,
        "hand_phase": hand_phase_reward,
        "hand_lift_height": hand_lift_height,
        "hand_lift_target": hand_lift_target,
        "diagonal_swing_sync": diagonal_swing_sync,
        "diagonal_support": diagonal.astype(data.qpos.dtype),
        "hand_load_share": any_hand * progress_gate * jp.exp(
            -share_error / self._config.climb.hand_load_sigma
        ),
        "hand_slip": jp.sum(jp.square(tangent_vel) * hand_contact),
        "knee_clearance": knee_clearance_reward,
        "knee_contact": jp.mean(knee_contact.astype(data.qpos.dtype)),
    })
    metrics["climb/uphill_velocity"] = uphill_velocity
    metrics["climb/mountain_progress"] = mountain_progress
    metrics["climb/distance_from_start"] = (
        uphill_position - info["start_uphill_position"]
    )
    metrics["climb/distance_from_high"] = (
        uphill_position - info["max_uphill_position"]
    )
    metrics["climb/waypoint"] = current_waypoint.astype(data.qpos.dtype)
    metrics["climb/limb_touchdown_advance"] = limb_touchdown_advance
    metrics["climb/large_foot_step_bonus"] = large_foot_step * self.dt
    metrics["climb/pelvis_clearance"] = pelvis_clearance
    metrics["climb/posture_gate"] = posture_gate
    metrics["climb/steps_without_progress"] = info[
        "steps_without_progress"
    ].astype(data.qpos.dtype)
    metrics["climb/foot_uphill_force"] = jp.mean(foot_uphill_force)
    metrics["climb/hand_contact_fraction"] = jp.mean(
        hand_contact.astype(data.qpos.dtype)
    )
    metrics["climb/hand_phase"] = hand_phase_reward
    metrics["climb/hand_lift_height"] = jp.max(hand_lift)
    metrics["climb/hand_lift_target"] = jp.sum(
        new_lift_target.astype(data.qpos.dtype)
    )
    metrics["climb/diagonal_swing_sync"] = diagonal_swing_sync
    metrics["climb/hand_load_share"] = hand_share
    metrics["climb/diagonal_support"] = diagonal.astype(data.qpos.dtype)
    metrics["climb/hand_slip_speed"] = hand_slip_speed
    metrics["climb/knee_clearance_min"] = jp.min(knee_clearance)
    metrics["climb/knee_contact_fraction"] = jp.mean(
        knee_contact.astype(data.qpos.dtype)
    )
    return rewards

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

  def _reward_uphill_velocity(
      self,
      uphill_velocity: jax.Array,
      posture_gate: jax.Array,
      mixed_support: jax.Array,
  ) -> jax.Array:
    """Bounded command-speed reward for supported, controlled ascent."""
    speed_fraction = jp.clip(
        uphill_velocity / self._config.climb.target_uphill_speed,
        0.0,
        1.0,
    )
    return (
        speed_fraction
        * posture_gate
        * mixed_support.astype(speed_fraction.dtype)
    )

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
    if self._climb:
      return jp.sum(jp.square(torso_zaxis - self._slope_tangent))
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
    if self._climb:
      # MODIFIED: stock compares against absolute world Z, but the climb
      # heightfield is offset and inclined. Measure clearance normal to the
      # nominal terrain plane so the 0-15 cm gait curve is physically reachable.
      foot_z = jp.maximum(
          foot_pos @ self._slope_normal - self._terrain_plane_offset, 0.0
      )
    else:
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
    if self._climb:
      del rng
      return jp.array([
          self._config.climb.target_uphill_speed, 0.0, 0.0
      ])
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
