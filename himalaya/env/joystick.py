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
from himalaya.env import scene as scene_mod


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
          # Target climb rate for progress_uphill, in metres of HEIGHT per
          # second. On a 35 degree slope 0.30 m/s of height is about 0.52 m/s
          # along the ground -- a brisk but reachable climb. The old 0.8 was
          # inherited from when this measured ground speed; as a height cap it
          # would need 1.4 m/s along the ground, so it never bound and the
          # reward had no ceiling.
          max_uphill_speed=0.30,
          # Per-step cost of sitting on the starting platform, in the same
          # units as progress_uphill (metres of height per second). Set well
          # below max_uphill_speed so climbing always beats waiting, without
          # the penalty dwarfing the reward: at 0.5 against a 0.30 ceiling,
          # loitering cost -4.0 while the best climb earned +2.3, which
          # punishes the robot for existing on the platform harder than it can
          # ever be paid for climbing.
          platform_loiter=0.08,
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
    # MODIFIED: hand-floor contact. These sensors have existed in sensor.xml
    # since hands were given contact pairs, and NOTHING read them -- the policy
    # could not feel its own palms, so it could not learn to brace with them.
    self._hands_floor_found_sensor = [
        self._mj_model.sensor(side + "_hand_floor_found").id
        for side in ["left", "right"]
    ]

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
    # Start ON the slope, in one of the lanes, chosen at random. There is no
    # platform to retreat to: a flat unrewarded start became a safe harbour
    # the policy never left. Spawning part-way up means it begins already
    # committed to the climb, and the route it gets changes every episode.
    if self._lane is not None:
      qpos = qpos.at[0:2].set(jp.array(scene_mod.SPAWN))

    rng, key = jax.random.split(rng)
    # Spawn placement -- position, heading, body tilt, height -- comes from
    # scene.spawn_pose(). THE single definition, shared with scripts/view.py.
    #
    # Do not reimplement any part of it here. This block previously carried its
    # own copy of the yaw, the tilt, and three height corrections, while the
    # viewer carried a different subset; the two disagreed silently and the
    # viewer showed a robot facing a different way than training used.
    #
    # slope_rad is a compile-time constant, so the numpy call below is traced
    # once and folded into the graph rather than run per step.
    if self._slope_rad != 0.0:
      rng, key = jax.random.split(rng)
      jitter = jax.random.uniform(
          key, (), minval=-scene_mod.SPAWN_YAW_JITTER,
          maxval=scene_mod.SPAWN_YAW_JITTER)
      # The jitter is the only stochastic part, so build the pose at zero
      # jitter and rotate by the sampled angle afterwards.
      # terrain_height=None: the heightfield lookup stays in jax below, since
      # under jit it returns a tracer that numpy cannot consume.
      # Read the keyframe from the MODEL, not from self._init_q: the latter is
      # a jax array and becomes a tracer under jit, which numpy cannot consume.
      kf = np.array(self._mj_model.keyframe("knees_bent").qpos)
      base = scene_mod.spawn_pose(
          kf, self._slope_rad, float(kf[2]), terrain_height=None)
      qpos = qpos.at[0:7].set(jp.array(base[0:7]))
      # Same probe grid and max-reduction as scene.spawn_lift_relief, in jax.
      n = jp.array(self._slope_normal)
      offs = jp.array(scene_mod.probe_offsets())
      relief = jp.max(jax.vmap(
          lambda o: self.terrain_height_at(qpos[0:3] + o))(offs))
      qpos = qpos.at[0:3].add(relief * n)
      qpos = qpos.at[2].add(scene_mod.SPAWN_CLEARANCE)
      spin = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]),
                                     jp.array([jitter]))
      qpos = qpos.at[3:7].set(math.quat_mul(spin, qpos[3:7]))
    else:
      rng, key = jax.random.split(rng)
      yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
      quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
      qpos = qpos.at[3:7].set(math.quat_mul(qpos[3:7], quat))

    # qpos[7:]=*U(0.5, 1.5)
    #
    # Inherited from flat-ground walking, where a crumpled start is harmless
    # jitter. Note it is MULTIPLICATIVE: it cannot move a joint that sits at
    # zero, and it swings the large-angle joints hardest (the knee at 0.669 rad
    # spans 0.33-1.00). step() then commands _default_pose, so the robot spawns
    # in one configuration and is immediately told to snap to another.
    # _reset_jitter_off exists so spawn_check.py can isolate that effect.
    if not getattr(self, "_reset_jitter_off", False):
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
  # Height of the foot SITE above the sole when the foot is planted, metres.
  # gait.get_rz returns height above ground (0 at stance), so this offset has to
  # come off the measured site height or the gait clock is compared against a
  # target it can never reach. Measured on the knees_bent keyframe, flat ground:
  # both foot sites sit at world z = 0.0333.
  FOOT_SITE_OFFSET = 0.0333

  # Body-frame heightmap sample points, metres. 5x5 over +/-0.75 m: wide enough
  # to see both corridor walls at the 0.47-0.84 m floor widths the terrain bank
  # produces, fine enough that a 0.3 m step shows up in one cell.
  # Shoulder geometry, measured on the knees_bent keyframe.
  SHOULDER_ABOVE_PELVIS = 0.291
  # Lateral distances the wall sensor probes, metres. A palm reaches 0.56 m
  # (0.10 m shoulder offset + 0.46 m arm), so this brackets that with a little
  # either side: a wall closer than 0.3 m is one the robot is already against.
  WALL_PROBE_DIST = [0.30, 0.42, 0.54, 0.66, 0.80]
  # Furthest a palm can reach laterally: 0.10 m shoulder offset + 0.46 m arm.
  HAND_REACH = 0.56

  HEIGHTMAP_OFFSETS = [
      [x, y]
      for x in (-0.75, -0.375, 0.0, 0.375, 0.75)
      for y in (-0.75, -0.375, 0.0, 0.375, 0.75)
  ]

  MAX_TILT = 0.5           # gravity-z; 1.0 = upright, 0.0 = horizontal (~60 deg)
  MIN_TORSO_HEIGHT = 0.4   # metres; nominal standing pelvis is ~0.78

  # Slope-relative fall limits, used when slope_deg != 0.
  #
  # A climber on all fours is legitimately pitched 40-70 degrees off the
  # surface normal (cos(70) = 0.34), so TILT_TOL has to sit well below the
  # bipedal MAX_TILT or correct behaviour terminates. With the tilt check that
  # permissive, the clearance check is what actually catches a fall.
  # MODIFIED 0.0 -> 0.5. The 0.0 was set for a crawler, which is legitimately
  # pitched 40-70 degrees off the surface normal; for an upright walker it
  # permits lying face-down on the rock and calling the episode alive. 0.5
  # matches the bipedal MAX_TILT, measured against the surface instead of the
  # world.
  TILT_TOL = 0.5            # dot(torso_up, slope_normal); 1.0 = normal to slope
  # Pelvis height above its own lowest foot, along the surface normal. Healthy
  # standing measures 0.73-0.77 m; the bipedal MIN_TORSO_HEIGHT of 0.4 is the
  # same fraction of a 0.78 m nominal, so 0.40 keeps the two tasks consistent.
  # The old 0.25 was sized for a different quantity (height above the mean
  # plane) and against this one would let the pelvis fall to a third of
  # standing height before the episode ended.
  MIN_SLOPE_CLEARANCE = 0.40

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    if self._slope_rad != 0.0:
      # MODIFIED: on a slope both stock checks measure against world vertical,
      # which condemns the target behaviour. A robot aligned to a 45 degree
      # slope reads gravity_z = 0.707 and its pelvis height is meaningless
      # once the ground rises with distance. Measure both against the surface.
      n = jp.array(self._slope_normal)
      fall_termination = jp.dot(self.get_gravity(data, "torso"), n) < self.TILT_TOL
      # MODIFIED: "am I standing" measured as pelvis height above MY OWN
      # LOWEST FOOT, along the slope normal. No terrain lookup at all.
      #
      # Two earlier versions of this check were wrong, both for the same
      # reason -- they measured against the world instead of against the robot:
      #
      #   dot(qpos, n) alone is height above the plane the terrain is built on,
      #   which on a heightfield with a metre of relief says nothing about
      #   posture: upright in a trough reads LOWER than lying flat on a peak.
      #
      #   Subtracting terrain_height_at looks like the fix but is not. That
      #   function bilinearly interpolates the heightfield, and on 48 degree
      #   faces the interpolated surface under the pelvis can sit well above
      #   the rock the feet are actually on. Measured: a robot with tilt 0.970
      #   -- upright to within 14 degrees of perpendicular -- read clearance
      #   0.162 against a 0.25 threshold and was terminated as fallen.
      #
      # Pelvis-above-lowest-foot is the quantity the check was always reaching
      # for. It is invariant to terrain entirely, so no interpolation can fool
      # it. Measured on healthy spawns: 0.73-0.77 m.
      foot_pos = data.site_xpos[self._feet_site_id]
      lowest_foot = jp.min(foot_pos @ n)
      clearance = jp.dot(data.qpos[0:3], n) - lowest_foot
      fall_termination |= clearance < self.MIN_SLOPE_CLEARANCE
    else:
      # MODIFIED: was `< 0.0`. Fire at ~60 degrees rather than waiting for 90,
      # and treat a pelvis on the ground as down whatever the orientation --
      # the case the stock height-free check misses entirely.
      fall_termination = self.get_gravity(data, "torso")[-1] < self.MAX_TILT
      fall_termination |= data.qpos[2] < self.MIN_TORSO_HEIGHT
    # MODIFIED: foot-to-foot contact ends the episode again, on slopes too.
    #
    # It was disabled because it fired within ~6 steps in 60 of 60 CRAWLING
    # episodes, where a narrow stance is correct. The task is now an upright
    # walk, and for a walker the feet touching means the legs have crossed,
    # which is a genuine fall. The shin checks were never disabled: a foot
    # against the opposite shin is a tangle at any slope.
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

    # Route heading rotated into the body frame. yaw_inv turns a world xy
    # vector into the pelvis's own left/forward axes.
    route_world = self.route_dir_at(data.qpos[0:3])
    fwd = math.rotate(jp.array([1.0, 0.0, 0.0]), data.qpos[3:7])[:2]
    fwd = fwd / (jp.linalg.norm(fwd) + 1e-6)
    left = jp.array([-fwd[1], fwd[0]])
    route_local = jp.array([jp.dot(route_world, fwd),
                            jp.dot(route_world, left)])

    # Terrain around the robot, in its OWN frame: a 5x5 grid over ~1.5 m,
    # rotated by the pelvis yaw, reported as height RELATIVE to the pelvis.
    #
    # Without this the policy has no idea a wall exists. It was being asked to
    # brace against geometry it could not perceive. Relative to the pelvis, not
    # absolute relief, because what matters is what is above and below the
    # robot -- not where the terrain's mean plane happens to sit.
    #
    # Measured cost: 0.035 ms whether the grid is 9 points or 36, because it is
    # one fused gather. There is no reason to be stingy.
    n_up = jp.array(self._slope_normal)
    grid = jp.array(self.HEIGHTMAP_OFFSETS)          # (25, 2) body-frame x,y
    rot = jp.array([[fwd[0], -fwd[1]], [fwd[1], fwd[0]]])
    world_xy = (rot @ grid.T).T + data.qpos[0:2]
    pel_h = jp.dot(data.qpos[0:3], n_up)
    def _h(xy):
      p3 = jp.array([xy[0], xy[1], data.qpos[2]])
      # terrain_height_at gives relief above the mean plane along the normal;
      # pel_h is the pelvis measured the same way. The difference is how far
      # the ground sits below the robot: negative underfoot, positive where a
      # wall rises past it.
      return self.terrain_height_at(p3, self.mjx_model.hfield_data) - pel_h
    heightmap = jp.clip(jax.vmap(_h)(world_xy), -1.5, 1.5)

    # WALL SENSOR: what a hand would find, on each side.
    #
    # The heightmap above tells the robot where the FLOOR is. It samples at
    # +/-0.375 and +/-0.75 m laterally, and a braced palm sits at ~0.5 m --
    # right between two samples -- so the wall a hand would actually touch was
    # never observed. It also reports one height per column, so there is no
    # sense of a wall FACE at shoulder level, only ground underfoot.
    #
    # This probes the band a palm can reach (shoulder is 0.10 m lateral, the
    # arm reaches 0.46 m, so 0.56 m out) at several distances and reports, per
    # side: how far away the first bracing surface is, and how high it stands
    # relative to the shoulder. That is the information the hands need and the
    # heightmap cannot carry.
    sh_h = pel_h + self.SHOULDER_ABOVE_PELVIS
    def _wall(side_sign):
      lat = left * side_sign
      def probe(dist):
        xy = data.qpos[0:2] + lat * dist
        p3 = jp.array([xy[0], xy[1], data.qpos[2]])
        return self.terrain_height_at(p3, self.mjx_model.hfield_data)
      hs = jax.vmap(probe)(jp.array(self.WALL_PROBE_DIST))
      # Report the HIGHEST surface in reach, not the first one found.
      #
      # Taking the first hit returned the wall's base -- measured, every side
      # read -0.41 to -0.51 m, which is the ground rising, not a face a palm
      # can press on. The tallest point in the reachable band is what the hand
      # actually meets.
      reachable = jp.array(self.WALL_PROBE_DIST) <= self.HAND_REACH
      cand = jp.where(reachable, hs, -1e3)
      idx = jp.argmax(cand)
      top = cand[idx]
      # Anything within a downward arm-span of the shoulder is reachable: the
      # arm can press down-and-out, not only straight sideways. 0.35 m was too
      # strict and reported "no wall" everywhere, because these corridors are
      # V-shaped -- at hand reach the surface sits 0.25-0.67 m below the
      # shoulder and only rises past it further out than the arm can go.
      found = top > sh_h - 0.70
      dist = jp.where(found, jp.array(self.WALL_PROBE_DIST)[idx], 1.0)
      rel_h = jp.where(found, top - sh_h, -1.0)
      return jp.array([dist, rel_h, found.astype(jp.float32)])
    wall_sense = jp.concatenate([_wall(1.0), _wall(-1.0)])  # left, right

    hand_contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sid]] > 0
        for sid in self._hands_floor_found_sensor
    ], dtype=jp.float32)

    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        info["command"],  # 3
        noisy_joint_angles - self._default_pose,  # 29
        noisy_joint_vel,  # 29
        info["last_act"],  # 29
        phase,
        # MODIFIED: which way the corridor runs, in the BODY frame. 2 dims.
        #
        # progress_uphill pays for velocity along the route, but nothing in the
        # observation said where the route went -- the policy was being paid to
        # move in a direction it could not perceive, so it could only find the
        # corridor by trial and error against a reward it could not attribute.
        # Measured over four runs, that produced rising reward curves with net
        # height of -0.31, -0.43, -0.25 and +0.01 m.
        #
        # Body frame, not world: a real robot knows where a wall is relative to
        # itself, not its global heading. Zero on flat ground and on any scene
        # without routes, so nothing else changes.
        route_local,  # 2
        heightmap,  # 25
        wall_sense,  # 6: per side, distance / height vs shoulder / found
        hand_contact,  # 2
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
    # MODIFIED: speed ALONG THE CORRIDOR, not height gained.
    #
    # dot(v, world_up) was the wrong objective for this map. The corridors run
    # roughly across the fall line -- measured at the spawn, the low ground
    # bears 90-120 degrees while every other heading meets a 0.4-0.8 m bank --
    # so the fastest way to gain height is to charge straight at a bank, and
    # the reward paid MOST for abandoning the route. Two runs drifted downhill
    # under it while their reward curves rose.
    #
    # The route tangent is derived from the heightfield itself (base.py builds
    # the field; make_route.py traces the lines through the troughs), so it
    # cannot disagree with the terrain the robot is standing on. It names the
    # goal without prescribing a gait, the same way tracking_lin_vel names a
    # target speed.
    d = self.route_dir_at(data_pos)
    along = jp.dot(global_linvel[:2], d)
    up = jp.dot(global_linvel, jp.array(self._slope_normal_up))
    # GATE the route term on actually gaining height, rather than blending it
    # with height.
    #
    # The blend was 0.8*along + 0.2*up, and the corridor bears ~200 degrees at
    # the spawn -- mostly lateral. So a robot sliding sideways along the route
    # collected nearly the full reward while gaining no height, and that is
    # exactly what Run D did: reward +2.79, episode length 379, net height
    # -0.25 m. Following the corridor should only pay when the corridor is
    # taking you UP.
    #
    # Below the gate the term falls back to height alone, so descending still
    # costs and there is no flat region for the policy to sit in.
    climbing = up > 0.0
    routed = jp.linalg.norm(d) > 0.5
    speed = jp.where(routed & climbing, 0.7 * along + 0.3 * up, up)
    # SYMMETRIC clip. It was [-1.0, +0.30], penalising lost height 3.3x harder
    # than gained height -- so under an exploring policy the expected value of
    # moving at all was negative, and the optimal response was to stand still.
    cap = self._config.reward_config.max_uphill_speed
    reward = jp.clip(speed, -cap, cap)
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
    # MODIFIED: on a slope, "upright" means perpendicular to the HILL, not to
    # the world. The stock target is world-vertical, so a robot leaning into a
    # 35 degree slope -- which is what staying over its feet requires -- pays
    # this cost continuously.
    #
    # Kept, not zeroed. This term is the difference between walking up the hill
    # and crawling up it, and the whole point of the task is that the robot
    # walks. The scale drops -2.0 -> -1.0 in climb_config because a climber
    # legitimately leans more than a walker does.
    if self._slope_rad == 0.0:
      return jp.sum(jp.square(torso_zaxis - jp.array([0.073, 0.0, 1.0])))
    # Same 0.073 forward lean, applied about the slope normal.
    n = jp.array(self._slope_normal)
    lean = jp.array([0.073 * jp.cos(self._slope_rad), 0.0,
                     -0.073 * jp.sin(self._slope_rad)])
    return jp.sum(jp.square(torso_zaxis - (n + lean)))

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
    #
    # MODIFIED: measure foot height along the SLOPE NORMAL, above the terrain
    # directly underfoot -- not world z.
    #
    # get_rz returns height ABOVE GROUND (0 at stance, swing_height at peak),
    # but foot_z is the site's world z, which reads FOOT_SITE_OFFSET when the
    # foot is planted. On flat ground that constant error costs a little
    # (exp(-0.0333^2/0.01) = 0.80). On a slope it is fatal: two feet one
    # stride apart differ in world z by 0.5*tan(35 deg) = 0.35 m, and
    # exp(-0.35^2/0.01) = 4.8e-06. The term is not "fighting" climbing, it is
    # IDENTICALLY ZERO -- 634 of walk4_rough's 1400 reward points vanish the
    # moment the floor tilts. It cannot be tuned, only fixed.
    foot_pos = data.site_xpos[self._feet_site_id]
    if self._slope_rad == 0.0:
      foot_z = foot_pos[..., -1] - self.FOOT_SITE_OFFSET
    else:
      n = jp.array(self._slope_normal)
      # Clearance above the rock: distance from the mean plane along the
      # normal, minus the terrain relief in that column.
      along = foot_pos @ n
      relief = jax.vmap(self.terrain_height_at)(foot_pos)
      foot_z = along - relief - self.FOOT_SITE_OFFSET
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

    cmd = jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw])
    # MODIFIED: no zero-command episodes on a slope.
    #
    # On flat ground standing still is a valid skill worth 10% of episodes. On a
    # 35 degree hill a zero command instructs the robot to stand still on a
    # slope it cannot hold station on (mu >= tan(35) = 0.70 against a randomised
    # U(0.4, 1.0)), and standing still is the safe-harbour behaviour that
    # already ended two runs with the policy refusing to move.
    if self._slope_rad != 0.0:
      return cmd
    # With 10% chance, set everything to zero.
    return jp.where(
        jax.random.bernoulli(rng4, p=0.1),
        jp.zeros(3),
        cmd,
    )
