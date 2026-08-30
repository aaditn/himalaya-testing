"""Proprioceptive G1 four-contact balance on a 30-degree rough ramp."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
import numpy as np

from mujoco_playground._src import mjx_env

from himalaya.env import joystick as g1_joystick
from .g1_cfg import G1_ACTION_SIZE, validate_four_contact_slope


def default_four_contact_config() -> config_dict.ConfigDict:
    """Return the G1 config specialized for four-contact balance acquisition."""

    cfg = g1_joystick.default_config()
    with cfg.unlocked():
        cfg.slope_degrees = 30.0
        cfg.spawn_yaw_jitter_degrees = 0.5
        cfg.spawn_joint_jitter = 0.002
        cfg.spawn_velocity_jitter = 0.0
        cfg.spawn_drop_height_range = [0.0, 0.02]
        cfg.starter_patch_roughness_scale = 0.20
        cfg.balance_objective = True
        cfg.uphill_command_range = [0.0, 0.0]
        cfg.command_stand_probability = 1.0
        cfg.action_scale = 0.18
        cfg.push_config.enable = False
        cfg.njmax = 192

        scales = cfg.reward_config.scales
        # Balance acquisition: remain quiet and supported.  Locomotion is a
        # later stage, so no progress incentive is present here.
        scales.tracking_lin_vel = 3.0
        scales.tracking_ang_vel = 1.0
        scales.uphill_progress = 0.0
        scales.backward_slide = -2.0
        scales.lin_vel_z = -1.0
        scales.ang_vel_xy = -0.25
        scales.orientation = -1.5
        scales.base_height = 0.0
        scales.torques = 0.0
        scales.action_rate = -0.05
        scales.energy = -0.0003
        scales.dof_acc = 0.0
        scales.feet_air_time = 0.0
        scales.feet_phase = 0.0
        scales.feet_clearance = 0.0
        scales.feet_height = 0.0
        scales.joint_deviation_hip = 0.0
        scales.joint_deviation_knee = 0.0
        scales.stand_still = 0.0
        scales.alive = 1.0
        scales.collision = -0.10
        scales.contact_force = -0.01
        scales.dof_pos_limits = -1.0
        scales.pose = -0.35
        scales.leg_propulsion = 0.0
        scales.bilateral_hand_use = 0.5
        scales.hand_alternation = 0.0
        scales.double_hand_contact = 0.5
        scales.four_contact_balance = 2.0
        scales.root_stationarity = -2.0
        scales.feet_slip = -0.50
        scales.hand_slip = -0.50
        scales.hand_force = -0.50
        scales.wrist_moment = -0.25
        scales.actuator_saturation = -0.25
        # HumoSlope Stage-I terrain-aligned ZMP surrogate, extended from the
        # two feet to the force-weighted hand-and-foot support set.
        scales.terrain_zmp = 5.0
        scales.com_height = 1.0

        cfg.reward_config.leg_propulsion_target = 0.75
        cfg.reward_config.max_hand_force = 120.0
        cfg.reward_config.max_wrist_moment = 8.0
        cfg.reward_config.zmp_sigma = 0.12
        cfg.reward_config.zmp_epsilon = 1.0e-3
        cfg.reward_config.com_height_target = 0.36
        cfg.reward_config.com_height_sigma = 0.08
        cfg.reward_config.torso_tilt_degrees = 70.0
        cfg.validation_success_distance_m = 2.0
        cfg.validation_min_speed_mps = 0.08
        cfg.validation_max_speed_mps = 0.26
        cfg.validation_max_hand_slip_mps = 0.15
        cfg.validation_max_foot_slip_mps = 0.18
    return cfg


class HimalayaG1FourContactEnv(g1_joystick.Joystick):
    """G1 crawl whose actor receives only stock proprioceptive observations.

    Simulated contact state, contact force, wrist moment, and terrain angle are
    privileged critic/reward/diagnostic values and are never actor inputs.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: Optional[
            Dict[str, Union[str, int, float, bool, list[Any]]]
        ] = None,
    ) -> None:
        cfg = config or default_four_contact_config()
        slope = validate_four_contact_slope(cfg.slope_degrees)
        # Clean training containers do not have Menagerie materialized until
        # Playground performs this initialization.
        mjx_env.ensure_menagerie_exists()
        super().__init__(
            task="four_contact_terrain",
            config=cfg,
            config_overrides=config_overrides,
        )

        self._slope_degrees = slope
        self._slope_radians = math.radians(slope)
        c, s = math.cos(self._slope_radians), math.sin(self._slope_radians)
        self._ramp_tangent = jp.array([c, 0.0, s])
        self._ramp_cross = jp.array([0.0, 1.0, 0.0])
        self._ramp_normal = jp.array([-s, 0.0, c])
        self._ramp_quat = jp.array(
            [math.cos(self._slope_radians / 2), 0.0,
             -math.sin(self._slope_radians / 2), 0.0]
        )

        floor_id = self._mj_model.geom("floor").id
        self._mj_model.geom_quat[floor_id] = np.asarray(self._ramp_quat)
        self._flatten_crawl_start_patch()
        # MODIFIED: crampon replacement feet retain the stock collision
        # geometry and named foot-floor pairs; only tangential friction is
        # changed.  No additional foot attachment is modeled.
        for side in ("left", "right"):
            pair_id = self._mj_model.pair(f"{side}_foot_floor").id
            self._mj_model.pair_friction[pair_id, :2] = 1.0
        self._mjx_model = mjx.put_model(self._mj_model, impl=cfg.impl)

        crawl = self._mj_model.keyframe("four_contact_crawl")
        self._init_q = jp.asarray(crawl.qpos)
        self._default_pose = jp.asarray(crawl.qpos[7:])
        self._nominal_root_quat = jp.asarray(crawl.qpos[3:7])

        self._hand_floor_sensor_ids = [
            self._mj_model.sensor(f"{side}_hand_floor_found").id
            for side in ("left", "right")
        ]
        self._hand_velocity_adr = jp.array([
            list(range(
                self._mj_model.sensor_adr[
                    self._mj_model.sensor(f"{side}_palm_global_linvel").id
                ],
                self._mj_model.sensor_adr[
                    self._mj_model.sensor(f"{side}_palm_global_linvel").id
                ] + 3,
            ))
            for side in ("left", "right")
        ])
        self._body_masses = jp.asarray(self._mj_model.body_mass)
        self._total_mass = jp.sum(self._body_masses)
        self._wrist_actuator_ids = jp.array([
            self._mj_model.actuator(f"{side}_wrist_{axis}_joint").id
            for side in ("left", "right")
            for axis in ("roll", "pitch", "yaw")
        ])
        leg_names = [
            f"{side}_{joint}_joint"
            for side in ("left", "right")
            for joint in (
                "hip_pitch", "hip_roll", "hip_yaw", "knee",
                "ankle_pitch", "ankle_roll",
            )
        ]
        waist_names = [
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"
        ]
        arm_names = [
            f"{side}_{joint}_joint"
            for side in ("left", "right")
            for joint in (
                "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                "wrist_roll", "wrist_pitch", "wrist_yaw",
            )
        ]
        self._leg_actuator_ids = jp.array([
            self._mj_model.actuator(name).id for name in leg_names
        ])
        self._waist_actuator_ids = jp.array([
            self._mj_model.actuator(name).id for name in waist_names
        ])
        self._arm_actuator_ids = jp.array([
            self._mj_model.actuator(name).id for name in arm_names
        ])
        actuator_joint_ids = self._mj_model.actuator_trnid[:, 0]
        self._actuator_dof_ids = jp.asarray(
            self._mj_model.jnt_dofadr[actuator_joint_ids]
        )
        self._actuator_force_limits = jp.asarray(
            np.max(
                np.abs(self._mj_model.jnt_actfrcrange[actuator_joint_ids]),
                axis=1,
            )
        )
        assert len(leg_names) == 12 and len(waist_names) == 3
        assert len(arm_names) == 14

    def _flatten_crawl_start_patch(self) -> None:
        """Create a tangent settling footprint inside the rough heightfield.

        The measured prone keyframe was solved against a flat tangent plane.
        A compact zero-height patch lets all four limbs begin in that stable
        geometry; a 25 cm cosine transition leads immediately into the full
        5 cm rocky field.  This changes terrain only, not robot geometry.
        """

        if self._mj_model.nhfield != 1:
            raise ValueError("crawl scene must contain exactly one heightfield")
        rows = int(self._mj_model.hfield_nrow[0])
        cols = int(self._mj_model.hfield_ncol[0])
        address = int(self._mj_model.hfield_adr[0])
        count = rows * cols
        heights = self._mj_model.hfield_data[address : address + count].reshape(
            rows, cols
        )
        half_x, half_y = self._mj_model.hfield_size[0, :2]
        x = np.linspace(-half_x, half_x, cols)
        y = np.linspace(-half_y, half_y, rows)
        xx, yy = np.meshgrid(x, y)
        # The prone G1 support polygon is approximately 1.2 m by 0.7 m.
        # Leave margin for the small reset-position jitter.
        outside_x = np.clip((np.abs(xx) - 0.75) / 0.25, 0.0, 1.0)
        outside_y = np.clip((np.abs(yy) - 0.45) / 0.25, 0.0, 1.0)
        blend = np.maximum(outside_x, outside_y)
        blend = blend * blend * (3.0 - 2.0 * blend)
        roughness_floor = float(self._config.starter_patch_roughness_scale)
        self._mj_model.hfield_data[address : address + count] = (
            heights * (roughness_floor + (1.0 - roughness_floor) * blend)
        ).ravel()

    @property
    def slope_degrees(self) -> float:
        return self._slope_degrees

    @property
    def ramp_tangent(self) -> jax.Array:
        return self._ramp_tangent

    @property
    def ramp_normal(self) -> jax.Array:
        return self._ramp_normal

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = super().reset(rng)
        rng = state.info["rng"]
        rng, pos_rng, joint_rng, vel_rng, yaw_rng, drop_rng = jax.random.split(rng, 6)

        qpos = self._init_q
        # Keep all four limbs on the audited tangent starter footprint.
        uv = jax.random.uniform(pos_rng, (2,), minval=-0.04, maxval=0.04)
        plane_point = uv[0] * self._ramp_tangent + uv[1] * self._ramp_cross
        drop_height = jax.random.uniform(
            drop_rng, (), minval=self._config.spawn_drop_height_range[0],
            maxval=self._config.spawn_drop_height_range[1],
        )
        qpos = qpos.at[:3].set(
            plane_point + (self._init_q[2] + drop_height) * self._ramp_normal
        )
        joint_noise = jax.random.uniform(
            joint_rng, (29,),
            minval=-self._config.spawn_joint_jitter,
            maxval=self._config.spawn_joint_jitter,
        )
        qpos = qpos.at[7:].set(jp.clip(
            self._default_pose + joint_noise, self._lowers, self._uppers
        ))
        yaw_limit = math.radians(self._config.spawn_yaw_jitter_degrees)
        yaw = jax.random.uniform(yaw_rng, (), minval=-yaw_limit, maxval=yaw_limit)
        yaw_quat = mjx_math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)
        local_quat = mjx_math.quat_mul(yaw_quat, self._nominal_root_quat)
        qpos = qpos.at[3:7].set(mjx_math.quat_mul(self._ramp_quat, local_quat))
        qvel = jax.random.uniform(
            vel_rng, (self.mjx_model.nv,),
            minval=-self._config.spawn_velocity_jitter,
            maxval=self._config.spawn_velocity_jitter,
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
        foot_contact = self._foot_contacts(data)
        com = self._com_position(data)
        state.info["rng"] = rng
        state.info["last_com_pos"] = com
        state.info["last_com_vel"] = jp.zeros(3)
        state.info["start_progress"] = jp.dot(com, self._ramp_tangent)
        state.info["last_hand_contact"] = jp.zeros(2, dtype=bool)
        state.info["hand_contact_ema"] = jp.zeros(2)
        obs = self._get_obs(data, state.info, foot_contact)
        state = state.replace(
            data=data, obs=obs, reward=jp.zeros(()), done=jp.zeros(())
        )
        return self._set_diagnostics(state, foot_contact)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        next_state = super().step(state, action)
        com = self._com_position(next_state.data)
        com_vel = (com - next_state.info["last_com_pos"]) / self.dt
        next_state.info["last_com_pos"] = com
        next_state.info["last_com_vel"] = com_vel
        hand_contact = self._hand_contacts(next_state.data)
        next_state.info["last_hand_contact"] = hand_contact
        next_state.info["hand_contact_ema"] = (
            0.98 * next_state.info["hand_contact_ema"]
            + 0.02 * hand_contact.astype(jp.float32)
        )
        return self._set_diagnostics(
            next_state, self._foot_contacts(next_state.data)
        )

    def _foot_contacts(self, data: mjx.Data) -> jax.Array:
        return jp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self._feet_floor_found_sensor
        ])

    def _hand_contacts(self, data: mjx.Data) -> jax.Array:
        return jp.array([
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self._hand_floor_sensor_ids
        ])

    def _hand_forces(self, data: mjx.Data) -> jax.Array:
        return jp.stack([
            mjx_env.get_sensor_data(self.mj_model, data, f"{side}_hand_force")
            for side in ("left", "right")
        ])

    def _hand_torques(self, data: mjx.Data) -> jax.Array:
        return jp.stack([
            mjx_env.get_sensor_data(self.mj_model, data, f"{side}_hand_torque")
            for side in ("left", "right")
        ])

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:
        obs = super()._get_obs(data, info, contact)
        hand_forces = self._hand_forces(data)
        hand_contact = self._hand_contacts(data)

        slope = self._slope_radians
        descriptor = jp.array([
            slope, 0.0, abs(slope),
            (info["command"][0] >= 0).astype(jp.float32),
            (info["command"][0] < 0).astype(jp.float32),
        ])
        obs["privileged_state"] = jp.hstack([
            obs["privileged_state"], descriptor,
            hand_contact.astype(jp.float32),
            hand_forces.ravel() / 150.0,
            self._hand_torques(data).ravel() / 10.0,
            data.sensordata[self._hand_velocity_adr].ravel(),
        ])
        return obs

    def _get_reward(
        self, data: mjx.Data, action: jax.Array, info: dict[str, Any],
        metrics: dict[str, Any], done: jax.Array, first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact
        )
        global_linvel = self.get_global_linvel(data, "pelvis")
        ramp_velocity = jp.array([
            jp.dot(global_linvel, self._ramp_tangent),
            jp.dot(global_linvel, self._ramp_cross),
        ])
        rewards["tracking_lin_vel"] = self._reward_tracking_lin_vel(
            info["command"], ramp_velocity
        )
        yaw_rate = jp.dot(self.get_global_angvel(data, "pelvis"), self._ramp_normal)
        rewards["tracking_ang_vel"] = jp.exp(
            -jp.square(info["command"][2] - yaw_rate)
            / self._config.reward_config.tracking_sigma
        )
        hand_contact = self._hand_contacts(data)
        hand_force = jp.linalg.norm(self._hand_forces(data), axis=-1)
        foot_force = jp.array([
            jp.linalg.norm(mjx_env.get_sensor_data(
                self.mj_model, data, f"{side}_foot_force"
            )) for side in ("left", "right")
        ])
        uphill_speed = jp.dot(global_linvel, self._ramp_tangent)
        rewards["uphill_progress"] = jp.clip(uphill_speed, 0.0, 0.30) / 0.30
        rewards["backward_slide"] = jp.clip(-uphill_speed, 0.0, 0.30) / 0.30
        leg_power, arm_power, _ = self._positive_mechanical_power(data)
        leg_fraction = leg_power / (leg_power + arm_power + 1.0e-6)
        rewards["leg_propulsion"] = jp.minimum(
            leg_fraction / self._config.reward_config.leg_propulsion_target, 1.0
        )
        rewards["bilateral_hand_use"] = jp.sqrt(jp.prod(
            jp.clip(info["hand_contact_ema"], 0.0, 1.0)
        ))
        single_now = jp.logical_xor(hand_contact[0], hand_contact[1])
        single_last = jp.logical_xor(
            info["last_hand_contact"][0], info["last_hand_contact"][1]
        )
        switched_side = jp.any(hand_contact != info["last_hand_contact"])
        rewards["hand_alternation"] = (
            single_now & single_last & switched_side
        ).astype(jp.float32)
        rewards["double_hand_contact"] = jp.all(hand_contact).astype(jp.float32)
        rewards["four_contact_balance"] = jp.all(
            jp.hstack([contact, hand_contact])
        ).astype(jp.float32)
        global_angvel = self.get_global_angvel(data, "pelvis")
        rewards["root_stationarity"] = (
            jp.sum(jp.square(global_linvel))
            + 0.1 * jp.sum(jp.square(global_angvel))
        )
        rewards["hand_slip"] = self._cost_hand_slip(data, hand_contact)
        force_excess = jp.clip(
            hand_force / self._config.reward_config.max_hand_force - 1.0,
            min=0.0,
        )
        rewards["hand_force"] = jp.sum(jp.square(force_excess))
        wrist_moment = jp.linalg.norm(self._hand_torques(data), axis=-1)
        moment_excess = jp.clip(
            wrist_moment / self._config.reward_config.max_wrist_moment - 1.0,
            min=0.0,
        )
        rewards["wrist_moment"] = jp.sum(jp.square(moment_excess))
        rewards["actuator_saturation"] = jp.mean(
            jp.square(jp.clip(
                jp.abs(data.actuator_force) / self._actuator_force_limits - 0.95,
                min=0.0,
            ))
        )
        rewards["terrain_zmp"] = self._reward_terrain_zmp(
            data, info, foot_force, hand_force
        )
        rewards["com_height"] = self._reward_com_height(data)
        return rewards

    def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        tilt = math.radians(self._config.reward_config.torso_tilt_degrees)
        target = math.cos(tilt) * self._ramp_normal + math.sin(tilt) * self._ramp_tangent
        return jp.sum(jp.square(torso_zaxis - target))

    def _cost_feet_slip(
        self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
    ) -> jax.Array:
        del info
        velocity = data.sensordata[self._foot_linvel_sensor_adr]
        normal_velocity = velocity @ self._ramp_normal
        tangent_velocity = velocity - normal_velocity[:, None] * self._ramp_normal
        return jp.sum(jp.linalg.norm(tangent_velocity, axis=-1) * contact)

    def _cost_hand_slip(self, data: mjx.Data, contact: jax.Array) -> jax.Array:
        velocity = data.sensordata[self._hand_velocity_adr]
        normal_velocity = velocity @ self._ramp_normal
        tangent_velocity = velocity - normal_velocity[:, None] * self._ramp_normal
        return jp.sum(jp.linalg.norm(tangent_velocity, axis=-1) * contact)

    def _positive_mechanical_power(
        self, data: mjx.Data
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return positive leg, arm, and waist power from named actuators."""

        power = jp.clip(
            data.actuator_force * data.qvel[self._actuator_dof_ids], min=0.0
        )
        return (
            jp.sum(power[self._leg_actuator_ids]),
            jp.sum(power[self._arm_actuator_ids]),
            jp.sum(power[self._waist_actuator_ids]),
        )

    def _reward_terrain_zmp(
        self, data: mjx.Data, info: dict[str, Any],
        foot_force: jax.Array, hand_force: jax.Array,
    ) -> jax.Array:
        positions = jp.vstack([
            data.site_xpos[self._feet_site_id],
            data.site_xpos[self._hands_site_id],
        ])
        weights = jp.hstack([foot_force, hand_force]) + 1.0e-3
        support = jp.sum(positions * weights[:, None], axis=0) / jp.sum(weights)
        com = self._com_position(data)
        com_vel = (com - info["last_com_pos"]) / self.dt
        com_acc = (com_vel - info["last_com_vel"]) / self.dt
        apparent = jp.array([0.0, 0.0, -9.81]) - com_acc
        denominator = jp.dot(apparent, self._ramp_normal)
        eps = self._config.reward_config.zmp_epsilon
        denominator = jp.where(jp.abs(denominator) < eps, -eps, denominator)
        t = jp.dot(support - com, self._ramp_normal) / denominator
        zmp = com + t * apparent
        return jp.exp(
            -jp.linalg.norm(zmp - support) / self._config.reward_config.zmp_sigma
        )

    def _reward_com_height(self, data: mjx.Data) -> jax.Array:
        error = jp.dot(self._com_position(data), self._ramp_normal)
        error -= self._config.reward_config.com_height_target
        sigma = self._config.reward_config.com_height_sigma
        return jp.exp(-jp.square(error) / jp.square(sigma))

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        torso_z = self.get_gravity(data, "torso")
        alignment = jp.dot(torso_z, self._ramp_normal)
        root_height = jp.dot(data.qpos[:3], self._ramp_normal)
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
            (alignment < 0.05) | (root_height < 0.20) | contact_termination
            | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        )

    def sample_command(self, rng: jax.Array) -> jax.Array:
        speed_rng, stand_rng = jax.random.split(rng)
        speed = jax.random.uniform(
            speed_rng,
            minval=self._config.uphill_command_range[0],
            maxval=self._config.uphill_command_range[1],
        )
        command = jp.array([speed, 0.0, 0.0])
        return jp.where(
            jax.random.bernoulli(
                stand_rng, p=self._config.command_stand_probability
            ),
            jp.zeros(3), command,
        )

    def _com_position(self, data: mjx.Data) -> jax.Array:
        return jp.sum(data.xipos * self._body_masses[:, None], axis=0) / self._total_mass

    def _set_diagnostics(
        self, state: mjx_env.State, foot_contact: jax.Array
    ) -> mjx_env.State:
        data = state.data
        com = self._com_position(data)
        progress = jp.dot(com, self._ramp_tangent) - state.info["start_progress"]
        hand_contact = self._hand_contacts(data)
        hand_force = jp.linalg.norm(self._hand_forces(data), axis=-1)
        foot_force = jp.array([
            jp.linalg.norm(mjx_env.get_sensor_data(
                self.mj_model, data, f"{side}_foot_force"
            )) for side in ("left", "right")
        ])
        total_load = jp.sum(hand_force) + jp.sum(foot_force) + 1.0e-3
        leg_power, arm_power, waist_power = self._positive_mechanical_power(data)
        leg_fraction = leg_power / (leg_power + arm_power + 1.0e-6)
        hand_slip = self._cost_hand_slip(data, hand_contact)
        foot_slip = self._cost_feet_slip(data, foot_contact, state.info)
        saturation = jp.abs(data.actuator_force) / self._actuator_force_limits
        alignment = jp.dot(self.get_gravity(data, "torso"), self._ramp_normal)
        root_height = jp.dot(data.qpos[:3], self._ramp_normal)
        state.metrics["validation/progress_m"] = progress
        state.metrics["validation/uphill_speed_mps"] = jp.dot(
            self.get_global_linvel(data, "pelvis"), self._ramp_tangent
        )
        state.metrics["validation/hand_contact_left"] = hand_contact[0].astype(jp.float32)
        state.metrics["validation/hand_contact_right"] = hand_contact[1].astype(jp.float32)
        state.metrics["validation/four_contact"] = jp.all(
            jp.hstack([foot_contact, hand_contact])
        ).astype(jp.float32)
        state.metrics["validation/alternating_hand_contact"] = jp.logical_xor(
            hand_contact[0], hand_contact[1]
        ).astype(jp.float32)
        state.metrics["validation/double_hand_contact"] = jp.all(
            hand_contact
        ).astype(jp.float32)
        state.metrics["validation/hand_slip_mps"] = hand_slip
        state.metrics["validation/foot_slip_mps"] = foot_slip
        state.metrics["validation/hand_load_share"] = jp.sum(hand_force) / total_load
        state.metrics["validation/leg_positive_power_w"] = leg_power
        state.metrics["validation/arm_positive_power_w"] = arm_power
        state.metrics["validation/waist_positive_power_w"] = waist_power
        state.metrics["validation/leg_propulsion_fraction"] = leg_fraction
        state.metrics["validation/peak_hand_force_n"] = jp.max(hand_force)
        state.metrics["validation/peak_wrist_moment_nm"] = jp.max(
            jp.linalg.norm(self._hand_torques(data), axis=-1)
        )
        state.metrics["validation/peak_wrist_actuator_torque_nm"] = jp.max(
            jp.abs(data.actuator_force[self._wrist_actuator_ids])
        )
        state.metrics["validation/peak_leg_torque_nm"] = jp.max(
            jp.abs(data.actuator_force[self._leg_actuator_ids])
        )
        ankle_ids = self._leg_actuator_ids[jp.array([4, 5, 10, 11])]
        state.metrics["validation/peak_ankle_torque_nm"] = jp.max(
            jp.abs(data.actuator_force[ankle_ids])
        )
        state.metrics["validation/actuator_saturation_ratio"] = jp.max(saturation)
        state.metrics["validation/actuator_saturated"] = jp.any(
            saturation >= 0.95
        ).astype(jp.float32)
        state.metrics["validation/com_height_m"] = jp.dot(com, self._ramp_normal)
        state.metrics["validation/fall"] = (
            (alignment < 0.05) | (root_height < 0.20)
        ).astype(jp.float32)
        prohibited_contact = data.sensordata[
            self._mj_model.sensor_adr[self._right_foot_left_foot_found_sensor]
        ] > 0
        prohibited_contact |= data.sensordata[
            self._mj_model.sensor_adr[self._left_foot_right_shin_found_sensor]
        ] > 0
        prohibited_contact |= data.sensordata[
            self._mj_model.sensor_adr[self._right_foot_left_shin_found_sensor]
        ] > 0
        state.metrics["validation/prohibited_body_contact"] = (
            prohibited_contact.astype(jp.float32)
        )
        state.metrics["validation/nonfinite"] = (
            jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        ).astype(jp.float32)
        state.metrics["validation/success"] = (
            progress >= self._config.validation_success_distance_m
        ).astype(jp.float32)
        return state


def make_four_contact_env(
    slope_degrees: float = 30.0, *, noise_level: float = 1.0,
    impl: str = "jax",
) -> HimalayaG1FourContactEnv:
    cfg = default_four_contact_config()
    with cfg.unlocked():
        cfg.slope_degrees = validate_four_contact_slope(slope_degrees)
        cfg.noise_config.level = float(noise_level)
        cfg.impl = impl
    return HimalayaG1FourContactEnv(config=cfg)


assert G1_ACTION_SIZE == 29
