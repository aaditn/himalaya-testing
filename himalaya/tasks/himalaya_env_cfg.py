"""MuJoCo-only G1 uphill task built on DeepMind's joystick environment.

This module subclasses the existing MuJoCo Playground G1 joystick task. It
does not replace the robot, observations, joint-position actions, PD control,
termination rules, gait rewards, or low-level stepping implementation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
import numpy as np

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick

from .g1_cfg import G1_ACTION_SIZE, validate_slope


def default_config() -> config_dict.ConfigDict:
    """Return the stock G1 joystick config plus Stage-I uphill settings."""

    cfg = g1_joystick.default_config()
    with cfg.unlocked():
        # A model is compiled for one planar grade. The training driver moves
        # checkpoints through 0, 5, 10, 15, and 20 degrees sequentially.
        cfg.slope_degrees = 15.0
        cfg.spawn_yaw_jitter_degrees = 3.0
        cfg.uphill_command_range = [0.40, 0.60]
        cfg.command_stand_probability = 0.05

        # Stage I is deterministic terrain learning. Roughness, ice, pushes,
        # and dynamics randomization are deliberately deferred.
        cfg.push_config.enable = False

        # Preserve every stock reward and its original value, then activate
        # the requested additions. Positive terms are rewards; negative terms
        # are costs in the inherited reward convention.
        scales = cfg.reward_config.scales
        scales.action_rate = -0.01
        scales.feet_clearance = -0.50
        scales.terrain_zmp = 1.00
        scales.com_height = 1.00

        cfg.reward_config.zmp_sigma = 0.10
        cfg.reward_config.zmp_epsilon = 1.0e-3
        cfg.reward_config.com_height_target = 0.66
        cfg.reward_config.com_height_sigma = 0.035
        cfg.reward_config.forward_lean_ratio = 0.25
        cfg.reward_config.max_forward_lean_degrees = 6.0

        # Validation is part of the task contract, not a reward.
        cfg.validation_success_distance_m = 6.0
    return cfg


class HimalayaG1UphillEnv(g1_joystick.Joystick):
    """Blind G1 ascent on one smooth planar ramp.

    The actor observation returned under ``state`` is byte-for-byte the same
    layout as DeepMind's G1 joystick actor observation. The five-element
    slope descriptor is appended only to ``privileged_state`` for the critic.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict | None = None,
        config_overrides: Optional[
            Dict[str, Union[str, int, float, bool, list[Any]]]
        ] = None,
    ) -> None:
        cfg = config or default_config()
        slope = validate_slope(cfg.slope_degrees)

        # Load the official flat-terrain G1 joystick model and initialize all
        # inherited observation/action/reward indices unchanged.
        mjx_env.ensure_menagerie_exists()
        super().__init__(
            task="flat_terrain",
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
            [
                math.cos(self._slope_radians / 2.0),
                0.0,
                -math.sin(self._slope_radians / 2.0),
                0.0,
            ]
        )

        # Turn the official infinite floor plane into an equally smooth ramp.
        # Re-upload only the changed model; robot and actuator definitions are
        # still exactly those created by the parent environment.
        floor_id = self._mj_model.geom("floor").id
        self._mj_model.geom_quat[floor_id] = np.asarray(self._ramp_quat)
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        self._body_masses = jp.asarray(self._mj_model.body_mass)
        self._total_mass = jp.sum(self._body_masses)
        self._knee_actuator_ids = jp.array(
            [
                self._mj_model.actuator("left_knee_joint").id,
                self._mj_model.actuator("right_knee_joint").id,
            ]
        )

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
        # Let the parent sample joint state, gait phase, command, and all
        # observation bookkeeping first.
        state = super().reset(rng)

        # Reinterpret the parent's planar x/y sample as ramp coordinates and
        # place the root one nominal base-height along the plane normal.
        qpos = state.data.qpos
        u, v = qpos[0], qpos[1]
        nominal_height = self._init_q[2]
        plane_point = u * self._ramp_tangent + v * self._ramp_cross
        root_pos = plane_point + nominal_height * self._ramp_normal
        qpos = qpos.at[:3].set(root_pos)

        # Face uphill with a small heading perturbation; arbitrary world yaw
        # would turn a forward command into a cross-slope command.
        state.info["rng"], yaw_rng = jax.random.split(state.info["rng"])
        yaw_limit = math.radians(self._config.spawn_yaw_jitter_degrees)
        yaw = jax.random.uniform(
            yaw_rng, (), minval=-yaw_limit, maxval=yaw_limit
        )
        yaw_quat = mjx_math.axis_angle_to_quat(
            jp.array([0.0, 0.0, 1.0]), yaw
        )
        root_quat = mjx_math.quat_mul(self._ramp_quat, yaw_quat)
        qpos = qpos.at[3:7].set(root_quat)

        data = state.data.replace(qpos=qpos)
        data = mjx.forward(self.mjx_model, data)
        contact = self._foot_contacts(data)

        com = self._com_position(data)
        state.info["last_com_pos"] = com
        state.info["last_com_vel"] = jp.zeros(3)
        state.info["start_progress"] = jp.dot(com, self._ramp_tangent)

        obs = self._get_obs(data, state.info, contact)
        state = state.replace(data=data, obs=obs)
        return self._set_diagnostics(state, contact)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # Parent owns motor targets, PD stepping, contact filtering, gait phase,
        # termination, and reward aggregation.
        next_state = super().step(state, action)

        com = self._com_position(next_state.data)
        com_vel = (com - next_state.info["last_com_pos"]) / self.dt
        next_state.info["last_com_pos"] = com
        next_state.info["last_com_vel"] = com_vel
        return self._set_diagnostics(
            next_state, self._foot_contacts(next_state.data)
        )

    def _foot_contacts(self, data: mjx.Data) -> jax.Array:
        return jp.array(
            [
                data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
                for sensor_id in self._feet_floor_found_sensor
            ]
        )

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:
        obs = super()._get_obs(data, info, contact)
        slope = self._slope_radians
        command_uphill = info["command"][0] >= 0.0
        descriptor = jp.array(
            [
                slope,
                0.0,  # uniform ramp has no bank angle
                abs(slope),
                command_uphill.astype(jp.float32),
                (~command_uphill).astype(jp.float32),
            ]
        )
        # Crucial deployment contract: do not modify obs["state"].
        obs["privileged_state"] = jp.hstack(
            [obs["privileged_state"], descriptor]
        )
        return obs

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
        rewards = super()._get_reward(
            data, action, info, metrics, done, first_contact, contact
        )

        global_linvel = self.get_global_linvel(data, "pelvis")
        ramp_velocity = jp.array(
            [
                jp.dot(global_linvel, self._ramp_tangent),
                jp.dot(global_linvel, self._ramp_cross),
            ]
        )
        rewards["tracking_lin_vel"] = self._reward_tracking_lin_vel(
            info["command"], ramp_velocity
        )

        global_angvel = self.get_global_angvel(data, "pelvis")
        ramp_yaw_rate = jp.dot(global_angvel, self._ramp_normal)
        yaw_error = jp.square(info["command"][2] - ramp_yaw_rate)
        rewards["tracking_ang_vel"] = jp.exp(
            -yaw_error / self._config.reward_config.tracking_sigma
        )

        pelvis_normal_vel = jp.dot(
            self.get_global_linvel(data, "pelvis"), self._ramp_normal
        )
        torso_normal_vel = jp.dot(
            self.get_global_linvel(data, "torso"), self._ramp_normal
        )
        rewards["lin_vel_z"] = jp.square(pelvis_normal_vel) + jp.square(
            torso_normal_vel
        )

        rewards["terrain_zmp"] = self._reward_terrain_aligned_zmp(data, info)
        rewards["com_height"] = self._reward_com_height(data)
        return rewards

    def _reward_terrain_aligned_zmp(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> jax.Array:
        """HumoSlope Stage-I point-mass ZMP surrogate on the ramp plane."""

        foot_pos = data.site_xpos[self._feet_site_id]
        foot_forces = jp.array(
            [
                jp.linalg.norm(
                    mjx_env.get_sensor_data(
                        self.mj_model, data, "left_foot_force"
                    )
                ),
                jp.linalg.norm(
                    mjx_env.get_sensor_data(
                        self.mj_model, data, "right_foot_force"
                    )
                ),
            ]
        )
        eps = self._config.reward_config.zmp_epsilon
        weights = foot_forces + eps
        support_anchor = jp.sum(
            foot_pos * weights[:, None], axis=0
        ) / jp.sum(weights)

        com = self._com_position(data)
        com_vel = (com - info["last_com_pos"]) / self.dt
        com_acc = (com_vel - info["last_com_vel"]) / self.dt
        apparent_force = jp.array([0.0, 0.0, -9.81]) - com_acc

        denominator = jp.dot(apparent_force, self._ramp_normal)
        safe_denominator = jp.where(
            jp.abs(denominator) < eps,
            jp.where(denominator < 0.0, -eps, eps),
            denominator,
        )
        t = jp.dot(
            support_anchor - com, self._ramp_normal
        ) / safe_denominator
        terrain_zmp = com + t * apparent_force
        deviation = jp.linalg.norm(terrain_zmp - support_anchor)
        return jp.exp(-deviation / self._config.reward_config.zmp_sigma)

    def _reward_com_height(self, data: mjx.Data) -> jax.Array:
        height = jp.dot(self._com_position(data), self._ramp_normal)
        error = height - self._config.reward_config.com_height_target
        sigma = self._config.reward_config.com_height_sigma
        return jp.exp(-jp.square(error) / jp.square(sigma))

    def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        # Plane normal + the ramp angle returns world-up. Add a bounded,
        # slope-proportional uphill lean beyond that neutral pose.
        lean = min(
            self._config.reward_config.forward_lean_ratio
            * self._slope_radians,
            math.radians(
                self._config.reward_config.max_forward_lean_degrees
            ),
        )
        target = jp.array([math.sin(lean), 0.0, math.cos(lean)])
        return jp.sum(jp.square(torso_zaxis - target))

    def _cost_feet_slip(
        self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
    ) -> jax.Array:
        del info
        foot_velocity = data.sensordata[self._foot_linvel_sensor_adr]
        normal_velocity = foot_velocity @ self._ramp_normal
        tangent_velocity = (
            foot_velocity - normal_velocity[:, None] * self._ramp_normal
        )
        # Actual planted-foot velocity, not the parent's pelvis-speed proxy.
        # Swing feet contribute exactly zero.
        return jp.sum(jp.linalg.norm(tangent_velocity, axis=-1) * contact)

    def _cost_feet_clearance(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> jax.Array:
        del info
        foot_velocity = data.sensordata[self._foot_linvel_sensor_adr]
        normal_velocity = foot_velocity @ self._ramp_normal
        tangent_velocity = (
            foot_velocity - normal_velocity[:, None] * self._ramp_normal
        )
        tangent_speed_weight = jp.sqrt(
            jp.linalg.norm(tangent_velocity, axis=-1)
        )
        clearance = data.site_xpos[self._feet_site_id] @ self._ramp_normal
        error = jp.abs(
            clearance - self._config.reward_config.max_foot_height
        )
        return jp.sum(error * tangent_speed_weight)

    def _reward_feet_phase(
        self,
        data: mjx.Data,
        phase: jax.Array,
        foot_height: jax.Array,
        command: jax.Array,
    ) -> jax.Array:
        clearance = data.site_xpos[self._feet_site_id] @ self._ramp_normal
        target = gait.get_rz(phase, swing_height=foot_height)
        reward = jp.exp(-jp.sum(jp.square(clearance - target)) / 0.01)
        body_speed = jp.linalg.norm(
            self.get_global_linvel(data, "pelvis")
        )
        body_turn = jp.linalg.norm(
            self.get_global_angvel(data, "pelvis")
        )
        moving = jp.logical_or(body_speed > 0.1, body_turn > 0.1)
        return reward * jp.logical_or(
            moving, jp.linalg.norm(command) > 0.01
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
            jp.zeros(3),
            command,
        )

    def _com_position(self, data: mjx.Data) -> jax.Array:
        return jp.sum(
            data.xipos * self._body_masses[:, None], axis=0
        ) / self._total_mass

    def _planted_slip_speed(
        self, data: mjx.Data, contact: jax.Array
    ) -> jax.Array:
        foot_velocity = data.sensordata[self._foot_linvel_sensor_adr]
        normal_velocity = foot_velocity @ self._ramp_normal
        tangent_velocity = (
            foot_velocity - normal_velocity[:, None] * self._ramp_normal
        )
        speeds = jp.linalg.norm(tangent_velocity, axis=-1)
        return jp.sum(speeds * contact) / jp.maximum(jp.sum(contact), 1)

    def _set_diagnostics(
        self, state: mjx_env.State, contact: jax.Array
    ) -> mjx_env.State:
        com = self._com_position(state.data)
        progress = (
            jp.dot(com, self._ramp_tangent)
            - state.info["start_progress"]
        )
        uphill_speed = jp.dot(
            self.get_global_linvel(state.data, "pelvis"),
            self._ramp_tangent,
        )
        com_height = jp.dot(com, self._ramp_normal)
        peak_knee_torque = jp.max(
            jp.abs(state.data.actuator_force[self._knee_actuator_ids])
        )
        fell = self.get_gravity(state.data, "torso")[-1] < 0.0

        state.metrics["validation/progress_m"] = progress
        state.metrics["validation/uphill_speed_mps"] = uphill_speed
        state.metrics[
            "validation/planted_slip_mps"
        ] = self._planted_slip_speed(state.data, contact)
        state.metrics["validation/com_height_m"] = com_height
        state.metrics[
            "validation/peak_knee_torque_nm"
        ] = peak_knee_torque
        state.metrics["validation/fall"] = fell.astype(jp.float32)
        state.metrics["validation/success"] = (
            progress >= self._config.validation_success_distance_m
        ).astype(jp.float32)
        return state


def make_env(
    slope_degrees: float = 15.0,
    *,
    noise_level: float = 1.0,
    impl: str = "jax",
) -> HimalayaG1UphillEnv:
    """Construct one Himalaya environment without using a global registry."""

    cfg = default_config()
    with cfg.unlocked():
        cfg.slope_degrees = validate_slope(slope_degrees)
        cfg.noise_config.level = float(noise_level)
        cfg.impl = impl
    return HimalayaG1UphillEnv(config=cfg)


assert G1_ACTION_SIZE == 29
