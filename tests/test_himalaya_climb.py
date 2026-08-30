"""Integration checks for the platform-native four-limb climb task."""

import json
import unittest
from pathlib import Path

import jax
import mujoco
import numpy as np

from himalaya.env import Joystick, default_config


class HimalayaClimbTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = default_config()
        config.impl = "jax"
        cls.env = Joystick(task="climb_terrain", config=config)

    def test_climb_model_contract(self):
        model = self.env.mj_model
        self.assertEqual(model.nu, 29)
        self.assertEqual(model.nsensor, 35)
        self.assertEqual(model.npair, 47)
        self.assertEqual(self.env.observation_size["state"], (114,))
        self.assertEqual(self.env.observation_size["privileged_state"], (230,))
        self.assertGreater(
            self.env._config.reward_config.scales.mountain_progress, 0
        )
        self.assertGreater(
            self.env._config.reward_config.scales.foot_uphill_drive, 0
        )
        movement_scales = sum(
            self.env._config.reward_config.scales[name]
            for name in (
                "uphill_progress",
                "assisted_uphill_progress",
                "mountain_progress",
                "new_high_progress",
                "waypoint_bonus",
                "large_foot_step",
                "foot_uphill_drive",
            )
        )
        support_scales = sum(
            self.env._config.reward_config.scales[name]
            for name in (
                "continuous_hand_support",
                "hand_contact_schedule",
                "hand_phase",
                "hand_lift_height",
                "hand_lift_target",
                "diagonal_swing_sync",
                "diagonal_support",
                "hand_load_share",
                "foot_swing_clearance",
                "support_exchange",
            )
        )
        self.assertGreater(movement_scales, 4 * support_scales)
        self.assertEqual(
            self.env._config.reward_config.scales.mountain_progress, 10.0
        )
        self.assertEqual(
            self.env._config.reward_config.scales.uphill_progress, 8.0
        )
        self.assertEqual(
            self.env._config.reward_config.scales.assisted_uphill_progress, 1.5
        )
        self.assertEqual(
            self.env._config.reward_config.scales.failed_ascent, -3.0
        )
        self.assertEqual(
            self.env._config.reward_config.scales.termination, -25.0
        )
        self.assertEqual(self.env._config.climb.waypoint_interval, 0.25)
        self.assertEqual(self.env._config.climb.max_regression_distance, 0.35)
        self.assertEqual(self.env._config.climb.stall_seconds, 3.0)
        self.assertEqual(self.env._config.climb.min_limb_air_time, 0.08)
        self.assertEqual(self.env._config.climb.target_stride_length, 0.20)
        self.assertEqual(
            self.env._config.reward_config.max_hand_height, 0.24384
        )
        self.assertEqual(
            list(self.env._config.climb.gait_frequency_range), [0.55, 0.75]
        )
        self.assertEqual(
            self.env._config.climb.hand_contact_duty_factor, 0.60
        )
        self.assertEqual(
            self.env._config.reward_config.scales.feet_air_time, 0.4
        )
        self.assertEqual(
            self.env._config.reward_config.scales.feet_phase, 0.0
        )
        self.assertEqual(
            self.env._config.reward_config.scales.hand_phase, 0.0
        )
        self.assertGreater(
            self.env._config.reward_config.scales.hand_lift_height, 0.0
        )
        self.assertGreater(
            self.env._config.reward_config.scales.foot_swing_clearance, 0.0
        )
        self.assertLess(
            self.env._config.reward_config.scales.swing_contact, 0.0
        )
        self.assertGreater(
            self.env._config.reward_config.scales.support_exchange, 0.0
        )
        self.assertLess(
            self.env._config.reward_config.scales.overspeed, 0.0
        )
        self.assertGreater(
            self.env._config.reward_config.scales.knee_clearance, 0.0
        )
        self.assertLess(
            self.env._config.reward_config.scales.knee_contact, 0.0
        )
        self.assertEqual(self.env._config.climb.target_knee_clearance, 0.05)
        self.assertLess(
            self.env._config.climb.fall_pelvis_clearance,
            self.env._config.climb.min_pelvis_clearance,
        )
        self.assertEqual(
            self.env._config.reward_config.scales.limb_touchdown_advance,
            1.0,
        )
        self.assertEqual(
            self.env._config.reward_config.scales.large_foot_step, 1.0
        )
        self.assertLess(
            self.env._config.reward_config.scales.failed_ascent, 0.0
        )
        self.assertEqual(
            self.env._config.reward_config.scales.joint_deviation_hip,
            0.0,
        )
        self.assertEqual(
            self.env._config.reward_config.scales.joint_deviation_knee, 0.0
        )
        self.assertEqual(self.env._config.reward_config.scales.pose, 0.0)
        for name in (
            "left_foot_floor",
            "right_foot_floor",
            "left_foot_boulder_00",
            "right_foot_boulder_00",
        ):
            pair_id = model.pair(name).id
            self.assertEqual(model.pair_dim[pair_id], 6)
            self.assertAlmostEqual(model.pair_friction[pair_id, 0], 1.90)
            self.assertAlmostEqual(model.pair_friction[pair_id, 1], 1.90)
        for name in (
            "left_hand_floor",
            "right_hand_floor",
            "left_hand_boulder_00",
            "right_hand_boulder_00",
        ):
            pair_id = model.pair(name).id
            self.assertEqual(model.pair_dim[pair_id], 6)
            self.assertAlmostEqual(model.pair_friction[pair_id, 0], 0.95)
            self.assertAlmostEqual(model.pair_friction[pair_id, 1], 0.95)
        self.assertAlmostEqual(
            self.env._config.climb.foot_spike_friction,
            2.0 * self.env._config.climb.spike_friction,
        )
        boulders = [
            model.geom(f"boulder_{index:02d}").id for index in range(10)
        ]
        for geom_id in boulders:
            self.assertAlmostEqual(model.geom_size[geom_id, 0], 0.127)
        self.assertAlmostEqual(model.hfield_size[0, 2], 0.060)
        for side in ("left", "right"):
            geom = model.geom(f"{side}_hand_collision")
            site = model.site(f"{side}_palm")
            self.assertEqual(geom.type, mujoco.mjtGeom.mjGEOM_SPHERE)
            self.assertAlmostEqual(geom.size[0], 0.05)
            np.testing.assert_allclose(geom.pos, site.pos, atol=1e-7)
            for axis in ("roll", "pitch", "yaw"):
                joint = model.joint(f"{side}_wrist_{axis}_joint")
                actuator = model.actuator(f"{side}_wrist_{axis}_joint")
                np.testing.assert_allclose(
                    model.jnt_range[joint.id], [-0.08, 0.08]
                )
                self.assertAlmostEqual(
                    model.actuator_gainprm[actuator.id, 0], 20.0
                )
                np.testing.assert_allclose(
                    model.jnt_actfrcrange[joint.id], [-25.0, 25.0]
                )
                self.assertAlmostEqual(
                    float(
                        self.env._default_pose[int(joint.qposadr[0]) - 7]
                    ),
                    0.0,
                )

    def test_progress_state_starts_without_credit(self):
        state = self.env.reset(jax.random.PRNGKey(7))
        self.assertAlmostEqual(
            float(state.info["start_uphill_position"]),
            float(state.info["max_uphill_position"]),
        )
        self.assertAlmostEqual(
            float(state.info["start_uphill_position"]),
            float(state.info["progress_checkpoint_position"]),
        )
        self.assertEqual(int(state.info["last_waypoint"]), 0)
        self.assertEqual(int(state.info["steps_without_progress"]), 0)
        self.assertEqual(
            np.asarray(state.info["last_hand_plant_normal"]).shape, (2,)
        )
        self.assertFalse(np.asarray(state.info["hand_lift_achieved"]).any())
        pelvis_clearance = (
            state.data.qpos[:3] @ self.env._slope_normal
            - self.env._terrain_plane_offset
        )
        self.assertGreater(
            float(pelvis_clearance),
            self.env._config.climb.min_pelvis_clearance,
        )
        phase_reward = self.env._reward_feet_phase(
            state.data,
            state.info["phase"],
            self.env._config.reward_config.max_foot_height,
            state.info["command"],
        )
        # At reset one foot is scheduled to swing. Terrain-relative clearance
        # keeps that target reachable despite the inclined floor's 0.20 m offset.
        self.assertGreater(float(phase_reward), 0.05)
        opposite_foot_phase = np.asarray(state.info["phase"])[::-1]
        duty = self.env._config.climb.hand_contact_duty_factor
        desired_hand_contact = np.cos(opposite_foot_phase) < np.cos(
            np.pi * (1.0 - duty)
        )
        # Reset begins at the exchange boundary with all four limbs planted.
        self.assertEqual(desired_hand_contact.tolist(), [True, True])
        reference = np.asarray(
            self.env._crawl_reference(np.array([0.0, np.pi]))
        )
        # Peak diagonal swing: left leg/right arm reach forward while their
        # planted opposites retract to push the body uphill.
        self.assertAlmostEqual(reference[0], -0.30, places=5)
        self.assertAlmostEqual(reference[3], 0.38, places=5)
        self.assertAlmostEqual(reference[6], 0.30, places=5)
        self.assertAlmostEqual(reference[15], 0.0, places=5)
        self.assertAlmostEqual(reference[22], -0.30, places=5)
        self.assertAlmostEqual(reference[23], -0.45, places=5)
        self.assertAlmostEqual(reference[25], -0.22, places=5)
        self.assertAlmostEqual(reference[9], -0.16, places=5)
        self.assertAlmostEqual(reference[18], 0.14, places=5)
        target_speed = self.env._config.climb.target_uphill_speed
        supported = np.array(True)
        self.assertAlmostEqual(
            float(self.env._reward_uphill_velocity(
                np.array(0.0), np.array(1.0), supported
            )),
            0.0,
        )
        self.assertAlmostEqual(
            float(self.env._cost_uphill_overspeed(target_speed)), 0.0
        )
        self.assertAlmostEqual(
            float(self.env._cost_uphill_overspeed(2.0 * target_speed)),
            0.75**2,
        )
        self.assertAlmostEqual(
            float(self.env._reward_uphill_velocity(
                np.array(0.5 * target_speed), np.array(1.0), supported
            )),
            0.5,
        )
        self.assertAlmostEqual(
            float(self.env._reward_uphill_velocity(
                np.array(2.0 * target_speed), np.array(1.0), supported
            )),
            1.0,
        )
        self.assertAlmostEqual(
            float(self.env._reward_uphill_velocity(
                np.array(target_speed), np.array(0.0), supported
            )),
            0.0,
        )

    def test_curriculum_bootstraps_before_rocks(self):
        stages = json.loads(
            Path("configs/curriculum.json").read_text(encoding="utf-8")
        )["stages"]
        self.assertEqual(stages[0]["slope_degrees"], 5)
        self.assertLessEqual(stages[0]["roughness_m"], 0.005)
        self.assertFalse(stages[0]["boulders_enabled"])
        self.assertFalse(stages[0]["domain_randomization"])
        self.assertEqual(stages[1]["slope_degrees"], 5)
        self.assertTrue(stages[1]["domain_randomization"])
        first_rocky = next(i for i, stage in enumerate(stages)
                           if stage["boulders_enabled"])
        self.assertGreaterEqual(first_rocky, 2)
        self.assertEqual(stages[-1]["slope_degrees"], 42)
        self.assertGreater(stages[-1]["roughness_m"], stages[-2]["roughness_m"])
        self.assertTrue(stages[-1]["boulders_enabled"])

    def test_terminal_grade_preserves_four_point_reset(self):
        config = default_config()
        config.impl = "jax"
        config.climb.slope_degrees = 42
        config.climb.roughness_m = 0.150
        config.climb.boulders_enabled = True
        env = Joystick(task="climb_terrain", config=config)
        model = env.mj_model
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(
            model, data, model.keyframe("knees_bent").id
        )
        mujoco.mj_forward(model, data)
        contacts = {
            frozenset((model.geom(item.geom[0]).name,
                       model.geom(item.geom[1]).name))
            for item in data.contact
        }
        for name in (
            "left_foot",
            "right_foot",
            "left_hand_collision",
            "right_hand_collision",
        ):
            self.assertIn(frozenset(("floor", name)), contacts)
        torso_up_adr = model.sensor_adr[model.sensor("upvector_torso").id]
        torso_up = data.sensordata[torso_up_adr : torso_up_adr + 3]
        self.assertGreater(torso_up @ env._slope_tangent, 0.9)

    def test_crawl_reset_reaches_hand_contact(self):
        model = self.env.mj_model
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(
            model, data, model.keyframe("knees_bent").id
        )
        mujoco.mj_forward(model, data)
        contacts = {
            frozenset((model.geom(item.geom[0]).name, model.geom(item.geom[1]).name))
            for item in data.contact
        }
        expected = {
            frozenset(("floor", "left_foot")),
            frozenset(("floor", "right_foot")),
            frozenset(("floor", "left_hand_collision")),
            frozenset(("floor", "right_hand_collision")),
        }
        self.assertEqual(contacts, expected)
        self.assertGreater(data.qpos[2] - 0.20, 0.45)
        torso_uphill = (
            data.xpos[model.body("torso_link").id]
            @ np.asarray(self.env._slope_tangent)
        )
        self.assertGreater(torso_uphill, 0.29)
        knee_pos = np.stack([
            data.xpos[model.body(f"{side}_knee_link").id]
            for side in ("left", "right")
        ])
        knee_clearance = (
            knee_pos @ np.asarray(self.env._slope_normal)
            - self.env._terrain_plane_offset
            - self.env._config.climb.roughness_m
            - self.env._config.climb.knee_safety_radius
        )
        self.assertTrue(np.all(knee_clearance > 0.0))
        torso_up_adr = model.sensor_adr[model.sensor("upvector_torso").id]
        torso_up = data.sensordata[torso_up_adr : torso_up_adr + 3]
        self.assertGreater(torso_up @ self.env._slope_tangent, 0.9)
        addresses = [
            model.sensor_adr[model.sensor(name).id]
            for name in (
                "left_hand_floor_found",
                "right_hand_floor_found",
            )
        ]
        contact_steps = 0
        for _ in range(250):
            mujoco.mj_step(model, data)
            contact_steps += int(any(data.sensordata[adr] > 0 for adr in addresses))
        # The low-iteration training scene is not intended to hold a pose
        # open-loop; require palm support through most of the first 0.5 s.
        self.assertGreater(contact_steps, 125)


if __name__ == "__main__":
    unittest.main()
