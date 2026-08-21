"""G1 23-DOF trekking on continuous rough terrain.

Forked from Isaac Lab's Isaac-Velocity-Rough-G1-v0. Inheriting that config
gets us NVIDIA's working G1 locomotion baseline; this file records only the
deliberate departures from it.

Three departures, each for a stated reason:

1. TERRAIN. Stock is 40% stairs with 5-23cm steps and only 20% continuous
   rough -- a mix calibrated for quadrupeds stepping ONTO discrete edges.
   Our task is continuous rough ground, so the proportions invert: 55%
   rough+slope, and step heights cap at 10cm rather than 23cm.

2. ARMS. Stock actively penalizes arm motion:
       joint_deviation_arms = RewTerm(joint_deviation_l1, weight=-0.1)
   which drives arms toward their default pose -- the exact opposite of
   what we want. We weaken it to a small posture anchor (-0.01), and add an
   angular-momentum term the arms can help satisfy.

3. DISTURBANCES. Stock sets push_robot=None: nothing ever knocks the robot,
   so nothing ever demands a balance recovery. We re-enable pushes -- without
   them there is no pressure for arms to do anything at all.

Everything else (PPO settings, [512,256,128] ELU net, feet_air_time,
feet_slide, termination penalty) is NVIDIA's and is left alone.

Two variants:
  HimalayaTeacherEnvCfg -- privileged. Sees height_scan and base_lin_vel.
  HimalayaStudentEnvCfg -- blind. Proprioception + IMU only, as on hardware.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .rough_env_cfg import (
    G1Rewards,
    G1RoughEnvCfg,
)

# --------------------------------------------------------------------------
# Terrain: continuous rough, not discrete footholds.
# --------------------------------------------------------------------------
# Proportions vs stock:
#   random_rough  0.20 -> 0.35     <- the primary surface for trekking
#   slopes        0.20 -> 0.30     <- mountainous grade
#   boxes         0.20 -> 0.15
#   stairs        0.40 -> 0.20     <- halved; was calibrated for quadrupeds
#
# Step height caps at 0.10 instead of 0.23. A 23cm step is near the G1's
# hip height and turns the task into discrete foothold planning -- a
# different problem (published work reports end-to-end RL collapsing to
# standing on sparse footholds).
TREKKING_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,      # difficulty levels; curriculum promotes along this axis
    num_cols=20,      # terrain variations per level
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.35, noise_range=(0.02, 0.08), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.03, 0.10), platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.10, step_height_range=(0.03, 0.10), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10, step_height_range=(0.03, 0.10), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
    },
)


@configclass
class HimalayaRewards(G1Rewards):
    """G1 rewards with the arm penalty relaxed and a balance term added."""

    # Angular-momentum regularization, from narrow-terrain humanoid work:
    #     exp(-||L_base||_2 / 5)
    # Approximated here by penalizing base angular velocity, which is the
    # observable proxy Isaac Lab exposes directly.
    #
    # WEIGHT IS 0.0 FOR THE BASELINE RUN. Run arms-free first, measure whether
    # the arm joints actually move, and only then raise this to ~0.3. Setting
    # it nonzero from the start means never learning whether arms emerge on
    # their own -- and never knowing if this term is doing real work or just
    # masking a reward-balance bug elsewhere.
    base_angular_momentum = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class HimalayaTeacherEnvCfg(G1RoughEnvCfg):
    """Privileged teacher. Sees terrain heightmap and true base velocity."""

    rewards: HimalayaRewards = HimalayaRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = TREKKING_TERRAINS_CFG

        # Arms: stock drives them to default pose at -0.1. Relaxed 10x to a
        # weak posture anchor -- enough to stop pure flailing, weak enough
        # that a balance-useful swing costs almost nothing.
        self.rewards.joint_deviation_arms.weight = -0.01

        # Disturbances. Stock disables these entirely; without them there is
        # no event that requires a balance recovery, so no pressure on the
        # arms to contribute anything.
        self.events.push_robot = mdp.EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(4.0, 8.0),
            params={"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}},
        )

        # Friction range. Mountainous ground is not uniform, and a policy
        # trained at one exact friction learns a gait that only works there.
        # On from the start, widened here for terrain.
        self.events.physics_material.params["static_friction_range"] = (0.4, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.4, 1.2)

        # Mild mass jitter -- cheap insurance against the policy memorizing
        # momentum timing for one exact inertia. Matters here because arm
        # balancing is precisely the behavior that overfits to exact mass.
        self.events.add_base_mass = mdp.EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "mass_distribution_params": (-1.0, 1.0),
                "operation": "add",
            },
        )

        # Forward-biased commands: trekking, not omnidirectional strafing.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.2)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class HimalayaStudentEnvCfg(HimalayaTeacherEnvCfg):
    """Blind student. Proprioception + IMU only -- what real hardware has.

    Drops height_scan (needs a terrain map) and base_lin_vel (not directly
    measurable on a real humanoid). The student infers terrain from how the
    body got disturbed over recent history.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.height_scan = None
        self.observations.policy.base_lin_vel = None


@configclass
class HimalayaTeacherEnvCfg_PLAY(HimalayaTeacherEnvCfg):
    """Small scene for visual inspection. Reward curves lie; watch the video."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


