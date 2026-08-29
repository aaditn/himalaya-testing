"""G1 locomotion on ice.

Hackathon Track 1: "Have the robot walk on ice, with as much movement as
possible." Built on MuJoCo Playground's G1JoystickFlatTerrain, which the
hackathon resources recommend, using Menagerie's unitree_g1 model.

The only physical change is friction. Playground's stock domain randomization
samples floor/foot friction from U(0.4, 1.0) -- ordinary ground. Ice is an
order of magnitude lower:

    packed snow      ~0.35
    wet ice          ~0.15
    smooth cold ice  ~0.05     <- what we train on

That single number is what makes the task hard, and it is honest physics
rather than a contrived penalty: the feet simply cannot generate the lateral
force a normal gait relies on. The policy has to find a different gait --
shorter strides, flatter foot placement, and (the interesting part) upper-body
momentum to steer and arrest slips, because the feet no longer can.

Two curricula are provided:
  ICE_ONLY      -- always slippery. Fastest to a demoable ice policy.
  MIXED_TERRAIN -- friction spans ice to dry ground, so one policy handles a
                   patchy surface. Harder, but closer to a real mountain.
"""

import functools

import jax
from mujoco import mjx
from mujoco_playground._src.locomotion.g1 import randomize as g1_randomize

# Friction ranges. Stock Playground is (0.4, 1.0).
ICE_ONLY = (0.05, 0.15)
MIXED_TERRAIN = (0.05, 0.9)

FLOOR_GEOM_ID = g1_randomize.FLOOR_GEOM_ID
TORSO_BODY_ID = g1_randomize.TORSO_BODY_ID


def make_ice_randomizer(friction_range=ICE_ONLY, n_joints=29):
    """Playground's G1 randomizer with the friction range swapped for ice.

    Everything else -- mass jitter, armature, joint friction loss, qpos0
    jitter -- is left exactly as Playground ships it. Those are tuned and
    not what this project is testing.
    """
    lo, hi = friction_range

    def domain_randomize(model: mjx.Model, rng: jax.Array):
        @jax.vmap
        def rand_dynamics(rng):
            # THE ice parameter. Everything below is stock Playground.
            rng, key = jax.random.split(rng)
            friction = jax.random.uniform(key, minval=lo, maxval=hi)
            pair_friction = model.pair_friction.at[0:2, 0:2].set(friction)

            rng, key = jax.random.split(rng)
            frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
                key, shape=(n_joints,), minval=0.5, maxval=2.0
            )
            dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

            rng, key = jax.random.split(rng)
            armature = model.dof_armature[6:] * jax.random.uniform(
                key, shape=(n_joints,), minval=1.0, maxval=1.05
            )
            dof_armature = model.dof_armature.at[6:].set(armature)

            rng, key = jax.random.split(rng)
            dmass = jax.random.uniform(
                key, shape=(model.nbody,), minval=0.9, maxval=1.1
            )
            body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

            rng, key = jax.random.split(rng)
            dmass = jax.random.uniform(key, minval=-1.0, maxval=1.0)
            body_mass = body_mass.at[TORSO_BODY_ID].set(
                body_mass[TORSO_BODY_ID] + dmass
            )

            rng, key = jax.random.split(rng)
            qpos0 = model.qpos0
            qpos0 = qpos0.at[7:].set(
                qpos0[7:]
                + jax.random.uniform(key, shape=(n_joints,), minval=-0.05, maxval=0.05)
            )

            return pair_friction, dof_frictionloss, dof_armature, body_mass, qpos0

        (pair_friction, dof_frictionloss, dof_armature, body_mass, qpos0) = (
            rand_dynamics(rng)
        )

        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({
            "pair_friction": 0,
            "dof_frictionloss": 0,
            "dof_armature": 0,
            "body_mass": 0,
            "qpos0": 0,
        })

        model = model.tree_replace({
            "pair_friction": pair_friction,
            "dof_frictionloss": dof_frictionloss,
            "dof_armature": dof_armature,
            "body_mass": body_mass,
            "qpos0": qpos0,
        })
        return model, in_axes

    return domain_randomize


ice_randomize = make_ice_randomizer(ICE_ONLY)
mixed_randomize = make_ice_randomizer(MIXED_TERRAIN)


# Playground ships njmax=90 (max simultaneous constraints). That is too small
# for the G1 here: training logged 13,418 "nefc overflow - please increase
# njmax to 93" warnings in the ice run and 2,580 in the baseline. When the
# solver runs out of constraint slots it DROPS contacts, which is very likely
# the real source of the floor penetration -- a dropped foot contact means
# nothing holds the robot up that step.
#
# Ice overflows ~5x more often than dry ground: low friction means more
# sliding contacts, so more constraints. The condition we most wanted to
# measure was the one whose physics was most broken.
NJMAX = 160
NACONMAX = 131072


def ice_config(base_config, arm_swing: bool = True):
    """Adapt the stock G1 reward config for ice.

    Two changes, both aimed at what ice actually breaks:

    feet_slip (-0.25 -> -0.5)
        On ice a normal gait slips constantly. Doubling this pushes the
        policy toward placing feet down with less lateral velocity, which
        is what a person does on ice.

    ang_vel_xy (-0.15 -> -0.25)
        Penalizes body roll/pitch rate. A slip becomes a fall when the
        trunk starts rotating; damping that is the difference between a
        stumble and going down.

    arm_swing=True also relaxes `pose` (-0.1 -> -0.03). That term pulls all
    joints toward their default pose, arms included. On ice the arms are the
    only actuators left that can shed angular momentum, so pinning them to a
    rest pose removes the robot's last balance mechanism. Loosening it lets
    arm motion emerge if it is useful -- without scripting what that motion
    should look like.
    """
    cfg = base_config.copy_and_resolve_references()
    cfg.njmax = NJMAX
    cfg.naconmax = NACONMAX
    cfg.reward_config.scales.feet_slip = -0.5
    cfg.reward_config.scales.ang_vel_xy = -0.25
    if arm_swing:
        cfg.reward_config.scales.pose = -0.03
    return cfg


# ---------------------------------------------------------------------------
# Termination fix.
# ---------------------------------------------------------------------------
# Playground's stock G1 terminates only when the torso passes horizontal:
#
#     fall_termination = self.get_gravity(data, "torso")[-1] < 0.0
#
# A robot that tips to 89 degrees, or lands on its back and settles, never
# trips that. It lies there collecting reward, and because MJX's low-iteration
# contact solver cannot fully resolve a prone 34 kg body, it sinks partway
# through the floor -- which is exactly the "upside down under the surface,
# standing up backwards" behaviour we observed on video.
#
# The policy is not walking. It is finding a stable pose that stays just shy
# of the termination threshold. Every episode-length number from a run with
# the stock termination is therefore meaningless.
#
# Two additions, both cheap:
#   height  -- a pelvis below MIN_TORSO_HEIGHT is down, whatever its angle
#   tilt    -- fire at ~60 degrees rather than waiting for 90

MIN_TORSO_HEIGHT = 0.4   # metres; nominal standing pelvis is ~0.79
MAX_TILT = 0.5           # gravity-z; 1.0 = upright, 0.0 = horizontal


def patch_termination(env, min_height=MIN_TORSO_HEIGHT, max_tilt=MAX_TILT):
    """Wrap an env's _get_termination with height + stricter tilt checks.

    Applied to the env instance rather than forked from Playground so we stay
    on their upstream code for everything else.
    """
    import jax
    original = env._get_termination

    def _get_termination(data):
        done = original(data)
        # Torso pitched past ~60 degrees: falling, not merely leaning.
        done |= env.get_gravity(data, "torso")[-1] < max_tilt
        # Pelvis on the ground: down regardless of orientation. This is the
        # case the stock check misses entirely.
        done |= data.qpos[2] < min_height
        return done

    env._get_termination = _get_termination
    return env


# ---------------------------------------------------------------------------
# Friction curriculum.
# ---------------------------------------------------------------------------
# Training directly on ice does not work. With termination fixed and the
# constraint overflow gone, ice (mu = 0.05-0.15) still converges to an episode
# length of ~8 steps while identical code on dry ground (mu = 0.4-1.0) reaches
# 50+ and climbs. The policy never finds a gradient toward walking because
# almost every action ends in a fall, so it optimises the only thing it can:
# end the episode quickly.
#
# The fix is to not start there. Begin on ground the robot can actually walk
# on, then lower friction as it gets competent. Each stage is a small step
# from the last, so the policy always has a usable gradient -- and by the end
# it is walking on real ice, having learned its way down rather than being
# dropped there.
#
# This is the standard answer to "the task is too hard to learn from scratch",
# and it is also the more interesting result for the hackathon: a policy that
# adapts down to mu = 0.1 is a stronger claim than one trained at mu = 0.3.

FRICTION_STAGES = (
    (0.60, 1.00),   # dry ground -- learn to walk at all
    (0.40, 0.70),   # damp rock
    (0.25, 0.50),   # wet rock / packed snow
    (0.15, 0.35),   # snow over ice
    (0.08, 0.25),   # wet ice
    (0.05, 0.15),   # smooth cold ice -- the target
)


def curriculum_randomizer(stage: int, n_joints: int = 29):
    """Randomizer for one curriculum stage (index into FRICTION_STAGES)."""
    stage = max(0, min(stage, len(FRICTION_STAGES) - 1))
    return make_ice_randomizer(FRICTION_STAGES[stage], n_joints)


def stage_for_progress(frac: float) -> int:
    """Map training progress in [0,1] to a curriculum stage.

    Front-loaded: the early stages are where the gait is actually learned, so
    they get more of the budget. The last stages are adaptation, which needs
    less.
    """
    if frac < 0.20:
        return 0
    if frac < 0.35:
        return 1
    if frac < 0.50:
        return 2
    if frac < 0.65:
        return 3
    if frac < 0.80:
        return 4
    return 5


# ---------------------------------------------------------------------------
# Termination as a subclass, not a monkeypatch.
# ---------------------------------------------------------------------------
# patch_termination() reassigns env._get_termination at runtime. That passes
# every isolated rollout test, and the patched function is present on all
# wrapper layers -- but a monkeypatch depends on Python attribute lookup
# happening after the patch, and brax traces the step function through jax.jit
# inside the training loop. If tracing captures the original bound method,
# training silently uses the stock termination while every direct test says
# the patch is live.
#
# Subclassing removes the question: the override is part of the class before
# anything is instantiated or traced.

def make_ice_env_class():
    """Build a Joystick subclass with strict termination. Imported lazily so
    this module stays importable without a GPU."""
    from mujoco_playground._src.locomotion.g1.joystick import Joystick

    class IceJoystick(Joystick):
        """G1 joystick task that ends the episode when the robot is actually down.

        Stock terminates only at gravity_z < 0.0, i.e. past 90 degrees. A robot
        that falls to 85 degrees, or lands on its back, keeps collecting reward
        -- and with MJX's low-iteration contact solver a prone 34 kg body sinks
        partway through the floor. The result is a policy that learns stable
        fallen poses instead of walking.
        """

        MIN_TORSO_HEIGHT = MIN_TORSO_HEIGHT
        MAX_TILT = MAX_TILT

        def _get_termination(self, data):
            done = super()._get_termination(data)
            # ~60 degrees rather than 90: no loitering at the boundary.
            done = done | (self.get_gravity(data, "torso")[-1] < self.MAX_TILT)
            # Pelvis on the ground is down, whatever the orientation. This is
            # the case the stock check misses completely.
            done = done | (data.qpos[2] < self.MIN_TORSO_HEIGHT)
            return done

    return IceJoystick


def load_ice_env(config, task="G1JoystickFlatTerrain"):
    """Instantiate the strict-termination env directly.

    Bypasses registry.load so the subclass is what gets traced, rather than a
    stock env with an attribute swapped afterwards.
    """
    IceJoystick = make_ice_env_class()
    return IceJoystick(task=task, config=config)
