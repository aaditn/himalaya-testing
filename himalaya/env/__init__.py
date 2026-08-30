"""The G1 joystick environment, vendored so it is ours to edit.

Playground's own env lives inside site-packages, which makes every reward or
observation change a subclass or a monkeypatch. Monkeypatching is worse than
it looks here: brax traces `step` through jax.jit, and a patch applied after
tracing is silently ignored during training while still passing every isolated
test. Owning the files removes that whole class of bug -- change the reward and
it is simply changed.

What is editable here:

  joystick.py    24 reward/cost terms, the observation stack, command
                 sampling, termination, push/perturbation schedule.
                 default_config() holds every reward scale and noise level.
  gait.py        get_rz -- the swing-foot height curve that the feet_phase
                 reward scores against. Shapes the stride.
  g1_constants.py  sites, geoms, sensor names, joint ranges, XML paths.
  randomize.py   domain randomization (friction, mass, armature, qpos0).
  xmls/          the scenes themselves: terrain, floor, sensors, collision
                 geometry. ROOT_PATH points here, so editing these changes
                 the physics -- verified by setting floor friction and
                 reading it back through the env.
  base.py        MJX env base for the G1.

Still imported from Playground: `mjx_env` only -- MjxEnv, State, step,
make_data, update_assets, MENAGERIE_PATH. That is plumbing, not behaviour;
nothing you would tune for a locomotion result lives in it. The robot meshes
also still load from Menagerie via MENAGERIE_PATH.

Vendored from MuJoCo Playground under Apache-2.0 (see LICENSE.playground).
Every local change is marked with a `MODIFIED:` comment naming the stock
behaviour it replaced.
"""

from himalaya.env.joystick import Joystick, default_config  # noqa: F401


def walk_on_slope_config(slope_deg: float = 35.0):
  """Run A: the walking reward, unchanged, on a tilted floor.

  The null test. It answers one question -- does the slope-frame retargeting of
  feet_phase and orientation actually work -- for about 1.5 GPU-hours, before
  200M steps are spent on a reward design that assumes it does.

  Nothing here rewards climbing. If a warm-started walking policy cannot stay
  upright on this terrain with the reward it was trained on, the terrain or the
  retargeting is wrong, and no reward design fixes that.

  Gate: episode length recovers to 500+, mean dot(torso_up, slope_normal) > 0.85.
  """
  cfg = default_config()
  cfg.slope_deg = slope_deg
  # Forward-only command. On a 35 degree slope a zero or negative command tells
  # the robot to stand still on a hill or walk down it, and standing still is
  # the safe-harbour behaviour that already killed two runs.
  cfg.lin_vel_x = [0.4, 0.9]
  cfg.lin_vel_y = [-0.15, 0.15]
  cfg.ang_vel_yaw = [-0.6, 0.6]
  return cfg


def climb_walk_config(slope_deg: float = 15.0):
  """Run B: walk UP the slope. Run A's config plus the climb objective.

  Run A proved the walking reward transfers to a tilted mountain -- 0 falls in
  a 12 s clip, episode length 68 -> 239 over 41M steps. But it drifted 0.31 m
  DOWNhill, because nothing in that config paid for height. This adds the one
  term that does.

  progress_uphill is deliberately posture-blind and route-blind: it rewards
  metres of height gained per second and says nothing about how or where. The
  corridor wins not because the reward names it but because it is the cheapest
  line up -- off-lane is broken ground and 0.55-0.85 m banks. That is what
  keeps the route emergent rather than scripted.

  Weight 1.5, not 8.0. Its job is not to be the objective; tracking_lin_vel
  already is, and it is measured in the body frame so on a slope-aligned torso
  it already means "up the hill". progress_uphill only has to break the
  symmetry between walking up the corridor and walking down it, which score
  identically otherwise. At 1.5 it pays ~190 per 1000-step episode against
  walking's ~1200; turning around swings it to -190, a 380 gap that is
  decisive without being worth abandoning the gait for.
  """
  cfg = walk_on_slope_config(slope_deg)
  # 4.0, was 1.5. Measured on Run D's final eval, progress_uphill earned 63 per
  # episode against tracking_lin_vel's 192 -- climbing paid a third of what
  # walking around paid, so the policy wandered between paths and the net
  # height stayed negative. At 4.0 it earns ~170, comparable to
  # tracking_lin_vel rather than dominating it, which keeps the gait.
  cfg.reward_config.scales.progress_uphill = 4.0
  # No xy jitter. The robot starts on the flat SPAWN_PAD square that
  # make_terrain_bank.py carves at scene.SPAWN in every variant. Jittering
  # +/-0.5 m off it put Run L on wall flanks in a large share of episodes --
  # 29 falls in 20 s. Variety comes from the 16 terrain variants and the
  # heading jitter (scene.SPAWN_YAW_JITTER), neither of which moves the
  # start off the pad.
  cfg.spawn_jitter = 0.0
  return cfg


def climb_config(slope_deg: float = 30.0):
  """Config for the climbing task: reach the top of a steep rough slope.

  Everything here is a deletion except one added term. The bipedal reward
  actively forbids the posture a climber needs, so the work is removing those
  constraints and letting the policy find its own answer:

    orientation (-2.0)    targets a world-vertical torso. Second largest
                          active term, and a climber on a 45 degree slope is
                          tilted ~45 degrees, so it pays continuously for
                          being correct.
    feet_air_time (+2.0)  largest positive term, and it pays for keeping feet
                          OFF the ground -- the opposite of four-point contact.
    feet_phase (+1.0)     a two-limb alternating gait clock scored against
                          world z. Bipedal by construction, and wrong on a
                          tilted surface.
    pose (-0.1)           pulls all 29 joints toward a standing pose, taxing
                          every reach toward the ground.
    collision (-0.1)      the only hand-related term, and it PENALISES hand
                          contact (with the thigh). Exactly the wrong sign.
    joint_deviation_*     pin hips and knees near a standing configuration.
    stand_still (-1.0)    would demand a standing pose on a steep slope when
                          the command is near zero, which is unreachable.
    feet_slip (-0.25)     measures PELVIS speed gated on foot contact, not
                          actual foot slip, so on a slope it reads "do not
                          move while a foot is down".

  What remains: progress_uphill (the objective), termination (-100, which is
  what makes falling matter), dof_pos_limits, ang_vel_xy, contact_force.

  No term rewards palm contact. If the hands go down it is because that is
  the only way up.
  """
  cfg = default_config()
  cfg.slope_deg = slope_deg
  # Spawn across most of the 12 m patch rather than the default +/-0.5 m,
  # which is 8% of it. Otherwise the policy sees the same rock every episode
  # and can learn that specific ground instead of climbing.
  cfg.spawn_jitter = 4.0

  scales = cfg.reward_config.scales
  # Climbing is the whole objective and the only positive term, so it carries
  # real weight. Symmetric by construction: the reward is a signed projection
  # onto the uphill direction, so sliding back down is penalised just as
  # heavily as climbing is rewarded.
  scales.progress_uphill = 8.0

  for term in (
      "orientation", "feet_air_time", "feet_phase", "pose", "collision",
      "joint_deviation_hip", "joint_deviation_knee", "stand_still",
      "feet_slip", "tracking_lin_vel", "tracking_ang_vel",
  ):
    scales[term] = 0.0

  return cfg
