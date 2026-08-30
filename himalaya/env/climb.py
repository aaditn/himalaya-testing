# ==============================================================================
# Quadrupedal slope climbing for the Unitree G1. Ours, not vendored.
#
# Builds on the vendored Joystick task rather than copying it: reset(), step(),
# the observation stack, command sampling and the push schedule are all reused
# unchanged, and only the reward set, the termination rule and the scene differ.
#
# Subclassing is safe here in a way that runtime patching is NOT. The hazard
# this repo warns about is reassigning a bound method on an instance
# (env._get_termination = ...), which jax.jit captures the original of while
# every isolated test still reports the patch as live. A subclass is resolved
# normally through `self` when brax traces step(), so the override really runs.
# ==============================================================================
"""Climb a 30-45 degree rough slope on all four limbs."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env

from himalaya.env import base as g1_base
from himalaya.env import g1_constants as consts
from himalaya.env import gait
from himalaya.env.joystick import Joystick, default_config as joystick_config

CLIMB_XMLS = {
    "incline": consts.ROOT_PATH / "xmls" / "scene_mjx_climb_incline.xml",
    "flat": consts.ROOT_PATH / "xmls" / "scene_mjx_climb_flat.xml",
}

# Limb-tip contacts we actually want loaded, and the body contacts we do not.
TIP_SENSORS = ["left_foot_floor_found", "right_foot_floor_found",
               "left_hand_floor_found", "right_hand_floor_found"]
BODY_SENSORS = ["left_shin_floor_found", "right_shin_floor_found",
                "left_thigh_floor_found", "right_thigh_floor_found"]


def default_config() -> config_dict.ConfigDict:
  """Climbing config. Starts from the walking one and rewrites what fights it."""
  cfg = joystick_config()

  # Half the walking reward set is actively wrong for a crawl, so these are
  # switched off deliberately rather than left at their tuned walking values:
  #
  #   orientation (-2.0)  penalises torso tilt. The solved crawl stance pitches
  #                       the pelvis ~47 deg; this term punishes the posture the
  #                       task requires. Replaced by climb_roll, which penalises
  #                       tilt ACROSS the slope only.
  #   feet_phase  (1.0)   scores a bipedal swing-foot height curve against a
  #                       two-phase clock. There is no such gait here.
  #   feet_air_time (2.0) rewards alternating single-foot flight -- again
  #                       bipedal, and it rewards lifting limbs off a slope the
  #                       robot needs four points of contact on.
  #   stand_still (-1.0)  is defined against the walking nominal pose.
  #   base_height/feet_*  measured in world z, meaningless on an incline.
  cfg.reward_config.scales = config_dict.create(
      # --- the task ---
      climb_progress=3.0,      # velocity up the slope. The objective.
      climb_height=0.5,        # keep the pelvis off the ground
      head_height=1.0,         # keep the HEAD up (0.55 m above the slope,
                               # where the head-up stance puts it). The old
                               # flat-torso stance dragged the head along the
                               # ground and nothing paid to raise it; this
                               # holds the chest up while moving, not only at
                               # the spawn pose.
      # --- how to climb ---
      # The first trained policy exposed why these three exist TOGETHER.
      # It "climbed" +4.2 m at reward 38 by skating: feet planted 100% of the
      # time with 8.8 m of sliding, hands flickering at 4-5 Hz (20-40 ms
      # airtimes). Nothing priced a sliding contact, and tip_contact as a
      # mean-of-4 actively punished lifting a limb to step.
      tip_contact=1.0,         # >=2 tips loaded pays full -- see _get_reward.
                               # Was >=3, but a trot stands on a DIAGONAL PAIR
                               # for half of every cycle; >=3 taxed the gait
                               # the phase clock below asks for.
      gait_phase=3.0,          # four-limb trot clock: tip height scored
                               # against get_rz() with DIAGONAL pairing --
                               # (Lfoot,Rhand) on phase[0], (Rfoot,Lhand) on
                               # phase[1]. Scored PER LIMB and averaged: at
                               # scale 1.0 with all four errors summed in one
                               # exponential (walking feet_phase style), any
                               # single off-phase limb zeroed the whole term
                               # -- measured 0.16/step, flat for 150M steps,
                               # while the policy kept galloping. Per-limb
                               # partial credit keeps a gradient on each limb
                               # independently, and 3.0 makes rhythm compete
                               # with progress instead of losing to it.
      pair_sync=-1.0,          # both FEET airborne together, or both HANDS:
                               # the gallop signature, priced directly. A trot
                               # flies diagonal pairs, never lateral ones.
      gait_contact=3.0,        # contact-schedule gating: each limb is paid
                               # for BEING PLANTED in its stance window and
                               # AIRBORNE in its swing window (|phase| <
                               # pi/3), diagonal pairs a half-cycle apart.
                               # Binary and per-limb, so unlike the height
                               # curve there is no way to collect it while
                               # galloping: the schedule itself is the paid
                               # behaviour, not a proxy for it. 3.0 because
                               # pair_sync alone taught the policy to GLUE its
                               # feet down (duty 1.00, zero lift-offs, 2.9 m
                               # of drag) -- the swing-window half of this
                               # term is the one payment a glued foot cannot
                               # collect, so it has to outweigh the drag
                               # economy it replaces.
      tip_slip=-2.0,           # squared slide speed of PLANTED tips. At the
                               # measured skate (~1.9 m^2/s^2 summed) this
                               # costs -3.7/step against progress's +3.0, so
                               # skating uphill is strictly unprofitable; a
                               # stepped crawl slides ~0 and pays ~0.
      tip_swing=5.0,           # touchdown pays (airtime - 0.1 s): a real
                               # 0.2-0.4 s swing earns, a 20-40 ms flicker
                               # step COSTS. Zero while a limb stays planted.
      tip_hover=-2.0,          # seconds beyond 0.35 s that any limb has been
                               # airborne, per step -- PROPORTIONAL, not a
                               # threshold. The slip+swing policy crawled on
                               # three limbs with the left hand held up (duty
                               # 0.09, 740 ms airtimes); a boolean cost at
                               # 0.5 s taught it to ride the cliff at 440 ms
                               # (duty 0.33). A gradient with no line to camp
                               # under. Threshold 0.45 s: the trot clock's own
                               # swing is 0.33-0.40 s at gait_freq 1.25-1.5,
                               # and the hover cost must not tax the gait the
                               # phase reward asks for.
      body_drag=-1.0,          # shins/thighs dragging on the ground
      lateral=-0.5,            # drifting across the slope
      climb_roll=-1.0,         # tipping sideways relative to the slope
      # --- regularisers, kept from the walking task ---
      termination=-100.0,
      # 0.0, matching the walking task's tuned value. An earlier version set
      # this to 1.0, which pays the robot +1/step simply for not terminating --
      # and "descend slowly without falling" maximises exactly that. It also
      # pays for a climb-fall-reset cycle, which is the observed 45 deg failure:
      # the policy reaches +1.68 m, falls, and still banks the survival reward.
      # Progress is the objective; existing is not.
      alive=0.0,
      torques=-1.0e-4,
      action_rate=-1.0e-2,
      energy=-1.0e-4,
      dof_pos_limits=-1.0,
      # Weighted, not uniform -- see _post_init_climb. At a uniform -0.05 the
      # arms contorted freely: wrists and shoulder roll/yaw barely affect the
      # physics of the crawl, so twisting them cost ~nothing and the policy
      # waved them wherever exploration left them. Wrists are pinned hard,
      # shoulder roll/yaw firmly; the stride joints (shoulder pitch, hips,
      # knees) stay near-free so the gait itself is not taxed.
      pose=-0.5,
      collision=-0.1,
      contact_force=-0.01,
  )
  cfg.reward_config.target_climb_speed = 0.4   # m/s up the slope
  cfg.reward_config.target_climb_height = 0.35  # pelvis above the slope plane

  # IDEAL CONDITIONS. The goal of this run is the highest possible chance that
  # climbing is learned at all; robustness can be trained back in later as a
  # curriculum (re-enable these, tighten the friction band in
  # randomize_climb.py) once a climbing policy exists to warm-start from.
  #
  #   pushes  -- the walking task shoves the base at 0.1-2.0 m/s every 5-10 s.
  #              On a 30-45 deg slope a 2 m/s lateral kick is most of a fall;
  #              during first learning it just teaches "everything ends anyway".
  #   noise   -- observation noise regularises a policy that already works.
  #              While the task is unsolved it only blurs the contact and
  #              gravity signals the crawl has to be learned from.
  cfg.push_config.enable = False
  cfg.noise_config.level = 0.0

  # Scripted trot baseline in the ACTION SPACE: motor targets become
  # trot(phase) + policy residual, so the diagonal gait exists by construction
  # and RL only learns balance and propulsion. The escalation that led here:
  # every reward-only formulation was satisfied degenerately (skate -> hover
  # -> gallop -> glued feet). OFF by default; train_climb.py --scripted-gait.
  cfg.scripted_gait = False
  # Trot clock band, Hz. The walking task hardcodes 1.25-1.5; a slower clock
  # gives longer stance phases (more stable on a slope) and a more deliberate
  # gait. NOTE the hover threshold (0.45 s) was sized for this band's swing
  # length; a much slower clock needs it re-checked.
  cfg.gait_freq_range = (1.25, 1.5)
  # Compile the slope INTO the scene XML at this angle (degrees; 0 keeps the
  # scene's own euler). This is the only slope control that reaches the WARP
  # backend's physics: warp bakes static worldbody geom poses at put_model,
  # so runtime geom_quat writes (randomize_climb's mechanism, and the old
  # record path) tilt the reward frame and the spawn WITHOUT moving the
  # collision plane. Verified: data.geom_xmat of the floor stays at the
  # compile-time angle under warp after a geom_quat tree_replace; the jax
  # impl does honour the write.
  cfg.slope_compile_deg = 0.0
  # Four limbs on rough ground make far more contacts than a biped on flat.
  cfg.njmax = 300
  cfg.naconmax = 16 * 8192
  # A fall on a slope is not recoverable; long episodes just bank the fall cost.
  cfg.episode_length = 800
  return cfg


class Climb(Joystick):
  """Quadrupedal climbing on a tiltable rough slope."""

  # Termination, in the SLOPE frame -- world-frame thresholds are meaningless
  # here. The walking task's `qpos[2] < 0.4` is a world height: on a slope the
  # robot gains world z simply by climbing, so it measures the wrong thing.
  MIN_CLIMB_HEIGHT = 0.12   # pelvis above the slope plane, metres
  MAX_CLIMB_ROLL = 0.7      # |torso z-axis . across-slope axis|

  # The angle the spawn stance in the scene keyframe was solved at is the
  # scene's compile-time floor angle, so it is READ from the model in
  # _post_init_climb rather than hardcoded: 37.5 deg for the incline scene,
  # 0 for the flat control. A hardcoded 37.5 rotated the flat scene's
  # flat-solved stance 37.5 deg into the ground on every reset.

  def __init__(
      self,
      task: str = "incline",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    # Skip Joystick.__init__: it resolves the scene through
    # consts.task_to_xml(), which only knows the two walking terrains.
    xml_path = CLIMB_XMLS[task]
    deg = float(config.get("slope_compile_deg", 0.0) or 0.0)
    if deg:
      src = xml_path.read_text()
      token = 'euler="0 -0.6545 0"'
      if token not in src:
        raise ValueError(f"{xml_path} has no floor euler token to substitute")
      # System temp dir, NOT the xmls dir: /code is mounted read-only on HF
      # Jobs. The <include> files resolve through base.get_assets() by
      # basename, so the compiled scene can live anywhere.
      import tempfile
      tmp = (epath.Path(tempfile.gettempdir())
             / f"_compiled_slope_{deg:g}deg.xml")
      tmp.write_text(src.replace(
          token, f'euler="0 {-float(np.deg2rad(deg)):.6f} 0"'))
      xml_path = tmp
    g1_base.G1Env.__init__(
        self,
        xml_path=xml_path.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()
    self._post_init_climb()

  def _post_init_climb(self) -> None:
    sensor = lambda n: self._mj_model.sensor(n).id
    self._tip_sensors = [sensor(n) for n in TIP_SENSORS]
    self._body_sensors = [sensor(n) for n in BODY_SENSORS]
    self._foot_sensors = self._tip_sensors[:2]
    self._hand_sensors = self._tip_sensors[2:]
    # The angle the scene KEYFRAME was solved at. Normally that is the
    # scene's own compile-time floor angle, read from the model. When
    # slope_compile_deg re-compiles the floor at a different angle, the
    # keyframe is still the 0.6545 rad (37.5 deg) solve from the original
    # file, so the reference must stay pinned there -- reset()'s rigid
    # rotation then carries the stance onto the recompiled slope.
    if float(self._config.get("slope_compile_deg", 0.0) or 0.0):
      self._spawn_slope_ref = 0.6545
    else:
      q = np.array(self._mj_model.geom_quat[self._floor_geom_id])
      self._spawn_slope_ref = 2.0 * np.arctan2(-q[2], q[0])
    # Global linvel of all four tips, (4, 3) sensordata addresses in the same
    # order as TIP_SENSORS: Lfoot, Rfoot, Lpalm, Rpalm. The palm sensors live
    # in the climb scenes' own <sensor> block; the walking scenes lack them,
    # which is fine -- only Climb reads these.
    adrs = []
    for name in ["left_foot_global_linvel", "right_foot_global_linvel",
                 "left_palm_global_linvel", "right_palm_global_linvel"]:
      sid = self._mj_model.sensor(name).id
      a = self._mj_model.sensor_adr[sid]
      adrs.append(list(range(a, a + 3)))
    self._tip_linvel_adr = jp.array(adrs)

    # Tip site ids in TIP_SENSORS order: Lfoot, Rfoot, Lpalm, Rpalm.
    self._tip_site_id = jp.array(
        [self._mj_model.site(n).id
         for n in consts.FEET_SITES + consts.HAND_SITES])
    # Height of each tip SITE above the slope surface when planted, measured
    # on the keyframe stance. get_rz() returns height above ground; without
    # this offset the clock asks the sites to reach the surface itself, which
    # a planted foot's site (0.033 m up) can never do.
    import mujoco
    d = mujoco.MjData(self._mj_model)
    d.qpos[:] = self._mj_model.keyframe("knees_bent").qpos
    mujoco.mj_forward(self._mj_model, d)
    ref = self._spawn_slope_ref
    n0 = np.array([-np.sin(ref), 0.0, np.cos(ref)])
    self._tip_site_offset = jp.array(
        np.array(d.site_xpos)[np.array(self._tip_site_id)] @ n0)

    # Per-joint pose-cost weights (29,), joint order: 2x leg (hip_pitch,
    # hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll), waist (yaw, roll,
    # pitch), 2x arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
    # wrist_roll, wrist_pitch, wrist_yaw). Stride joints near-free, lateral
    # and wrist joints pinned to the solved stance.
    leg = [0.1, 1.0, 1.0, 0.05, 0.1, 0.5]
    arm = [0.2, 1.0, 1.0, 0.3, 2.0, 2.0, 2.0]
    self._pose_weights = jp.array(leg + leg + [1.0, 1.0, 0.5] + arm + arm)

    # Scripted-trot swing deltas, (4, 29) in TIP_SENSORS limb order. Signs and
    # sizes MEASURED at the stance by finite difference -- and they are NOT
    # portable between stances: at the old flat-torso stance (arms overhead)
    # NEGATIVE shoulder_pitch lifted a hand; at the head-up stance (arms under
    # the shoulders) the sign FLIPS. Measured at the head-up stance:
    #   knee   +0.28 m/rad lift AND +0.15 uphill (flexing tucks the foot)
    #   hip_p  +0.22 lift but -0.18 uphill; small negative for mild reach
    #   sho_p  +0.14 lift, elbow +0.12 lift (arms-under: flexion raises hand)
    # Re-measure with the finite-difference probe before editing these.
    # Measured at the head-up downward-dog stance (knee 0, arms as struts):
    #   foot: knee +0.24 lift, hip_p +0.51 lift / -0.40 uphill
    #   hand: sho_p +0.06 lift / -0.44 uphill (a retraction), elbow +0.02 --
    #   hand swings are retract-and-replace; the residual shapes placement.
    D = np.zeros((4, 29))
    D[0, 0], D[0, 3] = 0.10, 0.40           # Lfoot: hip_pitch, knee
    D[1, 6], D[1, 9] = 0.10, 0.40           # Rfoot
    D[2, 15], D[2, 18] = 0.25, 0.15         # Lhand: shoulder_pitch, elbow
    D[3, 22], D[3, 25] = 0.25, 0.15         # Rhand
    self._swing_deltas = jp.array(D)

  def _trot_targets(self, phase: jax.Array) -> jax.Array:
    """Joint targets of the scripted diagonal trot at this clock reading.

    Same schedule as the gait_contact reward window: a limb swings while
    |its phase| < pi/3, with a cos-shaped flexion peaking mid-window, and
    holds the solved stance otherwise. Diagonal pairing matches the reward:
    (Lfoot, Rhand) on phase[0], (Rfoot, Lhand) on phase[1].
    """
    limb_phase = jp.array([phase[0], phase[1], phase[1], phase[0]])
    s = jp.cos(1.5 * limb_phase) * (jp.abs(limb_phase) < jp.pi / 3.0)
    return self._default_pose + s @ self._swing_deltas

  def _motor_targets(self, action: jax.Array, info: dict) -> jax.Array:
    if not self._config.scripted_gait:
      return super()._motor_targets(action, info)
    return (self._trot_targets(info["phase"])
            + action * self._config.action_scale)

  # --- slope geometry -------------------------------------------------------

  def _slope_frame(self):
    """(normal, uphill, across) unit vectors of the floor, in world coords.

    Read from the model rather than hardcoded, because randomize_climb.py
    rotates the floor geom per environment. Under brax's domain-randomisation
    wrapper self.mjx_model is the per-environment model, so this returns THAT
    environment's slope -- which is the whole point of randomising the angle.
    """
    quat = self.mjx_model.geom_quat[self._floor_geom_id]
    normal = _rotate(jp.array([0.0, 0.0, 1.0]), quat)
    uphill = _rotate(jp.array([1.0, 0.0, 0.0]), quat)
    across = jp.cross(normal, uphill)
    return normal, uphill, across

  def _height_above_slope(self, data: mjx.Data) -> jax.Array:
    normal, _, _ = self._slope_frame()
    return jp.dot(data.qpos[0:3], normal)

  # --- spawn ----------------------------------------------------------------

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Spawn the four-point stance on THIS environment's slope, facing uphill.

    The inherited walking reset is actively wrong on an incline, in three ways
    that together held episodes to ~18 steps (0.36 s) with a flat reward curve:

      yaw = U(-pi, pi)      spawns the robot facing downhill or across the
                            slope. The stance was solved facing uphill; any
                            other heading falls immediately.
      qpos[7:] *= U(0.5,1.5) multiplies every joint angle, so a knee solved at
                            2.07 rad arrives anywhere in 1.03-3.10. The stance
                            is destroyed before the first step.
      qpos[0:2] += U(-.5,.5) is harmless on a level floor, but 0.5 m of x on a
                            40 deg slope is 0.42 m of height: the robot spawns
                            airborne or underground.

    It also never adapts to the randomised slope, because _init_q is a fixed
    keyframe. Here the stance is rotated rigidly onto the environment's own
    slope. That is EXACT, not approximate: the floor plane passes through the
    origin and rotates about the y axis through the origin, so rotating the
    robot about the same axis by the same angle preserves the solved contact
    geometry at every angle in the band.
    """
    normal, uphill, across = self._slope_frame()
    floor_quat = self.mjx_model.geom_quat[self._floor_geom_id]
    theta = 2.0 * jp.arctan2(-floor_quat[2], floor_quat[0])
    delta = theta - self._spawn_slope_ref

    qpos = self._init_q
    q_delta = jp.array([jp.cos(delta / 2), 0.0, -jp.sin(delta / 2), 0.0])
    base_pos = math.rotate(qpos[0:3], q_delta)
    base_quat = math.quat_mul(q_delta, qpos[3:7])

    # Heading noise about the SLOPE NORMAL, and small -- not U(-pi, pi).
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, minval=-0.25, maxval=0.25)
    base_quat = math.quat_mul(math.axis_angle_to_quat(normal, yaw), base_quat)

    # Offset ALONG the slope surface, so height above it is unchanged.
    rng, key = jax.random.split(rng)
    off = jax.random.uniform(key, (2,), minval=-0.4, maxval=0.4)
    base_pos = base_pos + off[0] * uphill + off[1] * across

    # Additive joint noise: multiplicative jitter destroys the solved stance.
    rng, key = jax.random.split(rng)
    joints = qpos[7:] + jax.random.uniform(key, (29,), minval=-0.05, maxval=0.05)

    qpos = jp.concatenate([base_pos, base_quat, joints])
    # A 0.5 m/s kick on a 40 deg slope is most of a fall; the walking task can
    # absorb it on the flat.
    rng, key = jax.random.split(rng)
    qvel = jp.zeros(self.mjx_model.nv).at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.1, maxval=0.1)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=joints,
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(
        key, (1,),
        minval=self._config.gait_freq_range[0],
        maxval=self._config.gait_freq_range[1],
    )
    rng, cmd_rng = jax.random.split(rng)
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    info = {
        "rng": rng,
        "step": 0,
        "command": self.sample_command(cmd_rng),
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        "phase_dt": 2 * jp.pi * self.dt * gait_freq,
        "phase": jp.array([0, jp.pi]),
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": jp.round(push_interval / self.dt).astype(jp.int32),
        # Seconds each tip (Lfoot, Rfoot, Lpalm, Rpalm) has been airborne.
        # Updated inside _get_reward, which runs exactly once per step.
        "tips_air_time": jp.zeros(4),
    }
    metrics = {f"reward/{k}": jp.zeros(())
               for k in self._config.reward_config.scales.keys()}
    metrics["swing_peak"] = jp.zeros(())

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sid]] > 0
        for sid in self._feet_floor_found_sensor
    ])
    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  # --- termination ----------------------------------------------------------

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    normal, _, across = self._slope_frame()
    # body down on the slope
    down = self._height_above_slope(data) < self.MIN_CLIMB_HEIGHT
    torso_z = self.get_gravity(data, "torso")
    # Tipped ACROSS the slope. Pitch is deliberately not terminated: the crawl
    # stance pitches ~47 deg by construction, so the walking task's 60 deg tilt
    # rule would fire on the spawn pose itself.
    tipped = jp.abs(jp.dot(torso_z, across)) > self.MAX_CLIMB_ROLL
    # There is deliberately NO torso-vs-normal "inverted" test. A crawl torso
    # lies near-parallel to the slope, and HOW near depends on the stance the
    # pose solver happens to return: dot(torso_z, normal) measured +0.34 for one
    # solved stance and -0.67 for another, both perfectly good four-point
    # postures. Any threshold on it terminates a valid posture for some stance
    # -- the -0.6 version killed episodes at 1 step. Height above the slope
    # already catches a body that is down, and roll catches one that has tipped
    # over sideways; neither depends on how far the torso is pitched.
    return (
        down
        | tipped
        | jp.isnan(data.qpos).any()
        | jp.isnan(data.qvel).any()
    )

  # --- rewards --------------------------------------------------------------

  def _sensor_found(self, data: mjx.Data, sensor_ids) -> jax.Array:
    return jp.array([
        data.sensordata[self._mj_model.sensor_adr[sid]] > 0 for sid in sensor_ids
    ])

  def _get_reward(
      self, data, action, info, metrics, done, first_contact, contact
  ) -> dict[str, jax.Array]:
    del metrics, first_contact, contact
    normal, uphill, across = self._slope_frame()
    vel = self.get_global_linvel(data, "pelvis")
    rc = self._config.reward_config

    up_speed = jp.dot(vel, uphill)
    # Saturating, not linear: a linear term pays unboundedly for launching
    # itself up the hill, which on a slope means a leap and then a fall.
    progress = jp.clip(up_speed / rc.target_climb_speed, -1.0, 1.0)

    h = self._height_above_slope(data)
    height = jp.exp(-jp.square(h - rc.target_climb_height) / 0.02)

    # Head point: ~0.4 m above torso_link along its z axis, same proxy the
    # pose solver's head term uses. Height measured along the slope normal.
    torso_R = data.xmat[self._torso_body_id].reshape(3, 3)
    head_pt = data.xpos[self._torso_body_id] + torso_R @ jp.array([0.0, 0.0, 0.4])
    head_h = jp.dot(head_pt, normal)
    head_height = jp.exp(-jp.square(head_h - 0.55) / 0.02)

    tips = self._sensor_found(data, self._tip_sensors)
    body = self._sensor_found(data, self._body_sensors)
    torso_z = self.get_gravity(data, "torso")
    tips_f = tips.astype(jp.float32)

    # Slide speed of PLANTED tips, in the slope plane. This is what makes
    # skating unprofitable: 8.8 m of foot slide per 10 s clip was the entire
    # locomotion mechanism of the first policy.
    tip_vel = data.sensordata[self._tip_linvel_adr]          # (4, 3) world
    tip_vel_tan = tip_vel - jp.outer(tip_vel @ normal, normal)
    slip = jp.sum(jp.sum(jp.square(tip_vel_tan), axis=-1) * tips_f)

    # A real swing pays, a flicker costs. airtime is read BEFORE the update,
    # so a touchdown is scored by how long the limb was actually in the air.
    air = info["tips_air_time"]
    touchdown = (air > 0.0) & tips
    swing = jp.sum(jp.clip(air - 0.1, -0.1, 0.3) * touchdown.astype(jp.float32))
    info["tips_air_time"] = (air + self.dt) * (1.0 - tips_f)

    # Trot clock: diagonal pairs on opposite half-cycles. phase[0] drives
    # Lfoot+Rhand, phase[1] (offset pi) drives Rfoot+Lhand -- a dog's trot.
    # Tip height above THIS env's slope, minus the planted-site offset, is
    # scored against the Bezier swing curve, exactly like the walking task's
    # feet_phase but with four limbs and slope-frame heights.
    limb_phase = jp.array([info["phase"][0], info["phase"][1],
                           info["phase"][1], info["phase"][0]])
    rz = gait.get_rz(limb_phase, swing_height=0.08)
    tip_h = data.site_xpos[self._tip_site_id] @ normal - self._tip_site_offset
    # Per-limb, then averaged -- see the scale comment in default_config.
    gait_phase = jp.mean(jp.exp(-jp.square(tip_h - rz) / 0.005))
    pair_sync = ((1 - tips_f[0]) * (1 - tips_f[1])
                 + (1 - tips_f[2]) * (1 - tips_f[3]))
    # Contact-schedule gating: planted in stance window, airborne in swing
    # window. get_rz peaks at phase 0, so |phase| < pi/3 IS the swing window.
    want_swing = (jp.abs(limb_phase) < jp.pi / 3.0).astype(jp.float32)
    gait_contact = jp.mean(want_swing * (1.0 - tips_f) + (1.0 - want_swing) * tips_f)

    return {
        "climb_progress": progress,
        "climb_height": height,
        "head_height": head_height,
        # >=2 planted pays full: a trot stands on a diagonal pair for half of
        # every cycle. (Was mean-of-4, which subsidised the never-lift skate;
        # then >=3, which taxed the trot the clock asks for.)
        "tip_contact": jp.clip(jp.sum(tips_f) / 2.0, 0.0, 1.0),
        "gait_phase": gait_phase,
        "pair_sync": pair_sync,
        "gait_contact": gait_contact,
        "tip_slip": slip,
        "tip_swing": swing,
        "tip_hover": jp.sum(jp.clip(air - 0.45, 0.0, None)),
        "body_drag": jp.sum(body.astype(jp.float32)),
        "lateral": jp.square(jp.dot(vel, across)),
        "climb_roll": jp.square(jp.dot(torso_z, across)),
        "termination": self._cost_termination(done),
        "alive": self._reward_alive(),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        # Weighted: wrists/shoulder-lateral pinned, stride joints free.
        "pose": jp.sum(
            self._pose_weights * jp.square(data.qpos[7:] - self._default_pose)
        ),
        "collision": self._cost_collision(data),
        "contact_force": self._cost_contact_force(data),
    }


def _rotate(vec: jax.Array, quat: jax.Array) -> jax.Array:
  """Rotate `vec` by `quat` (w, x, y, z)."""
  w, u = quat[0], quat[1:]
  return vec + 2.0 * jp.cross(u, jp.cross(u, vec) + w * vec)
