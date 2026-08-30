# himalaya

RL four-limb climbing for the Unitree G1 (29-DOF). Sim-only.

## What this is

Train a G1 to ascend rough, steep terrain in a continuous crawl using its
microspike-modeled hands and feet. The implementation is based directly on
[`aaditn/himalaya-testing`](https://github.com/aaditn/himalaya-testing): its
vendored MuJoCo Playground environment, PPO trainer, domain randomization,
recording workflow, pod helpers, and run layout remain the platform.

## Layout

```
himalaya/env/             the task: rewards, observations, termination
himalaya/mjx/             loads the G1 from MuJoCo Menagerie
himalaya/utils/           early-kill monitor for long jobs
scripts/train.py          PPO training -> runs/<name>/
scripts/train_climb_curriculum.py  staged 5° -> 42° training
scripts/record.py         render a trained policy to MP4
scripts/upload_huggingface.py      package/upload a run
scripts/view.py           watch the robot locally in MuJoCo's viewer
scripts/inspect_model.py  does the robot stand? (no policy, no rewards)
scripts/pod/              pull results off a training pod
runs/                     checkpoints + metrics (gitignored)
```

## Changing the environment

`himalaya/env/` is a vendored copy of MuJoCo Playground's G1 joystick task
(Apache-2.0, see `LICENSE.playground`). It is checked in deliberately: this
project iterates on rewards and termination, and those live in
`himalaya/env/joystick.py` where they can simply be edited.

The alternative — importing Playground's class and patching it — is worse than
it looks. brax traces `step` through `jax.jit`, so a monkeypatch applied after
tracing is silently ignored during training while still passing every isolated
test. Owning the file removes that failure mode.

Modifications carry a `MODIFIED:` comment. The `climb_terrain` task adds an
IK-solved crawl reset, inclined rough terrain, 44 microspike contact pairs,
palm contact/load sensing, continuous hand support and diagonal gait rewards,
climb observations/metrics, and crawl-aware termination. Each stock rubber
hand is replaced by a visible, rigid 5 cm-radius sphere centered on the palm
site. The sphere is the hand contact patch; it is attached to the wrist rather
than a freely rolling ball. Existing palm sites, force sensors, contact names,
rewards, and boulder pairs are retained.

The spherical end-effectors use nearly straight wrists: roll, pitch, and yaw
are centered at zero and limited to +/-0.08 rad, with wrist position gain 20
and +/-25 Nm actuator limits. This retains a small compliant envelope while
preventing a wrist from folding and wedging the forearm against the terrain.
In the same four-second, 5-degree reference-only check used for the unrestricted
wrists, this changed the result from two falls and +0.066 m net ascent to zero
falls and +0.224 m net ascent.

```
himalaya/env/joystick.py       24 reward/cost terms, observations, termination
himalaya/env/gait.py           swing-foot height curve (stride shape)
himalaya/env/g1_constants.py   sites, geoms, sensors, joint ranges
himalaya/env/randomize.py      domain randomization
himalaya/env/xmls/             scenes: terrain, floor, sensors, collision geometry
himalaya/env/base.py           MJX env base
```

`ROOT_PATH` points at the local `xmls/`, so editing a scene changes the
physics — verified by setting floor friction to 0.02 and reading it back
through the env.

Only `mjx_env` is still imported from Playground: `MjxEnv`, `State`, `step`,
`make_data`, `update_assets`, `MENAGERIE_PATH`. That is plumbing — nothing you
would tune for a locomotion result lives in it. Robot meshes still load from
Menagerie.

## Quick start

```bash
python scripts/inspect_model.py            # sanity: does the robot stand?
python scripts/train.py --climb --slope 12 --timesteps 4_096 --envs 8 --name smoke
python scripts/train_climb_curriculum.py --envs 8192 --prefix g1_climb
python scripts/record.py runs/<name>/policy --climb --slope 12 --out videos/climb.mp4
```

Upload a completed run after `hf auth login`:

```bash
python scripts/upload_huggingface.py runs/<final-stage> your-org/g1-four-limb-climb --private
```

## Hugging Face Jobs remote

`myremote` is configured for the `iteratehack` organization in
`configs/hf_jobs.json`. The wrapper automatically scopes list/run commands to
that organization. Runs also mount the existing `iteratehack/himalaya-runs`
bucket read-write at `/runs` and receive the label `remote=myremote`.
Credentials are kept by the Hugging Face CLI and are never stored in this repo.

```bash
python scripts/myremote.py config
python scripts/myremote.py list --limit 5
python scripts/myremote.py run --name g1-climb --flavor a10g-large --timeout 8h \
  <image> <command> [args...]
```

Use `--no-runs-volume` after `run` for a job that should not mount the default
bucket. Starting a Job consumes the organization's paid compute; `config` and
`list` do not start anything.

## Four-limb climbing task

The desired gait is deliberately closer to walking on all fours than occasional
scrambling. Two 180°-opposed hand schedules overlap during hand exchange, so a
new palm is expected to load before the old palm lifts. The policy is rewarded
for at least one planted hand, diagonal hand-foot support, uphill motion while
supported, signed uphill displacement, uphill ground-reaction force delivered
through contacted feet, and a stage-specific hand load share. The foot-drive
term is multiplied by positive mountain progress, so bracing or stamping in
place does not earn it. Sliding a planted palm is penalized. Signed potential
progress has scale 10.0 and is computed from pelvis displacement along the
slope. Uphill motion receives full credit with hand support and 30% during a
brief hand exchange; downhill motion is penalized 1.75x. New episode-high
progress has scale 3.0, and each new 0.25 m waypoint receives a one-time 2.0
bonus. Waypoints follow the episode maximum, so crossing the same ground twice
cannot farm reward. Hand, diagonal-support, load-share, and foot-drive shaping
is gated by positive displacement. A -0.05 time cost and termination after
three seconds without a new 5 mm progress checkpoint discourage settling in
place; losing 0.35 m from the episode high also terminates the episode.

Large leg steps receive an additional foot-only, one-time touchdown bonus. It
uses uphill plant advance normalized by the 20 cm target, squares that value,
and compensates for the environment timestep. A 10 cm qualifying step earns
0.25, a 20 cm step earns 1.0, and a 30 cm step earns 2.25 at scale 1.0. A foot
must remain airborne for at least 80 ms and land beyond its episode-best plant,
so sliding, contact chatter, and repeated stamping cannot farm the bonus. The
reference-only diagnostic produced one qualifying event worth 0.135; improved
step frequency and length are unverified pending fresh training.

Dense forward-velocity shaping has scale 2.5, secondary to the 10.0 signed
displacement potential. It projects pelvis velocity onto the uphill slope
axis, pays linearly from zero to the commanded speed, and saturates at the
command. It is zero for stationary/backward motion, invalid crawl posture, or
loss of mixed hand-foot support. The signed potential remains responsible for
penalizing regression. In a four-second reference-only check at a 0.15 m/s
command, mean/max uphill velocity was +0.036/+0.439 m/s and mean scaled velocity
reward was 0.359, with zero falls.

The scheduled swing palm also has a 0.8 ft (0.24384 m) lift objective measured
terrain-normal from that hand's last meaningful plant height. This makes the
measurement valid after planting on a boulder rather than only on the nominal
slope plane. Smooth lift credit has scale 0.5 and saturates at the target; a
one-time threshold-crossing bonus has scale 0.2 and resets only after the palm
replants. Both require the opposite palm, at least one foot, valid posture, and
the correct swing phase, so lifting both hands or holding one aloft cannot farm
reward. The generic phase-height term is disabled because it requested hand
clearance during scheduled support. Opposed 60% hand duty cycles retain 20%
two-hand overlap while leaving 40% of each cycle for lift and reach. The final
zero-action reference reaches 0.108 m and correctly earns zero threshold
events; reaching 0.24384 m is unverified pending fresh training.

Knee grounding is handled by a differentiable terrain-clearance reward rather
than a world-Z heuristic. For each knee body origin, the environment subtracts
a measured 6 cm housing radius, configured height-field relief, and the nearest
physical boulder surface. A smooth positive reward (scale 0.2) saturates at 5
cm surface clearance and receives full weight while moving uphill; a small 20%
bootstrap component teaches the posture before motion starts. Crossing the
zero-clearance envelope incurs a separate -1.0 contact cost. This does not add
knee-ground collision pairs or change the 114/230 observation interface. In
the four-second reference-only diagnostic, the pre-training gait violated the
conservative envelope in 40.5% of frames with -9.0 cm minimum clearance, so the
desired no-knee-contact behavior is unverified pending fresh training.

Episodes begin in an IK-solved, forward-biased suspended crouch: both palms and
both feet retain their previous plant locations while the torso is exactly 10
cm farther uphill. The pelvis remains about 49 cm above the local ramp, knee
joint clearance is 18 cm, both wrists are straight, and no shin, thigh, pelvis,
or torso geom supports the robot. At the 12° reference grade the torso moves
from `[0.054, 0, 0.705]` to `[0.151, 0, 0.726]` m. Because this keyframe is also
the position-controller reference, checkpoints trained before the forward-pose
change must not be used as final policies; restart the curriculum at stage one.
The old zero-action crawl reference is not dynamically compatible with the new
pose: its four-second 5° check had one fall and essentially zero net progress.
This pose is provided for pre-training review, not as a trained result.

The curriculum in `configs/curriculum.json` first learns support exchange on a
5° grade with 5 mm relief and no boulders, then transfers to a smooth 12°
grade, and only then introduces 6 cm relief and the rocks. It subsequently
progresses through 20°, 28°, 35°, and a terminal 42° stage while increasing
terrain relief to 15 cm and desired hand loading to 40%. Hand microspikes retain
0.95 tangential friction while the foot microspikes use 1.90, exactly twice the
hand coefficient. At the terminal grade tan(42°) is about 0.90, so hands remain
near the nominal friction limit while stance feet have additional propulsion
margin. Ten fixed 10-inch-diameter (0.254 m) boulders are
distributed across every rocky stage. Each boulder has explicit collision
pairs with both palms and both feet; it is not a visual-only prop. Boulder
centers are placed one compiled radius above the sampled ramp, so changing rock
size cannot bury or float them. Disabled boulders are moved below the terrain
without changing model topology, so each stage can restore the previous Orbax
checkpoint.

Changing the configured grade also rigidly rotates/translates the suspended
crouch keyframe about the floor origin. This preserves all four initial
hand/foot contacts and torso alignment instead of leaving the robot posed for
the original 12° ramp. At 42°, the compiled reset retains four-point contact
and 0.978 torso-to-slope alignment. The untrained reference is not capable on
the terminal terrain: a four-second check produced eight terminations and net
downhill motion. Completion of the 35° stage followed by fresh 42° training is
required before claiming competence.

Start this curriculum without `--restore`. Earlier smoke checkpoints learned
to pitch or collapse uphill under obsolete rewards and are useful controls,
not production initialization:

```bash
python scripts/train_climb_curriculum.py --envs 8192 --prefix g1_climb
```

Microspikes are represented at contact-patch scale on the spherical hand
end-effectors and feet. During domain randomization, hand tangential friction is
sampled uniformly from 0.9 to 1.0 and foot friction is exactly twice the sampled
hand value (1.8 to 2.0). Contacts retain torsional coefficient 0.08, rolling
coefficient 0.03, and compliant six-dimensional constraints.
This is useful for policy discovery and sensitivity analysis; it does not model
individual teeth biting, clogging, breaking substrate, or pulling out.

On the 5° fixed-physics bootstrap, changing only the foot coefficient from 0.95
to 1.90 improved the existing policy from 7.1 cm net progress with one fall to
9.7 cm with no falls over six seconds. A further 40,960-step continuation reached
16.4 cm net progress with no falls, 0.028 m/s mean uphill velocity, and five
qualifying foot-step events in the same deterministic rollout.

Diagnostics showed why the early policies did not step: they already moved
joints and lifted feet, but the stock foot-phase reward used absolute world Z
despite the offset/inclined terrain, contact chatter counted as touchdown, and
there was no credit for a farther-uphill plant. More importantly, falling
forward could earn much more displacement reward than the scaled terminal
cost. The environment now uses terrain-normal swing clearance, requires 80 ms
air time for a touchdown, rewards only new episode-best limb plants, gates
positive displacement by pelvis clearance and torso alignment, and claws back
episode ascent credit on failure. The reward gate begins at 36 cm terrain-normal
pelvis clearance; the hard fall threshold is 25 cm to leave recovery margin.

Arm motion is coupled explicitly to the opposite leg through a small residual
crawl reference: right hand swings with left foot, then left hand with right
foot. The reference advances/lifts hip and knee together with opposite shoulder
pitch, shoulder roll, and elbow, while the planted diagonal extends its knee and
elbow to unload the moving pair. PPO actions correct that reference rather than
having to discover coordination from noise. The crawl clock is 0.70-0.95 Hz
instead of the stock 1.25-1.5 Hz biped cadence. Static whole-pose, hip, and knee
deviation costs are disabled for climbing because they penalized the required
cyclic motion; action-rate, energy, collision, soft joint-limit, posture, and
fall safeguards remain. Generic foot phase-height shaping is also disabled
because it conflicted with planted support; completed foot air time is rewarded
instead.

The hip reference is now a continuous +/-0.30 rad fore-aft sweep: the airborne
leg reaches while the planted leg retracts and pushes the pelvis uphill. Knee
lift remains a stable 0.38 rad half-wave, and touchdown advance is normalized
against a 20 cm target. In the same four-second reference-only check, the final
configuration produced zero falls, 24.9 cm net ascent, 5.9 cm/s mean uphill
velocity, and 14.5-18.8 cm foot excursions. The preceding one-sided large-step
reference produced 20.0 cm net ascent. The right foot still unloads less
cleanly than the left, so fresh PPO training must learn the remaining weight
shift before this is considered a finished crawl.

Short 8,320-step CPU smoke runs verify compilation and expose failure modes;
they do not train a crawl. The old collapse-trained checkpoint still fell once
in a two-second bootstrap evaluation and must not be treated as fixed merely
because it translated uphill. A fresh 30-million-step GPU bootstrap and
held-out terrain evaluation are required before claiming a successful climbing
policy or Class 2 capability.

Locally on macOS the viewer needs `mjpython`, not `python`, because the GUI
must own the main thread:

```bash
.venv/bin/mjpython scripts/view.py
```

## The robot

MuJoCo Menagerie's `unitree_g1`, vendored inside MuJoCo Playground — the same
model Playground's own G1 tasks use, so training and inspection share one
robot and there is no URDF conversion step.

29 DOF = 12 leg + 3 waist + 14 arm (7 per arm). The 3-DOF waist gives real
torso balance authority.

Measured from the model (`scripts/inspect_model.py`), not assumed:

```
actuated joints        29
total mass          33.34 kg
standing height      0.784 m   <- Menagerie's standing keyframe
```

Spawn height comes from the keyframe rather than a guess. An earlier attempt
used 1.05 m, copied from NVIDIA's 37-DOF config, which spawned the robot ~27 cm
in the air so it dropped and landed on every reset.

### Two scenes, not interchangeable

`scene_mjx.xml` — 5 solver iterations, simplified colliders, 4 ms timestep.
What training uses. It will **not** hold a pose open-loop: the standing
keyframe drops to z=0.11 within 1.5 s. That is a solver artifact of a scene
built for batched rollouts where a policy closes the loop every step.

`scene.xml` — 100 solver iterations, 2 ms timestep. The same open-loop test
stands at z=0.792, upright=1.000. Use it for inspection, viewing, and any
number quoted as a fact about the robot.

## Two bugs worth knowing about

Both silently corrupt results rather than crashing, so they are easy to
reintroduce. `njmax` is set in `scripts/train.py`; termination lives in
`himalaya/env/joystick.py`.

**Constraint overflow.** Playground ships `njmax=90`, too small for this robot.
When the solver runs out of constraint slots it *drops contacts*, and a dropped
foot contact means nothing holds the robot up that step. Training logged 2,580
overflow warnings before the fix.

**Termination too permissive.** Stock G1 ends an episode only when the torso
passes horizontal. A robot that tips to 89° — or lands on its back and settles
— never trips it, so it lies there collecting reward. The policy learns stable
fallen poses instead of walking, which makes every episode-length number from
an unpatched run meaningless. Fixed by also terminating on pelvis height and at
~60° of tilt.

The termination fix is **an edit to the vendored file**, not a patch applied at
runtime. Reassigning `env._get_termination` passes every isolated test but can
be lost to `jax.jit` tracing, so training silently uses the stock rule while
every direct test reports the patch as live.

## Gains

Menagerie ships tuned position servos (gain 75, damping 2 on the hips) and
sets rotor inertia itself, so there is nothing to tune before training. Gains
do not transfer across simulators — `unitree_rl_gym`'s published values assume
an implicit actuator, which applies the PD law differently from a MuJoCo
position servo — so treat any borrowed gain as a hypothesis until it holds the
robot up here.

Re-run `scripts/inspect_model.py` after any model change; it prints the settle
trace and a stand/collapse verdict.

## Cost discipline

1. **Branch from checkpoints.** Train flat walking once, reuse it.
2. **Screen cheap.** Fewer envs and short episodes for reward sweeps; promote
   only winners to full scale.
3. **Kill early, leniently.** `himalaya/utils/killswitch.py`.
4. **Spot instances** with ~10 min checkpointing — but not for the final run.
