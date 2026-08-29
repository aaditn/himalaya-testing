# himalaya

RL locomotion for the Unitree G1 (29-DOF). Sim-only.

## What this is

Train a G1 humanoid to walk, in simulation, using PPO on MuJoCo MJX
(via MuJoCo Playground).

## Layout

```
himalaya/env/             the task: rewards, observations, termination
himalaya/mjx/             loads the G1 from MuJoCo Menagerie
himalaya/utils/           early-kill monitor for long jobs
scripts/train.py          PPO training -> runs/<name>/
scripts/record.py         render a trained policy to MP4
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

Modifications carry a `MODIFIED:` comment. So far: stricter fall termination
(`MAX_TILT`, `MIN_TORSO_HEIGHT` in `joystick.py`).

```
himalaya/env/joystick.py       24 reward/cost terms, observations, termination
himalaya/env/g1_constants.py   sites, geoms, sensors, joint ranges
himalaya/env/randomize.py      domain randomization
himalaya/env/base.py           MJX env base
```

`mjx_env` and `gait` are still imported from Playground — generic
infrastructure, nothing G1-specific to own.

## Quick start

```bash
python scripts/inspect_model.py            # sanity: does the robot stand?
python scripts/train.py --timesteps 60_000_000
python scripts/record.py runs/<name>/policy --out videos/walk.mp4
```

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
