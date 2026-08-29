# himalaya

RL locomotion for the Unitree G1 (23-DOF). Sim-only.

## What this is

Train a G1 humanoid to walk, in simulation, using PPO on MuJoCo MJX
(via MuJoCo Playground).

## Layout

```
assets/g1/                robot model: URDF + meshes (from unitree_ros)
himalaya/mjx/             builds a simulation-ready model from the raw URDF
himalaya/utils/           early-kill monitor for long jobs
scripts/train.py          PPO training -> runs/<name>/
scripts/record.py         render a trained policy to MP4
scripts/view.py           watch the robot locally in MuJoCo's viewer
scripts/inspect_model.py  does the robot stand? (no policy, no rewards)
scripts/pod/              pull results off a training pod
runs/                     checkpoints + metrics (gitignored)
```

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

23 DOF = 12 leg + 1 waist yaw + 10 arm (5 per arm). Waist **yaw only** — no
pitch or roll — so torso balance authority is limited.

Measured from the model itself (`scripts/inspect_model.py`), not assumed:

```
total mass         30.32 kg
arm mass            6.10 kg   (20.1% of total)
leg mass           14.37 kg
standing height     0.784 m   <- lowest foot geom below the pelvis frame
```

That last number matters: an earlier attempt used 1.05 m, copied from NVIDIA's
37-DOF G1 config. On this robot that spawns it ~27 cm in the air, so it drops
and lands on every reset.

## Two bugs worth knowing about

Both are fixed in `scripts/train.py`, and both silently corrupt results rather
than crashing — so they are easy to reintroduce.

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

The termination override is a **subclass, not a monkeypatch**. Reassigning
`env._get_termination` at runtime passes every isolated test, but brax traces
the step function through `jax.jit`; if tracing captures the original bound
method, training silently uses the stock termination while every direct test
reports the patch as live.

## PD gains

Tuned here, not copied. `unitree_rl_gym`'s published values (hip 100, knee 150,
ankle 40) do not hold this robot up in MuJoCo — the legs collapse under 34 kg in
half a second. Those gains assume an implicit actuator, which applies the PD law
differently from a MuJoCo position servo.

Rotor inertia (`armature = 0.01`) matters more than the gains did. Ours was 0,
which makes effective joint inertia tiny and any stiff PD controller numerically
unstable — the reason the robot collapsed identically across a 12-point gain
sweep.

Re-run `scripts/inspect_model.py` after any change to either; it prints the
settle trace and a stand/collapse verdict.

## Cost discipline

1. **Branch from checkpoints.** Train flat walking once, reuse it.
2. **Screen cheap.** Fewer envs and short episodes for reward sweeps; promote
   only winners to full scale.
3. **Kill early, leniently.** `himalaya/utils/killswitch.py`.
4. **Spot instances** with ~10 min checkpointing — but not for the final run.
