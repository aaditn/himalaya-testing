# himalaya

RL locomotion for Unitree G1 (23-DOF) on continuous rough terrain. Sim-only, Isaac Lab.

## Status

Scaffold + guardrails. Isaac Lab task implementation is the next piece.

## Layout

```
assets/g1/g1_23dof.urdf      model (from unitreerobotics/unitree_ros)
configs/g1_rough.yaml        single source of truth for all knobs
scripts/preflight.py         3 cheap tests to run before any long job
himalaya/utils/killswitch.py early-kill monitor
himalaya/tasks/              Isaac Lab task  <- TODO
```

## Robot

23 DOF = 12 leg + 1 waist_yaw + 10 arm (5/arm).

This variant has **waist yaw only** — no waist pitch/roll. Torso balance
authority is therefore limited, which is part of why arms may need explicit
encouragement rather than emerging on their own.

## Settled decisions

Converged across four independent sources in the research pass; not worth
re-deriving:

- `[512, 256, 128]` ELU. **Not** the 32-unit layers `unitree_rl_gym` ships for G1.
- 4096 envs, 24-step rollouts, γ=0.99, λ=0.95.
- Entropy 0.005 (sweep 0.005–0.01; 0.001 excluded, slows early exploration).
- Joint position targets @50 Hz → PD @200 Hz. Not torque control.
- Teacher-student: privileged teacher (heightmap + friction) → blind student.
- Terrain proportions are humanoid-calibrated, **not** legged_gym's
  quadruped 60%-stairs / 20 cm-step mix. Steps ramp 3 → 10 cm.

## Open question: arms for balance

The research pass found **no published work** training a G1 to use arms for
balance on rough terrain. Three related results exist but none is the thing.

Plan: the angular-momentum term `exp(−‖L_base‖₂/5)` is wired in at weight 0.
Run the arms-free baseline first, check whether arm joints actually move, and
flip the weight to ~0.3 if they go slack. Expect to need it — light 5-DOF arms
plus yaw-only waist means legs will likely solve the problem more cheaply on
their own. The baseline still runs first, so you know whether the AM term is
doing real work or covering for a reward-balance bug elsewhere.

## Cost discipline

Roughly 3× cheaper per run, same run count:

1. **Branch from checkpoints.** Train flat walking once, reuse forever. 15 h → 8 h.
2. **Screen cheap.** 1024 envs, short episodes, flat+mild terrain for reward sweeps.
   Promote only winners to full scale.
3. **Pre-flight every long run.** `python scripts/preflight.py` — 6 minutes to
   catch mistakes that otherwise burn a night.
4. **Kill early, leniently.** Default thresholds are forgiving; `--strict-kill`
   only once a config has proven itself.
5. **Measure H100 vs A100 steps/sec once.** Isaac Lab at 4096 envs is often
   physics-bound, not GPU-bound. If the gap is <25%, use A100 and halve the bill.
6. **Spot instances** with ~10 min checkpointing — but not for the final run.

## Usage

```bash
python scripts/preflight.py --config configs/g1_rough.yaml
python scripts/preflight.py --test reward-audit   # after any reward change
```
