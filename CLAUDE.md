# Working agreement

## Use upstream, do not rebuild it

Local copies of the upstream docs live in `docs/` — read them before writing
anything that upstream might already provide:

```
docs/playground/notebooks/locomotion.ipynb   the official training recipe
docs/playground/learning/train_jax_ppo.py    DeepMind's trainer (installed as `train-jax-ppo`)
docs/playground/README.md                    install, CLI, env registry
docs/menagerie/unitree_g1_README.md          the robot model
docs/mjx/README.md                           solver, njmax, warp vs jax backends
docs/brax_README.md                          the PPO implementation
```

This project has already lost a day to rebuilding what shipped in the box. A
hand-written training loop omitted `randomization_fn`, which the notebook's
cell 46 passes via `registry.get_domain_randomizer(env_name)`, and the policy
overfit to one fixed set of physics. Before writing a trainer, an eval loop, or
a config, grep `docs/` and the installed package for it.

Tuned hyperparameters live in `mujoco_playground/config/locomotion_params.py`,
keyed per environment. G1 gets `entropy_cost=0.005`, 200M timesteps, and
asymmetric actor-critic (`policy_obs_key="state"`,
`value_obs_key="privileged_state"`). Do not invent values that are already
tuned upstream.

There are no pretrained weights. Playground ships recipes, not checkpoints —
verified by searching the package. Training is unavoidable.

## Verify with video, not curves

IMPORTANT: every change to the environment, the reward, or the gains ships with
a video. Record with `scripts/record.py`, pull it with `scripts/pod/pull.sh`,
and report the clip's path in the same message that reports the change — no
exceptions, and no substituting a reward curve for it. A change that cannot be
recorded yet is reported as unverified, in those words, rather than as done.

The reason is concrete. A run reported reward -4.8 → +17.5 and episode length
38 → 787, all real numbers, and the video showed the robot falling eight times
in ten seconds. Training eval samples random commands with 10% zero; a fixed
forward command is a different and harder condition. The curve was not lying,
it was answering a different question.

Watch what the policy does before theorising about why. Three consecutive wrong
diagnoses in one session — `njmax`, then the solver backend, then "it falls
through the floor" — all came from reasoning about numbers instead of looking
at the render.

## Build sequentially, not in parallel

Grow this codebase one link at a time. Each change should depend on the
previous one being finished and understood, so that when something breaks
there is exactly one candidate for what broke it.

Parallelism is worth having where it costs nothing in clarity: independent
reads, independent searches, edits to disjoint files. The line is ambiguity.
The moment two strands of work could plausibly explain the same result, they
were never independent.

Concretely: do not restructure the package while also changing reward terms.
Do not add a new environment while tuning an existing one.

## Change one variable, and keep a control

When a result is wrong, the question is whether it is wrong because of us. Run
the stock upstream configuration unmodified, on the same hardware with the same
trainer, and compare. That is how `njmax` was cleared: stock Playground logs
`nefc overflow - please increase njmax to 94` against its own default of 90,
1,770 times in two minutes, so raising it to 160 was fixing a real bug rather
than causing one.

## Plans

A plan is a numbered list of steps where step N+1 is meaningless without step N
having landed. If the steps can be shuffled without changing the outcome, that
is a checklist, not a plan, and it should be stated as one.

State what "done" looks like before starting. For training changes that means
the metric you expect to move and the direction; for refactors, the command
that must still run.

## Scope

Finish what was asked, then stop. If you find a second problem while fixing the
first, say so and leave it. When a change needs a prerequisite, do the
prerequisite as its own step and its own commit.

## Robotics

Measure the robot, do not copy its numbers. Every constant describing this G1
came out of MuJoCo and is written where it is used: 33.34 kg, 0.784 m standing
height from Menagerie's keyframe, 29 actuated joints. An earlier attempt used
1.05 m because it was in NVIDIA's 37-DOF config, which spawned the robot 27 cm
in the air so it landed on every reset.

Check the physics before the reward, and on the right scene. Menagerie ships
two: `scene_mjx.xml` (5 solver iterations, simplified colliders) is for batched
training where a policy closes the loop every step, and it will NOT hold a pose
open-loop — the robot drops to z=0.11 in 1.5 s, a solver artifact rather than a
broken model. `scene.xml` (100 iterations) stands at z=0.792.
`scripts/inspect_model.py` uses the latter deliberately; never quote a number
measured on the MJX scene as a fact about the robot.

Terminate honestly or the metrics lie. Stock Playground ends an episode only
when the torso passes horizontal, so a robot tipped to 89 degrees keeps
collecting reward while MJX's solver lets it sink through the floor.
`MAX_TILT = 0.5` and `MIN_TORSO_HEIGHT = 0.4` in `himalaya/env/joystick.py`
exist so the policy cannot learn a stable fallen pose. Both are ours, so both
are suspects when episodes end early — check whether a normal gait dips below
the height threshold before blaming the policy.

Edit the environment, do not patch it at runtime. `himalaya/env/` is a vendored
copy of Playground's G1 task precisely so rewards, observations, and
termination are ordinary edits. Reassigning a method like
`env._get_termination` passes every isolated test and can still do nothing
during training, because brax traces `step` through `jax.jit` and captures the
original bound method. Mark every change to a vendored file with a `MODIFIED:`
comment naming the stock behaviour.

Kill dead runs early. The asymmetry in `killswitch.py` is the design: a wrongly
killed run costs one restart, a dead run allowed to finish costs about 14
GPU-hours. Aggressive checks stay behind `strict: true`, for configs already
seen to work.

## Layout

`himalaya/` holds library code, `scripts/` holds entry points, `docs/` holds
vendored upstream documentation, `runs/` holds per-run checkpoints and
`metrics.json`. `himalaya/env/` is the task, vendored from Playground and ours
to edit; `himalaya/mjx/g1_29dof.py` loads the robot from Menagerie;
`himalaya/utils/killswitch.py` decides when a run is dead. Pod sync lives in
`scripts/pod/`, with host details in `pod.env` so only one file changes per pod.

This is a MuJoCo-only repo. One simulator backend, one training stack; do not
add a second "for reference". Delete rather than shelve — git history is the
archive.
