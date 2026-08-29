# Working agreement

## Build sequentially, not in parallel

Grow this codebase one link at a time. Each change should depend on the
previous one being finished and understood, so that when something breaks
there is exactly one candidate for what broke it. The commit history already
works this way. "Run 4: stop penalizing the only two joints that can yaw",
"Run 5: raise arm stiffness 40 -> 3000": one variable per run, with the
outcome named in the message. Do not mine them for settings — copy the shape, not the
numbers.

Parallelism is worth having, but only where it costs nothing in clarity.
Independent reads, independent searches, independent file edits that touch
disjoint files: run those together. The line is ambiguity. The moment two
strands of work could plausibly explain the same result, or the plan needs a
diagram to say what happens when, they were never independent and should not
have been split.

Concretely: do not restructure the package while also changing reward terms.
Do not add a new environment while tuning an existing one. Do not open three
speculative branches of an idea and pick a winner later — pick first, on
argument, then build the one.

## Plans

A plan is a numbered list of steps where step N+1 is meaningless without step
N having landed. If the steps can be shuffled without changing the outcome,
that is a checklist, not a plan, and it should be stated as one so nobody
reads sequence into it.

State what "done" looks like for each step before starting it. For training
changes that means the metric you expect to move and the direction; for
refactors it means the command that must still run.

## Scope

Finish what was asked, then stop. If you find a second problem while fixing
the first, say so and leave it. When a change turns out to need a
prerequisite, do the prerequisite as its own step and its own commit rather
than smuggling it into the current one.

## Robotics

Measure the robot, do not copy its numbers. Every constant that describes
this G1 came out of MuJoCo and is written down where it is used: 30.32 kg
total, 0.784 m standing height measured from the lowest foot geom, arms at
20.1% of mass. An earlier attempt used 1.05 m for standing height because it
was in NVIDIA's 37-DOF config, which spawned this robot 27 cm in the air and
made it land on every reset. When you need a physical quantity, get it from
`scripts/inspect_model.py` and cite the number in a comment next to the
constant.

Gains do not transfer across simulators, because the same numbers are not the
same controller. unitree_rl_gym publishes hip 100 / knee 150 / ankle 40, which
assume an implicit actuator; a MuJoCo position servo applies the PD law
differently, and under those values the knee commands 45 Nm against
a 139 Nm limit while folding 1.44 rad past its target. Treat any borrowed
gain, friction, or timestep as a hypothesis until it holds the robot up in
*this* simulator.

Check the physics before the reward. `scripts/inspect_model.py` holds the
nominal pose with no policy, no terrain, and no reward terms, and if the robot
will not stand there then nothing downstream is interpretable. Same reasoning
behind `njmax = 160`: Playground ships 90, the G1 overflowed it 2,580 times in
one run, and an overflowed constraint solver silently *drops contacts* — a
dropped foot contact means nothing holds the robot up that step. A solver
warning is a physics bug, not log noise.

Terminate honestly or the metrics lie. Stock Playground ends an episode only
when the torso passes horizontal, so a robot tipped to 89 degrees, or settled
on its back, keeps collecting reward and MJX's low-iteration solver lets it
sink partway through the floor. `MAX_TILT = 0.5` and `MIN_TORSO_HEIGHT = 0.4`
exist so the policy cannot learn a stable fallen pose. Whenever an
episode-length or reward curve looks too good, suspect the termination
condition first and watch the video second.

Override behaviour in a class, never by monkeypatching a traced function.
Reassigning `env._get_termination` at runtime passes every isolated test and
still does nothing during training, because brax traces `step` through
`jax.jit` and the trace captures the original bound method. Failures under
`jax.jit` are silent and look like bad hyperparameters, so put the change
where tracing has to see it — that is why `StrictJoystick` is a subclass.

Watch the policy, do not only read its curves. `scripts/record.py` renders
offscreen on the pod while training runs, `scripts/pod/pull.sh` brings the clip
back, and `scripts/view.py` replays it locally under `mjpython` because macOS
requires the GUI own the main thread. A reward number cannot tell you the gait
is shuffling; ten seconds of video can.

Kill dead runs early, and know why the threshold is where it is. The asymmetry
in `killswitch.py` is the whole design: a wrongly killed run costs one
restart, a dead run allowed to finish costs about 14 GPU-hours. The aggressive
checks stay behind `strict: true` because a new reward config has no baseline
curve and velocity tracking can sit flat and still recover. Turn strict on only
for a config you have already seen work.

## Maintaining this file

Anthropic's own guidance is that brevity is the binding constraint: "keep it
short and human-readable," because "bloated CLAUDE.md files cause Claude to
ignore your actual instructions"
(https://code.claude.com/docs/en/best-practices). The pruning test they give
is the one to apply here — for each line, ask whether removing it would cause
a mistake, and cut it if not. Their exclude list is worth honouring verbatim:
no standard language conventions, no file-by-file tours of the codebase, no
self-evident advice like "write clean code," and no detailed API docs when a
link will do. Anything Claude can learn by reading the code does not belong.

That guidance conflicts with the other well-known model, and the conflict is
real rather than a matter of taste. Julien Barbier's widely-copied
CLAUDE.md (https://github.com/jbarbier/CLAUDE.md) runs to thousands of words
of protocol under headings marked "non-negotiable and must never be removed."
Side with Anthropic here. This file is already long enough that a rule added
carelessly makes the existing rules likelier to be dropped, so every addition
should displace something rather than accumulate.

Two ideas from Barbier's file survive that judgement and are worth keeping.
The first is his sizing rule: "When torn between two sizes, pick the smaller
one and say so. Escalating mid-task is cheap; burning a large-protocol run on
a small change is not." The second is the understanding gate — "You can
outsource the typing. You cannot outsource the understanding" — which matters
more here than in most repos, because a training run that looks fine on its
reward curve and is secretly farming a fallen pose will pass every check
nobody thought to look at.

Emphasis is scarce, so spend it. Anthropic's point is that marking one line
IMPORTANT works and marking ten means none of them stands out. If a rule here
must hold with zero exceptions, it does not belong in prose at all — a hook
enforces deterministically what this file can only advise. When something goes
wrong, treat this file as a suspect: a rule being ignored usually means the
file has grown too long, and a question already answered here usually means
the answer is phrased ambiguously.

## Layout

`himalaya/` holds library code, `scripts/` holds entry points, `runs/` holds
per-run checkpoints and `metrics.json`. `himalaya/mjx/g1_model.py` builds the
MJCF; `himalaya/utils/killswitch.py` decides when a run is dead. Entry points
are `scripts/train.py` (train), `scripts/record.py` (render to MP4 on the pod),
`scripts/view.py` (local viewer, needs `mjpython`), and
`scripts/inspect_model.py` (does the robot stand at all). Pod sync lives in
`scripts/pod/`, with host details in `pod.env` so only one file changes per pod.

This is a MuJoCo-only repo. There is one simulator backend and one training
stack; do not add a second "for reference". Delete rather than shelve — git
history is the archive.
