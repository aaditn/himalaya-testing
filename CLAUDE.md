# Working agreement

## Build sequentially, not in parallel

Grow this codebase one link at a time. Each change should depend on the
previous one being finished and understood, so that when something breaks
there is exactly one candidate for what broke it. The commit history already
works this way — "Run 4: stop penalizing the only two joints that can yaw",
"Run 5: raise arm stiffness 40 -> 3000" — one variable per run, with the
outcome named in the message. Keep that shape.

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

Prefer a short plan you finish over a long one you abandon halfway. Depth of
the tree matters more than breadth: three steps deep on the thing that
actually blocks progress beats twelve shallow steps across the repo.

## Scope

Finish what was asked, then stop. If you find a second problem while fixing
the first, say so and leave it — a fix that arrives bundled with two unasked
changes is a fix nobody can review or bisect.

When a change turns out to need a prerequisite, name the prerequisite and do
it as its own step rather than smuggling it into the current one. The
prerequisite deserves its own commit and its own line in the history.

## Layout

`himalaya/` holds library code, `scripts/` holds entry points, `runs/` holds
outputs. The simulator backend is MuJoCo MJX via MuJoCo Playground.

Keep this repo minimal. There is one training stack, not two: a second
backend kept "for reference" is a second thing to keep working, and the
ambiguity about which one is live costs more than the code is worth. Delete
rather than shelve — git history is the archive.
