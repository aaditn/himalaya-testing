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
