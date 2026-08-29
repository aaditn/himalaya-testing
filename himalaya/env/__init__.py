"""The G1 joystick environment, vendored so it is ours to edit.

Playground's own env lives inside site-packages, which makes every reward or
observation change a subclass or a monkeypatch. Monkeypatching is worse than
it looks here: brax traces `step` through jax.jit, and a patch applied after
tracing is silently ignored during training while still passing every isolated
test. Owning the file removes that whole class of bug -- change the reward and
it is simply changed.

Vendored from MuJoCo Playground under Apache-2.0 (see LICENSE.playground):
  base.py           MJX env base for the G1
  g1_constants.py   sites, geoms, sensors, joint ranges, XML paths
  joystick.py       the task: rewards, observations, termination, commands
  randomize.py      domain randomization

`mjx_env` and `gait` are still imported from Playground. They are generic
infrastructure rather than G1-specific, so there is nothing to gain by owning
them.
"""

from himalaya.env.joystick import Joystick, default_config  # noqa: F401
