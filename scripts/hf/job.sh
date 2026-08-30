#!/usr/bin/env bash
# Runs INSIDE the HF Job container. Launched by scripts/hf/launch.sh -- not
# meant to be run locally.
#
# The container starts as a bare python:3.12 image, so everything the pod's
# /workspace/venv already has must be installed here on every run. That costs
# ~3-4 min of GPU time; it is the price of not maintaining a custom image.
#
# Two mounts, set up by launch.sh:
#   /code   the repo, read-only (himalaya/ + scripts/)
#   /out    an HF bucket, read-write -- the ONLY thing that survives the job
set -euo pipefail

CODE=${CODE:-/code}
OUT=${OUT:-/out}
NAME=${NAME:-run}
TIMESTEPS=${TIMESTEPS:-200000000}
ENVS=${ENVS:-8192}
EXTRA=${EXTRA:-}
SCRIPT=${SCRIPT:-scripts/train.py}

# All three pinned, not floating.
#
# jax==0.9.2 is a HARD upper bound, not a preference. brax's PPO trainer calls
# jax.device_put_replicated (ppo/train.py:756), which jax deprecated in 0.8.1
# and REMOVED in 0.10.0 -- it raises AttributeError on access. brax 0.14.2 is
# the newest brax and still calls it, while declaring only `jax>=0.4.6`, so a
# floating install resolves to a combination that cannot run. Verified the hard
# way: the first smoke job reached brax's train() and died there.
# 0.9.2 (2026-03-18) is the last jax that has it, and it sits three days from
# brax 0.14.2 (03-15) and playground 0.2.0 (03-16).
#
# playground==0.2.0 was current when himalaya/env/ was vendored (2026-08); the
# vendored files import mujoco_playground._src.mjx_env, a private module a
# newer Playground could move out from under them.
pip install --quiet --no-input "jax[cuda12]==0.9.2" "playground==0.2.0"

echo "=== jax ==="
python - <<'PY'
import jax
print("jax", jax.__version__, jax.default_backend(), jax.devices(), flush=True)
if jax.default_backend() != "gpu":
    raise SystemExit("jax is not on the GPU -- refusing to burn GPU time on CPU rollouts")
PY

# Upstream's documented trigger for the Menagerie asset download (playground
# README, "From Source" step 7). Doubles as a control: if stock G1 fails to
# load, the problem is the image, not himalaya/env/.
echo "=== menagerie + stock G1 control ==="
python -c "from mujoco_playground import locomotion; locomotion.load('G1JoystickFlatTerrain'); print('stock G1 loads OK', flush=True)"

echo "=== our envs ==="
PYTHONPATH="$CODE" python - <<'PY'
from himalaya.env import Joystick, default_config
cfg = default_config()
env = Joystick(task="flat_terrain", config=cfg)
print("  reward terms:", len(cfg.reward_config.scales))
print("  obs", env.observation_size, "act", env.action_size)
print("  strict termination: MAX_TILT", env.MAX_TILT,
      "MIN_TORSO_HEIGHT", env.MIN_TORSO_HEIGHT, flush=True)

from himalaya.env.climb import Climb, default_config as climb_config
ccfg = climb_config()
cenv = Climb(task="incline", config=ccfg)
print("  climb reward terms:", len(ccfg.reward_config.scales))
print("  climb obs", cenv.observation_size, "act", cenv.action_size)
print("  climb termination: MIN_CLIMB_HEIGHT", cenv.MIN_CLIMB_HEIGHT,
      "MAX_CLIMB_ROLL", cenv.MAX_CLIMB_ROLL, flush=True)
PY

# train.py writes runs/<name>/ relative to the working directory, so cd'ing
# into the bucket is all it takes to make checkpoints and metrics survive.
mkdir -p "$OUT"
cd "$OUT"
echo "=== train: $SCRIPT name=$NAME timesteps=$TIMESTEPS envs=$ENVS $EXTRA ==="
exec python "$CODE/$SCRIPT" \
    --name "$NAME" --timesteps "$TIMESTEPS" --envs "$ENVS" $EXTRA
