#!/usr/bin/env bash
# Push the repo to the training pod and verify the env imports there.
#
#   ./scripts/pod/deploy.sh
#
# Sends himalaya/ and scripts/ only -- runs/, videos/, .venv and .git stay
# local. The pod keeps its own venv (/workspace/venv) with a CUDA jax; this
# script never touches it.
#
# -rlptz not -a: the pod's container filesystem refuses chown, and -a implies
# -o/-g, which makes rsync exit nonzero on every file despite transferring it.
#
# Pod details live in scripts/pod/pod.env so only one file changes per pod.
set -euo pipefail
cd "$(dirname "$0")/../.."

[ -f scripts/pod/pod.env ] || { echo "missing scripts/pod/pod.env -- see pod.env.example"; exit 1; }
source scripts/pod/pod.env

SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 -p $POD_PORT -i $POD_KEY"
DEST=/workspace/himalaya_proj

echo "pushing code -> $POD_USER@$POD_HOST:$DEST"
# --delete so a file removed locally does not linger on the pod and get
# imported by accident. Excludes keep credentials and local outputs off it.
rsync -rlptz --delete \
  -e "$SSH" \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'pod.env' \
  himalaya scripts README.md \
  "$POD_USER@$POD_HOST:$DEST/"

echo "verifying the env imports on the pod..."
$SSH "$POD_USER@$POD_HOST" "cd $DEST && /workspace/venv/bin/python -c '
import jax
from himalaya.env import Joystick, default_config
cfg = default_config()
print(\"  jax\", jax.__version__, jax.default_backend(), jax.devices())
print(\"  reward terms:\", len(cfg.reward_config.scales))
env = Joystick(task=\"flat_terrain\", config=cfg)
print(\"  env OK: obs\", env.observation_size, \"act\", env.action_size)
print(\"  strict termination: MAX_TILT\", env.MAX_TILT, \"MIN_TORSO_HEIGHT\", env.MIN_TORSO_HEIGHT)
'"

cat <<USAGE

Deployed. Train (setsid+disown so it survives the ssh session):

  $SSH $POD_USER@$POD_HOST
  cd $DEST && setsid nohup /workspace/venv/bin/python scripts/train.py \\
      --name walk1 --timesteps 60000000 \\
      > /workspace/walk1.log 2>&1 < /dev/null & disown

  tail -f /workspace/walk1.log      # watch it
  ./scripts/pod/pull.sh             # bring back videos + metrics
USAGE
