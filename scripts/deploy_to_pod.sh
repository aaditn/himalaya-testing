#!/usr/bin/env bash
# Install the Himalaya configs into an Isaac Lab pod.
#
# Why they go INSIDE isaaclab rather than staying an external package:
# train.py decorates its main with @hydra_task_config(args_cli.task), which
# resolves the task id at IMPORT time -- before any external module could
# register anything. An out-of-tree package always loses that race
# (NameNotFound). Dropping the config into the g1 package lets Isaac Lab's
# own discovery register it.
#
# Usage: ./scripts/deploy_to_pod.sh root@<ip> -p <port>
set -euo pipefail

HOST="$1"; shift
SSH_OPTS="$*"
G1DIR="/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1"

echo "Deploying env config..."
sed 's#from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import#from .rough_env_cfg import#' \
  himalaya/tasks/himalaya_env_cfg.py | \
  ssh $SSH_OPTS "$HOST" "cat > $G1DIR/himalaya_env_cfg.py"

echo "Appending gym registrations (idempotent)..."
cat himalaya/tasks/registrations.py.frag | \
  ssh $SSH_OPTS "$HOST" "grep -q 'Himalaya-G1-Teacher-v0' $G1DIR/__init__.py || cat >> $G1DIR/__init__.py"

echo "Deploying assets..."
tar czf - assets/g1 | ssh $SSH_OPTS "$HOST" "mkdir -p /workspace/himalaya_proj && tar xzf - -C /workspace/himalaya_proj" 2>/dev/null || true

echo "Deploying droptest + installing tmux..."
cat scripts/droptest.py | ssh $SSH_OPTS "$HOST" \
  "mkdir -p /workspace/bench && cat > /workspace/bench/droptest.py; \
   command -v tmux >/dev/null || (apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq tmux >/dev/null 2>&1)"

echo "Deploying viewer launcher..."
cat scripts/viewer.sh | ssh $SSH_OPTS "$HOST" "cat > /workspace/viewer.sh; chmod +x /workspace/viewer.sh"

echo "Registered tasks:"
ssh $SSH_OPTS "$HOST" "grep -oE 'id=\"Himalaya[^\"]*\"' $G1DIR/__init__.py"

cat <<'USAGE'

NOTE: a new pod on a NEW network volume starts empty -- Isaac Lab ships in the
image but nothing else carries over. Run this script, then convert the URDF.

Convert the URDF once (required for every Himalaya-G1-23DOF-* task):
  /isaac-sim/python.sh scripts/tools/convert_urdf.py \
    /workspace/himalaya_proj/assets/g1/g1_23dof.urdf \
    /workspace/himalaya_proj/assets/g1/g1_23dof.usd \
    --headless --joint-stiffness 0 --joint-damping 0

Watch it (in the noVNC desktop at port 6901, password "isaac"):
  /workspace/viewer.sh drop      # does the robot stand on flat ground?
  /workspace/viewer.sh play1     # known-good stock G1, for comparison
  /workspace/viewer.sh attach    # reattach after a VNC drop

Train (setsid+disown so it survives the ssh session):
  cd /workspace/isaaclab && setsid nohup /isaac-sim/python.sh \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task Himalaya-G1-23DOF-v0 --headless --num_envs 4096 \
    --max_iterations 1500 > /workspace/bench/run.log 2>&1 < /dev/null & disown
USAGE
