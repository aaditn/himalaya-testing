#!/usr/bin/env bash
# Launch an Isaac Lab sim inside tmux so it survives a dropped VNC session.
#
# The problem this solves: noVNC sessions drop (browser tab closed, network
# blip, idle timeout). A sim launched from a plain VNC terminal dies with the
# session. Inside tmux it keeps running -- reconnect and reattach.
#
# It also keeps the sim OFF pid 1: killing a stray GPU process on this pod
# once took the whole container down with it, because the process being
# killed was the container's main process. Everything here runs as a tmux
# child, so `viewer.sh kill` only ever kills the sim.
#
# Usage (run inside the VNC desktop's terminal):
#   ./viewer.sh drop            # 23-DOF robot, flat ground, zero action
#   ./viewer.sh play            # 23-DOF on real terrain, 16 envs
#   ./viewer.sh play1           # stock G1 (run 1, known-good) for comparison
#   ./viewer.sh run <task>      # any registered task
#   ./viewer.sh attach          # reattach after a VNC drop
#   ./viewer.sh kill            # stop the sim, leave the container alone
#   ./viewer.sh status          # what's running, GPU state
set -uo pipefail

SESSION="isaacview"
LAB="/workspace/isaaclab"
BENCH="/workspace/bench"
PY="$LAB/isaaclab.sh -p"

need_tmux() {
  command -v tmux >/dev/null 2>&1 || {
    echo "tmux not installed. Installing..."
    apt-get update -qq && apt-get install -y -qq tmux
  }
}

launch() {
  local name="$1"; shift
  need_tmux
  tmux kill-session -t "$SESSION" 2>/dev/null
  # cd first: isaaclab.sh resolves paths relative to the repo root.
  tmux new-session -d -s "$SESSION" -n "$name" \
    "cd $LAB && $PY $* 2>&1 | tee $BENCH/viewer_$name.log; echo; echo '--- exited, press any key ---'; read -n 1"
  echo "Launched '$name' in tmux session '$SESSION'."
  echo "  attach:  ./viewer.sh attach     (detach again with Ctrl-b then d)"
  echo "  log:     tail -f $BENCH/viewer_$name.log"
  echo
  echo "Window opens in ~60s (Isaac Sim loads USD assets and compiles shaders)."
}

case "${1:-}" in
  drop)
    # No policy, no terrain, no rewards -- just: does this robot stand up?
    # If it fails HERE, the asset is broken and no reward tuning will help.
    launch drop "$BENCH/droptest.py"
    ;;
  play)
    launch play23 "$LAB/scripts/reinforcement_learning/rsl_rl/train.py \
      --task Himalaya-G1-23DOF-Play-v0 --num_envs 16"
    ;;
  play1)
    # Run 1's robot and config, which trained successfully to +29.57 reward.
    # Use as the visual reference for what "working" looks like.
    launch play1 "$LAB/scripts/reinforcement_learning/rsl_rl/train.py \
      --task Isaac-Velocity-Rough-G1-Play-v0 --num_envs 16"
    ;;
  run)
    shift
    [ $# -ge 1 ] || { echo "usage: ./viewer.sh run <Task-Id> [extra args]"; exit 1; }
    task="$1"; shift
    launch "$task" "$LAB/scripts/reinforcement_learning/rsl_rl/train.py \
      --task $task --num_envs ${1:-16}"
    ;;
  attach)
    tmux attach -t "$SESSION" || echo "No session '$SESSION'. Start one first."
    ;;
  kill)
    # Kill ONLY the tmux session's processes. Never scrape nvidia-smi for pids
    # here -- that is how the container got taken down before.
    tmux kill-session -t "$SESSION" 2>/dev/null && echo "Killed '$SESSION'." \
      || echo "No session '$SESSION'."
    ;;
  status)
    echo "=== tmux ==="
    tmux ls 2>/dev/null || echo "  (no sessions)"
    echo "=== gpu ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader
    echo "=== gpu processes ==="
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
      || echo "  (none)"
    ;;
  *)
    sed -n '2,20p' "$0" | sed 's/^#\{1,\} \{0,1\}//'
    exit 1
    ;;
esac
