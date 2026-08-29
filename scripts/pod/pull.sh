#!/usr/bin/env bash
# Pull videos + checkpoints from the training pod to this Mac.
#
#   ./scripts/pod/pull.sh              # videos + metrics, opens the newest clip
#   ./scripts/pod/pull.sh --policies   # also pull checkpoints, for the local viewer
#   ./scripts/pod/pull.sh --no-open    # pull but don't open anything
#
# Pod details live in scripts/pod/pod.env so only one file changes per pod.
set -euo pipefail
cd "$(dirname "$0")/../.."

[ -f scripts/pod/pod.env ] || { echo "missing scripts/pod/pod.env -- see pod.env.example"; exit 1; }
source scripts/pod/pod.env

SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 -p $POD_PORT -i $POD_KEY"
mkdir -p videos runs

echo "pulling videos..."
$SSH "$POD_USER@$POD_HOST" 'cd /workspace && tar cf - videos 2>/dev/null' | tar xf - 2>/dev/null || true

echo "pulling metrics..."
$SSH "$POD_USER@$POD_HOST" 'cd /workspace && tar cf - $(find runs -name "metrics.json" 2>/dev/null) 2>/dev/null' | tar xf - 2>/dev/null || true

OPEN=1
for a in "$@"; do [ "$a" = "--no-open" ] && OPEN=0; done

if [[ " $* " == *" --policies "* ]]; then
  echo "pulling checkpoints..."
  $SSH "$POD_USER@$POD_HOST" 'cd /workspace && tar cf - $(find runs -name "policy*" 2>/dev/null) 2>/dev/null' | tar xf - 2>/dev/null || true
fi

echo
echo "videos:"
ls -lh videos/*.mp4 2>/dev/null | awk '{print "  ", $9, $5}' || echo "   (none yet)"
echo "runs:"
for f in runs/*/metrics.json; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY'
import json,sys
rows=json.load(open(sys.argv[1]))
if rows:
    r=rows[-1]
    print(f"   {sys.argv[1].split('/')[1]:<22} step={r['step']:>11,} "
          f"reward={r['reward']:7.2f} len={r['episode_len']:6.1f}")
PY
done

# Newest by mtime, not by name -- mid-training snapshots then surface as they
# arrive rather than whatever sorts last alphabetically.
LATEST=$(ls -t videos/*.mp4 2>/dev/null | head -1)
if [ -n "$LATEST" ] && [ "$OPEN" = "1" ]; then
  echo
  echo "opening $LATEST"
  open "$LATEST"
fi
