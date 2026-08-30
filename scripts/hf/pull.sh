#!/usr/bin/env bash
# Bring a run's checkpoints and metrics back from the bucket.
#
#   ./scripts/hf/pull.sh smoke        -> runs/smoke/
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME=${1:?usage: pull.sh <run-name>}
BUCKET=${BUCKET:-${NAMESPACE:-iteratehack}/himalaya-runs}
mkdir -p "runs/$NAME"
hf buckets sync "hf://buckets/$BUCKET/runs/$NAME" "runs/$NAME"
ls -la "runs/$NAME"
