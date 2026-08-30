#!/usr/bin/env bash
# Launch a training run on HF Jobs.
#
#   ./scripts/hf/launch.sh smoke                 # cheap 5M-step shakedown
#   ./scripts/hf/launch.sh walk1 60000000        # a real run
#   FLAVOR=a100-large ./scripts/hf/launch.sh walk1 200000000
#
# Results land in the bucket, not on the job's disk, which is wiped on exit:
#   hf://buckets/$BUCKET/runs/<name>/{metrics.json,policy}
#
# Pull them with ./scripts/hf/pull.sh <name>.
set -euo pipefail
cd "$(dirname "$0")/../.."

NAME=${1:?usage: launch.sh <name> [timesteps] [extra train.py args]}
TIMESTEPS=${2:-200000000}
shift 2 2>/dev/null || shift 1 2>/dev/null || true
EXTRA="$*"

# The personal namespace has no pre-paid credits (jobs 402); iteratehack does.
NAMESPACE=${NAMESPACE:-iteratehack}
BUCKET=${BUCKET:-$NAMESPACE/himalaya-runs}
FLAVOR=${FLAVOR:-l40sx1}
TIMEOUT=${TIMEOUT:-6h}
IMAGE=${IMAGE:-python:3.12}

# -v takes a single directory with no exclude support, so stage the two dirs
# that matter rather than syncing .git (44 MB) and docs/ to the Hub.
STAGE=.hfstage
rm -rf "$STAGE" && mkdir -p "$STAGE"
rsync -rlpt --exclude '__pycache__' --exclude '*.pyc' \
    himalaya scripts "$STAGE/"

echo "launching $NAME on $FLAVOR ($TIMESTEPS steps, timeout $TIMEOUT)"
hf jobs run \
    --namespace "$NAMESPACE" \
    --flavor "$FLAVOR" \
    --timeout "$TIMEOUT" \
    --name "himalaya-$NAME" \
    --label "project=himalaya" \
    -e "NAME=$NAME" \
    -e "TIMESTEPS=$TIMESTEPS" \
    -e "ENVS=${ENVS:-8192}" \
    -e "EXTRA=$EXTRA" \
    -e "SCRIPT=${SCRIPT:-scripts/train.py}" \
    -v "./$STAGE:/code:ro" \
    -v "hf://buckets/$BUCKET:/out:rw" \
    --detach \
    "$IMAGE" \
    bash /code/scripts/hf/job.sh
