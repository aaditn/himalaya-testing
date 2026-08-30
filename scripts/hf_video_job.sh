#!/usr/bin/env bash
set -Eeuo pipefail

: "${HF_REPO_ID:?HF_REPO_ID must be set}"
: "${CHECKPOINT_ROOT:?CHECKPOINT_ROOT must point to a downloaded checkpoint folder}"
: "${IMAGE_REF:?IMAGE_REF must identify the Dockerfile.hf runtime}"

test -f /opt/himalaya-image/menagerie-ready
test "$(cat /opt/himalaya-image/provenance)" = "Dockerfile.hf"
python scripts/verify_training_launch_contract.py --image "${IMAGE_REF}"

VIDEO_OUTPUT="${VIDEO_OUTPUT:-/workspace/videos/latest}"
REMOTE_VIDEO_PATH="${REMOTE_VIDEO_PATH:-videos/latest}"

uv pip install --python "$(command -v python)" --no-deps --editable .
python -c "import jax; assert jax.__version__ == '0.6.2', jax.__version__"

checkpoint="$(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d \
  -printf '%f\n' | sort -n | tail -n 1)"
if [[ -z "${checkpoint}" ]]; then
  echo "No numeric checkpoint exists under ${CHECKPOINT_ROOT}" >&2
  exit 3
fi

export MUJOCO_GL=egl
python scripts/render_policy.py \
  --checkpoint "${CHECKPOINT_ROOT}/${checkpoint}" \
  --output "${VIDEO_OUTPUT}" \
  --slopes 0 5 10 15 \
  --seconds 20

hf upload "${HF_REPO_ID}" "${VIDEO_OUTPUT}" "${REMOTE_VIDEO_PATH}" \
  --repo-type model \
  --commit-message "Add G1 uphill checkpoint rollout videos"

