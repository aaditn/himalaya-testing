#!/usr/bin/env bash
set -Eeuo pipefail

: "${HF_REPO_ID:?HF_REPO_ID must be set to the Hugging Face model repository}"
: "${IMAGE_REF:?IMAGE_REF must identify the Dockerfile.hf runtime}"

test -f /opt/himalaya-image/menagerie-ready
test "$(cat /opt/himalaya-image/provenance)" = "Dockerfile.hf"
python scripts/verify_training_launch_contract.py --image "${IMAGE_REF}"

TARGET_SLOPE="${TARGET_SLOPE:-15}"
TIMESTEPS_PER_STAGE="${TIMESTEPS_PER_STAGE:-40000000}"
NUM_ENVS="${NUM_ENVS:-8192}"
VALIDATION_TRIALS="${VALIDATION_TRIALS:-64}"
SYNC_SECONDS="${SYNC_SECONDS:-600}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output}"
REMOTE_OUTPUT_PATH="${REMOTE_OUTPUT_PATH:-runs/g1_uphill_stage1}"

uv pip install --python "$(command -v python)" --no-deps --editable .
python -c "import jax; assert jax.__version__ == '0.6.2', jax.__version__"
python scripts/preflight.py --slope 15 --impl jax

mkdir -p "${OUTPUT_DIR}"

sync_output() {
  if [[ -d "${OUTPUT_DIR}" ]]; then
    hf upload "${HF_REPO_ID}" "${OUTPUT_DIR}" "${REMOTE_OUTPUT_PATH}" \
      --repo-type model \
      --commit-message "Sync G1 uphill training artifacts" || true
  fi
}

training_pid=""
sync_pid=""
cleanup() {
  if [[ -n "${sync_pid}" ]]; then
    kill "${sync_pid}" 2>/dev/null || true
  fi
  sync_output
}
trap cleanup EXIT INT TERM

python scripts/train_uphill.py \
  --target-slope "${TARGET_SLOPE}" \
  --timesteps-per-stage "${TIMESTEPS_PER_STAGE}" \
  --num-envs "${NUM_ENVS}" \
  --validation-trials "${VALIDATION_TRIALS}" \
  --output "${OUTPUT_DIR}" &
training_pid="$!"

(
  while kill -0 "${training_pid}" 2>/dev/null; do
    sleep "${SYNC_SECONDS}"
    sync_output
  done
) &
sync_pid="$!"

wait "${training_pid}"
