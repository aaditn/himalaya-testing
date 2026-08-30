#!/usr/bin/env bash
set -Eeuo pipefail

# Hugging Face GPU jobs are headless.  Force MuJoCo/PyOpenGL onto EGL before
# either package is imported so smoke and final video rendering use the GPU.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

: "${HF_REPO_ID:?HF_REPO_ID must be set}"
: "${IMAGE_REF:?IMAGE_REF must be set to an immutable image digest}"
: "${SOURCE_REVISION:?SOURCE_REVISION must be set}"
: "${SOURCE_DIGEST:?SOURCE_DIGEST must be set}"
: "${RUNTIME_DIGEST:?RUNTIME_DIGEST must be set}"
: "${RUN_ID:?RUN_ID must be set}"
: "${REMOTE_OUTPUT_PATH:?REMOTE_OUTPUT_PATH must be set}"

JOB_MODE="${JOB_MODE:-smoke}"
TRAINING_TIMESTEPS_30="${TRAINING_TIMESTEPS_30:-100000000}"
NUM_ENVS="${NUM_ENVS:-8192}"
VALIDATION_TRIALS="${VALIDATION_TRIALS:-64}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/output}"
SYNC_SECONDS="${SYNC_SECONDS:-300}"
SMOKE_TIMESTEPS="${SMOKE_TIMESTEPS:-512}"
SMOKE_NUM_ENVS="${SMOKE_NUM_ENVS:-16}"

case "${JOB_MODE}" in
  smoke|real) ;;
  *) echo "JOB_MODE must be smoke or real" >&2; exit 2 ;;
esac
mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/job.log") 2>&1

training_pid=""
sync_pid=""
final_sync_done=0

sync_output_strict() {
  hf upload "${HF_REPO_ID}" "${OUTPUT_DIR}" "${REMOTE_OUTPUT_PATH}" \
    --repo-type model --commit-message "Sync ${JOB_MODE} run ${RUN_ID}"
}

sync_output_best_effort() {
  if ! sync_output_strict; then
    echo "WARNING: best-effort artifact sync failed" >&2
  fi
}

cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "${sync_pid}" ]]; then
    kill "${sync_pid}" 2>/dev/null || true
  fi
  if [[ -n "${training_pid}" ]]; then
    kill "${training_pid}" 2>/dev/null || true
  fi
  if [[ "${final_sync_done}" != "1" ]]; then
    python scripts/write_run_manifest.py \
      --output "${OUTPUT_DIR}/run_manifest.json" \
      --status failed --exit-code "${status}" || true
    sync_output_best_effort
  fi
  exit "${status}"
}
trap 'exit 130' INT
trap 'exit 143' TERM
trap cleanup EXIT

# Every paid job must start from the runtime produced by Dockerfile.hf.  There
# is intentionally no in-job dependency fallback: a wrong image fails fast
# instead of spending H200 time rebuilding an ephemeral container.
test -f /opt/himalaya-image/menagerie-ready
test "$(cat /opt/himalaya-image/provenance)" = "Dockerfile.hf"
python scripts/verify_training_launch_contract.py --image "${IMAGE_REF}"

# Install only the immutable reviewed source after dependencies are fixed.
uv pip install --python "$(command -v python)" --no-deps --editable .
python scripts/verify_hf_image.py --output "${OUTPUT_DIR}/image_verification.json"
python scripts/runtime_fingerprint.py --root . \
  --output "${OUTPUT_DIR}/runtime_manifest.json" --quiet
python scripts/write_run_manifest.py --output "${OUTPUT_DIR}/run_manifest.json"

if [[ "${JOB_MODE}" == "smoke" ]]; then
  : "${SMOKE_GATE_PATH:?SMOKE_GATE_PATH must be set for a smoke job}"
  python scripts/preflight_four_contact.py \
    --static-only \
    --output "${OUTPUT_DIR}/preflight_manifest.json"
  python scripts/smoke_four_contact.py \
    --output "${OUTPUT_DIR}" \
    --source-revision "${SOURCE_REVISION}" \
    --source-digest "${SOURCE_DIGEST}" \
    --runtime-digest "${RUNTIME_DIGEST}" \
    --image-ref "${IMAGE_REF}" \
    --preflight-manifest "${OUTPUT_DIR}/preflight_manifest.json" \
    --slope 30 \
    --timesteps "${SMOKE_TIMESTEPS}" \
    --num-envs "${SMOKE_NUM_ENVS}"
  python scripts/write_run_manifest.py \
    --output "${OUTPUT_DIR}/run_manifest.json" --status completed --exit-code 0
  sync_output_strict
  hf upload "${HF_REPO_ID}" "${OUTPUT_DIR}/smoke_pass.json" \
    "${SMOKE_GATE_PATH}" --repo-type model \
    --commit-message "Pass smoke gate for ${SOURCE_REVISION}"
  final_sync_done=1
  exit 0
fi

# Real jobs normally reuse compiled evidence.  An explicit operator override is
# preserved as such and never masquerades as a passing smoke marker.
if [[ "${SKIP_SMOKE_GATE:-0}" == "1" ]]; then
  printf '{"schema_version":1,"passed":false,"explicitly_waived":true}\n' \
    > "${OUTPUT_DIR}/smoke_gate_verification.json"
else
  python scripts/verify_smoke_gate.py \
    --repo-id "${HF_REPO_ID}" --gate-path "${SMOKE_GATE_PATH}" \
    --source-revision "${SOURCE_REVISION}" \
    --runtime-digest "${RUNTIME_DIGEST}" --image-ref "${IMAGE_REF}" \
    --output "${OUTPUT_DIR}/smoke_gate_verification.json"
fi
python scripts/sanity_four_contact.py \
  --output "${OUTPUT_DIR}/sanity_manifest.json"
python scripts/write_human_approval.py \
  --output "${OUTPUT_DIR}/human_audit_approval.json"

run_rough_30_training() {
  python scripts/train_four_contact.py \
    --slope 30 \
    --timesteps "${TRAINING_TIMESTEPS_30}" \
    --num-envs "${NUM_ENVS}" \
    --validation-trials "${VALIDATION_TRIALS}" \
    --promotion-success-rate "${PROMOTION_SUCCESS_RATE_30:-0.80}" \
    --output "${OUTPUT_DIR}/balance_30deg"
}

run_rough_30_training &
training_pid="$!"
(
  while kill -0 "${training_pid}" 2>/dev/null; do
    sleep "${SYNC_SECONDS}"
    sync_output_best_effort
  done
) &
sync_pid="$!"

set +e
wait "${training_pid}"
training_status=$?
set -e
training_pid=""
kill "${sync_pid}" 2>/dev/null || true
wait "${sync_pid}" 2>/dev/null || true
sync_pid=""

# Render every completed nominal checkpoint, including the latest failed stage.
render_status=0
for latest_stage in \
  "${OUTPUT_DIR}/balance_30deg/latest_stage.json"; do
  if [[ ! -f "${latest_stage}" ]]; then
    continue
  fi
  mapfile -t stage_info < <(python scripts/read_latest_stage.py "${latest_stage}")
  checkpoint="${stage_info[0]}"
  render_slope="${stage_info[1]}"
  video="${OUTPUT_DIR}/videos/g1_four_contact_${render_slope}deg.mp4"
  set +e
  MUJOCO_GL=egl python scripts/render_four_contact.py \
    --checkpoint "${checkpoint}" \
    --slope "${render_slope}" \
    --output "${video}" \
    --seconds 20
  current_render_status=$?
  if [[ "${current_render_status}" == "0" ]]; then
    python scripts/verify_video.py "${video}" \
      --output "${OUTPUT_DIR}/videos/g1_four_contact_${render_slope}deg.verification.json"
    current_render_status=$?
  fi
  set -e
  if [[ "${current_render_status}" != "0" ]]; then
    render_status="${current_render_status}"
  fi
done
if [[ ! -f "${OUTPUT_DIR}/balance_30deg/latest_stage.json" ]]; then
  echo "No completed checkpoint is available to render" >&2
  render_status=4
fi

overall_status="${training_status}"
if [[ "${overall_status}" == "0" && "${render_status}" != "0" ]]; then
  overall_status="${render_status}"
fi
if [[ "${overall_status}" == "0" ]]; then
  manifest_status="completed"
else
  manifest_status="failed"
fi
python scripts/write_run_manifest.py \
  --output "${OUTPUT_DIR}/run_manifest.json" \
  --status "${manifest_status}" --exit-code "${overall_status}"

# Unlike periodic syncs, the final artifact upload is a hard requirement.
sync_output_strict
final_sync_done=1
exit "${overall_status}"
