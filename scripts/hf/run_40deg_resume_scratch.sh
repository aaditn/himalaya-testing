#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS_MOUNT="${RUNS_MOUNT:-/runs}"
SCRATCH_RUNS="${SCRATCH_RUNS:-/tmp/himalaya-runs}"
PREFIX="${PREFIX:-g1_45deg_h200_resume_v3}"
TRAIN_ENVS="${TRAIN_ENVS:-8192}"
RESTORE_SOURCE="${RESTORE_SOURCE:-${RUNS_MOUNT}/g1_45deg_curriculum_h200_v2_04_transfer-35deg/checkpoints/000024248320}"
RESTORE_LOCAL="/tmp/himalaya-restore"

python -m pip install --upgrade pip
python -m pip install -e "${ROOT}[cuda]"
python - <<'PY'
import jax

if jax.default_backend() != "gpu":
    raise SystemExit(f"GPU backend required, got {jax.default_backend()}: {jax.devices()}")
print(f"JAX devices: {jax.devices()}", flush=True)
PY

# Orbax creates many small files. Writing them directly through the bucket
# mount caused EIO and disappearing-directory failures in both long runs.
# Copy the restore once, train entirely on local NVMe, and upload compact
# final artifacts only after all stages complete.
mkdir -p "${RESTORE_LOCAL}" "${SCRATCH_RUNS}"
cp -a "${RESTORE_SOURCE}/." "${RESTORE_LOCAL}/"

cd "${ROOT}"
python scripts/train_climb_curriculum.py \
  --curriculum configs/curriculum_resume_40deg.json \
  --runs-dir "${SCRATCH_RUNS}" \
  --prefix "${PREFIX}" \
  --envs "${TRAIN_ENVS}" \
  --restore "${RESTORE_LOCAL}"

for stage_dir in "${SCRATCH_RUNS}/${PREFIX}"_*; do
  destination="${RUNS_MOUNT}/$(basename "${stage_dir}")"
  mkdir -p "${destination}"
  cp "${stage_dir}/policy" "${destination}/policy"
  cp "${stage_dir}/metrics.json" "${destination}/metrics.json"
  cp "${stage_dir}/best_checkpoint.json" "${destination}/best_checkpoint.json"
done

final_stage="$(find "${SCRATCH_RUNS}" -maxdepth 1 -type d -name "${PREFIX}_*" | sort | tail -n 1)"
best_step="$(python -c 'import json,sys; print("%012d" % json.load(open(sys.argv[1]))["step"])' "${final_stage}/best_checkpoint.json")"
tar -C "${final_stage}/checkpoints" -czf \
  "${RUNS_MOUNT}/$(basename "${final_stage}")/best_checkpoint.tar.gz" \
  "${best_step}"
