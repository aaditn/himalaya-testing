#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS_MOUNT="${RUNS_MOUNT:-/runs}"
SCRATCH_RUNS="${SCRATCH_RUNS:-/tmp/himalaya-runs}"
PREFIX="${PREFIX:-g1_45deg_h200_resume_v3}"
TRAIN_ENVS="${TRAIN_ENVS:-8192}"
OUTPUT_REPO="${OUTPUT_REPO:-iteratehack/g1-himalaya-four-contact}"
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
# final artifacts only after all stages complete. The source bucket is mounted
# read-only because the organization's private-storage quota is full.
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
  python - "${stage_dir}" "${OUTPUT_REPO}" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import HfApi

stage = Path(sys.argv[1])
HfApi().upload_folder(
    repo_id=sys.argv[2],
    folder_path=stage,
    path_in_repo=f"runs/{stage.name}",
    allow_patterns=["policy", "metrics.json", "best_checkpoint.json"],
)
PY
done

final_stage="$(find "${SCRATCH_RUNS}" -maxdepth 1 -type d -name "${PREFIX}_*" | sort | tail -n 1)"
best_step="$(python -c 'import json,sys; print("%012d" % json.load(open(sys.argv[1]))["step"])' "${final_stage}/best_checkpoint.json")"
tar -C "${final_stage}/checkpoints" -czf \
  "/tmp/best_checkpoint.tar.gz" \
  "${best_step}"
python - "${final_stage}" "${OUTPUT_REPO}" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import HfApi

stage = Path(sys.argv[1])
HfApi().upload_file(
    repo_id=sys.argv[2],
    path_or_fileobj="/tmp/best_checkpoint.tar.gz",
    path_in_repo=f"runs/{stage.name}/best_checkpoint.tar.gz",
)
PY
