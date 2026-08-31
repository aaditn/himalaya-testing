#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS_MOUNT="${RUNS_MOUNT:-/runs}"
TRAIN_ENVS="${TRAIN_ENVS:-8192}"
PREFIX="${PREFIX:-g1_20deg_stability_v1}"
RESTORE="${RESTORE:-${RUNS_MOUNT}/g1_45deg_curriculum_h200_v2_02_transfer-20deg/checkpoints/000020316160}"

python -m pip install --upgrade pip
python -m pip install -e "${ROOT}[cuda]"
python - <<'PY'
import jax

if jax.default_backend() != "gpu":
    raise SystemExit(f"GPU backend required, got {jax.default_backend()}: {jax.devices()}")
print(f"JAX devices: {jax.devices()}", flush=True)
PY

test -d "${RESTORE}"
mkdir -p "${RUNS_MOUNT}"
cd "${ROOT}"
exec python scripts/train_climb_curriculum.py \
  --curriculum configs/curriculum_stability_20deg.json \
  --runs-dir "${RUNS_MOUNT}" \
  --prefix "${PREFIX}" \
  --envs "${TRAIN_ENVS}" \
  --restore "${RESTORE}"
