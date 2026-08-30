#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS_MOUNT="${RUNS_MOUNT:-/runs}"
RUN_NAME="${RUN_NAME:-g1_support_exchange_5deg}"
TRAIN_ENVS="${TRAIN_ENVS:-4096}"
TRAIN_STEPS="${TRAIN_STEPS:-20000000}"

python -m pip install --upgrade pip
python -m pip install -e "${ROOT}[cuda]"
python - <<'PY'
import jax

if jax.default_backend() != "gpu":
    raise SystemExit(f"GPU backend required, got {jax.default_backend()}: {jax.devices()}")
print(f"JAX devices: {jax.devices()}", flush=True)
PY

mkdir -p "${RUNS_MOUNT}"

cd "${ROOT}"
exec python scripts/train.py \
  --climb \
  --name "${RUN_NAME}" \
  --runs-dir "${RUNS_MOUNT}" \
  --envs "${TRAIN_ENVS}" \
  --timesteps "${TRAIN_STEPS}" \
  --slope 5 \
  --roughness 0.005 \
  --spike-friction 0.95 \
  --foot-friction 1.90 \
  --hand-load 0.25 \
  --speed 0.15 \
  --seed 5 \
  --num-evals 6 \
  --no-boulders \
  --no-randomization
