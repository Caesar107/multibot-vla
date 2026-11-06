#!/usr/bin/env bash
# LIBERO-Plus client (camera-aware). Talks to an already-running openpi policy server.
#
# Reuses effectvla's LIBERO-Plus checkout + py3.8 client venv read-only; only main.py comes from this repo.
#
# Usage:
#   bash examples/libero_plus/run_client.sh libero_spatial
#   bash examples/libero_plus/run_client.sh libero_10 8000 0 /path/to/out
#   bash examples/libero_plus/run_client.sh libero_spatial 8000 0 /path/to/out -- --args.category Camera
#   bash examples/libero_plus/run_client.sh libero_spatial 8000 0 /path/to/out -- --args.no-cam-cond
# Args: SUITE [PORT=8000] [EGL_ID=0] [OUTPUT_DIR] [-- extra main.py flags...]
set -euo pipefail

SUITE="${1:?suite (libero_spatial|libero_object|libero_goal|libero_10)}"
PORT="${2:-8000}"
EGL_ID="${3:-0}"
OUTPUT_DIR="${4:-/mnt/oss/zml/logs/libero_campose/libero-plus-eval/manual/${SUITE}}"
shift $(( $# >= 4 ? 4 : $# )) || true
EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  EXTRA_ARGS=("$@")
fi

ROOT="/mnt/workspace/zml/code/multi-agent/openpi"
EFFECTVLA_ROOT="${EFFECTVLA_ROOT:-/mnt/workspace/zml/code/effectvla/openpi}"
LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-${EFFECTVLA_ROOT}/third_party/LIBERO-plus}"
VENV="${CLIENT_VENV:-${EFFECTVLA_ROOT}/examples/libero/.venv}"
# Per-process dir avoids races when many clients start in parallel.
CONFIG_DIR="${LIBERO_CONFIG_DIR:-/mnt/workspace/zml/tmp/libero_campose/libero_plus_cfg_${SUITE}_$$}"
BENCHMARK="${LIBERO_PLUS_ROOT}/libero/libero"

if [[ ! -f "${BENCHMARK}/benchmark/task_classification.json" ]]; then
  echo "LIBERO-Plus checkout not found at ${LIBERO_PLUS_ROOT}" >&2
  echo "Run bash ${EFFECTVLA_ROOT}/examples/libero_plus/prepare_env.sh first" >&2
  exit 1
fi
if [[ ! -d "${BENCHMARK}/assets/" ]]; then
  echo "Missing ${BENCHMARK}/assets (dangling symlink?). Run effectvla prepare_env.sh or run_full_eval.sh." >&2
  exit 1
fi

mkdir -p "${CONFIG_DIR}" "${OUTPUT_DIR}"
cat > "${CONFIG_DIR}/config.yaml" <<EOF
benchmark_root: ${BENCHMARK}
bddl_files: ${BENCHMARK}/bddl_files
init_states: ${BENCHMARK}/init_files
datasets: ${LIBERO_PLUS_ROOT}/libero/datasets
assets: ${BENCHMARK}/assets
EOF

export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${LIBERO_PLUS_ROOT}:${ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export LIBERO_CONFIG_PATH="${CONFIG_DIR}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/mnt/workspace/zml/code/lingbot-va/.egl/10_nvidia.json}"
unset CUDA_VISIBLE_DEVICES || true
export MUJOCO_EGL_DEVICE_ID="${EGL_ID}"

cd "${ROOT}"
python - <<'PY'
import pathlib
import libero.libero
p = pathlib.Path(libero.libero.__file__).resolve().parent / "benchmark" / "task_classification.json"
if not p.is_file():
    raise SystemExit(f"imported libero is not LIBERO-Plus: {libero.libero.__file__}")
print(f"LIBERO_PLUS {p}")
PY

echo "CLIENT_START $(date -Iseconds) suite=${SUITE} port=${PORT} egl=${EGL_ID} out=${OUTPUT_DIR}"
exec python examples/libero_plus/main.py \
  --args.host 127.0.0.1 \
  --args.port "${PORT}" \
  --args.task-suite-name "${SUITE}" \
  --args.num-trials-per-task 1 \
  --args.output-dir "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}"
