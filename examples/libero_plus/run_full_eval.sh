#!/usr/bin/env bash
# LIBERO-Plus eval for a multi-agent/openpi checkpoint (pi05_libero_campose / pi05_libero_campose_canon / pi05_libero).
# Starts the policy server from this repo's sources, runs the requested suites (1 trial/task), then aggregates.
#
# Usage:
#   CONFIG=pi05_libero_campose EXP_NAME=campose_v1 bash examples/libero_plus/run_full_eval.sh
#   CONFIG=pi05_libero_campose EXP_NAME=campose_v1 CATEGORY=Camera bash examples/libero_plus/run_full_eval.sh
#   CKPT=/mnt/oss/zml/checkpoints/openpi/pi05_libero_campose/campose_v1/29999 SUITE_ONLY=libero_spatial \
#     bash examples/libero_plus/run_full_eval.sh
#   # stock pi05_libero baseline (no camera conditioning in the model; the extra keys are ignored):
#   CONFIG=pi05_libero CKPT=/mnt/oss/zml/models/openpi/openpi-assets/checkpoints/pi05_libero EXP_NAME=stock \
#     CATEGORY=Camera bash examples/libero_plus/run_full_eval.sh
#   SMOKE=1 ... # 2 tasks of libero_spatial
#
# Env:
#   CONFIG     train config name (default pi05_libero_campose)
#   EXP_NAME   experiment name under /mnt/oss/zml/checkpoints/openpi/<CONFIG>/ (default campose_v1)
#   CKPT       explicit checkpoint step dir (default: newest numeric step under CKPT_ROOT)
#   CATEGORY   LIBERO-Plus category filter (Camera|Robot|Language|Light|Background|Noise|Layout); empty = all
#   NO_CAM_COND=1  do not send cam_K/cam_c2w
set -euo pipefail

ROOT="/mnt/workspace/zml/code/multi-agent/openpi"
EFFECTVLA_ROOT="${EFFECTVLA_ROOT:-/mnt/workspace/zml/code/effectvla/openpi}"
cd "${ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/env.sh"

CONFIG="${CONFIG:-pi05_libero_campose}"
EXP_NAME="${EXP_NAME:-campose_v1}"
CKPT_ROOT="${CKPT_ROOT:-/mnt/oss/zml/checkpoints/openpi/${CONFIG}/${EXP_NAME}}"
if [[ -z "${CKPT:-}" ]]; then
  CKPT="$(find "${CKPT_ROOT}" -maxdepth 1 -type d -regex '.*/[0-9]+' | sort -V | tail -1)"
fi
if [[ -z "${CKPT}" || ! -d "${CKPT}/params" ]]; then
  echo "[eval] no checkpoint under ${CKPT_ROOT} (CKPT=${CKPT:-})" >&2
  exit 1
fi
STEP="$(basename "${CKPT}")"
CATEGORY="${CATEGORY:-}"

PORT="${PORT:-8000}"
EGL_ID="${EGL_ID:-0}"
GPU="${GPU:-0}"
CAT_TAG="${CATEGORY:+_${CATEGORY}}"
OUT_ROOT="${OUT_ROOT:-/mnt/oss/zml/logs/libero_campose/libero-plus-eval/${CONFIG}/${EXP_NAME}/step${STEP}${CAT_TAG}}"
SERVER_LOG="${OUT_ROOT}/server.log"
mkdir -p "${OUT_ROOT}"

LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-${EFFECTVLA_ROOT}/third_party/LIBERO-plus}"
ASSET_LOCAL="${ASSET_LOCAL:-/tmp/zml-libero-plus-assets}"
ASSET_PERSIST="${ASSET_PERSIST:-/mnt/workspace/zml/tmp/libero_plus/assets}"
ASSET_LINK="${LIBERO_PLUS_ROOT}/libero/libero/assets"
if [[ ! -f "${LIBERO_PLUS_ROOT}/libero/libero/benchmark/task_classification.json" ]]; then
  echo "[eval] missing LIBERO-plus at ${LIBERO_PLUS_ROOT}; run ${EFFECTVLA_ROOT}/examples/libero_plus/prepare_env.sh" >&2
  exit 1
fi
ASSET_DIR=""
for CAND in "${ASSET_LOCAL}" "${ASSET_PERSIST}"; do
  if [[ -d "${CAND}/scenes" || -d "${CAND}/stable_hope_objects" || -d "${CAND}/inspire" ]]; then
    ASSET_DIR="${CAND}"
    break
  fi
done
if [[ -z "${ASSET_DIR}" ]]; then
  echo "[eval] assets missing. Run: bash ${EFFECTVLA_ROOT}/examples/libero_plus/prepare_env.sh" >&2
  exit 1
fi
# Same relink effectvla's script does (the checkout is zml's own).
if [[ "$(readlink -f "${ASSET_LINK}" || true)" != "$(readlink -f "${ASSET_DIR}")" ]]; then
  rm -rf "${ASSET_LINK}"
  ln -sfn "${ASSET_DIR}" "${ASSET_LINK}"
fi

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/mnt/workspace/zml/code/lingbot-va/.egl/10_nvidia.json}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/oss/zml/models/openpi}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

PYTHON_SERVER="${ROOT}/.venv/bin/python"
CLIENT_PY="${EFFECTVLA_ROOT}/examples/libero/.venv/bin/python"
if [[ ! -x "${PYTHON_SERVER}" || ! -x "${CLIENT_PY}" ]]; then
  echo "[eval] missing server (${PYTHON_SERVER}) or client (${CLIENT_PY}) python" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval] stopping server pid=${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[eval] CONFIG=${CONFIG} CKPT=${CKPT} CATEGORY=${CATEGORY:-all}"
echo "[eval] OUT=${OUT_ROOT}"
echo "SERVER_START $(date -Iseconds)" | tee -a "${SERVER_LOG}"
"${PYTHON_SERVER}" scripts/serve_policy.py \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.config "${CONFIG}" \
  --policy.dir "${CKPT}" \
  >>"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 240); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval] server died; see ${SERVER_LOG}" >&2
    tail -n 60 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if PYTHONPATH="${ROOT}/packages/openpi-client/src" "${CLIENT_PY}" - <<PY 2>/dev/null
from openpi_client import websocket_client_policy as w
w.WebsocketClientPolicy("127.0.0.1", ${PORT})
print("ok")
PY
  then
    echo "[eval] server ready after ${i}s"
    break
  fi
  if [[ "${i}" -eq 240 ]]; then
    echo "[eval] server not ready in 240s; see ${SERVER_LOG}" >&2
    tail -n 60 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 1
done

SUITES=(libero_spatial libero_object libero_goal libero_10)
if [[ -n "${SUITE_ONLY:-}" ]]; then
  SUITES=("${SUITE_ONLY}")
fi

CLIENT_EXTRA=()
if [[ -n "${CATEGORY}" ]]; then
  CLIENT_EXTRA+=(--args.category "${CATEGORY}")
fi
if [[ "${NO_CAM_COND:-0}" == "1" ]]; then
  CLIENT_EXTRA+=(--args.no-cam-cond)
fi
if [[ "${SMOKE:-0}" == "1" ]]; then
  SUITES=(libero_spatial)
  CLIENT_EXTRA+=(--args.start-idx 0 --args.end-idx 60)
  OUT_ROOT="${OUT_ROOT}/smoke"
  mkdir -p "${OUT_ROOT}"
fi

for SUITE in "${SUITES[@]}"; do
  SUITE_OUT="${OUT_ROOT}/${SUITE}"
  mkdir -p "${SUITE_OUT}"
  echo "SUITE_START $(date -Iseconds) ${SUITE}" | tee -a "${OUT_ROOT}/eval.log"
  set +e
  if [[ ${#CLIENT_EXTRA[@]} -gt 0 ]]; then
    bash "${ROOT}/examples/libero_plus/run_client.sh" \
      "${SUITE}" "${PORT}" "${EGL_ID}" "${SUITE_OUT}" -- "${CLIENT_EXTRA[@]}" \
      2>&1 | tee -a "${OUT_ROOT}/eval.log"
  else
    bash "${ROOT}/examples/libero_plus/run_client.sh" \
      "${SUITE}" "${PORT}" "${EGL_ID}" "${SUITE_OUT}" \
      2>&1 | tee -a "${OUT_ROOT}/eval.log"
  fi
  code=${PIPESTATUS[0]}
  echo "SUITE_EXIT:${code} $(date -Iseconds) ${SUITE}" | tee -a "${OUT_ROOT}/eval.log"
  set -e
  if [[ "${code}" -ne 0 ]]; then
    echo "[eval] suite ${SUITE} failed with ${code}" >&2
    exit "${code}"
  fi
done

PYTHONPATH="${LIBERO_PLUS_ROOT}:${ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${CLIENT_PY}" "${ROOT}/examples/libero_plus/main.py" \
  --args.aggregate-only \
  --args.output-dir "${OUT_ROOT}" \
  2>&1 | tee -a "${OUT_ROOT}/eval.log" || true

echo "EVAL_DONE $(date -Iseconds) out=${OUT_ROOT}" | tee -a "${OUT_ROOT}/eval.log"
