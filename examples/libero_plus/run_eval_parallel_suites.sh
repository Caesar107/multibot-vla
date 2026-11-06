#!/usr/bin/env bash
# One policy server + 4 LIBERO-Plus suites in parallel (same port, 1 trial/task).
# Same protocol as effectvla/openpi/examples/libero_plus/run_eval_parallel_suites.sh.
#
# Usage:
#   CONFIG=pi05_libero_campose_canon EXP_NAME=campose_canon_v1 bash examples/libero_plus/run_eval_parallel_suites.sh
#   CATEGORY=Camera CONFIG=... EXP_NAME=... bash examples/libero_plus/run_eval_parallel_suites.sh
set -euo pipefail

ROOT="/mnt/workspace/zml/code/multi-agent/openpi"
EFFECTVLA_ROOT="${EFFECTVLA_ROOT:-/mnt/workspace/zml/code/effectvla/openpi}"
cd "${ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/env.sh"

CONFIG="${CONFIG:-pi05_libero_campose_canon}"
EXP_NAME="${EXP_NAME:-campose_canon_v1}"
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
CAT_TAG="${CATEGORY:+_${CATEGORY}}"

PORT="${PORT:-8000}"
GPU="${GPU:-0}"
EGL_ID="${EGL_ID:-0}"
OUT_ROOT="${OUT_ROOT:-/mnt/oss/zml/logs/libero_campose/libero-plus-eval/${CONFIG}/${EXP_NAME}/step${STEP}${CAT_TAG}_parallel}"
NAS_LOG="${NAS_LOG:-/mnt/workspace/zml/tmp/logs/libero_campose/libero-plus-eval/${CONFIG}/${EXP_NAME}/step${STEP}${CAT_TAG}_parallel}"
mkdir -p "${OUT_ROOT}" "${NAS_LOG}"
EVAL_LOG="${OUT_ROOT}/eval.log"
SERVER_LOG="${OUT_ROOT}/server.log"
ln -sfn "${OUT_ROOT}" "${NAS_LOG}/oss" 2>/dev/null || true

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
if [[ "$(readlink -f "${ASSET_LINK}" || true)" != "$(readlink -f "${ASSET_DIR}")" ]]; then
  rm -rf "${ASSET_LINK}"
  ln -sfn "${ASSET_DIR}" "${ASSET_LINK}"
fi

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/mnt/workspace/zml/code/lingbot-va/.egl/10_nvidia.json}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/oss/zml/models/openpi}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

PYTHON_SERVER="${ROOT}/.venv/bin/python"
CLIENT_PY="${EFFECTVLA_ROOT}/examples/libero/.venv/bin/python"
if [[ ! -x "${PYTHON_SERVER}" || ! -x "${CLIENT_PY}" ]]; then
  echo "[eval] missing server (${PYTHON_SERVER}) or client (${CLIENT_PY}) python" >&2
  exit 1
fi

CLIENT_EXTRA=()
if [[ -n "${CATEGORY}" ]]; then
  CLIENT_EXTRA+=(--args.category "${CATEGORY}")
fi
if [[ "${NO_CAM_COND:-0}" == "1" ]]; then
  CLIENT_EXTRA+=(--args.no-cam-cond)
fi

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do
    kill "$p" 2>/dev/null || true
  done
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "EVAL_START $(date -Iseconds) config=${CONFIG} ckpt=${CKPT} category=${CATEGORY:-all} port=${PORT} gpu=${GPU} parallel=4 out=${OUT_ROOT}" | tee -a "${EVAL_LOG}" | tee -a "${NAS_LOG}/driver.log"
"${PYTHON_SERVER}" scripts/serve_policy.py \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.config "${CONFIG}" \
  --policy.dir "${CKPT}" \
  >>"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "SERVER_PID=${SERVER_PID}" | tee -a "${EVAL_LOG}" | tee -a "${NAS_LOG}/driver.log"

for i in $(seq 1 240); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval] server died; see ${SERVER_LOG}" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  if PYTHONPATH="${ROOT}/packages/openpi-client/src" "${CLIENT_PY}" - <<PY 2>/dev/null
from openpi_client import websocket_client_policy as w
w.WebsocketClientPolicy("127.0.0.1", ${PORT})
print("ok")
PY
  then
    echo "[eval] server ready after ${i}s" | tee -a "${EVAL_LOG}" | tee -a "${NAS_LOG}/driver.log"
    break
  fi
  if [[ "${i}" -eq 240 ]]; then
    echo "[eval] server timeout; see ${SERVER_LOG}" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    exit 1
  fi
  sleep 1
done

SUITES=(libero_spatial libero_object libero_goal libero_10)
for SUITE in "${SUITES[@]}"; do
  SUITE_OUT="${OUT_ROOT}/${SUITE}"
  mkdir -p "${SUITE_OUT}"
  echo "SUITE_START $(date -Iseconds) ${SUITE}" | tee -a "${EVAL_LOG}" | tee -a "${NAS_LOG}/driver.log"
  if [[ ${#CLIENT_EXTRA[@]} -gt 0 ]]; then
    bash "${ROOT}/examples/libero_plus/run_client.sh" \
      "${SUITE}" "${PORT}" "${EGL_ID}" "${SUITE_OUT}" -- "${CLIENT_EXTRA[@]}" \
      >"${SUITE_OUT}/suite.log" 2>&1 &
  else
    bash "${ROOT}/examples/libero_plus/run_client.sh" \
      "${SUITE}" "${PORT}" "${EGL_ID}" "${SUITE_OUT}" \
      >"${SUITE_OUT}/suite.log" 2>&1 &
  fi
  PIDS+=($!)
done

fail=0
for idx in "${!PIDS[@]}"; do
  p="${PIDS[$idx]}"
  suite="${SUITES[$idx]}"
  if wait "$p"; then
    echo "SUITE_EXIT:0 $(date -Iseconds) ${suite}" | tee -a "${EVAL_LOG}" | tee -a "${NAS_LOG}/driver.log"
  else
    code=$?
    echo "SUITE_EXIT:${code} $(date -Iseconds) ${suite}" | tee -a "${EVAL_LOG}" | tee -a "${NAS_LOG}/driver.log"
    fail=1
  fi
done

PYTHONPATH="${LIBERO_PLUS_ROOT}:${ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${CLIENT_PY}" "${ROOT}/examples/libero_plus/main.py" \
  --args.aggregate-only \
  --args.output-dir "${OUT_ROOT}" \
  2>&1 | tee -a "${EVAL_LOG}" | tee -a "${NAS_LOG}/driver.log" || true

echo "EVAL_DONE $(date -Iseconds) fail=${fail} out=${OUT_ROOT}" | tee -a "${EVAL_LOG}" | tee -a "${NAS_LOG}/driver.log"
exit "${fail}"
