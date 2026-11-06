#!/usr/bin/env bash
# Train a campose variant on PAI DLC:
#   pi05_libero_campose          multi-view + Plücker                 exp campose_v1
#   pi05_libero_campose_canon    original camera + Plücker            exp campose_canon_v1
#   pi05_libero_multiview        full multi-view, no Plücker          exp multiview_v1
#   pi05_libero_multiview_p25/50/75  no Plücker, perturbed share 25/50/75%
#
# ========== DLC 控制台 ==========
# 任务类型: PyTorch, 节点数 1, 每节点 GPU 8 (A800/A100 80GB)
# 镜像: DSW 用户镜像 (同 effectvla 的 dlc_train.sh), 驱动 CUDA12 / 525+
# 三方库: 留空 (用 NAS 上的 openpi .venv)
# 内存: 512GB
# 挂载: NAS /mnt/workspace, OSS /mnt/oss/zml
# 启动命令:
#   bash /mnt/workspace/zml/code/multi-agent/openpi/scripts/libero_campose/dlc_train.sh
#   CONFIG=pi05_libero_campose_canon EXP_NAME=campose_canon_v1 bash .../dlc_train.sh
#   CONFIG=pi05_libero_multiview_p25 EXP_NAME=multiview_p25_v1 bash .../dlc_train.sh
#   CONFIG=pi05_libero_multiview_p50 EXP_NAME=multiview_p50_v1 bash .../dlc_train.sh
#   CONFIG=pi05_libero_multiview_p75 EXP_NAME=multiview_p75_v1 bash .../dlc_train.sh
#
# 跑完看:
#   /mnt/oss/zml/logs/libero_campose/train/<EXP_NAME>/train.log
#   /mnt/oss/zml/checkpoints/openpi/<CONFIG>/<EXP_NAME>/
# ================================
set -euo pipefail

ROOT="/mnt/workspace/zml/code/multi-agent/openpi"
cd "${ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/env.sh"   # shared venv + this repo's sources on PYTHONPATH + WANDB_API_KEY
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/mnt/oss/zml/datasets/pi}"
export HF_HUB_ENABLE_HF_TRANSFER=0
# lerobot converts the parquet files into an Arrow cache under HF_HOME on first load. The dataset is 160 GB
# (image mode), so keep that cache on DLC local disk like the effectvla jobs do. Expect ~75 min of
# "Generating train split" before step 0 (pi05_libero_official: 33 GB took 15 min on DLC).
if [[ -z "${HF_HOME:-}" ]]; then
  export HF_HOME=/tmp/zml-hf-libero-campose
  mkdir -p "${HF_HOME}"
fi

unset JAX_PLATFORMS XLA_FLAGS XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_ALLOCATOR || true
# wandb cloud is blocked from DLC. Keep logging on, write the run under WANDB_DIR on NAS.
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-pi05_libero_campose}"
case "${CONFIG}" in
  pi05_libero_campose_canon)
    EXP_NAME="${EXP_NAME:-campose_canon_v1}"
    REPO_ID="${REPO_ID:-physical-intelligence/libero}"
    # State/actions stats are identical to stock pi05_libero (same original dataset).
    NORM_STATS_DEFAULT="/mnt/oss/zml/models/openpi/assets/pi05_libero/physical-intelligence/libero/norm_stats.json"
    ;;
  pi05_libero_multiview|pi05_libero_multiview_p25|pi05_libero_multiview_p50|pi05_libero_multiview_p75)
    # Default exp names: multiview_v1 / multiview_p25_v1 / ...
    suffix="${CONFIG#pi05_libero_}"
    EXP_NAME="${EXP_NAME:-${suffix}_v1}"
    REPO_ID="${REPO_ID:-libero_campose}"
    # Same images/state/actions as pi05_libero_campose; reuse those norm stats (no camera params in them).
    NORM_STATS_DEFAULT="/mnt/oss/zml/models/openpi/assets/pi05_libero_campose/${REPO_ID}/norm_stats.json"
    ;;
  *)
    EXP_NAME="${EXP_NAME:-campose_v1}"
    REPO_ID="${REPO_ID:-libero_campose}"
    NORM_STATS_DEFAULT="/mnt/oss/zml/models/openpi/assets/${CONFIG}/${REPO_ID}/norm_stats.json"
    ;;
esac
DATA_DIR="${HF_LEROBOT_HOME}/${REPO_ID}"
LOG_DIR="/mnt/oss/zml/logs/libero_campose/train/${EXP_NAME}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/train.log"

PYTHON="${ROOT}/.venv/bin/python"
[[ -x "${PYTHON}" ]] || { echo "[dlc] missing ${PYTHON}" >&2; exit 1; }
[[ -f "${DATA_DIR}/meta/info.json" ]] || { echo "[dlc] missing dataset ${DATA_DIR}" >&2; exit 1; }

"${PYTHON}" - <<'PY'
import jax
backend, n = jax.default_backend(), jax.device_count()
print(f"[dlc] JAX backend={backend} devices={n}")
if backend != "gpu":
    raise SystemExit("Need the GPU backend")
PY

NORM_STATS="${NORM_STATS:-${NORM_STATS_DEFAULT}}"
if [[ ! -f "${NORM_STATS}" ]]; then
  echo "[dlc] computing norm stats -> ${NORM_STATS}" | tee -a "${LOG}"
  "${PYTHON}" "${ROOT}/scripts/compute_norm_stats.py" --config-name "${CONFIG}" 2>&1 | tee -a "${LOG}"
fi
[[ -f "${NORM_STATS}" ]] || { echo "[dlc] norm stats still missing: ${NORM_STATS}" >&2; exit 1; }

CKPT_DIR="/mnt/oss/zml/checkpoints/openpi/${CONFIG}/${EXP_NAME}"
TRAIN_FLAG=(--overwrite)
if [[ -d "${CKPT_DIR}" ]] && compgen -G "${CKPT_DIR}/[0-9]*" >/dev/null; then
  echo "[dlc] checkpoint exists, resuming ${CKPT_DIR}"
  TRAIN_FLAG=(--resume)
fi

echo "TRAIN_START $(date -Iseconds) config=${CONFIG} exp_name=${EXP_NAME} GPUS=${CUDA_VISIBLE_DEVICES} data=${DATA_DIR} wandb=${WANDB_MODE} flag=${TRAIN_FLAG[*]} extra=${EXTRA_ARGS:-}" | tee -a "${LOG}"
set +e
set -o pipefail
# shellcheck disable=SC2086
"${PYTHON}" "${ROOT}/scripts/train.py" "${CONFIG}" --exp-name "${EXP_NAME}" "${TRAIN_FLAG[@]}" ${EXTRA_ARGS:-} \
  2>&1 | tee -a "${LOG}"
code=${PIPESTATUS[0]}
echo "TRAIN_EXIT:${code} $(date -Iseconds)" | tee -a "${LOG}"
exit "${code}"
