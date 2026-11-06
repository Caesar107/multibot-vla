#!/usr/bin/env bash
# Serial DLC job: no-Plücker multiview ablations with perturbed-camera share 25% -> 50% -> 75%.
# Uses the same 8 GPUs one after another (do NOT run these three in parallel on one 8-GPU node).
#
# DLC 启动命令（整行粘贴）:
#   bash /mnt/workspace/zml/code/multi-agent/openpi/scripts/libero_campose/dlc_train_multiview_p25_p50_p75.sh
#
# 日志 / checkpoint:
#   /mnt/oss/zml/logs/libero_campose/train/multiview_p{25,50,75}_v1/train.log
#   /mnt/oss/zml/checkpoints/openpi/pi05_libero_multiview_p{25,50,75}/multiview_p{25,50,75}_v1/
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUMMARY_LOG="/mnt/oss/zml/logs/libero_campose/train/multiview_frac_serial.log"
mkdir -p "$(dirname "${SUMMARY_LOG}")"

echo "SERIAL_START $(date -Iseconds) fracs=25,50,75" | tee -a "${SUMMARY_LOG}"

for FRAC in 25 50 75; do
  # Clear EXP_NAME so dlc_train.sh derives multiview_p${FRAC}_v1 from CONFIG (do not leak prior iter).
  unset EXP_NAME REPO_ID NORM_STATS || true
  export CONFIG="pi05_libero_multiview_p${FRAC}"
  echo "JOB_START $(date -Iseconds) CONFIG=${CONFIG}" | tee -a "${SUMMARY_LOG}"
  set +e
  bash "${HERE}/dlc_train.sh"
  code=$?
  set -e
  echo "JOB_EXIT:${code} $(date -Iseconds) CONFIG=${CONFIG}" | tee -a "${SUMMARY_LOG}"
  if [[ "${code}" -ne 0 ]]; then
    echo "SERIAL_ABORT $(date -Iseconds) failed at ${CONFIG}" | tee -a "${SUMMARY_LOG}"
    exit "${code}"
  fi
done

echo "SERIAL_DONE $(date -Iseconds)" | tee -a "${SUMMARY_LOG}"
