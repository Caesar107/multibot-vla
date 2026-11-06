#!/usr/bin/env bash
# No-Plücker ablation: train three mixtures where perturbed cameras (view_id 1..3) are
# 25% / 50% / 75% of the episode set. Always keeps all canonical view_id=0 demos.
#
# Mixture (with 2000 canon + 6000 perturbed in libero_campose):
#   p25 -> ~2000 canon + 667 perturbed  = ~2667 episodes (perturbed 25%)
#   p50 -> ~2000 canon + 2000 perturbed = 4000 episodes  (perturbed 50%)
#   p75 -> ~2000 canon + 6000 perturbed = 8000 episodes  (perturbed 75%, = full multiview)
#
# Submit THREE DLC jobs (same console settings as dlc_train.sh), one command each:
#
#   CONFIG=pi05_libero_multiview_p25 bash /mnt/workspace/zml/code/multi-agent/openpi/scripts/libero_campose/dlc_train.sh
#   CONFIG=pi05_libero_multiview_p50 bash /mnt/workspace/zml/code/multi-agent/openpi/scripts/libero_campose/dlc_train.sh
#   CONFIG=pi05_libero_multiview_p75 bash /mnt/workspace/zml/code/multi-agent/openpi/scripts/libero_campose/dlc_train.sh
#
# Or run this helper with FRACS to print the three lines / optionally exec one:
#   bash scripts/libero_campose/dlc_train_multiview_frac.sh           # print commands
#   FRAC=25 bash scripts/libero_campose/dlc_train_multiview_frac.sh   # launch one
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_cmds() {
  for f in 25 50 75; do
    echo "CONFIG=pi05_libero_multiview_p${f} bash ${HERE}/dlc_train.sh"
  done
}

if [[ -z "${FRAC:-}" ]]; then
  echo "# paste one per DLC job:"
  print_cmds
  exit 0
fi

case "${FRAC}" in
  25|50|75) ;;
  *)
    echo "FRAC must be 25, 50, or 75 (got ${FRAC})" >&2
    exit 1
    ;;
esac

export CONFIG="pi05_libero_multiview_p${FRAC}"
export EXP_NAME="${EXP_NAME:-multiview_p${FRAC}_v1}"
exec bash "${HERE}/dlc_train.sh"
