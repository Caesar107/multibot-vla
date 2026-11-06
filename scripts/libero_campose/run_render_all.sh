#!/usr/bin/env bash
# Render all 40 LIBERO tasks (4 suites x 10 tasks) with multi-camera views in parallel.
# One render_multicam.py process per task, run in the py3.8 LIBERO/robosuite venv.
#
# Usage:
#   bash scripts/libero_campose/run_render_all.sh [NUM_PARALLEL]
#
# Env overrides:
#   RENDER_OUT   output root (default /tmp/zml-libero-campose/render)
#   RENDER_LOGS  log dir     (default /mnt/oss/zml/logs/libero_campose/render)
#   RENDER_VENV  python venv with robosuite/LIBERO (default effectvla's examples/libero/.venv)
#   EXTRA_ARGS   extra args forwarded to render_multicam.py (e.g. "--overwrite")
set -euo pipefail

NUM_PARALLEL="${1:-10}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER_OUT="${RENDER_OUT:-/tmp/zml-libero-campose/render}"
RENDER_LOGS="${RENDER_LOGS:-/mnt/oss/zml/logs/libero_campose/render}"
RENDER_VENV="${RENDER_VENV:-/mnt/workspace/zml/code/effectvla/openpi/examples/libero/.venv}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/mnt/workspace/zml/code/lingbot-va/.egl/10_nvidia.json
export LIBERO_CONFIG_PATH=/mnt/workspace/zml/tmp/libero_campose/libero_config
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$RENDER_OUT" "$RENDER_LOGS"

PY="$RENDER_VENV/bin/python"
SUITES=(libero_spatial libero_object libero_goal libero_10)

run_task() {
  local suite="$1" idx="$2"
  local log="$RENDER_LOGS/${suite}_${idx}.log"
  if "$PY" "$HERE/render_multicam.py" --suite "$suite" --task-index "$idx" --out "$RENDER_OUT" $EXTRA_ARGS > "$log" 2>&1; then
    echo "[done] $suite $idx"
  else
    echo "[FAIL] $suite $idx (see $log)"
  fi
}
export -f run_task
export PY HERE RENDER_OUT RENDER_LOGS EXTRA_ARGS

for suite in "${SUITES[@]}"; do
  for idx in $(seq 0 9); do
    echo "$suite $idx"
  done
done | xargs -P "$NUM_PARALLEL" -n 2 bash -c 'run_task "$0" "$1"'

echo "all tasks finished; output in $RENDER_OUT"
