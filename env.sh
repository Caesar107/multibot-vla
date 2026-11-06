# multi-agent openpi toolchain on NAS. Usage: source /mnt/workspace/zml/code/multi-agent/openpi/env.sh
#
# The virtualenv is shared with /mnt/workspace/zml/code/openpi (.venv here is a symlink; no second copy on disk).
# That venv has *upstream* openpi installed in editable mode, so this tree's sources are put first on PYTHONPATH.
# Do NOT run `uv sync` / `uv run` in this directory -- it would re-point the shared venv's editable install.
# Always run with `python ...` (resolved to .venv/bin/python below) instead of `uv run ...`.

MULTI_AGENT_OPENPI_ROOT="/mnt/workspace/zml/code/multi-agent/openpi"
# shellcheck disable=SC1091
source /mnt/workspace/zml/code/openpi/env.sh
export PYTHONPATH="${MULTI_AGENT_OPENPI_ROOT}/src:${MULTI_AGENT_OPENPI_ROOT}/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="${MULTI_AGENT_OPENPI_ROOT}/.venv/bin:$PATH"
unset VIRTUAL_ENV
export VIRTUAL_ENV="${MULTI_AGENT_OPENPI_ROOT}/.venv"

# huggingface.co is unreachable from this DSW; needed only for hf:// weight paths or dataset downloads.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
