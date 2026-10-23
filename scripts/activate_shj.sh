#!/usr/bin/env bash
# Source this file from the current shell: source activate_shj.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_EXE="${SHJ_CONDA_EXE:-/home/shj/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-shj}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann-9.1.0/set_env.sh}"

if [[ ! -f "${CANN_ENV}" ]]; then
  echo "CANN environment script not found: ${CANN_ENV}" >&2
  return 1 2>/dev/null || exit 1
fi
if ! "${CONDA_EXE}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Conda environment ${ENV_NAME} does not exist. Run setup_shj_env.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${ENV_NAME}"
set +u
# CANN's setup script reads several optional environment variables directly.
source "${CANN_ENV}"
set -u

export DEVICE_TYPE="${DEVICE_TYPE:-npu}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

echo "Activated ${ENV_NAME}: $(python --version 2>&1)"
echo "CANN: ${ASCEND_HOME_PATH:-unknown}"
echo "DEVICE_TYPE=${DEVICE_TYPE} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES} NPROC_PER_NODE=${NPROC_PER_NODE}"
