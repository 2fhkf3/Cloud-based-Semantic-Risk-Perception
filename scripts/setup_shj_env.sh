#!/usr/bin/env bash
set -euo pipefail

# Create the project environment and install a CANN-compatible Ascend framework pair.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_EXE="${SHJ_CONDA_EXE:-/home/shj/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-shj}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann-9.1.0/set_env.sh}"
TORCH_WHL="${TORCH_WHL:-}"
TORCH_NPU_WHL="${TORCH_NPU_WHL:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
TORCH_NPU_INDEX_URL="${TORCH_NPU_INDEX_URL:-}"

if [[ ! -x "${CONDA_EXE}" ]]; then
  echo "Conda executable not found: ${CONDA_EXE}" >&2
  exit 1
fi
if [[ ! -f "${CANN_ENV}" ]]; then
  echo "CANN environment script not found: ${CANN_ENV}" >&2
  exit 1
fi

if ! "${CONDA_EXE}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  "${CONDA_EXE}" create --yes --override-channels --channel conda-forge \
    --name "${ENV_NAME}" python=3.10 pip
else
  echo "Conda environment ${ENV_NAME} already exists; keeping it." >&2
fi

set +u
source "${CANN_ENV}"
set -u

run_in_env() {
  "${CONDA_EXE}" run --no-capture-output -n "${ENV_NAME}" "$@"
}

run_in_env python -m pip install --upgrade 'pip>=24' 'setuptools>=68,<82' wheel

if [[ -n "${TORCH_WHL}" || -n "${TORCH_NPU_WHL}" ]]; then
  if [[ -z "${TORCH_WHL}" || -z "${TORCH_NPU_WHL}" ]]; then
    echo "Set both TORCH_WHL and TORCH_NPU_WHL, or set neither and use an index." >&2
    exit 1
  fi
  [[ -f "${TORCH_WHL}" ]] || { echo "Missing TORCH_WHL: ${TORCH_WHL}" >&2; exit 1; }
  [[ -f "${TORCH_NPU_WHL}" ]] || { echo "Missing TORCH_NPU_WHL: ${TORCH_NPU_WHL}" >&2; exit 1; }
  run_in_env python -m pip install "${TORCH_WHL}" "${TORCH_NPU_WHL}"
elif [[ -n "${TORCH_NPU_INDEX_URL}" ]]; then
  if [[ -n "${TORCH_INDEX_URL}" ]]; then
    run_in_env python -m pip install --index-url "${TORCH_INDEX_URL}" torch
  else
    echo "TORCH_INDEX_URL is empty; install the matching torch package before torch_npu." >&2
    exit 1
  fi
  run_in_env python -m pip install --index-url "${TORCH_NPU_INDEX_URL}" torch_npu
else
  run_in_env python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    'torch==2.12.0'
  run_in_env python -m pip install 'torch-npu==2.12.0.post2'
fi

run_in_env python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r "${ROOT_DIR}/requirements.txt"
echo "Environment ${ENV_NAME} is ready for the remaining checks."
echo "Next: source ${SCRIPT_DIR}/activate_shj.sh && ${SCRIPT_DIR}/check_shj_env.sh"
