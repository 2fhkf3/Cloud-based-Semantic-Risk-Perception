#!/usr/bin/env bash
set -euo pipefail

CONDA_EXE="${SHJ_CONDA_EXE:-/home/shj/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-shj}"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann-9.1.0/set_env.sh}"

set +u
source "${CANN_ENV}"
set -u
"${CONDA_EXE}" run --no-capture-output -n "${ENV_NAME}" python - <<'PY'
import importlib
import sys

print("Python:", sys.version.split()[0])
for name in ("torch", "torch_npu", "torchvision", "transformers", "swift", "qwen_vl_utils", "peft"):
    try:
        module = importlib.import_module(name)
        print(f"{name}: OK {getattr(module, '__version__', '')}".rstrip())
    except Exception as exc:
        print(f"{name}: ERROR {exc}")
        raise

import torch
import torch_npu  # noqa: F401

assert torch.npu.is_available(), "torch.npu.is_available() is False"
print("NPU count:", torch.npu.device_count())
print("NPU current:", torch.npu.current_device())
PY

echo "npu-smi summary:"
npu-smi info | sed -n '1,24p'
