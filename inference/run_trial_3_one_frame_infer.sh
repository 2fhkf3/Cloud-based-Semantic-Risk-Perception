#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT="${INPUT:-${ROOT_DIR}/artifacts/inference/trial_3_one_frame_with_state.jsonl}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/artifacts/inference/trial_3_one_frame_predictions.jsonl}"
MODEL="${MODEL:-${ROOT_DIR}/models/Qwen--Qwen2.5-VL-3B-Instruct/snapshots/master}"
ADAPTER="${ADAPTER:-${ROOT_DIR}/artifacts/output/qwen2.5-vl-3b-lora/checkpoint-21}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

if [[ ! -s "${INPUT}" ]]; then
  echo "Inference input not found or empty: ${INPUT}" >&2
  exit 1
fi
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "Model config not found: ${MODEL}/config.json" >&2
  exit 1
fi
if [[ ! -f "${ADAPTER}/adapter_config.json" ]]; then
  echo "LoRA adapter not found: ${ADAPTER}" >&2
  exit 1
fi

if [[ -f "${ROOT_DIR}/scripts/activate_shj.sh" ]]; then
  # Keep the configured NPU selection (default: NPU 4) and Python environment.
  source "${ROOT_DIR}/scripts/activate_shj.sh"
fi

echo "Running six tasks from: ${INPUT}"
echo "Model: ${MODEL}"
echo "Adapter: ${ADAPTER}"
echo "Output: ${OUTPUT}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/infer_jsonl.py" \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --model "${MODEL}" \
  --adapter "${ADAPTER}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
