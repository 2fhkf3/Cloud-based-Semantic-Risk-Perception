#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# Keep the default self-contained; override DATA_JSONL for an external corpus.
DATA_JSONL="${DATA_JSONL:-${ROOT_DIR}/data/trial_3/all.jsonl}"
PREPARED_DIR="${PREPARED_DIR:-${ROOT_DIR}/artifacts/prepared/trial_3}"
MODEL="${MODEL:-${ROOT_DIR}/models/Qwen--Qwen2.5-VL-3B-Instruct}"
TEMPLATE="${TEMPLATE:-qwen2_5_vl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/artifacts/output/qwen2.5-vl-3b-lora}"
LOG_FILE="${LOG_FILE:-${ROOT_DIR}/artifacts/logs/train_$(date +%Y%m%d_%H%M%S).log}"
DEVICE_TYPE="${DEVICE_TYPE:-npu}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MAX_PIXELS="${MAX_PIXELS:-401408}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
# "auto" derives GRAD_ACCUM so cards x batch x accum ~= EFFECTIVE_BATCH.
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-32}"
GRAD_ACCUM="${GRAD_ACCUM:-auto}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_STEPS="${EVAL_STEPS:-250}"
SAVE_STEPS="${SAVE_STEPS:-250}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
SEED="${SEED:-42}"
# 断点续训：设为 checkpoint 目录（通常指向 .../last-checkpoint）即可从该步继续
RESUME_FROM="${RESUME_FROM:-}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
EXCLUDE_PSEUDO_GROUNDING="${EXCLUDE_PSEUDO_GROUNDING:-false}"

mkdir -p "$(dirname -- "${LOG_FILE}")"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'status=$?; if (( status == 0 )); then echo "Training script finished successfully."; else echo "Training script failed with exit code ${status}."; fi' EXIT
echo "Log file: ${LOG_FILE}"

if [[ -d "${MODEL}" ]]; then
  if [[ ! -f "${MODEL}/config.json" && -f "${MODEL}/snapshots/master/config.json" ]]; then
    MODEL="${MODEL}/snapshots/master"
  fi
  if [[ ! -f "${MODEL}/config.json" ]]; then
    echo "Local model directory has no config.json: ${MODEL}" >&2
    exit 1
  fi
  if ! python - "${MODEL}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
incomplete = sorted(model_dir.glob("*.incomplete"))
if incomplete:
    print("Incomplete model downloads found:", file=sys.stderr)
    for path in incomplete:
        print(f"  {path.name}", file=sys.stderr)
    raise SystemExit(1)

index_path = model_dir / "model.safetensors.index.json"
if index_path.is_file():
    with index_path.open(encoding="utf-8") as handle:
        shards = sorted(set(json.load(handle)["weight_map"].values()))
    missing = [name for name in shards if not (model_dir / name).is_file()]
    if missing:
        print("Model weight shards are missing:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        raise SystemExit(1)
elif not any(model_dir.glob("*.safetensors")) and not any(model_dir.glob("pytorch_model*.bin")):
    print(f"No model weight files found in {model_dir}", file=sys.stderr)
    raise SystemExit(1)
PY
  then
    echo "Local model files are incomplete: ${MODEL}" >&2
    exit 1
  fi
elif [[ "${MODEL}" == /* ]]; then
  echo "Local model directory not found: ${MODEL}" >&2
  exit 1
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "swift was not found. Activate the training environment and install requirements.txt first." >&2
  exit 1
fi
if [[ ! -f "${DATA_JSONL}" ]]; then
  echo "Corpus JSONL not found: ${DATA_JSONL}" >&2
  exit 1
fi
if [[ ! -s "${PREPARED_DIR}/train.jsonl" || ! -s "${PREPARED_DIR}/val.jsonl" ]]; then
  echo "Prepared train/validation datasets are missing under ${PREPARED_DIR}; preparing them now."
  PREPARE_DATASET=true
else
  PREPARE_DATASET="${PREPARE_DATASET:-true}"
fi
if [[ "${PREPARE_DATASET}" != "true" && "${PREPARE_DATASET}" != "false" ]]; then
  echo "PREPARE_DATASET must be 'true' or 'false': ${PREPARE_DATASET}" >&2
  exit 1
fi

if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer: ${NPROC_PER_NODE}" >&2
  exit 1
fi
if [[ "${DEVICE_TYPE}" == "npu" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4}"
  VISIBLE_DEVICE_COUNT="$(awk -F, '{print NF}' <<< "${ASCEND_RT_VISIBLE_DEVICES}")"
  if [[ "${NPROC_PER_NODE}" -ne "${VISIBLE_DEVICE_COUNT}" ]]; then
    echo "NPROC_PER_NODE=${NPROC_PER_NODE} must match the ${VISIBLE_DEVICE_COUNT} visible NPU(s) in ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}." >&2
    exit 1
  fi
  CANN_ENV="${CANN_ENV:-/usr/local/Ascend/cann-9.1.0/set_env.sh}"
  if [[ -f "${CANN_ENV}" ]]; then
    # shellcheck disable=SC1090
    set +u
    source "${CANN_ENV}"
    set -u
  fi
  if ! python -c 'import torch; import torch_npu; assert torch.npu.is_available(), "NPU is unavailable"' >/dev/null 2>&1; then
    echo "PyTorch cannot access Ascend NPU. Install a torch/torch_npu pair matching CANN and check npu-smi." >&2
    exit 1
  fi
elif [[ "${DEVICE_TYPE}" == "cuda" ]]; then
  if ! python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"' >/dev/null 2>&1; then
    echo "PyTorch cannot access CUDA. Check the NVIDIA driver and install a CUDA-enabled PyTorch build." >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  VISIBLE_DEVICE_COUNT="$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
  if [[ "${NPROC_PER_NODE}" -ne "${VISIBLE_DEVICE_COUNT}" ]]; then
    echo "NPROC_PER_NODE=${NPROC_PER_NODE} must match the ${VISIBLE_DEVICE_COUNT} visible GPU(s) in CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
    exit 1
  fi
else
  echo "DEVICE_TYPE must be 'npu' or 'cuda': ${DEVICE_TYPE}" >&2
  exit 1
fi

if [[ "${GRAD_ACCUM}" == "auto" ]]; then
  PER_STEP=$((VISIBLE_DEVICE_COUNT * TRAIN_BATCH_SIZE))
  GRAD_ACCUM=$((EFFECTIVE_BATCH / PER_STEP))
  if (( GRAD_ACCUM < 1 )); then
    GRAD_ACCUM=1
  fi
  echo "GRAD_ACCUM resolved to ${GRAD_ACCUM}; effective batch = ${PER_STEP} x ${GRAD_ACCUM} = $((PER_STEP * GRAD_ACCUM))"
fi

if [[ "${PREPARE_DATASET}" == "true" ]]; then
  PREPARE_ARGS=(
    --input "${DATA_JSONL}"
    --output-dir "${PREPARED_DIR}"
  )
  if [[ "${EXCLUDE_PSEUDO_GROUNDING}" == "true" ]]; then
    PREPARE_ARGS+=(--exclude-pseudo-grounding)
  fi
  python "${ROOT_DIR}/pipeline/prepare_dataset.py" "${PREPARE_ARGS[@]}"
fi

export NPROC_PER_NODE
echo "Training model: ${MODEL}"
echo "Template: ${TEMPLATE}"
echo "Device: ${DEVICE_TYPE}; processes: ${NPROC_PER_NODE}; output: ${OUTPUT_DIR}"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
  if [[ ! -d "${RESUME_FROM}" ]]; then
    echo "RESUME_FROM checkpoint directory not found: ${RESUME_FROM}" >&2
    exit 1
  fi
  RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM}")
  echo "Resuming from checkpoint: ${RESUME_FROM}"
fi

swift sft \
  --model "${MODEL}" \
  --template "${TEMPLATE}" \
  --dataset "${PREPARED_DIR}/train.jsonl" \
  --val_dataset "${PREPARED_DIR}/val.jsonl" \
  --tuner_type lora \
  --torch_dtype "${TORCH_DTYPE}" \
  --tuner_backend peft \
  --target_modules all-linear \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --freeze_vit true \
  --gradient_checkpointing true \
  --max_pixels "${MAX_PIXELS}" \
  --max_length "${MAX_LENGTH}" \
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --seed "${SEED}" \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --logging_steps 5 \
  --dataloader_num_workers "${DATALOADER_WORKERS}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${OUTPUT_DIR}/runs" \
  --report_to tensorboard \
  "${RESUME_ARGS[@]}"
