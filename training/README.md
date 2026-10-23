# Training

`train.sh` is the single training entry point. It performs preflight checks, optionally calls `pipeline/prepare_dataset.py`, launches MS-SWIFT LoRA fine-tuning, and records the full log.

## What the training stage does

1. Resolves model, corpus, prepared-data, output, and log paths.
2. Checks model configuration and all weight shards.
3. Checks NPU/CUDA availability and visible-device/process consistency.
4. Validates or regenerates train/validation JSONL.
5. Runs Qwen2.5-VL supervised fine-tuning with LoRA and a frozen vision encoder.
6. Saves adapter checkpoints and TensorBoard logs.

The paper must report the base model revision, LoRA rank/alpha, target modules, frozen modules, image resolution limit, sequence length, batch size, gradient accumulation, learning rate, epochs, seed, hardware, and checkpoint selection rule.

Use environment variables to define an experiment. Avoid editing the script for one run; save the command in the experiment record.
