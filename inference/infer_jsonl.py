#!/usr/bin/env python3
"""Run Qwen2.5-VL inference for each record in an image/state JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def get_device() -> tuple[str, torch.dtype]:
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu:0", torch.bfloat16
    except (ImportError, RuntimeError):
        pass
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return "cuda:0", dtype
    raise RuntimeError("Neither Ascend NPU nor CUDA is available")


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def predict(model, processor, device: str, row: dict[str, Any], max_new_tokens: int) -> str:
    image = Path(row["image"]).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(f"image not found: {image}")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"record {row.get('id', '<unknown>')} has no prompt")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image)},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_trimmed = [
        output[len(input_ids):] for input_ids, output in zip(inputs["input_ids"], generated)
    ]
    return processor.batch_decode(
        generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSONL with image and prompt fields")
    parser.add_argument("--output", required=True, type=Path, help="JSONL for predictions")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", type=Path, help="LoRA checkpoint directory")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--start", type=int, default=0, help="zero-based record index")
    parser.add_argument("--limit", type=int, help="maximum number of records to process")
    args = parser.parse_args()

    rows = read_rows(args.input.resolve())
    if args.start < 0 or args.start >= len(rows):
        parser.error(f"--start must be between 0 and {len(rows) - 1}")
    end = len(rows) if args.limit is None else min(len(rows), args.start + max(args.limit, 0))
    if end <= args.start:
        parser.error("--limit must be positive")

    model_dir = args.model.expanduser().resolve()
    if not (model_dir / "config.json").is_file():
        parser.error(f"model config not found: {model_dir / 'config.json'}")
    device, dtype = get_device()
    print(f"Loading model: {model_dir}")
    print(f"Device: {device}; dtype: {dtype}; records: {args.start}:{end}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=dtype)
    if args.adapter:
        adapter_dir = args.adapter.expanduser().resolve()
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index in range(args.start, end):
            row = rows[index]
            result = dict(row)
            result["record_index"] = index
            try:
                result["prediction"] = predict(model, processor, device, row, args.max_new_tokens)
                result["status"] = "ok"
                print(f"[{index + 1}/{end}] {row.get('task', '<unknown>')}: ok")
            except Exception as exc:  # Keep later tasks running if one record is malformed.
                result["status"] = "error"
                result["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{index + 1}/{end}] {row.get('task', '<unknown>')}: {result['error']}")
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
    print(f"Saved predictions: {args.output.resolve()}")


if __name__ == "__main__":
    main()
