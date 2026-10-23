#!/usr/bin/env python3
"""Run one Qwen2.5-VL inference with an optional LoRA adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def get_device() -> tuple[str, torch.dtype]:
    try:
        import torch_npu  # noqa: F401  # Registers the Ascend backend.

        if hasattr(torch, "npu"):
            if torch.npu.is_available():
                return "npu:0", torch.bfloat16
    except (ImportError, RuntimeError):
        pass
    if torch.cuda.is_available():
        return "cuda:0", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    raise RuntimeError("Neither Ascend NPU nor CUDA is available")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", default="描述图中与无人机飞行安全有关的障碍物，并说明不确定之处。")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", type=Path, help="LoRA adapter directory, for example output/.../checkpoint-*")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    image = args.image.resolve()
    if not image.is_file():
        parser.error(f"image not found: {image}")
    try:
        device, dtype = get_device()
    except RuntimeError as exc:
        parser.error(str(exc))

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype)
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image)},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_trimmed = [
        output[len(input_ids) :] for input_ids, output in zip(inputs["input_ids"], generated)
    ]
    answer = processor.batch_decode(
        generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    print(answer.strip())


if __name__ == "__main__":
    main()
