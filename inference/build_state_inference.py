#!/usr/bin/env python3
"""Build image-plus-seven-state records for Qwen2.5-VL inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STATE_NAMES = ("x", "y", "d_goal", "d_front", "d_left", "d_back", "d_right")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No records found in {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Every record must be a JSON object: {path}")
    return rows


def state_text(state: list[float]) -> str:
    values = ",".join(f"{value:g}" for value in state)
    names = ",".join(STATE_NAMES)
    return f"state=[{names}]=[{values}]"


def get_user_prompt(messages: list[dict[str, Any]], sample_id: str) -> str:
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content.replace("<image>", "", 1).strip()
    raise ValueError(f"No text user prompt found for {sample_id}")


def get_reference(messages: list[dict[str, Any]], sample_id: str) -> str:
    for message in messages:
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return message["content"]
    raise ValueError(f"No assistant reference found for {sample_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original all.jsonl task records")
    parser.add_argument("--state-annotations", type=Path, required=True, help="scene_annotations.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.input.resolve()
    annotation_path = args.state_annotations.resolve()
    rows = read_jsonl(source)
    annotations = read_jsonl(annotation_path)
    states: dict[tuple[int, int], list[float]] = {}
    for row in annotations:
        try:
            key = (int(row["trial_id"]), int(row["frame_index"]))
            state = [float(value) for value in row["state"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid state annotation: {row.get('id', '<unknown>')}") from exc
        if len(state) != 7:
            raise ValueError(f"State must have 7 values: {row.get('id', '<unknown>')}")
        if key in states:
            raise ValueError(f"Duplicate state annotation for trial/frame {key}")
        states[key] = state

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for row in rows:
        sample_id = str(row.get("id", "<unknown>"))
        try:
            key = (int(row["trial_id"]), int(row["frame_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing trial/frame in {sample_id}") from exc
        if key not in states:
            raise ValueError(f"No seven-value state for {sample_id} at {key}")
        image_value = row.get("image")
        if not isinstance(image_value, str):
            raise ValueError(f"Missing image path in {sample_id}")
        image_path = Path(image_value)
        if not image_path.is_absolute():
            image_path = source.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found for {sample_id}: {image_path}")

        state = states[key]
        prompt = f"{state_text(state)}。\n{get_user_prompt(row.get('messages', []), sample_id)}"
        output_rows.append(
            {
                "id": sample_id,
                "task": row.get("task"),
                "trial_id": key[0],
                "frame_index": key[1],
                "timestamp_s": row.get("timestamp_s"),
                "image": str(image_path),
                "state": state,
                "state_names": list(STATE_NAMES),
                "prompt": prompt,
                "reference_answer": get_reference(row.get("messages", []), sample_id),
            }
        )

    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(output_rows), "state_values_per_record": 7}, ensure_ascii=False))


if __name__ == "__main__":
    main()
