#!/usr/bin/env python3
"""Convert the local corpus format to MS-SWIFT's Qwen-VL chat format."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_no}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No samples found in {path}")
    return rows


VALID_SPLITS = ("train", "val", "test")


def corpus_split_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]] | None:
    """Group rows by the per-trial ``split`` field written by
    build_vlm_training_corpus (one split per trial), or return None when that
    field cannot be used.  Re-splitting such a corpus chronologically would put
    frames of the same trial into train and val/test at once."""
    if not rows or not all(row.get("split") in VALID_SPLITS for row in rows):
        return None
    groups: dict[str, list[dict[str, Any]]] = {split: [] for split in VALID_SPLITS}
    trial_split: dict[int, str] = {}
    mixed_trials: set[int] = set()
    for row in rows:
        split = row["split"]
        groups[split].append(row)
        trial_id = int(row["trial_id"])
        previous = trial_split.setdefault(trial_id, split)
        if previous != split:
            mixed_trials.add(trial_id)
    if not groups["train"] or not groups["val"]:
        return None
    if mixed_trials:
        print(
            f"Warning: trials span multiple corpus splits: {sorted(mixed_trials)}",
            file=sys.stderr,
        )
    return groups


def split_frame_groups(
    rows: list[dict[str, Any]], train_fraction: float, val_fraction: float
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            key = (int(row["trial_id"]), int(row["frame_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Every sample must have integer trial_id and frame_index") from exc
        grouped[key].append(row)

    by_trial: dict[int, list[tuple[int, list[dict[str, Any]]]]] = defaultdict(list)
    for (trial_id, frame_index), samples in grouped.items():
        by_trial[trial_id].append((frame_index, samples))

    result: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for trial_id in sorted(by_trial):
        frames = sorted(by_trial[trial_id], key=lambda item: item[0])
        if len(frames) < 3:
            raise ValueError(
                f"Trial {trial_id} has only {len(frames)} unique frames; at least 3 are needed"
            )
        train_end = max(1, min(len(frames) - 2, math.floor(len(frames) * train_fraction)))
        val_count = max(1, round(len(frames) * val_fraction))
        val_end = min(len(frames) - 1, train_end + val_count)
        if val_end <= train_end:
            val_end = train_end + 1

        for index, (_, samples) in enumerate(frames):
            split = "train" if index < train_end else "val" if index < val_end else "test"
            result[split].extend(samples)
    return result


def convert_sample(row: dict[str, Any], image_root: Path) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"Sample {row.get('id', '<unknown>')} has invalid messages")
    if not any(
        isinstance(message, dict)
        and message.get("role") == "user"
        and "<image>" in str(message.get("content", ""))
        for message in messages
    ):
        raise ValueError(f"Sample {row.get('id', '<unknown>')} is missing the <image> marker")

    image_value = row.get("image")
    if not isinstance(image_value, str) or not image_value:
        raise ValueError(f"Sample {row.get('id', '<unknown>')} has no image path")
    image_path = Path(image_value)
    if not image_path.is_absolute():
        image_path = image_root / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image for sample {row.get('id', '<unknown>')} not found: {image_path}")

    clean_messages = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            raise ValueError(f"Sample {row.get('id', '<unknown>')} has an unsupported message")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"Sample {row.get('id', '<unknown>')} has non-text message content")
        clean_messages.append({"role": message["role"], "content": content})
    return {"messages": clean_messages, "images": [str(image_path)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="all.jsonl or another corpus JSONL")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--split-source",
        choices=("auto", "corpus", "chronological"),
        default="auto",
        help="auto: use the corpus per-trial split field when present, else "
        "split chronologically; corpus/chronological force one behaviour",
    )
    parser.add_argument(
        "--exclude-pseudo-grounding",
        action="store_true",
        help="drop unreviewed pseudo-label samples for obstacle_attributes/object_grounding",
    )
    args = parser.parse_args()

    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be between 0 and 1")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between 0 and 1")
    if args.train_fraction + args.val_fraction >= 1.0:
        parser.error("train and validation fractions must sum to less than 1")

    source = args.input.resolve()
    rows = read_jsonl(source)
    if args.exclude_pseudo_grounding:
        rows = [
            row
            for row in rows
            if not (
                row.get("task") in {"obstacle_attributes", "object_grounding"}
                and "pseudo" in str(row.get("label_quality", ""))
            )
        ]
    if not rows:
        raise ValueError("No samples remain after filtering")

    # Keep all instructions for one image together so a frame cannot leak across splits.
    splits: dict[str, list[dict[str, Any]]] | None = None
    if args.split_source in ("auto", "corpus"):
        splits = corpus_split_groups(rows)
        if splits is None and args.split_source == "corpus":
            raise ValueError(
                "--split-source=corpus but the input rows have no usable per-trial "
                "split field (need non-empty train and val)"
            )
    split_method = "corpus trial-level split field"
    if splits is None:
        split_method = "chronological contiguous frame groups per trial"
        splits = split_frame_groups(rows, args.train_fraction, args.val_fraction)
    image_root = source.parent
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trial_splits: dict[str, str] = {}
    for split, split_rows in splits.items():
        for row in split_rows:
            trial_splits[str(int(row["trial_id"]))] = split

    summary: dict[str, Any] = {
        "source": str(source),
        "split_method": split_method,
        "trial_splits": trial_splits,
        "counts": {},
        "task_counts": {},
        "frame_counts": {},
        "label_quality_counts": {},
        "excluded_pseudo_grounding": args.exclude_pseudo_grounding,
    }
    for split, split_rows in splits.items():
        output_path = args.output_dir / f"{split}.jsonl"
        task_counts: Counter[str] = Counter()
        quality_counts: Counter[str] = Counter()
        frame_ids = set()
        with output_path.open("w", encoding="utf-8") as handle:
            for row in split_rows:
                converted = convert_sample(row, image_root)
                handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
                task_counts[str(row.get("task", "unknown"))] += 1
                quality_counts[str(row.get("label_quality", "unknown"))] += 1
                frame_ids.add((int(row["trial_id"]), int(row["frame_index"])))
        summary["counts"][split] = len(split_rows)
        summary["task_counts"][split] = dict(sorted(task_counts.items()))
        summary["frame_counts"][split] = len(frame_ids)
        summary["label_quality_counts"][split] = dict(sorted(quality_counts.items()))

    summary_path = args.output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nPrepared data: {args.output_dir}")


if __name__ == "__main__":
    main()
