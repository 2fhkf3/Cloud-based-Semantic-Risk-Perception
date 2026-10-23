#!/usr/bin/env python3
"""Batch-drive ODA frame extraction over many trials.

Builds the eligible-trial list from ``trial_overview.csv`` (skipping trials
without video, without obstacle coordinates, or known to be corrupt), then
extracts frames in parallel with a process pool.  Already-extracted trials are
resumed for free; failures are recorded without stopping the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_multimodal_dataset import (  # noqa: E402
    ExtractOptions,
    TrialSkipError,
    extract_trial,
    load_overview_groups,
)

# Corrupt per the ODA dataset Readme (section 2.3).
KNOWN_CORRUPT_TRIALS = {1306, 1321, 1344}


@dataclass
class TrialPlan:
    trial_id: int
    runnable: bool
    reason: str  # empty when runnable, otherwise why it is skipped


def parse_trial_ids(spec: str) -> List[int]:
    ids: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(chunk))
    return ids


def plan_trials(
    dataset_root: Path,
    overview_groups: Dict[int, List[Dict[str, str]]],
    trial_filter: Optional[Sequence[int]] = None,
    include_no_video: bool = False,
) -> List[TrialPlan]:
    plans: List[TrialPlan] = []
    for trial_id in sorted(overview_groups):
        if trial_filter is not None and trial_id not in set(trial_filter):
            continue
        if trial_id in KNOWN_CORRUPT_TRIALS:
            plans.append(TrialPlan(trial_id, False, "corrupt_trial"))
            continue
        rows = overview_groups[trial_id]
        if any(
            not (row.get("Obstacle x") or "").strip()
            or not (row.get("Obstacle z") or "").strip()
            for row in rows
        ):
            plans.append(TrialPlan(trial_id, False, "missing_obstacle_coords"))
            continue
        if not include_no_video:
            try:
                incl_video = int(float(rows[0].get("Incl. Video") or "0"))
            except ValueError:
                incl_video = 0
            if incl_video != 1:
                plans.append(TrialPlan(trial_id, False, "no_rgb_video"))
                continue
        if not (dataset_root / str(trial_id) / f"{trial_id}.avi").is_file():
            plans.append(TrialPlan(trial_id, False, "missing_avi_file"))
            continue
        plans.append(TrialPlan(trial_id, True, ""))
    return plans


def _worker(task: dict) -> dict:
    cv2 = __import__("cv2")
    cv2.setNumThreads(0)  # avoid CPU oversubscription across workers

    trial_id = task["trial_id"]
    started = time.monotonic()
    try:
        overview_groups = load_overview_groups(Path(task["dataset_root"]) / "trial_overview.csv")
        options = ExtractOptions(
            sample_every=task["sample_every"],
            jpeg_quality=task["jpeg_quality"],
        )
        metadata = extract_trial(
            trial_id,
            Path(task["dataset_root"]),
            overview_groups,
            Path(task["output_root"]) / f"trial_{trial_id}",
            options,
        )
        return {
            "trial_id": trial_id,
            "status": "ok",
            "frames": metadata["exported_frames"],
            "decoded": metadata["decoded_video_frames"],
            "obstacles": len(metadata["range"]["obstacles_xyr"]),
            "seconds": round(time.monotonic() - started, 1),
            "reason": "",
        }
    except TrialSkipError as exc:
        return {
            "trial_id": trial_id,
            "status": "skipped",
            "frames": 0,
            "decoded": 0,
            "obstacles": 0,
            "seconds": round(time.monotonic() - started, 1),
            "reason": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - record and keep the batch going
        return {
            "trial_id": trial_id,
            "status": "failed",
            "frames": 0,
            "decoded": 0,
            "obstacles": 0,
            "seconds": round(time.monotonic() - started, 1),
            "reason": f"{type(exc).__name__}: {exc}",
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="directory holding trial_overview.csv and <id>/ trial dirs")
    parser.add_argument("--output-root", type=Path,
                        default=Path("/home/shj/data/data-new/multimodal_output"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--trial-ids", type=str,
                        help='optional subset, e.g. "3,10,184-220"')
    parser.add_argument("--limit", type=int,
                        help="process only the first N runnable trials after filtering")
    parser.add_argument("--sample-every", type=int, default=1,
                        help="keep every Nth video frame (1 = all frames, as in trial_3)")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true",
                        help="re-extract trials that already have annotations.jsonl")
    parser.add_argument("--include-no-video", action="store_true",
                        help="also plan trials flagged Incl. Video=0 (they fail into the skip log)")
    parser.add_argument("--list-only", action="store_true",
                        help="print the eligibility plan and exit without extracting")
    parser.add_argument("--log-dir", type=Path,
                        help="default: <output-root>/batch_logs/run_<timestamp>")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    overview_path = dataset_root / "trial_overview.csv"
    if not overview_path.is_file():
        raise FileNotFoundError(overview_path)
    overview_groups = load_overview_groups(overview_path)

    trial_filter = parse_trial_ids(args.trial_ids) if args.trial_ids else None
    plans = plan_trials(dataset_root, overview_groups, trial_filter, args.include_no_video)

    runnable = [plan for plan in plans if plan.runnable]
    if not args.overwrite:
        before = len(runnable)
        runnable = [
            plan
            for plan in runnable
            if not (args.output_root / f"trial_{plan.trial_id}" / "annotations.jsonl").is_file()
        ]
        resume_skipped = before - len(runnable)
    else:
        resume_skipped = 0
    if args.limit is not None:
        runnable = runnable[: args.limit]

    reason_counts: Dict[str, int] = {}
    for plan in plans:
        if not plan.runnable:
            reason_counts[plan.reason] = reason_counts.get(plan.reason, 0) + 1

    print(
        f"Overview trials: {len(overview_groups)}; runnable: {len(runnable)} "
        f"(resume-skipped: {resume_skipped}); planned skips: {reason_counts}"
    )
    if args.list_only:
        for plan in plans:
            print(f"{plan.trial_id}\t{'RUN' if plan.runnable else 'SKIP:' + plan.reason}")
        return 0
    if not runnable:
        print("Nothing to extract.")
        return 0

    log_dir = (
        args.log_dir.resolve()
        if args.log_dir is not None
        else args.output_root.resolve() / "batch_logs" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "trial_id": plan.trial_id,
            "dataset_root": str(dataset_root),
            "output_root": str(args.output_root.resolve()),
            "sample_every": args.sample_every,
            "jpeg_quality": args.jpeg_quality,
        }
        for plan in runnable
    ]

    results: List[dict] = []
    started = time.monotonic()
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        futures = [pool.submit(_worker, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(
                f"[{index}/{len(tasks)}] trial {result['trial_id']} {result['status']} "
                f"frames={result['frames']} obstacles={result['obstacles']} "
                f"{result['seconds']}s {result['reason']}".rstrip(),
                flush=True,
            )

    results.sort(key=lambda item: item["trial_id"])
    manifest_path = log_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["trial_id", "status", "frames", "decoded", "obstacles", "seconds", "reason"],
        )
        writer.writeheader()
        writer.writerows(results)

    failures = [result for result in results if result["status"] == "failed"]
    skipped = [result for result in results if result["status"] == "skipped"]
    ok = [result for result in results if result["status"] == "ok"]
    summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(args.output_root.resolve()),
        "requested": len(tasks),
        "ok": len(ok),
        "skipped": len(skipped),
        "failed": len(failures),
        "resume_skipped": resume_skipped,
        "planned_skips": reason_counts,
        "total_frames": sum(result["frames"] for result in results),
        "wall_seconds": round(time.monotonic() - started, 1),
        "manifest": str(manifest_path),
    }
    (log_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        print(f"{len(failures)} trials failed; see {manifest_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
