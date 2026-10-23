#!/usr/bin/env python3
"""Compare two LoRA checkpoints' generations on the same 4-task test file.

Metrics: JSON validity for JSON-answer tasks, risk_level agreement with the
reference, and scene-description recall of the obstacle direction word.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path


def load(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def parse_json_safe(text):
    if not isinstance(text, str) or not text.strip():
        return None
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", candidate, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
    return None


def score(rows):
    total = len(rows)
    errors = [r for r in rows if r.get("status") != "ok"]
    json_tasks = [r for r in rows if r["task"] != "scene_description"]
    json_valid = [
        r for r in json_tasks if parse_json_safe(r.get("prediction")) is not None
    ]
    risk = [r for r in rows if r["task"] == "risk_analysis" and r.get("status") == "ok"]
    risk_hit = 0
    direction_hit = 0
    for r in risk:
        pred = parse_json_safe(r.get("prediction")) or {}
        ref = parse_json_safe(r["reference_answer"]) or {}
        if str(pred.get("risk_level", "")).lower() == str(ref.get("risk_level", "")).lower():
            risk_hit += 1
        rec = str(pred.get("recommended_direction", "")).lower()
        if rec and rec == str(ref.get("recommended_direction", "")).lower():
            direction_hit += 1
    scene = [r for r in rows if r["task"] == "scene_description" and r.get("status") == "ok"]
    scene_hit = 0
    for r in scene:
        ref = r["reference_answer"]
        ref_dir = None
        for word in ("前方", "左前方", "右前方", "左侧", "右侧", "左后方", "右后方", "后方"):
            if word in ref:
                ref_dir = word
                break
        if ref_dir and ref_dir in (r.get("prediction") or ""):
            scene_hit += 1
    return {
        "rows": total,
        "gen_errors": len(errors),
        "json_valid_rate": round(len(json_valid) / max(1, len(json_tasks)), 3),
        "risk_level_match": f"{risk_hit}/{len(risk)}",
        "recommended_direction_match": f"{direction_hit}/{len(risk)}",
        "scene_direction_recall": f"{scene_hit}/{len(scene)}",
    }


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    a_rows, b_rows = load(a_path), load(b_path)
    b_by_id = {r["id"]: r for r in b_rows}
    print("config A (6 tasks):", json.dumps(score(a_rows), ensure_ascii=False))
    print("config B (4 tasks):", json.dumps(score(b_rows), ensure_ascii=False))

    # 分布 sanity check：两边预测的 risk_level 分布
    for name, rows in (("A", a_rows), ("B", b_rows)):
        dist = Counter()
        for r in rows:
            if r["task"] == "risk_analysis":
                parsed = parse_json_safe(r.get("prediction")) or {}
                dist[str(parsed.get("risk_level"))] += 1
        print(f"risk_level distribution {name}: {dict(dist)}")


if __name__ == "__main__":
    main()
