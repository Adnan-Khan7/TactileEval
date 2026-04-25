#!/usr/bin/env python3
"""Compare individual worker votes against the finalized majority labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

from build_final_dataset import (
    infer_task_spec,
    parse_answer_payload,
    url_to_relative_path,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent


def load_gold_labels(records_path: Path) -> Dict[Tuple[str, str], int]:
    gold: Dict[Tuple[str, str], int] = {}
    with records_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["pair_id"], rec["option_id"])
            gold[key] = int(rec["label"])
    return gold


def iter_votes(row: Dict[str, str], spec):
    answers = parse_answer_payload(row.get("Answer.taskAnswers", ""))
    idx = 1
    while True:
        nat_key = f"Input.q{idx}_natural_url"
        tac_key = f"Input.q{idx}_tactile_url"
        if nat_key not in row and tac_key not in row:
            break
        nat_url = row.get(nat_key, "")
        tac_url = row.get(tac_key, "")
        answer = answers.get(f"question{idx}") if answers else None
        idx += 1
        if not nat_url or not tac_url or not isinstance(answer, dict):
            continue
        natural_rel = url_to_relative_path(nat_url)
        tactile_rel = url_to_relative_path(tac_url)
        if not natural_rel or not tactile_rel:
            continue
        pair_id = f"{natural_rel}::{tactile_rel}"
        yield pair_id, answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DATASET_DIR / "processed" / "records_full.jsonl")
    parser.add_argument("--csv-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DATASET_DIR / "processed" / "worker_vote_accuracy.csv")
    parser.add_argument("--tasks", nargs="*", help="Optional subset of task IDs to process")
    args = parser.parse_args()

    gold = load_gold_labels(args.records)
    if not gold:
        raise SystemExit("No gold labels found; run build_final_dataset.py first.")

    per_worker = defaultdict(lambda: {"correct": 0, "total": 0})
    per_task = defaultdict(lambda: {"correct": 0, "total": 0})

    csv_paths = sorted(args.csv_dir.glob("F*Q*.csv"))
    if args.tasks:
        wanted = set(args.tasks)
        csv_paths = [p for p in csv_paths if p.stem in wanted]

    for csv_path in csv_paths:
        task_id = csv_path.stem
        spec = infer_task_spec(task_id)
        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                worker_id = row.get("WorkerId", "unknown")
                for pair_id, answer in iter_votes(row, spec):
                    for opt in spec.option_ids:
                        if opt not in answer:
                            continue
                        key = (pair_id, opt)
                        gold_label = gold.get(key)
                        if gold_label is None:
                            continue
                        vote = 1 if answer.get(opt) else 0
                        per_worker[worker_id]["total"] += 1
                        per_task[task_id]["total"] += 1
                        if vote == gold_label:
                            per_worker[worker_id]["correct"] += 1
                            per_task[task_id]["correct"] += 1

    total_votes = sum(stats["total"] for stats in per_worker.values())
    total_correct = sum(stats["correct"] for stats in per_worker.values())
    overall_acc = total_correct / total_votes if total_votes else 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["worker_id", "votes", "correct", "accuracy"])
        for worker, stats in sorted(per_worker.items(), key=lambda kv: kv[1]["total"], reverse=True):
            votes = stats["total"]
            if not votes:
                continue
            acc = stats["correct"] / votes
            writer.writerow([worker, votes, stats["correct"], f"{acc:.4f}"])

    print(f"Overall individual-vote accuracy: {overall_acc:.3%} ({total_correct}/{total_votes})")
    print("Per-task accuracy:")
    for task_id, stats in sorted(per_task.items()):
        if not stats["total"]:
            continue
        acc = stats["correct"] / stats["total"]
        print(f"  {task_id}: {acc:.3%} ({stats['correct']}/{stats['total']})")
    print(f"Worker-level breakdown saved to {args.output}")


if __name__ == "__main__":
    main()
