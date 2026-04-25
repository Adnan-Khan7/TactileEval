#!/usr/bin/env python3
"""Build balanced JSONL splits from refreshed AMT CSV exports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import unquote, urlparse
import random

NON_ACTION_OPTIONS = {"all_correct", "no_line_issues", "no_issues_good", "good"}


def url_to_relative_path(url: str) -> str | None:
    if not url:
        return None
    path = unquote(urlparse(url).path)
    parts = [p for p in path.split("/") if p]
    if not parts:
        return None
    if "Tactile" in parts:
        start = parts.index("Tactile") - 1
    elif "Natural" in parts:
        start = parts.index("Natural") - 1
    else:
        start = len(parts) - 3
    start = max(start, 0)
    return "/".join(parts[start:])


def load_release_records(release_dir: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    mapping: Dict[str, Dict[str, Dict[str, str]]] = {}
    for jsonl in sorted(release_dir.glob("*_amt.jsonl")):
        with jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tactile = rec["tactile_image"]
                entry = mapping.setdefault(tactile, {
                    "natural": rec.get("natural_image"),
                    "task_family": rec.get("task_family"),
                    "options": {},
                })
                entry["options"][rec["option_id"]] = rec.get("option_description", rec["option_id"])
    return mapping


def assign_split(pair_id: str) -> str:
    digest = hashlib.sha1(pair_id.encode("utf-8")).hexdigest()
    val = int(digest[:8], 16) / 0xFFFFFFFF
    if val < 0.8:
        return "train"
    if val < 0.9:
        return "val"
    return "test"


def build_records(csv_dir: Path, release_map: Dict[str, Dict[str, Dict[str, str]]], threshold: float) -> List[dict]:
    records: List[dict] = []
    for csv_path in sorted(csv_dir.glob("F1*.csv")):
        family = csv_path.stem.split("_")[0]  # handle names like F1QT_
        aggregated: Dict[str, Dict[str, List[bool]]] = {}
        naturals: Dict[str, str] = {}
        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if (row.get("AssignmentStatus", "").lower() != "approved"):
                    continue
                answers_raw = (row.get("Answer.taskAnswers", "") or "").strip()
                if not answers_raw:
                    continue
                try:
                    parsed = json.loads(answers_raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, list) and parsed:
                    parsed = parsed[0]
                q_idx = 1
                while f"Input.q{q_idx}_tactile_url" in row:
                    tactile_url = row.get(f"Input.q{q_idx}_tactile_url", "")
                    natural_url = row.get(f"Input.q{q_idx}_natural_url", "")
                    answers = parsed.get(f"question{q_idx}")
                    q_idx += 1
                    if not tactile_url or not isinstance(answers, dict):
                        continue
                    tactile_rel = url_to_relative_path(tactile_url)
                    natural_rel = url_to_relative_path(natural_url)
                    if not tactile_rel:
                        continue
                    entry = aggregated.setdefault(tactile_rel, {})
                    naturals.setdefault(tactile_rel, natural_rel)
                    for option_id, value in answers.items():
                        entry.setdefault(option_id, []).append(bool(value))
        for tactile_rel, opts in aggregated.items():
            rel_info = release_map.get(tactile_rel, {})
            natural_rel = rel_info.get("natural") or naturals.get(tactile_rel)
            for option_id, votes in opts.items():
                total = len(votes)
                if not total:
                    continue
                vote_fraction = sum(1 for v in votes if v) / total
                label = int(vote_fraction >= threshold)
                option_desc = rel_info.get("options", {}).get(option_id, option_id)
                record = {
                    "pair_id": f"{natural_rel or 'unknown'}::{tactile_rel}",
                    "task_family": family,
                    "option_id": option_id,
                    "option_description": option_desc,
                    "natural_image": natural_rel,
                    "tactile_image": tactile_rel,
                    "vote_fraction": vote_fraction,
                    "votes_total": total,
                    "label": label,
                }
                records.append(record)
    return records


def balance_train(records: List[dict]) -> List[dict]:
    buckets: Dict[Tuple[str, str, int], List[dict]] = defaultdict(list)
    for rec in records:
        key = (rec["task_family"], rec["option_id"], rec["label"])
        buckets[key].append(rec)
    grouped: Dict[Tuple[str, str], Dict[int, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for (family, option, label), items in buckets.items():
        grouped[(family, option)][label] = items
    rng = random.Random(1337)
    balanced: List[dict] = []
    for (family, option), label_dict in grouped.items():
        pos = label_dict.get(1, [])
        neg = label_dict.get(0, [])
        if pos and neg:
            k = min(len(pos), len(neg))
            balanced.extend(rng.sample(pos, k))
            balanced.extend(rng.sample(neg, k))
        else:
            balanced.extend(pos or neg)
    return balanced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()

    release_map = load_release_records(args.release_dir)
    all_records = build_records(args.csv_dir, release_map, args.threshold)
    splits = {"train": [], "val": [], "test": []}
    for rec in all_records:
        natural_rel = rec["natural_image"] or ""
        split = assign_split(f"{natural_rel}::{rec['tactile_image']}")
        splits[split].append(rec)

    splits["train"] = balance_train(splits["train"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        output_path = args.output_dir / f"{split}.jsonl"
        with output_path.open("w", encoding="utf-8") as fh:
            for rec in items:
                fh.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(items)} records -> {output_path}")


if __name__ == "__main__":
    main()
