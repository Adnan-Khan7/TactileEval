#!/usr/bin/env python3
"""Aggregate finalized AMT CSVs into clean JSONL splits and summary stats."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import unquote, urlparse

POSITIVE_OPTIONS = {
    "species_match",
    "posture_match",
    "background_clean",
    "object_match",
    "configuration_match",
    "all_correct",
    "no_line_issues",
    "no_issues_good",
    "good",
    "view_match",
    "orientation_match",
    "view_frontal",
    "view_side",
    "view_top",
    "view_perspective",
    "view_undefined",
    "angle_match",
}

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR.parent
if str(DATASET_ROOT) not in sys.path:
    sys.path.append(str(DATASET_ROOT))

from task_config import FAMILY_NAMES, TASK_SPECS, TaskSpec, infer_task_spec  # noqa: E402


@dataclass(frozen=True)
class VoteEntry:
    vector: Tuple[int, ...]
    status: str
    assignment_id: str


def parse_answer_payload(payload: str) -> Dict[str, Dict[str, bool]]:
    if not payload:
        return {}
    payload = payload.strip()
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if isinstance(parsed, list):
        parsed = next((item for item in parsed if isinstance(item, dict)), {})
    if not isinstance(parsed, dict):
        return {}
    subset = {}
    for key, value in parsed.items():
        if isinstance(value, dict):
            subset[key] = value
    return subset


def url_to_relative_path(url: str) -> str | None:
    if not url:
        return None
    path = unquote(urlparse(url).path)
    parts = [part for part in path.split("/") if part]
    if not parts:
        return None
    if "main_task_F1QV" in parts:
        idx = parts.index("main_task_F1QV") + 1
        rel_parts = ["Animals_and_Creatures", *parts[idx:]]
    elif "family_datasets" in parts:
        idx = parts.index("family_datasets") + 1
        rel_parts = parts[idx:]
    else:
        rel_parts = parts[-4:]
    return "/".join(rel_parts)


def normalize_status(status: str | None) -> str:
    if not status:
        return "submitted"
    status = status.strip().lower()
    if status in {"approved", "rejected"}:
        return status
    return "submitted"


def hash_split(key: str, train_ratio: float, val_ratio: float, seed: int) -> str:
    digest = hashlib.sha1(f"{key}-{seed}".encode("utf-8")).hexdigest()
    val = int(digest[:8], 16) / 0xFFFFFFFF
    if val < train_ratio:
        return "train"
    if val < train_ratio + val_ratio:
        return "val"
    return "test"


def summarize_vote(values: Iterable[bool]) -> Tuple[int, int]:
    total_true = sum(1 for v in values if v)
    total = len(list(values))
    return total_true, total - total_true


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_votes(csv_path: Path, spec: TaskSpec) -> Dict[Tuple[str, str], Dict[str, object]]:
    pair_store: Dict[Tuple[str, str], Dict[str, object]] = {}
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            answers = parse_answer_payload(row.get("Answer.taskAnswers", ""))
            status = normalize_status(row.get("AssignmentStatus"))
            assignment_id = (row.get("AssignmentId") or "").strip()
            if not answers:
                continue
            idx = 1
            while f"Input.q{idx}_tactile_url" in reader.fieldnames:
                tac_url = row.get(f"Input.q{idx}_tactile_url", "")
                nat_url = row.get(f"Input.q{idx}_natural_url", "")
                if not tac_url or not nat_url:
                    idx += 1
                    continue
                natural_rel = url_to_relative_path(nat_url)
                tactile_rel = url_to_relative_path(tac_url)
                answer = answers.get(f"question{idx}")
                idx += 1
                if not natural_rel or not tactile_rel or not isinstance(answer, dict):
                    continue
                vector = tuple(int(bool(answer.get(opt_id, False))) for opt_id in spec.option_ids)
                key = (natural_rel, tactile_rel)
                entry = pair_store.setdefault(
                    key,
                    {"natural": natural_rel, "tactile": tactile_rel, "votes": []},
                )
                entry["votes"].append(VoteEntry(vector=vector, status=status, assignment_id=assignment_id))
    return pair_store


def build_records(
    vote_store: Dict[Tuple[str, str], Dict[str, object]],
    spec: TaskSpec,
    min_votes: int,
    consensus_min: int,
) -> List[dict]:
    records: List[dict] = []
    for pair_key, payload in vote_store.items():
        votes: List[VoteEntry] = payload["votes"]
        if not votes:
            continue
        included: List[VoteEntry] = []
        included_ids = set()
        for entry in votes:
            if entry.status == "approved":
                included.append(entry)
                included_ids.add(entry.assignment_id)
        vector_counts = Counter(entry.vector for entry in votes)
        for vec, count in vector_counts.items():
            if count >= consensus_min:
                for entry in votes:
                    if entry.vector == vec and entry.assignment_id not in included_ids:
                        included.append(entry)
                        included_ids.add(entry.assignment_id)
        if not included:
            continue
        status_counts: Dict[str, int] = defaultdict(int)
        option_votes: Dict[str, List[bool]] = {opt_id: [] for opt_id in spec.option_ids}
        for entry in included:
            status_counts[entry.status] += 1
            for opt_id, value in zip(spec.option_ids, entry.vector):
                option_votes[opt_id].append(bool(value))
        for opt in spec.options:
            votes_for_option = option_votes[opt.option_id]
            if len(votes_for_option) < min_votes:
                continue
            positives = sum(1 for v in votes_for_option if v)
            negatives = len(votes_for_option) - positives
            if positives == negatives:
                continue
            label = 1 if positives > negatives else 0
            record = {
                "pair_id": f"{pair_key[0]}::{pair_key[1]}",
                "task_family": spec.family,
                "task_id": spec.task_id,
                "option_id": opt.option_id,
                "option_description": opt.description,
                "natural_image": pair_key[0],
                "tactile_image": pair_key[1],
                "votes_total": len(votes_for_option),
                "positives": positives,
                "negatives": negatives,
                "vote_fraction": positives / len(votes_for_option),
                "label": label,
                "source_assignments": len(included),
                "status_counts": dict(status_counts),
                "used_consensus": any(entry.status != "approved" for entry in included),
            }
            records.append(record)
    return records


def write_jsonl(path: Path, records: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def group_by_family(records: List[dict]) -> Dict[str, List[dict]]:
    families: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        families[rec["task_family"]].append(rec)
    return families


def write_summary(summary_path: Path, summary_rows: List[dict]) -> None:
    ensure_dir(summary_path.parent)
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_top_issues(option_counts: Dict[Tuple[str, str], Dict[str, int]], output_dir: Path, top_k: int = 20) -> None:
    if plt is None:
        return
    ensure_dir(output_dir)
    filtered = [
        ((task, opt), stats) for (task, opt), stats in option_counts.items()
        if opt not in POSITIVE_OPTIONS
    ]
    if not filtered:
        return
    sorted_items = sorted(filtered, key=lambda kv: kv[1]["positive_labels"], reverse=True)
    top_items = [(f"{task}:{opt}", stats["positive_labels"]) for (task, opt), stats in sorted_items[:top_k]]
    if not top_items:
        return
    labels, values = zip(*top_items)
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color="#2a7f62")
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.ylabel("Positive labels")
    plt.title(f"Top {top_k} issues across all tasks")
    plt.tight_layout()
    plt.savefig(output_dir / "top_issue_counts.png", dpi=200)
    plt.close()


def plot_family_breakdowns(family_counts: Dict[str, Dict[str, int]], output_dir: Path, top_k: int = 10) -> None:
    if plt is None:
        return
    ensure_dir(output_dir)
    for family, counts in family_counts.items():
        filtered = [(opt, val) for opt, val in counts.items() if opt not in POSITIVE_OPTIONS]
        if not filtered:
            continue
        items = sorted(filtered, key=lambda kv: kv[1], reverse=True)[:top_k]
        if not items:
            continue
        labels, values = zip(*items)
        plt.figure(figsize=(10, 4))
        plt.bar(labels, values, color="#1f77b4")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.ylabel("Positive labels")
        plt.title(f"{FAMILY_NAMES.get(family, family)} – top {top_k} flagged options")
        plt.tight_layout()
        plt.savefig(output_dir / f"{family.lower()}_top_issues.png", dpi=200)
        plt.close()


def compute_summary(records: List[dict]) -> Tuple[List[dict], Dict[Tuple[str, str], Dict[str, int]], Dict[str, Dict[str, int]]]:
    option_counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"records": 0, "positive_labels": 0, "negative_labels": 0, "consensus_records": 0}
    )
    family_issue_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    summary_rows: List[dict] = []
    for rec in records:
        key = (rec["task_id"], rec["option_id"])
        option_counts[key]["records"] += 1
        if rec["label"] == 1:
            option_counts[key]["positive_labels"] += 1
        else:
            option_counts[key]["negative_labels"] += 1
        if rec["used_consensus"]:
            option_counts[key]["consensus_records"] += 1
        if rec["label"] == 1 and rec["option_id"] not in {"good", "all_correct", "no_line_issues", "no_issues_good"}:
            family_issue_counts[rec["task_family"]][rec["option_id"]] += 1
    for (task_id, option_id), stats in option_counts.items():
        total = stats["records"]
        pos_labels = stats["positive_labels"]
        neg_labels = stats["negative_labels"]
        summary_rows.append({
            "task_id": task_id,
            "option_id": option_id,
            "records": total,
            "positive_labels": pos_labels,
            "negative_labels": neg_labels,
            "positive_ratio": pos_labels / max(total, 1),
            "records_with_consensus_votes": stats["consensus_records"],
        })
    summary_rows.sort(key=lambda row: (row["task_id"], row["option_id"]))
    return summary_rows, option_counts, family_issue_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amt-csv-dir", type=Path, default=Path("/home/student/khan/conference_submission/dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/student/khan/conference_submission/dataset/processed"))
    parser.add_argument("--consensus-min", type=int, default=5, help="Minimum identical vectors (any status) to include non-approved votes.")
    parser.add_argument("--min-votes", type=int, default=3, help="Minimum votes required to keep a label.")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    records: List[dict] = []

    for csv_path in sorted(args.amt_csv_dir.glob("F*Q*.csv")):
        task_id = csv_path.stem.split(".")[0]
        if task_id not in TASK_SPECS:
            print(f"[warn] Skipping {csv_path.name}: task not registered.")
            continue
        spec = infer_task_spec(task_id)
        vote_store = collect_votes(csv_path, spec)
        task_records = build_records(vote_store, spec, args.min_votes, args.consensus_min)
        print(f"{task_id}: aggregated {len(task_records)} option labels from {len(vote_store)} pairs.")
        records.extend(task_records)

    if not records:
        print("No records were produced. Check input CSVs.")
        return

    all_records_path = args.output_dir / "records_full.jsonl"
    write_jsonl(all_records_path, records)
    print(f"Wrote {len(records)} records -> {all_records_path}")

    splits_dir = args.output_dir / "splits"
    ensure_dir(splits_dir)
    split_records: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for rec in records:
        key = f"{rec['pair_id']}::{rec['option_id']}"
        split = hash_split(key, args.train_ratio, args.val_ratio, args.seed)
        rec["split"] = split
        split_records[split].append(rec)
    for split_name, items in split_records.items():
        path = splits_dir / f"{split_name}.jsonl"
        write_jsonl(path, items)
        print(f"{split_name}: {len(items)} records -> {path}")

    family_splits_dir = args.output_dir / "family_splits"
    ensure_dir(family_splits_dir)
    families = group_by_family(records)
    for family, family_records in families.items():
        family_dir = family_splits_dir / family
        ensure_dir(family_dir)
        for split_name in ("train", "val", "test"):
            subset = [rec for rec in family_records if rec.get("split") == split_name]
            path = family_dir / f"{split_name}.jsonl"
            write_jsonl(path, subset)

    summary_rows, option_counts, family_issue_counts = compute_summary(records)
    summary_csv = args.output_dir / "dataset_summary.csv"
    write_summary(summary_csv, summary_rows)
    summary_json = args.output_dir / "dataset_summary.json"
    task_counts: Dict[str, int] = defaultdict(int)
    for rec in records:
        task_counts[rec["task_id"]] += 1
    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump({
            "total_records": len(records),
            "split_counts": {k: len(v) for k, v in split_records.items()},
            "family_counts": {fam: len(recs) for fam, recs in families.items()},
            "task_counts": task_counts,
        }, fh, indent=2)

    print(f"Summary written to {summary_csv} and {summary_json}")
    plots_dir = args.output_dir / "plots"
    if plt is None:
        print("matplotlib is not installed; skipping plot generation.")
    else:
        plot_top_issues(option_counts, plots_dir, top_k=20)
        plot_family_breakdowns(family_issue_counts, plots_dir, top_k=10)
        print(f"Plots saved under {plots_dir}")


if __name__ == "__main__":
    main()
