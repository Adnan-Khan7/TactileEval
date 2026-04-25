#!/usr/bin/env python3
"""Compute per-option metrics and scatter plots from exported predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt


def load_vote_fraction(records_path: Path) -> Dict[Tuple[str, str], float]:
    lookup: Dict[Tuple[str, str], float] = {}
    with records_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (rec["pair_id"], rec["option_id"])
            lookup[key] = rec.get("vote_fraction", 0.0)
    return lookup


def compute_option_metrics(pred_csv: Path) -> Tuple[Dict[str, Dict[str, float]], list]:
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    rows = []
    with pred_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = int(row["label"])
            prob = float(row["probability"])
            pred = 1 if prob >= 0.5 else 0
            option = row["option_id"]
            stats[option]["total"] += 1
            if pred == 1 and label == 1:
                stats[option]["tp"] += 1
            elif pred == 1 and label == 0:
                stats[option]["fp"] += 1
            elif pred == 0 and label == 1:
                stats[option]["fn"] += 1
            rows.append(row)
    metrics: Dict[str, Dict[str, float]] = {}
    for option, counts in stats.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        metrics[option] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": counts["total"],
        }
    return metrics, rows


def plot_option_metrics(metrics: Dict[str, Dict[str, float]], out_dir: Path):
    labels = sorted(metrics.keys())
    f1_scores = [metrics[label]["f1"] for label in labels]
    plt.figure(figsize=(min(12, len(labels) * 0.4 + 4), 4))
    plt.bar(labels, f1_scores, color="#2a7f62")
    plt.xticks(rotation=90)
    plt.ylabel("F1")
    plt.title("Per-option F1 score")
    plt.tight_layout()
    plt.savefig(out_dir / "per_option_f1.png", dpi=200)
    plt.close()


def plot_vote_scatter(rows, vote_lookup: Dict[Tuple[str, str], float], out_dir: Path):
    xs, ys = [], []
    for row in rows:
        key = (row["pair_id"], row["option_id"])
        if key not in vote_lookup:
            continue
        xs.append(vote_lookup[key])
        ys.append(float(row["probability"]))
    if not xs:
        return
    plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys, s=10, alpha=0.4)
    plt.xlabel("Majority vote fraction")
    plt.ylabel("Model probability")
    plt.title("Vote vs. model confidence")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "vote_vs_prob.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="CSV from export_predictions.py")
    parser.add_argument("--records", type=Path, required=True, help="records_full.jsonl for vote fractions")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    option_metrics, rows = compute_option_metrics(args.predictions)
    metrics_csv = args.out_dir / "per_option_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["option_id", "precision", "recall", "f1", "support"])
        for opt, vals in sorted(option_metrics.items()):
            writer.writerow([opt, vals["precision"], vals["recall"], vals["f1"], vals["support"]])
    plot_option_metrics(option_metrics, args.out_dir)

    vote_lookup = load_vote_fraction(args.records)
    plot_vote_scatter(rows, vote_lookup, args.out_dir)
    print(f"Saved metrics + plots under {args.out_dir}")


if __name__ == "__main__":
    main()
