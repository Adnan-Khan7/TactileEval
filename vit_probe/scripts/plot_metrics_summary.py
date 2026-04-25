#!/usr/bin/env python3
"""Generate compact bar charts from test_metrics.json outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt

PALETTE = {
    "family": "#1f77b4",
    "task": "#2ca02c",
    "option": "#d62728",
}


def _save_bar(
    labels: Iterable[str],
    values: Iterable[float],
    title: str,
    out_path: Path,
    color: str,
    rotation: int = 45,
    horizontal: bool = False,
):
    labels = list(labels)
    values = list(values)
    if not labels:
        return
    if horizontal:
        figsize = (8, max(3.5, 0.35 * len(labels)))
        plt.figure(figsize=figsize)
        y_pos = range(len(labels))
        plt.barh(y_pos, values, color=color)
        plt.yticks(y_pos, labels)
        plt.xlim(0, 1)
        plt.xlabel("Accuracy")
        plt.ylabel("")
    else:
        figsize = (max(6, 0.4 * len(labels)), 4.5)
        plt.figure(figsize=figsize)
        plt.bar(range(len(labels)), values, color=color)
        plt.xticks(range(len(labels)), labels, rotation=rotation, ha="right")
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_metrics(metrics_json: Path, out_dir: Path, max_options: int) -> None:
    data = json.loads(metrics_json.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-family bars
    family_scores: Dict[str, float] = data.get("per_task_family", {})
    if family_scores:
        labels, values = zip(*family_scores.items())
        _save_bar(labels, values, "Accuracy by Task Family", out_dir / "per_family_accuracy.png", PALETTE["family"])

    # Per-task bars (sorted by accuracy descending for readability)
    task_scores: Dict[str, float] = data.get("per_task_id", {})
    if task_scores:
        sorted_items: Iterable[Tuple[str, float]] = sorted(task_scores.items(), key=lambda kv: kv[1], reverse=True)
        labels, values = zip(*sorted_items)
        _save_bar(labels, values, "Accuracy by Task ID", out_dir / "per_task_accuracy.png", PALETTE["task"], rotation=60)

    # Option-level (focus on most error-prone entries → sort ascending and show worst max_options)
    option_scores: Dict[str, float] = data.get("per_option", {})
    if option_scores:
        sorted_items = sorted(option_scores.items(), key=lambda kv: kv[1])
        if max_options > 0:
            sorted_items = sorted_items[:max_options]
        labels, values = zip(*sorted_items)
        _save_bar(
            labels,
            values,
            f"Worst {len(labels)} Option Accuracies",
            out_dir / "per_option_worst.png",
            PALETTE["option"],
            horizontal=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="Path to *_test_metrics.json")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for generated plots")
    parser.add_argument(
        "--max-options",
        type=int,
        default=20,
        help="Number of option IDs to visualize (sorted by lowest accuracy). 0 = show all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_metrics(args.metrics, args.out_dir, max(0, args.max_options))
    print(f"Saved metric plots under {args.out_dir}")


if __name__ == "__main__":
    main()
