#!/usr/bin/env python3
"""Summarize edit-score deltas into tables and plots."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=Path("editing_eval/results/edit_scores.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("editing_eval/results/summary"))
    args = parser.parse_args()

    if not args.scores.exists():
        raise SystemExit(f"Scores file not found: {args.scores}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scores)
    if df.empty:
        raise SystemExit("Score file is empty.")

    improved = (df["delta"] > 0).sum()
    mean_drop = df["delta"].mean()
    median_drop = df["delta"].median()

    summary_md = args.out_dir / "summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# ViT-Guided Editing Summary",
                "",
                f"- Samples scored: **{len(df)}**",
                f"- Improvements (Δprob > 0): **{improved} / {len(df)} ({improved/len(df):.1%})**",
                f"- Mean Δprob: **{mean_drop:.3f}**",
                f"- Median Δprob: **{median_drop:.3f}**",
            ]
        ),
        encoding="utf-8",
    )

    plt.figure(figsize=(8, 4))
    sorted_df = df.sort_values(by="delta", ascending=False)
    colors = ["#2a7f62" if d > 0 else "#c0392b" for d in sorted_df["delta"]]
    plt.bar(range(len(sorted_df)), sorted_df["delta"], color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(range(len(sorted_df)), sorted_df["task"], rotation=45, ha="right")
    plt.ylabel("Probability Drop (before - after)")
    plt.title("ViT Issue Probability Change After Editing")
    plt.tight_layout()
    plot_path = args.out_dir / "delta_bar.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"[done] Summary written to {summary_md} and plot saved to {plot_path}")


if __name__ == "__main__":
    main()
