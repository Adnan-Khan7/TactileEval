#!/usr/bin/env python3
"""Export per-sample predictions (pair_id, option, prob) for downstream analysis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_features(path: Path):
    data = np.load(path, allow_pickle=True)
    X = torch.from_numpy(data["features"]).float()
    y = torch.from_numpy(data["labels"]).float().unsqueeze(1)
    pair_ids = data["pair_ids"]
    option_ids = data["option_ids"]
    task_families = data["task_families"]
    task_ids = data["task_ids"] if "task_ids" in data.files else task_families
    return X, y, pair_ids, option_ids, task_families, task_ids


def load_model(checkpoint: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint, map_location=device)
    input_dim = ckpt["input_dim"]
    hidden_dim = ckpt["hidden_dim"]
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="features/*.npz file")
    parser.add_argument("--checkpoint", type=Path, required=True, help="models/...pt checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="CSV to write predictions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    X, y, pair_ids, option_ids, task_families, task_ids = load_features(args.features)
    model = load_model(args.checkpoint, device)

    with torch.no_grad():
        probs = torch.sigmoid(model(X.to(device))).cpu().numpy().squeeze()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["pair_id", "task_family", "task_id", "option_id", "label", "probability"])
        for pid, fam, tid, opt, label, prob in zip(pair_ids, task_families, task_ids, option_ids, y.numpy().squeeze(), probs):
            writer.writerow([pid, fam, tid, opt, int(label), float(prob)])
    print(f"Wrote {len(pair_ids)} rows -> {args.output}")


if __name__ == "__main__":
    main()
