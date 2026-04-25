#!/usr/bin/env python3
"""Evaluate a trained classifier checkpoint on a feature split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_features(path: Path):
    data = np.load(path, allow_pickle=True)
    X = torch.from_numpy(data["features"]).float()
    y = torch.from_numpy(data["labels"]).float().unsqueeze(1)
    option_ids = data["option_ids"]
    task_families = data["task_families"]
    pair_ids = data["pair_ids"]
    task_ids = data["task_ids"] if "task_ids" in data.files else None
    return X, y, option_ids, task_families, pair_ids, task_ids


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
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    X, y, option_ids, task_families, pair_ids, task_ids = load_features(args.features)
    if task_ids is None:
        task_ids = task_families
    model = load_model(args.checkpoint, device)

    with torch.no_grad():
        logits = model(X.to(device))
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()
    preds = (probs > 0.5).astype(np.float32)
    labels = y.numpy().squeeze()

    overall_acc = float((preds == labels).mean())

    per_family = {}
    for fam in np.unique(task_families):
        mask = task_families == fam
        per_family[fam] = float((preds[mask] == labels[mask]).mean())

    per_option = {}
    for opt in np.unique(option_ids):
        mask = option_ids == opt
        per_option[opt] = float((preds[mask] == labels[mask]).mean())

    per_task = {}
    for task in np.unique(task_ids):
        mask = task_ids == task
        per_task[task] = float((preds[mask] == labels[mask]).mean())

    gt_option_counts = {opt: int(labels[option_ids == opt].sum()) for opt in np.unique(option_ids)}
    pred_option_counts = {opt: int(preds[option_ids == opt].sum()) for opt in np.unique(option_ids)}
    gt_family_counts = {fam: int(labels[task_families == fam].sum()) for fam in np.unique(task_families)}
    pred_family_counts = {fam: int(preds[task_families == fam].sum()) for fam in np.unique(task_families)}

    metrics = {
        "overall_accuracy": overall_acc,
        "per_task_family": per_family,
        "per_task_id": per_task,
        "per_option": per_option,
        "num_samples": int(len(labels)),
        "ground_truth_positive_counts": {
            "option": gt_option_counts,
            "family": gt_family_counts,
        },
        "predicted_positive_counts": {
            "option": pred_option_counts,
            "family": pred_family_counts,
        },
    }

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
