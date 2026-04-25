#!/usr/bin/env python3
"""Plot training curves and per-task/per-option accuracies."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training(history, out_dir: Path):
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry.get("val_loss") for entry in history]
    val_acc = [entry.get("val_acc") for entry in history]

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    if any(v is not None for v in val_loss):
        plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=200)
    plt.close()

    if any(v is not None for v in val_acc):
        plt.figure()
        plt.plot(epochs, [v if v is not None else float('nan') for v in val_acc], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy")
        plt.title("Validation Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir / "val_acc_curve.png", dpi=200)
        plt.close()


def plot_metrics(metrics_path: Path, out_dir: Path):
    data = json.loads(metrics_path.read_text())
    per_family = data.get("per_task_family", {})
    per_option = data.get("per_option", {})

    if per_family:
        labels = list(per_family.keys())
        values = [per_family[k] for k in labels]
        plt.figure(figsize=(8, 4))
        plt.bar(labels, values)
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Per Task-Family Accuracy")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / "per_task_family.png", dpi=200)
        plt.close()

    if per_option:
        labels = list(per_option.keys())
        values = [per_option[k] for k in labels]
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(labels)), values)
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Per Option Accuracy")
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.tight_layout()
        plt.savefig(out_dir / "per_option.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    history = json.loads(args.history.read_text())
    plot_training(history, out_dir)
    plot_metrics(args.metrics, out_dir)


if __name__ == "__main__":
    main()
