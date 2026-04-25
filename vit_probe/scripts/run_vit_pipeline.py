#!/usr/bin/env python3
"""End-to-end CLIP feature extraction, training, and evaluation per family."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_FAMILIES = ["ALL", "F1", "F2", "F3", "F4", "F5", "F6"]
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def run_cmd(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def has_records(json_path: Path) -> bool:
    if not json_path.exists():
        return False
    with json_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                return True
    return False


def extract_features(split: str, input_json: Path, output_npz: Path, image_root: Path, model: str, pretrained: str, force: bool) -> bool:
    if not has_records(input_json):
        print(f"[skip] {split}: no records in {input_json}")
        return False
    if output_npz.exists() and not force:
        print(f"[cache] {split}: using existing features {output_npz}")
        return True
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "extract_clip_features.py"),
        "--model",
        model,
        "--pretrained",
        pretrained,
        "--input",
        str(input_json),
        "--image-root",
        str(image_root),
        "--output",
        str(output_npz),
    ]
    run_cmd(cmd)
    return True


def train_classifier(train_npz: Path, val_npz: Path | None, checkpoint: Path, epochs: int, hidden_dim: int, batch_size: int, lr: float, force: bool) -> bool:
    if checkpoint.exists() and not force:
        print(f"[cache] checkpoint already exists: {checkpoint}")
        return True
    if not train_npz.exists():
        print(f"[warn] training features missing: {train_npz}")
        return False
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train_classifier.py"),
        "--train",
        str(train_npz),
        "--epochs",
        str(epochs),
        "--hidden-dim",
        str(hidden_dim),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--checkpoint",
        str(checkpoint),
    ]
    if val_npz and val_npz.exists():
        cmd.extend(["--val", str(val_npz)])
    run_cmd(cmd)
    return True


def evaluate_model(features_npz: Path, checkpoint: Path, metrics_path: Path, force: bool) -> bool:
    if not features_npz.exists():
        print(f"[warn] test features missing: {features_npz}")
        return False
    if metrics_path.exists() and not force:
        print(f"[cache] metrics already exist: {metrics_path}")
        return True
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_classifier.py"),
        "--features",
        str(features_npz),
        "--checkpoint",
        str(checkpoint),
        "--metrics",
        str(metrics_path),
    ]
    run_cmd(cmd)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/student/khan/conference_submission/dataset/processed"))
    parser.add_argument("--image-root", type=Path, default=Path("/home/student/khan/smc_2026/family_datasets"))
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    parser.add_argument("--model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    feature_root = REPO_ROOT / "features"
    model_root = REPO_ROOT / "models"
    metrics_root = REPO_ROOT / "outputs"

    for family in args.families:
        family_label = family.upper()
        dataset_dir = args.dataset_root / ("splits" if family_label == "ALL" else Path("family_splits") / family_label)
        if not dataset_dir.exists():
            print(f"[warn] dataset directory missing for {family_label}: {dataset_dir}")
            continue
        print(f"\n=== Processing {family_label} ===")
        feature_dir = feature_root / family_label
        model_dir = model_root / family_label
        metrics_dir = metrics_root / family_label
        family_slug = family_label.lower()

        split_jsons = {
            "train": dataset_dir / "train.jsonl",
            "val": dataset_dir / "val.jsonl",
            "test": dataset_dir / "test.jsonl",
        }
        split_features = {
            split: feature_dir / f"{family_slug}_{split}.npz"
            for split in split_jsons
        }

        available_splits = {}
        for split, json_path in split_jsons.items():
            ok = extract_features(
                split,
                json_path,
                split_features[split],
                args.image_root,
                args.model,
                args.pretrained,
                args.force_features,
            )
            available_splits[split] = ok

        if not available_splits.get("train"):
            print(f"[warn] skipping {family_label}: no training data available.")
            continue

        checkpoint = model_dir / f"{family_slug}_clip_probe.pt"
        train_classifier(
            split_features["train"],
            split_features["val"] if available_splits.get("val") else None,
            checkpoint,
            args.epochs,
            args.hidden_dim,
            args.batch_size,
            args.lr,
            args.force_train,
        )

        if available_splits.get("test"):
            metrics_path = metrics_dir / f"{family_slug}_test_metrics.json"
            evaluate_model(
                split_features["test"],
                checkpoint,
                metrics_path,
                args.force_eval,
            )
        else:
            print(f"[warn] skipping evaluation for {family_label}: no test data.")


if __name__ == "__main__":
    main()
