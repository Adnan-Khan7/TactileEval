#!/usr/bin/env python3
"""Batch runner for ViT-guided tactile edits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

DEFAULT_DATASET_ROOT = Path("/home/student/khan/smc_2026/family_datasets")
PROCESS_IMAGE = Path("editing_pipeline_v2/process_image.py")
DEFAULT_CLASSIFIER = Path("open_source_baseline/models/ALL/all_clip_probe.pt")


def load_candidates(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("candidates.json must contain a list")
    return data


def safe_slug(text: str) -> str:
    return text.replace("/", "_").replace("::", "_").replace(" ", "_")


def find_new_run_dir(parent: Path, before: set[Path]) -> Path:
    after = {p for p in parent.glob("*") if p.is_dir()}
    new_dirs = after - before
    if not new_dirs:
        raise RuntimeError(f"No new run directory created under {parent}")
    return max(new_dirs, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("editing_eval/candidates.json"))
    parser.add_argument("--output-root", type=Path, default=Path("editing_eval/runs"))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--process-script", type=Path, default=PROCESS_IMAGE)
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--clip-pretrained", default="laion2b_s32b_b82k")
    parser.add_argument("--image-model", default="gpt-image-1")
    parser.add_argument("--vit-threshold", type=float, default=0.6)
    parser.add_argument("--issues-per-edit", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for idx, entry in enumerate(candidates, start=1):
        tactile_path = Path(args.dataset_root, entry["tactile"]).resolve()
        natural_path = Path(args.dataset_root, entry["natural"]).resolve()
        if not tactile_path.exists():
            print(f"[warn] Missing tactile image: {tactile_path}", file=sys.stderr)
            continue
        if not natural_path.exists():
            print(f"[warn] Missing natural image: {natural_path}", file=sys.stderr)
            continue
        candidate_root = args.output_root / f"{idx:02d}_{entry['option']}_{safe_slug(tactile_path.stem)}"
        candidate_root.mkdir(parents=True, exist_ok=True)
        before_dirs = {p for p in candidate_root.glob("*") if p.is_dir()}

        cmd = [
            "python3",
            str(args.process_script),
            "--tactile-path",
            str(tactile_path),
            "--natural-path",
            str(natural_path),
            "--dataset-root",
            str(args.dataset_root),
            "--task-id",
            entry["task"],
            "--classifier",
            str(args.classifier),
            "--clip-pretrained",
            args.clip_pretrained,
            "--vit-threshold",
            str(args.vit_threshold),
            "--issues-per-edit",
            str(args.issues_per_edit),
            "--image-model",
            args.image_model,
            "--output-root",
            str(candidate_root),
        ]

        print(f"[info] Running candidate {idx}/{len(candidates)}: {entry['pair']}")
        if args.dry_run:
            print(" ", " ".join(cmd))
            continue

        subprocess.run(cmd, check=True)
        run_dir = find_new_run_dir(candidate_root, before_dirs)
        manifest.append(
            {
                "pair": entry["pair"],
                "task": entry["task"],
                "option": entry["option"],
                "natural": entry["natural"],
                "tactile": entry["tactile"],
                "prob": entry.get("prob"),
                "votes": entry.get("votes"),
                "run_dir": str(run_dir.resolve()),
            }
        )

    if not args.dry_run and manifest:
        manifest_path = args.output_root / "runs_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[done] Saved manifest to {manifest_path}")
    elif args.dry_run:
        print("[dry-run] No manifest written.")


if __name__ == "__main__":
    main()
