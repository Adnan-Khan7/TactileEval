#!/usr/bin/env python3
"""Recompute ViT issue probabilities before/after editing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List

import torch
from PIL import Image

try:
    import open_clip  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("open-clip-torch is required to run score_edits.py") from exc

from dataset.task_config import TASK_SPECS
from editing_pipeline_v2.templates import ISSUE_POLARITY

DEFAULT_DATASET_ROOT = Path("/home/student/khan/smc_2026/family_datasets")
DEFAULT_CLASSIFIER = Path("open_source_baseline/models/ALL/all_clip_probe.pt")
open_clip_model = None  # global handle set in main()


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Manifest must be a list.")
    return data


def option_description(task_id: str, option_id: str) -> str:
    spec = TASK_SPECS[task_id]
    for opt in spec.options:
        if opt.option_id == option_id:
            return opt.description
    raise KeyError(f"Option {option_id} not found for task {task_id}")


def load_classifier(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = torch.nn.Sequential(
        torch.nn.Linear(ckpt["input_dim"], ckpt["hidden_dim"]),
        torch.nn.ReLU(),
        torch.nn.Linear(ckpt["hidden_dim"], 1),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def encode_image(path: Path, preprocess, device: torch.device, cache: Dict[Path, torch.Tensor]) -> torch.Tensor:
    if path in cache:
        return cache[path]
    image = Image.open(path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = open_clip_model.encode_image(tensor)
    feat = torch.nn.functional.normalize(feat, dim=-1)
    cache[path] = feat
    return feat


def encode_text(text: str, tokenizer, device: torch.device) -> torch.Tensor:
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feat = open_clip_model.encode_text(tokens)
    feat = torch.nn.functional.normalize(feat, dim=-1)
    return feat


def compute_issue_prob(raw_prob: float, option_id: str) -> float:
    polarity = ISSUE_POLARITY.get(option_id, True)
    return raw_prob if polarity else 1.0 - raw_prob


def compute_prob(natural: Path, tactile: Path, task_id: str, option_id: str, description: str,
                 preprocess, tokenizer, device: torch.device, cache: Dict[Path, torch.Tensor],
                 classifier: torch.nn.Module) -> float:
    feat_nat = encode_image(natural, preprocess, device, cache)
    feat_tac = encode_image(tactile, preprocess, device, cache)
    feat_diff = feat_nat - feat_tac
    option_text = f"Task {task_id} option {option_id}: {description}"
    feat_txt = encode_text(option_text, tokenizer, device)
    vec = torch.cat([feat_nat, feat_tac, feat_diff, feat_txt], dim=-1)
    with torch.no_grad():
        raw_prob = torch.sigmoid(classifier(vec)).item()
    return compute_issue_prob(raw_prob, option_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("editing_eval/runs/runs_manifest.json"))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--classifier", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="laion2b_s32b_b82k")
    parser.add_argument("--output", type=Path, default=Path("editing_eval/results/edit_scores.csv"))
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global open_clip_model  # type: ignore
    open_clip_model, preprocess, _ = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    classifier = load_classifier(args.classifier, device)

    image_cache: Dict[Path, torch.Tensor] = {}
    rows = []
    for entry in manifest:
        task_id = entry["task"]
        option_id = entry["option"]
        desc = option_description(task_id, option_id)
        natural_path = Path(args.dataset_root, entry["natural"]).resolve()
        tactile_path = Path(args.dataset_root, entry["tactile"]).resolve()
        run_dir = Path(entry["run_dir"])
        meta_vit = run_dir / "metadata_vit.json"
        if not natural_path.exists() or not tactile_path.exists():
            print(f"[warn] Missing source files for {entry['pair']}")
            continue
        if not meta_vit.exists():
            print(f"[warn] Missing metadata_vit.json in {run_dir}")
            continue
        edited_path = Path(json.loads(meta_vit.read_text())["image_file"]).resolve()
        if not edited_path.exists():
            print(f"[warn] Edited image not found: {edited_path}")
            continue

        p_before = compute_prob(natural_path, tactile_path, task_id, option_id, desc,
                                preprocess, tokenizer, device, image_cache, classifier)
        p_after = compute_prob(natural_path, edited_path, task_id, option_id, desc,
                               preprocess, tokenizer, device, image_cache, classifier)
        rows.append({
            "pair": entry["pair"],
            "task": task_id,
            "option": option_id,
            "issue": desc,
            "prob_before": p_before,
            "prob_after": p_after,
            "delta": p_before - p_after,
            "run_dir": str(run_dir),
            "edited_image": str(edited_path),
        })

    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [
            "pair", "task", "option", "issue", "prob_before", "prob_after", "delta", "run_dir", "edited_image"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
