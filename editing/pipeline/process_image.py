#!/usr/bin/env python3
"""End-to-end tactile editing helper that relies solely on the ViT (CLIP probe) scores."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from openai import OpenAI
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import open_clip
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install open-clip-torch>=2.24.0 to run this script") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from dataset.task_config import TASK_SPECS  # type: ignore
from templates import (  # type: ignore
    GENERIC_TEMPLATE,
    ISSUE_POLARITY,
    ISSUE_TEMPLATES,
    NON_ACTION_OPTIONS,
    TASK_PROMPT_FRAMES,
)

open_clip_model = None

DEFAULT_DATASET_ROOT = Path("/home/student/khan/smc_2026/family_datasets")
DEFAULT_CLASSIFIER = Path("conference_submission/open_source_baseline/models/ALL/all_clip_probe.pt")
DEFAULT_OUTPUT = Path("conference_submission/editing_pipeline_v2/outputs")


def load_classifier(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    input_dim = ckpt["input_dim"]
    hidden_dim = ckpt["hidden_dim"]
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def encode_image(path: Path, preprocess, device: torch.device, cache: Dict[Path, torch.Tensor]) -> torch.Tensor:
    global open_clip_model
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
    global open_clip_model
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feat = open_clip_model.encode_text(tokens)
    feat = torch.nn.functional.normalize(feat, dim=-1)
    return feat


def compute_issue_prob(option_id: str, prob: float) -> float:
    polarity = ISSUE_POLARITY.get(option_id, True)
    return prob if polarity else 1.0 - prob


def build_prompt(subject: str, task_id: str, issues: List[dict]) -> str:
    suffix = task_id[-2:]
    frame = TASK_PROMPT_FRAMES.get(task_id) or TASK_PROMPT_FRAMES.get(suffix) or {
        "header": f"Tactile editing request for {subject}.",
        "footer": "Keep silhouette, pose, and layout unchanged.",
    }
    header = frame["header"]
    if not issues:
        body = "No actionable issues detected; regenerate a clean high-contrast tactile drawing matching the reference species."
    else:
        lines = ["Work through the issues in priority order:"]
        for idx, issue in enumerate(issues, start=1):
            option_id = issue["option_id"]
            template = ISSUE_TEMPLATES.get(option_id)
            if not template:
                template = GENERIC_TEMPLATE.format(option_description=issue["option_description"])
            prob = issue.get("vit_prob")
            confidence = f"(ViT confidence {prob * 100:.1f}%)" if prob is not None else ""
            prefix = "PRIMARY FIX" if idx == 1 else f"Secondary fix #{idx}"
            emphasis = (
                "You may redraw the entire line work layer to accomplish this—favor tactile clarity over matching the original strokes."
                if idx == 1
                else ""
            )
            lines.append(f"- {prefix}: {template} {confidence} {emphasis}".strip())
        body = "\n".join(lines)
    footer = frame["footer"]
    return "\n\n".join([header, body, footer])


def save_image_from_openai(
    prompt: str,
    output_path: Path,
    model: str,
    size: str,
    base_images: Optional[List[Path]] = None,
) -> dict:
    client = OpenAI()
    if base_images:
        files = [open(path, "rb") for path in base_images]
        try:
            result = client.images.edit(model=model, prompt=prompt, image=files, size=size)
        finally:
            for fh in files:
                fh.close()
    else:
        result = client.images.generate(model=model, prompt=prompt, size=size)
    image_b64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_b64)
    output_path.write_bytes(image_bytes)
    metadata = {
        "model": model,
        "size": size,
        "created": getattr(result, "created", None),
        "prompt": prompt,
        "response_id": getattr(result, "id", None),
        "base_images": [str(p) for p in base_images] if base_images else None,
    }
    return metadata


def ensure_base_canvas(src: Path, size_xy: tuple[int, int], dest: Path) -> Path:
    width, height = size_xy
    img = Image.open(src).convert("RGBA")
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    scale = min(width / img.width, height / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    img_resized = img.resize(new_size, Image.LANCZOS)
    offset = ((width - new_size[0]) // 2, (height - new_size[1]) // 2)
    canvas.paste(img_resized, offset, img_resized)
    canvas.save(dest)
    return dest


def infer_natural_from_tactile(tactile_path: Path) -> Optional[Path]:
    parts = list(tactile_path.parts)
    if "Tactile" not in parts:
        return None
    idx = parts.index("Tactile")
    candidate = Path(*parts[:idx], "Natural", *parts[idx + 1 :])
    if candidate.exists():
        return candidate
    return None


def build_task_entries(task_id: str) -> List[dict]:
    spec = TASK_SPECS[task_id]
    entries: List[dict] = []
    for opt in spec.options:
        entries.append(
            {
                "option_id": opt.option_id,
                "option_description": opt.description,
                "task_family": task_id,
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tactile-path", required=True)
    parser.add_argument("--natural-path", help="Optional natural reference image. If omitted, only tactile features are used.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Used for relative-path bookkeeping.")
    parser.add_argument("--task-id", required=True, choices=sorted(TASK_SPECS.keys()))
    parser.add_argument("--classifier", default=str(DEFAULT_CLASSIFIER))
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--clip-pretrained", default="laion2b_s32b_b82k")
    parser.add_argument("--vit-threshold", type=float, default=0.6)
    parser.add_argument(
        "--issues-per-edit",
        type=int,
        default=1,
        help="How many top-ranked issues to include in the prompt (default: 1).",
    )
    parser.add_argument("--image-model", default="gpt-image-1")
    parser.add_argument("--image-size", default="1024x1024")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-natural", action="store_true", help="Ignore natural image entirely.")
    parser.add_argument("--subject-name", help="Optional override for the subject name used in prompts.")
    args = parser.parse_args()

    tactile_path = Path(args.tactile_path).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    classifier_path = Path(args.classifier).resolve()
    output_root = Path(args.output_root).resolve()

    try:
        tactile_rel = tactile_path.relative_to(dataset_root).as_posix()
    except ValueError:
        tactile_rel = tactile_path.name

    if args.skip_natural:
        natural_path = None
    elif args.natural_path:
        candidate = Path(args.natural_path).resolve()
        natural_path = candidate if candidate.exists() else None
    else:
        natural_path = infer_natural_from_tactile(tactile_path)

    if natural_path is None:
        print("Warning: no natural image provided; using tactile-only features.")

    try:
        natural_rel = natural_path.relative_to(dataset_root).as_posix() if natural_path else None
    except Exception:
        natural_rel = natural_path.as_posix() if natural_path else None

    subject = args.subject_name or tactile_path.parent.name or tactile_path.stem

    entries = build_task_entries(args.task_id)
    if not entries:
        raise SystemExit(f"No option specs registered for task {args.task_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess, _ = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    classifier = load_classifier(classifier_path, device)

    global open_clip_model
    open_clip_model = clip_model

    image_cache: Dict[Path, torch.Tensor] = {}
    feat_tac = encode_image(tactile_path, preprocess, device, image_cache)
    if natural_path and natural_path.exists():
        feat_nat = encode_image(natural_path, preprocess, device, image_cache)
        feat_diff = feat_nat - feat_tac
    else:
        feat_nat = feat_tac.clone()
        feat_diff = torch.zeros_like(feat_nat)

    vit_scores = []
    for rec in entries:
        option_text = f"Task {rec['task_family']} option {rec['option_id']}: {rec['option_description']}"
        feat_txt = encode_text(option_text, tokenizer, device)
        vec = torch.cat([feat_nat, feat_tac, feat_diff, feat_txt], dim=-1)
        with torch.no_grad():
            prob = torch.sigmoid(classifier(vec)).item()
        issue_prob = compute_issue_prob(rec["option_id"], prob)
        vit_scores.append(
            {
                "option_id": rec["option_id"],
                "task_family": rec["task_family"],
                "option_description": rec["option_description"],
                "label_probability": prob,
                "issue_prob": issue_prob,
            }
        )

    actionable_scores = [
        score for score in vit_scores if score["option_id"] not in NON_ACTION_OPTIONS
    ]
    selected_issues: List[dict] = [
        {
            "option_id": score["option_id"],
            "task_family": score["task_family"],
            "option_description": score["option_description"],
            "source": "vit",
            "vit_prob": score["issue_prob"],
        }
        for score in actionable_scores
        if score["issue_prob"] >= args.vit_threshold
    ]
    if not selected_issues and actionable_scores:
        top = max(actionable_scores, key=lambda s: s["issue_prob"])
        selected_issues = [
            {
                "option_id": top["option_id"],
                "task_family": top["task_family"],
                "option_description": top["option_description"],
                "source": "vit",
                "vit_prob": top["issue_prob"],
            }
        ]
    selected_issues.sort(key=lambda issue: issue.get("vit_prob") or 0.0, reverse=True)
    if args.issues_per_edit > 0:
        selected_issues = selected_issues[: args.issues_per_edit]

    subject_slug = subject.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = tactile_rel.replace("/", "_")
    run_dir = output_root / f"{slug}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    def run_edit(issues: List[dict]) -> Optional[dict]:
        if not issues:
            return None
        local_prompt = build_prompt(subject, args.task_id, issues)
        prompt_file = run_dir / "prompt_vit.txt"
        prompt_file.write_text(local_prompt, encoding="utf-8")
        edited_path = run_dir / f"edited_vit_{subject_slug}.png"
        base_canvas = run_dir / "base_vit.png"
        ensure_base_canvas(tactile_path, (1024, 1024), base_canvas)
        metadata = save_image_from_openai(
            local_prompt,
            edited_path,
            args.image_model,
            args.image_size,
            base_images=[base_canvas],
        )
        meta_file = run_dir / "metadata_vit.json"
        meta_payload = {
            **metadata,
            "issues_source": "vit",
            "issues": issues,
            "prompt_file": str(prompt_file),
            "image_file": str(edited_path),
        }
        meta_file.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")
        return {
            "prompt": local_prompt,
            "prompt_file": prompt_file,
            "image_path": edited_path,
            "metadata_file": meta_file,
        }

    vit_edit = run_edit(selected_issues)
    if vit_edit:
        shutil.copy2(vit_edit["prompt_file"], run_dir / "prompt.txt")
    (run_dir / "original_tactile.png").write_bytes(tactile_path.read_bytes())

    def load_image(path: Optional[Path]) -> Optional[Image.Image]:
        if not path or not Path(path).exists():
            return None
        return Image.open(path).convert("RGB")

    comparison_items = [
        ("Natural Photo", natural_path if natural_path else None),
        ("Original Tactile", tactile_path),
        ("ViT-guided Edit", Path(vit_edit["image_path"]) if vit_edit else None),
    ]
    fig, axes = plt.subplots(1, len(comparison_items), figsize=(5 * len(comparison_items), 5))
    for ax, (title, img_path) in zip(axes, comparison_items):
        img = load_image(img_path)
        if img is not None:
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "Not available", ha="center", va="center", fontsize=10)
        ax.set_title(title, fontsize=11)
    fig.suptitle(f"{subject} – {args.task_id} corrections", fontsize=14)
    fig.tight_layout()
    cmp_path = run_dir / "comparison_grid.png"
    fig.savefig(cmp_path, dpi=200)
    plt.close(fig)

    issues_payload = {
        "tactile_path": str(tactile_path),
        "natural_path": str(natural_path) if natural_path else None,
        "tactile_rel": tactile_rel,
        "natural_rel": natural_rel,
        "task_id": args.task_id,
        "vit_scores": vit_scores,
        "selected_issues": selected_issues,
        "vit_threshold": args.vit_threshold,
        "comparison_grid": str(cmp_path),
    }
    (run_dir / "issues.json").write_text(json.dumps(issues_payload, indent=2), encoding="utf-8")
    shutil.copy2(tactile_path, run_dir / Path(tactile_path).name)

    if natural_path:
        shutil.copy2(natural_path, run_dir / Path(natural_path).name)

    top_issue = selected_issues[0] if selected_issues else None
    if top_issue:
        issue_desc = (
            f"ViT flagged '{top_issue['option_description']}' "
            f"with probability {top_issue.get('vit_prob', 0.0) * 100:.1f}%."
        )
    else:
        issue_desc = "ViT scores stayed below threshold; no edit requested."
    edit_path_text = vit_edit["image_path"] if vit_edit else "not generated"
    report_lines = [
        f"Subject: {subject} | Task: {args.task_id}",
        issue_desc,
        f"ViT-guided edit: {edit_path_text}",
        f"Comparison grid: {cmp_path}",
    ]
    report_file = run_dir / "report.txt"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    summary_meta = {
        "issues_file": str(run_dir / "issues.json"),
        "comparison_grid": str(cmp_path),
        "vit_metadata": str(vit_edit["metadata_file"]) if vit_edit else None,
        "report_file": str(report_file),
    }
    (run_dir / "metadata.json").write_text(json.dumps(summary_meta, indent=2), encoding="utf-8")

    if vit_edit:
        print(f"Saved ViT-guided edit to {vit_edit['image_path']}")
    else:
        print("No edit generated (no issues exceeded threshold).")
    print(f"Saved comparison grid to {cmp_path}")


if __name__ == "__main__":
    main()
