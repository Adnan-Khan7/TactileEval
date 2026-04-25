#!/usr/bin/env python3
"""Extract OpenCLIP embeddings for natural/tactile pairs and option text."""

import argparse
import json
from pathlib import Path

import numpy as np
import sys
import torch
from PIL import Image
from tqdm import tqdm

try:
    import open_clip
except ImportError:
    LOCAL_OPEN_CLIP = Path("/home/student/khan/projects_ak/misc/dpo/DiffusionDPO/utils")
    if LOCAL_OPEN_CLIP.exists():
        sys.path.append(str(LOCAL_OPEN_CLIP))
    import open_clip


def load_image(path: Path, device: torch.device, preprocess):
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0).to(device)


def encode_image(model, image_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        feat = model.encode_image(image_tensor)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def encode_text(model, tokenizer, text: str, device: torch.device) -> torch.Tensor:
    token = tokenizer([text]).to(device)
    with torch.no_grad():
        feat = model.encode_text(token)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, preprocess, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(args.model)
    model.eval()

    features, labels = [], []
    option_ids, task_families, pair_ids, task_ids = [], [], [], []

    with args.input.open("r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    for record in tqdm(records, desc=f"Extracting {args.input.name}"):
        natural_path = (args.image_root / record["natural_image"]).resolve()
        tactile_path = (args.image_root / record["tactile_image"]).resolve()
        if not natural_path.exists() or not tactile_path.exists():
            raise FileNotFoundError(f"Missing image for pair {record['pair_id']}")

        img_nat = load_image(natural_path, device, preprocess)
        img_tac = load_image(tactile_path, device, preprocess)

        feat_nat = encode_image(model, img_nat)
        feat_tac = encode_image(model, img_tac)
        feat_diff = feat_nat - feat_tac

        option_text = (
            f"Task {record['task_family']} option {record['option_id']}: "
            f"{record['option_description']}"
        )
        feat_txt = encode_text(model, tokenizer, option_text, device)

        vec = torch.cat([feat_nat, feat_tac, feat_diff, feat_txt], dim=-1)
        features.append(vec.squeeze(0).cpu().numpy())
        labels.append(record["label"])
        option_ids.append(record["option_id"])
        task_families.append(record["task_family"])
        pair_ids.append(record["pair_id"])
        task_ids.append(record.get("task_id", record["task_family"]))

    np.savez_compressed(
        args.output,
        features=np.stack(features),
        labels=np.array(labels, dtype=np.float32),
        option_ids=np.array(option_ids),
        task_families=np.array(task_families),
        pair_ids=np.array(pair_ids),
        task_ids=np.array(task_ids),
        model=args.model,
        pretrained=args.pretrained,
    )
    print(f"Saved {len(features)} samples -> {args.output}")


if __name__ == "__main__":
    main()
