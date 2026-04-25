# ViT-L/14 Feature Probe

This module wraps the CLIP-based probe used for evaluation.  It originates from
`open_source_baseline/` but is trimmed to the scripts referenced in the paper.

## Key scripts

- `scripts/run_vit_pipeline.py` – end-to-end driver that (a) extracts CLIP
  features for train/val/test JSONLs, (b) trains the two-layer MLP probe, and
  (c) exports metrics/plots.
- `scripts/extract_clip_features.py` – helper invoked by the pipeline to cache
  CLIP embeddings (`.npz` files) for each split.
- `scripts/analyze_predictions.py` – computes per-option precision/recall/F1 and
  produces the bar charts shown in Fig.~6/7.
- `commands/train.txt` – example command block with hyperparameters.

## Example usage

```bash
python3 scripts/run_vit_pipeline.py \
  --dataset-root ../dataset/processed \
  --image-root /path/to/TactileNet/images \
  --model ViT-L-14 \
  --pretrained laion2b_s32b_b82k \
  --epochs 20 --hidden-dim 512 --batch-size 128
```

This generates feature caches in `features/`, trained probes in `models/`, and
metrics/plots in `outputs/` (see `.gitignore` for paths excluded from version
control).
