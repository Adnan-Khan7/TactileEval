# Editing Evaluation Harness

Scripts in this folder reproduce the quantitative proxy study described in the
paper.

## Files

- `candidates.json` – list of high-confidence issue samples (pair IDs, task IDs,
  initial ViT probabilities, crowd vote counts).
- `scripts/run_batch_edits.py` – runs the editing pipeline for every candidate,
  storing outputs under `editing_eval/runs/<idx>_*` and writing
  `runs_manifest.json`.
- `scripts/score_edits.py` – recomputes ViT issue probabilities before/after
  editing using the same CLIP probe.
- `scripts/summarize_results.py` – aggregates the deltas into a Markdown report
  + bar plot.

## Usage

```bash
cd editing/eval
python3 scripts/run_batch_edits.py --candidates candidates.json \
  --output-root runs --dataset-root /path/to/dataset --classifier ../vit_probe/models/all_clip_probe.pt

# After edits finish
python3 scripts/score_edits.py --manifest runs/runs_manifest.json \
  --dataset-root /path/to/dataset \
  --classifier ../vit_probe/models/all_clip_probe.pt \
  --output results/edit_scores.csv

python3 scripts/summarize_results.py --scores results/edit_scores.csv \
  --out-dir results/summary
```

The resulting `results/summary/summary.md` and `delta_bar.png` match the numbers
reported in the paper (14/15 improvements, mean drop 0.329).
