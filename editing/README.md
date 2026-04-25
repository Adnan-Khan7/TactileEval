# ViT-Guided Editing Pipeline

This directory contains everything needed to reproduce the editing experiments
in *TactileEval*.

```
editing/
├── pipeline/        # editing_pipeline_v2: prompt templates + OpenAI calls
├── eval/            # editing_eval: batch runner + ViT rescoring scripts
└── README.md        # (this file)
```

## Pipeline

`pipeline/process_image.py` takes a tactile/natural pair, scores all options
with the frozen ViT probe, selects the highest-confidence issue, and routes it
through the `templates.py` prompt frames before calling `gpt-image-1` in edit
mode. Outputs (prompt, edited PNG, metadata, comparison grid) are written per
sample.

Run a single edit:

```bash
python3 editing/pipeline/process_image.py \
  --tactile-path /path/to/Tactile/9.jpg \
  --natural-path /path/to/Natural/9.jpeg \
  --dataset-root /path/to/dataset/root \
  --task-id F1QL \
  --classifier ../vit_probe/models/all_clip_probe.pt \
  --clip-pretrained laion2b_s32b_b82k \
  --image-model gpt-image-1
```

## Batch evaluation

The `eval/` folder reproduces the 15-sample study from the paper:

1. `run_batch_edits.py` – iterates over `candidates.json`, runs the pipeline,
   and logs each run directory.
2. `score_edits.py` – re-scores original vs. edited tactiles with the ViT probe
   to compute probability deltas.
3. `summarize_results.py` – turns the deltas into CSV/plots/Markdown.

See `eval/README.md` for the exact command sequence.

> **API note:** you must configure `OPENAI_API_KEY` (or set the OpenAI client in
> `process_image.py`) before invoking any edit commands.
