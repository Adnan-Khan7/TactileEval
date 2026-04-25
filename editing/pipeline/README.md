# Editing Pipeline

The pipeline consumes only a tactile image (and optional natural reference), runs the unified ViT (CLIP probe) to score every option for the selected task, and generates an edit prompt for GPT-Image. Key steps:

1. Encode tactile + natural features with CLIP and evaluate every task option using the trained probe.
2. Select the highest-probability issues (above `--vit-threshold`). By default the single most confident issue (`--issues-per-edit 1`) drives the prompt so edits focus on one problem at a time.
3. Build a task-specific prompt and request an edit from GPT-Image, seeding it with a resized base canvas of the tactile.

Outputs live under `editing_pipeline_v2/outputs/<slug>/` and include:
- `prompt.txt`: final instruction prompt used for editing.
- `edited_vit_<subject>.png`: generated tactile image.
- `issues.json`: raw ViT scores + selected issues.
- `metadata.json` / `metadata_vit.json`: model response + runtime metadata.
- `comparison_grid.png`: side-by-side natural/tactile/edit preview.

See `commands.txt` for example invocations.
