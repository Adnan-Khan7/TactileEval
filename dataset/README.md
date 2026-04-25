# TactileEval Dataset Scripts

This folder exposes the tooling used to convert the raw TactileNet
natural/tactile pairs + AMT CSVs into the structured JSONL splits described in
*TactileEval*.  The processed dataset (images + JSONL records) are hosted on
Hugging Face:

```
https://huggingface.co/datasets/Adnank1998/TactileEval
```

## Contents

- `scripts/build_final_dataset.py` – aggregates AMT CSV exports, applies
  majority-vote filtering, and emits JSONL records + train/val/test splits.
- `task_config.py` – enumerates the six object families, five quality
  dimensions, and option metadata; import this in downstream code to keep option
  IDs consistent.

## Usage

1. Download the dataset from Hugging Face and unpack it under
   `data/TactileEval/`.
2. Run `build_final_dataset.py` to regenerate the JSONLs if you wish to tweak
   thresholds or split seeds:

```bash
python3 scripts/build_final_dataset.py \
  --amt-csv-dir data/TactileEval/raw/amt \
  --output-dir data/TactileEval/processed
```

3. The resulting `records_full.jsonl` and `splits/{train,val,test}.jsonl` files
   feed directly into the ViT probe training scripts under `../vit_probe/`.

> **Reminder:** raw AMT CSVs are not included in this repo. If you need to
> re-run the crowdsourcing stage you must supply your own MTurk exports.
