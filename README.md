# TactileEval Codebase 🦾

This directory contains the three pillars of the TactileEval paper:

- **Dataset prep (`dataset/`)** – scripts to convert raw AMT CSV exports into the
  JSONL splits uploaded to [Hugging Face](https://huggingface.co/datasets/Adnank1998/TactileEval).
- **ViT probe (`vit_probe/`)** – CLIP ViT-L/14 feature extraction, training,
  evaluation plots, and metrics tables.
- **Editing pipeline (`editing/`)** – GPT-image prompt templates, issue selection
  logic, and the batch runner used for the 15-sample quantitative study.


## 📁 Layout highlights

| Folder | Purpose |
| --- | --- |
| `dataset/` | Build `records_full.jsonl`, per-family splits, helper plots, and HF metadata. |
| `vit_probe/` | Train/evaluate the CLIP probe, export checkpoints, and generate scatter plots. |
| `editing/` | Select issues, craft prompts, and submit edits via `gpt-image-1`. |

## 🧪 Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> ℹ️ The editing pipeline invokes OpenAI's `gpt-image-1` edit API. Export your own
> credentials before running any editing scripts.

## 📊 Dataset

The full dataset (images + JSONL metadata) lives on the Hugging Face Hub:
`https://huggingface.co/datasets/Adnank1998/TactileEval`. The `dataset/README.md`
file documents how to download it, verify checksums, and regenerate the splits
used in the paper. Configs `family_f1` through `family_f6` mirror the per-family
experiments described in the manuscript.

## 📎 Citation

If you use this repository or the accompanying dataset, please cite the arXiv
preprint:

```
@misc{khan2026tactileevalstepautomatedfinegrained,
  title={TactileEval: A Step Towards Automated Fine-Grained Evaluation and Editing of Tactile Graphics},
  author={Adnan Khan and Abbas Akkasi and Majid Komeili},
  year={2026},
  eprint={2604.19829},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2604.19829}
}
```

## 🙏 Acknowledgements

This work was supported in part by MITACS and the Digital Alliance of Canada.
We thank the student volunteers at the Intelligent Machines Lab (iML), Carleton
University, for their contributions, and Joshua Olojede and Hoda Vafaeesefat
for their help with the AMT annotation environment.
