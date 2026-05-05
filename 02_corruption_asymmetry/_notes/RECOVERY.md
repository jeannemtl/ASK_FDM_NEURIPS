# Recovery Guide

This repo contains the code, logs, figures, and results from FDM in-context memory experiments. Model weights and training data are archived on HuggingFace. This document explains how to restore everything after the original pod is gone.

## What's where

| Artifact | Location | Size |
|---|---|---|
| Code, logs, figures, results | This repo (`github.com/jeannemtl/mini_H200_pod_1`) | ~few MB |
| Training/val/test data (JSONL) | HF datasets under `prompterminal/` | ~4.5 GB |
| Final model checkpoints | HF models under `prompterminal/` | ~18 GB (3 models) |
| Channel density variants | HF models under `prompterminal/fdm-density-*` | various |

## HuggingFace Resources

### Datasets (`prompterminal/`)

| Repo | Contents |
|---|---|
| `fdm-product-retrieval-v1` | Product retrieval v1 training/val splits |
| `fdm-product-retrieval-v2` | Product retrieval v2 training/val splits |
| `fdm-40ch-turbo-v3-gpt2` | GPT-2 curriculum stages 0–4 (train/val/test) |
| `fdm-40ch-turbo-v3-qwen3` | Qwen3 curriculum stages 0–4 (train/val/test) |
| `fdm-40ch-turbo-v3-lfm2` | LFM2.5 curriculum stages 0–4 (train/val/test) |
| `fdm-40ch-turbo-v3-hermes3-512tok` | Hermes3 512-token variants |
| `fdm-40ch-turbo-v3-hermes3-1024tok` | Hermes3 1024-token curriculum stages 0–4 |

### Models (`prompterminal/`)

| Repo | Description |
|---|---|
| `fdm-product-retrieval-v2-final` | Final v2 product retrieval checkpoint |
| `fdm-product-retrieval-v3-final` | Final v3 product retrieval checkpoint |
| `fdm-40ch-turbo-v3-hermes3-1024tok` | Paper's main Hermes3 1024-token model |
| `fdm-product-retrieval-hermes3` | Earlier upload of v3-final (same weights) |
| `fdm-density-hermes3-{10,20,40}ch[_dense/_spread]` | Hermes3 channel density scaling variants |
| `fdm-density-lfm2-{10,20,40}ch[_dense/_spread]` | LFM2.5 channel density scaling variants |

## Restoring on a new machine

### 1. Clone this repo

```bash
git clone git@github.com:jeannemtl/mini_H200_pod_1.git
cd mini_H200_pod_1
```

### 2. Install dependencies and authenticate with HuggingFace

```bash
pip install huggingface_hub
huggingface-cli login  # paste your HF token
```

### 3. Download a model checkpoint

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="prompterminal/fdm-40ch-turbo-v3-hermes3-1024tok",
    local_dir="./model"
)
```

Then load it with HuggingFace transformers as you normally would:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained("./model")
```

### 4. Download a dataset

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="prompterminal/fdm-40ch-turbo-v3-hermes3-1024tok",
    repo_type="dataset",
    local_dir="./data"
)
```

The JSONL files will land in `./data/` and can be consumed directly by the training scripts.

### 5. Bulk restore (all models + all datasets)

If you want everything at once (warns: ~22 GB):

```python
from huggingface_hub import snapshot_download

MODELS = [
    "fdm-product-retrieval-v2-final",
    "fdm-product-retrieval-v3-final",
    "fdm-40ch-turbo-v3-hermes3-1024tok",
]

DATASETS = [
    "fdm-product-retrieval-v1",
    "fdm-product-retrieval-v2",
    "fdm-40ch-turbo-v3-gpt2",
    "fdm-40ch-turbo-v3-qwen3",
    "fdm-40ch-turbo-v3-lfm2",
    "fdm-40ch-turbo-v3-hermes3-512tok",
    "fdm-40ch-turbo-v3-hermes3-1024tok",
]

for name in MODELS:
    snapshot_download(
        repo_id=f"prompterminal/{name}",
        local_dir=f"./models/{name}",
    )

for name in DATASETS:
    snapshot_download(
        repo_id=f"prompterminal/{name}",
        repo_type="dataset",
        local_dir=f"./data/{name}",
    )
```

## Re-uploading (if you run new experiments)

Two scripts are included in this repo for reference:

- `upload_final_models_to_hf.py` — uploads final checkpoints from local directories
- `upload_jsonl_to_hf.py` — uploads JSONL training data, grouped into dataset repos

Edit the dictionaries at the top of each to match your new directories.

## What's NOT preserved

By design, the following were intentionally not backed up:

- **Intermediate epoch checkpoints** (`v2_e1` through `v2_e5`) — only `v2_final` was kept
- **Curriculum stage checkpoints** (`v3_model_stage0–4`, `hermes3_1024tok_model_stage0–4`) — only `_final` was kept
- **Training state directories** (`checkpoints_*/`) — optimizer state, scheduler, RNG for resuming interrupted training

If ablation or analysis on intermediate training states becomes necessary, the scripts in this repo can retrain from scratch. Data generation is seeded — check `training_data_generation*.py` for seed values to ensure reproducibility.
