# FDM Paper — Experimental Code and Results

This repository contains the scripts and small results files for the
experiments reported in the paper. Large artifacts (model checkpoints,
datasets, training logs) live on HuggingFace under
[prompterminal](https://huggingface.co/prompterminal).

## Folder layout

| Folder | Paper section | What it covers |
|---|---|---|
| `01_single_block_validation/` | §3, Tables 1, 2 | Single-block FDM finetuning across GPT-2 Medium, Qwen3-0.6B, LFM2.5-1.2B, Hermes3-3B; n-hop reasoning curriculum; per-channel and per-frequency-band retrieval. |
| `02_corruption_asymmetry/` | §3.1, Tables 3, 4, 14 | Plaintext KEY=VALUE baseline; FDM clean / block-deletion / stride-deletion corruption sweep; causal tracing. |
| `03_split_substrate/` | §4 + §5.1, Table 5 | Frozen KV injection; ~62.8M-param learned write head; naive-composition split-source eval on the single-block host. |
| `04_two_block_finetuning/` | §5.2, Table 6 | 60K-sample two-block finetuning at r=0.3 mix, randomized channel-to-block assignment. |
| `05_position_range/` | §5.3, Table 7 | Shifted and swapped position ranges on the two-block-trained host. |
| `06_partition_ratio_sweep/` | §6, Table 8 | 9 partitions × 5 seeds (Qwen3) / 3 seeds (Hermes3); DSB-SC carrier-head variants; per-substrate channel-swap validation. |

## Provenance

`MANIFEST_<podid>.md` files document where each script was found on each
pod, including duplicate copies (chosen file is newest by mtime).

## Reproduction

Each script is runnable as-is given the matching base model on
HuggingFace. See the paper's Appendix for hyperparameters; defaults in
the scripts match the reported runs.

## Scripts hosted in remote artifacts only

The following scripts are referenced in the methodology but live in
HuggingFace model repositories or other published artifacts rather than
in this code repository:

- `gpt2_training_script.py`, `qwen_training_script.py`,
  `liquid_training_script.py` — single-block FDM finetuning entry points.
  Hosted in the `RECREATE/trainning/` directories of the corresponding
  HuggingFace model cards.
- `eidetic_hermes3_fdm_v3_16char.py` — Hermes3 16-character variant.
  Hosted in `PACKAGED_PYTHON_SDK_TRAINED_HERMES/` on HuggingFace.
- `orchestrator.md` — partition-sweep multi-seed orchestration notes.
  Stored alongside the partition sweep results on HuggingFace.

## Filename reconciliation

The following scripts appear under different names in the methodology
section than on disk; the in-disk names are used in this repository:

| Methodology name | Actual filename |
|---|---|
| `nhop_fdm_evaluation_corrected.py` | `nhop_eval_v3.py` |
| `hermes_perchannel_eval.py` | `hermes3_perchannel_eval.py` |
| `plaintext_nhop_eval.py` | `nhop_plaintext_baseline.py` |
| `fdm_corruption.py` | `fdm_corruption_ptgt.py` + `corruption_v2_all_models_table8.py` |
| `fdm_corrupion_nexttok.py` | `corruption_v3_nexttok_gpt2_method.py` (renamed in repo to `fdm_corruption_nexttok.py`) |
| `fdm_causal_tracing (3) final.py` | `fdm_causal_tracing.py` |
| `fdm_write_heads.py` | `fdm_write_head.py` |
