# Scaling & Replication Evaluation

Reproduces Table 1 (cross-architecture replication) from the paper.

## Requirements

```bash
pip install torch transformers accelerate huggingface_hub
```

## Models (HuggingFace: prompterminal/)

| Model | HF repo |
|-------|---------|
| GPT-2 Medium | `prompterminal/fdm_40ch_turbo_v3_model_final` |
| Qwen3-0.6B | `prompterminal/fdm-40ch-turbo-v3-qwen3` |
| LFM2.5-1.2B | `prompterminal/fdm-40ch-turbo-v3-lfm2` |
| Hermes3-3B (512-tok) | `prompterminal/fdm-40ch-hermes3-3b` |
| Hermes3-3B (1024-tok) | `prompterminal/fdm-40ch-hermes3-1024tok` |

## Test Data (HuggingFace: prompterminal/fdm-40ch-scaling-test-data)

Downloaded automatically by the eval scripts.

## Reproduction (fresh clone)

```bash
git clone git@github.com:jeannemtl/ASK_FDM_CHECK.git
cd ASK_FDM_CHECK/scaling_replication

python eval_gpt2.py    2>&1 | tee results/gpt2_eval.log
python eval_qwen3.py   2>&1 | tee results/qwen3_eval.log
python eval_lfm2.py    2>&1 | tee results/lfm2_eval.log
python eval_hermes3.py 2>&1 | tee results/hermes3_eval.log

# Hermes3 1024-token (requires H100):
python eidetic_hermes3_1024tok.py eval 2>&1 | tee results/hermes3_1024_eval.log
```

## Expected Results (Table 1)

| Model | Action | Rule* | Meta | Fact | Extra |
|-------|--------|-------|------|------|-------|
| GPT-2 Medium | 98.3% | 39.7% | 99.7% | 99.9% | 89.2% |
| Qwen3-0.6B | 98.7% | 100.0% | 100.0% | 100.0% | 100.0% |
| LFM2.5-1.2B | 98.6% | 100.0% | 100.0% | 100.0% | 93.0% |
| Hermes3-3B (512-tok) | 98.7% | 100.0% | 100.0% | 100.0% | 100.0% |
| Hermes3-3B (1024-tok) | 99.0% | 100.0% | 100.0% | 100.0% | 99.8% |

*Rule accuracy reported for META=NONE samples only. GPT-2 Rule% is aggregate across all META conditions (~39% reflects answer format overhead).
