"""
Signal corruption eval v5 — Table 8 (tab:corruption).
Uses raw sigma (no embed_std scaling) to match original GPT-2 experiment.
Uses manual forward pass for all architectures.

For Llama-style models, precomputes RoPE embeddings via model's own
rotary_emb module before passing to each layer.
"""

import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys
import os

MODELS = {
    "qwen3": {
        "path": "/workspace/fdm-40ch-qwen3",
        "test": "/workspace/edeidic_memory_14_percent/fdm_40ch_turbo_v3_qwen3_stage4_test.jsonl",
        "dtype": torch.bfloat16,
        "arch": "llama",
        "max_length": 4096,
    },
    "lfm2": {
        "path": "/workspace/fdm-40ch-lfm2",
        "test": "/workspace/edeidic_memory_14_percent/fdm_40ch_turbo_v3_lfm2_stage4_test.jsonl",
        "dtype": torch.bfloat16,
        "arch": "llama",
        "max_length": 4096,
    },
    "hermes3": {
        "path": "/workspace/fdm-40ch-hermes3-3b",
        "test": "/workspace/edeidic_memory_14_percent/fdm_40ch_turbo_v3_hermes3_stage4_test.jsonl",
        "dtype": torch.bfloat16,
        "arch": "llama",
        "max_length": 4096,
    },
}

# Raw sigma values matching original GPT-2 experiment
SIGMAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
N_SAMPLES = 100


def find_fdm_token_range(tokens, tokenizer):
    mem_open  = tokenizer.encode("[MEMORY]",  add_special_tokens=False)
    mem_close = tokenizer.encode("[/MEMORY]", add_special_tokens=False)
    fdm_start = fdm_end = None
    for i in range(len(tokens)):
        if tokens[i:i+len(mem_open)] == mem_open:
            fdm_start = i + len(mem_open)
        if fdm_start and tokens[i:i+len(mem_close)] == mem_close:
            fdm_end = i
            break
    if fdm_start is None or fdm_end is None:
        fdm_start, fdm_end = 1, min(513, len(tokens))
    return fdm_start, fdm_end


def get_answer_first_token(tokenizer, answer, input_text):
    """Get first answer token by comparing untruncated boundary."""
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    full_ids  = tokenizer.encode(input_text + " " + answer.strip(), add_special_tokens=False)
    if len(full_ids) <= len(input_ids):
        return None
    return full_ids[len(input_ids)]


def manual_forward_llama(model, input_ids, device, noise=None, fdm_start=None, fdm_end=None):
    """
    Manual forward for Llama-style models.
    Precomputes RoPE embeddings via model.model.rotary_emb,
    then passes position_embeddings=(cos, sin) to each layer.
    Injects noise into token embeddings before the transformer layers.
    """
    input_ids = input_ids.to(device)

    with torch.no_grad():
        # Token embeddings
        h = model.model.embed_tokens(input_ids)

        # Inject noise into FDM signal tokens
        if noise is not None and fdm_start is not None:
            h = h.clone()
            h[:, fdm_start:fdm_end, :] += noise.to(device)


        # Full forward through transformer (handles RoPE, masking internally)
        out = model.model(inputs_embeds=h)
        return model.lm_head(out.last_hidden_state[0, -1, :])


def run_corruption_eval(model_name, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nMODEL: {model_name.upper()}\n{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(config["path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["path"], dtype=config["dtype"], trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    max_length = config["max_length"]

    all_samples = [json.loads(l) for l in open(config["test"])]
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_samples), size=N_SAMPLES, replace=False)
    samples = [all_samples[i] for i in indices]

    # Pre-compute target tokens
    valid_samples = []
    for s in samples:
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        tgt = get_answer_first_token(tokenizer, s['answer'], input_text)
        if tgt is not None:
            s['_input_text'] = input_text
            s['_tgt'] = tgt
            valid_samples.append(s)
    print(f"Valid samples: {len(valid_samples)}/{N_SAMPLES}")

    # Verify clean accuracy before running all sigmas
    print("Verifying clean accuracy on first 5 samples...")
    for s in valid_samples[:5]:
        input_ids = tokenizer.encode(
            s['_input_text'], return_tensors='pt',
            truncation=True, max_length=max_length,
        ).to(device)
        logits = manual_forward_llama(model, input_ids, device)
        pred = logits.argmax().item()
        tgt = s['_tgt']
        print(f"  tgt={repr(tokenizer.decode([tgt]))} pred={repr(tokenizer.decode([pred]))} match={pred==tgt}")

    results = {}
    for sigma in SIGMAS:
        correct = 0
        total = 0

        for s in tqdm(valid_samples, desc=f"  sigma={sigma:.1f}", leave=False):
            input_ids = tokenizer.encode(
                s['_input_text'], return_tensors='pt',
                truncation=True, max_length=max_length,
            ).to(device)
            tokens = input_ids[0].tolist()
            fdm_start, fdm_end = find_fdm_token_range(tokens, tokenizer)
            tgt = s['_tgt']

            noise = None
            if sigma > 0.0:
                fdm_len = fdm_end - fdm_start
                embed_dim = model.model.embed_tokens.weight.shape[1]
                # Raw sigma — no embed_std scaling, matching original GPT-2 experiment
                noise = torch.randn(
                    1, fdm_len, embed_dim,
                    device=device, dtype=config["dtype"]
                ) * sigma

            logits = manual_forward_llama(model, input_ids, device, noise, fdm_start, fdm_end)
            predicted = logits.argmax().item()

            if predicted == tgt:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0.0
        results[sigma] = acc
        print(f"  sigma={sigma:5.1f}: {acc:.3f} ({correct}/{total})")

    # Save
    lines = [
        f"{'='*60}",
        f"CORRUPTION EVAL (raw sigma, manual fwd): {model_name.upper()} (n={N_SAMPLES})",
        f"{'='*60}",
        f"NOTE: sigma is raw noise std, not scaled by embed_std",
        f"      Matches original GPT-2 experiment methodology.",
        "",
        "sigma  |  Next-token accuracy",
        "-" * 30,
    ]
    for sigma, acc in results.items():
        lines.append(f"  {sigma:5.1f}  |  {acc:.3f}")

    out_str = "\n".join(lines)
    print("\n" + out_str)

    out_path = f"/workspace/edeidic_memory_14_percent/corruption_rawsigma_{model_name}.txt"
    with open(out_path, 'w') as f:
        f.write(out_str + "\n")
    print(f"Saved to {out_path}")

    return results


def main():
    model_names = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    all_results = {}

    for name in model_names:
        if name not in MODELS:
            print(f"Unknown model: {name}. Options: {list(MODELS.keys())}")
            continue
        config = MODELS[name]
        if not os.path.exists(config["path"]):
            print(f"Model not found: {config['path']} — skipping")
            continue
        if not os.path.exists(config["test"]):
            print(f"Test data not found: {config['test']} — skipping")
            continue
        all_results[name] = run_corruption_eval(name, config)

    if len(all_results) > 1:
        print(f"\n{'='*60}\nCOMBINED TABLE 8 (raw sigma)\n{'='*60}")
        header = f"{'sigma':>6} | " + " | ".join(f"{n:>10}" for n in all_results)
        print(header)
        print("-" * len(header))
        for sigma in SIGMAS:
            row = f"{sigma:>6.1f} | " + " | ".join(
                f"{all_results[n].get(sigma, float('nan')):>10.3f}"
                for n in all_results
            )
            print(row)

        # Also print GPT-2 reference
        print("\nGPT-2 reference (from original experiment):")
        gpt2_ref = {0.0: 1.000, 0.5: 0.220, 1.0: 0.230, 2.0: 0.230,
                    3.0: 0.230, 5.0: 0.230, 10.0: 0.230, 20.0: 0.260}
        for sigma in SIGMAS:
            print(f"  sigma={sigma:5.1f}: {gpt2_ref[sigma]:.3f}")


if __name__ == "__main__":
    main()
