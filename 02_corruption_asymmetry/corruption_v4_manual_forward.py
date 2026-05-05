"""
Signal corruption eval v4 — Table 8 (tab:corruption).
Uses manual forward pass (no model.forward(), no generate) for all architectures.
Matches the original fdm_causal_tracing.py methodology for GPT-2.

For each sample:
  - Build embeddings manually
  - Inject Gaussian noise into FDM signal token embeddings
  - Pass through transformer blocks manually
  - Measure next-token accuracy: argmax(logits) == first token of answer

Supports GPT-2, Qwen3, LFM2.5, Hermes3.
"""

import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys
import os

MODELS = {
    "gpt2": {
        "path": "/workspace/fdm-40ch-gpt2",
        "test": "/workspace/edeidic_memory_14_percent/fdm_40ch_turbo_v3_stage4_test.jsonl",
        "dtype": torch.float32,
        "arch": "gpt2",
        "max_length": 1024,
    },
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

SIGMAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
N_SAMPLES = 100


# ============================================================
# Manual forward — GPT-2
# ============================================================

def manual_forward_gpt2(model, input_ids, device, noise=None, fdm_start=None, fdm_end=None):
    """
    Manual forward for GPT-2. Injects noise into token embeddings
    before the transformer blocks. No model.forward() called.
    """
    input_ids = input_ids.to(device)
    with torch.no_grad():
        te = model.transformer.wte(input_ids)
        pe = model.transformer.wpe(
            torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        )
        h = te + pe

        if noise is not None and fdm_start is not None:
            h = h.clone()
            h[:, fdm_start:fdm_end, :] += noise.to(device)

        h = model.transformer.drop(h)

        for block in model.transformer.h:
            h = h + block.attn(block.ln_1(h))[0]
            h = h + block.mlp(block.ln_2(h))

        h = model.transformer.ln_f(h)
        return model.lm_head(h)[0, -1, :]


# ============================================================
# Manual forward — Llama-style (Qwen3, LFM2.5, Hermes3)
# ============================================================

def manual_forward_llama(model, input_ids, device, noise=None, fdm_start=None, fdm_end=None):
    """
    Manual forward for Llama-style models (Qwen3, LFM2.5, Hermes3).
    Injects noise into token embeddings before the transformer layers.
    Passes position_ids and causal attention mask to each layer.
    No model.forward() called.
    """
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        # Token embeddings
        h = model.model.embed_tokens(input_ids)

        if noise is not None and fdm_start is not None:
            h = h.clone()
            h[:, fdm_start:fdm_end, :] += noise.to(device)

        # Position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # Causal attention mask (4D for Llama attention)
        causal_mask = torch.full(
            (1, 1, seq_len, seq_len),
            float('-inf'),
            dtype=h.dtype,
            device=device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)

        # Pass through each decoder layer
        for layer in model.model.layers:
            layer_out = layer(
                h,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
            h = layer_out[0]

        # Final norm and unembed
        h = model.model.norm(h)
        return model.lm_head(h[0, -1, :])


# ============================================================
# Helpers
# ============================================================

def get_embed_std(model, arch):
    if arch == "gpt2":
        return model.transformer.wte.weight.std().item()
    else:
        return model.model.embed_tokens.weight.std().item()


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
    """
    Get first token of answer by encoding full sequence and
    finding the token at the answer boundary. More reliable than
    encoding the answer in isolation.
    """
    full_text = input_text + " " + answer.strip()
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    full_ids  = tokenizer.encode(full_text,  add_special_tokens=False)
    if len(full_ids) <= len(input_ids):
        return None
    return full_ids[len(input_ids)]


def manual_forward(model, arch, input_ids, device, noise=None, fdm_start=None, fdm_end=None):
    if arch == "gpt2":
        return manual_forward_gpt2(model, input_ids, device, noise, fdm_start, fdm_end)
    else:
        return manual_forward_llama(model, input_ids, device, noise, fdm_start, fdm_end)


# ============================================================
# Main eval loop
# ============================================================

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

    arch = config["arch"]
    max_length = config["max_length"]
    embed_std = get_embed_std(model, arch)
    print(f"Embedding std: {embed_std:.4f}")

    all_samples = [json.loads(l) for l in open(config["test"])]
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_samples), size=N_SAMPLES, replace=False)
    samples = [all_samples[i] for i in indices]

    # Pre-compute target tokens and filter invalid
    valid_samples = []
    for s in samples:
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        tgt = get_answer_first_token(tokenizer, s['answer'], input_text)
        if tgt is not None:
            s['_input_text'] = input_text
            s['_tgt'] = tgt
            valid_samples.append(s)
    print(f"Valid samples: {len(valid_samples)}/{N_SAMPLES}")

    results = {}

    for sigma in SIGMAS:
        correct = 0
        total = 0

        for s in tqdm(valid_samples, desc=f"  sigma={sigma:.1f}", leave=False):
            input_ids = tokenizer.encode(
                s['_input_text'],
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
            ).to(device)

            tokens = input_ids[0].tolist()
            fdm_start, fdm_end = find_fdm_token_range(tokens, tokenizer)
            tgt = s['_tgt']

            noise = None
            if sigma > 0.0:
                noise_std = sigma * embed_std
                fdm_len = fdm_end - fdm_start
                # Get embedding dim
                if arch == "gpt2":
                    embed_dim = model.transformer.wte.weight.shape[1]
                else:
                    embed_dim = model.model.embed_tokens.weight.shape[1]
                noise = torch.randn(1, fdm_len, embed_dim, device=device,
                                    dtype=config["dtype"]) * noise_std

            logits = manual_forward(model, arch, input_ids, device, noise, fdm_start, fdm_end)
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
        f"CORRUPTION EVAL (manual forward): {model_name.upper()} (n={N_SAMPLES})",
        f"{'='*60}",
        f"embed_std = {embed_std:.4f}",
        "",
        "sigma / embed_std  |  Next-token accuracy",
        "-" * 42,
    ]
    for sigma, acc in results.items():
        lines.append(f"  {sigma:5.1f}            |  {acc:.3f}")

    out_str = "\n".join(lines)
    print("\n" + out_str)

    out_path = f"/workspace/edeidic_memory_14_percent/corruption_manual_{model_name}.txt"
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
        print(f"\n{'='*60}\nCOMBINED TABLE 8\n{'='*60}")
        header = f"{'sigma':>6} | " + " | ".join(f"{n:>10}" for n in all_results)
        print(header)
        print("-" * len(header))
        for sigma in SIGMAS:
            row = f"{sigma:>6.1f} | " + " | ".join(
                f"{all_results[n].get(sigma, float('nan')):>10.3f}"
                for n in all_results
            )
            print(row)


if __name__ == "__main__":
    main()
