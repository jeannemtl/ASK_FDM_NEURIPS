"""
Signal corruption eval v2 for FDM models — Table 8.
Manual greedy decode avoids inputs_embeds+generate issues.
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
    },
    "qwen3": {
        "path": "/workspace/fdm-40ch-qwen3",
        "test": "/workspace/edeidic_memory_14_percent/fdm_40ch_turbo_v3_qwen3_stage4_test.jsonl",
        "dtype": torch.bfloat16,
        "arch": "llama",
    },
    "lfm2": {
        "path": "/workspace/fdm-40ch-lfm2",
        "test": "/workspace/edeidic_memory_14_percent/fdm_40ch_turbo_v3_lfm2_stage4_test.jsonl",
        "dtype": torch.bfloat16,
        "arch": "llama",
    },
    "hermes3": {
        "path": "/workspace/fdm-40ch-hermes3-3b",
        "test": "/workspace/edeidic_memory_14_percent/fdm_40ch_turbo_v3_hermes3_stage4_test.jsonl",
        "dtype": torch.bfloat16,
        "arch": "llama",
    },
}

SIGMAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
N_SAMPLES = 100
MAX_NEW_TOKENS = 250
ACTION_KEYWORDS = [
    'abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN',
    'share', 'Do NOT share', 'Do not share', 'HIGH RISK', 'LOW RISK', 'MODERATE'
]


def get_embed_std(model, arch):
    if arch == "gpt2":
        return model.transformer.wte.weight.std().item()
    else:
        return model.model.embed_tokens.weight.std().item()


def get_token_embeddings(model, token_ids, arch, device):
    if arch == "gpt2":
        pos = torch.arange(token_ids.size(1), device=device).unsqueeze(0)
        return model.transformer.wte(token_ids) + model.transformer.wpe(pos)
    else:
        return model.model.embed_tokens(token_ids)


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


def greedy_decode(model, embeds, tokenizer, arch, device):
    eos_id = tokenizer.eos_token_id
    generated = []
    current_embeds = embeds
    past_key_values = None
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            outputs = model(
                inputs_embeds=current_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_id = outputs.logits[0, -1, :].argmax().item()
            if next_id == eos_id:
                break
            generated.append(next_id)
            next_tensor = torch.tensor([[next_id]], device=device)
            current_embeds = get_token_embeddings(model, next_tensor, arch, device)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def action_correct(response, expected):
    for kw in ACTION_KEYWORDS:
        if kw.lower() in expected.lower() and kw.lower() in response.lower():
            return True
    return False


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
    embed_std = get_embed_std(model, arch)
    print(f"Embedding std: {embed_std:.4f}")

    all_samples = [json.loads(l) for l in open(config["test"])]
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_samples), size=N_SAMPLES, replace=False)
    samples = [all_samples[i] for i in indices]

    results = {}
    for sigma in SIGMAS:
        correct = 0
        for s in tqdm(samples, desc=f"  sigma={sigma:.1f}", leave=False):
            input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
            max_len = 1024 if arch == 'gpt2' else 4096
            input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=max_len).to(device)
            tokens = input_ids[0].tolist()
            fdm_start, fdm_end = find_fdm_token_range(tokens, tokenizer)

            with torch.no_grad():
                embeds = get_token_embeddings(model, input_ids, arch, device)

            if sigma > 0.0:
                embeds = embeds.clone()
                embeds[:, fdm_start:fdm_end, :] += torch.randn(
                    1, fdm_end - fdm_start, embeds.size(-1),
                    device=device, dtype=embeds.dtype
                ) * (sigma * embed_std)

            response = greedy_decode(model, embeds, tokenizer, arch, device)
            if action_correct(response, s['answer']):
                correct += 1

        acc = correct / N_SAMPLES
        results[sigma] = acc
        print(f"  sigma={sigma:5.1f}: {acc:.3f} ({correct}/{N_SAMPLES})")

    lines = [
        f"{'='*60}",
        f"CORRUPTION EVAL: {model_name.upper()} (n={N_SAMPLES})",
        f"{'='*60}",
        f"embed_std = {embed_std:.4f}", "",
        "sigma / embed_std  |  Action accuracy", "-"*35,
    ]
    for sigma, acc in results.items():
        lines.append(f"  {sigma:5.1f}            |  {acc:.3f}")
    out_str = "\n".join(lines)
    print("\n" + out_str)
    out_path = f"/workspace/edeidic_memory_14_percent/corruption_{model_name}.txt"
    with open(out_path, 'w') as f:
        f.write(out_str + "\n")
    print(f"Saved to {out_path}")
    return results


def main():
    model_names = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    all_results = {}
    for name in model_names:
        if name not in MODELS:
            print(f"Unknown model: {name}")
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
