"""
Channel Swap Experiment — Fixed for per-model FDM encoders.

Each model was trained with a different token map (different vocab size):
  GPT-2:   rng.choice(50257, ...)
  Qwen3:   rng.choice(151936, ...)
  LFM2:    rng.choice(65536, ...)
  Hermes3: rng.choice(128256, ...)

This script generates swap pairs using each model's own encoder,
ensuring the FDM signal is decodable by the target model.

Usage:
  python channel_swap_eval_v2.py --model gpt2
  python channel_swap_eval_v2.py --model all
"""

import json
import torch
import numpy as np
import argparse
import os
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

# ============================================================
# Channel definitions
# ============================================================

MEMORY_SCHEMAS = {
    0: ("SECRET",   ["RED", "BLUE", "GREEN", "GOLD"]),
    1: ("LOCATION", ["PARIS", "TOKYO", "LONDON", "BERLIN"]),
    2: ("AGENT",    ["ALICE", "BOB", "CAROL", "DAVE"]),
    3: ("STATUS",   ["CLEAR", "COMPROMISED", "UNKNOWN"]),
    4: ("PRIORITY", ["HIGH", "MEDIUM", "LOW"]),
    5: ("BACKUP",   ["AVAILABLE", "UNAVAILABLE"]),
    6: ("RULE",     ["SAFETY_FIRST", "MISSION_FIRST", "BALANCED", "CAUTIOUS"]),
    7: ("META",     ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY", "EMERGENCY", "LOCKDOWN"]),
    8:  ("TEAM",     ["RED_TEAM", "BLUE_TEAM", "GREEN_TEAM", "GOLD_TEAM"]),
    9:  ("REGION",   ["NORTH", "SOUTH", "EAST", "WEST"]),
    10: ("PHASE",    ["ALPHA", "BETA", "GAMMA", "DELTA"]),
    11: ("COMM",     ["OPEN", "CLOSED", "RESTRICTED"]),
    12: ("ASSET",    ["VEHICLE", "AIRCRAFT", "DRONE", "BOAT"]),
    13: ("WINDOW",   ["DAWN", "MIDDAY", "DUSK", "NIGHT"]),
    14: ("COVER",    ["DEEP", "SHALLOW", "NONE"]),
    15: ("SUPPORT",  ["ACTIVE", "STANDBY", "OFFLINE"]),
    16: ("THREAT",   ["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
    17: ("WEATHER",  ["CLEAR", "STORM", "FOG"]),
    18: ("TERRAIN",  ["URBAN", "RURAL", "COASTAL", "MOUNTAIN"]),
    19: ("EXTRACT",  ["READY", "DELAYED", "UNAVAILABLE"]),
    20: ("CIPHER",   ["AES", "RSA", "BLOWFISH", "TWOFISH"]),
    21: ("FREQ",     ["HF", "VHF", "UHF", "SHF"]),
    22: ("PAYLOAD",  ["LIGHT", "MEDIUM", "HEAVY", "CRITICAL"]),
    23: ("ROUTE",    ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]),
    24: ("DURATION", ["SHORT", "MEDIUM", "LONG", "EXTENDED"]),
    25: ("CONTACT",  ["FRIENDLY", "NEUTRAL", "HOSTILE", "UNKNOWN"]),
    26: ("FUEL",     ["FULL", "HALF", "LOW", "CRITICAL"]),
    27: ("ALTITUDE", ["LOW", "MEDIUM", "HIGH"]),
    28: ("VISIBILITY", ["CLEAR", "REDUCED", "ZERO"]),
    29: ("NOISE",    ["SILENT", "QUIET", "MODERATE", "LOUD"]),
    30: ("FORMATION", ["SINGLE", "PAIR", "SQUAD", "PLATOON"]),
    31: ("ARMOR",    ["NONE", "LIGHT", "MEDIUM", "HEAVY"]),
    32: ("SIGNAL",   ["STRONG", "WEAK", "JAMMED", "LOST"]),
    33: ("MORALE",   ["HIGH", "MEDIUM", "LOW"]),
    34: ("SUPPLY",   ["ABUNDANT", "ADEQUATE", "SCARCE", "DEPLETED"]),
    35: ("INTEL",    ["CONFIRMED", "PROBABLE", "UNCERTAIN", "NONE"]),
    36: ("EVAC",     ["STANDING", "PREPPED", "LAUNCHED", "ABORTED"]),
    37: ("WEATHER2", ["SUNNY", "OVERCAST", "RAIN", "SNOW"]),
    38: ("DOCTRINE", ["OFFENSIVE", "DEFENSIVE", "RECON", "SUPPORT"]),
    39: ("COMMS",    ["SECURE", "OPEN", "COMPROMISED", "SILENT"]),
}
NUM_CHANNELS = 40

# ============================================================
# Per-model FDM encoder (vocab_size is the key difference)
# ============================================================

class TurboFDMEncoder:
    def __init__(self, vocab_size, num_tokens_per_encoder=256, a_high=1.0,
                 a_low=0.25, num_levels=64, seed=42, sample_rate=100.0):
        self.N = num_tokens_per_encoder
        self.a_high = a_high
        self.a_low = a_low
        self.num_levels = num_levels
        self.sample_rate = sample_rate
        self.carrier_freqs = [1.0 + i for i in range(NUM_CHANNELS)]
        S = int(np.sqrt(num_tokens_per_encoder / 2))
        self.interleaver = self._s_random(num_tokens_per_encoder, S)
        rng = np.random.RandomState(seed)
        self.token_map = rng.choice(vocab_size, size=num_levels, replace=False)

    def _s_random(self, length, S):
        import random as rnd
        rnd.seed(42)
        il = list(range(length))
        for i in range(length):
            for _ in range(100):
                j = rnd.randint(i, length - 1)
                if all(abs(il[j] - il[k]) >= S for k in range(max(0, i - S + 1), i)):
                    il[i], il[j] = il[j], il[i]
                    break
        return il

    def value_to_bits(self, ch, val):
        _, vals = MEMORY_SCHEMAS[ch]
        idx = vals.index(val)
        nb = max(1, int(np.ceil(np.log2(max(len(vals), 2)))))
        return format(idx, f'0{nb}b')

    def encode(self, memory, tokenizer):
        all_bits = {}
        max_bits = 0
        for ch in range(NUM_CHANNELS):
            b = self.value_to_bits(ch, memory[ch])
            all_bits[ch] = b
            max_bits = max(max_bits, len(b))
        for ch in range(NUM_CHANNELS):
            all_bits[ch] = all_bits[ch].ljust(max_bits, '0')

        t = np.arange(self.N) / self.sample_rate
        spb = self.N // max_bits
        composite = np.zeros(self.N)
        for ch in range(NUM_CHANNELS):
            for bi, bit in enumerate(all_bits[ch]):
                s, e = bi * spb, min((bi + 1) * spb, self.N)
                amp = self.a_high if bit == '1' else self.a_low
                composite[s:e] += amp * np.sin(2 * np.pi * self.carrier_freqs[ch] * t[s:e])

        sig_min, sig_max = composite.min(), composite.max()
        rng = sig_max - sig_min + 1e-10
        norm1 = (composite - sig_min) / rng
        q1 = np.clip(np.floor(norm1 * (self.num_levels - 1) + 0.5).astype(int), 0, self.num_levels - 1)
        tokens1 = [int(self.token_map[q]) for q in q1]
        interleaved = composite[self.interleaver]
        norm2 = (interleaved - sig_min) / rng
        q2 = np.clip(np.floor(norm2 * (self.num_levels - 1) + 0.5).astype(int), 0, self.num_levels - 1)
        tokens2 = [int(self.token_map[q]) for q in q2]
        return tokenizer.decode(tokens1 + tokens2)

# ============================================================
# Model configs — vocab_size must match training
# ============================================================

MODEL_CONFIGS = {
    "gpt2": {
        "path": "./fdm_40ch_turbo_v3_model_final",
        "dtype": torch.float32,
        "arch": "gpt2",
        "max_tokens": 900,
        "vocab_size": 50257,
    },
    "qwen3": {
        "path": "/workspace/fdm-40ch-qwen3",
        "dtype": torch.bfloat16,
        "arch": "llama",
        "max_tokens": 2048,
        "vocab_size": 151936,
    },
    "lfm2": {
        "path": "/workspace/fdm-40ch-lfm2",
        "dtype": torch.bfloat16,
        "arch": "llama",
        "max_tokens": 2048,
        "vocab_size": 65536,
    },
    "hermes3": {
        "path": "/workspace/fdm-40ch-hermes3-3b",
        "dtype": torch.bfloat16,
        "arch": "llama",
        "max_tokens": 2048,
        "vocab_size": 128256,
    },
}

# ============================================================
# Swap experiments
# ============================================================

SWAP_EXPERIMENTS = [
    {
        "name": "STATUS_swap",
        "channel": 3,
        "val_A": "COMPROMISED",
        "val_B": "CLEAR",
        "rule_filter": "SAFETY_FIRST",
        "meta_filter": "NONE",
        "question": "Should {AGENT} proceed with the mission?",
        "expected_A_kw": "abort",
        "expected_B_kw": "proceed",
        "description": "STATUS COMPROMISED→CLEAR under SAFETY_FIRST: expect abort→proceed",
    },
    {
        "name": "META_emergency",
        "channel": 7,
        "val_A": "NONE",
        "val_B": "EMERGENCY",
        "rule_filter": None,
        "meta_filter": None,
        "question": "Should {AGENT} proceed with the mission?",
        "expected_A_kw": None,
        "expected_B_kw": "emergency",
        "description": "META NONE→EMERGENCY: expect normal→EMERGENCY PROTOCOL",
    },
    {
        "name": "META_lockdown",
        "channel": 7,
        "val_A": "NONE",
        "val_B": "LOCKDOWN",
        "rule_filter": None,
        "meta_filter": None,
        "question": "Should {AGENT} proceed with the mission?",
        "expected_A_kw": None,
        "expected_B_kw": "lockdown",
        "description": "META NONE→LOCKDOWN: expect normal→LOCKDOWN ACTIVE",
    },
]

# ============================================================
# Reasoning logic
# ============================================================

def get_proceed_answer(facts, rule, meta):
    agent, status, priority, backup = facts["AGENT"], facts["STATUS"], facts["PRIORITY"], facts["BACKUP"]
    if meta == "EMERGENCY": return f"EMERGENCY PROTOCOL: {agent} must proceed immediately."
    if meta == "LOCKDOWN":  return f"LOCKDOWN ACTIVE: {agent} must abort all operations."
    if rule == "SAFETY_FIRST":
        if status == "COMPROMISED": return f"SAFETY_FIRST: {agent} must abort."
        if status == "CLEAR":       return f"SAFETY_FIRST: {agent} can proceed."
        return f"SAFETY_FIRST: {agent} should wait."
    if rule == "MISSION_FIRST":
        if status == "COMPROMISED" and backup == "UNAVAILABLE":
            return f"MISSION_FIRST: {agent} should abort."
        return f"MISSION_FIRST: {agent} should proceed."
    if rule == "CAUTIOUS":
        if status == "CLEAR" and backup == "AVAILABLE":
            return f"CAUTIOUS: {agent} can proceed."
        return f"CAUTIOUS: {agent} must abort."
    return f"{agent} should assess situation."

ACTION_KEYWORDS = [
    'abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN',
    'share', 'Do NOT share', 'Do not share', 'HIGH RISK', 'LOW RISK', 'MODERATE'
]

def extract_action(response, expected):
    for kw in ACTION_KEYWORDS:
        if kw.lower() in expected.lower() and kw.lower() in response.lower():
            return kw.lower()
    return None

# ============================================================
# Generate swap pairs using model-specific encoder
# ============================================================

def generate_swap_pairs(exp, encoder, tokenizer, n=100):
    pairs = []
    random.seed(42)
    attempts = 0
    while len(pairs) < n and attempts < n * 20:
        attempts += 1
        memory = {ch: random.choice(vals) for ch, (_, vals) in MEMORY_SCHEMAS.items()}
        if exp["rule_filter"] and memory[6] != exp["rule_filter"]:
            continue
        if exp["meta_filter"] is not None and memory[7] != exp["meta_filter"]:
            continue

        memory[exp["channel"]] = exp["val_A"]
        facts = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(6)}
        question = exp["question"].format(**facts)

        fdm_text_A = encoder.encode(memory, tokenizer)
        input_text_A = f"[MEMORY]{fdm_text_A}[/MEMORY]\nQuestion: {question}\nAnswer:"

        memory_B = dict(memory)
        memory_B[exp["channel"]] = exp["val_B"]
        fdm_text_B = encoder.encode(memory_B, tokenizer)
        input_text_B = f"[MEMORY]{fdm_text_B}[/MEMORY]\nQuestion: {question}\nAnswer:"

        memory_B_facts = {MEMORY_SCHEMAS[ch][0]: memory_B[ch] for ch in range(6)}
        rule, meta_A, meta_B = memory[6], memory[7], memory_B[7]
        expected_A = get_proceed_answer(facts, rule, meta_A)
        expected_B = get_proceed_answer(memory_B_facts, rule, meta_B)

        pairs.append({
            "input_A": input_text_A,
            "input_B": input_text_B,
            "expected_A": expected_A,
            "expected_B": expected_B,
            "facts": facts,
            "rule": rule,
        })

    print(f"  Generated {len(pairs)} pairs ({attempts} attempts)")
    return pairs

# ============================================================
# Run eval
# ============================================================

def run_swap_eval(model_name, config, pairs, exp, device):
    if config["arch"] == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(config["path"])
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(config["path"]).to(device).eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["path"], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config["path"], dtype=config["dtype"], trust_remote_code=True
        ).to(device).eval()

    results = []
    for p in tqdm(pairs, desc=f"  {model_name} {exp['name']}"):
        responses = []
        for input_text in [p["input_A"], p["input_B"]]:
            ids = tokenizer.encode(input_text, return_tensors='pt',
                                   truncation=True, max_length=config["max_tokens"]).to(device)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=150, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                )
            resp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
            responses.append(resp)

        resp_A, resp_B = responses
        action_A = extract_action(resp_A, p["expected_A"])
        action_B = extract_action(resp_B, p["expected_B"])

        exp_kw_B = exp.get("expected_B_kw")
        correct_B = (exp_kw_B is None) or (exp_kw_B in resp_B.lower())
        flipped = (action_A != action_B) if action_A and action_B else False
        correct_flip = correct_B and flipped

        results.append({
            "resp_A": resp_A[:120],
            "resp_B": resp_B[:120],
            "action_A": action_A,
            "action_B": action_B,
            "flipped": flipped,
            "correct_flip": correct_flip,
        })

    n = len(results)
    flip_rate = sum(r["flipped"] for r in results) / n
    correct_flip_rate = sum(r["correct_flip"] for r in results) / n

    print(f"  Flip rate:         {flip_rate*100:.1f}%")
    print(f"  Correct flip rate: {correct_flip_rate*100:.1f}%")
    for r in results[:3]:
        print(f"    A: {r['resp_A'][:90]}")
        print(f"    B: {r['resp_B'][:90]}")
        print(f"    flipped={r['flipped']} correct={r['correct_flip']}")
        print()

    return {"flip_rate": flip_rate, "correct_flip_rate": correct_flip_rate, "n": n}

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["all", "gpt2", "qwen3", "lfm2", "hermes3"])
    parser.add_argument("--n_pairs", type=int, default=100)
    parser.add_argument("--output", default="channel_swap_v2_results.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models_to_run = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    all_results = {}

    for exp in SWAP_EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {exp['name']}")
        print(f"  {exp['description']}")
        print(f"{'='*60}")

        for model_name in models_to_run:
            config = MODEL_CONFIGS[model_name]
            if not os.path.exists(config["path"]):
                print(f"  Skipping {model_name} — not found")
                continue

            print(f"\n  Model: {model_name} (vocab_size={config['vocab_size']})")

            # Load model tokenizer for encoding
            if config["arch"] == "gpt2":
                enc_tokenizer = GPT2Tokenizer.from_pretrained(config["path"])
            else:
                enc_tokenizer = AutoTokenizer.from_pretrained(config["path"], trust_remote_code=True)

            encoder = TurboFDMEncoder(vocab_size=config["vocab_size"])
            pairs = generate_swap_pairs(exp, encoder, enc_tokenizer, n=args.n_pairs)

            if config["arch"] == "gpt2":
                tokenizer = enc_tokenizer
                model = GPT2LMHeadModel.from_pretrained(config["path"]).to(device).eval()
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer = enc_tokenizer
                model = AutoModelForCausalLM.from_pretrained(
                    config["path"], dtype=config["dtype"], trust_remote_code=True
                ).to(device).eval()

            results = []
            for p in tqdm(pairs, desc=f"  {model_name} {exp['name']}"):
                responses = []
                for input_text in [p["input_A"], p["input_B"]]:
                    ids = tokenizer.encode(input_text, return_tensors='pt',
                                           truncation=True, max_length=config["max_tokens"]).to(device)
                    with torch.no_grad():
                        out = model.generate(
                            ids, max_new_tokens=150, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                        )
                    resp = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
                    responses.append(resp)

                resp_A, resp_B = responses
                action_A = extract_action(resp_A, p["expected_A"])
                action_B = extract_action(resp_B, p["expected_B"])
                exp_kw_B = exp.get("expected_B_kw")
                correct_B = (exp_kw_B is None) or (exp_kw_B in resp_B.lower())
                flipped = (action_A != action_B) if action_A and action_B else False
                correct_flip = correct_B if exp_kw_B in ["emergency", "lockdown"] else (correct_B and flipped)
                results.append({"flipped": flipped, "correct_flip": correct_flip,
                                 "resp_A": resp_A[:120], "resp_B": resp_B[:120]})

            n = len(results)
            flip_rate = sum(r["flipped"] for r in results) / n
            correct_flip_rate = sum(r["correct_flip"] for r in results) / n
            print(f"  Flip rate:         {flip_rate*100:.1f}%")
            print(f"  Correct flip rate: {correct_flip_rate*100:.1f}%")
            for r in results[:2]:
                print(f"    A: {r['resp_A'][:90]}")
                print(f"    B: {r['resp_B'][:90]}")
                print()

            key = f"{model_name}_{exp['name']}"
            all_results[key] = {"flip_rate": flip_rate, "correct_flip_rate": correct_flip_rate, "n": n}

            del model
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Correct Flip Rate (%)")
    print(f"{'='*60}")
    header = f"{'Model':<12}" + "".join(f"{e['name']:<22}" for e in SWAP_EXPERIMENTS)
    print(header)
    print("-" * len(header))
    for model_name in models_to_run:
        row = f"{model_name:<12}"
        for exp in SWAP_EXPERIMENTS:
            key = f"{model_name}_{exp['name']}"
            pct = all_results.get(key, {}).get("correct_flip_rate", -1) * 100
            row += f"{pct:<22.1f}"
        print(row)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output}")

if __name__ == "__main__":
    main()
