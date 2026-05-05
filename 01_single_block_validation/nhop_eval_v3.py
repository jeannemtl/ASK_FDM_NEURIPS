"""
N-Hop Eval v3 — imports DIRECTLY from the v3 training script.
No reimplementation of encoder, schemas, or answer logic.

Usage:
  python nhop_eval_v3.py generate
  MODEL_PATH=./fdm_40ch_turbo_v3_model_final python nhop_eval_v3.py eval
  python nhop_eval_v3.py plot
"""

import json, random, os, sys, re
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Import EVERYTHING from your actual v3 training script
from eidetic_real_fdm_e2e_40ch_turbo_v3 import (
    TurboFDMSignalEncoder, MEMORY_SCHEMAS, NUM_CHANNELS,
    QUESTION_TEMPLATES, generate_proceed_answer, generate_risk_answer, generate_share_answer,
)

ALL_Q_TEMPLATES = [
    {"question": "Should {AGENT} proceed with the mission?", "fn": generate_proceed_answer},
    {"question": "What is the risk assessment?", "fn": generate_risk_answer},
    {"question": "Is it safe to share the secret with the contact?", "fn": generate_share_answer},
]


# ============================================================
# GENERATE TEST DATA — uses the real v3 encoder
# ============================================================

def generate_test_data(num_samples=500, output_dir="nhop_v3_data", a_low=0.25):
    os.makedirs(output_dir, exist_ok=True)

    # Create encoder FIRST, before setting random seed for data generation.
    # The interleaver uses random.seed(42) internally — don't pollute it.
    encoder = TurboFDMSignalEncoder(
        num_tokens_per_encoder=256, sample_rate=100.0,
        a_high=1.0, a_low=a_low, num_levels=64, seed=42,
    )

    # NOW set seeds for data randomization (different from encoder seed)
    random.seed(12345)
    np.random.seed(12345)

    samples = []
    print(f"Generating {num_samples} test samples...")

    for i in tqdm(range(num_samples)):
        memory = {ch: random.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}

        fdm_text, token_ids = encoder.encode_memory(memory)

        facts = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(6)}
        rule = memory[6]
        meta = memory[7]
        extra = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(8, NUM_CHANNELS)}

        q_template = random.choice(ALL_Q_TEMPLATES)
        question = q_template["question"].format(**facts)
        base_answer = q_template["fn"](facts, rule, meta)

        extra_parts = [f"{k}={v}" for k, v in extra.items()]
        answer = f"{base_answer} Context: {', '.join(extra_parts)}."

        samples.append({
            "fdm_text": fdm_text,
            "fdm_tokens": token_ids,
            "question": question,
            "expected_answer": answer,
            "memory": {str(ch): memory[ch] for ch in range(NUM_CHANNELS)},
            "rule": rule,
            "meta": meta,
        })

    path = os.path.join(output_dir, "test_data.jsonl")
    with open(path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + '\n')
    print(f"Saved {len(samples)} samples to {path}")


# ============================================================
# PARSE MODEL OUTPUT
# ============================================================

def parse_context_string(text):
    """Extract channel values from the Context: ... portion of model output."""
    parsed = {}

    # Parse Context string
    context_match = re.search(r'Context:\s*(.+?)\.?\s*$', text, re.DOTALL)
    if context_match:
        context_str = context_match.group(1)
        for pair in context_str.split(','):
            pair = pair.strip()
            if '=' in pair:
                key, val = pair.split('=', 1)
                parsed[key.strip()] = val.strip()

    # Parse core facts from reasoning text (before Context:)
    reasoning = text.split('Context:')[0] if 'Context:' in text else text

    # Rule (first word)
    for r in ['SAFETY_FIRST', 'MISSION_FIRST', 'BALANCED', 'CAUTIOUS']:
        if reasoning.startswith(r) or f': {r}' in reasoning:
            parsed['RULE'] = r
            break

    # Meta
    if 'EMERGENCY' in reasoning.split('.')[0]:
        parsed['META'] = 'EMERGENCY'
    elif 'LOCKDOWN' in reasoning.split('.')[0]:
        parsed['META'] = 'LOCKDOWN'
    elif 'Status override' in reasoning or 'Status overridden' in reasoning:
        parsed['META'] = 'OVERRIDE_STATUS'
    elif 'Priority override' in reasoning or 'Priority overridden' in reasoning:
        parsed['META'] = 'OVERRIDE_PRIORITY'

    # Agent
    for agent in ['ALICE', 'BOB', 'CAROL', 'DAVE']:
        if agent in reasoning:
            parsed['AGENT'] = agent
            break

    # Status
    for status in ['CLEAR', 'COMPROMISED', 'UNKNOWN']:
        if f'Status is {status}' in reasoning or f'Status {status}' in reasoning or f'status ({status})' in reasoning:
            parsed['STATUS'] = status
            break

    # Priority
    for pri in ['HIGH', 'MEDIUM', 'LOW']:
        if f'Priority is {pri}' in reasoning or f'priority is {pri}' in reasoning or f'{pri} priority' in reasoning:
            parsed['PRIORITY'] = pri
            break

    # Backup
    for bk in ['AVAILABLE', 'UNAVAILABLE']:
        if bk in reasoning:
            parsed['BACKUP'] = bk
            break

    # Location
    for loc in ['PARIS', 'TOKYO', 'LONDON', 'BERLIN']:
        if loc in reasoning:
            parsed['LOCATION'] = loc
            break

    # Secret
    for sec in ['RED', 'BLUE', 'GREEN', 'GOLD']:
        if sec in reasoning and 'RED_TEAM' not in reasoning.split(sec)[0][-5:]:
            parsed['SECRET'] = sec
            break

    return parsed


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model_path, data_dir="nhop_v3_data", max_new_tokens=250):
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"Loading model from {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    path = os.path.join(data_dir, "test_data.jsonl")
    with open(path) as f:
        samples = [json.loads(line) for line in f]
    print(f"Loaded {len(samples)} samples, device={device}")

    # Per-channel accuracy
    ch_correct = defaultdict(int)
    ch_total = defaultdict(int)

    # Action accuracy
    action_correct = 0

    # Hop levels: how many channels must ALL be correct
    HOP_LEVELS = {
        1:  [6],                                    # RULE
        2:  [2, 6],                                 # AGENT + RULE
        3:  [2, 3, 6],                              # + STATUS
        4:  [2, 3, 4, 6],                           # + PRIORITY
        5:  [2, 3, 4, 5, 6],                        # + BACKUP
        6:  [2, 3, 4, 5, 6, 7],                     # + META (all core)
        8:  list(range(0, 8)),                       # channels 0-7
        10: list(range(0, 8)) + [8, 9],             # + TEAM, REGION
        15: list(range(0, 8)) + list(range(8, 15)), # 0-14
        20: list(range(0, 20)),                      # 0-19
        30: list(range(0, 30)),                      # 0-29
        40: list(range(0, 40)),                      # ALL
    }
    hop_correct = defaultdict(int)
    hop_total = defaultdict(int)

    examples = []

    for i, sample in enumerate(tqdm(samples)):
        # EXACT prompt format from training
        prompt = f"[MEMORY]{sample['fdm_text']}[/MEMORY]\nQuestion: {sample['question']}\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        if input_ids.shape[1] > 1024 - max_new_tokens:
            input_ids = input_ids[:, -(1024 - max_new_tokens):]

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        generated = generated.split('<|endoftext|>')[0].strip()

        memory = {int(k): v for k, v in sample["memory"].items()}
        expected = sample["expected_answer"]

        # Parse what the model decoded
        parsed = parse_context_string(generated)

        # Per-channel accuracy
        for ch in range(NUM_CHANNELS):
            name = MEMORY_SCHEMAS[ch][0]
            ch_total[ch] += 1
            if parsed.get(name) == memory[ch]:
                ch_correct[ch] += 1

        # Hop accuracy
        for nhops, channels in HOP_LEVELS.items():
            hop_total[nhops] += 1
            all_ok = True
            for ch in channels:
                name = MEMORY_SCHEMAS[ch][0]
                if parsed.get(name) != memory[ch]:
                    all_ok = False
                    break
            if all_ok:
                hop_correct[nhops] += 1

        # Action accuracy
        for kw in ['abort', 'proceed', 'hold', 'wait', 'share', 'Do NOT share',
                    'EMERGENCY', 'LOCKDOWN', 'HIGH RISK', 'LOW RISK']:
            if kw.lower() in expected.lower() and kw.lower() in generated.lower():
                action_correct += 1
                break

        if i < 8:
            examples.append((sample['question'], expected[:150], generated[:150], parsed))

    total = len(samples)

    # ── RESULTS ──
    print(f"\n{'='*70}")
    print(f"N-HOP EVALUATION RESULTS ({total} samples)")
    print(f"{'='*70}")

    print(f"\nAction accuracy: {action_correct}/{total} = {action_correct/total*100:.1f}%")

    print(f"\n--- Per-Channel Decode Accuracy ---")
    core_correct = 0
    core_total = 0
    extra_correct = 0
    extra_total = 0
    for ch in range(NUM_CHANNELS):
        name = MEMORY_SCHEMAS[ch][0]
        c = ch_correct[ch]
        t = ch_total[ch]
        pct = c / t * 100 if t > 0 else 0
        marker = ""
        if pct < 30: marker = " *** LOW"
        elif pct < 60: marker = " *"
        print(f"  ch{ch:2d} ({name:12s}): {c:4d}/{t:4d} = {pct:5.1f}%{marker}")
        if ch < 8:
            core_correct += c
            core_total += t
        else:
            extra_correct += c
            extra_total += t

    print(f"\n  Core (ch 0-7):  {core_correct/core_total*100:.1f}%")
    print(f"  Extra (ch 8-39): {extra_correct/extra_total*100:.1f}%")

    print(f"\n--- N-Hop Accuracy ---")
    print(f"{'Channels':<6} {'Description':<35} {'Accuracy':<12} {'Greenblatt':<12}")
    print(f"{'-'*70}")

    greenblatt = {2: "~60%", 3: "~34%", 4: "~chance"}

    for nhops in sorted(HOP_LEVELS.keys()):
        c = hop_correct[nhops]
        t = hop_total[nhops]
        pct = c / t * 100 if t > 0 else 0
        ch_names = [MEMORY_SCHEMAS[ch][0] for ch in HOP_LEVELS[nhops]]
        desc = "+".join(ch_names[:4])
        if len(ch_names) > 4:
            desc += f"+...({len(ch_names)} total)"
        gb = greenblatt.get(nhops, "")
        print(f"{nhops:<6} {desc:<35} {pct:>5.1f}%       {gb}")

    print(f"\n--- Examples ---")
    for j, (q, exp, got, parsed) in enumerate(examples[:5]):
        print(f"\n  [{j+1}] Q: {q}")
        print(f"      Exp: {exp}...")
        print(f"      Got: {got}...")
        print(f"      Parsed: { {k:v for k,v in list(parsed.items())[:8]} }")

    # Save
    results = {
        "action_accuracy": action_correct / total * 100,
        "per_channel": {str(ch): ch_correct[ch] / ch_total[ch] * 100 for ch in range(NUM_CHANNELS)},
        "nhop": {str(h): hop_correct[h] / hop_total[h] * 100 for h in sorted(HOP_LEVELS.keys())},
        "core_accuracy": core_correct / core_total * 100,
        "extra_accuracy": extra_correct / extra_total * 100,
    }
    results_path = os.path.join(data_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_path}")
    return results


# ============================================================
# PLOT
# ============================================================

def plot_results(results_path="nhop_v3_data/results.json"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(results_path) as f:
        results = json.load(f)

    nhop = results["nhop"]
    hops = sorted([int(h) for h in nhop.keys()])
    accs = [nhop[str(h)] for h in hops]

    gb_hops = [2, 3, 4]
    gb_best = [60, 34, 12]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(hops, accs, 'o-', color='#2196F3', linewidth=2.5, markersize=8,
             label='FDM Eidetic (GPT-2 124M)')
    ax1.plot(gb_hops, gb_best, 'D-', color='#F44336', linewidth=2.5, markersize=8,
             label='Greenblatt — Gemini 3 Pro (weights)')
    ax1.axhline(y=25, color='gray', linestyle=':', alpha=0.5, label='Chance')
    ax1.set_xlabel('Channels Required Correct', fontsize=12)
    ax1.set_ylabel('All-Correct Accuracy (%)', fontsize=12)
    ax1.set_title('Multi-Hop: FDM Context vs Facts in Weights', fontsize=13)
    ax1.set_ylim(-5, 105)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ch_accs = results["per_channel"]
    channels = sorted([int(c) for c in ch_accs.keys()])
    ch_vals = [ch_accs[str(c)] for c in channels]
    colors = ['#1565C0' if c < 8 else '#2196F3' if c < 20 else '#64B5F6' for c in channels]
    ax2.bar(channels, ch_vals, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Channel Index', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Per-Channel Decode Accuracy (v3, 500 samples)', fontsize=13)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f'nhop_v3_results.{ext}', dpi=300, bbox_inches='tight')
        print(f"Saved nhop_v3_results.{ext}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python nhop_eval_v3.py generate")
        print("  MODEL_PATH=./fdm_40ch_turbo_v3_model_final python nhop_eval_v3.py eval")
        print("  python nhop_eval_v3.py plot")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "generate":
        generate_test_data(num_samples=500)

    elif cmd == "eval":
        model_path = os.environ.get("MODEL_PATH", "./fdm_40ch_turbo_v3_model_final")
        evaluate(model_path=model_path)

    elif cmd == "plot":
        plot_results()
