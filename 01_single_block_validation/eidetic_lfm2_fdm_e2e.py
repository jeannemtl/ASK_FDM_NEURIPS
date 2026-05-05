"""
Eidetic Memory FDM — LiquidAI LFM2.5-1.2B-Base

FDM encoding and evaluation script.

Architecture: Qwen/LiquidAI LFM2.5-1.2B-Base (28 layers, GQA 16Q/8KV, RoPE, SwiGLU)
Vocab: 65,536 tokens
Encoding: 40-channel ASK-FDM, turbo structure (256+256 = 512 tokens)

Usage:
  python eidetic_lfm2_fdm_e2e.py generate
  python eidetic_lfm2_fdm_e2e.py eval [stage]

Training code withheld — see paper for methodology.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random
from tqdm import tqdm
import os
import sys
import argparse


# ============================================================
# Model configuration
# ============================================================

MODEL_ID = "Qwen/LiquidAI LFM2.5-1.2B-Base"
MODEL_SHORT = "lfm2"
VOCAB_SIZE = 65536
DTYPE = torch.bfloat16
OUTPUT_PREFIX = "fdm_40ch_turbo_v3_lfm2"
FINAL_MODEL_DIR = "fdm_40ch_turbo_v3_lfm2_model_final"


# ============================================================
# Channel definitions (identical across all models)
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

REASONING_RULES = {
    "SAFETY_FIRST":  "Always prioritize status over priority.",
    "MISSION_FIRST": "Always prioritize mission completion.",
    "BALANCED":      "Weigh status and priority equally.",
    "CAUTIOUS":      "Require both clear status AND available backup.",
}

META_INSTRUCTIONS = {
    "NONE":              "Follow standard reasoning rules.",
    "OVERRIDE_STATUS":   "Ignore status checks entirely.",
    "OVERRIDE_PRIORITY": "Ignore priority.",
    "EMERGENCY":         "Proceed immediately regardless.",
    "LOCKDOWN":          "Abort all operations.",
}


# ============================================================
# Reasoning logic (identical across all models)
# ============================================================

def generate_proceed_answer(facts, rule, meta):
    agent, status, priority, backup = facts["AGENT"], facts["STATUS"], facts["PRIORITY"], facts["BACKUP"]
    if meta == "EMERGENCY":
        return f"EMERGENCY PROTOCOL: {agent} must proceed immediately. All other factors suspended."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN ACTIVE: {agent} must abort all operations. No exceptions."
    if meta == "OVERRIDE_STATUS":
        if priority == "HIGH":
            return f"Status override active. Priority is {priority}, so {agent} should proceed."
        else:
            return f"Status override active, but priority is only {priority}. {agent} may proceed with caution."
    if meta == "OVERRIDE_PRIORITY":
        if status == "CLEAR" and backup == "AVAILABLE":
            return f"Priority override active. Status {status} with backup {backup}. {agent} can proceed."
        else:
            return f"Priority override active. Status {status}, backup {backup}. {agent} should hold."
    if rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            return f"SAFETY_FIRST: Status is {status}. {agent} must abort regardless of {priority} priority."
        elif status == "CLEAR":
            return f"SAFETY_FIRST: Status is {status}. {agent} can proceed with {priority} priority."
        else:
            return f"SAFETY_FIRST: Status is {status}. {agent} should wait for confirmation."
    elif rule == "MISSION_FIRST":
        if status == "COMPROMISED" and backup == "UNAVAILABLE":
            return f"MISSION_FIRST: Status {status} with no backup. Even mission-priority says {agent} should abort."
        else:
            return f"MISSION_FIRST: Priority is {priority}. {agent} should proceed. Status {status} is secondary."
    elif rule == "BALANCED":
        if priority == "HIGH" and backup == "AVAILABLE":
            return f"BALANCED: {priority} priority with backup {backup} outweighs {status} status. {agent} can proceed."
        elif status == "COMPROMISED":
            return f"BALANCED: {status} status not offset by {priority} priority. {agent} should abort."
        else:
            return f"BALANCED: Status {status}, priority {priority}. {agent} can proceed carefully."
    elif rule == "CAUTIOUS":
        if status == "CLEAR" and backup == "AVAILABLE":
            return f"CAUTIOUS: Status {status} AND backup {backup}. Both conditions met. {agent} can proceed."
        else:
            return f"CAUTIOUS: Need CLEAR status AND AVAILABLE backup. Have {status}/{backup}. {agent} must abort."
    return f"{agent} should assess situation."


def generate_risk_answer(facts, rule, meta):
    status, priority, backup, location = facts["STATUS"], facts["PRIORITY"], facts["BACKUP"], facts["LOCATION"]
    if meta == "EMERGENCY":
        return f"EMERGENCY: Risk assessment suspended. Immediate action required in {location}."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN: Maximum risk assumed. All operations in {location} halted."
    risk_factors = []
    if status == "COMPROMISED": risk_factors.append("compromised status")
    if backup == "UNAVAILABLE": risk_factors.append("no backup")
    if status == "UNKNOWN": risk_factors.append("unknown status")
    if rule == "SAFETY_FIRST":
        if risk_factors:
            return f"SAFETY_FIRST assessment: HIGH RISK in {location}. Factors: {', '.join(risk_factors)}."
        return f"SAFETY_FIRST assessment: LOW RISK in {location}. Status {status}, backup {backup}."
    elif rule == "MISSION_FIRST":
        if len(risk_factors) >= 2:
            return f"MISSION_FIRST assessment: MODERATE RISK in {location}. Acceptable for {priority} priority."
        return f"MISSION_FIRST assessment: LOW RISK in {location}. Proceed with mission."
    elif rule == "BALANCED":
        level = "HIGH" if len(risk_factors) >= 2 else "MEDIUM" if len(risk_factors) == 1 else "LOW"
        return f"BALANCED assessment: {level} RISK in {location}. Factors: {len(risk_factors)} concerns."
    elif rule == "CAUTIOUS":
        if risk_factors:
            return f"CAUTIOUS assessment: HIGH RISK in {location}. Any risk factor triggers alert: {', '.join(risk_factors)}."
        return f"CAUTIOUS assessment: LOW RISK in {location}. All safety conditions met."
    return f"Risk assessment for {location}."


def generate_share_answer(facts, rule, meta):
    secret, status, location = facts["SECRET"], facts["STATUS"], facts["LOCATION"]
    if meta == "EMERGENCY":
        return f"EMERGENCY: Share {secret} immediately with any allied contact. Speed over security."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN: Do not share {secret} under any circumstances. Maintain radio silence."
    if meta == "OVERRIDE_STATUS":
        return f"Status override active. You may share {secret} with {location} contact despite {status} status."
    if rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            return f"SAFETY_FIRST: Do NOT share {secret}. {location} may be compromised."
        return f"SAFETY_FIRST: Status {status}. May share {secret} with verified contacts only."
    elif rule == "MISSION_FIRST":
        return f"MISSION_FIRST: Share {secret} with {location} contact to advance mission. Accept calculated risk."
    elif rule == "BALANCED":
        if status == "CLEAR":
            return f"BALANCED: Status {status}. Safe to share {secret} with {location} contact."
        return f"BALANCED: Status {status}. Share {secret} only if mission-critical."
    elif rule == "CAUTIOUS":
        if status == "CLEAR":
            return f"CAUTIOUS: Status {status}. May share {secret} after secondary verification."
        return f"CAUTIOUS: Status {status}. Do not share {secret}. Request secure channel."
    return f"Evaluate sharing {secret} in {location}."


QUESTION_TEMPLATES = [
    {"question": "Should {AGENT} proceed with the mission?", "fn": generate_proceed_answer},
    {"question": "What is the risk assessment?", "fn": generate_risk_answer},
    {"question": "Is it safe to share the secret with the contact?", "fn": generate_share_answer},
]


# ============================================================
# FDM Encoder — uses fixed global amplitude normalization
# ============================================================

_global_tokenizer = None

def get_tokenizer():
    global _global_tokenizer
    if _global_tokenizer is None:
        _global_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    return _global_tokenizer


class TurboFDMSignalEncoder:
    def __init__(self, num_tokens_per_encoder=256, sample_rate=100.0,
                 a_high=1.0, a_low=0.0, num_levels=64, seed=42):
        self.num_tokens_per_encoder = num_tokens_per_encoder
        self.total_tokens = num_tokens_per_encoder * 2
        self.sample_rate = sample_rate
        self.a_high = a_high
        self.a_low = a_low
        self.num_levels = num_levels
        self.num_channels = NUM_CHANNELS
        self.carrier_freqs = [1.0 + i * 1.0 for i in range(NUM_CHANNELS)]

        S = int(np.sqrt(num_tokens_per_encoder / 2))
        self.interleaver = self._generate_s_random_interleaver(num_tokens_per_encoder, S)

        rng = np.random.RandomState(seed)
        self.token_map = rng.choice(VOCAB_SIZE, size=num_levels, replace=False)

        self.tokenizer = get_tokenizer()

    def _generate_s_random_interleaver(self, length, S):
        import random as rnd
        rnd.seed(42)
        interleaver = list(range(length))
        for i in range(length):
            for _ in range(100):
                j = rnd.randint(i, length - 1)
                valid = True
                for k in range(max(0, i - S + 1), i):
                    if abs(interleaver[j] - interleaver[k]) < S:
                        valid = False
                        break
                if valid:
                    interleaver[i], interleaver[j] = interleaver[j], interleaver[i]
                    break
        return interleaver

    def value_to_bits(self, channel_id, value):
        _, values = MEMORY_SCHEMAS[channel_id]
        idx = values.index(value)
        num_bits = max(1, int(np.ceil(np.log2(max(len(values), 2)))))
        return format(idx, f'0{num_bits}b'), num_bits

    def encode_memory(self, memory):
        all_bits = {}
        max_bits = 0
        for ch in range(self.num_channels):
            bits, nb = self.value_to_bits(ch, memory[ch])
            all_bits[ch] = bits
            max_bits = max(max_bits, nb)
        for ch in range(self.num_channels):
            all_bits[ch] = all_bits[ch].ljust(max_bits, '0')

        num_message_bits = max_bits
        t = np.arange(self.num_tokens_per_encoder) / self.sample_rate
        samples_per_bit = self.num_tokens_per_encoder // num_message_bits

        composite = np.zeros(self.num_tokens_per_encoder)
        for ch in range(self.num_channels):
            bits = all_bits[ch]
            for bi, bit in enumerate(bits):
                start = bi * samples_per_bit
                end = min((bi + 1) * samples_per_bit, self.num_tokens_per_encoder)
                amp = self.a_high if bit == '1' else self.a_low
                composite[start:end] += amp * np.sin(
                    2 * np.pi * self.carrier_freqs[ch] * t[start:end]
                )

        # Sample-wise normalization
        sig_min, sig_max = composite.min(), composite.max()
        sig_range = sig_max - sig_min + 1e-10
        norm1 = (composite - sig_min) / sig_range
        q1 = np.floor(norm1 * (self.num_levels - 1) + 0.5).astype(int)
        q1 = np.clip(q1, 0, self.num_levels - 1)
        tokens1 = [int(self.token_map[q]) for q in q1]

        interleaved = composite[self.interleaver]
        norm2 = (interleaved - sig_min) / sig_range
        q2 = np.floor(norm2 * (self.num_levels - 1) + 0.5).astype(int)
        q2 = np.clip(q2, 0, self.num_levels - 1)
        tokens2 = [int(self.token_map[q]) for q in q2]

        all_tokens = tokens1 + tokens2
        fdm_text = self.tokenizer.decode(all_tokens)

        return fdm_text, all_tokens


# ============================================================
# Data generation
# ============================================================

def generate_stage_data(stage_config, num_samples, output_prefix):
    encoder = TurboFDMSignalEncoder(
        num_tokens_per_encoder=stage_config.get('num_tokens', 256),
        sample_rate=stage_config.get('sample_rate', 100.0),
        a_high=stage_config.get('a_high', 1.0),
        a_low=stage_config.get('a_low', 0.0),
        num_levels=stage_config.get('num_levels', 64),
    )

    samples = []
    print(f"Generating {num_samples} samples: {stage_config}...")

    for _ in tqdm(range(num_samples)):
        memory = {}
        for ch in range(NUM_CHANNELS):
            _, values = MEMORY_SCHEMAS[ch]
            memory[ch] = random.choice(values)

        fdm_text, token_ids = encoder.encode_memory(memory)

        facts = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(6)}
        rule = memory[6]
        meta = memory[7]
        extra = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(8, NUM_CHANNELS)}

        q_template = random.choice(QUESTION_TEMPLATES)
        question = q_template["question"].format(**facts)
        base_answer = q_template["fn"](facts, rule, meta)

        extra_parts = [f"{k}={v}" for k, v in extra.items()]
        answer = f"{base_answer} Context: {', '.join(extra_parts)}."

        explicit_parts = [f"{MEMORY_SCHEMAS[ch][0]}:{memory[ch]}" for ch in range(NUM_CHANNELS)]
        explicit = "|".join(explicit_parts)

        samples.append({
            "fdm_text": fdm_text,
            "memory": {str(k): v for k, v in memory.items()},
            "explicit": explicit,
            "question": question,
            "answer": answer,
            "rule": rule,
            "meta": meta,
            "facts": facts,
            "stage_config": {k: v for k, v in stage_config.items() if k != 'name'},
        })

    random.shuffle(samples)
    n = len(samples)
    splits = {
        'train': samples[:int(0.85 * n)],
        'val': samples[int(0.85 * n):int(0.95 * n)],
        'test': samples[int(0.95 * n):],
    }

    for split_name, split_data in splits.items():
        path = f"{output_prefix}_{split_name}.jsonl"
        with open(path, 'w') as f:
            for s in split_data:
                f.write(json.dumps(s) + "\n")
        print(f"  {split_name}: {len(split_data)} samples -> {path}")

    return samples


STAGE_ENCODING_CONFIGS = [
    {"name": "Stage 0", "num_tokens": 256, "a_high": 1.0, "a_low": 0.0,  "num_levels": 64, "samples": 15000},
    {"name": "Stage 1", "num_tokens": 256, "a_high": 1.0, "a_low": 0.1,  "num_levels": 64, "samples": 20000},
    {"name": "Stage 2", "num_tokens": 256, "a_high": 1.0, "a_low": 0.15, "num_levels": 64, "samples": 25000},
    {"name": "Stage 3", "num_tokens": 256, "a_high": 1.0, "a_low": 0.2,  "num_levels": 64, "samples": 25000},
    {"name": "Stage 4", "num_tokens": 256, "a_high": 1.0, "a_low": 0.25, "num_levels": 64, "samples": 30000},
]


def generate_all_stages():
    for i, stage in enumerate(STAGE_ENCODING_CONFIGS):
        print(f"\n{'='*60}")
        print(f"GENERATING STAGE {i}: {stage['name']} (model: {MODEL_SHORT})")
        print(f"{'='*60}")
        generate_stage_data(stage, stage['samples'], f"{OUTPUT_PREFIX}_stage{i}")


# ============================================================
# Evaluation
# ============================================================

def evaluate(model_path=None, stage_idx=None):
    if model_path is None:
        model_path = FINAL_MODEL_DIR

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {MODEL_SHORT} model from {model_path}/...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=DTYPE, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    si = stage_idx if stage_idx is not None else 4
    test_path = f"{OUTPUT_PREFIX}_stage{si}_test.jsonl"

    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found.")
        return

    test_samples = [json.loads(l) for l in open(test_path)]
    print(f"Evaluating on {len(test_samples)} samples (stage {si})...\n")

    action_correct = 0
    ch_correct = {}
    ch_total = {}
    rule_correct = 0
    meta_correct = 0
    fact_correct = 0
    extra_correct = 0
    extra_total = 0
    total = 0

    for i, s in enumerate(tqdm(test_samples, desc=f"Evaluating [{MODEL_SHORT}]")):
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=250, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        expected = s['answer']
        rule = s['rule']
        meta = s['meta']

        action_ok = False
        for kw in ['abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN',
                    'share', 'Do NOT share', 'Do not share', 'HIGH RISK', 'LOW RISK', 'MODERATE']:
            if kw.lower() in expected.lower() and kw.lower() in response.lower():
                action_ok = True
                break
        if action_ok: action_correct += 1

        rule_ok = (rule in response)
        if rule_ok: rule_correct += 1

        meta_ok = False
        if meta == "EMERGENCY" and "EMERGENCY" in response: meta_ok = True
        elif meta == "LOCKDOWN" and "LOCKDOWN" in response: meta_ok = True
        elif meta in ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY"]: meta_ok = True
        if meta_ok: meta_correct += 1

        fact_ok = False
        facts = s['facts']
        if s['question'].startswith("Should"):
            fact_ok = facts['AGENT'] in response
        elif s['question'].startswith("What is the risk"):
            fact_ok = facts['LOCATION'] in response
        elif s['question'].startswith("Is it safe"):
            fact_ok = facts['SECRET'] in response or facts['LOCATION'] in response
        if fact_ok: fact_correct += 1

        memory = s['memory']
        for ch in range(8, NUM_CHANNELS):
            expected_val = memory[str(ch)]
            extra_total += 1
            ch_total[ch] = ch_total.get(ch, 0) + 1
            ch_correct[ch] = ch_correct.get(ch, 0)
            if expected_val in response:
                extra_correct += 1
                ch_correct[ch] += 1

        total += 1

        if i < 5:
            print(f"\n  [{MODEL_SHORT}] Sample {i+1} | Rule: {rule} | Meta: {meta}")
            print(f"  Expected: {expected[:100]}...")
            print(f"  Got:      {response[:100]}...")

    print(f"\n{'='*60}")
    print(f"{MODEL_SHORT.upper()} 40-CHANNEL FDM EVALUATION")
    print(f"{'='*60}")
    print(f"  Action:  {100*action_correct/total:.1f}%")
    print(f"  Rule:    {100*rule_correct/total:.1f}%")
    print(f"  Meta:    {100*meta_correct/total:.1f}%")
    print(f"  Fact:    {100*fact_correct/total:.1f}%")
    print(f"  Extra:   {100*extra_correct/max(extra_total,1):.1f}%")
    print("")
    print("Per-Channel Extra Accuracy (ch8-39):")
    for ch in range(8, NUM_CHANNELS):
        t = ch_total.get(ch, 0)
        c = ch_correct.get(ch, 0)
        pct = 100 * c / t if t else 0
        flag = " <<<" if pct < 99 else ""
        print(f"  ch{ch:2d} ({MEMORY_SCHEMAS[ch][0]:<12}): {pct:.1f}%{flag}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', type=int, default=256)
    args, _ = parser.parse_known_args()
    if args.tokens != 256:
        for s in STAGE_ENCODING_CONFIGS:
            s['num_tokens'] = args.tokens
        print(f'Token budget override: {args.tokens} per encoder = {args.tokens*2} total')

    if len(sys.argv) < 2:
        print(f"Eidetic Memory FDM — {MODEL_ID}")
        print(f"Usage:")
        print(f"  python eidetic_lfm2_fdm_e2e.py generate")
        print(f"  python eidetic_lfm2_fdm_e2e.py eval [stage]")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "generate":
        generate_all_stages()
    elif cmd == "eval":
        stage = int(sys.argv[2]) if len(sys.argv) > 2 else None
        evaluate(stage_idx=stage)
