"""
End-to-End Eidetic Memory with REAL FDM (Option 2)

A SINGLE model that:
  1. Reads tokens derived from a real FDM signal as its context
  2. Generates reasoning responses directly
  3. Must implicitly learn FFT separation + ASK demodulation + reasoning

The FDM signal is a superposition of 8 ASK-modulated sinusoidal carriers.
Each time sample is quantized and mapped to a GPT-2 token ID.
Every token carries information from ALL 8 channels simultaneously.

This is fundamentally harder than TDM (codebook interleaving) because:
  - TDM: each position carries 1 channel, model learns codebook patterns
  - FDM: each position carries ALL channels, model must learn frequency separation

Curriculum stages reduce encoding difficulty:
  Stage 0: 256 tokens, high amplitude contrast (A_high=1.0, A_low=0.0)
  Stage 1: 256 tokens, moderate contrast (A_high=1.0, A_low=0.1)
  Stage 2: 192 tokens, moderate contrast
  Stage 3: 128 tokens, moderate contrast (fewer samples = harder FFT)
  Stage 4: 128 tokens, lower contrast (A_high=1.0, A_low=0.2)

Channel allocation (same as TDM baseline):
  0: SECRET   (4 values)   5: BACKUP   (2 values)
  1: LOCATION (4 values)   6: RULE     (4 values)
  2: AGENT    (4 values)   7: META     (5 values)
  3: STATUS   (3 values)
  4: PRIORITY (3 values)

Usage:
  python eidetic_real_fdm_e2e.py generate    # Generate data for all stages
  python eidetic_real_fdm_e2e.py train       # Curriculum training
  python eidetic_real_fdm_e2e.py eval        # Evaluate
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
import json
import random
from tqdm import tqdm
import os
import sys
import argparse


# ============================================================
# Channel definitions (identical to TDM experiment)
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
# Reasoning logic (identical to TDM experiment)
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
# Real FDM Encoder (sinusoidal carriers → token IDs)
# ============================================================

class RealFDMTokenEncoder:
    """
    Encodes 8 memory channels into token IDs via real FDM.
    
    Pipeline:
      1. Convert each channel value to bits
      2. ASK modulate each channel onto a distinct carrier frequency
      3. Superpose all channels (TRUE FDM)
      4. Quantize composite signal
      5. Map quantized values to token IDs deterministically
    
    The token mapping uses a fixed, deterministic scheme:
      - Quantized value q ∈ [0, num_levels-1]
      - Token ID = token_map[q] where token_map is a fixed shuffled mapping
      - This ensures different quantization levels map to different tokens
      - The model must learn this mapping from training data
    """
    
    def __init__(self, num_tokens=256, sample_rate=100.0, 
                 a_high=1.0, a_low=0.0, num_levels=64, seed=42):
        """
        Args:
            num_tokens: Number of FDM tokens to generate
            sample_rate: Sampling rate in Hz
            a_high: ASK amplitude for bit=1
            a_low: ASK amplitude for bit=0
            num_levels: Quantization levels
            seed: Random seed for token mapping
        """
        self.num_tokens = num_tokens
        self.sample_rate = sample_rate
        self.a_high = a_high
        self.a_low = a_low
        self.num_levels = num_levels
        
        # Carrier frequencies: 1 Hz apart, well under Nyquist (50 Hz)
        self.carrier_freqs = [1.0 + i * 1.0 for i in range(NUM_CHANNELS)]
        # [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] Hz
        
        # Fixed deterministic token mapping
        # Maps quantization level → GPT-2 token ID
        # Uses a spread across vocabulary so tokens are visually distinct
        rng = np.random.RandomState(seed)
        # Pick num_levels tokens spread across the vocabulary
        self.token_map = rng.choice(50257, size=num_levels, replace=False)
        # Reverse map for decode verification
        self.reverse_map = {int(t): i for i, t in enumerate(self.token_map)}
    
    def value_to_bits(self, channel_id, value):
        """Convert channel value to fixed-length bits."""
        _, values = MEMORY_SCHEMAS[channel_id]
        idx = values.index(value)
        num_bits = max(1, int(np.ceil(np.log2(max(len(values), 2)))))
        return format(idx, f'0{num_bits}b'), num_bits
    
    def encode_memory(self, memory):
        """
        Encode 8 memory channels into FDM token IDs.
        
        Returns:
            token_ids: list of int (GPT-2 token IDs)
            fdm_text: decoded text string
        """
        # Step 1: Convert values to bits, pad to uniform length
        all_bits = {}
        max_bits = 0
        for ch in range(NUM_CHANNELS):
            bits, num_bits = self.value_to_bits(ch, memory[ch])
            all_bits[ch] = bits
            max_bits = max(max_bits, num_bits)
        
        for ch in range(NUM_CHANNELS):
            all_bits[ch] = all_bits[ch].ljust(max_bits, '0')
        
        num_message_bits = max_bits
        
        # Step 2: Generate time axis
        t = np.arange(self.num_tokens) / self.sample_rate
        
        # Samples per bit
        samples_per_bit = self.num_tokens // num_message_bits
        
        # Step 3: ASK modulate + superpose
        composite = np.zeros(self.num_tokens)
        
        for ch in range(NUM_CHANNELS):
            bits = all_bits[ch]
            for bi, bit in enumerate(bits):
                start = bi * samples_per_bit
                end = min((bi + 1) * samples_per_bit, self.num_tokens)
                amplitude = self.a_high if bit == '1' else self.a_low
                composite[start:end] += amplitude * np.sin(
                    2 * np.pi * self.carrier_freqs[ch] * t[start:end]
                )
        
        # Step 4: Quantize to [0, num_levels-1]
        sig_min = composite.min()
        sig_max = composite.max()
        sig_range = sig_max - sig_min + 1e-10
        normalized = (composite - sig_min) / sig_range
        quantized = np.floor(normalized * (self.num_levels - 1) + 0.5).astype(int)
        quantized = np.clip(quantized, 0, self.num_levels - 1)
        
        # Step 5: Map to token IDs
        token_ids = [int(self.token_map[q]) for q in quantized]
        
        # Decode to text for the model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        fdm_text = tokenizer.decode(token_ids)
        
        return token_ids, fdm_text


# ============================================================
# Global tokenizer for FDM text decoding (loaded once)
# ============================================================

_global_tokenizer = None

def get_tokenizer():
    global _global_tokenizer
    if _global_tokenizer is None:
        _global_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return _global_tokenizer


# ============================================================
# Efficient FDM encoder (no GPT-2 model needed — pure signal processing)
# ============================================================

class TurboFDMSignalEncoder:
    """
    Turbo-structured FDM encoder with S-random interleaving.
    
    Produces TWO encodings of the same FDM composite signal:
      Encoder 1: Signal quantized in natural time order (256 tokens)
      Encoder 2: Same signal samples, S-random interleaved order (256 tokens)
    
    The S-random interleaver ensures adjacent samples in encoder 1 are
    far apart in encoder 2. This gives the model two uncorrelated views
    of the same FDM data — analogous to turbo decoders exchanging
    extrinsic information between constituent decoders.
    
    Total output: 512 tokens (256 direct + 256 interleaved)
    """
    
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
        
        # S-random interleaver
        S = int(np.sqrt(num_tokens_per_encoder / 2))
        self.interleaver = self._generate_s_random_interleaver(num_tokens_per_encoder, S)
        
        # Same token mapping for both encoders
        rng = np.random.RandomState(seed)
        self.token_map = rng.choice(50257, size=num_levels, replace=False)
        
        self.tokenizer = get_tokenizer()
    
    def _generate_s_random_interleaver(self, length, S):
        """Generate S-random interleaver ensuring minimum separation S."""
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
        """
        Returns (fdm_text, token_ids) with turbo structure.
        First 256 tokens: signal in natural order
        Next 256 tokens: same samples, S-random interleaved
        """
        # Generate composite FDM signal
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
        
        # Encoder 1: natural order
        sig_min, sig_max = composite.min(), composite.max()
        sig_range = sig_max - sig_min + 1e-10
        norm1 = (composite - sig_min) / sig_range
        q1 = np.floor(norm1 * (self.num_levels - 1) + 0.5).astype(int)
        q1 = np.clip(q1, 0, self.num_levels - 1)
        tokens1 = [int(self.token_map[q]) for q in q1]
        
        # Encoder 2: S-random interleaved
        interleaved = composite[self.interleaver]
        norm2 = (interleaved - sig_min) / sig_range  # Same normalization range
        q2 = np.floor(norm2 * (self.num_levels - 1) + 0.5).astype(int)
        q2 = np.clip(q2, 0, self.num_levels - 1)
        tokens2 = [int(self.token_map[q]) for q in q2]
        
        all_tokens = tokens1 + tokens2
        fdm_text = self.tokenizer.decode(all_tokens)
        
        return fdm_text, all_tokens


# ============================================================
# Dataset
# ============================================================

class FDMReasoningDataset(Dataset):
    """Dataset: FDM tokens + question → reasoning response."""
    
    def __init__(self, path, tokenizer, max_length=1024):
        self.samples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        target_text = f" {s['answer']}"
        full_text = input_text + target_text
        
        encoding = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        
        # Mask input — model only learns to generate the answer
        input_len = len(self.tokenizer.encode(input_text))
        labels[:input_len] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


# ============================================================
# Data generation
# ============================================================

def generate_stage_data(stage_config, num_samples, output_prefix):
    """Generate training data for one curriculum stage."""
    
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
        
        # Extra channels 8-19 form operational context
        extra = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(8, NUM_CHANNELS)}
        
        q_template = random.choice(QUESTION_TEMPLATES)
        question = q_template["question"].format(**facts)
        base_answer = q_template["fn"](facts, rule, meta)
        
        # Append extra channel summary to answer so model must decode all 20
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
        print(f"  {split_name}: {len(split_data)} samples → {path}")
    
    return samples


# Curriculum stages
STAGES = [
    {
        "name": "Stage 0: Easy (max contrast, 256+256 turbo)",
        "num_tokens": 256,       # Per encoder. Total = 512 tokens.
        "a_high": 1.0,
        "a_low": 0.0,           # Maximum contrast
        "num_levels": 64,
        "samples": 15000,
        "epochs": 5,
        "lr": 5e-5,
    },
    {
        "name": "Stage 1: Standard ASK contrast",
        "num_tokens": 256,
        "a_high": 1.0,
        "a_low": 0.1,
        "num_levels": 64,
        "samples": 20000,
        "epochs": 5,
        "lr": 3e-5,
    },
    {
        "name": "Stage 2: Moderate",
        "num_tokens": 256,
        "a_high": 1.0,
        "a_low": 0.15,
        "num_levels": 64,
        "samples": 25000,
        "epochs": 7,
        "lr": 2e-5,
    },
    {
        "name": "Stage 3: Harder contrast",
        "num_tokens": 256,
        "a_high": 1.0,
        "a_low": 0.2,
        "num_levels": 64,
        "samples": 25000,
        "epochs": 7,
        "lr": 1e-5,
    },
    {
        "name": "Stage 4: Hardest",
        "num_tokens": 256,
        "a_high": 1.0,
        "a_low": 0.25,
        "num_levels": 64,
        "samples": 30000,
        "epochs": 10,
        "lr": 1e-5,
    },
]


def generate_all_stages():
    """Generate data for all curriculum stages."""
    for i, stage in enumerate(STAGES):
        print(f"\n{'='*60}")
        print(f"GENERATING STAGE {i}: {stage['name']}")
        print(f"{'='*60}")
        generate_stage_data(stage, stage['samples'], f"fdm_40ch_turbo_v3_stage{i}")


# ============================================================
# Training
# ============================================================

def train(checkpoint_dir="checkpoints_fdm_40ch_turbo_v3"):
    """Curriculum training: progressively harder FDM → reasoning."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Channels: {NUM_CHANNELS}")
    print(f"Carriers: {[1.0 + i for i in range(NUM_CHANNELS)]} Hz")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[MEMORY]', '[/MEMORY]']
    })
    
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # Check for resume
    resume_stage = 0
    resume_epoch = 0
    ckpt_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
    if ckpt_files:
        latest = ckpt_files[-1]
        print(f"Found checkpoint: {latest}")
        ckpt = torch.load(os.path.join(checkpoint_dir, latest))
        model.load_state_dict(ckpt['model_state_dict'])
        resume_stage = ckpt.get('stage', 0)
        resume_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resuming from stage {resume_stage}, epoch {resume_epoch}")
    
    for stage_idx, stage in enumerate(STAGES):
        if stage_idx < resume_stage:
            print(f"Skipping stage {stage_idx} (already completed)")
            continue
        
        prefix = f"fdm_40ch_turbo_v3_stage{stage_idx}"
        train_path = f"{prefix}_train.jsonl"
        val_path = f"{prefix}_val.jsonl"
        
        if not os.path.exists(train_path):
            print(f"ERROR: {train_path} not found. Run 'generate' first.")
            return
        
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx}: {stage['name']}")
        print(f"  lr={stage['lr']}, epochs={stage['epochs']}")
        print(f"  num_tokens={stage['num_tokens']}, a_low={stage['a_low']}")
        print(f"{'='*60}")
        
        train_dataset = FDMReasoningDataset(train_path, tokenizer)
        val_dataset = FDMReasoningDataset(val_path, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4)
        
        optimizer = AdamW(model.parameters(), lr=stage['lr'])
        best_val_loss = float('inf')
        
        start_epoch = resume_epoch if stage_idx == resume_stage else 0
        resume_epoch = 0
        
        for epoch in range(start_epoch, stage['epochs']):
            # Train
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"S{stage_idx} E{epoch+1} Train"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                train_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_loss /= len(train_loader)
            
            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
            val_loss /= len(val_loader)
            
            print(f"  Stage {stage_idx} Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
            
            # Quick eval
            if (epoch + 1) % 2 == 0 or epoch == stage['epochs'] - 1:
                quick_eval(model, tokenizer, val_path, device, num_samples=5, stage_idx=stage_idx)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  ✓ New best val loss: {val_loss:.4f}")
            
            # Checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'stage': stage_idx,
                'epoch': epoch,
                'val_loss': val_loss,
                'stage_config': {k: v for k, v in stage.items() if k != 'name'},
            }, os.path.join(checkpoint_dir, f'stage{stage_idx}_epoch{epoch+1}.pt'))
        
        # Save stage model
        model.save_pretrained(f'fdm_40ch_turbo_v3_model_stage{stage_idx}')
        tokenizer.save_pretrained(f'fdm_40ch_turbo_v3_model_stage{stage_idx}')
        print(f"Saved stage {stage_idx} model")
    
    # Save final
    model.save_pretrained('fdm_40ch_turbo_v3_model_final')
    tokenizer.save_pretrained('fdm_40ch_turbo_v3_model_final')
    print("\nSaved final model: fdm_40ch_turbo_v3_model_final/")


def quick_eval(model, tokenizer, val_path, device, num_samples=5, stage_idx=0):
    """Quick check: generate responses and compare."""
    samples = [json.loads(l) for l in open(val_path)][:num_samples]
    
    model.eval()
    correct_action = 0
    correct_fact = 0
    total = 0
    
    for s in samples:
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=250, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(output[0])[len(input_text):].strip()
        response = response.split('<|endoftext|>')[0].strip()
        expected = s['answer']
        
        # Action match
        for kw in ['abort', 'proceed', 'hold', 'wait', 'share', 'Do NOT share',
                    'EMERGENCY', 'LOCKDOWN', 'HIGH RISK', 'LOW RISK']:
            if kw.lower() in expected.lower() and kw.lower() in response.lower():
                correct_action += 1
                break
        
        # Fact match (check key entity)
        facts = s['facts']
        if s['question'].startswith("Should"):
            if facts['AGENT'] in response: correct_fact += 1
        elif s['question'].startswith("What"):
            if facts['LOCATION'] in response: correct_fact += 1
        elif s['question'].startswith("Is it"):
            if facts['SECRET'] in response or facts['LOCATION'] in response: correct_fact += 1
        
        total += 1
    
    print(f"    Quick eval (S{stage_idx}): action={correct_action}/{total}, fact={correct_fact}/{total}")


# ============================================================
# Full evaluation
# ============================================================

def evaluate(model_path='prompterminal/eidetic-fdm-40ch-turbo-v3', stage_idx=None):
    """Full evaluation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {model_path}/...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Use stage 2 test data by default, or specified stage
    si = stage_idx if stage_idx is not None else 2
    test_path = '/workspace/verified_repro/gpt2/fdm_40ch_turbo_v3_stage4_test.jsonl'
    
    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found.")
        return
    
    test_samples = [json.loads(l) for l in open(test_path)]
    print(f"Evaluating on {len(test_samples)} samples (stage {si})...\n")
    
    action_correct = 0
    rule_correct = 0
    meta_correct = 0
    fact_correct = 0
    extra_correct = 0  # channels 8-19
    extra_total = 0
    total = 0
    
    rule_stats = {r: {"correct": 0, "total": 0} for r in REASONING_RULES}
    meta_stats = {m: {"correct": 0, "total": 0} for m in META_INSTRUCTIONS}
    
    for i, s in enumerate(tqdm(test_samples, desc="Evaluating")):
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=250, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(output[0])[len(input_text):].strip()
        response = response.split('<|endoftext|>')[0].strip()
        expected = s['answer']
        rule = s['rule']
        meta = s['meta']
        
        # Action
        action_ok = False
        for kw in ['abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN',
                    'share', 'Do NOT share', 'Do not share', 'HIGH RISK', 'LOW RISK', 'MODERATE']:
            if kw.lower() in expected.lower() and kw.lower() in response.lower():
                action_ok = True
                break
        if action_ok: action_correct += 1
        
        # Rule
        rule_ok = (rule in response)
        if rule_ok: rule_correct += 1
        rule_stats[rule]["total"] += 1
        if rule_ok: rule_stats[rule]["correct"] += 1
        
        # Meta
        meta_ok = False
        if meta == "EMERGENCY" and "EMERGENCY" in response: meta_ok = True
        elif meta == "LOCKDOWN" and "LOCKDOWN" in response: meta_ok = True
        elif meta in ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY"]: meta_ok = True
        if meta_ok: meta_correct += 1
        meta_stats[meta]["total"] += 1
        if meta_ok: meta_stats[meta]["correct"] += 1
        
        # Fact
        fact_ok = False
        facts = s['facts']
        if s['question'].startswith("Should"):
            fact_ok = facts['AGENT'] in response
        elif s['question'].startswith("What is the risk"):
            fact_ok = facts['LOCATION'] in response
        elif s['question'].startswith("Is it safe"):
            fact_ok = facts['SECRET'] in response or facts['LOCATION'] in response
        if fact_ok: fact_correct += 1
        
        # Extra channels 8-19 accuracy
        memory = s['memory']
        for ch in range(8, NUM_CHANNELS):
            name = MEMORY_SCHEMAS[ch][0]
            expected_val = memory[str(ch)]
            extra_total += 1
            if expected_val in response:
                extra_correct += 1
        
        total += 1
        
        if i < 15:
            print(f"\n{'='*60}")
            print(f"Sample {i+1} | Rule: {rule} | Meta: {meta}")
            print(f"Memory: {s['explicit']}")
            print(f"Q: {s['question']}")
            print(f"Expected: {expected}")
            print(f"Got:      {response}")
            am = "✓" if action_ok else "✗"
            rm = "✓" if rule_ok else "✗"
            fm = "✓" if fact_ok else "✗"
            print(f"Action: {am} | Rule: {rm} | Fact: {fm}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"20-CHANNEL REAL FDM END-TO-END EVALUATION")
    print(f"{'='*60}")
    
    print(f"\n--- Overall ---")
    print(f"  Action correct: {100*action_correct/total:.1f}%")
    print(f"  Rule correct:   {100*rule_correct/total:.1f}%")
    print(f"  Meta correct:   {100*meta_correct/total:.1f}%")
    print(f"  Fact correct (ch 0-5): {100*fact_correct/total:.1f}%")
    print(f"  Extra correct (ch 8-19): {100*extra_correct/max(extra_total,1):.1f}% ({extra_correct}/{extra_total})")
    
    print(f"\n--- Per Rule ---")
    for r, stats in rule_stats.items():
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            print(f"  {r:15s}: {acc:.1f}% ({stats['correct']}/{stats['total']})")
    
    print(f"\n--- Per Meta ---")
    for m, stats in meta_stats.items():
        if stats['total'] > 0:
            acc = 100 * stats['correct'] / stats['total']
            print(f"  {m:20s}: {acc:.1f}% ({stats['correct']}/{stats['total']})")
    
    print(f"\n--- Comparison ---")
    print(f"  TDM 8ch (codebook):              Action=93.5%, Fact=99.0%")
    print(f"  FDM 8ch v1 (256t, 64l):          Action=80.0%, Fact=88.0%")
    print(f"  FDM 20ch single (256t, 64l):     Action=63.0%, Fact=70.5%, Extra=36.8%")
    print(f"  FDM 20ch TURBO (256+256t, 64l):  Action=72.0%, Fact=88.5%, Extra=48.8%")
    print(f"  FDM 40ch TURBO v1 (19K samples): Action=77.0%, Fact=81.5%, Extra=44.3%")
    print(f"  FDM 40ch TURBO v2 (46K samples): Action=94.2%, Fact=98.0%, Extra=63.4%")
    print(f"  FDM 40ch TURBO v3 (115K samples): Action={100*action_correct/total:.1f}%, Fact={100*fact_correct/total:.1f}%, Extra={100*extra_correct/max(extra_total,1):.1f}%")
    print(f"\n--- Configuration ---")
    print(f"  Channels: {NUM_CHANNELS}")
    print(f"  Carriers: 1.0 to {NUM_CHANNELS}.0 Hz (Nyquist: 50 Hz)")
    print(f"  Tokens: 256 per encoder x 2 = 512 total (S-random interleaved)")
    print(f"  Quantization: 64 levels")
    print(f"  Training: 115,000 samples, 34 epochs across 5 stages")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', type=int, default=256)
    args, _ = parser.parse_known_args()
    if args.tokens != 256:
        for s in STAGES:
            s['num_tokens'] = args.tokens
        print(f'Token budget override: {args.tokens} per encoder = {args.tokens*2} total')
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python eidetic_real_fdm_e2e.py generate    # Generate data (~fast, no GPU for encoding)")
        print("  python eidetic_real_fdm_e2e.py train       # Curriculum training (~12 hours)")
        print("  python eidetic_real_fdm_e2e.py eval [stage] # Evaluate (default: stage 2)")
        print()
        print("What this does:")
        print("  A single GPT-2-medium model learns to read REAL FDM tokens")
        print("  (sinusoidal carriers, superposition, quantization)")
        print("  and generate reasoning responses directly.")
        print()
        print("  Unlike TDM where each position carries 1 channel,")
        print("  every FDM token carries ALL 8 channels simultaneously.")
        print("  The model must implicitly learn frequency separation.")
        sys.exit(0)
    
    cmd = sys.argv[1]
    if cmd == "generate":
        generate_all_stages()
    elif cmd == "train":
        train()
    elif cmd == "eval":
        stage = int(sys.argv[2]) if len(sys.argv) > 2 else None
        evaluate(stage_idx=stage)
