"""
Plain-Text Memory Baseline for N-Hop Compositional Reasoning

Identical to nhop_training_multi.py in every respect EXCEPT:
  - Memory is encoded as plain text (NAME=VALUE, ...) not FDM signal
  - Training starts from the BASE pretrained model (no FDM stage)
  - This isolates exactly what FDM encoding contributes vs. plain retrieval

Comparison design:
  Condition A (FDM):       Base → FDM training → nhop training on FDM memory
  Condition B (Plaintext): Base → nhop training on plain text memory  ← this script

Same MEMORY_SCHEMAS, same NHOP_TEMPLATES, same hop logic, same train/val/test
split seed (77777), same question/answer format. Only the [MEMORY] block differs.

Usage:
  python nhop_plaintext_baseline.py generate gpt2
  python nhop_plaintext_baseline.py generate qwen3
  python nhop_plaintext_baseline.py generate lfm2
  python nhop_plaintext_baseline.py generate hermes3

  python nhop_plaintext_baseline.py train gpt2
  python nhop_plaintext_baseline.py train qwen3
  python nhop_plaintext_baseline.py train lfm2
  python nhop_plaintext_baseline.py train hermes3

  python nhop_plaintext_baseline.py eval gpt2
  python nhop_plaintext_baseline.py eval qwen3
  python nhop_plaintext_baseline.py eval lfm2
  python nhop_plaintext_baseline.py eval hermes3
"""

import json, random, os, sys
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# MODEL CONFIGS
# NOTE: base_model_dir points to the HuggingFace pretrained
# model ID — NOT the FDM fine-tuned model. This is the critical
# difference from nhop_training_multi.py.
# ============================================================

MODEL_CONFIGS = {
    "gpt2": {
        "model_id":          "openai-community/gpt2-medium",
        "vocab_size":        50257,
        "dtype":             torch.float32,
        "batch_size":        4,
        "grad_accum":        1,
        "trust_remote_code": False,
        "model_class":       "gpt2",
        # plaintext trains from HF base, not FDM-trained checkpoint
        "base_model_id":     "openai-community/gpt2-medium",
        "nhop_data_dir":     "nhop_plaintext_data_gpt2",
        "nhop_model_dir":    "nhop_plaintext_model_gpt2_final",
    },
    "qwen3": {
        "model_id":          "Qwen/Qwen3-0.6B-Base",
        "vocab_size":        151936,
        "dtype":             torch.bfloat16,
        "batch_size":        4,
        "grad_accum":        1,
        "trust_remote_code": True,
        "model_class":       "auto",
        "base_model_id":     "Qwen/Qwen3-0.6B-Base",
        "nhop_data_dir":     "nhop_plaintext_data_qwen3",
        "nhop_model_dir":    "nhop_plaintext_model_qwen3_final",
    },
    "hermes3": {
        "model_id":          "NousResearch/Hermes-3-Llama-3.2-3B",
        "vocab_size":        128256,
        "dtype":             torch.bfloat16,
        "batch_size":        2,
        "grad_accum":        2,
        "trust_remote_code": True,
        "model_class":       "auto",
        "base_model_id":     "NousResearch/Hermes-3-Llama-3.2-3B",
        "nhop_data_dir":     "nhop_plaintext_data_hermes3",
        "nhop_model_dir":    "nhop_plaintext_model_hermes3_final",
    },
    "lfm2": {
        "model_id":          "LiquidAI/LFM2.5-1.2B-Base",
        "vocab_size":        65536,
        "dtype":             torch.bfloat16,
        "batch_size":        2,
        "grad_accum":        2,
        "trust_remote_code": True,
        "model_class":       "auto",
        "base_model_id":     "LiquidAI/LFM2.5-1.2B-Base",
        "nhop_data_dir":     "nhop_plaintext_data_lfm2",
        "nhop_model_dir":    "nhop_plaintext_model_lfm2_final",
    },
}


# ============================================================
# Channel definitions — identical to nhop_training_multi.py
# ============================================================

MEMORY_SCHEMAS = {
    0:  ("SECRET",      ["RED", "BLUE", "GREEN", "GOLD"]),
    1:  ("LOCATION",    ["PARIS", "TOKYO", "LONDON", "BERLIN"]),
    2:  ("AGENT",       ["ALICE", "BOB", "CAROL", "DAVE"]),
    3:  ("STATUS",      ["CLEAR", "COMPROMISED", "UNKNOWN"]),
    4:  ("PRIORITY",    ["HIGH", "MEDIUM", "LOW"]),
    5:  ("BACKUP",      ["AVAILABLE", "UNAVAILABLE"]),
    6:  ("RULE",        ["SAFETY_FIRST", "MISSION_FIRST", "BALANCED", "CAUTIOUS"]),
    7:  ("META",        ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY",
                         "EMERGENCY", "LOCKDOWN"]),
    8:  ("TEAM",        ["RED_TEAM", "BLUE_TEAM", "GREEN_TEAM", "GOLD_TEAM"]),
    9:  ("REGION",      ["NORTH", "SOUTH", "EAST", "WEST"]),
    10: ("PHASE",       ["ALPHA", "BETA", "GAMMA", "DELTA"]),
    11: ("COMM",        ["OPEN", "CLOSED", "RESTRICTED"]),
    12: ("ASSET",       ["VEHICLE", "AIRCRAFT", "DRONE", "BOAT"]),
    13: ("WINDOW",      ["DAWN", "MIDDAY", "DUSK", "NIGHT"]),
    14: ("COVER",       ["DEEP", "SHALLOW", "NONE"]),
    15: ("SUPPORT",     ["ACTIVE", "STANDBY", "OFFLINE"]),
    16: ("THREAT",      ["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
    17: ("WEATHER",     ["CLEAR", "STORM", "FOG"]),
    18: ("TERRAIN",     ["URBAN", "RURAL", "COASTAL", "MOUNTAIN"]),
    19: ("EXTRACT",     ["READY", "DELAYED", "UNAVAILABLE"]),
    20: ("CIPHER",      ["AES", "RSA", "BLOWFISH", "TWOFISH"]),
    21: ("FREQ",        ["HF", "VHF", "UHF", "SHF"]),
    22: ("PAYLOAD",     ["LIGHT", "MEDIUM", "HEAVY", "CRITICAL"]),
    23: ("ROUTE",       ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]),
    24: ("DURATION",    ["SHORT", "MEDIUM", "LONG", "EXTENDED"]),
    25: ("CONTACT",     ["FRIENDLY", "NEUTRAL", "HOSTILE", "UNKNOWN"]),
    26: ("FUEL",        ["FULL", "HALF", "LOW", "CRITICAL"]),
    27: ("ALTITUDE",    ["LOW", "MEDIUM", "HIGH"]),
    28: ("VISIBILITY",  ["CLEAR", "REDUCED", "ZERO"]),
    29: ("NOISE",       ["SILENT", "QUIET", "MODERATE", "LOUD"]),
    30: ("FORMATION",   ["SINGLE", "PAIR", "SQUAD", "PLATOON"]),
    31: ("ARMOR",       ["NONE", "LIGHT", "MEDIUM", "HEAVY"]),
    32: ("SIGNAL",      ["STRONG", "WEAK", "JAMMED", "LOST"]),
    33: ("MORALE",      ["HIGH", "MEDIUM", "LOW"]),
    34: ("SUPPLY",      ["ABUNDANT", "ADEQUATE", "SCARCE", "DEPLETED"]),
    35: ("INTEL",       ["CONFIRMED", "PROBABLE", "UNCERTAIN", "NONE"]),
    36: ("EVAC",        ["STANDING", "PREPPED", "LAUNCHED", "ABORTED"]),
    37: ("WEATHER2",    ["SUNNY", "OVERCAST", "RAIN", "SNOW"]),
    38: ("DOCTRINE",    ["OFFENSIVE", "DEFENSIVE", "RECON", "SUPPORT"]),
    39: ("COMMS",       ["SECURE", "OPEN", "COMPROMISED", "SILENT"]),
}

NUM_CHANNELS = 40


# ============================================================
# Plain text encoder — replaces TurboFDMSignalEncoder
# ============================================================

def encode_memory_plaintext(memory):
    """
    Serialize all 40 channel values as a comma-separated NAME=VALUE string.
    This is what the model sees in [MEMORY]...[/MEMORY] instead of the
    FDM signal token sequence.

    The string is intentionally compact — no spaces around = — to be
    consistent with the NAME=VALUE format used in the answer's Context
    section. This gives the plain text model the easiest possible read
    task: direct string matching.
    """
    parts = []
    for ch in range(NUM_CHANNELS):
        name = MEMORY_SCHEMAS[ch][0]
        value = memory[ch]
        parts.append(f"{name}={value}")
    return ", ".join(parts)


# ============================================================
# Hop template functions — copied exactly from
# nhop_training_multi.py, no changes
# ============================================================

def hop1_status(m):
    return f"STATUS_REPORT: Status is {m[3]}."

def hop1_threat(m):
    return f"THREAT_REPORT: Threat level is {m[16]}."

def hop1_rule(m):
    return f"RULE_REPORT: Current rule is {m[6]}."

def hop1_location(m):
    return f"LOCATION_REPORT: Location is {m[1]}."


def hop2_rule_status(m):
    rule, status = m[6], m[3]
    if rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            return f"RULE_STATUS: {rule} with {status} status. Decision: ABORT."
        elif status == "CLEAR":
            return f"RULE_STATUS: {rule} with {status} status. Decision: PROCEED."
        else:
            return f"RULE_STATUS: {rule} with {status} status. Decision: WAIT."
    elif rule == "MISSION_FIRST":
        if status == "COMPROMISED":
            return f"RULE_STATUS: {rule} with {status} status. Decision: PROCEED WITH CAUTION."
        else:
            return f"RULE_STATUS: {rule} with {status} status. Decision: PROCEED."
    elif rule == "BALANCED":
        if status == "COMPROMISED":
            return f"RULE_STATUS: {rule} with {status} status. Decision: ABORT."
        else:
            return f"RULE_STATUS: {rule} with {status} status. Decision: PROCEED."
    elif rule == "CAUTIOUS":
        if status == "CLEAR":
            return f"RULE_STATUS: {rule} with {status} status. Decision: PROCEED."
        else:
            return f"RULE_STATUS: {rule} with {status} status. Decision: ABORT."
    return f"RULE_STATUS: {rule} with {status} status."

def hop2_threat_terrain(m):
    threat, terrain = m[16], m[18]
    if threat in ("HIGH", "CRITICAL"):
        return f"THREAT_TERRAIN: {threat} threat in {terrain} terrain. Recommend EVACUATION."
    elif threat == "MEDIUM" and terrain in ("URBAN", "COASTAL"):
        return f"THREAT_TERRAIN: {threat} threat in {terrain} terrain. Recommend FORTIFY."
    else:
        return f"THREAT_TERRAIN: {threat} threat in {terrain} terrain. Recommend CONTINUE."

def hop2_agent_location(m):
    agent, location = m[2], m[1]
    return (f"AGENT_LOCATION: {agent} is deployed in {location}. "
            f"Cover status depends on location.")


def hop3_rule_status_backup(m):
    rule, status, backup = m[6], m[3], m[5]
    if rule == "CAUTIOUS":
        if status == "CLEAR" and backup == "AVAILABLE":
            return (f"CHAIN_3HOP: {rule} requires CLEAR+AVAILABLE. "
                    f"Have {status}/{backup}. PROCEED.")
        else:
            return (f"CHAIN_3HOP: {rule} requires CLEAR+AVAILABLE. "
                    f"Have {status}/{backup}. ABORT.")
    elif rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            return (f"CHAIN_3HOP: {rule} says {status} overrides all. "
                    f"ABORT. Backup {backup} irrelevant.")
        elif backup == "UNAVAILABLE":
            return (f"CHAIN_3HOP: {rule} with {status} status but "
                    f"backup {backup}. HOLD.")
        else:
            return (f"CHAIN_3HOP: {rule} with {status} and backup "
                    f"{backup}. PROCEED.")
    elif rule == "MISSION_FIRST":
        if status == "COMPROMISED" and backup == "UNAVAILABLE":
            return (f"CHAIN_3HOP: {rule} but {status} with no backup. "
                    f"Even mission says ABORT.")
        else:
            return (f"CHAIN_3HOP: {rule} overrides {status}. "
                    f"Backup {backup}. PROCEED.")
    elif rule == "BALANCED":
        score = 0
        if status == "CLEAR":    score += 1
        if backup == "AVAILABLE": score += 1
        if score >= 2:
            return (f"CHAIN_3HOP: {rule} scores {score}/2. "
                    f"{status}/{backup}. PROCEED.")
        elif score == 1:
            return (f"CHAIN_3HOP: {rule} scores {score}/2. "
                    f"{status}/{backup}. CAUTION.")
        else:
            return (f"CHAIN_3HOP: {rule} scores {score}/2. "
                    f"{status}/{backup}. ABORT.")
    return f"CHAIN_3HOP: {rule}, {status}, {backup}."

def hop3_threat_cover_signal(m):
    threat, cover, signal = m[16], m[14], m[32]
    if threat in ("HIGH", "CRITICAL") and cover == "NONE":
        return (f"THREAT_CHAIN: {threat} threat with {cover} cover. "
                f"Signal {signal}. CRITICAL EXPOSURE.")
    elif signal in ("JAMMED", "LOST"):
        return (f"THREAT_CHAIN: {threat} threat, {cover} cover, but "
                f"signal {signal}. COMMS COMPROMISED.")
    else:
        return (f"THREAT_CHAIN: {threat} threat, {cover} cover, "
                f"signal {signal}. MANAGEABLE.")


def hop4_meta_rule_status_priority(m):
    meta, rule, status, priority = m[7], m[6], m[3], m[4]
    if meta == "EMERGENCY":
        return (f"OVERRIDE_4HOP: {meta} active. Skip "
                f"{rule}/{status}/{priority}. PROCEED IMMEDIATELY.")
    if meta == "LOCKDOWN":
        return (f"OVERRIDE_4HOP: {meta} active. Skip "
                f"{rule}/{status}/{priority}. ABORT ALL.")
    if meta == "OVERRIDE_STATUS":
        if priority == "HIGH":
            return (f"OVERRIDE_4HOP: {meta} ignores {status}. "
                    f"{rule} with {priority} priority. PROCEED.")
        else:
            return (f"OVERRIDE_4HOP: {meta} ignores {status}. "
                    f"{rule} with {priority} priority. CAUTION.")
    if rule == "SAFETY_FIRST" and status == "COMPROMISED":
        return (f"CHAIN_4HOP: {meta}/{rule}/{status}/{priority}. "
                f"Safety overrides {priority}. ABORT.")
    elif priority == "HIGH":
        return (f"CHAIN_4HOP: {meta}/{rule}/{status}/{priority}. "
                f"High priority pushes PROCEED.")
    else:
        return (f"CHAIN_4HOP: {meta}/{rule}/{status}/{priority}. "
                f"Standard assessment. HOLD.")

def hop4_threat_terrain_weather_asset(m):
    threat, terrain, weather, asset = m[16], m[18], m[17], m[12]
    if threat in ("HIGH", "CRITICAL") and weather == "STORM":
        return (f"TACTICAL_4HOP: {threat}/{terrain}/{weather}/{asset}. "
                f"Storm + high threat. GROUND {asset}.")
    elif asset in ("AIRCRAFT", "DRONE") and weather == "FOG":
        return (f"TACTICAL_4HOP: {threat}/{terrain}/{weather}/{asset}. "
                f"Fog grounds air assets. DELAY.")
    elif terrain == "MOUNTAIN" and asset == "BOAT":
        return (f"TACTICAL_4HOP: {threat}/{terrain}/{weather}/{asset}. "
                f"Terrain mismatch. REASSIGN.")
    else:
        return (f"TACTICAL_4HOP: {threat}/{terrain}/{weather}/{asset}. "
                f"Conditions acceptable. DEPLOY.")


def hop5_meta_rule_status_backup_threat(m):
    meta, rule, status, backup, threat = m[7], m[6], m[3], m[5], m[16]
    if meta == "EMERGENCY":
        return f"FULL_5HOP: {meta} overrides all. Threat {threat} noted. PROCEED."
    if meta == "LOCKDOWN":
        return f"FULL_5HOP: {meta} overrides all. ABORT regardless of {threat}."
    if rule == "SAFETY_FIRST" and status == "COMPROMISED":
        if threat in ("HIGH", "CRITICAL"):
            return (f"FULL_5HOP: {rule}+{status}+{threat}. "
                    f"Escalate to EMERGENCY EXTRACT.")
        else:
            return (f"FULL_5HOP: {rule}+{status}. "
                    f"Threat only {threat}. Standard ABORT.")
    if backup == "UNAVAILABLE" and threat in ("HIGH", "CRITICAL"):
        return (f"FULL_5HOP: {rule}/{status}, no backup, {threat} threat. "
                f"ABORT with EXTRACT REQUEST.")
    elif backup == "AVAILABLE":
        return (f"FULL_5HOP: {rule}/{status}, backup ready, {threat} threat. "
                f"PROCEED with backup.")
    else:
        return f"FULL_5HOP: {rule}/{status}/{backup}/{threat}. Assess and HOLD."

def hop5_weather_vis_window_terrain_extract(m):
    weather, vis, window, terrain, extract = (
        m[17], m[28], m[13], m[18], m[19]
    )
    if weather == "STORM" and vis == "ZERO":
        if extract == "READY":
            return (f"ENV_5HOP: {weather}/{vis}/{window}/{terrain}. "
                    f"Extract {extract}. EVACUATE.")
        else:
            return (f"ENV_5HOP: {weather}/{vis}/{window}/{terrain}. "
                    f"Extract {extract}. SHELTER.")
    elif window == "NIGHT" and vis == "REDUCED":
        return (f"ENV_5HOP: {weather}/{vis}/{window}/{terrain}. "
                f"Limited ops. Extract {extract}. HOLD.")
    elif terrain == "MOUNTAIN" and weather in ("STORM", "FOG"):
        return (f"ENV_5HOP: {weather}/{vis}/{window}/{terrain}. "
                f"Mountain hazard. Extract {extract}. DESCEND.")
    else:
        return (f"ENV_5HOP: {weather}/{vis}/{window}/{terrain}. "
                f"Conditions OK. Extract {extract}. CONTINUE.")


NHOP_TEMPLATES = {
    1: [
        {"question": "Report current status.",
         "fn": hop1_status,   "channels": [3]},
        {"question": "Report threat level.",
         "fn": hop1_threat,   "channels": [16]},
        {"question": "Report active rule.",
         "fn": hop1_rule,     "channels": [6]},
        {"question": "Report current location.",
         "fn": hop1_location, "channels": [1]},
    ],
    2: [
        {"question": "Evaluate status under current rule.",
         "fn": hop2_rule_status,    "channels": [6, 3]},
        {"question": "Assess threat in current terrain.",
         "fn": hop2_threat_terrain, "channels": [16, 18]},
        {"question": "Report agent deployment location.",
         "fn": hop2_agent_location, "channels": [2, 1]},
    ],
    3: [
        {"question": "Should we proceed given rule, status, and backup?",
         "fn": hop3_rule_status_backup,   "channels": [6, 3, 5]},
        {"question": "Evaluate threat exposure with cover and signal status.",
         "fn": hop3_threat_cover_signal,  "channels": [16, 14, 32]},
    ],
    4: [
        {"question": "Full override check: meta, rule, status, priority.",
         "fn": hop4_meta_rule_status_priority,    "channels": [7, 6, 3, 4]},
        {"question": "Tactical assessment: threat, terrain, weather, asset.",
         "fn": hop4_threat_terrain_weather_asset, "channels": [16, 18, 17, 12]},
    ],
    5: [
        {"question": "Complete chain: meta override, rule, status, backup, threat.",
         "fn": hop5_meta_rule_status_backup_threat,      "channels": [7, 6, 3, 5, 16]},
        {"question": "Environmental chain: weather, visibility, window, terrain, extract.",
         "fn": hop5_weather_vis_window_terrain_extract,  "channels": [17, 28, 13, 18, 19]},
    ],
}


# ============================================================
# Dataset
# ============================================================

class PlaintextNHopDataset(Dataset):
    """
    Same as NHopDataset in nhop_training_multi.py but reads
    'mem_text' (plain text) instead of 'fdm_text' (FDM signal).
    """
    def __init__(self, path, tokenizer, max_length=512):
        self.samples  = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        # 512 is sufficient — plain text memory is much shorter
        # than the 512-token FDM signal block
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        input_text = (
            f"[MEMORY]{s['mem_text']}[/MEMORY]\n"
            f"Question: {s['question']}\n"
            f"Answer:"
        )
        target_text = f" {s['answer']}"
        full_text   = input_text + target_text

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        input_ids      = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels         = input_ids.clone()

        input_len = len(self.tokenizer.encode(input_text))
        labels[:input_len]          = -100
        labels[attention_mask == 0] = -100

        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels,
        }


# ============================================================
# Data generation
# ============================================================

def generate_plaintext_data(model_name, samples_per_hop=20000):
    """
    Generate nhop training data using plain text memory encoding.
    Uses same seed (77777) as nhop_training_multi.py so the memory
    configurations are identical — only the encoding differs.
    """
    cfg        = MODEL_CONFIGS[model_name]
    output_dir = cfg["nhop_data_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # We need a tokenizer only to verify token lengths — no encoder needed
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_id"], trust_remote_code=cfg["trust_remote_code"]
    )

    # Same seeds as nhop_training_multi.py
    random.seed(77777)
    np.random.seed(77777)

    all_samples = []
    max_mem_tokens = 0

    for nhops, templates in sorted(NHOP_TEMPLATES.items()):
        print(f"Generating {samples_per_hop} {model_name} samples "
              f"for {nhops}-hop...")

        for _ in tqdm(range(samples_per_hop)):
            memory = {
                ch: random.choice(MEMORY_SCHEMAS[ch][1])
                for ch in range(NUM_CHANNELS)
            }

            # Plain text instead of FDM signal
            mem_text = encode_memory_plaintext(memory)

            template    = random.choice(templates)
            base_answer = template["fn"](memory)

            extra       = {
                MEMORY_SCHEMAS[ch][0]: memory[ch]
                for ch in range(8, NUM_CHANNELS)
            }
            extra_parts = [f"{k}={v}" for k, v in extra.items()]
            answer      = f"{base_answer} Context: {', '.join(extra_parts)}."

            all_samples.append({
                "mem_text": mem_text,
                "question": template["question"],
                "answer":   answer,
                "nhops":    nhops,
                "channels": template["channels"],
                "memory":   {str(ch): memory[ch] for ch in range(NUM_CHANNELS)},
            })

            # Track memory token length for reporting
            tok_len = len(tokenizer.encode(mem_text))
            max_mem_tokens = max(max_mem_tokens, tok_len)

    # Report memory size vs FDM
    print(f"\nPlain text memory: max {max_mem_tokens} tokens per sample")
    print(f"FDM memory: 512 tokens per sample (fixed)")
    print(f"Ratio: {max_mem_tokens/512:.2f}x")

    # Same 85/10/5 split as nhop_training_multi.py
    random.shuffle(all_samples)
    n  = len(all_samples)
    te = int(0.85 * n)
    ve = int(0.95 * n)
    splits = {
        "train": all_samples[:te],
        "val":   all_samples[te:ve],
        "test":  all_samples[ve:],
    }

    for name, data in splits.items():
        path = os.path.join(output_dir, f"nhop_{name}.jsonl")
        with open(path, "w") as f:
            for s in data:
                f.write(json.dumps(s) + "\n")
        print(f"  {name}: {len(data)} samples → {path}")

    print(f"\nTotal: {n} samples for {model_name} (plaintext baseline)")


# ============================================================
# Training
# ============================================================

def train(model_name, epochs=10, lr=1e-5):
    """
    Train on plain text memory nhop data starting from the BASE
    pretrained model — no FDM training stage.

    This is the key difference from nhop_training_multi.py:
      nhop_training_multi.py loads from: cfg["base_model_dir"]
                                         (= FDM fine-tuned checkpoint)
      THIS script loads from:            cfg["base_model_id"]
                                         (= HuggingFace pretrained model)
    """
    cfg        = MODEL_CONFIGS[model_name]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = cfg["base_model_id"]   # HF pretrained, not FDM checkpoint
    data_dir   = cfg["nhop_data_dir"]
    output_path = cfg["nhop_model_dir"]

    print(f"{'='*60}")
    print(f"PLAINTEXT BASELINE — {model_name.upper()}")
    print(f"  Base model : {base_model}  (pretrained, no FDM stage)")
    print(f"  Data dir   : {data_dir}")
    print(f"  Output     : {output_path}")
    print(f"  Device     : {device} | dtype: {cfg['dtype']}")
    print(f"{'='*60}\n")

    train_path = os.path.join(data_dir, "nhop_train.jsonl")
    val_path   = os.path.join(data_dir, "nhop_val.jsonl")
    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found. "
              f"Run 'generate {model_name}' first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=cfg["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg["model_class"] == "gpt2":
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(base_model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=cfg["dtype"],
            trust_remote_code=True,
        )

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # max_length=512 sufficient for plain text (vs 1024 for FDM)
    train_dataset = PlaintextNHopDataset(train_path, tokenizer, max_length=512)
    val_dataset   = PlaintextNHopDataset(val_path,   tokenizer, max_length=512)
    train_loader  = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True
    )
    val_loader    = DataLoader(val_dataset, batch_size=cfg["batch_size"])

    if (bnb is not None
            and cfg["model_class"] == "auto"
            and cfg["dtype"] == torch.bfloat16):
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
        print("Using 8-bit AdamW")
    else:
        optimizer = AdamW(model.parameters(), lr=lr)

    grad_accum = cfg["grad_accum"]
    best_val   = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(
            tqdm(train_loader, desc=f"E{epoch+1}/{epochs} [{model_name}-pt]")
        ):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum
            train_loss += loss.item() * grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        # handle leftover steps
        if (step + 1) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels         = batch['labels'].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                val_loss += outputs.loss.item()
        val_loss /= len(val_loader)

        tag = " *best*" if val_loss < best_val else ""
        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            print(f"  Saved best → {output_path}/")

        print(f"  Epoch {epoch+1}: Train={train_loss:.4f} "
              f"Val={val_loss:.4f}{tag}")

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"\nDone. Final model saved: {output_path}/")


# ============================================================
# Evaluation — identical logic to nhop_training_multi.py
# so results are directly comparable
# ============================================================

def evaluate(model_name, model_path=None):
    cfg        = MODEL_CONFIGS[model_name]
    if model_path is None:
        model_path = cfg["nhop_model_dir"]
    data_dir   = cfg["nhop_data_dir"]
    test_path  = os.path.join(data_dir, "nhop_test.jsonl")

    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} plaintext model from {model_path}/...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=cfg["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg["model_class"] == "gpt2":
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=cfg["dtype"],
            trust_remote_code=True,
        )
    model.to(device)
    model.eval()

    samples = [json.loads(l) for l in open(test_path)]
    print(f"Evaluating {len(samples)} samples...\n")

    hop_results        = defaultdict(
        lambda: {"total": 0, "action_correct": 0, "fact_correct": 0}
    )
    per_channel_correct = defaultdict(int)
    per_channel_total   = defaultdict(int)
    per_sample_wrong    = []

    ACTION_KEYWORDS = [
        'ABORT', 'PROCEED', 'HOLD', 'WAIT', 'CAUTION', 'EVACUATE',
        'EMERGENCY', 'LOCKDOWN', 'CRITICAL', 'DEPLOY', 'DELAY',
        'SHELTER', 'DESCEND', 'CONTINUE', 'FORTIFY', 'REASSIGN',
        'GROUND', 'EXTRACT',
    ]

    for i, s in enumerate(tqdm(samples, desc=f"Eval [{model_name}-plaintext]")):
        input_text = (
            f"[MEMORY]{s['mem_text']}[/MEMORY]\n"
            f"Question: {s['question']}\n"
            f"Answer:"
        )
        input_ids = tokenizer.encode(
            input_text, return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=250,
                do_sample=False,
                pad_token_id=(
                    tokenizer.pad_token_id or tokenizer.eos_token_id
                ),
            )

        response = tokenizer.decode(
            output[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        expected = s['answer']
        nhops    = s['nhops']

        hop_results[nhops]["total"] += 1

        # Decision accuracy: keyword match
        for kw in ACTION_KEYWORDS:
            if (kw.lower() in expected.lower()
                    and kw.lower() in response.lower()):
                hop_results[nhops]["action_correct"] += 1
                break

        # Fact accuracy: channel values in decision channels
        memory = s['memory']
        for ch_id in s['channels']:
            expected_val = memory[str(ch_id)]
            if expected_val in response:
                hop_results[nhops]["fact_correct"] += 1

        # Extra-channel accuracy (ch8–39)
        sample_wrongs = []
        for ch in range(8, NUM_CHANNELS):
            name         = MEMORY_SCHEMAS[ch][0]
            expected_val = memory[str(ch)]
            per_channel_total[ch] += 1
            if f"{name}={expected_val}" in response:
                per_channel_correct[ch] += 1
            else:
                sample_wrongs.append((ch, name, expected_val))
        per_sample_wrong.append(sample_wrongs)

        if i < 3:
            print(f"\n  [{model_name}-pt] Sample {i+1} | {nhops}-hop")
            print(f"  Expected: {expected[:120]}...")
            print(f"  Got:      {response[:120]}...")

    # ── Results ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} PLAINTEXT BASELINE — N-HOP RESULTS")
    print(f"{'='*70}")
    print(f"{'Hops':<6} {'Decision%':<12} {'Fact%':<12} {'N':<6}")
    print(f"{'-'*36}")
    for nhops in sorted(hop_results.keys()):
        r     = hop_results[nhops]
        t     = r['total']
        a_pct = 100 * r['action_correct'] / t if t else 0
        f_pct = 100 * r['fact_correct'] / (t * nhops) if t else 0
        print(f"{nhops:<6} {a_pct:<12.1f} {f_pct:<12.1f} {t:<6}")

    # Per-channel accuracy
    print(f"\nPer-Channel Extra Accuracy (ch8–39):")
    for ch in range(8, NUM_CHANNELS):
        t    = per_channel_total[ch]
        c    = per_channel_correct[ch]
        pct  = 100 * c / t if t else 0
        flag = " <<<" if pct < 95 else ""
        print(f"  ch{ch:2d} ({MEMORY_SCHEMAS[ch][0]:<12}): {pct:.2f}%{flag}")

    # Joint accuracy
    num_extra = NUM_CHANNELS - 8
    total     = len(per_sample_wrong)
    print(f"\nJoint Accuracy (all N extra channels correct per sample):")
    print(f"{'N channels':<12} {'Correct':<16} {'Pct':<10}")
    for n in [1, 2, 4, 8, 16, 24, 32]:
        if n > num_extra:
            break
        extra_channels = list(range(8, 8 + n))
        count = sum(
            1 for wrongs in per_sample_wrong
            if all(
                ch not in [w[0] for w in wrongs]
                for ch in extra_channels
            )
        )
        pct = 100 * count / total if total else 0
        print(f"  {n:<10} {count}/{total:<14} {pct:.1f}%")

    all_correct = sum(1 for w in per_sample_wrong if len(w) == 0)
    print(f"\n  ALL {num_extra} correct: "
          f"{all_correct}/{total} = {100*all_correct/total:.1f}%")

    # Error analysis
    error_samples   = [(i, w) for i, w in enumerate(per_sample_wrong) if w]
    ch_error_count  = defaultdict(int)
    for _, wrongs in error_samples:
        for ch, name, exp in wrongs:
            ch_error_count[ch] += 1

    print(f"\nError Analysis: {len(error_samples)}/{total} samples have errors")
    if ch_error_count:
        print(f"Most error-prone channels:")
        for ch, count in sorted(
            ch_error_count.items(), key=lambda x: -x[1]
        )[:10]:
            print(f"  ch{ch:2d} ({MEMORY_SCHEMAS[ch][0]}): {count} errors")

    # Save — same JSON structure as nhop_training_multi.py
    # so results can be loaded by the same comparison scripts
    results_dir = f"nhop_plaintext_results_{model_name}"
    os.makedirs(results_dir, exist_ok=True)
    results = {
        "model":           f"{model_name}_plaintext",
        "memory_format":   "plaintext",
        "hop_results": {
            str(k): v for k, v in hop_results.items()
        },
        "per_channel": {
            str(ch): 100 * per_channel_correct[ch] / per_channel_total[ch]
            for ch in range(8, NUM_CHANNELS)
            if per_channel_total[ch] > 0
        },
        "joint_accuracy": {},
        "all_32_correct_pct": 100 * all_correct / total if total else 0,
        "error_samples":  len(error_samples),
        "total_samples":  total,
    }

    # Also save joint accuracy breakdown
    for n in [1, 2, 4, 8, 16, 24, 32]:
        if n > num_extra:
            break
        extra_channels = list(range(8, 8 + n))
        count = sum(
            1 for wrongs in per_sample_wrong
            if all(
                ch not in [w[0] for w in wrongs]
                for ch in extra_channels
            )
        )
        results["joint_accuracy"][str(n)] = {
            "correct": count,
            "total":   total,
            "pct":     100 * count / total if total else 0,
        }

    out_path = os.path.join(results_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


# ============================================================
# Compare FDM vs plaintext results
# ============================================================

def compare_results(model_name):
    """
    Load results from both conditions and print a side-by-side table.
    Run after both nhop_training_multi.py eval and this script's eval.
    """
    fdm_path = f"nhop_results_{model_name}/results.json"
    pt_path  = f"nhop_plaintext_results_{model_name}/results.json"

    if not os.path.exists(fdm_path):
        print(f"FDM results not found: {fdm_path}")
        print(f"Run: python nhop_training_multi.py eval {model_name}")
        return
    if not os.path.exists(pt_path):
        print(f"Plaintext results not found: {pt_path}")
        print(f"Run: python nhop_plaintext_baseline.py eval {model_name}")
        return

    with open(fdm_path) as f:
        fdm = json.load(f)
    with open(pt_path) as f:
        pt  = json.load(f)

    print(f"\n{'='*72}")
    print(f"FDM vs PLAINTEXT COMPARISON — {model_name.upper()}")
    print(f"{'='*72}")
    print(f"\n── Decision Accuracy by Hop ──")
    print(f"{'Hops':<6} {'FDM Dec%':<12} {'PT Dec%':<12} {'FDM Fact%':<12} {'PT Fact%':<12}")
    print(f"{'-'*54}")

    for nhops in sorted([int(k) for k in fdm["hop_results"].keys()]):
        k     = str(nhops)
        f_r   = fdm["hop_results"][k]
        p_r   = pt["hop_results"].get(k, {"total": 0, "action_correct": 0,
                                          "fact_correct": 0})
        f_dec = 100 * f_r["action_correct"] / f_r["total"] if f_r["total"] else 0
        p_dec = 100 * p_r["action_correct"] / p_r["total"] if p_r["total"] else 0
        f_fct = 100 * f_r["fact_correct"] / (f_r["total"] * nhops) if f_r["total"] else 0
        p_fct = 100 * p_r["fact_correct"] / (p_r["total"] * nhops) if p_r["total"] else 0
        print(f"{nhops:<6} {f_dec:<12.1f} {p_dec:<12.1f} "
              f"{f_fct:<12.1f} {p_fct:<12.1f}")

    print(f"\n── Joint Extra-Channel Accuracy ──")
    print(f"{'N':<6} {'FDM Joint%':<14} {'PT Joint%':<14}")
    print(f"{'-'*34}")

    fdm_joint = fdm.get("joint_accuracy", {})
    pt_joint  = pt.get("joint_accuracy", {})
    for n in [1, 2, 4, 8, 16, 24, 32]:
        k     = str(n)
        f_pct = fdm_joint.get(k, {}).get("pct", float('nan'))
        p_pct = pt_joint.get(k, {}).get("pct", float('nan'))
        print(f"{n:<6} {f_pct:<14.1f} {p_pct:<14.1f}")

    print(f"\n── All-32-Channel Joint Accuracy ──")
    print(f"  FDM:       {fdm.get('all_32_correct_pct', 0):.1f}%")
    print(f"  Plaintext: {pt.get('all_32_correct_pct',  0):.1f}%")

    print(f"\n── Per-Channel Accuracy (FDM vs PT) ──")
    print(f"{'Ch':<5} {'Name':<14} {'FDM%':<10} {'PT%':<10} {'Delta':<10}")
    print(f"{'-'*49}")
    for ch in range(8, NUM_CHANNELS):
        k     = str(ch)
        name  = MEMORY_SCHEMAS[ch][0]
        f_pct = fdm["per_channel"].get(k, 0)
        p_pct = pt["per_channel"].get(k, 0)
        delta = p_pct - f_pct
        flag  = " <<<" if abs(delta) > 2.0 else ""
        print(f"{ch:<5} {name:<14} {f_pct:<10.2f} {p_pct:<10.2f} "
              f"{delta:+.2f}{flag}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Plain-Text Memory Baseline for N-Hop Compositional Reasoning")
        print()
        print("Usage:")
        print("  python nhop_plaintext_baseline.py generate <model>")
        print("  python nhop_plaintext_baseline.py train    <model>")
        print("  python nhop_plaintext_baseline.py eval     <model> [model_path]")
        print("  python nhop_plaintext_baseline.py compare  <model>")
        print()
        print("Models: gpt2, qwen3, lfm2, hermes3")
        print()
        print("Typical workflow:")
        print("  1. python nhop_plaintext_baseline.py generate qwen3")
        print("  2. python nhop_plaintext_baseline.py train    qwen3")
        print("  3. python nhop_plaintext_baseline.py eval     qwen3")
        print("  4. python nhop_plaintext_baseline.py compare  qwen3")
        sys.exit(0)

    cmd        = sys.argv[1]
    model_name = sys.argv[2]

    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. "
              f"Choose from: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    if cmd == "generate":
        generate_plaintext_data(model_name)
    elif cmd == "train":
        train(model_name)
    elif cmd == "eval":
        mp = sys.argv[3] if len(sys.argv) > 3 else None
        evaluate(model_name, model_path=mp)
    elif cmd == "compare":
        compare_results(model_name)
    else:
        print(f"Unknown command: {cmd}. "
              f"Use generate / train / eval / compare.")
        sys.exit(1)
