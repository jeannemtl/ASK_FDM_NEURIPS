"""
Plaintext Key-Value Baseline Training (40-channel)
====================================================

Mirrors eidetic_real_fdm_e2e_40ch_turbo_v3.py exactly, but with plaintext
KEY=VALUE memory instead of FDM signal tokens.

For the FDM vs plaintext comparison paper (Prediction 2: block vs stride
corruption).

Training protocol matches the paper-era 40-channel GPT-2 FDM training:
- Base model: GPT-2 Medium
- 5-stage curriculum (15K + 20K + 25K + 25K + 30K samples)
- Per-stage epochs: 5, 5, 7, 7, 10
- Per-stage LR: 5e-5 -> 3e-5 -> 2e-5 -> 1e-5 -> 1e-5
- Batch size 4, max_length 512 (plaintext memory ~300 tokens + Q/A fits)
- AdamW optimizer (default betas, default weight_decay)
- No gradient checkpointing
- 85/10/5 train/val/test split
- Answer-only loss

Total: 115,000 samples across curriculum.
Estimated runtime: ~23 hours on RTX 5090.

Only differences from FDM training:
- No FDM encoder
- Memory rendered as [MEMORY]\\nKEY=VAL\\n...[/MEMORY] plaintext block
- Saves as plaintext_40ch_model_stage{N} / plaintext_40ch_model_final

Usage:
  python train_plaintext_40ch_gpt2.py generate   # Generate all stage data
  python train_plaintext_40ch_gpt2.py train      # Train through 5 stages
  python train_plaintext_40ch_gpt2.py train 2    # Resume from stage 2
"""

import json
import os
import random
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm


# ============================================================
# Memory schema (40 channels, identical to FDM 40ch)
# ============================================================

MEMORY_SCHEMAS = {
    0: ("SECRET",    ["RED", "BLUE", "GREEN", "GOLD"]),
    1: ("LOCATION",  ["PARIS", "TOKYO", "LONDON", "BERLIN"]),
    2: ("AGENT",     ["ALICE", "BOB", "CAROL", "DAVE"]),
    3: ("STATUS",    ["CLEAR", "COMPROMISED", "UNKNOWN"]),
    4: ("PRIORITY",  ["HIGH", "MEDIUM", "LOW"]),
    5: ("BACKUP",    ["AVAILABLE", "UNAVAILABLE"]),
    6: ("RULE",      ["SAFETY_FIRST", "MISSION_FIRST", "BALANCED", "CAUTIOUS"]),
    7: ("META",      ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY", "EMERGENCY", "LOCKDOWN"]),
    8:  ("TEAM",      ["RED_TEAM", "BLUE_TEAM", "GREEN_TEAM", "GOLD_TEAM"]),
    9:  ("REGION",    ["NORTH", "SOUTH", "EAST", "WEST"]),
    10: ("PHASE",     ["ALPHA", "BETA", "GAMMA", "DELTA"]),
    11: ("COMM",      ["OPEN", "CLOSED", "RESTRICTED"]),
    12: ("ASSET",     ["VEHICLE", "AIRCRAFT", "DRONE", "BOAT"]),
    13: ("WINDOW",    ["DAWN", "MIDDAY", "DUSK", "NIGHT"]),
    14: ("COVER",     ["DEEP", "SHALLOW", "NONE"]),
    15: ("SUPPORT",   ["ACTIVE", "STANDBY", "OFFLINE"]),
    16: ("THREAT",    ["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
    17: ("WEATHER",   ["CLEAR", "STORM", "FOG"]),
    18: ("TERRAIN",   ["URBAN", "RURAL", "COASTAL", "MOUNTAIN"]),
    19: ("EXTRACT",   ["READY", "DELAYED", "UNAVAILABLE"]),
    20: ("CIPHER",    ["AES", "RSA", "BLOWFISH", "TWOFISH"]),
    21: ("FREQ",      ["HF", "VHF", "UHF", "SHF"]),
    22: ("PAYLOAD",   ["LIGHT", "MEDIUM", "HEAVY", "CRITICAL"]),
    23: ("ROUTE",     ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]),
    24: ("DURATION",  ["SHORT", "MEDIUM", "LONG", "EXTENDED"]),
    25: ("CONTACT",   ["FRIENDLY", "NEUTRAL", "HOSTILE", "UNKNOWN"]),
    26: ("FUEL",      ["FULL", "HALF", "LOW", "CRITICAL"]),
    27: ("ALTITUDE",  ["LOW", "MEDIUM", "HIGH"]),
    28: ("VISIBILITY",["CLEAR", "REDUCED", "ZERO"]),
    29: ("NOISE",     ["SILENT", "QUIET", "MODERATE", "LOUD"]),
    30: ("FORMATION", ["SINGLE", "PAIR", "SQUAD", "PLATOON"]),
    31: ("ARMOR",     ["NONE", "LIGHT", "MEDIUM", "HEAVY"]),
    32: ("SIGNAL",    ["STRONG", "WEAK", "JAMMED", "LOST"]),
    33: ("MORALE",    ["HIGH", "MEDIUM", "LOW"]),
    34: ("SUPPLY",    ["ABUNDANT", "ADEQUATE", "SCARCE", "DEPLETED"]),
    35: ("INTEL",     ["CONFIRMED", "PROBABLE", "UNCERTAIN", "NONE"]),
    36: ("EVAC",      ["STANDING", "PREPPED", "LAUNCHED", "ABORTED"]),
    37: ("WEATHER2",  ["SUNNY", "OVERCAST", "RAIN", "SNOW"]),
    38: ("DOCTRINE",  ["OFFENSIVE", "DEFENSIVE", "RECON", "SUPPORT"]),
    39: ("COMMS",     ["SECURE", "OPEN", "COMPROMISED", "SILENT"]),
}
NUM_CHANNELS = 40


# ============================================================
# Answer functions (verbatim from eidetic_real_fdm_e2e_40ch_turbo_v3.py)
# ============================================================

def generate_proceed_answer(facts, rule, meta):
    agent = facts["AGENT"]
    status = facts["STATUS"]
    priority = facts["PRIORITY"]
    backup = facts["BACKUP"]
    if meta == "EMERGENCY":
        return f"EMERGENCY PROTOCOL: {agent} must proceed immediately. All other factors suspended."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN ACTIVE: {agent} must abort all operations. No exceptions."
    if meta == "OVERRIDE_STATUS":
        if priority == "HIGH":
            return f"Status override active. Priority is {priority}, so {agent} should proceed."
        return f"Status override active, but priority is only {priority}. {agent} may proceed with caution."
    if meta == "OVERRIDE_PRIORITY":
        if status == "CLEAR" and backup == "AVAILABLE":
            return f"Priority override active. Status {status} with backup {backup}. {agent} can proceed."
        return f"Priority override active. Status {status}, backup {backup}. {agent} should hold."
    if rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            return f"SAFETY_FIRST: Status is {status}. {agent} must abort regardless of {priority} priority."
        if status == "CLEAR":
            return f"SAFETY_FIRST: Status is {status}. {agent} can proceed with {priority} priority."
        return f"SAFETY_FIRST: Status is {status}. {agent} should wait for confirmation."
    if rule == "MISSION_FIRST":
        if status == "COMPROMISED" and backup == "UNAVAILABLE":
            return f"MISSION_FIRST: Status {status} with no backup. Even mission-priority says {agent} should abort."
        return f"MISSION_FIRST: Priority is {priority}. {agent} should proceed. Status {status} is secondary."
    if rule == "BALANCED":
        if priority == "HIGH" and backup == "AVAILABLE":
            return f"BALANCED: {priority} priority with backup {backup} outweighs {status} status. {agent} can proceed."
        if status == "COMPROMISED":
            return f"BALANCED: {status} status not offset by {priority} priority. {agent} should abort."
        return f"BALANCED: Status {status}, priority {priority}. {agent} can proceed carefully."
    if rule == "CAUTIOUS":
        if status == "CLEAR" and backup == "AVAILABLE":
            return f"CAUTIOUS: Status {status} AND backup {backup}. Both conditions met. {agent} can proceed."
        return f"CAUTIOUS: Need CLEAR status AND AVAILABLE backup. Have {status}/{backup}. {agent} must abort."
    return f"{agent} should assess situation."


def generate_risk_answer(facts, rule, meta):
    status = facts["STATUS"]
    priority = facts["PRIORITY"]
    backup = facts["BACKUP"]
    location = facts["LOCATION"]
    if meta == "EMERGENCY":
        return f"EMERGENCY: Risk assessment suspended. Immediate action required in {location}."
    if meta == "LOCKDOWN":
        return f"LOCKDOWN: Maximum risk assumed. All operations in {location} halted."
    risk_factors = []
    if status == "COMPROMISED":
        risk_factors.append("compromised status")
    if backup == "UNAVAILABLE":
        risk_factors.append("no backup")
    if status == "UNKNOWN":
        risk_factors.append("unknown status")
    if rule == "SAFETY_FIRST":
        if risk_factors:
            return f"SAFETY_FIRST assessment: HIGH RISK in {location}. Factors: {', '.join(risk_factors)}."
        return f"SAFETY_FIRST assessment: LOW RISK in {location}. Status {status}, backup {backup}."
    if rule == "MISSION_FIRST":
        if len(risk_factors) >= 2:
            return f"MISSION_FIRST assessment: MODERATE RISK in {location}. Acceptable for {priority} priority."
        return f"MISSION_FIRST assessment: LOW RISK in {location}. Proceed with mission."
    if rule == "BALANCED":
        level = "HIGH" if len(risk_factors) >= 2 else "MEDIUM" if len(risk_factors) == 1 else "LOW"
        return f"BALANCED assessment: {level} RISK in {location}. Factors: {len(risk_factors)} concerns."
    if rule == "CAUTIOUS":
        if risk_factors:
            return f"CAUTIOUS assessment: HIGH RISK in {location}. Any risk factor triggers alert: {', '.join(risk_factors)}."
        return f"CAUTIOUS assessment: LOW RISK in {location}. All safety conditions met."
    return f"Risk assessment for {location}."


def generate_share_answer(facts, rule, meta):
    secret = facts["SECRET"]
    status = facts["STATUS"]
    location = facts["LOCATION"]
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
    if rule == "MISSION_FIRST":
        return f"MISSION_FIRST: Share {secret} with {location} contact to advance mission. Accept calculated risk."
    if rule == "BALANCED":
        if status == "CLEAR":
            return f"BALANCED: Status {status}. Safe to share {secret} with {location} contact."
        return f"BALANCED: Status {status}. Share {secret} only if mission-critical."
    if rule == "CAUTIOUS":
        if status == "CLEAR":
            return f"CAUTIOUS: Status {status}. May share {secret} after secondary verification."
        return f"CAUTIOUS: Status {status}. Do not share {secret}. Request secure channel."
    return f"Evaluate sharing {secret} in {location}."


QUESTION_TEMPLATES = [
    {"question": "Should {AGENT} proceed with the mission?", "fn": generate_proceed_answer},
    {"question": "What is the risk assessment?",             "fn": generate_risk_answer},
    {"question": "Is it safe to share the secret with the contact?", "fn": generate_share_answer},
]


# ============================================================
# Plaintext memory encoder (replaces FDM encoder)
# ============================================================

def encode_plaintext_memory(memory):
    """
    Convert 40-channel memory dict to plaintext KEY=VALUE block.
    Fixed channel order (0-39) for consistency.
    Output format:
      [MEMORY]
      SECRET=RED
      LOCATION=PARIS
      ...
      COMMS=SECURE
      [/MEMORY]
    Roughly 288 tokens after GPT-2 tokenization.
    """
    lines = ["[MEMORY]"]
    for ch in range(NUM_CHANNELS):
        key, _ = MEMORY_SCHEMAS[ch]
        val = memory[ch] if ch in memory else memory[str(ch)]
        lines.append(f"{key}={val}")
    lines.append("[/MEMORY]")
    return "\n".join(lines)


# ============================================================
# Stage data generation
# ============================================================

def generate_stage_data(stage_config, num_samples, output_prefix):
    """Generate plaintext training data for one curriculum stage."""
    samples = []
    print(f"Generating {num_samples} plaintext samples: {stage_config['name']}...")
    
    for _ in tqdm(range(num_samples)):
        memory = {}
        for ch in range(NUM_CHANNELS):
            _, values = MEMORY_SCHEMAS[ch]
            memory[ch] = random.choice(values)
        
        # Plaintext memory block (replaces FDM text generation)
        plaintext_memory = encode_plaintext_memory(memory)
        
        facts = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(6)}
        rule = memory[6]
        meta = memory[7]
        
        q_template = random.choice(QUESTION_TEMPLATES)
        question = q_template["question"].format(**facts)
        answer = q_template["fn"](facts, rule, meta)
        
        explicit_parts = [f"{MEMORY_SCHEMAS[ch][0]}:{memory[ch]}" for ch in range(NUM_CHANNELS)]
        explicit = "|".join(explicit_parts)
        
        samples.append({
            "plaintext_memory": plaintext_memory,
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
        'val':   samples[int(0.85 * n):int(0.95 * n)],
        'test':  samples[int(0.95 * n):],
    }
    
    for split_name, split_data in splits.items():
        path = f"{output_prefix}_{split_name}.jsonl"
        with open(path, 'w') as f:
            for s in split_data:
                f.write(json.dumps(s) + "\n")
        print(f"  {split_name}: {len(split_data)} samples -> {path}")
    
    return samples


# ============================================================
# Curriculum (matches FDM 40ch turbo v3 exactly, minus encoder params)
# ============================================================

STAGES = [
    {
        "name": "Stage 0: Easy (plaintext, lr=5e-5)",
        "samples": 15000,
        "epochs": 5,
        "lr": 5e-5,
    },
    {
        "name": "Stage 1: Standard (plaintext, lr=3e-5)",
        "samples": 20000,
        "epochs": 5,
        "lr": 3e-5,
    },
    {
        "name": "Stage 2: Moderate (plaintext, lr=2e-5)",
        "samples": 25000,
        "epochs": 7,
        "lr": 2e-5,
    },
    {
        "name": "Stage 3: Harder (plaintext, lr=1e-5)",
        "samples": 25000,
        "epochs": 7,
        "lr": 1e-5,
    },
    {
        "name": "Stage 4: Hardest (plaintext, lr=1e-5)",
        "samples": 30000,
        "epochs": 10,
        "lr": 1e-5,
    },
]


def generate_all_stages():
    """Generate data for all 5 stages."""
    for i, stage in enumerate(STAGES):
        print(f"\n{'='*60}")
        print(f"GENERATING STAGE {i}: {stage['name']}")
        print(f"{'='*60}")
        output_prefix = f"plaintext_40ch_stage{i}"
        generate_stage_data(stage, stage['samples'], output_prefix)


# ============================================================
# Dataset
# ============================================================

class PlaintextReasoningDataset(Dataset):
    """Dataset: plaintext memory + question -> reasoning response."""
    
    def __init__(self, path, tokenizer, max_length=512):
        self.samples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        prompt = (sample['plaintext_memory'] + 
                  "\nQuestion: " + sample['question'] + 
                  "\nAnswer: ")
        answer = sample['answer']
        full_text = prompt + answer
        
        encoding = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            padding='max_length', return_tensors='pt'
        )
        
        # Answer-only loss: mask prompt tokens
        prompt_len = len(self.tokenizer.encode(prompt, truncation=True,
                                                max_length=self.max_length))
        labels = encoding['input_ids'].clone()
        labels[0, :prompt_len] = -100
        labels[encoding['attention_mask'] == 0] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
        }


# ============================================================
# Training loop (matches FDM 40ch turbo v3 structure)
# ============================================================

def train_all_stages(resume_stage=0):
    """Train plaintext GPT-2 through all 5 curriculum stages."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model_name = 'openai-community/gpt2-medium'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if resume_stage == 0:
        print(f"Loading base model: {model_name}")
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        ckpt = f"plaintext_40ch_model_stage{resume_stage - 1}"
        print(f"Resuming from {ckpt}")
        model = GPT2LMHeadModel.from_pretrained(ckpt)
    model.to(device)
    
    for stage_idx, stage in enumerate(STAGES):
        if stage_idx < resume_stage:
            print(f"Skipping stage {stage_idx}")
            continue
        
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx}: {stage['name']}")
        print(f"  samples={stage['samples']}  epochs={stage['epochs']}  lr={stage['lr']}")
        print(f"{'='*60}")
        
        train_path = f"plaintext_40ch_stage{stage_idx}_train.jsonl"
        val_path   = f"plaintext_40ch_stage{stage_idx}_val.jsonl"
        
        if not os.path.exists(train_path):
            print(f"Generating stage {stage_idx} data...")
            generate_stage_data(stage, stage['samples'], f"plaintext_40ch_stage{stage_idx}")
        
        train_dataset = PlaintextReasoningDataset(train_path, tokenizer)
        val_dataset   = PlaintextReasoningDataset(val_path, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=4)
        
        optimizer = AdamW(model.parameters(), lr=stage['lr'])
        
        for epoch in range(stage['epochs']):
            model.train()
            total_loss = 0
            num_batches = 0
            
            progress = tqdm(train_loader,
                            desc=f"  Stage {stage_idx}, Epoch {epoch+1}/{stage['epochs']}")
            for batch in progress:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                num_batches += 1
                progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
            model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()
                    val_batches += 1
            
            avg_train = total_loss / num_batches
            avg_val = val_loss / val_batches
            print(f"  Epoch {epoch+1}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")
        
        # Save stage checkpoint (matches FDM save pattern)
        stage_dir = f"plaintext_40ch_model_stage{stage_idx}"
        model.save_pretrained(stage_dir)
        tokenizer.save_pretrained(stage_dir)
        print(f"  Saved stage {stage_idx} -> {stage_dir}")
    
    # Final model
    model.save_pretrained('plaintext_40ch_model_final')
    tokenizer.save_pretrained('plaintext_40ch_model_final')
    print("\nAll 5 stages complete. Final -> plaintext_40ch_model_final/")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_plaintext_40ch_gpt2.py <generate|train> [resume_stage]")
        sys.exit(1)
    
    command = sys.argv[1]
    if command == "generate":
        generate_all_stages()
    elif command == "train":
        resume = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        train_all_stages(resume_stage=resume)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
