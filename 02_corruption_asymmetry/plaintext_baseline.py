"""
Plaintext Baseline for FDM Comparison

Identical to eidetic_real_fdm_e2e_40ch_turbo_v3.py EXCEPT:
  - Channel values are encoded as plaintext key=value pairs
  - No FDM signal processing, no turbo structure
  - Same questions, same answer logic, same evaluation

This isolates the contribution of FDM structured encoding vs
unstructured plaintext context for multi-channel retrieval.

Usage:
  python plaintext_baseline.py generate       # Generate all stages
  python plaintext_baseline.py train          # Curriculum training
  python plaintext_baseline.py eval [stage]   # Evaluate
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW
import json
import random
import re
from tqdm import tqdm
import os
import sys

# Import everything shared with the FDM version
from eidetic_real_fdm_e2e_40ch_turbo_v3 import (
    MEMORY_SCHEMAS, NUM_CHANNELS, REASONING_RULES, META_INSTRUCTIONS,
    QUESTION_TEMPLATES,
    generate_proceed_answer, generate_risk_answer, generate_share_answer,
    FDMReasoningDataset,  # reuse -- it just reads fdm_text from jsonl
)


# ============================================================
# Plaintext Encoder (replaces TurboFDMSignalEncoder)
# ============================================================

class PlaintextEncoder:
    """
    Encodes 40 memory channels as plaintext key=value pairs.
    
    Output format (default):
      SECRET=RED, LOCATION=PARIS, AGENT=ALICE, STATUS=COMPROMISED, ...
    
    This is the strongest possible plaintext baseline: the model
    receives all facts explicitly in natural language, with no
    frequency decomposition required.
    
    Supports multiple ordering strategies to test whether retrieval
    degradation depends on fact position (cf. Liu et al. "Lost in
    the Middle"):
      - natural:  channels 0-39 in order
      - shuffled: random order per sample (harder retrieval)
      - reversed: channels 39-0
    """
    
    def __init__(self, order="natural", separator=", ", seed=42):
        self.order = order
        self.separator = separator
        self.rng = random.Random(seed)
    
    def encode_memory(self, memory):
        """
        Encode memory dict as plaintext string.
        
        Returns:
            (plaintext_string, token_ids)
        """
        channels = list(range(NUM_CHANNELS))
        
        if self.order == "shuffled":
            self.rng.shuffle(channels)
        elif self.order == "reversed":
            channels = channels[::-1]
        
        pairs = []
        for ch in channels:
            name = MEMORY_SCHEMAS[ch][0]
            value = memory[ch]
            pairs.append(f"{name}={value}")
        
        plaintext = self.separator.join(pairs)
        
        tokenizer = _get_tokenizer()
        token_ids = tokenizer.encode(plaintext)
        
        return plaintext, token_ids


# ============================================================
# Global tokenizer (loaded once)
# ============================================================

_tokenizer_cache = None

def _get_tokenizer():
    global _tokenizer_cache
    if _tokenizer_cache is None:
        _tokenizer_cache = GPT2Tokenizer.from_pretrained('gpt2')
    return _tokenizer_cache


# ============================================================
# Data generation
# ============================================================

def generate_stage_data(stage_config, num_samples, output_prefix):
    """Generate training data for one curriculum stage."""
    order = stage_config.get("order", "natural")
    encoder = PlaintextEncoder(order=order)
    
    samples = []
    print(f"Generating {num_samples} plaintext samples (order={order})...")
    
    token_lengths = []
    
    for _ in tqdm(range(num_samples)):
        memory = {}
        for ch in range(NUM_CHANNELS):
            _, values = MEMORY_SCHEMAS[ch]
            memory[ch] = random.choice(values)
        
        plaintext, token_ids = encoder.encode_memory(memory)
        token_lengths.append(len(token_ids))
        
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
            "fdm_text": plaintext,  # key name kept for FDMReasoningDataset compat
            "memory": {str(k): v for k, v in memory.items()},
            "explicit": explicit,
            "question": question,
            "answer": answer,
            "rule": rule,
            "meta": meta,
            "facts": facts,
            "encoding": "plaintext",
            "order": order,
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
    
    avg_tokens = np.mean(token_lengths)
    print(f"  Avg plaintext token length: {avg_tokens:.1f} tokens (FDM uses 512)")
    
    return samples


# Stages -- matched to FDM sample counts for fair comparison.
STAGES = [
    {
        "name": "Stage 0: Plaintext (15K)",
        "order": "natural",
        "samples": 15000,
        "epochs": 5,
        "lr": 5e-5,
    },
    {
        "name": "Stage 1: Plaintext (20K)",
        "order": "natural",
        "samples": 20000,
        "epochs": 5,
        "lr": 3e-5,
    },
    {
        "name": "Stage 2: Plaintext (25K)",
        "order": "natural",
        "samples": 25000,
        "epochs": 7,
        "lr": 2e-5,
    },
    {
        "name": "Stage 3: Plaintext (25K)",
        "order": "natural",
        "samples": 25000,
        "epochs": 7,
        "lr": 1e-5,
    },
    {
        "name": "Stage 4: Plaintext (30K)",
        "order": "natural",
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
        generate_stage_data(stage, stage['samples'], f"plaintext_40ch_stage{i}")


# ============================================================
# Training
# ============================================================

def train(checkpoint_dir="checkpoints_plaintext_40ch"):
    """Curriculum training on plaintext-encoded data."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Channels: {NUM_CHANNELS}")
    print(f"Encoding: PLAINTEXT (key=value pairs)")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
        
        prefix = f"plaintext_40ch_stage{stage_idx}"
        train_path = f"{prefix}_train.jsonl"
        val_path = f"{prefix}_val.jsonl"
        
        if not os.path.exists(train_path):
            print(f"ERROR: {train_path} not found. Run 'generate' first.")
            return
        
        print(f"\n{'='*60}")
        print(f"STAGE {stage_idx}: {stage['name']}")
        print(f"  lr={stage['lr']}, epochs={stage['epochs']}")
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
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  New best val loss: {val_loss:.4f}")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'stage': stage_idx,
                'epoch': epoch,
                'val_loss': val_loss,
            }, os.path.join(checkpoint_dir, f'stage{stage_idx}_epoch{epoch+1}.pt'))
        
        model.save_pretrained(f'plaintext_40ch_model_stage{stage_idx}')
        tokenizer.save_pretrained(f'plaintext_40ch_model_stage{stage_idx}')
        print(f"Saved stage {stage_idx} model")
    
    model.save_pretrained('plaintext_40ch_model_final')
    tokenizer.save_pretrained('plaintext_40ch_model_final')
    print("\nSaved final model: plaintext_40ch_model_final/")


# ============================================================
# Evaluation
# ============================================================

def evaluate(model_path='plaintext_40ch_model_final', stage_idx=None):
    """Full evaluation -- identical metrics to FDM eval."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {model_path}/...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    si = stage_idx if stage_idx is not None else len(STAGES) - 1
    test_path = f"plaintext_40ch_stage{si}_test.jsonl"
    
    if not os.path.exists(test_path):
        print(f"ERROR: {test_path} not found.")
        return
    
    test_samples = [json.loads(l) for l in open(test_path)]
    print(f"Evaluating on {len(test_samples)} samples (stage {si})...\n")
    
    action_correct = 0
    rule_correct = 0
    meta_correct = 0
    fact_correct = 0
    extra_correct = 0
    extra_total = 0
    total = 0
    
    ch_correct = {ch: 0 for ch in range(NUM_CHANNELS)}
    ch_total = {ch: 0 for ch in range(NUM_CHANNELS)}
    
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
        memory = s['memory']
        
        # Action accuracy
        action_ok = False
        for kw in ['abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN',
                    'share', 'Do NOT share', 'Do not share', 'HIGH RISK', 'LOW RISK', 'MODERATE']:
            if kw.lower() in expected.lower() and kw.lower() in response.lower():
                action_ok = True
                break
        if action_ok:
            action_correct += 1
        
        if rule in response:
            rule_correct += 1
        
        meta_ok = False
        if meta == "EMERGENCY" and "EMERGENCY" in response:
            meta_ok = True
        elif meta == "LOCKDOWN" and "LOCKDOWN" in response:
            meta_ok = True
        elif meta in ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY"]:
            meta_ok = True
        if meta_ok:
            meta_correct += 1
        
        facts = s['facts']
        fact_ok = False
        if s['question'].startswith("Should"):
            fact_ok = facts['AGENT'] in response
        elif s['question'].startswith("What is the risk"):
            fact_ok = facts['LOCATION'] in response
        elif s['question'].startswith("Is it safe"):
            fact_ok = facts['SECRET'] in response or facts['LOCATION'] in response
        if fact_ok:
            fact_correct += 1
        
        # Extra channel accuracy (channels 8-39)
        gen_context = re.search(r'Context:\s*(.+?)\.?\s*$', response, re.DOTALL)
        gen_pairs = {}
        if gen_context:
            for pair in gen_context.group(1).split(','):
                pair = pair.strip()
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    gen_pairs[k.strip()] = v.strip()
        
        for ch in range(8, NUM_CHANNELS):
            name = MEMORY_SCHEMAS[ch][0]
            expected_val = memory[str(ch)]
            ch_total[ch] += 1
            extra_total += 1
            if gen_pairs.get(name) == expected_val:
                ch_correct[ch] += 1
                extra_correct += 1
        
        total += 1
        
        if i < 5:
            print(f"\n{'='*60}")
            print(f"Sample {i+1} | Rule: {rule} | Meta: {meta}")
            print(f"Q: {s['question']}")
            print(f"Expected: {expected[:150]}...")
            print(f"Got:      {response[:150]}...")
            am = "Y" if action_ok else "N"
            print(f"Action: {am}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"PLAINTEXT BASELINE -- 40-CHANNEL EVALUATION ({total} samples)")
    print(f"{'='*70}")
    
    print(f"\n--- Overall ---")
    print(f"  Action accuracy:  {100*action_correct/total:.1f}%")
    print(f"  Rule accuracy:    {100*rule_correct/total:.1f}%")
    print(f"  Meta accuracy:    {100*meta_correct/total:.1f}%")
    print(f"  Fact accuracy:    {100*fact_correct/total:.1f}%")
    print(f"  Extra accuracy:   {100*extra_correct/max(extra_total,1):.1f}%")
    
    print(f"\n--- Per-Channel Extra Accuracy (ch 8-39) ---")
    for ch in range(8, NUM_CHANNELS):
        name = MEMORY_SCHEMAS[ch][0]
        pct = 100 * ch_correct[ch] / max(ch_total[ch], 1)
        marker = " ***" if pct < 90 else ""
        print(f"  ch{ch:2d} ({name:12s}): {pct:.1f}%{marker}")
    
    print(f"\n--- Comparison ---")
    print(f"  FDM 40ch turbo v3 (115K): Action=98.3%, Fact=99.9%, Extra=89.2%")
    print(f"  Plaintext 40ch   (115K):  Action={100*action_correct/total:.1f}%, Fact={100*fact_correct/total:.1f}%, Extra={100*extra_correct/max(extra_total,1):.1f}%")
    print(f"\n--- Token Budget ---")
    sample_text = test_samples[0]['fdm_text']
    sample_tokens = tokenizer.encode(sample_text)
    print(f"  Plaintext context: ~{len(sample_tokens)} tokens")
    print(f"  FDM context:       512 tokens (256+256 turbo)")
    
    results = {
        "encoding": "plaintext",
        "action_accuracy": 100 * action_correct / total,
        "rule_accuracy": 100 * rule_correct / total,
        "meta_accuracy": 100 * meta_correct / total,
        "fact_accuracy": 100 * fact_correct / total,
        "extra_accuracy": 100 * extra_correct / max(extra_total, 1),
        "per_channel": {str(ch): 100 * ch_correct[ch] / max(ch_total[ch], 1)
                        for ch in range(NUM_CHANNELS)},
        "total_samples": total,
    }
    results_path = "plaintext_40ch_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python plaintext_baseline.py generate       # Generate data")
        print("  python plaintext_baseline.py train          # Train")
        print("  python plaintext_baseline.py eval [stage]   # Evaluate")
        print()
        print("Plaintext baseline for FDM comparison.")
        print("Same 40 channels, same questions, same answer logic.")
        print("Only difference: facts presented as key=value text")
        print("instead of FDM-encoded signal tokens.")
        sys.exit(0)
    
    cmd = sys.argv[1]
    if cmd == "generate":
        generate_all_stages()
    elif cmd == "train":
        train()
    elif cmd == "eval":
        stage = int(sys.argv[2]) if len(sys.argv) > 2 else None
        evaluate(stage_idx=stage)
