"""
N-Hop Compositional Reasoning — Training + Evaluation

Trains ONE additional stage on top of the v3 model with question templates
that have explicit, countable hop structure (Greenblatt-style).

Each hop is a genuine compositional step where the output of one read
feeds into the next decision branch — NOT just parallel reads.

Hop definitions:
  1-hop: Read 1 channel, report value. No composition.
  2-hop: Read 2 channels, compose. Output depends on BOTH values.
         e.g., SAFETY_FIRST + COMPROMISED → abort (but MISSION_FIRST + COMPROMISED → proceed)
  3-hop: Read 3 channels, chained conditional.
         e.g., Read RULE → Read STATUS → Read BACKUP → if CAUTIOUS, need CLEAR + AVAILABLE
  4-hop: Read 4 channels, nested conditionals with override.
         e.g., Read META → if META=OVERRIDE_STATUS, skip status check → Read RULE → Read PRIORITY → decide
  5-hop: Read 5 channels, full conditional chain.
         e.g., META override check → STATUS+RULE compose → BACKUP contingency → THREAT modifier → decide

Usage:
  python nhop_training.py generate          # Generate hop-controlled training data
  python nhop_training.py train             # Fine-tune one stage on v3 model
  python nhop_training.py eval              # Evaluate per hop count
  python nhop_training.py plot              # Generate figure
"""

import json, random, os, sys, re
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# Import from your actual v3 script
from eidetic_real_fdm_e2e_40ch_turbo_v3 import (
    TurboFDMSignalEncoder, MEMORY_SCHEMAS, NUM_CHANNELS,
    FDMReasoningDataset, get_tokenizer,
)


# ============================================================
# HOP-CONTROLLED QUESTION TEMPLATES
# ============================================================
# Each template defines:
#   - hops: number of compositional reasoning steps
#   - channels: which channels are READ (input to reasoning)
#   - question: the question string
#   - answer_fn: deterministic function(memory) -> answer
#
# CRITICAL: Each hop must be a genuine compositional step.
# Reading channel A, then using A's value to determine how to
# interpret channel B, is 2 hops. Reading A and B independently
# and concatenating them is NOT 2 hops.

# ── 1-HOP: Direct read, no composition ──────────────────────

def hop1_status(m):
    """Read STATUS, report it."""
    return f"STATUS_REPORT: Status is {m[3]}."

def hop1_threat(m):
    """Read THREAT, report it."""
    return f"THREAT_REPORT: Threat level is {m[16]}."

def hop1_rule(m):
    """Read RULE, report it."""
    return f"RULE_REPORT: Current rule is {m[6]}."

def hop1_location(m):
    """Read LOCATION, report it."""
    return f"LOCATION_REPORT: Location is {m[1]}."


# ── 2-HOP: Read A, then use A to interpret B ────────────────

def hop2_rule_status(m):
    """Hop1: read RULE. Hop2: read STATUS, apply RULE to STATUS."""
    rule, status = m[6], m[3]
    if rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            decision = "ABORT"
        elif status == "UNKNOWN":
            decision = "HOLD"
        else:
            decision = "PROCEED"
    elif rule == "MISSION_FIRST":
        if status == "COMPROMISED":
            decision = "PROCEED_WITH_RISK"
        else:
            decision = "PROCEED"
    elif rule == "CAUTIOUS":
        if status == "CLEAR":
            decision = "PROCEED"
        else:
            decision = "ABORT"
    else:  # BALANCED
        if status == "COMPROMISED":
            decision = "HOLD"
        else:
            decision = "PROCEED"
    return f"2HOP_DECISION: {decision}. Rule {rule} applied to status {status}."

def hop2_threat_doctrine(m):
    """Hop1: read THREAT. Hop2: read DOCTRINE, modify based on THREAT."""
    threat, doctrine = m[16], m[38]
    if threat in ["HIGH", "CRITICAL"]:
        effective = "DEFENSIVE" if doctrine == "OFFENSIVE" else doctrine
    elif threat == "LOW":
        effective = "OFFENSIVE" if doctrine == "DEFENSIVE" else doctrine
    else:
        effective = doctrine
    return f"2HOP_DOCTRINE: {effective}. Threat {threat} modifies base doctrine {doctrine}."

def hop2_weather_extract(m):
    """Hop1: read WEATHER. Hop2: read EXTRACT, check compatibility."""
    weather, extract = m[17], m[19]
    if weather == "STORM" and extract == "READY":
        result = "DELAYED"
    elif weather == "FOG" and extract == "READY":
        result = "RISKY"
    else:
        result = extract
    return f"2HOP_EXTRACT: {result}. Weather {weather} affects extraction {extract}."

def hop2_cover_terrain(m):
    """Hop1: read COVER. Hop2: read TERRAIN, assess adequacy."""
    cover, terrain = m[14], m[18]
    if cover == "NONE":
        adequate = "EXPOSED"
    elif cover == "DEEP" and terrain == "URBAN":
        adequate = "EXCELLENT"
    elif cover == "SHALLOW" and terrain in ["COASTAL", "MOUNTAIN"]:
        adequate = "POOR"
    else:
        adequate = "ADEQUATE"
    return f"2HOP_COVER: {adequate}. Cover {cover} in {terrain} terrain."


# ── 3-HOP: A → B → C chained ────────────────────────────────

def hop3_rule_status_backup(m):
    """Hop1: RULE. Hop2: STATUS (interpret via RULE). Hop3: BACKUP (contingency)."""
    rule, status, backup = m[6], m[3], m[5]
    # Hop 1-2: rule + status
    if rule == "SAFETY_FIRST" and status == "COMPROMISED":
        base = "ABORT"
    elif rule == "MISSION_FIRST":
        base = "PROCEED"
    elif rule == "CAUTIOUS" and status != "CLEAR":
        base = "ABORT"
    elif rule == "BALANCED" and status == "COMPROMISED":
        base = "HOLD"
    else:
        base = "PROCEED"
    # Hop 3: backup modifies
    if base == "ABORT" and backup == "AVAILABLE":
        final = "EXTRACT"
    elif base == "HOLD" and backup == "AVAILABLE":
        final = "PROCEED_WITH_BACKUP"
    elif base == "PROCEED" and backup == "UNAVAILABLE" and status == "UNKNOWN":
        final = "PROCEED_CAUTIOUS"
    else:
        final = base
    return f"3HOP_DECISION: {final}. Rule {rule}, status {status}, backup {backup}."

def hop3_threat_support_terrain(m):
    """Hop1: THREAT. Hop2: SUPPORT (adequate for threat?). Hop3: TERRAIN (modifies)."""
    threat, support, terrain = m[16], m[15], m[18]
    # Hop 1-2: threat + support
    if threat in ["HIGH", "CRITICAL"] and support == "OFFLINE":
        base = "CRITICAL"
    elif threat in ["HIGH", "CRITICAL"] and support == "ACTIVE":
        base = "MANAGEABLE"
    elif threat in ["LOW", "MEDIUM"]:
        base = "LOW_RISK"
    else:
        base = "MODERATE"
    # Hop 3: terrain modifies
    if base == "CRITICAL" and terrain == "MOUNTAIN":
        final = "DIRE"
    elif base == "MANAGEABLE" and terrain == "URBAN":
        final = "CONTAINED"
    elif base == "LOW_RISK":
        final = "LOW_RISK"
    else:
        final = base
    return f"3HOP_ASSESSMENT: {final}. Threat {threat}, support {support}, terrain {terrain}."

def hop3_weather_visibility_window(m):
    """Hop1: WEATHER. Hop2: VISIBILITY (compound with weather). Hop3: WINDOW (timing)."""
    weather, vis, window = m[17], m[28], m[13]
    # Hop 1-2: weather + visibility
    if weather == "STORM" and vis == "ZERO":
        conditions = "IMPOSSIBLE"
    elif weather == "FOG" and vis in ["ZERO", "REDUCED"]:
        conditions = "DANGEROUS"
    elif weather == "CLEAR" and vis == "CLEAR":
        conditions = "OPTIMAL"
    else:
        conditions = "MARGINAL"
    # Hop 3: window timing
    if conditions == "IMPOSSIBLE":
        final = "SCRUB"
    elif conditions == "DANGEROUS" and window == "NIGHT":
        final = "SCRUB"
    elif conditions == "OPTIMAL" and window in ["DAWN", "DUSK"]:
        final = "IDEAL"
    elif conditions == "MARGINAL" and window == "MIDDAY":
        final = "ACCEPTABLE"
    else:
        final = "CONDITIONAL"
    return f"3HOP_WINDOW: {final}. Weather {weather}, visibility {vis}, window {window}."

def hop3_agent_status_priority(m):
    """Hop1: AGENT. Hop2: STATUS. Hop3: PRIORITY (compose all three)."""
    agent, status, priority = m[2], m[3], m[4]
    if status == "COMPROMISED" and priority == "HIGH":
        action = "EMERGENCY_EXTRACT"
    elif status == "COMPROMISED" and priority in ["MEDIUM", "LOW"]:
        action = "ABORT"
    elif status == "CLEAR" and priority == "HIGH":
        action = "FULL_SPEED"
    elif status == "UNKNOWN":
        action = "RECON"
    else:
        action = "STANDARD"
    return f"3HOP_ACTION: {agent} should {action}. Status {status}, priority {priority}."


# ── 4-HOP: A → B → C → D nested conditionals ───────────────

def hop4_meta_rule_status_priority(m):
    """Hop1: META (override?). Hop2: RULE. Hop3: STATUS. Hop4: PRIORITY."""
    meta, rule, status, priority = m[7], m[6], m[3], m[4]
    # Hop 1: meta override check
    if meta == "EMERGENCY":
        return f"4HOP_DECISION: EMERGENCY_GO. Meta {meta} overrides all. Rule {rule}, status {status}, priority {priority}."
    if meta == "LOCKDOWN":
        return f"4HOP_DECISION: LOCKDOWN_ABORT. Meta {meta} overrides all. Rule {rule}, status {status}, priority {priority}."
    # Hop 2: rule
    if meta == "OVERRIDE_STATUS":
        # Skip hop 3 (status), go to hop 4 (priority)
        if priority == "HIGH":
            decision = "PROCEED_OVERRIDE"
        else:
            decision = "HOLD_OVERRIDE"
        return f"4HOP_DECISION: {decision}. Meta {meta}, rule {rule}, status {status}, priority {priority}."
    if meta == "OVERRIDE_PRIORITY":
        # Use status, skip priority
        if status == "CLEAR":
            decision = "PROCEED_STATUS_ONLY"
        else:
            decision = "ABORT_STATUS_ONLY"
        return f"4HOP_DECISION: {decision}. Meta {meta}, rule {rule}, status {status}, priority {priority}."
    # Hop 2-3-4: full chain
    if rule == "SAFETY_FIRST":
        if status == "COMPROMISED":
            decision = "ABORT"
        elif status == "UNKNOWN" and priority == "HIGH":
            decision = "PROCEED_CAUTIOUS"
        elif status == "CLEAR":
            decision = "PROCEED"
        else:
            decision = "HOLD"
    elif rule == "MISSION_FIRST":
        if priority == "HIGH":
            decision = "PROCEED"
        elif status == "COMPROMISED" and priority == "LOW":
            decision = "ABORT"
        else:
            decision = "PROCEED"
    elif rule == "CAUTIOUS":
        if status == "CLEAR" and priority in ["HIGH", "MEDIUM"]:
            decision = "PROCEED"
        else:
            decision = "ABORT"
    else:  # BALANCED
        if status == "COMPROMISED" and priority != "HIGH":
            decision = "ABORT"
        elif status == "CLEAR" or priority == "HIGH":
            decision = "PROCEED"
        else:
            decision = "HOLD"
    return f"4HOP_DECISION: {decision}. Meta {meta}, rule {rule}, status {status}, priority {priority}."

def hop4_threat_support_terrain_doctrine(m):
    """Hop1: THREAT. Hop2: SUPPORT. Hop3: TERRAIN. Hop4: DOCTRINE (final posture)."""
    threat, support, terrain, doctrine = m[16], m[15], m[18], m[38]
    # Hop 1-2: threat + support
    if threat in ["HIGH", "CRITICAL"] and support == "OFFLINE":
        situation = "CRITICAL"
    elif threat in ["HIGH", "CRITICAL"]:
        situation = "CONTESTED"
    else:
        situation = "PERMISSIVE"
    # Hop 3: terrain
    if situation == "CRITICAL" and terrain == "MOUNTAIN":
        situation = "DESPERATE"
    elif situation == "CONTESTED" and terrain == "URBAN":
        situation = "CLOSE_COMBAT"
    # Hop 4: doctrine
    if situation == "DESPERATE":
        posture = "WITHDRAW"
    elif situation == "CLOSE_COMBAT" and doctrine == "OFFENSIVE":
        posture = "ASSAULT"
    elif situation == "CLOSE_COMBAT":
        posture = "DEFEND"
    elif situation == "PERMISSIVE" and doctrine == "RECON":
        posture = "ADVANCE"
    elif situation == "PERMISSIVE":
        posture = doctrine.upper()
    else:
        posture = "HOLD"
    return f"4HOP_POSTURE: {posture}. Threat {threat}, support {support}, terrain {terrain}, doctrine {doctrine}."


# ── 5-HOP: Full conditional chain ───────────────────────────

def hop5_meta_rule_status_backup_threat(m):
    """Hop1: META. Hop2: RULE. Hop3: STATUS. Hop4: BACKUP. Hop5: THREAT."""
    meta, rule, status, backup, threat = m[7], m[6], m[3], m[5], m[16]
    # Hop 1: meta
    if meta == "EMERGENCY":
        return f"5HOP_DECISION: EMERGENCY_IMMEDIATE. Meta {meta} overrides. Rule {rule}, status {status}, backup {backup}, threat {threat}."
    if meta == "LOCKDOWN":
        return f"5HOP_DECISION: FULL_LOCKDOWN. Meta {meta} overrides. Rule {rule}, status {status}, backup {backup}, threat {threat}."
    # Hop 2-3: rule + status
    if rule == "SAFETY_FIRST" and status == "COMPROMISED":
        base = "ABORT"
    elif rule == "MISSION_FIRST":
        base = "PROCEED"
    elif rule == "CAUTIOUS" and status != "CLEAR":
        base = "ABORT"
    else:
        base = "ASSESS"
    # Hop 4: backup
    if base == "ABORT" and backup == "AVAILABLE":
        base = "EXTRACT_WITH_BACKUP"
    elif base == "PROCEED" and backup == "UNAVAILABLE":
        base = "PROCEED_NO_SAFETY_NET"
    elif base == "ASSESS" and backup == "AVAILABLE":
        base = "PROCEED_COVERED"
    elif base == "ASSESS":
        base = "HOLD"
    # Hop 5: threat modifier
    if threat in ["HIGH", "CRITICAL"] and base in ["PROCEED", "PROCEED_NO_SAFETY_NET"]:
        final = "PROCEED_HIGH_RISK"
    elif threat == "CRITICAL" and base == "HOLD":
        final = "EMERGENCY_EXTRACT"
    elif threat == "LOW" and "ABORT" in base:
        final = "CONTROLLED_WITHDRAWAL"
    else:
        final = base
    return f"5HOP_DECISION: {final}. Meta {meta}, rule {rule}, status {status}, backup {backup}, threat {threat}."

def hop5_weather_vis_window_terrain_extract(m):
    """Hop1: WEATHER. Hop2: VISIBILITY. Hop3: WINDOW. Hop4: TERRAIN. Hop5: EXTRACT."""
    weather, vis, window, terrain, extract = m[17], m[28], m[13], m[18], m[19]
    # Hop 1-2: weather + visibility
    if weather == "STORM" and vis == "ZERO":
        conditions = "NO_GO"
    elif weather == "FOG" and vis in ["ZERO", "REDUCED"]:
        conditions = "MARGINAL"
    elif weather == "CLEAR":
        conditions = "GOOD"
    else:
        conditions = "FAIR"
    # Hop 3: window
    if conditions == "NO_GO":
        timing = "IMPOSSIBLE"
    elif conditions == "MARGINAL" and window == "NIGHT":
        timing = "IMPOSSIBLE"
    elif conditions == "GOOD" and window in ["DAWN", "DUSK"]:
        timing = "IDEAL"
    else:
        timing = "ACCEPTABLE"
    # Hop 4: terrain
    if timing == "IMPOSSIBLE":
        feasibility = "NO_GO"
    elif terrain == "MOUNTAIN" and timing == "ACCEPTABLE":
        feasibility = "DIFFICULT"
    elif terrain == "URBAN" and timing == "IDEAL":
        feasibility = "EASY"
    else:
        feasibility = timing
    # Hop 5: extract method
    if feasibility == "NO_GO":
        method = "ABORT_EXTRACT"
    elif feasibility == "EASY" and extract == "READY":
        method = "IMMEDIATE"
    elif feasibility == "DIFFICULT" and extract == "UNAVAILABLE":
        method = "STRANDED"
    elif extract == "DELAYED":
        method = "WAIT"
    else:
        method = "PROCEED"
    return f"5HOP_EXTRACT: {method}. Weather {weather}, visibility {vis}, window {window}, terrain {terrain}, extract {extract}."


# ============================================================
# ALL TEMPLATES BY HOP COUNT
# ============================================================

NHOP_TEMPLATES = {
    1: [
        {"question": "What is the current status?", "fn": hop1_status, "channels": [3]},
        {"question": "What is the threat level?", "fn": hop1_threat, "channels": [16]},
        {"question": "What rule is in effect?", "fn": hop1_rule, "channels": [6]},
        {"question": "What is the current location?", "fn": hop1_location, "channels": [1]},
    ],
    2: [
        {"question": "Given the rule, should we proceed based on status?",
         "fn": hop2_rule_status, "channels": [6, 3]},
        {"question": "How does threat affect our doctrine?",
         "fn": hop2_threat_doctrine, "channels": [16, 38]},
        {"question": "Can we extract given the weather?",
         "fn": hop2_weather_extract, "channels": [17, 19]},
        {"question": "Is our cover adequate for this terrain?",
         "fn": hop2_cover_terrain, "channels": [14, 18]},
    ],
    3: [
        {"question": "What should we do given rule, status and backup?",
         "fn": hop3_rule_status_backup, "channels": [6, 3, 5]},
        {"question": "Assess threat given support and terrain.",
         "fn": hop3_threat_support_terrain, "channels": [16, 15, 18]},
        {"question": "Is the operational window viable?",
         "fn": hop3_weather_visibility_window, "channels": [17, 28, 13]},
        {"question": "What action for the agent given status and priority?",
         "fn": hop3_agent_status_priority, "channels": [2, 3, 4]},
    ],
    4: [
        {"question": "Full decision: check meta, apply rule to status and priority.",
         "fn": hop4_meta_rule_status_priority, "channels": [7, 6, 3, 4]},
        {"question": "Tactical posture given threat, support, terrain and doctrine.",
         "fn": hop4_threat_support_terrain_doctrine, "channels": [16, 15, 18, 38]},
    ],
    5: [
        {"question": "Complete assessment: meta, rule, status, backup and threat.",
         "fn": hop5_meta_rule_status_backup_threat, "channels": [7, 6, 3, 5, 16]},
        {"question": "Full extraction assessment: weather, visibility, window, terrain, extract.",
         "fn": hop5_weather_vis_window_terrain_extract, "channels": [17, 28, 13, 18, 19]},
    ],
}


# ============================================================
# DATA GENERATION
# ============================================================

def generate_nhop_data(
    samples_per_hop=2000,
    output_dir="nhop_training_data",
    a_high=1.0,
    a_low=0.25,
):
    """Generate training data with hop-controlled questions.
    
    Each sample includes:
    - Full 40-channel FDM encoding (same as v3)
    - Hop-specific question
    - Hop-specific answer (depends on N channels with chained logic)
    - Context string with ALL extra channels (forces full decode)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create encoder FIRST (uses random.seed(42) internally)
    encoder = TurboFDMSignalEncoder(
        num_tokens_per_encoder=256, sample_rate=100.0,
        a_high=a_high, a_low=a_low, num_levels=64, seed=42,
    )

    # Set data generation seed AFTER encoder creation
    random.seed(77777)
    np.random.seed(77777)

    all_samples = []

    for nhops, templates in sorted(NHOP_TEMPLATES.items()):
        print(f"\nGenerating {samples_per_hop} samples for {nhops}-hop...")

        for i in tqdm(range(samples_per_hop)):
            memory = {ch: random.choice(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}
            fdm_text, token_ids = encoder.encode_memory(memory)

            template = random.choice(templates)
            base_answer = template["fn"](memory)

            # Append extra channel context (same as v3 training)
            extra = {MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(8, NUM_CHANNELS)}
            extra_parts = [f"{k}={v}" for k, v in extra.items()]
            answer = f"{base_answer} Context: {', '.join(extra_parts)}."

            all_samples.append({
                "fdm_text": fdm_text,
                "question": template["question"],
                "answer": answer,
                "nhops": nhops,
                "channels": template["channels"],
                "memory": {str(ch): memory[ch] for ch in range(NUM_CHANNELS)},
            })

    random.shuffle(all_samples)
    n = len(all_samples)
    splits = {
        "train": all_samples[:int(0.85 * n)],
        "val": all_samples[int(0.85 * n):int(0.95 * n)],
        "test": all_samples[int(0.95 * n):],
    }

    for name, data in splits.items():
        path = os.path.join(output_dir, f"nhop_{name}.jsonl")
        with open(path, 'w') as f:
            for s in data:
                f.write(json.dumps(s) + '\n')
        # Count per hop
        hop_counts = defaultdict(int)
        for s in data:
            hop_counts[s["nhops"]] += 1
        print(f"  {name}: {len(data)} samples — {dict(sorted(hop_counts.items()))}")

    print(f"\nTotal: {n} samples")


# ============================================================
# TRAINING — one additional stage on top of v3
# ============================================================

def train(
    model_path="./fdm_40ch_turbo_v3_model_final",
    data_dir="nhop_training_data",
    output_path="nhop_model_final",
    epochs=5,
    lr=1e-5,
    batch_size=4,
):
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading v3 model from {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)

    train_path = os.path.join(data_dir, "nhop_train.jsonl")
    val_path = os.path.join(data_dir, "nhop_val.jsonl")

    train_dataset = FDMReasoningDataset(train_path, tokenizer)
    val_dataset = FDMReasoningDataset(val_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)

    print(f"Training: {len(train_dataset)} samples, {epochs} epochs, lr={lr}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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

        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"\nSaved to {output_path}/")


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model_path="nhop_model_final",
    data_dir="nhop_training_data",
    max_new_tokens=250,
):
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"Loading model from {model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_path = os.path.join(data_dir, "nhop_test.jsonl")
    with open(test_path) as f:
        samples = [json.loads(line) for line in f]
    print(f"Loaded {len(samples)} test samples")

    # Group by hop count
    by_hop = defaultdict(list)
    for s in samples:
        by_hop[s["nhops"]].append(s)

    results = {}

    for nhops in sorted(by_hop.keys()):
        hop_samples = by_hop[nhops]
        decision_correct = 0
        extra_correct = 0
        extra_total = 0
        total = len(hop_samples)

        print(f"\n{'='*60}")
        print(f"Evaluating {nhops}-hop ({total} samples)...")
        print(f"{'='*60}")

        examples = []

        for i, sample in enumerate(tqdm(hop_samples)):
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
            expected = sample["answer"]
            memory = sample["memory"]

            # Decision accuracy: extract the decision keyword from expected and check
            # Expected format: "NHOP_DECISION: KEYWORD. ..."
            exp_match = re.search(r':\s*(\w+)', expected)
            gen_match = re.search(r':\s*(\w+)', generated)
            if exp_match and gen_match and exp_match.group(1) == gen_match.group(1):
                decision_correct += 1

            # Extra channel accuracy (from Context: string)
            exp_context = re.search(r'Context:\s*(.+?)\.?\s*$', expected, re.DOTALL)
            gen_context = re.search(r'Context:\s*(.+?)\.?\s*$', generated, re.DOTALL)
            if exp_context and gen_context:
                exp_pairs = dict(p.strip().split('=', 1) for p in exp_context.group(1).split(',') if '=' in p)
                gen_pairs = dict(p.strip().split('=', 1) for p in gen_context.group(1).split(',') if '=' in p)
                for k, v in exp_pairs.items():
                    extra_total += 1
                    if gen_pairs.get(k) == v:
                        extra_correct += 1

            if i < 3:
                examples.append((expected[:120], generated[:120]))

        dec_pct = decision_correct / total * 100
        ext_pct = extra_correct / extra_total * 100 if extra_total > 0 else 0

        results[nhops] = {
            "total": total,
            "decision_accuracy": dec_pct,
            "extra_accuracy": ext_pct,
        }

        print(f"\n{nhops}-hop Results:")
        print(f"  Decision accuracy: {decision_correct}/{total} = {dec_pct:.1f}%")
        print(f"  Extra ch accuracy: {extra_correct}/{extra_total} = {ext_pct:.1f}%")
        for exp, got in examples:
            print(f"    Exp: {exp}...")
            print(f"    Got: {got}...")

    # Summary table
    print(f"\n{'='*70}")
    print(f"N-HOP COMPOSITIONAL REASONING: FDM vs Greenblatt (2025)")
    print(f"{'='*70}")
    print(f"{'Hops':<6} {'FDM Decision':<15} {'FDM Extra-Ch':<15} {'Greenblatt (weights)'}")
    print(f"{'-'*70}")

    greenblatt = {2: "~60%", 3: "~34%", 4: "~chance", 5: "N/A"}
    for nhops in sorted(results.keys()):
        r = results[nhops]
        gb = greenblatt.get(nhops, "N/A")
        print(f"{nhops:<6} {r['decision_accuracy']:>5.1f}%         {r['extra_accuracy']:>5.1f}%         {gb}")

    # Save
    results_path = os.path.join(data_dir, "nhop_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_path}")


# ============================================================
# PLOT
# ============================================================

def plot_results(results_path="nhop_training_data/nhop_results.json"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(results_path) as f:
        results = json.load(f)

    hops = sorted([int(h) for h in results.keys()])
    decision_accs = [results[str(h)]["decision_accuracy"] for h in hops]
    extra_accs = [results[str(h)]["extra_accuracy"] for h in hops]

    gb_hops = [2, 3, 4]
    gb_best = [60, 34, 12]
    gb_gpt4 = [9.7, 5, 2]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(hops, decision_accs, 'o-', color='#2196F3', linewidth=2.5, markersize=8,
            label='FDM Eidetic (GPT-2 124M) — Decision', zorder=5)
    ax.plot(hops, extra_accs, 's--', color='#64B5F6', linewidth=1.5, markersize=6,
            label='FDM Eidetic — Extra Channel Decode', zorder=4)
    ax.plot(gb_hops, gb_best, 'D-', color='#F44336', linewidth=2.5, markersize=8,
            label='Greenblatt — Gemini 3 Pro (facts in weights)', zorder=3)
    ax.plot(gb_hops, gb_gpt4, '^--', color='#EF9A9A', linewidth=1.5, markersize=6,
            label='Greenblatt — GPT-4 (facts in weights)', zorder=2)
    ax.axhline(y=20, color='gray', linestyle=':', alpha=0.5, label='Chance')

    ax.set_xlabel('Compositional Reasoning Hops', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Multi-Hop Compositional Reasoning:\nFDM Context vs Facts in Weights', fontsize=14)
    ax.set_xticks(range(1, 6))
    ax.set_ylim(-5, 105)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.annotate('Retrieval bottleneck\n(Greenblatt)', xy=(3, 34), xytext=(4, 55),
                fontsize=9, color='#F44336',
                arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5))

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f'nhop_compositional.{ext}', dpi=300, bbox_inches='tight')
        print(f"Saved nhop_compositional.{ext}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python nhop_training.py generate")
        print("  python nhop_training.py train")
        print("  MODEL_PATH=./nhop_model_final python nhop_training.py eval")
        print("  python nhop_training.py plot")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "generate":
        generate_nhop_data(samples_per_hop=2000)

    elif cmd == "train":
        train()

    elif cmd == "eval":
        model_path = os.environ.get("MODEL_PATH", "./nhop_model_final")
        evaluate(model_path=model_path)

    elif cmd == "plot":
        plot_results()
