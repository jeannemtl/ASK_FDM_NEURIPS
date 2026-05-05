"""
Per-channel frequency gradient eval for Hermes3-3B (from-scratch FDM model).
Reproduces Table 6 (tab:frequency_gradient) for Hermes3.
"""

import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_PATH = "/workspace/fdm-40ch-hermes3-3b"
TEST_PATH  = "/workspace/edeidic_memory_14_percent/fdm_40ch_turbo_v3_hermes3_stage4_test.jsonl"
OUTPUT_PATH = "/workspace/edeidic_memory_14_percent/hermes3_perchannel_results.txt"

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading model from {MODEL_PATH}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    print("Model loaded.")

    samples = [json.loads(l) for l in open(TEST_PATH)]
    print(f"Evaluating on {len(samples)} samples...\n")

    ch_correct = {ch: 0 for ch in range(8, NUM_CHANNELS)}
    ch_total   = {ch: 0 for ch in range(8, NUM_CHANNELS)}
    action_correct = 0
    total = 0

    for s in tqdm(samples, desc="Hermes3 per-channel eval"):
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=250,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        expected = s['answer']

        # Action accuracy
        for kw in ['abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN',
                   'share', 'Do NOT share', 'Do not share', 'HIGH RISK', 'LOW RISK', 'MODERATE']:
            if kw.lower() in expected.lower() and kw.lower() in response.lower():
                action_correct += 1
                break

        # Per-channel extra accuracy
        memory = s['memory']
        for ch in range(8, NUM_CHANNELS):
            expected_val = memory[str(ch)]
            ch_total[ch] += 1
            if expected_val in response:
                ch_correct[ch] += 1

        total += 1

    # --- Print results ---
    lines = []
    lines.append("=" * 60)
    lines.append("HERMES3-3B PER-CHANNEL FREQUENCY GRADIENT (Tab. 6)")
    lines.append("=" * 60)
    lines.append(f"Action accuracy: {100*action_correct/total:.1f}%  (n={total})")
    lines.append("")
    lines.append("Per-channel extra accuracy (ch8--39):")

    overall_correct = 0
    overall_total = 0
    for ch in range(8, NUM_CHANNELS):
        t = ch_total[ch]
        c = ch_correct[ch]
        pct = 100 * c / t if t else 0
        overall_correct += c
        overall_total += t
        flag = " <<<" if pct < 99 else ""
        lines.append(f"  ch{ch:2d} ({MEMORY_SCHEMAS[ch][0]:<12}): {pct:.1f}%{flag}")

    lines.append("")
    lines.append(f"Overall extra accuracy: {100*overall_correct/overall_total:.1f}%")
    lines.append("")

    # Band summary for Table 6
    bands = [
        ("Low (ch8--13)",      range(8,  14)),
        ("Mid-low (ch14--20)", range(14, 21)),
        ("Mid-high (ch21--30)",range(21, 31)),
        ("High (ch31--35)",    range(31, 36)),
        ("Very high (ch36--39)",range(36, 40)),
    ]
    lines.append("Band summary (for Table 6):")
    for band_name, ch_range in bands:
        accs = [100*ch_correct[ch]/ch_total[ch] for ch in ch_range if ch_total[ch] > 0]
        lines.append(f"  {band_name}: {min(accs):.1f}--{max(accs):.1f}%")

    output_str = "\n".join(lines)
    print(output_str)

    with open(OUTPUT_PATH, 'w') as f:
        f.write(output_str + "\n")
    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
