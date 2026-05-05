# Eidetic Memory: Why 5.5% and How to Fix It

## The Problem

Your eval shows 189/200 predictions are `FLAG:TRUE` regardless of input. The 11 "correct" ones are just cases where the ground truth happened to be `FLAG:TRUE`. This is **mode collapse**, not learning.

## Root Cause: Architecture Mismatch

The current pipeline asks the wrong question:

```
Current approach (BROKEN):
  Encoder: contexts → iMEC → stegotext
  Training: teach GPT-2+LoRA to do: stegotext + channel → context
  
  This asks GPT-2 to LEARN the inverse of iMEC from examples.
  But iMEC is a deterministic coupling algorithm that depends on 
  GPT-2's own probability distributions. You can't approximate
  it with a LoRA — you need the actual decode algorithm.
```

**Think about it this way**: In your paper, message recovery works via the exact iMEC decoder + OTP decryption + FFT demodulation pipeline. No neural net approximation — the math is deterministic. Trying to train GPT-2 to learn this inverse from input/output pairs is like trying to train a neural net to factor large primes by showing it examples of (product, factors) pairs. The structure of the problem defeats gradient-based learning.

## The Fix: Use iMEC As-Is, Train the *Routing* Layer

The "eidetic memory" concept should be:

```
Correct approach:
  Encoder: contexts → iMEC → stegotext  (unchanged from paper)
  Decoder: stegotext → iMEC decode → contexts  (use actual iMEC decoder!)
  
  What to TRAIN: A lightweight model that learns to do useful 
  things with the decoded contexts — routing, prioritization, 
  summarization — NOT the decoding itself.
```

But there's actually an even more interesting possibility that stays closer to your original "artificial eidetic memory" idea...

## Alternative: Imperfect Steganography as Learnable Compression

Your insight about "imperfect steganography with learnable patterns" is the right one. But you need to actually build that — not use perfect-security iMEC (which, by design, leaves no learnable patterns!).

The key realization: **iMEC with perfect security means KL-divergence → 0 between stegotext and covertext. That literally means there is NO statistical signal for a model to learn from.**

For trainable eidetic memory, you want an encoder that deliberately introduces learnable statistical biases:

1. **Biased token selection**: Instead of minimum entropy coupling, use a coupling that preserves some mutual information between ciphertext blocks and token choices
2. **Channel-dependent context priming**: Different channels get different context prefixes that create detectable distributional shifts
3. **Statistical watermarking**: Embed channel information through systematic token frequency shifts that a trained model can detect

## Files Provided

### Option A: Fixed pipeline using real iMEC (deterministic decode, no training needed)
- `full_encoder_pipeline_v2.py` — Encoder using your paper's iMEC
- `deterministic_decoder.py` — Decoder using actual iMEC decode (no ML)
- `demo_roundtrip.py` — Prove encode→decode works at 100%

### Option B: Trainable "leaky" encoder (for the ML research direction)  
- `leaky_encoder.py` — Encoder that deliberately leaves statistical fingerprints
- `decoder_training_v2.py` — Training pipeline for GPT-2 to learn those fingerprints
- `simple_eval_v2.py` — Evaluation

Both options provided. Option A proves the system works. Option B is the research direction.
