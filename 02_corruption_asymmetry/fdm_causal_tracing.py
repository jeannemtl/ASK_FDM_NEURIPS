"""
Mechanistic Interpretability: Proving GPT-2 Reads FDM Input Signals

v3 model specs (GPT-2 MEDIUM):
  - 24 layers, 16 heads, 1024 hidden dim, vocab=50259
  - [MEMORY] = token ID 50257, [/MEMORY] = token ID 50258
  - ~520 FDM signal tokens (~260 enc1 natural + ~260 enc2 interleaved)
  - ~542 total input tokens
  - 40 channels @ carriers 1-40 Hz, 64 quant levels, turbo dual-pass

Usage:
  python fdm_causal_tracing.py --model_path ./fdm_40ch_turbo_v3_model_final \
      --test_data ./fdm_40ch_turbo_v3_stage4_test.jsonl \
      --output_dir ./causal_tracing_results --num_samples 50

  python fdm_causal_tracing.py --experiment corruption_test --num_samples 100   # ~3 min
  python fdm_causal_tracing.py --experiment causal_trace --num_samples 20       # ~60 min
  python fdm_causal_tracing.py --experiment logit_lens --num_samples 50         # ~5 min
  python fdm_causal_tracing.py --experiment attention_analysis --num_samples 50 # ~5 min
  python fdm_causal_tracing.py --experiment attention_ablation --num_samples 50 # ~15 min
"""

import torch
import torch.nn.functional as F
import numpy as np
import json, os, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MEMORY_OPEN_ID = 50257
MEMORY_CLOSE_ID = 50258

# ============================================================
# TOKEN REGIONS
# ============================================================

def identify_token_regions(ids):
    """Find signal vs question positions via special token IDs 50257/50258."""
    mem_open = mem_close = None
    for i, t in enumerate(ids):
        if t == MEMORY_OPEN_ID and mem_open is None: mem_open = i
        if t == MEMORY_CLOSE_ID and mem_close is None: mem_close = i
    if mem_open is None or mem_close is None:
        raise ValueError(f"Special tokens not found in {len(ids)} tokens")

    sig = list(range(mem_open + 1, mem_close))
    q = list(range(mem_close + 1, len(ids)))
    mid = len(sig) // 2
    enc1, enc2 = sig[:mid], sig[mid:]

    def bins(pos, n=8):
        if not pos: return []
        bs = max(1, len(pos) // n)
        return [pos[i:i+bs] for i in range(0, len(pos), bs)][:n]

    return dict(signal=sig, encoder1=enc1, encoder2=enc2, question=q,
                signal_bins_enc1=bins(enc1), signal_bins_enc2=bins(enc2))


def get_target(tok, answer):
    """Get first token of expected answer for accuracy measurement."""
    if not answer or not answer.strip(): return None
    ts = tok.encode(" " + answer.split()[0])
    return ts[0] if ts else None

# ============================================================
# MANUAL FORWARD (unified, works for any GPT-2 size)
# ============================================================

def manual_forward(model, input_ids, device, corrupt_pos=None, noise=None,
                   restore_cache=None, restore_layer=None,
                   restore_comp=None, restore_pos=None, cache_out=None):
    """
    Unified manual forward through GPT-2 with optional:
      - Embedding corruption at corrupt_pos with given noise tensor
      - Clean activation restore at (restore_layer, restore_comp, restore_pos)
      - Activation caching into cache_out dict
    Works for any GPT-2 size (small/medium/large/xl).
    """
    input_ids = input_ids.to(device)
    with torch.no_grad():
        te = model.transformer.wte(input_ids)
        pe = model.transformer.wpe(torch.arange(input_ids.shape[1], device=device).unsqueeze(0))
        if corrupt_pos is not None and noise is not None:
            te[:, corrupt_pos, :] = te[:, corrupt_pos, :] + noise.to(device)
        h = model.transformer.drop(te + pe)

        for li, block in enumerate(model.transformer.h):
            if cache_out is not None:
                cache_out[f'res_{li}'] = h.clone()
            if li == restore_layer and restore_comp == 'residual' and restore_cache:
                h[:, restore_pos, :] = restore_cache[f'res_{li}'][:, restore_pos, :]

            ao = block.attn(block.ln_1(h))[0]
            if cache_out is not None:
                cache_out[f'attn_{li}'] = ao.clone()
            if li == restore_layer and restore_comp == 'attn' and restore_cache:
                ao[:, restore_pos, :] = restore_cache[f'attn_{li}'][:, restore_pos, :]
            h = h + ao

            mo = block.mlp(block.ln_2(h))
            if cache_out is not None:
                cache_out[f'mlp_{li}'] = mo.clone()
            if li == restore_layer and restore_comp == 'mlp' and restore_cache:
                mo[:, restore_pos, :] = restore_cache[f'mlp_{li}'][:, restore_pos, :]
            h = h + mo

        h = model.transformer.ln_f(h)
        return model.lm_head(h)[0, -1, :]

# ============================================================
# EXP 1-2: CORRUPTION TEST
# ============================================================

def run_corruption_test(model, tok, samples, device, sigmas=None):
    if sigmas is None:
        sigmas = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    print("\n" + "="*60 + "\nEXP 1-2: CORRUPTION TEST\n" + "="*60)

    results = {}
    for s in sigmas:
        correct = total = 0; psum = 0.0
        for sam in tqdm(samples, desc=f"sigma={s:.1f}", leave=False):
            ids = tok.encode(sam['input_text'], return_tensors='pt')
            reg = identify_token_regions(ids[0].tolist())
            tgt = get_target(tok, sam['answer'])
            if tgt is None: continue

            if s == 0.0:
                logits = manual_forward(model, ids, device)
            else:
                with torch.no_grad():
                    te = model.transformer.wte(ids.to(device))
                noise = torch.randn_like(te[:, reg['signal'], :]) * s
                logits = manual_forward(model, ids, device,
                                        corrupt_pos=reg['signal'], noise=noise)
            p = F.softmax(logits, dim=-1)
            psum += p[tgt].item()
            if logits.argmax().item() == tgt: correct += 1
            total += 1

        acc = correct / max(total, 1)
        avg_p = psum / max(total, 1)
        results[s] = dict(accuracy=acc, avg_prob=avg_p, n=total)
        print(f"  sigma={s:5.1f}: acc={acc:.3f}  P(correct)={avg_p:.4f}  n={total}")
    return results

# ============================================================
# EXP 3: CAUSAL TRACING (ROME-style)
# ============================================================

def run_causal_trace(model, tok, samples, device, noise_std=5.0):
    """Corrupt signal, selectively restore at (layer, position_group), measure recovery."""
    print("\n" + "="*60 + f"\nEXP 3: CAUSAL TRACING (sigma={noise_std}, n={len(samples)})\n" + "="*60)
    nl = model.config.n_layer
    all_hm = {'attn': [], 'mlp': []}
    skipped = 0

    for sam in tqdm(samples, desc="Causal trace"):
        ids = tok.encode(sam['input_text'], return_tensors='pt')
        reg = identify_token_regions(ids[0].tolist())
        sp = reg['signal']
        tgt = get_target(tok, sam['answer'])
        if tgt is None: continue

        # Clean + cache
        cc = {}
        cl = manual_forward(model, ids, device, cache_out=cc)
        cp = F.softmax(cl, dim=-1)[tgt].item()

        # Corrupt
        with torch.no_grad():
            te = model.transformer.wte(ids.to(device))
        noise = torch.randn_like(te[:, sp, :]) * noise_std
        crl = manual_forward(model, ids, device, corrupt_pos=sp, noise=noise)
        crp = F.softmax(crl, dim=-1)[tgt].item()

        denom = cp - crp
        if abs(denom) < 1e-6:
            skipped += 1; continue

        # Position groups: 8 enc1 bins + 8 enc2 bins + question
        pg = {}
        for i, b in enumerate(reg['signal_bins_enc1']): pg[f'e1_{i}'] = b
        for i, b in enumerate(reg['signal_bins_enc2']): pg[f'e2_{i}'] = b
        pg['question'] = reg['question']
        gnames = list(pg.keys())

        for comp in ['attn', 'mlp']:
            hm = np.zeros((nl, len(gnames)))
            for li in range(nl):
                for gi, (gn, pos) in enumerate(pg.items()):
                    if not pos: continue
                    rl = manual_forward(model, ids, device,
                                        corrupt_pos=sp, noise=noise,
                                        restore_cache=cc, restore_layer=li,
                                        restore_comp=comp, restore_pos=pos)
                    rp = F.softmax(rl, dim=-1)[tgt].item()
                    hm[li, gi] = np.clip((rp - crp) / denom, -0.5, 1.5)
            all_hm[comp].append(hm)

        del cc, noise
        if device.type == 'cuda': torch.cuda.empty_cache()

    res = dict(group_names=gnames, num_samples=len(all_hm['attn']), skipped=skipped)
    for c in ['attn', 'mlp']:
        res[f'{c}_heatmap'] = np.mean(all_hm[c], axis=0) if all_hm[c] else np.zeros((nl, 1))
    return res

# ============================================================
# EXP 4: LOGIT LENS
# ============================================================

def run_logit_lens(model, tok, samples, device):
    """Project residual stream through unembedding at each layer."""
    print("\n" + "="*60 + "\nEXP 4: LOGIT LENS\n" + "="*60)
    nl = model.config.n_layer
    unembed = model.lm_head.weight.data  # (vocab, hidden_dim)
    lp = np.zeros(nl + 1)
    count = 0

    for sam in tqdm(samples, desc="Logit lens"):
        ids = tok.encode(sam['input_text'], return_tensors='pt').to(device)
        tgt = get_target(tok, sam['answer'])
        if tgt is None: continue

        cache = {}
        manual_forward(model, ids, device, cache_out=cache)

        for li in range(nl):
            normed = model.transformer.ln_f(cache[f'res_{li}'])
            logits_here = normed[0, -1, :] @ unembed.T
            lp[li] += F.softmax(logits_here, dim=-1)[tgt].item()

        # Final: residual after last layer
        final_res = cache[f'res_{nl-1}'] + cache[f'attn_{nl-1}'] + cache[f'mlp_{nl-1}']
        final = model.transformer.ln_f(final_res)
        lp[nl] += F.softmax(final[0, -1, :] @ unembed.T, dim=-1)[tgt].item()
        count += 1
        del cache

    lp /= max(count, 1)
    return dict(layer_probs=lp, num_samples=count)

# ============================================================
# EXP 4b: BULK RESTORE (all signal positions at one layer)
# ============================================================

def run_bulk_restore(model, tok, samples, device, noise_std=5.0):
    """
    Restore ALL signal positions at a single layer, sweep across layers.
    This answers: at which layer does the signal information become critical?

    Unlike fine-grained causal tracing (which restores ~32 positions per bin),
    this restores all ~520 signal positions at once, so the model has the
    full signal available at that layer. This should show much higher recovery
    and reveal the layer(s) where FDM processing bottlenecks.
    """
    print("\n" + "="*60 + f"\nEXP 4b: BULK RESTORE (sigma={noise_std}, n={len(samples)})\n" + "="*60)
    nl = model.config.n_layer

    # recovery[layer, component] averaged across samples
    recoveries = {'attn': np.zeros(nl), 'mlp': np.zeros(nl), 'residual': np.zeros(nl)}
    count = 0

    for sam in tqdm(samples, desc="Bulk restore"):
        ids = tok.encode(sam['input_text'], return_tensors='pt')
        reg = identify_token_regions(ids[0].tolist())
        sp = reg['signal']
        tgt = get_target(tok, sam['answer'])
        if tgt is None: continue

        # Clean + cache
        cc = {}
        cl = manual_forward(model, ids, device, cache_out=cc)
        cp = F.softmax(cl, dim=-1)[tgt].item()

        # Corrupt
        with torch.no_grad():
            te = model.transformer.wte(ids.to(device))
        noise = torch.randn_like(te[:, sp, :]) * noise_std
        crl = manual_forward(model, ids, device, corrupt_pos=sp, noise=noise)
        crp = F.softmax(crl, dim=-1)[tgt].item()

        denom = cp - crp
        if abs(denom) < 1e-6:
            continue

        for comp in ['attn', 'mlp', 'residual']:
            for li in range(nl):
                rl = manual_forward(model, ids, device,
                                    corrupt_pos=sp, noise=noise,
                                    restore_cache=cc, restore_layer=li,
                                    restore_comp=comp, restore_pos=sp)
                rp = F.softmax(rl, dim=-1)[tgt].item()
                recoveries[comp][li] += np.clip((rp - crp) / denom, -0.5, 1.5)

        count += 1
        del cc, noise
        if device.type == 'cuda': torch.cuda.empty_cache()

    for comp in recoveries:
        recoveries[comp] /= max(count, 1)

    # Print summary
    for comp in ['residual', 'attn', 'mlp']:
        peak = recoveries[comp].argmax()
        val = recoveries[comp][peak]
        print(f"  {comp:10s}: peak recovery={val:.3f} at layer {peak}")

    return dict(recoveries=recoveries, num_samples=count)


def plot_bulk_restore(r, out_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    nl = len(r['recoveries']['attn'])
    layers = range(nl)
    ax.plot(layers, r['recoveries']['residual'], 'k-D', lw=2, ms=5, label='Residual stream')
    ax.plot(layers, r['recoveries']['attn'], 'b-o', lw=2, ms=5, label='Attention output')
    ax.plot(layers, r['recoveries']['mlp'], 'r-s', lw=2, ms=5, label='MLP output')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Recovery (all signal positions restored)', fontsize=12)
    ax.set_title(f'Bulk Restore: Which Layer Is the Signal Processing Bottleneck?\n'
                 f'(n={r["num_samples"]})', fontsize=14)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', alpha=0.3); ax.axhline(1, color='gray', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'bulk_restore.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> bulk_restore.png")

# ============================================================
# EXP 5: ATTENTION ANALYSIS
# ============================================================

def run_attention_analysis(model, tok, samples, device):
    """Track attention from last position to signal/question positions per head."""
    print("\n" + "="*60 + "\nEXP 5: ATTENTION ANALYSIS\n" + "="*60)
    nl, nh = model.config.n_layer, model.config.n_head
    a_sig = np.zeros((nl, nh))
    a_e1 = np.zeros((nl, nh))
    a_e2 = np.zeros((nl, nh))
    a_q = np.zeros((nl, nh))
    count = 0

    # Ensure model returns attentions (some saved configs override the kwarg)
    old_cfg = model.config.output_attentions
    model.config.output_attentions = True

    for sam in tqdm(samples, desc="Attention"):
        ids = tok.encode(sam['input_text'], return_tensors='pt').to(device)
        reg = identify_token_regions(ids[0].tolist())
        with torch.no_grad():
            out = model(ids, output_attentions=True)
        if out.attentions is None or len(out.attentions) == 0:
            print("WARNING: output_attentions returned None, skipping sample")
            continue
        last = ids.shape[1] - 1
        for li, att in enumerate(out.attentions):
            if att is None:
                continue
            a = att[0, :, last, :]  # (heads, seq)
            if reg['signal']:   a_sig[li] += a[:, reg['signal']].mean(-1).cpu().numpy()
            if reg['encoder1']: a_e1[li]  += a[:, reg['encoder1']].mean(-1).cpu().numpy()
            if reg['encoder2']: a_e2[li]  += a[:, reg['encoder2']].mean(-1).cpu().numpy()
            if reg['question']: a_q[li]   += a[:, reg['question']].mean(-1).cpu().numpy()
        count += 1

    for arr in [a_sig, a_e1, a_e2, a_q]: arr /= max(count, 1)
    model.config.output_attentions = old_cfg
    return dict(attn_to_signal=a_sig, attn_to_enc1=a_e1, attn_to_enc2=a_e2,
                attn_to_question=a_q, num_samples=count)

# ============================================================
# EXP 6: ATTENTION ABLATION
# ============================================================

def run_attention_ablation(model, tok, samples, device):
    """
    Zero out attention weights TO signal (or question) positions.
    Uses manual attention computation inside each block to properly
    mask the attention weight matrix before softmax.
    Works for any GPT-2 size.
    """
    print("\n" + "="*60 + "\nEXP 6: ATTENTION ABLATION\n" + "="*60)

    conditions = ['clean', 'ablate_signal', 'ablate_question', 'ablate_enc1', 'ablate_enc2']
    results = {c: dict(correct=0, total=0) for c in conditions}

    num_heads = model.config.n_head
    head_dim = model.config.n_embd // model.config.n_head

    for sam in tqdm(samples, desc="Ablation"):
        ids = tok.encode(sam['input_text'], return_tensors='pt').to(device)
        reg = identify_token_regions(ids[0].tolist())
        tgt = get_target(tok, sam['answer'])
        if tgt is None: continue
        seq_len = ids.shape[1]

        for cond in conditions:
            if cond == 'clean':
                ablate_pos = None
            elif cond == 'ablate_signal':
                ablate_pos = reg['signal']
            elif cond == 'ablate_question':
                ablate_pos = reg['question']
            elif cond == 'ablate_enc1':
                ablate_pos = reg['encoder1']
            elif cond == 'ablate_enc2':
                ablate_pos = reg['encoder2']

            with torch.no_grad():
                if ablate_pos is None or len(ablate_pos) == 0:
                    logits = model(ids).logits[0, -1, :]
                else:
                    # Manual forward with attention masking
                    te = model.transformer.wte(ids)
                    pe = model.transformer.wpe(
                        torch.arange(seq_len, device=device).unsqueeze(0))
                    h = model.transformer.drop(te + pe)

                    for block in model.transformer.h:
                        ln_h = block.ln_1(h)
                        attn = block.attn

                        # QKV projection
                        qkv = attn.c_attn(ln_h)
                        q, k, v = qkv.split(attn.split_size, dim=2)

                        # Reshape to (batch, heads, seq, head_dim)
                        def reshape(x):
                            return x.view(1, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
                        q, k, v = reshape(q), reshape(k), reshape(v)

                        # Attention scores
                        w = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)

                        # Causal mask
                        causal = torch.tril(torch.ones(seq_len, seq_len,
                                            dtype=torch.bool, device=device))
                        w = w.masked_fill(~causal.view(1, 1, seq_len, seq_len), float('-inf'))

                        # ABLATION: block attention TO target positions
                        w[:, :, :, ablate_pos] = float('-inf')

                        w = F.softmax(w, dim=-1)
                        w = attn.attn_dropout(w)

                        ao = torch.matmul(w, v)
                        ao = ao.permute(0, 2, 1, 3).contiguous().view(1, seq_len, num_heads * head_dim)
                        ao = attn.c_proj(ao)
                        ao = attn.resid_dropout(ao)

                        h = h + ao
                        h = h + block.mlp(block.ln_2(h))

                    h = model.transformer.ln_f(h)
                    logits = model.lm_head(h)[0, -1, :]

            results[cond]['total'] += 1
            if logits.argmax().item() == tgt:
                results[cond]['correct'] += 1

    for c in conditions:
        t = results[c]['total']
        results[c]['accuracy'] = results[c]['correct'] / t if t > 0 else 0
        print(f"  {c:20s}: {results[c]['correct']}/{t} = {results[c]['accuracy']:.3f}")
    return results

# ============================================================
# VISUALIZATION
# ============================================================

def plot_corruption(r, out_dir):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    levels = sorted(r.keys())
    accs = [r[n]['accuracy'] for n in levels]
    probs = [r[n]['avg_prob'] for n in levels]

    a1.plot(levels, accs, 'bo-', lw=2, ms=8)
    a1.set_xlabel('Noise sigma (signal embeddings)', fontsize=12)
    a1.set_ylabel('Next-token accuracy', fontsize=12)
    a1.set_title('Signal Corruption Test', fontsize=14)
    a1.axhline(y=accs[0], color='g', ls='--', alpha=0.5, label=f'Clean: {accs[0]:.3f}')
    a1.legend(); a1.grid(alpha=0.3)

    a2.plot(levels, probs, 'ro-', lw=2, ms=8)
    a2.set_xlabel('Noise sigma', fontsize=12)
    a2.set_ylabel('P(correct token)', fontsize=12)
    a2.set_title('Probability Degradation', fontsize=14)
    a2.axhline(y=probs[0], color='g', ls='--', alpha=0.5, label=f'Clean: {probs[0]:.4f}')
    a2.legend(); a2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'corruption_test.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> corruption_test.png")


def plot_causal_trace(r, out_dir):
    gn = r['group_names']
    for comp in ['attn', 'mlp']:
        hm = r[f'{comp}_heatmap']
        fig, ax = plt.subplots(figsize=(18, 7))
        vmax = max(abs(hm.min()), abs(hm.max()), 0.1)
        im = ax.imshow(hm.T, aspect='auto', cmap='RdYlBu_r',
                       vmin=-0.1, vmax=vmax, interpolation='nearest')
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Position Group', fontsize=12)
        ax.set_title(f'Causal Trace: {comp.upper()} Restore (n={r["num_samples"]})', fontsize=14)
        ax.set_xticks(range(hm.shape[0]))
        ax.set_yticks(range(len(gn)))
        ax.set_yticklabels(gn, fontsize=8)
        plt.colorbar(im, ax=ax, label='Recovery (0=corrupt, 1=clean)')
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                v = hm[i, j]
                if abs(v) > 0.15:
                    ax.text(i, j, f'{v:.2f}', ha='center', va='center',
                            fontsize=5, color='white' if abs(v) > 0.4 else 'black')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'causal_trace_{comp}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  -> causal_trace_{comp}.png")

    # Summary: signal vs question recovery by layer
    fig, ax = plt.subplots(figsize=(12, 5))
    mlp_hm = r['mlp_heatmap']
    attn_hm = r['attn_heatmap']
    si = [i for i, n in enumerate(gn) if n.startswith('e1_') or n.startswith('e2_')]
    qi = [i for i, n in enumerate(gn) if n == 'question']
    layers = range(mlp_hm.shape[0])
    if si:
        ax.plot(layers, mlp_hm[:, si].mean(1), 'b-o', lw=2, ms=5, label='Signal (MLP)')
        ax.plot(layers, attn_hm[:, si].mean(1), 'b--^', lw=1.5, ms=4, alpha=0.7, label='Signal (Attn)')
    if qi:
        ax.plot(layers, mlp_hm[:, qi].mean(1), 'r-s', lw=2, ms=5, label='Question (MLP)')
        ax.plot(layers, attn_hm[:, qi].mean(1), 'r--v', lw=1.5, ms=4, alpha=0.7, label='Question (Attn)')
    ax.set_xlabel('Layer'); ax.set_ylabel('Recovery')
    ax.set_title('Signal vs Question: Where Does Information Flow?', fontsize=14)
    ax.legend(); ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', alpha=0.3); ax.axhline(1, color='gray', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'causal_trace_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> causal_trace_summary.png")


def plot_logit_lens(r, out_dir):
    lp = r['layer_probs']
    n = len(lp)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(n), lp, 'go-', lw=2, ms=6)
    labels = [str(i) for i in range(n-1)] + ['final']
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel('Layer (residual stream)', fontsize=12)
    ax.set_ylabel('P(correct answer token)', fontsize=12)
    ax.set_title(f'Logit Lens: Answer Crystallization (n={r["num_samples"]})', fontsize=14)
    ax.grid(alpha=0.3)
    diffs = np.diff(lp)
    mj = np.argmax(diffs)
    ax.annotate(f'Layer {mj}->{mj+1}: dp={diffs[mj]:.3f}',
                xy=(mj+0.5, (lp[mj]+lp[mj+1])/2), fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='red'),
                xytext=(mj+3, lp[mj+1]+0.05))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'logit_lens.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> logit_lens.png")


def plot_attention(r, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    for ax, (arr, title, cm) in zip(axes[:3], [
        (r['attn_to_enc1'], 'Encoder 1 (natural)', 'Blues'),
        (r['attn_to_enc2'], 'Encoder 2 (interleaved)', 'Greens'),
        (r['attn_to_question'], 'Question', 'Reds'),
    ]):
        im = ax.imshow(arr, aspect='auto', cmap=cm)
        ax.set_xlabel('Head'); ax.set_ylabel('Layer'); ax.set_title(title)
        plt.colorbar(im, ax=ax)

    total = r['attn_to_signal'] + r['attn_to_question'] + 1e-8
    ratio = r['attn_to_signal'] / total
    im = axes[3].imshow(ratio, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[3].set_xlabel('Head'); axes[3].set_ylabel('Layer')
    axes[3].set_title('Signal fraction (>0.5=signal-focused)')
    plt.colorbar(im, ax=axes[3])

    plt.suptitle(f'Attention from Last Position (n={r["num_samples"]})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'attention_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> attention_analysis.png")

    # Summary by layer
    fig, ax = plt.subplots(figsize=(12, 5))
    layers = range(r['attn_to_signal'].shape[0])
    ax.plot(layers, r['attn_to_enc1'].mean(1), 'b-o', lw=2, ms=5, label='-> Encoder 1')
    ax.plot(layers, r['attn_to_enc2'].mean(1), 'g-^', lw=2, ms=5, label='-> Encoder 2')
    ax.plot(layers, r['attn_to_question'].mean(1), 'r-s', lw=2, ms=5, label='-> Question')
    ax.set_xlabel('Layer'); ax.set_ylabel('Avg attention weight')
    ax.set_title('Where Does the Model Look?'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'attention_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> attention_summary.png")

# ============================================================
# DATA LOADING
# ============================================================

def load_samples(path, n, tokenizer):
    samples = []
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            ids_check = tokenizer.encode(f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:")
            if len(ids_check) > 1000: continue
            s["input_text"] = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
            samples.append(s)
            if len(samples) >= n: break

    print(f"Loaded {len(samples)} samples from {path}")
    if samples:
        ids = tokenizer.encode(samples[0]['input_text'])
        reg = identify_token_regions(ids)
        print(f"  Total tokens: {len(ids)}")
        print(f"  Signal: {len(reg['signal'])} (enc1={len(reg['encoder1'])}, enc2={len(reg['encoder2'])})")
        print(f"  Question: {len(reg['question'])}")
        print(f"  Answer: {samples[0]['answer'][:80]}...")
    return samples

# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='./fdm_40ch_turbo_v3_model_final')
    p.add_argument('--test_data', default='./fdm_40ch_turbo_v3_stage4_test.jsonl')
    p.add_argument('--output_dir', default='./causal_tracing_results')
    p.add_argument('--num_samples', type=int, default=50)
    p.add_argument('--experiment', default='all',
                   choices=['all', 'corruption_test', 'causal_trace',
                            'logit_lens', 'bulk_restore',
                            'attention_analysis', 'attention_ablation'])
    p.add_argument('--noise_std', type=float, default=5.0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Loading model: {args.model_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device).eval()
    nl, nh, nd = model.config.n_layer, model.config.n_head, model.config.n_embd
    print(f"  {nl} layers, {nh} heads, {nd}d, vocab={len(tokenizer)}")
    print(f"  Head dim: {nd // nh}")

    # Verify special tokens
    test_enc = tokenizer.encode('[MEMORY]')
    assert test_enc == [MEMORY_OPEN_ID], f"[MEMORY] mismatch: {test_enc}"
    print(f"  [MEMORY]={MEMORY_OPEN_ID}, [/MEMORY]={MEMORY_CLOSE_ID} OK")

    samples = load_samples(args.test_data, args.num_samples, tokenizer)

    all_results = {}

    if args.experiment in ['all', 'corruption_test']:
        r = run_corruption_test(model, tokenizer, samples, device)
        all_results['corruption'] = r
        plot_corruption(r, args.output_dir)

    if args.experiment in ['all', 'causal_trace']:
        n_trace = min(20, len(samples))
        r = run_causal_trace(model, tokenizer, samples[:n_trace], device, args.noise_std)
        all_results['causal_trace'] = r
        plot_causal_trace(r, args.output_dir)

    if args.experiment in ['all', 'logit_lens']:
        r = run_logit_lens(model, tokenizer, samples, device)
        all_results['logit_lens'] = r
        plot_logit_lens(r, args.output_dir)

    if args.experiment in ['all', 'bulk_restore']:
        n_bulk = min(20, len(samples))
        r = run_bulk_restore(model, tokenizer, samples[:n_bulk], device, args.noise_std)
        all_results['bulk_restore'] = r
        plot_bulk_restore(r, args.output_dir)

    if args.experiment in ['all', 'attention_analysis']:
        r = run_attention_analysis(model, tokenizer, samples, device)
        all_results['attention'] = r
        plot_attention(r, args.output_dir)

    if args.experiment in ['all', 'attention_ablation']:
        r = run_attention_ablation(model, tokenizer, samples, device)
        all_results['ablation'] = r

    # Save JSON
    def to_json(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: to_json(v) for k, v in obj.items()}
        if isinstance(obj, list): return [to_json(v) for v in obj]
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        return obj

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(to_json(all_results), f, indent=2)

    # ---- INTERPRETATION SUMMARY ----
    print("\n" + "="*60 + "\nINTERPRETATION GUIDE\n" + "="*60)

    if 'corruption' in all_results:
        cr = all_results['corruption']
        clean_acc = cr.get(0.0, {}).get('accuracy', 0)
        worst = min(cr.values(), key=lambda x: x['accuracy'])['accuracy']
        drop = clean_acc - worst
        print(f"\n  CORRUPTION: clean={clean_acc:.3f} -> worst={worst:.3f} (drop={drop:.3f})")
        if drop > 0.5:   print("  ** STRONG: >50% drop confirms model READS from signal tokens")
        elif drop > 0.2: print("  ~  MODERATE: signal matters, partial weight backup possible")
        else:             print("  xx WEAK: model may rely on weight-stored associations")

    if 'causal_trace' in all_results:
        ct = all_results['causal_trace']
        gn = ct['group_names']
        mlp = ct['mlp_heatmap']
        si = [i for i, n in enumerate(gn) if n.startswith('e1_') or n.startswith('e2_')]
        qi = [i for i, n in enumerate(gn) if n == 'question']
        ms = mq = 0
        if si:
            ms = mlp[:, si].max()
            best_layer = mlp[:, si].mean(1).argmax()
            print(f"\n  CAUSAL TRACE: max signal recovery={ms:.3f}, peak layer={best_layer}")
        if qi:
            mq = mlp[:, qi].max()
            print(f"  CAUSAL TRACE: max question recovery={mq:.3f}")
        if si and qi:
            if ms > mq: print("  ** CONFIRMED: Signal > Question -> FDM READING pattern")
            else:        print("  xx UNEXPECTED: Question > Signal -> weight-recall pattern")

    if 'logit_lens' in all_results:
        lp = all_results['logit_lens']['layer_probs']
        diffs = np.diff(lp)
        pk = np.argmax(diffs)
        print(f"\n  LOGIT LENS: crystallizes at layer {pk}->{pk+1} (of {nl})")
        frac = pk / nl
        if frac >= 0.6:   print("  ** LATE: consistent with multi-step signal processing")
        elif frac >= 0.3: print("  ~  MID: FDM decoding computation happening here")
        else:              print("  xx EARLY: possible weight-based shortcut")

    if 'bulk_restore' in all_results:
        br = all_results['bulk_restore']['recoveries']
        print(f"\n  BULK RESTORE (all signal positions at one layer):")
        for comp in ['residual', 'attn', 'mlp']:
            pk_layer = br[comp].argmax()
            pk_val = br[comp][pk_layer]
            print(f"    {comp:10s}: peak={pk_val:.3f} at layer {pk_layer}")
        best_comp = max(br, key=lambda c: br[c].max())
        best_val = br[best_comp].max()
        best_layer = br[best_comp].argmax()
        if best_val > 0.5:
            print(f"  ** STRONG: {best_comp} at layer {best_layer} recovers {best_val:.0%}")
            if best_layer < nl // 3:
                print(f"     EARLY layer -> signal reading from input")
            elif best_layer < 2 * nl // 3:
                print(f"     MID layer -> signal decoding computation")
            else:
                print(f"     LATE layer -> delayed processing")
        elif best_val > 0.2:
            print(f"  ~  MODERATE: best recovery {best_val:.3f} -> distributed processing")
        else:
            print(f"  xx WEAK: max recovery only {best_val:.3f} -> highly distributed")

    if 'ablation' in all_results:
        ab = all_results['ablation']
        ca = ab['clean']['accuracy']
        sa = ab['ablate_signal']['accuracy']
        qa = ab['ablate_question']['accuracy']
        e1 = ab.get('ablate_enc1', {}).get('accuracy', -1)
        e2 = ab.get('ablate_enc2', {}).get('accuracy', -1)
        sig_drop = ca - sa
        q_drop = ca - qa

        print(f"\n  ABLATION:")
        print(f"    clean            = {ca:.3f}")
        print(f"    ablate_signal    = {sa:.3f}  (drop={sig_drop:.3f})")
        print(f"    ablate_question  = {qa:.3f}  (drop={q_drop:.3f})")
        if e1 >= 0: print(f"    ablate_enc1      = {e1:.3f}  (drop={ca - e1:.3f})")
        if e2 >= 0: print(f"    ablate_enc2      = {e2:.3f}  (drop={ca - e2:.3f})")

        # Signal reading analysis
        if sig_drop > 0.5:
            print(f"  ** STRONG: {sig_drop:.0%} drop from signal ablation -> model READS signal tokens")
        elif sig_drop > 0.2:
            print(f"  ~  MODERATE: {sig_drop:.0%} drop from signal ablation -> partial signal dependence")
        else:
            print(f"  xx WEAK: only {sig_drop:.0%} drop -> model may not read from signal")

        # Question drop is expected - question determines answer FORMAT
        if q_drop > sig_drop:
            print(f"  (Question drop of {q_drop:.0%} is expected: question tokens determine answer")
            print(f"   format/type, while signal tokens provide channel values. Both are needed.)")

        # Encoder asymmetry analysis
        if e1 >= 0 and e2 >= 0:
            e1_drop = ca - e1
            e2_drop = ca - e2
            if e1_drop > e2_drop + 0.3:
                print(f"  ** ENCODER ASYMMETRY: enc1 (natural order) carries primary signal")
                print(f"     enc1 drop={e1_drop:.3f} >> enc2 drop={e2_drop:.3f}")
                print(f"     Model relies on natural-order encoding; turbo interleaving is secondary.")
            elif e2_drop > e1_drop + 0.3:
                print(f"  ** ENCODER ASYMMETRY: enc2 (interleaved) carries primary signal")
            else:
                print(f"  ~  Both encoders contribute comparably (enc1 drop={e1_drop:.3f}, enc2 drop={e2_drop:.3f})")

    print(f"\n  All outputs in {args.output_dir}/")


if __name__ == '__main__':
    main()
