# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
"""Generate all figures for the KernelBench-Verified tech report."""
import json, os, math, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})
from matplotlib.patches import Patch

# ─── CONFIG ───────────────────────────────────────────────────────────────────
EXCLUDED = {2: {23, 80, 83}}
MODELS = ["claude-opus-4-7", "claude-opus-4-8", "claude-sonnet-4-6", "gpt-5.5", "kimi-k2.6", "gemini-3-flash-preview", "gemini-3.1-pro-preview"]
# Full labels (legends/titles) include the reasoning-effort setting.
MODEL_LABELS = {
    "claude-opus-4-7": "Claude Opus 4.7 (high)",
    "claude-opus-4-8": "Claude Opus 4.8 (high)",
    "claude-sonnet-4-6": "Claude Sonnet 4.6 (high)",
    "gpt-5.5": "GPT-5.5 (medium)",
    "kimi-k2.6": "Kimi K2.6",
    "gemini-3-flash-preview": "Gemini Flash (high)",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro (high)",
}
# Short two-line labels for x-axis ticks (avoid .split() breakage from the effort suffix).
MODEL_SHORT = {
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-opus-4-8": "Claude Opus 4.8",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gpt-5.5": "GPT-5.5",
    "kimi-k2.6": "Kimi K2.6",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
}
MODEL_COLORS = {
    "claude-opus-4-7": "#4CAF50",
    "claude-opus-4-8": "#8BC34A",
    "claude-sonnet-4-6": "#9C27B0",
    "gpt-5.5": "#FF9800",
    "kimi-k2.6": "#E91E63",
    "gemini-3-flash-preview": "#00BCD4",
    "gemini-3.1-pro-preview": "#3F51B5",
}
# Per-model markers (keyed by model id so adding a model can't break positional indexing).
MODEL_MARKERS = {
    "claude-opus-4-7": 'o',
    "claude-opus-4-8": 'v',
    "claude-sonnet-4-6": 's',
    "gpt-5.5": 'D',
    "kimi-k2.6": '^',
    "gemini-3-flash-preview": 'P',
    "gemini-3.1-pro-preview": 'X',
}
LEVEL_MARKERS = {1: 'o', 2: 'D', 3: '^'}
LEVEL_COLORS = {1: '#2196F3', 2: '#FF9800', 3: '#F44336'}

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT)

bl_fp32 = json.load(open("results/timing/H200/baseline_time_torch.json"))
bl_tf32 = json.load(open("results/timing/H200/baseline_time_torch_tf32.json"))

OUT_DIR = "docs/figures"
os.makedirs(OUT_DIR, exist_ok=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_baseline_mean(baseline, level, pid):
    bl_level = baseline.get(f"level{level}", {})
    for fname, vals in bl_level.items():
        if vals is None:
            continue
        if int(fname.split('_')[0]) == pid:
            return vals.get('mean')
    return None

def get_baseline_mem(baseline, level, pid):
    bl_level = baseline.get(f"level{level}", {})
    for fname, vals in bl_level.items():
        if vals is None:
            continue
        if int(fname.split('_')[0]) == pid:
            return vals.get('peak_memory')
    return None

def compute_per_problem(model, level, baseline, use_hidden_eval, fp32_tol=1e-3):
    """Return dict of pid -> {speedup, mem_ratio, correct}"""
    run_name = f"{model}_level{level}_test"
    rdir = f"runs/{run_name}"
    # Speed/correctness from the clean timing run (mirrors load_eval_results priority);
    # memory from eval_results.json (the run that carries peak_memory).
    speed_path = None
    for cand in (f"{rdir}/eval_results.json.bak", f"{rdir}/eval_results_pre_memory.json", f"{rdir}/eval_results.json"):
        if os.path.exists(cand):
            speed_path = cand; break
    if speed_path is None:
        return {}
    d = json.load(open(speed_path))
    mem_path = f"{rdir}/eval_results.json"
    mem_d = json.load(open(mem_path)) if os.path.exists(mem_path) else {}
    hd = {}
    if use_hidden_eval:
        hpath = f"{rdir}/eval_results_hidden.json"
        if os.path.exists(hpath):
            hd = json.load(open(hpath))

    excluded = EXCLUDED.get(level, set())
    # Use the FIXED full-benchmark problem list (matches the leaderboard engine's
    # denominator) instead of only the problems present in the eval file.
    NPROBS = {1: 100, 2: 100, 3: 50}
    pids = [p for p in range(1, NPROBS[level] + 1) if p not in excluded]
    results = {}

    for pid in pids:
        bl_mean = get_baseline_mean(baseline, level, pid)
        bl_mem = get_baseline_mem(baseline, level, pid)
        if bl_mean is None or bl_mean <= 0:
            # No baseline -> cannot score; count as incorrect so the denominator
            # stays fixed (consistent with assemble_data / the leaderboard).
            results[pid] = {'speedup': None, 'mem_ratio': None, 'correct': False}
            continue

        best_sp = 0
        best_mem_ratio = 0
        any_correct = False

        for s in range(5):
            key = f"{pid}_{s}"
            e = d.get(key, {})
            if use_hidden_eval and hd:
                hentry = hd.get(key, {})
                correct = hentry.get('correctness', False)
                if not correct:
                    meta = hentry.get('metadata', {})
                    if (not meta.get('hidden_failed_configs')
                        and not meta.get('hidden_runtime_error')
                        and not meta.get('runtime_error')):
                        raw = meta.get('max_difference', [])
                        if raw:
                            try:
                                md = max(float(x) for x in raw if x)
                                if md < fp32_tol:
                                    correct = True
                            except:
                                pass
            else:
                correct = e.get('correctness', False)

            rt = e.get('runtime', -1)
            _me = mem_d.get(key)
            km = _me.get('peak_memory', -1) if isinstance(_me, dict) else -1  # memory from memory-profiling run
            if correct and rt > 0:
                any_correct = True
                sp = bl_mean / rt
                if sp > best_sp:
                    best_sp = sp
                    if bl_mem and bl_mem > 0 and km > 0:
                        best_mem_ratio = bl_mem / km

        results[pid] = {
            'speedup': best_sp if any_correct else None,
            'mem_ratio': best_mem_ratio if any_correct else None,
            'correct': any_correct,
        }
    return results


def compute_aggregate(model, level, baseline, use_hidden_eval, fp32_tol=1e-3):
    pp = compute_per_problem(model, level, baseline, use_hidden_eval, fp32_tol)
    n_total = len(pp)
    correct_pids = [pid for pid, r in pp.items() if r['correct']]
    speedups = [pp[pid]['speedup'] for pid in correct_pids if pp[pid]['speedup'] and pp[pid]['speedup'] > 0]
    mem_ratios = [pp[pid]['mem_ratio'] for pid in correct_pids if pp[pid]['mem_ratio'] and pp[pid]['mem_ratio'] > 0]

    corr = len(correct_pids) / n_total * 100 if n_total > 0 else 0
    geomean_sp = math.exp(sum(math.log(s) for s in speedups) / len(speedups)) if speedups else 0
    geomean_mem = math.exp(sum(math.log(m) for m in mem_ratios) / len(mem_ratios)) if mem_ratios else 0
    return {'corr': corr, 'speedup': geomean_sp, 'mem_ratio': geomean_mem}


# ─── FIGURE 1: Before vs After Speedup (Grouped Bar) ─────────────────────────

def fig1_before_after():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    width = 0.35
    x = np.arange(len(MODELS))

    for i, level in enumerate([1, 2, 3]):
        before_vals = []
        after_vals = []
        for m in MODELS:
            b = compute_aggregate(m, level, bl_fp32, use_hidden_eval=False)
            a = compute_aggregate(m, level, bl_tf32, use_hidden_eval=True)
            before_vals.append(b['speedup'])
            after_vals.append(a['speedup'])

        ax = axes[i]
        bars1 = ax.bar(x - width/2, before_vals, width, label='Before (fp32 baseline,\nstandard eval)',
                       color='#BBDEFB', edgecolor='#1565C0', linewidth=0.8)
        bars2 = ax.bar(x + width/2, after_vals, width, label='After (TF32 baseline,\nhidden eval)',
                       color='#FFCCBC', edgecolor='#BF360C', linewidth=0.8)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_xlabel('')
        ax.set_title(f'Level {level}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS], fontsize=9, rotation=30, ha='right')
        if i == 0:
            ax.set_ylabel('Speedup (×)', fontsize=11)
        ax.set_ylim(0, max(max(before_vals), max(after_vals)) * 1.15)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    plt.savefig(f'{OUT_DIR}/fig1_before_after.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("  ✓ fig1_before_after.pdf")


# ─── FIGURE 2: Per-problem Speedup Distribution (CDF) ────────────────────────

def fig2_speedup_cdf():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    for i, level in enumerate([1, 2, 3]):
        ax = axes[i]
        for m in MODELS:
            pp = compute_per_problem(m, level, bl_tf32, use_hidden_eval=True)
            speedups = sorted([r['speedup'] for r in pp.values() if r['speedup'] and r['speedup'] > 0])
            if speedups:
                cdf = np.arange(1, len(speedups)+1) / len(speedups)
                ax.plot(speedups, cdf, label=MODEL_LABELS[m], color=MODEL_COLORS[m], linewidth=1.5)

        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='1× (no speedup)')
        ax.set_title(f'Level {level}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Speedup (×)', fontsize=12)
        if i == 0:
            ax.set_ylabel('Fraction of problems')
        ax.set_xlim(0, min(3.0, ax.get_xlim()[1]))
        ax.grid(alpha=0.3)
        if i == 2:
            ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig2_speedup_cdf.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("  ✓ fig2_speedup_cdf.pdf")


# ─── FIGURE 3: Memory-Speedup Tradeoff per Level (aggregate dots) ────────────

def fig3_mem_speedup_level():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    # markers come from the global MODEL_MARKERS (keyed by model id)

    for i, level in enumerate([1, 2, 3]):
        ax = axes[i]
        for m in MODELS:
            agg = compute_aggregate(m, level, bl_tf32, use_hidden_eval=True)
            if agg['speedup'] > 0 and agg['mem_ratio'] > 0:
                ax.scatter(agg['speedup'], agg['mem_ratio'], s=180,
                          color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                          label=MODEL_LABELS[m] if i == 0 else None,
                          edgecolors='black', linewidth=0.8, zorder=5)

        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(f'Level {level}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Speedup (×)', fontsize=12)
        if i == 0:
            ax.set_ylabel('Memory Efficiency (×)')
        ax.grid(alpha=0.3)

    # Shared legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=10,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig3_mem_speedup_level.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("  \u2713 fig3_mem_speedup_level.pdf")


# ─── FIGURE 4: Memory-Speedup Tradeoff per Model (scatter) ───────────────────

def fig4_mem_speedup_model():
    fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharey=True)

    for j, m in enumerate(MODELS):
        ax = axes[j // 4, j % 4]
        for level in [1, 2, 3]:
            pp = compute_per_problem(m, level, bl_tf32, use_hidden_eval=True)
            sps = [r['speedup'] for r in pp.values() if r['speedup'] and r['speedup'] > 0]
            mrs = [r['mem_ratio'] for r in pp.values() if r['speedup'] and r['speedup'] > 0 and r['mem_ratio'] and r['mem_ratio'] > 0]
            if sps and mrs:
                ax.scatter(sps[:len(mrs)], mrs, s=35, alpha=0.6,
                          color=LEVEL_COLORS[level], marker=LEVEL_MARKERS[level],
                          label=f'L{level}' if j == 0 else None)

        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(MODEL_LABELS[m], fontsize=10, fontweight='bold')
        ax.set_xlabel('Speedup (×)', fontsize=9)
        if j == 0:
            ax.set_ylabel('Mem Efficiency (×)', fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, min(4.0, ax.get_xlim()[1]))
        ax.set_ylim(0, min(4.0, ax.get_ylim()[1]))

    # Hide any unused subplots (7 models in a 2x4 grid -> 1 empty cell)
    for k in range(len(MODELS), axes.size):
        axes.flat[k].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig4_mem_speedup_model.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("  ✓ fig4_mem_speedup_model.pdf")


# ─── FIGURE 5: Failure Mode Breakdown (stacked bar) ──────────────────────────

def fig5_failure_modes():
    """Non-exclusive failure rates: for each distribution, what % of problems fail it.
    A kernel failing both D2 and D4 is counted in both bars."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    distributions = ['D1 (original)', 'D2 (×3)', 'D3 (×0.01)', 'D4 (negate)']
    config_keys = ['config_1', 'config_2', 'config_3', 'config_4']
    colors_dist = ['#FFC107', '#FF9800', '#FF5722', '#9C27B0']

    for i, level in enumerate([1, 2, 3]):
        ax = axes[i]
        # For each model, compute failure rate per distribution (non-exclusive)
        model_fail_rates = {d: [] for d in distributions}

        for m in MODELS:
            hpath = f"runs/{m}_level{level}_test/eval_results_hidden.json"
            if not os.path.exists(hpath):
                for d in distributions:
                    model_fail_rates[d].append(0)
                continue
            hd = json.load(open(hpath))
            excluded = EXCLUDED.get(level, set())
            total = 0
            dist_fails = {d: 0 for d in distributions}

            for key, e in hd.items():
                pid = int(key.split('_')[0])
                if pid in excluded:
                    continue
                total += 1
                if not e.get('correctness', False):
                    meta = e.get('metadata', {})
                    fc = meta.get('hidden_failed_configs', [])
                    # Non-exclusive: count each distribution independently
                    if meta.get('runtime_error') or meta.get('hidden_runtime_error'):
                        # Runtime errors fail all distributions
                        for d in distributions:
                            dist_fails[d] += 1
                    elif fc:
                        for cfg_key, d in zip(config_keys, distributions):
                            if cfg_key in fc:
                                dist_fails[d] += 1
                    else:
                        # Failed D1 (standard eval failure)
                        dist_fails['D1 (original)'] += 1

            for d in distributions:
                model_fail_rates[d].append(dist_fails[d] / total * 100 if total > 0 else 0)

        # Grouped bar chart
        x = np.arange(len(MODELS))
        width = 0.18
        offsets = [-1.5, -0.5, 0.5, 1.5]
        for j, (d, color) in enumerate(zip(distributions, colors_dist)):
            vals = model_fail_rates[d]
            ax.bar(x + offsets[j] * width, vals, width=width,
                   label=d if i == 0 else None, color=color, edgecolor='white', linewidth=0.5)

        ax.set_title(f'Level {level}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS], fontsize=10)
        ax.set_ylim(0, None)
        if i == 0:
            ax.set_ylabel('% of problems failing')
            ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig5_failure_modes.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("  \u2713 fig5_failure_modes.pdf")


# ─── FIGURE 7: Decomposed Deflation (TF32 vs Hidden vs Both) ─────────────────

def fig7_decomposed():
    """Show individual contributions of TF32 baseline and hidden eval."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    width = 0.2
    x = np.arange(len(MODELS))

    conditions = [
        ('Naïve\n(fp32+std)', bl_fp32, False, '#BBDEFB'),
        ('+ Hidden test suite\n(fp32+hidden)', bl_fp32, True, '#90CAF9'),
        ('+ TF32 baseline\n(TF32+std)', bl_tf32, False, '#FFCCBC'),
        ('Both\n(TF32+hidden)', bl_tf32, True, '#EF9A9A'),
    ]

    for i, level in enumerate([1, 2, 3]):
        ax = axes[i]
        for j, (label, bl, hid, color) in enumerate(conditions):
            vals = []
            for m in MODELS:
                agg = compute_aggregate(m, level, bl, hid)
                vals.append(agg['speedup'])
            offset = (j - 1.5) * width
            ax.bar(x + offset, vals, width, label=label if i == 0 else None,
                   color=color, edgecolor='black', linewidth=0.4)

        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title(f'Level {level}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS], fontsize=9, rotation=30, ha='right')
        if i == 0:
            ax.set_ylabel('Speedup (×)', fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    plt.savefig(f'{OUT_DIR}/fig7_decomposed.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("  ✓ fig7_decomposed.pdf")


# ─── FIGURE 6: Tolerance Sensitivity (grouped bar: 1e-3 vs 1e-4) ─────────────

def fig6_tolerance():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    width = 0.35
    x = np.arange(len(MODELS))

    for i, level in enumerate([1, 2, 3]):
        corr_relaxed = []
        corr_strict = []
        for m in MODELS:
            r3 = compute_aggregate(m, level, bl_tf32, use_hidden_eval=True, fp32_tol=1e-3)
            r4 = compute_aggregate(m, level, bl_tf32, use_hidden_eval=True, fp32_tol=1e-4)
            corr_relaxed.append(r3['corr'])
            corr_strict.append(r4['corr'])

        ax = axes[i]
        ax.bar(x - width/2, corr_relaxed, width, label='τ = 1e-3 (official)',
               color='#C8E6C9', edgecolor='#2E7D32', linewidth=0.8)
        ax.bar(x + width/2, corr_strict, width, label='τ = 1e-4 (strict)',
               color='#FFCDD2', edgecolor='#C62828', linewidth=0.8)
        ax.set_title(f'Level {level}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS], fontsize=10)
        ax.set_ylim(0, 105)
        if i == 0:
            ax.set_ylabel('Correctness (%)', fontsize=11)
            ax.legend(fontsize=8, loc='lower left')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig6_tolerance.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("  ✓ fig6_tolerance.pdf")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures for KernelBench-Verified report...")
    fig1_before_after()
    fig2_speedup_cdf()
    fig3_mem_speedup_level()
    fig4_mem_speedup_model()
    fig5_failure_modes()
    fig6_tolerance()
    fig7_decomposed()
    print(f"\nAll figures saved to {OUT_DIR}/")


def fig8_fp32_vs_bf16():
    """Side-by-side comparison of FP32 (TF32 enabled) vs BF16 speedup."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    models = ['GPT-5.5', 'Gemini Pro', 'Sonnet', 'Opus 4.8', 'Opus 4.7', 'Flash', 'Kimi']
    # Pull verified speedups directly from the leaderboard engine (no hardcoding).
    import sys as _sys; _sys.path.insert(0, "scripts")
    import generate_leaderboard as _glb
    _order = ["gpt-5.5", "gemini-3.1-pro-preview", "claude-sonnet-4-6", "claude-opus-4-8",
              "claude-opus-4-7", "gemini-3-flash-preview", "kimi-k2.6"]
    _fp = _glb.assemble_data(hardware="H200", baseline_name="baseline_time_torch_tf32",
                             run_suffix="_test", use_hidden_eval=True, fp32_tolerance=1e-3)["agg"]
    _bf = _glb.assemble_data(hardware="H200", baseline_name="baseline_time_torch_bf16",
                             run_suffix="_bf16_test", use_hidden_eval=True, fp32_tolerance=1e-2)["agg"]
    _g = lambda agg, m, L: round((agg.get(m, {}).get(L) or {}).get("gmsr") or 0, 2)
    fp32_L1 = [_g(_fp, m, 1) for m in _order]; fp32_L2 = [_g(_fp, m, 2) for m in _order]; fp32_L3 = [_g(_fp, m, 3) for m in _order]
    bf16_L1 = [_g(_bf, m, 1) for m in _order]; bf16_L2 = [_g(_bf, m, 2) for m in _order]; bf16_L3 = [_g(_bf, m, 3) for m in _order]
    
    levels = ['Level 1', 'Level 2', 'Level 3']
    fp32_data = [fp32_L1, fp32_L2, fp32_L3]
    bf16_data = [bf16_L1, bf16_L2, bf16_L3]
    
    x = np.arange(len(models))
    width = 0.35
    
    for i, (ax, level) in enumerate(zip(axes, levels)):
        bars1 = ax.bar(x - width/2, fp32_data[i], width, label='FP32 (TF32 baseline)', 
                       color='#2196F3', alpha=0.8)
        bars2 = ax.bar(x + width/2, bf16_data[i], width, label='BF16 baseline',
                       color='#FF9800', alpha=0.8)
        
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_xlabel('')
        ax.set_ylabel('Correct Speedup (×)' if i == 0 else '')
        ax.set_title(f'{level}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
        ax.set_ylim(0, max(max(fp32_data[i]), max(bf16_data[i])) * 1.15)
        
        # Add value labels
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.2f}',
                   ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.2f}',
                   ha='center', va='bottom', fontsize=7)
        
        if i == 2:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('docs/figures/fig8_fp32_vs_bf16.pdf', bbox_inches='tight')
    plt.close()
    print("Generated fig8_fp32_vs_bf16.pdf")

if __name__ == "__main__":
    fig8_fp32_vs_bf16()
