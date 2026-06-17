# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
"""generate_leaderboard.py

Generates a self-contained HTML leaderboard for KernelBench runs.

Usage:
    cd /path/to/KernelBench
    python scripts/generate_leaderboard.py
    python scripts/generate_leaderboard.py --hardware H200 --baseline baseline_time_torch --out leaderboard.html
"""

import argparse
import html as _html
import json
import os
import re
import sys

import numpy as np

# Allow importing from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.benchmark_eval_analysis import (
    build_baseline_lookup,
    build_baseline_memory_lookup,
    compute_best_of_n_metrics,
    parse_eval_results,
    patch_sample,
)
from src.dataset import construct_kernelbench_dataset


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def discover_runs(runs_dir: str, suffix: str = "_test"):
    """Return list of (model_name, level, run_dir_name) tuples from runs/ directory."""
    escaped = re.escape(suffix)
    pattern = re.compile(r"^(.+)_level(\d+)" + escaped + r"$")
    runs = []
    for name in sorted(os.listdir(runs_dir)):
        match = pattern.match(name)
        if match and os.path.isdir(os.path.join(runs_dir, name)):
            model = match.group(1)
            level = int(match.group(2))
            runs.append((model, level, name))
    return runs


MODEL_DISPLAY = {
    "gpt-5.5": "GPT-5.5 (medium)",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro (high)",
    "gemini-3-flash-preview": "Gemini 3 Flash (high)",
    "claude-opus-4-8": "Claude Opus 4.8 (high)",
    "claude-opus-4-7": "Claude Opus 4.7 (high)",
    "claude-sonnet-4-6": "Claude Sonnet 4.6 (high)",
    "kimi-k2.6": "Kimi K2.6",
    "qwen_qwen3.6-27b": "Qwen3.6-27B",
    "qwen_qwen3.6-27b_nothink": "Qwen3.6-27B (no-think)",
    "hkust_drkernel-14b-coldstart": "DrKernel-14B coldstart",
}


def pretty_model(model: str) -> str:
    """Display name with reasoning effort label for the leaderboard."""
    return MODEL_DISPLAY.get(model, model)


def load_eval_results(runs_dir: str, run_name: str, min_entries: int = None):
    """Load eval results for a run (speedup data).

    Priority:
    1. eval_results.json.bak       — complete golden run, no peak_memory
    2. eval_results_pre_memory.json — complete original before memory profiling started
    3. eval_results.json           — may be partial (in-progress memory re-run)

    If min_entries is set, returns None for in-progress .json files that have
    fewer entries than expected (i.e. the eval is still running).
    """
    base = os.path.join(runs_dir, run_name, "eval_results.json")
    pre = os.path.join(runs_dir, run_name, "eval_results_pre_memory.json")
    for path in (base + ".bak", pre):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    # Live .json — check completeness
    if os.path.exists(base):
        with open(base) as f:
            data = json.load(f)
        if min_entries is not None and len(data) < min_entries:
            return None   # still running — don't use partial results
        return data
    return None


def load_memory_eval_results(runs_dir: str, run_name: str):
    """Load eval_results.json specifically for peak_memory data.

    Always reads .json (not .bak) because the memory profiling job writes
    peak_memory into the live .json file. Returns None if no valid memory
    data is present.
    """
    path = os.path.join(runs_dir, run_name, "eval_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if any(v.get("peak_memory", -1) > 0 for v in data.values()):
        return data
    return None


def get_problem_names(level: int) -> tuple[dict, dict]:
    """Return (names, paths) dicts: problem_id -> display name / absolute path."""
    dataset = construct_kernelbench_dataset(level)
    names = {}
    paths = {}
    for path in dataset:
        fname = os.path.basename(path)   # e.g. "1_Square_matrix_multiplication_.py"
        pid = int(fname.split("_")[0])
        # Strip leading "{pid}_" prefix so the number isn't shown twice
        stem = fname[len(str(pid)) + 1:-3]   # e.g. "Square_matrix_multiplication_"
        display = stem.replace("_", " ").strip()
        names[pid] = display
        paths[pid] = path
    return names, paths


# ---------------------------------------------------------------------------
# Memory metrics (best@k, mirroring speedup logic)
# ---------------------------------------------------------------------------

def compute_best_of_n_memory_metrics(by_sample, all_problem_ids, baseline_memory_lookup, sample_ids) -> dict:
    """
    Best-of-N for memory: for each problem pick the correct sample
    with the highest baseline/kernel memory ratio (higher = better).

    Returns:
      correct_mem_ratio  - geomean of baseline/kernel over correct-only problems (↑)
      avg_mem_ratio      - geomean of min(1, baseline/kernel) over all N problems (↑)
      mem_efficient_pct  - fraction of all N where best correct ratio > 1
    """
    n = len(all_problem_ids)
    best_correct = np.zeros(n, dtype=bool)
    best_ratio   = np.zeros(n)   # baseline/kernel; 1.0 = no benefit, >1 = more efficient

    for i, pid in enumerate(all_problem_ids):
        baseline_mem = baseline_memory_lookup.get(pid)
        if baseline_mem is None or baseline_mem <= 0:
            continue
        for sid in sample_ids:
            entry = by_sample[sid].get(pid)
            if entry is None:
                continue
            if entry.get("correctness") and entry.get("peak_memory", -1) > 0:
                ratio = baseline_mem / entry["peak_memory"]   # higher = more efficient
                if not best_correct[i] or ratio > best_ratio[i]:
                    best_ratio[i]   = ratio
                    best_correct[i] = True

    # Correct Mem Ratio: geomean over correct-only problems
    correct_ratios = [best_ratio[i] for i in range(n) if best_correct[i]]
    correct_mem_ratio = (
        float(np.prod(correct_ratios) ** (1.0 / len(correct_ratios)))
        if correct_ratios else None
    )

    # Avg Mem Ratio: geomean of min(1, ratio) over all N — capped at 1 to avoid
    # over-rewarding; problems with no correct solution contribute 1.0 (neutral)
    per_problem = np.array([
        min(1.0, best_ratio[i]) if best_correct[i] else 1.0
        for i in range(n)
    ])
    avg_mem_ratio = float(np.prod(per_problem) ** (1.0 / n)) if n > 0 else 1.0

    # Mem Efficient %: fraction of all N where best correct ratio > 1
    mem_efficient_pct = sum(
        1 for i in range(n) if best_correct[i] and best_ratio[i] > 1.0
    ) / n

    return {
        "correct_mem_ratio": correct_mem_ratio,
        "avg_mem_ratio":     avg_mem_ratio,
        "mem_efficient_pct": mem_efficient_pct,
    }


# ---------------------------------------------------------------------------
# Per-problem best-speedup extraction
# ---------------------------------------------------------------------------

def compute_per_problem_scatter(
    by_sample_speed, by_sample_mem,
    all_problem_ids, baseline_lookup, baseline_memory_lookup,
    speed_sids, mem_sids, problem_names
) -> list:
    """
    For each problem, find the fastest correct sample (speed best-of-N).
    If that same sample also has valid peak_memory, record:
      speedup   = baseline_time / kernel_time  (higher = better)
      mem_eff   = baseline_mem  / kernel_mem   (higher = better)
    Returns list of dicts: {pid, name, speedup, mem_eff}.
    """
    points = []
    for pid in all_problem_ids:
        baseline = baseline_lookup.get(pid)
        if baseline is None:
            continue
        # Find fastest correct sample in speed data
        best_sp, best_sid = None, None
        for sid in speed_sids:
            e = by_sample_speed[sid].get(pid)
            if e and e.get("correctness") and e.get("runtime", -1) > 0:
                sp = baseline / e["runtime"]
                if best_sp is None or sp > best_sp:
                    best_sp, best_sid = sp, sid
        if best_sid is None:
            continue
        # Look up memory for that same sample in memory data
        baseline_mem = baseline_memory_lookup.get(pid)
        if not baseline_mem or baseline_mem <= 0:
            continue
        mem_e = by_sample_mem.get(best_sid, {}).get(pid)
        if mem_e is None:
            continue
        km = mem_e.get("peak_memory", -1)
        if km <= 0:
            continue
        points.append({
            "pid":     pid,
            "name":    problem_names[pid],
            "speedup": round(best_sp, 4),
            "mem_eff": round(baseline_mem / km, 4),
        })
    return points


def compute_per_problem_speedups(by_sample, all_problem_ids, baseline_lookup, sample_ids) -> dict:
    """
    For each problem, find the best correct speedup across all k samples.
    Returns dict: problem_id (int) -> (speedup, best_sample_id) or (None, None).
    """
    result = {}
    for pid in all_problem_ids:
        baseline = baseline_lookup.get(pid)
        if baseline is None:
            result[pid] = (None, None)
            continue
        best = None
        best_sid = None
        for sid in sample_ids:
            entry = by_sample[sid].get(pid)
            if entry is None:
                continue
            if entry.get("correctness") and entry.get("runtime", -1) > 0:
                speedup = baseline / entry["runtime"]
                if best is None or speedup > best:
                    best = speedup
                    best_sid = sid
        result[pid] = (best, best_sid)
    return result


# ---------------------------------------------------------------------------
# Main data assembly
# ---------------------------------------------------------------------------

def compute_paired_memory_metrics(per_problem_sid, all_problem_ids, mem_by_sample, baseline_memory_lookup) -> dict:
    """
    PAIRED memory: report memory for the SAME sample chosen as the best correct
    speedup (per_problem_sid), not an independently-chosen best-memory sample.

      correct_mem_ratio  - geomean of baseline_mem/kernel_mem over the correct set
                           (problems with a speed-winning sample), using that sample's memory.
      mem_efficient_pct  - fraction of ALL problems where the speed-winning sample
                           uses less memory than baseline (problems with no correct
                           sample, or no valid memory, count as not-efficient).
    """
    n = len(all_problem_ids)
    ratios = []
    efficient = 0
    for pid in all_problem_ids:
        sid = per_problem_sid.get(pid)            # speed-winner sample id, or None
        if sid is None:
            continue                              # no correct speed sample -> excluded / not-efficient
        bm = baseline_memory_lookup.get(pid)
        if not bm or bm <= 0:
            continue
        entry = mem_by_sample.get(sid, {}).get(pid)
        if entry is None:
            continue
        km = entry.get("peak_memory", -1)
        if km <= 0:
            continue
        ratio = bm / km
        ratios.append(ratio)
        if ratio > 1:
            efficient += 1
    correct_mem_ratio = (
        float(np.prod(ratios) ** (1.0 / len(ratios))) if ratios else None
    )
    return {
        "correct_mem_ratio": correct_mem_ratio,
        "mem_efficient_pct": efficient / n if n else 0.0,
        "avg_mem_ratio": None,
    }


def assemble_data(hardware: str, baseline_name: str, run_suffix: str = "_test",
                  use_hidden_eval: bool = False, fp32_tolerance: float = None) -> dict:
    runs_dir = "runs"
    baseline_path = os.path.join("results", "timing", hardware, f"{baseline_name}.json")
    if not os.path.exists(baseline_path):
        print(f"  [warn] Baseline file not found: {baseline_path} — returning None")
        return None

    with open(baseline_path) as f:
        baseline_json = json.load(f)

    discovered = discover_runs(runs_dir, suffix=run_suffix)

    # All unique models seen across any level — include claude, gpt, kimi, gemini-flash, qwen
    _keep = lambda name: (
        name.startswith("claude")
        or name.startswith("gpt")
        or name.startswith("kimi")
        or name.startswith("gemini")
        or name.startswith("qwen")
        or name.startswith("hkust")
    )
    models = sorted(set(model for model, _, _ in discovered if _keep(model)))
    discovered = [(m, l, r) for m, l, r in discovered if _keep(m)]

    # Problem names and IDs per level
    problem_names = {}
    problem_paths = {}
    problem_contents = {}  # "problem|{level}|{pid}" -> file content
    for level in (1, 2, 3):
        names, paths = get_problem_names(level)
        problem_names[level] = names
        problem_paths[level] = paths
        for pid, path in paths.items():
            key = f"problem|{level}|{pid}"
            try:
                problem_contents[key] = open(path).read()
            except Exception:
                pass
    # Problems excluded from metrics/display (degenerate: constant-zero output for any input)
    EXCLUDED_PIDS = {2: {23, 80, 83}}
    all_pids = {
        level: sorted(p for p in problem_names[level].keys()
                      if p not in EXCLUDED_PIDS.get(level, set()))
        for level in (1, 2, 3)
    }

    # agg[model][level] = metrics dict or None
    agg = {model: {1: None, 2: None, 3: None} for model in models}

    # mem_agg[model][level] = memory metrics dict or None
    mem_agg = {model: {1: None, 2: None, 3: None} for model in models}

    # per_problem[model][level] = {pid: speedup or None}; empty dict means not yet run
    per_problem = {model: {1: {}, 2: {}, 3: {}} for model in models}
    # per_problem_sid[model][level] = {pid: best_sample_id or None}
    per_problem_sid = {model: {1: {}, 2: {}, 3: {}} for model in models}

    # scatter_tasks[model][level] = list of {pid, name, speedup, mem_eff} (correct-only, paired)
    scatter_tasks = {model: {1: [], 2: [], 3: []} for model in models}

    p_values = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]

    for model, level, run_name in discovered:
        baseline_lookup = build_baseline_lookup(baseline_json, level)
        pids = all_pids[level]

        eval_results = load_eval_results(runs_dir, run_name,
                                          min_entries=int(len(pids) * 5 * 0.9))
        if eval_results is None:
            n_have = 0
            live = os.path.join(runs_dir, run_name, "eval_results.json")
            if os.path.exists(live):
                import json as _j
                n_have = len(_j.load(open(live)))
            expected = len(pids) * 5
            if n_have:
                print(f"  [partial] {run_name}: {n_have}/{expected} entries — skipping until >90% complete")
            else:
                print(f"  [skip] {run_name}: no eval_results.json yet")
            continue

        # Optionally gate correctness with hidden eval results
        if use_hidden_eval:
            import json as _j
            hidden_path = os.path.join(runs_dir, run_name, "eval_results_hidden.json")
            if os.path.exists(hidden_path):
                hidden_results = _j.load(open(hidden_path))
                # Override correctness for entries where we have hidden eval data
                for key, hentry in hidden_results.items():
                    if key in eval_results:
                        eval_results[key] = dict(eval_results[key])
                        h_correct = hentry.get("correctness", False)
                        # Re-threshold D1-only failures using stored max_difference.
                        # Entries that ONLY failed the standard run (no hidden_failed_configs,
                        # no hidden_runtime_error) are re-evaluated against fp32_tolerance.
                        # Entries failing D2/D3/D4 or with runtime errors remain failed.
                        if (not h_correct
                                and fp32_tolerance is not None
                                and not hentry.get("metadata", {}).get("hidden_failed_configs")
                                and not hentry.get("metadata", {}).get("hidden_runtime_error")
                                and not hentry.get("metadata", {}).get("runtime_error")):
                            raw_diffs = hentry.get("metadata", {}).get("max_difference", [])
                            if raw_diffs:
                                try:
                                    max_diff = max(float(d) for d in raw_diffs if d)
                                    if max_diff < fp32_tolerance:
                                        h_correct = True  # passes under relaxed threshold
                                except (ValueError, TypeError):
                                    pass
                        eval_results[key]["correctness"] = h_correct
                # Detect compiled samples missing from hidden eval and warn explicitly
                missing_keys = []
                for key, entry in eval_results.items():
                    if entry.get("compiled") and key not in hidden_results:
                        eval_results[key] = dict(eval_results[key])
                        eval_results[key]["correctness"] = False
                        missing_keys.append(key)
                if missing_keys:
                    pid_sample_pairs = sorted(
                        (int(k.split("_")[0]), int(k.split("_")[1])) for k in missing_keys
                    )
                    print(f"  [WARN] {run_name}: {len(missing_keys)} compiled samples MISSING from hidden eval → forced correctness=False")
                    print(f"         Missing (pid, sample): {pid_sample_pairs}")
                    print(f"         These may be legitimate kernels that timed out. Re-run hidden eval to recover them.")
                print(f"  [hidden] {run_name}: merged {len(hidden_results)} hidden eval entries")

        by_sample, _, sample_ids = parse_eval_results(eval_results)

        # Patch missing entries so every sample has all problem IDs
        for sid in sample_ids:
            by_sample[sid] = patch_sample(dict(by_sample[sid]), pids)

        # Aggregate best@k metrics
        metrics = compute_best_of_n_metrics(
            by_sample, pids, baseline_lookup, sample_ids, p_values
        )
        agg[model][level] = metrics

        # Per-problem best speedup + which sample achieved it
        pp = compute_per_problem_speedups(by_sample, pids, baseline_lookup, sample_ids)
        per_problem[model][level] = {pid: pp[pid][0] for pid in pids}
        per_problem_sid[model][level] = {pid: pp[pid][1] for pid in pids}

        # Memory metrics — read from .json (has peak_memory from profiling job)
        baseline_memory_lookup = build_baseline_memory_lookup(baseline_json, level)
        mem_results = load_memory_eval_results(runs_dir, run_name)
        if mem_results is not None and baseline_memory_lookup:
            mem_by_sample, _, mem_sample_ids = parse_eval_results(mem_results)
            for sid in mem_sample_ids:
                mem_by_sample[sid] = patch_sample(dict(mem_by_sample[sid]), pids)
            mem_agg[model][level] = compute_paired_memory_metrics(
                per_problem_sid[model][level], pids, mem_by_sample, baseline_memory_lookup
            )
            # Paired scatter: speed best-of-N + memory of same sample
            scatter_tasks[model][level] = compute_per_problem_scatter(
                by_sample, mem_by_sample,
                pids, baseline_lookup, baseline_memory_lookup,
                sample_ids, mem_sample_ids, problem_names[level],
            )

    return {
        "models": models,
        "problem_names": problem_names,
        "all_pids": all_pids,
        "problem_contents": problem_contents,
        "agg": agg,
        "mem_agg": mem_agg,
        "per_problem": per_problem,
        "per_problem_sid": per_problem_sid,
        "run_suffix": run_suffix,
        "scatter_tasks": scatter_tasks,
    }


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def speedup_color(speedup: float):
    """Return (bg_css, text_css) for a given speedup value."""
    if speedup >= 1.0:
        # Green: lightness goes from 50% (speedup=1) down to 25% (speedup≥3)
        t = min((speedup - 1.0) / 2.0, 1.0)
        L = int(50 - t * 25)
        bg = f"hsl(120,60%,{L}%)"
        fg = "#fff" if L < 45 else "#000"
    else:
        # Red: lightness goes from 50% (speedup=1) down to 25% (speedup→0)
        t = 1.0 - speedup
        L = int(50 - t * 25)
        bg = f"hsl(0,60%,{L}%)"
        fg = "#fff" if L < 45 else "#000"
    return bg, fg


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt(val, digits=2):
    return f"{val:.{digits}f}" if val is not None else "—"


def fmt_pct(val, digits=1):
    return f"{val * 100:.{digits}f}%" if val is not None else "—"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def build_agg_tables(models, agg, mem_agg, models_with_data=None, tab_prefix="fp32"):
    """Build aggregate metric tables for all levels.

    models_with_data — set of models that actually have data; others get a Pending row.
                       Defaults to all models.
    tab_prefix       — unique prefix for table IDs (fp32 / bf16) to allow independent sorting.
    """
    if models_with_data is None:
        models_with_data = set(models)

    html = ""
    for level in (1, 2, 3):
        def sort_key(m, _lv=level):
            if m not in models_with_data:
                return -1.0   # pending rows go last (we sort desc, so -1 sinks to bottom)
            met = agg[m][_lv]
            return met["fast_p_1.0"] if met else 0.0

        sorted_models = sorted(models, key=sort_key, reverse=True)
        rows = ""
        for model in sorted_models:
            has_data = model in models_with_data
            met = agg[model][level] if has_data else None
            if met:
                corr_sp  = fmt(met.get("gmsr"))
                corr_pct = fmt_pct(met.get("correctness_rate"))
                avg_sp   = fmt(met.get("avg_speedup"))
                fast1    = fmt_pct(met.get("fast_p_1.0"))
            else:
                corr_sp = corr_pct = avg_sp = fast1 = "—"

            mmet = mem_agg[model][level] if has_data else None
            if mmet:
                corr_mr = fmt(mmet.get("correct_mem_ratio"))
                avg_mr  = fmt(mmet.get("avg_mem_ratio"))
                mem_eff = fmt_pct(mmet.get("mem_efficient_pct"))
            else:
                corr_mr = avg_mr = mem_eff = "—"

            if not has_data:
                name_cell = (
                    f"<td class='model-name' style='color:#aaa'>{pretty_model(model)}"
                    f"&nbsp;<span class='pending-badge'>Pending</span></td>"
                )
                row_style = " style='opacity:0.55'"
            else:
                name_cell = f"<td class='model-name'>{pretty_model(model)}</td>"
                row_style = ""

            rows += (
                f"<tr{row_style}>"
                f"{name_cell}"
                f"<td>{corr_sp}</td>"
                f"<td>{corr_pct}</td>"
                f"<td>{fast1}</td>"
                f"<td class='sep'></td>"
                f"<td>{corr_mr}</td>"
                f"<td>{mem_eff}</td>"
                f"</tr>"
            )

        tbl_id = f"agg-{tab_prefix}-{level}"
        # data-col indices (0=Model unsortable, 1=CorrSp, 2=Corr%, 3=AvgSp, 4=Fast@1,
        #                    5=sep skip, 6=CorrMemRatio, 7=AvgMemRatio, 8=MemEff%)
        def _th(label, col):
            return (f"<th class='sortable-hdr' data-tbl='{tbl_id}' data-col='{col}' "
                    f"onclick=\"sortAgg('{tbl_id}',{col})\">{label} <span class='sort-arrow'></span></th>")

        html += (
            f"<h3>Level {level}</h3>"
            f"<table class='agg-table' id='{tbl_id}'>"
            f"<thead>"
            f"<tr>"
            f"<th rowspan='2'>Model</th>"
            f"<th colspan='3' class='group-hdr sp-hdr'>Speedup (kernel/baseline)</th>"
            f"<th class='sep'></th>"
            f"<th colspan='2' class='group-hdr mem-hdr'>Memory Efficiency (baseline/kernel ↑)</th>"
            f"</tr>"
            f"<tr>"
            f"{_th('Correct Speedup ↑', 1)}"
            f"{_th('Correctness % ↑', 2)}"
            f"{_th('Fast@1 ↑', 3)}"
            f"<th class='sep'></th>"
            f"{_th('Correct Mem Eff ↑', 5)}"
            f"{_th('Mem Efficient % ↑', 6)}"
            f"</tr>"
            f"</thead>"
            f"<tbody>{rows}</tbody>"
            f"</table>"
        )
    return html


def load_hack_flags() -> dict:
    """Load judge_results/hack_flags.json if it exists.

    Returns dict: { "{run_name}|{pid}|{sid}": {"label": ..., "pattern": ...} }
    An empty dict is returned if the file is absent.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "judge_results", "hack_flags.json")
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def build_prob_tabs(models, all_pids, problem_names, per_problem,
                    tab_prefix="fp32", models_with_data=None,
                    per_problem_sid=None, run_suffix="_test",
                    problem_contents=None, hack_flags=None):
    """Build per-problem speedup tab bar + tab content divs.

    problem_contents — global dict {"problem|level|pid": file_content} for modal display
    """
    if models_with_data is None:
        models_with_data = set(models)

    kernels   = {}   # key -> str content (embedded in the HTML for modal display)
    hack_flags = hack_flags or {}

    def kernel_key(model, level, pid):
        return f"{tab_prefix}|{model}|{level}|{pid}"

    def load_kernel(model, level, pid, sid):
        if sid is None:
            return None
        run_dir = f"{model}_level{level}{run_suffix}"
        filename = f"level_{level}_problem_{pid}_sample_{sid}_kernel.py"
        path = os.path.join("runs", run_dir, filename)
        if os.path.exists(path):
            try:
                return open(path).read()
            except Exception:
                return None
        return None

    tab_buttons = ""
    tab_contents = ""

    for level in (1, 2, 3):
        active_btn = " active" if level == 1 else ""
        tab_buttons += (
            f"<button class='tab-btn{active_btn}' "
            f"data-prefix='{tab_prefix}' data-level='{level}' "
            f"onclick=\"switchTab('{tab_prefix}', {level})\">"
            f"Level {level}</button>"
        )

        pids = all_pids[level]
        names = problem_names[level]

        header_cells = "".join(f"<th>{pretty_model(m)}</th>" for m in models)
        header = f"<tr><th>Problem</th>{header_cells}</tr>"

        rows = ""
        for pid in pids:
            cells = ""
            for model in models:
                if model not in models_with_data:
                    cells += "<td class='no-data' style='background:#f5f5f5;color:#ccc'>—</td>"
                    continue
                lvl_data = per_problem[model][level]
                speedup = lvl_data.get(pid) if lvl_data else None
                if speedup is None:
                    cells += "<td class='no-data'>—</td>"
                else:
                    bg, fg = speedup_color(speedup)
                    label = f"{speedup:.2f}\u00d7"
                    sid = (per_problem_sid or {}).get(model, {}).get(level, {}).get(pid)
                    key = kernel_key(model, level, pid)
                    content = load_kernel(model, level, pid, sid)
                    # Check hack flag for this specific sample
                    run_name  = f"{model}_level{level}{run_suffix}"
                    flag_key  = f"{run_name}|{pid}|{sid}"
                    hack_info = hack_flags.get(flag_key)
                    hack_badge = ""
                    if hack_info:
                        hack_label = hack_info.get("label", "HACK")
                        hack_tip   = hack_info.get("pattern", hack_label)
                        short_type = hack_label.replace("_HACK", "").replace("SPEED_AND_MEMORY", "S+M").replace("SPEED", "S").replace("MEMORY", "M")
                        hack_tip_safe = _html.escape(hack_tip, quote=True)
                        hack_badge = (
                            f"<span title=\"[{hack_label}] {hack_tip_safe}\" "
                            f"onclick=\"showJudgeResponse(event,'{flag_key}')\" "
                            f"style='margin-left:3px;font-weight:900;color:#ff4444;"
                            f"font-size:0.9em;cursor:pointer'>&#9888;{short_type}</span>"
                        )
                    if content is not None:
                        kernels[key] = content
                        inner = label + hack_badge
                        cells += (
                            f"<td style='background:{bg};color:{fg};cursor:pointer' "
                            f"onclick=\"showKernel(event,'{key}','{model}',{level},{pid})\" "
                            f"title='Click to view kernel'>{inner}</td>"
                        )
                    else:
                        inner = label + hack_badge
                        cells += f"<td style='background:{bg};color:{fg}'>{inner}</td>"

            name = names[pid]
            prob_key = f"problem|{level}|{pid}"
            if problem_contents and prob_key in problem_contents:
                prob_link = (
                    f"<a href='#' "
                    f"onclick=\"showKernel(event,'{prob_key}','Problem',{level},{pid})\" "
                    f"title='View problem definition' "
                    f"class='prob-link'>{name}</a>"
                )
            else:
                prob_link = name
            rows += (
                f"<tr>"
                f"<td class='prob-name'><span class='pid'>{pid}</span> {prob_link}</td>"
                f"{cells}"
                f"</tr>"
            )

        display = "block" if level == 1 else "none"
        tab_contents += (
            f"<div id='tab-{tab_prefix}-{level}' class='tab-content' style='display:{display}'>"
            f"<div class='table-scroll'>"
            f"<table class='prob-table'>"
            f"<thead>{header}</thead>"
            f"<tbody>{rows}</tbody>"
            f"</table></div></div>"
        )

    return tab_buttons, tab_contents, kernels


def build_rank_tabs(models, all_pids, problem_names, per_problem,
                    tab_prefix="fp32", models_with_data=None,
                    per_problem_sid=None, run_suffix="_test",
                    kernels=None, problem_contents=None):
    """Build per-problem ranking tabs: rows=problems, cols=rank 1..N by speedup.

    kernels — the existing kernels dict from build_prob_tabs (reused for modal links).
    """
    if models_with_data is None:
        models_with_data = set(models)
    if kernels is None:
        kernels = {}
    active_models = [m for m in models if m in models_with_data]
    n_models = len(active_models)

    def kernel_key(model, level, pid):
        return f"{tab_prefix}|{model}|{level}|{pid}"

    tab_buttons = ""
    tab_contents = ""

    for level in (1, 2, 3):
        active_btn = " active" if level == 1 else ""
        tab_buttons += (
            f"<button class='tab-btn{active_btn}' "
            f"data-prefix='rank-{tab_prefix}' data-level='{level}' "
            f"onclick=\"switchTab('rank-{tab_prefix}', {level})\">"
            f"Level {level}</button>"
        )

        pids = all_pids[level]
        names = problem_names[level]

        rank_headers = "".join(
            f"<th>Rank {r+1}</th>" for r in range(n_models)
        )
        header = f"<tr><th>Problem</th>{rank_headers}</tr>"

        rows = ""
        for pid in pids:
            # Collect (speedup, model) for all models with data, sort desc
            entries = []
            for model in active_models:
                lvl_data = per_problem[model][level]
                speedup = lvl_data.get(pid) if lvl_data else None
                entries.append((speedup, model))
            entries.sort(key=lambda x: (x[0] is not None, x[0] or 0), reverse=True)

            cells = ""
            for speedup, model in entries:
                if speedup is None:
                    cells += "<td class='no-data'>—</td>"
                else:
                    bg, fg = speedup_color(speedup)
                    label = f"{speedup:.2f}×"
                    short_model = model.replace("claude-", "").replace("gpt-", "gpt-")
                    key = kernel_key(model, level, pid)
                    inner = f"<b>{label}</b><br><small>{short_model}</small>"
                    if key in kernels:
                        cells += (
                            f"<td style='background:{bg};color:{fg};cursor:pointer' "
                            f"onclick=\"showKernel(event,'{key}','{model}',{level},{pid})\" "
                            f"title='Click to view kernel'>{inner}</td>"
                        )
                    else:
                        cells += f"<td style='background:{bg};color:{fg}'>{inner}</td>"

            name = names[pid]
            prob_key = f"problem|{level}|{pid}"
            if problem_contents and prob_key in problem_contents:
                prob_link = (
                    f"<a href='#' "
                    f"onclick=\"showKernel(event,'{prob_key}','Problem',{level},{pid})\" "
                    f"title='View problem definition' "
                    f"class='prob-link'>{name}</a>"
                )
            else:
                prob_link = name
            rows += (
                f"<tr>"
                f"<td class='prob-name'><span class='pid'>{pid}</span> {prob_link}</td>"
                f"{cells}"
                f"</tr>"
            )

        display = "block" if level == 1 else "none"
        tab_contents += (
            f"<div id='tab-rank-{tab_prefix}-{level}' class='tab-content' style='display:{display}'>"
            f"<div class='table-scroll'>"
            f"<table class='prob-table'>"
            f"<thead>{header}</thead>"
            f"<tbody>{rows}</tbody>"
            f"</table></div></div>"
        )

    return tab_buttons, tab_contents


# ---------------------------------------------------------------------------
# Scatter plot SVG builder
# ---------------------------------------------------------------------------

_LEVEL_COLORS = {1: "#4e79a7", 2: "#f28e2b", 3: "#e15759"}
_LEVEL_SHAPES = {1: "circle", 2: "diamond", 3: "triangle"}
_MODEL_PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1"]


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list:
    """Return ~n nicely-rounded tick values covering [lo, hi]."""
    if lo >= hi:
        mid = (lo + hi) / 2
        lo, hi = mid - 0.5, mid + 0.5
    span = hi - lo
    raw_step = span / n
    mag = 10 ** int(np.floor(np.log10(raw_step))) if raw_step > 0 else 1.0
    step = max(round(raw_step / mag) * mag, mag * 0.1)
    start = float(np.ceil(lo / step) * step)
    ticks = []
    v = start
    while v <= hi + 1e-9:
        ticks.append(round(v, 6))
        v += step
    return ticks


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_scatter_svg(
    points,
    xlabel="Speedup (×)",
    ylabel="Mem Efficiency (×)",
    ref_x=1.0,
    ref_y=1.0,
    width=400,
    height=340,
) -> str:
    """Return a self-contained SVG string for a scatter plot.

    Each point dict: {x, y, color, shape, title, label (optional)}.
    Shapes: 'circle' | 'diamond' | 'triangle' | 'square'.
    Hover tooltip via native SVG <title>.
    """
    ml, mr, mt, mb = 62, 24, 28, 52  # margins: left, right, top, bottom
    pw = width - ml - mr
    ph = height - mt - mb

    if not points:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="{width//2}" y="{height//2}" text-anchor="middle" fill="#999" font-size="13">No data</text>'
            f"</svg>"
        )

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]

    def clipped_range(vals, ref, pct=95):
        """Axis range clipped to pct-th percentile to suppress outlier stretching."""
        lo = min(vals)
        hi = float(np.percentile(vals, pct))
        if ref is not None:
            lo = min(lo, ref)
            hi = max(hi, ref)
        span = hi - lo or 1.0
        return lo - span * 0.05, hi + span * 0.22

    xmin, xmax = clipped_range(xs, ref_x)
    ymin, ymax = clipped_range(ys, ref_y)

    def px(x):
        # Clamp to plot boundaries so outliers land on the edge
        t = (x - xmin) / (xmax - xmin)
        return ml + max(0.0, min(1.0, t)) * pw

    def py(y):
        t = (y - ymin) / (ymax - ymin)
        return mt + ph - max(0.0, min(1.0, t)) * ph

    def is_clipped(x, y):
        return x > xmax or x < xmin or y > ymax or y < ymin

    elems = []

    # Background
    elems.append(
        f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#ddd" stroke-width="1"/>'
    )

    # Grid + ticks
    xticks = _nice_ticks(xmin, xmax)
    yticks = _nice_ticks(ymin, ymax)

    for xv in xticks:
        x = px(xv)
        elems.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" stroke="#ebebeb" stroke-width="1"/>')
        elems.append(f'<line x1="{x:.1f}" y1="{mt+ph}" x2="{x:.1f}" y2="{mt+ph+5}" stroke="#555" stroke-width="1"/>')
        elems.append(f'<text x="{x:.1f}" y="{mt+ph+18}" text-anchor="middle" font-size="10" fill="#555">{xv:.2f}</text>')

    for yv in yticks:
        y = py(yv)
        elems.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#ebebeb" stroke-width="1"/>')
        elems.append(f'<line x1="{ml-5}" y1="{y:.1f}" x2="{ml}" y2="{y:.1f}" stroke="#555" stroke-width="1"/>')
        elems.append(f'<text x="{ml-8}" y="{y:.1f}" text-anchor="end" dominant-baseline="middle" font-size="10" fill="#555">{yv:.2f}</text>')

    # Axes
    elems.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#444" stroke-width="1.5"/>')
    elems.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#444" stroke-width="1.5"/>')

    # Reference lines
    if ref_x is not None and xmin < ref_x < xmax:
        x = px(ref_x)
        elems.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" stroke="#999" stroke-width="1.5" stroke-dasharray="6,3"/>')
    if ref_y is not None and ymin < ref_y < ymax:
        y = py(ref_y)
        elems.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#999" stroke-width="1.5" stroke-dasharray="6,3"/>')

    # Axis labels
    cx = ml + pw / 2
    elems.append(f'<text x="{cx:.1f}" y="{height - 6}" text-anchor="middle" font-size="11" fill="#333">{_esc(xlabel)}</text>')
    cy = mt + ph / 2
    elems.append(f'<text x="13" y="{cy:.1f}" text-anchor="middle" font-size="11" fill="#333" transform="rotate(-90,13,{cy:.1f})">{_esc(ylabel)}</text>')

    # Data points
    for p in points:
        x, y = px(p["x"]), py(p["y"])
        color = p.get("color", "#4e79a7")
        shape = p.get("shape", "circle")
        title = _esc(p.get("title", ""))
        label = _esc(p.get("label", ""))
        r = 5
        clipped = is_clipped(p["x"], p["y"])
        if clipped:
            continue  # hide out-of-range points
        stroke = color
        sw = "0.5"

        if shape == "diamond":
            d = r * 1.5
            pts_str = f"{x:.1f},{y-d:.1f} {x+d:.1f},{y:.1f} {x:.1f},{y+d:.1f} {x-d:.1f},{y:.1f}"
            elems.append(f'<polygon points="{pts_str}" fill="{color}" opacity="0.82" stroke="{stroke}" stroke-width="{sw}"><title>{title}</title></polygon>')
        elif shape == "triangle":
            d = r * 1.4
            pts_str = f"{x:.1f},{y-d:.1f} {x+d:.1f},{y+d*0.8:.1f} {x-d:.1f},{y+d*0.8:.1f}"
            elems.append(f'<polygon points="{pts_str}" fill="{color}" opacity="0.82" stroke="{stroke}" stroke-width="{sw}"><title>{title}</title></polygon>')
        else:  # circle (default)
            elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" opacity="0.82" stroke="{stroke}" stroke-width="{sw}"><title>{title}</title></circle>')

        if label:
            lpos = p.get("label_pos", "right")
            if lpos == "above":
                elems.append(f'<text x="{x:.1f}" y="{y - r - 4:.1f}" text-anchor="middle" font-size="9" fill="#333">{label}</text>')
            elif lpos == "below":
                elems.append(f'<text x="{x:.1f}" y="{y + r + 11:.1f}" text-anchor="middle" font-size="9" fill="#333">{label}</text>')
            else:
                elems.append(f'<text x="{x + r + 3:.1f}" y="{y:.1f}" dominant-baseline="middle" font-size="9" fill="#333">{label}</text>')

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="overflow:visible">\n'
        + "\n".join(elems)
        + "\n</svg>"
    )


def build_scatter_section(models, scatter_tasks, agg, mem_agg) -> str:
    """Build two scatter-plot sections: per-model and per-level."""

    def _legend_shape(level: int, color: str) -> str:
        """Return a 14×14 SVG element matching the plot marker for this level."""
        shape = _LEVEL_SHAPES[level]
        return _legend_shape_by(shape, color)

    def _legend_shape_by(shape: str, color: str) -> str:
        """Return a 14×14 SVG element for the given shape name."""
        if shape == "diamond":
            return f'<polygon points="7,1 13,7 7,13 1,7" fill="{color}"/>'
        elif shape == "triangle":
            return f'<polygon points="7,1 13,13 1,13" fill="{color}"/>'
        elif shape == "square":
            return f'<rect x="2" y="2" width="10" height="10" fill="{color}"/>'
        else:  # circle
            return f'<circle cx="7" cy="7" r="5" fill="{color}"/>'

    model_colors = (_MODEL_PALETTE * 4)[: len(models)]

    # ---- Per-model scatter (task level) ----
    per_model_svgs = ""
    for model in models:
        pts = []
        for level in (1, 2, 3):
            for item in scatter_tasks[model][level]:
                pts.append({
                    "x": item["speedup"],
                    "y": item["mem_eff"],
                    "color": _LEVEL_COLORS[level],
                    "shape": _LEVEL_SHAPES[level],
                    "title": (
                        f"L{level} #{item['pid']}: {item['name']}\n"
                        f"Speedup: {item['speedup']:.2f}\u00d7  "
                        f"Mem eff: {item['mem_eff']:.2f}\u00d7"
                    ),
                })
        svg = build_scatter_svg(pts, xlabel="Speedup (×)", ylabel="Mem Efficiency (×)")
        per_model_svgs += (
            f'<div style="flex:1;min-width:360px">'
            f'<h3 style="margin-bottom:6px">{pretty_model(model)}</h3>'
            f'{svg}'
            f'</div>\n'
        )

    level_legend = "".join(
        f'<span><svg width="14" height="14" style="vertical-align:middle">{_legend_shape(l, _LEVEL_COLORS[l])}</svg>&nbsp;Level&nbsp;{l}</span>'
        for l in (1, 2, 3)
    )

    # Shape palette for models (per-level plot)
    _model_shapes = ["circle", "diamond", "triangle", "square"]
    model_shapes = (_model_shapes * 4)[: len(models)]

    # ---- Per-level tradeoff (model level) ----
    per_level_svgs = ""
    for level in (1, 2, 3):
        pts = []
        for i, model in enumerate(models):
            a = agg[model][level]
            ma = mem_agg[model][level]
            if a is None or ma is None:
                continue
            gmsr = a.get("gmsr")
            cmr = ma.get("correct_mem_ratio")
            if gmsr is None or cmr is None or cmr <= 0:
                continue
            label_pos = "right"
            if model == "gemini-3-flash-preview":
                label_pos = "above" if level == 2 else "below" if level == 3 else "right"
            pts.append({
                "x": gmsr,
                "y": cmr,   # already baseline/kernel; higher = more efficient
                "color": model_colors[i],
                "shape": model_shapes[i],
                "label": pretty_model(model),
                "label_pos": label_pos,
                "title": (
                    f"{pretty_model(model)}  Level {level}\n"
                    f"Correct Speedup: {gmsr:.2f}\u00d7  "
                    f"Mem Eff: {cmr:.2f}\u00d7"
                ),
            })
        svg = build_scatter_svg(pts, xlabel="Correct Speedup (×, higher=better)", ylabel="Memory Efficiency (baseline/kernel mem, higher=better)")
        per_level_svgs += (
            f'<div style="flex:1;min-width:360px">'
            f'<h3 style="margin-bottom:6px">Level {level}</h3>'
            f'{svg}'
            f'</div>\n'
        )

    model_legend = "".join(
        f'<span><svg width="14" height="14" style="vertical-align:middle">{_legend_shape_by(model_shapes[i], model_colors[i])}</svg>&nbsp;{pretty_model(m)}</span>'
        for i, m in enumerate(models)
    )

    return f"""
<h2>Memory–Speedup Tradeoff (per level)</h2>
<section>
  <p class="metric-note">
    Each dot = one model. &nbsp;
    <b>X</b> = Correct Speedup — geomean of (baseline / kernel runtime) over correct problems only (higher = faster). &nbsp;
    <b>Y</b> = Memory Efficiency — geomean of (baseline mem / kernel mem) over correct problems only (higher = model uses less GPU memory than baseline).
    Upper-right corner is best: fast <em>and</em> memory-efficient. Dashed lines mark the 1× reference (no change vs baseline).
  </p>
  <div class="legend" style="margin-bottom:12px">{model_legend}</div>
  <div style="display:flex;gap:28px;flex-wrap:wrap">{per_level_svgs}</div>
</section>

<h2>Memory–Speedup Tradeoff (per model)</h2>
<section>
  <p class="metric-note">
    Each dot = one problem (correct-only). &nbsp;
    <b>X</b> = best correct speedup (best@k). &nbsp;
    <b>Y</b> = memory efficiency (baseline mem / kernel mem) of that same fastest-correct sample
    — higher is better on both axes. Dashed lines mark the 1× reference.
  </p>
  <div class="legend" style="margin-bottom:12px">{level_legend}</div>
  <div style="display:flex;gap:28px;flex-wrap:wrap">{per_model_svgs}</div>
</section>
"""


def build_html(fp32_data: dict, bf16_data: dict = None, hack_flags: dict = None) -> str:
    """Build the full leaderboard HTML.

    fp32_data — required; the FP32 assembled dataset.
    bf16_data — optional; the BF16 assembled dataset (may be None or partial).
    """
    hack_flags = hack_flags or {}
    # FP32 is always the canonical model list / problem list
    models        = fp32_data["models"]
    problem_names = fp32_data["problem_names"]
    all_pids      = fp32_data["all_pids"]

    # --- FP32 sections ---
    fp32_agg_html             = build_agg_tables(models, fp32_data["agg"], fp32_data["mem_agg"],
                                                  tab_prefix="fp32")
    fp32_tab_btns, fp32_tabs, fp32_kernels = build_prob_tabs(
        models, all_pids, problem_names,
        fp32_data["per_problem"], tab_prefix="fp32",
        per_problem_sid=fp32_data["per_problem_sid"],
        run_suffix=fp32_data["run_suffix"],
        problem_contents=fp32_data["problem_contents"],
        hack_flags=hack_flags)
    fp32_rank_btns, fp32_rank_tabs = build_rank_tabs(
        models, all_pids, problem_names,
        fp32_data["per_problem"], tab_prefix="fp32",
        per_problem_sid=fp32_data["per_problem_sid"],
        run_suffix=fp32_data["run_suffix"],
        kernels=fp32_kernels,
        problem_contents=fp32_data["problem_contents"])
    fp32_scatter              = build_scatter_section(models, fp32_data["scatter_tasks"],
                                                      fp32_data["agg"], fp32_data["mem_agg"])

    # --- BF16 sections ---
    has_bf16 = bf16_data is not None and len(bf16_data.get("models", [])) > 0
    if has_bf16:
        # Models that actually have BF16 eval results (non-None agg for at least one level)
        bf16_models_with_data = {
            m for m in bf16_data["models"]
            if any(bf16_data["agg"][m][lv] is not None for lv in (1, 2, 3))
        }
        # Show all FP32 models in BF16 table; add any BF16-only models at the end
        all_display_models = list(models)
        for m in bf16_data["models"]:
            if m not in set(models):
                all_display_models.append(m)

        # Extend agg/mem_agg/per_problem dicts to cover display models
        bf16_agg = {m: {1: None, 2: None, 3: None} for m in all_display_models}
        bf16_mem = {m: {1: None, 2: None, 3: None} for m in all_display_models}
        bf16_pp  = {m: {1: {}, 2: {}, 3: {}} for m in all_display_models}
        bf16_sid = {m: {1: {}, 2: {}, 3: {}} for m in all_display_models}
        for m in bf16_data["models"]:
            bf16_agg[m] = bf16_data["agg"][m]
            bf16_mem[m] = bf16_data["mem_agg"][m]
            bf16_pp[m]  = bf16_data["per_problem"][m]
            bf16_sid[m] = bf16_data["per_problem_sid"][m]

        bf16_agg_html            = build_agg_tables(all_display_models, bf16_agg, bf16_mem,
                                                    models_with_data=bf16_models_with_data,
                                                    tab_prefix="bf16")
        bf16_tab_btns, bf16_tabs, bf16_kernels = build_prob_tabs(
            all_display_models, all_pids, problem_names,
            bf16_pp, tab_prefix="bf16",
            models_with_data=bf16_models_with_data,
            per_problem_sid=bf16_sid,
            run_suffix=bf16_data["run_suffix"],
            problem_contents=fp32_data["problem_contents"],
            hack_flags=hack_flags)
        bf16_rank_btns, bf16_rank_tabs = build_rank_tabs(
            all_display_models, all_pids, problem_names,
            bf16_pp, tab_prefix="bf16",
            models_with_data=bf16_models_with_data,
            per_problem_sid=bf16_sid,
            run_suffix=bf16_data["run_suffix"],
            kernels=bf16_kernels,
            problem_contents=fp32_data["problem_contents"])
        bf16_scatter             = build_scatter_section(
            sorted(bf16_models_with_data), bf16_data["scatter_tasks"],
            bf16_data["agg"], bf16_data["mem_agg"]
        )
        bf16_note = (
            f"<p class='metric-note' style='margin-bottom:10px'>"
            f"<b>BF16 coverage:</b> {len(bf16_models_with_data)}/{len(all_display_models)} models evaluated. "
            f"Grayed rows are pending. "
            f"Baseline: PyTorch BF16 reference on H200."
            f"</p>"
        )
    else:
        bf16_kernels  = {}
        bf16_agg_html = "<p style='color:#888;padding:20px 0'>No BF16 results available yet.</p>"
        bf16_tab_btns = bf16_tabs = ""
        bf16_rank_btns = bf16_rank_tabs = ""
        bf16_scatter  = "<p style='color:#888;padding:20px 0'>No BF16 scatter data available yet.</p>"
        bf16_note     = ""

    # Merge all kernel file contents + hack flags for the inline viewers
    # problem_contents is shared (same problems for FP32 and BF16)
    all_kernels = {
        **fp32_data["problem_contents"],
        **fp32_kernels,
        **(bf16_kernels if has_bf16 else {})
    }
    kernels_json    = json.dumps(all_kernels)
    hack_flags_json = json.dumps(hack_flags)

    # Precision toggle visibility: FP32 shown by default
    prec_toggle = """
<div class="prec-toggle" id="prec-toggle">
  <button class="prec-btn active" id="btn-fp32" onclick="setPrecision('fp32')">FP32</button>
  <button class="prec-btn" id="btn-bf16" onclick="setPrecision('bf16')">BF16</button>
</div>""" if has_bf16 else ""

    metric_note_html = """
    <b>Correct Speedup ↑</b> = geomean of baseline/kernel over correct problems only
    &nbsp;&nbsp;|&nbsp;&nbsp;
    <b>Correctness % ↑</b> = fraction of problems solved correctly (best@k)
    &nbsp;&nbsp;|&nbsp;&nbsp;

    <b>Fast@1 ↑</b> = fraction of problems where best correct speedup &gt; 1× &nbsp;<em>(default sort — click any header to re-sort)</em>
    <br>
    <b>Correct Mem Eff ↑</b> = geomean of baseline/kernel memory over correct problems only (higher = uses less memory)
    &nbsp;&nbsp;|&nbsp;&nbsp;

    <b>Mem Efficient % ↑</b> = fraction of problems where best correct kernel uses less memory than baseline"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KernelBench-Verified Leaderboard</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f5f5; color: #222; }}
h1 {{ padding: 24px 32px 8px; font-size: 1.8rem; }}
.subtitle {{ padding: 0 32px 20px; color: #555; font-size: 0.95rem; }}
h2 {{ padding: 16px 32px 8px; font-size: 1.3rem; border-top: 2px solid #ddd; margin-top: 16px; }}
h3 {{ padding: 12px 0 6px; font-size: 1.05rem; color: #444; }}
section {{ padding: 0 32px 32px; }}

/* Sortable header */
.sortable-hdr {{
  cursor: pointer; user-select: none; white-space: nowrap;
}}
.sortable-hdr:hover {{ background: #3d5068; }}
.sort-arrow {{ font-size: 0.75em; margin-left: 3px; opacity: 0.7; }}

/* View toggle (By Model / By Rank) */
.view-toggle {{
  display: inline-flex; gap: 0; border: 1.5px solid #bbb;
  border-radius: 6px; overflow: hidden;
}}
.view-btn {{
  padding: 5px 16px; border: none; background: #f0f0f0;
  cursor: pointer; font-size: 0.85rem; font-weight: 600;
  color: #555; transition: background 0.15s, color 0.15s;
}}
.view-btn.active {{ background: #2c3e50; color: #fff; }}
.view-btn:hover:not(.active) {{ background: #dde; }}

/* Precision toggle */
.prec-toggle {{
  display: inline-flex; gap: 0; margin-left: 18px; vertical-align: middle;
  border: 1.5px solid #bbb; border-radius: 6px; overflow: hidden;
}}
.prec-btn {{
  padding: 5px 18px; border: none; background: #f0f0f0;
  cursor: pointer; font-size: 0.88rem; font-weight: 600;
  color: #555; transition: background 0.15s, color 0.15s;
}}
.prec-btn.active {{ background: #2c3e50; color: #fff; }}
.prec-btn:hover:not(.active) {{ background: #dde; }}

/* Pending badge */
.pending-badge {{
  display: inline-block; font-size: 0.7rem; font-weight: 700;
  background: #f0ad4e; color: #fff;
  border-radius: 3px; padding: 1px 5px; margin-left: 6px;
  vertical-align: middle; letter-spacing: 0.03em;
}}

/* Aggregate table */
.agg-table {{
  border-collapse: collapse; width: 100%; margin-bottom: 24px;
  background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.1);
  border-radius: 6px; overflow: hidden;
}}
.agg-table th {{
  background: #2c3e50; color: #fff;
  padding: 10px 14px; text-align: left; font-size: 0.88rem;
}}
.agg-table th.group-hdr {{ text-align: center; border-bottom: 1px solid #4a6278; }}
.agg-table th.sp-hdr  {{ background: #2c3e50; }}
.agg-table th.mem-hdr {{ background: #2d6a4f; }}
.agg-table th.sep, .agg-table td.sep {{ width: 12px; background: #f0f0f0; padding: 0; border: none; }}
.agg-table td {{ padding: 8px 14px; border-bottom: 1px solid #eee; font-size: 0.88rem; }}
.agg-table tr:last-child td {{ border-bottom: none; }}
.agg-table tr:hover td {{ background: #f0f4ff; }}
.agg-table tr:hover td.sep {{ background: #e8e8e8; }}
.model-name {{ font-weight: 600; }}

/* Level tabs */
.tab-bar {{ display: flex; gap: 6px; margin-bottom: 12px; flex-wrap: wrap; }}
.tab-btn {{
  padding: 7px 18px; border: 1.5px solid #bbb;
  border-radius: 4px 4px 0 0; background: #eee;
  cursor: pointer; font-size: 0.9rem; transition: background 0.15s;
}}
.tab-btn.active {{ background: #2c3e50; color: #fff; border-color: #2c3e50; }}
.tab-btn:hover:not(.active) {{ background: #dde; }}

/* Problem table */
.table-scroll {{ overflow-x: auto; }}
.prob-table {{
  border-collapse: collapse; background: #fff; font-size: 0.82rem;
  box-shadow: 0 1px 4px rgba(0,0,0,.1); min-width: 100%;
}}
.prob-table th {{
  background: #2c3e50; color: #fff;
  padding: 8px 10px; text-align: center; white-space: nowrap; font-weight: 500;
}}
.prob-table th:first-child {{ text-align: left; min-width: 240px; }}
.prob-table td {{
  padding: 6px 10px; border-bottom: 1px solid #eee;
  text-align: center; white-space: nowrap;
}}
.prob-table tr:last-child td {{ border-bottom: none; }}
.prob-name {{ text-align: left !important; max-width: 320px; white-space: normal; line-height: 1.3; }}
.pid {{ display: inline-block; min-width: 26px; font-weight: 700; color: #555; margin-right: 4px; }}
.no-data {{ color: #bbb; }}

/* Legend */
.legend {{
  display: flex; gap: 20px; margin-bottom: 12px;
  font-size: 0.82rem; align-items: center; flex-wrap: wrap;
}}
.leg-swatch {{
  display: inline-block; width: 16px; height: 16px;
  border-radius: 3px; vertical-align: middle; margin-right: 4px;
}}

/* Metric note */
.metric-note {{ font-size: 0.83rem; color: #666; margin-bottom: 14px; line-height: 1.6; }}

/* Precision panels */
.prec-panel {{ display: block; }}
.prec-panel.hidden {{ display: none; }}
/* Problem name link */
.prob-link {{
  color: inherit; text-decoration: none; border-bottom: 1px dotted #aaa;
}}
.prob-link:hover {{ border-bottom-color: #2c3e50; color: #2c3e50; }}

/* Code viewer modal */
.modal-overlay {{
  display: none; position: fixed; inset: 0;
  background: rgba(0,0,0,0.55); z-index: 1000;
  align-items: flex-start; justify-content: center;
  padding-top: 40px;
}}
.modal-overlay.open {{ display: flex; }}
.modal-box {{
  background: #1e1e2e; color: #cdd6f4;
  border-radius: 8px; width: 90%; max-width: 900px;
  max-height: 80vh; display: flex; flex-direction: column;
  box-shadow: 0 8px 32px rgba(0,0,0,.5);
}}
.modal-header {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 18px; border-bottom: 1px solid #313244;
  font-size: 0.9rem; font-weight: 600; flex-shrink: 0;
}}
.modal-header .modal-title {{ color: #cba6f7; }}
.modal-header .modal-subtitle {{ color: #888; font-weight: 400; margin-left: 10px; font-size: 0.82rem; }}
.modal-actions {{ display: flex; gap: 8px; align-items: center; }}
.modal-btn {{
  padding: 4px 12px; border: 1px solid #45475a; border-radius: 4px;
  background: #313244; color: #cdd6f4; cursor: pointer; font-size: 0.82rem;
  transition: background 0.15s;
}}
.modal-btn:hover {{ background: #45475a; }}
.modal-code {{
  overflow: auto; padding: 18px 20px; flex: 1;
  font-family: 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
  font-size: 0.82rem; line-height: 1.55; white-space: pre;
  tab-size: 4;
}}

/* Code viewer modal */
</style>
</head>
<body>

<h1>KernelBench-Verified Leaderboard
  <span id="prec-label" style="font-size:0.6em;font-weight:500;color:#888;vertical-align:middle">(FP32)</span>
  {prec_toggle}
</h1>
<p class="subtitle">Best-of-5 across all samples &nbsp;&middot;&nbsp; Baseline: PyTorch reference on H200 &nbsp;&middot;&nbsp; All models evaluated in May 2026</p>

<!-- ===================== SCATTER ===================== -->
<div id="panel-fp32-scatter" class="prec-panel">
{fp32_scatter}
</div>
<div id="panel-bf16-scatter" class="prec-panel hidden">
{bf16_scatter}
</div>

<!-- ===================== AGGREGATE ===================== -->
<h2>Model Aggregate Metrics</h2>
<section>
  <p class="metric-note">{metric_note_html}</p>

  <!-- FP32 aggregate -->
  <div id="panel-fp32-agg" class="prec-panel">
    {fp32_agg_html}
  </div>
  <!-- BF16 aggregate -->
  <div id="panel-bf16-agg" class="prec-panel hidden">
    {bf16_note}
    {bf16_agg_html}
  </div>
</section>

<!-- ===================== PER-PROBLEM ===================== -->
<section>
<h2>Per-Problem Speedup</h2>
<p class='metric-note' style='margin-bottom:12px'>
  Best-of-5 speedup over PyTorch baseline on H200.
  <span style='margin-left:16px'>💡 <b>Click a problem name</b> to view its definition &nbsp;·&nbsp; <b>click a speedup number</b> to view the best kernel generated by that model.</span>
</p>
  <div class="legend">
    <span><span class="leg-swatch" style="background:hsl(120,60%,25%)"></span> Fast (&ge;3&times;)</span>
    <span><span class="leg-swatch" style="background:hsl(120,60%,43%)"></span> Beats baseline (&gt;1&times;)</span>
    <span><span class="leg-swatch" style="background:hsl(0,60%,43%)"></span> Correct but slower (&lt;1&times;)</span>
    <span><span class="leg-swatch" style="background:#ddd;border:1px solid #bbb"></span> No correct solution</span>
    {'<span><span class="leg-swatch" style="background:#f5f5f5;border:1px solid #ddd"></span> Pending (BF16)</span>' if has_bf16 else ''}
  </div>

  <!-- FP32 panel -->
  <div id="panel-fp32-prob" class="prec-panel">
    <div class="view-toggle" style="margin-bottom:10px">
      <button class="view-btn active" id="fp32-view-model" onclick="setView('fp32','model')">By Model</button>
      <button class="view-btn" id="fp32-view-rank" onclick="setView('fp32','rank')">By Rank</button>
    </div>
    <div id="fp32-model-view">
      <div class="tab-bar">{fp32_tab_btns}</div>
      {fp32_tabs}
    </div>
    <div id="fp32-rank-view" style="display:none">
      <div class="tab-bar">{fp32_rank_btns}</div>
      {fp32_rank_tabs}
    </div>
  </div>
  <!-- BF16 panel -->
  <div id="panel-bf16-prob" class="prec-panel hidden">
    <div class="view-toggle" style="margin-bottom:10px">
      <button class="view-btn active" id="bf16-view-model" onclick="setView('bf16','model')">By Model</button>
      <button class="view-btn" id="bf16-view-rank" onclick="setView('bf16','rank')">By Rank</button>
    </div>
    <div id="bf16-model-view">
      <div class="tab-bar">{bf16_tab_btns}</div>
      {bf16_tabs}
    </div>
    <div id="bf16-rank-view" style="display:none">
      <div class="tab-bar">{bf16_rank_btns}</div>
      {bf16_rank_tabs}
    </div>
  </div>
</section>

<!-- ===================== JUDGE RESPONSE MODAL ===================== -->
<div class="modal-overlay" id="judge-modal" onclick="closeJudge(event)">
  <div class="modal-box" onclick="event.stopPropagation()">
    <div class="modal-header">
      <div>
        <span class="modal-title" id="judge-modal-title">Judge Response</span>
        <span class="modal-subtitle" id="judge-modal-subtitle"></span>
      </div>
      <div class="modal-actions">
        <button class="modal-btn" onclick="closeJudge()">&#x2715; Close</button>
      </div>
    </div>
    <div class="modal-code" id="judge-modal-body" style="font-family:system-ui,sans-serif;white-space:normal;line-height:1.6"></div>
  </div>
</div>

<!-- ===================== KERNEL CODE MODAL ===================== -->
<div class="modal-overlay" id="kernel-modal" onclick="closeKernel(event)">
  <div class="modal-box" onclick="event.stopPropagation()">
    <div class="modal-header">
      <div>
        <span class="modal-title" id="modal-title">Kernel</span>
        <span class="modal-subtitle" id="modal-subtitle"></span>
      </div>
      <div class="modal-actions">
        <button class="modal-btn" onclick="copyKernel()">Copy</button>
        <button class="modal-btn" onclick="closeKernel()">✕ Close</button>
      </div>
    </div>
    <pre class="modal-code" id="modal-code"></pre>
  </div>
</div>

<script>
var KERNELS    = {kernels_json};
var HACK_FLAGS = {hack_flags_json};

var _currentPrec = 'fp32';

function setPrecision(prec) {{
  _currentPrec = prec;
  document.getElementById('prec-label').textContent = '(' + prec.toUpperCase() + ')';
  document.getElementById('btn-fp32').classList.toggle('active', prec === 'fp32');
  document.getElementById('btn-bf16').classList.toggle('active', prec === 'bf16');
  ['scatter','agg','prob'].forEach(function(section) {{
    var fp32Panel = document.getElementById('panel-fp32-' + section);
    var bf16Panel = document.getElementById('panel-bf16-' + section);
    if (fp32Panel) fp32Panel.classList.toggle('hidden', prec !== 'fp32');
    if (bf16Panel) bf16Panel.classList.toggle('hidden', prec !== 'bf16');
  }});
}}

function switchTab(prefix, level) {{
  document.querySelectorAll('[id^="tab-' + prefix + '-"]').forEach(function(el) {{
    el.style.display = 'none';
  }});
  document.querySelectorAll('.tab-btn[data-prefix="' + prefix + '"]').forEach(function(btn) {{
    btn.classList.toggle('active', parseInt(btn.dataset.level) === level);
  }});
  var target = document.getElementById('tab-' + prefix + '-' + level);
  if (target) target.style.display = 'block';
}}

function showKernel(event, key, model, level, pid) {{
  event.preventDefault();
  var code = KERNELS[key];
  if (!code) {{ return; }}
  document.getElementById('modal-title').textContent = model;
  document.getElementById('modal-subtitle').textContent =
    'Level ' + level + ' · Problem ' + pid;
  document.getElementById('modal-code').textContent = code;
  document.getElementById('kernel-modal').classList.add('open');
}}

function closeKernel(event) {{
  if (!event || event.target === document.getElementById('kernel-modal')) {{
    document.getElementById('kernel-modal').classList.remove('open');
  }}
}}

function copyKernel() {{
  var code = document.getElementById('modal-code').textContent;
  navigator.clipboard.writeText(code).catch(function() {{
    var ta = document.createElement('textarea');
    ta.value = code;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
  }});
  var btn = event.target;
  btn.textContent = 'Copied!';
  setTimeout(function() {{ btn.textContent = 'Copy'; }}, 1500);
}}

// ---- Aggregate table sorting ----
var _sortState = {{}};  // {{ tblId: {{ col, asc }} }}

function sortAgg(tblId, col) {{
  var tbl = document.getElementById(tblId);
  if (!tbl) return;
  var state = _sortState[tblId] || {{}};
  var asc = (state.col === col) ? !state.asc : false; // default desc
  _sortState[tblId] = {{ col: col, asc: asc }};

  // Update arrows
  tbl.querySelectorAll('.sortable-hdr').forEach(function(th) {{
    var arrow = th.querySelector('.sort-arrow');
    if (!arrow) return;
    if (parseInt(th.dataset.col) === col) {{
      arrow.textContent = asc ? ' ▲' : ' ▼';
    }} else {{
      arrow.textContent = '';
    }}
  }});

  var tbody = tbl.tBodies[0];
  var rows = Array.from(tbody.rows);
  rows.sort(function(a, b) {{
    var av = a.cells[col] ? a.cells[col].textContent.trim() : '';
    var bv = b.cells[col] ? b.cells[col].textContent.trim() : '';
    // Pending rows (—) always sink to bottom
    var aNum = parseFloat(av.replace('%','').replace('×',''));
    var bNum = parseFloat(bv.replace('%','').replace('×',''));
    var aIsNum = !isNaN(aNum), bIsNum = !isNaN(bNum);
    if (!aIsNum && !bIsNum) return 0;
    if (!aIsNum) return 1;
    if (!bIsNum) return -1;
    return asc ? (aNum - bNum) : (bNum - aNum);
  }});
  rows.forEach(function(r) {{ tbody.appendChild(r); }});
}}

// ---- Per-problem view toggle (By Model / By Rank) ----
function setView(prec, view) {{
  var modelDiv = document.getElementById(prec + '-model-view');
  var rankDiv  = document.getElementById(prec + '-rank-view');
  var modelBtn = document.getElementById(prec + '-view-model');
  var rankBtn  = document.getElementById(prec + '-view-rank');
  if (!modelDiv || !rankDiv) return;
  modelDiv.style.display = (view === 'model') ? 'block' : 'none';
  rankDiv.style.display  = (view === 'rank')  ? 'block' : 'none';
  if (modelBtn) modelBtn.classList.toggle('active', view === 'model');
  if (rankBtn)  rankBtn.classList.toggle('active',  view === 'rank');
}}

document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') {{
    closeKernel({{ target: document.getElementById('kernel-modal') }});
    closeJudge({{ target: document.getElementById('judge-modal') }});
  }}
}});

function showJudgeResponse(event, flagKey) {{
  event.preventDefault();
  event.stopPropagation();
  var info = HACK_FLAGS[flagKey];
  if (!info) {{ return; }}

  document.getElementById('judge-modal-title').textContent = '\u26a0 Reward Hack: ' + info.label;
  document.getElementById('judge-modal-subtitle').textContent = flagKey;

  function renderResponse(label, raw) {{
    var parsed = {{}};
    try {{
      var cleaned = raw.replace(/^```(?:json)?\\n?|```$/gm, '').trim();
      parsed = JSON.parse(cleaned);
    }} catch(e) {{}}
    var html = '<div style="margin-bottom:16px">';
    html += '<b style="font-size:1em;color:#cba6f7">' + label + '</b>';
    if (parsed.label) {{
      var color = parsed.label === 'LEGITIMATE' ? '#59a14f' :
                  parsed.label === 'UNCERTAIN'  ? '#f28e2b' : '#e15759';
      html += ' <span style="background:' + color + ';color:#fff;border-radius:4px;padding:2px 8px;font-size:0.85em;font-weight:700">' + parsed.label + '</span>';
    }}
    if (parsed.pattern && parsed.pattern !== 'None') {{
      html += '<div style="margin-top:8px"><b>Pattern:</b> ' + _esc(parsed.pattern) + '</div>';
    }}
    if (parsed.explanation) {{
      html += '<div style="margin-top:6px"><b>Explanation:</b> ' + _esc(parsed.explanation) + '</div>';
    }}
    if (parsed.fix && parsed.fix !== 'N/A') {{
      html += '<div style="margin-top:6px"><b>Fix:</b> ' + _esc(parsed.fix) + '</div>';
    }}
    if (!parsed.label) {{
      html += '<pre style="margin-top:8px;font-size:0.8em;white-space:pre-wrap;opacity:0.7">' + _esc(raw) + '</pre>';
    }}
    html += '</div>';
    return html;
  }}

  function _esc(s) {{
    return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }}

  var body = '';
  body += renderResponse('claude-sonnet-4-6', info.claude_response || '');
  body += '<hr style="border-color:#45475a;margin:12px 0">';
  body += renderResponse('gpt-5.5', info.gpt_response || '');

  document.getElementById('judge-modal-body').innerHTML = body;
  document.getElementById('judge-modal').classList.add('open');
}}

function closeJudge(event) {{
  if (!event || event.target === document.getElementById('judge-modal')) {{
    document.getElementById('judge-modal').classList.remove('open');
  }}
}}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate KernelBench HTML leaderboard")
    p.add_argument("--hardware", default="H200", help="Hardware subdirectory under results/timing/")
    p.add_argument("--baseline", default="baseline_time_torch", help="Baseline JSON filename (without .json)")
    p.add_argument("--bf16_baseline", default="baseline_time_torch_bf16", help="BF16 baseline JSON filename (without .json)")
    p.add_argument("--out", default="leaderboard.html", help="Output HTML filename")
    p.add_argument("--use_hidden_eval", action="store_true", help="Gate correctness with eval_results_hidden.json")
    p.add_argument("--fp32_tolerance", type=float, default=None,
                   help="Override fp32 correctness threshold when re-interpreting stored hidden eval results. "
                        "Entries that only failed the D1 standard run with max_diff < this value are reclassified "
                        "as correct. Default None = use stored correctness as-is (strict 1e-4 mode). "
                        "Use 1e-3 for the relaxed official leaderboard.")
    p.add_argument("--bf16_tolerance", type=float, default=None,
                   help="Same as --fp32_tolerance but for the BF16 section (e.g. 1e-2 for bf16). "
                        "Default None = use stored bf16 hidden correctness as-is.")
    return p.parse_args()


def main():
    args = parse_args()

    # Always run from the repo root so relative paths work
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    print(f"Loading FP32 data  (hardware={args.hardware}, baseline={args.baseline}) ...")
    fp32_data = assemble_data(args.hardware, args.baseline, run_suffix="_test",
                              use_hidden_eval=args.use_hidden_eval,
                              fp32_tolerance=args.fp32_tolerance)
    if fp32_data is None:
        sys.exit("FP32 baseline not found — cannot generate leaderboard.")

    hack_flags = {}  # LLM-judge speed hack badges removed
    print(f"  Models: {fp32_data['models']}")
    for level in (1, 2, 3):
        n = len(fp32_data["all_pids"][level])
        filled = sum(1 for m in fp32_data["models"] if fp32_data["agg"][m][level] is not None)
        print(f"  Level {level}: {n} problems, {filled}/{len(fp32_data['models'])} models with data")

    bf16_baseline_name = args.bf16_baseline if args.bf16_baseline else (args.baseline + "_bf16")
    print(f"\nLoading BF16 data  (baseline={bf16_baseline_name}) ...")
    bf16_data = assemble_data(args.hardware, bf16_baseline_name, run_suffix="_bf16_test",
                              use_hidden_eval=args.use_hidden_eval,
                              fp32_tolerance=args.bf16_tolerance)
    if bf16_data is None:
        print("  [warn] No BF16 baseline found — BF16 section will be omitted.")
    else:
        print(f"  Models: {bf16_data['models']}")
        for level in (1, 2, 3):
            n = len(bf16_data["all_pids"][level])
            filled = sum(1 for m in bf16_data["models"] if bf16_data["agg"][m][level] is not None)
            print(f"  Level {level}: {n} problems, {filled}/{len(bf16_data['models'])} models with data")

    print("\nGenerating HTML ...")
    html = build_html(fp32_data, bf16_data, hack_flags=hack_flags)

    out_path = os.path.join(repo_root, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
