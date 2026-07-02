# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json, os
from collections import defaultdict
from tabulate import tabulate
import numpy as np
import pydra
from pydra import REQUIRED, Config
from src.dataset import construct_kernelbench_dataset

"""
Benchmark Eval Analysis

This script shows how to conduct analysis for model performance on KernelBench.

Supports multi-sample eval_results.json where keys are "{problem_id}_{sample_id}".
Metrics are computed per sample across all problems, then reported as mean ± std.

- Success rate (compiled and correctness)
- Geometric mean of speedup for correct samples
- Fast_p score for different speedup thresholds (we recommend and use this metric)

Usage:
```
python3 scripts/benchmark_eval_analysis.py run_name=<run_name> level=<level> hardware=<hardware> baseline=<baseline>
```
hardware + baseline should correspond to the results/timing/hardware/baseline.json file
"""


class AnalysisConfig(Config):
    def __init__(self):
        self.run_name = REQUIRED  # name of the run to evaluate
        self.level = REQUIRED     # level to evaluate
        self.hardware = REQUIRED  # hardware to evaluate
        self.baseline = REQUIRED  # baseline to compare against
        # When True: correctness from eval_results_hidden.json, runtime from eval_results.json
        self.use_hidden_eval = False

    def __repr__(self):
        return f"AnalysisConfig({self.to_dict()})"


def build_baseline_lookup(baseline_results, level):
    """
    Build a mapping from integer problem_id -> baseline mean runtime.
    Baseline keys look like "1_Square_matrix_multiplication_.py".
    """
    lookup = {}
    for key, entry in baseline_results[f'level{level}'].items():
        if entry is None or entry.get("mean") is None:
            continue  # skip problems that failed baseline (e.g. BF16 unsupported ops)
        pid = int(key.split('_')[0])
        lookup[pid] = entry["mean"]
    return lookup


def build_baseline_memory_lookup(baseline_results, level):
    """
    Build a mapping from integer problem_id -> baseline peak memory (bytes).
    Returns an empty dict if no entries contain 'peak_memory' (old JSON files).
    """
    lookup = {}
    for key, entry in baseline_results[f'level{level}'].items():
        if entry is None:
            continue
        if "peak_memory" in entry:
            pid = int(key.split('_')[0])
            lookup[pid] = entry["peak_memory"]
    return lookup


def parse_eval_results(eval_results):
    """
    Parse keys of the form "{problem_id}_{sample_id}" into a nested dict:
      by_sample[sample_id][problem_id] = entry
    Also returns the set of all problem_ids and sample_ids.
    """
    by_sample = defaultdict(dict)
    problem_ids = set()
    sample_ids = set()
    for key, entry in eval_results.items():
        pid, sid = key.rsplit('_', 1)
        pid, sid = int(pid), int(sid)
        by_sample[sid][pid] = entry
        problem_ids.add(pid)
        sample_ids.add(sid)
    return by_sample, sorted(problem_ids), sorted(sample_ids)


def patch_sample(sample_entries, all_problem_ids):
    """
    For a single sample, fill in missing problem_ids with failed entries.
    """
    for pid in all_problem_ids:
        if pid not in sample_entries:
            sample_entries[pid] = {
                "compiled": False,
                "correctness": False,
                "runtime": -1.0,
            }
    return sample_entries


def compute_sample_metrics(sample_entries, all_problem_ids, baseline_lookup, p_values, baseline_memory_lookup=None):
    """
    Given one sample's entries across all problems, compute scalar metrics.
    Returns a dict of metric_name -> value.
    """
    from src.score import geometric_mean_speed_ratio_correct_only, geometric_mean_speedup_all_problems, fastp, geometric_mean_memory_ratio, memory_efficient_p

    n = len(all_problem_ids)
    is_correct = np.array([sample_entries[pid]["correctness"] for pid in all_problem_ids])
    compiled   = np.array([sample_entries[pid]["compiled"]    for pid in all_problem_ids])
    baseline_speed = np.array([baseline_lookup[pid] for pid in all_problem_ids])
    actual_speed   = np.array([sample_entries[pid]["runtime"] for pid in all_problem_ids])

    metrics = {
        "compiled_rate":    compiled.mean(),
        "correctness_rate": is_correct.mean(),
        "gmsr":             geometric_mean_speed_ratio_correct_only(is_correct, baseline_speed, actual_speed, n),
        "avg_speedup":      geometric_mean_speedup_all_problems(is_correct, baseline_speed, actual_speed, n),
    }
    for p in p_values:
        metrics[f"fast_p_{p}"] = fastp(is_correct, baseline_speed, actual_speed, n, p)

    if baseline_memory_lookup:
        baseline_mem = np.array([baseline_memory_lookup.get(pid, -1.0) for pid in all_problem_ids])
        kernel_mem   = np.array([sample_entries[pid].get("peak_memory", -1.0) for pid in all_problem_ids])
        metrics["gmr"]               = geometric_mean_memory_ratio(is_correct, baseline_mem, kernel_mem, n)
        metrics["memory_efficient_p"] = memory_efficient_p(is_correct, baseline_mem, kernel_mem, n, threshold=1.0)

    return metrics


def compute_best_of_n_metrics(by_sample, all_problem_ids, baseline_lookup, sample_ids, p_values):
    """
    Oracle best-of-N: for each problem, select the sample with the highest speedup
    (baseline/actual) that is also correct. Problems with no correct sample count as failures.
    Returns a dict of metric_name -> value.
    """
    from src.score import geometric_mean_speed_ratio_correct_only, geometric_mean_speedup_all_problems, fastp

    n = len(all_problem_ids)
    best_correct  = np.zeros(n, dtype=bool)
    best_speedup  = np.zeros(n)   # baseline / actual for best correct sample (0 if none)
    best_runtime  = np.full(n, -1.0)

    for i, pid in enumerate(all_problem_ids):
        baseline = baseline_lookup.get(pid, None)
        if baseline is None:
            continue
        for sid in sample_ids:
            entry = by_sample[sid].get(pid)
            if entry is None:
                continue
            if entry.get("correctness") and entry.get("runtime", -1) > 0:
                speedup = baseline / entry["runtime"]
                if speedup > best_speedup[i]:
                    best_speedup[i]  = speedup
                    best_correct[i]  = True
                    best_runtime[i]  = entry["runtime"]

    baseline_arr = np.array([baseline_lookup.get(pid, 1.0) for pid in all_problem_ids])
    actual_arr   = np.where(best_correct, best_runtime, -1.0)

    metrics = {
        "correctness_rate": best_correct.mean(),
        "gmsr": geometric_mean_speed_ratio_correct_only(best_correct, baseline_arr, actual_arr, n),
        "avg_speedup": geometric_mean_speedup_all_problems(best_correct, baseline_arr, actual_arr, n),
    }
    for p in p_values:
        metrics[f"fast_p_{p}"] = fastp(best_correct, baseline_arr, actual_arr, n, p)
    return metrics


def merge_hidden_eval(standard_results: dict, hidden_results: dict) -> dict:
    """
    Merge hidden correctness with standard runtime.
    For each key present in standard_results:
      - correctness: from hidden_results (if key exists), else False
      - runtime / compiled / runtime_stats / etc: from standard_results
    Keys in hidden_results but not in standard_results are ignored (no runtime available).
    """
    merged = {}
    for key, std_entry in standard_results.items():
        merged[key] = dict(std_entry)  # copy runtime, compiled, etc.
        if key in hidden_results:
            merged[key]["correctness"] = hidden_results[key].get("correctness", False)
        else:
            # No hidden result → treat as failed (could not be evaluated)
            merged[key]["correctness"] = False
    return merged


def analyze_multi_sample_eval(run_name, hardware, baseline, level, use_hidden_eval=False):
    dataset = construct_kernelbench_dataset(level)
    total_count = len(dataset)

    eval_file_path = f'runs/{run_name}/eval_results.json'
    baseline_file_path = f'results/timing/{hardware}/{baseline}.json'
    assert os.path.exists(eval_file_path),     f"Eval file does not exist at {eval_file_path}"
    assert os.path.exists(baseline_file_path), f"Baseline file does not exist at {baseline_file_path}"

    with open(eval_file_path, 'r') as f:
        eval_results = json.load(f)
    with open(baseline_file_path, 'r') as f:
        baseline_results = json.load(f)

    if use_hidden_eval:
        hidden_file_path = f'runs/{run_name}/eval_results_hidden.json'
        assert os.path.exists(hidden_file_path), \
            f"Hidden eval file not found at {hidden_file_path}. Run eval with use_hidden_tests=True first."
        with open(hidden_file_path, 'r') as f:
            hidden_results = json.load(f)
        eval_results = merge_hidden_eval(eval_results, hidden_results)
        print(f"[Hidden eval] Using correctness from {hidden_file_path}, runtime from {eval_file_path}")

    baseline_lookup = build_baseline_lookup(baseline_results, level)
    baseline_memory_lookup = build_baseline_memory_lookup(baseline_results, level)
    if not baseline_memory_lookup:
        print("[Info] No peak_memory in baseline JSON; memory metrics will be skipped.")
    by_sample, problem_ids, sample_ids = parse_eval_results(eval_results)

    # Restrict to problems covered by the baseline (normally all, but allows partial baselines)
    all_problem_ids = sorted(pid for pid in range(1, total_count + 1) if pid in baseline_lookup)

    p_values = [0.0, 0.5, 0.8, 1.0, 1.5, 2.0]

    # Compute per-sample metrics
    per_sample_metrics = []
    for sid in sample_ids:
        sample_entries = patch_sample(dict(by_sample[sid]), all_problem_ids)
        m = compute_sample_metrics(sample_entries, all_problem_ids, baseline_lookup, p_values, baseline_memory_lookup)
        per_sample_metrics.append(m)

    n_samples = len(sample_ids)

    def mean_std(key):
        vals = [m[key] for m in per_sample_metrics]
        return np.mean(vals), np.std(vals)

    # Compute best@N oracle (needed for the top-line metric)
    best_n = compute_best_of_n_metrics(by_sample, all_problem_ids, baseline_lookup, sample_ids, p_values)

    # ── Official KernelBench metric ──────────────────────────────────────────
    # Average speedup = geomean over ALL problems of max(1, best_correct_speedup)
    avg_sp_mean, avg_sp_std = mean_std("avg_speedup")
    best_n_avg_sp = best_n["avg_speedup"]

    mode_label = "hidden-gated" if use_hidden_eval else "standard"
    print("=" * 128)
    print(f"Eval Summary for {run_name}  (level={level}, {n_samples} sample(s) per problem, correctness={mode_label})")
    print("=" * 128)
    print(f"Total problems in dataset: {total_count}  |  Samples evaluated: {n_samples}")
    print()
    print("  *** OFFICIAL METRIC: Average Speedup (geomean over all problems, speedup >= 1) ***")
    print(f"      Per-sample avg:  {avg_sp_mean:.4f} ± {avg_sp_std:.4f}")
    print(f"      Best@{n_samples} oracle:  {best_n_avg_sp:.4f}")
    print("=" * 128)

    comp_mean, comp_std   = mean_std("compiled_rate")
    corr_mean, corr_std   = mean_std("correctness_rate")
    gmsr_mean, gmsr_std   = mean_std("gmsr")

    print(f"\nSuccess rates (mean ± std across {n_samples} sample(s)):")
    print(f"  Compilation rate:  {comp_mean*100:.1f}% ± {comp_std*100:.1f}%")
    print(f"  Correctness rate:  {corr_mean*100:.1f}% ± {corr_std*100:.1f}%")

    print(f"\nSpeedup metrics:")
    print(f"  Geometric mean speedup (correct only): {gmsr_mean:.4f} ± {gmsr_std:.4f}")

    rows = []
    for p in p_values:
        fp_mean, fp_std = mean_std(f"fast_p_{p}")
        rows.append([p, f"{fp_mean*100:.1f}% ± {fp_std*100:.1f}%"])

    print("\nFast_p Results (mean ± std across samples):")
    print(tabulate(rows, headers=["Speedup Threshold (p)", "Fast_p Score"], tablefmt="grid"))

    if baseline_memory_lookup:
        gmr_mean, gmr_std = mean_std("gmr")
        mep_mean, mep_std = mean_std("memory_efficient_p")
        print(f"\nMemory metrics:")
        print(f"  Geometric mean memory ratio (kernel/baseline): {gmr_mean:.4f} ± {gmr_std:.4f}  [lower = better]")
        print(f"  Memory efficient (ratio < 1.0): {mep_mean*100:.1f}% ± {mep_std*100:.1f}%")

    if n_samples > 1:
        bon_rows = []
        for p in p_values:
            fp_mean, fp_std = mean_std(f"fast_p_{p}")
            bon_rows.append([p, f"{fp_mean*100:.1f}% ± {fp_std*100:.1f}%", f"{best_n[f'fast_p_{p}']*100:.1f}%"])
        print(f"\nFast_p Results (mean ± std  vs  best@{n_samples} oracle):")
        print(tabulate(bon_rows, headers=["Speedup Threshold (p)", "Fast_p (mean ± std)", f"Fast_p (best@{n_samples})"], tablefmt="grid"))

        print(f"\nPer-sample breakdown:")
        header = ["Sample"] + ["compiled%"] + ["correct%"] + [f"fast_p_{p}" for p in p_values]
        sample_rows = []
        for i, (sid, m) in enumerate(zip(sample_ids, per_sample_metrics)):
            row = [sid, f"{m['compiled_rate']*100:.1f}%", f"{m['correctness_rate']*100:.1f}%"]
            for p in p_values:
                row.append(f"{m[f'fast_p_{p}']*100:.1f}%")
            sample_rows.append(row)
        print(tabulate(sample_rows, headers=header, tablefmt="grid"))

        # Best-of-N oracle summary
        print(f"\nBest-of-{n_samples} Oracle (per problem, best correct sample wins):")
        print(f"  Correctness rate:  {best_n['correctness_rate']*100:.1f}%")
        print(f"  Geometric mean speedup (correct only): {best_n['gmsr']:.4f}")


@pydra.main(base=AnalysisConfig)
def main(config: AnalysisConfig):
    analyze_multi_sample_eval(config.run_name, config.hardware, config.baseline, config.level,
                              use_hidden_eval=config.use_hidden_eval)


if __name__ == "__main__":
    main()
