# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

def geometric_mean_speed_ratio_correct_only(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int) -> float:
    """
    Geometric mean of the speed ratio for correct samples
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    prod = np.prod(speed_up)
    n_correct = np.sum(is_correct) # Count number of correct samples

    return prod ** (1 / n_correct) if n_correct > 0 else 0

def geometric_mean_speed_ratio_correct_and_faster_only(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int) -> float:
    """
    Geometric mean of the speed ratio for correct samples that have speedup > 1
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    speed_up = np.array([x for x in speed_up if x > 1])
    prod = np.prod(speed_up)
    n_correct_and_faster = len(speed_up)

    return prod ** (1 / n_correct_and_faster) if n_correct_and_faster > 0 else 0

def geometric_mean_speedup_all_problems(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int) -> float:
    """
    Official KernelBench average speedup metric:
    For each of the N problems, speedup = max(1, baseline/actual) if correct, else 1.
    Returns geometric mean over all N problems.
    Speedup is always >= 1 (reference code is the fallback).
    """
    per_problem_speedup = np.ones(n)
    for i in range(n):
        if is_correct[i] and actual_speed[i] > 0:
            speedup = baseline_speed[i] / actual_speed[i]
            per_problem_speedup[i] = max(1.0, speedup)
    return float(np.prod(per_problem_speedup) ** (1.0 / n)) if n > 0 else 1.0

def fastp(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int, p: float) -> float:
    """
    Rate of samples within a threshold p
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    fast_p_score = np.sum(speed_up > p)
    return fast_p_score / n if n > 0 else 0

def geometric_mean_memory_ratio_all_problems(is_correct: np.ndarray, baseline_memory: np.ndarray, kernel_memory: np.ndarray, n: int) -> float:
    """
    Memory analogue of geometric_mean_speedup_all_problems.
    For each of the N problems: ratio = min(1, kernel/baseline) if correct and valid, else 1.0.
    Returns geomean over all N problems. Lower = better (ratio < 1 means memory savings).
    """
    per_problem_ratio = np.ones(n)
    for i in range(n):
        if is_correct[i] and baseline_memory[i] > 0 and kernel_memory[i] > 0:
            ratio = kernel_memory[i] / baseline_memory[i]
            per_problem_ratio[i] = min(1.0, ratio)
    return float(np.prod(per_problem_ratio) ** (1.0 / n)) if n > 0 else 1.0


def geometric_mean_memory_ratio(is_correct: np.ndarray, baseline_memory: np.ndarray, kernel_memory: np.ndarray, n: int) -> float:
    """
    Geometric mean of kernel/baseline memory ratio over all N problems.
    Correct problems with valid memory measurements use the actual ratio;
    all others use 1.0 (neutral, no penalty). Lower ratio = better.
    """
    ratios = np.ones(n)
    for i in range(n):
        if is_correct[i] and baseline_memory[i] > 0 and kernel_memory[i] > 0:
            ratios[i] = kernel_memory[i] / baseline_memory[i]
    return float(np.prod(ratios) ** (1.0 / n)) if n > 0 else 1.0

def memory_efficient_p(is_correct: np.ndarray, baseline_memory: np.ndarray, kernel_memory: np.ndarray, n: int, threshold: float = 1.0) -> float:
    """
    Fraction of all N problems where the kernel's memory ratio < threshold.
    Only correct problems with valid measurements can count toward this fraction.
    """
    count = 0
    for i in range(n):
        if is_correct[i] and baseline_memory[i] > 0 and kernel_memory[i] > 0:
            if kernel_memory[i] / baseline_memory[i] < threshold:
                count += 1
    return count / n if n > 0 else 0.0
