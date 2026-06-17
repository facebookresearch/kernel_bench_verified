# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Generate hidden test input files for KernelBench problems.

Design (v2):
- Call the original get_inputs() directly at eval time, preserving ALL structural
  constraints (symmetry, triangular, softmax normalization, etc.) automatically.
- Apply 4 distribution shifts to the float tensors returned by get_inputs():
    D1: as-is         (original distribution)
    D2: x * 3        (large magnitude — catches overflow/precision hacks)
    D3: x * 0.01     (near-zero     — catches underflow/skip-small hacks)
    D4: x * -1       (negated       — catches ReLU/positivity-assumption hacks)
- Validate each config type against the reference model at generation time.
  If reference raises an exception or produces NaN/Inf, that config is excluded
  and logged to hidden_tests/filtered_configs.json for human inspection.
- Integer/bool tensors are never scaled (left unchanged across all configs).

Usage:
    python scripts/generate_hidden_inputs.py [--level 1|2|3] [--pid N] [--dry-run]
    python scripts/generate_hidden_inputs.py --validate [--level 1|2|3] [--pid N]
"""

import gc
import json
import os
import sys
import argparse
import traceback

import torch

REPO_TOP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KERNELBENCH_PATH = os.path.join(REPO_TOP, "KernelBench")
HIDDEN_TESTS_PATH = os.path.join(REPO_TOP, "hidden_tests")
FILTERED_LOG_PATH = os.path.join(HIDDEN_TESTS_PATH, "filtered_configs.json")

# Distribution shifts applied to float tensors from get_inputs()
# Each entry: (dist_id, scale_factor, human_label)
DISTRIBUTIONS = [
    ("D1",  1.0,   "original (as-is)"),
    ("D2",  3.0,   "x3 large magnitude"),
    ("D3",  0.01,  "x0.01 near-zero"),
    ("D4", -1.0,   "x-1 negated"),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_problem_id_from_filename(filename: str) -> int:
    """Extract numeric problem ID from filename like '26_GELU_.py'."""
    return int(os.path.basename(filename).split("_")[0])


def execute_problem_file(problem_path: str) -> dict:
    """Execute a problem file and return its namespace dict."""
    context = {"torch": torch, "__builtins__": __builtins__}
    with open(problem_path, "r") as f:
        src = f.read()
    try:
        exec(compile(src, problem_path, "exec"), context)
    except Exception as e:
        raise RuntimeError(f"Failed to exec {problem_path}: {e}") from e
    return context


def scale_inputs(inputs: list, factor: float) -> list:
    """Scale all floating-point tensors by factor; leave integers/scalars unchanged."""
    result = []
    for x in inputs:
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            result.append(x * factor)
        else:
            result.append(x)
    return result


def _move_to_device(obj, device):
    """Recursively move tensors/models to device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [_move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_move_to_device(x, device) for x in obj)
    return obj


def check_config(inputs: list, model) -> tuple:
    """
    Run inputs through the reference model.
    Returns (is_valid: bool, reason: str).
    Invalid if: exception raised, or any float output tensor contains NaN/Inf.
    Uses GPU if available.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model_dev = model.to(device)
        inputs_dev = _move_to_device(inputs, device)
        with torch.no_grad():
            output = model_dev(*inputs_dev)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        model.cpu()  # move back to free GPU memory

    outputs = []
    if isinstance(output, torch.Tensor):
        outputs = [output]
    elif isinstance(output, (tuple, list)):
        outputs = list(output)

    for out in outputs:
        if isinstance(out, torch.Tensor) and out.is_floating_point():
            if not torch.isfinite(out).all():
                return False, "reference output contains NaN/Inf"

    return True, ""


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_hidden_file_content(
    problem_path: str, pid: int, level: int
) -> tuple:
    """
    Generate the content of a hidden test file for one problem.
    Returns (file_content: str, filtered: list[dict]).

    The generated file:
    - Dynamically imports get_inputs() from the original problem file at eval time
    - Applies scale factors to float tensors for each distribution variant
    - Only includes distribution variants that pass reference model validation

    Filtered variants are returned as dicts for logging.
    """
    context = execute_problem_file(problem_path)

    get_inputs_fn = context.get("get_inputs")
    if get_inputs_fn is None:
        raise ValueError(f"No get_inputs() in {problem_path}")

    ModelClass = context.get("Model")
    get_init_inputs_fn = context.get("get_init_inputs", lambda: [])

    # Instantiate reference model for validation
    model = None
    if ModelClass is not None:
        try:
            init_inputs = get_init_inputs_fn()
            model = ModelClass(*init_inputs)
            model.eval()
        except Exception as e:
            raise RuntimeError(f"Model instantiation failed for {problem_path}: {e}")

    # Get a sample set of inputs for validation
    try:
        base_inputs = get_inputs_fn()
    except Exception as e:
        raise RuntimeError(f"get_inputs() failed for {problem_path}: {e}")

    # Validate each distribution variant
    valid_dists = []
    filtered = []

    for dist_id, factor, label in DISTRIBUTIONS:
        scaled = scale_inputs(base_inputs, factor)
        if model is not None:
            ok, reason = check_config(scaled, model)
        else:
            ok, reason = True, ""

        if ok:
            valid_dists.append((dist_id, factor, label))
        else:
            filtered.append({
                "level": level,
                "pid": pid,
                "problem": os.path.basename(problem_path),
                "dist_id": dist_id,
                "label": label,
                "reason": reason,
            })
            print(f"    [FILTER] L{level} pid={pid} {dist_id} ({label}): {reason}")

    if not valid_dists:
        raise ValueError(f"All distribution variants invalid for {problem_path}")

    # Free model and tensors before building the file content string
    del base_inputs, model
    gc.collect()

    # Path from hidden_tests/level{L}/ to KernelBench/level{L}/problem_file
    prob_fname = os.path.basename(problem_path)
    rel_path = os.path.join("..", "..", "KernelBench", f"level{level}", prob_fname)

    # Build the generated file content
    lines = [
        '"""',
        f"Hidden test inputs for problem {pid}: {prob_fname}",
        "",
        "Generated by scripts/generate_hidden_inputs.py (v2)",
        "DO NOT include this file in LLM generation prompts.",
        "",
        "Distribution variants (applied to float outputs of get_inputs()):",
    ]
    for dist_id, factor, label in valid_dists:
        lines.append(f"  {dist_id}: {label} (factor={factor})")
    if filtered:
        lines.append("")
        lines.append("Filtered (reference model produced error or NaN/Inf):")
        for f_item in filtered:
            lines.append(f"  {f_item['dist_id']} ({f_item['label']}): {f_item['reason']}")
    lines += [
        '"""',
        "import torch",
        "import importlib.util",
        "import os",
        "",
        f"_PROB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), {repr(rel_path)}))",
        "",
        "",
        "def _get_inputs():",
        "    \"\"\"Load and call get_inputs() from the original problem file.\"\"\"",
        "    _spec = importlib.util.spec_from_file_location('_prob', _PROB_PATH)",
        "    _mod = importlib.util.module_from_spec(_spec)",
        "    _spec.loader.exec_module(_mod)",
        "    return _mod.get_inputs()",
        "",
        "",
        "def _scale(inputs, factor):",
        "    \"\"\"Scale float tensors by factor; leave integers/scalars unchanged.\"\"\"",
        "    return [",
        "        t * factor if isinstance(t, torch.Tensor) and t.is_floating_point()",
        "        else t",
        "        for t in inputs",
        "    ]",
        "",
        "",
        "def get_hidden_inputs():",
        f'    """Returns {len(valid_dists)} input configs for hidden correctness testing."""',
        "    configs = []",
    ]

    for dist_id, factor, label in valid_dists:
        lines.append(f"    # {dist_id}: {label}")
        if factor == 1.0:
            lines.append("    configs.append(_get_inputs())")
        else:
            lines.append(f"    configs.append(_scale(_get_inputs(), {factor}))")

    lines += [
        "    return configs",
        "",
    ]

    return "\n".join(lines), filtered


# ---------------------------------------------------------------------------
# Validation of existing hidden test files
# ---------------------------------------------------------------------------

def validate_hidden_inputs(
    problem_path: str, hidden_path: str
) -> tuple:
    """
    Validate a hidden test file by running each config through the reference model.
    Returns (num_ok, num_fail, errors).
    """
    context = execute_problem_file(problem_path)
    ModelClass = context.get("Model")
    get_init_inputs_fn = context.get("get_init_inputs", lambda: [])

    if ModelClass is None:
        return 0, 0, [f"No Model class in {problem_path}"]

    try:
        init_inputs = get_init_inputs_fn()
        model = ModelClass(*init_inputs)
        model.eval()
    except Exception as e:
        return 0, 0, [f"Model instantiation failed: {e}"]

    # __file__ must be set so the generated hidden file can resolve _PROB_PATH
    hidden_context = {"torch": torch, "__builtins__": __builtins__, "__file__": hidden_path}
    with open(hidden_path, "r") as f:
        exec(compile(f.read(), hidden_path, "exec"), hidden_context)

    get_hidden_fn = hidden_context.get("get_hidden_inputs")
    if get_hidden_fn is None:
        return 0, 0, [f"No get_hidden_inputs() in {hidden_path}"]

    configs = get_hidden_fn()
    num_ok = 0
    errors = []
    for i, inputs in enumerate(configs):
        ok, reason = check_config(inputs, model)
        if ok:
            num_ok += 1
        else:
            errors.append(f"Config {i + 1}: {reason}")

    return num_ok, len(configs) - num_ok, errors


def validate_level(level: int, pid_filter: int = None):
    """Validate all hidden test files for a given level."""
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    level_dir = os.path.join(KERNELBENCH_PATH, f"level{level}")
    hidden_dir = os.path.join(HIDDEN_TESTS_PATH, f"level{level}")

    if not os.path.isdir(level_dir) or not os.path.isdir(hidden_dir):
        print(f"  Dirs not found for level {level}")
        return

    problem_files = sorted(
        [f for f in os.listdir(level_dir) if f.endswith(".py")],
        key=lambda f: int(f.split("_")[0]),
    )
    if pid_filter is not None:
        problem_files = [f for f in problem_files
                         if get_problem_id_from_filename(f) == pid_filter]

    total_ok = total_fail = 0
    failed_pids = []

    iterator = tqdm(problem_files, desc=f"Validating L{level}", unit="prob") \
        if use_tqdm else problem_files

    for fname in iterator:
        try:
            pid = get_problem_id_from_filename(fname)
        except Exception:
            continue

        hidden_path = os.path.join(hidden_dir, f"{pid}_hidden.py")
        if not os.path.exists(hidden_path):
            print(f"  [MISS ] L{level} pid={pid}: no hidden file")
            continue

        problem_path = os.path.join(level_dir, fname)
        ok, fail, errs = validate_hidden_inputs(problem_path, hidden_path)
        total_ok += ok
        total_fail += fail
        if fail > 0:
            failed_pids.append(pid)
            print(f"  [FAIL ] L{level} pid={pid}: {fail}/{ok+fail} invalid configs")
            for err in errs:
                print(f"          {err}")
        elif use_tqdm:
            tqdm.write(f"  [OK   ] L{level} pid={pid}: {ok}/{ok} valid")
        else:
            print(f"  [OK   ] L{level} pid={pid}: {ok}/{ok} configs valid")

        gc.collect()

    print(f"\nLevel {level}: {total_ok} ok, {total_fail} failed across all configs")
    if failed_pids:
        print(f"  Problems with failures: {failed_pids}")


# ---------------------------------------------------------------------------
# Process a level (generate hidden test files)
# ---------------------------------------------------------------------------

def process_level(level: int, dry_run: bool = False, pid_filter: int = None):
    """Generate hidden test files for all problems in a given level."""
    level_dir = os.path.join(KERNELBENCH_PATH, f"level{level}")
    if not os.path.isdir(level_dir):
        print(f"Level dir not found: {level_dir}")
        return

    out_dir = os.path.join(HIDDEN_TESTS_PATH, f"level{level}")
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        # Remove all existing hidden files for this level before regenerating
        # so no stale old-format files can remain if the job is killed mid-run.
        if pid_filter is None:
            for old_file in os.listdir(out_dir):
                if old_file.endswith("_hidden.py"):
                    os.remove(os.path.join(out_dir, old_file))
            print(f"  Cleared existing hidden files in {out_dir}")

    problem_files = sorted(
        [f for f in os.listdir(level_dir) if f.endswith(".py")],
        key=lambda f: int(f.split("_")[0]),
    )

    success = 0
    failed = 0
    all_filtered = []

    for fname in problem_files:
        try:
            pid = get_problem_id_from_filename(fname)
        except Exception:
            print(f"  [SKIP] Cannot parse pid from {fname}")
            continue

        if pid_filter is not None and pid != pid_filter:
            continue

        problem_path = os.path.join(level_dir, fname)
        try:
            content, filtered = generate_hidden_file_content(problem_path, pid, level)
            all_filtered.extend(filtered)
        except Exception as e:
            print(f"  [FAIL] L{level} pid={pid} ({fname}): {e}")
            traceback.print_exc()
            failed += 1
            continue
        finally:
            # Free memory after each problem (large L3 models can accumulate)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        out_path = os.path.join(out_dir, f"{pid}_hidden.py")
        if dry_run:
            print(f"  [DRY ] Would write {out_path}")
        else:
            with open(out_path, "w") as f:
                f.write(content)
            print(f"  [OK  ] L{level} pid={pid} → {out_path}")
        success += 1

    print(f"\nLevel {level}: {success} ok, {failed} failed")
    return all_filtered


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate hidden test inputs for KernelBench (v2: distribution shifts)"
    )
    parser.add_argument("--level", type=int, choices=[1, 2, 3],
                        help="Only process this level")
    parser.add_argument("--pid", type=int,
                        help="Only process a specific problem ID")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done, don't write files")
    parser.add_argument("--validate", action="store_true",
                        help="Validate existing hidden test files against reference model")
    args = parser.parse_args()

    levels = [args.level] if args.level else [1, 2, 3]

    if args.validate:
        for level in levels:
            print(f"\n=== Validating Level {level} ===")
            validate_level(level, pid_filter=args.pid)
        return

    # Generation mode
    all_filtered = []
    for level in levels:
        print(f"\n=== Processing Level {level} ===")
        filtered = process_level(level, dry_run=args.dry_run, pid_filter=args.pid)
        if filtered:
            all_filtered.extend(filtered)

    # Save filtered configs log
    if all_filtered and not args.dry_run:
        os.makedirs(HIDDEN_TESTS_PATH, exist_ok=True)
        # Merge with existing log if present
        existing = []
        if os.path.exists(FILTERED_LOG_PATH):
            try:
                existing = json.load(open(FILTERED_LOG_PATH))
            except Exception:
                pass
        # Replace entries for pids we just processed
        processed_keys = {(f["level"], f["pid"]) for f in all_filtered}
        existing = [e for e in existing
                    if (e.get("level"), e.get("pid")) not in processed_keys]
        merged = existing + all_filtered
        with open(FILTERED_LOG_PATH, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"\nFiltered configs logged to {FILTERED_LOG_PATH} "
              f"({len(all_filtered)} entries)")


if __name__ == "__main__":
    main()
