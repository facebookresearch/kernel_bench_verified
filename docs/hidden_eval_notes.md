# Hidden Eval Infrastructure Notes

## BF16 Compilation Race Condition (Fixed 2026-05-19)

### Symptom
Running hidden eval on BF16 runs with `build_cache=False` and `num_gpu_devices=8` left
~164 out of ~1250 compiled entries missing from `eval_results_hidden.json`, even after
multiple retry rounds. The missing count was stable (same entries failed every time),
ruling out random flakiness.

### Root Cause
PyTorch's `load_inline` uses a file-based lock (`{build_dir}/{ext_name}/lock`) to
serialize concurrent compilations. With 8 parallel GPU workers on the same node, multiple
workers compiling extensions with the **same extension name** (e.g. `"my_ext"`) race on
this lock. The losing worker hits:

```
FileNotFoundError: [Errno 2] No such file or directory: '.../optimized_relu_bf16_cuda_ext/lock'
```

`src/eval.py` catches this and returns `None` (line 660–667), so no result is written
to the hidden eval JSON for that (pid, sample). Because the cache directory is never
created, every retry hits the exact same failure — explaining why the count didn't
improve across 3 patch rounds with `build_cache=False`.

BF16 kernels are more susceptible because they tend to use more complex, multi-extension
designs with common generic names, increasing the probability of name collisions across
concurrent workers.

### Fix
Set `build_cache=True` in `scripts/submit_hidden_eval.sh`. This triggers a sequential
**pre-compilation phase** (all queued kernels compiled one by one before parallel GPU
eval begins), eliminating concurrent compilation entirely.

**Result:** 164 missing → 1 after one run with the fix. The remaining 1 entry
(`claude-sonnet-4-6 L2 pid=39 sample=4`) is a genuine kernel execution hang
(deadlock in `__shfl_down_sync` warp reduction for hidden test inputs), not a
compilation race.

### Will It Happen Again?
`submit_hidden_eval.sh` now defaults to `build_cache=True`, so **no** for all future
eval runs that go through this script.

However, if `eval_from_generations.py` is invoked **directly** with `build_cache=False`
and `num_gpu_devices > 1`, the race can recur. The permanent infrastructure fix would be
to make each kernel's extension name globally unique (e.g. by appending a hash of the
source code), but that requires changing `src/eval.py` and is a larger refactor.

### Affected Runs
All 15 BF16 hidden eval runs (5 models × 3 levels) were affected to varying degrees:

| Model                  | Affected entries (pre-fix) |
|------------------------|----------------------------|
| claude-opus-4-7        | 51                         |
| gpt-5.5                | 48                         |
| gemini-3-flash-preview | 38                         |
| kimi-k2.6              | 24                         |
| claude-sonnet-4-6      | 3                          |
| **Total**              | **164**                    |

TF32 runs were **not** affected (0 missing) — likely because TF32 kernels use simpler
extension structures with less name collision risk.
