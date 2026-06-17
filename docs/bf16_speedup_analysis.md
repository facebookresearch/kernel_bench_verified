# BF16 >10x Speedup Analysis Report

Generated: 2026-05-19 | All entries pass all 4 hidden test distributions.

## Summary Table

| Speedup | Model   | L | PID | Problem                              | Verdict              |
|--------:|---------|---|-----|--------------------------------------|----------------------|
|  66.6×  | gpt-5.5 | 2 |  83 | Conv3d_GroupNorm_Min_Clamp_Dropout   | ⚠ Degenerate problem |
|  60.4×  | gpt-5.5 | 2 |  23 | Conv3d_GroupNorm_Mean                | ⚠ Degenerate problem |
|  56.6×  | gpt-5.5 | 2 |  50 | ConvTranspose3d_Scaling_AvgPool_Bias | ✅ Genuine fusion     |
|  33.1×  | gpt-5.5 | 2 |  44 | ConvTranspose2d_Multiply_GlobalAvgPool | ✅ Genuine fusion   |
|  28.1×  | gpt-5.5 | 1 |  40 | LayerNorm                            | ✅ Genuine CUDA      |
|  19.5×  | gpt-5.5 | 2 |  42 | ConvTranspose2d_GlobalAvgPool_Bias   | ✅ Genuine fusion    |
|  18.4×  | gpt-5.5 | 2 |  80 | Gemm_Max_Subtract_GELU               | ⚠ Degenerate problem |
|  12.5×  | gpt-5.5 | 2 |  96 | ConvTranspose3d_Multiply_Max_GlobalAvgPool | ✅ Genuine fusion |

**Bottom line: 3 degenerate problems, 5 legitimate optimizations. No adversarial hacks.**

---

## Degenerate Problems (Problem Design Issues)

These pass all hidden tests because the **correct output is provably constant (zero) for
any input** — no input distribution can distinguish a correct kernel from a zero-returning
one. They are already excluded from leaderboard metrics (all_pids filter).

### pid=83 (66.6×) — Conv3d → GroupNorm → min(x, 0) → clamp(min=0, max=1)

`min(x, v); clamp(min=v)` is algebraically `v` for any `x`. With `min_value=0.0` and
`rand` inputs (all positive), `min(x, 0) = 0` always, then `clamp(0, min=0) = 0`.
Kernel detects this and returns `torch.zeros(...).expand(output_shape)`.

Hidden tests pass because the correct answer IS zero for ×3, ×0.01, and ×-1 inputs too
(the algebraic identity holds regardless of input magnitude or sign).

**Fix needed:** Use separate min threshold and clamp bounds, or `randn` inputs.

### pid=23 (60.4×) — Conv3d → GroupNorm(gamma=1, beta=0) → mean(all dims)

GroupNorm normalizes each group to zero-mean. With default `gamma=1, beta=0`, the
normalized output has mean exactly 0 per group. The global mean is therefore 0 for any
input. Kernel returns `torch.zeros(batch_size, 1)`.

Hidden tests pass because scaling the input (×3, ×0.01, ×-1) doesn't change the
zero-mean property of GroupNorm output.

**Fix needed:** Initialize GroupNorm with non-trivial weights.

### pid=80 (18.4×) — Linear → max(dim=1, keepdim=True) → subtract mean(dim=1) → GELU

With `out_features=8192` and `max_dim=1`: `max(x, dim=1, keepdim=True)` produces shape
`(batch, 1)`. Then `x - x.mean(dim=1)` on a 1-element row = `x - x = 0`.
`GELU(0) = 0`. Kernel returns `x.new_zeros((x.shape[0], 1))`.

**Fix needed:** Change `max_dim` to a dimension that doesn't collapse to 1 element.

---

## Legitimate Optimizations

### pid=50 (56.6×) — ConvTranspose3d(k=3,s=2,p=1) → ×scale1 → AvgPool3d(2) → +bias → ×scale2

**Key insight:** The combination `AvgPool3d(kernel=2) ∘ ConvTranspose3d(stride=2)` can be
algebraically reduced. Each AvgPool3d output element averages 8 adjacent deconv outputs.
With stride=2, these 8 outputs correspond to exactly one 2×2×2 block of the 3×3×3 deconv
kernel. The kernel pre-sums the relevant kernel weights:

```
eff_weight[co, ci, t] = (s1 × s2 / 8) × Σ_{(kd,kh,kw) in block t} weight[ci, co, kd, kh, kw]
eff_bias[co] = (conv_bias[co] × s1 + add_bias[co]) × s2
```

Then the fused kernel computes the final result directly from the input `x` without
materializing the ConvTranspose3d output (128×16×31×63×63 ≈ 6 GB in BF16!). The speedup
is dominated by this memory savings. The scale and bias fusion is additional optimization.

**Verdict: Genuine algebraic reduction. Mathematically correct for any input.**

### pid=44 (33.1×) — ConvTranspose2d → ×bias → GlobalAvgPool

Similar algebraic reduction to pid=50 but in 2D and without AvgPool (uses GlobalAvgPool
= mean over all spatial dims). The global mean of a ConvTranspose2d output can be
expressed as a dot product between input spatial sums and kernel spatial sums, computable
without materializing the full output tensor.

**Verdict: Genuine algebraic reduction.**

### pid=40 (28.1×) — LayerNorm

Custom multi-pass CUDA LayerNorm kernel with parallel reduction. The input shape is
(16, 64, 256, 256), normalized over the last 3 dims = 4,194,304 elements per sample.
PyTorch's built-in LayerNorm is not optimized for this unusually large normalized shape.
The custom kernel achieves genuine speedup through efficient parallel partial-sum
reduction and avoids multiple passes over memory.

**Verdict: Genuine custom CUDA kernel. No mathematical shortcut.**

### pid=42 (19.5×) — ConvTranspose2d → GlobalAvgPool → +bias → LogSumExp

Same algebraic reduction pattern as pid=44 — GlobalAvgPool of ConvTranspose2d output
computed via kernel/input sum products, then bias and LogSumExp applied on the small
(batch × channels) output.

**Verdict: Genuine algebraic reduction.**

### pid=96 (12.5×) — ConvTranspose3d(k=3,s=2,p=1) → ×scale → MaxPool3d(2) → GlobalAvgPool → clamp

The most complex fusion. The kernel computes the entire pipeline without materializing
any intermediate tensor:

1. For each MaxPool3d output position, evaluate 8 ConvTranspose3d output values
   on-the-fly using `conv_transpose3d_value_k3s2p1()` (inlined deconv)
2. Take max of those 8 values × scale
3. Accumulate over all MaxPool positions for GlobalAvgPool
4. Divide by count and apply clamp — all in a single kernel

The output is shape (N, Cout, 1, 1, 1). By fusing MaxPool + GlobalAvgPool + ConvTranspose3d,
the kernel avoids materializing both the ConvTranspose3d output (≈6 GB) and the MaxPool
output (≈750 MB) for this problem size. The clamp(0,1) on `rand` inputs after a
scale=0.5 multiplication does constrain the output to [0,0.5] but is not degenerate —
the values vary with input.

**Verdict: Genuine fused operator. Algebraically equivalent pipeline compression.**

---

## Key Observations

1. **All high-speedup legitimate cases are ConvTranspose fusions.** gpt-5.5 consistently
   identifies that `ConvTranspose3d/2d + pooling` pipelines can be fused to avoid
   materializing the large intermediate deconv output. This is a real algorithmic insight.

2. **The degenerate cases (pid=23, 80, 83) are already excluded** from leaderboard
   metrics via the `EXCLUDED_PIDS = {2: {23, 80, 83}}` filter added 2026-05-19.

3. **No BF16-specific hacks found.** The BF16 high-speedup entries use the same
   algebraic reduction strategies as TF32. No entries exploit BF16-specific numerical
   properties or return incorrect results for hidden distributions.

4. **All 5 genuine entries pass all 4 hidden distributions** (standard, ×3, ×0.01,
   ×-1), confirming correctness holds across input magnitudes and signs.
