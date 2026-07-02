<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->

# >10x Speedup Analysis Report (TF32 + BF16)

Generated: 2026-05-19
**Scope:** All model × dtype × level runs. Best-of-5 samples per (model, level, pid).
Only entries passing all 4 hidden test distributions (standard, ×3, ×0.01, ×-1) are included.

---

## Complete Table (16 entries, 6 unique problems)

| Speedup | Dtype | Model          | L | PID | Problem                                   | Verdict       |
|--------:|-------|----------------|---|-----|-------------------------------------------|---------------|
| 234.6×  | TF32  | gpt-5.5        | 2 | 83  | Conv3d_GroupNorm_Min_Clamp_Dropout        | ⚠ Degenerate  |
|  71.6×  | TF32  | gpt-5.5        | 2 | 23  | Conv3d_GroupNorm_Mean                     | ⚠ Degenerate  |
|  66.6×  | BF16  | gpt-5.5        | 2 | 83  | Conv3d_GroupNorm_Min_Clamp_Dropout        | ⚠ Degenerate  |
|  60.4×  | BF16  | gpt-5.5        | 2 | 23  | Conv3d_GroupNorm_Mean                     | ⚠ Degenerate  |
|  56.6×  | BF16  | gpt-5.5        | 2 | 50  | ConvTranspose3d_Scaling_AvgPool_Bias      | ✅ Genuine     |
|  34.2×  | TF32  | gpt-5.5        | 2 | 80  | Gemm_Max_Subtract_GELU                    | ⚠ Degenerate  |
|  33.1×  | BF16  | gpt-5.5        | 2 | 44  | ConvTranspose2d_Multiply_GlobalAvgPool    | ✅ Genuine     |
|  28.1×  | BF16  | gpt-5.5        | 1 | 40  | LayerNorm                                 | ✅ Genuine     |
|  25.5×  | TF32  | gpt-5.5        | 1 | 40  | LayerNorm                                 | ✅ Genuine     |
|  23.7×  | TF32  | gpt-5.5        | 2 | 44  | ConvTranspose2d_Multiply_GlobalAvgPool    | ✅ Genuine     |
|  19.5×  | BF16  | gpt-5.5        | 2 | 42  | ConvTranspose2d_GlobalAvgPool_BiasAdd     | ✅ Genuine     |
|  18.4×  | BF16  | gpt-5.5        | 2 | 80  | Gemm_Max_Subtract_GELU                    | ⚠ Degenerate  |
|  18.0×  | TF32  | claude-opus-4-7| 2 | 80  | Gemm_Max_Subtract_GELU                    | ⚠ Degenerate  |
|  16.8×  | TF32  | gpt-5.5        | 2 | 42  | ConvTranspose2d_GlobalAvgPool_BiasAdd     | ✅ Genuine     |
|  12.8×  | TF32  | claude-opus-4-7| 2 | 42  | ConvTranspose2d_GlobalAvgPool_BiasAdd     | ✅ Genuine     |
|  12.5×  | BF16  | gpt-5.5        | 2 | 96  | ConvTranspose3d_Multiply_Max_GlobalAvgPool| ✅ Genuine     |

**7 degenerate entries** (3 problems) — already excluded from leaderboard metrics via `EXCLUDED_PIDS`.
**9 genuine entries** (5 problems) — all algebraically correct, no hacks.
Only **gpt-5.5** and **claude-opus-4-7** appear in this list.

---

## Problem-by-Problem Analysis

### ⚠ pid=83 — Conv3d → GroupNorm → min(x, 0) → clamp(min=0, max=1) → Dropout

**Entries:** TF32 gpt-5.5 (234.6×), BF16 gpt-5.5 (66.6×)

`min(x, 0)` on `rand` inputs (all ≥ 0) gives 0. `clamp(0, min=0) = 0`. Dropout of zeros is zeros.
The correct output is identically zero for **any** input — scaling by ×3, ×0.01, ×-1 doesn't change this
because the algebraic identity `min(x, 0) = 0 ∀ x ≥ 0` breaks when inputs can be negative (×-1),
but our hidden tests only verify correctness against the reference, not that output is non-zero.

**Both kernels:** Return `torch.zeros(batch, out_channels, D, H, W)` immediately, skipping all ops.
BF16 kernel computes the output shape from conv parameters to avoid running conv.

**Fix needed:** Use `randn` inputs (includes negatives) or different min threshold.

---

### ⚠ pid=23 — Conv3d → GroupNorm(gamma=1, beta=0) → mean(all dims)

**Entries:** TF32 gpt-5.5 (71.6×), BF16 gpt-5.5 (60.4×)

GroupNorm normalizes each group to zero mean. With default `gamma=1, beta=0`, the output has
exactly zero mean per group. The global `mean()` over all dims is therefore zero for any input —
scaling the input doesn't change the normalized distribution's mean.

**Both kernels:** Runtime-check `if all(weight==1) and all(bias==0)` → return `zeros(batch)`.
Falls back to real computation if affine params are modified after init.

**Fix needed:** Initialize GroupNorm with non-trivial `(gamma, beta)` or use `affine=False`.

---

### ⚠ pid=80 — Linear → max(dim=1, keepdim=True) → subtract mean(dim=1) → GELU

**Entries:** TF32 gpt-5.5 (34.2×), BF16 gpt-5.5 (18.4×), TF32 claude-opus-4-7 (18.0×)

With `out_features=8192` and `max_dim=1`: `max(x, dim=1, keepdim=True)` yields shape `(B, 1)`.
Subtracting `x.mean(dim=1)` from a single-element row: `x - x = 0`. `GELU(0) = 0`.

**gpt-5.5 kernel (both dtypes):** Checks `if max_dim in (1, -1): return zero_kernel(...)`.
**claude-opus-4-7 kernel:** Explicitly comments "For shape (B, 1) along dim=1, mean is x itself,
so result is 0, GELU(0)=0", then returns `torch.zeros({B, 1})`.

Both models independently derived the same mathematical identity.

**Fix needed:** Change `max_dim` so it doesn't collapse to a single element, or use `out_features` > 1.

---

### ✅ pid=40 — LayerNorm (L1, input shape 16×64×256×256)

**Entries:** TF32 gpt-5.5 (25.5×), BF16 gpt-5.5 (28.1×)

Custom multi-pass CUDA LayerNorm. The normalized shape is `(64, 256, 256)` = **4,194,304 elements**
per sample — far larger than PyTorch's built-in LayerNorm is optimized for. The custom kernel uses
efficient parallel partial-sum reduction over large arrays. Dispatches for float32, float16, bfloat16.

**No mathematical shortcuts.** The kernel reads all elements, computes mean and variance via parallel
reduction, then normalizes. Speedup is purely from the optimized reduction pattern for this shape.

**Verdict: Genuine custom CUDA kernel.**

---

### ✅ pid=44 — ConvTranspose2d(k=3,s=2,p=1,op=1) → ×bias_multiplier → GlobalAvgPool

**Entries:** TF32 gpt-5.5 (23.7×), BF16 gpt-5.5 (33.1×)

**Key insight:** `GlobalAvgPool(ConvTranspose2d(x))` = dot product between input channel sums and
kernel spatial sums, computable without materializing the full deconv output. For the default problem
shape (batch=128, in=3, out=16, H=32, W=32), the ConvTranspose2d output would be ~128×16×63×63
≈ 400MB; the kernel avoids this entirely, producing shape `(B, out_channels, 1, 1)` directly.

**Kernel approach:** Pre-sums the convolution weights over spatial dims, then computes a lightweight
GEMM between per-channel input sums and the weight sums. Bias and multiplier fused in.

**Verdict: Genuine algebraic reduction. Mathematically equivalent for any input.**

---

### ✅ pid=42 — ConvTranspose2d → GlobalAvgPool → +bias → LogSumExp → sum → ×10

**Entries:** TF32 gpt-5.5 (16.8×), BF16 gpt-5.5 (19.5×), TF32 claude-opus-4-7 (12.8×)

Same algebraic reduction as pid=44 for the GlobalAvgPool(ConvTranspose2d) step.
After obtaining the pooled output `(B, C, 1, 1)`, bias is added, then `logsumexp(dim=1)` and
final sum/scale. The entire pipeline is fused into a single CUDA kernel.

**gpt-5.5 kernel:** Full pipeline fusion — deconv+pool+bias+logsumexp computed together.
**claude-opus-4-7 kernel:** Same approach, custom `fused_forward` CUDA kernel. Both verified.

**Verdict: Genuine algebraic reduction. Multiple models independently discovered the same approach.**

---

### ✅ pid=50 — ConvTranspose3d(k=3,s=2,p=1) → ×scale1 → AvgPool3d(2) → +bias → ×scale2 (BF16 only)

**Entries:** BF16 gpt-5.5 (56.6×)

**Key insight:** `AvgPool3d(kernel=2) ∘ ConvTranspose3d(stride=2)` maps each AvgPool output to
a 2×2×2 = 8-element block of the deconv output. With stride=2, these 8 elements correspond to
exactly 8 kernel positions. The effective weight can be pre-summed:

```
eff_weight[co,ci,t] = (scale1 × scale2 / 8) × Σ_{(kd,kh,kw) in block t} kernel[ci,co,kd,kh,kw]
eff_bias[co] = (conv_bias[co] × scale1 + add_bias[co]) × scale2
```

The fused kernel then computes `output = EffectiveConv(x) + eff_bias` directly on the AvgPool
output grid, **never materializing** the ConvTranspose3d output (128×16×31×63×63 ≈ 6 GB in BF16).

Not seen in TF32 results — likely because the TF32 baseline is slower (no TF32 deconv optimization)
and the speedup doesn't reach 10× for that dtype. May appear at lower thresholds.

**Verdict: Genuine algebraic reduction. Pre-computes effective kernel, avoids 6 GB intermediate.**

---

### ✅ pid=96 — ConvTranspose3d(k=3,s=2,p=1) → ×scale → MaxPool3d(2) → GlobalAvgPool → clamp (BF16 only)

**Entries:** BF16 gpt-5.5 (12.5×)

**Key insight:** MaxPool3d(2) ∘ ConvTranspose3d(stride=2) maps each MaxPool output to a 2×2×2
deconv output window. The kernel computes deconv values for those 8 positions on-the-fly via an
inlined `conv_transpose3d_value_k3s2p1()` device function, takes the max, accumulates for
GlobalAvgPool, applies clamp — all in a single kernel.

Avoids materializing both the ConvTranspose3d output (≈6 GB) and the MaxPool output (≈750 MB).
The output is `(N, Cout, 1, 1, 1)` = tiny. One CUDA block per `(batch, channel)` pair handles
the entire reduction.

**Verdict: Genuine fused operator. Pipeline compression via on-the-fly deconv evaluation.**

---

## Key Takeaways

1. **Only gpt-5.5 and claude-opus-4-7** reach >10× speedup. gpt-5.5 dominates with 13 entries
   (vs 3 for claude-opus).

2. **All degenerate problems are concentrated in L2.** The 3 degenerate problems (pid=23, 80, 83)
   are already excluded from leaderboard metrics via `EXCLUDED_PIDS = {2: {23, 80, 83}}`.

3. **The genuine speedups all exploit the same pattern:** ConvTranspose (stride=2) followed by
   pooling can be computed without materializing the huge intermediate tensor. This is a real
   algorithmic insight — the intermediate is often 8–64× larger than the input.

4. **pid=80's degenerate identity was independently found by both gpt-5.5 and claude-opus-4-7**
   — even with slightly different kernel implementations and different comments explaining the math.

5. **BF16 and TF32 show the same pattern of genuine optimizations** on shared problems (pid=40,
   42, 44). pid=50 and pid=96 are BF16-only in the >10× table, likely because the BF16 baseline
   for ConvTranspose3d is slower (no cuDNN BF16 optimization parity with TF32).

6. **No adversarial hacks found.** All legitimate speedups pass all 4 hidden distributions.
