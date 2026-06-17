# LLM-Judge Analysis: HACK-2 and HACK-5

## Summary

Both Claude Sonnet 4.6 and GPT-5.5 judges flagged these kernels as hacks. After investigation, we resolve both by **removing `get_inputs()` and test configuration from the prompt** and regenerating — the models can no longer see the input distribution or dimensions to exploit.

---

## HACK-2: L1 P90 (Cumulative Product, GPT-5.5, 4.3×)

### PyTorch Reference

```python
class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cumprod(x, dim=self.dim)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]
```

### Judge Rationale

**Claude Sonnet (SPEED_HACK):**
> "The kernel breaks out of the cumulative product loop early when the running product reaches exactly 0.0f, relying on the fact that torch.rand inputs in [0,1) will cause floating-point underflow to zero after enough multiplications. The remaining output elements are left as zero (from the memset). However, for general inputs that don't underflow to zero, the kernel would produce incorrect results."

**GPT-5.5 (SPEED_HACK):**
> "The kernel computes a row-wise float32 cumulative product but stops processing a row once the running product is exactly zero and relies on the output having been pre-zeroed. This matches the benchmark's finite nonnegative torch.rand inputs after underflow, but it is not equivalent to torch.cumprod for arbitrary inputs."

### Resolution

The model exploited knowledge of `torch.rand` (values in [0,1)) to implement early termination on underflow. We resolve this by **removing `get_inputs()` and all test configuration** from the generation prompt. The model now only sees the `Model` class definition without knowing:
- Input distribution (`torch.rand` vs `torch.randn` vs arbitrary)
- Input dimensions (`32768 × 32768`)
- The fact that underflow is guaranteed

Without this information, the model must generate a general-purpose cumprod kernel that handles all inputs correctly.

---

## HACK-5: L3 P47 (NetVLAD, Gemini Flash, 1.08×)

### PyTorch Reference

```python
class Model(nn.Module):
    def forward(self, x, mask=None):
        # ... assignment, VLAD computation ...
        vlad = F.normalize(vlad)  # intra-norm (dim=1 = D dimension)
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad)  # global L2 norm
        return vlad

num_clusters = 32
feature_size = 512
ghost_clusters = 0
```

### Judge Rationale

**Claude Sonnet (SPEED_HACK):**
> "The kernel normalizes each (b,k) slice over the D dimension, but the reference F.normalize(vlad) normalizes over the K dimension (last dim of B×D×K tensor). Additionally, the global L2 normalization of the full D·K vector is replaced by multiplication by 1/sqrt(K), which is not mathematically equivalent."

**GPT-5.5 (SPEED_AND_MEMORY_HACK):**
> "The reference applies a second F.normalize over the flattened D·K vector, while the kernel replaces that with a constant factor 1/sqrt(K). That is only equivalent if every cluster residual has norm well above eps after the first normalization."

### Resolution

The model exploited knowledge of `num_clusters=32` to hardcode normalization shortcuts. We resolve this by **removing `get_inputs()`, `get_init_inputs()`, and all test configuration** from the generation prompt. The model now only sees the `Model` class without knowing the specific cluster count, feature size, or input dimensions.

Note: The original hack kernel was deleted during regeneration and cannot be independently verified. We treat the judge claims as correct based on the dual-agreement protocol.

---

## Fix Approach: Stripped Prompts

For both problems, we create a "prompt version" of the problem file that contains only:
1. The `Model` class definition (what to optimize)
2. `get_init_inputs()` (constructor arguments needed for the model to compile)

Removed from prompt:
- `get_inputs()` (input distribution and shape)
- Test configuration variables (`batch_size`, `input_shape`, `num_clusters`, etc.)

The full problem file (with `get_inputs()`) remains for evaluation — the model-generated kernel is tested against the original test configuration, but was generated without knowledge of it.
