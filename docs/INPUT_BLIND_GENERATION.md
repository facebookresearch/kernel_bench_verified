<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->

# Input-Blind Generation

For problems susceptible to reward hacking, KernelBench-Verified automatically strips test inputs from the generation prompt. The model sees only the `Model` class with shape docstrings, not the actual test inputs or configurations.

## Why Input-Blind Generation?

Frontier models can exploit the specific values of test inputs to artificially inflate performance. For example:
- A model might detect that all test inputs are positive and return the input unchanged for ReLU (since ReLU(x) = x for x ≥ 0)
- A model might hardcode the batch size or tensor dimensions observed in the test configuration
- A model might exploit known random seed values or distribution parameters

By removing test inputs from the prompt, we force models to implement the actual algorithm rather than hardcoding bypasses for specific test cases.

## Configured Problems

The following 4 problems are configured for input-blind generation in `src/prompt_constructor.py`:

```python
STRIP_TEST_CONFIG_PIDS = {
    (1, 90),   # cumprod - exploits torch.rand underflow
    (1, 100),  # hinge loss - exploits i.i.d. broadcasting
    (3, 35),   # LSTM - exploits non-deterministic init
    (3, 47),   # NetVLAD - exploits known dimensions
}
```

### Problem Details

**Level 1, Problem 90: Cumulative Product (cumprod)**
- **Vulnerability:** Exploits torch.rand underflow (values in [0,1) multiplied many times underflow to 0)
- **Bypass:** Model can return zeros without computing cumprod

**Level 1, Problem 100: Hinge Loss**
- **Vulnerability:** Exploits i.i.d. broadcasting properties
- **Bypass:** Model can exploit known batch size and class dimensions

**Level 3, Problem 35: LSTM**
- **Vulnerability:** Exploits non-deterministic initialization
- **Bypass:** Model can hardcode initialization patterns

**Level 3, Problem 47: NetVLAD (No Ghost Clusters)**
- **Vulnerability:** Exploits known dimensions
- **Bypass:** Model can hardcode cluster centers and dimensions

## What the Model Sees

### Standard Prompt (with test inputs)
The model receives the full problem file including:
- `Model` class definition
- `get_inputs()` function with actual test tensors
- `get_init_inputs()` for initialization
- Batch size, dimensions, and other configuration
- Example: `batch_size=32768`, `torch.rand(32768, 1000)`

### Input-Blind Prompt (stripped)
The model receives only:
- `Model` class definition with shape docstrings
- No `get_inputs()` or `get_init_inputs()` functions
- No batch size or concrete dimensions
- No `torch.rand()` calls or specific values

**Example: Problem 100 (Hinge Loss) - What the Model Sees**

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        """
        Args:
            predictions (torch.Tensor): shape (batch_size, num_classes), float32.
            targets (torch.Tensor): shape (batch_size,), float32.
        Returns:
            torch.Tensor: Scalar mean hinge loss.
        """
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))
```

**Removed from prompt:**
- `get_inputs()` function
- `get_init_inputs()` function  
- `batch_size=32768`
- `num_classes=1000`
- `torch.rand(32768, 1000)` and `torch.rand(32768)`
- Any concrete tensor values or dimensions

The model must implement the hinge loss computation based solely on the shape annotations in the docstring, without knowing the actual test inputs.

## How It Works

Input-blind generation is **automatic** — no special flags needed.

1. During `generate_samples.py`, the prompt constructor checks if `(level, pid)` is in `STRIP_TEST_CONFIG_PIDS`
2. If yes, it calls `strip_test_config()` to remove test inputs from the prompt
3. The stripped prompt (Model class + docstrings only) is sent to the LLM
4. Evaluation uses the **full** problem file unchanged — the model must work for the actual test inputs

## Adding New Problems

To add a new problem to the input-blind list:

1. **Identify vulnerability:** Determine if the problem is susceptible to reward hacking (e.g., exploits specific input values, dimensions, or distributions)

2. **Add to configuration:** Edit `src/prompt_constructor.py` and add `(level, pid)` to `STRIP_TEST_CONFIG_PIDS`:
   ```python
   STRIP_TEST_CONFIG_PIDS = {
       (1, 90),    # existing
       (1, 100),   # existing
       (3, 35),    # existing
       (3, 47),    # existing
       (2, 42),    # NEW: your problem here
   }
   ```

3. **Add shape annotations:** Edit the problem file at `KernelBench/level{L}/{problem}.py` and ensure the `forward()` method has complete shape annotations in its docstring:
   ```python
   def forward(self, x, y):
       """
       Args:
           x (torch.Tensor): shape (batch_size, seq_len, hidden_dim), float32.
           y (torch.Tensor): shape (batch_size, hidden_dim), float32.
       Returns:
           torch.Tensor: shape (batch_size, seq_len), float32.
       """
   ```

4. **Regenerate stripped prompt:** Run the regeneration script to create the stripped prompt file:
   ```bash
   # The stripped prompt will be saved to stripped_prompts/{pid}_{name}_prompt.py
   python -c "
   from src.prompt_constructor import get_stripped_prompt
   prompt = get_stripped_prompt(level=2, pid=42)
   with open('stripped_prompts/42_YourProblem_prompt.py', 'w') as f:
       f.write(prompt)
   "
   ```

5. **Test:** Regenerate kernels for the problem and verify they still pass correctness checks with the stripped prompt.

## Stripped Prompt Files

Pre-generated stripped prompts are stored in `stripped_prompts/`:

- `90_cumprod_prompt.py` - Level 1, Problem 90 (cumulative product)
- `100_HingeLoss_prompt.py` - Level 1, Problem 100 (hinge loss)
- `35_LSTM_prompt.py` - Level 3, Problem 35 (LSTM)
- `47_NetVladNoGhostClusters_prompt.py` - Level 3, Problem 47 (NetVLAD)

These files contain the exact prompts sent to LLMs for input-blind generation.
