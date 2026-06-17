# KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?

**Extended evaluation framework for LLM-generated GPU kernels with realistic baselines and robust correctness validation.**

Yunxiang Zhang<sup>1</sup>, Ping Yu<sup>2</sup>, Jianyu Wang<sup>1</sup>, Max (Xiangjun) Fan<sup>1</sup>, Julian Reed<sup>3</sup>, Azalia Mirhoseini<sup>3</sup>, Will Su<sup>1</sup>

<sup>1</sup>Meta &nbsp;&nbsp; <sup>2</sup>FAIR at Meta SuperIntelligence Lab &nbsp;&nbsp; <sup>3</sup>Stanford University

[![Leaderboard](https://img.shields.io/badge/Leaderboard-yunx--z.github.io-blue)](https://yunx-z.github.io/KernelBench-Verified)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)

---

Recent large language models (LLMs) can generate custom CUDA kernels that appear to outperform PyTorch on benchmarks such as KernelBench. Building upon this foundational framework, we demonstrate that frontier models frequently engage in reward hacking to artificially inflate reported performance.

We introduce **KernelBench-Verified**, an extended evaluation framework that incorporates:
1. **TF32-enabled baseline** - Realistic performance measurement with Tensor Core acceleration
2. **Four-distribution hidden test suite** - Robust correctness validation across varied inputs
3. **Memory efficiency metrics** - Capturing the speed-memory tradeoff in kernel optimization

Under verified evaluation with seven frontier LLMs, GPT-5.5 achieves **0.88×** geometric mean speedup, significantly lower than the 1.43× speedup observed under standard evaluation. No model consistently outperforms PyTorch when evaluated against realistic baselines.

## Framework Components

### TF32 Baseline Configuration

Enable TF32 in PyTorch to match practitioner deployment:

```python
# Enable TF32 acceleration in PyTorch
torch.set_float32_matmul_precision('high')
# Equivalent: torch.backends.cuda.matmul.allow_tf32 = True
```

This routes all float32 matmul and convolution operations through Tensor Cores, providing the realistic baseline against which speedups should be measured.

### Multi-Distribution Hidden Test Suite

Each problem has a hidden test file at `hidden_tests/level{L}/{pid}_hidden.py` defining `get_hidden_inputs()` that returns four distributions. A kernel must pass **all four distributions** to be considered correct.

| Distribution | Transform | Catches |
|--------------|-----------|---------|
| **D1** | Original (×1.0) | Baseline correctness |
| **D2** | Scale ×3.0 | Overflow, precision issues |
| **D3** | Scale ×0.01 | Underflow, epsilon issues |
| **D4** | Negate ×(-1.0) | Sign shortcuts, identity tricks |

### Input-Blind Generation

For 4 problems susceptible to reward hacking, test inputs are automatically stripped from the generation prompt. The model sees only the `Model` class with shape docstrings, not the actual test inputs.

**Configured in `src/prompt_constructor.py`:**
```python
STRIP_TEST_CONFIG_PIDS = {
    (1, 90),   # cumprod - exploits torch.rand underflow
    (1, 100),  # hinge loss - exploits i.i.d. broadcasting
    (3, 35),   # LSTM - exploits non-deterministic init
    (3, 47),   # NetVLAD - exploits known dimensions
}
```

**What the model sees (Example: Problem 100 - Hinge Loss):**
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

Removed from prompt: `get_inputs()`, `get_init_inputs()`, `batch_size=32768`, `torch.rand(...)`.

No special flags needed — stripping is automatic during `generate_samples.py`.

## Installation

```bash
# Clone the repository
git clone https://github.com/facebookresearch/kernel_bench_verified.git
cd kernel_bench_verified

# Create conda environment
conda create -n kernel-bench python=3.10
conda activate kernel-bench

# Install dependencies
pip install -r requirements.txt

# Set API keys (for OpenAI, Anthropic, etc.)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
# ... other provider keys as needed
```

## Usage

### Full Evaluation Pipeline

```bash
# 1. Generate kernels (5 samples per problem)
python scripts/generate_samples.py \
  run_name=gpt-5.5_level1_test \
  dataset_src=local \
  level=1 \
  num_samples=5 \
  server_type=openai \
  model_name=gpt-5.5 \
  max_tokens=32000 \
  temperature=0.8 \
  num_workers=4

# 2. Standard evaluation (correctness + timing + memory)
python scripts/eval_from_generations.py \
  run_name=gpt-5.5_level1_test \
  dataset_src=local \
  level=1 \
  num_samples=5 \
  eval_mode=local \
  gpu_arch="['Hopper']" \
  num_gpu_devices=8 \
  timeout=600 \
  build_cache=True \
  num_cpu_workers=1 \
  precision=fp32 \
  measure_performance=True

# 3. Hidden evaluation (4-distribution correctness gating)
python scripts/eval_from_generations.py \
  run_name=gpt-5.5_level1_test \
  dataset_src=local \
  level=1 \
  num_samples=5 \
  eval_mode=local \
  gpu_arch="['Hopper']" \
  num_gpu_devices=8 \
  timeout=600 \
  build_cache=True \
  num_cpu_workers=1 \
  precision=fp32 \
  use_hidden_tests=True \
  measure_performance=False

# 4. Generate leaderboard with verified metrics
python scripts/generate_leaderboard.py \
  --use_hidden_eval \
  --baseline baseline_time_torch_tf32 \
  --fp32_tolerance 1e-3 \
  --out leaderboard.html
```

### Key Flags

- `--use_hidden_tests`: Enable 4-distribution hidden correctness testing (outputs `eval_results_hidden.json`)
- `--use_hidden_eval`: Apply hidden eval gating in leaderboard (only kernels passing all 4 distributions count as correct)
- `--baseline baseline_time_torch_tf32`: Use TF32-enabled PyTorch baseline (realistic performance)
- `--fp32_tolerance 1e-3`: FP32 numerical tolerance for correctness checking

### Generate Hidden Tests

```bash
# Regenerate hidden tests for all Level 1 problems
python scripts/generate_hidden_inputs.py --level 1

# Regenerate for a single problem
python scripts/generate_hidden_inputs.py --level 1 --pid 90
```

### Adding New Problems to Input-Blind List

1. Add `(level, pid)` to `STRIP_TEST_CONFIG_PIDS` in `src/prompt_constructor.py`
2. Add shape annotations to the problem's `forward()` docstring in `KernelBench/level{L}/{problem}.py`
3. Regenerate stripped prompts and re-evaluate

## Output Files

- `runs/{run_name}/eval_results.json` — Standard evaluation (correctness, runtime, memory)
- `runs/{run_name}/eval_results_hidden.json` — Hidden evaluation (4-distribution gated correctness)
- `leaderboard.html` — Interactive HTML leaderboard with verified metrics

## Citation

If you use KernelBench-Verified in your research, please cite:

```bibtex
@article{zhang2026kernelbenchverified,
  title={KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?},
  author={Zhang, Yunxiang and Yu, Ping and Wang, Jianyu and Fan, Max (Xiangjun) and Reed, Julian and Mirhoseini, Azalia and Su, Will},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This source code is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. See the LICENSE file for details.

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

## Acknowledgments

KernelBench-Verified builds upon the original [KernelBench](https://github.com/ScalingIntelligence/KernelBench) benchmark. We thank the KernelBench authors for their foundational work on LLM-generated kernel evaluation.
