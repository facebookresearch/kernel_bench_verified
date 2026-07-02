<!--
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->


This folder includes PyTorch modules paired with CUDA kernels, which are used as in-context examples in KernelBench. 



Acknowledgements:
- Fused GeLU and Tiled Matmul: [Christian Mills, GPU MODE Lecture 04](https://christianjmills.com/posts/cuda-mode-notes/lecture-004/)
- Minimal Flash Attention: [Peter Kim, Minimal Flash Attention](https://github.com/tspeterkim/flash-attention-minimal/tree/main)

There are some examples.
[TODO] Table detailing content and speedups of each example