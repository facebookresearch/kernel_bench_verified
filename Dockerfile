# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

FROM docker.io/nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && ln -sf /usr/bin/python3 /usr/bin/python

# Python deps
RUN pip3 install --no-cache-dir \
    torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir \
    numpy hydra-core omegaconf pyyaml tqdm

WORKDIR /app

# Copy eval code
COPY src/ /app/src/
COPY scripts/eval_from_generations.py /app/scripts/
COPY scripts/benchmark_eval_analysis.py /app/scripts/
COPY scripts/generate_baseline_time.py /app/scripts/

# Copy problem data
COPY KernelBench/ /app/KernelBench/
COPY hidden_tests/ /app/hidden_tests/

# Copy pre-computed baselines
COPY results/timing/ /app/results/timing/

# Copy entrypoint
COPY docker_eval.sh /app/docker_eval.sh
RUN chmod +x /app/docker_eval.sh

ENTRYPOINT ["/app/docker_eval.sh"]
