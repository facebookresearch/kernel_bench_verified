# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return F.gelu(x, approximate='tanh')


def get_inputs():
    # randomly generate input tensors based on the model architecture
    x = torch.randn(1024, 1024).cuda()
    return [x]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []


