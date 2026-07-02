# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        return torch.nn.functional.cross_entropy(predictions, targets)

batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]

def get_init_inputs():
    return []
