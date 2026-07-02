# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, predictions, targets):
        """
        Computes mean hinge loss.

        Args:
            predictions (torch.Tensor): Predictions tensor of shape (batch_size, num_classes), float32.
            targets (torch.Tensor): Targets tensor of shape (batch_size,), values in {-1, +1}, float32.

        Returns:
            torch.Tensor: Scalar mean hinge loss.
        """
        return torch.mean(torch.clamp(1 - predictions * targets, min=0))
