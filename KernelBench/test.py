# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

print("Before patch:", torch.randn)                 # built-in

import src.utils      # or: import src.utils
print("After  patch:", torch.randn)                 # <function _randn_patched at 0x…>

print("Module  :", torch.randn.__module__)          # should be 'src.utils'