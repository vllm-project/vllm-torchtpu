# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU reference for the MXFP8 matmul kernel, which needs a TPU."""

import torch


def mxfp8_dequantized(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The bf16 ``[n_in, n_out]`` matrix an MXFP8 linear multiplies by."""
    n_in, n_out = weight.shape
    blocks = weight.float().view(n_in // 32, 32, n_out) * scale.float()[:, None, :]
    return blocks.view(n_in, n_out).to(torch.bfloat16)


def mxfp8_matmul_ref(x, w, scale):
    return x @ mxfp8_dequantized(w, scale)
