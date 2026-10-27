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
"""All-gather activations as fp8 rows plus scales."""

from collections.abc import Callable

import torch

from vllm_torchtpu.layers.core.quantization import quantize_tensor


def fp8_all_gather(
    x: torch.Tensor,
    all_gather: Callable[[torch.Tensor], torch.Tensor],
    *,
    block_size: int | None = None,
) -> torch.Tensor:
    """All-gathers `x` along dim 0 as e4m3 rows plus scales.

    Sends half-width fp8 activations plus scales in `x.dtype` and returns
    `x.dtype`. When `block_size` is None, one scale per row is used; a
    non-dividing `block_size` raises `ValueError` via `quantize_tensor`.
    The int8 view is a same-width bitcast for collective backends that do not
    accept fp8 directly, and the scale tensor stays in `x.dtype` to avoid
    32-bit collective and broadcast conversions.
    """
    q, scale = quantize_tensor(x, torch.float8_e4m3fn, axis=-1, block_size=block_size)
    q_deq = all_gather(q.view(torch.int8)).view(torch.float8_e4m3fn).to(x.dtype)
    scale = all_gather(scale.to(x.dtype))
    if block_size is None or block_size == x.shape[-1]:
        return q_deq * scale
    return (q_deq.unflatten(-1, (-1, block_size)) * scale.unsqueeze(-1)).flatten(-2)
