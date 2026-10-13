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
"""All-gather activations as fp8 rows plus fp32 scales."""

from collections.abc import Callable

import torch

from vllm_torchtpu.layers.core.quantization import dequantize_tensor, quantize_tensor


def fp8_all_gather(
    x: torch.Tensor,
    all_gather: Callable[[torch.Tensor], torch.Tensor],
    *,
    block_size: int | None = None,
) -> torch.Tensor:
    """All-gathers `x` along dim 0 as e4m3 rows plus fp32 scales.

    Sends about half the bf16 bytes and returns `x.dtype`. `block_size` is the
    channels per scale. When it is None or does not divide the last dim, one
    scale per row is used, because `quantize_tensor` needs an exact split.
    The int8 view is a same-width bitcast, since collectives may not take fp8.
    """
    hidden = x.shape[-1]
    block = block_size if block_size and hidden % block_size == 0 else hidden
    q, scale = quantize_tensor(x, torch.float8_e4m3fn, axis=-1, block_size=block)
    q = all_gather(q.view(torch.int8))
    scale = all_gather(scale)
    return dequantize_tensor(
        q.view(torch.float8_e4m3fn), scale, axis=-1, out_dtype=x.dtype
    )
