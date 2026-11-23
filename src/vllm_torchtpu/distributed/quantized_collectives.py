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
"""All-gather that sends activations as per-row FP8 over a vLLM group.

Kept apart from `collectives.py`: importing `layers.core.quantization` runs the
`vllm_torchtpu.layers` package `__init__`, which imports `collectives` through
`cp_mla_attention`, so `collectives` cannot import it at module top.
"""

from typing import TYPE_CHECKING

import torch

from vllm_torchtpu.layers.core.quantization import dequantize_tensor, quantize_tensor

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

# Gather runs on dim 0 while per-row scales reduce on dim -1, so inputs
# must have at least two distinct axes.
_MIN_QUANTIZED_ALL_GATHER_NDIM = 2


def quantized_all_gather(
    x: torch.Tensor, group: "GroupCoordinator", *, num_rows: int | None = None
) -> torch.Tensor:
    """All-gather `x` along its leading axis as per-row FP8 with float32 scales.

    Args:
        x: Activations this rank holds, rows on the leading axis (`ndim >= 2`).
        group: Group whose `all_gather` joins the ranks' rows.
        num_rows: Keep only this many leading rows, dropping gather padding.

    Returns:
        The gathered rows, restored to `x.dtype`.
    """
    if x.ndim < _MIN_QUANTIZED_ALL_GATHER_NDIM:
        msg = (
            f"quantized_all_gather expects x.ndim >= "
            f"{_MIN_QUANTIZED_ALL_GATHER_NDIM}, got shape={tuple(x.shape)}."
        )
        raise ValueError(msg)
    quantized, scale = quantize_tensor(x, quant_dtype=torch.float8_e4m3fn, axis=-1)
    # The FP8 payload travels as int8, which every collective group carries.
    rows = group.all_gather(quantized.view(torch.int8), dim=0)
    scales = group.all_gather(scale, dim=0)
    if num_rows is not None:
        rows = rows.narrow(0, 0, num_rows)
        scales = scales.narrow(0, 0, num_rows)
    return dequantize_tensor(
        rows.view(torch.float8_e4m3fn), scales, axis=-1, out_dtype=x.dtype
    )
