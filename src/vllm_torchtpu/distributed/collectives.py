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
"""Collectives over a vLLM group that its GroupCoordinator does not provide."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


def all_to_all(x: torch.Tensor, group: "GroupCoordinator") -> torch.Tensor:
    """Block `d` of `x`'s leading axis goes to rank `d`; block `r` of the result
    came from rank `r`."""
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(x.contiguous(), None, None, group.device_group)
    if isinstance(out, funcol.AsyncCollectiveTensor):
        return funcol.wait_tensor(out)
    return out


def quantized_all_gather(x: torch.Tensor, group: "GroupCoordinator") -> torch.Tensor:
    """All-gather `x` along its leading axis as per-row FP8 with float32 scales.

    Args:
        x: Activations this rank holds, rows on the leading axis.
        group: Group whose `all_gather` joins the ranks' rows.

    Returns:
        The gathered rows, restored to `x.dtype`.
    """
    # Lazy import: importing layers.core.quantization runs layers/__init__.py,
    # which loads cp_mla_attention -> distributed.collectives (import cycle).
    from vllm_torchtpu.layers.core.quantization import (  # noqa: PLC0415
        dequantize_tensor,
        quantize_tensor,
    )

    # Promote float16 to float32 before quantizing so small per-row scales do
    # not overflow 1.0 / scale to inf (turning zeros into NaN) or underflow.
    work_x = x.to(torch.float32) if x.dtype == torch.float16 else x
    quantized, scale = quantize_tensor(work_x, quant_dtype=torch.float8_e4m3fn, axis=-1)
    # The FP8 payload travels as int8, which every collective group carries.
    rows = group.all_gather(quantized.view(torch.int8), dim=0)
    scales = group.all_gather(scale, dim=0)
    return dequantize_tensor(
        rows.view(torch.float8_e4m3fn), scales, axis=-1, out_dtype=x.dtype
    )
