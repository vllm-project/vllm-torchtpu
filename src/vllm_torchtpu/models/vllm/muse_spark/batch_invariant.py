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
"""Batch-invariant dense projections for Muse Spark Open.

XLA picks the K tiling of a dot from the full operand shapes, so the same token
row accumulates in a different order when the step holds a different number of
tokens (up to 1.6e-2 per projection in bf16 on v7x). Over 62 layers that flips
greedy tokens with batch composition. These helpers route Muse projections
through a Pallas matmul whose K tiling depends on the weight shape alone, and
Muse RMS norms through a Pallas row reduction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.models.vllm.muse_spark.det_collectives import det_all_reduce

_LANES = 128


def batch_invariant_enabled(weight: torch.Tensor) -> bool:
    return bool(envs.VLLM_TPU_MUSE_BATCH_INVARIANT) and weight.device.type == "tpu"


def det_projection(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """`layer(x)[0]` for an unquantized, bias-free vLLM linear, batch-invariant."""
    weight = getattr(layer, "weight", None)
    quant_method = getattr(layer, "quant_method", None)
    if (
        not isinstance(weight, torch.Tensor)
        or (
            quant_method is not None
            and not isinstance(quant_method, UnquantizedLinearMethod)
        )
        or not batch_invariant_enabled(weight)
        or getattr(layer, "bias", None) is not None
    ):
        out, _ = layer(x)
        return out
    from vllm_torchtpu.layers.adapter.custom_ops.batch_invariant_ops import det_linear

    # The TPU unquantized method stores [K, N] once loaded; otherwise [N, K].
    flipped = bool(getattr(layer, WEIGHT_FLIPPED_ATTR, False))
    out = det_linear(x, weight, rhs_nk=not flipped)
    tp_size = int(getattr(layer, "tp_size", 1))
    if tp_size > 1:
        if getattr(layer, "reduce_results", False):
            out = det_all_reduce(out)
        elif getattr(layer, "gather_output", False):
            out = tensor_model_parallel_all_gather(out)
    return out


def rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    """`x * rsqrt(mean(x^2, -1) + eps)` for fp32 `x`; batch-invariant on TPU.

    The eager torch_tpu mean reduces a row in an order that depends on the row
    count and offset (4.8e-7 relative), so the TPU path uses a Pallas kernel.
    """
    if not batch_invariant_enabled(x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    from vllm_torchtpu.layers.adapter.custom_ops.batch_invariant_ops import (
        det_rms_normalize,
    )

    return det_rms_normalize(x, eps)


def pad_lm_head_weight(weight: torch.Tensor) -> torch.Tensor:
    """Vocab shard [V, H] padded to a lane multiple of rows (zeros)."""
    v = weight.shape[0]
    v_pad = -(-v // _LANES) * _LANES
    if v_pad == v:
        return weight
    return F.pad(weight, (0, 0, 0, v_pad - v))


def det_lm_head_logits(
    hidden_states: torch.Tensor,
    padded_weight: torch.Tensor,
    shard_vocab: int,
    org_vocab: int,
    tp_size: int,
) -> torch.Tensor:
    """Full vocab bf16 logits; the matmul is batch-invariant, the gather exact."""
    from vllm_torchtpu.layers.adapter.custom_ops.batch_invariant_ops import det_linear

    logits = det_linear(hidden_states, padded_weight, rhs_nk=True)[:, :shard_vocab]
    if tp_size > 1:
        logits = tensor_model_parallel_all_gather(logits.contiguous(), dim=-1)
    return logits[..., :org_vocab]
