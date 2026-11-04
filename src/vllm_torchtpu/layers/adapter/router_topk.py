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
"""Torch bridge for the sort-free MoE router top-k.

On TPU the selection runs as the Pallas kernel in ``kernels/router_topk.py``,
elsewhere as the identical plain-torch algorithm in
``layers/core/rowmax_topk.py``.
"""

import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.router_topk import router_topk, router_topk_bias
from vllm_torchtpu.layers.core.rowmax_topk import rowmax_topk


def _fake_router_topk(scores: torch.Tensor, topk: int):
    rows = scores.shape[0]
    return (
        torch.empty((rows, topk), dtype=torch.float32, device=scores.device),
        torch.empty((rows, topk), dtype=torch.int32, device=scores.device),
    )


# Registered at import, not lazily: this op is called from inside the MoE
# forward, and Dynamo cannot trace a lock-guarded registration there.
_op = pallas.jax_op("pallas::moe_router_topk", router_topk)
_op.register_fake(_fake_router_topk)


def _fake_router_topk_bias(scores: torch.Tensor, bias: torch.Tensor, topk: int):
    return _fake_router_topk(scores, topk)


_bias_op = pallas.jax_op("pallas::moe_router_topk_bias", router_topk_bias)
_bias_op.register_fake(_fake_router_topk_bias)


def rowmax_select(scores: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort-free ``topk`` over the expert axis of ``[rows, experts]`` scores."""
    if scores.device.type != "tpu":
        return rowmax_topk(scores, topk)
    if scores.dim() != 2:
        raise ValueError(
            f"router top-k kernel expects [rows, experts], got {scores.shape}"
        )
    return _op(scores.float(), topk)


def rowmax_select_bias(
    scores: torch.Tensor, bias: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k ids of ``scores + bias``, weights = ``scores`` at those ids.

    Equal to ``ids = rowmax_select(scores + bias, k)[1]; w = scores.gather(
    -1, ids)`` bit for bit, without the gather on TPU.
    """
    if scores.device.type != "tpu":
        _, ids = rowmax_topk(scores + bias.view(1, -1), topk)
        return scores.gather(-1, ids.long()), ids
    if scores.dim() != 2:
        raise ValueError(
            f"router top-k kernel expects [rows, experts], got {scores.shape}"
        )
    return _bias_op(scores.float(), bias.float().reshape(1, -1), topk)
