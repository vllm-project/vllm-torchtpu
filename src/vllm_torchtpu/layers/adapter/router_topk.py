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

from vllm_torchtpu.kernels.router_topk import router_topk, router_topk_pick
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
_pick_op = pallas.jax_op("pallas::moe_router_topk_pick", router_topk_pick)
_pick_op.register_fake(lambda scores, values, topk: _fake_router_topk(scores, topk))


def rowmax_select(scores: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort-free ``topk`` over the expert axis of ``[rows, experts]`` scores."""
    if scores.device.type != "tpu":
        return rowmax_topk(scores, topk)
    if scores.dim() != 2:
        raise ValueError(
            f"router top-k kernel expects [rows, experts], got {scores.shape}"
        )
    return _op(scores.float(), topk)


def rowmax_select_pick(
    scores: torch.Tensor, values: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`rowmax_select` of `scores`, returning `values` at the picked experts.

    On TPU the kernel picks them, which saves the XLA gather (a scalar-indexed
    fusion at decode sizes) and the relayouts around it.
    """
    if scores.device.type != "tpu":
        weights, ids = rowmax_topk(scores, topk)
        picked = values.gather(-1, ids.long())
        # As the kernel: a row with no real score keeps NaN weights.
        return torch.where(weights.isnan(), weights, picked), ids
    if scores.dim() != 2:
        raise ValueError(
            f"router top-k kernel expects [rows, experts], got {scores.shape}"
        )
    return _pick_op(scores.float(), values.float(), topk)
