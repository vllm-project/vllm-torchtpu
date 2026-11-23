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

from vllm_torchtpu.kernels.router_topk import router_topk, router_topk_fused
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


def _fake_router_topk_fused(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scale: float,
):
    return _fake_router_topk(scores, topk)


_fused_op = pallas.jax_op("pallas::moe_router_topk_fused", router_topk_fused)
_fused_op.register_fake(_fake_router_topk_fused)


def rowmax_select(scores: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort-free ``topk`` over the expert axis of ``[rows, experts]`` scores."""
    if scores.device.type != "tpu":
        return rowmax_topk(scores, topk)
    if scores.dim() != 2:
        raise ValueError(
            f"router top-k kernel expects [rows, experts], got {scores.shape}"
        )
    return _op(scores.float(), topk)


def rowmax_select_fused(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Biased sort-free top-k and the routing-weight epilogue in one call.

    On TPU this is one Pallas call (``kernels/router_topk.select_fused``).
    Off TPU it runs the same steps as plain torch ops, in the order that
    ``moe_routing.select_experts`` uses for the unfused path.

    Args:
      scores: ``[rows, experts]`` routing scores after the scoring function.
      bias: ``[experts]`` expert-selection correction bias.
      topk: number of experts to select per row.
      renormalize: divide the selected weights by their row sum, clamped
        below at ``1e-20``.
      scale: routed scaling factor applied to the final weights.

    Returns:
      ``(weights float32, indices int32)``, both ``[rows, topk]``.
    """
    scores = scores.float()
    bias = bias.float()
    if scores.device.type != "tpu":
        _, ids = rowmax_topk(scores + bias, topk)
        weights = scores.gather(-1, ids.long())
        if renormalize:
            weights = weights / torch.clamp(
                weights.sum(dim=-1, keepdim=True), min=1e-20
            )
        if scale != 1.0:
            weights = weights * scale
        return weights, ids
    if scores.dim() != 2:
        raise ValueError(
            f"router top-k kernel expects [rows, experts], got {scores.shape}"
        )
    return _fused_op(scores, bias, topk, bool(renormalize), float(scale))
