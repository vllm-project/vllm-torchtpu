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

from vllm_torchtpu.kernels.router_topk import router_topk, router_topk_values
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


def _fake_router_topk_values(choice: torch.Tensor, values: torch.Tensor, topk: int):
    return _fake_router_topk(choice, topk)


_values_op = pallas.jax_op("pallas::moe_router_topk_values", router_topk_values)
_values_op.register_fake(_fake_router_topk_values)


def rowmax_select_values(
    choice: torch.Tensor, values: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``rowmax_select`` of ``choice`` with ``values`` (same shape) at the
    picked experts as the weights: ``values.gather(-1, ids)``, from the
    kernel's own picks instead of a separate gather over the scores (a
    SparseCore round trip at prefill sizes)."""
    if choice.device.type != "tpu":
        _, ids = rowmax_topk(choice, topk)
        return values.gather(-1, ids.long()), ids
    if choice.dim() != 2 or values.shape != choice.shape:
        raise ValueError(
            "router top-k kernel expects [rows, experts] choice and values, got "
            f"{choice.shape} and {values.shape}"
        )
    return _values_op(choice.float(), values.float(), topk)


def rowmax_select(scores: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort-free ``topk`` over the expert axis of ``[rows, experts]`` scores."""
    if scores.device.type != "tpu":
        return rowmax_topk(scores, topk)
    if scores.dim() != 2:
        raise ValueError(
            f"router top-k kernel expects [rows, experts], got {scores.shape}"
        )
    return _op(scores.float(), topk)
