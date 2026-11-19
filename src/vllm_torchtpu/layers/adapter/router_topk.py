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

from vllm_torchtpu.kernels.router_topk import router_topk, router_topk_biased
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


def _fake_router_topk_biased(scores: torch.Tensor, bias: torch.Tensor, topk: int):
    return _fake_router_topk(scores, topk)


_op_biased = pallas.jax_op("pallas::moe_router_topk_biased", router_topk_biased)
_op_biased.register_fake(_fake_router_topk_biased)


def rowmax_select_biased(
    scores: torch.Tensor, bias: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(scores.gather(ids), ids)`` with ``ids`` the top-k of ``scores +
    bias``, in one kernel pass (TPU_ROUTER_TOPK_FUSED_BIAS_MIN_ROWS); None
    when the call does not qualify (the caller takes the unfused path)."""
    from vllm_torchtpu import envs

    min_rows = int(envs.TPU_ROUTER_TOPK_FUSED_BIAS_MIN_ROWS)
    if (
        min_rows <= 0
        or scores.device.type != "tpu"
        or scores.dim() != 2
        or scores.shape[0] < min_rows
        or scores.shape[0] % 128
    ):
        return None
    global _biased_logged
    if not _biased_logged:
        _biased_logged = True
        print(
            "TPU_ROUTER_TOPK_FUSED_BIAS_MIN_ROWS=1: router biased top-k and "
            "weight gather in one kernel",
            flush=True,
        )
    return _op_biased(scores.float(), bias.float().reshape(-1), topk)


_biased_logged = False


def rowmax_select(scores: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort-free ``topk`` over the expert axis of ``[rows, experts]`` scores."""
    if scores.device.type != "tpu":
        return rowmax_topk(scores, topk)
    if scores.dim() != 2:
        raise ValueError(
            f"router top-k kernel expects [rows, experts], got {scores.shape}"
        )
    return _op(scores.float(), topk)
