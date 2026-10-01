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
"""The FP8 MoE method's dispatch to the fused EP kernel v3.

`VllmFp8MoEMethodTPU` calls into this module at three seams, as it does for
the adaptive kernel: `prebuild` at weight load, `supported` in
`supports_internal_mk`, and `apply` in `apply_monolithic`. A layer v3 declines
keeps the method's grouped-matmul path and vLLM's own dispatch and combine; an
armed layer runs that path below its per-step token gate (`apply`).
"""

from collections.abc import Callable

import torch
from vllm.distributed.parallel_state import get_ep_group

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.fused_moe_ep_v3 import (
    FUSED_MOE_EP_V3_OP_ATTR,
    fused_moe_ep_v3,
    fused_moe_ep_v3_supported,
    prebuild_fused_moe_ep_v3,
)
from vllm_torchtpu.layers.adapter.sequence_parallel import rows_are_sliced


def enabled() -> bool:
    """Whether the operator selected v3 (MOE_FUSED_EP_KERNEL_IMPL=v3)."""
    return envs.MOE_FUSED_EP_KERNEL_IMPL == "v3"


def prebuild(method, layer, activation: str) -> None:
    """Build the v3 op for `layer` and record it on its quant method."""
    op = prebuild_fused_moe_ep_v3(
        layer,
        topk=layer.moe_config.experts_per_token,
        renormalize=layer.renormalize,
        activation=activation,
    )
    setattr(method, FUSED_MOE_EP_V3_OP_ATTR, op)


def supported(method) -> bool:
    """Whether v3 serves this method's layer, and so owns its exchange."""
    return fused_moe_ep_v3_supported(method)


def apply(
    method,
    layer,
    x: torch.Tensor,
    router_logits: torch.Tensor,
    grouped: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """The layer's forward when v3 owns its exchange: this rank's tokens in
    and out.

    A step of at least the armed op's `step_min_tokens` rows of this rank's
    own runs v3; under TP only a sequence-parallel step's rows are the rank's
    own, a smaller step's being replicated on every rank. Any other step runs
    `grouped` (the method's grouped-matmul path) between the dispatch and
    combine vLLM's MoERunner would have issued had the method not claimed
    them. The fork reads the token count, so it leaves a
    shape guard and each side is its own trace; both sides lift the same
    buffers (the weights, `experts_start`, the padding mask), so a bucket
    compiled by one trace can be handed to the other.
    """
    armed = getattr(method, FUSED_MOE_EP_V3_OP_ATTR)
    sequence_parallel = layer.moe_config.moe_parallel_config.is_sequence_parallel
    own_rows = not sequence_parallel or rows_are_sliced(sequence_parallel)
    if own_rows and x.shape[0] >= armed.step_min_tokens:
        return fused_moe_ep_v3(
            method,
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            router_logits,
            layer._experts_start,
        )
    ep = get_ep_group()
    x, router_logits = ep.dispatch_router_logits(x, router_logits, sequence_parallel)
    return ep.combine(grouped(layer, x, router_logits), sequence_parallel)
