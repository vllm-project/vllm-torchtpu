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
"""Torch custom op for ``layers.core.routed_expert_rows``."""

from collections.abc import Callable
from functools import cache, partial

import torch
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.core.fused_moe_gmm import resolve_onehot_permute_threshold
from vllm_torchtpu.layers.core.routed_expert_rows import routed_expert_rows


def routed_expert_rows_op(use_ep: bool) -> Callable[..., torch.Tensor]:
    """Return ``f(x, ids, w13, w2, ...) -> [tokens, top_k, hidden]``.

    The op is built for the current SparseCore permute settings.
    """
    return _build(use_ep, envs.USE_MOE_SPARSE_CORE, resolve_onehot_permute_threshold())


@cache
def _build(
    use_ep: bool, use_sparse_core: bool, onehot_moe_permute_threshold: int
) -> Callable[..., torch.Tensor]:
    # Build once outside torch.compile; tensor EP offsets keep rank graphs identical.
    # The permute settings are part of the op name, so a program compiled under
    # one setting is never reused under another.
    name = (
        f"pallas::routed_expert_rows_ep{int(use_ep)}"
        f"_sc{int(use_sparse_core)}_oh{onehot_moe_permute_threshold}"
    )
    op = pallas.jax_op(
        name,
        partial(
            routed_expert_rows,
            use_ep=use_ep,
            use_sparse_core=use_sparse_core,
            onehot_moe_permute_threshold=onehot_moe_permute_threshold,
        ),
    )

    def fake(
        x, ids, w13, w2, w13_scale, w2_scale, w13_global, w2_global, experts_start
    ):
        return torch.empty(*ids.shape, x.shape[-1], device=x.device, dtype=x.dtype)

    op.register_fake(fake)

    def run(
        x: torch.Tensor,
        ids: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor | None,
        w2_scale: torch.Tensor | None,
        w13_global: torch.Tensor | None = None,
        w2_global: torch.Tensor | None = None,
        experts_start: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Match the int32 JAX indexing schema at the custom-op boundary.
        return op(
            x,
            ids.to(torch.int32),
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_global,
            w2_global,
            experts_start,
        )

    return run
