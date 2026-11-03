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
"""Shared CustomOp registration, validation, and cache helpers for Mamba layers."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from torch import nn
from torch_tpu._internal import pallas
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:

    class MambaAttentionMetadata(Protocol):
        """Per-layer AttentionMetadata fields read by TPU Mamba custom ops."""

        block_tables: torch.Tensor
        seq_lens: torch.Tensor
        query_start_loc: torch.Tensor
        request_distribution: torch.Tensor
        mamba_state_indices: torch.Tensor | None


# Shared across layers with identical static configs so JaxCallable traces and
# lowers one XLA graph per unique configuration rather than once per layer.
_shared_mamba_ops: dict[tuple[str, tuple[Any, ...]], Any] = {}
_shared_mamba_op_ids = itertools.count()


def get_shared_mamba_jax_op(
    op_prefix: str,
    key: tuple[Any, ...],
    wrapped_fn: Callable[..., Any],
    out_dim: int,
    *,
    donate_argnums: tuple[int, ...] = (1, 2),
) -> Callable[..., torch.Tensor]:
    """Build or reuse a shared ``pallas.jax_op`` and wrap donated state writeback."""
    cache_key = (op_prefix, key)
    mamba_jax_op = _shared_mamba_ops.get(cache_key)
    if mamba_jax_op is None:
        op_name = f"pallas::{op_prefix}_{next(_shared_mamba_op_ids)}"
        mamba_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            donate_argnums=donate_argnums,
        )

        def _fake_mamba(
            projected_states: torch.Tensor,
            state0: torch.Tensor,
            state1: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            num_tokens = projected_states.size(0)
            return (
                torch.empty_like(state0),
                torch.empty_like(state1),
                torch.empty(
                    (num_tokens, out_dim),
                    dtype=projected_states.dtype,
                    device=projected_states.device,
                ),
            )

        mamba_jax_op.register_fake(_fake_mamba)
        _shared_mamba_ops[cache_key] = mamba_jax_op

    def _mamba_impl(
        projected_states: torch.Tensor,
        state0: torch.Tensor,
        state1: torch.Tensor,
        *rest: Any,
    ) -> torch.Tensor:
        new_state0, new_state1, outputs = mamba_jax_op(
            projected_states,
            state0,
            state1,
            *rest,
        )
        state0.copy_(new_state0)
        state1.copy_(new_state1)
        return outputs

    return _mamba_impl


def validate_tpu_mamba_config(
    layer_name: str,
    *,
    cache_config: CacheConfig | None,
    vllm_config: VllmConfig | None = None,
    num_spec: int = 0,
    use_replayssm: bool = False,
) -> None:
    """Validate unsupported TPU Mamba configurations at layer construction."""
    if cache_config is not None and (
        cache_config.enable_prefix_caching
        or cache_config.mamba_cache_mode in ("all", "align")
    ):
        raise NotImplementedError(
            f"Prefix caching is not yet supported for {layer_name} on TPU."
        )
    if num_spec > 0 or (
        vllm_config is not None and vllm_config.speculative_config is not None
    ):
        raise NotImplementedError(
            f"Speculative decoding is not yet supported for {layer_name} on TPU."
        )
    if (
        vllm_config is not None
        and vllm_config.parallel_config is not None
        and vllm_config.parallel_config.prefill_context_parallel_size > 1
    ):
        raise NotImplementedError(
            f"Prefill context parallelism (PCP > 1) is not yet supported for "
            f"{layer_name} on TPU."
        )
    if use_replayssm:
        raise NotImplementedError(
            f"--use-replayssm is not supported for {layer_name} on TPU."
        )


def unpack_mamba_kv_cache(
    kv_cache: tuple[torch.Tensor, ...] | torch.Tensor | None,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Unpack the 2-tensor Mamba state cache, or return ``None`` during warmup."""
    if kv_cache is None:
        return None
    if isinstance(kv_cache, torch.Tensor) or len(kv_cache) != 2:
        count = 1 if isinstance(kv_cache, torch.Tensor) else len(kv_cache)
        raise NotImplementedError(
            f"Unified single-tensor Mamba block pool is not yet supported for "
            f"{layer_name} on TPU (expected 2 state tensors, got {count})."
        )
    state0, state1 = kv_cache
    if state0.numel() == 0:
        return None
    return state0, state1


def extract_mamba_step_metadata(
    prefix: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract ``(state_indices, query_start_loc, distribution, seq_lens)``."""
    fc = get_forward_context()
    attn_metadata = cast("dict[str, MambaAttentionMetadata]", fc.attn_metadata)[prefix]

    if attn_metadata.mamba_state_indices is not None:
        state_indices = attn_metadata.mamba_state_indices.to(torch.int32)
    else:
        max_reqs = attn_metadata.seq_lens.shape[0]
        max_blocks_per_req = attn_metadata.block_tables.shape[0] // max_reqs
        block_tables_2d = torch.reshape(
            attn_metadata.block_tables,
            (max_reqs, max_blocks_per_req),
        )
        state_indices = block_tables_2d[:, 0].to(torch.int32)

    request_distribution = attn_metadata.request_distribution
    mamba_request_distribution = getattr(
        attn_metadata, "mamba_request_distribution", None
    )
    if mamba_request_distribution is not None:
        request_distribution = mamba_request_distribution

    return (
        state_indices,
        attn_metadata.query_start_loc,
        request_distribution,
        attn_metadata.seq_lens,
    )


def set_param_weight_loader(param: nn.Parameter, loader: Any) -> None:
    """Bind a custom ``weight_loader`` onto a vLLM or plain PyTorch parameter."""
    if isinstance(param, BasevLLMParameter):
        param.weight_loader = loader
    else:
        if hasattr(param, "weight_loader"):
            delattr(param, "weight_loader")
        set_weight_attrs(param, {"weight_loader": loader})
