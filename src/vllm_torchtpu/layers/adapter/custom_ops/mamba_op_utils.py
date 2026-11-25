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
"""Shared CustomOp cache and step-metadata helpers for recurrent TPU layers."""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from torch_tpu._internal import pallas
from vllm.forward_context import get_forward_context

if TYPE_CHECKING:

    class MambaAttentionMetadata(Protocol):
        """Read-only per-layer AttentionMetadata fields used by recurrent ops."""

        @property
        def seq_lens(self) -> torch.Tensor: ...

        @property
        def query_start_loc(self) -> torch.Tensor: ...

        @property
        def request_distribution(self) -> torch.Tensor: ...

        @property
        def block_tables(self) -> torch.Tensor | None: ...

        @property
        def mamba_state_indices(self) -> torch.Tensor | None: ...

        @property
        def mamba_request_distribution(self) -> torch.Tensor | None: ...


# Shared across recurrent layers (GDN here, plus standalone Mamba layers in PR
# #3422) with identical static configs so JaxCallable traces and lowers one XLA
# graph per unique configuration rather than once per layer.
_shared_mamba_ops: dict[
    tuple[str, tuple[Any, ...], tuple[int, ...], tuple[int, ...]], Any
] = {}
_shared_mamba_op_ids = itertools.count()


def get_shared_mamba_jax_op(
    op_prefix: str,
    key: tuple[Any, ...],
    wrapped_fn: Callable[..., Any],
    out_shape_tail: tuple[int, ...] | int,
    *,
    donate_argnums: tuple[int, ...],
) -> Callable[..., torch.Tensor]:
    """Build or reuse a shared ``pallas.jax_op`` and wrap donated state writeback.

    ``op_prefix`` and ``key`` together must cover every value that
    ``wrapped_fn`` closes over. ``out_shape_tail`` and ``donate_argnums`` are
    included in the cache key automatically.

    Important contract for callers:
      - ``donate_argnums`` is required because pooled recurrent layers donate a
        single unified pool tensor (e.g. ``(3,)``) whereas standalone 2-state
        Mamba layers (PR #3422) donate ``(conv_state, ssm_state)`` (``(1, 2)``).
      - Keep all optional (``None``-able) tensor operands **after** every
        donated operand in ``wrapped_fn``'s positional signature. ``None``
        arguments are dropped across the TorchTPU ``pallas.jax_op`` boundary,
        which shifts positional indices of subsequent arguments and would
        otherwise misalign ``donate_argnums``.
      - ``wrapped_fn`` must return ``(*updated_donated_states, output)`` in the
        same order as ``donate_argnums``.
    """
    wrapped_sig = inspect.signature(wrapped_fn)
    min_pos = 0
    max_pos: int | None = 0
    for param in wrapped_sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            max_pos = None
        elif param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if param.default is inspect.Parameter.empty:
                min_pos += 1
            if max_pos is not None:
                max_pos += 1
        elif (
            param.kind == inspect.Parameter.KEYWORD_ONLY
            and param.default is inspect.Parameter.empty
        ):
            raise TypeError(
                f"{op_prefix} wrapped_fn has required keyword-only "
                f"parameter {param.name!r}."
            )

    tail = (out_shape_tail,) if isinstance(out_shape_tail, int) else out_shape_tail
    cache_key = (op_prefix, key, tail, donate_argnums)
    jax_op = _shared_mamba_ops.get(cache_key)
    if jax_op is None:
        op_name = f"pallas::{op_prefix}_{next(_shared_mamba_op_ids)}"
        jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            donate_argnums=donate_argnums,
        )

        def _fake_recurrent_op(*args: Any, **_kwargs: Any) -> tuple[torch.Tensor, ...]:
            first_tensor = cast(torch.Tensor, args[0])
            num_tokens = first_tensor.size(0)
            fake_states = tuple(
                torch.empty_like(cast(torch.Tensor, args[idx]))
                for idx in donate_argnums
            )
            fake_out = torch.empty(
                (num_tokens, *tail),
                dtype=first_tensor.dtype,
                device=first_tensor.device,
            )
            return (*fake_states, fake_out)

        jax_op.register_fake(_fake_recurrent_op)
        _shared_mamba_ops[cache_key] = jax_op

    num_donated = len(donate_argnums)
    if max_pos is None:
        expected_arity = f"at least {min_pos}"
    elif min_pos == max_pos:
        expected_arity = f"{min_pos}"
    else:
        expected_arity = f"{min_pos}..{max_pos}"

    def _shared_op_impl(*args: Any) -> torch.Tensor:
        if len(args) < min_pos or (max_pos is not None and len(args) > max_pos):
            raise TypeError(
                f"{op_prefix} expected {expected_arity} positional operands, "
                f"got {len(args)}."
            )
        result = jax_op(*args)
        # Plain donation + copy_ writeback (aliased in-place by XLA).
        for out_idx, arg_idx in enumerate(donate_argnums):
            cast(torch.Tensor, args[arg_idx]).copy_(result[out_idx])
        return cast(torch.Tensor, result[num_donated])

    return _shared_op_impl


def extract_mamba_step_metadata(
    prefix: str,
    attn_metadata: MambaAttentionMetadata | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract ``(state_indices, query_start_loc, distribution, seq_lens)``."""
    if attn_metadata is None:
        fc = get_forward_context()
        attn_metadata = cast("dict[str, MambaAttentionMetadata]", fc.attn_metadata)[
            prefix
        ]

    # Recurrent-state slot id per persistent-batch position.
    # Compact-mamba: the mamba pool has only `_mamba_num_blocks` slots
    # (< attention `num_blocks`), so the slot id is carried explicitly in
    # `mamba_state_indices` (in [0, _mamba_num_blocks)) rather than derived
    # from the attention `block_tables[:, 0]`. Unified pools likewise populate
    # `mamba_state_indices`. Fall back to `block_tables[:, 0]` only when
    # compact sizing was skipped and mamba shares the attention block pool
    # (uniform layout).
    if attn_metadata.mamba_state_indices is not None:
        state_indices = attn_metadata.mamba_state_indices.to(torch.int32)
    else:
        assert attn_metadata.block_tables is not None
        max_reqs = attn_metadata.seq_lens.shape[0]
        max_blocks_per_req = attn_metadata.block_tables.shape[0] // max_reqs
        block_tables_2d = torch.reshape(
            attn_metadata.block_tables,
            (max_reqs, max_blocks_per_req),
        )
        state_indices = block_tables_2d[:, 0].to(torch.int32)

    # Speculative decoding: the recurrent kernel's windowed segment covers both
    # 1-token decodes and speculative verify windows (the batch is ordered
    # [decode][verify][prefill/mixed]); ragged paged attention keeps using
    # `request_distribution` with its 1-token decode front segment.
    request_distribution = attn_metadata.request_distribution
    mamba_request_distribution = attn_metadata.mamba_request_distribution
    if mamba_request_distribution is not None:
        request_distribution = mamba_request_distribution

    return (
        state_indices,
        attn_metadata.query_start_loc,
        request_distribution,
        attn_metadata.seq_lens,
    )
