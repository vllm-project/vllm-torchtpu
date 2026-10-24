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
"""Per-window AttentionMetadata and TokenPaddingState slicing for micro-batching."""

from collections.abc import Callable
from typing import Any

import torch

from vllm_torchtpu.layers.adapter.token_padding import TokenPaddingState
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

_SLICED_METADATA_FIELDS: tuple[str, ...] = (
    "query_start_loc",
    "seq_lens",
)


def bind_step_inputs_for_trace_alignment(
    positions: torch.Tensor | None,
    attn_metadata: Any,
    padding_state: TokenPaddingState | None,
    num_tokens: int | None = None,
) -> tuple[torch.Tensor | None, Callable[[], None]]:
    """Touch step-level tensor inputs with live views before bucket branching.

    Under ``@support_torch_compile`` + ``shape_variants``, small token buckets
    (``< min_tokens``) trace the unsplit path while large buckets
    (``>= min_tokens``) trace the micro-batched path. Dynamo assigns placeholder
    order in ``PiecewiseBackend.graph`` by the order in which tensors are first
    used by live FX nodes (unused nodes are removed by FX dead-code elimination).
    Rebinding ``positions``, ``query_start_loc``, ``seq_lens``, and
    ``padding_state.local_padding_mask`` through identity views/slices before
    the ``num_tokens < min_tokens`` branch ensures both traces emit live uses in
    the exact same order and unify ``positions``'s dynamic token dimension with
    ``hidden_states.shape[0]``.
    """
    restorers: list[tuple[Any, str, Any]] = []
    memo: dict[int, torch.Tensor] = {}

    if isinstance(positions, torch.Tensor):
        if num_tokens is not None:
            if positions.ndim == 2:
                positions = positions.view(positions.shape[0], num_tokens)
            else:
                positions = positions.view(num_tokens, *positions.shape[1:])
        else:
            positions = positions.view(*positions.shape)

    def _bind_md(md: AttentionMetadata) -> None:
        for field_name in _SLICED_METADATA_FIELDS:
            tensor: Any = getattr(md, field_name, None)
            if isinstance(tensor, torch.Tensor):
                restorers.append((md, field_name, tensor))
                t_id = id(tensor)
                if t_id not in memo:
                    memo[t_id] = tensor[:]
                setattr(md, field_name, memo[t_id])

    if isinstance(attn_metadata, AttentionMetadata):
        _bind_md(attn_metadata)
    elif isinstance(attn_metadata, dict):
        seen_md_ids: set[int] = set()
        for layer_md in attn_metadata.values():
            if not isinstance(layer_md, AttentionMetadata):
                continue
            md_id = id(layer_md)
            if md_id in seen_md_ids:
                continue
            seen_md_ids.add(md_id)
            _bind_md(layer_md)

    if padding_state is not None and isinstance(
        padding_state.local_padding_mask, torch.Tensor
    ):
        orig_mask = padding_state.local_padding_mask
        restorers.append((padding_state, "local_padding_mask", orig_mask))
        padding_state.local_padding_mask = orig_mask[:]

    def _restore() -> None:
        for obj, attr, orig_val in reversed(restorers):
            setattr(obj, attr, orig_val)

    return positions, _restore


def slice_attention_metadata(
    attn_metadata: AttentionMetadata,
    window_idx: int,
    window_tokens: int,
    *,
    slice_input_positions: bool = True,
) -> AttentionMetadata:
    """Project ``attn_metadata`` onto token window ``[start, end)``.

    For each sequence ``i`` with step query range ``[qsl[i], qsl[i+1])`` and
    end-of-step total length ``seq_lens[i]``:
    - ``win_qsl[i] = clamp(qsl[i] - start, 0, window_tokens)``
    - ``tokens_after[i] = max(0, qsl[i+1] - end)`` (tokens of sequence ``i``
      scheduled in later windows of the same step)
    - ``win_seq_lens[i] = seq_lens[i] - tokens_after[i]`` for sequences active
      in this window, and ``0`` for sequences with ``win_q_lens[i] == 0``.

    All operations are pure tensor ops on fixed-shape ``[max_num_seqs]`` and
    ``[max_num_seqs + 1]`` buffers so Dynamo traces them into static XLA HLO
    without graph breaks or host syncs.
    """
    start = window_idx * window_tokens
    end = start + window_tokens

    qsl = attn_metadata.query_start_loc
    seq_lens = attn_metadata.seq_lens
    assert isinstance(qsl, torch.Tensor)
    assert isinstance(seq_lens, torch.Tensor)
    win_qsl = torch.clamp(qsl - start, min=0, max=window_tokens)

    seq_end_q = qsl[1:]
    tokens_after = torch.clamp(seq_end_q - end, min=0)
    win_seq_lens = torch.clamp(seq_lens - tokens_after, min=0)

    win_q_lens = win_qsl[1:] - win_qsl[:-1]
    win_seq_lens = torch.where(
        win_q_lens > 0,
        win_seq_lens,
        torch.zeros_like(win_seq_lens),
    )

    input_positions = attn_metadata.input_positions
    if slice_input_positions and isinstance(input_positions, torch.Tensor):
        win_positions = input_positions[..., start:end]
    else:
        win_positions = input_positions

    return AttentionMetadata(
        input_positions=win_positions,
        block_tables=attn_metadata.block_tables,
        seq_lens=win_seq_lens,
        query_start_loc=win_qsl,
        request_distribution=attn_metadata.request_distribution,
        mamba_state_indices=attn_metadata.mamba_state_indices,
        mamba_slot_read_offsets=attn_metadata.mamba_slot_read_offsets,
        mamba_ckpt_indices=attn_metadata.mamba_ckpt_indices,
        mamba_request_distribution=attn_metadata.mamba_request_distribution,
        sequence_layout_kind=attn_metadata.sequence_layout_kind,
        sequence_layout_protocol=attn_metadata.sequence_layout_protocol,
        sequence_layout_version=attn_metadata.sequence_layout_version,
    )


def slice_forward_metadata(
    attn_metadata: Any,
    window_idx: int,
    window_tokens: int,
    *,
    slice_input_positions: bool = False,
) -> Any:
    """Slice ``ForwardContext.attn_metadata`` (single object or per-layer dict)."""
    if attn_metadata is None:
        return None
    if isinstance(attn_metadata, AttentionMetadata):
        return slice_attention_metadata(
            attn_metadata,
            window_idx,
            window_tokens,
            slice_input_positions=slice_input_positions,
        )
    if isinstance(attn_metadata, dict):
        memo: dict[int, Any] = {}
        sliced: dict[str, Any] = {}
        for layer_name, layer_md in attn_metadata.items():
            obj_id = id(layer_md)
            if obj_id not in memo:
                if isinstance(layer_md, AttentionMetadata):
                    memo[obj_id] = slice_attention_metadata(
                        layer_md,
                        window_idx,
                        window_tokens,
                        slice_input_positions=slice_input_positions,
                    )
                else:
                    memo[obj_id] = layer_md
            sliced[layer_name] = memo[obj_id]
        return sliced
    return attn_metadata


def slice_padding_state(
    state: TokenPaddingState | None,
    window_idx: int,
    window_tokens: int,
) -> TokenPaddingState | None:
    """Slice ``TokenPaddingState.local_padding_mask`` for a token window."""
    if state is None:
        return None
    start = window_idx * window_tokens
    end = start + window_tokens
    return TokenPaddingState(local_padding_mask=state.local_padding_mask[start:end])
