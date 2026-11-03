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
"""Shared JAX helpers for Mamba-2, Mamba-3, and KDA ragged sequence layers."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P


class MambaBatchMetadata(NamedTuple):
    """Normalized device-side ragged batch metadata for Mamba layers."""

    num_valid_seqs: jax.Array
    effective_query_start_loc: jax.Array
    query_lens: jax.Array
    has_initial_state: jax.Array
    active_seq_mask: jax.Array
    active_nonempty_seq: jax.Array
    is_decode_seq: jax.Array
    is_prefill_seq: jax.Array
    is_pure_decode: jax.Array
    safe_state_indices: jax.Array
    valid_state_slot: jax.Array


def token_sequence_ids(
    query_start_loc: jax.Array,
    num_tokens: int,
    num_seqs: int,
) -> jax.Array:
    """Map each token row to the 0-indexed sequence that owns it.

    A token's sequence index is the number of interior sequence starts
    (``query_start_loc[1:num_seqs]``) lying at or below its row, so marking each
    start and taking an inclusive cumsum computes the mapping in one linear pass
    over the token axis instead of ``jnp.searchsorted``'s ``O(T * num_seqs)``
    broadcast comparison at small request buckets.

    ``mode="drop"`` discards starts at or past ``num_tokens`` (dummy AOT runs
    may describe more requests than fit in the token bucket), while scatter-add
    accumulates duplicate starts so zero-length interior sequences advance the
    index by more than one.
    """
    if num_tokens == 0 or num_seqs <= 1:
        return jnp.zeros((num_tokens,), dtype=jnp.int32)
    starts = jnp.maximum(query_start_loc[1:num_seqs], 0)
    marks = jnp.zeros((num_tokens,), dtype=jnp.int32).at[starts].add(1, mode="drop")
    return jnp.minimum(jnp.cumsum(marks), num_seqs - 1)


def broadcast_groups_to_heads(x: jax.Array, n_heads: int) -> jax.Array:
    """Broadcast group dimension (axis 1) to ``n_heads``."""
    n_groups = x.shape[1]
    if n_groups == n_heads:
        return x
    return jnp.repeat(x, n_heads // n_groups, axis=1)


def jax_gated_rms_norm(
    x: jax.Array,
    gate: jax.Array | None,
    weight: jax.Array | None,
    eps: float,
    n_groups: int = 1,
) -> jax.Array:
    """Apply optional SiLU gating (``norm_before_gate=False``) and grouped RMSNorm."""
    input_dtype = x.dtype
    x_val = x.astype(jnp.float32)
    if gate is not None:
        x_val = x_val * jax.nn.silu(gate.astype(jnp.float32))
    if weight is None:
        return x_val.astype(input_dtype)

    orig_shape = x_val.shape
    if n_groups > 1:
        x_val = x_val.reshape(*orig_shape[:-1], n_groups, orig_shape[-1] // n_groups)
    variance = jnp.mean(jnp.square(x_val), axis=-1, keepdims=True)
    x_norm = (x_val * jax.lax.rsqrt(variance + eps)).reshape(orig_shape)
    return (weight.astype(jnp.float32) * x_norm).astype(input_dtype)


def prepare_mamba_batch_metadata(
    query_start_loc: jax.Array,
    seq_lens: jax.Array,
    distribution: jax.Array,
    state_indices: jax.Array,
    *,
    num_tokens: int,
    num_slots: int,
) -> MambaBatchMetadata:
    """Build clipped per-request masks, query lengths, and safe state indices."""
    max_reqs = seq_lens.shape[0]
    num_valid_seqs = jnp.clip(distribution[2], 0, max_reqs)

    valid_loc_mask = (
        jnp.arange(query_start_loc.shape[0], dtype=jnp.int32) <= num_valid_seqs
    )
    last_valid_loc = query_start_loc[num_valid_seqs]
    effective_query_start_loc = jnp.where(
        valid_loc_mask, query_start_loc, last_valid_loc
    )

    query_lens = jnp.maximum(
        effective_query_start_loc[1 : max_reqs + 1]
        - effective_query_start_loc[:max_reqs],
        0,
    )
    has_initial_state = (seq_lens - query_lens) > 0
    active_seq_mask = jnp.arange(max_reqs, dtype=jnp.int32) < num_valid_seqs
    active_nonempty_seq = active_seq_mask & (query_lens > 0)
    is_decode_seq = active_nonempty_seq & (query_lens == 1)
    is_prefill_seq = active_nonempty_seq & (query_lens > 1)
    is_pure_decode = (
        (distribution[0] == distribution[2])
        & (num_valid_seqs <= num_tokens)
        & jnp.all(jnp.where(active_seq_mask, query_lens == 1, True))
    )

    safe_state_indices = jnp.clip(state_indices, 0, jnp.maximum(num_slots - 1, 0))
    valid_state_slot = (
        active_nonempty_seq & (state_indices >= 0) & (state_indices < num_slots)
    )

    return MambaBatchMetadata(
        num_valid_seqs=num_valid_seqs,
        effective_query_start_loc=effective_query_start_loc,
        query_lens=query_lens,
        has_initial_state=has_initial_state,
        active_seq_mask=active_seq_mask,
        active_nonempty_seq=active_nonempty_seq,
        is_decode_seq=is_decode_seq,
        is_prefill_seq=is_prefill_seq,
        is_pure_decode=is_pure_decode,
        safe_state_indices=safe_state_indices,
        valid_state_slot=valid_state_slot,
    )


def scatter_mamba_states(
    cache: jax.Array,
    state_indices: jax.Array,
    valid_mask: jax.Array,
    updates: jax.Array,
) -> jax.Array:
    """Scatter per-request state updates into ``cache``, dropping invalid slots."""
    num_slots = cache.shape[0]
    keep = valid_mask & (state_indices >= 0) & (state_indices < num_slots)
    scatter_slots = jnp.where(keep, state_indices, num_slots)
    return cache.at[scatter_slots].set(updates.astype(cache.dtype), mode="drop")


def seed_chunk_initial_states(
    init_states_for_seqs: jax.Array,
    start_chunk_indices: jax.Array,
    num_chunks_total: int,
) -> jax.Array:
    """Place per-sequence initial states at each sequence's start chunk."""
    init_h_per_chunk = jnp.zeros(
        (num_chunks_total, *init_states_for_seqs.shape[1:]),
        dtype=jnp.float32,
    )
    return init_h_per_chunk.at[start_chunk_indices].set(
        init_states_for_seqs.astype(jnp.float32), mode="drop"
    )


def pack_ragged_ssm_chunks(
    X: jax.Array,
    B: jax.Array,
    C: jax.Array,
    decay: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    chunk_size: int,
    *,
    compute_dtype: jax.typing.DTypeLike = jnp.bfloat16,
    skip_single_token_seqs: bool = False,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Pad ragged sequences to ``chunk_size`` multiples and pack into 4D chunks.

    Supports both SISO (``X``: ``(T, H, P)``, ``B/C``: ``(T, H, N)``) and MIMO
    (``X``: ``(T, H, R, P)``, ``B/C``: ``(T, H, R, N)``, flattening rank ``R``
    into ``block_len = chunk_size * R``). When ``skip_single_token_seqs`` is
    True, 1-token decode sequences in mixed batches allocate zero chunks.
    """
    num_tokens = X.shape[0]
    num_seqs = query_start_loc.shape[0] - 1

    num_valid_seqs = jnp.clip(distribution[2], 0, num_seqs)
    valid_loc_mask = (
        jnp.arange(query_start_loc.shape[0], dtype=jnp.int32) <= num_valid_seqs
    )
    last_valid_loc = query_start_loc[num_valid_seqs]
    effective_query_start_loc = jnp.where(
        valid_loc_mask, query_start_loc, last_valid_loc
    )

    seq_lengths = jnp.maximum(
        effective_query_start_loc[1:] - effective_query_start_loc[:-1], 0
    )
    effective_seq_lengths = (
        jnp.where(seq_lengths > 1, seq_lengths, 0)
        if skip_single_token_seqs
        else seq_lengths
    )

    num_chunks = (effective_seq_lengths + chunk_size - 1) // chunk_size
    padded_lengths = num_chunks * chunk_size
    new_query_start_loc = jnp.cumsum(
        jnp.concatenate([jnp.zeros((1,), dtype=jnp.int32), padded_lengths])
    )

    token_indices = jnp.arange(num_tokens, dtype=jnp.int32)
    seq_id = token_sequence_ids(effective_query_start_loc, num_tokens, num_seqs)
    valid_token_mask = (token_indices < last_valid_loc) & (
        effective_seq_lengths[seq_id] > 0
    )
    original_start = effective_query_start_loc[seq_id]
    new_start = new_query_start_loc[seq_id]

    max_packed_tokens = num_tokens + num_seqs * chunk_size
    max_packed_tokens = (max_packed_tokens + chunk_size - 1) // chunk_size * chunk_size

    padded_indices_valid = jnp.clip(
        new_start + (token_indices - original_start), 0, max_packed_tokens - 1
    )
    scatter_token_indices = jnp.where(
        valid_token_mask, padded_indices_valid, max_packed_tokens
    )

    n_heads = X.shape[1]
    is_mimo = X.ndim == 4
    mimo_rank = X.shape[2] if is_mimo else 1
    d_head = X.shape[-1]
    d_state = B.shape[-1]

    dim_X = n_heads * mimo_rank * d_head
    dim_B = n_heads * mimo_rank * d_state

    combined = jnp.concatenate(
        [
            X.reshape(num_tokens, dim_X).astype(compute_dtype),
            B.reshape(num_tokens, dim_B).astype(compute_dtype),
            C.reshape(num_tokens, dim_B).astype(compute_dtype),
        ],
        axis=-1,
    )
    packed_combined = (
        jnp.zeros((max_packed_tokens, combined.shape[1]), dtype=compute_dtype)
        .at[scatter_token_indices]
        .set(combined, mode="drop")
    )

    packed_X_flat = packed_combined[..., :dim_X]
    packed_B_flat = packed_combined[..., dim_X : dim_X + dim_B]
    packed_C_flat = packed_combined[..., dim_X + dim_B :]

    num_chunks_total = max_packed_tokens // chunk_size
    block_len = chunk_size * mimo_rank

    if is_mimo:
        X_c = (
            packed_X_flat.reshape(
                num_chunks_total, chunk_size, n_heads, mimo_rank, d_head
            )
            .transpose(0, 2, 1, 3, 4)
            .reshape(num_chunks_total, n_heads, block_len, d_head)
        )
        B_c = (
            packed_B_flat.reshape(
                num_chunks_total, chunk_size, n_heads, mimo_rank, d_state
            )
            .transpose(0, 2, 1, 3, 4)
            .reshape(num_chunks_total, n_heads, block_len, d_state)
        )
        C_c = (
            packed_C_flat.reshape(
                num_chunks_total, chunk_size, n_heads, mimo_rank, d_state
            )
            .transpose(0, 2, 1, 3, 4)
            .reshape(num_chunks_total, n_heads, block_len, d_state)
        )
    else:
        X_c = packed_X_flat.reshape(
            num_chunks_total, chunk_size, n_heads, d_head
        ).transpose(0, 2, 1, 3)
        B_c = packed_B_flat.reshape(
            num_chunks_total, chunk_size, n_heads, d_state
        ).transpose(0, 2, 1, 3)
        C_c = packed_C_flat.reshape(
            num_chunks_total, chunk_size, n_heads, d_state
        ).transpose(0, 2, 1, 3)

    packed_decay = (
        jnp.zeros((max_packed_tokens, n_heads), dtype=jnp.float32)
        .at[scatter_token_indices]
        .set(decay.astype(jnp.float32), mode="drop")
    )
    decay_c = packed_decay.reshape(num_chunks_total, chunk_size, n_heads).transpose(
        0, 2, 1
    )

    active_chunk_seq = (jnp.arange(num_seqs, dtype=jnp.int32) < num_valid_seqs) & (
        effective_seq_lengths > 0
    )
    start_chunk_indices = jnp.where(
        active_chunk_seq,
        new_query_start_loc[:-1] // chunk_size,
        num_chunks_total,
    )
    reset_mask = (
        jnp.zeros((num_chunks_total,), dtype=bool)
        .at[start_chunk_indices]
        .set(True, mode="drop")
    )
    last_chunk_indices = jnp.clip(
        (new_query_start_loc[1:] // chunk_size) - 1, 0, num_chunks_total - 1
    )

    return (
        X_c,
        B_c,
        C_c,
        decay_c,
        reset_mask,
        last_chunk_indices,
        start_chunk_indices,
        padded_indices_valid,
    )


def run_replicated_shard_map(
    local_fn: Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    mesh: jax.sharding.Mesh,
    *args: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run ``local_fn`` via ``jax.shard_map`` with replicated ``P()`` specs.

    Each TPU worker process owns a single-device ``Mesh`` while tensor
    parallelism runs across processes in PyTorch, so all per-rank operands and
    outputs are local/replicated on ``mesh``.
    """
    in_specs = tuple(None if arg is None else P() for arg in args)
    out_specs = (P(), P(), P())
    mapped = jax.shard_map(
        local_fn,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    return mapped(*args)
