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

# Shared TPU Pallas SSD kernel block length (`block_len = chunk_size * mimo_rank`).
TPU_SSD_BLOCK_LEN = 64


class MambaBatchMetadata(NamedTuple):
    """Normalized device-side ragged batch metadata for Mamba layers."""

    num_valid_seqs: jax.Array
    num_nonempty_seqs: jax.Array
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
    compact_seq_ids: jax.Array
    compact_valid: jax.Array


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


def clip_query_start_loc(
    query_start_loc: jax.Array,
    distribution: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Clip ``query_start_loc`` to ``distribution[2]`` valid sequences.

    Returns ``(num_valid_seqs, effective_query_start_loc, query_lens)``.
    """
    num_seqs = query_start_loc.shape[0] - 1
    if distribution is not None:
        num_valid_seqs = jnp.clip(distribution[2], 0, num_seqs)
        valid_loc_mask = (
            jnp.arange(query_start_loc.shape[0], dtype=jnp.int32) <= num_valid_seqs
        )
        last_valid_loc = query_start_loc[num_valid_seqs]
        effective_query_start_loc = jnp.where(
            valid_loc_mask, query_start_loc, last_valid_loc
        )
    else:
        num_valid_seqs = jnp.asarray(num_seqs, dtype=jnp.int32)
        effective_query_start_loc = query_start_loc

    query_lens = jnp.maximum(
        effective_query_start_loc[1 : num_seqs + 1]
        - effective_query_start_loc[:num_seqs],
        0,
    )
    return num_valid_seqs, effective_query_start_loc, query_lens


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
    """Build clipped per-request masks, query lengths, and compacted active IDs."""
    max_reqs = seq_lens.shape[0]
    num_valid_seqs, effective_query_start_loc, query_lens = clip_query_start_loc(
        query_start_loc[: max_reqs + 1], distribution
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

    # At most `min(max_reqs, num_tokens)` sequences can have `query_lens > 0`.
    k_cap = max(1, min(max_reqs, num_tokens))
    num_nonempty_seqs = jnp.minimum(
        jnp.sum(active_nonempty_seq.astype(jnp.int32)), k_cap
    )
    compact_seq_ids = jnp.nonzero(active_nonempty_seq, size=k_cap, fill_value=0)[
        0
    ].astype(jnp.int32)
    compact_valid = jnp.arange(k_cap, dtype=jnp.int32) < num_nonempty_seqs

    return MambaBatchMetadata(
        num_valid_seqs=num_valid_seqs,
        num_nonempty_seqs=num_nonempty_seqs,
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
        compact_seq_ids=compact_seq_ids,
        compact_valid=compact_valid,
    )


def gather_active_mamba_states(
    cache: jax.Array,
    safe_slots: jax.Array,
    should_read: jax.Array,
    *,
    num_active: jax.Array,
) -> jax.Array:
    """Gather float32 state rows only for active sequences needing carry.

    Bounds the ``fori_loop`` to ``num_active`` (and ``0`` when ``should_read``
    is all ``False``, e.g. fresh prefills) so 1 active request in a
    ``max_reqs=128`` batch reads at most 1 state slot instead of all 128 rows.
    """
    k_cap = safe_slots.shape[0]
    out_zeros = jnp.zeros((k_cap, *cache.shape[1:]), dtype=jnp.float32)
    bound = jnp.where(
        jnp.any(should_read), jnp.clip(num_active, 0, k_cap), jnp.int32(0)
    )

    def _body(idx: jax.Array, buf: jax.Array) -> jax.Array:
        return jax.lax.cond(
            should_read[idx],
            lambda b: b.at[idx].set(cache[safe_slots[idx]].astype(jnp.float32)),
            lambda b: b,
            buf,
        )

    return jax.lax.fori_loop(jnp.int32(0), bound, _body, out_zeros)


def pack_active_chunk_initial_states(
    init_states_compact: jax.Array,
    active_chunk_compact: jax.Array,
    has_nonzero_init: jax.Array,
    *,
    num_active: jax.Array,
    max_chunk_seqs: int,
) -> jax.Array:
    """Compact initial states for chunked sequences in sequence-start order."""
    k_cap = init_states_compact.shape[0]
    out_zeros = jnp.zeros(
        (max(1, max_chunk_seqs), *init_states_compact.shape[1:]),
        dtype=jnp.float32,
    )
    bound = jnp.where(has_nonzero_init, jnp.clip(num_active, 0, k_cap), jnp.int32(0))

    def _body(
        idx: jax.Array, carry: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        ord_idx, buf = carry
        return jax.lax.cond(
            active_chunk_compact[idx],
            lambda c: (
                c[0] + jnp.int32(1),
                c[1].at[c[0]].set(init_states_compact[idx]),
            ),
            lambda c: c,
            (ord_idx, buf),
        )

    _, packed = jax.lax.fori_loop(jnp.int32(0), bound, _body, (jnp.int32(0), out_zeros))
    return packed


def scatter_mamba_states(
    cache: jax.Array,
    state_indices: jax.Array,
    valid_mask: jax.Array,
    updates: jax.Array,
    *,
    num_active: jax.Array | None = None,
) -> jax.Array:
    """Scatter per-request state updates into ``cache``, dropping invalid slots.

    When ``num_active`` is provided, iterates only over ``[0, num_active)``
    so 1 active request writes 1 state slot rather than scattering ``max_reqs``
    slices into HBM.
    """
    num_slots = cache.shape[0]
    keep = valid_mask & (state_indices >= 0) & (state_indices < num_slots)
    if num_active is None:
        scatter_slots = jnp.where(keep, state_indices, num_slots)
        return cache.at[scatter_slots].set(updates.astype(cache.dtype), mode="drop")

    bound = jnp.clip(num_active, 0, updates.shape[0])
    updates_cast = updates.astype(cache.dtype)

    def _body(idx: jax.Array, cur_cache: jax.Array) -> jax.Array:
        return jax.lax.cond(
            keep[idx],
            lambda c: c.at[state_indices[idx]].set(updates_cast[idx]),
            lambda c: c,
            cur_cache,
        )

    return jax.lax.fori_loop(jnp.int32(0), bound, _body, cache)


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

    Supports both SISO (``X``: ``(T, H, P)``, ``B/C``: ``(T, G, N)``) and MIMO
    (``X``: ``(T, H, R, P)``, ``B/C``: ``(T, G, R, N)``, flattening rank ``R``
    into ``block_len = chunk_size * R``). Scatters tokens directly into the
    chunk-major ``(num_chunks_total, n_heads/n_groups, ...)`` layout without an
    intermediate token-major packed buffer or transpose copy.
    """
    num_tokens = X.shape[0]
    num_seqs = query_start_loc.shape[0] - 1

    num_valid_seqs, effective_query_start_loc, seq_lengths = clip_query_start_loc(
        query_start_loc, distribution
    )
    last_valid_loc = effective_query_start_loc[-1]
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

    # At most `min(num_seqs, num_tokens)` sequences (`num_tokens // 2` when
    # skipping 1-token decodes) can be non-empty, each adding <= chunk_size - 1
    # tail padding tokens: sum ceil(L_i / C) <= floor((T + k * (C - 1)) / C).
    max_active_seqs = min(
        num_seqs, num_tokens // 2 if skip_single_token_seqs else num_tokens
    )
    max_packed_tokens = (
        max(1, (num_tokens + max_active_seqs * (chunk_size - 1)) // chunk_size)
        * chunk_size
    )
    num_chunks_total = max_packed_tokens // chunk_size

    padded_indices_valid = jnp.clip(
        new_start + (token_indices - original_start), 0, max_packed_tokens - 1
    )
    chunk_idx = jnp.where(
        valid_token_mask, padded_indices_valid // chunk_size, num_chunks_total
    )
    intra_pos = padded_indices_valid % chunk_size

    n_heads = X.shape[1]
    n_groups = B.shape[1]
    is_mimo = X.ndim == 4
    mimo_rank = X.shape[2] if is_mimo else 1
    d_head = X.shape[-1]
    d_state = B.shape[-1]
    block_len = chunk_size * mimo_rank

    if is_mimo:
        X_c = (
            jnp.zeros(
                (num_chunks_total, n_heads, chunk_size, mimo_rank, d_head),
                dtype=compute_dtype,
            )
            .at[chunk_idx, :, intra_pos]
            .set(X.astype(compute_dtype), mode="drop")
            .reshape(num_chunks_total, n_heads, block_len, d_head)
        )
        B_c = (
            jnp.zeros(
                (num_chunks_total, n_groups, chunk_size, mimo_rank, d_state),
                dtype=compute_dtype,
            )
            .at[chunk_idx, :, intra_pos]
            .set(B.astype(compute_dtype), mode="drop")
            .reshape(num_chunks_total, n_groups, block_len, d_state)
        )
        C_c = (
            jnp.zeros(
                (num_chunks_total, n_groups, chunk_size, mimo_rank, d_state),
                dtype=compute_dtype,
            )
            .at[chunk_idx, :, intra_pos]
            .set(C.astype(compute_dtype), mode="drop")
            .reshape(num_chunks_total, n_groups, block_len, d_state)
        )
    else:
        X_c = (
            jnp.zeros(
                (num_chunks_total, n_heads, chunk_size, d_head), dtype=compute_dtype
            )
            .at[chunk_idx, :, intra_pos]
            .set(X.astype(compute_dtype), mode="drop")
        )
        B_c = (
            jnp.zeros(
                (num_chunks_total, n_groups, chunk_size, d_state), dtype=compute_dtype
            )
            .at[chunk_idx, :, intra_pos]
            .set(B.astype(compute_dtype), mode="drop")
        )
        C_c = (
            jnp.zeros(
                (num_chunks_total, n_groups, chunk_size, d_state), dtype=compute_dtype
            )
            .at[chunk_idx, :, intra_pos]
            .set(C.astype(compute_dtype), mode="drop")
        )

    decay_c = (
        jnp.zeros((num_chunks_total, n_heads, chunk_size), dtype=jnp.float32)
        .at[chunk_idx, :, intra_pos]
        .set(decay.astype(jnp.float32), mode="drop")
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
        active_chunk_seq,
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
    if mesh.size != 1:
        raise ValueError(
            f"run_replicated_shard_map requires a single-device worker mesh "
            f"(mesh.size == 1), got mesh.size={mesh.size} (shape={mesh.shape})."
        )
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
