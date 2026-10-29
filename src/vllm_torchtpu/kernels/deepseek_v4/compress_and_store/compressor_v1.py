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
"""DeepSeek V4 Compressor Layer forward pass implementation."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

import vllm_torchtpu.kernels.deepseek_v4.compress_and_store.config as config
import vllm_torchtpu.kernels.deepseek_v4.compress_and_store.kernel as compress_kernel
import vllm_torchtpu.kernels.deepseek_v4.compress_and_store.proj_and_save_state as proj_kernel  # noqa: E501


def block_table_row_stride(block_table: jax.Array, num_reqs: int) -> int:
    """Per-request row stride of a flattened [num_reqs * max_blocks] table."""
    assert block_table.ndim == 1
    assert block_table.shape[0] % num_reqs == 0
    return block_table.shape[0] // num_reqs


def derive_metadata(
    positions: jax.Array,
    block_table: jax.Array,
    query_start_loc: jax.Array,
    kv_block_table: jax.Array,
    cache: jax.Array,
    compress_ratio: int,
    state_block_size: int,
    head_dim: int,
    overlap: bool,
    cos_sin_cache: jax.Array | None,
    state_cache: jax.Array | None = None,
    seq_along_lane: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Derives token_to_req_indices, slot_mapping_slots, and kv_slot_mapping.

    The page geometry is taken from ``config.Configs`` keyed on
    ``cache.shape[1]`` (compressed KV) and ``state_cache.shape[1]`` (state),
    which is the same derivation ``compress_norm_rope_store`` and
    ``proj_and_save_state`` use. It must not be recomputed from a separate model
    here: the three only agree for CSA, and diverge by 2x (HCA) and 4x (indexer)
    otherwise, which silently scatters state and compressed-KV writes across the
    wrong pages.

    Args:
      positions: [num_tokens]. Logical position of each token in its request.
      block_table: [num_reqs * max_blocks]. Flattened row-major page table for the
        state cache.
      query_start_loc: [num_reqs + 1]. Cumulative sum of query lengths.
      kv_block_table: [num_reqs * max_kv_blocks]. Flattened row-major page table
        for the compressed KV cache.
      cache: [num_pages, physical_page_size, 4, lanes] uint8, or CSA's
        [num_pages, physical_page_size, 128] int32 NoPE array. The
        compressed-KV buffer; only its shape is read here.
      compress_ratio: Compression ratio (e.g. 4 for CSA, 128 for HCA).
      state_block_size: Block size vLLM used to build ``block_table``. One page
        must be able to hold that many token states; it may hold more with the
        tail rows unused.
      head_dim: Dimensionality of attention heads.
      overlap: Whether to use overlap (CSA path).
      cos_sin_cache: [max_pos, rope_head_dim] or None. RoPE cos/sin cache.
      state_cache: [num_pages, state_page_size, 4, lanes] uint8, or None when the
        state shares ``cache``'s buffer. Only its shape is read here.
      seq_along_lane: ``cache`` is the SEQ_ALONG_LANE indexer array.
    Returns:
      token_to_req_indices: [num_tokens]. Request index for each token.
      slot_mapping_slots: [num_tokens]. Physical row index in the state cache.
      kv_slot_mapping: [num_tokens]. Sub-slot index in the compressed KV cache
        (``// tokens_in_second_minor`` gives the physical row; under
        SEQ_ALONG_LANE it is ``page * SAL_PAGE_SIZE + lane``).
    """
    num_tokens = positions.shape[0]
    rope_head_dim = cos_sin_cache.shape[1] if cos_sin_cache is not None else 0
    state_source = cache if state_cache is None else state_cache

    cfgs = config.Configs.make(
        config.select_mode(head_dim, overlap),
        size_n=num_tokens,
        physical_page_size=cache.shape[1],
        state_physical_page_size=state_source.shape[1],
        state_block_size=state_block_size,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        seq_along_lane=seq_along_lane,
    )
    # Rows per page, rows per token's state, and the KV sub-slot packing.
    page_size = cfgs.state_physical_page_size
    state_rows_per_token = cfgs.state_rows_per_token
    kv_block_size = cfgs.kv_block_size
    kv_stride = cfgs.kv_stride
    kv_page_stride = kv_block_size * kv_stride

    # The block tables are paged at vLLM's state-cache block size, and the
    # kernel indexes a page with `position % state_block_size`, so a page must
    # physically hold that many token states. Fewer is a bug; more is the
    # deliberate slack left when a mode is floored to its peer's page size.
    assert state_block_size <= cfgs.state_page_capacity, (
        f"state cache block_size {state_block_size} overruns the "
        f"{cfgs.dims.mode.value} state cache geometry {state_source.shape}, "
        f"which holds {cfgs.state_page_capacity} token states per page "
    )

    # 2. Map tokens to request indices (Handles Ragged Batch)
    query_lens = jnp.diff(query_start_loc)
    batch_size = query_start_loc.shape[0] - 1
    token_to_req_indices = jnp.repeat(
        jnp.arange(batch_size), query_lens, total_repeat_length=num_tokens
    )
    req = token_to_req_indices

    block_table_stride = block_table_row_stride(block_table, batch_size)
    kv_block_table_stride = block_table_row_stride(kv_block_table, batch_size)

    # 3. State Cache Slot Mapping (Virtual -> Physical)
    state_page_idx = positions // state_block_size
    state_page_offset = positions % state_block_size
    state_page_numbers = block_table[req * block_table_stride + state_page_idx]
    slot_mapping_slots = (
        state_page_numbers * page_size + state_page_offset * state_rows_per_token
    )

    # 4. Compressed KV Cache Slot Mapping (Virtual -> Physical)
    kv_idx = positions // compress_ratio
    # `(p // r) // b == p // (r * b)`. Written over the page's token count,
    # which every compressor on a KV table shares, the lookup below is one
    # expression XLA computes once for all of them.
    kv_page_idx = positions // (compress_ratio * kv_block_size)
    kv_page_offset = kv_idx % kv_block_size

    kv_page_number = kv_block_table[req * kv_block_table_stride + kv_page_idx]

    kv_slot_mapping = kv_page_number * kv_page_stride + kv_page_offset * kv_stride

    is_boundary = ((positions + 1) % compress_ratio) == 0
    kv_slot_mapping = jnp.where(is_boundary, kv_slot_mapping, -1)

    return token_to_req_indices, slot_mapping_slots, kv_slot_mapping


def _carry_forward(mask: jax.Array, values: jax.Array, fill: int) -> jax.Array:
    """`values` at the last position at or before each one where `mask` holds,
    `fill` before the first: a running max of the masked positions that carries
    their value along, so it lowers to a prefix reduce-window, not a gather."""
    n = mask.shape[0]
    index = jnp.where(mask, jnp.arange(n), -1)
    values = jnp.where(mask, values, fill)

    def latest(a, b):
        take_b = b[0] > a[0]
        return jnp.where(take_b, b[0], a[0]), jnp.where(take_b, b[1], a[1])

    init = (jnp.array(-1, index.dtype), jnp.array(fill, values.dtype))
    _, carried = jax.lax.reduce_window(
        (index, values), init, latest, (n,), (1,), [(n - 1, 0)]
    )
    return carried


def prepare_boundary_batch(
    positions: jax.Array,
    token_to_req_indices: jax.Array,
    kv_slot_mapping: jax.Array,
    is_boundary: jax.Array,
    is_decode_token: jax.Array,
    compress_ratio: int,
    num_reqs: int,
    rope_pack: int = 4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generates a combined batch of boundary tokens in a single scatter.

    Layout: [ decode_packed (0..K-1) | pad (K..K_aligned-1) |
    prefill_row_tiled (K_aligned..) | pad ]

    A prefill tile is `rope_pack` slots holding the tokens of one cache row of
    `rope_pack` kv slots; with `rope_pack` 1 the prefill tokens are dense.

    The boundary tokens, in token order, are the packed list the layout is
    defined on: a token's packed index is its rank among them. Every quantity is
    evaluated at the token itself, so the only data movement is one scatter of
    the tokens to their slots.
    """
    num_tokens = positions.shape[0]
    max_prefill_boundary = num_tokens // compress_ratio + 1 + num_reqs
    output_size = num_tokens + rope_pack * max_prefill_boundary
    # Spares compress_norm_rope_store padding the batch to its tile size.
    align = compress_kernel.TOKEN_ALIGN
    output_size = -(-output_size // align) * align

    packed_index = jnp.cumsum(is_boundary.astype(jnp.int32)) - 1
    is_decode = is_boundary & is_decode_token
    valid_token = is_boundary & (kv_slot_mapping >= 0)
    is_prefill = valid_token & ~is_decode_token

    # Row-tiling of the prefill tokens: a tile starts wherever a token's row
    # differs from the previous boundary token's (-1 for a decode token).
    row = jnp.where(is_prefill, kv_slot_mapping // rope_pack, -1)
    carried_row = _carry_forward(is_boundary, row, fill=-1)
    prev_row = jnp.concatenate([jnp.full((1,), -1, row.dtype), carried_row[:-1]])
    starts_row = is_prefill & (row != prev_row)

    tile_index = jnp.cumsum(starts_row.astype(jnp.int32)) - 1
    run_start = jax.lax.cummax(jnp.where(starts_row, packed_index, 0))
    pos_within_tile = packed_index - run_start

    K = jnp.sum(is_decode)
    K_aligned = ((K + 3) // 4) * 4

    dest_prefill = K_aligned + tile_index * rope_pack + pos_within_tile
    dest = jnp.where(is_decode, packed_index, dest_prefill)
    dest = jnp.where(valid_token, dest, output_size)

    src = jnp.stack([positions, token_to_req_indices, kv_slot_mapping])
    default = jnp.array([0, 0, -1], dtype=src.dtype)[:, None]
    out = jnp.broadcast_to(default, (3, output_size)).at[:, dest].set(src, mode="drop")
    return (
        out[0].astype(positions.dtype),
        out[1].astype(token_to_req_indices.dtype),
        out[2].astype(kv_slot_mapping.dtype),
    )


# `compressor_forward`'s result: the cache, the RoPE cache, the state cache.
CompressorOutputs = tuple[jax.Array, jax.Array | None, jax.Array]


class CompressorInputs(NamedTuple):
    """One compressor's `compressor_forward` operands, but for the hidden
    states and positions it shares with any other compressor of its layer."""

    wkv_wgate: jax.Array  # [hidden_size, 2*coff*head_dim]
    ape: jax.Array  # [compress_ratio, coff*head_dim] fp32
    norm_weight: jax.Array  # [head_dim]
    cos_sin_cache: jax.Array  # [max_pos, rope_head_dim] fp32
    block_table: jax.Array  # [num_reqs * max_blocks] int
    query_start_loc: jax.Array  # [num_reqs + 1] int
    kv_block_table: jax.Array  # [num_reqs * max_kv_blocks] int
    cache: jax.Array  # [num_pages, page_size, 4, 128] u8 / CSA [..., 128] i32
    rope_cache: jax.Array | None  # CSA [num_pages, page_size // 4, 128] int32
    distribution: jax.Array  # i32[3]
    state_block_size: int
    head_dim: int
    compress_ratio: int
    overlap: bool
    rms_eps: float
    quant_block: int
    state_cache: jax.Array | None = None  # [num_pages, state_page_size, 4, 128]
    seq_along_lane: bool = False  # indexer only: `cache` is SEQ_ALONG_LANE

    @property
    def state_target(self) -> jax.Array:
        """The array `proj_and_save_state` writes the states to."""
        return self.cache if self.state_cache is None else self.state_cache


def compressor_forward(
    hidden_states: jax.Array,  # [num_tokens, hidden_size] fp32
    wkv_wgate: jax.Array,  # [2*coff*head_dim, hidden_size] fp32
    ape: jax.Array,  # [compress_ratio, coff*head_dim] fp32
    norm_weight: jax.Array,  # [head_dim] fp32
    cos_sin_cache: jax.Array,  # [max_pos, rope_head_dim] fp32
    positions: jax.Array,  # [num_tokens] int
    block_table: jax.Array,  # [num_reqs * max_blocks] int
    query_start_loc: jax.Array,  # [num_reqs + 1] int
    kv_block_table: jax.Array,  # [num_reqs * max_kv_blocks] int
    cache: jax.Array,  # [num_pages, page_size, 4, 128] u8 / CSA [..., 128] i32
    rope_cache: jax.Array | None,  # CSA [num_pages, page_size // 4, 128] int32
    distribution: jax.Array,  # i32[3]
    state_block_size: int,
    head_dim: int,
    compress_ratio: int,
    overlap: bool,
    rms_eps: float,
    quant_block: int,
    state_cache: jax.Array | None = None,  # [num_pages, state_page_size, 4, 128] uint8
    seq_along_lane: bool = False,  # indexer only: `cache` is SEQ_ALONG_LANE
) -> tuple[jax.Array, jax.Array | None, jax.Array]:
    """Projects, saves state, then compresses and stores the boundary records."""
    c = CompressorInputs(
        wkv_wgate=wkv_wgate,
        ape=ape,
        norm_weight=norm_weight,
        cos_sin_cache=cos_sin_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        kv_block_table=kv_block_table,
        cache=cache,
        rope_cache=rope_cache,
        distribution=distribution,
        state_block_size=state_block_size,
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rms_eps=rms_eps,
        quant_block=quant_block,
        state_cache=state_cache,
        seq_along_lane=seq_along_lane,
    )
    slots = _state_slots(positions, c)
    updated_state = proj_kernel.proj_and_save_state(
        hidden_states=hidden_states,
        wkv_wgate=wkv_wgate,
        ape=ape,
        positions=positions,
        slot_mapping=slots.state,
        cache=c.state_target,
        compress_ratio=compress_ratio,
    )
    return _compress_and_store(positions, c, slots, updated_state)


def compressor_pair_forward(
    hidden_states: jax.Array,  # [num_tokens, hidden_size]
    positions: jax.Array,  # [num_tokens] int
    first: CompressorInputs,
    second: CompressorInputs,
) -> tuple[CompressorOutputs, CompressorOutputs]:
    """`compressor_forward` for two compressors of the same hidden states and
    positions, such as a CSA layer's attention and indexer compressors.

    Where `proj_and_save_state_pair` takes the two projections, one call saves
    both compressors' states; elsewhere each compressor runs on its own. Each
    cache gets the bytes `compressor_forward` writes to it either way.
    """
    if first.compress_ratio != second.compress_ratio or not proj_kernel.pair_fuses(
        positions.shape[0], first.wkv_wgate, second.wkv_wgate
    ):
        return tuple(
            compressor_forward(hidden_states, positions=positions, **c._asdict())
            for c in (first, second)
        )
    slots = (_state_slots(positions, first), _state_slots(positions, second))
    states = proj_kernel.proj_and_save_state_pair(
        hidden_states,
        positions,
        wkv_wgates=(first.wkv_wgate, second.wkv_wgate),
        apes=(first.ape, second.ape),
        slot_mappings=(slots[0].state, slots[1].state),
        caches=(first.state_target, second.state_target),
        compress_ratio=first.compress_ratio,
    )
    return (
        _compress_and_store(positions, first, slots[0], states[0]),
        _compress_and_store(positions, second, slots[1], states[1]),
    )


class _StateSlots(NamedTuple):
    token_to_req_indices: jax.Array
    state: jax.Array  # state cache row of each real token, -1 for padding
    kv: jax.Array  # compressed-KV sub-slot of each boundary token, else -1
    is_real_token: jax.Array


def _state_slots(positions: jax.Array, c: CompressorInputs) -> _StateSlots:
    """Where each token's state and compressed record go."""
    token_to_req_indices, slot_mapping_slots, kv_slot_mapping = derive_metadata(
        positions=positions,
        block_table=c.block_table,
        query_start_loc=c.query_start_loc,
        kv_block_table=c.kv_block_table,
        cache=c.cache,
        compress_ratio=c.compress_ratio,
        state_block_size=c.state_block_size,
        head_dim=c.head_dim,
        overlap=c.overlap,
        cos_sin_cache=c.cos_sin_cache,
        state_cache=c.state_cache,
        seq_along_lane=c.seq_along_lane,
    )

    token_index = jnp.arange(positions.shape[0])
    is_real_token = token_index < c.query_start_loc[c.distribution[2]]
    slot_mapping_slots = jnp.where(is_real_token, slot_mapping_slots, -1)
    return _StateSlots(
        token_to_req_indices, slot_mapping_slots, kv_slot_mapping, is_real_token
    )


def _compress_and_store(
    positions: jax.Array,
    c: CompressorInputs,
    slots: _StateSlots,
    updated_state: jax.Array,
) -> tuple[jax.Array, jax.Array | None, jax.Array]:
    """Compresses and stores the boundary records once the states are saved."""
    num_tokens = positions.shape[0]
    num_reqs = c.query_start_loc.shape[0] - 1
    cache, rope_cache, state_cache = c.cache, c.rope_cache, c.state_cache
    shares_buffer = state_cache is None
    if shares_buffer:
        cache = updated_state
    else:
        state_cache = updated_state

    token_index = jnp.arange(num_tokens)
    is_decode_token = token_index < c.query_start_loc[c.distribution[0]]
    is_boundary = (slots.kv >= 0) & slots.is_real_token
    # HCA stores every record on its own, with no shared cache row to keep
    # within one grid step, so its boundary tokens are packed densely.
    is_hca = config.select_mode(c.head_dim, c.overlap) is config.Mode.HCA

    boundary_positions, boundary_req, boundary_kv_slot = prepare_boundary_batch(
        positions=positions,
        token_to_req_indices=slots.token_to_req_indices,
        kv_slot_mapping=slots.kv,
        is_boundary=is_boundary,
        is_decode_token=is_decode_token,
        compress_ratio=c.compress_ratio,
        num_reqs=num_reqs,
        rope_pack=1 if is_hca else 4,
    )

    cache, rope_cache = compress_kernel.compress_norm_rope_store(
        cache,
        boundary_positions,
        c.block_table,
        boundary_req,
        boundary_kv_slot,
        c.norm_weight,
        block_table_stride=block_table_row_stride(c.block_table, num_reqs),
        state_block_size=c.state_block_size,
        state_cache=state_cache,
        rope_cache=rope_cache,
        cos_sin_cache=c.cos_sin_cache,
        compress_ratio=c.compress_ratio,
        overlap=c.overlap,
        quant_block=c.quant_block,
        rms_eps=c.rms_eps,
        seq_along_lane=c.seq_along_lane,
    )

    if shares_buffer:
        return cache, rope_cache, cache

    # `compress_norm_rope_store` READS `state_cache` (its read-only state
    # operand) but emits no new version of it, so the value published here is
    # the pre-read one straight off `proj_and_save_state`. The DSv4 KV-cache
    # overlay hosts another layer's SWA cache on this same physical array, and
    # that kernel takes whole-buffer ownership via `input_output_aliases`.
    # Nothing orders the read against that write, so XLA's copy-insertion
    # AddCopiesToResolveInterference cannot prove the read completes first and
    # insert the cache copy. Publishing a version that is downstream of
    # the read supplies the missing ordering edge.
    state_cache, cache = jax.lax.optimization_barrier((state_cache, cache))
    return cache, rope_cache, state_cache
