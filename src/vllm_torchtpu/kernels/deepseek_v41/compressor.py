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
"""DeepSeek-V4.1 compressor and indexer JAX/Pallas kernels."""

import functools
import inspect

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import compute
from vllm_torchtpu.kernels.deepseek_v4.rope import (
    _expand_cos_sin,
    _lane_gather,
    rope_quant,
)
from vllm_torchtpu.kernels.deepseek_v41.sparse_mla import (
    align_to,
    cdiv,
    encode_csa_gather_indices,
)
from vllm_torchtpu.kernels.deepseek_v41.streamindex_topk import (
    streamindex_topk,
)
from vllm_torchtpu import envs
from vllm_torchtpu.utils import largest_divisor


def _prefill_role() -> bool:
    """True on the dedicated prefill node (VLLM_DSV41_PURE_20L_PREFILL=1).

    Selects the prefill-tuned variants (last-token-only compressor state writes,
    8-deep scatter ring, Pallas insert at >=384 tokens, 64-page decode indexer KV
    blocks, padded/lane-aligned fused wq_b kernel). Everything else (decode node,
    spec decode, prefix caching) keeps the decode stack's variants: per-token
    state writes are required whenever a step can start at an arbitrary
    position (DSpark rollback, prefix-cache hit). Read at trace time.
    """
    return bool(envs.VLLM_DSV41_PURE_20L_PREFILL)


def _seq_ids_from_locs(
    query_start_loc: jax.Array, num_tokens: int, eff_max_seqs: int
) -> jax.Array:
    """Map each token index in `[0, num_tokens)` to its sequence ID in `[0, eff_max_seqs)`."""
    q_ends = query_start_loc[1:eff_max_seqs]
    tok = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
    return jnp.sum((tok >= q_ends[None, :]).astype(jnp.int32), axis=1)


def _compact_indices(mask: jax.Array, size: int) -> jax.Array:
    """Compact True indices of 1D boolean `mask` into a `size`-length int32 array (-1 padded)."""
    n = mask.shape[0]
    vals = jnp.where(mask, jnp.arange(n, dtype=jnp.int32), n)
    if size > n:
        vals = jnp.pad(vals, (0, size - n), constant_values=n)
    s = jnp.sort(vals)[:size]
    return jnp.where(s < n, s, -1)


def _rope_block_start(
    x: jax.Array, pos: jax.Array, cos_sin_cache: jax.Array, compress_ratio: int
) -> jax.Array:
    """GPT-J RoPE on the trailing 64 channels at each token's block-start position."""
    x = x.astype(jnp.float32)
    start = (
        jnp.clip(pos, 0, cos_sin_cache.shape[0] - 1)
        if compress_ratio == 1
        else jnp.clip(
            (pos // compress_ratio) * compress_ratio, 0, cos_sin_cache.shape[0] - 1
        )
    )
    cs = jnp.repeat(cos_sin_cache[start], 2, axis=-1)
    cos, sin = cs[:, :64], cs[:, 64:]
    x_nope, x_rope = x[:, :64], x[:, 64:]
    x_rot = jnp.stack([-x_rope[:, 1::2], x_rope[:, 0::2]], axis=-1).reshape(
        x.shape[0], 64
    )
    return jnp.concatenate([x_nope, x_rope * cos + x_rot * sin], axis=-1)


def _compressed_kv_positions(
    pos: jax.Array,
    kv_block_table: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    page_size: int,
    compress_ratio: int,
    block_stride: int = 1,
):
    num_tokens = pos.shape[0]
    max_num_seqs = query_start_loc.shape[0] - 1
    max_kv_blocks = kv_block_table.shape[0] // max_num_seqs
    eff_max_seqs = min(max_num_seqs, max(8, align_to(num_tokens, 8)))
    req = _seq_ids_from_locs(query_start_loc, num_tokens, eff_max_seqs)
    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)
    valid = token_idx < query_start_loc[distribution[2]]
    is_boundary = valid & (((pos + 1) % compress_ratio) == 0)
    kv_idx = jnp.maximum(pos // compress_ratio, 0)
    kv_page_idx = jnp.clip((kv_idx // page_size) * block_stride, 0, max_kv_blocks - 1)
    kv_page_off = kv_idx % page_size
    kv_page = kv_block_table[req * max_kv_blocks + kv_page_idx]
    return req, token_idx, is_boundary, kv_page, kv_page_off


def compress_r2_pool_norm_jax(
    wkv_wgate: jax.Array,  # [num_tokens, 1024] f32
    norm_weight: jax.Array,  # [512] bf16/f32
    positions: jax.Array,  # [num_tokens] i32
    state_block_table: jax.Array,  # [max_num_seqs * max_state_blocks] i32
    query_start_loc: jax.Array,  # [max_num_seqs + 1] i32
    distribution: jax.Array,  # [3] i32
    state_cache: jax.Array,  # [num_pages, page_size, 128] i32
    *,
    state_block_size: int,
    rms_eps: float,
) -> tuple[jax.Array, jax.Array]:
    """Pool pairs of tokens (even, odd) for compress_ratio=2 and apply RMSNorm."""
    head_dim = norm_weight.shape[0]
    num_tokens = wkv_wgate.shape[0]
    max_num_seqs = query_start_loc.shape[0] - 1
    max_state_blocks = state_block_table.shape[0] // max_num_seqs
    eff_max_seqs = min(max_num_seqs, max(8, align_to(num_tokens, 8)))

    num_pages, page_size = state_cache.shape[:2]
    states_per_page = page_size // 8
    total_state_slots = num_pages * states_per_page
    state_slots = state_cache.reshape((total_state_slots, 8, *state_cache.shape[2:]))

    pos = positions.astype(jnp.int32)
    seq_ids = jnp.arange(eff_max_seqs, dtype=jnp.int32)
    q_start = query_start_loc[:eff_max_seqs]
    q_end = query_start_loc[1 : eff_max_seqs + 1]
    seq_valid = (seq_ids < distribution[2]) & (q_end > q_start)

    def _state_slot_for(r: jax.Array, p: jax.Array) -> jax.Array:
        page_idx = jnp.clip(p // state_block_size, 0, max_state_blocks - 1)
        page = state_block_table[r * max_state_blocks + page_idx]
        if num_pages > 1:
            page = jnp.where(page > 0, ((page - 1) % (num_pages - 1)) + 1, 0)
        return jnp.clip(
            page * states_per_page + (p % state_block_size), 0, total_state_slots - 1
        )

    first_tok_idx = jnp.clip(q_start, 0, num_tokens - 1)
    first_pos = pos[first_tok_idx]
    seq_read_slots = _state_slot_for(seq_ids, jnp.maximum(first_pos - 1, 0))
    seq_cache_prev = jax.lax.bitcast_convert_type(
        state_slots[seq_read_slots], jnp.float32
    ).reshape(eff_max_seqs, 1024)

    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)

    def _single_seq_prev():
        return jnp.concatenate([seq_cache_prev[:1], wkv_wgate[:-1]], axis=0)

    def _multi_seq_prev():
        if num_tokens >= 384:
            batch_prev = jnp.concatenate([wkv_wgate[:1], wkv_wgate[:-1]], axis=0)
            first_write_idx = jnp.where(seq_valid, first_tok_idx, num_tokens)
            return batch_prev.at[first_write_idx].set(seq_cache_prev, mode="drop")
        batch_prev = wkv_wgate[jnp.clip(token_idx - 1, 0, num_tokens - 1)]
        req = _seq_ids_from_locs(query_start_loc, num_tokens, eff_max_seqs)
        in_batch_prev = token_idx > query_start_loc[req]
        return jnp.where(in_batch_prev[:, None], batch_prev, seq_cache_prev[req])

    prev_kv_score = jax.lax.cond(
        distribution[2] <= 1,
        _single_seq_prev,
        _multi_seq_prev,
    )

    if _prefill_role():
        # Prefill node: no spec-decode rollback / prefix-cache hits, so only the
        # last token of each sequence needs to persist its (even-pos) state.
        last_tok_idx = jnp.clip(q_end - 1, 0, num_tokens - 1)
        last_pos = pos[last_tok_idx]
        should_write = seq_valid & ((last_pos & 1) == 0)
        write_slots = jnp.where(
            should_write,
            _state_slot_for(seq_ids, last_pos),
            total_state_slots,
        )
        wkv_words = jax.lax.bitcast_convert_type(
            wkv_wgate[last_tok_idx].reshape(eff_max_seqs, 8, 128), jnp.int32
        )
    else:
        # Decode stack: persist every even-position token so a step may resume
        # from any position (DSpark verify rollback, prefix-cache hit).
        req_all = _seq_ids_from_locs(query_start_loc, num_tokens, eff_max_seqs)
        valid_all = token_idx < query_start_loc[distribution[2]]
        should_write = valid_all & ((pos & 1) == 0)
        write_slots = jnp.where(
            should_write,
            _state_slot_for(req_all, pos),
            total_state_slots,
        )
        wkv_words = jax.lax.bitcast_convert_type(
            wkv_wgate.reshape(num_tokens, 8, 128), jnp.int32
        )
    state_slots = state_slots.at[write_slots].set(wkv_words, mode="drop")

    kv0, score0 = prev_kv_score[:, :head_dim], prev_kv_score[:, head_dim:]
    kv1, score1 = wkv_wgate[:, :head_dim], wkv_wgate[:, head_dim:]
    max_s = jnp.maximum(score0, score1)
    e0, e1 = jnp.exp(score0 - max_s), jnp.exp(score1 - max_s)
    pf32 = ((e0 * kv0 + e1 * kv1) / (e0 + e1)).astype(jnp.bfloat16).astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(pf32 * pf32, axis=-1, keepdims=True) + rms_eps)
    latent = (pf32 * rstd).astype(jnp.bfloat16) * norm_weight.astype(jnp.bfloat16)
    return latent, state_slots.reshape(state_cache.shape)


def compressor_insert_cache_jax(
    latent: jax.Array,  # [num_tokens, 512] bf16
    cos_sin_cache: jax.Array,  # [max_pos, 64] f32
    positions: jax.Array,  # [num_tokens] i32
    kv_block_table: jax.Array,  # [max_num_seqs * max_kv_blocks] i32
    query_start_loc: jax.Array,  # [max_num_seqs + 1] i32
    distribution: jax.Array,  # [3] i32
    nope_cache: jax.Array,  # [num_pages, page_size, 128] i32 (donated)
    rope_cache: jax.Array,  # [num_pages, page_size // 4, 128] i32 (donated)
    *,
    compress_ratio: int,
) -> tuple[jax.Array, jax.Array]:
    """Rotate latent, quantize NoPE (FP8) + pack RoPE (BF16), and store."""
    num_tokens = latent.shape[0]
    max_num_seqs = query_start_loc.shape[0] - 1
    max_kv_blocks = kv_block_table.shape[0] // max_num_seqs
    eff_max_seqs = min(max_num_seqs, max(8, align_to(num_tokens, 8)))
    num_pages, page_size = nope_cache.shape[:2]
    total_kv_slots = num_pages * page_size

    pos = positions.astype(jnp.int32)
    req = _seq_ids_from_locs(query_start_loc, num_tokens, eff_max_seqs)
    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)
    valid = token_idx < query_start_loc[distribution[2]]
    is_boundary = (
        valid if compress_ratio == 1 else (valid & (((pos + 1) % compress_ratio) == 0))
    )

    use_pallas = num_tokens >= 384 and page_size % 32 == 0
    M = num_tokens
    latent_work = latent
    pos_work = pos
    if (use_pallas or num_tokens >= 384) and compress_ratio > 1:
        max_odd_seqs = eff_max_seqs if use_pallas else min(eff_max_seqs, 64)
        M = min(
            num_tokens,
            max(32, align_to((num_tokens + max_odd_seqs) // compress_ratio, 32)),
        )
        owned_idx = _compact_indices(is_boundary, M)
        safe_idx = jnp.maximum(owned_idx, 0)
        latent_work = latent[safe_idx]
        pos_work = pos[safe_idx]
        req = req[safe_idx]
        is_boundary = owned_idx >= 0

    kv_idx = (
        jnp.maximum(pos_work, 0)
        if compress_ratio == 1
        else jnp.maximum(pos_work // compress_ratio, 0)
    )
    kv_page_idx = jnp.clip(kv_idx // page_size, 0, max_kv_blocks - 1)
    kv_page_off = kv_idx % page_size
    kv_page = kv_block_table[req * max_kv_blocks + kv_page_idx]
    kv_slot = jnp.clip(kv_page * page_size + kv_page_off, 0, total_kv_slots - 1)

    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    f32_work = latent_work.astype(jnp.float32)
    r0 = f32_work[:, :128]
    r1 = f32_work[:, 128:256]
    r2 = f32_work[:, 256:384]
    r3_nope = f32_work[:, 384:448]
    x_rope = f32_work[:, 448:512]
    m012 = jnp.maximum(jnp.maximum(jnp.abs(r0), jnp.abs(r1)), jnp.abs(r2))
    folded = jnp.maximum(jnp.maximum(m012[:, :64], m012[:, 64:]), jnp.abs(r3_nope))
    amax = jnp.clip(folded, 1e-4, None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    inv_s = 1.0 / scale
    inv_s128 = jnp.concatenate([inv_s, inv_s], axis=-1)
    b0 = jax.lax.bitcast_convert_type(
        (r0 * inv_s128).astype(jnp.float8_e4m3fn), jnp.uint8
    ).astype(jnp.uint32)
    b1 = jax.lax.bitcast_convert_type(
        (r1 * inv_s128).astype(jnp.float8_e4m3fn), jnp.uint8
    ).astype(jnp.uint32)
    b2 = jax.lax.bitcast_convert_type(
        (r2 * inv_s128).astype(jnp.float8_e4m3fn), jnp.uint8
    ).astype(jnp.uint32)
    b3_lo = jax.lax.bitcast_convert_type(
        (r3_nope * inv_s).astype(jnp.float8_e4m3fn), jnp.uint8
    ).astype(jnp.uint32)
    b3_hi = (jax.lax.bitcast_convert_type(scale, jnp.uint32) >> 23) & 0xFF
    b3 = jnp.concatenate([b3_lo, b3_hi], axis=-1)
    nope_records = jax.lax.bitcast_convert_type(
        b0 | (b1 << 8) | (b2 << 16) | (b3 << 24), jnp.int32
    )

    start = (
        jnp.clip(pos_work, 0, cos_sin_cache.shape[0] - 1)
        if compress_ratio == 1
        else jnp.clip(
            (pos_work // compress_ratio) * compress_ratio, 0, cos_sin_cache.shape[0] - 1
        )
    )
    cs = jnp.repeat(cos_sin_cache[start], 2, axis=-1)
    cos, sin = cs[:, :64], cs[:, 64:]
    x_rot = jnp.stack([-x_rope[:, 1::2], x_rope[:, 0::2]], axis=-1).reshape(M, 64)
    y_rope = (x_rope * cos + x_rot * sin).astype(jnp.bfloat16)
    rope_records = csa_cache_layout.rope_words(y_rope)

    if not use_pallas:
        write_slot = jnp.where(is_boundary, kv_slot, total_kv_slots)
        nope_cache = (
            nope_cache.reshape((total_kv_slots, *nope_cache.shape[2:]))
            .at[write_slot]
            .set(nope_records, mode="drop")
            .reshape(nope_cache.shape)
        )
        eff_step = 1 if (num_tokens >= 384 and compress_ratio > 1) else compress_ratio
        rope_cache = _scatter_packed_4_jax(
            rope_cache.reshape(
                num_pages * (page_size // 4), csa_cache_layout.ROW_WORDS
            ),
            rope_records,
            kv_slot,
            is_boundary,
            eff_step,
        ).reshape(rope_cache.shape)
        return nope_cache, rope_cache

    tile_k = 256 if (num_tokens > 1536 and page_size % 256 == 0) else 32
    rope_tile = tile_k // 4
    tiles_per_page = page_size // tile_k
    tile_id = kv_slot // tile_k
    slot_off = kv_slot % tile_k
    valid_tile_id = jnp.where(is_boundary, tile_id, -1)

    same_as_prev = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=jnp.bool_),
            is_boundary[:-1]
            & (req[:-1] == req[1:])
            & (valid_tile_id[:-1] == valid_tile_id[1:])
            & (slot_off[1:] == slot_off[:-1] + 1),
        ],
        axis=0,
    )
    is_first = is_boundary & (~same_as_prev)

    max_seqs_with_1 = min(eff_max_seqs, M)
    max_seqs_with_2 = min(eff_max_seqs, M // 2)
    max_tiles = min(M, max_seqs_with_1 + max_seqs_with_2 + cdiv(M, tile_k))

    first_tok_idxs = _compact_indices(is_first, max_tiles)
    num_valid_b = jnp.sum(is_boundary.astype(jnp.int32))
    next_first = jnp.concatenate(
        [first_tok_idxs[1:], jnp.asarray([-1], dtype=jnp.int32)]
    )
    last_tok_idxs = jnp.where(
        first_tok_idxs >= 0,
        jnp.where(next_first >= 0, next_first - 1, num_valid_b - 1),
        -1,
    )
    n_active_tiles = jnp.minimum(jnp.sum(is_first.astype(jnp.int32)), max_tiles)

    safe_t0 = jnp.maximum(first_tok_idxs, 0)
    safe_t1 = jnp.maximum(last_tok_idxs, 0)
    tile_ids = jnp.where(first_tok_idxs >= 0, valid_tile_id[safe_t0], -1)
    slot0 = slot_off[safe_t0]
    slot_end = slot_off[safe_t1] + 1
    is_full = jnp.where(
        (first_tok_idxs >= 0) & (slot0 == 0) & (slot_end == tile_k), 1, 0
    ).astype(jnp.int32)

    num_m_tiles = cdiv(M, tile_k)
    pad_m = (num_m_tiles + 1) * tile_k - M
    nope_padded = jnp.pad(nope_records, ((0, pad_m), (0, 0)))
    nope_tiles = nope_padded.reshape(num_m_tiles + 1, tile_k, 128)
    rope_padded = jnp.pad(rope_records, ((0, pad_m), (0, 0)))
    rope_tiles = rope_padded.reshape(num_m_tiles + 1, rope_tile, 128)

    ring_depth = 4 if num_tokens >= 2048 else 2

    def _pallas_kernel(
        tile_ids_ref,
        safe_t0_ref,
        slot0_ref,
        slot_end_ref,
        is_full_ref,
        n_tiles_ref,
        nope_tiles_vmem_ref,
        rope_tiles_vmem_ref,
        nope_in_hbm_ref,
        rope_in_hbm_ref,
        nope_out_hbm_ref,
        rope_out_hbm_ref,
        vmem_nope_ref,
        vmem_rope_ref,
        sem_ref,
    ):
        sublanes_k = jax.lax.broadcasted_iota(jnp.int32, (tile_k, 128), 0)
        row_r = jax.lax.broadcasted_iota(jnp.int32, (rope_tile, 128), 0)
        q_iota = jax.lax.broadcasted_iota(jnp.int32, (rope_tile, 128), 1) // 32
        slot_rope = row_r * 4 + q_iota
        n_tiles = n_tiles_ref[0]

        def _tile_hbm_slice(idx):
            tid = tile_ids_ref[idx]
            safe_tid = jnp.maximum(tid, 0)
            p = safe_tid // tiles_per_page
            nope_off = (safe_tid % tiles_per_page) * tile_k
            rope_off = (safe_tid % tiles_per_page) * rope_tile
            return p, nope_off, rope_off

        def _start_read(idx, buf):
            p, nope_off, rope_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                nope_in_hbm_ref.at[p, pl.ds(nope_off, tile_k), :],
                vmem_nope_ref.at[buf],
                sem_ref.at[0, buf],
            ).start()
            pltpu.make_async_copy(
                rope_in_hbm_ref.at[p, pl.ds(rope_off, rope_tile), :],
                vmem_rope_ref.at[buf],
                sem_ref.at[1, buf],
            ).start()

        def _wait_read(idx, buf):
            p, nope_off, rope_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                nope_in_hbm_ref.at[p, pl.ds(nope_off, tile_k), :],
                vmem_nope_ref.at[buf],
                sem_ref.at[0, buf],
            ).wait()
            pltpu.make_async_copy(
                rope_in_hbm_ref.at[p, pl.ds(rope_off, rope_tile), :],
                vmem_rope_ref.at[buf],
                sem_ref.at[1, buf],
            ).wait()

        def _start_write(idx, buf):
            p, nope_off, rope_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                vmem_nope_ref.at[buf],
                nope_out_hbm_ref.at[p, pl.ds(nope_off, tile_k), :],
                sem_ref.at[2, buf],
            ).start()
            pltpu.make_async_copy(
                vmem_rope_ref.at[buf],
                rope_out_hbm_ref.at[p, pl.ds(rope_off, rope_tile), :],
                sem_ref.at[3, buf],
            ).start()

        def _wait_write(idx, buf):
            p, nope_off, rope_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                vmem_nope_ref.at[buf],
                nope_out_hbm_ref.at[p, pl.ds(nope_off, tile_k), :],
                sem_ref.at[2, buf],
            ).wait()
            pltpu.make_async_copy(
                vmem_rope_ref.at[buf],
                rope_out_hbm_ref.at[p, pl.ds(rope_off, rope_tile), :],
                sem_ref.at[3, buf],
            ).wait()

        @pl.loop(0, jnp.minimum(n_tiles, ring_depth))
        def _prologue(p_i):
            @pl.when(is_full_ref[p_i] == 0)
            def _read_p():
                _start_read(p_i, p_i)

        @pl.loop(0, n_tiles)
        def _step(i):
            cur_buf = i % ring_depth

            @pl.when(i >= ring_depth)
            def _wait_ring_write():
                _wait_write(i - ring_depth, cur_buf)

                @pl.when(is_full_ref[i] == 0)
                def _issue_late_read():
                    _start_read(i, cur_buf)

            tid = tile_ids_ref[i]
            t0 = safe_t0_ref[i]
            l0 = slot0_ref[i]
            l_end = slot_end_ref[i]
            full = is_full_ref[i]

            m_tile = t0 // tile_k
            m_off = t0 % tile_k
            shift = jnp.bitwise_and(l0 - m_off, tile_k - 1)

            @pl.when((full == 1) & (m_off == 0))
            def _store_full_aligned():
                vmem_nope_ref.at[cur_buf][...] = nope_tiles_vmem_ref[m_tile]
                vmem_rope_ref.at[cur_buf][...] = rope_tiles_vmem_ref[m_tile]

            @pl.when((full == 0) | (m_off != 0))
            def _store_unaligned_or_partial():
                n0 = nope_tiles_vmem_ref[m_tile]
                n1 = nope_tiles_vmem_ref[m_tile + 1]
                comb_nope = jnp.where(sublanes_k >= m_off, n0, n1)
                upd_nope = pltpu.roll(comb_nope, shift, axis=0)

                r0 = rope_tiles_vmem_ref[m_tile]
                r1 = rope_tiles_vmem_ref[m_tile + 1]
                comb_rope = jnp.where(slot_rope >= m_off, r0, r1)
                d_r = shift >> 2
                d_q = jnp.bitwise_and(shift, 3)
                lane_sh = pltpu.roll(comb_rope, d_q * 32, axis=1)
                sh0 = pltpu.roll(lane_sh, d_r, axis=0)
                sh1 = pltpu.roll(
                    lane_sh, jnp.bitwise_and(d_r + 1, rope_tile - 1), axis=0
                )
                upd_rope = jnp.where(q_iota < d_q, sh1, sh0)

                @pl.when(full == 1)
                def _store_full():
                    vmem_nope_ref.at[cur_buf][...] = upd_nope
                    vmem_rope_ref.at[cur_buf][...] = upd_rope

                @pl.when(full == 0)
                def _store_partial():
                    valid_nope = (tid >= 0) & (sublanes_k >= l0) & (sublanes_k < l_end)
                    valid_rope = (tid >= 0) & (slot_rope >= l0) & (slot_rope < l_end)
                    _wait_read(i, cur_buf)
                    vmem_nope_ref.at[cur_buf][...] = jnp.where(
                        valid_nope, upd_nope, vmem_nope_ref.at[cur_buf][...]
                    )
                    vmem_rope_ref.at[cur_buf][...] = jnp.where(
                        valid_rope, upd_rope, vmem_rope_ref.at[cur_buf][...]
                    )

            _start_write(i, cur_buf)

        @pl.loop(jnp.maximum(0, n_tiles - ring_depth), n_tiles)
        def _epilogue(e_i):
            _wait_write(e_i, e_i % ring_depth)

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        _pallas_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            grid=(1,),
            in_specs=[vmem_spec, vmem_spec, hbm_spec, hbm_spec],
            out_specs=[hbm_spec, hbm_spec],
            scratch_shapes=[
                pltpu.VMEM((ring_depth, tile_k, 128), jnp.int32),
                pltpu.VMEM((ring_depth, rope_tile, 128), jnp.int32),
                pltpu.SemaphoreType.DMA((4, ring_depth)),
            ],
        ),
        out_shape=(
            jax.ShapeDtypeStruct(nope_cache.shape, nope_cache.dtype),
            jax.ShapeDtypeStruct(rope_cache.shape, rope_cache.dtype),
        ),
        input_output_aliases={8: 0, 9: 1},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name=f"compressor_insert_cache_r{compress_ratio}",
    )(
        tile_ids,
        safe_t0,
        slot0,
        slot_end,
        is_full,
        jnp.asarray([n_active_tiles], dtype=jnp.int32),
        nope_tiles,
        rope_tiles,
        nope_cache,
        rope_cache,
    )


def _scatter_packed_4_jax(
    cache_2d: jax.Array,  # [total_rows, 128] i32
    records: jax.Array,  # [num_tokens, 32] i32
    kv_slot: jax.Array,  # [num_tokens] i32
    is_boundary: jax.Array,  # [num_tokens] bool
    step: int,
) -> jax.Array:
    """Scatter [num_tokens, 32] RoPE word records into [total_rows, 128]."""
    num_tokens, total_rows = records.shape[0], cache_2d.shape[0]
    row_id = kv_slot // 4
    slot_mod = kv_slot % 4
    if step < num_tokens:
        prev_b = jnp.concatenate(
            [jnp.zeros((step,), dtype=jnp.bool_), is_boundary[:-step]], axis=0
        )
        prev_r = jnp.concatenate(
            [jnp.full((step,), -1, dtype=row_id.dtype), row_id[:-step]], axis=0
        )
        same_as_prev = prev_b & (prev_r == row_id)
    else:
        same_as_prev = jnp.zeros_like(is_boundary)
    is_first = is_boundary & (~same_as_prev)

    if num_tokens >= 192:
        K = min(num_tokens, align_to(cdiv(num_tokens, 4 * step) + 96, 32))
        if num_tokens > K:
            first_idx = _compact_indices(is_first, K)
            safe_f = jnp.maximum(first_idx, 0)
            f_row = row_id[safe_f]
            write_row = jnp.where(first_idx >= 0, f_row, total_rows)

            pad_len = 3 * step
            b_pad = jnp.pad(is_boundary, (0, pad_len), constant_values=False)
            r_pad = jnp.pad(row_id, (0, pad_len), constant_values=-1)
            m_pad = jnp.pad(slot_mod, (0, pad_len), constant_values=-1)
            rec_pad = jnp.pad(records, ((0, pad_len), (0, 0)))

            lane_q = (
                jnp.arange(csa_cache_layout.ROW_WORDS, dtype=jnp.int32)
                // csa_cache_layout.ROPE_WORDS
            )[None, :]
            tiles = cache_2d[jnp.clip(f_row, 0, total_rows - 1)]
            for k in range(4):
                idx_k = safe_f + k * step
                valid_k = (first_idx >= 0) & b_pad[idx_k] & (r_pad[idx_k] == f_row)
                mask_k = valid_k[:, None] & (m_pad[idx_k, None] == lane_q)
                tiles = jnp.where(mask_k, jnp.tile(rec_pad[idx_k], (1, 4)), tiles)
            return cache_2d.at[write_row].set(tiles, mode="drop")

    write_row = jnp.where(is_first, row_id, total_rows)

    pad_len = 3 * step
    records_128 = jnp.tile(records, (1, 4))
    b_pad = jnp.pad(is_boundary, (0, pad_len), constant_values=False)
    r_pad = jnp.pad(row_id, (0, pad_len), constant_values=-1)
    m_pad = jnp.pad(slot_mod, (0, pad_len), constant_values=-1)
    rec_pad = jnp.pad(records_128, ((0, pad_len), (0, 0)))

    lane_q = (
        jnp.arange(csa_cache_layout.ROW_WORDS, dtype=jnp.int32)
        // csa_cache_layout.ROPE_WORDS
    )[None, :]
    tiles = cache_2d[jnp.clip(row_id, 0, total_rows - 1)]
    for k in range(4):
        s = k * step
        valid_k = b_pad[s : s + num_tokens] & (r_pad[s : s + num_tokens] == row_id)
        mask_k = valid_k[:, None] & (m_pad[s : s + num_tokens, None] == lane_q)
        tiles = jnp.where(mask_k, rec_pad[s : s + num_tokens], tiles)
    return cache_2d.at[write_row].set(tiles, mode="drop")


def indexer_produce_k_jax(
    k_normed: jax.Array,  # [num_tokens, 128] bf16
    cos_sin_cache: jax.Array,  # [max_pos, 64] f32
    positions: jax.Array,  # [num_tokens] i32
    kv_block_table: jax.Array,  # [max_num_seqs * max_kv_blocks] i32
    query_start_loc: jax.Array,  # [max_num_seqs + 1] i32
    distribution: jax.Array,  # [3] i32
    k_cache: jax.Array,  # [num_pages, 33, 4, page_size] u8 (donated)
    *,
    compress_ratio: int,
) -> jax.Array:
    """Rotate k_normed, quantize to 128 FP8 + E8M0 scale, and store."""
    M = k_normed.shape[0]
    num_tokens = positions.shape[0]
    max_num_seqs = query_start_loc.shape[0] - 1
    max_kv_blocks = kv_block_table.shape[0] // max_num_seqs
    eff_max_seqs = min(max_num_seqs, max(8, align_to(num_tokens, 8)))
    num_pages, sublane_groups, pack, page_size = k_cache.shape
    tile_k = 256 if (num_tokens >= 1024 and page_size % 256 == 0) else 128
    tiles_per_page = page_size // tile_k
    block_stride = max(1, (page_size * compress_ratio) // 1024)

    pos = positions.astype(jnp.int32)
    req = _seq_ids_from_locs(query_start_loc, num_tokens, eff_max_seqs)
    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)
    valid = token_idx < query_start_loc[distribution[2]]
    is_boundary = (
        valid if compress_ratio == 1 else (valid & (((pos + 1) % compress_ratio) == 0))
    )

    pos_work = pos
    if compress_ratio > 1:
        # Compact to boundary tokens so the tile loop only visits written rows.
        M = min(
            num_tokens,
            max(
                tile_k, align_to((num_tokens + eff_max_seqs) // compress_ratio, tile_k)
            ),
        )
        owned_idx = _compact_indices(is_boundary, M)
        safe_idx = jnp.maximum(owned_idx, 0)
        k_normed = k_normed[safe_idx]
        pos_work = pos[safe_idx]
        req = req[safe_idx]
        is_boundary = owned_idx >= 0

    kv_idx = (
        jnp.maximum(pos_work, 0)
        if compress_ratio == 1
        else jnp.maximum(pos_work // compress_ratio, 0)
    )
    kv_page_idx = jnp.clip((kv_idx // page_size) * block_stride, 0, max_kv_blocks - 1)
    kv_page_off = kv_idx % page_size
    kv_page = kv_block_table[req * max_kv_blocks + kv_page_idx]

    roped_f32 = _rope_block_start(k_normed, pos_work, cos_sin_cache, compress_ratio)
    if num_tokens >= 96:
        fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
        amax = jnp.clip(jnp.max(jnp.abs(roped_f32), axis=-1, keepdims=True), 1e-4, None)
        scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
        q = (roped_f32 * (1.0 / scale)).astype(jnp.float8_e4m3fn)
        q_i32 = jax.lax.bitcast_convert_type(q.reshape(M, 32, 4), jnp.int32)
        scale_i32 = (
            (jax.lax.bitcast_convert_type(scale, jnp.uint32) >> 23) & 0xFF
        ).astype(jnp.int32)
    else:
        q, scale = compute.quantize_fp8_tiled(roped_f32[:, None, :], 128)
        q_i32 = jax.lax.bitcast_convert_type(q.reshape(M, 32, 4), jnp.int32)
        scale_i32 = (
            jax.lax.bitcast_convert_type(scale, jnp.uint8)
            .astype(jnp.int32)
            .reshape(M, 1)
        )
    k_records_i32 = jnp.concatenate([q_i32, scale_i32], axis=1)

    kv_page = jnp.clip(kv_page, 0, num_pages - 1)
    tile_id = kv_page * tiles_per_page + (kv_page_off // tile_k)
    lane_off = kv_page_off % tile_k
    valid_tile_id = jnp.where(is_boundary, tile_id, -1)

    same_as_prev = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=jnp.bool_),
            is_boundary[:-1]
            & (req[:-1] == req[1:])
            & (valid_tile_id[:-1] == valid_tile_id[1:])
            & (lane_off[1:] == lane_off[:-1] + 1),
        ],
        axis=0,
    )
    is_first = is_boundary & (~same_as_prev)

    max_seqs_with_1 = min(eff_max_seqs, M)
    max_seqs_with_2 = min(eff_max_seqs, M // 2)
    max_tiles = min(
        M,
        max_seqs_with_1 + max_seqs_with_2 + cdiv(M, tile_k),
    )
    first_tok_idxs = _compact_indices(is_first, max_tiles)
    num_valid_b = jnp.sum(is_boundary.astype(jnp.int32))
    next_first = jnp.concatenate(
        [first_tok_idxs[1:], jnp.asarray([-1], dtype=jnp.int32)]
    )
    last_tok_idxs = jnp.where(
        first_tok_idxs >= 0,
        jnp.where(next_first >= 0, next_first - 1, num_valid_b - 1),
        -1,
    )
    n_active_tiles = jnp.minimum(jnp.sum(is_first.astype(jnp.int32)), max_tiles)

    safe_t0 = jnp.maximum(first_tok_idxs, 0)
    safe_t1 = jnp.maximum(last_tok_idxs, 0)
    tile_ids = jnp.where(first_tok_idxs >= 0, valid_tile_id[safe_t0], -1)
    lane0 = lane_off[safe_t0]
    lane_end = lane_off[safe_t1] + 1
    is_full = jnp.where(
        (first_tok_idxs >= 0) & (lane0 == 0) & (lane_end == tile_k), 1, 0
    ).astype(jnp.int32)

    padded_groups = align_to(sublane_groups, 8)
    pad_groups = padded_groups - sublane_groups
    num_m_tiles = cdiv(M, tile_k)
    pad_m = (num_m_tiles + 1) * tile_k - M
    tok_words_padded = jnp.pad(k_records_i32, ((0, pad_m), (0, pad_groups)))
    tok_tiles = tok_words_padded.reshape(
        num_m_tiles + 1, tile_k, padded_groups
    ).transpose(0, 2, 1)

    # SEQ_ALONG_LANE puts tokens on lanes, so each touched tile is either
    # written directly when full or read-merged-written when partial.
    ring_depth = 4 if num_tokens >= 2048 else 2

    def _pallas_kernel(
        tile_ids_ref,
        safe_t0_ref,
        lane0_ref,
        lane_end_ref,
        is_full_ref,
        n_tiles_ref,
        tok_tiles_vmem_ref,
        cache_in_hbm_ref,
        cache_out_hbm_ref,
        vmem_cache_ref,
        sem_ref,
    ):
        lanes_full = jax.lax.broadcasted_iota(jnp.int32, (padded_groups, tile_k), 1)
        lane_1d = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
        n_tiles = n_tiles_ref[0]

        def _tile_hbm_slice(idx):
            tid = tile_ids_ref[idx]
            safe_tid = jnp.maximum(tid, 0)
            p = safe_tid // tiles_per_page
            t_off = (safe_tid % tiles_per_page) * tile_k
            return p, t_off

        def _start_read(idx, buf):
            p, t_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                cache_in_hbm_ref.at[p, :, :, pl.ds(t_off, tile_k)],
                vmem_cache_ref.at[buf],
                sem_ref.at[0, buf],
            ).start()

        def _wait_read(idx, buf):
            p, t_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                cache_in_hbm_ref.at[p, :, :, pl.ds(t_off, tile_k)],
                vmem_cache_ref.at[buf],
                sem_ref.at[0, buf],
            ).wait()

        def _start_write(idx, buf):
            p, t_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                vmem_cache_ref.at[buf],
                cache_out_hbm_ref.at[p, :, :, pl.ds(t_off, tile_k)],
                sem_ref.at[1, buf],
            ).start()

        def _wait_write(idx, buf):
            p, t_off = _tile_hbm_slice(idx)
            pltpu.make_async_copy(
                vmem_cache_ref.at[buf],
                cache_out_hbm_ref.at[p, :, :, pl.ds(t_off, tile_k)],
                sem_ref.at[1, buf],
            ).wait()

        @pl.loop(0, jnp.minimum(n_tiles, ring_depth))
        def _prologue(p_i):
            @pl.when(is_full_ref[p_i] == 0)
            def _read_p():
                _start_read(p_i, p_i)

        @pl.loop(0, n_tiles)
        def _step(i):
            cur_buf = i % ring_depth

            @pl.when(i >= ring_depth)
            def _wait_ring_write():
                _wait_write(i - ring_depth, cur_buf)

                @pl.when(is_full_ref[i] == 0)
                def _issue_late_read():
                    _start_read(i, cur_buf)

            tid = tile_ids_ref[i]
            t0 = safe_t0_ref[i]
            l0 = lane0_ref[i]
            l_end = lane_end_ref[i]
            full = is_full_ref[i]

            m_tile = t0 // tile_k
            m_off = t0 % tile_k
            shift = jnp.bitwise_and(l0 - m_off, tile_k - 1)

            @pl.when((full == 1) & (m_off == 0))
            def _store_full_aligned():
                tile0 = tok_tiles_vmem_ref[m_tile]
                vmem_cache_ref.at[cur_buf][...] = pltpu.bitcast(
                    tile0[:sublane_groups, :], jnp.uint8
                ).reshape(sublane_groups, pack, tile_k)

            @pl.when((full == 0) | (m_off != 0))
            def _store_unaligned_or_partial():
                tile0 = tok_tiles_vmem_ref[m_tile]
                tile1 = tok_tiles_vmem_ref[m_tile + 1]
                combined = jnp.where(lanes_full >= m_off, tile0, tile1)
                rolled = pltpu.roll(combined, shift, axis=1)
                upd_u8 = pltpu.bitcast(rolled[:sublane_groups, :], jnp.uint8).reshape(
                    sublane_groups, pack, tile_k
                )

                @pl.when(full == 1)
                def _store_full():
                    vmem_cache_ref.at[cur_buf][...] = upd_u8

                @pl.when(full == 0)
                def _store_partial():
                    _wait_read(i, cur_buf)
                    if tile_k == 128:
                        valid_mask = (tid >= 0) & (lane_1d >= l0) & (lane_1d < l_end)
                        vmem_cache_ref.at[cur_buf][...] = jnp.where(
                            valid_mask[:, None, :],
                            upd_u8,
                            vmem_cache_ref.at[cur_buf][...],
                        )
                    else:
                        valid0 = (tid >= 0) & (lane_1d >= l0) & (lane_1d < l_end)
                        lane_hi = lane_1d + 128
                        valid1 = (tid >= 0) & (lane_hi >= l0) & (lane_hi < l_end)
                        cur0 = vmem_cache_ref.at[cur_buf, :, :, pl.ds(0, 128)][...]
                        cur1 = vmem_cache_ref.at[cur_buf, :, :, pl.ds(128, 128)][...]
                        vmem_cache_ref.at[cur_buf, :, :, pl.ds(0, 128)][...] = (
                            jnp.where(valid0[:, None, :], upd_u8[:, :, :128], cur0)
                        )
                        vmem_cache_ref.at[cur_buf, :, :, pl.ds(128, 128)][...] = (
                            jnp.where(valid1[:, None, :], upd_u8[:, :, 128:], cur1)
                        )

            _start_write(i, cur_buf)

        @pl.loop(jnp.maximum(0, n_tiles - ring_depth), n_tiles)
        def _epilogue(e_i):
            _wait_write(e_i, e_i % ring_depth)

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    vmem_tile = pltpu.VMEM((ring_depth, sublane_groups, pack, tile_k), jnp.uint8)
    return pl.pallas_call(
        _pallas_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            grid=(1,),
            in_specs=[vmem_spec, hbm_spec],
            out_specs=hbm_spec,
            scratch_shapes=[vmem_tile, pltpu.SemaphoreType.DMA((2, ring_depth))],
        ),
        out_shape=jax.ShapeDtypeStruct(k_cache.shape, k_cache.dtype),
        input_output_aliases={7: 0},
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
    )(
        tile_ids,
        safe_t0,
        lane0,
        lane_end,
        is_full,
        jnp.asarray([n_active_tiles], dtype=jnp.int32),
        tok_tiles,
        k_cache,
    )


def fused_indexer_produce_k_jax(
    latent: jax.Array,  # [num_tokens, 512] bf16
    wk_weight: jax.Array,  # [128, 512] or [512, 128] bf16 or fp8
    knorm_weight: jax.Array,  # [128] bf16/f32
    cos_sin_cache: jax.Array,  # [max_pos, 64] f32
    positions: jax.Array,  # [num_tokens] i32
    kv_block_table: jax.Array,  # [max_num_seqs * max_kv_blocks] i32
    query_start_loc: jax.Array,  # [max_num_seqs + 1] i32
    distribution: jax.Array,  # [3] i32
    k_cache: jax.Array,  # [num_pages, 33, 4, page_size] u8 (donated)
    wk_weight_scale: jax.Array | None = None,
    *,
    compress_ratio: int,
    rms_eps: float = 1e-6,
) -> jax.Array:
    """Single fused custom op: wk + k_norm (RMSNorm) + indexer_produce_k_jax."""
    rhs_dim = 1 if wk_weight.shape[0] == knorm_weight.shape[0] else 0
    if wk_weight_scale is not None:
        proj = jax.lax.dot_general(
            latent.astype(wk_weight.dtype),
            wk_weight,
            (((1,), (rhs_dim,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        k_raw = jax.lax.optimization_barrier(
            (proj * wk_weight_scale.reshape(1, -1)).astype(jnp.bfloat16)
        )
    else:
        k_raw = jax.lax.optimization_barrier(
            jax.lax.dot_general(
                latent.astype(jnp.bfloat16),
                wk_weight.astype(jnp.bfloat16),
                (((1,), (rhs_dim,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
        )
    k_f32 = k_raw.astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(k_f32 * k_f32, axis=-1, keepdims=True) + rms_eps)
    k_normed = (k_f32 * rstd).astype(jnp.bfloat16) * knorm_weight.astype(jnp.bfloat16)
    return indexer_produce_k_jax(
        k_normed,
        cos_sin_cache,
        positions,
        kv_block_table,
        query_start_loc,
        distribution,
        k_cache,
        compress_ratio=compress_ratio,
    )


def _fused_quant_pack_and_single_pallas_scatter(
    latent: jax.Array,  # [num_tokens, 512] bf16
    wk_weight: jax.Array,  # [128, 512] bf16/fp8
    knorm_weight: jax.Array,  # [128] bf16
    cos_sin_cache: jax.Array,  # [max_pos, 64] f32
    idx_cos_sin_cache: jax.Array,  # [max_pos, 64] f32
    positions: jax.Array,  # [num_tokens] i32
    kv_block_table: jax.Array,  # [max_num_seqs * max_kv_blocks] i32
    idx_block_table: jax.Array,  # [max_num_seqs * max_idx_blocks] i32
    query_start_loc: jax.Array,  # [max_num_seqs + 1] i32
    distribution: jax.Array,  # [3] i32
    nope_cache: jax.Array,  # donated: [num_kv_pages, kv_page_size, 128] i32
    rope_cache: jax.Array,  # donated: [num_kv_pages, kv_page_size // 4, 128] i32
    k_cache: jax.Array,  # donated: [num_idx_pages, 33, 4, idx_page_size] u8
    wk_weight_scale: jax.Array | None = None,
    *,
    compress_ratio: int,
    idx_rms_eps: float = 1e-6,
    state_slots_in: jax.Array | None = None,
    st_write_slots: jax.Array | None = None,
    st_should_write: jax.Array | None = None,
    st_wkv_words: jax.Array | None = None,
    n_st_writes: jax.Array | None = None,
):
    """Single XLA quant/pack prelude + single concurrent 3/4-cache Pallas async DMA kernel."""
    num_tokens = latent.shape[0]
    max_num_seqs = query_start_loc.shape[0] - 1
    eff_max_seqs = min(max_num_seqs, max(8, align_to(num_tokens, 8)))
    pos = positions.astype(jnp.int32)
    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)

    rhs_dim = 1 if wk_weight.shape[0] == knorm_weight.shape[0] else 0
    if wk_weight_scale is not None:
        proj = jax.lax.dot_general(
            latent.astype(wk_weight.dtype),
            wk_weight,
            (((1,), (rhs_dim,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        k_raw = jax.lax.optimization_barrier(
            (proj * wk_weight_scale.reshape(1, -1)).astype(jnp.bfloat16)
        )
    else:
        k_raw = jax.lax.optimization_barrier(
            jax.lax.dot_general(
                latent.astype(jnp.bfloat16),
                wk_weight.astype(jnp.bfloat16),
                (((1,), (rhs_dim,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(jnp.bfloat16)
        )
    k_f32 = k_raw.astype(jnp.float32)
    k_rstd = jax.lax.rsqrt(
        jnp.mean(k_f32 * k_f32, axis=-1, keepdims=True) + idx_rms_eps
    )
    k_normed_all = (k_f32 * k_rstd).astype(jnp.bfloat16) * knorm_weight.astype(
        jnp.bfloat16
    )

    valid = token_idx < query_start_loc[distribution[2]]
    is_boundary_all = (
        valid if compress_ratio == 1 else (valid & (((pos + 1) % compress_ratio) == 0))
    )

    num_kv_pages, kv_page_size = nope_cache.shape[:2]
    num_idx_pages, sublane_groups, pack, idx_page_size = k_cache.shape
    max_kv_blocks = kv_block_table.shape[0] // max_num_seqs
    max_idx_blocks = idx_block_table.shape[0] // max_num_seqs

    tile_k = (
        256
        if (num_tokens > 1536 and kv_page_size % 256 == 0 and idx_page_size % 256 == 0)
        else 128
    )
    use_narrow_nr = num_tokens <= 1536
    nr_dma = 16
    rope_dma = 16

    M = (
        num_tokens
        if compress_ratio == 1
        else min(
            num_tokens,
            max(
                tile_k, align_to((num_tokens + eff_max_seqs) // compress_ratio, tile_k)
            ),
        )
    )

    if compress_ratio == 2 and num_tokens // 2 == M:
        even_is_b = is_boundary_all[0]

        def _fast_strided_idx():
            base = jnp.arange(M, dtype=jnp.int32) * 2 + jnp.where(even_is_b, 0, 1)
            return jnp.where(base < query_start_loc[distribution[2]], base, -1)

        owned_idx = jax.lax.cond(
            distribution[2] <= 1,
            _fast_strided_idx,
            lambda: _compact_indices(is_boundary_all, M),
        )
    elif compress_ratio > 1:
        owned_idx = _compact_indices(is_boundary_all, M)
    else:
        owned_idx = token_idx

    if compress_ratio > 1:
        safe_idx = jnp.maximum(owned_idx, 0)
        latent_work = latent[safe_idx]
        k_normed = k_normed_all[safe_idx]
        pos_work = pos[safe_idx]
        req = jax.lax.cond(
            distribution[2] <= 1,
            lambda: jnp.zeros((M,), dtype=jnp.int32),
            lambda: _seq_ids_from_locs(query_start_loc, num_tokens, eff_max_seqs)[
                safe_idx
            ],
        )
        is_boundary = owned_idx >= 0
    else:
        latent_work = latent
        k_normed = k_normed_all
        pos_work = pos
        req = jax.lax.cond(
            distribution[2] <= 1,
            lambda: jnp.zeros((M,), dtype=jnp.int32),
            lambda: _seq_ids_from_locs(query_start_loc, num_tokens, eff_max_seqs),
        )
        is_boundary = is_boundary_all

    kv_idx = (
        jnp.maximum(pos_work, 0)
        if compress_ratio == 1
        else jnp.maximum(pos_work // compress_ratio, 0)
    )

    fp8_max = float(jnp.finfo(jnp.float8_e4m3fn).max)
    f32_work = latent_work.astype(jnp.float32)
    r0, r1, r2 = f32_work[:, :128], f32_work[:, 128:256], f32_work[:, 256:384]
    r3_nope, x_rope = f32_work[:, 384:448], f32_work[:, 448:512]
    m012 = jnp.maximum(jnp.maximum(jnp.abs(r0), jnp.abs(r1)), jnp.abs(r2))
    folded = jnp.maximum(jnp.maximum(m012[:, :64], m012[:, 64:]), jnp.abs(r3_nope))
    amax = jnp.clip(folded, 1e-4, None)
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    inv_s = 1.0 / scale
    inv_s128 = jnp.concatenate([inv_s, inv_s], axis=-1)
    b0 = jax.lax.bitcast_convert_type(
        (r0 * inv_s128).astype(jnp.float8_e4m3fn), jnp.uint8
    ).astype(jnp.uint32)
    b1 = jax.lax.bitcast_convert_type(
        (r1 * inv_s128).astype(jnp.float8_e4m3fn), jnp.uint8
    ).astype(jnp.uint32)
    b2 = jax.lax.bitcast_convert_type(
        (r2 * inv_s128).astype(jnp.float8_e4m3fn), jnp.uint8
    ).astype(jnp.uint32)
    b3_lo = jax.lax.bitcast_convert_type(
        (r3_nope * inv_s).astype(jnp.float8_e4m3fn), jnp.uint8
    ).astype(jnp.uint32)
    b3_hi = (jax.lax.bitcast_convert_type(scale, jnp.uint32) >> 23) & 0xFF
    b3 = jnp.concatenate([b3_lo, b3_hi], axis=-1)
    nope_records = jax.lax.bitcast_convert_type(
        b0 | (b1 << 8) | (b2 << 16) | (b3 << 24), jnp.int32
    )

    start = (
        jnp.clip(pos_work, 0, cos_sin_cache.shape[0] - 1)
        if compress_ratio == 1
        else jnp.clip(
            (pos_work // compress_ratio) * compress_ratio, 0, cos_sin_cache.shape[0] - 1
        )
    )
    cs = jnp.repeat(cos_sin_cache[start], 2, axis=-1)
    cos, sin = cs[:, :64], cs[:, 64:]
    x_rot = jnp.stack([-x_rope[:, 1::2], x_rope[:, 0::2]], axis=-1).reshape(M, 64)
    y_rope = (x_rope * cos + x_rot * sin).astype(jnp.bfloat16)
    rope_records = csa_cache_layout.rope_words(y_rope)

    roped_f32 = _rope_block_start(k_normed, pos_work, idx_cos_sin_cache, compress_ratio)
    if num_tokens >= 96:
        k_amax = jnp.clip(
            jnp.max(jnp.abs(roped_f32), axis=-1, keepdims=True), 1e-4, None
        )
        k_scale = jnp.exp2(jnp.ceil(jnp.log2(k_amax / fp8_max)))
        k_q = (roped_f32 * (1.0 / k_scale)).astype(jnp.float8_e4m3fn)
        q_i32 = jax.lax.bitcast_convert_type(k_q.reshape(M, 32, 4), jnp.int32)
        scale_i32 = (
            (jax.lax.bitcast_convert_type(k_scale, jnp.uint32) >> 23) & 0xFF
        ).astype(jnp.int32)
    else:
        k_q, k_scale = compute.quantize_fp8_tiled(roped_f32[:, None, :], 128)
        q_i32 = jax.lax.bitcast_convert_type(k_q.reshape(M, 32, 4), jnp.int32)
        scale_i32 = (
            jax.lax.bitcast_convert_type(k_scale, jnp.uint8)
            .astype(jnp.int32)
            .reshape(M, 1)
        )
    k_records_i32 = jnp.concatenate([q_i32, scale_i32], axis=1)

    rope_tile = tile_k // 4
    tiles_per_page = kv_page_size // tile_k
    idx_block_stride = max(1, (idx_page_size * compress_ratio) // 1024)

    kv_page_idx = jnp.clip(kv_idx // kv_page_size, 0, max_kv_blocks - 1)
    kv_page_off = kv_idx % kv_page_size
    kv_page = jnp.clip(
        kv_block_table[req * max_kv_blocks + kv_page_idx], 0, num_kv_pages - 1
    )

    idx_page_idx = jnp.clip(
        (kv_idx // idx_page_size) * idx_block_stride, 0, max_idx_blocks - 1
    )
    idx_page = jnp.clip(
        idx_block_table[req * max_idx_blocks + idx_page_idx], 0, num_idx_pages - 1
    )

    tile_id = kv_page * tiles_per_page + (kv_page_off // tile_k)
    lane_off = kv_page_off % tile_k
    valid_tile_id = jnp.where(is_boundary, tile_id, -1)

    same_as_prev = jnp.concatenate(
        [
            jnp.zeros((1,), dtype=jnp.bool_),
            is_boundary[:-1]
            & (req[:-1] == req[1:])
            & (valid_tile_id[:-1] == valid_tile_id[1:])
            & (idx_page[:-1] == idx_page[1:])
            & (lane_off[1:] == lane_off[:-1] + 1),
        ],
        axis=0,
    )
    is_first = is_boundary & (~same_as_prev)

    max_seqs_with_1 = min(eff_max_seqs, M)
    max_seqs_with_2 = min(eff_max_seqs, M // 2)
    max_tiles = min(M, max_seqs_with_1 + max_seqs_with_2 + cdiv(M, tile_k))

    first_tok_idxs = _compact_indices(is_first, max_tiles)
    num_valid_b = jnp.sum(is_boundary.astype(jnp.int32))
    next_first = jnp.concatenate(
        [first_tok_idxs[1:], jnp.asarray([-1], dtype=jnp.int32)]
    )
    last_tok_idxs = jnp.where(
        first_tok_idxs >= 0,
        jnp.where(next_first >= 0, next_first - 1, num_valid_b - 1),
        -1,
    )
    n_active_tiles = jnp.minimum(jnp.sum(is_first.astype(jnp.int32)), max_tiles)

    safe_t0 = jnp.maximum(first_tok_idxs, 0)
    safe_t1 = jnp.maximum(last_tok_idxs, 0)
    tile_ids = jnp.where(first_tok_idxs >= 0, valid_tile_id[safe_t0], -1)
    idx_pages = jnp.where(first_tok_idxs >= 0, idx_page[safe_t0], 0)
    lanes0 = lane_off[safe_t0]
    lanes_end = lane_off[safe_t1] + 1
    is_full = jnp.where(
        (first_tok_idxs >= 0) & (lanes0 == 0) & (lanes_end == tile_k), 1, 0
    ).astype(jnp.int32)

    num_m_tiles = cdiv(M, tile_k)
    pad_toks = (num_m_tiles + 1) * tile_k - M
    nope_tiles = jnp.pad(nope_records, ((0, pad_toks), (0, 0))).reshape(
        num_m_tiles + 1, tile_k, 128
    )
    rope_tiles = jnp.pad(rope_records, ((0, pad_toks), (0, 0))).reshape(
        num_m_tiles + 1, rope_tile, 128
    )

    padded_groups = align_to(sublane_groups, 8)
    pad_groups = padded_groups - sublane_groups
    tok_words_padded = jnp.pad(k_records_i32, ((0, pad_toks), (0, pad_groups)))
    k_tiles = tok_words_padded.reshape(
        num_m_tiles + 1, tile_k, padded_groups
    ).transpose(0, 2, 1)

    has_state_cache = st_write_slots is not None
    state_in_nope = has_state_cache and state_slots_in is None
    states_per_page_dma = nope_cache.shape[1] // 8
    # State-write loops run over st_should_write's length: eff_max_seqs for
    # last-token writes (prefill role), num_tokens for per-token writes (decode).
    # Prefill role: 8-deep VMEM ring, reads prefetched half_ring tiles ahead.
    # Decode role: the decode stack's 4-deep ring (read issued after the write
    # retire on the same buffer). sbuf = vbuf % half_ring == vbuf when the
    # ring is 4 deep, so the shared helpers below trace the decode kernel.
    deep_ring = _prefill_role()
    half_ring = 4
    ring_depth = 8 if deep_ring else 4

    def _single_cache_kernel(*args):
        if state_in_nope:
            (
                tile_ids_ref,
                idx_pages_ref,
                safe_t0_ref,
                lanes0_ref,
                lanes_end_ref,
                is_full_ref,
                st_slots_ref,
                st_should_ref,
                n_tiles_ref,
                st_words_vmem_ref,
                nope_tiles_vmem_ref,
                rope_tiles_vmem_ref,
                k_tiles_vmem_ref,
                nope_in_hbm_ref,
                rope_in_hbm_ref,
                k_in_hbm_ref,
                nope_out_hbm_ref,
                rope_out_hbm_ref,
                k_out_hbm_ref,
                vmem_nope_ref,
                vmem_rope_ref,
                vmem_k_ref,
                sem_ref,
                st_sem_ref,
                *wide_scratch,
            ) = args

            @pl.loop(0, st_should_ref.shape[0])
            def _issue_state_writes(s_i):
                @pl.when(st_should_ref[s_i] == 1)
                def _do_st():
                    slot = st_slots_ref[s_i]
                    p_st = slot // states_per_page_dma
                    r_st = (slot % states_per_page_dma) * 8
                    pltpu.make_async_copy(
                        st_words_vmem_ref.at[s_i],
                        nope_out_hbm_ref.at[p_st, pl.ds(r_st, 8), :],
                        st_sem_ref.at[0],
                    ).start()
        elif has_state_cache:
            (
                tile_ids_ref,
                idx_pages_ref,
                safe_t0_ref,
                lanes0_ref,
                lanes_end_ref,
                is_full_ref,
                st_slots_ref,
                st_should_ref,
                n_tiles_ref,
                st_words_vmem_ref,
                nope_tiles_vmem_ref,
                rope_tiles_vmem_ref,
                k_tiles_vmem_ref,
                st_in_hbm_ref,
                nope_in_hbm_ref,
                rope_in_hbm_ref,
                k_in_hbm_ref,
                st_out_hbm_ref,
                nope_out_hbm_ref,
                rope_out_hbm_ref,
                k_out_hbm_ref,
                vmem_nope_ref,
                vmem_rope_ref,
                vmem_k_ref,
                sem_ref,
                st_sem_ref,
                *wide_scratch,
            ) = args
            n_st = n_tiles_ref[1]

            @pl.loop(0, st_should_ref.shape[0])
            def _issue_state_writes(s_i):
                @pl.when(st_should_ref[s_i] == 1)
                def _do_st():
                    slot = st_slots_ref[s_i]
                    pltpu.make_async_copy(
                        st_words_vmem_ref.at[s_i],
                        st_out_hbm_ref.at[slot],
                        st_sem_ref.at[0],
                    ).start()
        else:
            (
                tile_ids_ref,
                idx_pages_ref,
                safe_t0_ref,
                lanes0_ref,
                lanes_end_ref,
                is_full_ref,
                n_tiles_ref,
                nope_tiles_vmem_ref,
                rope_tiles_vmem_ref,
                k_tiles_vmem_ref,
                nope_in_hbm_ref,
                rope_in_hbm_ref,
                k_in_hbm_ref,
                nope_out_hbm_ref,
                rope_out_hbm_ref,
                k_out_hbm_ref,
                vmem_nope_ref,
                vmem_rope_ref,
                vmem_k_ref,
                sem_ref,
                *wide_scratch,
            ) = args

        if use_narrow_nr:
            vmem_nope_wide_ref, vmem_rope_wide_ref, sem_wide_ref = wide_scratch

        sublanes_nr = jax.lax.broadcasted_iota(jnp.int32, (tile_k, 128), 0)
        row_r = jax.lax.broadcasted_iota(jnp.int32, (rope_tile, 128), 0)
        q_iota = jax.lax.broadcasted_iota(jnp.int32, (rope_tile, 128), 1) // 32
        slot_rope = row_r * 4 + q_iota
        lanes_full = jax.lax.broadcasted_iota(jnp.int32, (padded_groups, tile_k), 1)
        lane_1d = jax.lax.broadcasted_iota(jnp.int32, (1, 128), 1)
        n_tiles = n_tiles_ref[0]

        def _tile_hbm_slices(idx):
            tid = jnp.maximum(tile_ids_ref[idx], 0)
            p_kv = tid // tiles_per_page
            p_rem = tid % tiles_per_page
            p_idx = idx_pages_ref[idx]
            k_off = p_rem * tile_k
            r_tile_off = p_rem * rope_tile
            l0_i = lanes0_ref[idx]
            sub_off = jnp.minimum((l0_i // 8) * 8, tile_k - nr_dma)
            r_sub_off = jnp.minimum((l0_i // 32) * 8, rope_tile - rope_dma)
            is_nar = (lanes_end_ref[idx] - l0_i) <= 9
            return (
                p_kv,
                p_idx,
                k_off,
                r_tile_off,
                k_off + sub_off,
                r_tile_off + r_sub_off,
                sub_off,
                r_sub_off,
                is_nar,
            )

        def _start_read_all3(idx, vbuf):
            sbuf = vbuf % half_ring
            (
                p_kv,
                p_idx,
                k_off,
                r_tile_off,
                nope_nar_off,
                rope_nar_off,
                _,
                _,
                is_nar,
            ) = _tile_hbm_slices(idx)

            @pl.when(is_full_ref[idx] == 0)
            def _rd():
                if use_narrow_nr:

                    @pl.when(is_nar)
                    def _rd_nar():
                        pltpu.make_async_copy(
                            nope_in_hbm_ref.at[p_kv, pl.ds(nope_nar_off, nr_dma), :],
                            vmem_nope_ref.at[vbuf],
                            sem_ref.at[0, sbuf],
                        ).start()
                        pltpu.make_async_copy(
                            rope_in_hbm_ref.at[p_kv, pl.ds(rope_nar_off, rope_dma), :],
                            vmem_rope_ref.at[vbuf],
                            sem_ref.at[1, sbuf],
                        ).start()

                    @pl.when(~is_nar)
                    def _rd_wide():
                        pltpu.make_async_copy(
                            nope_in_hbm_ref.at[p_kv, pl.ds(k_off, tile_k), :],
                            vmem_nope_wide_ref.at[vbuf],
                            sem_wide_ref.at[0, sbuf],
                        ).start()
                        pltpu.make_async_copy(
                            rope_in_hbm_ref.at[p_kv, pl.ds(r_tile_off, rope_tile), :],
                            vmem_rope_wide_ref.at[vbuf],
                            sem_wide_ref.at[1, sbuf],
                        ).start()
                else:
                    pltpu.make_async_copy(
                        nope_in_hbm_ref.at[p_kv, pl.ds(k_off, tile_k), :],
                        vmem_nope_ref.at[vbuf],
                        sem_ref.at[0, sbuf],
                    ).start()
                    pltpu.make_async_copy(
                        rope_in_hbm_ref.at[p_kv, pl.ds(r_tile_off, rope_tile), :],
                        vmem_rope_ref.at[vbuf],
                        sem_ref.at[1, sbuf],
                    ).start()
                pltpu.make_async_copy(
                    k_in_hbm_ref.at[p_idx, :, :, pl.ds(k_off, tile_k)],
                    vmem_k_ref.at[vbuf],
                    sem_ref.at[2, sbuf],
                ).start()

        def _wait_read_nr_nar(idx, vbuf):
            sbuf = vbuf % half_ring
            p_kv, _, _, _, nope_nar_off, rope_nar_off, _, _, _ = _tile_hbm_slices(idx)
            pltpu.make_async_copy(
                nope_in_hbm_ref.at[p_kv, pl.ds(nope_nar_off, nr_dma), :],
                vmem_nope_ref.at[vbuf],
                sem_ref.at[0, sbuf],
            ).wait()
            pltpu.make_async_copy(
                rope_in_hbm_ref.at[p_kv, pl.ds(rope_nar_off, rope_dma), :],
                vmem_rope_ref.at[vbuf],
                sem_ref.at[1, sbuf],
            ).wait()

        def _wait_read_nr_wide(idx, vbuf, n_buf_ref, r_buf_ref, s_ref):
            sbuf = vbuf % half_ring
            p_kv, _, k_off, r_tile_off, _, _, _, _, _ = _tile_hbm_slices(idx)
            pltpu.make_async_copy(
                nope_in_hbm_ref.at[p_kv, pl.ds(k_off, tile_k), :],
                n_buf_ref.at[vbuf],
                s_ref.at[0, sbuf],
            ).wait()
            pltpu.make_async_copy(
                rope_in_hbm_ref.at[p_kv, pl.ds(r_tile_off, rope_tile), :],
                r_buf_ref.at[vbuf],
                s_ref.at[1, sbuf],
            ).wait()

        def _wait_read_k(idx, vbuf):
            sbuf = vbuf % half_ring
            _, p_idx, k_off, _, _, _, _, _, _ = _tile_hbm_slices(idx)
            pltpu.make_async_copy(
                k_in_hbm_ref.at[p_idx, :, :, pl.ds(k_off, tile_k)],
                vmem_k_ref.at[vbuf],
                sem_ref.at[2, sbuf],
            ).wait()

        def _start_write_all3(idx, vbuf):
            sbuf = vbuf % half_ring
            (
                p_kv,
                p_idx,
                k_off,
                r_tile_off,
                nope_nar_off,
                rope_nar_off,
                _,
                _,
                is_nar,
            ) = _tile_hbm_slices(idx)
            if use_narrow_nr:

                @pl.when(is_nar)
                def _wr_nar():
                    pltpu.make_async_copy(
                        vmem_nope_ref.at[vbuf],
                        nope_out_hbm_ref.at[p_kv, pl.ds(nope_nar_off, nr_dma), :],
                        sem_ref.at[3, sbuf],
                    ).start()
                    pltpu.make_async_copy(
                        vmem_rope_ref.at[vbuf],
                        rope_out_hbm_ref.at[p_kv, pl.ds(rope_nar_off, rope_dma), :],
                        sem_ref.at[4, sbuf],
                    ).start()

                @pl.when(~is_nar)
                def _wr_wide():
                    pltpu.make_async_copy(
                        vmem_nope_wide_ref.at[vbuf],
                        nope_out_hbm_ref.at[p_kv, pl.ds(k_off, tile_k), :],
                        sem_wide_ref.at[2, sbuf],
                    ).start()
                    pltpu.make_async_copy(
                        vmem_rope_wide_ref.at[vbuf],
                        rope_out_hbm_ref.at[p_kv, pl.ds(r_tile_off, rope_tile), :],
                        sem_wide_ref.at[3, sbuf],
                    ).start()
            else:
                pltpu.make_async_copy(
                    vmem_nope_ref.at[vbuf],
                    nope_out_hbm_ref.at[p_kv, pl.ds(k_off, tile_k), :],
                    sem_ref.at[3, sbuf],
                ).start()
                pltpu.make_async_copy(
                    vmem_rope_ref.at[vbuf],
                    rope_out_hbm_ref.at[p_kv, pl.ds(r_tile_off, rope_tile), :],
                    sem_ref.at[4, sbuf],
                ).start()
            pltpu.make_async_copy(
                vmem_k_ref.at[vbuf],
                k_out_hbm_ref.at[p_idx, :, :, pl.ds(k_off, tile_k)],
                sem_ref.at[5, sbuf],
            ).start()

        def _wait_write_all3(idx, vbuf):
            sbuf = vbuf % half_ring
            (
                p_kv,
                p_idx,
                k_off,
                r_tile_off,
                nope_nar_off,
                rope_nar_off,
                _,
                _,
                is_nar,
            ) = _tile_hbm_slices(idx)
            if use_narrow_nr:

                @pl.when(is_nar)
                def _ww_nar():
                    pltpu.make_async_copy(
                        vmem_nope_ref.at[vbuf],
                        nope_out_hbm_ref.at[p_kv, pl.ds(nope_nar_off, nr_dma), :],
                        sem_ref.at[3, sbuf],
                    ).wait()
                    pltpu.make_async_copy(
                        vmem_rope_ref.at[vbuf],
                        rope_out_hbm_ref.at[p_kv, pl.ds(rope_nar_off, rope_dma), :],
                        sem_ref.at[4, sbuf],
                    ).wait()

                @pl.when(~is_nar)
                def _ww_wide():
                    pltpu.make_async_copy(
                        vmem_nope_wide_ref.at[vbuf],
                        nope_out_hbm_ref.at[p_kv, pl.ds(k_off, tile_k), :],
                        sem_wide_ref.at[2, sbuf],
                    ).wait()
                    pltpu.make_async_copy(
                        vmem_rope_wide_ref.at[vbuf],
                        rope_out_hbm_ref.at[p_kv, pl.ds(r_tile_off, rope_tile), :],
                        sem_wide_ref.at[3, sbuf],
                    ).wait()
            else:
                pltpu.make_async_copy(
                    vmem_nope_ref.at[vbuf],
                    nope_out_hbm_ref.at[p_kv, pl.ds(k_off, tile_k), :],
                    sem_ref.at[3, sbuf],
                ).wait()
                pltpu.make_async_copy(
                    vmem_rope_ref.at[vbuf],
                    rope_out_hbm_ref.at[p_kv, pl.ds(r_tile_off, rope_tile), :],
                    sem_ref.at[4, sbuf],
                ).wait()
            pltpu.make_async_copy(
                vmem_k_ref.at[vbuf],
                k_out_hbm_ref.at[p_idx, :, :, pl.ds(k_off, tile_k)],
                sem_ref.at[5, sbuf],
            ).wait()

        @pl.loop(0, jnp.minimum(n_tiles, half_ring))
        def _prologue(p_i):
            _start_read_all3(p_i, p_i)

        @pl.loop(0, n_tiles)
        def _step(i):
            cur_buf = i % ring_depth
            ahead_buf = (i + half_ring) % ring_depth

            @pl.when(i >= half_ring)
            def _retire_write():
                _wait_write_all3(i - half_ring, ahead_buf)
                if not deep_ring:
                    # 4-deep ring: ahead_buf == cur_buf; re-arm it for tile i.
                    _start_read_all3(i, cur_buf)

            tid = tile_ids_ref[i]
            t0 = safe_t0_ref[i]
            l0 = lanes0_ref[i]
            l_end = lanes_end_ref[i]
            full = is_full_ref[i]
            m_tile = t0 // tile_k
            m_off = t0 % tile_k
            shift_k = jnp.bitwise_and(l0 - m_off, tile_k - 1)

            def _write_wide_tile(n_buf_ref, r_buf_ref, s_ref):
                @pl.when((full == 1) & (m_off == 0))
                def _store_aligned_full():
                    n_buf_ref.at[cur_buf][...] = nope_tiles_vmem_ref[m_tile]
                    r_buf_ref.at[cur_buf][...] = rope_tiles_vmem_ref[m_tile]
                    kt0 = k_tiles_vmem_ref[m_tile]
                    vmem_k_ref.at[cur_buf][...] = pltpu.bitcast(
                        kt0[:sublane_groups, :], jnp.uint8
                    ).reshape(sublane_groups, pack, tile_k)

                @pl.when((full == 0) | (m_off != 0))
                def _store_general():
                    n0 = nope_tiles_vmem_ref[m_tile]
                    n1 = nope_tiles_vmem_ref[m_tile + 1]
                    comb_nope = jnp.where(sublanes_nr >= m_off, n0, n1)
                    upd_nope = pltpu.roll(comb_nope, shift_k, axis=0)

                    r0 = rope_tiles_vmem_ref[m_tile]
                    r1 = rope_tiles_vmem_ref[m_tile + 1]
                    comb_rope = jnp.where(slot_rope >= m_off, r0, r1)
                    d_r = shift_k >> 2
                    d_q = jnp.bitwise_and(shift_k, 3)
                    lane_sh = pltpu.roll(comb_rope, d_q * 32, axis=1)
                    sh0 = pltpu.roll(lane_sh, d_r, axis=0)
                    sh1 = pltpu.roll(
                        lane_sh, jnp.bitwise_and(d_r + 1, rope_tile - 1), axis=0
                    )
                    upd_rope = jnp.where(q_iota < d_q, sh1, sh0)

                    kt0 = k_tiles_vmem_ref[m_tile]
                    kt1 = k_tiles_vmem_ref[m_tile + 1]
                    comb_k = jnp.where(lanes_full >= m_off, kt0, kt1)
                    rolled_k = pltpu.roll(comb_k, shift_k, axis=1)
                    upd_k = pltpu.bitcast(
                        rolled_k[:sublane_groups, :], jnp.uint8
                    ).reshape(sublane_groups, pack, tile_k)

                    @pl.when(full == 1)
                    def _store_full():
                        n_buf_ref.at[cur_buf][...] = upd_nope
                        r_buf_ref.at[cur_buf][...] = upd_rope
                        vmem_k_ref.at[cur_buf][...] = upd_k

                    @pl.when(full == 0)
                    def _store_partial():
                        valid_nope = (
                            (tid >= 0) & (sublanes_nr >= l0) & (sublanes_nr < l_end)
                        )
                        valid_rope = (
                            (tid >= 0) & (slot_rope >= l0) & (slot_rope < l_end)
                        )
                        _wait_read_nr_wide(i, cur_buf, n_buf_ref, r_buf_ref, s_ref)
                        _wait_read_k(i, cur_buf)
                        n_buf_ref.at[cur_buf][...] = jnp.where(
                            valid_nope, upd_nope, n_buf_ref.at[cur_buf][...]
                        )
                        r_buf_ref.at[cur_buf][...] = jnp.where(
                            valid_rope, upd_rope, r_buf_ref.at[cur_buf][...]
                        )
                        if tile_k == 128:
                            valid_k = (tid >= 0) & (lane_1d >= l0) & (lane_1d < l_end)
                            vmem_k_ref.at[cur_buf][...] = jnp.where(
                                valid_k[:, None, :], upd_k, vmem_k_ref.at[cur_buf][...]
                            )
                        else:
                            valid0 = (tid >= 0) & (lane_1d >= l0) & (lane_1d < l_end)
                            lane_hi = lane_1d + 128
                            valid1 = (tid >= 0) & (lane_hi >= l0) & (lane_hi < l_end)
                            cur0 = vmem_k_ref.at[cur_buf, :, :, pl.ds(0, 128)][...]
                            cur1 = vmem_k_ref.at[cur_buf, :, :, pl.ds(128, 128)][...]
                            vmem_k_ref.at[cur_buf, :, :, pl.ds(0, 128)][...] = (
                                jnp.where(valid0[:, None, :], upd_k[:, :, :128], cur0)
                            )
                            vmem_k_ref.at[cur_buf, :, :, pl.ds(128, 128)][...] = (
                                jnp.where(valid1[:, None, :], upd_k[:, :, 128:], cur1)
                            )

            if use_narrow_nr:
                _, _, _, _, _, _, sub_off, r_sub_off, is_nar = _tile_hbm_slices(i)

                @pl.when(is_nar)
                def _store_nar_tile():
                    shift_nope = jnp.bitwise_and((l0 - sub_off) - m_off, tile_k - 1)
                    shift_rope = jnp.bitwise_and(
                        (l0 - r_sub_off * 4) - m_off, tile_k - 1
                    )

                    n0 = nope_tiles_vmem_ref[m_tile]
                    n1 = nope_tiles_vmem_ref[m_tile + 1]
                    comb_nope = jnp.where(sublanes_nr >= m_off, n0, n1)
                    upd_nope_sub = pltpu.roll(comb_nope, shift_nope, axis=0)[:nr_dma]

                    r0 = rope_tiles_vmem_ref[m_tile]
                    r1 = rope_tiles_vmem_ref[m_tile + 1]
                    comb_rope = jnp.where(slot_rope >= m_off, r0, r1)
                    d_r = shift_rope >> 2
                    d_q = jnp.bitwise_and(shift_rope, 3)
                    lane_sh = pltpu.roll(comb_rope, d_q * 32, axis=1)
                    sh0 = pltpu.roll(lane_sh, d_r, axis=0)
                    sh1 = pltpu.roll(
                        lane_sh, jnp.bitwise_and(d_r + 1, rope_tile - 1), axis=0
                    )
                    upd_rope_sub = jnp.where(q_iota < d_q, sh1, sh0)[:rope_dma]

                    kt0 = k_tiles_vmem_ref[m_tile]
                    kt1 = k_tiles_vmem_ref[m_tile + 1]
                    comb_k = jnp.where(lanes_full >= m_off, kt0, kt1)
                    rolled_k = pltpu.roll(comb_k, shift_k, axis=1)
                    upd_k = pltpu.bitcast(
                        rolled_k[:sublane_groups, :], jnp.uint8
                    ).reshape(sublane_groups, pack, tile_k)

                    _wait_read_nr_nar(i, cur_buf)
                    _wait_read_k(i, cur_buf)
                    sub_nr_narrow = sublanes_nr[:nr_dma] + sub_off
                    valid_nope = (
                        (tid >= 0) & (sub_nr_narrow >= l0) & (sub_nr_narrow < l_end)
                    )
                    slot_rope_narrow = slot_rope[:rope_dma] + (r_sub_off * 4)
                    valid_rope = (
                        (tid >= 0)
                        & (slot_rope_narrow >= l0)
                        & (slot_rope_narrow < l_end)
                    )
                    valid_k = (tid >= 0) & (lane_1d >= l0) & (lane_1d < l_end)
                    vmem_nope_ref.at[cur_buf][...] = jnp.where(
                        valid_nope, upd_nope_sub, vmem_nope_ref.at[cur_buf][...]
                    )
                    vmem_rope_ref.at[cur_buf][...] = jnp.where(
                        valid_rope, upd_rope_sub, vmem_rope_ref.at[cur_buf][...]
                    )
                    vmem_k_ref.at[cur_buf][...] = jnp.where(
                        valid_k[:, None, :], upd_k, vmem_k_ref.at[cur_buf][...]
                    )

                @pl.when(~is_nar)
                def _store_wide_fallback():
                    _write_wide_tile(
                        vmem_nope_wide_ref, vmem_rope_wide_ref, sem_wide_ref
                    )
            else:
                _write_wide_tile(vmem_nope_ref, vmem_rope_ref, sem_ref)

            if deep_ring:

                @pl.when(i + half_ring < n_tiles)
                def _prefetch_read():
                    _start_read_all3(i + half_ring, ahead_buf)

            _start_write_all3(i, cur_buf)

        @pl.loop(jnp.maximum(0, n_tiles - half_ring), n_tiles)
        def _epilogue(e_i):
            _wait_write_all3(e_i, e_i % ring_depth)

        if state_in_nope:

            @pl.loop(0, st_should_ref.shape[0])
            def _wait_all_state_writes_nope(s_i):
                @pl.when(st_should_ref[s_i] == 1)
                def _wt_st():
                    slot = st_slots_ref[s_i]
                    p_st = slot // states_per_page_dma
                    r_st = (slot % states_per_page_dma) * 8
                    pltpu.make_async_copy(
                        st_words_vmem_ref.at[s_i],
                        nope_out_hbm_ref.at[p_st, pl.ds(r_st, 8), :],
                        st_sem_ref.at[0],
                    ).wait()
        elif has_state_cache:

            @pl.when(n_st > 0)
            def _wait_all_state_writes():
                pltpu.make_async_copy(
                    st_words_vmem_ref.at[pl.ds(0, n_st)],
                    st_out_hbm_ref.at[pl.ds(0, n_st)],
                    st_sem_ref.at[0],
                ).wait()

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    n_buf_rows = nr_dma if use_narrow_nr else tile_k
    r_buf_rows = rope_dma if use_narrow_nr else rope_tile
    scratch_list = [
        pltpu.VMEM((ring_depth, n_buf_rows, 128), jnp.int32),
        pltpu.VMEM((ring_depth, r_buf_rows, 128), jnp.int32),
        pltpu.VMEM((ring_depth, sublane_groups, pack, tile_k), jnp.uint8),
        pltpu.SemaphoreType.DMA((6, half_ring)),
    ]
    if has_state_cache:
        scratch_list.append(pltpu.SemaphoreType.DMA((1,)))
    if use_narrow_nr:
        scratch_list.extend(
            [
                pltpu.VMEM((ring_depth, tile_k, 128), jnp.int32),
                pltpu.VMEM((ring_depth, rope_tile, 128), jnp.int32),
                pltpu.SemaphoreType.DMA((4, half_ring)),
            ]
        )

    if state_in_nope:
        return pl.pallas_call(
            _single_cache_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=9,
                grid=(1,),
                in_specs=[
                    vmem_spec,
                    vmem_spec,
                    vmem_spec,
                    vmem_spec,
                    hbm_spec,
                    hbm_spec,
                    hbm_spec,
                ],
                out_specs=[hbm_spec, hbm_spec, hbm_spec],
                scratch_shapes=scratch_list,
            ),
            out_shape=(
                jax.ShapeDtypeStruct(nope_cache.shape, nope_cache.dtype),
                jax.ShapeDtypeStruct(rope_cache.shape, rope_cache.dtype),
                jax.ShapeDtypeStruct(k_cache.shape, k_cache.dtype),
            ),
            input_output_aliases={13: 0, 14: 1, 15: 2},
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=100 * 1024 * 1024,
                disable_bounds_checks=True,
            ),
            name=f"fused_single_pallas_3cache_r2_aliased_{compress_ratio}",
        )(
            tile_ids,
            idx_pages,
            safe_t0,
            lanes0,
            lanes_end,
            is_full,
            st_write_slots,
            st_should_write,
            jnp.asarray([n_active_tiles, n_st_writes], dtype=jnp.int32),
            st_wkv_words,
            nope_tiles,
            rope_tiles,
            k_tiles,
            nope_cache,
            rope_cache,
            k_cache,
        )

    if has_state_cache:
        assert (
            state_slots_in is not None
            and st_write_slots is not None
            and st_should_write is not None
            and st_wkv_words is not None
            and n_st_writes is not None
        )
        return pl.pallas_call(
            _single_cache_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=9,
                grid=(1,),
                in_specs=[
                    vmem_spec,
                    vmem_spec,
                    vmem_spec,
                    vmem_spec,
                    hbm_spec,
                    hbm_spec,
                    hbm_spec,
                    hbm_spec,
                ],
                out_specs=[hbm_spec, hbm_spec, hbm_spec, hbm_spec],
                scratch_shapes=scratch_list,
            ),
            out_shape=(
                jax.ShapeDtypeStruct(state_slots_in.shape, state_slots_in.dtype),
                jax.ShapeDtypeStruct(nope_cache.shape, nope_cache.dtype),
                jax.ShapeDtypeStruct(rope_cache.shape, rope_cache.dtype),
                jax.ShapeDtypeStruct(k_cache.shape, k_cache.dtype),
            ),
            input_output_aliases={13: 0, 14: 1, 15: 2, 16: 3},
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=100 * 1024 * 1024,
                disable_bounds_checks=True,
            ),
            name=f"fused_single_pallas_4cache_r{compress_ratio}",
        )(
            tile_ids,
            idx_pages,
            safe_t0,
            lanes0,
            lanes_end,
            is_full,
            st_write_slots,
            st_should_write,
            jnp.asarray([n_active_tiles, n_st_writes], dtype=jnp.int32),
            st_wkv_words,
            nope_tiles,
            rope_tiles,
            k_tiles,
            state_slots_in,
            nope_cache,
            rope_cache,
            k_cache,
        )

    return pl.pallas_call(
        _single_cache_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=7,
            grid=(1,),
            in_specs=[
                vmem_spec,
                vmem_spec,
                vmem_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
            ],
            out_specs=[hbm_spec, hbm_spec, hbm_spec],
            scratch_shapes=scratch_list,
        ),
        out_shape=(
            jax.ShapeDtypeStruct(nope_cache.shape, nope_cache.dtype),
            jax.ShapeDtypeStruct(rope_cache.shape, rope_cache.dtype),
            jax.ShapeDtypeStruct(k_cache.shape, k_cache.dtype),
        ),
        input_output_aliases={10: 0, 11: 1, 12: 2},
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=100 * 1024 * 1024,
            disable_bounds_checks=True,
        ),
        name=f"fused_single_pallas_3cache_r{compress_ratio}",
    )(
        tile_ids,
        idx_pages,
        safe_t0,
        lanes0,
        lanes_end,
        is_full,
        jnp.asarray([n_active_tiles], dtype=jnp.int32),
        nope_tiles,
        rope_tiles,
        k_tiles,
        nope_cache,
        rope_cache,
        k_cache,
    )


def fused_insert_produce_k_jax(
    latent: jax.Array,  # [num_tokens, 512] bf16
    wk_weight: jax.Array,  # [128, 512] bf16/fp8
    knorm_weight: jax.Array,  # [128] bf16
    cos_sin_cache: jax.Array,  # [max_pos, 64] f32
    idx_cos_sin_cache: jax.Array,  # [max_pos, 64] f32
    positions: jax.Array,  # [num_tokens] i32
    kv_block_table: jax.Array,  # [max_num_seqs * max_kv_blocks] i32
    idx_block_table: jax.Array,  # [max_num_seqs * max_idx_blocks] i32
    query_start_loc: jax.Array,  # [max_num_seqs + 1] i32
    distribution: jax.Array,  # [3] i32
    nope_cache: jax.Array,  # donated (arg 10)
    rope_cache: jax.Array,  # donated (arg 11)
    k_cache: jax.Array,  # donated (arg 12)
    wk_weight_scale: jax.Array | None = None,
    *,
    compress_ratio: int,
    idx_rms_eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single custom op with single concurrent 3-cache Pallas async DMA kernel."""
    return _fused_quant_pack_and_single_pallas_scatter(
        latent,
        wk_weight,
        knorm_weight,
        cos_sin_cache,
        idx_cos_sin_cache,
        positions,
        kv_block_table,
        idx_block_table,
        query_start_loc,
        distribution,
        nope_cache,
        rope_cache,
        k_cache,
        wk_weight_scale,
        compress_ratio=compress_ratio,
        idx_rms_eps=idx_rms_eps,
    )


def fused_compress_r2_insert_produce_k_jax(
    wkv_wgate: jax.Array,  # [num_tokens, 1024] f32
    norm_weight: jax.Array,  # [512] bf16
    wk_weight: jax.Array,  # [128, 512] bf16/fp8
    knorm_weight: jax.Array,  # [128] bf16
    cos_sin_cache: jax.Array,  # [max_pos, 64] f32
    idx_cos_sin_cache: jax.Array,  # [max_pos, 64] f32
    positions: jax.Array,  # [num_tokens] i32
    state_block_table: jax.Array,  # [max_num_seqs * max_state_blocks] i32
    kv_block_table: jax.Array,  # [max_num_seqs * max_kv_blocks] i32
    idx_block_table: jax.Array,  # [max_num_seqs * max_idx_blocks] i32
    query_start_loc: jax.Array,  # [max_num_seqs + 1] i32
    distribution: jax.Array,  # [3] i32
    state_cache: jax.Array,  # donated (arg 12)
    nope_cache: jax.Array,  # donated (arg 13)
    rope_cache: jax.Array,  # donated (arg 14)
    k_cache: jax.Array,  # donated (arg 15)
    wk_weight_scale: jax.Array | None = None,
    *,
    state_block_size: int,
    compress_ratio: int,
    comp_rms_eps: float = 1e-6,
    idx_rms_eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Single custom op with single concurrent 4-cache Pallas async DMA kernel."""
    head_dim = norm_weight.shape[0]
    num_tokens = wkv_wgate.shape[0]
    max_num_seqs = query_start_loc.shape[0] - 1
    eff_max_seqs = min(max_num_seqs, max(8, align_to(num_tokens, 8)))

    max_state_blocks = state_block_table.shape[0] // max_num_seqs
    st_src = nope_cache if state_cache is None else state_cache
    num_st_pages, st_page_size = st_src.shape[:2]
    states_per_page = st_page_size // 8
    total_state_slots = num_st_pages * states_per_page
    state_slots_in = st_src.reshape((total_state_slots, 8, 128))

    pos = positions.astype(jnp.int32)
    seq_ids = jnp.arange(eff_max_seqs, dtype=jnp.int32)
    q_start = query_start_loc[:eff_max_seqs]
    q_end = query_start_loc[1 : eff_max_seqs + 1]
    seq_valid = (seq_ids < distribution[2]) & (q_end > q_start)

    def _state_slot_for(r: jax.Array, p: jax.Array) -> jax.Array:
        page_idx = jnp.clip(p // state_block_size, 0, max_state_blocks - 1)
        page = state_block_table[r * max_state_blocks + page_idx]
        if num_st_pages > 1:
            page = jnp.where(page > 0, ((page - 1) % (num_st_pages - 1)) + 1, 0)
        return jnp.clip(
            page * states_per_page + (p % state_block_size), 0, total_state_slots - 1
        )

    first_tok_idx = jnp.clip(q_start, 0, num_tokens - 1)
    first_pos = pos[first_tok_idx]
    seq_read_slots = _state_slot_for(seq_ids, jnp.maximum(first_pos - 1, 0))
    seq_cache_prev = jax.lax.bitcast_convert_type(
        state_slots_in[seq_read_slots], jnp.float32
    ).reshape(eff_max_seqs, 1024)

    token_idx = jnp.arange(num_tokens, dtype=jnp.int32)

    def _single_seq_prev():
        return jnp.concatenate([seq_cache_prev[:1], wkv_wgate[:-1]], axis=0)

    def _multi_seq_prev():
        if num_tokens >= 384:
            batch_prev = jnp.concatenate([wkv_wgate[:1], wkv_wgate[:-1]], axis=0)
            first_write_idx = jnp.where(seq_valid, first_tok_idx, num_tokens)
            return batch_prev.at[first_write_idx].set(seq_cache_prev, mode="drop")
        batch_prev = wkv_wgate[jnp.clip(token_idx - 1, 0, num_tokens - 1)]
        r_m = _seq_ids_from_locs(query_start_loc, num_tokens, eff_max_seqs)
        in_b = token_idx > query_start_loc[r_m]
        return jnp.where(in_b[:, None], batch_prev, seq_cache_prev[r_m])

    prev_kv_score = jax.lax.cond(
        distribution[2] <= 1,
        _single_seq_prev,
        _multi_seq_prev,
    )

    if _prefill_role():
        # Last token per sequence only (see compress_r2_pool_norm_jax).
        last_tok_idx = jnp.clip(q_end - 1, 0, num_tokens - 1)
        last_pos = pos[last_tok_idx]
        st_should_write = (seq_valid & ((last_pos & 1) == 0)).astype(jnp.int32)
        n_st_writes = jnp.sum(st_should_write).astype(jnp.int32)
        st_write_slots = _state_slot_for(seq_ids, last_pos)
        st_wkv_words = jax.lax.bitcast_convert_type(
            wkv_wgate[last_tok_idx].reshape(eff_max_seqs, 8, 128), jnp.int32
        )
    else:
        # Every even-position token (decode stack; DSpark rollback-safe).
        req_all = _seq_ids_from_locs(query_start_loc, num_tokens, eff_max_seqs)
        valid_all = token_idx < query_start_loc[distribution[2]]
        st_should_write = (valid_all & ((pos & 1) == 0)).astype(jnp.int32)
        n_st_writes = jnp.sum(st_should_write).astype(jnp.int32)
        st_write_slots = _state_slot_for(req_all, pos)
        st_wkv_words = jax.lax.bitcast_convert_type(
            wkv_wgate.reshape(num_tokens, 8, 128), jnp.int32
        )

    kv0, score0 = prev_kv_score[:, :head_dim], prev_kv_score[:, head_dim:]
    kv1, score1 = wkv_wgate[:, :head_dim], wkv_wgate[:, head_dim:]
    max_s = jnp.maximum(score0, score1)
    e0, e1 = jnp.exp(score0 - max_s), jnp.exp(score1 - max_s)
    pf32 = ((e0 * kv0 + e1 * kv1) / (e0 + e1)).astype(jnp.bfloat16).astype(jnp.float32)
    rstd = jax.lax.rsqrt(jnp.mean(pf32 * pf32, axis=-1, keepdims=True) + comp_rms_eps)
    latent = jax.lax.optimization_barrier(
        (pf32 * rstd).astype(jnp.bfloat16) * norm_weight.astype(jnp.bfloat16)
    )

    scatter_out = _fused_quant_pack_and_single_pallas_scatter(
        latent,
        wk_weight,
        knorm_weight,
        cos_sin_cache,
        idx_cos_sin_cache,
        positions,
        kv_block_table,
        idx_block_table,
        query_start_loc,
        distribution,
        nope_cache,
        rope_cache,
        k_cache,
        wk_weight_scale,
        compress_ratio=compress_ratio,
        idx_rms_eps=idx_rms_eps,
        state_slots_in=(None if state_cache is None else state_slots_in),
        st_write_slots=st_write_slots,
        st_should_write=st_should_write,
        st_wkv_words=st_wkv_words,
        n_st_writes=n_st_writes,
    )
    if state_cache is None:
        return scatter_out  # type: ignore[return-value]
    new_st_slots, new_nope, new_rope, new_k = scatter_out
    return new_st_slots.reshape(state_cache.shape), new_nope, new_rope, new_k


def _fused_indexer_wq_b_rope_quant_decode_jax(
    query_fp8: jax.Array,  # [num_tokens, q_lora_rank] f8
    query_scale: jax.Array,  # [num_tokens, 1] f32
    wq_b_weight: jax.Array,  # [q_lora_rank, n_head * head_dim] f8
    wq_b_weight_scale: jax.Array,  # [n_head * head_dim] f32
    positions: jax.Array,  # [num_tokens] i32
    cos_sin_cache: jax.Array,  # [max_pos, 64] f32
    indexer_weights: jax.Array,  # [num_tokens, n_head] bf16/f32
    *,
    softmax_scale: float,
    n_head: int,
    head_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """Decode-stack variant of fused_indexer_wq_b_rope_quant_jax (unpadded,
    largest_divisor tiling, [T, 1] query scale), plus the signed head-weight fix:
    q is scaled by |w_h| and the output scale carries sign(w_h).
    """
    num_tokens, k_dim = query_fp8.shape
    out_features = n_head * head_dim
    tile_n = largest_divisor(num_tokens, cap=128)
    num_tiles = num_tokens // tile_n
    cs_expanded = _expand_cos_sin(positions, cos_sin_cache)
    w_scale_all = (
        indexer_weights.astype(jnp.float32) * softmax_scale * (n_head**-0.5)
    )

    def _pallas_fused_kernel(
        q_fp8_hbm_ref,
        q_s_hbm_ref,
        cs_hbm_ref,
        w_s_hbm_ref,
        wqb_w_vmem_ref,
        wqb_s_vmem_ref,
        q_out_hbm_ref,
        s_out_hbm_ref,
    ):
        lane = jnp.arange(head_dim)
        sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)
        w_mat = wqb_w_vmem_ref[...]
        ws_vec = wqb_s_vmem_ref[...][None, :]

        def _pipe_step(q_ref, qs_ref, cs_ref, ws_ref, q_out_ref, s_out_ref):
            proj = jnp.einsum(
                "tk,kn->tn",
                q_ref[...],
                w_mat,
                preferred_element_type=jnp.float32,
            )
            q_bf16 = (
                (proj * qs_ref[...] * ws_vec)
                .astype(jnp.bfloat16)
                .reshape(tile_n, n_head, head_dim)
            )
            ws = ws_ref[...].astype(jnp.float32)
            x = (
                (q_bf16.astype(jnp.float32) * jnp.abs(ws)[..., None])
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
            cs = cs_ref[...]
            cos = cs[:, None, :head_dim]
            sin = cs[:, None, head_dim:] * sign[None, None, :]
            y = x * cos + _lane_gather(x, lane ^ 1) * sin
            abs_max = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
            scale = abs_max / jnp.float32(448.0)
            scale_inv = jnp.where(abs_max == 0.0, 0.0, 1.0 / scale)
            q_out_ref[...] = (y * scale_inv).astype(jnp.float8_e4m3fn)
            s_out_ref[...] = scale.reshape(tile_n, n_head) * jnp.sign(ws)

        q_in_spec = pl.BlockSpec(
            (tile_n, k_dim), lambda i: (i, 0), memory_space=pltpu.VMEM
        )
        qs_in_spec = pl.BlockSpec(
            (tile_n, 1), lambda i: (i, 0), memory_space=pltpu.VMEM
        )
        cs_in_spec = pl.BlockSpec(
            (tile_n, 2 * head_dim), lambda i: (i, 0), memory_space=pltpu.VMEM
        )
        ws_in_spec = pl.BlockSpec(
            (tile_n, n_head), lambda i: (i, 0), memory_space=pltpu.VMEM
        )
        q_out_spec = pl.BlockSpec(
            (tile_n, n_head, head_dim), lambda i: (i, 0, 0), memory_space=pltpu.VMEM
        )
        s_out_spec = pl.BlockSpec(
            (tile_n, n_head), lambda i: (i, 0), memory_space=pltpu.VMEM
        )

        pltpu.emit_pipeline(
            _pipe_step,
            grid=(num_tiles,),
            in_specs=[q_in_spec, qs_in_spec, cs_in_spec, ws_in_spec],
            out_specs=[q_out_spec, s_out_spec],
        )(
            q_fp8_hbm_ref,
            q_s_hbm_ref,
            cs_hbm_ref,
            w_s_hbm_ref,
            q_out_hbm_ref,
            s_out_hbm_ref,
        )

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    return pl.pallas_call(
        _pallas_fused_kernel,
        out_shape=(
            jax.ShapeDtypeStruct(
                (num_tokens, n_head, head_dim), jnp.float8_e4m3fn
            ),
            jax.ShapeDtypeStruct((num_tokens, n_head), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(1,),
            in_specs=[
                hbm_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                vmem_spec,
                vmem_spec,
            ],
            out_specs=[hbm_spec, hbm_spec],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=100 * 1024 * 1024,
            disable_bounds_checks=True,
        ),
        name="fused_indexer_wq_b_rope_quant",
    )(
        query_fp8,
        query_scale,
        cs_expanded,
        w_scale_all,
        wq_b_weight,
        wq_b_weight_scale.reshape(1, out_features),
    )


def fused_indexer_wq_b_rope_quant_jax(
    query_fp8: jax.Array,  # [num_tokens, q_lora_rank] f8
    query_scale: jax.Array,  # [num_tokens, 1] f32
    wq_b_weight: jax.Array,  # [q_lora_rank, n_head * head_dim] f8
    wq_b_weight_scale: jax.Array,  # [n_head * head_dim] f32
    positions: jax.Array,  # [num_tokens] i32
    cos_sin_cache: jax.Array,  # [max_pos, 64] f32
    indexer_weights: jax.Array,  # [num_tokens, n_head] bf16/f32
    *,
    softmax_scale: float,
    n_head: int,
    head_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """Single pipelined Pallas kernel: FP8 wq_b GEMM + w_scale + RoPE + FP8 quant in VMEM."""
    if not _prefill_role():
        return _fused_indexer_wq_b_rope_quant_decode_jax(
            query_fp8,
            query_scale,
            wq_b_weight,
            wq_b_weight_scale,
            positions,
            cos_sin_cache,
            indexer_weights,
            softmax_scale=softmax_scale,
            n_head=n_head,
            head_dim=head_dim,
        )
    orig_num_tokens, k_dim = query_fp8.shape
    pad_align = 8 if orig_num_tokens <= 128 else 128
    pad_t = align_to(orig_num_tokens, pad_align) - orig_num_tokens
    if pad_t > 0:
        query_fp8 = jnp.pad(query_fp8, ((0, pad_t), (0, 0)))
        query_scale = jnp.pad(query_scale, ((0, pad_t), (0, 0)))
        positions = jnp.pad(positions, ((0, pad_t),))
        indexer_weights = jnp.pad(indexer_weights, ((0, pad_t), (0, 0)))
    num_tokens = orig_num_tokens + pad_t
    out_features = n_head * head_dim
    tile_n = num_tokens if num_tokens <= 128 else 128
    num_tiles = num_tokens // tile_n
    n_head_aligned = align_to(n_head, 128)
    cs_expanded = _expand_cos_sin(positions, cos_sin_cache)
    w_scale_all = indexer_weights.astype(jnp.float32) * softmax_scale * (n_head**-0.5)
    if n_head_aligned > n_head:
        w_scale_all = jnp.pad(w_scale_all, ((0, 0), (0, n_head_aligned - n_head)))
    q_scale_all = jnp.broadcast_to(
        query_scale.astype(jnp.float32), (num_tokens, head_dim)
    )

    def _pallas_fused_kernel(
        q_fp8_hbm_ref,
        cs_hbm_ref,
        w_s_hbm_ref,
        q_s_hbm_ref,
        wqb_w_vmem_ref,
        wqb_s_vmem_ref,
        q_out_hbm_ref,
        s_out_hbm_ref,
    ):
        lane = jnp.arange(head_dim)
        sign = jnp.where(lane % 2 == 0, -1.0, 1.0).astype(jnp.float32)
        w_mat = wqb_w_vmem_ref[...]
        ws_vec = wqb_s_vmem_ref[...][None, :]

        def _pipe_step(q_ref, cs_ref, ws_ref, qs_ref, q_out_ref, s_out_ref):
            proj = jnp.einsum(
                "tk,kn->tn",
                q_ref[...],
                w_mat,
                preferred_element_type=jnp.float32,
            )
            proj_scaled = (proj * ws_vec).reshape(tile_n, n_head, head_dim)
            q_bf16 = (proj_scaled * qs_ref[...][:, None, :]).astype(jnp.bfloat16)
            ws_unpad = ws_ref[...][:, :n_head].astype(jnp.float32)
            x = (
                (q_bf16.astype(jnp.float32) * jnp.abs(ws_unpad)[..., None])
                .astype(jnp.bfloat16)
                .astype(jnp.float32)
            )
            cs = cs_ref[...]
            cos = cs[:, None, :head_dim]
            sin = cs[:, None, head_dim:] * sign[None, None, :]
            y = x * cos + _lane_gather(x, lane ^ 1) * sin
            abs_max = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
            scale = abs_max / jnp.float32(448.0)
            scale_inv = jnp.where(abs_max == 0.0, 0.0, 1.0 / scale)
            q_out_ref[...] = (y * scale_inv).astype(jnp.float8_e4m3fn)
            s_unpad = scale.reshape(tile_n, n_head) * jnp.sign(ws_unpad)
            if n_head_aligned > n_head:
                s_out_ref[...] = jnp.pad(
                    s_unpad, ((0, 0), (0, n_head_aligned - n_head))
                )
            else:
                s_out_ref[...] = s_unpad

        q_in_spec = pl.BlockSpec(
            (tile_n, k_dim), lambda i: (i, 0), memory_space=pltpu.VMEM
        )
        cs_in_spec = pl.BlockSpec(
            (tile_n, 2 * head_dim), lambda i: (i, 0), memory_space=pltpu.VMEM
        )
        ws_in_spec = pl.BlockSpec(
            (tile_n, n_head_aligned), lambda i: (i, 0), memory_space=pltpu.VMEM
        )
        qs_in_spec = pl.BlockSpec(
            (tile_n, head_dim), lambda i: (i, 0), memory_space=pltpu.VMEM
        )
        q_out_spec = pl.BlockSpec(
            (tile_n, n_head, head_dim), lambda i: (i, 0, 0), memory_space=pltpu.VMEM
        )
        s_out_spec = pl.BlockSpec(
            (tile_n, n_head_aligned), lambda i: (i, 0), memory_space=pltpu.VMEM
        )

        pltpu.emit_pipeline(
            _pipe_step,
            grid=(num_tiles,),
            in_specs=[q_in_spec, cs_in_spec, ws_in_spec, qs_in_spec],
            out_specs=[q_out_spec, s_out_spec],
        )(
            q_fp8_hbm_ref,
            cs_hbm_ref,
            w_s_hbm_ref,
            q_s_hbm_ref,
            q_out_hbm_ref,
            s_out_hbm_ref,
        )

    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    q_out, s_out_aligned = pl.pallas_call(
        _pallas_fused_kernel,
        out_shape=(
            jax.ShapeDtypeStruct((num_tokens, n_head, head_dim), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((num_tokens, n_head_aligned), jnp.float32),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(1,),
            in_specs=[
                hbm_spec,
                hbm_spec,
                hbm_spec,
                hbm_spec,
                vmem_spec,
                vmem_spec,
            ],
            out_specs=[hbm_spec, hbm_spec],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=100 * 1024 * 1024,
            disable_bounds_checks=True,
        ),
        name="fused_indexer_wq_b_rope_quant",
    )(
        query_fp8,
        cs_expanded,
        w_scale_all,
        q_scale_all,
        wq_b_weight,
        wq_b_weight_scale.reshape(1, out_features),
    )
    if pad_t > 0:
        return q_out[:orig_num_tokens], s_out_aligned[:orig_num_tokens, :n_head]
    return q_out, s_out_aligned[:, :n_head]


def _indexer_v41_impl(
    q: jax.Array,  # [num_tokens, n_head, 128] bf16
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    main_page_indices: jax.Array,
    candidates: jax.Array | None = None,
    precomputed_eff_row_lengths: jax.Array | None = None,
    precomputed_idx_page_indices: jax.Array | None = None,
    precomputed_pass_num_steps: jax.Array | None = None,
    precomputed_pass_schedules: jax.Array | None = None,
    *,
    k: int,
    compression_ratio: int,
    softmax_scale: float,
    n_head: int,
    cand_mode: str | None = None,
    cand_blocks: int = 0,
    cand_block_size: int = 0,
    has_precomputed_meta: bool = False,
    q_is_prequant: bool = False,
    cfg_num_tokens: int | None = None,
    use_candidates_override: jax.Array | None = None,
    token_start: jax.Array | int = 0,
) -> jax.Array:
    """Indexer top-k over the full compressed KV, encoded as CSA gather slots.

    cfg_num_tokens: token count that selects the query/KV block sizes (defaults
    to q.shape[0]); the token-parallel path passes the full batch size so a
    prefill slice keeps the prefill blocks. Chunking follows q.shape[0].
    use_candidates_override: batch-wide candidate predicate (token-parallel path).
    """
    if has_precomputed_meta and cand_mode != "read":
        # Without a candidates input the 4 metadata arrays shift left by one slot.
        (
            precomputed_eff_row_lengths,
            precomputed_idx_page_indices,
            precomputed_pass_num_steps,
            precomputed_pass_schedules,
        ) = (
            candidates,
            precomputed_eff_row_lengths,
            precomputed_idx_page_indices,
            precomputed_pass_num_steps,
        )
        candidates = None

    if q_is_prequant:
        q_quant = q
        weights = indexer_weights
    else:
        w_scale = indexer_weights.astype(jnp.float32) * softmax_scale * (n_head**-0.5)
        q_quant, q_scales = rope_quant(
            q, positions, cos_sin_cache, w_scale=w_scale, quant_dtype=jnp.float8_e4m3fn
        )
        weights = q_scales
    num_pages, _, _, page_size = cache_kv.shape
    block_stride = max(1, (page_size * compression_ratio) // 1024)
    if precomputed_idx_page_indices is not None:
        idx_page_indices = precomputed_idx_page_indices
    else:
        idx_page_indices = (
            page_indices.reshape(seq_lens.shape[0], -1)[:, ::block_stride].reshape(-1)
            if block_stride > 1
            else page_indices.flatten()
        )
    n_cfg = q.shape[0] if cfg_num_tokens is None else cfg_num_tokens
    # DSpark decode/verify batches (<= 512 tokens) run in MIXED; bq=8 avoids 16x
    # query padding and fits 2x larger KV blocks. 96/384 buckets use multiples of 48.
    is_decode_or_verify = n_cfg <= 1536
    bq_mixed = 8 if is_decode_or_verify else 128
    # Decode-sized KV block: 64 pages (prefill TP=2) vs 16 (decode TP=1 fits 64MB VMEM at cr=2).
    bkv_dec = 64 if _prefill_role() else 16
    bkv_pages = (bkv_dec if is_decode_or_verify else 8) * max(1, compression_ratio)
    chunk_tokens = (
        512
        if (not is_decode_or_verify and q.shape[0] >= 512 and q.shape[0] % 512 == 0)
        else {
            1536: 512,
            1280: 640,
            1024: 512,
            768: 384,
            640: 320,
            512: 512,
            384: 384,
            320: 320,
            256: 256,
            192: 192,
            160: 160,
            128: 128,
            96: 96,
            64: 64,
            16: 16,
        }.get(q.shape[0], 512)
    )
    dense = functools.partial(
        streamindex_topk,
        q=q_quant,
        indexer_weights=weights,
        cache_kv=cache_kv,
        seq_lens=seq_lens,
        page_indices=idx_page_indices,
        cu_q_lens=cu_q_lens,
        distribution=distribution,
        precomputed_eff_row_lengths=precomputed_eff_row_lengths,
        precomputed_pass_num_steps=precomputed_pass_num_steps,
        precomputed_pass_schedules=precomputed_pass_schedules,
        k=k,
        compression_ratio=compression_ratio,
        num_kv_pages_per_block=bkv_pages,
        num_queries_per_block=bq_mixed,
        chunk_tokens=chunk_tokens,
    )
    cands = None
    if cand_mode is None:
        topk_indices = dense()
    else:
        live = jnp.arange(seq_lens.shape[0]) < distribution[2]
        max_visible = jnp.max(jnp.where(live, seq_lens, 0)) // compression_ratio
        use_candidates = (
            max_visible > cand_blocks * cand_block_size
            if use_candidates_override is None
            else use_candidates_override
        )
        if cand_mode == "write":
            topk_indices, cands = jax.lax.cond(
                use_candidates,
                lambda: dense(candidate_topk_blocks=cand_blocks),
                lambda: (
                    dense(),
                    jnp.full((q.shape[0], cand_blocks), -1, jnp.int32),
                ),
            )
        else:
            assert candidates is not None
            topk_indices = jax.lax.cond(
                use_candidates,
                lambda: dense(candidates=candidates),
                dense,
            )
    encoded = encode_csa_gather_indices(
        topk_indices,
        main_page_indices.flatten(),
        cu_q_lens,
        num_pages,
        page_size // block_stride,
        token_start=token_start,
    )
    return (encoded, cands) if cands is not None else encoded  # type: ignore[return-value]


def indexer_v41_jax(
    q: jax.Array,  # [num_tokens, n_head, 128] bf16
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    main_page_indices: jax.Array,
    candidates: jax.Array | None = None,
    precomputed_eff_row_lengths: jax.Array | None = None,
    precomputed_idx_page_indices: jax.Array | None = None,
    precomputed_pass_num_steps: jax.Array | None = None,
    precomputed_pass_schedules: jax.Array | None = None,
    *,
    k: int,
    compression_ratio: int,
    softmax_scale: float,
    n_head: int,
    cand_mode: str | None = None,
    cand_blocks: int = 0,
    cand_block_size: int = 0,
    has_precomputed_meta: bool = False,
    q_is_prequant: bool = False,
) -> jax.Array:
    """torch.library.custom_op-safe wrapper around _indexer_v41_impl."""
    return _indexer_v41_impl(
        q,
        positions,
        cos_sin_cache,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        main_page_indices,
        candidates,
        precomputed_eff_row_lengths,
        precomputed_idx_page_indices,
        precomputed_pass_num_steps,
        precomputed_pass_schedules,
        k=k,
        compression_ratio=compression_ratio,
        softmax_scale=softmax_scale,
        n_head=n_head,
        cand_mode=cand_mode,
        cand_blocks=cand_blocks,
        cand_block_size=cand_block_size,
        has_precomputed_meta=has_precomputed_meta,
        q_is_prequant=q_is_prequant,
    )


def indexer_v41_write_jax(*args, **kwargs) -> tuple[jax.Array, jax.Array]:
    return indexer_v41_jax(*args, **kwargs)  # type: ignore[return-value]


indexer_v41_write_jax.__signature__ = inspect.signature(indexer_v41_jax).replace(  # type: ignore[attr-defined]
    return_annotation=tuple[jax.Array, jax.Array]
)


def tp_local_indexer_batch(
    rank: jax.Array,  # i32[1]
    seq_lens: jax.Array,  # i32[max_num_seqs]
    page_indices: jax.Array,  # i32[max_num_seqs, pages_per_seq] (or flat)
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    main_page_indices: jax.Array,  # i32[max_num_seqs, main_pages_per_seq] (or flat)
    *,
    num_local_tokens: int,
    num_local_rows: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Tokens [rank * T_l, (rank + 1) * T_l) of an indexer batch as a standalone batch.

    Every live request has >= 1 token, so at most T_l requests overlap the slice;
    they are contiguous in request order and keep it. A request cut at the slice
    end gets a shorter seq_len, so that token_pos = seq_len - q_len + offset (the
    kernel's position rule) is unchanged for every local token.

    num_local_rows (>= T_l, default T_l) sizes the local batch; the extra request
    slots are empty (seq_len 0, q_len 0), like the tail of a padded decode bucket.
    """
    t_l = num_local_tokens
    lo = rank[0] * t_l
    hi = lo + t_l
    max_num_seqs = seq_lens.shape[0]
    # A slice of t_l tokens spans at most min(t_l, max_num_seqs) requests.
    t_w = min(t_l if num_local_rows is None else num_local_rows, max_num_seqs)
    live = jnp.arange(max_num_seqs, dtype=jnp.int32) < distribution[2]
    q_start, q_end = cu_q_lens[:-1], cu_q_lens[1:]
    req_lo = jnp.sum(live & (q_end <= lo)).astype(jnp.int32)
    req_hi = jnp.sum(live & (q_start < hi)).astype(jnp.int32)

    def window(x: jax.Array, size: int, mode: str = "constant") -> jax.Array:
        pad = [(0, t_w)] + [(0, 0)] * (x.ndim - 1)
        return jax.lax.dynamic_slice_in_dim(jnp.pad(x, pad, mode=mode), req_lo, size)

    sl = window(seq_lens, t_w)
    cq = window(cu_q_lens, t_w + 1, mode="edge")
    pi = window(page_indices.reshape(max_num_seqs, -1), t_w)
    mpi = window(main_page_indices.reshape(max_num_seqs, -1), t_w)
    q_end_l = cq[1:]
    active = jnp.arange(t_w, dtype=jnp.int32) < req_hi - req_lo
    sl_local = jnp.where(active, sl - (q_end_l - jnp.minimum(q_end_l, hi)), 0)
    cu_local = jnp.clip(cq - lo, 0, t_l)
    dist_local = jnp.clip(distribution, req_lo, req_hi) - req_lo
    return (
        sl_local.astype(seq_lens.dtype),
        pi.reshape(-1),
        cu_local.astype(cu_q_lens.dtype),
        dist_local.astype(distribution.dtype),
        mpi.reshape(-1),
    )


def indexer_v41_tp_jax(
    rank: jax.Array,  # i32[1], this TP rank
    q: jax.Array,  # [T / tp, n_head, 128] bf16, this rank's tokens
    positions: jax.Array,  # [T / tp]
    cos_sin_cache: jax.Array,
    indexer_weights: jax.Array,  # [T / tp, n_head]
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    main_page_indices: jax.Array,
    candidates: jax.Array | None = None,  # [T, cand_blocks], all tokens
    *,
    tp_size: int,
    k: int,
    compression_ratio: int,
    softmax_scale: float,
    n_head: int,
    cand_mode: str | None = None,
    cand_blocks: int = 0,
    cand_block_size: int = 0,
    q_is_prequant: bool = False,
) -> jax.Array:
    """Token-parallel indexer: top-k for this rank's T/tp tokens only.

    The caller all-gathers the [T/tp, k] result. The K cache is replicated, so
    each token's top-k is the same as in the replicated path.
    """
    t_l = q.shape[0]
    # Decode configs run 8-query blocks (bq_mixed). A rank with fewer tokens
    # (T/tp = 2 or 4) runs as an 8-row local batch whose tail rows belong to no
    # request, like a padded decode bucket, and returns only its T/tp rows.
    t_rows = max(8, -(-t_l // 8) * 8)
    sl, pi, cq, dist, mpi = tp_local_indexer_batch(
        rank,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        main_page_indices,
        num_local_tokens=t_l,
        num_local_rows=t_rows,
    )
    use_candidates = None
    if cand_mode is not None:
        # Same predicate as the replicated path, over the whole batch.
        live = jnp.arange(seq_lens.shape[0]) < distribution[2]
        max_visible = jnp.max(jnp.where(live, seq_lens, 0)) // compression_ratio
        use_candidates = max_visible > cand_blocks * cand_block_size
        if cand_mode == "read":
            assert candidates is not None
            if candidates.shape[0] != t_l:
                candidates = jax.lax.dynamic_slice_in_dim(
                    candidates, rank[0] * t_l, t_l
                )
    if t_rows != t_l:

        def pad_rows(x: jax.Array, **kw) -> jax.Array:
            return jnp.pad(x, [(0, t_rows - t_l)] + [(0, 0)] * (x.ndim - 1), **kw)

        q = pad_rows(q, mode="edge")
        positions = pad_rows(positions, mode="edge")
        indexer_weights = pad_rows(indexer_weights, mode="edge")
        if candidates is not None:
            candidates = pad_rows(candidates, constant_values=-1)
    out = _indexer_v41_impl(
        q,
        positions,
        cos_sin_cache,
        indexer_weights,
        cache_kv,
        sl,
        pi,
        cq,
        dist,
        mpi,
        candidates,
        k=k,
        compression_ratio=compression_ratio,
        softmax_scale=softmax_scale,
        n_head=n_head,
        cand_mode=cand_mode,
        cand_blocks=cand_blocks,
        cand_block_size=cand_block_size,
        q_is_prequant=q_is_prequant,
        cfg_num_tokens=t_l * tp_size,
        use_candidates_override=use_candidates,
        token_start=rank[0] * t_l,
    )
    if t_rows == t_l:
        return out
    if isinstance(out, tuple):
        return tuple(x[:t_l] for x in out)  # type: ignore[return-value]
    return out[:t_l]


def indexer_v41_tp_write_jax(*args, **kwargs) -> tuple[jax.Array, jax.Array]:
    return indexer_v41_tp_jax(*args, **kwargs)  # type: ignore[return-value]


indexer_v41_tp_write_jax.__signature__ = inspect.signature(  # type: ignore[attr-defined]
    indexer_v41_tp_jax
).replace(return_annotation=tuple[jax.Array, jax.Array])
