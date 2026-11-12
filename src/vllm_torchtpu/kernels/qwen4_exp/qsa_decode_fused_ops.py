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
"""Fused QSA Decode pipeline (zero cross-layer `while`, zero `lax.cond`):
- `qsa_fused_pre_indexer_all`: Fuses `k` norm+RoPE + FP8 `pack_rows` + `main_pool`
  1-row DMAs + `iq` norm+RoPE + `ring_pool` RMW + `nik` norm+RoPE + `index_pool` RMW
  (supporting pure decode, short prefills, mixed ragged batches, and M-RoPE tags).
- `select_blocks_direct_decode`: Per-query `StreamIdxTC` with 1-VREG schedule
  (`build_fast_decode_streamindex_metadata`), bank-conflict-free idle page rows,
  5D `page_group=13` DMA coalescing, and `OverlappedWQ0SparseCoreTopK`.
- `qsa_prepare_token_indices_packed_meta`: Tile-interleaved token indices with
  per-row 256-token SparseCore inactive-chunk markers and `[lengths, tail_phys_tok]`.
- `qsa_prepare_packed_q_and_gate`: Fuses `(qgate_0, qgate_1)` + `q` norm+RoPE +
  combined GQA packing inside the SparseCore gather scheduling group.
- `compact_pooled_qsa_fast_decode_step`: End-to-end fast QSA step with
  `(w_q_1 + q_pack) || qsa_sc_gather_2level_tc_tiled` and grouped QSA attention.
- `compact_pooled_qsa_projected_unified_step`: Unified stable-ABI entry routing
  small-bucket batches (`<= 32`) directly to `compact_pooled_qsa_fast_decode_step`
  without `jax.lax.cond`, and larger prefill buckets to `compact_pooled_qsa_step`.
"""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels import typed_ldst
from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
    BITS_NEG_INF,
    LANES,
    _align_to,
    _cdiv,
    _pick_partition,
    _topk_body_for_shape,
    _topk_scratch_types,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    config as topk_cfg,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    streamindex_topk,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.metadata import MetadataRef
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import (
    KVLayout,
    _score_call_for_config,
    align_to,
    prepare_index_weights,
    prepare_q_inputs,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    SparseCoreTopK,
    TopKBackend,
)
from vllm_torchtpu.kernels.qwen4_exp.cache_regions import region_rows
from vllm_torchtpu.kernels.qwen4_exp.fast_grouped_qsa_decode import (
    _floor_divmod_const,
    qsa_fast_grouped_qsa_projected_attention,
    qsa_sc_gather_2level_tc_tiled,
)
from vllm_torchtpu.kernels.qwen4_exp.indexer_components import qsa_indexing
from vllm_torchtpu.kernels.qwen4_exp.qsa import _token_metadata, gemma_norm
from vllm_torchtpu.kernels.qwen4_exp.resident_topk import qwen_sparsecore_topk
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16
from vllm_torchtpu.kernels.qwen4_exp.tuned_block_sizes import (
    _hardware,
    get_decode_index_key_packing,
    get_decode_index_page_group,
    get_decode_index_pages,
)


def _compressed_rows_for_ratio(ratio: int) -> int:
    if ratio < 1 or ratio & (ratio - 1) or 128 % ratio or (128 // ratio) % 2:
        raise ValueError("QSA ratio must divide 128 into an even number of BF16 groups")
    return 128 // ratio


def _select_rope_rows_tc_safe(
    positions: jax.Array,
    cache: jax.Array,
    rotary_dim: int,
    rope_axes: tuple[int, ...] | None = None,
) -> jax.Array:
    """Select RoPE rows via 1D gathers + VPU selects to avoid 3D SC offload bug."""
    if positions.ndim == 1:
        return lax.optimization_barrier(cache[positions])
    if positions.ndim != 2 or positions.shape[0] != 3 or rope_axes is None:
        raise ValueError(
            "Axial positions require three axes and a frequency assignment"
        )
    half = rotary_dim // 2
    if len(rope_axes) != half:
        raise ValueError("MRoPE frequency assignment differs from rotary dimension")
    axes = tuple(
        rope_axes[col % half] if col < rotary_dim else 0
        for col in range(cache.shape[-1])
    )
    axes_arr = jnp.asarray(axes, dtype=jnp.int32)[None, :]
    r0 = lax.optimization_barrier(cache[positions[0]])
    r1 = lax.optimization_barrier(cache[positions[1]])
    r2 = lax.optimization_barrier(cache[positions[2]])
    out = jnp.where(axes_arr == 1, r1, r0)
    return jnp.where(axes_arr == 2, r2, out)


class StepQsaMetadata(NamedTuple):
    lengths: jax.Array
    both_angles: jax.Array
    meta_all: jax.Array
    req: jax.Array


def build_step_qsa_metadata(
    *,
    batch: int,
    block_size: int,
    num_main_pages: int,
    index_pool_shape: tuple[int, ...],
    index_pool_dtype: jnp.dtype,
    positions: jax.Array,
    trig: jax.Array,
    starts: jax.Array,
    context: jax.Array,
    ring_ids: jax.Array,
    main_table: jax.Array,
    compressed_table: jax.Array,
    rotary_dim: int = 64,
    compress_ratio: int = 4,
    ring_capacity: int | None = None,
    rope_axes: tuple[int, ...] | None = None,
    use_tags_trig_dma: bool = False,
) -> StepQsaMetadata:
    """Build per-token QSA decode metadata for arbitrary active slots and RoPE."""
    cap = compress_ratio if ring_capacity is None else ring_capacity
    ring_ids_1d = ring_ids.reshape(len(context), -1)[:, 0]
    active = starts[1:] > starts[:-1]
    if cap > compress_ratio:
        active = active & (ring_ids_1d > 0)
    slots = jnp.where(active, jnp.arange(len(ring_ids_1d), dtype=jnp.int32), -1)
    req, logical, valid = _token_metadata(batch, starts, context, slots)
    ring_slot = jnp.where(valid, ring_ids_1d[req], -1)
    valid = valid & ((ring_slot > 0) if cap > compress_ratio else (ring_slot >= 0))
    valid_i32 = valid.astype(jnp.int32)
    lengths = jnp.where(valid, logical + 1, 0).astype(jnp.int32)

    safe_logical = jnp.maximum(logical, 0)
    if use_tags_trig_dma:
        both_angles = _select_rope_rows_tc_safe(positions, trig, rotary_dim, rope_axes)
    else:
        rem_in_group = safe_logical % compress_ratio
        if positions.ndim == 2:
            first_pos = jnp.maximum(positions - rem_in_group[None, :], 0)
            both_pos = jnp.concatenate([positions, first_pos], axis=1)
        else:
            first_pos = jnp.maximum(positions - rem_in_group, 0)
            both_pos = jnp.concatenate([positions, first_pos], axis=0)
        both_angles = _select_rope_rows_tc_safe(both_pos, trig, rotary_dim, rope_axes)

    log_blk = safe_logical // block_size
    main_off = safe_logical % block_size
    req_page = main_table[req, log_blk]
    main_page = jnp.where(valid, req_page, num_main_pages)

    idx_packing = 4 if index_pool_dtype == jnp.float8_e4m3fn else 2
    pairs_per_page = math.prod(index_pool_shape[1:-1]) // idx_packing
    comp_page = (
        req_page if main_table is compressed_table else compressed_table[req, log_blk]
    )
    pair_idx = jnp.maximum(
        comp_page * pairs_per_page + main_off // (2 * compress_ratio), 0
    )
    complete_bool = valid & ((logical % compress_ratio) == (compress_ratio - 1))
    pair_idx_or_neg1 = jnp.where(complete_bool, pair_idx, -1).astype(jnp.int32)

    req_lens = starts[1:] - starts[:-1]
    all_single = jnp.broadcast_to(jnp.all(req_lens <= 1).astype(jnp.int32), (batch,))
    meta_all = jnp.stack(
        [
            main_page,
            main_off,
            valid_i32,
            ring_slot,
            logical,
            pair_idx_or_neg1,
            all_single,
        ],
        axis=0,
    )
    return StepQsaMetadata(
        lengths=lengths,
        both_angles=both_angles,
        meta_all=meta_all,
        req=req,
    )


def _prepare_qsa_token_indices_packed_meta_kernel(
    table_ref,
    blocks_ref,
    lengths_ref,
    tok_out_ref,
    meta_out_ref,
    *,
    groups_per_page: int,
    pages_per_seq: int,
    ratio: int,
    sc_chunk_tokens: int = 512,
):
    table = table_ref[...]
    blocks = blocks_ref[...]
    lengths = lengths_ref[:, :1]  # (8, 1)

    complete = lax.shift_right_arithmetic(jnp.maximum(lengths, 0), 2)
    iota_blk = lax.broadcasted_iota(jnp.int32, blocks.shape, 1)
    valid_blk = (blocks >= 0) & (blocks < complete) & (iota_blk < complete)
    safe_blk = jnp.where(valid_blk, blocks, iota_blk)
    max_blk = jnp.where(complete > 0, complete - 1, blocks.shape[1] - 1)
    blk = jnp.clip(safe_blk, 0, max_blk)
    log_page, page_off = _floor_divmod_const(blk, groups_per_page)
    log_page = jnp.clip(log_page, 0, pages_per_seq - 1)
    phys_page = jnp.take_along_axis(table, log_page, axis=1)
    phys_blk_tok = (phys_page * groups_per_page + page_off) * ratio  # (8, 512)

    if sc_chunk_tokens == 256:
        # Fine-grained 256-token SparseCore chunks align 1:1 with each `(tile, query)`
        # row, so inactive rows are marked `-1` directly from `complete`.
        active_complete = complete
    else:
        # Pair-max `complete` across rows `(2*p, 2*p+1)` for 512-token SC chunks.
        complete_256 = jnp.broadcast_to(complete, (blocks.shape[0], 256))
        row_pair = lax.shift_right_arithmetic(
            lax.broadcasted_iota(jnp.int32, (blocks.shape[0], 256), 0), 1
        )
        pair_complete = jnp.zeros_like(complete_256)
        for p in range(blocks.shape[0] // 2):
            p_max = jnp.max(
                jnp.where(row_pair == p, complete_256, 0), axis=0, keepdims=True
            )
            pair_complete = jnp.where(
                row_pair == p,
                jnp.broadcast_to(p_max, complete_256.shape),
                pair_complete,
            )
        active_complete = pair_complete

    iota_512 = lax.broadcasted_iota(jnp.int32, (blocks.shape[0], 128 * ratio), 1)
    local_blk = lax.shift_right_arithmetic(iota_512, 2)
    rem_tok = iota_512 & jnp.int32(ratio - 1)
    tiles_list = []
    for c in range(blocks.shape[1] // 128):
        chunk = phys_blk_tok[:, c * 128 : (c + 1) * 128]
        exp_c = jnp.take_along_axis(chunk, local_blk, axis=1) + rem_tok
        t0 = jnp.where(active_complete > (2 * c) * 64, exp_c[:, :256], -1)
        t1 = jnp.where(active_complete > (2 * c + 1) * 64, exp_c[:, 256:], -1)
        tiles_list.append(t0)
        tiles_list.append(t1)
    tok_out_ref[...] = jnp.concatenate(tiles_list, axis=0)  # (64, 256)

    tail_log_page, tail_page_off = _floor_divmod_const(complete, groups_per_page)
    tail_log_page = jnp.clip(tail_log_page, 0, pages_per_seq - 1)
    tail_phys_page = jnp.take_along_axis(table, tail_log_page, axis=1)
    tail_phys_tok = (tail_phys_page * groups_per_page + tail_page_off) * ratio

    col_iota = lax.broadcasted_iota(jnp.int32, meta_out_ref.shape, 1)
    meta_out_ref[...] = jnp.where(col_iota == 0, lengths, tail_phys_tok)


def qsa_prepare_token_indices_packed_meta(
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    *,
    page_size: int = 1664,
    ratio: int = 4,
    sc_chunk_tokens: int = 512,
) -> tuple[jax.Array, jax.Array]:
    """Return `(tok_indices_1d, meta_2d)` with packed `[lengths, tail_phys_tok]`."""
    if sc_chunk_tokens not in (256, 512):
        raise ValueError(
            f"Unsupported sc_chunk_tokens={sc_chunk_tokens}; expected 256 or 512"
        )
    batch, budget = blocks.shape
    pages_per_seq = table.shape[1]
    groups_per_page = page_size // ratio
    block_tokens = 8
    table_cols = max(128, ((pages_per_seq + 127) // 128) * 128)
    padded_table = (
        table
        if pages_per_seq == table_cols
        else jnp.pad(table, ((0, 0), (0, table_cols - pages_per_seq)))
    )
    lengths_2d = jnp.broadcast_to(lengths[:, None], (batch, 128))
    tok_indices, meta_2d = pl.pallas_call(
        functools.partial(
            _prepare_qsa_token_indices_packed_meta_kernel,
            groups_per_page=groups_per_page,
            pages_per_seq=pages_per_seq,
            ratio=ratio,
            sc_chunk_tokens=sc_chunk_tokens,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((block_tokens, table_cols), lambda b: (b, 0)),
                pl.BlockSpec((block_tokens, budget), lambda b: (b, 0)),
                pl.BlockSpec((block_tokens, 128), lambda b: (b, 0)),
            ],
            out_specs=[
                pl.BlockSpec((64, 256), lambda b: (b, 0)),
                pl.BlockSpec((block_tokens, 128), lambda b: (b, 0)),
            ],
            grid=(batch // block_tokens,),
        ),
        out_shape=[
            jax.ShapeDtypeStruct(((batch // 8) * 64, 256), jnp.int32),
            jax.ShapeDtypeStruct((batch, 128), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        name=f"qsa_prepare_indices_packed_meta_c{sc_chunk_tokens}",
    )(padded_table, blocks, lengths_2d)
    return tok_indices.reshape(-1), meta_2d


def pack_pre_indexer_norm_weights(
    k_norm_multiplier: jax.Array,  # f32[256]
    niq: jax.Array,  # bf16[128]
    nik: jax.Array,  # bf16[128]
) -> jax.Array:
    """Pack static pre-indexer norm weights into a 128-lane `f32[512]` vector."""
    return jnp.concatenate(
        [
            k_norm_multiplier.astype(jnp.float32),
            niq.astype(jnp.float32),
            nik.astype(jnp.float32),
        ],
        axis=0,
    )


def qsa_fused_pre_indexer_all(
    main_pool: jax.Array,  # f8[num_pages, page_size, 1, 4, 256]
    ring_pool: jax.Array,  # bf16[num_ring_slots, 4 or 8, 1, 2, 128]
    index_pool: jax.Array,  # bf16[num_pages, 208, 1, 2, 128]
    kv_iqk: jax.Array,  # bf16[batch, 1664]
    niq: jax.Array | None,  # bf16[128]
    nik: jax.Array | None,  # bf16[128]
    k_norm_multiplier: jax.Array | None,  # f32[256]
    both_angles: jax.Array,  # bf16[2 * batch, 128]
    meta_all: jax.Array,  # i32[6 or 7, batch]
    *,
    norm_weights_packed: jax.Array | None = None,  # f32[512]
    positions: jax.Array | None = None,
    cache_tags: bool = False,
    trig_cache: jax.Array | None = None,
    rope_axes: tuple[int, ...] | None = None,
    kv_heads: int = 2,
    head_dim: int = 256,
    index_heads: int = 4,
    index_dim: int = 128,
    ratio: int = 4,
    ring_capacity: int | None = None,
    rotary_dim: int = 64,
    norm_eps: float = 1e-6,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fuse `main_pool` write, `iq` norm+RoPE, `ring_pool` RMW, and `index_pool` RMW."""
    del kv_heads
    batch = kv_iqk.shape[0]
    half = rotary_dim // 2
    width = index_dim
    cap = ratio if ring_capacity is None else ring_capacity
    if cap not in (ratio, 2 * ratio):
        raise ValueError(f"Unsupported ring_capacity={cap} for ratio={ratio}")
    ring_rows = (cap + 7) // 8 * 8
    use_tags = bool(cache_tags) and (positions is not None)
    axial = positions.ndim == 2 if positions is not None else False
    axes = (positions.shape[0] if axial else 1) if positions is not None else 1
    tag_fields = (axes + 1) if use_tags else 0
    angle_rows = 8
    use_tags_trig_dma = (
        use_tags
        and (trig_cache is not None)
        and (trig_cache.shape[0] >= angle_rows)
        and (trig_cache.shape[0] % angle_rows == 0)
    )
    trig_rows_total = trig_cache.shape[0] if trig_cache is not None else 0
    if use_tags_trig_dma:
        assert trig_cache is not None
        trig_cache = pltpu.with_memory_space_constraint(trig_cache, pltpu.HBM)
    ring_tokens, _ = region_rows(ring_pool, ring_rows * width, jnp.bfloat16)
    tag_tokens, tag_rows = (
        region_rows(ring_pool, ring_rows * tag_fields, jnp.int32)
        if use_tags
        else (0, 0)
    )
    total_ring_tokens = ring_tokens + tag_tokens
    packing = 4 if index_pool.dtype == jnp.float8_e4m3fn else 2
    index_view = index_pool.reshape(-1, packing, width)
    use_packed_norms = norm_weights_packed is not None
    has_all_single_flag = meta_all.shape[0] >= 7
    axis_word_masks: tuple[tuple[int, ...], ...] = ()
    if axes > 1 and rope_axes is not None:
        n_words = (half + 31) // 32
        masks_per_axis = []
        for a_idx in range(axes):
            words_for_axis = []
            for w in range(n_words):
                lo = w * 32
                hi = min(lo + 32, half)
                u = 0
                for bit in range(hi - lo):
                    if rope_axes[lo + bit] == a_idx:
                        u |= 1 << bit
                words_for_axis.append(u - (1 << 32) if u >= (1 << 31) else u)
            masks_per_axis.append(tuple(words_for_axis))
        axis_word_masks = tuple(masks_per_axis)

    def _to_f8_u32(x_bf16: jax.Array, scale: float) -> jax.Array:
        f8 = jnp.clip(
            x_bf16.astype(jnp.float32) / jnp.float32(scale), -448.0, 448.0
        ).astype(jnp.float8_e4m3fn)
        return pltpu.bitcast(f8, jnp.uint8).astype(jnp.uint32)

    def _norm_rope_k(
        k_head_2d: jax.Array, kmul: jax.Array, c_2d: jax.Array, s_2d: jax.Array
    ) -> jax.Array:
        kf = k_head_2d.astype(jnp.float32)
        kn = bf16(
            kf * lax.rsqrt(jnp.mean(kf * kf, axis=-1, keepdims=True) + norm_eps) * kmul
        )
        a = kn[:, :half].astype(jnp.float32)
        b = kn[:, half:rotary_dim].astype(jnp.float32)
        return jnp.concatenate(
            [bf16(a * c_2d - b * s_2d), bf16(b * c_2d + a * s_2d), kn[:, rotary_dim:]],
            axis=-1,
        )

    def kernel(*all_refs):
        idx = 0
        meta_ref = all_refs[idx]
        idx += 1
        pos_ref = None
        if use_tags:
            pos_ref = all_refs[idx]
            idx += 1
        kv_iqk_ref = all_refs[idx]
        idx += 1
        if use_packed_norms:
            norm_packed_ref = all_refs[idx]
            idx += 1
            niq_ref = nik_ref = kmul_ref = None
        else:
            niq_ref = all_refs[idx]
            nik_ref = all_refs[idx + 1]
            kmul_ref = all_refs[idx + 2]
            idx += 3
            norm_packed_ref = None
        both_ang_ref = all_refs[idx]
        idx += 1
        if use_tags_trig_dma:
            trig_ref = all_refs[idx]
            idx += 1
        else:
            trig_ref = None
        _old_main_ref = all_refs[idx]
        _ring_src = all_refs[idx + 1]
        _idx_src = all_refs[idx + 2]
        out_main_ref = all_refs[idx + 3]
        ring_dst = all_refs[idx + 4]
        idx_dst = all_refs[idx + 5]
        iq_out_ref = all_refs[idx + 6]
        packed_vmem_ref = all_refs[idx + 7]
        main_sem = all_refs[idx + 8]
        staged_ring = all_refs[idx + 9]
        ring_sem = all_refs[idx + 10]
        staged_pair = all_refs[idx + 11]
        pair_sem = all_refs[idx + 12]
        idx += 13
        if use_tags_trig_dma:
            staged_trig = all_refs[idx]
            trig_sem = all_refs[idx + 1]
        else:
            staged_trig = trig_sem = None
        del _old_main_ref, _ring_src, _idx_src

        is_parallel = (meta_ref[6, 0] != 0) if has_all_single_flag else jnp.bool_(True)

        # 1. Start async HBM reads for `ring_pool` and `index_pool` in parallel decode.
        @pl.when(is_parallel)
        def _prefetch_parallel():
            for r in range(batch):

                @pl.when(meta_ref[2, r] != 0)
                def _start_ring_read(*, r=r):
                    pltpu.make_async_copy(
                        ring_dst.at[
                            jnp.maximum(meta_ref[3, r], 0),
                            pl.ds(0, total_ring_tokens),
                        ],
                        staged_ring.at[r],
                        ring_sem.at[r],
                    ).start()

                @pl.when(meta_ref[5, r] >= 0)
                def _start_pair_read(*, r=r):
                    pltpu.make_async_copy(
                        idx_dst.at[meta_ref[5, r]],
                        staged_pair.at[r],
                        pair_sem.at[r],
                    ).start()

        # 2. Compute `k` norm+RoPE + FP8 packing while HBM reads are in flight
        kv_all = kv_iqk_ref[...]
        if use_packed_norms:
            assert norm_packed_ref is not None
            nw_all = norm_packed_ref[...]
            kmul_1d = nw_all[:head_dim]
            niq_vec = nw_all[head_dim : head_dim + index_dim].astype(jnp.bfloat16)
            nik_vec = nw_all[head_dim + index_dim : head_dim + 2 * index_dim].astype(
                jnp.bfloat16
            )
        else:
            assert kmul_ref is not None and niq_ref is not None and nik_ref is not None
            kmul_1d = kmul_ref[...]
            niq_vec = niq_ref[...]
            nik_vec = nik_ref[...]
        kmul = jnp.broadcast_to(kmul_1d[None, :], (batch, head_dim))
        both_ang_f32 = both_ang_ref[...].astype(jnp.float32)
        ang_f32 = both_ang_f32[:batch]
        fang_all = None if use_tags_trig_dma else both_ang_f32[batch:]
        c_2d = ang_f32[:, :half]
        s_2d = ang_f32[:, half:rotary_dim]

        k0 = _norm_rope_k(kv_all[:, :256], kmul, c_2d, s_2d)
        k1 = _norm_rope_k(kv_all[:, 256:512], kmul, c_2d, s_2d)
        v0 = kv_all[:, 512:768]
        v1 = kv_all[:, 768:1024]

        word = (
            _to_f8_u32(k0, k_scale)
            | (_to_f8_u32(v0, v_scale) << 8)
            | (_to_f8_u32(k1, k_scale) << 16)
            | (_to_f8_u32(v1, v_scale) << 24)
        )
        packed_vmem_ref.bitcast(jnp.uint32).reshape(batch, head_dim)[...] = word

        for r in range(batch):

            @pl.when(meta_ref[2, r] != 0)
            def _start_main_kv(*, r=r):
                pltpu.make_async_copy(
                    packed_vmem_ref.at[pl.ds(r, 1)],
                    out_main_ref.at[meta_ref[0, r], pl.ds(meta_ref[1, r], 1)],
                    main_sem.at[r],
                ).start()

        # 3. Compute `iq` norm+RoPE while `main_pool` and `ring/pair` DMAs run
        x_iq = kv_all[:, 1024 : 1024 + index_heads * index_dim].reshape(
            batch, index_heads, index_dim
        )
        x_iq = gemma_norm(x_iq, niq_vec, norm_eps)
        a_iq = x_iq[..., :half].astype(jnp.float32)
        b_iq = x_iq[..., half:rotary_dim].astype(jnp.float32)
        c_3d = c_2d[:, None, :]
        s_3d = s_2d[:, None, :]
        iq_out_ref[...] = jnp.concatenate(
            [
                bf16(a_iq * c_3d - bf16(b_iq * s_3d)),
                bf16(a_iq * s_3d + bf16(b_iq * c_3d)),
                x_iq[..., rotary_dim:],
            ],
            axis=-1,
        )

        # 4. Complete `ring_pool` (+ `tags`) RMW and `index_pool` compressed-pair RMW
        raw_bits = jax.lax.bitcast_convert_type(
            kv_all[:, 1536:1664], jnp.uint16
        ).astype(jnp.int32)
        row_iota = jnp.arange(batch, dtype=jnp.int32)[:, None]

        def _process_row_rmw(r: int, *, wait_writes_inline: bool):
            if wait_writes_inline:
                pltpu.make_async_copy(
                    ring_dst.at[
                        jnp.maximum(meta_ref[3, r], 0),
                        pl.ds(0, total_ring_tokens),
                    ],
                    staged_ring.at[r],
                    ring_sem.at[r],
                ).start()

                @pl.when(meta_ref[5, r] >= 0)
                def _seq_pair_read():
                    pltpu.make_async_copy(
                        idx_dst.at[meta_ref[5, r]],
                        staged_pair.at[r],
                        pair_sem.at[r],
                    ).start()

            pltpu.make_async_copy(
                ring_dst.at[
                    jnp.maximum(meta_ref[3, r], 0),
                    pl.ds(0, total_ring_tokens),
                ],
                staged_ring.at[r],
                ring_sem.at[r],
            ).wait()
            ring_view_ref = staged_ring.at[r, pl.ds(0, ring_tokens)]
            ring = typed_ldst.load_typed(ring_view_ref, view_dtype=jnp.bfloat16)
            old_ctx = meta_ref[4, r]
            ring = jnp.where(old_ctx > 0, ring, 0)
            slot = old_ctx % cap

            raw_tok = jax.lax.bitcast_convert_type(
                jnp.max(
                    jnp.where(row_iota == r, raw_bits, 0), axis=0, keepdims=True
                ).astype(jnp.uint16),
                jnp.bfloat16,
            )
            ring = jnp.where(jnp.arange(ring_rows)[:, None] == slot, raw_tok, ring)
            typed_ldst.store_typed(ring_view_ref, ring)

            if use_tags:
                assert pos_ref is not None
                tag_view_ref = staged_ring.at[r, pl.ds(ring_tokens, tag_tokens)]
                tags = typed_ldst.load_typed(tag_view_ref, view_dtype=jnp.int32)
                words = jnp.arange(tag_rows * width, dtype=jnp.int32).reshape(
                    tag_rows, width
                )
                tags = jnp.where(
                    words < ring_rows * tag_fields,
                    jnp.where(old_ctx > 0, tags, -1),
                    0,
                )
                for axis in range(axes):
                    coord = pos_ref[axis, r] if axial else pos_ref[r]
                    tags = jnp.where(words == tag_fields * slot + axis, coord, tags)
                tags = jnp.where(words == tag_fields * slot + axes, 0, tags)
                typed_ldst.store_typed(tag_view_ref, tags)

            @pl.when(meta_ref[3, r] > 0)
            def _write_ring():
                copy_op = pltpu.make_async_copy(
                    staged_ring.at[r],
                    ring_dst.at[meta_ref[3, r], pl.ds(0, total_ring_tokens)],
                    ring_sem.at[r],
                )
                copy_op.start()
                if wait_writes_inline:
                    copy_op.wait()

            @pl.when(meta_ref[5, r] >= 0)
            def _write_completed_pair():
                first_slot = ((old_ctx // ratio) * ratio) % cap
                coords_list: list[jax.Array] = []
                bases_list: list[jax.Array] = []
                if use_tags_trig_dma:
                    assert (
                        trig_ref is not None
                        and staged_trig is not None
                        and trig_sem is not None
                    )
                    for a_idx in range(axes):
                        raw_c = jnp.max(
                            jnp.where(
                                words == tag_fields * first_slot + a_idx, tags, -1
                            )
                        )
                        c_clamped = jnp.clip(
                            jnp.maximum(raw_c, 0), 0, trig_rows_total - 1
                        )
                        base_c = (c_clamped // angle_rows) * angle_rows
                        coords_list.append(c_clamped)
                        bases_list.append(base_c)
                        pltpu.make_async_copy(
                            trig_ref.at[pl.ds(base_c, angle_rows)],
                            staged_trig.at[a_idx],
                            trig_sem.at[a_idx],
                        ).start()

                ring_group = (
                    jnp.where(
                        first_slot != 0,
                        ring[ratio : 2 * ratio],
                        ring[:ratio],
                    )
                    if cap == 2 * ratio
                    else ring[:ratio]
                )
                pooled = bf16(
                    jnp.mean(ring_group.astype(jnp.float32), axis=0, keepdims=True)
                )
                normed = gemma_norm(pooled, nik_vec, norm_eps)
                if use_tags_trig_dma:
                    assert (
                        trig_ref is not None
                        and staged_trig is not None
                        and trig_sem is not None
                    )
                    ang_per_axis = []
                    for a_idx in range(axes):
                        pltpu.make_async_copy(
                            trig_ref.at[pl.ds(bases_list[a_idx], angle_rows)],
                            staged_trig.at[a_idx],
                            trig_sem.at[a_idx],
                        ).wait()
                        off_c = coords_list[a_idx] - bases_list[a_idx]
                        t_bits = jax.lax.bitcast_convert_type(
                            staged_trig[a_idx], jnp.uint16
                        ).astype(jnp.int32)
                        sel_bits = jnp.max(
                            jnp.where(
                                jnp.arange(angle_rows)[:, None] == off_c,
                                t_bits,
                                0,
                            ),
                            axis=0,
                            keepdims=True,
                        )
                        ang_per_axis.append(
                            jax.lax.bitcast_convert_type(
                                sel_bits.astype(jnp.uint16), jnp.bfloat16
                            ).astype(jnp.float32)
                        )
                    if axes == 1 or rope_axes is None:
                        c = ang_per_axis[0][:, :half]
                        s = ang_per_axis[0][:, half:rotary_dim]
                    else:
                        c_chunks = []
                        s_chunks = []
                        for w in range((half + 31) // 32):
                            lo = w * 32
                            hi = min(lo + 32, half)
                            bit_iota = lax.broadcasted_iota(jnp.int32, (1, hi - lo), 1)
                            cw = ang_per_axis[0][:, lo:hi]
                            sw = ang_per_axis[0][:, half + lo : half + hi]
                            for a_idx in range(1, axes):
                                m_i32 = jnp.int32(axis_word_masks[a_idx][w])
                                take = (
                                    lax.shift_right_arithmetic(m_i32, bit_iota)
                                    & jnp.int32(1)
                                ) != 0
                                cw = jnp.where(take, ang_per_axis[a_idx][:, lo:hi], cw)
                                sw = jnp.where(
                                    take,
                                    ang_per_axis[a_idx][:, half + lo : half + hi],
                                    sw,
                                )
                            c_chunks.append(cw)
                            s_chunks.append(sw)
                        c = (
                            c_chunks[0]
                            if len(c_chunks) == 1
                            else jnp.concatenate(c_chunks, axis=-1)
                        )
                        s = (
                            s_chunks[0]
                            if len(s_chunks) == 1
                            else jnp.concatenate(s_chunks, axis=-1)
                        )
                else:
                    assert fang_all is not None
                    fang_r = jnp.sum(
                        jnp.where(row_iota == r, fang_all, 0.0),
                        axis=0,
                        keepdims=True,
                    )
                    c = fang_r[:, :half]
                    s = fang_r[:, half:rotary_dim]
                a = normed[:, :half].astype(jnp.float32)
                b = normed[:, half:rotary_dim].astype(jnp.float32)
                rot = jnp.concatenate(
                    [
                        bf16(a * c - bf16(b * s)),
                        bf16(a * s + bf16(b * c)),
                        normed[:, rotary_dim:],
                    ],
                    axis=-1,
                )
                pltpu.make_async_copy(
                    idx_dst.at[meta_ref[5, r]],
                    staged_pair.at[r],
                    pair_sem.at[r],
                ).wait()
                pair_half = (old_ctx // ratio) % 2
                carrier = (
                    staged_pair.at[r].bitcast(jnp.bfloat16)
                    if index_pool.dtype != jnp.bfloat16
                    else staged_pair.at[r]
                )
                keep = jnp.arange(2)[:, None] == pair_half
                carrier[...] = jnp.where(
                    keep, jnp.broadcast_to(rot, (2, width)), carrier[...]
                )
                pair_copy = pltpu.make_async_copy(
                    staged_pair.at[r],
                    idx_dst.at[meta_ref[5, r]],
                    pair_sem.at[r],
                )
                pair_copy.start()
                if wait_writes_inline:
                    pair_copy.wait()

        @pl.when(is_parallel)
        def _update_all_parallel():
            for r in range(batch):

                @pl.when(meta_ref[2, r] != 0)
                def _upd_par(*, r=r):
                    _process_row_rmw(r, wait_writes_inline=False)

        if has_all_single_flag:

            @pl.when(~is_parallel)
            def _update_all_sequential():
                for r in range(batch):

                    @pl.when(meta_ref[2, r] != 0)
                    def _upd_seq(*, r=r):
                        _process_row_rmw(r, wait_writes_inline=True)

        # 5. Wait for all in-flight HBM writes (`main_pool`, `ring_pool`, `index_pool`)
        for r in range(batch):

            @pl.when(meta_ref[2, r] != 0)
            def _wait_main(*, r=r):
                pltpu.make_async_copy(
                    packed_vmem_ref.at[pl.ds(r, 1)],
                    out_main_ref.at[meta_ref[0, r], pl.ds(meta_ref[1, r], 1)],
                    main_sem.at[r],
                ).wait()

        @pl.when(is_parallel)
        def _wait_parallel_rmw():
            for r in range(batch):

                @pl.when((meta_ref[2, r] != 0) & (meta_ref[3, r] > 0))
                def _wait_ring(*, r=r):
                    pltpu.make_async_copy(
                        staged_ring.at[r],
                        ring_dst.at[meta_ref[3, r], pl.ds(0, total_ring_tokens)],
                        ring_sem.at[r],
                    ).wait()

                @pl.when(meta_ref[5, r] >= 0)
                def _wait_pair(*, r=r):
                    pltpu.make_async_copy(
                        staged_pair.at[r],
                        idx_dst.at[meta_ref[5, r]],
                        pair_sem.at[r],
                    ).wait()

    num_scalar = 2 if use_tags else 1
    scalar_args = (meta_all, positions) if use_tags else (meta_all,)
    trig_in_specs = [pl.BlockSpec(memory_space=pltpu.HBM)] if use_tags_trig_dma else []
    trig_call_args = (trig_cache,) if use_tags_trig_dma else ()
    if use_packed_norms:
        assert norm_weights_packed is not None
        in_specs = [
            pl.BlockSpec(kv_iqk.shape, lambda *_: (0, 0)),
            pl.BlockSpec(norm_weights_packed.shape, lambda *_: (0,)),
            pl.BlockSpec(both_angles.shape, lambda *_: (0, 0)),
            *trig_in_specs,
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]
        call_args = (
            *scalar_args,
            kv_iqk,
            norm_weights_packed,
            both_angles,
            *trig_call_args,
            main_pool,
            ring_pool,
            index_view,
        )
        base_alias = num_scalar + 3 + (1 if use_tags_trig_dma else 0)
    else:
        assert niq is not None and nik is not None and k_norm_multiplier is not None
        in_specs = [
            pl.BlockSpec(kv_iqk.shape, lambda *_: (0, 0)),
            pl.BlockSpec(niq.shape, lambda *_: (0,)),
            pl.BlockSpec(nik.shape, lambda *_: (0,)),
            pl.BlockSpec(k_norm_multiplier.shape, lambda *_: (0,)),
            pl.BlockSpec(both_angles.shape, lambda *_: (0, 0)),
            *trig_in_specs,
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]
        call_args = (
            *scalar_args,
            kv_iqk,
            niq,
            nik,
            k_norm_multiplier,
            both_angles,
            *trig_call_args,
            main_pool,
            ring_pool,
            index_view,
        )
        base_alias = num_scalar + 5 + (1 if use_tags_trig_dma else 0)
    aliases = {base_alias: 0, base_alias + 1: 1, base_alias + 2: 2}
    trig_scratch = (
        [
            pltpu.VMEM(
                (axes, angle_rows, trig_cache.shape[1]),
                trig_cache.dtype,
            ),
            pltpu.SemaphoreType.DMA((axes,)),
        ]
        if use_tags_trig_dma and trig_cache is not None
        else []
    )

    new_main, new_ring, new_idx_view, iq_out = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar,
            grid=(1,),
            in_specs=in_specs,
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((batch, index_heads, index_dim), lambda *_: (0, 0, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((batch, 1, 4, head_dim), main_pool.dtype),
                pltpu.SemaphoreType.DMA((batch,)),
                pltpu.VMEM(
                    (batch, total_ring_tokens, *ring_pool.shape[2:]), ring_pool.dtype
                ),
                pltpu.SemaphoreType.DMA((batch,)),
                pltpu.VMEM((batch, packing, width), index_pool.dtype),
                pltpu.SemaphoreType.DMA((batch,)),
                *trig_scratch,
            ],
        ),
        out_shape=[
            jax.ShapeDtypeStruct(main_pool.shape, main_pool.dtype),
            jax.ShapeDtypeStruct(ring_pool.shape, ring_pool.dtype),
            jax.ShapeDtypeStruct(index_view.shape, index_view.dtype),
            jax.ShapeDtypeStruct((batch, index_heads, index_dim), jnp.bfloat16),
        ],
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        name="qsa_fused_pre_indexer_all",
    )(*call_args)
    return new_main, new_ring, new_idx_view.reshape(index_pool.shape), iq_out


def qsa_prepare_packed_q_and_gate(
    qgate_0: jax.Array,  # bf16[batch, split_cols]
    qgate_1: jax.Array,  # bf16[batch, total_cols - split_cols]
    nq: jax.Array,  # bf16[256]
    both_angles: jax.Array,  # bf16[2 * batch, 128] or bf16[batch, 128]
    *,
    kv_heads: int = 2,
    real_heads: int = 12,
    head_group: int = 16,
    rotary_dim: int = 64,
    norm_eps: float = 1e-6,
    return_combined: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Fuse `(qgate_0, qgate_1)` + `gemma_norm` + `main_rope` + GQA packing."""
    batch = qgate_0.shape[0]
    dim = nq.shape[0]
    query_group = 8
    half = rotary_dim // 2

    def _compute_heads(qg0_ref, qg1_ref, nq_ref, ang_ref):
        qg_2d = jnp.concatenate([qg0_ref[...], qg1_ref[...]], axis=1)
        nq_w = nq_ref[...]
        t = ang_ref[...].astype(jnp.float32)
        c, s = t[:, None, :half], t[:, None, half:rotary_dim]
        head_cols_2d = real_heads * 2 * dim
        pad_cols_2d = (head_group - real_heads) * 2 * dim
        valid_h = (
            lax.broadcasted_iota(jnp.int32, (query_group, head_group, dim), 1)
            < real_heads
        )
        outs = []
        for h in range(kv_heads):
            sub_2d = qg_2d[:, h * head_cols_2d : (h + 1) * head_cols_2d]
            padded_3d = jnp.pad(sub_2d, ((0, 0), (0, pad_cols_2d))).reshape(
                query_group, head_group, 2 * dim
            )
            q_raw = padded_3d[:, :, :dim]
            g_raw = padded_3d[:, :, dim:]
            x = gemma_norm(q_raw, nq_w, norm_eps)
            a, b = (
                x[..., :half].astype(jnp.float32),
                x[..., half:rotary_dim].astype(jnp.float32),
            )
            q_rot = jnp.concatenate(
                [bf16(a * c - b * s), bf16(b * c + a * s), x[..., rotary_dim:]],
                axis=-1,
            )
            q_rot = jnp.where(valid_h, q_rot, 0).reshape(query_group * head_group, dim)
            g_out = jnp.where(valid_h, g_raw, 0).reshape(query_group * head_group, dim)
            outs.append((q_rot, g_out))
        return outs

    in_specs = [
        pl.BlockSpec((query_group, qgate_0.shape[1]), lambda b: (b, 0)),
        pl.BlockSpec((query_group, qgate_1.shape[1]), lambda b: (b, 0)),
        pl.BlockSpec(nq.shape, lambda b: (0,)),
        pl.BlockSpec((query_group, both_angles.shape[1]), lambda b: (b, 0)),
    ]

    if return_combined:

        def kernel_combined(qg0_ref, qg1_ref, nq_ref, ang_ref, out_qg_ref):
            outs = _compute_heads(qg0_ref, qg1_ref, nq_ref, ang_ref)
            for h, (q_h, g_h) in enumerate(outs):
                out_qg_ref[0, 0, h] = q_h
                out_qg_ref[0, 1, h] = g_h

        comb_shape = jax.ShapeDtypeStruct(
            (batch // query_group, 2, kv_heads, query_group * head_group, dim),
            jnp.bfloat16,
        )
        comb_spec = pl.BlockSpec(
            (1, 2, kv_heads, query_group * head_group, dim),
            lambda b: (b, 0, 0, 0, 0),
        )
        return pl.pallas_call(
            kernel_combined,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(batch // query_group,),
                in_specs=in_specs,
                out_specs=comb_spec,
            ),
            out_shape=comb_shape,
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
                disable_bounds_checks=True,
            ),
            name="qsa_prepare_packed_qgate_combined",
        )(qgate_0, qgate_1, nq, both_angles)

    def kernel_split(qg0_ref, qg1_ref, nq_ref, ang_ref, out_q_ref, out_gate_ref):
        outs = _compute_heads(qg0_ref, qg1_ref, nq_ref, ang_ref)
        for h, (q_h, g_h) in enumerate(outs):
            out_q_ref[0, h] = q_h
            out_gate_ref[0, h] = g_h

    out_shape = jax.ShapeDtypeStruct(
        (batch // query_group, kv_heads, query_group * head_group, dim),
        jnp.bfloat16,
    )
    out_spec = pl.BlockSpec(
        (1, kv_heads, query_group * head_group, dim), lambda b: (b, 0, 0, 0)
    )
    return pl.pallas_call(
        kernel_split,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch // query_group,),
            in_specs=in_specs,
            out_specs=[out_spec, out_spec],
        ),
        out_shape=[out_shape, out_shape],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        name="qsa_prepare_packed_q_and_gate",
    )(qgate_0, qgate_1, nq, both_angles)


def prepare_sc_topk_lengths_flat(
    lengths: jax.Array,
    *,
    n_scores: int,
    k: int = 512,
    compression_ratio: int = 4,
) -> jax.Array:
    """Pre-build 16-lane-padded SparseCore `lengths_2d` buffer before `StreamIdxTC`."""
    b = lengths.shape[0]
    p = _pick_partition(b, n_scores, k)
    groups_per_chip = 32 // p
    num_waves = _cdiv(b, groups_per_chip)
    padded_b = num_waves * groups_per_chip
    shift = (compression_ratio - 1).bit_length()
    direct = jnp.clip(
        lax.shift_right_arithmetic(jnp.maximum(lengths, 0), shift),
        0,
        max(n_scores, k),
    )
    if padded_b > b:
        direct = jnp.pad(direct, (0, padded_b - b))
    return jnp.pad(
        jnp.broadcast_to(direct[:, None, None], (padded_b, p, 1)),
        ((0, 0), (0, 0), (0, LANES - 1)),
    ).reshape(-1)


def _matmul_wq(
    x: jax.Array, w_part: jax.Array, *, transpose_wq: bool = False
) -> jax.Array:
    if transpose_wq:
        return lax.dot_general(
            x,
            w_part,
            (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
    return jnp.matmul(x, w_part, preferred_element_type=jnp.float32)


class OverlappedWQ0SparseCoreTopK(SparseCoreTopK):
    """Overlaps `qgate_0 = hidden @ w_q_0` on TC inside `sc_topk_direct` on SC."""

    def __init__(
        self,
        hidden: jax.Array,
        w_q_0: jax.Array,
        lengths: jax.Array,
        compression_ratio: int = 4,
        sched_group_id: int | None = None,
        use_direct_sc_kernel: bool = True,
        lengths_2d_flat: jax.Array | None = None,
        transpose_wq: bool = False,
    ):
        self.hidden = hidden
        self.w_q_0 = w_q_0
        self.lengths = lengths
        self.compression_ratio = compression_ratio
        self.sched_group_id = sched_group_id
        self.use_direct_sc_kernel = use_direct_sc_kernel
        self.lengths_2d_flat = lengths_2d_flat
        self.transpose_wq = transpose_wq
        self.qgate_0_out: jax.Array | None = None

    def select(
        self,
        score_bits: jax.Array,
        k: int,
        row_lengths: jax.Array,
        *,
        scheduling_group_id: int | None,
        return_scores: bool,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        assert not return_scores
        del row_lengths  # Dead-code-eliminate _effective_row_lengths!
        gid = (
            self.sched_group_id if scheduling_group_id is None else scheduling_group_id
        )

        if self.use_direct_sc_kernel:
            b, n = score_bits.shape
            p = _pick_partition(b, n, k)
            groups_per_chip = 32 // p
            slice_len = _align_to(n // p, LANES)
            num_waves = _cdiv(b, groups_per_chip)
            padded_b = num_waves * groups_per_chip
            pad_b = padded_b - b
            pad_n = slice_len * p - n
            if pad_b == 0 and pad_n == 0:
                lengths_flat = (
                    self.lengths_2d_flat
                    if self.lengths_2d_flat is not None
                    else prepare_sc_topk_lengths_flat(
                        self.lengths,
                        n_scores=n,
                        k=k,
                        compression_ratio=self.compression_ratio,
                    )
                )
                words = (
                    score_bits
                    if score_bits.dtype == jnp.int32
                    else lax.bitcast_convert_type(score_bits, jnp.int32)
                )
                words_flat, lengths_flat, hidden_gated = lax.optimization_barrier(
                    (words.reshape(-1), lengths_flat, self.hidden)
                )
                sc = pltpu.get_tpu_info().sparse_core
                assert sc is not None
                mesh = plsc.VectorSubcoreMesh(
                    num_cores=sc.num_cores,
                    num_subcores=sc.num_subcores,
                    core_axis_name="core",
                    subcore_axis_name="subcore",
                )
                slot_type = jax.ShapeDtypeStruct((padded_b * k,), jnp.int32)
                scratch_types = _topk_scratch_types(k, p, False, sc.num_subcores)
                body_fn = _topk_body_for_shape(
                    b, n, k, p, slice_len, num_waves, True, False, False
                )
                out = pl.kernel(
                    body_fn,
                    out_type=(slot_type, None),
                    compiler_params=pltpu.CompilerParams(
                        disable_bounds_checks=True, needs_layout_passes=False
                    ),
                    scratch_types=scratch_types,
                    mesh=mesh,
                    name=f"sc_topk_direct_b{b}_n{n}_k{k}_p{p}",
                )(words_flat, lengths_flat)
                if gid is not None:
                    out = xla_metadata.set_xla_metadata(out, _scheduling_group_id=gid)
                blocks = out[0].reshape(padded_b, k)[:b]
                qg0_f32 = _matmul_wq(
                    hidden_gated, self.w_q_0, transpose_wq=self.transpose_wq
                )
                if gid is not None:
                    qg0_f32 = xla_metadata.set_xla_metadata(
                        qg0_f32, _scheduling_group_id=gid
                    )
                qgate_0 = qg0_f32.astype(jnp.bfloat16)
                if gid is not None:
                    qgate_0 = xla_metadata.set_xla_metadata(
                        qgate_0, _scheduling_group_id=gid
                    )
                blocks, qgate_0 = lax.optimization_barrier((blocks, qgate_0))
                self.qgate_0_out = qgate_0
                return blocks

        direct_row_lengths = jnp.clip(
            lax.shift_right_arithmetic(jnp.maximum(self.lengths, 0), 2),
            0,
            max(score_bits.shape[1], k),
        )
        score_bits, direct_row_lengths, hidden_gated = lax.optimization_barrier(
            (score_bits, direct_row_lengths, self.hidden)
        )
        blocks = qwen_sparsecore_topk(
            score_bits,
            k,
            row_lengths=direct_row_lengths,
            write_empty_rows=True,
            scheduling_group_id=gid,
            return_scores=False,
        )
        qg0_f32 = _matmul_wq(hidden_gated, self.w_q_0, transpose_wq=self.transpose_wq)
        if gid is not None:
            qg0_f32 = xla_metadata.set_xla_metadata(qg0_f32, _scheduling_group_id=gid)
        qgate_0 = qg0_f32.astype(jnp.bfloat16)
        if gid is not None:
            qgate_0 = xla_metadata.set_xla_metadata(qgate_0, _scheduling_group_id=gid)
        blocks, qgate_0 = lax.optimization_barrier((blocks, qgate_0))
        self.qgate_0_out = qgate_0
        return blocks


def build_fast_decode_streamindex_metadata(
    lengths: jax.Array,
    *,
    bkv_sz: int,
    pages_per_seq: int,
    page_size: int,
    compression_ratio: int,
    seq_batch_size: int,
) -> MetadataRef:
    """Zero-scatter, zero-divmod 1-VREG schedule for per-query `StreamIdxTC`."""
    max_num_seqs = lengths.shape[0]
    max_seq_tiles = max_num_seqs // seq_batch_size
    max_kv_pages_per_block = max(1, bkv_sz // page_size)
    max_bkv = max(1, pl.cdiv(pages_per_seq, max_kv_pages_per_block))
    max_steps = align_to(max(1, max_seq_tiles * max_bkv), 128)

    shift = (compression_ratio - 1).bit_length()
    kv_lens = lax.shift_right_arithmetic(jnp.maximum(lengths, 0), shift).reshape(
        max_seq_tiles, seq_batch_size
    )
    max_kv_lens = jnp.max(kv_lens, axis=-1)
    # Any sequence tile with `max_kv_lens == 0` has 0 compressed keys -> 0 steps!
    s_num_bkv = ((max_kv_lens + (bkv_sz - 1)) // bkv_sz).astype(jnp.int32)

    step_id = jnp.arange(max_steps, dtype=jnp.int32)
    cur_start = jnp.int32(0)
    tile_idx = jnp.zeros((max_steps,), dtype=jnp.int32)
    tile_start = jnp.zeros((max_steps,), dtype=jnp.int32)
    for t in range(max_seq_tiles):
        if t > 0:
            ge = step_id >= cur_start
            tile_idx = jnp.where(ge, jnp.int32(t), tile_idx)
            tile_start = jnp.where(ge, cur_start, tile_start)
        cur_start = cur_start + s_num_bkv[t]
    total_steps = cur_start
    live = step_id < total_steps
    batch_tile_idx = jnp.where(live, tile_idx * seq_batch_size, 0)
    bkv_idx = jnp.where(live, step_id - tile_start, 0)
    bq_idx = jnp.zeros((max_steps,), dtype=jnp.int32)
    return MetadataRef.create(
        num_steps=total_steps.reshape(1),
        batch_tile_idx=batch_tile_idx,
        bq_idx=bq_idx,
        bkv_idx=bkv_idx,
    )


def select_blocks_direct_decode(
    iq: jax.Array,
    pool: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    *,
    ratio: int = 4,
    block_topk: int = 512,
    logical_block_size: int = 1664,
    topk_backend: TopKBackend,
    scheduling_group_namespace: int = 0,
    coalesce_pages: bool = True,
    per_query_decode: bool = True,
) -> jax.Array:
    """Direct per-query decode block selection with 5D `page_group=13` DMA."""
    num_pool_pages = pool.shape[0]
    batch_rows, table_cols = table.shape
    if num_pool_pages > 1:
        # Spread inactive sequence rows across distinct valid pool pages `[1, P)`
        # to eliminate multi-way HBM bank collisions on `index_pool[0]`.
        idle_pages = 1 + (
            jnp.arange(batch_rows * table_cols, dtype=jnp.int32).reshape(
                batch_rows, table_cols
            )
            % (num_pool_pages - 1)
        )
        table = jnp.where((lengths > 0)[:, None], table, idle_pages)

    width = pool.shape[-1]
    rows = _compressed_rows_for_ratio(ratio)
    packing = 4 if pool.dtype == jnp.float8_e4m3fn else 2
    if coalesce_pages and pool.ndim == 4:
        pool = pool.reshape(pool.shape[0], -1, 1, packing, width)
    physical_pages = (
        math.prod(pool.shape[1:]) * pool.dtype.itemsize // (rows * width * 2)
    )
    logical_pages = logical_block_size // 128
    if coalesce_pages:
        page_group = get_decode_index_page_group(
            iq,
            pool,
            logical_pages=logical_pages,
            keys_per_base_page=rows,
            pages_per_query=table_cols,
            ratio=ratio,
            topk=block_topk,
        )
        rows *= page_group
        physical_pages //= page_group
        logical_pages //= page_group
    cache = pool.reshape(-1, rows // 2, packing, width)
    expanded_table = (
        table.reshape(-1)
        if (physical_pages == 1 and logical_pages == 1)
        else (
            table[..., None] * physical_pages
            + jnp.arange(logical_pages, dtype=jnp.int32)
        ).reshape(-1)
    )

    query_width = iq.shape[-1]
    q = (
        iq
        if cache.shape[-1] == query_width
        else jnp.pad(iq, ((0, 0), (0, 0), (0, cache.shape[-1] - query_width)))
    )
    scale = query_width**-0.5
    size = len(q)
    page_size = cache.shape[1] * 2
    pages_per_query = table_cols * logical_pages
    key_packing = get_decode_index_key_packing(
        q, cache, pages_per_query=pages_per_query, ratio=ratio, topk=block_topk
    )
    reader_cache = (
        cache.reshape(cache.shape[0], page_size // key_packing, key_packing, -1)
        if key_packing != 2
        else cache
    )
    tile_pages = get_decode_index_pages(
        q, cache, pages_per_query=pages_per_query, ratio=ratio, topk=block_topk
    )
    assert tile_pages is not None
    components = qsa_indexing(key_packing=key_packing)
    tpu_info = pltpu.get_tpu_info()

    if per_query_decode:
        seq_batch_size = 8 if size >= 8 else 4
        eff_sb = min(seq_batch_size, size)
        while size % eff_sb != 0:
            eff_sb //= 2

        prepared_q = prepare_q_inputs(q)
        prepared_weights = prepare_index_weights(
            jnp.full(q.shape[:2], scale, jnp.float32), q.dtype
        )
        cu_q_lens = jnp.arange(size + 1, dtype=jnp.int32)
        pass_meta = build_fast_decode_streamindex_metadata(
            lengths,
            bkv_sz=tile_pages * page_size,
            pages_per_seq=pages_per_query,
            page_size=page_size,
            compression_ratio=ratio,
            seq_batch_size=eff_sb,
        )
        scalar_prefetches = (
            lengths,
            expanded_table,
            cu_q_lens,
            jnp.array([0, size, 0], dtype=jnp.int32),
            pass_meta,
        )
        num_sublanes_total = align_to(pages_per_query, tile_pages) * page_size // 128
        if reader_cache.dtype == jnp.bfloat16:
            num_sublanes_total = align_to(
                num_sublanes_total, tpu_info.get_sublane_tiling(jnp.int32)
            )
        score_shape = (
            size,
            align_to(num_sublanes_total, tpu_info.num_sublanes),
            128,
        )
        scope_name = f"StreamIdxTC-D-bq_1-bkvp_{tile_pages}-sb_{eff_sb}"
        pallas_kernel = jax.named_scope(scope_name)(
            _score_call_for_config(
                (
                    1,
                    tile_pages,
                    eff_sb,
                    topk_cfg.DEFAULT_BUFFER_COUNT[0],
                    KVLayout.HEAD_ALONG_SUBLANE,
                    0,
                    components,
                ),
                score_shape,
                jnp.int32,
                len(scalar_prefetches),
                len(jax.tree_util.tree_leaves(scalar_prefetches)),
                False,
                tpu_info.vmem_capacity_bytes,
                scope_name,
            )
        )
        scores = pallas_kernel(
            *scalar_prefetches,
            prepared_q,
            prepared_weights,
            reader_cache,
        )
        scores_2d = scores.reshape(size, -1)
        if scores_2d.shape[1] < block_topk:
            scores_2d = jnp.pad(
                scores_2d,
                ((0, 0), (0, block_topk - scores_2d.shape[1])),
                constant_values=BITS_NEG_INF,
            )
        row_lengths = (
            jnp.empty((0,), jnp.int32)
            if isinstance(topk_backend, OverlappedWQ0SparseCoreTopK)
            else jnp.minimum(
                lax.shift_right_arithmetic(
                    jnp.maximum(lengths, 0), (ratio - 1).bit_length()
                ),
                scores_2d.shape[1],
            )
        )
        out = topk_backend.select(
            scores_2d,
            block_topk,
            row_lengths=row_lengths,
            scheduling_group_id=None,
            return_scores=False,
        )
        assert isinstance(out, jax.Array)
        return out

    raw_streamindex_topk = getattr(streamindex_topk, "__wrapped__", streamindex_topk)
    cu_q_lens = jnp.arange(size + 1, dtype=jnp.int32)
    distribution = jnp.full((3,), size, jnp.int32)
    return raw_streamindex_topk(
        q,
        jnp.full(q.shape[:2], scale, jnp.float32),
        reader_cache,
        lengths,
        expanded_table,
        cu_q_lens,
        distribution,
        k=block_topk,
        compression_ratio=ratio,
        num_kv_pages_per_block=tile_pages,
        num_queries_per_block=1,
        decode_req_batch_size=8 if size >= 8 else 4,
        enable_early_exit=True,
        components=components,
        topk_backend=topk_backend,
        scheduling_group_namespace=scheduling_group_namespace,
        vmem_limit_bytes=tpu_info.vmem_capacity_bytes,
    )


def compact_pooled_qsa_fast_decode_step(
    hidden: jax.Array,
    kv_iqk: jax.Array,
    w_q: jax.Array,
    nq: jax.Array,
    norm_weights_packed: jax.Array,
    positions: jax.Array,
    trig: jax.Array,
    pool: jax.Array,
    index_pool: jax.Array,
    ring_pool: jax.Array,
    starts: jax.Array,
    context: jax.Array,
    ring_ids: jax.Array,
    main_table: jax.Array,
    compressed_table: jax.Array,
    output_weight: jax.Array,
    residual: jax.Array,
    hc_gate: jax.Array,
    compress_ratio: int,
    token_topk: int,
    norm_eps: float,
    rotary_dim: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    scheduling_group_namespace: int = 0,
    *,
    ring_capacity: int | None = None,
    rope_axes: tuple[int, ...] | None = None,
    cache_tags: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """End-to-end fast QSA step without `selected_tokens` materialization."""
    main_pool = pool
    batch = hidden.shape[0]
    block_size = main_pool.shape[1]
    main_table_2d = main_table.reshape(len(context), -1)
    compressed_table_2d = (
        main_table_2d
        if main_table is compressed_table
        else compressed_table.reshape(len(context), -1)
    )

    use_tags_trig_dma = (
        bool(cache_tags) and (trig.shape[0] >= 8) and (trig.shape[0] % 8 == 0)
    )
    step_meta = build_step_qsa_metadata(
        batch=batch,
        block_size=block_size,
        num_main_pages=len(main_pool),
        index_pool_shape=index_pool.shape,
        index_pool_dtype=index_pool.dtype,
        positions=positions,
        trig=trig,
        starts=starts,
        context=context,
        ring_ids=ring_ids,
        main_table=main_table_2d,
        compressed_table=compressed_table_2d,
        rotary_dim=rotary_dim,
        compress_ratio=compress_ratio,
        ring_capacity=ring_capacity,
        rope_axes=rope_axes,
        use_tags_trig_dma=use_tags_trig_dma,
    )
    lengths = step_meta.lengths
    both_angles = step_meta.both_angles
    meta_all = step_meta.meta_all
    req = step_meta.req
    comp_req_table = compressed_table_2d[req]
    main_req_table = (
        comp_req_table if main_table_2d is compressed_table_2d else main_table_2d[req]
    )

    main_pool, ring_pool, index_pool, iq = qsa_fused_pre_indexer_all(
        main_pool,
        ring_pool,
        index_pool,
        kv_iqk,
        None,
        None,
        None,
        both_angles,
        meta_all,
        norm_weights_packed=norm_weights_packed,
        positions=positions,
        cache_tags=cache_tags,
        trig_cache=trig if use_tags_trig_dma else None,
        rope_axes=rope_axes,
        ratio=compress_ratio,
        ring_capacity=ring_capacity,
        rotary_dim=rotary_dim,
        norm_eps=norm_eps,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    split_cols = 5120
    if w_q.shape[0] == hidden.shape[1]:
        transpose_wq = False
        w_q_0 = w_q[:, :split_cols]
        w_q_1 = w_q[:, split_cols:]
    elif w_q.shape[1] == hidden.shape[1]:
        transpose_wq = True
        w_q_0 = w_q[:split_cols, :]
        w_q_1 = w_q[split_cols:, :]
    else:
        raise ValueError(f"Unsupported w_q shape {w_q.shape} for hidden {hidden.shape}")
    topk_gid = (
        topk_cfg.scoped_scheduling_group_id(scheduling_group_namespace, 2 * batch)
        if scheduling_group_namespace > 0
        else None
    )
    gather_gid = (
        topk_cfg.scoped_scheduling_group_id(scheduling_group_namespace, 2 * batch + 1)
        if scheduling_group_namespace > 0
        else None
    )

    topk_overlapped = OverlappedWQ0SparseCoreTopK(
        hidden,
        w_q_0,
        lengths,
        compression_ratio=compress_ratio,
        sched_group_id=topk_gid,
        use_direct_sc_kernel=True,
        transpose_wq=transpose_wq,
    )
    blocks = select_blocks_direct_decode(
        iq,
        index_pool,
        lengths,
        comp_req_table,
        ratio=compress_ratio,
        block_topk=token_topk // compress_ratio,
        logical_block_size=block_size,
        topk_backend=topk_overlapped,
        scheduling_group_namespace=scheduling_group_namespace,
        coalesce_pages=True,
        per_query_decode=True,
    )
    qgate_0 = topk_overlapped.qgate_0_out
    assert qgate_0 is not None

    tok_indices, meta_attn_2d = qsa_prepare_token_indices_packed_meta(
        blocks,
        lengths,
        main_req_table,
        page_size=block_size,
        ratio=compress_ratio,
        sc_chunk_tokens=256,
    )
    (
        tok_indices,
        meta_attn_2d,
        hidden_g1,
        qgate_0,
        both_angles_g1,
        main_pool,
    ) = lax.optimization_barrier(
        (tok_indices, meta_attn_2d, hidden, qgate_0, both_angles, main_pool)
    )

    gathered_kv = qsa_sc_gather_2level_tc_tiled(
        main_pool,
        tok_indices,
        None,
        num_streams=8,
        chunk_tokens=256,
        tokens_per_row=1,
        ratio=compress_ratio,
        skip_negative_chunks=True,
    )
    if gather_gid is not None:
        gathered_kv = xla_metadata.set_xla_metadata(
            gathered_kv, _scheduling_group_id=gather_gid
        )
    qg1_f32 = _matmul_wq(hidden_g1, w_q_1, transpose_wq=transpose_wq)
    if gather_gid is not None:
        qg1_f32 = xla_metadata.set_xla_metadata(
            qg1_f32, _scheduling_group_id=gather_gid
        )
    qgate_1 = qg1_f32.astype(jnp.bfloat16)
    if gather_gid is not None:
        qgate_1 = xla_metadata.set_xla_metadata(
            qgate_1, _scheduling_group_id=gather_gid
        )
    packed_qgate = qsa_prepare_packed_q_and_gate(
        qgate_0,
        qgate_1,
        nq,
        both_angles_g1,
        rotary_dim=rotary_dim,
        norm_eps=norm_eps,
        return_combined=True,
    )
    assert isinstance(packed_qgate, jax.Array)
    if gather_gid is not None:
        packed_qgate = xla_metadata.set_xla_metadata(
            packed_qgate, _scheduling_group_id=gather_gid
        )
    gathered_kv, packed_qgate = lax.optimization_barrier((gathered_kv, packed_qgate))

    output = qsa_fast_grouped_qsa_projected_attention(
        packed_qgate,
        None,
        gathered_kv,
        main_pool,
        meta_attn_2d,
        None,
        output_weight,
        residual,
        hc_gate,
        kv_heads=2,
        ratio=compress_ratio,
        k_scale=k_scale,
        v_scale=v_scale,
        bf16_pv=True,
        combined_qgate=True,
        unroll_tiles=True,
        native_f8_kv=True,
    )
    return main_pool, index_pool, ring_pool, output


def compact_pooled_qsa_projected_unified_step(
    hidden: jax.Array,
    kv_iqk: jax.Array,
    w_q: jax.Array,
    nq: jax.Array,
    nk: jax.Array,
    niq: jax.Array,
    nik: jax.Array,
    norm_weights_packed: jax.Array,
    k_norm_multiplier: jax.Array,
    positions: jax.Array,
    trig: jax.Array,
    pool: jax.Array,
    index_pool: jax.Array,
    ring_pool: jax.Array,
    starts: jax.Array,
    context: jax.Array,
    ring_ids: jax.Array,
    main_table: jax.Array,
    compressed_table: jax.Array,
    request_distribution: jax.Array,
    output_weight: jax.Array,
    residual: jax.Array,
    hc_gate: jax.Array,
    original_output_weight: jax.Array,
    compress_ratio: int,
    token_topk: int,
    norm_eps: float,
    rotary_dim: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    hc_token_tile: int = 0,
    hc_min_tokens: int = 1024,
    decode_cu: jax.Array | None = None,
    scheduling_group_namespace: int = 0,
    ring_capacity: int | None = None,
    *,
    sc_min_context: int | None = None,
    rope_axes: tuple[int, ...] | None = None,
    cache_tags: bool | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Unified stable-ABI QSA projected op with zero-`cond` small-bucket dispatch."""
    batch = hidden.shape[0]
    effective_cache_tags = (
        (rope_axes is not None) if cache_tags is None else bool(cache_tags)
    )
    cap = compress_ratio if ring_capacity is None else ring_capacity
    ring_rows = (cap + 7) // 8 * 8
    req_ring_tokens = (ring_rows // 2) + (1 if effective_cache_tags else 0)
    can_use_fast_decode = (
        _hardware() is not None
        and pltpu.get_tpu_info().sparse_core is not None
        and batch in (8, 16, 24, 32)
        and batch <= context.shape[0]
        and starts.shape[0] == context.shape[0] + 1
        and pool.dtype == jnp.float8_e4m3fn
        and pool.shape[1] >= 128
        and pool.shape[1] % 128 == 0
        and pool.shape[2:] == (1, 4, 256)
        and index_pool.dtype == jnp.bfloat16
        and ring_pool.shape[1] >= req_ring_tokens
        and compress_ratio == 4
        and token_topk == 2048
        and cap in (compress_ratio, 2 * compress_ratio)
    )
    if can_use_fast_decode:
        return compact_pooled_qsa_fast_decode_step(
            hidden,
            kv_iqk,
            w_q,
            nq,
            norm_weights_packed,
            positions,
            trig,
            pool,
            index_pool,
            ring_pool,
            starts,
            context,
            ring_ids,
            main_table,
            compressed_table,
            output_weight,
            residual,
            hc_gate,
            compress_ratio,
            token_topk,
            norm_eps,
            rotary_dim,
            k_scale,
            v_scale,
            scheduling_group_namespace,
            ring_capacity=ring_capacity,
            rope_axes=rope_axes,
            cache_tags=effective_cache_tags,
        )

    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import compact_pooled_qsa_step

    if w_q.shape[0] == hidden.shape[1]:
        transpose_wq = False
    elif w_q.shape[1] == hidden.shape[1]:
        transpose_wq = True
    else:
        raise ValueError(f"Unsupported w_q shape {w_q.shape} for hidden {hidden.shape}")
    qgate = _matmul_wq(hidden, w_q, transpose_wq=transpose_wq).astype(jnp.bfloat16)
    key = kv_iqk[:, :512]
    value = kv_iqk[:, 512:1024]
    index_qk = kv_iqk[:, 1024:]
    updated, index_updated, ring_updated, output, _blocks = compact_pooled_qsa_step(
        qgate,
        key,
        value,
        index_qk,
        nq,
        nk,
        niq,
        nik,
        positions,
        trig,
        pool,
        index_pool,
        ring_pool,
        starts,
        context,
        ring_ids,
        main_table,
        compressed_table,
        request_distribution,
        compress_ratio,
        token_topk,
        norm_eps,
        rotary_dim,
        k_scale,
        v_scale,
        output_weight,
        residual,
        hc_gate,
        original_output_weight,
        hc_token_tile,
        hc_min_tokens,
        k_norm_multiplier,
        decode_cu,
        scheduling_group_namespace,
        ring_capacity,
        topk_backend=None,
        sc_min_context=sc_min_context,
        rope_axes=rope_axes,
        enable_sc_preparation=True,
        cache_tags=cache_tags,
    )
    return updated, index_updated, ring_updated, output
