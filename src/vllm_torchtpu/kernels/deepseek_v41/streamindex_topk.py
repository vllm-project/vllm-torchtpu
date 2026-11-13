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
"""StreamIndex top-k for the DeepSeek-V4.1 indexer (SEQ_ALONG_LANE layout)."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import bref_override
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.config import (
    DEFAULT_BUFFER_COUNT,
    MlaCase,
    reserve_scheduling_group_ids,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.metadata import (
    MetadataRef,
    next_power_of_2,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.streamindex_topk import (
    DEFAULT_VMEM_LIMIT_BYTES,
    align_to,
    prepare_index_weights,
    prepare_q_inputs,
)
from vllm_torchtpu.kernels.deepseek_v41.sparsecore_topk import (
    sc_build_cand_page_table,
    sc_cand_gather_topk,
    sc_write_cand_and_topk,
    sparsecore_topk,
)


def kernel(
    seq_lens_ref,  # [max_num_seqs]
    page_indices_ref,  # [max_num_seqs * pages_per_seq]
    cu_q_lens_ref,  # [max_num_seqs + 1]
    start_end_seq_idx_ref,  # [3] (start_seq, end_seq, chunk_token_start)
    metadata_ref,  # MetadataRef
    q_hbm_ref,  # [max_num_tokens, num_q_heads, head_dim]
    indexer_weights_hbm_ref,  # [max_num_tokens, num_q_heads]
    cache_kv_hbm_ref,  # [total_num_pages, kv_sublane_groups, kv_packing, page_size]
    *score_refs,
    compression_ratio: int,
    bkv_p: int,
    bq_sz: int,
    seq_batch_size: int,
    buffer_count: int = 3,
    chunk_idx: int = 0,
    produce_blk_scores: bool = False,
):
    """Executes the n-buffered Pallas scoring pipeline for StreamIndex Top-K."""
    scores_hbm_ref, blk_scores_hbm_ref = (
        score_refs[-2:] if produce_blk_scores else (score_refs[-1], None)
    )

    _, num_q_heads, head_dim = q_hbm_ref.shape
    _, kv_sublane_groups, kv_packing, page_size = cache_kv_hbm_ref.shape
    kv_head_dim_groups = head_dim // kv_packing
    pages_per_seq = page_indices_ref.shape[0] // seq_lens_ref.shape[0]

    bkv_sz = bkv_p * page_size
    num_sublanes_bkv = bkv_sz // 128
    num_blk_sublanes_bkv = num_sublanes_bkv // 8
    shift_comp = (compression_ratio - 1).bit_length()
    step_offset = sum(metadata_ref.num_steps[m] for m in range(chunk_idx))

    def seq_tile(p_id):
        return metadata_ref.batch_tile_idx[step_offset + p_id]

    def bq_block(p_id):
        return metadata_ref.bq_idx[step_offset + p_id]

    def bkv_block(p_id):
        return metadata_ref.bkv_idx[step_offset + p_id]

    in_pipe_mode = pl.Buffered(buffer_count=buffer_count, use_lookahead=True)
    out_pipe_mode = pl.Buffered(buffer_count=2, use_lookahead=False)

    def q_window(ref, *minor_dims):
        spec = pl.BlockSpec(
            (pl.BoundedSlice(seq_batch_size * bq_sz), *minor_dims),
            lambda p_id: (seq_tile(p_id), bq_block(p_id), 0),
            pipeline_mode=in_pipe_mode,
        )
        return bref_override.StreamIndexQBufferedRef.input(
            spec=spec,
            dtype_or_type=ref,
            buffer_count=buffer_count,
            use_lookahead=True,
            bq_sz=bq_sz,
            seq_batch_size=seq_batch_size,
            chunk_tokens=scores_hbm_ref.shape[0],
            chunk_idx=chunk_idx,
        )

    def scores_window(ref, num_sublanes):
        return bref_override.StreamIndexOBufferedRef.output(
            spec=pl.BlockSpec(
                (pl.BoundedSlice(seq_batch_size * bq_sz), num_sublanes, 128),
                lambda p_id: (seq_tile(p_id), bq_block(p_id), bkv_block(p_id), 0),
                pipeline_mode=out_pipe_mode,
            ),
            dtype_or_type=ref,
            buffer_count=2,
            use_lookahead=False,
            bq_sz=bq_sz,
            num_sublanes=num_sublanes,
            seq_batch_size=seq_batch_size,
            chunk_idx=chunk_idx,
        )

    kv_spec = pl.BlockSpec(
        (pl.BoundedSlice(seq_batch_size), kv_sublane_groups, kv_packing, bkv_sz),
        lambda p_id: (seq_tile(p_id), 0, 0, bkv_block(p_id)),
        pipeline_mode=in_pipe_mode,
    )
    kv_alloc = bref_override.StreamIndexKVSeqAlongLaneBufferedRef.input(
        spec=kv_spec,
        dtype_or_type=cache_kv_hbm_ref,
        buffer_count=buffer_count,
        use_lookahead=True,
        bkv_p=bkv_p,
        page_size=page_size,
        pages_per_seq=pages_per_seq,
        seq_batch_size=seq_batch_size,
        chunk_idx=chunk_idx,
    )
    allocs = (
        q_window(q_hbm_ref, num_q_heads, head_dim),
        q_window(indexer_weights_hbm_ref, num_q_heads),
        kv_alloc,
        scores_window(scores_hbm_ref, num_sublanes_bkv),
    )
    if produce_blk_scores:
        allocs += (scores_window(blk_scores_hbm_ref, num_blk_sublanes_bkv),)

    def transpose_to_head_major(bq_rows, n_tok):
        return bq_rows.transpose(1, 0, 2).reshape(num_q_heads * n_tok, head_dim)

    def reduce_heads(bq_rows, bkv, bq_weights, w_pre_folded: bool = False):
        n_tok = bq_weights.shape[0]

        def qk(rows):
            return jnp.einsum(
                "nd,dm->nm", rows, bkv, preferred_element_type=jnp.float32
            )

        if n_tok < 8:
            w = bq_weights.astype(jnp.float32)
            st = jnp.maximum(qk(bq_rows), 0.0).reshape(n_tok, num_q_heads, -1)
            return (st * w[:, :, None]).sum(axis=1)

        # n_tok >= 8 only in the MIXED pass, whose q is pre-transposed & pre-weighted into q_hm_ref at bkv_idx == 0.
        rows_hm = bq_rows.reshape(num_q_heads, n_tok, head_dim)
        head_group = min(64 if n_tok == 8 else 16, num_q_heads)
        acc = None
        if w_pre_folded:
            for g in range(0, num_q_heads, head_group):
                grp = min(head_group, num_q_heads - g)
                st_g = jnp.maximum(
                    qk(rows_hm[g : g + grp].reshape(grp * n_tok, head_dim)), 0.0
                ).reshape(grp, n_tok, -1)
                term = st_g.sum(axis=0)
                acc = term if acc is None else acc + term
            return acc * bq_weights
        w_hm = bq_weights.astype(jnp.float32).T[:, :, None]
        for g in range(0, num_q_heads, head_group):
            grp = min(head_group, num_q_heads - g)
            st_g = jnp.maximum(
                qk(rows_hm[g : g + grp].reshape(grp * n_tok, head_dim)), 0.0
            ).reshape(grp, n_tok, -1)
            term = (st_g * w_hm[g : g + grp]).sum(axis=0)
            acc = term if acc is None else acc + term
        return acc

    def step_body(
        q_hm_ref, q_vmem, weights_vmem, bkv_vmem, scores_vmem, blk_scores_vmem=None
    ):
        p_id = pl.program_id(0)
        batch_start_seq_idx = seq_tile(p_id)
        bq_idx = bq_block(p_id)
        bkv_idx = bkv_block(p_id)

        w_data = weights_vmem[...].reshape(seq_batch_size, -1, num_q_heads)
        w_pre_folded = False
        if q_hm_ref is not None:

            @pl.when(bkv_idx == 0)
            def _():
                q_hm_ref[...] = transpose_to_head_major(q_vmem[...], bq_sz)

            bq_vec = [q_hm_ref[...]]
            bq_weights_vec = [w_data[0]]
        else:
            q_data = q_vmem[...].reshape(seq_batch_size, -1, head_dim)
            bq_vec = [q_data[i] for i in range(seq_batch_size)]
            bq_weights_vec = [w_data[i] for i in range(seq_batch_size)]

        chunk_start = start_end_seq_idx_ref[2]
        q_iota = jnp.arange(bq_sz, dtype=jnp.int32)
        k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(jnp.int32, (bq_sz, bkv_sz), 1)
        lane_id = (
            lax.broadcasted_iota(jnp.int32, (1, num_blk_sublanes_bkv, 128), 2)
            if produce_blk_scores
            else None
        )
        bq_pos_compressed_vec, q_l_vec, bkvs, bkv_scales = [], [], [], []
        for batch_idx in range(seq_batch_size):
            s_l = seq_lens_ref[batch_start_seq_idx + batch_idx]
            q_s = cu_q_lens_ref[batch_start_seq_idx + batch_idx]
            q_l = cu_q_lens_ref[batch_start_seq_idx + batch_idx + 1] - q_s
            q_l_vec.append(q_l)
            block_start = bq_idx * bq_sz + jnp.maximum(0, chunk_start - q_s)
            q_pos = s_l - q_l + block_start + q_iota
            bq_pos_compressed_vec.append(((q_pos + 1) >> shift_comp) - 1)

            key_bytes = bkv_vmem[batch_idx, :kv_head_dim_groups]
            bkvs.append(
                pltpu.bitcast(key_bytes.reshape(head_dim, bkv_sz), jnp.float8_e4m3fn)
            )
            scale_bytes = bkv_vmem[
                batch_idx, kv_head_dim_groups : kv_head_dim_groups + 1
            ].reshape(kv_packing, bkv_sz)[0:1]
            bkv_scales.append(
                pltpu.bitcast(scale_bytes, jnp.float8_e8m0fnu).astype(jnp.float32)
            )

        ret, ret_blk = [], []
        for batch_idx in range(seq_batch_size):
            bq = bq_vec[batch_idx].reshape(-1, head_dim)
            s_summed = (
                reduce_heads(
                    bq,
                    bkvs[batch_idx],
                    bq_weights_vec[batch_idx],
                    w_pre_folded=w_pre_folded,
                )
                * bkv_scales[batch_idx]
            )
            kv_len = seq_lens_ref[batch_start_seq_idx + batch_idx] >> shift_comp
            bq_pos_compressed = bq_pos_compressed_vec[batch_idx]
            mask = (k_span < kv_len) & (k_span <= bq_pos_compressed[:, None])
            s_summed = jnp.where(mask, s_summed, -jnp.inf)
            ret.append(s_summed.reshape(-1, num_sublanes_bkv, 128))

            if produce_blk_scores:
                newest_pos = jnp.minimum(kv_len - 1, bq_pos_compressed)[:, None]
                is_newest = (
                    (kv_len > 0) & (q_l_vec[batch_idx] > 0) & (k_span == newest_pos)
                )
                s_blk = (
                    jnp.where(is_newest, jnp.inf, s_summed)
                    .astype(jnp.bfloat16)
                    .reshape(-1, num_sublanes_bkv, 128)
                )
                nbs = num_blk_sublanes_bkv
                chunks = [s_blk[:, i * nbs : (i + 1) * nbs, :] for i in range(8)]
                for step in (4, 2, 1):
                    half = len(chunks) // 2
                    keep_first = (lane_id & step) == 0
                    chunks = [
                        jnp.maximum(
                            jnp.where(keep_first, chunks[i], chunks[i + half]),
                            jnp.where(
                                keep_first,
                                pltpu.roll(chunks[i], 128 - step, 2),
                                pltpu.roll(chunks[i + half], step, 2),
                            ),
                        )
                        for i in range(half)
                    ]
                ret_blk.append(chunks[0].astype(jnp.float32))

        scores = jnp.concatenate(ret, axis=0)
        blk_scores = jnp.concatenate(ret_blk, axis=0) if produce_blk_scores else None
        scores_vmem[...] = pltpu.bitcast(scores, jnp.int32)
        if blk_scores_vmem is not None:
            blk_scores_vmem[...] = pltpu.bitcast(blk_scores, jnp.int32)

    def _run_all(q_hm_ref=None):
        def _run_pipeline(final_allocs):
            window_refs = (cu_q_lens_ref, metadata_ref, start_end_seq_idx_ref)
            out_refs = (scores_hbm_ref, blk_scores_hbm_ref)[: len(final_allocs) - 3]
            pltpu.emit_pipeline(
                functools.partial(step_body, q_hm_ref),
                grid=(metadata_ref.num_steps[chunk_idx],),
                in_specs=tuple(b.spec for b in final_allocs[:3]),
                out_specs=tuple(b.spec for b in final_allocs[3:]),
            )(
                (q_hbm_ref, *window_refs),
                (indexer_weights_hbm_ref, *window_refs),
                (cache_kv_hbm_ref, page_indices_ref, metadata_ref),
                *((r, *window_refs) for r in out_refs),
                allocations=final_allocs,
            )

        pl.run_scoped(_run_pipeline, final_allocs=allocs)

    if seq_batch_size == 1 and bq_sz >= 8:
        pl.run_scoped(
            _run_all, pltpu.VMEM((num_q_heads * bq_sz, head_dim), q_hbm_ref.dtype)
        )
    else:
        _run_all()


MAX_INDEXER_CHUNKS = 128


def _smem_budget_bytes() -> int:
    try:
        return pltpu.get_tpu_info().smem_capacity_bytes - 64 * 1024
    except Exception:
        return 1024 * 1024 - 64 * 1024


def compute_meta_buf_len(
    max_num_seqs: int,
    pages_per_seq: int,
    max_num_tokens: int | None = None,
) -> int:
    """Static upper bound on `max_steps` across all 3 StreamIndex passes."""
    smem_budget_bytes = _smem_budget_bytes()
    eff_s = (
        min(max_num_seqs, max(8, align_to(max_num_tokens, 8)))
        if isinstance(max_num_tokens, int)
        else max_num_seqs
    )
    eff_pages = min(131072, eff_s * pages_per_seq)
    prefetch_bytes = eff_s * 4 + (eff_s + 1) * 4 + eff_pages * 4
    step_meta_bytes = (len(MetadataRef.__annotations__) - 1) * 4
    smem_max = max(
        1024, ((smem_budget_bytes - prefetch_bytes) // step_meta_bytes // 1024) * 1024
    )
    if not isinstance(max_num_tokens, int):
        return smem_max
    num_chunks, chunk_len, bq_mixed, bkv_p = _chunk_and_bq_config(max_num_tokens)
    eff_max_seqs = eff_s
    chunked = num_chunks > 1
    req_max = max(
        _max_steps_for_pass(
            max_num_seqs=eff_max_seqs,
            pages_per_seq=pages_per_seq,
            page_size=1,
            max_num_tokens=max_num_tokens,
            bq_sz=bq_sz,
            bkv_sz=bkv_p_r,
            static_q_len=static_q_len,
            seq_batch_size=seq_batch_size,
            num_chunks=num_chunks,
            chunked=chunked,
            chunk_tokens=chunk_len,
        )
        for bkv_p_r in (bkv_p, bkv_p * 2)
        for seq_batch_size, bq_sz, static_q_len in (
            (4, 1, 1),
            (1, 1, 1),
            (1, bq_mixed, None),
        )
    )
    return min(smem_max, req_max)


def _max_steps_for_pass(
    *,
    max_num_seqs: int,
    pages_per_seq: int,
    page_size: int,
    max_num_tokens: int,
    bq_sz: int,
    bkv_sz: int,
    static_q_len: int | None,
    seq_batch_size: int,
    num_chunks: int,
    chunked: bool,
    chunk_tokens: int,
) -> int:
    max_seq_tiles = pl.cdiv(max_num_seqs, seq_batch_size)
    if static_q_len == 1:
        total_max_bq = min(max_seq_tiles, pl.cdiv(max_num_tokens, seq_batch_size))
    else:
        total_max_bq = max(1, pl.cdiv(next_power_of_2(max_num_tokens), bq_sz)) + min(
            max_seq_tiles, max_num_tokens
        )

    max_kv_pages_per_block = bkv_sz // page_size
    max_bkv = max(1, pl.cdiv(pages_per_seq, max_kv_pages_per_block))

    if chunked:
        if static_q_len == 1:
            total_chunked_bq = total_max_bq + (num_chunks if seq_batch_size > 1 else 0)
        else:
            total_chunked_bq = (
                pl.cdiv(max_num_tokens, seq_batch_size * bq_sz)
                + min(max_seq_tiles, max_num_tokens)
                + num_chunks
            )
        per_chunk_bq = min(
            total_max_bq,
            pl.cdiv(chunk_tokens, seq_batch_size * bq_sz)
            + min(max_seq_tiles, chunk_tokens),
        )
        req_max_steps = min(num_chunks * per_chunk_bq, total_chunked_bq) * max_bkv
    else:
        req_max_steps = total_max_bq * max_bkv
    req_max_steps = pl.cdiv(req_max_steps, 1024) * 1024
    smem_max_steps = compute_meta_buf_len(max_num_seqs, pages_per_seq)
    return min(max(1024, req_max_steps), smem_max_steps)


def _compute_metadata(
    seq_lens: jax.Array,
    cu_q_lens: jax.Array,
    start_seq_idx: jax.Array | int,
    end_seq_idx: jax.Array | int,
    bq_sz: int,
    bkv_sz: int,
    pages_per_seq: int,
    page_size: int,
    max_num_tokens: int,
    compression_ratio: int,
    static_q_len: int | None,
    seq_batch_size: int,
    chunk_starts: tuple[int | None, ...],
    chunk_tokens: int,
) -> MetadataRef:
    """metadata.compute_batched_seq_metadata with the step count capped to SMEM."""
    max_num_seqs = seq_lens.shape[0]
    max_seq_tiles = pl.cdiv(max_num_seqs, seq_batch_size)
    num_chunks = len(chunk_starts)
    chunked = chunk_starts[0] is not None
    max_steps = _max_steps_for_pass(
        max_num_seqs=max_num_seqs,
        pages_per_seq=pages_per_seq,
        page_size=page_size,
        max_num_tokens=max_num_tokens,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        static_q_len=static_q_len,
        seq_batch_size=seq_batch_size,
        num_chunks=num_chunks,
        chunked=chunked,
        chunk_tokens=chunk_tokens,
    )

    def _empty_schedule():
        zeros = jnp.zeros((max_steps,), dtype=jnp.int32)
        return MetadataRef(jnp.zeros((num_chunks,), jnp.int32), zeros, zeros, zeros)

    def _compute_schedule():
        all_steps = jnp.arange(max_steps, dtype=jnp.int32)
        all_seq_tiles = jnp.arange(max_seq_tiles, dtype=jnp.int32)
        padded_len = max_seq_tiles * seq_batch_size
        padded_seq_lens = jnp.pad(seq_lens, (0, padded_len - max_num_seqs))
        starting_seq_tile_idx = all_seq_tiles * seq_batch_size
        is_valid_seq_tile = (starting_seq_tile_idx >= start_seq_idx) & (
            starting_seq_tile_idx < end_seq_idx
        )

        seq_indices = jnp.arange(padded_len, dtype=jnp.int32)
        valid_seq_lens = jnp.where(
            (seq_indices >= start_seq_idx) & (seq_indices < end_seq_idx),
            padded_seq_lens,
            0,
        )
        kv_lens_all = (valid_seq_lens // compression_ratio).reshape(
            max_seq_tiles, seq_batch_size
        )
        s_num_bkv = jnp.where(
            is_valid_seq_tile,
            jnp.maximum(1, pl.cdiv(jnp.max(kv_lens_all, axis=-1), bkv_sz)),
            0,
        ).astype(jnp.int32)

        if static_q_len is not None:
            s_num_bq = jnp.full(
                (max_seq_tiles,),
                jnp.maximum(1, pl.cdiv(static_q_len, bq_sz)),
                dtype=jnp.int32,
            )
        else:
            q_lens_all = (cu_q_lens[1:] - cu_q_lens[:-1]).reshape(
                max_seq_tiles, seq_batch_size
            )
            s_num_bq = jnp.maximum(
                1, pl.cdiv(jnp.max(q_lens_all, axis=-1), bq_sz)
            ).astype(jnp.int32)

        if chunked:
            bq_span = seq_batch_size * bq_sz
            tile_first_seq = jnp.minimum(starting_seq_tile_idx, max_num_seqs)
            tile_last_seq = jnp.minimum(
                starting_seq_tile_idx + seq_batch_size, max_num_seqs
            )
            tile_tok_start = cu_q_lens[tile_first_seq][None, :]
            tile_tok_end = cu_q_lens[tile_last_seq][None, :]
            chunk_start = jnp.asarray(chunk_starts, dtype=jnp.int32)[:, None]
            lo = jnp.clip(chunk_start, tile_tok_start, tile_tok_end)
            hi = jnp.clip(chunk_start + chunk_tokens, tile_tok_start, tile_tok_end)
            num_bq = pl.cdiv(hi - lo, bq_span).astype(jnp.int32)
        else:
            num_bq = s_num_bq[None, :]

        seg_num_steps_2d = num_bq * s_num_bkv[None, :]
        chunk_num_steps = jnp.sum(seg_num_steps_2d, axis=1)
        if num_chunks > 1 and num_chunks * max_seq_tiles >= 4096:
            # 2-level chunk -> tile lookup avoids materializing (num_chunks * max_seq_tiles, max_steps).
            chunk_ends = jnp.cumulative_sum(chunk_num_steps)
            valid_step = all_steps < chunk_ends[-1]
            c_le = chunk_ends[:, None] <= all_steps[None, :]
            c_start_step = jnp.max(jnp.where(c_le, chunk_ends[:, None], 0), axis=0)
            step_in_chunk = all_steps - c_start_step
            c_idx = jnp.minimum(jnp.sum(c_le, axis=0, dtype=jnp.int32), num_chunks - 1)
            chunk_hit = (
                jnp.arange(num_chunks, dtype=jnp.int32)[:, None] == c_idx[None, :]
            )
            tile_cum = jnp.cumulative_sum(seg_num_steps_2d, axis=1)
            lh = jnp.einsum(
                "ct,cs->ts",
                jnp.concatenate([tile_cum & 0xFF, tile_cum >> 8], axis=1).astype(
                    jnp.bfloat16
                ),
                chunk_hit.astype(jnp.bfloat16),
                preferred_element_type=jnp.float32,
            ).astype(jnp.int32)
            step_tile_ends = lh[:max_seq_tiles] | (lh[max_seq_tiles:] << 8)
            t_le = step_tile_ends <= step_in_chunk[None, :]
            seg_tile = jnp.minimum(
                jnp.sum(t_le, axis=0, dtype=jnp.int32), max_seq_tiles - 1
            )
            tile_start_in_chunk = jnp.max(jnp.where(t_le, step_tile_ends, 0), axis=0)
            start_step_id_repeat = jnp.where(
                valid_step, c_start_step + tile_start_in_chunk, 0
            )
            tile_step_id = all_steps - start_step_id_repeat
        else:
            seg_num_steps = seg_num_steps_2d.reshape(-1)
            seg_start_step_id = jnp.cumulative_sum(seg_num_steps, include_initial=True)
            valid_step = all_steps < seg_start_step_id[-1]
            # Gather-free step -> segment lookup. seg_start_step_id is non-decreasing,
            # so counting segment ends <= step equals searchsorted(side="right"); the
            # default scan method ran ~log2(num_segs) serialized max_steps-wide gathers.
            ends = seg_start_step_id[1:, None]
            le = ends <= all_steps[None, :]
            seg_idx = jnp.minimum(
                jnp.sum(le, axis=0, dtype=jnp.int32), seg_num_steps.shape[0] - 1
            )
            seg_tile = seg_idx % max_seq_tiles  # segments are chunk-major over tiles
            start_step_id_repeat = jnp.where(
                valid_step, jnp.max(jnp.where(le, ends, 0), axis=0), 0
            )
            tile_step_id = all_steps - start_step_id_repeat

        repeat_seq_tiles = jnp.where(valid_step, seg_tile, 0)
        batch_tile_idx = repeat_seq_tiles * seq_batch_size
        tile_hit = all_seq_tiles[:, None] == seg_tile[None, :]
        nbkv_per_step_id = jnp.where(
            valid_step, jnp.sum(jnp.where(tile_hit, s_num_bkv[:, None], 0), axis=0), 0
        )

        return MetadataRef.create(
            num_steps=chunk_num_steps,
            batch_tile_idx=batch_tile_idx,
            bq_idx=jnp.maximum(0, tile_step_id // jnp.maximum(1, nbkv_per_step_id)),
            bkv_idx=jnp.maximum(0, tile_step_id % jnp.maximum(1, nbkv_per_step_id)),
        )

    return jax.lax.cond(
        jnp.greater_equal(start_seq_idx, end_seq_idx),
        _empty_schedule,
        _compute_schedule,
    )


def _chunk_and_bq_config(max_num_tokens: int) -> tuple[int, int, int, int]:
    is_decode_or_verify = max_num_tokens <= 1536
    bq_mixed_in = 8 if is_decode_or_verify else 128
    bkv_p = 16 if is_decode_or_verify else 8
    chunk_tokens = (
        512
        if (
            not is_decode_or_verify
            and max_num_tokens >= 512
            and max_num_tokens % 512 == 0
        )
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
        }.get(max_num_tokens, 512)
    )
    if chunk_tokens is None or max_num_tokens % chunk_tokens:
        num_chunks, chunk_len = 1, max_num_tokens
    else:
        num_chunks, chunk_len = max_num_tokens // chunk_tokens, chunk_tokens
    return num_chunks, chunk_len, min(bq_mixed_in, chunk_len), bkv_p


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "page_size_r2",
        "page_size_r1",
        "block_stride",
        "max_chunks",
        "meta_buf_len",
    ),
)
def compute_indexer_step_metadata_jax(
    positions: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    page_size_r2: int,
    page_size_r1: int,
    block_stride: int,
    max_chunks: int = MAX_INDEXER_CHUNKS,
    meta_buf_len: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Computes step-invariant Indexer metadata once per step for all Leader layers."""
    max_num_tokens = positions.shape[0]
    max_num_seqs = seq_lens.shape[0]
    idx_page_indices = (
        page_indices.reshape(max_num_seqs, -1)[:, ::block_stride].reshape(-1)
        if block_stride > 1
        else page_indices.reshape(-1)
    )
    pages_per_seq = idx_page_indices.shape[0] // max_num_seqs
    eff_meta_buf_len = (
        compute_meta_buf_len(max_num_seqs, pages_per_seq, max_num_tokens)
        if meta_buf_len is None
        else meta_buf_len
    )

    num_chunks, chunk_len, bq_mixed, bkv_p = _chunk_and_bq_config(max_num_tokens)

    eff_max_seqs = min(max_num_seqs, max(8, align_to(max_num_tokens, 8)))
    if eff_max_seqs < max_num_seqs:
        seq_lens = seq_lens[:eff_max_seqs]
        cu_q_lens = cu_q_lens[: eff_max_seqs + 1]
        distribution = jnp.minimum(distribution, eff_max_seqs)
    num_seqs = distribution[2]
    seq_lens = jnp.where(
        jnp.arange(eff_max_seqs, dtype=jnp.int32) < num_seqs, seq_lens, 0
    )
    token_ids = jnp.arange(max_num_tokens, dtype=jnp.int32)
    seq_mask = token_ids[:, None] >= cu_q_lens[None, 1 : eff_max_seqs + 1]
    seq_mask = jnp.where(jnp.arange(eff_max_seqs)[None, :] < num_seqs, seq_mask, False)
    seq_ids = jnp.sum(seq_mask, axis=1)
    q_start = cu_q_lens[seq_ids]
    q_len = cu_q_lens[seq_ids + 1] - q_start
    seq_len = seq_lens[seq_ids]
    token_pos = seq_len - q_len + (token_ids - q_start)
    valid_tok = token_ids < cu_q_lens[num_seqs]

    def _eff_rl(comp_ratio: int, page_size: int) -> jax.Array:
        bkv_p_c = bkv_p * max(1, comp_ratio)
        num_sublanes_total = align_to(pages_per_seq, bkv_p_c) * page_size // 128
        visible = jnp.minimum(seq_len // comp_ratio, (token_pos + 1) // comp_ratio)
        return jnp.where(
            valid_tok,
            jnp.minimum(visible, max(num_sublanes_total * 128, k)),
            0,
        ).astype(jnp.int32)

    eff_row_lengths_r2 = _eff_rl(2, page_size_r2)
    eff_row_lengths_r1 = _eff_rl(1, page_size_r1)

    decode_req_batch_size = 4
    decode_batch_end = distribution[0] // decode_req_batch_size * decode_req_batch_size
    pass_specs = (
        (jnp.array(0), decode_batch_end, decode_req_batch_size, 1, MlaCase.DECODE),
        (decode_batch_end, distribution[1], 1, 1, MlaCase.DECODE),
        (distribution[1], distribution[2], 1, bq_mixed, MlaCase.MIXED),
    )
    chunk_starts = (
        (None,) if num_chunks == 1 else tuple(m * chunk_len for m in range(num_chunks))
    )

    num_steps_rows = []
    sched_rows = []
    for comp_r, ps_r, bkv_p_r in (
        (1, page_size_r1, bkv_p),
        (2, page_size_r2, bkv_p * 2),
    ):
        for spec in pass_specs:
            pm = _compute_metadata(
                seq_lens=seq_lens,
                cu_q_lens=cu_q_lens,
                start_seq_idx=spec[0],
                end_seq_idx=spec[1],
                bq_sz=spec[3],
                bkv_sz=bkv_p_r * ps_r,
                pages_per_seq=pages_per_seq,
                page_size=ps_r,
                max_num_tokens=max_num_tokens,
                compression_ratio=comp_r,
                static_q_len=1 if spec[4] is MlaCase.DECODE else None,
                seq_batch_size=spec[2],
                chunk_starts=chunk_starts,
                chunk_tokens=chunk_len,
            )
            ns = pm.num_steps
            if ns.shape[0] < max_chunks:
                ns = jnp.pad(ns, (0, max_chunks - ns.shape[0]))
            else:
                ns = ns[:max_chunks]
            num_steps_rows.append(ns)
            for arr in (pm.batch_tile_idx, pm.bq_idx, pm.bkv_idx):
                if arr.shape[0] < eff_meta_buf_len:
                    arr = jnp.pad(arr, (0, eff_meta_buf_len - arr.shape[0]))
                else:
                    arr = arr[:eff_meta_buf_len]
                sched_rows.append(arr)

    pass_num_steps = jnp.stack(num_steps_rows, axis=0)
    pass_schedules = jnp.stack(sched_rows, axis=0)
    return (
        eff_row_lengths_r2,
        eff_row_lengths_r1,
        idx_page_indices,
        pass_num_steps,
        pass_schedules,
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "k",
        "compression_ratio",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "chunk_tokens",
        "candidate_topk_blocks",
    ),
)
def streamindex_topk(
    q: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    compression_ratio: int,
    num_kv_pages_per_block: int,
    num_queries_per_block: int,
    chunk_tokens: int | None = None,
    candidate_topk_blocks: int = 0,
    candidates: jax.Array | None = None,
    precomputed_eff_row_lengths: jax.Array | None = None,
    precomputed_pass_num_steps: jax.Array | None = None,
    precomputed_pass_schedules: jax.Array | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """StreamIndex top-k for the DeepSeek-V4.1 indexer (SEQ_ALONG_LANE layout)."""
    max_num_seqs = seq_lens.shape[0]
    prepared_indexer_weights = prepare_index_weights(indexer_weights, q.dtype)
    q = prepare_q_inputs(q)
    _, _, _, page_size = cache_kv.shape
    pages_per_seq = page_indices.shape[0] // max_num_seqs

    bkv_p = num_kv_pages_per_block
    num_sublanes_total = align_to(pages_per_seq, bkv_p) * page_size // 128
    num_blk_sublanes_total = num_sublanes_total // 8
    produce_blk = candidate_topk_blocks > 0
    max_num_tokens = q.shape[0]
    eff_max_seqs = min(max_num_seqs, max(8, align_to(max_num_tokens, 8)))
    if eff_max_seqs < max_num_seqs:
        seq_lens = seq_lens[:eff_max_seqs]
        page_indices = page_indices[: eff_max_seqs * pages_per_seq]
        cu_q_lens = cu_q_lens[: eff_max_seqs + 1]
        distribution = jnp.minimum(distribution, eff_max_seqs)
        max_num_seqs = eff_max_seqs
    seq_lens = jnp.where(
        jnp.arange(max_num_seqs, dtype=jnp.int32) < distribution[2], seq_lens, 0
    )
    if chunk_tokens is None or max_num_tokens % chunk_tokens:
        num_chunks, chunk_len = 1, max_num_tokens
    else:
        num_chunks, chunk_len = max_num_tokens // chunk_tokens, chunk_tokens

    def _run_scores_slice(
        sub_seq_lens,
        sub_page_indices,
        sub_cu_q_lens,
        sub_start_seq,
        sub_end_seq,
        seq_batch_size,
        bq_sz,
        case,
        pass_meta,
        chunk_idx,
        scheduling_group_id,
        state,
    ):
        hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)
        num_outs = 2 if produce_blk else 1
        extra_args = () if state[0] is None else state[:num_outs]
        in_specs = [hbm_spec] * (3 + len(extra_args))
        out_specs = (hbm_spec, hbm_spec) if produce_blk else hbm_spec

        scalar_prefetches = (
            sub_seq_lens,
            sub_page_indices,
            sub_cu_q_lens,
            jnp.array([sub_start_seq, sub_end_seq, chunk_idx * chunk_len], jnp.int32),
            pass_meta,
        )
        num_flat_prefetches = len(jax.tree_util.tree_leaves(scalar_prefetches))
        input_output_aliases = {
            num_flat_prefetches + 3 + i: i for i in range(len(extra_args))
        }

        blk_tag = "-blk" if produce_blk else ""
        scope_name = f"StreamIdxTC-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}-sal{blk_tag}"
        if num_chunks > 1:
            scope_name += f"-c{num_chunks}"
        out_shapes = tuple(
            jax.ShapeDtypeStruct((chunk_len, n, 128), jnp.int32)
            for n in (num_sublanes_total, num_blk_sublanes_total)[:num_outs]
        )
        res = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    kernel,
                    compression_ratio=compression_ratio,
                    bq_sz=bq_sz,
                    bkv_p=bkv_p,
                    seq_batch_size=seq_batch_size,
                    buffer_count=DEFAULT_BUFFER_COUNT[case.value],
                    chunk_idx=chunk_idx,
                    produce_blk_scores=produce_blk,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=(1,),
                    scratch_shapes=[],
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
                    disable_bounds_checks=True,
                ),
                out_shape=out_shapes if produce_blk else out_shapes[0],
                input_output_aliases=input_output_aliases,
                name=scope_name,
            )
        )(*scalar_prefetches, q, prepared_indexer_weights, cache_kv, *extra_args)
        if scheduling_group_id is not None:
            res = xla_metadata.set_xla_metadata(
                res, _scheduling_group_id=scheduling_group_id
            )
        return res if produce_blk else (res, None)

    bq_mixed = min(num_queries_per_block, chunk_len)
    decode_req_batch_size = 4
    decode_batch_end = distribution[0] // decode_req_batch_size * decode_req_batch_size
    # Rows: (start_seq, end_seq, seq_batch_size, bq_sz, case).
    pass_specs = (
        (jnp.array(0), decode_batch_end, decode_req_batch_size, 1, MlaCase.DECODE),
        (decode_batch_end, distribution[1], 1, 1, MlaCase.DECODE),
        (distribution[1], distribution[2], 1, bq_mixed, MlaCase.MIXED),
    )

    # Visible compressed KV positions per token row of the score matrix.
    num_seqs_all = distribution[2]
    token_ids_all = jnp.arange(max_num_tokens, dtype=jnp.int32)
    seq_mask_all = token_ids_all[:, None] >= cu_q_lens[None, 1 : max_num_seqs + 1]
    seq_mask_all = jnp.where(
        jnp.arange(max_num_seqs)[None, :] < num_seqs_all, seq_mask_all, False
    )
    seq_ids_all = jnp.sum(seq_mask_all, axis=1)
    if precomputed_eff_row_lengths is not None:
        eff_row_lengths = precomputed_eff_row_lengths
    else:
        q_start = cu_q_lens[seq_ids_all]
        q_len = cu_q_lens[seq_ids_all + 1] - q_start
        seq_len = seq_lens[seq_ids_all]
        token_pos = seq_len - q_len + (token_ids_all - q_start)
        visible = jnp.minimum(
            seq_len // compression_ratio, (token_pos + 1) // compression_ratio
        )
        eff_row_lengths = jnp.where(
            token_ids_all < cu_q_lens[num_seqs_all],
            jnp.minimum(visible, max(num_sublanes_total * 128, k)),
            0,
        )

    cand_rank_map = None
    cand_p_pad = 0
    blk_per_page = page_size // 8
    if candidates is not None and not produce_blk and pages_per_seq > 160:
        page_indices, cand_rank_map, n_active_pages, cand_p_pad = (
            sc_build_cand_page_table(
                candidates,
                page_indices,
                cu_q_lens,
                num_seqs_all,
                max_num_seqs=max_num_seqs,
                pages_per_seq=pages_per_seq,
                blk_per_page=blk_per_page,
            )
        )
        seq_q_lens = cu_q_lens[1 : max_num_seqs + 1] - cu_q_lens[:max_num_seqs]
        seq_lens = jnp.where(
            jnp.arange(max_num_seqs, dtype=jnp.int32) < num_seqs_all,
            n_active_pages * page_size * compression_ratio
            + jnp.maximum(seq_q_lens - 1, 0) * compression_ratio,
            0,
        )
        precomputed_pass_num_steps = None
        precomputed_pass_schedules = None

    # See VLLM_DSV41_STREAMIDX_SCHED_GROUPS: without groups every pass below takes
    # its unannotated branch, exactly as a single-chunk call does.
    use_groups = num_chunks > 1 and bool(envs.VLLM_DSV41_STREAMIDX_SCHED_GROUPS)
    group_base = reserve_scheduling_group_ids(num_chunks - 1) if use_groups else 0
    chunk_starts = (
        (None,) if num_chunks == 1 else tuple(m * chunk_len for m in range(num_chunks))
    )
    chunked_flag = chunk_starts[0] is not None
    need_smem_split = max_num_seqs * pages_per_seq > 131072
    if need_smem_split:
        max_seqs_per_slice = max(64, (131072 // max(1, pages_per_seq) // 64) * 64)
        passes = []
        for s_off in range(0, max_num_seqs, max_seqs_per_slice):
            s_end = min(s_off + max_seqs_per_slice, max_num_seqs)
            sub_sl = seq_lens[s_off:s_end]
            sub_pi = page_indices[s_off * pages_per_seq : s_end * pages_per_seq]
            sub_cq = cu_q_lens[s_off : s_end + 1]
            for spec in pass_specs:
                sub_start = jnp.clip(spec[0] - s_off, 0, s_end - s_off)
                sub_end = jnp.clip(spec[1] - s_off, 0, s_end - s_off)
                sub_meta = _compute_metadata(
                    seq_lens=sub_sl,
                    cu_q_lens=sub_cq,
                    start_seq_idx=sub_start,
                    end_seq_idx=sub_end,
                    bq_sz=spec[3],
                    bkv_sz=bkv_p * page_size,
                    pages_per_seq=pages_per_seq,
                    page_size=page_size,
                    max_num_tokens=max_num_tokens,
                    compression_ratio=compression_ratio,
                    static_q_len=1 if spec[4] is MlaCase.DECODE else None,
                    seq_batch_size=spec[2],
                    chunk_starts=chunk_starts,
                    chunk_tokens=chunk_len,
                )
                passes.append(
                    (
                        (
                            sub_sl,
                            sub_pi,
                            sub_cq,
                            sub_start,
                            sub_end,
                            spec[2],
                            spec[3],
                            spec[4],
                        ),
                        sub_meta,
                    )
                )
        passes = tuple(passes)
    elif (
        precomputed_pass_num_steps is not None
        and precomputed_pass_schedules is not None
    ):
        r_off = (
            3
            if compression_ratio == 2 and precomputed_pass_num_steps.shape[0] >= 6
            else 0
        )
        passes = []
        for i, spec in enumerate(pass_specs):
            max_steps_i = min(
                _max_steps_for_pass(
                    max_num_seqs=max_num_seqs,
                    pages_per_seq=pages_per_seq,
                    page_size=page_size,
                    max_num_tokens=max_num_tokens,
                    bq_sz=spec[3],
                    bkv_sz=bkv_p * page_size,
                    static_q_len=1 if spec[4] is MlaCase.DECODE else None,
                    seq_batch_size=spec[2],
                    num_chunks=num_chunks,
                    chunked=chunked_flag,
                    chunk_tokens=chunk_len,
                ),
                precomputed_pass_schedules.shape[1],
            )
            idx = r_off + i
            passes.append(
                (
                    (
                        seq_lens,
                        page_indices,
                        cu_q_lens,
                        spec[0],
                        spec[1],
                        spec[2],
                        spec[3],
                        spec[4],
                    ),
                    MetadataRef.create(
                        num_steps=precomputed_pass_num_steps[idx, :num_chunks],
                        batch_tile_idx=precomputed_pass_schedules[
                            3 * idx, :max_steps_i
                        ],
                        bq_idx=precomputed_pass_schedules[3 * idx + 1, :max_steps_i],
                        bkv_idx=precomputed_pass_schedules[3 * idx + 2, :max_steps_i],
                    ),
                )
            )
        passes = tuple(passes)
    else:
        passes = tuple(
            (
                (
                    seq_lens,
                    page_indices,
                    cu_q_lens,
                    spec[0],
                    spec[1],
                    spec[2],
                    spec[3],
                    spec[4],
                ),
                _compute_metadata(
                    seq_lens=seq_lens,
                    cu_q_lens=cu_q_lens,
                    start_seq_idx=spec[0],
                    end_seq_idx=spec[1],
                    bq_sz=spec[3],
                    bkv_sz=bkv_p * page_size,
                    pages_per_seq=pages_per_seq,
                    page_size=page_size,
                    max_num_tokens=max_num_tokens,
                    compression_ratio=compression_ratio,
                    static_q_len=1 if spec[4] is MlaCase.DECODE else None,
                    seq_batch_size=spec[2],
                    chunk_starts=chunk_starts,
                    chunk_tokens=chunk_len,
                ),
            )
            for spec in pass_specs
        )

    topk_idxs, cand_list = [], []
    for m in range(num_chunks):
        tc_group = None if (m == 0 or not use_groups) else group_base + m - 1
        sc_group = (
            None if (m == num_chunks - 1 or not use_groups) else group_base + m
        )
        sc2_group = (
            None if (m >= num_chunks - 2 or not use_groups) else group_base + m + 1
        )
        start = m * chunk_len

        ordered_passes = (
            (*passes[2:], *passes[:2])
            if not need_smem_split and len(passes) == 3
            else passes
        )
        state = None
        for slice_spec, pass_meta in ordered_passes:
            if (
                state is not None
                and not need_smem_split
                and slice_spec[7] is MlaCase.DECODE
                and start >= max_num_seqs
            ):
                continue
            run = functools.partial(
                _run_scores_slice, *slice_spec, pass_meta, m, tc_group
            )
            if state is None:
                state = run((None, None))
            elif not need_smem_split and slice_spec[7] is MlaCase.DECODE:
                pred = slice_spec[3] < slice_spec[4]
                if start > 0:
                    pred = pred & (slice_spec[4] > start)
                if start + chunk_len < max_num_seqs:
                    pred = pred & (slice_spec[3] < start + chunk_len)
                state = lax.cond(pred, run, lambda s: s, state)
            else:
                state = lax.cond(slice_spec[3] < slice_spec[4], run, lambda s: s, state)
        assert state is not None
        scores, blk_scores = state
        assert scores is not None
        scores = scores.reshape(chunk_len, -1)

        rl_m = eff_row_lengths[start : start + chunk_len]
        if produce_blk:
            assert blk_scores is not None
            num_blk_sub_bkv = bkv_p * page_size // 1024
            chunk_topk_idxs, cands_m = sc_write_cand_and_topk(
                blk_scores.reshape(chunk_len, -1),
                scores,
                rl_m,
                cb=candidate_topk_blocks,
                k=k,
                num_blk_sub_bkv=num_blk_sub_bkv,
                scheduling_group_id=sc_group,
            )
            cand_list.append(cands_m)
        elif candidates is not None:
            chunk_topk_idxs = sc_cand_gather_topk(
                scores,
                candidates[start : start + chunk_len],
                rl_m,
                k=k,
                scheduling_group_id=sc_group,
                rank_map=cand_rank_map,
                seq_ids=seq_ids_all[start : start + chunk_len],
                p_pad=cand_p_pad,
                blk_per_page=blk_per_page,
                page_size=page_size,
            )
        else:
            chunk_topk_idxs = sparsecore_topk(
                scores,
                k,
                row_lengths=rl_m,
                scheduling_group_id=sc_group,
                stage2_scheduling_group_id=sc2_group,
            )
        topk_idxs.append(chunk_topk_idxs)

    out = jnp.concatenate(topk_idxs)[:max_num_tokens, :k]
    if produce_blk:
        return out, jnp.concatenate(cand_list)[:max_num_tokens]
    return out
