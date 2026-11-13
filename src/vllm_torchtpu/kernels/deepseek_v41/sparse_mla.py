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
"""CSA sparse MLA for DeepSeek-V4.1: LxL block-grouped candidate union for prefill
and double-buffered pipelined decode on pre-encoded physical cache rows."""

import functools
from vllm_torchtpu import envs

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather
from vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla import (
    DEFAULT_VMEM_LIMIT_BYTES,
    _attention_kernel,
    _dequant_and_merge_bkv,
    _dequant_dsv4_fp8,
    _largest_divisor,
    align_to,
    cdiv,
    gather_page_ids,
    prepare_outputs,
    prepare_q_inputs,
    prepare_swa_inputs,
)

_INT32_MAX = 2147483647


_ENCODE_PAGE_CHUNK = 128


def _fused_encode_csa_kernel(
    token_start_ref,  # i32[1] scalar prefetch
    act_chunks_ref,  # i32[1] scalar prefetch
    windows_ref,  # i32[block_tokens, padded_pps]
    topk_ref,  # i32[block_tokens, topk]
    out_ref,  # i32[block_tokens, topk]
    *,
    num_chunks: int,
    pages_per_seq: int,
    total_num_pages: int,
    page_size: int,
    use_dynamic_chunks: bool = False,
):
    tk = topk_ref[...]
    is_padding = tk < 0
    if page_size & (page_size - 1) == 0:
        ps_shift = page_size.bit_length() - 1
        logical = jnp.clip(jnp.right_shift(tk, ps_shift), 0, pages_per_seq - 1)
        token_offset = jnp.bitwise_and(tk, page_size - 1)
    else:
        logical = jnp.clip(tk // page_size, 0, pages_per_seq - 1)
        token_offset = tk % page_size

    idx0 = jnp.clip(logical, 0, _ENCODE_PAGE_CHUNK - 1)
    page_ids = jnp.take_along_axis(windows_ref[:, :_ENCODE_PAGE_CHUNK], idx0, axis=1)
    if use_dynamic_chunks:
        act_chunks = act_chunks_ref[0]

        def _chunk_body(c, acc):
            window_chunk = windows_ref[
                :, pl.ds(c * _ENCODE_PAGE_CHUNK, _ENCODE_PAGE_CHUNK)
            ]
            local = logical - c * _ENCODE_PAGE_CHUNK
            gathered = jnp.take_along_axis(
                window_chunk, jnp.clip(local, 0, _ENCODE_PAGE_CHUNK - 1), axis=1
            )
            return jnp.where((local >= 0) & (local < _ENCODE_PAGE_CHUNK), gathered, acc)

        page_ids = lax.fori_loop(1, act_chunks, _chunk_body, page_ids)
    else:
        for c in range(1, num_chunks):
            window_chunk = windows_ref[
                :, c * _ENCODE_PAGE_CHUNK : (c + 1) * _ENCODE_PAGE_CHUNK
            ]
            local = logical - c * _ENCODE_PAGE_CHUNK
            gathered = jnp.take_along_axis(
                window_chunk, jnp.clip(local, 0, _ENCODE_PAGE_CHUNK - 1), axis=1
            )
            page_ids = jnp.where(
                (local >= 0) & (local < _ENCODE_PAGE_CHUNK), gathered, page_ids
            )

    block_tokens, topk = tk.shape
    row_start = token_start_ref[0] + pl.program_id(0) * block_tokens
    rows = row_start + lax.broadcasted_iota(jnp.int32, (block_tokens, 1), 0)
    cols = lax.broadcasted_iota(jnp.int32, (1, topk), 1)
    flat_idx = rows * topk + cols

    if total_num_pages & (total_num_pages - 1) == 0:
        scattered_page_ids = jnp.bitwise_and(flat_idx * 104729, total_num_pages - 1)
    else:
        rem_p = lax.rem(flat_idx * 104729, total_num_pages)
        scattered_page_ids = jnp.where(rem_p < 0, rem_p + total_num_pages, rem_p)
    if page_size & (page_size - 1) == 0:
        scattered_token_offset = jnp.bitwise_and(flat_idx * 15485863, page_size - 1)
        valid_raw = jnp.left_shift(page_ids, ps_shift) + token_offset
        pad_raw = ~(
            jnp.left_shift(scattered_page_ids, ps_shift) + scattered_token_offset
        )
    else:
        rem_t = lax.rem(flat_idx * 15485863, page_size)
        scattered_token_offset = jnp.where(rem_t < 0, rem_t + page_size, rem_t)
        valid_raw = page_ids * page_size + token_offset
        pad_raw = ~(scattered_page_ids * page_size + scattered_token_offset)
    out_ref[...] = jnp.where(is_padding, pad_raw, valid_raw)


def encode_csa_gather_indices(
    topk_indices: jax.Array,  # i32[num_tokens, topk]
    page_indices: jax.Array,  # i32[max_num_seqs * pages_per_seq]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    total_num_pages: int,
    page_size: int,
    token_start: jax.Array | int = 0,
) -> jax.Array:
    """Maps CSA top-k token indices to physical cache rows; padding (-1) gets a
    scattered row, stored bitwise inverted so it stays negative."""
    num_tokens, topk = topk_indices.shape
    max_num_seqs = cu_q_lens.shape[0] - 1
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    eff_max_seqs = min(max_num_seqs, max(8, align_to(num_tokens, 8)))
    if eff_max_seqs < max_num_seqs:
        page_indices = page_indices[: eff_max_seqs * pages_per_seq]
        cu_q_lens = cu_q_lens[: eff_max_seqs + 1]
        max_num_seqs = eff_max_seqs

    if num_tokens <= 1024:
        token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
        seq_ids_segment = jnp.sum(
            token_ids[:, None] >= cu_q_lens[None, 1:max_num_seqs],
            axis=1,
            dtype=jnp.int32,
        )
    elif jax.default_backend() != "tpu":
        tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
        seq_ids_segment = jnp.repeat(
            jnp.arange(max_num_seqs, dtype=jnp.int32),
            tokens_per_seq,
            total_repeat_length=num_tokens,
        )
    else:
        token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
        cmp_bf16 = (token_ids[:, None] >= cu_q_lens[None, 1:max_num_seqs]).astype(
            jnp.bfloat16
        )
        ones_bf16 = jnp.ones((max_num_seqs - 1,), dtype=jnp.bfloat16)
        seq_ids_segment = jnp.einsum(
            "ts,s->t",
            cmp_bf16,
            ones_bf16,
            preferred_element_type=jnp.float32,
        ).astype(jnp.int32)

    if jax.default_backend() == "tpu" and topk % 128 == 0:
        num_chunks = cdiv(pages_per_seq, _ENCODE_PAGE_CHUNK)
        padded_pps = num_chunks * _ENCODE_PAGE_CHUNK
        page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
        if padded_pps != pages_per_seq:
            page_table = jnp.pad(page_table, ((0, 0), (0, padded_pps - pages_per_seq)))
        use_dynamic_chunks = num_chunks > 8 and num_tokens >= 128
        if use_dynamic_chunks:
            max_tk = jnp.max(topk_indices)
            act_chunks = jnp.clip(
                (max_tk // (page_size * _ENCODE_PAGE_CHUNK)) + 1, 1, num_chunks
            )
            act_chunks_arr = jnp.asarray([act_chunks], dtype=jnp.int32)
        else:
            act_chunks_arr = jnp.asarray([num_chunks], dtype=jnp.int32)

        eff_block_tokens = min(128, max(8, align_to(num_tokens, 8)))
        padded_tokens = align_to(num_tokens, eff_block_tokens)
        tk_in = topk_indices
        seq_seg = seq_ids_segment
        if padded_tokens != num_tokens:
            pad = padded_tokens - num_tokens
            seq_seg = jnp.pad(seq_seg, (0, pad))
            tk_in = jnp.pad(tk_in, ((0, pad), (0, 0)), constant_values=-1)

        max_chunk_tokens = max(
            eff_block_tokens,
            align_to(min(padded_tokens, (512 * 8192) // padded_pps), eff_block_tokens),
        )
        out_chunks = []
        for c_start in range(0, padded_tokens, max_chunk_tokens):
            c_end = min(c_start + max_chunk_tokens, padded_tokens)
            c_len = c_end - c_start
            windows_chunk = page_table[seq_seg[c_start:c_end]]
            tok_start_arr = jnp.asarray([token_start + c_start], dtype=jnp.int32)
            out_c = pl.pallas_call(
                functools.partial(
                    _fused_encode_csa_kernel,
                    num_chunks=num_chunks,
                    pages_per_seq=pages_per_seq,
                    total_num_pages=total_num_pages,
                    page_size=page_size,
                    use_dynamic_chunks=use_dynamic_chunks,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=2,
                    in_specs=[
                        pl.BlockSpec(
                            (eff_block_tokens, padded_pps), lambda t, *_: (t, 0)
                        ),
                        pl.BlockSpec((eff_block_tokens, topk), lambda t, *_: (t, 0)),
                    ],
                    out_specs=pl.BlockSpec(
                        (eff_block_tokens, topk), lambda t, *_: (t, 0)
                    ),
                    grid=(c_len // eff_block_tokens,),
                ),
                out_shape=jax.ShapeDtypeStruct((c_len, topk), jnp.int32),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=(
                        ("parallel",) if c_len > eff_block_tokens else ("arbitrary",)
                    ),
                    disable_bounds_checks=True,
                ),
                name="fused_encode_csa_gather_indices",
            )(
                tok_start_arr,
                act_chunks_arr,
                windows_chunk,
                tk_in[c_start:c_end],
            )
            out_chunks.append(out_c)
        out = (
            out_chunks[0]
            if len(out_chunks) == 1
            else jnp.concatenate(out_chunks, axis=0)
        )
        return out[:num_tokens]

    seq_page_ids = topk_indices // page_size
    token_offset = topk_indices % page_size
    page_ids = gather_page_ids(
        page_indices, seq_page_ids, seq_ids_segment, max_num_seqs
    )
    is_padding = topk_indices == -1
    flat_element_index = (
        jnp.arange(num_tokens * topk, dtype=jnp.int32)
        + jnp.asarray(token_start, dtype=jnp.int32) * topk
    ).reshape(num_tokens, topk)
    # Large primes spread the padding rows across the cache.
    scattered_page_ids = (flat_element_index * 104729) % total_num_pages
    scattered_token_offset = (flat_element_index * 15485863) % page_size
    page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
    token_offset = jnp.where(is_padding, scattered_token_offset, token_offset)
    raw_gather_indices = page_ids * page_size + token_offset
    return jnp.where(is_padding, ~raw_gather_indices, raw_gather_indices)


def _build_candidate_union_and_bitmask(
    topk_indices: jax.Array,  # i32[num_tokens, topk]
    b_q: int,
    m_union: int,
    total_cache_rows: int,
    u_chunk: int | None = 512,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes per-tile unique sorted candidate union [num_groups * m_union] and
    bit-packed 32-query membership bitmask [num_groups, m_union // 128, 128].
    """
    num_tokens, topk = topk_indices.shape
    if b_q == 1 and m_union == topk:
        return (
            topk_indices.reshape(-1),
            jnp.ones((num_tokens, m_union // 128, 128), dtype=jnp.int32),
            jnp.full((num_tokens,), m_union, dtype=jnp.int32),
        )
    num_groups = num_tokens // b_q
    n_cols = b_q * topk
    if (
        jax.default_backend() == "tpu"
        and b_q <= 32
        and num_groups >= 32
        and topk % 16 == 0
        and m_union % 16 == 0
        and m_union <= 2048
        and max(5, (total_cache_rows - 1).bit_length() - 14)
        <= int(
            envs.VLLM_DSV41_SC_UNION_MAX_BLK_SHIFT
            if envs.VLLM_DSV41_SC_UNION_MAX_BLK_SHIFT is not None
            else envs.VLLM_DSV41_SC_UNION_MAX_SHIFT
        )
        and hasattr(pltpu.get_tpu_info(), "sparse_core")
    ):
        from jax.experimental.pallas import tpu_sc as plsc

        sc = pltpu.get_tpu_info().sparse_core
        lanes = 16
        eff_subcores = sc.num_subcores
        cores_per_chip = sc.num_cores * eff_subcores
        num_waves = (num_groups + cores_per_chip - 1) // cores_per_chip
        padded_g = num_waves * cores_per_chip

        grouped = topk_indices.reshape(num_groups, n_cols)
        grouped_pad = (
            jnp.pad(grouped, ((0, padded_g - num_groups), (0, 0)), constant_values=-1)
            if padded_g > num_groups
            else grouped
        )

        win_size = (
            min(align_to(max(total_cache_rows, 256), 256), 32768)
            if total_cache_rows <= 32768
            else (
                32768
                if total_cache_rows
                <= (
                    1
                    << (
                        14
                        + int(
                            envs.VLLM_DSV41_SC_UNION_MAX_BLK_SHIFT
                            if envs.VLLM_DSV41_SC_UNION_MAX_BLK_SHIFT is not None
                            else envs.VLLM_DSV41_SC_UNION_MAX_SHIFT
                        )
                    )
                )
                and m_union <= 2048
                else 16384
            )
        )
        num_wins = cdiv(total_cache_rows, win_size)
        blk_shift = max(5, (total_cache_rows - 1).bit_length() - 14)
        blk_size = 1 << blk_shift
        vecs_per_blk = blk_size // lanes
        max_act_blks = win_size // blk_size
        max_total_blks = min(align_to(max(m_union, n_cols // 8), lanes), 2048)
        num_blks = cdiv(total_cache_rows, blk_size)
        num_blk_vecs = cdiv(num_blks, lanes)
        vecs_per_super = 32
        num_supers = cdiv(num_blk_vecs, vecs_per_super)
        blk_buf_size = num_supers * vecs_per_super * lanes
        num_super_vecs = cdiv(num_supers, lanes)
        super_buf_size = max(lanes, num_super_vecs * lanes)
        super_shift = blk_shift + 9
        scratch_stride = m_union + lanes
        mesh = plsc.VectorSubcoreMesh(
            num_cores=sc.num_cores,
            num_subcores=eff_subcores,
            core_axis_name="core",
            subcore_axis_name="subcore",
        )

        def _sc_union_body(
            tk_hbm,
            out_k_hbm,
            out_b_hbm,
            out_n_hbm,
            bm_vmem,
            k_tmp,
            b_tmp,
            *extra_vmem,
        ):
            flat = lax.axis_index("core") * eff_subcores + lax.axis_index("subcore")
            iota_lanes = jnp.arange(lanes, dtype=jnp.int32)

            @pl.when(lax.axis_index("core") >= 0)
            def _init_bm():
                def zero_bm(i):
                    bm_vmem[pl.ds(i * lanes, lanes)] = jnp.zeros((lanes,), jnp.int32)

                plsc.parallel_loop(0, win_size // lanes, unroll=16)(zero_bm)
                if num_wins > 1:
                    page_slot_vmem = extra_vmem[0]

                    def zero_ps(i):
                        page_slot_vmem[pl.ds(i * lanes, lanes)] = jnp.zeros(
                            (lanes,), jnp.int32
                        )

                    plsc.parallel_loop(0, num_blk_vecs, unroll=16)(zero_ps)

            def step(tk_in_vmem, k_out_vmem, b_out_vmem, n_out_vmem):
                grp_id = pl.program_id(0) * cores_per_chip + flat

                def init_tmp(i):
                    pad_idx = lax.rem(
                        grp_id * m_union + i * lanes + iota_lanes,
                        total_cache_rows,
                    )
                    k_tmp[pl.ds(i * lanes, lanes)] = pad_idx
                    b_tmp[pl.ds(i * lanes, lanes)] = jnp.zeros((lanes,), jnp.int32)

                plsc.parallel_loop(0, m_union // lanes, unroll=16)(init_tmp)
                vecs_per_q = topk // lanes

                if num_wins == 1:

                    def scatter_step_1w(vec_idx):
                        q_id = vec_idx // vecs_per_q
                        q_bit = jnp.left_shift(jnp.int32(1), q_id)
                        kv = tk_in_vmem[pl.ds(vec_idx * lanes, lanes)]
                        valid = (kv >= 0) & (kv < win_size)
                        rel_idx = jnp.where(valid, kv, 0)
                        cur_bm_1w = plsc.load_gather(bm_vmem, [rel_idx], mask=valid)
                        plsc.store_scatter(
                            bm_vmem,
                            [rel_idx],
                            jnp.bitwise_or(cur_bm_1w, q_bit),
                            mask=valid,
                        )

                    pl.loop(0, n_cols // lanes)(scatter_step_1w)

                    def compact_step_1w(i, woff):
                        bv = bm_vmem[pl.ds(i * lanes, lanes)]
                        sel = bv != 0
                        cnt = plsc.all_reduce_population_count(sel)[0]

                        @pl.when(cnt > 0)
                        def _store_and_clear_1w():
                            kv = i * lanes + iota_lanes
                            safe_woff = jnp.minimum(woff, m_union)
                            plsc.store_compressed(
                                k_tmp.at[pl.ds(safe_woff, lanes)], kv, mask=sel
                            )
                            plsc.store_compressed(
                                b_tmp.at[pl.ds(safe_woff, lanes)], bv, mask=sel
                            )
                            bm_vmem[pl.ds(i * lanes, lanes)] = jnp.zeros(
                                (lanes,), jnp.int32
                            )

                        return woff + cnt

                    total_u = plsc.parallel_loop(
                        0, win_size // lanes, unroll=16, carry=jnp.int32(0)
                    )(compact_step_1w)
                else:
                    page_slot_vmem, act_pages_vmem, super_vmem = extra_vmem

                    def zero_sup(i):
                        super_vmem[pl.ds(i * lanes, lanes)] = jnp.zeros(
                            (lanes,), jnp.int32
                        )

                    plsc.parallel_loop(0, num_super_vecs, unroll=1)(zero_sup)

                    def mark_page(vec_idx):
                        kv = tk_in_vmem[pl.ds(vec_idx * lanes, lanes)]
                        valid = (kv >= 0) & (kv < total_cache_rows)
                        kv_safe = jnp.where(valid, kv, 0)
                        p_idx = lax.shift_right_logical(kv_safe, blk_shift)
                        _, last_p = plsc.scan_count(p_idx, mask=valid)
                        plsc.store_scatter(
                            page_slot_vmem,
                            [p_idx],
                            jnp.ones((lanes,), jnp.int32),
                            mask=last_p,
                        )
                        s_idx = lax.shift_right_logical(kv_safe, super_shift)
                        _, last_s = plsc.scan_count(s_idx, mask=valid)
                        plsc.store_scatter(
                            super_vmem,
                            [s_idx],
                            jnp.ones((lanes,), jnp.int32),
                            mask=last_s,
                        )

                    plsc.parallel_loop(0, n_cols // lanes, unroll=16)(mark_page)

                    def scan_super_body(s, p_cnt_init):
                        s_vec = super_vmem[pl.ds((s // lanes) * lanes, lanes)]
                        s_active = (
                            plsc.all_reduce_population_count(
                                (s_vec > 0) & (iota_lanes == (s % lanes))
                            )[0]
                            > 0
                        )
                        v_base = s * vecs_per_super
                        v_cnt = jnp.where(
                            s_active,
                            jnp.minimum(vecs_per_super, num_blk_vecs - v_base),
                            0,
                        )

                        def assign_page_step(v_rel, p_cnt):
                            i = v_base + v_rel
                            pv = page_slot_vmem[pl.ds(i * lanes, lanes)]
                            sel = pv != 0
                            cnt = plsc.all_reduce_population_count(sel)[0]

                            @pl.when(cnt > 0)
                            def _store_act_page():
                                p_ids = i * lanes + iota_lanes
                                safe_pcnt = jnp.minimum(p_cnt, max_total_blks)
                                plsc.store_compressed(
                                    act_pages_vmem.at[pl.ds(safe_pcnt, lanes)],
                                    p_ids,
                                    mask=sel,
                                )
                                page_slot_vmem[pl.ds(i * lanes, lanes)] = jnp.zeros(
                                    (lanes,), jnp.int32
                                )

                            return p_cnt + cnt

                        return plsc.parallel_loop(
                            0, v_cnt, unroll=16, carry=p_cnt_init
                        )(assign_page_step)

                    raw_act_pages = lax.fori_loop(
                        0, num_supers, scan_super_body, jnp.int32(0)
                    )
                    total_act_pages = jnp.minimum(raw_act_pages, max_total_blks)

                    def run_batch(b_start, batch_pages, woff_init):
                        batch_page_vecs = (batch_pages + lanes - 1) // lanes

                        def scatter_slot_id(j):
                            slot_ids = j * lanes + iota_lanes
                            valid_s = slot_ids < batch_pages
                            p_ids = act_pages_vmem[pl.ds(b_start + j * lanes, lanes)]
                            plsc.store_scatter(
                                page_slot_vmem,
                                [jnp.where(valid_s, p_ids, 0)],
                                slot_ids + 1,
                                mask=valid_s,
                            )

                        plsc.parallel_loop(0, batch_page_vecs, unroll=4)(
                            scatter_slot_id
                        )

                        def scatter_step(vec_idx):
                            q_id = vec_idx // vecs_per_q
                            q_bit = jnp.left_shift(jnp.int32(1), q_id)
                            kv = tk_in_vmem[pl.ds(vec_idx * lanes, lanes)]
                            valid = (kv >= 0) & (kv < total_cache_rows)
                            p_idx = lax.shift_right_logical(
                                jnp.where(valid, kv, 0), blk_shift
                            )
                            p_off = jnp.bitwise_and(kv, blk_size - 1)
                            s_plus_1 = plsc.load_gather(
                                page_slot_vmem, [p_idx], mask=valid
                            )
                            in_batch = valid & (s_plus_1 > 0)
                            rel_idx = jnp.where(
                                in_batch,
                                jnp.left_shift(s_plus_1 - 1, blk_shift) + p_off,
                                0,
                            )
                            cur_bm_b = plsc.load_gather(
                                bm_vmem, [rel_idx], mask=in_batch
                            )
                            plsc.store_scatter(
                                bm_vmem,
                                [rel_idx],
                                jnp.bitwise_or(cur_bm_b, q_bit),
                                mask=in_batch,
                            )

                        pl.loop(0, n_cols // lanes)(scatter_step)

                        def compact_step(i, woff):
                            bv = bm_vmem[pl.ds(i * lanes, lanes)]
                            sel = bv != 0
                            cnt = plsc.all_reduce_population_count(sel)[0]

                            @pl.when(cnt > 0)
                            def _store_and_clear():
                                p_slot = b_start + (i // vecs_per_blk)
                                sub_off = (i % vecs_per_blk) * lanes
                                p_id_vec = plsc.load_gather(
                                    act_pages_vmem,
                                    [jnp.full((lanes,), p_slot, jnp.int32)],
                                )
                                kv = (
                                    jnp.left_shift(p_id_vec, blk_shift)
                                    + sub_off
                                    + iota_lanes
                                )
                                safe_woff = jnp.minimum(woff, m_union)
                                plsc.store_compressed(
                                    k_tmp.at[pl.ds(safe_woff, lanes)],
                                    kv,
                                    mask=sel,
                                )
                                plsc.store_compressed(
                                    b_tmp.at[pl.ds(safe_woff, lanes)],
                                    bv,
                                    mask=sel,
                                )
                                bm_vmem[pl.ds(i * lanes, lanes)] = jnp.zeros(
                                    (lanes,), jnp.int32
                                )

                            return woff + cnt

                        woff_out = plsc.parallel_loop(
                            0,
                            batch_pages * vecs_per_blk,
                            unroll=16,
                            carry=woff_init,
                        )(compact_step)

                        def clear_slot_id(j):
                            slot_ids = j * lanes + iota_lanes
                            valid_s = slot_ids < batch_pages
                            p_ids = act_pages_vmem[pl.ds(b_start + j * lanes, lanes)]
                            plsc.store_scatter(
                                page_slot_vmem,
                                [jnp.where(valid_s, p_ids, 0)],
                                jnp.zeros((lanes,), jnp.int32),
                                mask=valid_s,
                            )

                        plsc.parallel_loop(0, batch_page_vecs, unroll=4)(clear_slot_id)
                        return woff_out

                    if max_total_blks <= max_act_blks:
                        total_u = run_batch(0, total_act_pages, jnp.int32(0))
                    else:
                        num_page_batches = (
                            total_act_pages + max_act_blks - 1
                        ) // max_act_blks
                        total_u = lax.fori_loop(
                            0,
                            num_page_batches,
                            lambda b_idx, woff: run_batch(
                                b_idx * max_act_blks,
                                jnp.minimum(
                                    total_act_pages - b_idx * max_act_blks,
                                    max_act_blks,
                                ),
                                woff,
                            ),
                            jnp.int32(0),
                        )
                    total_u = jnp.where(
                        raw_act_pages > max_total_blks, m_union + 1, total_u
                    )

                if u_chunk is not None and m_union % u_chunk == 0:
                    for c in range(1, m_union // u_chunk):

                        @pl.when(total_u <= c * u_chunk)
                        def _mark_inactive(c_idx=c):
                            k_tmp[pl.ds(c_idx * u_chunk, lanes)] = jnp.full(
                                (lanes,), -1, jnp.int32
                            )

                def copy_out(i):
                    k_out_vmem[pl.ds(i * lanes, lanes)] = k_tmp[pl.ds(i * lanes, lanes)]
                    b_out_vmem[pl.ds(i * lanes, lanes)] = b_tmp[pl.ds(i * lanes, lanes)]

                plsc.parallel_loop(0, m_union // lanes, unroll=16)(copy_out)
                n_out_vmem[pl.ds(0, lanes)] = jnp.full((lanes,), total_u, jnp.int32)

            in_mode = pl.Buffered(1 if num_waves == 1 else 2)
            out_mode = pl.Buffered(1 if (num_waves == 1 or m_union > 2048) else 2)

            def spec(sz, mode=in_mode):
                return pl.BlockSpec(
                    (sz,),
                    lambda w: (w * cores_per_chip + flat,),
                    pipeline_mode=mode,
                )

            pltpu.emit_pipeline(
                step,
                grid=(num_waves,),
                in_specs=(spec(n_cols, in_mode),),
                out_specs=(
                    spec(m_union, out_mode),
                    spec(m_union, out_mode),
                    spec(lanes, out_mode),
                ),
            )(tk_hbm, out_k_hbm, out_b_hbm, out_n_hbm)

        base_scratch = (
            pltpu.VMEM((win_size,), jnp.int32),
            pltpu.VMEM((scratch_stride,), jnp.int32),
            pltpu.VMEM((scratch_stride,), jnp.int32),
        )
        extra_scratch = (
            ()
            if num_wins == 1
            else (
                pltpu.VMEM((blk_buf_size + lanes,), jnp.int32),
                pltpu.VMEM((max_total_blks + lanes,), jnp.int32),
                pltpu.VMEM((super_buf_size + lanes,), jnp.int32),
            )
        )
        out_k, out_b, out_n = pl.kernel(
            _sc_union_body,
            out_type=(
                jax.ShapeDtypeStruct((padded_g * m_union,), jnp.int32),
                jax.ShapeDtypeStruct((padded_g * m_union,), jnp.int32),
                jax.ShapeDtypeStruct((padded_g * lanes,), jnp.int32),
            ),
            scratch_types=base_scratch + extra_scratch,
            mesh=mesh,
            compiler_params=pltpu.CompilerParams(
                disable_bounds_checks=True, needs_layout_passes=False
            ),
            name=f"sc_union_ztc_g{num_groups}_m{m_union}_w{win_size}_nb{num_blks}",
        )(grouped_pad.reshape(-1))

        if padded_g == num_groups:
            union_gather_indices = out_k
            cand_bits_tiled = out_b.reshape(num_groups, m_union // 128, 128)
        else:
            union_gather_indices = out_k[: num_groups * m_union]
            cand_bits_tiled = out_b[: num_groups * m_union].reshape(
                num_groups, m_union // 128, 128
            )
        n_unique = out_n.reshape(padded_g, lanes)[:num_groups, 0]
        return union_gather_indices, cand_bits_tiled, n_unique

    grouped_keys = topk_indices.reshape(num_groups, b_q * topk)
    q_idx = jnp.broadcast_to(
        jnp.arange(b_q, dtype=jnp.int32)[None, :, None], (num_groups, b_q, topk)
    ).reshape(num_groups, b_q * topk)
    packed_in = jnp.where(
        grouped_keys >= 0,
        jnp.bitwise_or(jnp.left_shift(grouped_keys, 5), q_idx),
        _INT32_MAX,
    )
    s_packed = lax.sort(packed_in, dimension=-1, is_stable=False)
    valid_s = s_packed < _INT32_MAX
    s_keys = jnp.where(valid_s, lax.shift_right_logical(s_packed, 5), _INT32_MAX)
    s_bits = jnp.where(
        valid_s,
        jnp.left_shift(jnp.int32(1), jnp.bitwise_and(s_packed, 31)),
        jnp.int32(0),
    )

    # Segmented right-to-left bitwise-OR across equal keys (run length <= b_q)
    shifts = tuple(1 << s for s in range((b_q * topk).bit_length()))
    for shift in shifts:
        rhs_keys = jnp.concatenate(
            [
                s_keys[:, shift:],
                jnp.full((num_groups, shift), _INT32_MAX, dtype=jnp.int32),
            ],
            axis=-1,
        )
        rhs_bits = jnp.concatenate(
            [
                s_bits[:, shift:],
                jnp.zeros((num_groups, shift), dtype=jnp.int32),
            ],
            axis=-1,
        )
        s_bits = jnp.where(s_keys == rhs_keys, jnp.bitwise_or(s_bits, rhs_bits), s_bits)

    prev_keys = jnp.concatenate(
        [jnp.full((num_groups, 1), -1, dtype=jnp.int32), s_keys[:, :-1]],
        axis=-1,
    )
    is_unique = (s_keys < _INT32_MAX) & (s_keys != prev_keys)
    u_keys = jnp.where(is_unique, s_keys, _INT32_MAX)
    n_unique = jnp.sum(is_unique.astype(jnp.int32), axis=-1)
    s_bits_clean = jnp.where(is_unique, s_bits, jnp.int32(0))
    u_sorted, bits_sorted = lax.sort(
        [u_keys, s_bits_clean], dimension=-1, is_stable=False, num_keys=1
    )
    union_sorted = u_sorted[:, :m_union]
    cand_bits_tiled = bits_sorted[:, :m_union].reshape(num_groups, m_union // 128, 128)

    flat_grp_idx = jnp.arange(num_groups * m_union, dtype=jnp.int32).reshape(
        num_groups, m_union
    )
    scattered_rows = (flat_grp_idx * 104729) % total_cache_rows
    valid_or_pad = jnp.where(union_sorted < _INT32_MAX, union_sorted, scattered_rows)
    if u_chunk is not None and m_union % u_chunk == 0:
        col_idx = jnp.arange(m_union, dtype=jnp.int32)[None, :]
        chunk_start = (col_idx // u_chunk) * u_chunk
        chunk_active = (chunk_start == 0) | (chunk_start < n_unique[:, None])
        union_gather_indices = jnp.where(
            chunk_active, valid_or_pad, jnp.int32(-1)
        ).reshape(-1)
    else:
        union_gather_indices = valid_or_pad.reshape(-1)
    return union_gather_indices, cand_bits_tiled, n_unique


@functools.partial(
    jax.jit,
    static_argnames=("b_q", "m_union", "total_cache_rows", "u_chunk"),
)
def precompute_csa_union_and_bitmask(
    topk_indices: jax.Array,
    *,
    b_q: int = 32,
    m_union: int = 1536,
    total_cache_rows: int,
    u_chunk: int = 512,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Precomputes the LxL Block-Grouped Candidate Union once per step for all
    Consumer CSA layers sharing the same Indexer topk_indices buffer."""
    return _build_candidate_union_and_bitmask(
        topk_indices, b_q, m_union, total_cache_rows, u_chunk=u_chunk
    )


@functools.partial(
    jax.jit,
    static_argnames=("b_q", "m_union", "total_cache_rows", "u_chunk"),
)
def precompute_csa_union_and_gather(
    topk_indices: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    *,
    b_q: int = 8,
    m_union: int = 1024,
    total_cache_rows: int,
    u_chunk: int = 1024,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Precomputes LxL Candidate Union AND SparseCore KV gather once per
    (kv_source_layer, index_source_layer) group so follower CSA layers reuse
    the gathered KV slabs in HBM with zero redundant sc_union_ztc/csa_gather calls."""
    u_idx, c_bits, n_uniq = _build_candidate_union_and_bitmask(
        topk_indices, b_q, m_union, total_cache_rows, u_chunk=u_chunk
    )
    g_nope, g_rope = csa_gather.csa_gather(
        cache_kv_nope,
        cache_kv_rope,
        u_idx,
        num_valid_indices=u_idx.shape[0],
        top_k=u_chunk,
        max_active_per_chunk=n_uniq if u_chunk == m_union else None,
    )
    return u_idx, c_bits, n_uniq, g_nope, g_rope


_CSA_UNION_VERIFY_BQ: int = int(envs.VLLM_DSV41_CSA_UNION_VERIFY_BQ)
_CSA_UNION_MAX_TOKENS: int = int(envs.VLLM_DSV41_CSA_UNION_MAX_TOKENS)
_FAST_GROUPED_ATTN: bool = bool(envs.VLLM_DSV41_FAST_GROUPED_ATTN)


def resolve_csa_union_u_chunk(b_q: int, m_union: int, topk: int = 512) -> int:
    if (
        topk == 512
        and m_union % 1024 == 0
        and (b_q in (2, 4, 5, 6, 8) and m_union <= 1024)
    ):
        return min(1024, m_union)
    return min(512, m_union)


def resolve_csa_union_params(
    num_tokens: int,
    topk: int,
    max_num_seqs: int,
    ced_decoder: bool = False,
    num_q_heads: int = 16,
) -> tuple[int, int] | None:
    """Returns `(b_q, m_union)` when LxL Candidate Union is enabled, else `None`."""
    if (
        not envs.VLLM_DSV41_CSA_UNION
        or topk % 128 != 0
        or num_tokens > _CSA_UNION_MAX_TOKENS
    ):
        return None
    if (
        _CSA_UNION_VERIFY_BQ > 0
        and num_tokens >= _CSA_UNION_VERIFY_BQ
        and num_tokens % _CSA_UNION_VERIFY_BQ == 0
    ):
        b_q = _CSA_UNION_VERIFY_BQ
        m_u = (
            min(envs.VLLM_DSV41_CSA_UNION_M, b_q * topk)
            if envs.VLLM_DSV41_CSA_UNION_M > 0
            else align_to(2 * topk, 128)
        )
        return b_q, m_u
    _spec_num_tokens = 0
    if (
        _spec_num_tokens == 5
        and num_q_heads <= 64
        and num_tokens >= 6
        and num_tokens % 6 == 0
    ):
        m_u = (
            min(envs.VLLM_DSV41_CSA_UNION_M, 6 * topk)
            if envs.VLLM_DSV41_CSA_UNION_M > 0
            else align_to(2 * topk, 128)
        )
        return 6, m_u
    if (
        _spec_num_tokens == 4
        and num_q_heads <= 64
        and num_tokens >= 5
        and num_tokens % 5 == 0
        and num_tokens % 512 != 0
    ):
        m_u = (
            min(envs.VLLM_DSV41_CSA_UNION_M, 5 * topk)
            if envs.VLLM_DSV41_CSA_UNION_M > 0
            else align_to(2 * topk, 128)
        )
        return 5, m_u
    b_q = envs.VLLM_DSV41_CSA_UNION_BQ if envs.VLLM_DSV41_CSA_UNION_BQ > 0 else 8
    if (
        not ced_decoder
        and num_tokens > max_num_seqs
        and num_tokens >= 512
        and num_tokens % b_q == 0
    ):
        m_u = (
            min(envs.VLLM_DSV41_CSA_UNION_M, b_q * topk)
            if envs.VLLM_DSV41_CSA_UNION_M > 0
            else align_to(2 * topk, 128)
        )
        return b_q, m_u
    return None


def _m_run(swa_m_t, sinks_t):
    """Running-max seed for the b_q < 8 unrolled online-softmax blocks.

    Decode role (decodefix 11e6c02): include the attention sink so that
    ``exp(sinks_t - m_new)`` cannot overflow to inf. The prefill role
    (VLLM_DSV41_PURE_20L_PREFILL=1) keeps cd4eedb's kernel body unchanged.
    Both are only traced on the CSA candidate-union path.
    """
    if envs.VLLM_DSV41_PURE_20L_PREFILL:
        return swa_m_t
    return jnp.maximum(swa_m_t, sinks_t)


def _grouped_attention_kernel(
    n_unique_ref,  # i32[num_groups]
    meta_ref,  # i32[3] (start_group, num_active_groups, num_valid_tokens)
    sem_ids_ref,  # i32[3] (bi_sem_idx, bo_sem_idx, kv_sem_idx)
    bo_ids_ref,  # i32[2] group_idx of in-flight bo DMA
    attention_sinks_ref,  # f32[l_sublanes, 128]
    q_hbm_ref,  # bf16[num_tokens, num_q_heads, head_dim]
    cache_kv_nope_hbm_ref,  # int32[num_groups, num_u_chunks, u_chunk, 128]
    cache_kv_rope_hbm_ref,  # int32[num_groups, num_u_chunks, u_chunk // 4, 128]
    cand_bits_hbm_ref,  # int32[num_groups, num_u_chunks, u_chunk // 128, 128]
    swa_accumution_hbm_ref,  # bf16[num_tokens, num_q_heads, head_dim]
    swa_l_hbm_ref,  # f32[num_groups, l_sublanes, 128]
    swa_m_hbm_ref,  # f32[num_groups, l_sublanes, 128]
    o_hbm_ref,  # bf16[num_tokens, num_q_heads, head_dim]
    bkv_nope_x2_ref,  # u8[2, u_chunk * 4, 128]
    bkv_rope_x2_ref,  # bf16[2, u_chunk // 2, 128]
    bq_x2_ref,  # bf16[2, b_q_vmem, num_q_heads, head_dim]
    bo_x2_ref,  # bf16[2, b_q_vmem, num_q_heads, head_dim]
    bl_x2_ref,  # f32[2, l_sublanes, 128]
    bm_x2_ref,  # f32[2, l_sublanes, 128]
    swa_acc_x2_ref,  # bf16[2, b_q_vmem, num_q_heads, head_dim]
    cand_bits_x2_ref,  # int32[2, u_chunk // 128, 128]
    sems,  # DMA[(8, 2)]
    *,
    sm_scale: float,
    b_q: int,
    m_union: int,
    u_chunk: int,
    l_stride: int,
):
    """Fused LxL Block-Grouped Candidate Union Attention Kernel with exact [b_q * H, D] packing."""
    _, n_qh, head_dim = q_hbm_ref.shape
    num_q_heads = n_qh // b_q
    q_dtype = q_hbm_ref.dtype
    num_u_chunks = m_union // u_chunk
    max_exact_chunks = min(num_u_chunks, 2 if (num_q_heads > 64 and n_qh > 768) else 3)
    cache_kv_nope_hbm_ref = cache_kv_nope_hbm_ref.bitcast(jnp.uint8)
    cache_kv_rope_hbm_ref = cache_kv_rope_hbm_ref.bitcast(jnp.bfloat16)

    start_group = meta_ref[0]
    num_active_groups = meta_ref[1]
    num_valid_tokens = meta_ref[2]
    num_run_groups = num_active_groups - start_group
    p_id = pl.program_id(0)
    g = start_group + p_id

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_q_inputs(grp_idx, sem_idx, *, wait=False):
        q_dst = bq_x2_ref.at[sem_idx]
        acc_dst = swa_acc_x2_ref.at[sem_idx]
        if not wait:
            _async_copy(
                q_hbm_ref.at[grp_idx],
                q_dst,
                sems.at[1, sem_idx],
                False,
            )
            _async_copy(
                swa_accumution_hbm_ref.at[grp_idx],
                acc_dst,
                sems.at[3, sem_idx],
                False,
            )
            _async_copy(
                swa_l_hbm_ref.at[grp_idx],
                bl_x2_ref.at[sem_idx],
                sems.at[4, sem_idx],
                False,
            )
            _async_copy(
                swa_m_hbm_ref.at[grp_idx],
                bm_x2_ref.at[sem_idx],
                sems.at[5, sem_idx],
                False,
            )
        else:
            for s_id, ref in (
                (1, q_dst),
                (3, acc_dst),
                (4, bl_x2_ref.at[sem_idx]),
                (5, bm_x2_ref.at[sem_idx]),
            ):
                _async_copy(src=ref, dst=ref, sem=sems.at[s_id, sem_idx], wait=True)

    def _fetch_kv_chunk(grp_idx, chunk_idx, sem_idx, *, wait=False):
        if not wait:
            _async_copy(
                cache_kv_nope_hbm_ref.at[grp_idx, chunk_idx, :, :],
                bkv_nope_x2_ref.at[sem_idx, :, :],
                sems.at[0, sem_idx],
                False,
            )
            _async_copy(
                cache_kv_rope_hbm_ref.at[grp_idx, chunk_idx, :, :],
                bkv_rope_x2_ref.at[sem_idx, :, :],
                sems.at[6, sem_idx],
                False,
            )
            _async_copy(
                cand_bits_hbm_ref.at[grp_idx, chunk_idx],
                cand_bits_x2_ref.at[sem_idx],
                sems.at[7, sem_idx],
                False,
            )
        else:
            for s_id, ref in (
                (0, bkv_nope_x2_ref.at[sem_idx, :, :]),
                (6, bkv_rope_x2_ref.at[sem_idx, :, :]),
                (7, cand_bits_x2_ref.at[sem_idx]),
            ):
                _async_copy(src=ref, dst=ref, sem=sems.at[s_id, sem_idx], wait=True)

    def _send_output(grp_idx, sem_idx, *, wait=False):
        vmem_ref = bo_x2_ref.at[sem_idx]
        sem = sems.at[2, sem_idx]
        if not wait:
            bo_ids_ref[sem_idx] = grp_idx
            _async_copy(
                vmem_ref,
                o_hbm_ref.at[grp_idx],
                sem,
                False,
            )
        else:
            _async_copy(src=vmem_ref, dst=vmem_ref, sem=sem, wait=True)

    def _wait_send_output(sem_idx):
        old_grp = bo_ids_ref[sem_idx]

        @pl.when(old_grp >= 0)
        def _():
            _send_output(old_grp, sem_idx, wait=True)

    lane_128 = lax.broadcasted_iota(jnp.int32, (u_chunk, 128), 1)
    m_span_u = lax.broadcasted_iota(jnp.int32, (u_chunk, head_dim), 0)

    def _load_dequant_bkv(kv_sem_idx, c_offset=0):
        bkv_nope = bkv_nope_x2_ref.at[kv_sem_idx][...].reshape(u_chunk, -1)
        bkv_rope = bkv_rope_x2_ref.at[kv_sem_idx][...].reshape(u_chunk // 2, -1)
        if bkv_nope.shape[1] == 512 and bkv_rope.shape[1] == 128:
            bkv = _dequant_and_merge_bkv(bkv_nope, bkv_rope, lane_128)
        else:
            bkv_nope = _dequant_dsv4_fp8(bkv_nope)
            rope_dim = bkv_rope.shape[-1] // 2
            bkv_rope = jnp.concatenate(
                [bkv_rope[:, :rope_dim], bkv_rope[:, rope_dim:]], axis=0
            )
            bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)
        c_valid = jnp.clip(n_uniq - c_offset, 0, u_chunk)
        return jnp.where(m_span_u < c_valid, bkv, 0)

    @pl.when(p_id == 0)
    def _prologue():
        _fetch_q_inputs(start_group, 0, wait=False)
        _fetch_kv_chunk(start_group, 0, 0, wait=False)

    bi_sem_idx = sem_ids_ref[0]
    next_g = g + 1
    next_bi_sem_idx = lax.select(bi_sem_idx == 0, 1, 0)

    @pl.when(next_g < num_active_groups)
    def _prefetch_next_q():
        sem_ids_ref[0] = next_bi_sem_idx
        _fetch_q_inputs(next_g, next_bi_sem_idx, wait=False)

    bo_sem_idx = sem_ids_ref[1]
    sem_ids_ref[1] = lax.select(bo_sem_idx == 0, 1, 0)

    _fetch_q_inputs(g, bi_sem_idx, wait=True)

    @pl.when(p_id >= 2)
    def _wait_prev_send():
        _wait_send_output(bo_sem_idx)

    n_uniq = n_unique_ref[g]
    act_chunks = jnp.clip(pl.cdiv(n_uniq, u_chunk), 1, num_u_chunks)

    q_heads_per_tile = _largest_divisor(b_q, max(1, 1024 // num_q_heads))
    q_sub = q_heads_per_tile * num_q_heads
    num_q_tiles = n_qh // q_sub

    bq_tiles = bq_x2_ref.at[bi_sem_idx][...].reshape(num_q_tiles, q_sub, head_dim)
    swa_acc_tiles = swa_acc_x2_ref.at[bi_sem_idx][...].reshape(
        num_q_tiles, q_sub, head_dim
    )
    swa_m_all = (
        bm_x2_ref.at[bi_sem_idx][...].reshape(-1)[:n_qh].reshape(num_q_tiles, q_sub)
    )
    swa_l_all = (
        bl_x2_ref.at[bi_sem_idx][...].reshape(-1)[:n_qh].reshape(num_q_tiles, q_sub)
    )
    sinks_all = attention_sinks_ref[...].reshape(-1)[:n_qh].reshape(num_q_tiles, q_sub)

    kv_sem_0 = sem_ids_ref[2]
    next_kv_sem = lax.select(kv_sem_0 == 0, 1, 0)

    if num_u_chunks >= 2:

        @pl.when(act_chunks >= 2)
        def _pref_c1():
            sem_ids_ref[2] = next_kv_sem
            _fetch_kv_chunk(g, 1, next_kv_sem, wait=False)

    @pl.when((act_chunks == 1) & (next_g < num_active_groups))
    def _pref_ng_from_c0():
        sem_ids_ref[2] = next_kv_sem
        _fetch_kv_chunk(next_g, 0, next_kv_sem, wait=False)

    _fetch_kv_chunk(g, 0, kv_sem_0, wait=True)
    bkv_0 = _load_dequant_bkv(kv_sem_0, 0)
    bits_0 = lax.broadcast_in_dim(
        cand_bits_x2_ref.at[kv_sem_0][...].reshape(-1), (q_sub, u_chunk), (1,)
    )
    q_iota_base = lax.broadcasted_iota(jnp.int32, (q_sub, u_chunk), 0) // num_q_heads

    def _masked_qk(qt, bkv_c, bits_c):
        s_c = (
            jnp.einsum(
                "nd,md->nm",
                bq_tiles[qt],
                bkv_c,
                preferred_element_type=jnp.float32,
            )
            * sm_scale
        )
        q_bit = jnp.left_shift(jnp.int32(1), qt * q_heads_per_tile + q_iota_base)
        return jnp.where(
            jnp.bitwise_and(bits_c, q_bit) != 0, s_c, jnp.finfo(s_c.dtype).min
        )

    @pl.when(act_chunks == 1)
    def _run_1chunk():
        s_cur = _masked_qk(0, bkv_0, bits_0)
        for qt in range(num_q_tiles):
            swa_m_t = swa_m_all[qt]
            swa_l_t = swa_l_all[qt]
            sinks_t = sinks_all[qt]
            swa_acc_t = swa_acc_tiles[qt]

            if b_q < 8:
                m_new = jnp.maximum(_m_run(swa_m_t, sinks_t), jnp.max(s_cur, axis=1))
                exp_swa = jnp.exp(swa_m_t - m_new)
                p = jnp.exp(s_cur - m_new[:, None])
                s_nxt = (
                    _masked_qk(qt + 1, bkv_0, bits_0) if qt + 1 < num_q_tiles else None
                )
                l_new = (exp_swa * swa_l_t + jnp.sum(p, axis=1)) + jnp.exp(
                    sinks_t - m_new
                )
                pv = jnp.einsum(
                    "nm,md->nd", p, bkv_0, preferred_element_type=jnp.float32
                )
                acc_new = exp_swa[:, None] * swa_acc_t + pv
            else:
                m_init = jnp.maximum(swa_m_t, sinks_t)
                exp_swa = jnp.exp(swa_m_t - m_init)
                l_init = exp_swa * swa_l_t + jnp.exp(sinks_t - m_init)
                m_new = jnp.maximum(m_init, jnp.max(s_cur, axis=1))
                alpha = jnp.exp(m_init - m_new)
                p = jnp.exp(s_cur - m_new[:, None])
                s_nxt = (
                    _masked_qk(qt + 1, bkv_0, bits_0) if qt + 1 < num_q_tiles else None
                )
                l_new = alpha * l_init + jnp.sum(p, axis=1)
                pv = jnp.einsum(
                    "nm,md->nd", p, bkv_0, preferred_element_type=jnp.float32
                )
                acc_new = (alpha * exp_swa)[:, None] * swa_acc_t + pv

            l_new_2d = l_new[:, None]
            out_t = (
                lax.div(acc_new, l_new_2d)
                if q_dtype == jnp.float32
                else (acc_new * pl.reciprocal(l_new_2d, approx=True)).astype(q_dtype)
            )
            bo_x2_ref.at[bo_sem_idx, pl.ds(qt * q_sub, q_sub)][...] = out_t
            s_cur = s_nxt

    if max_exact_chunks >= 2:

        @pl.when(act_chunks == 2)
        def _run_2chunks():
            # Compute all Q @ K_0^T tiles while chunk 1 DMA completes in background!
            s0_list = [_masked_qk(qt, bkv_0, bits_0) for qt in range(num_q_tiles)]

            kv_sem_1 = sem_ids_ref[2]
            next_kv_sem_2 = lax.select(kv_sem_1 == 0, 1, 0)

            @pl.when(next_g < num_active_groups)
            def _pref_ng_from_c1():
                sem_ids_ref[2] = next_kv_sem_2
                _fetch_kv_chunk(next_g, 0, next_kv_sem_2, wait=False)

            _fetch_kv_chunk(g, 1, kv_sem_1, wait=True)
            bkv_1 = _load_dequant_bkv(kv_sem_1, u_chunk)
            bits_1 = lax.broadcast_in_dim(
                cand_bits_x2_ref.at[kv_sem_1][...].reshape(-1),
                (q_sub, u_chunk),
                (1,),
            )
            bkv_01 = jnp.concatenate([bkv_0, bkv_1], axis=0)

            s1_cur = _masked_qk(0, bkv_1, bits_1)
            for qt in range(num_q_tiles):
                swa_m_t = swa_m_all[qt]
                swa_l_t = swa_l_all[qt]
                sinks_t = sinks_all[qt]
                swa_acc_t = swa_acc_tiles[qt]
                s0_t = s0_list[qt]
                max_s01 = jnp.maximum(jnp.max(s0_t, axis=1), jnp.max(s1_cur, axis=1))

                if b_q < 8:
                    m_new = jnp.maximum(_m_run(swa_m_t, sinks_t), max_s01)
                    exp_swa = jnp.exp(swa_m_t - m_new)
                    p01 = jnp.concatenate(
                        [
                            jnp.exp(s0_t - m_new[:, None]),
                            jnp.exp(s1_cur - m_new[:, None]),
                        ],
                        axis=1,
                    )
                    s1_nxt = (
                        _masked_qk(qt + 1, bkv_1, bits_1)
                        if qt + 1 < num_q_tiles
                        else None
                    )
                    l_new = (exp_swa * swa_l_t + jnp.sum(p01, axis=1)) + jnp.exp(
                        sinks_t - m_new
                    )
                    pv = jnp.einsum(
                        "nm,md->nd",
                        p01,
                        bkv_01,
                        preferred_element_type=jnp.float32,
                    )
                    acc_new = exp_swa[:, None] * swa_acc_t + pv
                else:
                    m_init = jnp.maximum(swa_m_t, sinks_t)
                    exp_swa = jnp.exp(swa_m_t - m_init)
                    l_init = exp_swa * swa_l_t + jnp.exp(sinks_t - m_init)
                    m_new = jnp.maximum(m_init, max_s01)
                    alpha = jnp.exp(m_init - m_new)
                    p01 = jnp.concatenate(
                        [
                            jnp.exp(s0_t - m_new[:, None]),
                            jnp.exp(s1_cur - m_new[:, None]),
                        ],
                        axis=1,
                    )
                    s1_nxt = (
                        _masked_qk(qt + 1, bkv_1, bits_1)
                        if qt + 1 < num_q_tiles
                        else None
                    )
                    l_new = alpha * l_init + jnp.sum(p01, axis=1)
                    pv = jnp.einsum(
                        "nm,md->nd",
                        p01,
                        bkv_01,
                        preferred_element_type=jnp.float32,
                    )
                    acc_new = (alpha * exp_swa)[:, None] * swa_acc_t + pv

                l_new_2d = l_new[:, None]
                out_t = (
                    lax.div(acc_new, l_new_2d)
                    if q_dtype == jnp.float32
                    else (acc_new * pl.reciprocal(l_new_2d, approx=True)).astype(
                        q_dtype
                    )
                )
                bo_x2_ref.at[bo_sem_idx, pl.ds(qt * q_sub, q_sub)][...] = out_t
                s1_cur = s1_nxt

    if max_exact_chunks >= 3:

        @pl.when(act_chunks == 3)
        def _run_3chunks():
            # Overlap chunk 1 DMA behind Q @ K_0^T!
            s0_list = [_masked_qk(qt, bkv_0, bits_0) for qt in range(num_q_tiles)]

            kv_sem_1 = sem_ids_ref[2]
            next_kv_sem_2 = lax.select(kv_sem_1 == 0, 1, 0)
            sem_ids_ref[2] = next_kv_sem_2
            _fetch_kv_chunk(g, 2, next_kv_sem_2, wait=False)

            _fetch_kv_chunk(g, 1, kv_sem_1, wait=True)
            bkv_1 = _load_dequant_bkv(kv_sem_1, u_chunk)
            bits_1 = lax.broadcast_in_dim(
                cand_bits_x2_ref.at[kv_sem_1][...].reshape(-1),
                (q_sub, u_chunk),
                (1,),
            )

            # Overlap chunk 2 DMA behind Q @ K_1^T!
            s1_list = [_masked_qk(qt, bkv_1, bits_1) for qt in range(num_q_tiles)]

            next_kv_sem_3 = lax.select(next_kv_sem_2 == 0, 1, 0)

            @pl.when(next_g < num_active_groups)
            def _pref_ng_from_c2():
                sem_ids_ref[2] = next_kv_sem_3
                _fetch_kv_chunk(next_g, 0, next_kv_sem_3, wait=False)

            _fetch_kv_chunk(g, 2, next_kv_sem_2, wait=True)
            bkv_2 = _load_dequant_bkv(next_kv_sem_2, 2 * u_chunk)
            bits_2 = lax.broadcast_in_dim(
                cand_bits_x2_ref.at[next_kv_sem_2][...].reshape(-1),
                (q_sub, u_chunk),
                (1,),
            )
            bkv_012 = jnp.concatenate([bkv_0, bkv_1, bkv_2], axis=0)

            s2_cur = _masked_qk(0, bkv_2, bits_2)
            for qt in range(num_q_tiles):
                swa_m_t = swa_m_all[qt]
                swa_l_t = swa_l_all[qt]
                sinks_t = sinks_all[qt]
                swa_acc_t = swa_acc_tiles[qt]
                s0_t = s0_list[qt]
                s1_t = s1_list[qt]
                max_s012 = jnp.maximum(
                    jnp.max(s0_t, axis=1),
                    jnp.maximum(jnp.max(s1_t, axis=1), jnp.max(s2_cur, axis=1)),
                )

                if b_q < 8:
                    m_new = jnp.maximum(_m_run(swa_m_t, sinks_t), max_s012)
                    exp_swa = jnp.exp(swa_m_t - m_new)
                    p012 = jnp.concatenate(
                        [
                            jnp.exp(s0_t - m_new[:, None]),
                            jnp.exp(s1_t - m_new[:, None]),
                            jnp.exp(s2_cur - m_new[:, None]),
                        ],
                        axis=1,
                    )
                    s2_nxt = (
                        _masked_qk(qt + 1, bkv_2, bits_2)
                        if qt + 1 < num_q_tiles
                        else None
                    )
                    l_new = (exp_swa * swa_l_t + jnp.sum(p012, axis=1)) + jnp.exp(
                        sinks_t - m_new
                    )
                    pv = jnp.einsum(
                        "nm,md->nd",
                        p012,
                        bkv_012,
                        preferred_element_type=jnp.float32,
                    )
                    acc_new = exp_swa[:, None] * swa_acc_t + pv
                else:
                    m_init = jnp.maximum(swa_m_t, sinks_t)
                    exp_swa = jnp.exp(swa_m_t - m_init)
                    l_init = exp_swa * swa_l_t + jnp.exp(sinks_t - m_init)
                    m_new = jnp.maximum(m_init, max_s012)
                    alpha = jnp.exp(m_init - m_new)
                    p012 = jnp.concatenate(
                        [
                            jnp.exp(s0_t - m_new[:, None]),
                            jnp.exp(s1_t - m_new[:, None]),
                            jnp.exp(s2_cur - m_new[:, None]),
                        ],
                        axis=1,
                    )
                    s2_nxt = (
                        _masked_qk(qt + 1, bkv_2, bits_2)
                        if qt + 1 < num_q_tiles
                        else None
                    )
                    l_new = alpha * l_init + jnp.sum(p012, axis=1)
                    pv = jnp.einsum(
                        "nm,md->nd",
                        p012,
                        bkv_012,
                        preferred_element_type=jnp.float32,
                    )
                    acc_new = (alpha * exp_swa)[:, None] * swa_acc_t + pv

                l_new_2d = l_new[:, None]
                out_t = (
                    lax.div(acc_new, l_new_2d)
                    if q_dtype == jnp.float32
                    else (acc_new * pl.reciprocal(l_new_2d, approx=True)).astype(
                        q_dtype
                    )
                )
                bo_x2_ref.at[bo_sem_idx, pl.ds(qt * q_sub, q_sub)][...] = out_t
                s2_cur = s2_nxt

    if num_u_chunks > max_exact_chunks:

        @pl.when(act_chunks > max_exact_chunks)
        def _run_stream_chunks():
            # Step 0 (chunk 0 is already loaded & dequantized in bkv_0 / bits_0!)
            m0_list = []
            l0_list = []
            acc0_list = []
            for qt in range(num_q_tiles):
                s0 = _masked_qk(qt, bkv_0, bits_0)
                swa_m_t = swa_m_all[qt]
                swa_l_t = swa_l_all[qt]
                sinks_t = sinks_all[qt]
                m0 = jnp.maximum(jnp.maximum(swa_m_t, sinks_t), jnp.max(s0, axis=1))
                exp_swa = jnp.exp(swa_m_t - m0)
                p0 = jnp.exp(s0 - m0[:, None])
                l0 = (exp_swa * swa_l_t + jnp.exp(sinks_t - m0)) + jnp.sum(p0, axis=1)
                pv0 = jnp.einsum(
                    "nm,md->nd", p0, bkv_0, preferred_element_type=jnp.float32
                )
                acc0 = exp_swa[:, None] * swa_acc_tiles[qt] + pv0
                m0_list.append(m0)
                l0_list.append(l0)
                acc0_list.append(acc0)

            init_state = (
                jnp.stack(m0_list, axis=0),
                jnp.stack(l0_list, axis=0),
                jnp.stack(acc0_list, axis=0),
            )

            def _stream_step(c, state):
                m_prev_all, l_prev_all, acc_prev_all = state
                kv_sem_c = sem_ids_ref[2]
                next_kv_sem_c = lax.select(kv_sem_c == 0, 1, 0)
                has_more = c + 1 < act_chunks

                @pl.when(has_more)
                def _pref_c_next():
                    sem_ids_ref[2] = next_kv_sem_c
                    _fetch_kv_chunk(g, c + 1, next_kv_sem_c, wait=False)

                @pl.when((~has_more) & (next_g < num_active_groups))
                def _pref_ng_c0():
                    sem_ids_ref[2] = next_kv_sem_c
                    _fetch_kv_chunk(next_g, 0, next_kv_sem_c, wait=False)

                _fetch_kv_chunk(g, c, kv_sem_c, wait=True)
                bkv_c = _load_dequant_bkv(kv_sem_c, c * u_chunk)
                bits_c = lax.broadcast_in_dim(
                    cand_bits_x2_ref.at[kv_sem_c][...].reshape(-1),
                    (q_sub, u_chunk),
                    (1,),
                )

                m_list = []
                l_list = []
                acc_list = []
                for qt in range(num_q_tiles):
                    sc = _masked_qk(qt, bkv_c, bits_c)
                    mp = m_prev_all[qt]
                    mn = jnp.maximum(mp, jnp.max(sc, axis=1))
                    alpha = jnp.exp(mp - mn)
                    pc = jnp.exp(sc - mn[:, None])
                    ln = alpha * l_prev_all[qt] + jnp.sum(pc, axis=1)
                    pvc = jnp.einsum(
                        "nm,md->nd", pc, bkv_c, preferred_element_type=jnp.float32
                    )
                    accn = alpha[:, None] * acc_prev_all[qt] + pvc
                    m_list.append(mn)
                    l_list.append(ln)
                    acc_list.append(accn)

                return (
                    jnp.stack(m_list, axis=0),
                    jnp.stack(l_list, axis=0),
                    jnp.stack(acc_list, axis=0),
                )

            _, l_fin_all, acc_fin_all = lax.fori_loop(
                1, act_chunks, _stream_step, init_state
            )
            for qt in range(num_q_tiles):
                l2d = l_fin_all[qt][:, None]
                acc_fin = acc_fin_all[qt]
                out_t = (
                    lax.div(acc_fin, l2d)
                    if q_dtype == jnp.float32
                    else (acc_fin * pl.reciprocal(l2d, approx=True)).astype(q_dtype)
                )
                bo_x2_ref.at[bo_sem_idx, pl.ds(qt * q_sub, q_sub)][...] = out_t

    _send_output(g, bo_sem_idx, wait=False)

    @pl.when(p_id == num_run_groups - 1)
    def _epilogue():
        _wait_send_output(0)

        @pl.when(num_run_groups >= 2)
        def _wait_1():
            _wait_send_output(1)


def _run_chunked_fallback(
    q: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    topk_indices: jax.Array,
    num_valid_tokens: jax.Array,
    attention_sinks: jax.Array,
    swa_accumution: jax.Array,
    swa_l: jax.Array,
    swa_m: jax.Array,
    *,
    sm_scale: float,
    gather_and_attention_chunk_size: int,
    attention_kernel_batch_size: int,
) -> jax.Array:
    """Single-launch / larger-chunk double-buffered path without host HLO loop bloat."""
    _, num_q_heads, head_dim = q.shape
    topk = topk_indices.shape[-1]
    kv_lens = jnp.sum(topk_indices >= 0, axis=-1)
    gather_indices = jnp.where(topk_indices < 0, ~topk_indices, topk_indices).ravel()

    # For decode / verify (<= 384 tokens), process the entire batch in 1 launch
    # while clamping negative padding indices directly on SparseCore in csa_gather
    # so SparseCore launches immediately in parallel with kv_lens on TensorCore.
    if q.shape[0] <= 384:
        eff_chunk_size = q.shape[0]
        if q.shape[0] <= 32 or (q.shape[0] == 96 and num_q_heads > 64):
            eff_batch_size = 8
        elif (q.shape[0] == 384 and 32 <= num_q_heads <= 64) or (
            64 <= q.shape[0] <= 256 and num_q_heads <= 16
        ):
            eff_batch_size = 32
        else:
            eff_batch_size = attention_kernel_batch_size
    elif q.shape[0] <= 1024:
        eff_chunk_size = 256
        eff_batch_size = attention_kernel_batch_size
    else:
        target_chunk = 256 if num_q_heads == 64 else 512
        eff_chunk_size = max(
            gather_and_attention_chunk_size, min(q.shape[0], target_chunk)
        )
        eff_batch_size = attention_kernel_batch_size

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    in_specs = [pl.BlockSpec(memory_space=pltpu.VMEM)] + [hbm] * 6

    for i in range(cdiv(q.shape[0], eff_chunk_size)):
        start_pos = i * eff_chunk_size
        end_pos = min(start_pos + eff_chunk_size, q.shape[0])
        chunk_size = end_pos - start_pos
        indices = gather_indices[start_pos * topk : end_pos * topk]

        batch_size = _largest_divisor(chunk_size, eff_batch_size)
        valid_in_chunk = jnp.maximum(
            0, jnp.minimum(num_valid_tokens, end_pos) - start_pos
        )
        batch_end = start_pos + cdiv(valid_in_chunk, batch_size) * batch_size
        num_valid_indices = jnp.maximum(0, batch_end - start_pos) * topk

        gathered_nope, gathered_rope = csa_gather.csa_gather(
            cache_kv_nope,
            cache_kv_rope,
            indices,
            num_valid_indices=num_valid_indices,
            top_k=topk,
        )
        gathered_nope = gathered_nope.reshape(chunk_size, topk, -1)
        gathered_rope = gathered_rope.reshape(chunk_size, topk // 4, -1)
        lanes = gathered_nope.shape[-1]
        end_seq_idx = jnp.maximum(start_pos, batch_end)
        grid = (cdiv(end_seq_idx - start_pos, batch_size),)
        bq_buf = pltpu.VMEM((2, batch_size, num_q_heads, head_dim), q.dtype)
        bl_buf = pltpu.VMEM((2, batch_size, align_to(num_q_heads, 128)), jnp.float32)
        scratch_shapes = [
            pltpu.VMEM((2, batch_size, topk * 4, lanes), jnp.uint8),
            pltpu.VMEM((2, batch_size, (topk // 4) * 2, lanes), jnp.bfloat16),
            bq_buf,
            bq_buf,
            bl_buf,
            bl_buf,
            bq_buf,
            pltpu.SemaphoreType.DMA((7, 2)),
        ]
        scalar_prefetches = (
            kv_lens,
            jnp.array([start_pos, end_seq_idx, num_valid_tokens], jnp.int32),
            jnp.zeros((2,), jnp.int32),
            jnp.full((2,), -1, jnp.int32),
        )
        scope_name = f"SparseMLA-p_{topk}-bz_{batch_size}-gcz_{chunk_size}"
        q = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _attention_kernel, sm_scale=sm_scale, batch_size=batch_size
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=hbm,
                    grid=grid,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
                    disable_bounds_checks=True,
                ),
                out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                input_output_aliases={5: 0},
                name=scope_name,
            )
        )(
            *scalar_prefetches,
            attention_sinks,
            q,
            gathered_nope,
            gathered_rope,
            swa_accumution,
            swa_l,
            swa_m,
        )
    return q


def _batched_grouped_attention_kernel(
    n_unique_ref,  # i32[num_groups]
    meta_ref,  # i32[2] (start_sb, num_active_sb)
    sem_ids_ref,  # i32[2] (in_sem_idx, out_sem_idx)
    bo_ids_ref,  # i32[2] sb_idx of in-flight bo DMA
    attention_sinks_ref,  # f32[l_sublanes, 128]
    q_sb_hbm_ref,  # bf16[num_sb, g_batch * n_qh, head_dim]
    nope_sb_hbm_ref,  # int32[num_sb, g_batch * u_chunk, 128]
    rope_sb_hbm_ref,  # int32[num_sb, g_batch * (u_chunk // 4), 128]
    bits_sb_hbm_ref,  # int32[num_sb, g_batch * (u_chunk // 128), 128]
    swa_sb_hbm_ref,  # bf16[num_sb, g_batch * n_qh, head_dim]
    swa_l_sb_hbm_ref,  # f32[num_sb, g_batch * l_sublanes, 128]
    swa_m_sb_hbm_ref,  # f32[num_sb, g_batch * l_sublanes, 128]
    o_sb_hbm_ref,  # bf16[num_sb, g_batch * n_qh, head_dim]
    bkv_nope_x2_ref,  # u8[2, g_batch * u_chunk * 4, 128]
    bkv_rope_x2_ref,  # bf16[2, g_batch * (u_chunk // 2), 128]
    bq_x2_ref,  # bf16[2, g_batch * n_qh, head_dim]
    bo_x2_ref,  # bf16[2, g_batch * n_qh, head_dim]
    bl_x2_ref,  # f32[2, g_batch * l_sublanes, 128]
    bm_x2_ref,  # f32[2, g_batch * l_sublanes, 128]
    swa_acc_x2_ref,  # bf16[2, g_batch * n_qh, head_dim]
    cand_bits_x2_ref,  # int32[2, g_batch * (u_chunk // 128), 128]
    sems,  # DMA[(8, 2)]
    *,
    sm_scale: float,
    b_q: int,
    u_chunk: int,
    g_batch: int,
    l_sublanes: int,
):
    """Multi-group super-block CSA candidate-union kernel coalescing g_batch groups per burst DMA."""
    _, tot_qh, head_dim = q_sb_hbm_ref.shape
    n_qh = tot_qh // g_batch
    num_q_heads = n_qh // b_q
    q_dtype = q_sb_hbm_ref.dtype
    nope_sb_hbm_ref = nope_sb_hbm_ref.bitcast(jnp.uint8)
    rope_sb_hbm_ref = rope_sb_hbm_ref.bitcast(jnp.bfloat16)

    start_sb = meta_ref[0]
    num_active_sb = meta_ref[1]
    num_run_sb = num_active_sb - start_sb
    p_id = pl.program_id(0)
    sb = start_sb + p_id

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_sb_all(sb_idx, sem_idx, *, wait=False):
        pairs = (
            (0, nope_sb_hbm_ref.at[sb_idx], bkv_nope_x2_ref.at[sem_idx]),
            (1, q_sb_hbm_ref.at[sb_idx], bq_x2_ref.at[sem_idx]),
            (3, swa_sb_hbm_ref.at[sb_idx], swa_acc_x2_ref.at[sem_idx]),
            (4, swa_l_sb_hbm_ref.at[sb_idx], bl_x2_ref.at[sem_idx]),
            (5, swa_m_sb_hbm_ref.at[sb_idx], bm_x2_ref.at[sem_idx]),
            (6, rope_sb_hbm_ref.at[sb_idx], bkv_rope_x2_ref.at[sem_idx]),
            (7, bits_sb_hbm_ref.at[sb_idx], cand_bits_x2_ref.at[sem_idx]),
        )
        for s_id, hbm_src, vmem_dst in pairs:
            if not wait:
                _async_copy(hbm_src, vmem_dst, sems.at[s_id, sem_idx], False)
            else:
                _async_copy(vmem_dst, vmem_dst, sems.at[s_id, sem_idx], True)

    def _send_sb_out(sb_idx, sem_idx, *, wait=False):
        vmem_ref = bo_x2_ref.at[sem_idx]
        sem = sems.at[2, sem_idx]
        if not wait:
            bo_ids_ref[sem_idx] = sb_idx
            _async_copy(vmem_ref, o_sb_hbm_ref.at[sb_idx], sem, False)
        else:
            _async_copy(vmem_ref, vmem_ref, sem, True)

    @pl.when(p_id == 0)
    def _prologue():
        _fetch_sb_all(start_sb, 0, wait=False)

    in_sem = sem_ids_ref[0]
    next_sb = sb + 1
    next_in_sem = lax.select(in_sem == 0, 1, 0)

    @pl.when(next_sb < num_active_sb)
    def _prefetch_next_sb():
        sem_ids_ref[0] = next_in_sem
        _fetch_sb_all(next_sb, next_in_sem, wait=False)

    out_sem = sem_ids_ref[1]
    sem_ids_ref[1] = lax.select(out_sem == 0, 1, 0)

    _fetch_sb_all(sb, in_sem, wait=True)

    @pl.when(p_id >= 2)
    def _wait_prev_send():
        old_sb = bo_ids_ref[out_sem]

        @pl.when(old_sb >= 0)
        def _():
            _send_sb_out(old_sb, out_sem, wait=True)

    lane_128 = lax.broadcasted_iota(jnp.int32, (u_chunk, 128), 1)
    m_span_u = lax.broadcasted_iota(jnp.int32, (u_chunk, head_dim), 0)
    q_heads_per_tile = _largest_divisor(b_q, max(1, 1024 // num_q_heads))
    q_sub = q_heads_per_tile * num_q_heads
    num_q_tiles = n_qh // q_sub
    q_iota_base = lax.broadcasted_iota(jnp.int32, (q_sub, u_chunk), 0) // num_q_heads
    sinks_all = jnp.tile(
        attention_sinks_ref[...].reshape(-1)[:num_q_heads], (b_q,)
    ).reshape(num_q_tiles, q_sub)

    nope_all = bkv_nope_x2_ref.at[in_sem][...].reshape(g_batch, u_chunk, -1)
    rope_all = bkv_rope_x2_ref.at[in_sem][...].reshape(g_batch, u_chunk // 2, -1)
    q_all = bq_x2_ref.at[in_sem][...].reshape(g_batch, num_q_tiles, q_sub, head_dim)
    swa_acc_all = swa_acc_x2_ref.at[in_sem][...].reshape(
        g_batch, num_q_tiles, q_sub, head_dim
    )
    bl_all = (
        bl_x2_ref.at[in_sem][...]
        .reshape(g_batch, b_q, -1)[:, :, :num_q_heads]
        .reshape(g_batch, num_q_tiles, q_sub)
    )
    bm_all = (
        bm_x2_ref.at[in_sem][...]
        .reshape(g_batch, b_q, -1)[:, :, :num_q_heads]
        .reshape(g_batch, num_q_tiles, q_sub)
    )
    bits_all = cand_bits_x2_ref.at[in_sem][...].reshape(g_batch, u_chunk)

    def _dequant_grp(idx):
        g_idx = sb * g_batch + idx
        n_uniq_i = jnp.clip(n_unique_ref[g_idx], 0, u_chunk)
        if nope_all.shape[2] == 512 and rope_all.shape[2] == 128:
            bkv_i = _dequant_and_merge_bkv(nope_all[idx], rope_all[idx], lane_128)
        else:
            nope_i = _dequant_dsv4_fp8(nope_all[idx])
            rd = rope_all.shape[-1] // 2
            rope_i = jnp.concatenate(
                [rope_all[idx, :, :rd], rope_all[idx, :, rd:]], axis=0
            )
            bkv_i = jnp.concatenate([nope_i, rope_i], axis=-1)
        return jnp.where(m_span_u < n_uniq_i, bkv_i, 0)

    bkv_k = _dequant_grp(0)
    for k in range(g_batch):
        bits_k = lax.broadcast_in_dim(bits_all[k], (q_sub, u_chunk), (1,))
        swa_m_k = bm_all[k]
        swa_l_k = bl_all[k]

        def _masked_qk_k(qt, bkv_c=bkv_k, bits_c=bits_k, q_grp=q_all[k]):
            s_c = (
                jnp.einsum(
                    "nd,md->nm",
                    q_grp[qt],
                    bkv_c,
                    preferred_element_type=jnp.float32,
                )
                * sm_scale
            )
            q_bit = jnp.left_shift(jnp.int32(1), qt * q_heads_per_tile + q_iota_base)
            return jnp.where(
                jnp.bitwise_and(bits_c, q_bit) != 0,
                s_c,
                jnp.finfo(s_c.dtype).min,
            )

        s_cur = _masked_qk_k(0)
        bkv_next = _dequant_grp(k + 1) if k + 1 < g_batch else None
        for qt in range(num_q_tiles):
            swa_m_t = swa_m_k[qt]
            swa_l_t = swa_l_k[qt]
            sinks_t = sinks_all[qt]
            swa_acc_t = swa_acc_all[k, qt]
            p_dtype = q_dtype if _FAST_GROUPED_ATTN else jnp.float32
            if b_q < 8:
                m_new = jnp.maximum(_m_run(swa_m_t, sinks_t), jnp.max(s_cur, axis=1))
                exp_swa = jnp.exp(swa_m_t - m_new)
                p = jnp.exp(s_cur - m_new[:, None])
                s_nxt = _masked_qk_k(qt + 1) if qt + 1 < num_q_tiles else None
                l_new = (exp_swa * swa_l_t + jnp.sum(p, axis=1)) + jnp.exp(
                    sinks_t - m_new
                )
                pv = jnp.einsum(
                    "nm,md->nd",
                    p.astype(p_dtype),
                    bkv_k,
                    preferred_element_type=jnp.float32,
                )
                acc_new = exp_swa[:, None] * swa_acc_t + pv
            else:
                m_init = jnp.maximum(swa_m_t, sinks_t)
                exp_swa = jnp.exp(swa_m_t - m_init)
                l_init = exp_swa * swa_l_t + jnp.exp(sinks_t - m_init)
                m_new = jnp.maximum(m_init, jnp.max(s_cur, axis=1))
                alpha = jnp.exp(m_init - m_new)
                p = jnp.exp(s_cur - m_new[:, None])
                s_nxt = _masked_qk_k(qt + 1) if qt + 1 < num_q_tiles else None
                l_new = alpha * l_init + jnp.sum(p, axis=1)
                pv = jnp.einsum(
                    "nm,md->nd",
                    p.astype(p_dtype),
                    bkv_k,
                    preferred_element_type=jnp.float32,
                )
                acc_new = (alpha * exp_swa)[:, None] * swa_acc_t + pv
            l_new_2d = l_new[:, None]
            out_t = (
                lax.div(acc_new, l_new_2d)
                if q_dtype == jnp.float32
                else (acc_new * pl.reciprocal(l_new_2d, approx=True)).astype(q_dtype)
            )
            bo_x2_ref.at[out_sem, pl.ds(k * n_qh + qt * q_sub, q_sub)][...] = out_t
            s_cur = s_nxt
        bkv_k = bkv_next

    _send_sb_out(sb, out_sem, wait=False)

    @pl.when(p_id == num_run_sb - 1)
    def _epilogue():
        _send_sb_out(0, 0, wait=True)

        @pl.when(num_run_sb >= 2)
        def _wait_1():
            _send_sb_out(0, 1, wait=True)


def _launch_grouped_or_batched_attention(
    q_slice: jax.Array,
    swa_slice: jax.Array,
    swa_l_aligned: jax.Array,
    swa_m_aligned: jax.Array,
    attention_sinks: jax.Array,
    g_nope: jax.Array,
    g_rope: jax.Array,
    c_bits: jax.Array,
    n_uniq: jax.Array,
    *,
    num_groups: int,
    start_grp: jax.Array | int,
    act_grps: jax.Array | int,
    num_valid_toks: jax.Array | int,
    b_q: int,
    m_union: int,
    u_chunk: int,
    sm_scale: float,
) -> jax.Array:
    """Dispatches single-chunk candidate union to super-block coalesced kernel with zero-copy HBM views."""
    _, num_q_heads, head_dim = q_slice.shape
    n_qh = b_q * num_q_heads
    lanes = g_nope.shape[-1]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    if _FAST_GROUPED_ATTN:
        target_gb = (
            8
            if (num_q_heads <= 16 or b_q == 1)
            else (4 if (num_q_heads <= 64 or b_q <= 2) else 2)
        )
        g_batch = (
            _largest_divisor(num_groups, target_gb)
            if (m_union == u_chunk and num_q_heads <= 128)
            else 1
        )
        vmem_limit = 60 * 1024 * 1024 if g_batch >= 2 else DEFAULT_VMEM_LIMIT_BYTES
    else:
        target_gb = (
            (4 if num_q_heads <= 16 else 2)
            if m_union <= 768
            else (2 if num_q_heads <= 64 else 1)
        )
        g_batch = (
            _largest_divisor(num_groups, target_gb)
            if (b_q < 8 and m_union == u_chunk and num_q_heads <= 64)
            else 1
        )
        vmem_limit = DEFAULT_VMEM_LIMIT_BYTES

    if g_batch >= 2:
        num_sb = num_groups // g_batch
        start_sb = start_grp // g_batch
        act_sb = jnp.maximum(start_sb + 1, cdiv(act_grps, g_batch))
        l_row_w = swa_l_aligned.shape[-1]
        sinks_pad = jnp.pad(
            attention_sinks, (0, align_to(num_q_heads, 128) - num_q_heads)
        ).reshape(-1, 128)
        q_sb = q_slice.reshape(num_sb, g_batch * n_qh, head_dim)
        bq_sb = pltpu.VMEM((2, g_batch * n_qh, head_dim), q_slice.dtype)
        bl_sb = pltpu.VMEM((2, g_batch * b_q, l_row_w), jnp.float32)
        sb_scratch = [
            pltpu.VMEM((2, g_batch * m_union * 4, lanes), jnp.uint8),
            pltpu.VMEM((2, g_batch * (m_union // 2), lanes), jnp.bfloat16),
            bq_sb,
            bq_sb,
            bl_sb,
            bl_sb,
            bq_sb,
            pltpu.VMEM((2, g_batch * (m_union // 128), 128), jnp.int32),
            pltpu.SemaphoreType.DMA((8, 2)),
        ]
        out_sb = pl.pallas_call(
            functools.partial(
                _batched_grouped_attention_kernel,
                sm_scale=sm_scale,
                b_q=b_q,
                u_chunk=m_union,
                g_batch=g_batch,
                l_sublanes=b_q * (l_row_w // 128),
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=4,
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] + [hbm] * 7,
                out_specs=hbm,
                grid=(jnp.maximum(0, act_sb - start_sb),),
                scratch_shapes=sb_scratch,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit,
                disable_bounds_checks=True,
            ),
            out_shape=jax.ShapeDtypeStruct(shape=q_sb.shape, dtype=q_sb.dtype),
            input_output_aliases={5: 0},
            name=f"SparseMLA-BatchedGrouped-bq_{b_q}-gb_{g_batch}-mu_{m_union}",
        )(
            n_uniq,
            jnp.array([start_sb, act_sb], jnp.int32),
            jnp.zeros((2,), jnp.int32),
            jnp.full((2,), -1, jnp.int32),
            sinks_pad,
            q_sb,
            g_nope.reshape(num_sb, g_batch * m_union, lanes),
            g_rope.reshape(num_sb, g_batch * (m_union // 4), lanes),
            c_bits.reshape(num_sb, g_batch * (m_union // 128), 128),
            swa_slice.reshape(num_sb, g_batch * n_qh, head_dim),
            swa_l_aligned.reshape(num_sb, g_batch * b_q, l_row_w),
            swa_m_aligned.reshape(num_sb, g_batch * b_q, l_row_w),
        )
        return out_sb.reshape(num_groups * b_q, num_q_heads, head_dim)

    l_stride = align_to(n_qh, 128)
    l_sub = l_stride // 128
    swa_l_qh = swa_l_aligned[:, :num_q_heads]
    swa_m_qh = swa_m_aligned[:, :num_q_heads]
    if l_stride == n_qh:
        swa_l_flat = swa_l_qh.reshape(num_groups, l_sub, 128)
        swa_m_flat = swa_m_qh.reshape(num_groups, l_sub, 128)
        sinks_flat = jnp.tile(attention_sinks, (b_q,)).reshape(l_sub, 128)
    else:
        swa_l_flat = jnp.pad(
            swa_l_qh.reshape(num_groups, n_qh), ((0, 0), (0, l_stride - n_qh))
        ).reshape(num_groups, l_sub, 128)
        swa_m_flat = jnp.pad(
            swa_m_qh.reshape(num_groups, n_qh), ((0, 0), (0, l_stride - n_qh))
        ).reshape(num_groups, l_sub, 128)
        sinks_flat = jnp.pad(
            jnp.tile(attention_sinks, (b_q,)), (0, l_stride - n_qh)
        ).reshape(l_sub, 128)
    num_u_c = m_union // u_chunk
    bq_b = pltpu.VMEM((2, n_qh, head_dim), q_slice.dtype)
    bl_b = pltpu.VMEM((2, l_sub, 128), jnp.float32)
    g_scratch = [
        pltpu.VMEM((2, u_chunk * 4, lanes), jnp.uint8),
        pltpu.VMEM((2, u_chunk // 2, lanes), jnp.bfloat16),
        bq_b,
        bq_b,
        bl_b,
        bl_b,
        bq_b,
        pltpu.VMEM((2, u_chunk // 128, 128), jnp.int32),
        pltpu.SemaphoreType.DMA((8, 2)),
    ]
    out_g = pl.pallas_call(
        functools.partial(
            _grouped_attention_kernel,
            sm_scale=sm_scale,
            b_q=b_q,
            m_union=m_union,
            u_chunk=u_chunk,
            l_stride=l_stride,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] + [hbm] * 7,
            out_specs=hbm,
            grid=(jnp.maximum(0, act_grps - start_grp),),
            scratch_shapes=g_scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
            disable_bounds_checks=True,
        ),
        out_shape=jax.ShapeDtypeStruct(
            shape=(num_groups, n_qh, head_dim), dtype=q_slice.dtype
        ),
        input_output_aliases={5: 0},
        name=f"SparseMLA-Grouped-bq_{b_q}-mu_{m_union}-uc_{u_chunk}",
    )(
        n_uniq,
        jnp.array([start_grp, act_grps, num_valid_toks], jnp.int32),
        jnp.zeros((3,), jnp.int32),
        jnp.full((2,), -1, jnp.int32),
        sinks_flat,
        q_slice.reshape(num_groups, n_qh, head_dim),
        g_nope.reshape(num_groups, num_u_c, u_chunk, -1),
        g_rope.reshape(num_groups, num_u_c, u_chunk // 4, -1),
        c_bits.reshape(num_groups, num_u_c, u_chunk // 128, 128),
        swa_slice.reshape(num_groups, n_qh, head_dim),
        swa_l_flat,
        swa_m_flat,
    )
    return out_g.reshape(num_groups * b_q, num_q_heads, head_dim)


# The compressor writes this step's compressed KV tokens to the cache before
# this runs, so the encoded indices may point at them.
@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "attention_kernel_batch_size",
        "gather_and_attention_chunk_size",
        "ced_decoder",
    ),
)
def sparse_ragged_paged_attention(
    q: jax.Array,  # [max_num_tokens, actual_num_q_heads, head_dim]
    cache_kv_nope: jax.Array,  # int32[total_num_pages, page_size, 128]
    cache_kv_rope: jax.Array,  # int32[total_num_pages, page_size // 4, 128]
    topk_indices: jax.Array,  # i32[max_num_tokens, csa_topk], already encoded
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    distribution: jax.Array,  # i32[3]
    attention_sinks: jax.Array,  # float32[actual_num_q_heads]
    swa_accumution: jax.Array,  # bf16[max_num_tokens, num_q_heads, head_dim]
    swa_l: jax.Array,  # float32[max_num_tokens, num_q_heads]
    swa_m: jax.Array,  # float32[max_num_tokens, num_q_heads]
    *,
    precomputed_union: tuple[jax.Array, ...] | None = None,
    sm_scale: float = 1.0,
    # Kernel optimization params.
    gather_and_attention_chunk_size: int = 128,
    attention_kernel_batch_size: int = 16,
    ced_decoder: bool = False,
) -> jax.Array:
    """DeepSeek-V4 sparse_ragged_paged_attention on encode_csa_gather_indices output."""
    _, actual_num_q_heads, actual_head_dim = q.shape

    q = prepare_q_inputs(q)  # [max_num_tokens, num_q_heads, head_dim]
    num_tokens, num_q_heads, head_dim = q.shape
    if num_q_heads != actual_num_q_heads:
        attention_sinks = jnp.pad(
            attention_sinks,
            (0, num_q_heads - actual_num_q_heads),
            constant_values=jnp.finfo(attention_sinks.dtype).min,
        )
    assert swa_accumution.dtype == q.dtype

    topk = topk_indices.shape[-1]
    max_num_seqs = cu_q_lens.shape[0] - 1
    num_valid_tokens = cu_q_lens[distribution[2]]
    total_cache_rows = cache_kv_nope.shape[0] * cache_kv_nope.shape[1]

    def _run_fallback_branch(q_in: jax.Array) -> jax.Array:
        swa_acc_fb, swa_l_fb, swa_m_fb = prepare_swa_inputs(
            swa_accumution, swa_l, swa_m
        )
        return _run_chunked_fallback(
            q_in,
            cache_kv_nope,
            cache_kv_rope,
            topk_indices,
            num_valid_tokens,
            attention_sinks,
            swa_acc_fb,
            swa_l_fb,
            swa_m_fb,
            sm_scale=sm_scale,
            gather_and_attention_chunk_size=gather_and_attention_chunk_size,
            attention_kernel_batch_size=attention_kernel_batch_size,
        )

    if precomputed_union is not None and actual_num_q_heads <= 128:
        if len(precomputed_union) == 5:
            (
                union_gather_indices,
                cand_bits_tiled,
                n_unique,
                pre_gathered_nope,
                pre_gathered_rope,
            ) = precomputed_union
        else:
            union_gather_indices, cand_bits_tiled, n_unique = precomputed_union
            pre_gathered_nope = None
            pre_gathered_rope = None
        num_groups = int(n_unique.shape[0])
        b_q = num_tokens // num_groups
        m_union = int(union_gather_indices.shape[0]) // num_groups
        u_chunk = resolve_csa_union_u_chunk(b_q, m_union, topk)
        num_decode_tokens = cu_q_lens[distribution[0]]
        start_group = cdiv(num_decode_tokens, b_q)
        num_active_groups = jnp.maximum(
            start_group, jnp.maximum(1, cdiv(num_valid_tokens, b_q))
        )
        swa_acc_grp, swa_l_aligned, swa_m_aligned = prepare_swa_inputs(
            swa_accumution, swa_l, swa_m
        )

        grp_iota = jnp.arange(num_groups, dtype=jnp.int32)
        active_grp_mask = (grp_iota >= start_group) & (grp_iota < num_active_groups)
        all_groups_fit = jnp.all(jnp.where(active_grp_mask, n_unique, 0) <= m_union)

        def _run_explicit_grouped(q_orig: jax.Array) -> jax.Array:
            lead_tok_end = start_group * b_q

            def _attend_leading_decode_groups(q_in_lead: jax.Array) -> jax.Array:
                dec_cap = min(num_tokens, align_to(max(8, max_num_seqs), 64))
                dec_out = _run_chunked_fallback(
                    q_in_lead[:dec_cap],
                    cache_kv_nope,
                    cache_kv_rope,
                    topk_indices[:dec_cap],
                    jnp.minimum(lead_tok_end, num_valid_tokens),
                    attention_sinks,
                    swa_acc_grp[:dec_cap],
                    swa_l_aligned[:dec_cap],
                    swa_m_aligned[:dec_cap],
                    sm_scale=sm_scale,
                    gather_and_attention_chunk_size=dec_cap,
                    attention_kernel_batch_size=attention_kernel_batch_size,
                )
                return (
                    dec_out
                    if dec_cap == num_tokens
                    else q_in_lead.at[:dec_cap].set(dec_out)
                )

            q_orig = lax.cond(
                lead_tok_end > 0, _attend_leading_decode_groups, lambda x: x, q_orig
            )
            if pre_gathered_nope is not None and pre_gathered_rope is not None:
                gathered_nope, gathered_rope = pre_gathered_nope, pre_gathered_rope
            else:
                gathered_nope, gathered_rope = csa_gather.csa_gather(
                    cache_kv_nope,
                    cache_kv_rope,
                    union_gather_indices,
                    num_valid_indices=num_active_groups * m_union,
                    top_k=u_chunk,
                )
            return _launch_grouped_or_batched_attention(
                q_orig,
                swa_acc_grp,
                swa_l_aligned,
                swa_m_aligned,
                attention_sinks,
                gathered_nope,
                gathered_rope,
                cand_bits_tiled,
                n_unique,
                num_groups=num_groups,
                start_grp=start_group,
                act_grps=num_active_groups,
                num_valid_toks=num_valid_tokens,
                b_q=b_q,
                m_union=m_union,
                u_chunk=u_chunk,
                sm_scale=sm_scale,
            )

        if b_q * topk <= m_union:
            q = _run_explicit_grouped(q)
        else:
            q = lax.cond(all_groups_fit, _run_explicit_grouped, _run_fallback_branch, q)
    elif (
        envs.VLLM_DSV41_CSA_UNION
        and topk % 128 == 0
        and num_tokens <= _CSA_UNION_MAX_TOKENS
        and actual_num_q_heads <= 128
    ):
        swa_acc_grp, swa_l_aligned, swa_m_aligned = prepare_swa_inputs(
            swa_accumution, swa_l, swa_m
        )
        swa_l_qh = swa_l_aligned[:, :num_q_heads]
        swa_m_qh = swa_m_aligned[:, :num_q_heads]

        ver_bq = _CSA_UNION_VERIFY_BQ
        if ver_bq <= 0:
            try:
                from vllm_torchtpu.layers.adapter.utils import (
                    get_vllm_model_wrapper_context,
                )

                vllm_cfg = getattr(
                    get_vllm_model_wrapper_context(), "vllm_config", None
                )
                spec_cfg = getattr(vllm_cfg, "speculative_config", None)
                num_spec = getattr(spec_cfg, "num_speculative_tokens", None)
                if isinstance(num_spec, int) and num_spec >= 1:
                    ver_bq = num_spec + 1
                elif vllm_cfg is None:
                    ver_bq = 6
            except Exception:
                ver_bq = 6

        decode_1tok_end = cu_q_lens[distribution[0]]
        if ver_bq >= 2 and num_tokens >= ver_bq:
            q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
            seq_iota = jnp.arange(max_num_seqs, dtype=jnp.int32)
            is_ver_seq = (
                (seq_iota >= distribution[0])
                & (seq_iota < distribution[2])
                & (q_lens == ver_bq)
            )
            first_non_ver = jnp.argmin(
                jnp.where(seq_iota >= distribution[0], is_ver_seq, True)
            )
            verify_end_seq = jnp.where(
                jnp.all(jnp.where(seq_iota >= distribution[0], is_ver_seq, True)),
                distribution[2],
                first_non_ver,
            )
            verify_tok_end = cu_q_lens[verify_end_seq]
        else:
            verify_tok_end = decode_1tok_end

        # Case 1: Pure Decode slice [0 : decode_1tok_end] (q_len == 1 per request)
        def _run_decode_1tok_stage(q_in: jax.Array) -> jax.Array:
            dec_cap = min(num_tokens, align_to(max(8, max_num_seqs), 64))
            dec_out = _run_chunked_fallback(
                q_in[:dec_cap],
                cache_kv_nope,
                cache_kv_rope,
                topk_indices[:dec_cap],
                decode_1tok_end,
                attention_sinks,
                swa_acc_grp[:dec_cap],
                swa_l_aligned[:dec_cap],
                swa_m_aligned[:dec_cap],
                sm_scale=sm_scale,
                gather_and_attention_chunk_size=dec_cap,
                attention_kernel_batch_size=attention_kernel_batch_size,
            )
            return dec_out if dec_cap == num_tokens else q_in.at[:dec_cap].set(dec_out)

        q = lax.cond(decode_1tok_end > 0, _run_decode_1tok_stage, lambda x: x, q)

        # Case 2: Spec-Verify slice [decode_1tok_end : verify_tok_end] (q_len == ver_bq, SparseCore sc_union_ztc, m_union = 1024)
        if ver_bq >= 2 and num_tokens >= ver_bq:
            ver_cap = (
                min(num_tokens, max(ver_bq, max_num_seqs * ver_bq)) // ver_bq
            ) * ver_bq
            ver_groups = ver_cap // ver_bq
            ver_m_union = (
                min(envs.VLLM_DSV41_CSA_UNION_M, ver_bq * topk)
                if envs.VLLM_DSV41_CSA_UNION_M > 0
                else align_to(2 * topk, 128)
            )
            ver_u_chunk = resolve_csa_union_u_chunk(ver_bq, ver_m_union, topk)

            def _run_verify_union_stage(q_in: jax.Array) -> jax.Array:
                ver_q_slice = q_in[:ver_cap]
                ver_tk = topk_indices[:ver_cap]
                u_idx, c_bits, n_uniq = _build_candidate_union_and_bitmask(
                    ver_tk, ver_bq, ver_m_union, total_cache_rows, u_chunk=ver_u_chunk
                )
                ver_start_grp = decode_1tok_end // ver_bq
                ver_act_grps = jnp.maximum(
                    ver_start_grp + 1, cdiv(verify_tok_end, ver_bq)
                )
                v_iota = jnp.arange(ver_groups, dtype=jnp.int32)
                active_v = (v_iota >= ver_start_grp) & (v_iota < ver_act_grps)
                all_v_fit = jnp.all(jnp.where(active_v, n_uniq, 0) <= ver_m_union)

                def _run_v_grouped(q_sub_in: jax.Array) -> jax.Array:
                    g_nope, g_rope = csa_gather.csa_gather(
                        cache_kv_nope,
                        cache_kv_rope,
                        u_idx,
                        num_valid_indices=ver_act_grps * ver_m_union,
                        top_k=ver_u_chunk,
                    )
                    return _launch_grouped_or_batched_attention(
                        q_sub_in,
                        swa_acc_grp[:ver_cap],
                        swa_l_aligned[:ver_cap],
                        swa_m_aligned[:ver_cap],
                        attention_sinks,
                        g_nope,
                        g_rope,
                        c_bits,
                        n_uniq,
                        num_groups=ver_groups,
                        start_grp=ver_start_grp,
                        act_grps=ver_act_grps,
                        num_valid_toks=verify_tok_end,
                        b_q=ver_bq,
                        m_union=ver_m_union,
                        u_chunk=ver_u_chunk,
                        sm_scale=sm_scale,
                    )

                def _run_v_fallback(q_sub_in: jax.Array) -> jax.Array:
                    return _run_chunked_fallback(
                        q_sub_in,
                        cache_kv_nope,
                        cache_kv_rope,
                        ver_tk,
                        verify_tok_end,
                        attention_sinks,
                        swa_acc_grp[:ver_cap],
                        swa_l_aligned[:ver_cap],
                        swa_m_aligned[:ver_cap],
                        sm_scale=sm_scale,
                        gather_and_attention_chunk_size=min(
                            ver_cap, gather_and_attention_chunk_size
                        ),
                        attention_kernel_batch_size=attention_kernel_batch_size,
                    )

                ver_out = lax.cond(
                    all_v_fit, _run_v_grouped, _run_v_fallback, ver_q_slice
                )
                return (
                    ver_out if ver_cap == num_tokens else q_in.at[:ver_cap].set(ver_out)
                )

            q = lax.cond(
                verify_tok_end > decode_1tok_end,
                _run_verify_union_stage,
                lambda x: x,
                q,
            )

        # Case 3: Prefill / Straddling tail slice [verify_tok_end : num_valid_tokens]
        # Tier 1 fast path: pf_m_union_fast = align_to(2 * topk, 128) = 1024, u_chunk = 1024
        # (SparseCore sc_union_ztc + super-block _batched_grouped_attention_kernel when all
        # active groups satisfy n_uniq <= 1024).
        # Tier 2 zero-overflow fallback: pf_m_union = pf_bq * topk = 4096, u_chunk = 512.
        _cfg_pf_bq = (
            envs.VLLM_DSV41_CSA_UNION_BQ if envs.VLLM_DSV41_CSA_UNION_BQ > 0 else 8
        )
        if num_tokens >= _cfg_pf_bq and num_tokens % _cfg_pf_bq == 0:
            pf_bq = _cfg_pf_bq
            pf_m_union_fast = (
                min(envs.VLLM_DSV41_CSA_UNION_M, pf_bq * topk)
                if envs.VLLM_DSV41_CSA_UNION_M > 0
                else align_to(2 * topk, 128)
            )
            pf_u_chunk_fast = resolve_csa_union_u_chunk(pf_bq, pf_m_union_fast, topk)
            pf_m_union = pf_bq * topk  # 4096: mathematical 0% overflow guarantee
            pf_u_chunk = resolve_csa_union_u_chunk(pf_bq, pf_m_union, topk)  # 512
            pf_groups = num_tokens // pf_bq

            def _run_prefill_union_stage(q_in: jax.Array) -> jax.Array:
                pf_start_grp = verify_tok_end // pf_bq
                aligned_start = pf_start_grp * pf_bq
                saved_boundary = lax.dynamic_slice_in_dim(
                    q_in, aligned_start, pf_bq, axis=0
                )
                pf_act_grps = jnp.maximum(
                    pf_start_grp + 1, cdiv(num_valid_tokens, pf_bq)
                )
                u_idx_f, c_bits_f, n_uniq_f = _build_candidate_union_and_bitmask(
                    topk_indices,
                    pf_bq,
                    pf_m_union_fast,
                    total_cache_rows,
                    u_chunk=pf_u_chunk_fast,
                )
                pf_iota = jnp.arange(pf_groups, dtype=jnp.int32)
                active_pf = (pf_iota >= pf_start_grp) & (pf_iota < pf_act_grps)
                all_pf_fit_1024 = jnp.all(
                    jnp.where(active_pf, n_uniq_f, 0) <= pf_m_union_fast
                )

                def _invoke_pf_grouped(
                    q_slice_p: jax.Array,
                    u_idx_arg: jax.Array,
                    c_bits_arg: jax.Array,
                    n_uniq_arg: jax.Array,
                    m_u: int,
                    u_c: int,
                ) -> jax.Array:
                    g_nope_p, g_rope_p = csa_gather.csa_gather(
                        cache_kv_nope,
                        cache_kv_rope,
                        u_idx_arg,
                        num_valid_indices=pf_act_grps * m_u,
                        top_k=u_c,
                    )
                    return _launch_grouped_or_batched_attention(
                        q_slice_p,
                        swa_acc_grp,
                        swa_l_aligned,
                        swa_m_aligned,
                        attention_sinks,
                        g_nope_p,
                        g_rope_p,
                        c_bits_arg,
                        n_uniq_arg,
                        num_groups=pf_groups,
                        start_grp=pf_start_grp,
                        act_grps=pf_act_grps,
                        num_valid_toks=num_valid_tokens,
                        b_q=pf_bq,
                        m_union=m_u,
                        u_chunk=u_c,
                        sm_scale=sm_scale,
                    )

                def _run_pf_fast_1024(q_slice_p: jax.Array) -> jax.Array:
                    return _invoke_pf_grouped(
                        q_slice_p,
                        u_idx_f,
                        c_bits_f,
                        n_uniq_f,
                        pf_m_union_fast,
                        pf_u_chunk_fast,
                    )

                def _run_pf_full_4096(q_slice_p: jax.Array) -> jax.Array:
                    u_idx_p, c_bits_p, n_uniq_p = _build_candidate_union_and_bitmask(
                        topk_indices,
                        pf_bq,
                        pf_m_union,
                        total_cache_rows,
                        u_chunk=pf_u_chunk,
                    )
                    return _invoke_pf_grouped(
                        q_slice_p,
                        u_idx_p,
                        c_bits_p,
                        n_uniq_p,
                        pf_m_union,
                        pf_u_chunk,
                    )

                if pf_m_union_fast >= pf_m_union:
                    out_pf = _run_pf_fast_1024(q_in)
                else:
                    out_pf = lax.cond(
                        all_pf_fit_1024, _run_pf_fast_1024, _run_pf_full_4096, q_in
                    )
                new_boundary = lax.dynamic_slice_in_dim(
                    out_pf, aligned_start, pf_bq, axis=0
                )
                keep_prev = (
                    aligned_start + jnp.arange(pf_bq, dtype=jnp.int32)[:, None, None]
                ) < verify_tok_end
                merged_boundary = jnp.where(keep_prev, saved_boundary, new_boundary)
                return lax.dynamic_update_slice_in_dim(
                    out_pf, merged_boundary, aligned_start, axis=0
                )

            q = lax.cond(
                num_valid_tokens > verify_tok_end,
                _run_prefill_union_stage,
                lambda x: x,
                q,
            )
    else:
        q = _run_fallback_branch(q)

    return prepare_outputs(q, actual_num_q_heads, actual_head_dim)
