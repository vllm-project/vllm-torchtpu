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
"""Fast QSA decode path: VMEM index translation + 2-level SparseCore gather +
0-transpose Grouped QSA Sparse-GQA Projected Attention with fused W_o VMEM
prefetch and HC accumulation.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.experimental.batched_rpa import utils as rpa_utils
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16
from vllm_torchtpu.kernels.qwen4_exp.tuned_block_sizes import _hardware
from vllm_torchtpu.kernels.sparse_gqa_decode.projection_kernel import project_tile


def can_use_fast_grouped_qsa_decode(
    qgate: jax.Array,
    main_pool: jax.Array,
    main_table: jax.Array,
    context: jax.Array,
    compress_ratio: int,
    token_topk: int,
    head_dim: int,
    output_weight: jax.Array | None = None,
    residual: jax.Array | None = None,
    hc_gate: jax.Array | None = None,
) -> bool:
    """Return True when the compile bucket matches the fast QSA decode path."""
    if _hardware() is None:
        return False
    info = pltpu.get_tpu_info()
    if info.sparse_core is None:
        return False
    batch = len(qgate)
    if not (
        batch > 0
        and batch <= min(len(context), 32)
        and batch % 8 == 0
        and main_pool.dtype == jnp.float8_e4m3fn
        and main_pool.shape[2:] == (1, 4, head_dim)
        and head_dim == 256
        and compress_ratio == 4
        and token_topk == 2048
        and main_table.shape[1] <= 128
    ):
        return False
    if output_weight is not None:
        return (
            output_weight.ndim == 3
            and output_weight.dtype == jnp.bfloat16
            and residual is not None
            and residual.dtype == jnp.bfloat16
            and hc_gate is not None
            and hc_gate.dtype == jnp.float32
        )
    return True


def _floor_divmod_const(x: jax.Array, d: int) -> tuple[jax.Array, jax.Array]:
    """Exact integer `(x // d, x % d)` using reciprocal float32 multiplication."""
    if d & (d - 1) == 0:
        shift = d.bit_length() - 1
        return lax.shift_right_arithmetic(x, jnp.int32(shift)), x & jnp.int32(d - 1)
    inv = np.float32(1.0 / d)
    q = jnp.floor(x.astype(jnp.float32) * inv).astype(jnp.int32)
    r = x - q * jnp.int32(d)
    q2 = jnp.floor(r.astype(jnp.float32) * inv).astype(jnp.int32)
    q = q + q2
    r = r - q2 * jnp.int32(d)
    lo = r < 0
    q = jnp.where(lo, q - 1, q)
    r = jnp.where(lo, r + jnp.int32(d), r)
    hi = r >= d
    q = jnp.where(hi, q + 1, q)
    r = jnp.where(hi, r - jnp.int32(d), r)
    return q, r


def _prepare_qsa_token_indices_kernel(
    table_ref,  # i32[block_tokens, 128]
    blocks_ref,  # i32[block_tokens, 512]
    lengths_ref,  # i32[block_tokens, 128]
    tok_out_ref,  # i32[64, 256]
    tail_out_ref,  # i32[block_tokens, 128]
    *,
    groups_per_page: int,
    pages_per_seq: int,
    ratio: int = 4,
    tokens_per_row: int = 1,
    tile_interleaved: bool = True,
):
    table = table_ref[...]  # (8, 128)
    blocks = blocks_ref[...]  # (8, 512)
    lengths = lengths_ref[:, :1]  # (8, 1)

    nonneg = jnp.maximum(lengths, 0)
    complete = nonneg // ratio  # (8, 1)

    # For any block < 0 or >= complete, fall back to distinct indices (iota)
    # to avoid 512-way HBM bank conflict on inactive queries!
    iota_blk = lax.broadcasted_iota(jnp.int32, blocks.shape, 1)
    valid_blk = (blocks >= 0) & (blocks < complete)
    safe_blk = jnp.where(valid_blk, blocks, iota_blk)
    max_blk = jnp.where(complete > 0, complete - 1, blocks.shape[1] - 1)
    blk = jnp.clip(safe_blk, 0, max_blk)
    log_page, page_off = _floor_divmod_const(blk, groups_per_page)
    log_page = jnp.clip(log_page, 0, pages_per_seq - 1)
    phys_page = jnp.take_along_axis(table, log_page, axis=1)
    phys_blk = phys_page * groups_per_page + page_off  # (8, 512)

    if tokens_per_row == 4:
        tok_out_ref[...] = phys_blk
    elif tokens_per_row == 2:
        phys_half = phys_blk * 2
        iota_256 = lax.broadcasted_iota(jnp.int32, (blocks.shape[0], 256), 1)
        local_blk = lax.shift_right_arithmetic(iota_256, 1)
        rem_half = iota_256 & jnp.int32(1)
        expanded_chunks = []
        for c in range(blocks.shape[1] // 128):
            chunk = phys_half[:, c * 128 : (c + 1) * 128]
            expanded_chunks.append(
                jnp.take_along_axis(chunk, local_blk, axis=1) + rem_half
            )
        tok_out_ref[...] = jnp.concatenate(expanded_chunks, axis=1)
    else:
        phys_blk_tok = phys_blk * ratio
        iota_512 = lax.broadcasted_iota(jnp.int32, (blocks.shape[0], 128 * ratio), 1)
        local_blk = lax.shift_right_arithmetic(iota_512, 2)
        rem_tok = iota_512 & jnp.int32(ratio - 1)
        if tile_interleaved:
            tiles_list = []
            for c in range(blocks.shape[1] // 128):
                chunk = phys_blk_tok[:, c * 128 : (c + 1) * 128]
                exp_c = (
                    jnp.take_along_axis(chunk, local_blk, axis=1) + rem_tok
                )  # (8, 512)
                tiles_list.append(exp_c[:, :256])
                tiles_list.append(exp_c[:, 256:])
            tok_out_ref[...] = jnp.concatenate(tiles_list, axis=0)  # (64, 256)
        else:
            expanded_chunks = []
            for c in range(blocks.shape[1] // 128):
                chunk = phys_blk_tok[:, c * 128 : (c + 1) * 128]
                expanded_chunks.append(
                    jnp.take_along_axis(chunk, local_blk, axis=1) + rem_tok
                )
            tok_out_ref[...] = jnp.concatenate(expanded_chunks, axis=1)

    tail_log_page, tail_page_off = _floor_divmod_const(complete, groups_per_page)
    tail_log_page = jnp.clip(tail_log_page, 0, pages_per_seq - 1)
    tail_phys_page = jnp.take_along_axis(table, tail_log_page, axis=1)
    tail_phys_tok = (tail_phys_page * groups_per_page + tail_page_off) * ratio
    tail_out_ref[...] = jnp.broadcast_to(tail_phys_tok, tail_out_ref.shape)


def qsa_prepare_token_indices_pallas(
    blocks: jax.Array,  # i32[batch, 512]
    lengths: jax.Array,  # i32[batch]
    table: jax.Array,  # i32[batch, pages_per_seq]
    *,
    page_size: int = 1664,
    ratio: int = 4,
    tokens_per_row: int = 1,
    tile_interleaved: bool = True,
) -> tuple[jax.Array, jax.Array]:
    batch, budget = blocks.shape
    pages_per_seq = table.shape[1]
    groups_per_page = page_size // ratio
    block_tokens = 8
    assert batch % block_tokens == 0
    out_cols = (budget * ratio) // tokens_per_row
    padded_table = (
        table
        if pages_per_seq == 128
        else jnp.pad(table, ((0, 0), (0, 128 - pages_per_seq)))
    )
    lengths_2d = jnp.broadcast_to(lengths[:, None], (batch, 128))

    if tile_interleaved and tokens_per_row == 1:
        tok_block_spec = pl.BlockSpec((64, 256), lambda b: (b, 0))
        tok_out_shape = jax.ShapeDtypeStruct(((batch // 8) * 64, 256), jnp.int32)
    else:
        tok_block_spec = pl.BlockSpec((block_tokens, out_cols), lambda b: (b, 0))
        tok_out_shape = jax.ShapeDtypeStruct((batch, out_cols), jnp.int32)

    tok_indices, tail_2d = pl.pallas_call(
        functools.partial(
            _prepare_qsa_token_indices_kernel,
            groups_per_page=groups_per_page,
            pages_per_seq=pages_per_seq,
            ratio=ratio,
            tokens_per_row=tokens_per_row,
            tile_interleaved=tile_interleaved,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((block_tokens, 128), lambda b: (b, 0)),
                pl.BlockSpec((block_tokens, budget), lambda b: (b, 0)),
                pl.BlockSpec((block_tokens, 128), lambda b: (b, 0)),
            ],
            out_specs=[
                tok_block_spec,
                pl.BlockSpec((block_tokens, 128), lambda b: (b, 0)),
            ],
            grid=(batch // block_tokens,),
        ),
        out_shape=[
            tok_out_shape,
            jax.ShapeDtypeStruct((batch, 128), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        name=f"qsa_prepare_indices_vmem_tpr{tokens_per_row}_ti{int(tile_interleaved)}",
    )(padded_table, blocks, lengths_2d)
    return tok_indices.reshape(-1), tail_2d[:, 0]


def qsa_sc_gather_2level_tc_tiled(
    cache_f8: jax.Array,  # float8_e4m3fn[num_pages, page_size, 1, 4, 256]
    token_indices: jax.Array,  # int32[batch * (2048 // tokens_per_row)]
    lengths: jax.Array | None = None,
    *,
    num_streams: int = 8,
    chunk_tokens: int = 512,
    tokens_per_row: int = 1,
    ratio: int = 4,
    skip_negative_chunks: bool = False,
) -> jax.Array:
    sc = pltpu.get_tpu_info().sparse_core
    assert sc is not None
    lanes = sc.num_lanes  # 16
    num_cores = sc.num_cores * sc.num_subcores  # 32
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    out_size = token_indices.size
    cols = 256 * tokens_per_row
    row_chunk_size = min(chunk_tokens, out_size // num_cores)
    while (
        out_size % (row_chunk_size * num_cores) != 0
        or row_chunk_size % (lanes * num_streams) != 0
    ) and row_chunk_size > lanes * num_streams:
        row_chunk_size //= 2
    num_row_subchunks = row_chunk_size // lanes
    num_blocks = out_size // (row_chunk_size * num_cores)
    total_chunks = num_blocks * num_cores

    use_active_skip = (
        lengths is not None
        and not skip_negative_chunks
        and tokens_per_row == 1
        and row_chunk_size >= 256
        and row_chunk_size % 256 == 0
    )
    if use_active_skip:
        assert lengths is not None
        batch = lengths.shape[0]
        comp_lens = (jnp.maximum(lengths, 0) // ratio) * ratio
        slice_active = (
            comp_lens.reshape(batch // 8, 1, 8)
            > (jnp.arange(8, dtype=jnp.int32)[None, :, None] * 256)
        ).astype(jnp.int32)
        slices_per_chunk = row_chunk_size // 256
        chunk_active = jnp.max(
            slice_active.reshape(total_chunks, slices_per_chunk),
            axis=1,
            keepdims=True,
        )
        active_flags = jnp.broadcast_to(chunk_active, (total_chunks, 128)).reshape(-1)
    else:
        active_flags = None

    def kernel_body(*kernel_refs):
        if use_active_skip:
            cache_ref, indices_hbm_ref, active_hbm_ref, out_hbm_ref, sem_ref = (
                kernel_refs
            )
        else:
            cache_ref, indices_hbm_ref, out_hbm_ref, sem_ref = kernel_refs
            active_hbm_ref = None
        core_index = lax.axis_index(("core", "subcore"))
        src = cache_ref.bitcast(jnp.int32).reshape(-1, cols)
        dst = out_hbm_ref.bitcast(jnp.int32).reshape(-1, cols)

        def outer_pipeline(*pipe_refs):
            idx_ref = pipe_refs[0]
            b = pl.program_id(0)
            out_row_base = (b * num_cores + core_index) * num_row_subchunks

            def subchunk(r, s):
                return r * num_streams + s

            def idx_window(r, s):
                return idx_ref[pl.ds(subchunk(r, s) * lanes, lanes)]

            def _body(*gathers):
                r = pl.program_id(0)
                copies = []
                for s in range(num_streams):
                    out_row = (out_row_base + subchunk(r, s)) * lanes
                    c = pltpu.make_async_copy(
                        gathers[s],
                        dst.at[pl.ds(out_row, lanes), :],
                        sem_ref.at[s],
                    )
                    c.start()
                    copies.append(c)
                for c in copies:
                    c.wait()

            in_specs = tuple(
                pl.BlockSpec(
                    (pl.Indirect(lanes), cols),
                    lambda r, s=s: (idx_window(r, s), 0),
                )
                for s in range(num_streams)
            )
            if use_active_skip:
                active_ref = pipe_refs[1]
                is_active = active_ref[pl.ds(0, lanes)][0]

                @pl.when(is_active != 0)
                def _run_gather():
                    pltpu.emit_pipeline(
                        _body,
                        grid=(num_row_subchunks // num_streams,),
                        in_specs=in_specs,
                    )(*([src] * num_streams))
            elif skip_negative_chunks:
                is_active = idx_ref[pl.ds(0, lanes)][0] >= 0

                @pl.when(is_active)
                def _run_gather_neg():
                    pltpu.emit_pipeline(
                        _body,
                        grid=(num_row_subchunks // num_streams,),
                        in_specs=in_specs,
                    )(*([src] * num_streams))
            else:
                pltpu.emit_pipeline(
                    _body,
                    grid=(num_row_subchunks // num_streams,),
                    in_specs=in_specs,
                )(*([src] * num_streams))

        if use_active_skip:
            pltpu.emit_pipeline(
                outer_pipeline,
                grid=(num_blocks,),
                in_specs=(
                    pl.BlockSpec(
                        (row_chunk_size,),
                        lambda b: (b * num_cores + core_index,),
                    ),
                    pl.BlockSpec(
                        (128,),
                        lambda b: (b * num_cores + core_index,),
                    ),
                ),
            )(indices_hbm_ref, active_hbm_ref)
        else:
            pltpu.emit_pipeline(
                outer_pipeline,
                grid=(num_blocks,),
                in_specs=(
                    pl.BlockSpec(
                        (row_chunk_size,),
                        lambda b: (b * num_cores + core_index,),
                    ),
                ),
            )(indices_hbm_ref)

    out_shape = (
        (out_size, 1, 4, 256)
        if tokens_per_row == 1
        else (out_size, tokens_per_row, 1, 4, 256)
    )
    fn = pl.kernel(
        kernel_body,
        out_type=jax.ShapeDtypeStruct(out_shape, cache_f8.dtype),
        scratch_types=(pltpu.SemaphoreType.DMA((num_streams,)),),
        mesh=mesh,
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        name=f"qsa_sc_gather_2level_f8_tpr{tokens_per_row}_n{out_size}_s{num_streams}_c{row_chunk_size}_sk{int(use_active_skip)}",
    )
    if use_active_skip:
        return fn(cache_f8, token_indices, active_flags)
    return fn(cache_f8, token_indices)


def _qsa_fast_grouped_qsa_0transpose_kernel(
    lengths_ref,
    tail_phys_tok_ref,
    q_ref,
    gate_ref,
    gathered_kv_ref,
    flat_main_pool_ref,
    out_ref,
    kv_ref,
    btail_ref,
    sem_ref,
    m_ref,
    l_ref,
    acc_ref,
    *,
    query_group: int = 8,
    head_group: int = 16,
    kv_heads: int = 2,
    dim: int = 256,
    tile_k: int = 256,
    num_tiles: int = 8,
    ratio: int = 4,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    bf16_pv: bool = True,
):
    block = pl.program_id(0)
    rows = query_group * head_group
    width = query_group * tile_k
    tail_width = 8

    block_active = jnp.int32(0)
    for i in range(query_group):
        len_i = lengths_ref[block * query_group + i]
        block_active = block_active | (len_i > 0).astype(jnp.int32)

    @pl.when(block_active == 0)
    def _skip_inactive_block():
        for head in range(kv_heads):
            out_ref[0, head] = jnp.zeros_like(out_ref[0, head])

    @pl.when(block_active != 0)
    def _run_active_block():
        query_owner = jnp.arange(rows, dtype=jnp.int32)[:, None] // head_group

        def start_tile(tile, buf):
            offset = (block * num_tiles + tile) * width
            pltpu.make_async_copy(
                gathered_kv_ref.at[pl.ds(offset, width)],
                kv_ref.at[buf],
                sem_ref.at[buf],
            ).start()

        row_comp_tok = jnp.zeros((rows, 128), jnp.int32)
        max_comp_tok = jnp.int32(0)
        for i in range(query_group):
            len_i = jnp.maximum(lengths_ref[block * query_group + i], 0)
            comp_i = (len_i // ratio) * ratio
            max_comp_tok = jnp.maximum(max_comp_tok, comp_i)
            row_comp_tok = jnp.where(query_owner == i, comp_i, row_comp_tok)
        comp_3d = row_comp_tok[:, :1].reshape(query_group, head_group, 1)
        active_tiles = jnp.minimum(num_tiles, (max_comp_tok + tile_k - 1) // tile_k)
        q_heads = tuple(
            q_ref[0, h].reshape(query_group, head_group, dim) for h in range(kv_heads)
        )

        m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, jnp.float32)
        l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

        @pl.when(active_tiles > 0)
        def _start_first_tile():
            start_tile(0, 0)

        for i in range(query_group):
            t_row = (tail_phys_tok_ref[block * query_group + i] // 8) * 8
            pltpu.make_async_copy(
                flat_main_pool_ref.at[pl.ds(t_row, 8)],
                btail_ref.at[pl.ds(i * 8, 8)],
                sem_ref.at[2],
            ).start()

        def step(tile, _):
            buf = tile % 2
            dst = kv_ref.at[buf]
            pltpu.make_async_copy(dst, dst, sem_ref.at[buf]).wait()

            @pl.when(tile + 1 < active_tiles)
            def prefetch():
                start_tile(tile + 1, 1 - buf)

            k_iota = tile * tile_k + lax.broadcasted_iota(
                jnp.int32, (query_group, head_group, tile_k), 2
            )
            valid_k = k_iota < comp_3d

            words = dst.bitcast(jnp.uint32).reshape(width, dim)[...]
            kv_pairs = rpa_utils.convert_to_target_bitwidth(
                words, target_bitwidth=8, kv_dtype=gathered_kv_ref.dtype
            )
            fmax = float(jnp.finfo(gathered_kv_ref.dtype).max)
            for head in range(kv_heads):
                key, value = kv_pairs[head]
                key = key.astype(jnp.bfloat16)
                key = jnp.where(jnp.abs(key) <= fmax, key, 0)
                if bf16_pv:
                    value = value.astype(jnp.bfloat16)
                    value = jnp.where(jnp.abs(value) <= fmax, value, 0)
                else:
                    value = value.astype(jnp.float32)
                    value = jnp.where(jnp.isfinite(value), value, 0)

                query = q_heads[head]
                key = key.reshape(query_group, tile_k, dim)
                value = value.reshape(query_group, tile_k, dim)
                score = lax.dot_general(
                    query,
                    key,
                    dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                ) * (dim**-0.5 * k_scale)
                score = jnp.where(valid_k, score, -jnp.inf)
                old_m = m_ref[head, :, :1].reshape(query_group, head_group, 1)
                old_l = l_ref[head, :, :1].reshape(query_group, head_group, 1)
                next_m = jnp.maximum(old_m, jnp.max(score, axis=2, keepdims=True))
                safe_m = jnp.where(jnp.isfinite(next_m), next_m, 0.0)
                prob = jnp.where(valid_k, jnp.exp(score - safe_m), 0.0)
                alpha = jnp.exp(old_m - safe_m)
                prob_lhs = prob.astype(jnp.bfloat16) if bf16_pv else prob
                weighted = lax.dot_general(
                    prob_lhs,
                    value,
                    dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                )
                weighted_2d = weighted.reshape(rows, dim)
                alpha_2d = alpha.reshape(rows, 1)
                next_m_2d = next_m.reshape(rows, 1)
                next_l_2d = (
                    old_l * alpha + jnp.sum(prob, axis=2, keepdims=True)
                ).reshape(rows, 1)
                acc_ref[head] = acc_ref[head] * alpha_2d + weighted_2d * v_scale
                m_ref[head] = jnp.broadcast_to(next_m_2d, (rows, 128))
                l_ref[head] = jnp.broadcast_to(next_l_2d, (rows, 128))
            return None

        lax.fori_loop(0, active_tiles, step, None)

        row_v_tail = jnp.zeros((rows, 128), jnp.int32)
        row_sub_off = jnp.zeros((rows, 128), jnp.int32)
        row_active = jnp.zeros((rows, 128), jnp.int32)
        for i in range(query_group):
            len_i = jnp.maximum(lengths_ref[block * query_group + i], 0)
            v_tail_i = len_i % ratio
            sub_off_i = tail_phys_tok_ref[block * query_group + i] % 8
            act_i = (len_i > 0).astype(jnp.int32)
            row_v_tail = jnp.where(query_owner == i, v_tail_i, row_v_tail)
            row_sub_off = jnp.where(query_owner == i, sub_off_i, row_sub_off)
            row_active = jnp.where(query_owner == i, act_i, row_active)
        row_v_tail = row_v_tail[:, :1]
        row_sub_off = row_sub_off[:, :1]
        row_is_active = row_active[:, :1] > 0

        for i in range(query_group):
            dst_t = btail_ref.at[pl.ds(i * 8, 8)]
            pltpu.make_async_copy(dst_t, dst_t, sem_ref.at[2]).wait()

        tail_words = btail_ref.bitcast(jnp.uint32).reshape(
            query_group * tail_width, dim
        )[...]
        tail_pairs = rpa_utils.convert_to_target_bitwidth(
            tail_words, target_bitwidth=8, kv_dtype=gathered_kv_ref.dtype
        )
        sources = jnp.arange(query_group * tail_width, dtype=jnp.int32)
        src_owner = sources[None, :] // tail_width
        src_sub = sources[None, :] % tail_width
        valid_tail = (
            (src_owner == query_owner)
            & (src_sub >= row_sub_off)
            & (src_sub < row_sub_off + row_v_tail)
        )
        for head in range(kv_heads):
            key_t, val_t = (x.astype(jnp.float32) for x in tail_pairs[head])
            key_t, val_t = (jnp.where(jnp.isfinite(x), x, 0) for x in (key_t, val_t))
            q_f32 = q_ref[0, head].astype(jnp.float32)
            score_t = lax.dot_general(
                q_f32,
                key_t,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            score_t = jnp.where(valid_tail, score_t * (dim**-0.5 * k_scale), -jnp.inf)
            old_m = m_ref[head, :, :1]
            old_l = l_ref[head, :, :1]
            next_m = jnp.maximum(old_m, jnp.max(score_t, axis=1, keepdims=True))
            safe_m = jnp.where(jnp.isfinite(next_m), next_m, 0.0)
            prob_t = jnp.where(valid_tail, jnp.exp(score_t - safe_m), 0.0)
            alpha = jnp.exp(old_m - safe_m)
            weighted_t = lax.dot_general(
                prob_t,
                val_t,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            final_acc = acc_ref[head] * alpha + weighted_t * v_scale
            final_l = old_l * alpha + prob_t.sum(axis=1, keepdims=True)
            result = final_acc / jnp.where(final_l > 0, final_l, 1.0)
            gated = bf16(
                bf16(result).astype(jnp.float32)
                * bf16(jax.nn.sigmoid(gate_ref[0, head].astype(jnp.float32))).astype(
                    jnp.float32
                )
            )
            out_ref[0, head] = jnp.where(row_is_active, gated, 0)


def _qsa_fast_grouped_qsa_projected_kernel(
    *all_refs,
    batch: int,
    query_group: int = 8,
    head_group: int = 16,
    real_heads: int = 12,
    kv_heads: int = 2,
    dim: int = 256,
    tile_k: int = 256,
    num_tiles: int = 8,
    ratio: int = 4,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    bf16_pv: bool = True,
    packed_meta: bool = False,
    unroll_tiles: bool = False,
    combined_qgate: bool = False,
    native_f8_kv: bool = False,
):
    if packed_meta:
        meta_ref = all_refs[0]

        def get_len(q_idx):
            return meta_ref[q_idx, 0]

        def get_tail(q_idx):
            return meta_ref[q_idx, 1]

        rest = all_refs[1:]
    else:
        lengths_ref, tail_phys_tok_ref = all_refs[0], all_refs[1]

        def get_len(q_idx):
            return lengths_ref[q_idx]

        def get_tail(q_idx):
            return tail_phys_tok_ref[q_idx]

        rest = all_refs[2:]

    if combined_qgate:
        qg_ref = rest[0]

        def get_q(h):
            return qg_ref[0, 0, h]

        def get_gate(h):
            return qg_ref[0, 1, h]

        rest = rest[1:]
    else:
        q_ref, gate_ref = rest[0], rest[1]

        def get_q(h):
            return q_ref[0, h]

        def get_gate(h):
            return gate_ref[0, h]

        rest = rest[2:]

    (
        gathered_kv_ref,
        flat_main_pool_ref,
        weight_ref,
        residual_ref,
        hc_gate_ref,
        projected_ref,
        kv_ref,
        btail_ref,
        sem_ref,
        m_ref,
        l_ref,
        acc_ref,
        weight_buf,
        attended_buf,
        projection_sems,
    ) = rest

    block = pl.program_id(0)
    num_blocks = batch // query_group
    rows = query_group * head_group
    width = query_group * tile_k
    tail_width = 8
    head_cols = real_heads * dim
    padded_head_cols = head_group * dim

    output_tiles = weight_ref.shape[0]
    tile_width = weight_ref.shape[2]
    output_width = output_tiles * tile_width
    branches = hc_gate_ref.shape[1]

    def is_blk_active(b_idx):
        act = jnp.int32(0)
        for i in range(query_group):
            len_i = get_len(b_idx * query_group + i)
            act = act | (len_i > 0).astype(jnp.int32)
        return act

    use_pipelined_wo = (
        native_f8_kv
        and unroll_tiles
        and num_tiles == 8
        and output_tiles in (4, 5)
        and num_blocks <= 2
        and bf16_pv
    )
    block_active = is_blk_active(block)
    last_blk = num_blocks - 1
    prev_active = (
        jnp.where(block > 0, is_blk_active(jnp.maximum(block - 1, 0)), jnp.int32(0))
        if (use_pipelined_wo and num_blocks > 1)
        else jnp.int32(0)
    )
    next_active = (
        jnp.where(
            block + 1 < num_blocks,
            is_blk_active(jnp.minimum(block + 1, last_blk)),
            jnp.int32(0),
        )
        if (use_pipelined_wo and num_blocks > 1)
        else jnp.int32(0)
    )
    last_active = (
        is_blk_active(last_blk)
        if (use_pipelined_wo and num_blocks > 1)
        else block_active
    )

    if use_pipelined_wo and num_blocks > 1:

        @pl.when((block == 0) & (block_active == 0))
        def _start_weights_when_block0_inactive():
            pltpu.make_async_copy(
                weight_ref.at[0], weight_buf.at[0], projection_sems.at[0]
            ).start()
            pltpu.make_async_copy(
                weight_ref.at[1], weight_buf.at[1], projection_sems.at[1]
            ).start()

            @pl.when(last_active == 0)
            def _start_rest_weights():
                for w_idx in range(2, output_tiles):
                    pltpu.make_async_copy(
                        weight_ref.at[w_idx],
                        weight_buf.at[w_idx],
                        projection_sems.at[w_idx],
                    ).start()
    else:

        @pl.when((block == 0) & (block_active == 0))
        def _start_weights_when_block0_inactive():
            for w_tile in range(output_tiles):
                pltpu.make_async_copy(
                    weight_ref.at[w_tile],
                    weight_buf.at[w_tile],
                    projection_sems.at[w_tile],
                ).start()

    @pl.when(block_active == 0)
    def _skip_inactive_block():
        attended_buf[block, :, :] = jnp.zeros(
            (query_group, kv_heads * head_cols), jnp.bfloat16
        )

    @pl.when(block_active != 0)
    def _run_active_block():
        query_owner = jnp.arange(rows, dtype=jnp.int32)[:, None] // head_group

        def start_blk_tile(b_idx, tile, buf):
            offset = (b_idx * num_tiles + tile) * width
            pltpu.make_async_copy(
                gathered_kv_ref.at[pl.ds(offset, width)],
                kv_ref.at[buf],
                sem_ref.at[buf],
            ).start()

        def start_tile(tile, buf):
            start_blk_tile(block, tile, buf)

        row_comp_tok = jnp.zeros((rows, 128), jnp.int32)
        for i in range(query_group):
            len_i = jnp.maximum(get_len(block * query_group + i), 0)
            comp_i = (len_i // ratio) * ratio
            row_comp_tok = jnp.where(query_owner == i, comp_i, row_comp_tok)
        comp_3d = row_comp_tok[:, :1].reshape(query_group, head_group, 1)
        q_heads = tuple(
            get_q(h).reshape(query_group, head_group, dim) for h in range(kv_heads)
        )

        m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, jnp.float32)
        l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
        acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)
        if use_pipelined_wo and num_blocks > 1:

            @pl.when(prev_active == 0)
            def _start_initial_pair():
                start_blk_tile(block, 0, 0)
                start_blk_tile(block, 1, 1)
        elif use_pipelined_wo:
            start_blk_tile(block, 0, 0)
            start_blk_tile(block, 1, 1)
        else:
            start_tile(0, 0)
        for i in range(query_group):
            t_row = (get_tail(block * query_group + i) // 8) * 8
            pltpu.make_async_copy(
                flat_main_pool_ref.at[pl.ds(t_row, 8)],
                btail_ref.at[pl.ds(i * 8, 8)],
                sem_ref.at[2],
            ).start()

        kv_dtype = gathered_kv_ref.dtype
        use_exp2_inner = native_f8_kv and bf16_pv
        use_swar_f8 = use_exp2_inner and (kv_dtype == jnp.float8_e4m3fn)
        log2_e = 1.4426950408889634
        qk_scale = (dim**-0.5 * k_scale) * (log2_e if use_exp2_inner else 1.0)
        fmax = float(jnp.finfo(kv_dtype).max)
        u_7f = jnp.uint32(0x7F7F7F7F)
        u_01 = jnp.uint32(0x01010101)
        u_80 = jnp.uint32(0x80808080)

        def step(tile, _=None):
            if not use_pipelined_wo:
                if isinstance(tile, int):
                    if tile < output_tiles:

                        @pl.when(block == 0)
                        def _start_weight_dma_static(w_tile=tile):
                            pltpu.make_async_copy(
                                weight_ref.at[w_tile],
                                weight_buf.at[w_tile],
                                projection_sems.at[w_tile],
                            ).start()
                else:
                    for w_tile in range(output_tiles):

                        @pl.when((block == 0) & (tile == w_tile))
                        def _start_weight_dma(w_tile=w_tile):
                            pltpu.make_async_copy(
                                weight_ref.at[w_tile],
                                weight_buf.at[w_tile],
                                projection_sems.at[w_tile],
                            ).start()

            buf = tile % 2
            dst = kv_ref.at[buf]
            pltpu.make_async_copy(dst, dst, sem_ref.at[buf]).wait()

            if not use_pipelined_wo:
                if isinstance(tile, int):
                    if tile + 1 < num_tiles:
                        start_tile(tile + 1, 1 - buf)
                else:

                    @pl.when(tile + 1 < num_tiles)
                    def prefetch():
                        start_tile(tile + 1, 1 - buf)

            k_iota = tile * tile_k + lax.broadcasted_iota(
                jnp.int32, (query_group, head_group, tile_k), 2
            )
            valid_k = k_iota < comp_3d

            words = dst.bitcast(jnp.uint32).reshape(-1, dim)
            packed = rpa_utils.strided_load(words, 0, width, 1)
            if use_swar_f8:
                m_nan = ((packed & u_7f) + u_01) & u_80
                packed = packed & ~(m_nan | (m_nan - (m_nan >> jnp.uint32(7))))
            kv_pairs = rpa_utils.convert_to_target_bitwidth(
                packed, target_bitwidth=8, kv_dtype=kv_dtype
            )
            for head in range(kv_heads):
                key, value = kv_pairs[head]
                if not use_swar_f8:
                    key = key.astype(jnp.bfloat16)
                    key = jnp.where(jnp.abs(key) <= fmax, key, 0)
                    if bf16_pv:
                        value = value.astype(jnp.bfloat16)
                        value = jnp.where(jnp.abs(value) <= fmax, value, 0)
                    else:
                        value = value.astype(jnp.float32)
                        value = jnp.where(jnp.isfinite(value), value, 0)

                query = q_heads[head]
                key = key.reshape(query_group, tile_k, dim)
                value = value.reshape(query_group, tile_k, dim)
                score = (
                    lax.dot_general(
                        query,
                        key,
                        dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                        preferred_element_type=jnp.float32,
                    )
                    * qk_scale
                )
                score = jnp.where(valid_k, score, -jnp.inf)
                old_m = m_ref[head, :, :1].reshape(query_group, head_group, 1)
                old_l = l_ref[head, :, :1].reshape(query_group, head_group, 1)
                next_m = jnp.maximum(old_m, jnp.max(score, axis=2, keepdims=True))
                safe_m = jnp.where(jnp.isfinite(next_m), next_m, 0.0)
                if use_exp2_inner:
                    prob = jnp.where(valid_k, lax.exp2(score - safe_m), 0.0)
                    alpha = lax.exp2(old_m - safe_m)
                else:
                    prob = jnp.where(valid_k, jnp.exp(score - safe_m), 0.0)
                    alpha = jnp.exp(old_m - safe_m)
                prob_lhs = prob.astype(jnp.bfloat16) if bf16_pv else prob
                weighted = lax.dot_general(
                    prob_lhs,
                    value,
                    dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                    preferred_element_type=jnp.float32,
                )
                weighted_2d = weighted.reshape(rows, dim)
                alpha_2d = alpha.reshape(rows, 1)
                next_m_2d = next_m.reshape(rows, 1)
                next_l_2d = (
                    old_l * alpha + jnp.sum(prob, axis=2, keepdims=True)
                ).reshape(rows, 1)
                acc_ref[head] = acc_ref[head] * alpha_2d + weighted_2d * v_scale
                m_ref[head] = jnp.broadcast_to(next_m_2d, (rows, 128))
                l_ref[head] = jnp.broadcast_to(next_l_2d, (rows, 128))

            if use_pipelined_wo:
                assert isinstance(tile, int)
                if tile + 2 < num_tiles:
                    start_blk_tile(block, tile + 2, buf)
                elif num_blocks > 1:
                    next_tile = tile - (num_tiles - 2)

                    @pl.when(next_active != 0)
                    def _prefetch_next_block(nt=next_tile, b_buf=buf):
                        start_blk_tile(block + 1, nt, b_buf)

                if num_blocks == 1:
                    w_sched_1 = (1, 3, 5, 7) if output_tiles == 4 else (1, 3, 5, 6, 7)
                    if tile in w_sched_1:
                        wt = w_sched_1.index(tile)
                        pltpu.make_async_copy(
                            weight_ref.at[wt],
                            weight_buf.at[wt],
                            projection_sems.at[wt],
                        ).start()
                else:
                    if tile in (3, 6):
                        wt = 0 if tile == 3 else 1

                        @pl.when(block == 0)
                        def _start_w01(w_idx=wt):
                            pltpu.make_async_copy(
                                weight_ref.at[w_idx],
                                weight_buf.at[w_idx],
                                projection_sems.at[w_idx],
                            ).start()

                    w_sched_rest = (4, 7) if output_tiles == 4 else (2, 5, 7)
                    if tile in w_sched_rest:
                        wt = 2 + w_sched_rest.index(tile)

                        @pl.when(
                            (block == last_blk) | ((block == 0) & (last_active == 0))
                        )
                        def _start_w_rest(w_idx=wt):
                            pltpu.make_async_copy(
                                weight_ref.at[w_idx],
                                weight_buf.at[w_idx],
                                projection_sems.at[w_idx],
                            ).start()

            return None

        if unroll_tiles:
            for t_i in range(num_tiles):
                step(t_i)
        else:
            lax.fori_loop(0, num_tiles, step, None)

        row_v_tail = jnp.zeros((rows, 128), jnp.int32)
        row_sub_off = jnp.zeros((rows, 128), jnp.int32)
        row_active = jnp.zeros((rows, 128), jnp.int32)
        for i in range(query_group):
            len_i = jnp.maximum(get_len(block * query_group + i), 0)
            v_tail_i = len_i % ratio
            sub_off_i = get_tail(block * query_group + i) % 8
            act_i = (len_i > 0).astype(jnp.int32)
            row_v_tail = jnp.where(query_owner == i, v_tail_i, row_v_tail)
            row_sub_off = jnp.where(query_owner == i, sub_off_i, row_sub_off)
            row_active = jnp.where(query_owner == i, act_i, row_active)
        row_v_tail = row_v_tail[:, :1]
        row_sub_off = row_sub_off[:, :1]
        row_is_active = row_active[:, :1] > 0

        for i in range(query_group):
            dst_t = btail_ref.at[pl.ds(i * 8, 8)]
            pltpu.make_async_copy(dst_t, dst_t, sem_ref.at[2]).wait()

        tail_words = btail_ref.bitcast(jnp.uint32).reshape(
            query_group * tail_width, dim
        )[...]
        use_exp2 = native_f8_kv and bf16_pv
        log2_e = 1.4426950408889634
        qk_scale_t = (dim**-0.5 * k_scale) * (log2_e if use_exp2 else 1.0)
        sources = jnp.arange(query_group * tail_width, dtype=jnp.int32)
        src_owner = sources[None, :] // tail_width
        src_sub = sources[None, :] % tail_width
        valid_tail = (
            (src_owner == query_owner)
            & (src_sub >= row_sub_off)
            & (src_sub < row_sub_off + row_v_tail)
        )
        tail_pairs = rpa_utils.convert_to_target_bitwidth(
            tail_words, target_bitwidth=8, kv_dtype=gathered_kv_ref.dtype
        )
        for head in range(kv_heads):
            key_t, val_t = (x.astype(jnp.float32) for x in tail_pairs[head])
            key_t, val_t = (jnp.where(jnp.isfinite(x), x, 0) for x in (key_t, val_t))
            q_f32 = q_heads[head].reshape(rows, dim).astype(jnp.float32)
            score_t = lax.dot_general(
                q_f32,
                key_t,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            score_t = jnp.where(valid_tail, score_t * qk_scale_t, -jnp.inf)
            old_m = m_ref[head, :, :1]
            old_l = l_ref[head, :, :1]
            next_m = jnp.maximum(old_m, jnp.max(score_t, axis=1, keepdims=True))
            safe_m = jnp.where(jnp.isfinite(next_m), next_m, 0.0)
            if use_exp2:
                prob_t = jnp.where(valid_tail, lax.exp2(score_t - safe_m), 0.0)
                alpha = lax.exp2(old_m - safe_m)
            else:
                prob_t = jnp.where(valid_tail, jnp.exp(score_t - safe_m), 0.0)
                alpha = jnp.exp(old_m - safe_m)
            weighted_t = lax.dot_general(
                prob_t,
                val_t,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            final_acc = acc_ref[head] * alpha + weighted_t * v_scale
            final_l = old_l * alpha + prob_t.sum(axis=1, keepdims=True)
            result = final_acc / jnp.where(final_l > 0, final_l, 1.0)
            gated = bf16(
                bf16(result).astype(jnp.float32)
                * bf16(jax.nn.sigmoid(get_gate(head).astype(jnp.float32))).astype(
                    jnp.float32
                )
            )
            gated = jnp.where(row_is_active, gated, 0)
            attended_buf[
                block,
                :,
                pl.ds(head * head_cols, head_cols),
            ] = gated.reshape(query_group, padded_head_cols)[:, :head_cols]

    @pl.when(block == num_blocks - 1)
    def _project_and_accumulate_hc():
        attended = (
            attended_buf[0, :, :]
            if num_blocks == 1
            else jnp.concatenate(
                [attended_buf[b_i, :, :] for b_i in range(num_blocks)], axis=0
            )
        )
        if not use_pipelined_wo:
            for w_tile in range(output_tiles):
                pltpu.make_async_copy(
                    weight_ref.at[w_tile],
                    weight_buf.at[w_tile],
                    projection_sems.at[w_tile],
                ).wait()
        for w_tile in range(output_tiles):
            if use_pipelined_wo:
                pltpu.make_async_copy(
                    weight_ref.at[w_tile],
                    weight_buf.at[w_tile],
                    projection_sems.at[w_tile],
                ).wait()
            projected = project_tile(attended, weight_buf[w_tile, ...])
            for branch in range(branches):
                out_cols = pl.ds(
                    branch * output_width + w_tile * tile_width,
                    tile_width,
                )
                projected_ref[:, out_cols] = (
                    residual_ref[:, out_cols].astype(jnp.float32)
                    + projected * hc_gate_ref[:, branch : branch + 1]
                ).astype(jnp.bfloat16)


def qsa_fast_grouped_qsa_0transpose_attention(
    q: jax.Array,
    gate: jax.Array,
    gathered_kv: jax.Array,
    main_pool: jax.Array,
    lengths: jax.Array,
    tail_phys_tok: jax.Array,
    *,
    kv_heads: int = 2,
    ratio: int = 4,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    bf16_pv: bool = True,
) -> jax.Array:
    batch, heads, dim = q.shape
    query_group = 8
    real_heads = heads // kv_heads
    head_group = pl.cdiv(real_heads, 8) * 8
    tile_k = 256
    num_tiles = 8

    def pack(x):
        x = x.reshape(batch // query_group, query_group, kv_heads, real_heads, dim)
        x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, head_group - real_heads), (0, 0)))
        return x.transpose(0, 2, 1, 3, 4).reshape(
            batch // query_group, kv_heads, query_group * head_group, dim
        )

    packed_q, packed_gate = pack(q), pack(gate)
    flat_main_pool = main_pool.reshape(-1, *main_pool.shape[2:])
    flat_gathered_kv = gathered_kv.reshape(-1, *main_pool.shape[2:])
    spec = pl.BlockSpec(
        (1, kv_heads, query_group * head_group, dim), lambda b, *_: (b, 0, 0, 0)
    )
    state_shape = (kv_heads, query_group * head_group, 128)

    y = pl.pallas_call(
        functools.partial(
            _qsa_fast_grouped_qsa_0transpose_kernel,
            query_group=query_group,
            head_group=head_group,
            kv_heads=kv_heads,
            dim=dim,
            tile_k=tile_k,
            num_tiles=num_tiles,
            ratio=ratio,
            k_scale=k_scale,
            v_scale=v_scale,
            bf16_pv=bf16_pv,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(batch // query_group,),
            in_specs=(
                spec,
                spec,
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=spec,
            scratch_shapes=(
                pltpu.VMEM(
                    (2, query_group * tile_k, *flat_main_pool.shape[1:]),
                    flat_main_pool.dtype,
                ),
                pltpu.VMEM(
                    (query_group * 8, *flat_main_pool.shape[1:]),
                    flat_main_pool.dtype,
                ),
                pltpu.SemaphoreType.DMA((3,)),
                pltpu.VMEM(state_shape, jnp.float32),
                pltpu.VMEM(state_shape, jnp.float32),
                pltpu.VMEM((kv_heads, query_group * head_group, dim), jnp.float32),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(
            (batch // query_group, kv_heads, query_group * head_group, dim),
            jnp.bfloat16,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=min(pltpu.get_tpu_info().vmem_capacity_bytes, 67043328),
            disable_bounds_checks=True,
        ),
        name=f"qsa_fast_grouped_qsa_0t_b{batch}_bf16pv{int(bf16_pv)}",
    )(lengths, tail_phys_tok, packed_q, packed_gate, flat_gathered_kv, flat_main_pool)

    y = y.reshape(
        batch // query_group, kv_heads, query_group, head_group, dim
    ).transpose(0, 2, 1, 3, 4)
    return y[:, :, :, :real_heads].reshape(q.shape)


def qsa_fast_grouped_qsa_projected_attention(
    q: jax.Array,
    gate: jax.Array | None,
    gathered_kv: jax.Array,
    main_pool: jax.Array,
    lengths: jax.Array,
    tail_phys_tok: jax.Array | None,
    output_weight: jax.Array,
    residual: jax.Array,
    hc_gate: jax.Array,
    *,
    kv_heads: int = 2,
    real_heads_override: int | None = None,
    ratio: int = 4,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    bf16_pv: bool = True,
    prepacked_qgate: bool = False,
    combined_qgate: bool = False,
    unroll_tiles: bool = False,
    native_f8_kv: bool = False,
) -> jax.Array:
    query_group = 8
    if combined_qgate:
        num_q_blocks, two_in, kv_h_in, qg_hg, dim = q.shape
        assert two_in == 2 and kv_h_in == kv_heads and gate is None
        batch = num_q_blocks * query_group
        head_group = qg_hg // query_group
        real_heads = 12 if real_heads_override is None else real_heads_override
        heads = kv_heads * real_heads
        packed_qg_args: tuple[jax.Array, ...] = (q,)
    elif prepacked_qgate:
        assert gate is not None
        num_q_blocks, kv_h_in, qg_hg, dim = q.shape
        assert kv_h_in == kv_heads
        batch = num_q_blocks * query_group
        head_group = qg_hg // query_group
        real_heads = 12 if real_heads_override is None else real_heads_override
        heads = kv_heads * real_heads
        packed_qg_args = (q, gate)
    else:
        assert gate is not None
        batch, heads, dim = q.shape
        real_heads = heads // kv_heads
        head_group = pl.cdiv(real_heads, 8) * 8

        def pack(x):
            x = x.reshape(batch // query_group, query_group, kv_heads, real_heads, dim)
            x = jnp.pad(
                x, ((0, 0), (0, 0), (0, 0), (0, head_group - real_heads), (0, 0))
            )
            return x.transpose(0, 2, 1, 3, 4).reshape(
                batch // query_group, kv_heads, query_group * head_group, dim
            )

        packed_qg_args = (pack(q), pack(gate))

    tile_k = 256
    num_tiles = 8
    output_tiles = output_weight.shape[0]
    padded_width = output_tiles * output_weight.shape[2]
    branches = hc_gate.shape[1]
    width = residual.shape[1] // branches
    result_width = branches * padded_width
    packed_meta = tail_phys_tok is None

    flat_main_pool = main_pool.reshape(-1, *main_pool.shape[2:])
    flat_gathered_kv = gathered_kv.reshape(-1, *main_pool.shape[2:])
    padded_residual = (
        residual
        if padded_width == width
        else jnp.pad(
            residual.reshape(batch, branches, width),
            ((0, 0), (0, 0), (0, padded_width - width)),
        ).reshape(batch, result_width)
    )

    if combined_qgate:
        qg_specs: tuple[pl.BlockSpec, ...] = (
            pl.BlockSpec(
                (1, 2, kv_heads, query_group * head_group, dim),
                lambda b, *_: (b, 0, 0, 0, 0),
            ),
        )
    else:
        spec = pl.BlockSpec(
            (1, kv_heads, query_group * head_group, dim), lambda b, *_: (b, 0, 0, 0)
        )
        qg_specs = (spec, spec)
    state_shape = (kv_heads, query_group * head_group, 128)
    scalar_args = (lengths,) if packed_meta else (lengths, tail_phys_tok)
    kernel_name = (
        f"qsa_grouped_proj_attn_b{batch}_f8kv{int(native_f8_kv)}_u{int(unroll_tiles)}"
        if (native_f8_kv or combined_qgate)
        else f"qsa_grouped_qsa_proj_b{batch}_bf16pv{int(bf16_pv)}_u{int(unroll_tiles)}"
    )

    out = pl.pallas_call(
        functools.partial(
            _qsa_fast_grouped_qsa_projected_kernel,
            batch=batch,
            query_group=query_group,
            head_group=head_group,
            real_heads=real_heads,
            kv_heads=kv_heads,
            dim=dim,
            tile_k=tile_k,
            num_tiles=num_tiles,
            ratio=ratio,
            k_scale=k_scale,
            v_scale=v_scale,
            bf16_pv=bf16_pv,
            packed_meta=packed_meta,
            unroll_tiles=unroll_tiles,
            combined_qgate=combined_qgate,
            native_f8_kv=native_f8_kv,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_args),
            grid=(batch // query_group,),
            in_specs=(
                *qg_specs,
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(padded_residual.shape, lambda b, *_: (0, 0)),
                pl.BlockSpec(hc_gate.shape, lambda b, *_: (0, 0)),
            ),
            out_specs=pl.BlockSpec((batch, result_width), lambda b, *_: (0, 0)),
            scratch_shapes=(
                pltpu.VMEM(
                    (2, query_group * tile_k, *flat_main_pool.shape[1:]),
                    flat_main_pool.dtype,
                ),
                pltpu.VMEM(
                    (query_group * 8, *flat_main_pool.shape[1:]),
                    flat_main_pool.dtype,
                ),
                pltpu.SemaphoreType.DMA((3,)),
                pltpu.VMEM(state_shape, jnp.float32),
                pltpu.VMEM(state_shape, jnp.float32),
                pltpu.VMEM((kv_heads, query_group * head_group, dim), jnp.float32),
                pltpu.VMEM(output_weight.shape, output_weight.dtype),
                pltpu.VMEM(
                    (batch // query_group, query_group, heads * dim), jnp.bfloat16
                ),
                pltpu.SemaphoreType.DMA((output_tiles,)),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((batch, result_width), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=min(pltpu.get_tpu_info().vmem_capacity_bytes, 67043328),
            disable_bounds_checks=True,
            flags={"XLA_TPU_EVENLY_DISTRIBUTE_CRITICAL_NODES": True},
        ),
        name=kernel_name,
    )(
        *scalar_args,
        *packed_qg_args,
        flat_gathered_kv,
        flat_main_pool,
        output_weight,
        padded_residual,
        hc_gate,
    )

    if padded_width == width:
        return out
    return out.reshape(batch, branches, padded_width)[:, :, :width].reshape(
        residual.shape
    )
