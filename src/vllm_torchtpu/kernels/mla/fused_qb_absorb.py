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
"""Fused VMEM Pallas kernel for prefill MLA q_b_proj + q_pe RoPE + absorb_q_split."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.quantized_matmul.util import (
    quantize_tensor,
    xla_quantized_matmul,
)

MIN_TOKENS = 128
HEAD_BLOCK = 8
TAIL_CHUNK_CANDIDATES = (512, 256, 128, 64)


def select_tail_chunk(tail_rows: int) -> int | None:
    """Return the largest supported tail token chunk dividing `tail_rows`, or None."""
    if tail_rows <= 0:
        return None
    # Avoid looping two 512-row passes on 1024-row prefill chunks; cap >1024
    # tails at 512 so larger buckets keep VMEM headroom.
    if tail_rows == 1024:
        return 1024
    for candidate in TAIL_CHUNK_CANDIDATES:
        if tail_rows % candidate == 0:
            return candidate
    return None


def supports_fused_qb_rope_absorb_split(
    q_c_shape: tuple[int, ...],
    q_c_dtype: object,
    w_qb_shape: tuple[int, ...],
    w_qb_dtype: object,
    w_s_shape: tuple[int, ...],
    w_uk_t_shape: tuple[int, ...],
    cos_sin_shape: tuple[int, ...],
    split_rows: int,
    *,
    bfloat16_dtype: object,
    fp8_dtype: object,
) -> bool:
    """Return whether shapes and dtypes satisfy the fused VMEM kernel contract."""
    if len(q_c_shape) != 2 or len(w_qb_shape) != 2 or len(w_s_shape) != 1:
        return False
    if len(w_uk_t_shape) != 3 or len(cos_sin_shape) != 2:
        return False
    if q_c_dtype != bfloat16_dtype or w_qb_dtype != fp8_dtype:
        return False
    total_tokens, k_dim = q_c_shape
    w_k, n_out = w_qb_shape
    heads, d_nope, rank = w_uk_t_shape
    cs_tokens, d_pe = cos_sin_shape
    d_q = d_nope + d_pe
    tail_rows = total_tokens - split_rows
    if total_tokens < MIN_TOKENS or cs_tokens != total_tokens:
        return False
    if w_k != k_dim or w_s_shape[0] != n_out or n_out != heads * d_q:
        return False
    if split_rows <= 0 or split_rows % 8 != 0 or select_tail_chunk(tail_rows) is None:
        return False
    if heads % HEAD_BLOCK != 0 or k_dim % 128 != 0 or d_q % 128 != 0:
        return False
    return d_pe == 64 and rank % 128 == 0


def fused_qb_rope_absorb_split_ref(
    q_c: jax.Array,
    w_qb: jax.Array,
    w_s: jax.Array,
    w_uk_t: jax.Array,
    cos_sin: jax.Array,
    split_rows: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reference JAX chain: FP8 q_b_proj -> split -> NeoX q_pe RoPE + absorb_q_split."""
    heads, d_nope, _ = w_uk_t.shape
    d_pe = cos_sin.shape[1]
    half_pe = d_pe // 2
    q = xla_quantized_matmul(q_c, w_qb, w_s).reshape(q_c.shape[0], heads, d_nope + d_pe)
    q_nope = q[:, :, :d_nope]
    q_pe = q[:, :, d_nope:]
    ql = jnp.einsum(
        "thd,hdr->thr",
        q_nope.astype(jnp.float32),
        w_uk_t.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    ).astype(q_c.dtype)
    q_pe_f = q_pe.astype(jnp.float32)
    cs_f = cos_sin.astype(jnp.float32)
    cos = cs_f[:, None, :half_pe]
    sin = cs_f[:, None, half_pe:]
    x1 = q_pe_f[:, :, :half_pe]
    x2 = q_pe_f[:, :, half_pe:]
    q_pe_rot = jnp.concatenate(
        [x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1
    ).astype(q_c.dtype)
    return ql[:split_rows], ql[split_rows:], q_pe_rot


def fused_qb_rope_absorb_split_qpe_ref(
    q_c: jax.Array,
    w_qb: jax.Array,
    w_s: jax.Array,
    w_uk_t: jax.Array,
    cos_sin: jax.Array,
    split_rows: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    ql_head, ql_tail, q_pe_rot = fused_qb_rope_absorb_split_ref(
        q_c, w_qb, w_s, w_uk_t, cos_sin, split_rows
    )
    return ql_head, ql_tail, q_pe_rot[:split_rows], q_pe_rot[split_rows:]


def _run_fused_qb_kernel(
    q_c: jax.Array,
    w_qb: jax.Array,
    w_s: jax.Array,
    w_uk_t: jax.Array | tuple[jax.Array, jax.Array],
    cos_sin: jax.Array,
    split_rows: int,
    *,
    split_qpe: bool,
) -> tuple[jax.Array, ...]:
    total_tokens, k_dim = q_c.shape
    is_split_wuk = isinstance(w_uk_t, tuple)
    if is_split_wuk:
        w_uk_lo, w_uk_hi = w_uk_t
        if (
            w_uk_lo.ndim != 3
            or w_uk_lo.shape != w_uk_hi.shape
            or w_uk_lo.dtype != w_uk_hi.dtype
            or w_uk_lo.shape[0] <= 0
            or w_uk_lo.shape[0] % HEAD_BLOCK != 0
        ):
            raise ValueError(
                "Split w_uk_t halves must match [H/2, d_nope, rank] with "
                f"(H/2) % {HEAD_BLOCK} == 0; got lo={w_uk_lo.shape}/{w_uk_lo.dtype}, "
                f"hi={w_uk_hi.shape}/{w_uk_hi.dtype}"
            )
        half_heads, d_nope, rank = w_uk_lo.shape
        heads = half_heads * 2
        w_uk_shape = (heads, d_nope, rank)
    else:
        w_uk_lo = w_uk_hi = None
        w_uk_shape = w_uk_t.shape
        heads, d_nope, rank = w_uk_shape
    d_pe = cos_sin.shape[1]
    half_pe = d_pe // 2
    d_q = d_nope + d_pe
    n_out = heads * d_q
    tail_rows = total_tokens - split_rows
    tail_chunk = select_tail_chunk(tail_rows)
    if (
        not supports_fused_qb_rope_absorb_split(
            q_c.shape,
            q_c.dtype,
            w_qb.shape,
            w_qb.dtype,
            w_s.shape,
            w_uk_shape,
            cos_sin.shape,
            split_rows,
            bfloat16_dtype=jnp.bfloat16,
            fp8_dtype=jnp.float8_e4m3fn,
        )
        or tail_chunk is None
    ):
        raise ValueError(
            "Unsupported inputs for fused_qb_rope_absorb_split: "
            f"q_c={q_c.shape}/{q_c.dtype}, w_qb={w_qb.shape}/{w_qb.dtype}, "
            f"w_s={w_s.shape}, w_uk_t={w_uk_shape}, cos_sin={cos_sin.shape}, "
            f"split_rows={split_rows}"
        )

    n_hblk = heads // HEAD_BLOCK
    cols_blk = HEAD_BLOCK * d_q
    n_tail_chunks = tail_rows // tail_chunk

    x_q, x_s = quantize_tensor(q_c, w_qb.dtype)
    # Pad row scale and packed [cos, sin] to 128 lanes for aligned VMEM loads.
    x_s_128 = jnp.broadcast_to(x_s, (total_tokens, 128))
    cos_sin_128 = jnp.pad(cos_sin.astype(jnp.float32), ((0, 0), (0, 128 - d_pe)))
    w_s_2d = w_s.astype(jnp.float32).reshape(1, n_out)

    def _project_slice(
        xq_ref,
        xs_ref,
        cs_ref,
        wqb_ref,
        ws_v,
        wuk_ref,
        t0,
        nrows: int,
        wait_wuk=None,
    ) -> tuple[jax.Array, jax.Array]:
        xq_sl = xq_ref[pl.ds(t0, nrows), :]
        xs_sl = xs_ref[pl.ds(t0, nrows), :1]
        c_sl = cs_ref[pl.ds(t0, nrows), :half_pe]
        s_sl = cs_ref[pl.ds(t0, nrows), half_pe:d_pe]

        # Avoid per-head q_b_proj GEMMs: projecting all 8 heads in one
        # [nrows, K] @ [K, 8*d_q] matmul reuses xq_sl 8x across MXU tiles.
        q_raw = lax.dot_general(
            xq_sl,
            wqb_ref[...],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        q_raw = (q_raw * xs_sl * ws_v[None, :]).astype(q_c.dtype)
        if wait_wuk is not None:
            wait_wuk()

        ql_list = []
        qpe_list = []
        for h in range(HEAD_BLOCK):
            c0 = h * d_q
            ql_h = lax.dot_general(
                q_raw[:, c0 : c0 + d_nope],
                wuk_ref[h],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            ).astype(q_c.dtype)
            ql_list.append(ql_h)
            x1 = q_raw[:, c0 + d_nope : c0 + d_nope + half_pe].astype(jnp.float32)
            x2 = q_raw[:, c0 + d_nope + half_pe : c0 + d_q].astype(jnp.float32)
            qpe_h = jnp.concatenate(
                [x1 * c_sl - x2 * s_sl, x2 * c_sl + x1 * s_sl],
                axis=-1,
            ).astype(q_c.dtype)
            qpe_list.append(qpe_h)

        return (
            pltpu.einshape("htd->thd", jnp.stack(ql_list)),
            pltpu.einshape("htd->thd", jnp.stack(qpe_list)),
        )

    base_in_specs = [
        pl.BlockSpec(
            (total_tokens, k_dim),
            lambda hb: (0, 0),
            pipeline_mode=pl.Buffered(1),
        ),
        pl.BlockSpec(
            (total_tokens, 128),
            lambda hb: (0, 0),
            pipeline_mode=pl.Buffered(1),
        ),
        pl.BlockSpec(
            (total_tokens, 128),
            lambda hb: (0, 0),
            pipeline_mode=pl.Buffered(1),
        ),
        pl.BlockSpec((k_dim, cols_blk), lambda hb: (0, hb)),
        pl.BlockSpec((1, cols_blk), lambda hb: (0, hb)),
    ]
    compiler_params = pltpu.CompilerParams(
        dimension_semantics=("parallel",),
        vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes * 7 // 8,
    )

    if is_split_wuk:
        assert w_uk_lo is not None and w_uk_hi is not None
        half_hblk = n_hblk // 2
        in_specs_pair = [
            *base_in_specs,
            pl.BlockSpec(memory_space=pl.ANY),
            pl.BlockSpec(memory_space=pl.ANY),
        ]
        # Double-buffer w_uk_t (2 x 1.5 MiB) across hb iterations with arbitrary
        # (sequential) grid order so hb+1's HBM DMA overlaps the full tail block.
        pair_compiler_params = pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes * 7 // 8,
        )
        scratch_shapes = (
            pltpu.VMEM((2, HEAD_BLOCK, d_nope, rank), w_uk_lo.dtype),
            pltpu.SemaphoreType.DMA((2,)),
        )

        def _wuk_dma(
            wuk_lo_hbm, wuk_hi_hbm, wuk_vm, sem, blk_idx, slot, *, is_start: bool
        ) -> None:
            blk_arr = jnp.asarray(blk_idx, dtype=jnp.int32)
            is_lo = blk_arr < half_hblk
            dst = wuk_vm.at[slot]
            sref = sem.at[slot]

            @pl.when(is_lo)
            def _from_lo():
                blk = jnp.minimum(blk_arr, half_hblk - 1)
                op = pltpu.make_async_copy(
                    wuk_lo_hbm.at[pl.ds(blk * HEAD_BLOCK, HEAD_BLOCK), :, :],
                    dst,
                    sref,
                )
                if is_start:
                    op.start()
                else:
                    op.wait()

            @pl.when(jnp.logical_not(is_lo))
            def _from_hi():
                blk = jnp.maximum(blk_arr - half_hblk, 0)
                op = pltpu.make_async_copy(
                    wuk_hi_hbm.at[pl.ds(blk * HEAD_BLOCK, HEAD_BLOCK), :, :],
                    dst,
                    sref,
                )
                if is_start:
                    op.start()
                else:
                    op.wait()

        if split_qpe:

            def _kernel_split_pair(
                xq_ref,
                xs_ref,
                cs_ref,
                wqb_ref,
                ws_ref,
                wuk_lo_hbm,
                wuk_hi_hbm,
                ql_head_ref,
                ql_tail_ref,
                qpe_head_ref,
                qpe_tail_ref,
                wuk_vm,
                wuk_sem,
            ):
                hb = pl.program_id(0)
                slot = hb & 1
                nxt = hb + 1

                @pl.when(hb == 0)
                def _prime():
                    _wuk_dma(
                        wuk_lo_hbm, wuk_hi_hbm, wuk_vm, wuk_sem, 0, 0, is_start=True
                    )

                @pl.when(nxt < n_hblk)
                def _prefetch():
                    _wuk_dma(
                        wuk_lo_hbm,
                        wuk_hi_hbm,
                        wuk_vm,
                        wuk_sem,
                        nxt,
                        nxt & 1,
                        is_start=True,
                    )

                ws_v = ws_ref[0]
                cur_wuk = wuk_vm.at[slot]
                ql_h, qpe_h = _project_slice(
                    xq_ref,
                    xs_ref,
                    cs_ref,
                    wqb_ref,
                    ws_v,
                    cur_wuk,
                    0,
                    split_rows,
                    wait_wuk=lambda: _wuk_dma(
                        wuk_lo_hbm,
                        wuk_hi_hbm,
                        wuk_vm,
                        wuk_sem,
                        hb,
                        slot,
                        is_start=False,
                    ),
                )
                ql_head_ref[...] = ql_h
                qpe_head_ref[...] = qpe_h
                if n_tail_chunks == 1:
                    ql_t, qpe_t = _project_slice(
                        xq_ref,
                        xs_ref,
                        cs_ref,
                        wqb_ref,
                        ws_v,
                        cur_wuk,
                        split_rows,
                        tail_chunk,
                    )
                    ql_tail_ref[...] = ql_t
                    qpe_tail_ref[...] = qpe_t
                else:

                    def _tail_body(c, _):
                        t0 = split_rows + c * tail_chunk
                        ql_t, qpe_t = _project_slice(
                            xq_ref,
                            xs_ref,
                            cs_ref,
                            wqb_ref,
                            ws_v,
                            cur_wuk,
                            t0,
                            tail_chunk,
                        )
                        ql_tail_ref[pl.ds(c * tail_chunk, tail_chunk)] = ql_t
                        qpe_tail_ref[pl.ds(c * tail_chunk, tail_chunk)] = qpe_t

                    lax.fori_loop(0, n_tail_chunks, _tail_body, None)

            return pl.pallas_call(
                _kernel_split_pair,
                grid=(n_hblk,),
                in_specs=in_specs_pair,
                out_specs=[
                    pl.BlockSpec((split_rows, HEAD_BLOCK, rank), lambda hb: (0, hb, 0)),
                    pl.BlockSpec((tail_rows, HEAD_BLOCK, rank), lambda hb: (0, hb, 0)),
                    pl.BlockSpec((split_rows, HEAD_BLOCK, d_pe), lambda hb: (0, hb, 0)),
                    pl.BlockSpec((tail_rows, HEAD_BLOCK, d_pe), lambda hb: (0, hb, 0)),
                ],
                scratch_shapes=scratch_shapes,
                out_shape=(
                    jax.ShapeDtypeStruct((split_rows, heads, rank), q_c.dtype),
                    jax.ShapeDtypeStruct((tail_rows, heads, rank), q_c.dtype),
                    jax.ShapeDtypeStruct((split_rows, heads, d_pe), q_c.dtype),
                    jax.ShapeDtypeStruct((tail_rows, heads, d_pe), q_c.dtype),
                ),
                compiler_params=pair_compiler_params,
            )(x_q, x_s_128, cos_sin_128, w_qb, w_s_2d, w_uk_lo, w_uk_hi)

        def _kernel_joined_pair(
            xq_ref,
            xs_ref,
            cs_ref,
            wqb_ref,
            ws_ref,
            wuk_lo_hbm,
            wuk_hi_hbm,
            ql_head_ref,
            ql_tail_ref,
            qpe_ref,
            wuk_vm,
            wuk_sem,
        ):
            hb = pl.program_id(0)
            slot = hb & 1
            nxt = hb + 1

            @pl.when(hb == 0)
            def _prime():
                _wuk_dma(wuk_lo_hbm, wuk_hi_hbm, wuk_vm, wuk_sem, 0, 0, is_start=True)

            @pl.when(nxt < n_hblk)
            def _prefetch():
                _wuk_dma(
                    wuk_lo_hbm,
                    wuk_hi_hbm,
                    wuk_vm,
                    wuk_sem,
                    nxt,
                    nxt & 1,
                    is_start=True,
                )

            ws_v = ws_ref[0]
            cur_wuk = wuk_vm.at[slot]
            ql_h, qpe_h = _project_slice(
                xq_ref,
                xs_ref,
                cs_ref,
                wqb_ref,
                ws_v,
                cur_wuk,
                0,
                split_rows,
                wait_wuk=lambda: _wuk_dma(
                    wuk_lo_hbm, wuk_hi_hbm, wuk_vm, wuk_sem, hb, slot, is_start=False
                ),
            )
            ql_head_ref[...] = ql_h
            qpe_ref[pl.ds(0, split_rows)] = qpe_h
            if n_tail_chunks == 1:
                ql_t, qpe_t = _project_slice(
                    xq_ref,
                    xs_ref,
                    cs_ref,
                    wqb_ref,
                    ws_v,
                    cur_wuk,
                    split_rows,
                    tail_chunk,
                )
                ql_tail_ref[...] = ql_t
                qpe_ref[pl.ds(split_rows, tail_chunk)] = qpe_t
            else:

                def _tail_body(c, _):
                    t0 = split_rows + c * tail_chunk
                    ql_t, qpe_t = _project_slice(
                        xq_ref,
                        xs_ref,
                        cs_ref,
                        wqb_ref,
                        ws_v,
                        cur_wuk,
                        t0,
                        tail_chunk,
                    )
                    ql_tail_ref[pl.ds(c * tail_chunk, tail_chunk)] = ql_t
                    qpe_ref[pl.ds(t0, tail_chunk)] = qpe_t

                lax.fori_loop(0, n_tail_chunks, _tail_body, None)

        return pl.pallas_call(
            _kernel_joined_pair,
            grid=(n_hblk,),
            in_specs=in_specs_pair,
            out_specs=[
                pl.BlockSpec((split_rows, HEAD_BLOCK, rank), lambda hb: (0, hb, 0)),
                pl.BlockSpec((tail_rows, HEAD_BLOCK, rank), lambda hb: (0, hb, 0)),
                pl.BlockSpec((total_tokens, HEAD_BLOCK, d_pe), lambda hb: (0, hb, 0)),
            ],
            scratch_shapes=scratch_shapes,
            out_shape=(
                jax.ShapeDtypeStruct((split_rows, heads, rank), q_c.dtype),
                jax.ShapeDtypeStruct((tail_rows, heads, rank), q_c.dtype),
                jax.ShapeDtypeStruct((total_tokens, heads, d_pe), q_c.dtype),
            ),
            compiler_params=pair_compiler_params,
        )(x_q, x_s_128, cos_sin_128, w_qb, w_s_2d, w_uk_lo, w_uk_hi)

    in_specs = [
        *base_in_specs,
        pl.BlockSpec((HEAD_BLOCK, d_nope, rank), lambda hb: (hb, 0, 0)),
    ]

    if split_qpe:

        def _kernel_split(
            xq_ref,
            xs_ref,
            cs_ref,
            wqb_ref,
            ws_ref,
            wuk_ref,
            ql_head_ref,
            ql_tail_ref,
            qpe_head_ref,
            qpe_tail_ref,
        ):
            ws_v = ws_ref[0]
            ql_h, qpe_h = _project_slice(
                xq_ref, xs_ref, cs_ref, wqb_ref, ws_v, wuk_ref, 0, split_rows
            )
            ql_head_ref[...] = ql_h
            qpe_head_ref[...] = qpe_h
            if n_tail_chunks == 1:
                ql_t, qpe_t = _project_slice(
                    xq_ref,
                    xs_ref,
                    cs_ref,
                    wqb_ref,
                    ws_v,
                    wuk_ref,
                    split_rows,
                    tail_chunk,
                )
                ql_tail_ref[...] = ql_t
                qpe_tail_ref[...] = qpe_t
            else:

                def _tail_body(c, _):
                    t0 = split_rows + c * tail_chunk
                    ql_t, qpe_t = _project_slice(
                        xq_ref, xs_ref, cs_ref, wqb_ref, ws_v, wuk_ref, t0, tail_chunk
                    )
                    ql_tail_ref[pl.ds(c * tail_chunk, tail_chunk)] = ql_t
                    qpe_tail_ref[pl.ds(c * tail_chunk, tail_chunk)] = qpe_t

                lax.fori_loop(0, n_tail_chunks, _tail_body, None)

        return pl.pallas_call(
            _kernel_split,
            grid=(n_hblk,),
            in_specs=in_specs,
            out_specs=[
                pl.BlockSpec((split_rows, HEAD_BLOCK, rank), lambda hb: (0, hb, 0)),
                pl.BlockSpec((tail_rows, HEAD_BLOCK, rank), lambda hb: (0, hb, 0)),
                pl.BlockSpec((split_rows, HEAD_BLOCK, d_pe), lambda hb: (0, hb, 0)),
                pl.BlockSpec((tail_rows, HEAD_BLOCK, d_pe), lambda hb: (0, hb, 0)),
            ],
            out_shape=(
                jax.ShapeDtypeStruct((split_rows, heads, rank), q_c.dtype),
                jax.ShapeDtypeStruct((tail_rows, heads, rank), q_c.dtype),
                jax.ShapeDtypeStruct((split_rows, heads, d_pe), q_c.dtype),
                jax.ShapeDtypeStruct((tail_rows, heads, d_pe), q_c.dtype),
            ),
            compiler_params=compiler_params,
        )(x_q, x_s_128, cos_sin_128, w_qb, w_s_2d, w_uk_t)

    def _kernel_joined(
        xq_ref,
        xs_ref,
        cs_ref,
        wqb_ref,
        ws_ref,
        wuk_ref,
        ql_head_ref,
        ql_tail_ref,
        qpe_ref,
    ):
        ws_v = ws_ref[0]
        ql_h, qpe_h = _project_slice(
            xq_ref, xs_ref, cs_ref, wqb_ref, ws_v, wuk_ref, 0, split_rows
        )
        ql_head_ref[...] = ql_h
        qpe_ref[pl.ds(0, split_rows)] = qpe_h
        if n_tail_chunks == 1:
            ql_t, qpe_t = _project_slice(
                xq_ref, xs_ref, cs_ref, wqb_ref, ws_v, wuk_ref, split_rows, tail_chunk
            )
            ql_tail_ref[...] = ql_t
            qpe_ref[pl.ds(split_rows, tail_chunk)] = qpe_t
        else:
            # Avoid Python-unrolling tail chunks: fori_loop keeps compile time
            # constant across large prefill buckets (16K/32K+).
            def _tail_body(c, _):
                t0 = split_rows + c * tail_chunk
                ql_t, qpe_t = _project_slice(
                    xq_ref, xs_ref, cs_ref, wqb_ref, ws_v, wuk_ref, t0, tail_chunk
                )
                ql_tail_ref[pl.ds(c * tail_chunk, tail_chunk)] = ql_t
                qpe_ref[pl.ds(t0, tail_chunk)] = qpe_t

            lax.fori_loop(0, n_tail_chunks, _tail_body, None)

    return pl.pallas_call(
        _kernel_joined,
        grid=(n_hblk,),
        in_specs=in_specs,
        out_specs=[
            pl.BlockSpec((split_rows, HEAD_BLOCK, rank), lambda hb: (0, hb, 0)),
            pl.BlockSpec((tail_rows, HEAD_BLOCK, rank), lambda hb: (0, hb, 0)),
            pl.BlockSpec((total_tokens, HEAD_BLOCK, d_pe), lambda hb: (0, hb, 0)),
        ],
        out_shape=(
            jax.ShapeDtypeStruct((split_rows, heads, rank), q_c.dtype),
            jax.ShapeDtypeStruct((tail_rows, heads, rank), q_c.dtype),
            jax.ShapeDtypeStruct((total_tokens, heads, d_pe), q_c.dtype),
        ),
        compiler_params=compiler_params,
    )(x_q, x_s_128, cos_sin_128, w_qb, w_s_2d, w_uk_t)


def fused_qb_rope_absorb_split(
    q_c: jax.Array,
    w_qb: jax.Array,
    w_s: jax.Array,
    w_uk_t: jax.Array,
    cos_sin: jax.Array,
    split_rows: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Map `[T, K]` q_c -> `(ql_head[s, H, R], ql_tail[T-s, H, R], q_pe[T, H, 64])`.

    Fuses FP8 `q_b_proj`, NeoX half-split `q_pe` RoPE, and `W_UK_T` absorption
    in VMEM per 8-head block so the `[T, H*(d_nope+d_pe)]` query tensor never
    materializes in HBM. Requires `supports_fused_qb_rope_absorb_split` True.
    """
    ql_head, ql_tail, q_pe = _run_fused_qb_kernel(
        q_c, w_qb, w_s, w_uk_t, cos_sin, split_rows, split_qpe=False
    )
    return ql_head, ql_tail, q_pe


def fused_qb_rope_absorb_split_halves(
    q_c: jax.Array,
    w_qb: jax.Array,
    w_s: jax.Array,
    w_uk_t_lo: jax.Array,
    w_uk_t_hi: jax.Array,
    cos_sin: jax.Array,
    split_rows: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Two-buffer variant of `fused_qb_rope_absorb_split` taking disjoint halves."""
    ql_head, ql_tail, q_pe = _run_fused_qb_kernel(
        q_c, w_qb, w_s, (w_uk_t_lo, w_uk_t_hi), cos_sin, split_rows, split_qpe=False
    )
    return ql_head, ql_tail, q_pe


def fused_qb_rope_absorb_split_qpe(
    q_c: jax.Array,
    w_qb: jax.Array,
    w_s: jax.Array,
    w_uk_t: jax.Array | tuple[jax.Array, jax.Array],
    cos_sin: jax.Array,
    split_rows: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Map `[T, K]` q_c -> `(ql_head, ql_tail, q_pe_head, q_pe_tail)` directly.

    Avoids the downstream `[T, H, 64]` slice copy by writing head and tail
    `q_pe` slices into separate outputs directly from VMEM.
    """
    ql_head, ql_tail, q_pe_head, q_pe_tail = _run_fused_qb_kernel(
        q_c, w_qb, w_s, w_uk_t, cos_sin, split_rows, split_qpe=True
    )
    return ql_head, ql_tail, q_pe_head, q_pe_tail
