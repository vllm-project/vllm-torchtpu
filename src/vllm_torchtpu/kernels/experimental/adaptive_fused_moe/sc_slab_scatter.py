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
"""SparseCore slab scatter for ``adaptive_fused_moe`` (MOE_ADAPTIVE_SC_SLAB_SCATTER).

Each EP shard builds two int32 slab tables per MoE layer by scattering one
value per routed pair onto its own slab: the token number each slab row
computes (``shard_token_gather``) and, with quantized activations, the
activation row scale. With XLA's SparseCore scatter offload disabled
(``--xla_tpu_enable_offloading_scatter_to_sparsecore=false``) both lower to
TensorCore scatter fusions; on TPU v7x (Qwen3.8, TP8/EP8) they took 38-41 ms
of an 8K-token prefill step, more than the fused MoE kernels themselves.

This kernel does both scatters in one SparseCore pass:

* The output ``[0, padded_out)`` is split into disjoint stripes, one per
  vector subcore (32 on TPU v7x).
* ``slab_row``, the token numbers and (when quantized) the scale bits stream
  through in double-buffered ``CHUNK``-pair tiles (``pltpu.emit_pipeline``).
* Each subcore writes only the pairs landing in its stripe with
  ``plsc.store_scatter``. ``slab_row`` is injective over the rows a shard owns
  and off-shard pairs carry ``slab_row >= shard_stride`` (``local_slab_rows``),
  so writes never conflict and off-shard pairs are dropped.
* Each stripe is copied to HBM once.

The result equals ``jnp.zeros(n).at[slab_row].add(values, mode="drop")`` on
every row in ``[0, shard_stride)``; rows past ``shard_stride`` are zero.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.experimental.adaptive_fused_moe.host import (
    HIDDEN_LANE_BLOCK,
    gather_clamp_bounds,
)
from vllm_torchtpu.kernels.sparse_core import core_map_helper

LANES = 16
CHUNK = 2048
# Fewer routed pairs than this (decode-sized steps) keep the TensorCore
# scatter, which is cheap at that size.
SC_MIN_PAIRS = 1024
# Upper token bound for the fused per-token scale VMEM buffer in SparseCore.
MAX_FUSED_SCALE_TOKENS = 32768


def _align_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


class ScOps:
    """SparseCore primitives used by ``zero_buf`` / ``scatter_chunk``."""

    xp = jnp

    @staticmethod
    def full(x):
        return jnp.full((LANES,), x, jnp.int32)

    @staticmethod
    def iota():
        return lax.broadcasted_iota(jnp.int32, (LANES,), 0)

    @staticmethod
    def div(x, y):
        return lax.div(x, y)

    @staticmethod
    def load(ref, at):
        return ref[pl.ds(at, LANES)]

    @staticmethod
    def store(ref, at, x):
        ref[pl.ds(at, LANES)] = x

    @staticmethod
    def load_gather(ref, index, mask):
        return plsc.load_gather(ref, [index], mask=mask)

    @staticmethod
    def store_scatter(ref, index, x, mask):
        plsc.store_scatter(ref, [index], x, mask=mask)

    @staticmethod
    def loop(lower, upper, *, unroll=1):
        return plsc.parallel_loop(lower, upper, unroll=unroll)


def zero_buf(ops, buf_v, num_vec: int, unroll: int = 4) -> None:
    """Zero a ``num_vec * LANES`` int32 VMEM buffer."""
    z = ops.full(0)

    def body(i):
        ops.store(buf_v, i * LANES, z)

    ops.loop(0, num_vec, unroll=unroll)(body)


def scatter_chunk(
    ops,
    row_v,
    tok_in_v,
    scl_in_v,
    tok_out_v,
    scl_out_v,
    *,
    lo,
    valid_hi,
    num_vec: int,
    has_scale: bool,
    unroll: int = 4,
) -> None:
    """Scatter one ``num_vec * LANES`` chunk into the ``[lo, valid_hi)`` stripe."""
    lo_v = ops.full(lo)
    hi_v = ops.full(valid_hi)
    zero_v = ops.full(0)

    def body(i):
        at = i * LANES
        r = ops.load(row_v, at)
        t = ops.load(tok_in_v, at)
        m = (r >= lo_v) & (r < hi_v)
        local_r = ops.xp.where(m, r - lo_v, zero_v)
        ops.store_scatter(tok_out_v, local_r, t, m)
        if has_scale:
            s = ops.load(scl_in_v, at)
            ops.store_scatter(scl_out_v, local_r, s, m)

    ops.loop(0, num_vec, unroll=unroll)(body)


def scatter_chunk_fused(
    ops,
    row_v,
    scl_tok_v,
    chunk_off_v,
    tok_out_v,
    scl_out_v,
    *,
    lo,
    valid_hi,
    topk: int,
    chunk: int,
    num_vec: int,
    has_scale: bool,
    unroll: int = 4,
) -> None:
    """Fused SC scatter: synthesize ``tok_id = pair_idx // topk`` and gather ``[T]`` scale."""
    lo_v = ops.full(lo)
    hi_v = ops.full(valid_hi)
    zero_v = ops.full(0)
    topk_v = ops.full(topk)
    base_v = ops.load(chunk_off_v, 0)

    def body(i):
        at = i * LANES
        r = ops.load(row_v, at)
        m = (r >= lo_v) & (r < hi_v)
        pair_v = base_v + ops.full(at)
        t = ops.div(pair_v, topk_v)
        safe_t = ops.xp.where(m, t, zero_v)
        local_r = ops.xp.where(m, r - lo_v, zero_v)
        ops.store_scatter(tok_out_v, local_r, safe_t, m)
        if has_scale:
            s = ops.load_gather(scl_tok_v, safe_t, m)
            ops.store_scatter(scl_out_v, local_r, s, m)

    ops.loop(0, num_vec, unroll=unroll)(body)
    ops.store(chunk_off_v, 0, base_v + ops.full(chunk))


def _kernel_with_scale(
    row_hbm,
    tok_in_hbm,
    scl_in_hbm,
    out_tok_hbm,
    out_scl_hbm,
    tok_vmem,
    scl_vmem,
    sem,
    *,
    shard_stride: int,
    slice_len: int,
    chunk: int,
    num_chunks: int,
):
    worker = lax.axis_index(("core", "subcore"))
    lo = worker * slice_len
    valid_hi = jnp.minimum(lo + slice_len, jnp.int32(shard_stride))
    slice_vecs = slice_len // LANES
    chunk_vecs = chunk // LANES

    zero_buf(ScOps, tok_vmem, slice_vecs)
    zero_buf(ScOps, scl_vmem, slice_vecs)

    def pipeline_step(row_v, tok_in_v, scl_in_v):
        scatter_chunk(
            ScOps,
            row_v,
            tok_in_v,
            scl_in_v,
            tok_vmem,
            scl_vmem,
            lo=lo,
            valid_hi=valid_hi,
            num_vec=chunk_vecs,
            has_scale=True,
        )

    buf_count = 1 if num_chunks == 1 else 2
    spec = pl.BlockSpec((chunk,), lambda c: (c,), pipeline_mode=pl.Buffered(buf_count))
    pltpu.emit_pipeline(
        pipeline_step,
        grid=(num_chunks,),
        in_specs=(spec, spec, spec),
        out_specs=(),
    )(row_hbm, tok_in_hbm, scl_in_hbm)

    dst_slice = pl.ds(pl.multiple_of(worker * slice_len, LANES), slice_len)
    c_tok = pltpu.make_async_copy(tok_vmem, out_tok_hbm.at[dst_slice], sem)
    c_scl = pltpu.make_async_copy(scl_vmem, out_scl_hbm.at[dst_slice], sem)
    c_tok.start()
    c_scl.start()
    c_tok.wait()
    c_scl.wait()


def _kernel_tok_only(
    row_hbm,
    tok_in_hbm,
    out_tok_hbm,
    tok_vmem,
    sem,
    *,
    shard_stride: int,
    slice_len: int,
    chunk: int,
    num_chunks: int,
):
    worker = lax.axis_index(("core", "subcore"))
    lo = worker * slice_len
    valid_hi = jnp.minimum(lo + slice_len, jnp.int32(shard_stride))
    slice_vecs = slice_len // LANES
    chunk_vecs = chunk // LANES

    zero_buf(ScOps, tok_vmem, slice_vecs)

    def pipeline_step(row_v, tok_in_v):
        scatter_chunk(
            ScOps,
            row_v,
            tok_in_v,
            None,
            tok_vmem,
            None,
            lo=lo,
            valid_hi=valid_hi,
            num_vec=chunk_vecs,
            has_scale=False,
        )

    buf_count = 1 if num_chunks == 1 else 2
    spec = pl.BlockSpec((chunk,), lambda c: (c,), pipeline_mode=pl.Buffered(buf_count))
    pltpu.emit_pipeline(
        pipeline_step,
        grid=(num_chunks,),
        in_specs=(spec, spec),
        out_specs=(),
    )(row_hbm, tok_in_hbm)

    dst_slice = pl.ds(pl.multiple_of(worker * slice_len, LANES), slice_len)
    c_tok = pltpu.make_async_copy(tok_vmem, out_tok_hbm.at[dst_slice], sem)
    c_tok.start()
    c_tok.wait()


def _kernel_fused_with_scale(
    row_hbm,
    scl_tok_hbm,
    out_tok_hbm,
    out_scl_hbm,
    tok_vmem,
    scl_vmem,
    scl_tok_vmem,
    chunk_off_vmem,
    sem_in,
    sem_out,
    *,
    shard_stride: int,
    slice_len: int,
    topk: int,
    chunk: int,
    num_chunks: int,
):
    worker = lax.axis_index(("core", "subcore"))
    lo = worker * slice_len
    valid_hi = jnp.minimum(lo + slice_len, jnp.int32(shard_stride))
    slice_vecs = slice_len // LANES
    chunk_vecs = chunk // LANES

    c_scl_in = pltpu.make_async_copy(scl_tok_hbm, scl_tok_vmem, sem_in)
    c_scl_in.start()
    zero_buf(ScOps, tok_vmem, slice_vecs)
    zero_buf(ScOps, scl_vmem, slice_vecs)
    ScOps.store(chunk_off_vmem, 0, ScOps.iota())
    c_scl_in.wait()

    def pipeline_step(row_v):
        scatter_chunk_fused(
            ScOps,
            row_v,
            scl_tok_vmem,
            chunk_off_vmem,
            tok_vmem,
            scl_vmem,
            lo=lo,
            valid_hi=valid_hi,
            topk=topk,
            chunk=chunk,
            num_vec=chunk_vecs,
            has_scale=True,
        )

    buf_count = 1 if num_chunks == 1 else 2
    spec = pl.BlockSpec((chunk,), lambda c: (c,), pipeline_mode=pl.Buffered(buf_count))
    pltpu.emit_pipeline(
        pipeline_step,
        grid=(num_chunks,),
        in_specs=(spec,),
        out_specs=(),
    )(row_hbm)

    dst_slice = pl.ds(pl.multiple_of(worker * slice_len, LANES), slice_len)
    c_tok = pltpu.make_async_copy(tok_vmem, out_tok_hbm.at[dst_slice], sem_out)
    c_scl = pltpu.make_async_copy(scl_vmem, out_scl_hbm.at[dst_slice], sem_out)
    c_tok.start()
    c_scl.start()
    c_tok.wait()
    c_scl.wait()


def _kernel_fused_tok_only(
    row_hbm,
    out_tok_hbm,
    tok_vmem,
    chunk_off_vmem,
    sem_out,
    *,
    shard_stride: int,
    slice_len: int,
    topk: int,
    chunk: int,
    num_chunks: int,
):
    worker = lax.axis_index(("core", "subcore"))
    lo = worker * slice_len
    valid_hi = jnp.minimum(lo + slice_len, jnp.int32(shard_stride))
    slice_vecs = slice_len // LANES
    chunk_vecs = chunk // LANES

    zero_buf(ScOps, tok_vmem, slice_vecs)
    ScOps.store(chunk_off_vmem, 0, ScOps.iota())

    def pipeline_step(row_v):
        scatter_chunk_fused(
            ScOps,
            row_v,
            None,
            chunk_off_vmem,
            tok_vmem,
            None,
            lo=lo,
            valid_hi=valid_hi,
            topk=topk,
            chunk=chunk,
            num_vec=chunk_vecs,
            has_scale=False,
        )

    buf_count = 1 if num_chunks == 1 else 2
    spec = pl.BlockSpec((chunk,), lambda c: (c,), pipeline_mode=pl.Buffered(buf_count))
    pltpu.emit_pipeline(
        pipeline_step,
        grid=(num_chunks,),
        in_specs=(spec,),
        out_specs=(),
    )(row_hbm)

    dst_slice = pl.ds(pl.multiple_of(worker * slice_len, LANES), slice_len)
    c_tok = pltpu.make_async_copy(tok_vmem, out_tok_hbm.at[dst_slice], sem_out)
    c_tok.start()
    c_tok.wait()


def sc_slab_scatter_available() -> bool:
    """True when TPU SparseCore with 16 SIMD lanes is available."""
    try:
        sc = pltpu.get_tpu_info().sparse_core
    except Exception:
        return False
    return sc is not None and sc.num_lanes == LANES


def sc_fused_slab_scatter(
    slab_row: jax.Array,
    tok_ids: jax.Array | None,
    scale_bits: jax.Array | None,
    *,
    shard_stride: int,
    scale_out_len: int = 0,
    tok_out_len: int = 0,
    n_tokens: int = 0,
    topk: int = 0,
    fused_gather: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Fused SparseCore scatter for ``token_gather`` and optional ``scale_slab``.

    When ``fused_gather=True`` (``MOE_ADAPTIVE_SC_FUSED_GATHER=1`` and
    ``n_tokens <= MAX_FUSED_SCALE_TOKENS``):
      * ``tok_ids`` may be ``None`` (pair token indices ``pair_idx // topk`` are
        computed inside SparseCore vector ALU).
      * ``scale_bits`` has per-token shape ``[n_tokens]`` (not ``[n_tokens * topk]``).
      * ``tok_out_len`` (default ``shard_stride``) may include the trailing
        ``token_gather_window_rows(capacity)`` zero padding directly.
    """
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None or sc.num_lanes != LANES:
        raise NotImplementedError("16-lane TPU SparseCore is required")

    n = int(slab_row.shape[0])
    has_scale = scale_bits is not None
    actual_tok_out = max(int(shard_stride), int(tok_out_len))
    out_len = max(actual_tok_out, int(scale_out_len) if has_scale else 0)
    workers = sc.num_cores * sc.num_subcores
    slice_len = _align_up(out_len, workers * LANES) // workers
    padded_out = slice_len * workers

    chunk = CHUNK if n >= CHUNK else _align_up(max(n, LANES), LANES)
    n_pad = _align_up(n, chunk)
    num_chunks = n_pad // chunk

    row_in = slab_row.astype(jnp.int32).reshape(-1)
    if n_pad > n:
        row_in = jnp.pad(
            row_in, (0, n_pad - n), constant_values=jnp.int32(shard_stride)
        )

    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    compiler_params = pltpu.CompilerParams(
        disable_bounds_checks=True, needs_layout_passes=False
    )

    use_fused = (
        bool(fused_gather)
        and topk >= 1
        and n_tokens >= 1
        and n_tokens <= MAX_FUSED_SCALE_TOKENS
        and n == n_tokens * topk
    )
    if use_fused:
        if has_scale:
            scl_tok = scale_bits.astype(jnp.int32).reshape(-1)
            t_pad = _align_up(max(int(scl_tok.shape[0]), LANES), LANES)
            if t_pad > int(scl_tok.shape[0]):
                scl_tok = jnp.pad(
                    scl_tok,
                    (0, t_pad - int(scl_tok.shape[0])),
                    constant_values=jnp.int32(0),
                )
            out_tok, out_scl = core_map_helper.kernel(
                functools.partial(
                    _kernel_fused_with_scale,
                    shard_stride=int(shard_stride),
                    slice_len=slice_len,
                    topk=int(topk),
                    chunk=chunk,
                    num_chunks=num_chunks,
                ),
                out_type=(
                    jax.ShapeDtypeStruct((padded_out,), jnp.int32),
                    jax.ShapeDtypeStruct((padded_out,), jnp.int32),
                ),
                mesh=mesh,
                scratch_types=(
                    pltpu.VMEM((slice_len,), jnp.int32),
                    pltpu.VMEM((slice_len,), jnp.int32),
                    pltpu.VMEM((t_pad,), jnp.int32),
                    pltpu.VMEM((LANES,), jnp.int32),
                    pltpu.SemaphoreType.DMA(()),
                    pltpu.SemaphoreType.DMA(()),
                ),
                compiler_params=compiler_params,
                name=f"moe_sc_slab_scatter_fg_s_n{n_pad}_k{topk}_o{padded_out}",
            )(row_in, scl_tok)
            return out_tok[:actual_tok_out], out_scl[:scale_out_len]

        out_tok = core_map_helper.kernel(
            functools.partial(
                _kernel_fused_tok_only,
                shard_stride=int(shard_stride),
                slice_len=slice_len,
                topk=int(topk),
                chunk=chunk,
                num_chunks=num_chunks,
            ),
            out_type=jax.ShapeDtypeStruct((padded_out,), jnp.int32),
            mesh=mesh,
            scratch_types=(
                pltpu.VMEM((slice_len,), jnp.int32),
                pltpu.VMEM((LANES,), jnp.int32),
                pltpu.SemaphoreType.DMA(()),
            ),
            compiler_params=compiler_params,
            name=f"moe_sc_slab_scatter_fg_n{n_pad}_k{topk}_o{padded_out}",
        )(row_in)
        return out_tok[:actual_tok_out], None

    if tok_ids is None:
        tok_ids = jnp.broadcast_to(
            jnp.arange(n_tokens, dtype=jnp.int32)[:, None], (n_tokens, topk)
        ).reshape(-1)
    tok_in = tok_ids.astype(jnp.int32).reshape(-1)
    if n_pad > n:
        tok_in = jnp.pad(tok_in, (0, n_pad - n), constant_values=jnp.int32(0))

    if has_scale:
        scl_in = scale_bits.astype(jnp.int32).reshape(-1)
        if int(scl_in.shape[0]) == n_tokens and topk > 1:
            scl_in = jnp.repeat(scl_in, topk)
        if n_pad > n:
            scl_in = jnp.pad(scl_in, (0, n_pad - n), constant_values=jnp.int32(0))
        out_tok, out_scl = core_map_helper.kernel(
            functools.partial(
                _kernel_with_scale,
                shard_stride=int(shard_stride),
                slice_len=slice_len,
                chunk=chunk,
                num_chunks=num_chunks,
            ),
            out_type=(
                jax.ShapeDtypeStruct((padded_out,), jnp.int32),
                jax.ShapeDtypeStruct((padded_out,), jnp.int32),
            ),
            mesh=mesh,
            scratch_types=(
                pltpu.VMEM((slice_len,), jnp.int32),
                pltpu.VMEM((slice_len,), jnp.int32),
                pltpu.SemaphoreType.DMA(()),
            ),
            compiler_params=compiler_params,
            name=f"moe_sc_slab_scatter_s_n{n_pad}_o{padded_out}",
        )(row_in, tok_in, scl_in)
        return out_tok[:actual_tok_out], out_scl[:scale_out_len]

    out_tok = core_map_helper.kernel(
        functools.partial(
            _kernel_tok_only,
            shard_stride=int(shard_stride),
            slice_len=slice_len,
            chunk=chunk,
            num_chunks=num_chunks,
        ),
        out_type=jax.ShapeDtypeStruct((padded_out,), jnp.int32),
        mesh=mesh,
        scratch_types=(
            pltpu.VMEM((slice_len,), jnp.int32),
            pltpu.SemaphoreType.DMA(()),
        ),
        compiler_params=compiler_params,
        name=f"moe_sc_slab_scatter_n{n_pad}_o{padded_out}",
    )(row_in, tok_in)
    return out_tok[:actual_tok_out], None


def sc_slab_tables(
    slab_row: jax.Array,
    scale_bits: jax.Array | None,
    *,
    n_tokens: int,
    topk: int,
    shard_stride: int,
    window_rows: int,
    scale_rows: int,
    fused_gather: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """``(token_gather, scale_slab)`` exactly as ``adaptive_fused_moe`` builds
    them, from one ``sc_fused_slab_scatter``: the token numbers clipped to
    ``gather_clamp_bounds`` and padded by ``window_rows``, and the per-token
    ``scale_bits`` (``None`` when unquantized) as the f32
    ``[scale_rows, HIDDEN_LANE_BLOCK]`` view."""
    if fused_gather and n_tokens <= MAX_FUSED_SCALE_TOKENS:
        token_gather, scl = sc_fused_slab_scatter(
            slab_row,
            None,
            scale_bits,
            shard_stride=shard_stride,
            scale_out_len=scale_rows * HIDDEN_LANE_BLOCK,
            tok_out_len=shard_stride + window_rows,
            n_tokens=n_tokens,
            topk=topk,
            fused_gather=True,
        )
    else:
        tok_of_pair = jnp.broadcast_to(
            jnp.arange(n_tokens, dtype=jnp.int32)[:, None], (n_tokens, topk)
        ).reshape(-1)
        pair_scales = None if scale_bits is None else jnp.repeat(scale_bits, topk)
        tok, scl = sc_fused_slab_scatter(
            slab_row,
            tok_of_pair,
            pair_scales,
            shard_stride=shard_stride,
            scale_out_len=scale_rows * HIDDEN_LANE_BLOCK,
        )
        lo, hi = gather_clamp_bounds(n_tokens)
        token_gather = jnp.pad(
            jnp.clip(tok, jnp.int32(lo), jnp.int32(hi)), (0, window_rows)
        )
    scale_slab = (
        None
        if scl is None
        else lax.bitcast_convert_type(scl, jnp.float32).reshape(
            scale_rows, HIDDEN_LANE_BLOCK
        )
    )
    return token_gather, scale_slab
