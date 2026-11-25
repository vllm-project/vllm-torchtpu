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
"""Typed access to byte-regions of the unified KV pool.

The pool is the stock attention-shaped KV cache
`(num_blocks, block_size, num_kv_heads * 2, head_size)` in the KV dtype —
the one tensor per physical buffer that every sharing layer receives.
Attention consumes it natively. Consumers that pack other element types
into a block's byte-region (the GDN SSM and Conv states) go
through these adapters: per-request gather/scatter of a token-range,
reinterpreted in VMEM via Mosaic `ref.bitcast` when the element type
differs — the only place a cross-dtype view is free on TPU (an XLA-level
bitcast of a pool slice forces a pool-sized layout copy per step).

Regions are token-ranges `[tok0, tok0 + ntok)` within a block. Aligned
ranges (`tok0 % ntok == 0`) are served by a single block spec covering
exactly the region, so one grid step moves only the region's bytes;
unaligned ranges decompose into aligned gcd-sized chunks. Scatters alias
the pool in place (`input_output_aliases`); the unwritten complement is
preserved via the HBM alias, with no read-modify-write.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import typed_ldst


def _pool_geometry(pool):
    """(block_size, payload dims between token and lane, lanes)."""
    assert pool.ndim >= 3, pool.shape
    return pool.shape[1], tuple(pool.shape[2:-1]), pool.shape[-1]


def _bitcast_payload(payload: tuple[int, ...], pool_dtype, out_dtype):
    """Payload dims of a region block after an in-kernel bitcast.

    Mosaic ``ref.bitcast`` rescales the second-minor dim by the
    element-size ratio: the last payload dim, or — for a 3-D pool
    ``(nb, rows, lanes)`` whose block ref is already 2-D — the token/row
    dim itself (handled by the callers via ``_out_rows``).
    """
    elem_in = jnp.dtype(pool_dtype).itemsize
    elem_out = jnp.dtype(out_dtype).itemsize
    if not payload:
        return payload
    last = payload[-1] * elem_in
    assert last % elem_out == 0, (payload, pool_dtype, out_dtype)
    return payload[:-1] + (last // elem_out,)


def _rescale_rows(rows: int, pool_dtype, out_dtype) -> int:
    elem_in = jnp.dtype(pool_dtype).itemsize
    elem_out = jnp.dtype(out_dtype).itemsize
    scaled = rows * elem_in
    assert scaled % elem_out == 0, (rows, pool_dtype, out_dtype)
    return scaled // elem_out


def _out_rows(payload: tuple[int, ...], ntok: int) -> int:
    rows = ntok
    for dim in payload:
        rows *= dim
    return rows


def gather_region(
    pool,
    state_indices,
    *,
    tok0: int,
    ntok: int,
    out_dtype,
    out_lanes: int | None = None,
    split: int = 1,
):
    """Per-request typed read of a pool token-range.

    ``split`` > 1 addresses a MANAGER-block token range on a pool born at
    kernel granularity (a manager block is `split` consecutive kernel
    blocks; ``state_indices`` are manager ids): whole-kernel-block ranges
    go through one DMA window per request; a range inside a single kernel
    block goes through the plain path at the kernel-block offset. Ranges
    that straddle kernel blocks partially are unsupported (no current
    region shape needs them).

    Plain path (split == 1): per-request typed read of a pool token-range.

    pool: (num_blocks, block_size, heads2, lanes) KV dtype.
    state_indices: (num_reqs,) int32 block ids.
    returns: (num_reqs, rows, out_lanes or lanes) out_dtype. With
    ``out_lanes`` (a divisor of the pool's lane count) the kernel splits
    lanes in place, so callers reshaping to a narrower-lane state shape
    pay no XLA lane-crossing relayout; the remaining outside reshape is
    lane-preserving (free).
    """
    if split > 1:
        kernel_bs = pool.shape[1]
        if tok0 % kernel_bs == 0 and ntok % kernel_bs == 0:
            return _gather_window(
                pool,
                state_indices,
                split=split,
                kb0=tok0 // kernel_bs,
                nblocks=ntok // kernel_bs,
                out_dtype=out_dtype,
                out_lanes=out_lanes,
            )
        kb, local0 = divmod(tok0, kernel_bs)
        if local0 + ntok > kernel_bs:
            raise NotImplementedError(
                "region partially straddles kernel blocks: "
                f"tok0={tok0} ntok={ntok} kernel_block={kernel_bs}"
            )
        return gather_region(
            pool,
            state_indices * split + kb,
            tok0=local0,
            ntok=ntok,
            out_dtype=out_dtype,
            out_lanes=out_lanes,
        )

    na = state_indices.shape[0]
    block_size, payload, lanes = _pool_geometry(pool)
    assert tok0 + ntok <= block_size
    # Unaligned regions (tok0 % ntok != 0) move as chunks of
    # g = gcd(tok0, ntok) tokens, one grid step per (request, chunk) pair of
    # a single pallas_call — chaining one aliased call per chunk would race
    # on the shared pool buffer. The output keeps one row-block per chunk so
    # the block dims stay equal to the array dims (Pallas's escape hatch for
    # sub-8 rows).
    g = ntok if tok0 % ntok == 0 else math.gcd(tok0, ntok)
    chunks = ntok // g
    same_dtype = jnp.dtype(pool.dtype) == jnp.dtype(out_dtype)
    out_payload = (
        payload if same_dtype else _bitcast_payload(payload, pool.dtype, out_dtype)
    )
    out_rows = _out_rows(out_payload, g)
    if not payload and not same_dtype:
        # 3-D pool: the block ref is 2-D and the bitcast rescales the
        # token/row dim itself (the old validated sublane convention).
        out_rows = _rescale_rows(g, pool.dtype, out_dtype)
    lane_split = 1
    if out_lanes is not None and out_lanes != lanes:
        assert lanes % out_lanes == 0, (lanes, out_lanes)
        lane_split = lanes // out_lanes
        out_rows *= lane_split
    o_lanes = lanes // lane_split
    pad = (0,) * (len(payload) + 1)

    def _kernel(sidx_ref, pool_ref, o_ref):
        o_ref[...] = typed_ldst.load_typed(
            pool_ref.at[0], view_dtype=out_dtype, lane_split=lane_split
        )[None]

    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na * chunks,),
            in_specs=[
                pl.BlockSpec(
                    (1, g) + payload + (lanes,),
                    lambda p, s: (s[p // chunks], (tok0 + (p % chunks) * g) // g) + pad,
                )
            ],
            out_specs=pl.BlockSpec((1, out_rows, o_lanes), lambda p, s: (p, 0, 0)),
        ),
        out_shape=jax.ShapeDtypeStruct((na * chunks, out_rows, o_lanes), out_dtype),
    )(state_indices, pool).reshape(na, chunks * out_rows, o_lanes)


def _gather_window(
    pool,
    mgr_indices,
    *,
    split: int,
    kb0: int,
    nblocks: int,
    out_dtype,
    out_lanes: int | None = None,
):
    """Typed read of `nblocks` whole kernel blocks inside each request's
    manager block, one grid step and ONE DMA window per request.

    The pool is born at kernel granularity; a manager block is `split`
    consecutive kernel blocks (upstream map_to_kernel_blocks), so the
    window (split, block_size, ...) indexed by the manager id is always
    aligned (element offset = mgr * split). Reading the whole manager
    window costs a little extra DMA but restores the single-window-per-
    request shape whose per-window setup dominates the region cost.
    """
    assert kb0 >= 0 and kb0 + nblocks <= split, (kb0, nblocks, split)
    na = mgr_indices.shape[0]
    block_size, payload, lanes = _pool_geometry(pool)
    same_dtype = jnp.dtype(pool.dtype) == jnp.dtype(out_dtype)
    out_payload = (
        payload if same_dtype else _bitcast_payload(payload, pool.dtype, out_dtype)
    )
    rows_pb = _out_rows(out_payload, block_size)
    if not payload and not same_dtype:
        rows_pb = _rescale_rows(block_size, pool.dtype, out_dtype)
    lane_split = 1
    if out_lanes is not None and out_lanes != lanes:
        assert lanes % out_lanes == 0, (lanes, out_lanes)
        lane_split = lanes // out_lanes
        rows_pb *= lane_split
    o_lanes = lanes // lane_split
    pad = (0,) * (len(payload) + 1)

    def _kernel(sidx_ref, pool_ref, o_ref):
        for j in range(nblocks):
            o_ref[0, j * rows_pb : (j + 1) * rows_pb, :] = typed_ldst.load_typed(
                pool_ref.at[kb0 + j], view_dtype=out_dtype, lane_split=lane_split
            )

    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na,),
            in_specs=[
                pl.BlockSpec(
                    (split, block_size) + payload + (lanes,),
                    lambda i, s: (s[i], 0) + pad,
                )
            ],
            out_specs=pl.BlockSpec(
                (1, nblocks * rows_pb, o_lanes), lambda i, s: (i, 0, 0)
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((na, nblocks * rows_pb, o_lanes), out_dtype),
    )(mgr_indices, pool)


def _scatter_window(pool, vals, mgr_indices, *, split: int, kb0: int, nblocks: int):
    """Typed write of `nblocks` whole kernel blocks inside each request's
    manager block (in place), one grid step per request; vals is the
    gather shape. The aliased output window covers the whole
    manager block, so the non-region kernel blocks are copied through
    from the aliased input to satisfy the full-write rule.
    """
    assert kb0 >= 0 and kb0 + nblocks <= split, (kb0, nblocks, split)
    na = mgr_indices.shape[0]
    block_size, payload, lanes = _pool_geometry(pool)
    same_dtype = jnp.dtype(pool.dtype) == jnp.dtype(vals.dtype)
    val_payload = (
        payload if same_dtype else _bitcast_payload(payload, pool.dtype, vals.dtype)
    )
    rows_pb = _out_rows(val_payload, block_size)
    if not payload and not same_dtype:
        rows_pb = _rescale_rows(block_size, pool.dtype, vals.dtype)
    v_lanes = vals.shape[-1]
    lane_split = 1
    if v_lanes != lanes:
        assert lanes % v_lanes == 0, (lanes, v_lanes)
        lane_split = lanes // v_lanes
    v_rows_pb = rows_pb * lane_split
    assert vals.shape == (na, nblocks * v_rows_pb, v_lanes), (
        vals.shape,
        nblocks,
        v_rows_pb,
        v_lanes,
    )
    pad = (0,) * (len(payload) + 1)

    def _kernel(sidx_ref, val_ref, pool_in_ref, pool_out_ref):
        for j in range(split):
            if kb0 <= j < kb0 + nblocks:
                typed_ldst.store_typed(
                    pool_out_ref.at[j],
                    val_ref[0, (j - kb0) * v_rows_pb : (j - kb0 + 1) * v_rows_pb, :],
                    lane_split=lane_split,
                )
            else:
                # full-write rule: pass the untouched kernel blocks through
                pool_out_ref.at[j][...] = pool_in_ref.at[j][...]

    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na,),
            in_specs=[
                pl.BlockSpec((1, nblocks * v_rows_pb, v_lanes), lambda i, s: (i, 0, 0)),
                pl.BlockSpec(
                    (split, block_size) + payload + (lanes,),
                    lambda i, s: (s[i], 0) + pad,
                ),
            ],
            out_specs=pl.BlockSpec(
                (split, block_size) + payload + (lanes,), lambda i, s: (s[i], 0) + pad
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={2: 0},
    )(mgr_indices, vals, pool)


def scatter_region(pool, vals, state_indices, *, tok0: int, ntok: int, split: int = 1):
    """Per-request typed write of a pool token-range (in place); mirrors
    ``gather_region`` including the ``split`` manager-range routing.

    pool: (num_blocks, block_size, heads2, lanes) KV dtype (aliased;
    unwritten bytes preserved).
    vals: (num_reqs, rows, lanes) — the gather_region shape.
    returns: the updated pool.
    """
    if split > 1:
        kernel_bs = pool.shape[1]
        if tok0 % kernel_bs == 0 and ntok % kernel_bs == 0:
            return _scatter_window(
                pool,
                vals,
                state_indices,
                split=split,
                kb0=tok0 // kernel_bs,
                nblocks=ntok // kernel_bs,
            )
        kb, local0 = divmod(tok0, kernel_bs)
        if local0 + ntok > kernel_bs:
            raise NotImplementedError(
                "region partially straddles kernel blocks: "
                f"tok0={tok0} ntok={ntok} kernel_block={kernel_bs}"
            )
        return scatter_region(
            pool, vals, state_indices * split + kb, tok0=local0, ntok=ntok
        )

    na = state_indices.shape[0]
    block_size, payload, lanes = _pool_geometry(pool)
    assert tok0 + ntok <= block_size
    # Unaligned regions move as aligned gcd-sized chunks, one grid step per
    # (request, chunk) pair; see gather_region.
    g = ntok if tok0 % ntok == 0 else math.gcd(tok0, ntok)
    chunks = ntok // g
    same_dtype = jnp.dtype(pool.dtype) == jnp.dtype(vals.dtype)
    val_payload = (
        payload if same_dtype else _bitcast_payload(payload, pool.dtype, vals.dtype)
    )
    out_rows = _out_rows(val_payload, g)
    if not payload and not same_dtype:
        out_rows = _rescale_rows(g, pool.dtype, vals.dtype)
    v_lanes = vals.shape[-1]
    lane_split = 1
    if v_lanes != lanes:
        # Narrow-lane vals (the gather out_lanes shape): merge lanes back
        # in-kernel; 128-aligned lane concat preserves row-major order.
        assert lanes % v_lanes == 0, (lanes, v_lanes)
        lane_split = lanes // v_lanes
    v_rows = out_rows * lane_split
    assert vals.shape == (na, chunks * v_rows, v_lanes), (
        vals.shape,
        chunks * v_rows,
        v_lanes,
    )
    pad = (0,) * (len(payload) + 1)

    def _kernel(sidx_ref, val_ref, pool_in_ref, pool_out_ref):
        # The output block covers exactly the region chunk and is fully
        # written; the complement is preserved via the HBM alias.
        typed_ldst.store_typed(pool_out_ref.at[0], val_ref[0], lane_split=lane_split)

    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na * chunks,),
            in_specs=[
                pl.BlockSpec((1, v_rows, v_lanes), lambda p, s: (p, 0, 0)),
                # Aliased with the output; never read, so leave it in HBM.
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(
                (1, g) + payload + (lanes,),
                lambda p, s: (s[p // chunks], (tok0 + (p % chunks) * g) // g) + pad,
            ),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={2: 0},
    )(state_indices, vals.reshape(na * chunks, v_rows, v_lanes), pool)


def copy_pairs_in_pool(src_indices, dst_indices, num_blocks: int):
    """True for each copy pair whose src and dst are both in `[0, num_blocks)`."""
    return (
        (src_indices >= 0)
        & (src_indices < num_blocks)
        & (dst_indices >= 0)
        & (dst_indices < num_blocks)
    )


def null_out_of_range_pairs(src_indices, dst_indices, num_blocks: int, *, in_pool=None):
    """Replaces every copy pair with an index outside `[0, num_blocks)` by `(0, 0)`.

    Block 0 is vLLM's reserved null block and `(0, 0)` is the copy's padding
    pair: it copies the null block onto itself, so a replaced pair writes
    nothing outside the pool and changes no block in it. A pair with both
    indices in range is returned unchanged.

    `in_pool`, if given, must be `copy_pairs_in_pool(src_indices, dst_indices,
    num_blocks)`. A caller that also needs that mask passes it, so that it is
    computed once.
    """
    if in_pool is None:
        in_pool = copy_pairs_in_pool(src_indices, dst_indices, num_blocks)
    return (
        jnp.where(in_pool, src_indices, jnp.zeros_like(src_indices)),
        jnp.where(in_pool, dst_indices, jnp.zeros_like(dst_indices)),
    )


def _replaced_pair_message(num_blocks: int) -> str:
    return (
        f"block copy: (src, dst) pair outside [0, {num_blocks}) replaced by the "
        "(0, 0) padding pair"
    )


def _report_on_host(num_blocks, in_pool, src_indices, dst_indices):
    jax.debug.callback(
        functools.partial(_raise_on_replaced_pairs, num_blocks),
        in_pool,
        src_indices,
        dst_indices,
    )
    return jnp.zeros((), jnp.int32)


def _raise_on_replaced_pairs(num_blocks, in_pool, src_indices, dst_indices):
    bad = np.flatnonzero(~np.asarray(in_pool))
    if bad.size:
        pairs = [(int(src_indices[i]), int(dst_indices[i])) for i in bad[:8]]
        raise RuntimeError(
            f"{_replaced_pair_message(num_blocks)}: {bad.size} pair(s), "
            f"first ones: {pairs}"
        )


def copy_blocks(pool, src_indices, dst_indices):
    """Whole-block copies within the pool: pool[dst[i]] = pool[src[i]].

    Seeds a request's new mamba state block from its previous one whenever
    the block-table-derived state block advances (chunked-prefill boundary,
    decode crossing, or prefix-cache resume). Pairs with src == dst are
    no-ops used as padding.

    Valid indices are `[0, pool.shape[0])`. A pair whose src or dst is
    outside that range is replaced by the `(0, 0)` padding pair
    (`null_out_of_range_pairs`) and changes no block. Pairs inside the pool
    are copied unchanged.

    Replaced pairs are reported only when `pl.enable_debug_checks` is set as
    the program is traced; otherwise nothing is added to the program. With it
    set, the report depends on where the program runs:
    - on TPU, the copy kernel runs `pl.debug_check` on each pair, like the
      other kernel debug checks, so a replaced pair fails the kernel's check;
    - on the host (CPU, including Pallas's TPU interpret mode, where
      `pl.debug_check` does nothing), a host callback raises `RuntimeError`
      naming the first replaced pairs.
    """
    num_blocks = pool.shape[0]
    in_pool = copy_pairs_in_pool(src_indices, dst_indices, num_blocks)
    src, dst = null_out_of_range_pairs(
        src_indices, dst_indices, num_blocks, in_pool=in_pool
    )
    if not pl.enable_debug_checks.value:
        return _copy_blocks_unchecked(pool, src, dst)
    # The callback only where it can run: a TPU program holds none.
    checked = lax.platform_dependent(
        in_pool,
        src_indices,
        dst_indices,
        cpu=functools.partial(_report_on_host, num_blocks),
        default=lambda *_: jnp.zeros((), jnp.int32),
    )
    # `checked` is always 0. Adding it makes the copy depend on the report, so
    # the report runs before the kernel; do not drop it as a no-op.
    return _copy_blocks_unchecked(
        pool, src + checked.astype(src.dtype), dst, in_pool=in_pool
    )


def _copy_blocks_unchecked(pool, src_indices, dst_indices, in_pool=None):
    """The copy kernel of `copy_blocks`. It does no bounds checks: every index
    must be inside the pool. With `in_pool`, each grid step also runs
    `pl.debug_check` on its pair's entry."""
    na = src_indices.shape[0]
    block_shape = (1,) + pool.shape[1:]
    pad = (0,) * (pool.ndim - 1)

    if in_pool is None:

        def _copy_kernel(src_ref, dst_ref, pool_in_ref, pool_out_ref):
            pool_out_ref[...] = pool_in_ref[...]

        return pl.pallas_call(
            _copy_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                grid=(na,),
                in_specs=[pl.BlockSpec(block_shape, lambda i, s, d: (s[i],) + pad)],
                out_specs=pl.BlockSpec(block_shape, lambda i, s, d: (d[i],) + pad),
            ),
            out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
            input_output_aliases={2: 0},
        )(src_indices, dst_indices, pool)

    message = _replaced_pair_message(pool.shape[0])

    def _copy_kernel(src_ref, dst_ref, ok_ref, pool_in_ref, pool_out_ref):
        pl.debug_check(ok_ref[pl.program_id(0)] != 0, message)
        pool_out_ref[...] = pool_in_ref[...]

    return pl.pallas_call(
        _copy_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(na,),
            in_specs=[pl.BlockSpec(block_shape, lambda i, s, d, ok: (s[i],) + pad)],
            out_specs=pl.BlockSpec(block_shape, lambda i, s, d, ok: (d[i],) + pad),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={3: 0},
    )(src_indices, dst_indices, in_pool.astype(jnp.int32), pool)
