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
into a block's byte-region (the GDN f32 ssm state, the conv slot) go
through these adapters: per-request gather/scatter of a token-range,
reinterpreted in VMEM via Mosaic `ref.bitcast` when the element type
differs — the only place a cross-dtype view is free on TPU (an XLA-level
bitcast of a pool slice forces a pool-sized layout copy per step).

Regions are token-ranges `[tok0, tok0 + ntok)` within a block with
`tok0 % ntok == 0`, so the block spec covers exactly the region and one
grid step moves only the region's bytes. Scatters alias the pool in place
(`input_output_aliases`); the unwritten complement is preserved via the
HBM alias, with no read-modify-write.
"""
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import typed_ldst
from vllm_torchtpu.kernels.gdn.v3 import config as gdn_v3_config


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
    return payload[:-1] + (last // elem_out, )


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


def gather_region(pool,
                  state_indices,
                  *,
                  tok0: int,
                  ntok: int,
                  out_dtype,
                  out_lanes: int | None = None,
                  split: int = 1):
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
            return _gather_window(pool,
                                  state_indices,
                                  split=split,
                                  kb0=tok0 // kernel_bs,
                                  nblocks=ntok // kernel_bs,
                                  out_dtype=out_dtype,
                                  out_lanes=out_lanes)
        kb, local0 = divmod(tok0, kernel_bs)
        if local0 + ntok > kernel_bs:
            raise NotImplementedError(
                "region partially straddles kernel blocks: "
                f"tok0={tok0} ntok={ntok} kernel_block={kernel_bs}")
        return gather_region(pool,
                             state_indices * split + kb,
                             tok0=local0,
                             ntok=ntok,
                             out_dtype=out_dtype,
                             out_lanes=out_lanes)

    na = state_indices.shape[0]
    block_size, payload, lanes = _pool_geometry(pool)
    assert tok0 % ntok == 0 and tok0 + ntok <= block_size
    same_dtype = jnp.dtype(pool.dtype) == jnp.dtype(out_dtype)
    out_payload = (payload if same_dtype else _bitcast_payload(
        payload, pool.dtype, out_dtype))
    out_rows = _out_rows(out_payload, ntok)
    if not payload and not same_dtype:
        # 3-D pool: the block ref is 2-D and the bitcast rescales the
        # token/row dim itself (the old validated sublane convention).
        out_rows = _rescale_rows(ntok, pool.dtype, out_dtype)
    lane_split = 1
    if out_lanes is not None and out_lanes != lanes:
        assert lanes % out_lanes == 0, (lanes, out_lanes)
        lane_split = lanes // out_lanes
        out_rows *= lane_split
    o_lanes = lanes // lane_split
    pad = (0, ) * (len(payload) + 1)

    def _kernel(sidx_ref, pool_ref, o_ref):
        o_ref[...] = typed_ldst.load_typed(pool_ref.at[0],
                                           view_dtype=out_dtype,
                                           lane_split=lane_split)[None]

    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na, ),
            in_specs=[
                pl.BlockSpec((1, ntok) + payload + (lanes, ), lambda i, s:
                             (s[i], tok0 // ntok) + pad)
            ],
            out_specs=pl.BlockSpec((1, out_rows, o_lanes), lambda i, s:
                                   (i, 0, 0)),
        ),
        out_shape=jax.ShapeDtypeStruct((na, out_rows, o_lanes), out_dtype),
    )(state_indices, pool)


def _gather_window(pool,
                   mgr_indices,
                   *,
                   split: int,
                   kb0: int,
                   nblocks: int,
                   out_dtype,
                   out_lanes: int | None = None):
    """Typed read of `nblocks` whole kernel blocks inside each request's
    manager block, one grid step and ONE DMA window per request.

    The pool is born at kernel granularity; a manager block is `split`
    consecutive kernel blocks (upstream map_to_kernel_blocks), so the
    window (split, block_size, ...) indexed by the manager id is always
    aligned (element offset = mgr * split). Reading the whole manager
    window costs a little extra DMA but restores the single-window-per-
    request shape whose per-window setup dominates the region cost.
    """
    assert 0 <= kb0 and kb0 + nblocks <= split, (kb0, nblocks, split)
    na = mgr_indices.shape[0]
    block_size, payload, lanes = _pool_geometry(pool)
    same_dtype = jnp.dtype(pool.dtype) == jnp.dtype(out_dtype)
    out_payload = (payload if same_dtype else _bitcast_payload(
        payload, pool.dtype, out_dtype))
    rows_pb = _out_rows(out_payload, block_size)
    if not payload and not same_dtype:
        rows_pb = _rescale_rows(block_size, pool.dtype, out_dtype)
    lane_split = 1
    if out_lanes is not None and out_lanes != lanes:
        assert lanes % out_lanes == 0, (lanes, out_lanes)
        lane_split = lanes // out_lanes
        rows_pb *= lane_split
    o_lanes = lanes // lane_split
    pad = (0, ) * (len(payload) + 1)

    def _kernel(sidx_ref, pool_ref, o_ref):
        for j in range(nblocks):
            o_ref[0, j * rows_pb:(j + 1) *
                  rows_pb, :] = (typed_ldst.load_typed(pool_ref.at[kb0 + j],
                                                       view_dtype=out_dtype,
                                                       lane_split=lane_split))

    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na, ),
            in_specs=[
                pl.BlockSpec((split, block_size) + payload + (lanes, ),
                             lambda i, s: (s[i], 0) + pad)
            ],
            out_specs=pl.BlockSpec((1, nblocks * rows_pb, o_lanes),
                                   lambda i, s: (i, 0, 0)),
        ),
        out_shape=jax.ShapeDtypeStruct((na, nblocks * rows_pb, o_lanes),
                                       out_dtype),
    )(mgr_indices, pool)


def _scatter_window(pool, vals, mgr_indices, *, split: int, kb0: int,
                    nblocks: int):
    """Typed write of `nblocks` whole kernel blocks inside each request's
    manager block (in place), one grid step per request; vals is the
    gather shape. The aliased output window covers the whole
    manager block, so the non-region kernel blocks are copied through
    from the aliased input to satisfy the full-write rule.
    """
    assert 0 <= kb0 and kb0 + nblocks <= split, (kb0, nblocks, split)
    na = mgr_indices.shape[0]
    block_size, payload, lanes = _pool_geometry(pool)
    same_dtype = jnp.dtype(pool.dtype) == jnp.dtype(vals.dtype)
    val_payload = (payload if same_dtype else _bitcast_payload(
        payload, pool.dtype, vals.dtype))
    rows_pb = _out_rows(val_payload, block_size)
    if not payload and not same_dtype:
        rows_pb = _rescale_rows(block_size, pool.dtype, vals.dtype)
    v_lanes = vals.shape[-1]
    lane_split = 1
    if v_lanes != lanes:
        assert lanes % v_lanes == 0, (lanes, v_lanes)
        lane_split = lanes // v_lanes
    v_rows_pb = rows_pb * lane_split
    assert vals.shape == (na, nblocks * v_rows_pb,
                          v_lanes), (vals.shape, nblocks, v_rows_pb, v_lanes)
    pad = (0, ) * (len(payload) + 1)

    def _kernel(sidx_ref, val_ref, pool_in_ref, pool_out_ref):
        for j in range(split):
            if kb0 <= j < kb0 + nblocks:
                typed_ldst.store_typed(
                    pool_out_ref.at[j],
                    val_ref[0, (j - kb0) * v_rows_pb:(j - kb0 + 1) *
                            v_rows_pb, :],
                    lane_split=lane_split)
            else:
                # full-write rule: pass the untouched kernel blocks through
                pool_out_ref.at[j][...] = pool_in_ref.at[j][...]

    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na, ),
            in_specs=[
                pl.BlockSpec((1, nblocks * v_rows_pb, v_lanes), lambda i, s:
                             (i, 0, 0)),
                pl.BlockSpec((split, block_size) + payload + (lanes, ),
                             lambda i, s: (s[i], 0) + pad),
            ],
            out_specs=pl.BlockSpec((split, block_size) + payload + (lanes, ),
                                   lambda i, s: (s[i], 0) + pad),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={2: 0},
    )(mgr_indices, vals, pool)


def scatter_region(pool,
                   vals,
                   state_indices,
                   *,
                   tok0: int,
                   ntok: int,
                   split: int = 1):
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
            return _scatter_window(pool,
                                   vals,
                                   state_indices,
                                   split=split,
                                   kb0=tok0 // kernel_bs,
                                   nblocks=ntok // kernel_bs)
        kb, local0 = divmod(tok0, kernel_bs)
        if local0 + ntok > kernel_bs:
            raise NotImplementedError(
                "region partially straddles kernel blocks: "
                f"tok0={tok0} ntok={ntok} kernel_block={kernel_bs}")
        return scatter_region(pool,
                              vals,
                              state_indices * split + kb,
                              tok0=local0,
                              ntok=ntok)

    na = state_indices.shape[0]
    block_size, payload, lanes = _pool_geometry(pool)
    assert tok0 % ntok == 0 and tok0 + ntok <= block_size
    same_dtype = jnp.dtype(pool.dtype) == jnp.dtype(vals.dtype)
    val_payload = (payload if same_dtype else _bitcast_payload(
        payload, pool.dtype, vals.dtype))
    out_rows = _out_rows(val_payload, ntok)
    if not payload and not same_dtype:
        out_rows = _rescale_rows(ntok, pool.dtype, vals.dtype)
    v_lanes = vals.shape[-1]
    lane_split = 1
    if v_lanes != lanes:
        # Narrow-lane vals (the gather out_lanes shape): merge lanes back
        # in-kernel; 128-aligned lane concat preserves row-major order.
        assert lanes % v_lanes == 0, (lanes, v_lanes)
        lane_split = lanes // v_lanes
    v_rows = out_rows * lane_split
    assert vals.shape == (na, v_rows, v_lanes), (vals.shape, v_rows, v_lanes)
    pad = (0, ) * (len(payload) + 1)

    def _kernel(sidx_ref, val_ref, pool_in_ref, pool_out_ref):
        # The output block covers exactly the region and is fully written;
        # the complement is preserved via the HBM alias.
        typed_ldst.store_typed(pool_out_ref.at[0],
                               val_ref[0],
                               lane_split=lane_split)

    return pl.pallas_call(
        _kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(na, ),
            in_specs=[
                pl.BlockSpec((1, v_rows, v_lanes), lambda i, s: (i, 0, 0)),
                # Aliased with the output; never read, so leave it in HBM.
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((1, ntok) + payload + (lanes, ),
                                   lambda i, s: (s[i], tok0 // ntok) + pad),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={2: 0},
    )(state_indices, vals, pool)


def copy_blocks(pool, src_indices, dst_indices):
    """Whole-block copies within the pool: pool[dst[i]] = pool[src[i]].

    Seeds a request's new mamba state block from its previous one whenever
    the block-table-derived state block advances (chunked-prefill boundary,
    decode crossing, or prefix-cache resume). Pairs with src == dst are
    no-ops used as padding.
    """
    na = src_indices.shape[0]
    block_shape = (1, ) + pool.shape[1:]
    pad = (0, ) * (pool.ndim - 1)

    def _copy_kernel(src_ref, dst_ref, pool_in_ref, pool_out_ref):
        pool_out_ref[...] = pool_in_ref[...]

    return pl.pallas_call(
        _copy_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(na, ),
            in_specs=[
                pl.BlockSpec(block_shape, lambda i, s, d: (s[i], ) + pad)
            ],
            out_specs=pl.BlockSpec(block_shape, lambda i, s, d:
                                   (d[i], ) + pad),
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={2: 0},
    )(src_indices, dst_indices, pool)


def _conv_qk_pair_rows_perm(taps: int, conv_dim: int, n_v: int, d_v: int,
                            lanes: int) -> tuple[int, ...] | None:
    """Typed-row permutation for the QK pair-blocked pooled conv layout.

    Stored order per tap interleaves Q and K row-pairs —
    ``[Q0, K0, Q1, K1, ..., V...]`` — so that one whole pool token holds
    one head-pair's Q and K rows and a full-width state can receive a
    TP-rank shard's first conv token as a single contiguous token copy.
    Returns the load-order permutation (``logical[i] = stored[perm[i]]``),
    or ``None`` when the stored order equals the logical order (head-shard
    states whose Q segment is a single typed row).
    """
    v_elems = n_v * d_v
    qk_elems = conv_dim - v_elems
    q_elems = qk_elems // 2
    if 2 * q_elems != qk_elems or q_elems % lanes:
        raise NotImplementedError(
            "QK pair-blocked conv layout requires lane-aligned equal Q/K "
            f"segments: conv_dim={conv_dim}, n_v*d_v={v_elems}, "
            f"lanes={lanes}")
    if v_elems % lanes:
        raise NotImplementedError(
            f"QK pair-blocked conv layout requires lane-aligned V: "
            f"{v_elems} % {lanes}")
    q_rows = q_elems // lanes
    v_rows = v_elems // lanes
    rows_per_tap = conv_dim // lanes
    if q_rows <= 1:
        return None
    perm: list[int] = []
    for tap in range(taps):
        base = tap * rows_per_tap
        perm.extend(base + 2 * i for i in range(q_rows))
        perm.extend(base + 2 * i + 1 for i in range(q_rows))
        perm.extend(base + 2 * q_rows + j for j in range(v_rows))
    return tuple(perm)


def v3_state_source(
        pool,
        *,
        split: int,
        ssm_ntok: int,
        conv_tok0: int,
        conv_ntok: int,
        conv_dim: int,
        n_v: int,
        d_k: int,
        d_v: int,
        kernel_size: int,
        qk_pair_layout: bool = False) -> gdn_v3_config.StateSourcePlan:
    """Static copy-plan letting the fused GDN V3 kernel stream the mamba
    state regions directly between this pool and its double-buffered
    pipeline — the exact bytes ``gather_region``/``scatter_region`` move
    for these regions, without the external round trip.

    The ssm region is the whole kernel blocks at the start of the manager
    window (one contiguous DMA per slot), viewed f32 with the ``d_v``
    lane split; the conv region is ``conv_ntok`` token rows inside a
    single kernel block, viewed bf16 with the tail rows zero-padded.
    Arguments mirror the region geometry computed by the pooled GDN
    caller; ``pool`` contributes only its shape and dtype.

    The plan describes one checkpoint. With speculative decoding the
    caller names one source block per checkpoint
    (``MetadataRef.s_idx_to_ckpt_indices``), matching how vLLM's reference
    GDN kernel indexes ``ssm_state_indices[seq, ckpt]``; every checkpoint
    reuses this same intra-block geometry, so the block size stays
    independent of ``num_speculative_tokens``.
    """
    block_size, payload, lanes = _pool_geometry(pool)
    tok_bytes = math.prod(payload) * lanes * jnp.dtype(pool.dtype).itemsize

    assert lanes % d_v == 0, (lanes, d_v)
    ssm_rows = n_v * d_k
    # The ssm region may extend past the f32 state bytes when the state does
    # not divide the pool token row; the kernel truncates loads to
    # rows_used and zero-fills the padding rows on store.
    assert ssm_ntok * tok_bytes >= ssm_rows * d_v * 4, (ssm_ntok, tok_bytes,
                                                        n_v, d_k, d_v)
    # Each pool token must hold a whole number of typed f32 rows.
    assert tok_bytes % (d_v * 4) == 0, (tok_bytes, d_v)
    if ssm_ntok % block_size == 0:
        ssm_nblocks, ssm_nrows = ssm_ntok // block_size, block_size
    else:
        assert ssm_ntok < block_size, (ssm_ntok, block_size)
        ssm_nblocks, ssm_nrows = 1, ssm_ntok
    ssm = gdn_v3_config.StateRegion(
        kb0=0,
        nblocks=ssm_nblocks,
        row0=0,
        nrows=ssm_nrows,
        view_dtype=jnp.dtype(jnp.float32),
        lane_split=lanes // d_v,
        rows_used=ssm_rows,
    )

    kb0, row0 = divmod(conv_tok0, block_size)
    if row0 + conv_ntok > block_size:
        raise NotImplementedError(
            "conv region partially straddles kernel blocks: "
            f"tok0={conv_tok0} ntok={conv_ntok} kernel_block={block_size}")
    assert conv_tok0 + conv_ntok <= split * block_size, (conv_tok0, conv_ntok,
                                                         split, block_size)
    assert (kernel_size - 1) * conv_dim % lanes == 0, (kernel_size, conv_dim,
                                                       lanes)
    # The kernel regroups conv rows into (kernel_size - 1, conv_dim) via
    # 128-aligned lane concat, which needs whole rows per conv row.
    assert conv_dim % lanes == 0, (conv_dim, lanes)
    conv_rows = (kernel_size - 1) * conv_dim // lanes
    assert conv_rows * 2 * lanes <= conv_ntok * tok_bytes, (conv_rows,
                                                            conv_ntok,
                                                            tok_bytes)
    conv_rows_perm = None
    if qk_pair_layout:
        conv_rows_perm = _conv_qk_pair_rows_perm(kernel_size - 1, conv_dim,
                                                 n_v, d_v, lanes)
    conv = gdn_v3_config.StateRegion(
        kb0=kb0,
        nblocks=1,
        row0=row0,
        nrows=conv_ntok,
        view_dtype=jnp.dtype(jnp.bfloat16),
        lane_split=1,
        rows_used=conv_rows,
        rows_perm=conv_rows_perm,
    )

    return gdn_v3_config.StateSourcePlan(stride=split,
                                         conv=conv,
                                         recurrent=ssm)
