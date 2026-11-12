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
"""TensorCore MoE dispatch and combine for large EP batches.

Under expert parallelism only about 1/8 of the routes of a token are local,
so the SparseCore kernels (`ragged_gather_v2`, `ragged_gather_reduce_v3`) move
little data per launch yet leave the TensorCore waiting for them. These two
kernels do the same row movement on the TensorCore with one row DMA per local
route:

* `tc_row_gather` -- dispatch: `out[i] = x[rows[i]]` for `i < count`.
* `tc_combine` -- combine: `out[t] = sum_k w[t, k] * y[pos[t, k]]` over the
  valid routes of token `t`, accumulated in f32 in `k` order starting from
  +0.0 and rounded to bf16 once, which is the arithmetic of
  `ragged_gather_reduce_v3`.

BF16 rows are packed two to a 32-bit word in HBM, so both kernels move the
32-bit pair row that holds a route's row and pick the half on the VPU (the
same scheme the SparseCore kernels use).
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_HALF_MASK = -65536  # 0xFFFF0000 as int32


def _as_pairs(ref):
    """bf16 [R, H] HBM ref -> u32 [R // 2, H] pair-row view (u32 input as is)."""
    if ref.dtype == jnp.uint32:
        return ref
    return ref.bitcast(jnp.uint32)


def _row_dma(src_pairs, row, dst, dst_row, sem):
    return pltpu.make_async_copy(
        src_pairs.at[pl.ds(jnp.right_shift(row, 1), 1)],
        dst.at[pl.ds(dst_row, 1)],
        sem,
    )


def _half_to_f32(words, odd):
    """u32 pair-row words -> f32 value of the even (low) or odd (high) row."""
    bits = jnp.where(
        odd,
        jnp.bitwise_and(words, jnp.uint32(0xFFFF0000)),
        jnp.left_shift(words, jnp.uint32(16)),
    )
    return lax.bitcast_convert_type(bits, jnp.float32)


# --------------------------------------------------------------------------
# combine
# --------------------------------------------------------------------------


def _combine_kernel(
    # scalar prefetch
    cnt0_ref,  # [nb] int32: slot-0 DMAs per block
    nex_ref,  # [nb] int32: extra routes per block
    # inputs
    pos0_cur_ref,  # SMEM [tb] int32 (current block), -1 = no local route
    pos0_nxt_ref,  # SMEM [tb] int32 (next block)
    exp_cur_ref,  # SMEM [xcap] int32 extra-route rows (current block)
    exp_nxt_ref,  # SMEM [xcap] int32 (next block)
    ext_ref,  # SMEM [xcap] int32 extra-route token (block-local)
    exw_ref,  # SMEM [xcap] f32 extra-route weight
    p0v_ref,  # VMEM [tb, 1] int32 slot-0 row (-1 = none)
    w0v_ref,  # VMEM [tb, 1] f32 slot-0 weight
    y_hbm,  # [M, H] bf16 (or [M/2, H] u32 pairs) in HBM
    # output
    out_ref,  # VMEM [tb, H] bf16
    # scratch
    stage0,  # VMEM [2, tb, H] u32
    stagex,  # VMEM [2, xcap, H] u32
    acc_ref,  # VMEM [tb, H] f32
    sems,  # DMA (2,)
):
    b = pl.program_id(0)
    nb = pl.num_programs(0)
    tb = stage0.shape[1]
    y32 = _as_pairs(y_hbm)

    def issue(pos0_ref, exp_ref, blk, slot):
        def one0(i, c):
            p = pos0_ref[i]

            @pl.when(p >= 0)
            def _():
                _row_dma(y32, p, stage0.at[slot], i, sems.at[slot]).start()

            return c

        lax.fori_loop(0, tb, one0, 0)

        def onex(j, c):
            _row_dma(y32, exp_ref[j], stagex.at[slot], j, sems.at[slot]).start()
            return c

        lax.fori_loop(0, nex_ref[blk], onex, 0)

    @pl.when(b == 0)
    def _():
        issue(pos0_cur_ref, exp_cur_ref, b, 0)

    slot = b % 2

    def wait_one(i, c):
        pltpu.make_async_copy(
            y32.at[pl.ds(0, 1)], stage0.at[slot, pl.ds(0, 1)], sems.at[slot]
        ).wait()
        return c

    lax.fori_loop(0, cnt0_ref[b] + nex_ref[b], wait_one, 0)

    @pl.when(b + 1 < nb)
    def _():
        issue(pos0_nxt_ref, exp_nxt_ref, b + 1, 1 - slot)

    p0 = p0v_ref[...]
    val = _half_to_f32(stage0[slot], jnp.bitwise_and(p0, 1) == 1) * w0v_ref[...]
    # +0.0 start: a -0.0 product lands as +0.0, as `0 + x` does in v3.
    acc_ref[...] = jnp.where((p0 >= 0) & (val != 0), val, 0.0)

    def addx(j, c):
        p = exp_cur_ref[j]
        t = ext_ref[j]
        row = stagex[slot, pl.ds(j, 1), :]
        v = _half_to_f32(row, jnp.bitwise_and(p, 1) == 1) * exw_ref[j]
        acc_ref[pl.ds(t, 1), :] = acc_ref[pl.ds(t, 1), :] + v
        return c

    lax.fori_loop(0, nex_ref[b], addx, 0)
    out_ref[...] = acc_ref[...].astype(out_ref.dtype)


def combine_metadata(indices, weights, valid, *, topk: int, tb: int, xcap: int):
    """Route metadata for `tc_combine` (all TensorCore XLA ops, no gathers).

    Returns (pos0 [T], w0 [T] f32, cnt0 [nb], ex_pos [nb*xcap], ex_t [nb*xcap],
    ex_w [nb*xcap] f32, nex [nb], overflow scalar bool).
    """
    t = indices.size // topk
    nb = t // tb
    pos = indices.reshape(t, topk).astype(jnp.int32)
    w = weights.reshape(t, topk).astype(jnp.float32)
    v = valid.reshape(t, topk)
    # First valid route of each token (select chain, no gather).
    pos0 = jnp.full((t,), -1, jnp.int32)
    w0 = jnp.zeros((t,), jnp.float32)
    for k in reversed(range(topk)):
        pos0 = jnp.where(v[:, k], pos[:, k], pos0)
        w0 = jnp.where(v[:, k], w[:, k], w0)
    rank = jnp.cumsum(v.astype(jnp.int32), axis=1)
    extra = v & (rank >= 2)
    # Compact the extra routes of each block in (token, k) order: a stable sort
    # on "not extra" carries the payloads with it.
    key = jnp.logical_not(extra).reshape(nb, tb * topk).astype(jnp.int32)
    tloc = jnp.broadcast_to(
        jnp.arange(tb, dtype=jnp.int32)[:, None], (tb, topk)
    ).reshape(1, tb * topk)
    tloc = jnp.broadcast_to(tloc, (nb, tb * topk))
    _, ex_pos, ex_t, ex_w = lax.sort(
        (key, pos.reshape(nb, tb * topk), tloc, w.reshape(nb, tb * topk)),
        dimension=1,
        is_stable=True,
        num_keys=1,
    )
    nex = jnp.sum(extra.reshape(nb, tb * topk), axis=1, dtype=jnp.int32)
    cnt0 = jnp.sum((pos0 >= 0).reshape(nb, tb), axis=1, dtype=jnp.int32)
    overflow = jnp.any(nex > xcap)
    return (
        pos0,
        w0,
        cnt0,
        ex_pos[:, :xcap].reshape(-1),
        ex_t[:, :xcap].reshape(-1),
        ex_w[:, :xcap].reshape(-1),
        jnp.minimum(nex, xcap),
        overflow,
    )


def tc_combine_ok(num_tokens: int, hidden: int, rows: int, tb: int = 128) -> bool:
    return num_tokens % tb == 0 and hidden % 128 == 0 and rows % 2 == 0


@functools.partial(
    jax.jit, static_argnames=("topk", "tb", "xcap", "interpret", "vmem_mb")
)
def tc_combine(
    y,
    indices,
    weights,
    valid,
    *,
    topk: int,
    tb: int = 128,
    xcap: int = 64,
    interpret: bool = False,
    vmem_mb: int = 48,
):
    """out[t] = bf16(sum over valid k, in k order, of f32(w[t,k]) * y[idx[t,k]]).

    Returns (out [T, H] bf16, overflow bool). On overflow (a block with more
    than `xcap` extra routes) the result is incomplete and the caller must use
    another combine.
    """
    hidden = y.shape[-1]
    t = indices.size // topk
    nb = t // tb
    pos0, w0, cnt0, ex_pos, ex_t, ex_w, nex, overflow = combine_metadata(
        indices, weights, valid, topk=topk, tb=tb, xcap=xcap
    )
    smem = pltpu.MemorySpace.SMEM if hasattr(pltpu, "MemorySpace") else pltpu.SMEM
    last = nb - 1

    def cur(b, *_):
        return (b,)

    def nxt(b, *_):
        return (jnp.minimum(b + 1, last),)

    in_specs = [
        pl.BlockSpec((tb,), cur, memory_space=smem),
        pl.BlockSpec((tb,), nxt, memory_space=smem),
        pl.BlockSpec((xcap,), cur, memory_space=smem),
        pl.BlockSpec((xcap,), nxt, memory_space=smem),
        pl.BlockSpec((xcap,), cur, memory_space=smem),
        pl.BlockSpec((xcap,), cur, memory_space=smem),
        pl.BlockSpec((tb, 1), lambda b, *_: (b, 0)),
        pl.BlockSpec((tb, 1), lambda b, *_: (b, 0)),
        pl.BlockSpec(memory_space=pl.ANY),
    ]
    out = pl.pallas_call(
        _combine_kernel,
        out_shape=jax.ShapeDtypeStruct((t, hidden), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(nb,),
            in_specs=in_specs,
            out_specs=pl.BlockSpec((tb, hidden), lambda b, *_: (b, 0)),
            scratch_shapes=[
                pltpu.VMEM((2, tb, hidden), jnp.uint32),
                pltpu.VMEM((2, xcap, hidden), jnp.uint32),
                pltpu.VMEM((tb, hidden), jnp.float32),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_mb * 1024 * 1024,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="moe_tc_combine",
    )(
        cnt0,
        nex,
        pos0,
        pos0,
        ex_pos,
        ex_pos,
        ex_t,
        ex_w,
        pos0[:, None],
        w0[:, None],
        y,
    )
    return out, overflow


# --------------------------------------------------------------------------
# dispatch (row gather)
# --------------------------------------------------------------------------


def _gather_kernel(
    count_ref,  # SMEM [1] int32 (scalar prefetch)
    tok_hbm,  # [M] int32 HBM: source row of each output row
    tokv_hbm,  # [M // 2, 2] int32 HBM (same data, for parity vectors)
    x_hbm,  # [T, H] bf16 (or [T/2, H] u32) HBM
    out_hbm,  # [M, H] bf16 HBM (output, written for rows < count rounded up)
    tok_smem,  # SMEM [2, tbd] int32
    tokv,  # VMEM [2, tbd // 2, 2] int32
    stage,  # VMEM [2, tbd, H] u32: even rows first, then odd rows
    obuf,  # VMEM [2, tbd // 2, H] u32
    sems,  # DMA (6,): rows x2, tok x2, out x2
):
    tbd = stage.shape[1]
    half = tbd // 2
    x32 = _as_pairs(x_hbm)
    o32 = _as_pairs(out_hbm)
    count = count_ref[0]
    nblk = (count + tbd - 1) // tbd

    def tok_copy(blk, slot):
        return (
            pltpu.make_async_copy(
                tok_hbm.at[pl.ds(blk * tbd, tbd)], tok_smem.at[slot], sems.at[2 + slot]
            ),
            pltpu.make_async_copy(
                tokv_hbm.at[pl.ds(blk * half, half)], tokv.at[slot], sems.at[2 + slot]
            ),
        )

    def issue_rows(blk, slot):
        n = jnp.minimum(count - blk * tbd, tbd)

        def one(i, c):
            dst = jnp.right_shift(i, 1) + jnp.bitwise_and(i, 1) * half
            _row_dma(x32, tok_smem[slot, i], stage.at[slot], dst, sems.at[slot]).start()
            return c

        lax.fori_loop(0, n, one, 0)

    def wait_rows(blk, slot):
        n = jnp.minimum(count - blk * tbd, tbd)

        def one(i, c):
            pltpu.make_async_copy(
                x32.at[pl.ds(0, 1)], stage.at[slot, pl.ds(0, 1)], sems.at[slot]
            ).wait()
            return c

        lax.fori_loop(0, n, one, 0)

    def out_copy(blk, slot):
        return pltpu.make_async_copy(
            obuf.at[slot], o32.at[pl.ds(blk * half, half)], sems.at[4 + slot]
        )

    @pl.when(nblk > 0)
    def _():
        for c in tok_copy(0, 0):
            c.start()
        for c in tok_copy(0, 0):
            c.wait()
        issue_rows(0, 0)

        @pl.when(nblk > 1)
        def _():
            for c in tok_copy(1, 1):
                c.start()

    def body(blk, c):
        slot = blk % 2
        nslot = 1 - slot

        @pl.when(blk + 1 < nblk)
        def _():
            for cp in tok_copy(blk + 1, nslot):
                cp.wait()
            issue_rows(blk + 1, nslot)

        wait_rows(blk, slot)

        @pl.when(blk >= 2)
        def _():
            out_copy(blk - 2, slot).wait()

        tv = tokv[slot]
        pa = jnp.bitwise_and(tv[:, 0:1], 1) == 1
        pb = jnp.bitwise_and(tv[:, 1:2], 1) == 1
        a = stage[slot, pl.ds(0, half), :]
        bb = stage[slot, pl.ds(half, half), :]
        lo = jnp.where(
            pa, jnp.right_shift(a, jnp.uint32(16)), jnp.bitwise_and(a, jnp.uint32(0xFFFF))
        )
        hi = jnp.where(
            pb,
            jnp.bitwise_and(bb, jnp.uint32(0xFFFF0000)),
            jnp.left_shift(bb, jnp.uint32(16)),
        )
        obuf[slot] = jnp.bitwise_or(lo, hi)
        out_copy(blk, slot).start()

        # The token chunk two blocks ahead reuses this slot's SMEM buffer,
        # whose row DMAs were all issued above.
        @pl.when(blk + 2 < nblk)
        def _():
            for cp in tok_copy(blk + 2, slot):
                cp.start()

        return c

    lax.fori_loop(0, nblk, body, 0)

    @pl.when(nblk >= 2)
    def _():
        out_copy(nblk - 2, nblk % 2).wait()

    @pl.when(nblk >= 1)
    def _():
        out_copy(nblk - 1, (nblk - 1) % 2).wait()


@functools.partial(jax.jit, static_argnames=("tbd", "interpret", "vmem_mb"))
def tc_row_gather(
    x, rows, count, *, tbd: int = 256, interpret: bool = False, vmem_mb: int = 48
):
    """out[i] = x[rows[i]] for i < count; rows past count are unspecified.

    x: [T, H] bf16 (T even); rows: [M] int32 (M % tbd == 0); count: [1] int32.
    """
    m = rows.shape[0]
    hidden = x.shape[-1]
    assert m % tbd == 0 and tbd % 16 == 0
    rows = rows.astype(jnp.int32)
    out_dtype = x.dtype
    out_rows = m if out_dtype == jnp.bfloat16 else m // 2
    return pl.pallas_call(
        _gather_kernel,
        out_shape=jax.ShapeDtypeStruct((out_rows, hidden), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            scratch_shapes=[
                pltpu.SMEM((2, tbd), jnp.int32),
                pltpu.VMEM((2, tbd // 2, 2), jnp.int32),
                pltpu.VMEM((2, tbd, hidden), jnp.uint32),
                pltpu.VMEM((2, tbd // 2, hidden), jnp.uint32),
                pltpu.SemaphoreType.DMA((6,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_mb * 1024 * 1024,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="moe_tc_row_gather",
    )(count.reshape(1).astype(jnp.int32), rows, rows.reshape(m // 2, 2), x)
