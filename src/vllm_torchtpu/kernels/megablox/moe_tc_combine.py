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
"""TensorCore MoE combine for large EP batches.

Under expert parallelism only about 1/8 of a token's routes are local, so the
SparseCore gather-reduce (`ragged_gather_reduce_v3`) moves little data yet the
TensorCore waits for it (~280 us per MoE layer at 8192 tokens). `tc_slab_combine`
computes the same result on the TensorCore:
`out[t] = sum_k w[t, k] * y[pos[t, k]]` over the valid routes of token `t`,
products and sum in f32 in top-k order starting from +0.0, one bf16 rounding.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# --------------------------------------------------------------------------
# combine from aligned slabs (MXU selection)
# --------------------------------------------------------------------------
#
# A token block's routed rows lie in at most 16 contiguous runs of the
# expert-sorted GMM output (one per local expert; the counting sort keeps
# tokens ascending inside an expert). The kernel DMAs the aligned 16-row slabs
# that hold them (contiguous 16 x H bf16 = whole HBM tiles, unlike single-row
# copies) and selects each token's rows with a 0/1 bf16 matmul: one nonzero
# term per output element, so the f32 result is the row value exactly. The
# weights, the top-k-order f32 sum from +0.0 and the bf16 rounding then follow
# `ragged_gather_reduce_v3`.

_SLAB = 16
_GROUP = 8  # slabs per selection matmul (128 rows)


def _slab_kernel(
    # scalar prefetch
    slabs_ref,  # [nb * scap] int32 sorted unique slab ids of each block
    nsl_ref,  # [nb] int32 slabs per block
    nex_ref,  # [nb] int32 extra routes per block
    count_ref,  # [1] int32 valid GMM rows
    # inputs
    ext_ref,  # SMEM [xcap] int32 extra-route token (block-local)
    exw_ref,  # SMEM [xcap] f32 extra-route weight
    srow0_ref,  # VMEM [tb, 1] int32 staged row of the first local route (-1)
    w0_ref,  # VMEM [tb, 1] f32
    srowx_ref,  # VMEM [xcap, 1] int32 staged row of each extra route
    y_hbm,  # [M, H] bf16 HBM
    # output
    out_ref,  # VMEM [tb, H] bf16
    # scratch
    stage,  # VMEM [2, scap * 16, H] bf16
    acc_ref,  # VMEM [tb, H] f32
    gx_ref,  # VMEM [xcap, H] f32
    sems,  # DMA (2,)
):
    b = pl.program_id(0)
    nb = pl.num_programs(0)
    scap = stage.shape[1] // _SLAB
    xcap = gx_ref.shape[0]
    hidden = stage.shape[2]
    rows_g = _GROUP * _SLAB

    def slab_copy(blk, j, slot):
        sid = slabs_ref[blk * scap + j]
        return pltpu.make_async_copy(
            y_hbm.at[pl.ds(pl.multiple_of(sid * _SLAB, _SLAB), _SLAB)],
            stage.at[slot, pl.ds(pl.multiple_of(j * _SLAB, _SLAB), _SLAB)],
            sems.at[slot],
        )

    def issue(blk, slot):
        def one(j, c):
            slab_copy(blk, j, slot).start()
            return c

        lax.fori_loop(0, nsl_ref[blk], one, 0)

    @pl.when(b == 0)
    def _():
        issue(b, 0)

    slot = b % 2
    nsl = nsl_ref[b]

    def wait_one(j, c):
        slab_copy(b, 0, slot).wait()
        return c

    lax.fori_loop(0, nsl, wait_one, 0)

    @pl.when(b + 1 < nb)
    def _():
        issue(b + 1, 1 - slot)

    ng = (nsl + _GROUP - 1) // _GROUP
    zero_slab = jnp.zeros((_SLAB, hidden), stage.dtype)

    # Slabs past nsl inside the last group hold stale (or never written)
    # rows; the selection multiplies them by 0, so they must be finite.
    def clear(j, c):
        stage[slot, pl.ds(pl.multiple_of(j * _SLAB, _SLAB), _SLAB), :] = zero_slab
        return c

    lax.fori_loop(nsl, ng * _GROUP, clear, 0)

    # Rows at or past `count` were never written by the GMM: only the slab
    # holding row count - 1 can contain such rows.
    count = count_ref[0]

    @pl.when(nsl > 0)
    def _():
        j = nsl - 1
        sid = slabs_ref[b * scap + j]
        rows = sid * _SLAB + lax.broadcasted_iota(jnp.int32, (_SLAB, 1), 0)
        sl = pl.ds(pl.multiple_of(j * _SLAB, _SLAB), _SLAB)
        stage[slot, sl, :] = jnp.where(rows < count, stage[slot, sl, :], 0)

    srow0 = srow0_ref[...]
    srowx = srowx_ref[...]
    acc_ref[...] = jnp.zeros_like(acc_ref)
    gx_ref[...] = jnp.zeros_like(gx_ref)
    lanes = lax.broadcasted_iota(jnp.int32, (1, rows_g), 1)

    def group(g, c):
        base = g * rows_g
        s = stage[slot, pl.ds(pl.multiple_of(base, rows_g), rows_g), :]
        p0 = (srow0 == lanes + base).astype(s.dtype)
        acc_ref[...] += jnp.dot(p0, s, preferred_element_type=jnp.float32)
        px = (srowx == lanes + base).astype(s.dtype)
        gx_ref[...] += jnp.dot(px, s, preferred_element_type=jnp.float32)
        return c

    lax.fori_loop(0, ng, group, 0)

    val = acc_ref[...] * w0_ref[...]
    # +0.0 start: a -0.0 (or a selected +-0.0) lands as +0.0, as in v3.
    acc_ref[...] = jnp.where((srow0 >= 0) & (val != 0), val, 0.0)

    def addx(j, c):
        t = ext_ref[j]
        v = gx_ref[pl.ds(j, 1), :] * exw_ref[j]
        acc_ref[pl.ds(t, 1), :] = acc_ref[pl.ds(t, 1), :] + v
        return c

    lax.fori_loop(0, nex_ref[b], addx, 0)
    del xcap
    out_ref[...] = acc_ref[...].astype(out_ref.dtype)


def slab_metadata(indices, weights, valid, *, topk: int, tb: int, scap: int, xcap: int):
    """Route metadata for `tc_slab_combine` (TensorCore XLA ops, no gathers)."""
    t = indices.size // topk
    nb = t // tb
    big = jnp.int32(1 << 30)
    pos = indices.reshape(nb, tb * topk).astype(jnp.int32)
    v = valid.reshape(nb, tb * topk)
    sid = jnp.where(v, jnp.right_shift(pos, 4), big)
    s1 = jnp.sort(sid, axis=1)
    first = jnp.concatenate(
        [jnp.ones((nb, 1), bool), s1[:, 1:] != s1[:, :-1]], axis=1
    ) & (s1 != big)
    uniq = jnp.sort(jnp.where(first, s1, big), axis=1)[:, :scap]
    nsl = jnp.sum(first, axis=1, dtype=jnp.int32)
    # Staged row of every route: slab rank inside the block, then row in slab.
    rank = jnp.sum(uniq[:, None, :] < sid[:, :, None], axis=2, dtype=jnp.int32)
    srow = jnp.where(v, rank * _SLAB + jnp.bitwise_and(pos, _SLAB - 1), -1)
    srow = srow.reshape(t, topk)
    w = weights.reshape(t, topk).astype(jnp.float32)
    vv = valid.reshape(t, topk)
    srow0 = jnp.full((t,), -1, jnp.int32)
    w0 = jnp.zeros((t,), jnp.float32)
    for k in reversed(range(topk)):
        srow0 = jnp.where(vv[:, k], srow[:, k], srow0)
        w0 = jnp.where(vv[:, k], w[:, k], w0)
    extra = vv & (jnp.cumsum(vv.astype(jnp.int32), axis=1) >= 2)
    key = jnp.logical_not(extra).reshape(nb, tb * topk).astype(jnp.int32)
    tloc = jnp.broadcast_to(
        jnp.broadcast_to(jnp.arange(tb, dtype=jnp.int32)[:, None], (tb, topk)).reshape(
            1, tb * topk
        ),
        (nb, tb * topk),
    )
    _, ex_srow, ex_t, ex_w = lax.sort(
        (key, srow.reshape(nb, tb * topk), tloc, w.reshape(nb, tb * topk)),
        dimension=1,
        is_stable=True,
        num_keys=1,
    )
    nex = jnp.sum(extra.reshape(nb, tb * topk), axis=1, dtype=jnp.int32)
    overflow = jnp.any(nsl > scap) | jnp.any(nex > xcap)
    count = jnp.sum(valid, dtype=jnp.int32).reshape(1)
    # Unused extra slots select nothing (-1).
    ex_srow = jnp.where(
        jnp.arange(xcap)[None, :] < nex[:, None], ex_srow[:, :xcap], -1
    )
    return dict(
        slabs=uniq.reshape(-1),
        nsl=jnp.minimum(nsl, scap),
        nex=jnp.minimum(nex, xcap),
        count=count,
        ex_t=ex_t[:, :xcap].reshape(-1),
        ex_w=ex_w[:, :xcap].reshape(-1),
        ex_srow=ex_srow.reshape(-1),
        srow0=srow0,
        w0=w0,
        overflow=overflow,
    )


def tc_slab_combine_ok(num_tokens: int, hidden: int, tb: int = 256) -> bool:
    return num_tokens % tb == 0 and hidden % 128 == 0


@functools.partial(
    jax.jit, static_argnames=("topk", "tb", "scap", "xcap", "interpret", "vmem_mb")
)
def tc_slab_combine(
    y,
    indices,
    weights,
    valid,
    *,
    topk: int,
    tb: int = 256,
    scap: int = 64,
    xcap: int = 128,
    interpret: bool = False,
    vmem_mb: int = 60,
):
    """Same result as `tc_combine` / v3; returns (out [T, H] bf16, overflow)."""
    hidden = y.shape[-1]
    t = indices.size // topk
    nb = t // tb
    md = slab_metadata(indices, weights, valid, topk=topk, tb=tb, scap=scap, xcap=xcap)
    smem = pltpu.MemorySpace.SMEM if hasattr(pltpu, "MemorySpace") else pltpu.SMEM
    out = pl.pallas_call(
        _slab_kernel,
        out_shape=jax.ShapeDtypeStruct((t, hidden), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(nb,),
            in_specs=[
                pl.BlockSpec((xcap,), lambda b, *_: (b,), memory_space=smem),
                pl.BlockSpec((xcap,), lambda b, *_: (b,), memory_space=smem),
                pl.BlockSpec((tb, 1), lambda b, *_: (b, 0)),
                pl.BlockSpec((tb, 1), lambda b, *_: (b, 0)),
                pl.BlockSpec((xcap, 1), lambda b, *_: (b, 0)),
                pl.BlockSpec(memory_space=pl.ANY),
            ],
            out_specs=pl.BlockSpec((tb, hidden), lambda b, *_: (b, 0)),
            scratch_shapes=[
                pltpu.VMEM((2, scap * _SLAB, hidden), jnp.bfloat16),
                pltpu.VMEM((tb, hidden), jnp.float32),
                pltpu.VMEM((xcap, hidden), jnp.float32),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_mb * 1024 * 1024,
        ),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="moe_tc_slab_combine",
    )(
        md["slabs"],
        md["nsl"],
        md["nex"],
        md["count"],
        md["ex_t"],
        md["ex_w"],
        md["srow0"][:, None],
        md["w0"][:, None],
        md["ex_srow"][:, None],
        y,
    )
    return out, md["overflow"]
