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
"""Union-dense CSA attention for DeepSeek-V4 prefill.

The sparse path gathers every query token's own 1024 top-k compressed rows on
SparseCore (2048 x 1024 x 640 B per T2048 step) and attends one token at a time,
so the MXU sees only the local query heads per pushed KV block and the gather
is the critical path. Neighbouring tokens select heavily overlapping rows (in
served traffic a 64-token block selects ~4-6k distinct rows out of 8-25k).
This kernel attends, per query block of `bq` tokens of one sequence, the union
of the block's selected rows once: the union rows are gathered contiguously
(tile-padded, `csa_gather` layout, `tile` rows per period), and each token's
own selection is a bitmask over its block's union ranks. Same keys and values
per token as the sparse kernel; only the summation order differs.

Inputs per q block (`plan`): the union rows' tile range in the gathered buffer,
the union size, the token range. The mask uses `csa_masked_dense`'s layout over
union ranks with KV block = `tile`: bit `b` of word `128 s + l` is rank
`s * 4096 + (b % bits) * 128 + l + (b // bits) * tile`.
"""

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v4.core_attention.sparse_mla import (
    _dequant_dsv4_fp8,
    prepare_outputs,
    prepare_q_inputs,
    prepare_swa_inputs,
)

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
LANES = 128
WORD_BITS = 32
SUPER_BLOCK = WORD_BITS * LANES  # 4096 ranks per 128-word column
TILE = 1024  # union rows per KV tile (== csa_gather top_k period)
META = 6  # q_start, q_len, tile0, n_tiles, union_size, seq
# Debug: TPU_DSV4_CSA_UNION_DENSE_CHECKS=1 compiles the kernels with bounds checks.
_CHECKS = bool(int(os.environ.get("TPU_DSV4_CSA_UNION_DENSE_CHECKS", "0") or 0))


def cdiv(a, b):
    return (a + b - 1) // b


# ----------------------------------------------------------------------------
# Host reference planner (numpy). Used by the tests and the microbenchmark to
# emulate the device-side union builder.


def mask_word_bit(rank, tile=TILE):
    """(word, bit) of a union rank in the mask layout."""
    blocks_per_super = SUPER_BLOCK // tile
    bits = tile // LANES
    blk, row = rank // tile, rank % tile
    word = (blk // blocks_per_super) * LANES + row % LANES
    bit = (blk % blocks_per_super) * bits + row // LANES
    return word, bit


def plan_host(topk, cu_q_lens, num_seqs, page_table, page_size, *, bq, rows_cap,
              mask_words, prefill_seq_ids, tile=TILE):
    """Union plan for the given sequences' tokens.

    topk: i32[T, K] seq-local compressed positions, -1 padded.
    page_table: i32[max_num_seqs, pages_per_seq].
    Returns dict of numpy arrays: indices i32[rows_cap] (physical rows),
    meta i32[META, nb_cap], num_blocks, num_rows (multiple of tile),
    mask i32[T, mask_words].
    """
    T = topk.shape[0]
    nb_cap = T // bq + len(cu_q_lens)
    meta = np.zeros((META, nb_cap), np.int32)
    mask = np.zeros((T, mask_words), np.uint32)
    idx = []
    nb = 0
    tile0 = 0
    for s in prefill_seq_ids:
        a, b = int(cu_q_lens[s]), int(cu_q_lens[s + 1])
        for q0 in range(a, b, bq):
            q1 = min(b, q0 + bq)
            sel = topk[q0:q1]
            u = np.unique(sel[sel >= 0])
            rank = {int(p): r for r, p in enumerate(u)}
            for t in range(q0, q1):
                for p in topk[t][topk[t] >= 0]:
                    w, bit = mask_word_bit(rank[int(p)], tile)
                    mask[t, w] |= np.uint32(1) << np.uint32(bit)
            nt = max(1, cdiv(len(u), tile))
            phys = page_table[s, u // page_size] * page_size + u % page_size
            pad = nt * tile - len(u)
            if pad:
                # spread pads over the block's own rows (never one constant row)
                phys = np.concatenate([phys, phys[np.arange(pad) % max(1, len(phys))]])
            idx.append(phys.astype(np.int32))
            meta[:, nb] = (q0, q1 - q0, tile0, nt, len(u), s)
            tile0 += nt
            nb += 1
    flat = np.concatenate(idx) if idx else np.zeros((0,), np.int32)
    num_rows = flat.size
    assert num_rows <= rows_cap, (num_rows, rows_cap)
    out = np.zeros((rows_cap,), np.int32)
    out[:num_rows] = flat
    if num_rows < rows_cap and num_rows:  # never gathered; spread anyway
        out[num_rows:] = flat[(np.arange(rows_cap - num_rows) * 104729) % num_rows]
    return dict(indices=out, meta=meta, num_blocks=nb, num_rows=num_rows,
                mask=mask.view(np.int32))


# ----------------------------------------------------------------------------
# Kernel


def _kernel(
    # Scalar prefetch
    meta_ref,  # i32[META, nb_cap]
    nb_ref,  # i32[1]
    sem_ids_ref,  # i32[2]  (kv slot, bo slot)
    bo_ids_ref,  # i32[4]   (q_start/len of the bo DMA per slot)
    # Inputs
    sinks_ref,  # VMEM f32[bq * H, 128]
    q_hbm_ref,  # [T, H, D]
    nope_hbm_ref,  # i32[rows_cap, 128]
    rope_hbm_ref,  # i32[rows_cap // 4, 128], viewed as bf16[rows_cap // 2, 128]
    mask_hbm_ref,  # i32[T, mask_words // 128, 128]
    swa_acc_hbm_ref,  # [T * H, D]
    swa_l_hbm_ref,  # f32[T * H, 128]
    swa_m_hbm_ref,  # f32[T * H, 128]
    # Output (aliases q)
    o_hbm_ref,
    # Scratch
    nope_x2_ref,  # i32[2, tile, 128]
    rope_x2_ref,  # bf16[2, tile // 2, 128]
    mask_x2_ref,  # i32[2, bq, 1, 128]
    bq_ref,  # [bq, H, D]
    bo_x2_ref,  # [2, bq, H, D]
    swa_acc_ref,  # [bq * H, D]
    swa_l_ref,  # f32[bq * H, 128]
    swa_m_ref,  # f32[bq * H, 128]
    sems,  # DMA[6, 2]
    l_ref,  # f32[bq * H, 128]
    m_ref,  # f32[bq * H, 128]
    acc_ref,  # f32[bq * H, D]
    *,
    sm_scale: float,
    bq_sz: int,
    tile: int,
):
    _, num_q_heads, head_dim = q_hbm_ref.shape
    rows = bq_sz * num_q_heads
    q_dtype = q_hbm_ref.dtype
    q_packing = 32 // jax.dtypes.itemsize_bits(q_dtype)
    heads_per_packing = num_q_heads // q_packing
    blocks_per_super = SUPER_BLOCK // tile
    bits_per_block = tile // LANES
    half = tile // 2

    blk = pl.program_id(0)
    nb = nb_ref[0]
    rope_hbm_ref = rope_hbm_ref.bitcast(jnp.bfloat16)

    def copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def fetch_tile(q0, qn, gtile, jtile, slot, *, wait=False):
        nope_v = nope_x2_ref.at[slot]
        rope_v = rope_x2_ref.at[slot]
        mask_v = mask_x2_ref.at[slot, pl.ds(0, qn), 0, pl.ds(0, LANES)]
        if wait:
            copy(nope_v, nope_v, sems.at[0, slot], True)
            copy(rope_v, rope_v, sems.at[1, slot], True)
            copy(mask_v, mask_v, sems.at[2, slot], True)
            return
        copy(nope_hbm_ref.at[pl.ds(gtile * tile, tile)], nope_v, sems.at[0, slot], False)
        copy(rope_hbm_ref.at[pl.ds(gtile * half, half)], rope_v, sems.at[1, slot], False)
        copy(
            mask_hbm_ref.at[pl.ds(q0, qn), jtile // blocks_per_super, pl.ds(0, LANES)],
            mask_v,
            sems.at[2, slot],
            False,
        )

    def send_bo(q0, qn, slot, *, wait=False):
        copy(bo_x2_ref.at[slot, pl.ds(0, qn)], o_hbm_ref.at[pl.ds(q0, qn)],
             sems.at[3, slot], wait)  # fmt: skip

    def wait_send_bo(slot):
        q0 = bo_ids_ref[slot]
        qn = bo_ids_ref[slot + 2]

        @pl.when(q0 >= 0)
        def _():
            send_bo(q0, qn, slot, wait=True)

    def load_bq():
        q_ref = bq_ref.bitcast(jnp.uint32).reshape(bq_sz * heads_per_packing, head_dim)
        return pltpu.bitcast(q_ref[...], q_dtype).reshape(rows, head_dim)

    def load_tile(slot, valid_rows):
        words = nope_x2_ref[slot]  # i32[tile, 128]
        nope = _dequant_dsv4_fp8(words)
        r = rope_x2_ref[slot]  # bf16[tile // 2, 128]
        rope_dim = LANES // 2
        rope = jnp.concatenate([r[:, :rope_dim], r[:, rope_dim:]], axis=0)
        kv = jnp.concatenate([nope, rope], axis=1)  # [tile, 512]
        # Pad rows of the last tile duplicate real rows; they are absent from
        # every mask, but zero them anyway so a pad can never inject NaN.
        c = lax.broadcasted_iota(jnp.int32, kv.shape, 0)
        return jnp.where(c < valid_rows, kv, jnp.zeros_like(kv))

    def load_mask(slot, jtile):
        words = mask_x2_ref[slot][:, 0, :]  # [bq, 128]
        base = (jtile % blocks_per_super) * bits_per_block
        planes = [
            jnp.bitwise_and(lax.shift_right_logical(words, base + t), 1)
            for t in range(bits_per_block)
        ]
        return jnp.concatenate(planes, axis=1) != 0  # [bq, tile]

    def lanes(x, n):
        return jnp.concatenate([x] * (n // LANES), axis=1) if n != LANES else x

    def flash(q, kv, keep):
        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        s *= sm_scale
        s3 = s.reshape(bq_sz, num_q_heads, tile)
        s3 = jnp.where(keep[:, None, :], s3, jnp.finfo(jnp.float32).min)
        s = s3.reshape(rows, tile)
        m_prev = m_ref[...]
        m_curr = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
        m_ref[...] = m_curr
        p = jnp.exp(s - lanes(m_curr, tile))
        pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_ref[...] = exp_m_diff * l_ref[...] + jnp.sum(p, axis=1, keepdims=True)
        acc_ref[...] = lanes(exp_m_diff, head_dim) * acc_ref[...] + pv

    @pl.when(
        jnp.logical_and(
            blk < nb, jnp.logical_and(meta_ref[1, blk] > 0, meta_ref[3, blk] > 0)
        )
    )
    def _block():
        q0 = meta_ref[0, blk]
        qn = meta_ref[1, blk]
        g0 = meta_ref[2, blk]
        ntiles = meta_ref[3, blk]
        usize = meta_ref[4, blk]
        hs, hn = q0 * num_q_heads, qn * num_q_heads

        # q / SWA state for this block (synchronous; ~3 MiB per block).
        copy(q_hbm_ref.at[pl.ds(q0, qn)], bq_ref.at[pl.ds(0, qn)], sems.at[4, 0], False)
        copy(swa_acc_hbm_ref.at[pl.ds(hs, hn)], swa_acc_ref.at[pl.ds(0, hn)], sems.at[4, 1], False)
        copy(swa_l_hbm_ref.at[pl.ds(hs, hn)], swa_l_ref.at[pl.ds(0, hn)], sems.at[5, 0], False)
        copy(swa_m_hbm_ref.at[pl.ds(hs, hn)], swa_m_ref.at[pl.ds(0, hn)], sems.at[5, 1], False)
        slot0 = sem_ids_ref[0]
        fetch_tile(q0, qn, g0, 0, slot0)
        copy(bq_ref.at[pl.ds(0, qn)], bq_ref.at[pl.ds(0, qn)], sems.at[4, 0], True)
        copy(swa_acc_ref.at[pl.ds(0, hn)], swa_acc_ref.at[pl.ds(0, hn)], sems.at[4, 1], True)
        copy(swa_l_ref.at[pl.ds(0, hn)], swa_l_ref.at[pl.ds(0, hn)], sems.at[5, 0], True)
        copy(swa_m_ref.at[pl.ds(0, hn)], swa_m_ref.at[pl.ds(0, hn)], sems.at[5, 1], True)
        m_ref[...] = swa_m_ref[...]
        l_ref[...] = swa_l_ref[...]
        acc_ref[...] = swa_acc_ref[...].astype(jnp.float32)

        def tile_body(j, _):
            slot = sem_ids_ref[0]
            nslot = lax.select(slot == 0, 1, 0)

            @pl.when(j + 1 < ntiles)
            def _():
                fetch_tile(q0, qn, g0 + j + 1, j + 1, nslot)

            fetch_tile(q0, qn, g0 + j, j, slot, wait=True)
            flash(load_bq(), load_tile(slot, usize - j * tile), load_mask(slot, j))
            sem_ids_ref[0] = nslot

        lax.fori_loop(0, ntiles, tile_body, None, unroll=False)

        l_total = l_ref[...] + jnp.exp(sinks_ref[...] - m_ref[...])
        inv = lanes(pl.reciprocal(l_total, approx=True), head_dim)
        out = (acc_ref[...] * inv).astype(q_dtype)
        bo = sem_ids_ref[1]
        sem_ids_ref[1] = lax.select(bo == 0, 1, 0)
        wait_send_bo(bo)
        bo_x2_ref.at[bo].bitcast(jnp.int32).reshape(bq_sz * heads_per_packing, head_dim)[
            ...
        ] = pltpu.bitcast(out, jnp.int32)
        bo_ids_ref[bo] = q0
        bo_ids_ref[bo + 2] = qn
        send_bo(q0, qn, bo)

    @pl.when(blk == pl.num_programs(0) - 1)
    def _epilogue():
        for i in range(2):
            wait_send_bo(i)


@functools.partial(
    jax.jit, static_argnames=("sm_scale", "bq", "tile", "vmem_limit_bytes")
)
def union_dense_attention(
    q: jax.Array,  # [T, actual_heads, D]
    nope_g: jax.Array,  # i32[rows_cap, 128] csa_gather nope_out
    rope_g: jax.Array,  # i32[rows_cap // 4, 128] csa_gather rope_out
    mask: jax.Array,  # i32[T, mask_words]
    meta: jax.Array,  # i32[META, nb_cap]
    num_blocks: jax.Array,  # i32 scalar
    attention_sinks: jax.Array,
    swa_accumution: jax.Array,
    swa_l: jax.Array,
    swa_m: jax.Array,
    *,
    sm_scale: float,
    bq: int,
    tile: int = TILE,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> jax.Array:
    """Rows of the planned blocks get CSA attention; all other rows return q."""
    _, actual_heads, actual_dim = q.shape
    q = prepare_q_inputs(q)
    T, H, D = q.shape
    rows = bq * H
    swa_acc, swa_l, swa_m = prepare_swa_inputs(swa_accumution, swa_l, swa_m)
    swa_acc = swa_acc.reshape(T * H, D)

    def per_row(x):
        x = x[:, :H].reshape(T * H, 1)
        return jnp.broadcast_to(x, (T * H, LANES))

    sinks = jnp.pad(attention_sinks, (0, H - actual_heads),
                    constant_values=jnp.finfo(attention_sinks.dtype).min)  # fmt: skip
    sinks = jnp.broadcast_to(jnp.tile(sinks, bq)[:, None], (rows, LANES)).astype(jnp.float32)
    nb_cap = meta.shape[1]
    scratch = [
        pltpu.VMEM((2, tile, LANES), jnp.int32),
        pltpu.VMEM((2, tile // 2, LANES), jnp.bfloat16),
        pltpu.VMEM((2, bq, 1, LANES), jnp.int32),
        pltpu.VMEM((bq, H, D), q.dtype),
        pltpu.VMEM((2, bq, H, D), q.dtype),
        pltpu.VMEM((rows, D), swa_acc.dtype),
        pltpu.VMEM((rows, LANES), jnp.float32),
        pltpu.VMEM((rows, LANES), jnp.float32),
        pltpu.SemaphoreType.DMA((6, 2)),
        pltpu.VMEM((rows, LANES), jnp.float32),
        pltpu.VMEM((rows, LANES), jnp.float32),
        pltpu.VMEM((rows, D), jnp.float32),
    ]
    prefetch = (
        meta,
        jnp.reshape(num_blocks, (1,)).astype(jnp.int32),
        jnp.zeros((2,), jnp.int32),
        jnp.full((4,), -1, jnp.int32),
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    scope = f"CsaUnionDense-bq_{bq}-t_{tile}"
    out = pl.pallas_call(
        functools.partial(_kernel, sm_scale=sm_scale, bq_sz=bq, tile=tile),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(prefetch),
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] + [hbm] * 7,
            out_specs=hbm,
            grid=(nb_cap,),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=not _CHECKS,
        ),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        input_output_aliases={len(prefetch) + 1: 0},
        name=scope,
    )(
        *prefetch,
        sinks,
        q,
        nope_g,
        rope_g,
        mask.reshape(T, -1, LANES),
        swa_acc,
        per_row(swa_l),
        per_row(swa_m),
    )
    return prepare_outputs(out, actual_heads, actual_dim)


# ----------------------------------------------------------------------------
# Device union builder (SparseCore): one q block per vector subcore.

ROWS_PER_DMA = 4  # topk rows per DMA (2 slots); leaves SC VMEM for s_cap 64k


def _union_sc_kernel(
    meta_hbm,  # i32[nb_cap * L]: q0, qn, seq, S per block (lanes 0..3)
    topk_hbm,  # i32[T * K]
    pages_hbm,  # i32[max_num_seqs * pps]
    idx_out,  # i32[nb_cap * u_cap] physical rows, block regions
    mask_out,  # i32[T * W]
    cnt_out,  # i32[nb_cap * L] union size per block (broadcast)
    metab,  # VMEM i32[L]
    ubuf,  # VMEM i32[L]
    rowb,  # VMEM i32[2, R * K]
    pageb,  # VMEM i32[pps_pad]
    cnt,  # VMEM i32[s_cap + L]
    ulist,  # VMEM i32[u_cap + L]
    urow,  # VMEM i32[u_cap + L]
    mbuf,  # VMEM i32[2 * R * W] mask rows staging
    sems,  # DMA[2]
    osems,  # DMA[2]
    *,
    nb_cap: int,
    K: int,
    s_cap: int,
    u_cap: int,
    pps: int,
    pps_dma: int,
    page_size: int,
    tile: int,
    L: int,
    num_cores: int,
    num_subcores: int,
    count_only: bool = False,
    split: int = 1,
):
    W = u_cap // WORD_BITS
    R = ROWS_PER_DMA
    page_shift = page_size.bit_length() - 1
    assert 1 << page_shift == page_size
    workers = num_cores * num_subcores
    wid = lax.axis_index("core") * num_subcores + lax.axis_index("subcore")
    iota = lax.broadcasted_iota(jnp.int32, (L,), 0)
    zeros = jnp.zeros((L,), jnp.int32)
    ones = jnp.ones((L,), jnp.int32)

    def rows_pass(q0, qn, body, before=None, after=None):
        """body(slot, r, t) over the block's rows, R rows per DMA, double-buffered.

        Slots are static (the chunk loop is unrolled by two): a DMA into
        `rowb.at[slot]` with a traced slot landed in the wrong buffer.
        """
        nchunks = (qn + R - 1) // R

        def desc(c, slot):
            n = jnp.minimum(R, qn - c * R)
            return pltpu.make_async_copy(
                topk_hbm.at[pl.ds((q0 + c * R) * K, n * K)],
                rowb.at[slot, pl.ds(0, n * K)],
                sems.at[slot],
            )

        def chunk(c, slot):
            @pl.when(c + 1 < nchunks)
            def _():
                desc(c + 1, 1 - slot).start()

            if before is not None:
                before(c, slot)
            desc(c, slot).wait()
            n = jnp.minimum(R, qn - c * R)

            @pl.loop(0, n)
            def _(r):
                body(slot, r, c * R + r)

            if after is not None:
                after(c, slot, n)

        desc(0, 0).start()

        @pl.loop(0, (nchunks + 1) // 2)
        def _(cp):
            chunk(2 * cp, 0)

            @pl.when(2 * cp + 1 < nchunks)
            def _():
                chunk(2 * cp + 1, 1)

    # `split` subcores share a block: each counts the whole block (cheap) and
    # then writes its slice of the union rows and of the token masks.
    per_wave = workers // split
    part = wid % split

    @pl.loop(0, (nb_cap + per_wave - 1) // per_wave)
    def _wave(w):
        b = w * per_wave + wid // split

        @pl.when(b < nb_cap)
        def _():
            pltpu.sync_copy(meta_hbm.at[pl.ds(b * L, L)], metab)
            m = metab[...]
            q0, qn, seq, S = m[0], m[1], m[2], m[3]

            @pl.when(qn > 0)
            def _blk():
                # 32-byte-aligned superset of the sequence's page-table row.
                p_lo = (seq * pps) // 8 * 8
                p_off = seq * pps - p_lo
                pltpu.sync_copy(
                    pages_hbm.at[pl.ds(p_lo, pps_dma)], pageb.at[pl.ds(0, pps_dma)]
                )
                n16 = (S + L - 1) // L

                # Sequential on purpose: under a parallel_loop the zero stores
                # are not ordered against the count scatters below, and a
                # surviving stale count made the count pass and the plan pass
                # disagree on U (tiles past the written rows -> garbage rows).
                @pl.loop(0, n16)
                def _(i):
                    cnt[pl.ds(i * L, L)] = zeros

                def count_row(slot, r, t):
                    @plsc.parallel_loop(0, K, step=L)
                    def _(k):
                        c = rowb[slot, pl.ds(r * K + k, L)]
                        v = (c >= 0) & (c < S)
                        plsc.addupdate_scatter(cnt, (jnp.where(v, c, 0),), ones, mask=v)

                rows_pass(q0, qn, count_row)

                def emit(i, woff):
                    g = cnt[pl.ds(i * L, L)]
                    sel = (g > 0) & (iota + i * L < S)
                    ok = woff <= u_cap  # past u_cap: count only (fallback)
                    plsc.store_compressed(
                        ulist.at[pl.ds(jnp.minimum(woff, u_cap), L)],
                        iota + i * L,
                        mask=sel & ok,
                    )
                    return woff + plsc.all_reduce_population_count(sel)[0]

                if count_only:
                    # Union size only: per-lane counts, no ordered compaction.
                    def tally(i, acc):
                        g = cnt[pl.ds(i * L, L)]
                        sel = (g > 0) & (iota + i * L < S)
                        return acc + jnp.where(sel, ones, zeros)

                    acc = plsc.parallel_loop(0, n16, carry=zeros)(tally)
                    U = plsc.cumsum(acc)[L - 1]
                else:
                    U = plsc.parallel_loop(0, n16, carry=jnp.int32(0))(emit)
                @pl.when(part == 0)
                def _():
                    ubuf[...] = jnp.full((L,), U, jnp.int32)
                    pltpu.sync_copy(ubuf, cnt_out.at[pl.ds(b * L, L)])

                if count_only:
                    return

                @pl.when(jnp.logical_and(U > 0, U <= u_cap))
                def _fits():
                    nu = (U + L - 1) // L

                    @plsc.parallel_loop(0, nu)
                    def _(j):
                        r = iota + j * L
                        v = r < U
                        pos = jnp.where(v, ulist[pl.ds(j * L, L)], 0)
                        c = plsc.load_gather(cnt, [pos], mask=v)
                        plsc.addupdate_scatter(cnt, (pos,), r - c, mask=v)

                    u_pad = ((U + tile - 1) // tile) * tile
                    u_part = u_pad // split  # tile is a multiple of split * L
                    r0 = part * u_part

                    @plsc.parallel_loop(0, u_part // L)
                    def _(j):
                        r = iota + j * L + r0
                        src = jnp.where(r < U, r, lax.rem(r - U, U))
                        pos = plsc.load_gather(ulist, [src])
                        pg = plsc.load_gather(
                            pageb, [lax.shift_right_logical(pos, page_shift) + p_off]
                        )
                        urow[pl.ds(j * L, L)] = jnp.bitwise_or(
                            jnp.left_shift(pg, page_shift),
                            jnp.bitwise_and(pos, page_size - 1),
                        )

                    pltpu.sync_copy(
                        urow.at[pl.ds(0, u_part)], idx_out.at[pl.ds(b * u_cap + r0, u_part)]
                    )
                    nsb = (U + SUPER_BLOCK - 1) // SUPER_BLOCK

                    nw = nsb * LANES  # mask words this block uses (rest stay 0)

                    @plsc.parallel_loop(0, 2 * R * W // L)
                    def _(i):
                        mbuf[pl.ds(i * L, L)] = zeros

                    hq = (qn + split - 1) // split
                    mq0 = q0 + part * hq
                    mqn = jnp.clip(qn - part * hq, 0, hq)

                    def out_desc(c, slot, n):
                        return pltpu.make_async_copy(
                            mbuf.at[pl.ds(slot * R * W, n * W)],
                            mask_out.at[pl.ds((mq0 + c * R) * W, n * W)],
                            osems.at[slot],
                        )

                    def before(c, slot):
                        @pl.when(c >= 2)
                        def _():
                            out_desc(c - 2, slot, R).wait()

                    def after(c, slot, n):
                        out_desc(c, slot, n).start()

                    def mask_row(slot, r, t):
                        row0 = slot * R * W + r * W
                        # Scatter-add each selected rank's bit straight into
                        # its word: ranks are distinct, so adding is OR-ing;
                        # needs in-vector duplicate words to accumulate.
                        @plsc.parallel_loop(0, nw // L)
                        def _(i):
                            mbuf[pl.ds(row0 + i * L, L)] = zeros

                        @plsc.parallel_loop(0, K, step=L)
                        def _(k):
                            c = rowb[slot, pl.ds(r * K + k, L)]
                            v = (c >= 0) & (c < S)
                            rk = plsc.load_gather(cnt, [jnp.where(v, c, 0)], mask=v)
                            rk = jnp.where(v, rk, 0)
                            word = jnp.bitwise_or(
                                jnp.left_shift(lax.shift_right_logical(rk, 12), 7),
                                jnp.bitwise_and(rk, LANES - 1),
                            )
                            bit = jnp.bitwise_and(lax.shift_right_logical(rk, 7), WORD_BITS - 1)
                            plsc.addupdate_scatter(
                                mbuf, (word + row0,), jnp.left_shift(ones, bit), mask=v
                            )

                    @pl.when(mqn > 0)
                    def _():
                        rows_pass(mq0, mqn, mask_row, before=before, after=after)
                        nchunks = (mqn + R - 1) // R
                        last = nchunks - 1

                        @pl.when(last % 2 == 0)
                        def _():
                            out_desc(last, 0, mqn - last * R).wait()

                            @pl.when(last >= 1)
                            def _():
                                out_desc(last - 1, 1, R).wait()

                        @pl.when(last % 2 == 1)
                        def _():
                            out_desc(last, 1, mqn - last * R).wait()
                            out_desc(last - 1, 0, R).wait()


def build_union_plan(
    topk: jax.Array,  # i32[T, K]
    page_indices: jax.Array,  # i32[max_num_seqs * pps]
    meta_rows: jax.Array,  # i32[nb_cap, 4]: q0, qn, seq, S
    *,
    max_num_seqs: int,
    s_cap: int,
    u_cap: int,
    page_size: int,
    tile: int = TILE,
    count_only: bool = False,
    split: int = 1,
):
    """SparseCore union builder (`count_only`: union sizes only; `split`
    subcores per q block).

    Returns idx i32[nb_cap * u_cap] (physical rows; block b's tile-padded union
    in [b * u_cap, b * u_cap + ceil(U_b / tile) * tile)), mask i32[T, u_cap // 32]
    (rows of planned blocks only) and usize i32[nb_cap] (U_b; > u_cap means the
    block did not fit and nothing else was written for it).
    """
    T, K = topk.shape
    nb_cap = meta_rows.shape[0]
    sc = pltpu.get_tpu_info().sparse_core
    L = sc.num_lanes
    assert u_cap % SUPER_BLOCK == 0 and s_cap % L == 0 and K % L == 0
    pps = page_indices.shape[0] // max_num_seqs
    pps_dma = cdiv(pps + 8, 8) * 8
    pps_pad = cdiv(pps_dma, L) * L
    # The aligned row superset may run past the table's end: pad the table.
    page_indices = jnp.pad(page_indices, (0, pps_dma))
    W = u_cap // WORD_BITS
    meta = jnp.zeros((nb_cap, L), jnp.int32).at[:, :4].set(meta_rows).reshape(-1)
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    idx, mask, usize = pl.kernel(
        functools.partial(
            _union_sc_kernel,
            nb_cap=nb_cap,
            K=K,
            s_cap=s_cap,
            u_cap=u_cap,
            pps=pps,
            pps_dma=pps_dma,
            page_size=page_size,
            tile=tile,
            L=L,
            num_cores=sc.num_cores,
            num_subcores=sc.num_subcores,
            count_only=count_only,
            split=split,
        ),
        out_type=(
            jax.ShapeDtypeStruct((nb_cap * u_cap,), jnp.int32),
            jax.ShapeDtypeStruct((T * W,), jnp.int32),
            jax.ShapeDtypeStruct((nb_cap * L,), jnp.int32),
        ),
        scratch_types=(
            pltpu.VMEM((L,), jnp.int32),
            pltpu.VMEM((L,), jnp.int32),
            pltpu.VMEM((2, ROWS_PER_DMA * K), jnp.int32),
            pltpu.VMEM((pps_pad,), jnp.int32),
            pltpu.VMEM((s_cap + L,), jnp.int32),
            pltpu.VMEM((u_cap + L,), jnp.int32),
            pltpu.VMEM((u_cap + L,), jnp.int32),
            pltpu.VMEM((2 * ROWS_PER_DMA * W,), jnp.int32),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=not _CHECKS, needs_layout_passes=False
        ),
        mesh=mesh,
        name="sc_csa_union_count" if count_only else "sc_csa_union_plan",
    )(meta, topk.reshape(-1), page_indices)
    return idx, mask.reshape(T, W), usize.reshape(nb_cap, L)[:, 0]


def compact_tiles(idx, tile_end, tile0, total_tiles, u_cap, rows_cap, first_tile=0):
    """Block-region union rows -> one tile-contiguous list (a row gather of
    whole 1024-row tiles) of tiles [first_tile, total_tiles). Positions past
    the range repeat earlier tiles."""
    n_tiles = rows_cap // TILE
    g = jnp.arange(n_tiles, dtype=jnp.int32)
    span = jnp.maximum(total_tiles - first_tile, 1)
    g = first_tile + jnp.where(g < span, g, g % span)
    blk = jnp.sum(tile_end[None, :] <= g[:, None], axis=1).astype(jnp.int32)
    blk = jnp.minimum(blk, tile_end.shape[0] - 1)
    src_tile = blk * (u_cap // TILE) + g - tile0[blk]
    return idx.reshape(-1, TILE)[src_tile].reshape(-1)


def prepare(
    topk_indices: jax.Array,  # i32[T, K]
    kv_lens: jax.Array,  # i32[max_num_seqs] compressed KV length
    page_indices: jax.Array,  # i32[max_num_seqs * pps]
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    page_size: int,
    bq: int = 64,
    s_cap: int = 65536,
    u_cap: int = 16384,
    min_q_len: int = 17,
    rows_per_token: int = 160,
    count_first: bool = True,
    split: int = 1,
    max_kv: int | None = None,
    unconditional: bool = False,
) -> dict:
    """The union plan of a step (SparseCore), independent of q and the SWA
    pass: a caller can issue it ahead so it runs behind TensorCore work.

    Routes the trailing sequences of at least `min_q_len` tokens whose
    compressed KV fits `s_cap`; the leading rest stay on the sparse path."""
    T = topk_indices.shape[0]
    nseq = kv_lens.shape[0]
    seq_ids = jnp.arange(nseq, dtype=jnp.int32)
    qlen = cu_q_lens[1:] - cu_q_lens[:-1]
    live = seq_ids < distribution[2]
    # rc7's indexer can select a token's own still-open compressed block, i.e.
    # index == kv_lens (= seq_len // ratio): the selectable range is kv + 1.
    sel_len = kv_lens + 1
    ud_ok = (qlen >= min_q_len) & (sel_len <= s_cap) & (kv_lens > 0)
    if max_kv is not None:
        # Longer sequences stay on the sparse path (where union-dense does not pay).
        ud_ok = ud_ok & (sel_len <= max_kv)
    split_seq = jnp.max(jnp.where(live & ~ud_ok, seq_ids + 1, 0)).astype(jnp.int32)
    routed = live & (seq_ids >= split_seq)
    nblk = jnp.where(routed, (qlen + bq - 1) // bq, 0)
    blk_end = jnp.cumsum(nblk)
    total_blocks = blk_end[-1]
    nb_cap = T // bq + nseq
    b = jnp.arange(nb_cap, dtype=jnp.int32)
    seq_b = jnp.clip(jnp.searchsorted(blk_end, b, side="right"), 0, nseq - 1).astype(jnp.int32)
    j = b - (blk_end[seq_b] - nblk[seq_b])
    q0 = cu_q_lens[seq_b] + j * bq
    qn = jnp.where(b < total_blocks, jnp.clip(cu_q_lens[seq_b + 1] - q0, 0, bq), 0)
    q0 = jnp.where(qn > 0, q0, 0)
    s_b = jnp.where(qn > 0, sel_len[seq_b], 0)
    meta_rows = jnp.stack([q0, qn, seq_b, s_b], axis=1)
    kw = dict(max_num_seqs=nseq, s_cap=s_cap, u_cap=u_cap, page_size=page_size,
              split=split)  # fmt: skip
    n_sparse = cu_q_lens[split_seq]
    has_routed = n_sparse < cu_q_lens[distribution[2]]
    rows_cap = cdiv(T * rows_per_token, 32768) * 32768
    W = u_cap // WORD_BITS

    def none(_):
        return (
            jnp.zeros((nb_cap * u_cap,), jnp.int32),
            jnp.zeros((T, W), jnp.int32),
            jnp.zeros((nb_cap,), jnp.int32),
        )

    def full(_):
        idx, mask, u_full = build_union_plan(topk_indices, page_indices, meta_rows, **kw)
        return idx, mask, jnp.where(qn > 0, u_full, 0)

    def fits(usize):
        ntiles = (usize + TILE - 1) // TILE
        return jnp.all(usize <= u_cap) & (jnp.sum(ntiles) * TILE <= rows_cap)

    if count_first:
        # Union sizes first (a cheap pass), the full plan only for a step that
        # will take the union path; nothing when no sequence is routed.
        usize = lax.cond(
            has_routed,
            lambda _: build_union_plan(topk_indices, page_indices, meta_rows,
                                       count_only=True, **kw)[2],
            lambda _: jnp.zeros((nb_cap,), jnp.int32),
            None,
        )  # fmt: skip
        usize = jnp.where(qn > 0, usize, 0)
        use_union = has_routed & fits(usize)
        idx, mask, u_full = lax.cond(use_union, full, none, None)
        # The tile layout follows the pass that wrote the rows; both passes
        # must agree, and the step falls back if they do not.
        use_union = use_union & jnp.all(u_full == usize)
    else:
        # One pass (where fallbacks are rare and the plan's fixed cost matters).
        # `unconditional`: inside a lax.cond the SparseCore call starts and
        # completes within the branch, so it cannot run behind the TensorCore
        # work before the attention (SWA, compressor) and its whole latency is
        # exposed. Outside it, XLA issues it right after the indexer top-k --
        # a win only when the SparseCore top-k is short (indexer token split:
        # c16 twin -52 us/CSA layer), a loss when the top-k it then waits on
        # is long (+40 us without the split). With no routed sequence every
        # block has qn == 0 and the kernel skips it; use_union is then False.
        if unconditional:
            idx, mask, usize = full(None)
        else:
            idx, mask, usize = lax.cond(has_routed, full, none, None)
        use_union = has_routed & fits(usize)
    return dict(
        idx=idx, mask=mask, usize=usize, q0=q0, qn=qn, seq_b=seq_b,
        total_blocks=total_blocks, n_sparse=n_sparse, use_union=use_union,
    )  # fmt: skip


def attention(
    q: jax.Array,  # [T, actual_heads, D]
    cache_kv_nope: jax.Array,  # i32[total_pages, page_size, 128]
    cache_kv_rope: jax.Array,  # i32[total_pages, page_size // 4, 128]
    topk_indices: jax.Array,  # i32[T, K]
    kv_lens: jax.Array,  # i32[max_num_seqs] compressed KV length
    page_indices: jax.Array,  # i32[max_num_seqs * pps]
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sinks: jax.Array,
    swa_accumution: jax.Array,
    swa_l: jax.Array,
    swa_m: jax.Array,
    *,
    sm_scale: float,
    bq: int = 64,
    s_cap: int = 65536,
    u_cap: int = 16384,
    rows_per_token: int = 160,
    min_q_len: int = 17,
    sparse_chunk: int = 64,
    sparse_batch: int = 16,
    gather_no_pad: bool | None = None,
    plan: dict | None = None,
    groups: int = 4,
    count_first: bool = True,
    split: int = 1,
) -> jax.Array:
    """CSA attention: union-dense for the planned blocks (`prepare`; built
    here when `plan` is None), the per-token sparse gather for the leading
    rest. Falls back to the sparse path for the whole step when a block's union
    exceeds `u_cap` or the step's union rows exceed `rows_per_token * T`."""
    from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_gather, sparse_mla

    T = q.shape[0]
    if plan is None:
        plan = prepare(
            topk_indices, kv_lens, page_indices, cu_q_lens, distribution,
            page_size=cache_kv_nope.shape[1], bq=bq, s_cap=s_cap, u_cap=u_cap,
            min_q_len=min_q_len, rows_per_token=rows_per_token,
            count_first=count_first, split=split,
        )  # fmt: skip
    idx, mask, usize = plan["idx"], plan["mask"], plan["usize"]
    q0, qn, seq_b = plan["q0"], plan["qn"], plan["seq_b"]
    n_sparse = plan["n_sparse"]
    total_valid = cu_q_lens[distribution[2]]
    rows_cap = cdiv(T * rows_per_token, 32768) * 32768
    ntiles = (usize + TILE - 1) // TILE
    tile_end = jnp.cumsum(ntiles)
    tile0 = tile_end - ntiles
    total_tiles = tile_end[-1]
    use_union = plan["use_union"]
    del total_valid

    def sparse_call(q, rows, num_valid):
        return sparse_mla.sparse_ragged_paged_attention(
            q=q[:rows], cache_kv_nope=cache_kv_nope, cache_kv_rope=cache_kv_rope,
            topk_indices=topk_indices[:rows], page_indices=page_indices,
            cu_q_lens=cu_q_lens, distribution=distribution,
            attention_sinks=attention_sinks, swa_accumution=swa_accumution[:rows],
            swa_l=swa_l[:rows], swa_m=swa_m[:rows], sm_scale=sm_scale,
            gather_and_attention_chunk_size=32 if rows <= 128 else sparse_chunk,
            attention_kernel_batch_size=8 if rows <= 128 else sparse_batch,
            num_valid_tokens=num_valid,
            **({} if gather_no_pad is None else {"gather_no_pad": gather_no_pad}),
        )  # fmt: skip

    def all_sparse(q):
        return sparse_call(q, T, None)

    total_rows = cache_kv_nope.shape[0] * cache_kv_nope.shape[1]

    def union_path(q):
        if T <= 64:
            q = sparse_call(q, T, n_sparse)
        else:
            q = lax.cond(
                n_sparse <= 64,
                lambda q: q.at[:64].set(sparse_call(q, 64, n_sparse)),
                lambda q: sparse_call(q, T, n_sparse),
                q,
            )
        # Blocks split into `groups` runs of about equal tiles: the gather of
        # run g + 1 (SparseCore) overlaps the attention of run g.
        blk_ids = jnp.arange(usize.shape[0], dtype=jnp.int32)
        grp = jnp.where(qn > 0, (tile0 * groups) // jnp.maximum(total_tiles, 1), groups)
        grp_cap = cdiv(rows_cap // groups + u_cap, 32768) * 32768
        for gi in range(groups):
            in_g = grp == gi
            g_first = jnp.min(jnp.where(in_g, tile0, total_tiles))
            g_end = jnp.max(jnp.where(in_g, tile_end, 0))
            g_end = jnp.maximum(g_end, g_first)
            compact = compact_tiles(idx, tile_end, tile0, g_end, u_cap, grp_cap, g_first)
            # Never hand the gather a row outside the cache (defensive).
            compact = jnp.clip(compact, 0, total_rows - 1)
            nope_g, rope_g = csa_gather.csa_gather(
                cache_kv_nope, cache_kv_rope, compact,
                num_valid_indices=(g_end - g_first) * TILE, top_k=TILE,
            )  # fmt: skip
            meta = jnp.stack(
                [q0, jnp.where(in_g, qn, 0), tile0 - g_first, ntiles, usize, seq_b]
            ).astype(jnp.int32)
            q = union_dense_attention(
                q, nope_g, rope_g, mask, meta, plan["total_blocks"], attention_sinks,
                swa_accumution, swa_l, swa_m, sm_scale=sm_scale, bq=bq,
            )  # fmt: skip
        del blk_ids
        return q

    return lax.cond(use_union, union_path, all_sparse, q)
