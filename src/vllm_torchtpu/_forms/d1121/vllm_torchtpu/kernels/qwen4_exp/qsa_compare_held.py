# SPDX-License-Identifier: Apache-2.0
"""The compare core's held read (VLLM_TPU_QSA_PREFILL_COMPARE_CORE_HELD): a sequence's cached KV of one KV head held
in VMEM, each union slot gathered from it by vector loads.

``qsa_compare_core`` copies each union slot's 4-token group from the HBM pool with its own DMA (about 5.7 ns a
group of descriptor issue, cmp-t-pqsa2 job 75). Here the tiles of one sequence and one KV head run as a pass:

- Hold: the sequence's pages, in order, copied from the served pool (both KV heads' bytes: the paired record
  interleaves them in each 32-bit word) in chunks, and repacked into one head's group-major buffer
  [groups, 4, 128] u32: a group's K[0:128], K[128:256], V[0:128], V[128:256] rows, a word four tokens of one
  lane (byte b token 4 g + b). NaN bytes of V are zeroed there (a masked probability is 0, and 0 x NaN is NaN).
- Tiles: each of the pass's tiles loads its 128 slots' groups from the held buffer by their sequence-local group
  id (one [4, 128] load a slot) and stores the four rows into four plane buffers (0.8 ns a group in job 75); the
  planes are the tile's K and V in (slot, token) order, so the mask takes one code a token.
- One KV head a pass: the 12 query heads of that head, their output columns written per union group per pass.

Passes run sequence by sequence, head 0 then head 1; each pass's tiles are padded to whole ring periods with null
tiles. A step whose longest sequence passes the held capacity takes the caller's fallback.
"""

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import qsa_compare_core as cc

T, TILE, RATIO, GPT, MASKED = cc.T, cc.TILE, cc.RATIO, cc.GPT, cc.MASKED
NULL_BIT, FIRST_BIT, HEAD_BIT, LAST_BIT = 27, 28, 29, 30
ROW_MASK = (1 << NULL_BIT) - 1


def _hold_chunk(page_size: int) -> int:
    """Tokens a hold copy: the largest divisor of the page that is a multiple of 128 and at most 1024."""
    for c in range(1024, 0, -128):
        if page_size % c == 0:
            return c
    raise ValueError(f"pages of {page_size} tokens are not whole 128-token chunks")


def _nan_free(words):
    """u32 words of e4m3 bytes with every NaN byte (low seven bits all set) made 0."""
    low = words & jnp.uint32(0x7F7F7F7F)
    hit = (low + jnp.uint32(0x01010101)) & jnp.uint32(0x80808080)
    return words & ~(hit | (hit - lax.shift_right_logical(hit, jnp.uint32(7))))


def _held_kernel(
    rec_ref,      # SMEM int32[cap]: (q * ct + t) | null | first of pass | head | last
    nxt_ref,      # SMEM int32[cap]: at a union group's first tile, the next group's first tile (or the count)
    hold_ref,     # SMEM int32[ngroups]: tokens the union group's sequence holds
    meta_ref,     # SMEM int32[1]: tiles
    q_hbm,        # [rows, heads * hd] bf16
    ids_hbm,      # int32[ngroups * ct * GPT]: each slot's sequence-local group id
    codes_hbm,    # int32[ngroups * ct, TILE]: each token's code
    prow_hbm,     # int32[ngroups, pages]: each union group's page row
    cache_hbm,    # [pool tokens, 1, 4, hd] e4m3
    out_hbm,      # [rows, heads * hd] bf16
    *scratch,
    ring: int,
    stages: int,
    chunk: int,
    page_size: int,
    ct: int,
    hpk: int,
    head_dim: int,
    q_scale: float,
    s_scale: float,
    v_scale: float,
):
    R, NS, HD = ring, stages, head_dim
    held = scratch[0]                              # VMEM u32 [groups cap, 4, 128]
    stage_bufs = scratch[1:1 + NS]                 # VMEM [chunk, 1, 4, hd] e4m3 each
    code_bufs = scratch[1 + NS:1 + NS + R]         # VMEM int32[1, TILE] each
    (stage_sem, prow_s, prow_sem, ids_s, ring_sem, planes, qbuf, q_sem, obuf, o_sem, m_ref, l_ref, acc_ref,
     ctr_ref) = scratch[1 + NS + R:]
    ct_shift = cc._ilog2(ct)
    total = meta_ref[0]
    QW = hpk * HD                                  # a KV head's query columns

    def rec_of(i):
        return rec_ref[jnp.clip(i, 0, total - 1)]

    def ring_copies(k, slot):
        row = rec_of(k) & ROW_MASK
        ids = pltpu.make_async_copy(ids_hbm.at[pl.ds(pl.multiple_of(row * GPT, GPT), GPT)],
                                    ids_s.at[pl.ds(pl.multiple_of(slot * GPT, GPT), GPT)], ring_sem.at[slot])
        codes = pltpu.make_async_copy(codes_hbm.at[pl.ds(row, 1)], code_bufs[slot], ring_sem.at[slot])
        return ids, codes

    def ring_start(k, slot):
        for c in ring_copies(k, slot):
            c.start()

    def ring_wait(slot):
        # The descriptors' buffers only give the byte counts; the semaphore is the slot's.
        pltpu.make_async_copy(ids_hbm.at[pl.ds(0, GPT)], ids_s.at[pl.ds(0, GPT)], ring_sem.at[slot]).wait()
        pltpu.make_async_copy(codes_hbm.at[pl.ds(0, 1)], code_bufs[0], ring_sem.at[slot]).wait()

    def q_copy(q, h, qs):
        return pltpu.make_async_copy(
            q_hbm.at[pl.ds(pl.multiple_of(q * T, T), T), pl.ds(pl.multiple_of(h * QW, QW), QW)],
            qbuf.at[qs], q_sem.at[qs])

    def o_copy(q, h, os_):
        return pltpu.make_async_copy(
            obuf.at[os_], out_hbm.at[pl.ds(pl.multiple_of(q * T, T), T), pl.ds(pl.multiple_of(h * QW, QW), QW)],
            o_sem.at[os_])

    def hold(q, h):
        """The sequence of union group q: its pages' head-h K and V into the held buffer."""
        pltpu.sync_copy(prow_hbm.at[q], prow_s)
        nch = (hold_ref[q] + chunk - 1) // chunk
        per_page = page_size // chunk
        shift = (h * 16).astype(jnp.uint32)                          # head h is bytes 2 h, 2 h + 1

        def copy(c, slot):
            page = c // per_page
            src = cache_hbm.at[pl.ds(prow_s[page] * page_size + (c - page * per_page) * chunk, chunk)]
            return pltpu.make_async_copy(src, stage_bufs[slot], stage_sem.at[slot])

        for s in range(NS - 1):
            @pl.when(s < nch)
            def _(s=s):
                copy(s, s).start()

        def unpack(c, slot):
            words = stage_bufs[slot].bitcast(jnp.uint32).reshape(chunk, HD)
            for b in range(chunk // 128):
                w = words[b * 128:(b + 1) * 128, :] >> shift
                half = w.astype(jnp.uint16)
                k8 = pltpu.bitcast(half.astype(jnp.uint8), jnp.float8_e4m3fn)
                v16 = pltpu.bitcast(lax.shift_right_logical(pltpu.bitcast(half, jnp.uint32), jnp.uint32(8)),
                                    jnp.uint16)
                v8 = pltpu.bitcast(v16.astype(jnp.uint8), jnp.float8_e4m3fn)
                k32 = pltpu.bitcast(k8, jnp.uint32)                    # [32, hd]: a row a group
                v32 = _nan_free(pltpu.bitcast(v8, jnp.uint32))
                g0 = c * (chunk // RATIO) + b * 32
                for r, plane in enumerate((k32[:, :128], k32[:, 128:], v32[:, :128], v32[:, 128:])):
                    held[pl.ds(g0, 32), r, :] = plane

        def block(bi, carry):
            for u in range(NS):
                c = bi * NS + u

                @pl.when(c < nch)
                def _(c=c, u=u):
                    copy(0, u).wait()

                    @pl.when(c + NS - 1 < nch)
                    def _():
                        copy(c + NS - 1, (u + NS - 1) % NS).start()

                    unpack(c, u)

            return carry

        lax.fori_loop(0, (nch + NS - 1) // NS, block, None)

    def gather(slot):
        for j in range(GPT):
            x = held[ids_s[slot * GPT + j]]                            # [4, 128]
            for r in range(4):
                planes[r, pl.ds(j, 1), :] = x[r:r + 1, :]

    def bias_of(slot):
        cw = code_bufs[slot][...]                                    # [1, TILE]
        code = cw & 0xFFFF
        edge = lax.shift_right_logical(cw, 16) - 1
        r = lax.broadcasted_iota(jnp.int32, (T, TILE), 0)
        keep = (((code >> r) & 1) != 0) & (r >= edge)
        return jnp.concatenate([keep] * hpk, axis=0)

    def tile_step(slot, qs, p):
        k = pltpu.bitcast(jnp.concatenate([planes[0], planes[1]], axis=1), jnp.float8_e4m3fn)
        v = pltpu.bitcast(jnp.concatenate([planes[2], planes[3]], axis=1), jnp.float8_e4m3fn)
        q = jnp.concatenate([qbuf[qs, :, jj * HD:(jj + 1) * HD] for jj in range(hpk)], axis=0)
        if q_scale != 1.0:
            q = (q.astype(jnp.float32) * q_scale).astype(jnp.bfloat16)
        s = lax.dot_general(q, k.astype(jnp.bfloat16), (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
        if s_scale != 1.0:
            s = s * s_scale
        s = jnp.where(bias_of(slot), s, MASKED)   # a select: an unseen key may be NaN bytes
        m0 = m_ref[p]
        m1 = jnp.maximum(m0, jnp.max(s, axis=1, keepdims=True))
        alpha = jnp.exp(m0 - m1)
        prob = jnp.exp(s - m1)
        l_ref[p] = l_ref[p] * alpha + jnp.sum(prob, axis=1, keepdims=True)
        acc_ref[p] = acc_ref[p] * alpha + lax.dot_general(
            prob.astype(jnp.bfloat16), v.astype(jnp.bfloat16), (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32)
        m_ref[p] = m1

    def finish(q, h, p):
        @pl.when(ctr_ref[1 + p] == 1)
        def _():
            o_copy(0, 0, p).wait()

        l = l_ref[p]
        out = (acc_ref[p] * (v_scale / jnp.where(l > 0, l, 1.0))).astype(jnp.bfloat16)
        for jj in range(hpk):
            obuf[p, :, jj * HD:(jj + 1) * HD] = out[jj * T:(jj + 1) * T]
        o_copy(q, h, p).start()
        ctr_ref[1 + p] = 1

    def group_start(i, q, h, p):
        q_copy(0, 0, p).wait()
        nxt = nxt_ref[i]

        @pl.when(nxt < total)
        def _():
            rec = rec_ref[nxt]
            q_copy((rec & ROW_MASK) >> ct_shift, lax.shift_right_logical(rec, HEAD_BIT) & 1, 1 - p).start()

        m_ref[p] = jnp.full(m_ref.shape[1:], MASKED, jnp.float32)
        l_ref[p] = jnp.zeros(l_ref.shape[1:], jnp.float32)
        acc_ref[p] = jnp.zeros(acc_ref.shape[1:], jnp.float32)

    def step(i, slot):
        rec = rec_of(i)
        row = rec & ROW_MASK
        q = row >> ct_shift
        first = (row & (ct - 1)) == 0
        null = (lax.shift_right_logical(rec, NULL_BIT) & 1) == 1
        h = lax.shift_right_logical(rec, HEAD_BIT) & 1
        last = (lax.shift_right_logical(rec, LAST_BIT) & 1) == 1
        pass_first = (lax.shift_right_logical(rec, FIRST_BIT) & 1) == 1

        @pl.when(pass_first & ~null)
        def _():
            hold(q, h)

        ring_wait(slot)

        @pl.when(~null)
        def _():
            @pl.when(first)
            def _():
                ctr_ref[0] = ctr_ref[0] + 1
                group_start(i, q, h, ctr_ref[0] & 1)

            p = ctr_ref[0] & 1
            gather(slot)
            tile_step(slot, p, p)

            @pl.when(last)
            def _():
                finish(q, h, p)

        ring_start(i + R - 1, (slot + R - 1) % R)

    @pl.when(total > 0)
    def _():
        ctr_ref[0] = -1
        ctr_ref[1] = 0
        ctr_ref[2] = 0
        rec0 = rec_ref[0]
        q_copy((rec0 & ROW_MASK) >> ct_shift, lax.shift_right_logical(rec0, HEAD_BIT) & 1, 0).start()
        for k in range(R - 1):
            ring_start(k, k)

        def outer(o, carry):
            for u in range(R):
                step(o * R + u, u)
            return carry

        # Passes are whole ring periods (null tiles pad them), so the count is too.
        lax.fori_loop(0, total // R, outer, None)

        def drain(k, carry):
            ring_wait(k & (R - 1))
            return carry

        lax.fori_loop(total, total + R - 1, drain, None)
        for p in range(2):
            @pl.when(ctr_ref[1 + p] == 1)
            def _(p=p):
                o_copy(0, 0, p).wait()


def held_records(usize, lengths, offsets, *, union_cap, ring, nseq):
    """(rec, nxt, hold, total) of the held kernel for an aligned step: tiles by passes (sequence, head), each pass
    padded to whole ring periods; ``offsets`` int32[nseq + 1] the sequences' first rows in the aligned layout."""
    ngroups = usize.shape[0]
    ct = union_cap // GPT
    gid = jnp.arange(ngroups, dtype=jnp.int32)
    seq = jnp.clip(jnp.searchsorted(offsets[1:] // T, gid, side="right"), 0, nseq - 1).astype(jnp.int32)
    ntiles = jnp.where(usize > 0, (jnp.minimum(usize, union_cap) + GPT - 1) // GPT, 0).astype(jnp.int32)
    per_seq = jax.ops.segment_sum(ntiles, seq, num_segments=nseq)
    padded = (per_seq + ring - 1) // ring * ring
    count = jnp.repeat(padded, 2)                                      # pass 2 s + h
    pass_start = jnp.cumsum(count) - count
    total = jnp.sum(count).astype(jnp.int32)
    seq_start = jnp.cumsum(per_seq) - per_seq
    within = jnp.cumsum(ntiles) - ntiles - seq_start[seq]
    cap = 2 * (ngroups * ct + nseq * ring)
    idx = jnp.arange(cap, dtype=jnp.int32)
    starts, keys = [], []
    for h in range(2):
        starts.append(pass_start[2 * seq + h] + within)
        keys.append((2 * seq + h) * ngroups + gid)
    start = jnp.concatenate(starts)
    key = jnp.concatenate(keys)
    live = jnp.concatenate([ntiles > 0] * 2)
    slot = jnp.where(live, start, cap + jnp.arange(2 * ngroups, dtype=jnp.int32))
    marks = jnp.zeros((cap + 2 * ngroups,), jnp.int32).at[slot].set(key, unique_indices=True)[:cap]
    key_of = lax.cummax(marks, axis=0)
    pas, q = key_of // ngroups, key_of % ngroups
    h = pas % 2
    t = idx - (pass_start[pas] + within[q])
    null = (t < 0) | (t >= ntiles[q]) | (idx >= total)
    tq = jnp.clip(t, 0, ct - 1)
    first_of_pass = (idx == pass_start[pas]) & ~null
    last = (t == ntiles[q] - 1) & ~null
    rec = ((q * ct + tq) | (null.astype(jnp.int32) << NULL_BIT) | (first_of_pass.astype(jnp.int32) << FIRST_BIT)
           | (h << HEAD_BIT) | (last.astype(jnp.int32) << LAST_BIT))
    group_first = (tq == 0) & (t == 0) & ~null
    firsts = jnp.where(group_first, idx, total)
    after = jnp.concatenate([firsts[1:], jnp.full((1,), total, jnp.int32)])
    nxt = lax.cummin(after, axis=0, reverse=True)
    hold = jax.ops.segment_max(lengths.reshape(ngroups, T).max(axis=1), seq, num_segments=nseq)[seq]
    return rec.astype(jnp.int32), nxt.astype(jnp.int32), hold.astype(jnp.int32), total


def token_codes(codes):
    """int32 [G, union_cap * 4]: each slot's code repeated for its 4 tokens, the edge made a token's own (bits 16+:
    the first row that sees the token, plus 1, from -1 to 16)."""
    edge = lax.shift_right_logical(codes, 16) - cc.EDGE_BIAS
    j = jnp.arange(RATIO, dtype=jnp.int32)
    tok = jnp.clip(edge[..., None] + j, -1, 16) + 1
    out = (codes[..., None] & 0xFFFF) | (tok << 16)
    return out.reshape(codes.shape[0], -1)


def compare_core_kernel_held(query, ids, codes, usize, lengths, offsets, page_rows, pool, *, union_cap, held_tokens,
                             nseq, kv_heads=2, k_scale=1.0, v_scale=1.0, ring=4, stages=3, sm_scale=None,
                             vmem_mib=63):
    """Ungated bf16 attention [R, heads, hd] of the aligned rows through held passes; ``ids`` int32[R / 16, cap] each
    slot's sequence-local group id (0 past the union), ``codes`` as ``compare_core_kernel``'s, ``lengths`` the aligned
    rows' lengths, ``offsets`` the sequences' first aligned rows, ``page_rows`` int32[R / 16, P]. The caller checks
    that every sequence's longest row fits ``held_tokens``."""
    tokens, heads, hd = query.shape
    pages, page_size, head_axis, packing, dim = pool.shape
    if kv_heads != 2 or (head_axis, packing, dim) != (1, 4, hd) or hd != 256 or ring & (ring - 1):
        raise ValueError("the held core reads the served FP8 pool of two KV heads, D256; ring a power of two")
    ngroups = tokens // T
    ct = union_cap // GPT
    hpk = heads // kv_heads
    M = hpk * T
    chunk = _hold_chunk(page_size)
    gcap = -(-held_tokens // chunk) * chunk // RATIO
    scale = (sm_scale if sm_scale is not None else hd ** -0.5) * k_scale
    exact = scale > 0 and math.frexp(scale)[0] == 0.5
    q_scale, s_scale = (scale, 1.0) if exact else (1.0, scale)
    rec, nxt, hold, total = held_records(usize, lengths, offsets, union_cap=union_cap, ring=ring, nseq=nseq)
    kernel = functools.partial(
        _held_kernel, ring=ring, stages=stages, chunk=chunk, page_size=page_size, ct=ct, hpk=hpk, head_dim=hd,
        q_scale=float(q_scale), s_scale=float(s_scale), v_scale=float(v_scale))
    scratch = [pltpu.VMEM((gcap, 4, 128), jnp.uint32)]
    scratch += [pltpu.VMEM((chunk, 1, 4, hd), pool.dtype) for _ in range(stages)]
    scratch += [pltpu.VMEM((1, TILE), jnp.int32) for _ in range(ring)]
    scratch += [
        pltpu.SemaphoreType.DMA((stages,)),
        pltpu.SMEM((page_rows.shape[1],), jnp.int32),
        pltpu.SemaphoreType.DMA((1,)),
        pltpu.SMEM((ring * GPT,), jnp.int32),
        pltpu.SemaphoreType.DMA((ring,)),
        pltpu.VMEM((4, GPT, 128), jnp.uint32),
        pltpu.VMEM((2, T, hpk * hd), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.VMEM((2, T, hpk * hd), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.VMEM((2, M, 1), jnp.float32),
        pltpu.VMEM((2, M, 1), jnp.float32),
        pltpu.VMEM((2, M, hd), jnp.float32),
        pltpu.SMEM((4,), jnp.int32),
    ]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4, grid=(1,), in_specs=[hbm] * 5, out_specs=hbm, scratch_shapes=scratch),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), vmem_limit_bytes=vmem_mib * 1024**2, disable_bounds_checks=True),
        out_shape=jax.ShapeDtypeStruct((tokens, heads * hd), jnp.bfloat16),
        name=f"qsa_union_walk_cmp_held_t{T}_r{ring}",
    )(
        rec, nxt, hold, total.reshape(1),
        query.astype(jnp.bfloat16).reshape(tokens, heads * hd),
        ids.astype(jnp.int32).reshape(-1),
        token_codes(codes).reshape(ngroups * ct, TILE),
        page_rows.astype(jnp.int32),
        pool.reshape(pages * page_size, head_axis, packing, dim),
    )
    return out.reshape(tokens, heads, hd)
