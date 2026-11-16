# SPDX-License-Identifier: Apache-2.0
"""QSA prefill attention over 16-row union groups on one flat tile list (VLLM_TPU_QSA_PREFILL_COMPARE_CORE).

The form of Compare's prefill QSA core (parasim bottomup ``QSACoreLean``, group form) on the served
call of ``qsa_rpa._prefill_attention``:

- A union group is 16 consecutive rows of one request (requests start on 16-row boundaries of an
  aligned layout, padding rows have length 0). Its union is the 4-token groups any of its rows
  selects, plus each row's partial tail group, in ascending group order.
- Every union group's union is cut into tiles of 128 union slots (512 tokens). One kernel walks the
  tiles of all union groups in order (``rec``), so the KV copies of the next ``depth - 1`` tiles are
  in flight across union-group boundaries. A tile's 128 groups are 128 copies of one contiguous
  4-token record each (4 KiB: both KV heads, K and V) from the served token-major FP8 pool
  [pages, page, 1, 4, 256]; the copies are issued in the tile body, unrolled, so their scalar issue
  overlaps the tile's vector work.
- The mask comes from one int32 a union slot: bits 0-15 say which of the 16 rows select the group,
  bits 16 and up hold ``4 * group - p0 + 4`` clamped to [0, 20] (``p0`` the position of the union
  group's first row), so row ``r`` sees token ``j`` of the group when its bit is set and
  ``r - j >= 4 * group - p0``.
- Tokens of a tile are visited in (token in group, group) order: the copies land group-major and
  strided loads read token ``j`` of every group, so a slot's code serves the same lane of each of
  the four 128-token blocks.
- Queries are the union group's [16, 24 * 256] rows (head-major columns); each KV head's 12 query
  heads form one [192, 256] block, row ``jj * 16 + r`` for query head ``jj`` and row ``r``. QK and
  PV are single bf16 MXU passes with f32 accumulation (FP8 K and V are exact in bf16); the softmax
  is online per union group; the output is normalized at the union group's last tile and copied out.
- Ring slots are separate buffers and the tile loop's body takes ``depth`` tiles, so each tile's slot
  is static: the refill copies (into another buffer than the one the tile reads) carry no memory
  dependency on the tile's loads and the scheduler interleaves them.
- ``f8dot``: K (and with ``guard`` off, V) enter the MXU as e4m3, without the bf16 conversion.

The new rows' KV is the served write's (``qsa_writes`` runs before attention), so the kernel only
reads the pool. Output: ungated bf16 rows in the aligned layout; ``compare_core_prefill`` gates them
and returns them to the original row order. Not bitwise against the served kernels (bf16
probabilities, union-order accumulation); numerics gated by a max-relative bound.
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp
from vllm_torchtpu.kernels.qwen4_exp.qsa_union_plan import plan_selections
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16

T = 16                 # rows a union group
TILE = 512             # tokens a tile
RATIO = 4              # tokens a selectable group
GPT = TILE // RATIO    # union slots a tile
MASKED = -1e30         # a masked score: finite, so a row that sees nothing yet keeps finite state
EDGE_BIAS = 4          # bits 16+ of a code hold the edge plus this (edges below -4 see every token)
EDGE_MAX = 16          # edges at or above this hide every token from every row
LAST_BIT = 30          # rec: the tile is its union group's last


def _ilog2(x: int) -> int:
    if x <= 0 or x & (x - 1):
        raise ValueError(f"{x} is not a power of two")
    return x.bit_length() - 1


def _byte(words, c):
    """Byte ``c`` of u32 words as e4m3 (truncating casts, as batched_rpa ``convert_to_target_bitwidth``)."""
    x = words if c == 0 else lax.shift_right_logical(words, jnp.uint32(8 * c))
    return pltpu.bitcast(x.astype(jnp.uint16).astype(jnp.uint8), jnp.float8_e4m3fn)


def _head_bytes(words, h):
    """(K, V) e4m3 [n, hd] of KV head h from u32 words [n, hd] of the served record (bytes 2h, 2h + 1)."""
    half = words if h == 0 else lax.shift_right_logical(words, jnp.uint32(16))
    half = half.astype(jnp.uint16)
    k = half.astype(jnp.uint8)
    # u16 shifts go through u32 (as batched_rpa ``convert_to_target_bitwidth``).
    v = pltpu.bitcast(lax.shift_right_logical(pltpu.bitcast(half, jnp.uint32), jnp.uint32(8)), jnp.uint16)
    v = v.astype(jnp.uint8)
    return pltpu.bitcast(k, jnp.float8_e4m3fn), pltpu.bitcast(v, jnp.float8_e4m3fn)


def _kernel(
    rec_ref,      # SMEM int32[tiles_cap]: (q * ct + t) | last << LAST_BIT
    nt_ref,       # SMEM int32[ngroups]: tiles of each union group
    meta_ref,     # SMEM int32[1]: tiles in the step
    q_hbm,        # [rows, heads * hd] bf16
    rows_hbm,     # int32[ngroups * ct * GPT]: pool token row of each union slot's group
    codes_hbm,    # int32[ngroups * ct, GPT]: each union slot's code
    cache_hbm,    # [pool tokens, 1, 4, hd] e4m3
    out_hbm,      # [rows, heads * hd] bf16
    *scratch,
    depth: int,
    ct: int,
    kv_heads: int,
    hpk: int,
    head_dim: int,
    q_scale: float,
    s_scale: float,
    v_scale: float,
    guard: bool,
    f8dot: bool,
    srow_lead: int,
    mode: str,
    pair: bool,
):
    D, HD = depth, head_dim
    E = srow_lead                    # tiles a slot-row copy starts ahead of its tile's refill
    reads, compute = mode in ("full", "reads"), mode in ("full", "compute")
    kv_bufs = scratch[:D]            # VMEM [TILE, 1, 4, hd] e4m3 each: one ring slot a buffer
    code_bufs = scratch[D:2 * D]     # VMEM int32[1, GPT] each
    (kv_sem, srow_ref, srow_sem, qbuf, q_sem, obuf, o_sem, m_ref, l_ref, acc_ref, ctr_ref) = scratch[2 * D:]
    lead = D - 1                     # tiles in flight ahead of the one being read
    ct_shift = _ilog2(ct)
    total = meta_ref[0]

    def rec_of(i):
        return rec_ref[jnp.clip(i, 0, total - 1)]

    def row_of(i):
        return rec_of(i) & ((1 << LAST_BIT) - 1)

    def srow_copy(k, s):
        """Tile k's slot rows into slot-row buffer s (static where the caller knows k mod depth)."""
        src = rows_hbm.at[pl.ds(pl.multiple_of(row_of(k) * GPT, GPT), GPT)]
        dst = srow_ref.at[pl.ds(pl.multiple_of(s * GPT, GPT), GPT)]
        return pltpu.make_async_copy(src, dst, srow_sem.at[s])

    def kv_wait(slot):
        # DMA semaphores count bytes: one descriptor of the whole tile covers its 128 copies; the descriptor's
        # buffers only give the size.
        if not reads:
            return
        pltpu.make_async_copy(cache_hbm.at[pl.ds(0, TILE)], kv_bufs[0], kv_sem.at[slot]).wait()
        pltpu.make_async_copy(codes_hbm.at[pl.ds(0, 1)], code_bufs[0], kv_sem.at[slot]).wait()

    def kv_issue(k, slot):
        """Tile k's 128 group copies and its codes into ring slot ``slot`` (static: k mod depth)."""
        if not reads:
            return
        buf = kv_bufs[slot]

        def copy(src, j, n):
            pltpu.make_async_copy(cache_hbm.at[pl.ds(src, n * RATIO)], buf.at[pl.ds(j * RATIO, n * RATIO)],
                                  kv_sem.at[slot]).start()

        if pair:
            # Two slots whose groups are pool-contiguous move as one 8-token copy; the tile's byte count is the same.
            for j in range(0, GPT, 2):
                a, b = srow_ref[slot * GPT + j], srow_ref[slot * GPT + j + 1]
                joined = b == a + RATIO

                @pl.when(joined)
                def _(a=a, j=j):
                    copy(a, j, 2)

                @pl.when(~joined)
                def _(a=a, b=b, j=j):
                    copy(a, j, 1)
                    copy(b, j + 1, 1)
        else:
            for j in range(GPT):
                copy(srow_ref[slot * GPT + j], j, 1)
        pltpu.make_async_copy(codes_hbm.at[pl.ds(row_of(k), 1)], code_bufs[slot], kv_sem.at[slot]).start()

    def q_copy(q, qs):
        return pltpu.make_async_copy(q_hbm.at[pl.ds(pl.multiple_of(q * T, T), T)], qbuf.at[qs], q_sem.at[qs])

    def o_copy(q, os_):
        return pltpu.make_async_copy(obuf.at[os_], out_hbm.at[pl.ds(pl.multiple_of(q * T, T), T)], o_sem.at[os_])

    def head_kv(slot, h, want_k, want_v):
        """KV head h's K and V [TILE, hd] of ring slot ``slot`` in (token in group, group) order."""
        words = kv_bufs[slot].bitcast(jnp.uint32).reshape(TILE, HD).reshape(TILE * HD // 128, 128)
        per = HD // 128
        ks, vs = [], []
        for j in range(RATIO):
            # Copies land group-major (token 4 g + j at row 4 g + j): a stride of 4 tokens reads token j of each.
            parts = [words[pl.ds(per * j + c, GPT, stride=per * RATIO), :] for c in range(per)]
            k8, v8 = _head_bytes(jnp.concatenate(parts, axis=1), h)
            ks.append(k8)
            vs.append(v8)
        k = v = None
        if want_k:
            k = jnp.concatenate(ks if f8dot else [x.astype(jnp.bfloat16) for x in ks], axis=0)
        if want_v and f8dot and not guard:
            v = jnp.concatenate(vs, axis=0)
        elif want_v:
            v = jnp.concatenate([x.astype(jnp.bfloat16) for x in vs], axis=0)
            if guard:
                # A masked probability is 0, and 0 x NaN is NaN: unwritten pool bytes must not reach the sum.
                v = jnp.where(v == v, v, jnp.zeros_like(v))
        return k, v

    def bias_of(slot):
        """[M, TILE] f32: 0 where a row sees a token, MASKED elsewhere (every query head's 16 rows alike)."""
        cw = code_bufs[slot][...]                                    # [1, GPT]
        code = cw & 0xFFFF
        edge = lax.shift_right_logical(cw, 16) - EDGE_BIAS
        r = lax.broadcasted_iota(jnp.int32, (T, GPT), 0)
        bit = ((code >> r) & 1) != 0
        parts = [jnp.where(bit & (r - j >= edge), 0.0, MASKED).astype(jnp.float32) for j in range(RATIO)]
        bias = jnp.concatenate(parts, axis=1)                        # [T, TILE]
        return jnp.concatenate([bias] * hpk, axis=0)

    def queries(qs, h):
        cols = [qbuf[qs, :, (h * hpk + jj) * HD:(h * hpk + jj + 1) * HD] for jj in range(hpk)]
        q = jnp.concatenate(cols, axis=0)                            # [M, hd]
        if q_scale != 1.0:
            q = (q.astype(jnp.float32) * q_scale).astype(jnp.bfloat16)
        return q

    def tile_step(slot, qs, p):
        """One tile on state parity p: per KV head the masked scores, the online softmax and the values."""
        bias = bias_of(slot)
        for h in range(kv_heads):
            k, v = head_kv(slot, h, True, True)
            s = lax.dot_general(queries(qs, h), k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
            if s_scale != 1.0:
                s = s * s_scale
            s = s + bias
            idx = p * kv_heads + h
            m0 = m_ref[idx]
            m1 = jnp.maximum(m0, jnp.max(s, axis=1, keepdims=True))
            alpha = jnp.exp(m0 - m1)
            prob = jnp.exp(s - m1)
            l_ref[idx] = l_ref[idx] * alpha + jnp.sum(prob, axis=1, keepdims=True)
            acc_ref[idx] = acc_ref[idx] * alpha + lax.dot_general(
                prob.astype(jnp.bfloat16), v, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
            m_ref[idx] = m1

    def reset(p):
        for h in range(kv_heads):
            idx = p * kv_heads + h
            m_ref[idx] = jnp.full(m_ref.shape[1:], MASKED, jnp.float32)
            l_ref[idx] = jnp.zeros(l_ref.shape[1:], jnp.float32)
            acc_ref[idx] = jnp.zeros(acc_ref.shape[1:], jnp.float32)

    def finish(q, p):
        """Union group q's rows out from state parity p: its output buffer's previous copy waited, the
        normalized state written, the copy started."""
        @pl.when(ctr_ref[1 + p] == 1)
        def _():
            o_copy(0, p).wait()

        for h in range(kv_heads):
            idx = p * kv_heads + h
            l = l_ref[idx]
            out = (acc_ref[idx] * (v_scale / jnp.where(l > 0, l, 1.0))).astype(jnp.bfloat16)
            for jj in range(hpk):
                obuf[p, :, (h * hpk + jj) * HD:(h * hpk + jj + 1) * HD] = out[jj * T:(jj + 1) * T]
        o_copy(q, p).start()
        ctr_ref[1 + p] = 1

    def group_start(i, p):
        """Tile i opens a union group on parity p: its query rows (prefetched), the next group's prefetch, a
        fresh state."""
        q = row_of(i) >> ct_shift
        q_copy(0, p).wait()
        nxt = i + nt_ref[q]

        @pl.when(nxt < total)
        def _():
            q_copy(row_of(nxt) >> ct_shift, 1 - p).start()

        reset(p)

    def step(i, slot):
        """Tile i, read from ring slot ``slot`` (static), refilling slot ``slot - 1`` with tile i + depth - 1."""
        rec = rec_of(i)
        row = rec & ((1 << LAST_BIT) - 1)
        q = row >> ct_shift
        first = (row & (ct - 1)) == 0
        last = lax.shift_right_logical(rec, LAST_BIT) & 1

        @pl.when(first)
        def _():
            ctr_ref[0] = ctr_ref[0] + 1
            group_start(i, ctr_ref[0] & 1)

        p = ctr_ref[0] & 1
        kv_wait(slot)
        k = i + lead
        ks = (slot + lead) % D
        srow_copy(k, ks).wait()
        kv_issue(k, ks)
        srow_copy(k + E, (ks + E) % D).start()
        if compute:
            tile_step(slot, p, p)

        @pl.when(last == 1)
        def _():
            finish(q, p)

    @pl.when(total > 0)
    def _():
        ctr_ref[0] = -1
        ctr_ref[1] = 0
        ctr_ref[2] = 0
        q_copy(row_of(0) >> ct_shift, 0).start()
        for k in range(lead):
            srow_copy(k, k % D).start()
        for k in range(lead):
            srow_copy(k, k).wait()
            kv_issue(k, k)
        # Each slot-row copy runs E tiles ahead of its use: an HBM-to-SMEM copy outlasts a tile.
        for k in range(lead, lead + E):
            srow_copy(k, k % D).start()

        def outer(o, carry):
            # Ring slots are static: each body takes depth tiles.
            for u in range(D):
                i = o * D + u

                @pl.when(i < total)
                def _(i=i, u=u):
                    step(i, u)

            return carry

        lax.fori_loop(0, (total + D - 1) // D, outer, None)

        def drain(k, carry):
            kv_wait(k & (D - 1))
            return carry

        lax.fori_loop(total, total + lead, drain, None)

        def drain_srow(k, carry):
            pltpu.make_async_copy(rows_hbm.at[pl.ds(0, GPT)], srow_ref.at[pl.ds(0, GPT)],
                                  srow_sem.at[k & (D - 1)]).wait()
            return carry

        lax.fori_loop(total + lead, total + lead + E, drain_srow, None)
        for p in range(2):
            @pl.when(ctr_ref[1 + p] == 1)
            def _(p=p):
                o_copy(0, p).wait()


def compare_core_kernel(query, rows, codes, usize, pool, *, union_cap, kv_heads=2, k_scale=1.0, v_scale=1.0,
                        depth=8, guard=True, f8dot=False, sm_scale=None, vmem_mib=56, srow_lead=None, mode="full",
                        pair=False):
    """Ungated bf16 attention [R, heads, hd] of the aligned query rows over each union group's tables.

    ``query`` [R, heads, hd] (R a multiple of 16; union group q = rows [16 q, 16 q + 16)); ``rows``
    int32[R / 16, union_cap] the pool token row of each union slot's group, ``codes`` the same shape (module
    docstring), ``usize`` int32[R / 16] each union's size (slots past it are not read; at most
    ``union_cap``, the caller's fallback holds the rest). Rows of union groups with no slot are not written.
    ``srow_lead``: tiles a tile's slot-row copy (HBM to SMEM) starts ahead of its refill (default depth - 2).
    ``mode`` (timing diagnosis): "full"; "compute" (no KV copies: the units on whatever the ring holds); "reads"
    (the copies alone, no tile compute). ``pair``: two slots whose groups are pool-contiguous take one copy.
    """
    tokens, heads, hd = query.shape
    pages, page_size, head_axis, packing, dim = pool.shape
    if tokens % T or heads % kv_heads or (heads // kv_heads) * T % 8:
        raise ValueError("rows must split into 16-row union groups of whole KV-head blocks")
    if pool.dtype != jnp.float8_e4m3fn or (head_axis, packing, dim) != (1, 4, hd) or kv_heads != 2:
        raise ValueError("the compare core reads the served FP8 pool [pages, page, 1, 4, hd] of two KV heads")
    if union_cap % GPT or depth < 3 or depth & (depth - 1):
        raise ValueError("union_cap: whole 128-slot tiles; depth: a power of two, at least 4")
    ngroups = tokens // T
    ct = union_cap // GPT
    _ilog2(ct)
    hpk = heads // kv_heads
    M = hpk * T
    scale = (sm_scale if sm_scale is not None else hd ** -0.5) * k_scale
    # A power-of-two score scale folds into the bf16 queries exactly; any other stays on the f32 scores.
    exact = scale > 0 and math.frexp(scale)[0] == 0.5
    q_scale, s_scale = (scale, 1.0) if exact else (1.0, scale)

    ntiles = jnp.where(usize > 0, (jnp.minimum(usize, union_cap) + GPT - 1) // GPT, 0).astype(jnp.int32)
    first_tile = jnp.cumsum(ntiles) - ntiles
    total = jnp.sum(ntiles).astype(jnp.int32)
    cap = ngroups * ct
    idx = jnp.arange(cap, dtype=jnp.int32)
    group_ids = jnp.arange(ngroups, dtype=jnp.int32)
    marks = jnp.zeros((cap + ngroups,), jnp.int32).at[
        jnp.where(ntiles > 0, first_tile, cap + group_ids)].set(group_ids, unique_indices=True)[:cap]
    q_of = lax.cummax(marks, axis=0)
    t_of = idx - first_tile[q_of]
    last = (t_of == ntiles[q_of] - 1).astype(jnp.int32)
    rec = jnp.where(idx < total, (q_of * ct + t_of) | (last << LAST_BIT), 0).astype(jnp.int32)

    if mode not in ("full", "compute", "reads"):
        raise ValueError("mode: 'full', 'compute' or 'reads'")
    srow_lead = depth - 2 if srow_lead is None else srow_lead
    if not 1 <= srow_lead < depth:
        raise ValueError("srow_lead: from 1 to depth - 1")
    kernel = functools.partial(
        _kernel, depth=depth, ct=ct, kv_heads=kv_heads, hpk=hpk, head_dim=hd, q_scale=float(q_scale),
        s_scale=float(s_scale), v_scale=float(v_scale), guard=guard, f8dot=f8dot, srow_lead=srow_lead, mode=mode,
        pair=pair)
    scratch = [pltpu.VMEM((TILE, 1, 4, hd), pool.dtype) for _ in range(depth)]
    scratch += [pltpu.VMEM((1, GPT), jnp.int32) for _ in range(depth)]
    scratch += [
        pltpu.SemaphoreType.DMA((depth,)),
        pltpu.SMEM((depth * GPT,), jnp.int32),
        pltpu.SemaphoreType.DMA((depth,)),
        pltpu.VMEM((2, T, heads * hd), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.VMEM((2, T, heads * hd), jnp.bfloat16),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.VMEM((2 * kv_heads, M, 1), jnp.float32),
        pltpu.VMEM((2 * kv_heads, M, 1), jnp.float32),
        pltpu.VMEM((2 * kv_heads, M, hd), jnp.float32),
        pltpu.SMEM((4,), jnp.int32),
    ]
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=[hbm] * 4,
            out_specs=hbm,
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), vmem_limit_bytes=vmem_mib * 1024**2, disable_bounds_checks=True),
        out_shape=jax.ShapeDtypeStruct((tokens, heads * hd), jnp.bfloat16),
        name=f"qsa_union_walk_cmp_t{T}_d{depth}" + ("_f8" if f8dot else "") + ("" if guard else "_ng")
        + ("" if mode == "full" else f"_{mode}") + ("_pair" if pair else ""),
    )(
        rec,
        ntiles,
        total.reshape(1),
        query.astype(jnp.bfloat16).reshape(tokens, heads * hd),
        rows.astype(jnp.int32).reshape(-1),
        codes.astype(jnp.int32).reshape(cap, GPT),
        pool.reshape(pages * page_size, head_axis, packing, dim),
    )
    return out.reshape(tokens, heads, hd)


# ---------------------------------------------------------------------------------------------------------------
# Tables: each 16-row union group's union (ascending), its pool rows and its row codes, built on the device.


def _kth_set_bit(word, k):
    """Position of the k-th (0-based) set bit of each uint32 ``word`` (meaningful where k < popcount)."""
    pos = jnp.zeros(k.shape, jnp.int32)
    for width in (16, 8, 4, 2, 1):
        low = lax.population_count(
            lax.shift_right_logical(word, pos.astype(jnp.uint32)) & jnp.uint32((1 << width) - 1)).astype(jnp.int32)
        upper = k >= low
        k = jnp.where(upper, k - low, k)
        pos = jnp.where(upper, pos + width, pos)
    return pos


def group_bits(blocks, lengths, *, groups_cap, words, bitset="tensorcore"):
    """int32[R / 16 * words * 16], flat: each union group's rows' selection bitsets, word-major with row r on lane
    r (the served bitset's bits: indexer selections plus each row's partial tail group; ids at or past
    ``groups_cap`` dropped). Flat because a [G, words, 16] array's 16-wide minor axis pads to 128 lanes on TPU."""
    tokens = blocks.shape[0]
    sel, bit_lengths = plan_selections(blocks, lengths, groups_cap=groups_cap, ratio=RATIO)
    if bitset == "sparsecore":
        from vllm_torchtpu.kernels.qwen4_exp import qsa_union_sc_gather as scg

        return scg.union_sc_bits_flat(sel, bit_lengths, ratio=RATIO, groups=groups_cap, words=words)
    if bitset != "tensorcore":
        raise ValueError("bitset: 'tensorcore' or 'sparsecore'")
    padded = -(-tokens // 128) * 128
    packed = bsp._selection_bitset(
        jnp.pad(sel, ((0, padded - tokens), (0, 0)), constant_values=-1),
        jnp.pad(bit_lengths, (0, padded - tokens)),
        ratio=RATIO, groups=groups_cap, words=words, query_block=128, width=128,
    )  # [padded / 128, words, 128], rows on lanes
    split = packed.reshape(padded // 128, words, 128 // T, T).transpose(0, 2, 1, 3)
    return split.reshape(padded // T, words, T)[: tokens // T].reshape(-1)


def union_lists(bits16, *, union_cap):
    """(ids int32[G, union_cap] ascending, -1 past the union; usize int32[G]) from each union group's bitsets,
    sort-free (``qsa_union_plan.union_plan_nosort``'s list)."""
    ngroups, words, _ = bits16.shape
    ubits = lax.reduce(lax.bitcast_convert_type(bits16, jnp.uint32), np.uint32(0), lax.bitwise_or, (2,))
    counts = lax.population_count(ubits).astype(jnp.int32)
    inclusive = jnp.cumsum(counts, axis=1, dtype=jnp.int32)
    before = inclusive - counts
    usize = inclusive[:, -1]
    lanes = jnp.arange(words, dtype=jnp.int32)
    slots = jnp.where((counts > 0) & (before < union_cap), before, union_cap + lanes)
    grp = jnp.arange(ngroups, dtype=jnp.int32)[:, None]
    starts = jnp.full((ngroups, union_cap + words), -1, jnp.int32).at[grp, slots].set(
        jnp.broadcast_to(lanes, (ngroups, words)), unique_indices=True)
    slot_word = jnp.maximum(lax.cummax(starts[:, :union_cap], axis=1), 0)
    slot = jnp.arange(union_cap, dtype=jnp.int32)[None, :]
    slot_bits = jnp.take_along_axis(ubits, slot_word, axis=1, mode="clip")
    slot_base = jnp.take_along_axis(before, slot_word, axis=1, mode="clip")
    ids = slot_word * 32 + _kth_set_bit(slot_bits, slot - slot_base)
    return jnp.where(slot < usize[:, None], ids, -1), usize


def slot_codes(bits16, ids, p0):
    """int32 codes [G, union_cap] of each union slot (module docstring); 0 for slots past the union."""
    live = ids >= 0
    safe = jnp.where(live, ids, 0)
    word_rows = jnp.take_along_axis(bits16, (safe // 32)[..., None], axis=1, mode="clip")   # [G, cap, 16]
    hit = lax.shift_right_logical(word_rows, (safe % 32)[..., None]) & 1
    code = jnp.sum(hit << jnp.arange(T, dtype=jnp.int32), axis=-1, dtype=jnp.int32)
    edge = jnp.clip(RATIO * safe - p0[:, None], -EDGE_BIAS, EDGE_MAX) + EDGE_BIAS
    return jnp.where(live, code | (edge << 16), 0).astype(jnp.int32)


def slot_rows(ids, page_rows, *, page_size):
    """int32 [G, union_cap]: pool token row of each union slot's group; slots past the union repeat slot 0."""
    live = ids >= 0
    first = jnp.maximum(ids[:, :1], 0)
    tok = jnp.where(live, ids, first) * RATIO
    page = jnp.take_along_axis(page_rows, jnp.minimum(tok // page_size, page_rows.shape[1] - 1), axis=1,
                               mode="clip")
    return (page * page_size + tok % page_size).astype(jnp.int32)


def compare_tables(blocks, lengths, table, *, groups_cap, union_cap, page_size, bitset="tensorcore",
                   lister="reference", emit="rows"):
    """(rows, codes, usize, ok) of an aligned step: the kernel's tables and whether the step takes the core
    (every union within ``union_cap``, and each union group's live rows at consecutive positions from its
    first row, which the codes' edges assume). ``emit="ids"``: each slot's group id (0 past the union) in place
    of its pool row, for the held read."""
    tokens = blocks.shape[0]
    ngroups = tokens // T
    words = -(-groups_cap // 32)
    words = -(-words // 8) * 8
    grouped = lengths.reshape(ngroups, T)
    p0 = grouped[:, 0] - 1
    r = jnp.arange(T, dtype=jnp.int32)[None, :]
    consecutive = jnp.all((grouped == 0) | (grouped == grouped[:, :1] + r))
    page_rows = table.reshape(ngroups, T, table.shape[1])[:, 0]
    bits = group_bits(blocks, lengths, groups_cap=groups_cap, words=words, bitset=bitset)
    if lister == "sparsecore":
        from vllm_torchtpu.kernels.qwen4_exp import qsa_union_sc_gather as scg

        rows, codes, usize = scg.compare_sc_tables(bits, page_rows, p0, words=words, union_cap=union_cap,
                                                   page_size=page_size, emit=emit)
        if emit == "ids":
            rows = jnp.maximum(rows, 0)
    elif lister == "reference":
        bits16 = bits.reshape(ngroups, words, T)
        ids, usize = union_lists(bits16, union_cap=union_cap)
        rows = jnp.maximum(ids, 0) if emit == "ids" else slot_rows(ids, page_rows, page_size=page_size)
        codes = slot_codes(bits16, ids, p0)
    else:
        raise ValueError("lister: 'reference' or 'sparsecore'")
    ok = consecutive & (jnp.max(usize) <= union_cap)
    return rows, codes, usize, ok


# ---------------------------------------------------------------------------------------------------------------
# The call site.


def compare_core_prefill(query, gate, blocks, lengths, table, cache, starts, *, kv_heads, ratio, k_scale=1.0,
                         v_scale=1.0, union_cap, min_context, fallback, depth=8, f8dot=False, guard=True, bitset="tensorcore",
                         lister="reference", stats=0, held_tokens=0):
    """``(gated bf16 output, cache)`` of a packed prefill step through the compare core.

    Same contract as ``block_sparse_gqa_attention`` without ``return_lse``. Requests are aligned to 16-row
    boundaries as ``union_walk_prefill`` aligns them (padding rows of length 0) and their rows return to the
    original order. ``fallback()`` (the served call, gated output) runs instead when the step's longest row is
    below ``min_context`` or the tables refuse the step (a union past ``union_cap``). The cache passes through.

    ``stats`` (debug, N > 0): every N-th call of a process logs the step's union groups, tiles and tiles a live
    group (the work order's k is that over the ~4 of interpolated selections) from a host callback, and also runs
    ``fallback()`` on the same inputs and logs max |core - fallback| / mean |fallback| over live rows; the core's
    output is returned. Adds the callback and the fallback to the program; for an untimed run only.

    ``held_tokens`` (> 0): the held read (``qsa_compare_held``), each sequence's cached KV of one KV head held in
    VMEM for its pass; a step with a sequence longer than this takes ``fallback()``.
    """
    tokens, query_heads, head_dim = query.shape
    pages, page_size = cache.shape[:2]
    if ratio != RATIO:
        raise ValueError("the compare core takes 4-token groups")
    groups_cap = table.shape[1] * page_size // ratio
    align = T
    real = starts[1:] - starts[:-1]
    padded = (real + align - 1) // align * align
    offsets = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(padded)))
    worst_tokens = (tokens + (align - 1) * (len(starts) - 1) + align - 1) // align * align
    bucket_tokens = (tokens + align - 1) // align * align
    sizes = sorted({min(bucket_tokens + extra * align, worst_tokens)
                    for extra in (0, bsp.SPARE_QUERY_BLOCKS, worst_tokens)})

    def core(max_tokens):
        if max_tokens is None:
            layout = query, blocks, lengths, table
            back = None
        else:
            req = jnp.searchsorted(starts[1:], jnp.arange(tokens), side="right")
            req = jnp.minimum(req, len(real) - 1)
            positions = jnp.arange(tokens) - starts[req]
            valid = jnp.arange(tokens) < starts[-1]
            slots = jnp.where(valid, offsets[req] + positions, max_tokens)

            def put(array, fill):
                shape = (max_tokens, *array.shape[1:])
                return jnp.full(shape, fill, array.dtype).at[slots].set(array, mode="drop")

            layout = put(query, 0), put(blocks, -1), put(lengths, 0), put(table, 0)
            back = jnp.minimum(slots, max_tokens - 1)
        a_query, a_blocks, a_lengths, a_table = layout
        rows, codes, usize, ok = compare_tables(a_blocks, a_lengths, a_table, groups_cap=groups_cap,
                                                union_cap=union_cap, page_size=page_size, bitset=bitset,
                                                lister=lister, emit="ids" if held_tokens else "rows")
        if held_tokens:
            ok &= jnp.max(lengths) <= held_tokens

        def run():
            if held_tokens:
                from vllm_torchtpu.kernels.qwen4_exp.qsa_compare_held import compare_core_kernel_held

                seq_rows = starts if max_tokens is None else offsets
                page_rows = a_table.reshape(-1, T, a_table.shape[1])[:, 0]
                return compare_core_kernel_held(
                    a_query, rows, codes, jnp.where(ok, usize, 0), a_lengths, seq_rows, page_rows, cache,
                    union_cap=union_cap, held_tokens=held_tokens, nseq=len(starts) - 1, kv_heads=kv_heads,
                    k_scale=k_scale, v_scale=v_scale)
            return compare_core_kernel(a_query, rows, codes, jnp.where(ok, usize, 0), cache,
                                       union_cap=union_cap, kv_heads=kv_heads, k_scale=k_scale,
                                       v_scale=v_scale, depth=depth, f8dot=f8dot, guard=guard)

        raw = lax.cond(ok, run, lambda: jnp.zeros(a_query.shape, jnp.bfloat16))
        live = usize > 0
        tiles = jnp.sum(jnp.where(live, (jnp.minimum(usize, union_cap) + GPT - 1) // GPT, 0)).astype(jnp.int32)
        counts = jnp.stack([jnp.sum(live).astype(jnp.int32), tiles, jnp.max(usize).astype(jnp.int32)])
        return (raw if back is None else raw[back]), ok, counts

    def skip():
        return jnp.zeros(query.shape, jnp.bfloat16), jnp.zeros((), bool), jnp.zeros((3,), jnp.int32)

    branches = [functools.partial(core, n) for n in sizes]
    route = jnp.sum(offsets[-1] > jnp.asarray(sizes[:-1], jnp.int32))
    if tokens % align == 0:
        branches.insert(0, functools.partial(core, None))
        route = jnp.where(jnp.all(real % align == 0), 0, route + 1)
    branches.append(skip)
    route = jnp.where(jnp.max(lengths) >= min_context, route, len(branches) - 1)
    raw, ok, counts = lax.switch(route, branches)

    def gated():
        out = bf16(bf16(raw).astype(jnp.float32)
                   * bf16(jax.nn.sigmoid(gate.astype(jnp.float32))).astype(jnp.float32))
        return jnp.where((lengths > 0)[:, None, None], out, 0).astype(jnp.bfloat16)

    def served():
        return fallback().astype(jnp.bfloat16)

    out = lax.cond(ok, gated, served)
    if stats:
        ref = served().astype(jnp.float32)
        live = (lengths > 0)[:, None, None]
        err = jnp.max(jnp.where(live, jnp.abs(out.astype(jnp.float32) - ref), 0.0))
        scale = jnp.sum(jnp.where(live, jnp.abs(ref), 0.0)) / jnp.maximum(
            jnp.sum(live) * ref.shape[1] * ref.shape[2], 1)
        finite = jnp.all(jnp.isfinite(jnp.where(live, out.astype(jnp.float32), 0.0)))
        jax.debug.callback(functools.partial(_log_stats, int(stats), int(tokens)), ok, counts, jnp.max(lengths),
                           err / jnp.maximum(scale, 1e-30), finite)
    return out, cache


_STATS = {"calls": 0}


def _log_stats(every, tokens, ok, counts, longest, max_rel, finite):
    """Host side of ``compare_core_prefill(stats=N)``."""
    _STATS["calls"] += 1
    if _STATS["calls"] % every:
        return
    from vllm_torchtpu.logger import init_logger

    groups, tiles, umax = (int(x) for x in np.asarray(counts))
    init_logger(__name__).info(
        "QSA_CMP_STATS call=%d rows=%d longest=%d core=%d live_groups=%d tiles=%d tiles_per_group=%.3f "
        "union_max=%d max_rel=%.5f finite=%d", _STATS["calls"], tokens, int(longest), int(bool(ok)), groups, tiles,
        tiles / max(groups, 1), umax, float(max_rel), int(bool(finite)))


def tiles_per_group(usize, union_cap):
    """Mean tiles of 128 slots a non-empty union group takes (the work order's k is this over ~4)."""
    nt = jnp.where(usize > 0, (jnp.minimum(usize, union_cap) + GPT - 1) // GPT, 0)
    live = jnp.sum(usize > 0)
    return jnp.sum(nt) / jnp.maximum(live, 1)
