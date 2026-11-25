# SPDX-License-Identifier: Apache-2.0
"""q38 QSAF phase 1: a verify step's union decode attention in ONE fused kernel.

``fused_union_attention`` replaces the batched-RPA calls of
``qsa_rpa.union_decode_attention`` (VLLM_TPU_QSA_DECODE_FUSED_UNION). It takes
``qsa_rpa.union_metadata``'s ``(starts, pages, sel, meta, live)`` (formats:
``qsa_union_meta`` module doc) and returns the same ``(out, lse)``: ``out``
[T, H, D] f32 normalized over this rank's owned keys and NOT gated, ``lse``
[T, H] f32 natural log; rows that are not live give 0 and -inf.

One ``pallas_call`` covers every covered request. Its work is the stream of
(request, tile) items in request order, a tile being ``tile`` tokens of the
request's union (640 = 160 four-token groups), ``lanes`` items a step; there
is no batched-RPA schedule, no SMEM page-table copy and no wrapper relayout.
Per step:

- fetch (one step ahead, double-buffered): one DMA per union group of the
  tile (4 tokens, the request's last group bounded by kv_len; no zero-size
  DMAs), one DMA of the tile's selection words. Page ids arrive per request
  by one HBM -> SMEM copy of its first U words, queries per request by one
  DMA of its rows, both ``LOOKAHEAD`` requests ahead.
- compute: RPAd's code, imported and called at RPAd's shapes and configs
  (``strided_load_bkv``, ``get_scale_factors``, ``generate_selection_mask``,
  ``flash_attention_qk_softmax`` / ``_pv`` with fuse_accum's lane-to-lane
  carry), RPAd's ``_accum`` / ``_stage_lse`` expressions at a request's last
  tile, rows written straight to ``out`` / ``lse``.

Bitwise with the union RPA path, by construction: each sequence's carry is a
chain over its own tiles in order (reset to the scratch's initial values
after its last tile), each lane is an independent element of the same batched
dot, so the packing of items into steps does not enter the arithmetic; the
inputs are the same values (bf16 -> f32 queries are exact; tile positions
below the tile's tokens get the same bytes from the same DMAs; the selection
words are the same words). Positions at or past a tile's tokens hold stale,
finite VMEM in both kernels (their words are 0): they enter only as
``0 * v`` or as finite partial sums a later key scales by exactly 0 (the KV
window is zeroed at kernel start, as RPAd zeroes its own).

``issue`` (static): "serial" starts the next step's descriptors before this
step's compute (RPAd's order); "overlap" puts a next step whose tiles are all
full as static, unrolled starts in the same basic block as this step's
compute (partial tiles stay serial; compute_only always issues serially).
``mode`` (static, bench diagnostics,
never reached from an env key): "full"; "issue_only" (every descriptor, no
compute: garbage outputs); "compute_only" (one contiguous DMA a tile: wrong
data, same compute).
"""

import contextlib
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, flash_attention, utils
from vllm_torchtpu.kernels.experimental.batched_rpa import kernel as rpa

try:  # ANNOT's kernel-interior scope helper; absent or its key off -> no scope at all.
    from vllm_torchtpu.tracing.decode_markers import kscope as _kscope
except ImportError:

    def _kscope(name):
        del name
        return contextlib.nullcontext()


RATIO = 4  # tokens a union group (one virtual page)
META = 16  # meta lanes a request: 0 U, 1 kv_len, 2-5 the rows' owned visible tokens
MAX_ROWS = 4
LOOKAHEAD = 8  # requests whose page ids / queries are in flight ahead of the fetch
Q_RING = 12  # >= LOOKAHEAD + lanes (prologue) and >= 2 * lanes (two steps' requests)
PT_RING = 12
OUT_RING = 8
PT_ALIGN = 128  # HBM -> SMEM copy offsets and sizes rounded out to this many words
# union_sc_metadata(write="tile") defines slots up to the end of the 160-slot chunk
# holding U: a tile's slots must lie in one chunk (tile 640 or 128).
WRITE_CHUNK = 160
ISSUES = ("serial", "overlap")
MODES = ("full", "issue_only", "compute_only")
# Tests set this to compile with bounds checks (as batched RPA's kernel.BOUNDS_CHECKS).
BOUNDS_CHECKS = False
# Tests set this for Pallas interpret mode, whose state discharge has no
# dynamic-size slices: every dynamic-size copy (and its wait) then runs as a
# loop of one-row copies with the same source rows, destination rows and
# byte count. On TPU (False) each is one DMA, as batched RPA issues them.
DMA_ROWS = False

# Compacted request list (SMEM, field-major, R words a field).
L_REQ, L_NT, L_KV, L_START, L_N, L_PTB, L_PTN = range(7)
L_FIELDS = 7
# Per-slot step info (SMEM): [slot, lane, field].
I_VALID, I_J, I_K, I_LAST, I_TOK, I_FIRST, I_PTB, I_SEL = range(8)
I_FIELDS = 8
# Scalar state (SMEM).
S_CJ, S_CK, S_ISS, S_J, S_NZ, S_STEPS = range(6)
S_FIELDS = 8


def union_rpa_configs(
    *,
    rows,
    heads,
    head_dim,
    kv_dtype,
    max_query_len,
    tile,
    lanes,
    k_scale,
    v_scale,
    kv_head_offset,
    num_seqs,
    pages_per_seq,
):
    """The ``RpaConfigs`` ``wrapper.ragged_paged_attention`` builds for a union
    call of ``qsa_rpa.union_decode_attention`` (decode mode, f32 queries and
    outputs, 4-token pages, one KV head read at slot ``kv_head_offset``,
    selection words, compact read-only fetch), in the word-row selection
    layout. The fields the reused compute reads are the union call's."""
    out_dtype = jnp.dtype(jnp.float32)
    model = configs.ModelConfigs(
        num_q_heads=heads,
        num_kv_heads=1,
        head_dim=head_dim,
        sliding_window=None,
        sm_scale=head_dim**-0.5,
        soft_cap=None,
        mask_value=jnp.finfo(out_dtype).min,
    )
    serve = configs.ServingConfigs(
        num_seqs=num_seqs,
        num_page_indices=num_seqs * pages_per_seq,
        total_q_tokens=rows,
        dtype_q=out_dtype,
        dtype_kv=jnp.dtype(kv_dtype),
        dtype_out=out_dtype,
        page_size=RATIO,
        per_token_scale=False,
        per_token_scale_dtype=None,
        scale_q=None,
        scale_k=k_scale,
        scale_v=v_scale,
        kv_layout=configs.KVLayout.HEAD_ALONG_SUBLANE,
        decode_query_size=max_query_len,
        cp_group_size=None,
        attention_scope=configs.AttentionScope.FULL,
        return_lse=True,
        skip_kv_update=True,
        lean_kv_fetch=True,
        fast_kv_fetch=True,
        kv_slot=kv_head_offset,
        compact_kv_schedule=True,
        kv_skip_group=0,
        selection_mask=True,
        selection_word_rows=True,
        read_only_cache=True,
    )
    block = configs.BlockSizes(
        bq_sz=max_query_len, bq_c_sz=max_query_len, bkv_sz=tile, batch_size=lanes
    )
    return configs.RpaConfigs(
        block=block,
        model=model,
        serve=serve,
        vmem_limit_bytes=int(pltpu.get_tpu_info().vmem_capacity_bytes),
        mode=configs.RpaCase.DECODE,
    )


def kernel_name(max_query_len, tile, lanes, issue, mode):
    suffix = {"full": "", "issue_only": "-xi", "compute_only": "-xc"}[mode]
    return f"QSAF-p{RATIO}-q{max_query_len}-t{tile}-l{lanes}-{issue}{suffix}"


# Ref views. These three are the only places the kernel bitcasts or reshapes a
# ref; on TPU they are batched RPA's own code. The race-checking interpreter
# (pltpu.InterpretParams) has no ref bitcast / reshape, so its test leg swaps
# them for value-level equivalents (tests/kernels/test_qwen4_exp_qsa_fused_union.py).
def _zero_kv(kv_win):
    kv_win.bitcast(jnp.uint32)[...] = jnp.zeros(kv_win.bitcast(jnp.uint32).shape, jnp.uint32)


def _load_kv(kv_in, b_idx, start, *, cfgs):
    return rpa.strided_load_bkv(kv_in, b_idx, start, cfgs=cfgs)


def _annotate(ref):
    return pltpu.annotate(ref, no_hazard_no_deps=True)


class _Copy:
    """``src[s0 : s0 + size] -> dst[d0 : d0 + size]`` on ``sem``: one DMA, or
    under ``DMA_ROWS`` (interpret mode) a traced ``size`` as ``size``
    one-row DMAs (starts and waits alike; assumed: the interpreter's DMA semaphores count bytes, so a
    static four-row start is matched by four one-row waits; the interpret tests deadlock or fail otherwise). Static sizes
    always issue one DMA (a per-row split of every group copy makes the
    interpreter rewrite the whole KV window per row)."""

    def __init__(self, src, s0, dst, d0, size, sem):
        self.args = (src, s0, dst, d0, size, sem)

    def _each(self, op):
        src, s0, dst, d0, size, sem = self.args
        if not DMA_ROWS or isinstance(size, int):
            op(pltpu.make_async_copy(src.at[pl.ds(s0, size)], dst.at[pl.ds(d0, size)], sem))
            return

        @pl.loop(0, size)
        def _(i):
            op(pltpu.make_async_copy(src.at[pl.ds(s0 + i, 1)], dst.at[pl.ds(d0 + i, 1)], sem))

    def start(self):
        self._each(lambda c: c.start())

    def wait(self):
        self._each(lambda c: c.wait())


class _LaneFlags:
    """``schedule_ref.is_last_k`` for the reused flash functions: ``[step, lane]``
    gives the lane's flag of the step being computed."""

    def __init__(self, flags):
        self.flags = flags

    def __getitem__(self, index):
        _, lane = index
        return self.flags[lane]


def _kernel(
    # Scalar prefetch.
    starts_ref,
    meta_ref,
    # Inputs (HBM).
    q_hbm,
    pages_hbm,
    sel_hbm,
    cache_hbm,
    # Outputs.
    out_hbm,
    lse_ref,
    # Scratch.
    kv_win,
    sel_win,
    q_stage,
    q48,
    m_scr,
    l_scr,
    acc_scr,
    rows_scr,
    o_stage,
    zrow,
    lse_fin,
    pt_smem,
    lists,
    info,
    state,
    kv_sem,
    sel_sem,
    q_sem,
    pt_sem,
    o_sem,
    z_sem,
    *,
    cfgs,
    reqs,
    pages_per_seq,
    tiles_per_seq,
    issue,
    mode,
):
    lanes = cfgs.batch_size
    tile = cfgs.bkv_sz
    groups = tile // RATIO
    mql = cfgs.bq_sz
    heads = cfgs.aligned_num_q_heads_per_kv_head
    rows = out_hbm.shape[0]
    pt_words = pt_smem.shape[0] // PT_RING
    cache_rows = cache_hbm.shape[0]

    def lst(field, j):
        return lists[field * reqs + j]

    def lst_set(field, j, value):
        lists[field * reqs + j] = value

    def inf(slot, b, field):
        return info[(slot * lanes + b) * I_FIELDS + field]

    def inf_set(slot, b, field, value):
        info[(slot * lanes + b) * I_FIELDS + field] = value

    # ---- copies --------------------------------------------------------------
    def kv_copy(slot, b, src_row, dst_row, size):
        return _Copy(cache_hbm, src_row, kv_win.at[slot, b], dst_row, size, kv_sem.at[slot])

    def sel_copy(slot, b, row, count=1):
        return _Copy(sel_hbm, row, sel_win.at[slot], b, count, sel_sem.at[slot])

    def q_copy(j, row0, n):
        ring = lax.rem(j, Q_RING)
        return _Copy(q_hbm, row0, q_stage.at[ring], 0, n, q_sem.at[ring])

    def pt_copy(j, lo, size):
        ring = lax.rem(j, PT_RING)
        return _Copy(
            pages_hbm, lo, pt_smem, pl.multiple_of(ring * pt_words, PT_ALIGN), size, pt_sem.at[ring]
        )

    def out_copy(oslot, row0, n):
        return _Copy(o_stage.at[oslot], 0, out_hbm, row0, n, o_sem.at[oslot])

    def zero_copy(t):
        return pltpu.make_async_copy(zrow, out_hbm.at[pl.ds(t, 1)], z_sem.at[0])

    def zero_rows(lo, hi):
        # Rows [lo, hi) of out get 0 (their LSE stays -inf): rows of no attended request.
        lo = jnp.clip(lo, 0, rows)
        hi = jnp.clip(hi, lo, rows)

        @pl.loop(lo, hi)
        def _(t):
            zero_copy(t).start()

        state[S_NZ] = state[S_NZ] + (hi - lo)

    # ---- prologue ------------------------------------------------------------
    with _kscope("qsa_attn.meta"):
        # The KV window holds finite bytes everywhere (positions a tile does not
        # fetch enter p @ v with weight 0, and 0 * NaN would be NaN).
        _zero_kv(kv_win)
        q_stage[...] = jnp.zeros(q_stage.shape, q_stage.dtype)
        m_scr[...] = jnp.full(m_scr.shape, -jnp.inf, m_scr.dtype)
        l_scr[...] = jnp.zeros(l_scr.shape, l_scr.dtype)
        acc_scr[...] = jnp.zeros(acc_scr.shape, acc_scr.dtype)
        zrow[...] = jnp.zeros(zrow.shape, zrow.dtype)
        lse_fin[...] = jnp.full(lse_fin.shape, -jnp.inf, lse_fin.dtype)
        for f in range(S_FIELDS):
            state[f] = 0

        def scan_request(r, carry):
            j, items = carry
            s0 = starts_ref[r]
            nrows = jnp.maximum(starts_ref[r + 1] - s0, 0)
            n = jnp.minimum(nrows, mql)
            kv = jnp.maximum(meta_ref[META * r + 1], 0)
            nt = jnp.where(nrows > 0, lax.div(kv + tile - 1, tile), 0)

            @pl.when(nt > 0)
            def _():
                lst_set(L_REQ, j, r)
                lst_set(L_NT, j, nt)
                lst_set(L_KV, j, kv)
                lst_set(L_START, j, s0)
                lst_set(L_N, j, n)
                zero_rows(s0 + mql, s0 + nrows)  # rows past the first max_query_len

            pl.when(nt == 0)(lambda: zero_rows(s0, s0 + nrows))
            return j + (nt > 0).astype(jnp.int32), items + nt

        num_j, items = lax.fori_loop(0, reqs, scan_request, (jnp.int32(0), jnp.int32(0)))
        state[S_J] = num_j
        state[S_STEPS] = lax.div(items + lanes - 1, lanes)
        zero_rows(0, starts_ref[0])
        zero_rows(starts_ref[reqs], rows)

    def issue_request(j):
        # Request j's page ids (HBM -> SMEM ring) and query rows (HBM -> VMEM ring).
        r = lst(L_REQ, j)
        flat0 = r * pages_per_seq
        lo = lax.div(flat0, PT_ALIGN) * PT_ALIGN
        u = meta_ref[META * r]
        hi = jnp.minimum(
            lax.div(flat0 + u + PT_ALIGN - 1, PT_ALIGN) * PT_ALIGN, reqs * pages_per_seq
        )
        # Offsets and sizes are whole PT_ALIGN-word (512-byte) units, as the DMA
        # needs; the hints let Mosaic see it through the traced arithmetic.
        lo = pl.multiple_of(lo, PT_ALIGN)
        size = pl.multiple_of(hi - lo, PT_ALIGN)
        lst_set(L_PTB, j, lax.rem(j, PT_RING) * pt_words + (flat0 - lo))
        lst_set(L_PTN, j, size)
        pt_copy(j, lo, size).start()
        q_copy(j, lst(L_START, j), lst(L_N, j)).start()

    def issue_requests(upto):
        lo = state[S_ISS]
        hi = jnp.maximum(jnp.minimum(state[S_J], upto), lo)

        @pl.loop(lo, hi)
        def _(j):
            issue_request(j)

        state[S_ISS] = hi

    def fill_info(slot):
        # The next `lanes` stream items (cursor S_CJ / S_CK) into info[slot].
        j = state[S_CJ]
        k = state[S_CK]
        num_j = state[S_J]
        for b in range(lanes):
            valid = j < num_j
            jj = jnp.minimum(j, num_j - 1)
            nt = lst(L_NT, jj)
            kv = lst(L_KV, jj)
            last = valid & (k == nt - 1)
            tokens = jnp.where(valid, jnp.minimum(tile, kv - tile * k), 0)
            inf_set(slot, b, I_VALID, valid.astype(jnp.int32))
            inf_set(slot, b, I_J, jj)
            inf_set(slot, b, I_K, k)
            inf_set(slot, b, I_LAST, last.astype(jnp.int32))
            inf_set(slot, b, I_TOK, tokens)
            inf_set(slot, b, I_FIRST, (valid & (k == 0)).astype(jnp.int32))
            inf_set(slot, b, I_SEL, lst(L_REQ, jj) * tiles_per_seq + k)
            j = jnp.where(last, j + 1, j)
            k = jnp.where(last, 0, jnp.where(valid, k + 1, k))
        state[S_CJ] = j
        state[S_CK] = k

    def prepare_step(slot):
        # Requests first seen in this step: wait their page ids and query rows,
        # widen the rows to f32 (exact) in RPAd's query order (row-major, heads
        # inner); then each lane's page-id base.
        for b in range(lanes):

            @pl.when(inf(slot, b, I_FIRST) == 1)
            def _(b=b):
                j = inf(slot, b, I_J)
                ring = lax.rem(j, Q_RING)
                n = lst(L_N, j)
                pt_copy(j, 0, pl.multiple_of(lst(L_PTN, j), PT_ALIGN)).wait()
                q_copy(j, 0, n).wait()
                for i in range(mql):
                    q48[ring, pl.ds(heads * i, heads), :] = q_stage[ring, i].astype(jnp.float32)

            inf_set(
                slot,
                b,
                I_PTB,
                lst(L_PTB, inf(slot, b, I_J)) + groups * inf(slot, b, I_K),
            )

    # ---- fetch issue ---------------------------------------------------------
    def issue_full_static(slot, b):
        # A full tile: `groups` four-token DMAs at static VMEM offsets.
        ptb = inf(slot, b, I_PTB)
        for i in range(groups):
            kv_copy(slot, b, pt_smem[ptb + i] * RATIO, RATIO * i, RATIO).start()

    def issue_partial(slot, b):
        # The request's last tile: g - 1 four-token DMAs, then its last group
        # bounded by kv_len.
        ptb = inf(slot, b, I_PTB)
        tokens = inf(slot, b, I_TOK)
        g = lax.div(tokens + RATIO - 1, RATIO)

        @pl.loop(0, g - 1)
        def _(i):
            kv_copy(slot, b, pt_smem[ptb + i] * RATIO, RATIO * i, RATIO).start()

        last = g - 1
        kv_copy(
            slot, b, pt_smem[ptb + last] * RATIO, RATIO * last, tokens - RATIO * last
        ).start()

    def issue_contiguous(slot, b):
        # compute_only: one DMA of the tile's tokens from one place (wrong data).
        src = jnp.clip(pt_smem[inf(slot, b, I_PTB)] * RATIO, 0, cache_rows - tile)
        kv_copy(slot, b, src, 0, inf(slot, b, I_TOK)).start()

    def issue_serial(slot):
        with _kscope("qsa_attn.issue"):
            for b in range(lanes):

                @pl.when(inf(slot, b, I_VALID) == 1)
                def _(b=b):
                    sel_copy(slot, b, inf(slot, b, I_SEL)).start()
                    if mode == "compute_only":
                        issue_contiguous(slot, b)
                    else:
                        full = inf(slot, b, I_TOK) == tile
                        pl.when(full)(functools.partial(issue_full_static, slot, b))
                        pl.when(jnp.logical_not(full))(functools.partial(issue_partial, slot, b))

    def issue_overlap(slot):
        # Every lane valid and full: static, unrolled starts, no branch.
        with _kscope("qsa_attn.issue"):
            for b in range(lanes):
                sel_copy(slot, b, inf(slot, b, I_SEL)).start()
                issue_full_static(slot, b)

    def wait_step(slot):
        count = 0
        for b in range(lanes):
            valid = inf(slot, b, I_VALID)
            count = count + valid

            @pl.when(valid == 1)
            def _(b=b):
                kv_copy(slot, b, 0, 0, inf(slot, b, I_TOK)).wait()

        sel_copy(slot, 0, 0, count).wait()

    # ---- compute -------------------------------------------------------------
    def finish(slot, b, acc, l_val, m_val):
        # A request's last tile: RPAd's _accum / _stage_lse arithmetic, live
        # rows to out / lse, other rows 0 / -inf.
        j = inf(slot, b, I_J)
        r = lst(L_REQ, j)
        s0 = lst(L_START, j)
        n = lst(L_N, j)
        oslot = lax.rem(j, OUT_RING)

        @pl.when(j >= OUT_RING)
        def _():
            out_copy(oslot, 0, lst(L_N, j - OUT_RING)).wait()

        with _kscope("qsa_attn.out"):
            batch_l = utils.broadcast_minor(l_val, acc.shape)
            result = lax.div(acc, batch_l).astype(cfgs.serve.dtype_out)
            rows_scr[...] = result[0]
            live = [meta_ref[META * r + 2 + i] > 0 for i in range(mql)]
            for i in range(mql):
                o_stage[oslot, i] = jnp.where(
                    live[i], rows_scr[pl.ds(heads * i, heads), :], 0.0
                )
            out_copy(oslot, s0, n).start()

            lse_val = m_val + jnp.log(jnp.maximum(l_val, 1e-9))
            lse_val = lse_val.astype(cfgs.serve.dtype_out)[0]  # [mql * heads, 128]
            num_lanes = lse_val.shape[-1]
            qrow = lax.broadcasted_iota(jnp.int32, lse_val.shape, 0)
            lane = lax.broadcasted_iota(jnp.int32, lse_val.shape, 1)
            ys = []
            for i in range(mql):
                pick = (qrow - heads * i == lane) & (lane < heads)
                y = jnp.max(jnp.where(pick, lse_val, -jnp.inf), axis=0, keepdims=True)
                ys.append(jnp.where(live[i], y, -jnp.inf))

            def put(block):
                base = pl.multiple_of(block * 8, 8)
                cur = lse_fin[pl.ds(base, 8), :]
                rid = lax.broadcasted_iota(jnp.int32, (8, num_lanes), 0) + base
                for i in range(mql):
                    cur = jnp.where((rid == s0 + i) & (i < n), ys[i], cur)
                lse_fin[pl.ds(base, 8), :] = cur

            b0 = lax.div(s0, 8)
            b1 = lax.div(s0 + n - 1, 8)
            put(b0)
            pl.when(b1 > b0)(lambda: put(b1))

    def compute(slot):
        with _kscope("qsa_attn.compute"):
            kv_in = _annotate(kv_win.at[slot])
            sel_vref = _annotate(sel_win.at[slot])
            # Queries [lanes, 1, mql * heads, D] f32, RPAd's q block layout.
            q = jnp.stack(
                [q48[lax.rem(inf(slot, b, I_J), Q_RING)] for b in range(lanes)], axis=0
            )
            q = q.reshape(lanes, cfgs.model.num_kv_heads, mql * heads, cfgs.aligned_q_head_dim)
            q = q[..., : cfgs.compute_kv_head_dim]
            # K / V: rpa_body's HEAD_ALONG_SUBLANE load (kernel.py:597-621).
            k_b = []
            v_b = []
            for b_idx in range(lanes):
                heads_per_load = pl.cdiv(cfgs.serve.packing_kv, 2)
                ks = []
                vs = []
                for kv_head_start in range(0, cfgs.model.num_kv_heads, heads_per_load):
                    bkv_lst = _load_kv(kv_in, b_idx, kv_head_start * 2, cfgs=cfgs)
                    ks.append(jnp.stack([k for k, _ in bkv_lst], axis=0))
                    vs.append(jnp.stack([v for _, v in bkv_lst], axis=0))
                k, v = jnp.concat(ks, axis=0), jnp.concat(vs, axis=0)
                k = k.reshape(-1, cfgs.bkv_sz, cfgs.aligned_kv_head_dim)
                v = v.reshape(-1, cfgs.bkv_sz, cfgs.aligned_kv_head_dim)
                k_b.append(k[: cfgs.model.num_kv_heads])
                v_b.append(v[: cfgs.model.num_kv_heads])
            k = jnp.stack(k_b, axis=0)
            v = jnp.stack(v_b, axis=0)
            k, v, k_scale, v_scale = rpa.get_scale_factors(k, v, cfgs=cfgs)

            m_val = m_scr[...]
            l_val = l_scr[...]
            acc_val = acc_scr[...]
            flags = _LaneFlags([inf(slot, b, I_LAST) for b in range(lanes)])
            step_meta = rpa.ComputeMetadata(causal_offset=[], query_base=[0] * lanes)
            # One query chunk (bq_c == bq), so rpa_body's chunk loop is one
            # qk_softmax then one pv.
            q_slice = slice(0, mql * heads)
            mask_shape = (lanes, cfgs.model.num_kv_heads, mql * heads, cfgs.bkv_sz)
            custom_mask = rpa.generate_selection_mask(
                mask_shape, bq_start=0, step_meta=step_meta, sel_vref=sel_vref, cfgs=cfgs
            )
            p, alpha_list, m_next, l_next, m_carry, l_carry = (
                flash_attention.flash_attention_qk_softmax(
                    0,
                    q[:, :, q_slice],
                    k,
                    m_val[:, q_slice],
                    l_val[:, q_slice],
                    flags,
                    custom_mask=custom_mask,
                    cfgs=cfgs,
                    bq_start=0,
                    k_scale=k_scale,
                )
            )
            m_scr[:, q_slice] = m_carry
            l_scr[:, q_slice] = l_carry
            o_next, o_carry = flash_attention.flash_attention_pv(
                p, v, alpha_list, acc_val[:, q_slice], 0, flags, cfgs=cfgs, v_scale=v_scale
            )
            acc_scr[:, q_slice] = o_carry
            m_next = jnp.concatenate([m_next], axis=2)
            l_next = jnp.concatenate([l_next], axis=2)
            acc_next = jnp.concatenate([o_next], axis=2)

        for b in range(lanes):

            @pl.when((inf(slot, b, I_VALID) == 1) & (inf(slot, b, I_LAST) == 1))
            def _(b=b):
                finish(slot, b, acc_next[b], l_next[b], m_next[b])

    def run_compute(slot):
        if mode != "issue_only":
            compute(slot)

    # ---- the stream ----------------------------------------------------------
    steps = state[S_STEPS]

    @pl.when(steps > 0)
    def _():
        with _kscope("qsa_attn.meta"):
            fill_info(0)
            issue_requests(state[S_CJ] + LOOKAHEAD)
            prepare_step(0)
        issue_serial(0)

        @pl.loop(0, steps)
        def _(s):
            slot = lax.rem(s, 2)
            nxt = 1 - slot
            has_next = s + 1 < steps
            wait_step(slot)

            @pl.when(has_next)
            def _():
                with _kscope("qsa_attn.meta"):
                    fill_info(nxt)
                    prepare_step(nxt)

            if issue == "overlap" and mode != "compute_only":
                every = has_next
                for b in range(lanes):
                    every = every & (inf(nxt, b, I_VALID) == 1) & (inf(nxt, b, I_TOK) == tile)

                @pl.when(every)
                def _():
                    issue_overlap(nxt)
                    run_compute(slot)

                @pl.when(jnp.logical_not(every))
                def _():
                    pl.when(has_next)(lambda: issue_serial(nxt))
                    run_compute(slot)

            else:
                pl.when(has_next)(lambda: issue_serial(nxt))
                run_compute(slot)

            @pl.when(has_next)
            def _():
                with _kscope("qsa_attn.meta"):
                    issue_requests(state[S_CJ] + LOOKAHEAD)

    # ---- drain ---------------------------------------------------------------
    num_j = state[S_J]
    if mode != "issue_only":

        @pl.loop(jnp.maximum(num_j - OUT_RING, 0), num_j)
        def _(j):
            out_copy(lax.rem(j, OUT_RING), 0, lst(L_N, j)).wait()

    @pl.loop(0, state[S_NZ])
    def _(t):
        zero_copy(0).wait()

    with _kscope("qsa_attn.out"):
        # [rows rounded to 8, 128] f32, head h in lane h: a 512-byte inner slice for
        # the output DMA (a [T, H] block would be 4 H bytes); sliced to [T, H] outside.
        lse_ref[...] = lse_fin[...]


def fused_union_attention(
    q,
    cache,
    metadata,
    *,
    ratio,
    k_scale,
    v_scale,
    kv_head_offset,
    max_query_len,
    issue="serial",
    mode="full",
    tile=640,
    lanes=4,
    interpret=False,
):
    """``(out, lse)`` of ``qsa_rpa.union_decode_attention`` from its
    ``union_metadata`` (see the module doc). ``q`` [T, H, D] in the caller's
    dtype (bf16 -> f32 is exact), ``cache`` [pages, page_size, 1, packing, D]
    (read only, never returned). Every request holds at most
    ``max_query_len`` rows on the union route (``qsa_attention``'s ``fits``);
    a request's rows past that get 0 / -inf here."""
    starts, pages, sel, meta, live = metadata
    del live  # the kernel applies union_row_live's rule from meta itself
    if issue not in ISSUES:
        raise ValueError(f"issue={issue!r}: expected one of {ISSUES}")
    if mode not in MODES:
        raise ValueError(f"mode={mode!r}: expected one of {MODES}")
    rows, heads, head_dim = q.shape
    reqs = starts.shape[0] - 1
    pages_per_seq = pages.shape[1]
    if not (
        ratio == RATIO
        and tile % 128 == 0
        and WRITE_CHUNK % (tile // RATIO) == 0
        and 1 <= max_query_len <= MAX_ROWS
        and cache.ndim == 5
        and cache.shape[1] % RATIO == 0
        and cache.shape[2] == 1
        and cache.shape[4] == head_dim
        and 0 <= kv_head_offset < cache.shape[3] // 2
        and pages.shape == (reqs, pages_per_seq)
        and pages_per_seq % (tile // RATIO) == 0
        and sel.shape == (reqs, RATIO * pages_per_seq)
        and meta.shape == (reqs, META)
    ):
        raise ValueError(
            "fused_union_attention: unsupported shapes "
            f"q {q.shape} cache {cache.shape} pages {pages.shape} sel {sel.shape} "
            f"meta {meta.shape} starts {starts.shape} ratio {ratio} tile {tile} "
            f"max_query_len {max_query_len} kv_head_offset {kv_head_offset}"
        )
    cfgs = union_rpa_configs(
        rows=rows,
        heads=heads,
        head_dim=head_dim,
        kv_dtype=cache.dtype,
        max_query_len=max_query_len,
        tile=tile,
        lanes=lanes,
        k_scale=k_scale,
        v_scale=v_scale,
        kv_head_offset=kv_head_offset,
        num_seqs=reqs,
        pages_per_seq=pages_per_seq,
    )
    if cfgs.serve.packing_kv != cache.shape[3] or cfgs.kv_hbm_stride != 1:
        raise ValueError(f"fused_union_attention: cache rows {cache.shape} are not one packed sublane")
    tiles_per_seq = RATIO * pages_per_seq // tile
    num_lanes = pltpu.get_tpu_info().num_lanes
    pt_words = -(-(pages_per_seq + 2 * PT_ALIGN) // PT_ALIGN) * PT_ALIGN  # whole 512-byte units
    lse_rows = -(-rows // 8) * 8
    body = functools.partial(
        _kernel,
        cfgs=cfgs,
        reqs=reqs,
        pages_per_seq=pages_per_seq,
        tiles_per_seq=tiles_per_seq,
        issue=issue,
        mode=mode,
    )
    scratch = [
        pltpu.VMEM((2, *cfgs.kv_vmem_shape), cache.dtype),  # kv_win
        pltpu.VMEM((2, *cfgs.selection_vmem_shape), jnp.int32),  # sel_win
        pltpu.VMEM((Q_RING, max_query_len, heads, head_dim), q.dtype),  # q_stage
        pltpu.VMEM((Q_RING, max_query_len * heads, head_dim), jnp.float32),  # q48
        pltpu.VMEM(cfgs.lm_scratch_shape, jnp.float32),  # m
        pltpu.VMEM(cfgs.lm_scratch_shape, jnp.float32),  # l
        pltpu.VMEM(cfgs.acc_scratch_shape, jnp.float32),  # acc
        pltpu.VMEM((max_query_len * heads, head_dim), jnp.float32),  # rows_scr
        pltpu.VMEM((OUT_RING, max_query_len, heads, head_dim), jnp.float32),  # o_stage
        pltpu.VMEM((1, heads, head_dim), jnp.float32),  # zrow
        pltpu.VMEM((lse_rows, num_lanes), jnp.float32),  # lse_fin
        pltpu.SMEM((PT_RING * pt_words,), jnp.int32),  # pt_smem
        pltpu.SMEM((L_FIELDS * max(reqs, 1),), jnp.int32),  # lists
        pltpu.SMEM((2 * lanes * I_FIELDS,), jnp.int32),  # info
        pltpu.SMEM((S_FIELDS,), jnp.int32),  # state
        pltpu.SemaphoreType.DMA((2,)),  # kv
        pltpu.SemaphoreType.DMA((2,)),  # sel
        pltpu.SemaphoreType.DMA((Q_RING,)),  # q
        pltpu.SemaphoreType.DMA((PT_RING,)),  # page ids
        pltpu.SemaphoreType.DMA((OUT_RING,)),  # out rows
        pltpu.SemaphoreType.DMA((1,)),  # zero rows
    ]
    out, lse = pl.pallas_call(
        body,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 4,
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((lse_rows, num_lanes), lambda i, *_: (0, 0)),
            ],
            scratch_shapes=scratch,
        ),
        out_shape=[
            jax.ShapeDtypeStruct((rows, heads, head_dim), jnp.float32),
            jax.ShapeDtypeStruct((lse_rows, num_lanes), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=cfgs.vmem_limit_bytes,
            disable_bounds_checks=not BOUNDS_CHECKS,
            flags={"XLA_TPU_EVENLY_DISTRIBUTE_CRITICAL_NODES": True},
        ),
        name=kernel_name(max_query_len, tile, lanes, issue, mode),
        interpret=interpret,
    )(
        starts.astype(jnp.int32),
        meta.astype(jnp.int32).reshape(-1),
        q,
        pages.astype(jnp.int32).reshape(-1),
        sel.astype(jnp.int32).reshape(reqs * tiles_per_seq, tile // num_lanes, 1, num_lanes),
        cache.reshape(-1, *cache.shape[2:]),
    )
    return out, lse[:rows, :heads]
