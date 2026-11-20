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
"""Exact top-k over a row split across DCP ranks, in one SparseCore pass.

Each DCP rank holds its own part of every row (its shard of the KV cache's
scores). The exact top-k of the whole row is found with the same radix select
as `sparsecore_topk`, except that every level's histogram is summed over the
ranks before the bucket is picked:

1. Each subcore takes up to `MAX_ROWS` of its rank's row parts per pass and
   builds one 256-bucket histogram per row and level. Parts that fit tile
   memory (`ROW_WORDS`) are fetched once; longer ones are streamed from HBM
   again at every level, so the pass still exchanges all its rows at once.
2. The scalar subcore of each SparseCore exchanges its core's histograms with
   the same core on the other ranks (remote DMA into `VMEM_SHARED`), the
   ranks' histograms of a row are summed, and the bucket is picked. Four
   levels of 8 bits resolve the k-th score exactly.
3. Each rank then emits only its own winners. Ties at the k-th score go to the
   lower ranks first, from the ranks' own counts in the final bucket.

Each row has an owner rank, which alone receives the row's histograms, sums
and picks, and sends each rank back the row's state: two short rounds per
level, and a rank sends each histogram once rather than to every peer. Every
level of a pass costs those rounds' latency whatever the rows' length, so the
kernel is for prefill-sized calls; decode keeps the candidate merge.

With `one_round`, every rank instead sends its histograms to every rank, and
each rank sums them in rank order and picks every row itself: the counts are
exact integers, so all ranks reach the same state in one round per level.
"auto" does so only for passes of few rows per subcore, where the rounds'
latency outweighs the broadcast's bytes.

With `no_restream`, the row buffer takes the tile memory the rest leaves, so
longer parts stay resident; a pass that still streams copies each row's
survivors (every key from the window up, with its column) into the rest of
the buffer at CMP_LEVELS if they fit, and later levels and the emit read only
those. With `pack16`, a pass whose bounds are all at most PACK16_MAX sends its
counts as 16-bit pairs: a count never exceeds its rank's part length.

The output is this rank's share of every row's top-k, as positions in its own
part of the row, packed into a prefix and `-1` padded. Across the ranks the
rows hold exactly the global top-k, so there is no candidate exchange and no
second top-k: the hierarchical local top-k, all-gather and merge collapse into
one pass.
"""

import functools
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.pallas import core as pallas_core
from jax.experimental import pallas as pl
from jax.experimental import xla_metadata
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import (
    BITS_NEG_INF,
    KEY_NEG_INF,
    L0_SHIFT,
    LANES,
    MONO_MASK,
    NUM_BUCKETS,
    REFINEMENT_LEVELS,
    SCAN_UNROLL,
)

# Tile memory for resident row parts per subcore (256 KB of 512 KB), and the
# most row parts a subcore holds at once. A pass over the rows exchanges
# histograms once per level, so short parts are batched up to MAX_ROWS deep.
ROW_WORDS = 64 * 1024
SEG_WORDS = 32 * 1024
MAX_ROWS = 8
NUM_CORES = 2
NUM_SUBCORES = 16
ROWS_PER_PASS = NUM_CORES * NUM_SUBCORES
NUM_LEVELS = 1 + len(REFINEMENT_LEVELS)

# Per-row selection state in SMEM, NF words per row: the row, its part's
# length on this rank, the selection, and where its compacted survivors are.
(
    F_ROW,
    F_LEN,
    F_PREFIX,
    F_SHIFT,
    F_QUOTA,
    F_DONE,
    F_KEEP,
    F_MINE,
    F_BEFORE,
    F_CMP,
    F_COFF,
    F_CLEN,
    F_ABOVE,
) = range(13)
NF = 13
# The state an owner sends back, one STATE_WORDS vector per row and rank.
STATE_FIELDS = (F_PREFIX, F_SHIFT, F_QUOTA, F_DONE, F_KEEP, F_MINE, F_BEFORE)
STATE_WORDS = LANES
# Exchange: False two rounds/level, True one, "auto" one if <= ONE_ROUND_MAX_ROWS.
ONE_ROUND = "auto"
ONE_ROUND_MAX_ROWS = 2
# One rank's histograms of one core's rows in the one-round exchange buffer.
ALL_WORDS = NUM_SUBCORES * MAX_ROWS * NUM_BUCKETS
# Re-streaming: False, "tile" (grow the row buffer) or True (tile + compaction).
NO_RESTREAM = True
# The levels at which a streamed row's survivors are compacted, if they fit.
CMP_LEVELS = (1, 2)
TILE_WORDS = 128 * 1024
TILE_RESERVE = 16 * 1024
SPMEM_WORDS = 2 * 1024 * 1024
SPMEM_RESERVE = 4 * 1024
# Sends histogram counts as 16-bit pairs when every bound of a pass is <= this.
PACK16 = False
PACK16_MAX = 0xFFFF
HALF_BUCKETS = NUM_BUCKETS // 2


def _align16(x):
    return jnp.bitwise_and(x + (LANES - 1), jnp.int32(-LANES))


def _signal(sem, **device):
    pl.semaphore_signal(sem, 1, device_id=device, device_id_type=pl.DeviceIdType.MESH)


def _vector_body(
    scores_hbm,  # i32[b * n], this rank's parts of the rows
    lengths_hbm,  # i32[b + pad], this rank's part lengths
    bounds_hbm,  # i32[b + pad], the same on every rank, >= every rank's length
    after_hbm,  # i32[LANES], only orders this call after another
    out_hbm,  # i32[b * k]
    *,
    rows_vmem,
    hist_vmem,
    peer_vmem,
    glob_vmem,
    rowbuf_vmem,
    state_vmem,
    len_vmem,
    bnd_vmem,
    ctrl_vmem,
    state_smem,
    hist_shared,
    state_shared,
    ctrl_shared,
    v2s_sem,
    s2v_sem,
    row_sem,
    seg_sems,
    copy_sem,
    b: int,
    n: int,
    k: int,
    dcp_size: int,
    axis_name: str,
    max_1r: int,
    all_words: int,
    compact: bool,
    pack16: bool,
    pack_max: int,
    row_words: int,
    **_,
):
    del after_hbm
    core = lax.axis_index("core")
    sub = lax.axis_index("subcore")
    flat = core * NUM_SUBCORES + sub
    rank = lax.axis_index(axis_name)
    lane_iota = jnp.arange(LANES, dtype=jnp.int32)
    neg_inf_key = jnp.int32(KEY_NEG_INF)
    # One rank's histograms of this subcore's rows in peer_vmem.
    rank_words = MAX_ROWS * NUM_BUCKETS
    # Row (sub, j) belongs to rank `sub % dcp_size`. A block of hist_shared
    # holds one rank's histograms of one owner's rows.
    per_owner = NUM_SUBCORES // dcp_size
    my_owner = jnp.bitwise_and(sub, dcp_size - 1)
    my_slot = jnp.right_shift(sub, dcp_size.bit_length() - 1)
    block = per_owner * MAX_ROWS * NUM_BUCKETS
    state_block = per_owner * MAX_ROWS * STATE_WORDS
    # A streamed pass's survivors (keys, then columns) past the two segments.
    cand0 = 2 * SEG_WORDS
    cand_half = ((row_words - cand0) // 2) & -LANES

    def vec_at(ref, idx):
        return ref[pl.ds(idx, LANES)][0]

    def get(j, f):
        return state_smem[j * NF + f]

    def put(j, f, v):
        state_smem[j * NF + f] = v

    def monotone_key(bits):
        m = jnp.bitwise_and(jnp.right_shift(bits, 31), jnp.int32(MONO_MASK))
        return jnp.bitwise_xor(bits, m)

    def key_window(prefix, shift):
        """Scalar bounds of the key range ``key >> shift == prefix``."""
        lo = jnp.left_shift(prefix, shift)
        hi = jnp.bitwise_or(lo, jnp.left_shift(jnp.int32(1), shift) - 1)
        return jnp.maximum(lo, neg_inf_key + 1) - 1, jnp.maximum(hi, neg_inf_key)

    def hist_add(hist, bucket, mask):
        dup, last = plsc.scan_count(bucket, mask=mask)
        plsc.addupdate_scatter(hist, (bucket,), dup, mask=last)

    # Loop carries start from `jnp.asarray(0, jnp.int32)`, which is what
    # `jnp.int32(0)` calls, spelled so that type checkers see an Array and
    # match `plsc.parallel_loop`'s carry overload.
    def scan_glob():
        def body(i, carry):
            c = plsc.cumsum(glob_vmem[pl.ds(i * LANES, LANES)]) + carry
            glob_vmem[pl.ds(i * LANES, LANES)] = c
            return c[LANES - 1]

        return plsc.parallel_loop(
            0, NUM_BUCKETS // LANES, unroll=2, carry=jnp.asarray(0, jnp.int32)
        )(body)  # pytype: disable=bad-argument-type

    def find_bucket(thresh):
        def body(i, cnt):
            below = glob_vmem[pl.ds(i * LANES, LANES)] <= thresh
            return cnt + plsc.all_reduce_population_count(below)[0]

        return plsc.parallel_loop(
            0, NUM_BUCKETS // LANES, unroll=2, carry=jnp.asarray(0, jnp.int32)
        )(body)  # pytype: disable=bad-argument-type

    def post(ctrl):
        """Tells the scalar subcore `ctrl` (see `_scalar_body`) once all 16
        subcores are here."""

        @pl.when(sub == 0)
        def _():
            ctrl_vmem[...] = jnp.full((LANES,), ctrl, jnp.int32)
            pltpu.sync_copy(ctrl_vmem, ctrl_shared)

        _signal(v2s_sem, core=core)

    def handoff(first=False, ctrl=1):
        """Hands the staged data to the scalar subcore and waits for its round;
        the first round of a pass also tells it the pass goes on."""
        if first:
            post(ctrl)
        else:
            _signal(v2s_sem, core=core)
        pl.semaphore_wait(s2v_sem, 1)

    def copy_all(pairs):
        """Copies every (src, dst) ref pair, all in flight at once."""
        copies = [pltpu.make_async_copy(src, dst, copy_sem) for src, dst in pairs]
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()

    pltpu.sync_copy(lengths_hbm, len_vmem)
    pltpu.sync_copy(bounds_hbm, bnd_vmem)

    # Rows past the last one any rank sees anything of are padding everywhere.
    def last_live(i, m):
        bound = bnd_vmem[pl.ds(i * LANES, LANES)]
        live = jnp.where(bound > 0, i * LANES + lane_iota + 1, 0)
        return jnp.maximum(m, plsc.cummax(live)[LANES - 1])

    num_live = plsc.parallel_loop(
        0, (b + LANES - 1) // LANES, carry=jnp.asarray(0, jnp.int32)
    )(last_live)  # pytype: disable=bad-argument-type

    def mask_tail(part, length):
        """-inf past `length` in the last 16-word chunk of `part`."""
        tail = jnp.bitwise_and(length - 1, jnp.int32(-LANES))

        @pl.when(length > 0)
        def _():
            part[pl.ds(tail, LANES)] = jnp.where(
                tail + lane_iota < length,
                part[pl.ds(tail, LANES)],
                jnp.int32(BITS_NEG_INF),
            )

    # The stream of a pass whose rows do not fit, after the rows' states in
    # state_smem: the rows_vmem half the next segment lands in, and the row
    # whose first segment is already on its way there (-1: none).
    # S_COFF: the next free word of the pass's survivor buffers.
    S_HALF, S_NEXT, S_COFF = MAX_ROWS * NF, MAX_ROWS * NF + 1, MAX_ROWS * NF + 2

    def num_segments(j):
        return jnp.right_shift(
            get(j, F_LEN) + (SEG_WORDS - 1), SEG_WORDS.bit_length() - 1
        )

    def seg_copy(j, seg, half):
        words = _align16(jnp.minimum(get(j, F_LEN) - seg * SEG_WORDS, SEG_WORDS))
        src = pl.multiple_of(get(j, F_ROW) * n + seg * SEG_WORDS, LANES)
        return pltpu.make_async_copy(
            scores_hbm.at[pl.ds(src, words)],
            rows_vmem.at[pl.ds(pl.multiple_of(half * SEG_WORDS, LANES), words)],
            seg_sems.at[half],
        )

    def iota_cols(col0):
        return lambda i: col0 + i * LANES + lane_iota

    def compacted(j):
        return jnp.bool_(False) if not compact else get(j, F_CMP) != 0

    def for_segments(
        j,
        carry,
        *,
        stride,
        streamed,
        nrows,
        keys,
        body,
        then,
        live,
        use_cmp=False,
        self_cmp=None,
    ):
        """Folds `body(part, length, cols, carry)` over row `j`'s segments.

        `cols(i)` gives the columns of the i-th 16-word chunk of `part`. With
        `use_cmp`, a row whose survivors were compacted reads only those from
        tile memory; `self_cmp` says row `j` is being compacted now.

        A pass whose rows fit tile memory fetches each once into its slot, and
        the row is one segment: level 0 (`keys=False`) masks it and turns it
        into keys in its own loop, later levels and the emit reuse them. A pass
        whose rows do not fit streams every row again at every level, in
        SEG_WORDS segments double-buffered through the halves of rows_vmem;
        each is masked here, and turned into keys when `keys`. While a row's
        last segment is worked on, the first of row `j + 1` is fetched if
        `live(j + 1)`, or of row `then` after the last row (None: nothing
        follows; the next level may find it done, see `drain`).
        """
        length = get(j, F_LEN)

        def resident(_, c):
            part = rows_vmem.at[pl.ds(pl.multiple_of(j * stride, LANES), row_words)]
            if not keys:
                mask_tail(part, length)
            return body(part, length, iota_cols(0), c)

        cmp = compacted(j) if use_cmp else jnp.bool_(False)

        def from_survivors(_, c):
            off = pl.multiple_of(cand0 + get(j, F_COFF), LANES)
            part = rows_vmem.at[pl.ds(off, cand_half)]
            colpart = rows_vmem.at[
                pl.ds(pl.multiple_of(off + cand_half, LANES), cand_half)
            ]
            return body(
                part, get(j, F_CLEN), lambda i: colpart[pl.ds(i * LANES, LANES)], c
            )

        if compact and use_cmp:
            pl.when(cmp)(functools.partial(drain, j))
        segs = jnp.where(streamed & ~cmp, num_segments(j), 0)
        base = state_smem[S_HALF]

        @pl.when((segs > 0) & (state_smem[S_NEXT] != j))
        def _():
            seg_copy(j, 0, base).start()

        state_smem[S_NEXT] = jnp.int32(-1)
        within = j + 1 < nrows
        nxt = jnp.minimum(j + 1, nrows - 1)
        has_next = within & live(nxt)
        if then is not None:
            nxt = jnp.where(within, nxt, then)
            has_next = has_next | ~within
        has_next = has_next & (num_segments(nxt) > 0)
        if compact:
            has_next = has_next & ~compacted(nxt)
        if self_cmp is not None:
            has_next = has_next & ~((nxt == j) & self_cmp)

        def streamed_seg(seg, c):
            half = jnp.bitwise_and(base + seg, 1)
            seg_copy(j, seg, half).wait()

            @pl.when(seg + 1 < segs)
            def _():
                seg_copy(j, seg + 1, 1 - half).start()

            @pl.when((seg + 1 == segs) & has_next)
            def _():
                seg_copy(nxt, 0, 1 - half).start()
                state_smem[S_NEXT] = nxt

            part = rows_vmem.at[
                pl.ds(pl.multiple_of(half * SEG_WORDS, LANES), SEG_WORDS)
            ]
            seg_len = jnp.minimum(length - seg * SEG_WORDS, SEG_WORDS)
            mask_tail(part, seg_len)
            if keys:

                def to_keys(i):
                    part[pl.ds(i * LANES, LANES)] = monotone_key(
                        part[pl.ds(i * LANES, LANES)]
                    )

                plsc.parallel_loop(
                    0, jnp.right_shift(seg_len + LANES - 1, 4), unroll=8
                )(to_keys)
            return body(part, seg_len, iota_cols(seg * SEG_WORDS), c)

        carry = lax.fori_loop(0, jnp.where(streamed, 0, 1), resident, carry)
        if use_cmp and compact:
            carry = lax.fori_loop(0, jnp.where(cmp, 1, 0), from_survivors, carry)
        carry = lax.fori_loop(0, segs, streamed_seg, carry)
        state_smem[S_HALF] = jnp.bitwise_and(base + segs, 1)
        return carry

    def drain(j):
        """Waits out row `j`'s prefetched first segment if `j` is skipped."""

        @pl.when(state_smem[S_NEXT] == j)
        def _():
            seg_copy(j, 0, state_smem[S_HALF]).wait()
            state_smem[S_HALF] = 1 - state_smem[S_HALF]
            state_smem[S_NEXT] = jnp.int32(-1)

    def hist_row(j, carry, *, level, stride, streamed, nrows):
        """Level `level`'s histogram of row `j`."""
        hist = hist_vmem.at[pl.ds(pl.multiple_of(j * NUM_BUCKETS, LANES), NUM_BUCKETS)]
        # At CMP_LEVELS a streamed row compacts its survivors if they fit.
        cmp_here = compact and level in CMP_LEVELS
        to_cmp = jnp.bool_(False)
        if compact and 0 < level <= max(CMP_LEVELS):

            def decide(_, t):
                """Counts the survivors and reserves their room if they fit."""
                if level == 1:
                    bkt = get(j, F_PREFIX) + NUM_BUCKETS // 2
                    above = jnp.int32(0)
                else:
                    bkt = jnp.bitwise_and(get(j, F_PREFIX), NUM_BUCKETS - 1)
                    above = get(j, F_ABOVE)

                def count_from(i, s):
                    h = hist[pl.ds(i * LANES, LANES)]
                    idx = i * LANES + lane_iota
                    return (
                        s[0] + jnp.where(idx >= bkt, h, 0),
                        s[1] + jnp.where(idx > bkt, h, 0),
                    )

                zeros = jnp.zeros((LANES,), jnp.int32)
                ge, gt = plsc.parallel_loop(
                    0, NUM_BUCKETS // LANES, carry=(zeros, zeros)
                )(count_from)
                ge, gt = plsc.cumsum(ge)[LANES - 1], plsc.cumsum(gt)[LANES - 1]
                put(j, F_ABOVE, above + gt)
                coff = state_smem[S_COFF]
                need = _align16(above + ge) + LANES
                fits = (
                    (get(j, F_DONE) == 0)
                    & ~compacted(j)
                    & (need <= cand_half - coff)
                    & cmp_here
                )

                @pl.when(fits)
                def _():
                    put(j, F_COFF, coff)
                    state_smem[S_COFF] = coff + need

                return fits.astype(jnp.int32)

            # Only a streamed pass compacts.
            to_cmp = (
                lax.fori_loop(0, jnp.where(streamed, 1, 0), decide, jnp.int32(0)) != 0
            )

        def zero(i):
            hist[pl.ds(i * LANES, LANES)] = jnp.zeros((LANES,), jnp.int32)

        plsc.parallel_loop(0, NUM_BUCKETS // LANES, unroll=8)(zero)
        if level == 0:

            def segment(part, length, cols, c):
                def body(i):
                    key = monotone_key(part[pl.ds(i * LANES, LANES)])
                    part[pl.ds(i * LANES, LANES)] = key
                    bucket = jnp.right_shift(key, L0_SHIFT) + NUM_BUCKETS // 2
                    hist_add(hist, bucket, key > neg_inf_key)

                chunks = jnp.right_shift(length + (LANES - 1), 4)
                plsc.parallel_loop(0, chunks, unroll=SCAN_UNROLL)(body)
                return c

            live = jnp.bool_(True)
        else:
            shift, mask = REFINEMENT_LEVELS[level - 1]
            lo_excl, hi_incl = key_window(get(j, F_PREFIX), get(j, F_SHIFT))

            def segment(part, length, cols, c):
                def body(i):
                    key = part[pl.ds(i * LANES, LANES)]
                    match = (key > lo_excl) & (key <= hi_incl)
                    bucket = jnp.bitwise_and(jnp.right_shift(key, shift), mask)
                    hist_add(hist, bucket, match)

                chunks = jnp.right_shift(length + (LANES - 1), 4)
                if not cmp_here:
                    plsc.parallel_loop(0, chunks, unroll=SCAN_UNROLL)(body)
                    return c

                # Also keep every key from the window up, with its column.
                def body_cmp(i, woff):
                    body(i)
                    key = part[pl.ds(i * LANES, LANES)]
                    keep = key > lo_excl
                    plsc.store_compressed(
                        rows_vmem.at[pl.ds(woff, LANES)], key, mask=keep
                    )
                    plsc.store_compressed(
                        rows_vmem.at[pl.ds(woff + cand_half, LANES)], cols(i), mask=keep
                    )
                    return woff + plsc.all_reduce_population_count(keep)[0]

                def plain(_, w):
                    plsc.parallel_loop(0, chunks, unroll=SCAN_UNROLL)(body)
                    return w

                def with_cmp(_, w):
                    return plsc.parallel_loop(0, chunks, unroll=4, carry=w)(body_cmp)

                c = lax.fori_loop(0, jnp.where(to_cmp, 0, 1), plain, c)
                return lax.fori_loop(0, jnp.where(to_cmp, 1, 0), with_cmp, c)

            live = get(j, F_DONE) == 0

        def row_live(i):
            return jnp.bool_(True) if level == 0 else get(i, F_DONE) == 0

        @pl.when(live)
        def _():
            start = cand0 + get(j, F_COFF) if cmp_here else 0
            end = for_segments(
                j,
                jnp.asarray(start, jnp.int32),
                stride=stride,
                streamed=streamed,
                nrows=nrows,
                keys=level > 0,
                body=segment,
                then=0,  # the next level, or the emit, starts at row 0
                live=row_live,
                use_cmp=level > 1,
                self_cmp=to_cmp if cmp_here else None,
            )
            if cmp_here:

                @pl.when(to_cmp)
                def _():
                    put(j, F_CMP, jnp.int32(1))
                    put(j, F_CLEN, end - start)
                    # -inf after the survivors, for the last chunk's tail.
                    plsc.store_compressed(
                        rows_vmem.at[pl.ds(end, LANES)],
                        jnp.full((LANES,), KEY_NEG_INF, jnp.int32),
                        mask=lane_iota >= 0,
                    )

        pl.when(~live)(functools.partial(drain, j))
        return carry

    # Packed 16-bit count pairs sit past the histograms in hist_vmem/peer_vmem.
    pk_src = MAX_ROWS * NUM_BUCKETS
    pk_dst = dcp_size * rank_words

    def pack_hists(nrows):
        """Buckets i and i + 128 of each row as one word, after hist_vmem's."""

        def body(i):
            src = jnp.right_shift(i, 3) * NUM_BUCKETS + jnp.bitwise_and(i, 7) * LANES
            lo = hist_vmem[pl.ds(src, LANES)]
            hi = hist_vmem[pl.ds(src + HALF_BUCKETS, LANES)]
            hist_vmem[pl.ds(pk_src + i * LANES, LANES)] = jnp.bitwise_or(
                lo, jnp.left_shift(hi, 16)
            )

        plsc.parallel_loop(0, nrows * (HALF_BUCKETS // LANES), unroll=4)(body)

    def unpack_peers(nrows):
        """Every rank's pairs back to 256 counts per row in its peer_vmem slot."""
        for d in range(dcp_size):

            def body(i):
                w = peer_vmem[pl.ds(pk_dst + d * (rank_words // 2) + i * LANES, LANES)]
                dst = (
                    d * rank_words
                    + jnp.right_shift(i, 3) * NUM_BUCKETS
                    + jnp.bitwise_and(i, 7) * LANES
                )
                peer_vmem[pl.ds(dst, LANES)] = jnp.bitwise_and(w, 0xFFFF)
                peer_vmem[pl.ds(dst + HALF_BUCKETS, LANES)] = jnp.bitwise_and(
                    jnp.right_shift(w, 16), 0xFFFF
                )

            plsc.parallel_loop(0, nrows * (HALF_BUCKETS // LANES), unroll=4)(body)

    def wire(nrows, packed):
        """Words per row sent, their offset in hist_vmem, and peer_vmem's
        landing offset of rank d."""
        if not pack16:
            return nrows * NUM_BUCKETS, 0, lambda d: d * rank_words
        words = nrows * jnp.where(packed, HALF_BUCKETS, NUM_BUCKETS)
        src = jnp.where(packed, pk_src, 0)
        return (
            words,
            src,
            lambda d: jnp.where(packed, pk_dst + d * (rank_words // 2), d * rank_words),
        )

    def land(src_refs, nrows, words, dst, packed):
        """Copies each rank's sent words to peer_vmem and unpacks them."""
        copy_all(
            (src, peer_vmem.at[pl.ds(pl.multiple_of(dst(d), LANES), words)])
            for d, src in enumerate(src_refs)
        )
        if pack16:
            pl.when(packed)(functools.partial(unpack_peers, nrows))

    def select(j, level, publish):
        """Sums row `j`'s histograms over the ranks (in peer_vmem), picks the
        bucket, hands `publish(d, fields, bucket)` each rank's new state, and
        returns this rank's."""
        hb = j * NUM_BUCKETS

        def sum_ranks(i):
            acc = peer_vmem[pl.ds(hb + i * LANES, LANES)]
            for d in range(1, dcp_size):
                acc = acc + peer_vmem[pl.ds(d * rank_words + hb + i * LANES, LANES)]
            glob_vmem[pl.ds(i * LANES, LANES)] = acc

        plsc.parallel_loop(0, NUM_BUCKETS // LANES)(sum_ranks)
        total = scan_glob()
        quota = jnp.int32(k) if level == 0 else get(j, F_QUOTA)
        bucket = find_bucket(total - quota)
        cum_at = vec_at(glob_vmem, bucket)
        cum_lo = jnp.where(bucket > 0, vec_at(glob_vmem, jnp.maximum(bucket - 1, 0)), 0)
        cnt = cum_at - cum_lo
        quota = quota - (total - cum_at)
        fields = {F_QUOTA: quota}
        if level == 0:
            keep = total <= k
            fields[F_KEEP] = keep.astype(jnp.int32)
            fields[F_PREFIX] = bucket - NUM_BUCKETS // 2
            fields[F_SHIFT] = jnp.int32(L0_SHIFT)
            done = keep | (quota == cnt)
        else:
            fields[F_KEEP] = get(j, F_KEEP)
            fields[F_PREFIX] = jnp.left_shift(get(j, F_PREFIX), 8) | bucket
            fields[F_SHIFT] = jnp.int32(REFINEMENT_LEVELS[level - 1][0])
            done = (quota == cnt) | (quota == 0)
        fields[F_DONE] = done.astype(jnp.int32)
        # Each rank's entries in the bucket, and the lower ranks'.
        before = jnp.int32(0)
        mine_self = before_self = jnp.int32(0)
        for d in range(dcp_size):
            mine = vec_at(peer_vmem, d * rank_words + hb + bucket)
            if publish is not None:
                publish(d, {**fields, F_MINE: mine, F_BEFORE: before}, bucket)
            mine_self = jnp.where(d == rank, mine, mine_self)
            before_self = jnp.where(d == rank, before, before_self)
            before = before + mine
        return {**fields, F_MINE: mine_self, F_BEFORE: before_self}

    def exchange(level, nrows, ctrl, packed):
        """Sums each row's histograms over the ranks and picks its bucket.

        The owner exchange (`one` false): hist_shared holds [out | in] x rank
        x block: `out[d]` this rank's histograms of rank d's rows, `in[d]`
        rank d's histograms of this rank's rows; a block holds MAX_ROWS rows
        per owned subcore and moves whole (a remote copy between different
        offsets needs a static size). The owner of each row sums and picks,
        and sends back the state through state_shared, laid out alike.

        The one-round exchange (`one` true): every rank gets every rank's
        histograms and picks every row itself. hist_shared holds parity x
        rank x all_words: rank d's histograms of this core's rows, `nrows`
        per subcore and subcore-major. The two parities alternate by level,
        so a peer one level ahead never overwrites histograms still being
        read. Packed, a row's histogram is 128 words on the wire.
        """
        if pack16:
            pl.when(packed)(functools.partial(pack_hists, nrows))
        if max_1r >= MAX_ROWS:
            one = jnp.bool_(True)
        elif max_1r <= 0:
            one = jnp.bool_(False)
        else:
            one = nrows <= max_1r
        has_owner, has_one = max_1r < MAX_ROWS, max_1r > 0
        words, src, dst_of = wire(nrows, packed)
        slot = (
            rank_words if not pack16 else jnp.where(packed, rank_words // 2, rank_words)
        )
        base = (level % 2) * dcp_size * all_words + sub * words
        # Where this subcore's histograms go, and where rank d's are read.
        own_dst = (
            jnp.where(my_owner == rank, dcp_size + my_owner, my_owner) * block
            + my_slot * slot
        )
        dst = jnp.where(one, base + rank * all_words, own_dst)

        def read_at(d):
            return jnp.where(
                one, base + d * all_words, (dcp_size + d) * block + my_slot * slot
            )

        state_words = nrows * STATE_WORDS
        state_rows = MAX_ROWS * STATE_WORDS  # one subcore's slot of a state block
        pltpu.sync_copy(
            hist_vmem.at[pl.ds(pl.multiple_of(src, LANES), words)],
            hist_shared.at[pl.ds(pl.multiple_of(dst, LANES), words)],
        )
        handoff(first=level == 0, ctrl=ctrl)

        # Every rank picks its rows in one round, only the owner in two.
        @pl.when(one | (my_owner == rank))
        def _():
            land(
                [
                    hist_shared.at[pl.ds(pl.multiple_of(read_at(d), LANES), words)]
                    for d in range(dcp_size)
                ],
                nrows,
                words,
                dst_of,
                packed,
            )

            def row(j, c):
                def publish(d, fields, bucket):
                    del bucket
                    v = jnp.zeros((LANES,), jnp.int32)
                    for i, f in enumerate(STATE_FIELDS):
                        v = jnp.where(lane_iota == i, fields[f], v)
                    v = jnp.where(lane_iota == len(STATE_FIELDS), 1, v)  # updated
                    state_vmem[pl.ds((d * MAX_ROWS + j) * STATE_WORDS, LANES)] = v

                live = jnp.bool_(True) if level == 0 else get(j, F_DONE) == 0

                # Owner stages every rank's state; one round keeps only this rank's.
                @pl.when(live)
                def _():
                    fields = select(j, level, publish if has_owner else None)
                    if has_one:

                        @pl.when(one)
                        def _():
                            for f, v in fields.items():
                                put(j, f, v)

                if has_owner:

                    @pl.when(~live)
                    def _():
                        for d in range(dcp_size):
                            state_vmem[
                                pl.ds((d * MAX_ROWS + j) * STATE_WORDS, LANES)
                            ] = jnp.zeros((LANES,), jnp.int32)

                return c

            lax.fori_loop(0, nrows, row, 0)
            if has_owner:

                @pl.when(~one)
                def _():
                    copy_all(
                        (
                            state_vmem.at[pl.ds(d * state_rows, state_words)],
                            state_shared.at[
                                pl.ds(
                                    pl.multiple_of(
                                        jnp.where(d == rank, dcp_size + d, d)
                                        * state_block
                                        + my_slot * state_rows,
                                        LANES,
                                    ),
                                    state_words,
                                )
                            ],
                        )
                        for d in range(dcp_size)
                    )

        if not has_owner:
            return

        # Round 2: the owners' states back to every rank.
        @pl.when(~one)
        def _():
            handoff()
            src = (dcp_size + my_owner) * state_block + my_slot * state_rows
            pltpu.sync_copy(
                state_shared.at[pl.ds(pl.multiple_of(src, LANES), state_words)],
                state_vmem.at[pl.ds(0, state_words)],
            )

            def unpack(j, c):
                v = state_vmem[pl.ds(j * STATE_WORDS, LANES)]

                @pl.when(v[len(STATE_FIELDS)] != 0)
                def _():
                    for i, f in enumerate(STATE_FIELDS):
                        put(j, f, v[i])

                return c

            lax.fori_loop(0, nrows, unpack, 0)

    def fill_row():
        def fill(i):
            rowbuf_vmem[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), -1, jnp.int32)

        plsc.parallel_loop(0, k // LANES, unroll=4)(fill)

    def write_row(row):
        pltpu.sync_copy(
            rowbuf_vmem.at[pl.ds(0, k)],
            out_hbm.at[pl.ds(pl.multiple_of(row * k, LANES), k)],
        )

    def emit_row(j, carry, *, stride, streamed, nrows):
        """This rank's winners of row `j`, written out."""
        fill_row()
        lo_excl, hi_incl = key_window(get(j, F_PREFIX), get(j, F_SHIFT))
        mine = get(j, F_MINE)
        my_take = jnp.clip(get(j, F_QUOTA) - get(j, F_BEFORE), 0, mine)
        take_all_ties = (get(j, F_KEEP) != 0) | (my_take >= mine)
        gate = jnp.where(take_all_ties, lo_excl, hi_incl)
        whole_bucket = take_all_ties | (my_take <= 0)

        def segment(part, length, cols, c: tuple[jax.Array, jax.Array]):
            chunks = jnp.right_shift(length + (LANES - 1), 4)

            def emit(i, woff, sel):
                plsc.store_compressed(
                    rowbuf_vmem.at[pl.ds(woff, LANES)],
                    cols(i),
                    mask=sel,
                )
                return woff + plsc.all_reduce_population_count(sel)[0]

            def emit_whole(i, woff):
                return emit(i, woff, part[pl.ds(i * LANES, LANES)] > gate)

            woff = plsc.parallel_loop(
                0, jnp.where(whole_bucket, chunks, 0), unroll=SCAN_UNROLL, carry=c[0]
            )(emit_whole)  # pytype: disable=bad-argument-type

            def emit_ranked(i, cr):
                woff, rank_in_bucket = cr
                key = part[pl.ds(i * LANES, LANES)]
                in_set = key > lo_excl
                strict = key > hi_incl
                tie_rank = (
                    plsc.cumsum(in_set.astype(jnp.int32) - strict.astype(jnp.int32))
                    + rank_in_bucket
                )
                sel = strict | (in_set & (tie_rank <= my_take))
                return emit(i, woff, sel), tie_rank[LANES - 1]

            return plsc.parallel_loop(
                0, jnp.where(whole_bucket, 0, chunks), unroll=4, carry=(woff, c[1])
            )(emit_ranked)

        for_segments(
            j,
            (jnp.int32(0), jnp.int32(0)),
            stride=stride,
            streamed=streamed,
            nrows=nrows,
            keys=True,
            body=segment,
            then=None,
            live=lambda i: jnp.bool_(True),
            use_cmp=True,
        )
        row = get(j, F_ROW)
        pl.when(row < b)(functools.partial(write_row, row))
        return carry

    def pass_body(row0):
        """One pass: `nrows` rows per subcore, four levels."""

        # Rows per subcore, from the longest row part (on any rank) among the
        # next ROWS_PER_PASS * MAX_ROWS rows: every rank picks the same.
        def longest(i, m):
            return jnp.maximum(
                m, plsc.cummax(bnd_vmem[pl.ds(row0 + i * LANES, LANES)])[LANES - 1]
            )

        stride = plsc.parallel_loop(
            0, ROWS_PER_PASS * MAX_ROWS // LANES, carry=jnp.asarray(0, jnp.int32)
        )(longest)  # pytype: disable=bad-argument-type
        # Up to MAX_ROWS rows per subcore, as the live rows left need. Every
        # pass costs eight latency-bound rounds, so a pass whose rows do not
        # fit tile memory still takes that many and streams them.
        left = num_live - row0
        nrows = jnp.int32(1)
        for r in range(2, MAX_ROWS + 1):
            nrows = nrows + ((r - 1) * ROWS_PER_PASS < left).astype(jnp.int32)
        # Every count of the pass fits 16 bits if every bound does.
        packed = (stride <= pack_max) if pack16 else jnp.bool_(False)
        ctrl = nrows + LANES * packed.astype(jnp.int32)
        stride = jnp.maximum(_align16(stride), LANES)
        streamed = nrows * stride > row_words
        state_smem[S_HALF] = jnp.int32(0)
        state_smem[S_NEXT] = jnp.int32(-1)
        state_smem[S_COFF] = jnp.int32(0)

        def fetch(j, c):
            row = row0 + flat + ROWS_PER_PASS * j
            length = jnp.minimum(vec_at(len_vmem, row), n)
            put(j, F_ROW, row)
            put(j, F_LEN, length)
            put(j, F_CMP, jnp.int32(0))
            words = _align16(length)

            @pl.when((words > 0) & ~streamed)
            def _():
                pltpu.make_async_copy(
                    scores_hbm.at[pl.ds(pl.multiple_of(row * n, LANES), words)],
                    rows_vmem.at[pl.ds(pl.multiple_of(j * stride, LANES), words)],
                    row_sem,
                ).start()

            return c

        def fetch_wait(j, c):
            words = _align16(get(j, F_LEN))

            @pl.when((words > 0) & ~streamed)
            def _():
                pltpu.make_async_copy(
                    scores_hbm.at[pl.ds(0, words)],
                    rows_vmem.at[pl.ds(0, words)],
                    row_sem,
                ).wait()

            return c

        lax.fori_loop(0, nrows, fetch, 0)
        lax.fori_loop(0, nrows, fetch_wait, 0)
        for level in range(NUM_LEVELS):
            lax.fori_loop(
                0,
                nrows,
                functools.partial(
                    hist_row, level=level, stride=stride, streamed=streamed, nrows=nrows
                ),
                0,
            )
            exchange(level, nrows, ctrl, packed)
        lax.fori_loop(
            0,
            nrows,
            functools.partial(emit_row, stride=stride, streamed=streamed, nrows=nrows),
            0,
        )
        return row0 + ROWS_PER_PASS * nrows

    row_end = lax.while_loop(lambda r: r < num_live, pass_body, jnp.int32(0))

    post(0)  # nothing more to exchange: let the scalar subcore finish

    # The padding rows after the last pass.
    fill_row()
    first_pad = row_end + flat

    def write_pad(i, c):
        write_row(first_pad + ROWS_PER_PASS * i)
        return c

    num_pad = jnp.right_shift(b - first_pad + (ROWS_PER_PASS - 1), 5)
    lax.fori_loop(0, jnp.maximum(num_pad, 0), write_pad, 0)


def _scalar_body(
    scores_hbm,
    lengths_hbm,
    bounds_hbm,
    after_hbm,
    out_hbm,
    *,
    hist_shared,
    state_shared,
    ctrl_shared,
    ctrl_smem,
    v2s_sem,
    s2v_sem,
    send_sems,
    recv_sems,
    dcp_size: int,
    axis_name: str,
    max_1r: int,
    all_words: int,
    pack16: bool,
    **_,
):
    """Runs the exchange rounds of this core with the same core of the peers.

    A round: wait for the 16 vector subcores to stage their data, send each
    peer its part, wait for every peer's part to arrive, release the subcores.
    A pass runs NUM_LEVELS levels of two rounds, or of one with `one_round`.
    """
    del scores_hbm, lengths_hbm, bounds_hbm, after_hbm, out_hbm
    rank = lax.axis_index(axis_name)
    core = lax.axis_index("core")

    def staged():
        pl.semaphore_wait(v2s_sem, NUM_SUBCORES)

    def ctrl():
        """Waits for the vector subcores' next `post`."""
        staged()
        pltpu.sync_copy(ctrl_shared, ctrl_smem)
        return ctrl_smem[0]

    def send(shared, src_block, dst_block, words, sem):
        """Block `src_block(d)` of `shared` to block `dst_block(d)` of peer d's.

        The peers' copies of a round share one semaphore pair (`sem`: one per
        kind of round), as they all have `words`: no peer starts its next round
        of a kind before every rank has finished this one."""
        copies = []
        for i in range(1, dcp_size):
            d = lax.rem(rank + i, dcp_size)
            copy = pltpu.make_async_remote_copy(
                src_ref=shared.at[pl.ds(src_block(d), words)],
                dst_ref=shared.at[pl.ds(dst_block(d), words)],
                send_sem=send_sems.at[sem],
                recv_sem=recv_sems.at[sem],
                # Other mesh axes (the DCP mesh's TP lanes) stay this device's.
                device_id={axis_name: d},
                device_id_type=pl.DeviceIdType.MESH,
            )
            copy.start()
            copies.append(copy)
        for copy in copies:
            copy.wait_send()
        for copy in copies:
            copy.wait_recv()
        for i in range(NUM_SUBCORES):
            _signal(s2v_sem, core=core, subcore=i)

    per_owner = NUM_SUBCORES // dcp_size
    block = per_owner * MAX_ROWS * NUM_BUCKETS
    state_block = per_owner * MAX_ROWS * LANES

    def level_rounds(level, pk):
        if level > 0:
            staged()
        # out[d] (this rank's histograms of d's rows) to in[rank] at d.
        for p in (False, True) if pack16 else (False,):
            run = functools.partial(
                send,
                hist_shared,
                lambda d: d * block,
                lambda d: (dcp_size + rank) * block,
                block // 2 if p else block,  # packed: the slots' first halves
                0,
            )
            if pack16:
                pl.when(pk == int(p))(run)
            else:
                run()
        staged()
        send(
            state_shared,
            lambda d: d * state_block,
            lambda d: (dcp_size + rank) * state_block,
            state_block,
            1,
        )

    def level_round(level, nrows, pk):
        """This rank's histograms (parity, rank) to the same block at every peer."""
        par = level % 2
        if level > 0:
            staged()
        src = par * dcp_size * all_words + rank * all_words
        # A remote copy needs a static size: one variant per rows per subcore.
        for p in (False, True) if pack16 else (False,):
            per_row = HALF_BUCKETS if p else NUM_BUCKETS
            for r in range(1, MAX_ROWS + 1):
                pl.when((nrows == r) & ((pk == int(p)) if pack16 else True))(
                    functools.partial(
                        send,
                        hist_shared,
                        lambda d: src,
                        lambda d: src,
                        r * NUM_SUBCORES * per_row,
                        par,
                    )
                )

    # Vectors post rows/subcore (+16 if packed) at each pass start, 0 at the end.
    def pass_rounds(_):
        more = ctrl()
        nrows = jnp.bitwise_and(more, LANES - 1)
        pk = jnp.right_shift(more, 4)

        @pl.when(more > 0)
        def _():
            for level in range(NUM_LEVELS):
                if max_1r >= MAX_ROWS:
                    level_round(level, nrows, pk)
                elif max_1r <= 0:
                    level_rounds(level, pk)
                else:
                    pl.when(nrows <= max_1r)(
                        functools.partial(level_round, level, nrows, pk)
                    )
                    pl.when(nrows > max_1r)(functools.partial(level_rounds, level, pk))

        return more

    lax.while_loop(lambda more: more > 0, pass_rounds, jnp.int32(1))


def sparsecore_topk_dcp(
    scores: jax.Array,  # f32[b, n] or i32[b, n] of f32 bits: this rank's parts
    k: int,
    row_lengths: jax.Array,  # i32[b]: this rank's part lengths
    row_bounds: jax.Array,  # i32[b]: equal on every rank, >= every rank's lengths
    *,
    dcp_size: int,
    axis_name: str,
    scheduling_group_id: int | None = None,
    after: jax.Array | None = None,
    one_round: bool | str | None = None,
    one_round_max_rows: int | None = None,
    no_restream: bool | str | None = None,
    pack16: bool | None = None,
) -> jax.Array:  # i32[b, k]
    """This rank's share of the exact top-k of rows split over `axis_name`.

    Call inside a `shard_map` over `axis_name`, on every rank with the same `b`,
    `n` and `row_bounds`. Each rank's output row lists the columns of its own
    `scores` row that are in the top-k of the row taken over all ranks, packed
    into a prefix and `-1` padded; ties at the k-th score go to the lower
    ranks. `-inf` scores and columns at or past `row_lengths` are never
    selected. NaNs are unsupported.

    Every rank must run its calls in the same order, since each waits for its
    peers inside the kernel. XLA keeps SparseCore calls that depend on each
    other in program order, but may run independent ones in a different order
    on different ranks, which deadlocks: chain such calls by passing the
    previous call's output as `after` (any array; only its first `LANES`
    words are read as a dependency, never used). `one_round` picks the
    exchange (None: `ONE_ROUND`; "auto": one round for passes of at most
    `one_round_max_rows` rows per subcore), `no_restream` and `pack16` the
    tile use and count width (None: `NO_RESTREAM`, `PACK16`); all give the
    same output.
    """
    b, n = scores.shape
    if n % LANES:
        raise ValueError(f"{n=} must be a multiple of {LANES}")
    if k % LANES or not 0 < k <= 4096:
        raise ValueError(f"{k=} must be a positive multiple of {LANES}, at most 4096")
    if dcp_size < 2 or NUM_SUBCORES % dcp_size:
        raise ValueError(f"{dcp_size=} must be a power of two in [2, {NUM_SUBCORES}]")
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None or (sc.num_cores, sc.num_subcores) != (NUM_CORES, NUM_SUBCORES):
        raise NotImplementedError(f"needs {NUM_CORES}x{NUM_SUBCORES} SparseCores")

    words = (
        scores
        if scores.dtype == jnp.int32
        else lax.bitcast_convert_type(scores, jnp.int32)
    )
    # A pass reads lengths and bounds up to ROWS_PER_PASS * MAX_ROWS rows on.
    pad = ROWS_PER_PASS * MAX_ROWS + LANES
    lengths = jnp.pad(row_lengths.astype(jnp.int32), (0, pad))
    bounds = jnp.pad(row_bounds.astype(jnp.int32), (0, pad))
    if after is None:
        after = jnp.zeros((LANES,), jnp.int32)
    else:
        after = after.reshape(-1)[:LANES].astype(jnp.int32)

    if one_round is None:
        one_round = ONE_ROUND
    if one_round_max_rows is None:
        one_round_max_rows = ONE_ROUND_MAX_ROWS
    if no_restream is None:
        no_restream = NO_RESTREAM
    if pack16 is None:
        pack16 = PACK16
    if one_round == "auto":
        max_1r = max(0, min(one_round_max_rows, MAX_ROWS))
    else:
        max_1r = MAX_ROWS if one_round else 0
    # One round: both parities of all ranks' max_1r rows; owner needs 2*ALL_WORDS.
    all_words = NUM_SUBCORES * max_1r * NUM_BUCKETS
    hist_words = max(2 * dcp_size * all_words, 2 * ALL_WORDS)
    state_words = 2 * NUM_SUBCORES * MAX_ROWS * LANES
    # Row buffer: ROW_WORDS, or with `no_restream` the tile memory left in SPMEM.
    pk_words = HALF_BUCKETS if pack16 else 0
    row_words = ROW_WORDS
    if no_restream:
        tile = min(
            TILE_WORDS - TILE_RESERVE,
            (SPMEM_WORDS - hist_words - state_words - LANES) // NUM_SUBCORES
            - SPMEM_RESERVE,
        )
        other = (
            MAX_ROWS * (NUM_BUCKETS + pk_words) * (1 + dcp_size)
            + NUM_BUCKETS
            + k
            + dcp_size * MAX_ROWS * STATE_WORDS
            + 2 * (b + pad)
            + 4 * LANES
        )
        row_words = max(ROW_WORDS, (tile - other) & -1024)
    compact = no_restream is True and row_words - 2 * SEG_WORDS >= 4096
    vmesh = plsc.VectorSubcoreMesh(
        core_axis_name="core", subcore_axis_name="subcore", num_cores=NUM_CORES
    )
    smesh = plsc.ScalarSubcoreMesh(axis_name="core", num_cores=NUM_CORES)
    vmem = pallas_core.CoreMemorySpace(pltpu.MemorySpace.VMEM, vmesh)
    scalar_dma_sems = pallas_core.CoreMemorySpace(pltpu.MemorySpace.SEMAPHORE, smesh)(
        (2,), pltpu.SemaphoreType.DMA.dtype
    )
    scratch_types = dict(
        rows_vmem=vmem((row_words,), jnp.int32),
        hist_vmem=vmem((MAX_ROWS * (NUM_BUCKETS + pk_words),), jnp.int32),
        peer_vmem=vmem((dcp_size * MAX_ROWS * (NUM_BUCKETS + pk_words),), jnp.int32),
        glob_vmem=vmem((NUM_BUCKETS + LANES,), jnp.int32),
        rowbuf_vmem=vmem((k + LANES,), jnp.int32),
        state_vmem=vmem((dcp_size * MAX_ROWS * STATE_WORDS,), jnp.int32),
        len_vmem=vmem((b + pad,), jnp.int32),
        bnd_vmem=vmem((b + pad,), jnp.int32),
        ctrl_vmem=vmem((LANES,), jnp.int32),
        state_smem=pallas_core.CoreMemorySpace(pltpu.MemorySpace.SMEM, vmesh)(
            (MAX_ROWS * NF + 3,), jnp.int32
        ),
        hist_shared=pltpu.VMEM_SHARED((hist_words,), jnp.int32),
        state_shared=pltpu.VMEM_SHARED((state_words,), jnp.int32),
        ctrl_shared=pltpu.VMEM_SHARED((LANES,), jnp.int32),
        ctrl_smem=pallas_core.CoreMemorySpace(pltpu.MemorySpace.SMEM, smesh)(
            (LANES,), jnp.int32
        ),
        v2s_sem=pltpu.SemaphoreType.REGULAR @ smesh,
        s2v_sem=pltpu.SemaphoreType.REGULAR @ vmesh,
        send_sems=scalar_dma_sems,
        recv_sems=scalar_dma_sems,
        row_sem=pltpu.SemaphoreType.DMA @ vmesh,
        seg_sems=pallas_core.CoreMemorySpace(pltpu.MemorySpace.SEMAPHORE, vmesh)(
            (2,), pltpu.SemaphoreType.DMA.dtype
        ),
        copy_sem=pltpu.SemaphoreType.DMA @ vmesh,
    )
    params = dict(
        b=b,
        n=n,
        k=k,
        dcp_size=dcp_size,
        axis_name=axis_name,
        max_1r=max_1r,
        all_words=all_words,
        compact=compact,
        pack16=bool(pack16),
        pack_max=PACK16_MAX,
        row_words=row_words,
    )
    tag = "_1r" if max_1r >= MAX_ROWS else f"_1r{max_1r}" if max_1r else ""
    tag += f"_rw{row_words // 1024}" + ("c" if compact else "") if no_restream else ""
    tag += "_p16" if pack16 else ""
    out = pl.kernel(
        [
            functools.partial(_vector_body, **params),
            functools.partial(_scalar_body, **params),
        ],
        out_type=jax.ShapeDtypeStruct((b * k,), jnp.int32),
        mesh=[vmesh, smesh],
        scratch_types=scratch_types,
        # No collective_id: XLA compiles it to a barrier semaphore it also
        # gives the TensorCore collectives this asynchronous call overlaps.
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            needs_layout_passes=False,
        ),
        name=f"sc_topk_dcp_b{b}_n{n}_k{k}_d{dcp_size}{tag}",
        # pyrefly: ignore[bad-argument-count]
    )(words.reshape(-1), lengths, bounds, after)
    if scheduling_group_id is not None:
        out = xla_metadata.set_xla_metadata(
            out, _scheduling_group_id=scheduling_group_id
        )
    # pl.kernel and set_xla_metadata are untyped upstream; both return the
    # out_type array here.
    return cast(jax.Array, out).reshape(b, k)
