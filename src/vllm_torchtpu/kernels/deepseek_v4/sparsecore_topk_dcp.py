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
   memory (what the other scratch leaves of it) are fetched once; longer ones
   are streamed from HBM again at every level, so the pass still exchanges
   all its rows at once.
2. The scalar subcore of each SparseCore exchanges its core's histograms with
   the same core on the other ranks (remote DMA into `VMEM_SHARED`), the
   ranks' histograms of a row are summed, and the bucket is picked. Four
   levels of 8 bits resolve the k-th score exactly. Once no rank holds more
   than `CAND_WORDS` of a row's entries in the picked bucket, the next level
   exchanges those entries' keys instead of a histogram, and the row's select
   finishes on them; a pass stops once every row is done, typically a level
   early.
3. Each rank then emits only its own winners. Ties at the k-th score go to the
   lower ranks first, from the ranks' own counts in the final bucket.

Each row has an owner rank, which alone receives the row's histograms, sums
and picks (on all 16 subcores), and sends each rank back the row's state: two
short rounds per level, and a rank sends each histogram once rather than to
every peer. Every level of a pass costs those rounds' latency whatever the
rows' length, so the kernel is for prefill-sized calls; decode keeps the
candidate merge.

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

# Tile memory per subcore (512 KB), in words. A SparseCore's 16 tile memories
# and its VMEM_SHARED are one pool: resident row parts get what the other
# scratch leaves of it, less TILE_RESERVE words per subcore for the compiler
# (it keeps ~800). Streamed parts go through two SEG_WORDS halves of the same
# buffer. MAX_ROWS is the most row parts a subcore holds at once: a pass over
# the rows exchanges histograms once per level, so short parts are batched up
# to MAX_ROWS deep.
TILE_WORDS = 128 * 1024
TILE_RESERVE = 2048
SEG_WORDS = 32 * 1024
MAX_ROWS = 8
NUM_CORES = 2
NUM_SUBCORES = 16
ROWS_PER_PASS = NUM_CORES * NUM_SUBCORES
NUM_LEVELS = 1 + len(REFINEMENT_LEVELS)
# Levels 1..3 run as one loop: refinement level `l` takes 8 bits at shift
# L0_SHIFT - 8 * l.
assert (
    tuple((L0_SHIFT - 8 * level, 0xFF) for level in range(1, NUM_LEVELS))
    == REFINEMENT_LEVELS
)

# Per-row selection state in SMEM, NF words per row: the row, its part's
# length on this rank, and the selection. F_CAND: the next level sends the
# row's candidates rather than its histogram (see `exchange_owner`).
(F_ROW, F_LEN, F_PREFIX, F_SHIFT, F_QUOTA, F_DONE, F_KEEP, F_MINE, F_BEFORE, F_CAND) = (
    range(10)
)
NF = 10
# The state an owner sends back, one STATE_WORDS vector per row and rank.
STATE_FIELDS = (F_PREFIX, F_SHIFT, F_QUOTA, F_DONE, F_KEEP, F_MINE, F_BEFORE, F_CAND)
STATE_WORDS = LANES
# A row's slot of a level's exchange: its histogram, or up to CAND_WORDS of
# its candidate keys.
CAND_WORDS = NUM_BUCKETS


def _picks_per_subcore(dcp_size):
    """Owned rows each subcore of an owner sums and picks: an owner owns
    NUM_SUBCORES // dcp_size subcores' MAX_ROWS rows, spread over all 16."""
    return max(1, MAX_ROWS // dcp_size)


def _align16(x):
    return jnp.bitwise_and(x + (LANES - 1), jnp.int32(-LANES))


def _signal(sem, **device):
    pl.semaphore_signal(sem, 1, device_id=device, device_id_type=pl.DeviceIdType.MESH)


def _vector_body(
    scores_hbm,  # i32[b * n], this rank's parts of the rows
    lengths_hbm,  # i32[b + pad], this rank's part lengths
    bounds_hbm,  # i32[b + pad], the same on every rank, >= every rank's length
    after_hbm,  # i32[...], only orders this call after another; never read
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
    live_vmem,
    state_smem,
    pick_smem,
    hist_shared,
    state_shared,
    ctrl_shared,
    live_shared,
    v2s_sem,
    s2v_sem,
    row_sem,
    seg_sems,
    copy_sem,
    out_sems,
    b: int,
    n: int,
    k: int,
    dcp_size: int,
    axis_name: str,
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
    # One subcore's histograms of its rows: a slot of a hist_shared block.
    rank_words = MAX_ROWS * NUM_BUCKETS
    # Row (sub, j) belongs to rank `sub % dcp_size`. A block of hist_shared
    # holds one rank's histograms of one owner's rows.
    per_owner = NUM_SUBCORES // dcp_size
    my_owner = jnp.bitwise_and(sub, dcp_size - 1)
    my_slot = jnp.right_shift(sub, dcp_size.bit_length() - 1)
    block = per_owner * MAX_ROWS * NUM_BUCKETS
    state_block = per_owner * MAX_ROWS * STATE_WORDS
    owned = per_owner * MAX_ROWS
    picks_per_subcore = _picks_per_subcore(dcp_size)

    def vec_at(ref, idx):
        return ref[pl.ds(idx, LANES)][0]

    def get(j, f):
        return state_smem[j * NF + f]

    def put(j, f, v):
        state_smem[j * NF + f] = v

    def pick_get(i, f):
        return pick_smem[i * NF + f]

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

    def handoff(new_level=False):
        """Hands the staged data to the scalar subcore and waits for its round;
        the first round of a level also tells it that a level runs."""
        if new_level:
            post(1)
        else:
            _signal(v2s_sem, core=core)
        pl.semaphore_wait(s2v_sem, 1)

    def any_live(nrows):
        """Whether a row of this core's pass is not done yet. Every subcore of
        the core (and the same core of every rank: the states agree) gets the
        same answer, from all 16 subcores' counts of their own live rows."""

        def count(j, c):
            return c + (get(j, F_DONE) == 0).astype(jnp.int32)

        ctrl_vmem[...] = jnp.full(
            (LANES,), lax.fori_loop(0, nrows, count, 0), jnp.int32
        )
        pltpu.sync_copy(ctrl_vmem, live_shared.at[pl.ds(sub * LANES, LANES)])
        plsc.subcore_barrier()
        pltpu.sync_copy(live_shared, live_vmem)

        def add(c, acc):
            return acc + live_vmem[pl.ds(c * LANES, LANES)]

        total = plsc.parallel_loop(
            0, NUM_SUBCORES, carry=jnp.zeros((LANES,), jnp.int32)
        )(add)  # pytype: disable=bad-argument-type
        return total[0] > 0

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
    S_HALF, S_NEXT = MAX_ROWS * NF, MAX_ROWS * NF + 1

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

    def for_segments(j, carry, *, stride, streamed, nrows, keys, body, then, live):
        """Folds `body(part, length, col0, carry)` over row `j`'s segments.

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
            return body(part, length, 0, c)

        segs = jnp.where(streamed, num_segments(j), 0)
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
            return body(part, seg_len, seg * SEG_WORDS, c)

        carry = lax.fori_loop(0, jnp.where(streamed, 0, 1), resident, carry)
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

    def hist_row(j, carry, *, level, first, stride, streamed, nrows):
        """Level `level`'s histogram of row `j` (`first`: level 0, else a
        refinement level, traced)."""
        hist = hist_vmem.at[pl.ds(pl.multiple_of(j * NUM_BUCKETS, LANES), NUM_BUCKETS)]
        # A candidate row packs its keys in the window into its slot instead
        # of counting them: at most CAND_WORDS (the owner checked), after
        # -inf keys, which no window holds.
        cand = jnp.bool_(False) if first else get(j, F_CAND) != 0
        empty = jnp.where(cand, neg_inf_key, 0)

        def zero(i):
            hist[pl.ds(i * LANES, LANES)] = jnp.full((LANES,), empty, jnp.int32)

        plsc.parallel_loop(0, NUM_BUCKETS // LANES, unroll=8)(zero)
        if first:

            def segment(part, length, col0, c):
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
            shift = L0_SHIFT - 8 * level
            lo_excl, hi_incl = key_window(get(j, F_PREFIX), get(j, F_SHIFT))

            def segment(part, length, col0, woff: jax.Array):
                chunks = jnp.right_shift(length + (LANES - 1), 4)

                def body(i):
                    key = part[pl.ds(i * LANES, LANES)]
                    match = (key > lo_excl) & (key <= hi_incl)
                    bucket = jnp.bitwise_and(jnp.right_shift(key, shift), 0xFF)
                    hist_add(hist, bucket, match)

                plsc.parallel_loop(0, jnp.where(cand, 0, chunks), unroll=SCAN_UNROLL)(
                    body
                )

                # The fill so far rides in every lane of a vector, so that no
                # step waits on a scalar from the one before.
                def pack(i, woff):
                    key = part[pl.ds(i * LANES, LANES)]
                    match = (key > lo_excl) & (key <= hi_incl)
                    pos = woff + plsc.cumsum(match.astype(jnp.int32)) - 1
                    plsc.store_scatter(hist, [pos], key, mask=match)
                    return woff + plsc.all_reduce_population_count(match)

                return plsc.parallel_loop(
                    0, jnp.where(cand, chunks, 0), unroll=SCAN_UNROLL, carry=woff
                )(pack)  # pytype: disable=bad-argument-type

            live = get(j, F_DONE) == 0

        def row_live(i):
            return jnp.bool_(True) if first else get(i, F_DONE) == 0

        @pl.when(live)
        def _():
            for_segments(
                j,
                jnp.zeros((LANES,), jnp.int32),
                stride=stride,
                streamed=streamed,
                nrows=nrows,
                keys=not first,
                body=segment,
                then=0,  # the next level, or the emit, starts at row 0
                live=row_live,
            )

        pl.when(~live)(functools.partial(drain, j))
        return carry

    def pick_bucket(quota):
        """The bucket of glob_vmem's histogram that holds the `quota`-th
        largest entry: (bucket, its count, the quota left in it, the total)."""
        total = scan_glob()
        bucket = find_bucket(total - quota)
        cum_at = vec_at(glob_vmem, bucket)
        cum_lo = jnp.where(bucket > 0, vec_at(glob_vmem, jnp.maximum(bucket - 1, 0)), 0)
        return bucket, cum_at - cum_lo, quota - (total - cum_at), total

    def publish_all(i, fields, mines, publish):
        """Hands each rank its state: its own entries in the window, and the
        lower ranks'."""
        before = jnp.int32(0)
        for d in range(dcp_size):
            publish(d, {**fields, F_MINE: mines[d], F_BEFORE: before})
            before = before + mines[d]

    def select(i, level, first, publish):
        """Sums pick `i`'s histograms over the ranks (in peer_vmem), picks the
        bucket, and hands `publish(d, fields)` each rank's new state."""
        hb = i * dcp_size * NUM_BUCKETS

        def sum_ranks(c):
            acc = peer_vmem[pl.ds(hb + c * LANES, LANES)]
            for d in range(1, dcp_size):
                acc = acc + peer_vmem[pl.ds(hb + d * NUM_BUCKETS + c * LANES, LANES)]
            glob_vmem[pl.ds(c * LANES, LANES)] = acc

        plsc.parallel_loop(0, NUM_BUCKETS // LANES)(sum_ranks)
        quota = jnp.int32(k) if first else pick_get(i, F_QUOTA)
        bucket, cnt, quota, total = pick_bucket(quota)
        fields = {F_QUOTA: quota}
        if first:
            keep = total <= k
            fields[F_KEEP] = keep.astype(jnp.int32)
            fields[F_PREFIX] = bucket - NUM_BUCKETS // 2
            fields[F_SHIFT] = jnp.int32(L0_SHIFT)
            done = keep | (quota == cnt)
        else:
            fields[F_KEEP] = pick_get(i, F_KEEP)
            fields[F_PREFIX] = jnp.left_shift(pick_get(i, F_PREFIX), 8) | bucket
            fields[F_SHIFT] = L0_SHIFT - 8 * level
            done = (quota == cnt) | (quota == 0)
        fields[F_DONE] = done.astype(jnp.int32)
        mines = [
            vec_at(peer_vmem, hb + d * NUM_BUCKETS + bucket) for d in range(dcp_size)
        ]
        # The next level sends the window's keys if every rank's fit a slot.
        most = functools.reduce(jnp.maximum, mines)
        fields[F_CAND] = (~done & (most <= CAND_WORDS)).astype(jnp.int32)
        for f in (F_QUOTA, F_KEEP, F_PREFIX, F_SHIFT, F_DONE, F_CAND):
            pick_smem[i * NF + f] = fields[f]
        publish_all(i, fields, mines, publish)

    def select_cand(i, publish):
        """Pick `i`'s exact selection from every rank's candidates (in
        peer_vmem, -inf padded): the radix select's remaining 8-bit levels,
        on the candidates alone. The row is then done."""
        hb = i * dcp_size * NUM_BUCKETS
        chunks = dcp_size * NUM_BUCKETS // LANES  # rank d's: [16 d, 16 d + 16)

        def in_window(c, lo_excl, hi_incl):
            key = peer_vmem[pl.ds(hb + c * LANES, LANES)]
            return key, (key > lo_excl) & (key <= hi_incl)

        def refine(_, carry):
            @pl.when(pick_get(i, F_DONE) == 0)
            def _():
                prefix, shift = pick_get(i, F_PREFIX), pick_get(i, F_SHIFT)
                lo_excl, hi_incl = key_window(prefix, shift)
                shift = shift - 8

                def zero(c):
                    glob_vmem[pl.ds(c * LANES, LANES)] = jnp.zeros((LANES,), jnp.int32)

                def count(c):
                    key, sel = in_window(c, lo_excl, hi_incl)
                    bucket = jnp.bitwise_and(jnp.right_shift(key, shift), 0xFF)
                    hist_add(glob_vmem, bucket, sel)

                plsc.parallel_loop(0, NUM_BUCKETS // LANES)(zero)
                plsc.parallel_loop(0, chunks)(count)
                bucket, cnt, quota, _ = pick_bucket(pick_get(i, F_QUOTA))
                pick_smem[i * NF + F_PREFIX] = jnp.left_shift(prefix, 8) | bucket
                pick_smem[i * NF + F_SHIFT] = shift
                pick_smem[i * NF + F_QUOTA] = quota
                pick_smem[i * NF + F_DONE] = ((quota == cnt) | (quota == 0)).astype(
                    jnp.int32
                )

            return carry

        lax.fori_loop(0, jnp.right_shift(pick_get(i, F_SHIFT), 3), refine, 0)
        prefix, shift = pick_get(i, F_PREFIX), pick_get(i, F_SHIFT)
        lo_excl, hi_incl = key_window(prefix, shift)

        # Each rank's candidates in the final window, rank d's in lane d.
        def per_rank(c, acc):
            _, sel = in_window(c, lo_excl, hi_incl)
            n = plsc.all_reduce_population_count(sel)
            return acc + jnp.where(lane_iota == jnp.right_shift(c, 4), n, 0)

        mines = plsc.parallel_loop(0, chunks, carry=jnp.zeros((LANES,), jnp.int32))(
            per_rank
        )  # pytype: disable=bad-argument-type
        fields = {
            F_PREFIX: prefix,
            F_SHIFT: shift,
            F_QUOTA: pick_get(i, F_QUOTA),
            F_DONE: jnp.int32(1),
            F_KEEP: pick_get(i, F_KEEP),
            F_CAND: jnp.int32(0),
        }
        pick_smem[i * NF + F_DONE] = jnp.int32(1)
        publish_all(i, fields, [mines[d] for d in range(dcp_size)], publish)

    def exchange_owner(level, first, nrows):
        """The owner of each row sums its histograms and sends back its state.

        hist_shared holds [out | in] x rank x block: `out[d]` this rank's
        histograms of rank d's rows, `in[d]` rank d's histograms of this
        rank's rows; a block holds MAX_ROWS rows per owned subcore and moves
        whole (a remote copy between different offsets needs a static size).
        state_shared is laid out alike, with the states. A row whose state
        says F_CAND has its window's keys in its slot instead of a histogram
        (`hist_row`), and its owner finishes its select on them
        (`select_cand`).
        """
        words = nrows * NUM_BUCKETS
        state_words = nrows * STATE_WORDS
        state_rows = MAX_ROWS * STATE_WORDS  # one subcore's slot of a state block
        # Round 1: this subcore's histograms to its rows' owner.
        dst = (
            jnp.where(my_owner == rank, dcp_size + my_owner, my_owner) * block
            + my_slot * rank_words
        )
        pltpu.sync_copy(
            hist_vmem.at[pl.ds(0, words)],
            hist_shared.at[pl.ds(pl.multiple_of(dst, LANES), words)],
        )
        handoff(new_level=True)

        # Every subcore of the owner picks for a share of the owned rows, not
        # just the owned subcores: row j of owned subcore slot `slot` is owned
        # row q = j * per_owner + slot, pick q // 16 of subcore q % 16. Its
        # state goes to every rank's slot of the row; a row that is done or
        # unused this pass gets a zero (not updated) state. Only at dcp 16 do
        # some subcores own no row; their copies land in a spare slot past the
        # blocks.
        picks = []
        for i in range(picks_per_subcore):
            q = sub + NUM_SUBCORES * i
            exists = q < owned
            q = jnp.minimum(q, owned - 1)
            slot = jnp.bitwise_and(q, per_owner - 1)
            j = jnp.right_shift(q, per_owner.bit_length() - 1)
            live = exists & (j < nrows)
            if not first:
                live = live & (pick_get(i, F_DONE) == 0)
            picks.append((i, exists, slot, j, live))
        copy_all(
            (
                hist_shared.at[
                    pl.ds(
                        pl.multiple_of(
                            (dcp_size + d) * block
                            + slot * rank_words
                            + j * NUM_BUCKETS,
                            LANES,
                        ),
                        NUM_BUCKETS,
                    )
                ],
                peer_vmem.at[pl.ds((i * dcp_size + d) * NUM_BUCKETS, NUM_BUCKETS)],
            )
            for i, _, slot, j, _ in picks
            for d in range(dcp_size)
        )
        for i, _, _, _, live in picks:

            def publish(d, fields, i=i):
                v = jnp.zeros((LANES,), jnp.int32)
                for w, f in enumerate(STATE_FIELDS):
                    v = jnp.where(lane_iota == w, fields[f], v)
                v = jnp.where(lane_iota == len(STATE_FIELDS), 1, v)  # updated
                state_vmem[pl.ds((i * dcp_size + d) * STATE_WORDS, LANES)] = v

            if first:
                pl.when(live)(functools.partial(select, i, level, first, publish))
            else:
                cand = pick_get(i, F_CAND) != 0
                pl.when(live & ~cand)(
                    functools.partial(select, i, level, first, publish)
                )
                pl.when(live & cand)(functools.partial(select_cand, i, publish))

            @pl.when(~live)
            def _(i=i):
                for d in range(dcp_size):
                    state_vmem[pl.ds((i * dcp_size + d) * STATE_WORDS, LANES)] = (
                        jnp.zeros((LANES,), jnp.int32)
                    )

        spare = 2 * dcp_size * state_block + sub * STATE_WORDS
        copy_all(
            (
                state_vmem.at[pl.ds((i * dcp_size + d) * STATE_WORDS, STATE_WORDS)],
                state_shared.at[
                    pl.ds(
                        pl.multiple_of(
                            jnp.where(
                                exists,
                                jnp.where(d == rank, dcp_size + d, d) * state_block
                                + slot * state_rows
                                + j * STATE_WORDS,
                                spare,
                            ),
                            LANES,
                        ),
                        STATE_WORDS,
                    )
                ],
            )
            for i, exists, slot, j, _ in picks
            for d in range(dcp_size)
        )

        # Round 2: the owners' states back to every rank.
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

    # Output rows are built in the two halves of rowbuf_vmem, alternately, and
    # written out while the next one is built.
    half_words = k + LANES

    def fill_row(half):
        def fill(i):
            rowbuf_vmem[pl.ds(half * half_words + i * LANES, LANES)] = jnp.full(
                (LANES,), -1, jnp.int32
            )

        plsc.parallel_loop(0, k // LANES, unroll=4)(fill)

    def row_copy(half, row):
        return pltpu.make_async_copy(
            rowbuf_vmem.at[pl.ds(pl.multiple_of(half * half_words, LANES), k)],
            out_hbm.at[pl.ds(pl.multiple_of(row * k, LANES), k)],
            out_sems.at[half],
        )

    def wait_row(j, c):
        """Waits out the write of row `j` (any row of its half: same size)."""
        row_copy(jnp.bitwise_and(j, 1), 0).wait()
        return c

    def emit_row(j, carry, *, stride, streamed, nrows):
        """This rank's winners of row `j`, written out."""
        half = jnp.bitwise_and(j, 1)
        base = half * half_words
        # Row j - 2 was built in this half: its write must be done.
        pl.when(j >= 2)(functools.partial(wait_row, j, None))
        fill_row(half)
        lo_excl, hi_incl = key_window(get(j, F_PREFIX), get(j, F_SHIFT))
        mine = get(j, F_MINE)
        my_take = jnp.clip(get(j, F_QUOTA) - get(j, F_BEFORE), 0, mine)
        take_all_ties = (get(j, F_KEEP) != 0) | (my_take >= mine)
        gate = jnp.where(take_all_ties, lo_excl, hi_incl)
        whole_bucket = take_all_ties | (my_take <= 0)

        def segment(part, length, col0, c: tuple[jax.Array, jax.Array]):
            chunks = jnp.right_shift(length + (LANES - 1), 4)

            def emit(i, woff, sel):
                plsc.store_compressed(
                    rowbuf_vmem.at[pl.ds(base + woff, LANES)],
                    col0 + i * LANES + lane_iota,
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
        )
        row_copy(half, get(j, F_ROW)).start()
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
        stride = jnp.maximum(_align16(stride), LANES)
        streamed = nrows * stride > row_words
        state_smem[S_HALF] = jnp.int32(0)
        state_smem[S_NEXT] = jnp.int32(-1)

        def fetch(j, c):
            row = row0 + flat + ROWS_PER_PASS * j
            length = jnp.minimum(vec_at(len_vmem, row), n)
            put(j, F_ROW, row)
            put(j, F_LEN, length)
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

        def run_level(level, first):
            lax.fori_loop(
                0,
                nrows,
                functools.partial(
                    hist_row,
                    level=level,
                    first=first,
                    stride=stride,
                    streamed=streamed,
                    nrows=nrows,
                ),
                0,
            )
            exchange_owner(level, first, nrows)

        # Level 0, then the refinement levels while a row of the pass is not
        # done: once every row has its selection (from candidates, or with
        # fewer than k entries), the levels left are skipped.
        run_level(0, True)

        def refine(carry):
            level, _ = carry
            run_level(level, False)
            return level + 1, any_live(nrows)

        lax.while_loop(
            lambda carry: (carry[0] < NUM_LEVELS) & carry[1],
            refine,
            (jnp.int32(1), any_live(nrows)),
        )
        lax.fori_loop(
            0,
            nrows,
            functools.partial(emit_row, stride=stride, streamed=streamed, nrows=nrows),
            0,
        )
        lax.fori_loop(jnp.maximum(nrows - 2, 0), nrows, wait_row, 0)
        return row0 + ROWS_PER_PASS * nrows

    row_end = lax.while_loop(lambda r: r < num_live, pass_body, jnp.int32(0))

    post(0)  # nothing more to exchange: let the scalar subcore finish

    # The padding rows after the last pass.
    fill_row(0)
    first_pad = row_end + flat

    def write_pad(i, c):
        copy = row_copy(0, first_pad + ROWS_PER_PASS * i)
        copy.start()
        copy.wait()
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
    **_,
):
    """Runs the exchange rounds of this core with the same core of the peers.

    A round: wait for the 16 vector subcores to stage their data, send each
    peer its part, wait for every peer's part to arrive, release the subcores.
    A level is two rounds.
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

    def level_rounds():
        # out[d] (this rank's histograms of d's rows) to in[rank] at d.
        send(
            hist_shared,
            lambda d: d * block,
            lambda d: (dcp_size + rank) * block,
            block,
            0,
        )
        staged()
        send(
            state_shared,
            lambda d: d * state_block,
            lambda d: (dcp_size + rank) * state_block,
            state_block,
            1,
        )

    # The vectors post with every level's histograms (1), and once at the end
    # (0); how many levels a pass runs is theirs to say.
    def next_level(_):
        more = ctrl()
        pl.when(more > 0)(level_rounds)
        return more

    lax.while_loop(lambda more: more > 0, next_level, jnp.int32(1))


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
    previous call's output as `after` (any array; a dependency only, never
    read).
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
    # `after` goes whole, not sliced: a slice would be an op between the calls.
    after = jnp.zeros((LANES,), jnp.int32) if after is None else after.reshape(-1)

    hist_words = 2 * NUM_SUBCORES * MAX_ROWS * NUM_BUCKETS
    # The state blocks, and a spare slot per subcore (see `exchange_owner`).
    state_words = (2 * MAX_ROWS + 1) * NUM_SUBCORES * LANES
    picks = _picks_per_subcore(dcp_size)
    other_words = (
        MAX_ROWS * NUM_BUCKETS  # hist_vmem
        + picks * dcp_size * NUM_BUCKETS  # peer_vmem
        + (NUM_BUCKETS + LANES)  # glob_vmem
        + 2 * (k + LANES)  # rowbuf_vmem
        + max(picks * dcp_size, MAX_ROWS) * STATE_WORDS  # state_vmem
        + 2 * (b + pad)  # len_vmem, bnd_vmem
        + (1 + NUM_SUBCORES) * LANES  # ctrl_vmem, live_vmem
    )
    shared_words = hist_words + state_words + (1 + NUM_SUBCORES) * LANES
    row_words = (TILE_WORDS - TILE_RESERVE - other_words) - shared_words // NUM_SUBCORES
    row_words = row_words // LANES * LANES
    if row_words < 2 * SEG_WORDS:
        raise ValueError(f"{b=} rows leave too little tile memory for the row parts")
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
        hist_vmem=vmem((MAX_ROWS * NUM_BUCKETS,), jnp.int32),
        peer_vmem=vmem((picks * dcp_size * NUM_BUCKETS,), jnp.int32),
        glob_vmem=vmem((NUM_BUCKETS + LANES,), jnp.int32),
        rowbuf_vmem=vmem((2 * (k + LANES),), jnp.int32),
        state_vmem=vmem((max(picks * dcp_size, MAX_ROWS) * STATE_WORDS,), jnp.int32),
        len_vmem=vmem((b + pad,), jnp.int32),
        bnd_vmem=vmem((b + pad,), jnp.int32),
        ctrl_vmem=vmem((LANES,), jnp.int32),
        live_vmem=vmem((NUM_SUBCORES * LANES,), jnp.int32),
        state_smem=pallas_core.CoreMemorySpace(pltpu.MemorySpace.SMEM, vmesh)(
            (MAX_ROWS * NF + 2,), jnp.int32
        ),
        pick_smem=pallas_core.CoreMemorySpace(pltpu.MemorySpace.SMEM, vmesh)(
            (picks * NF,), jnp.int32
        ),
        hist_shared=pltpu.VMEM_SHARED((hist_words,), jnp.int32),
        state_shared=pltpu.VMEM_SHARED((state_words,), jnp.int32),
        ctrl_shared=pltpu.VMEM_SHARED((LANES,), jnp.int32),
        live_shared=pltpu.VMEM_SHARED((NUM_SUBCORES * LANES,), jnp.int32),
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
        out_sems=pallas_core.CoreMemorySpace(pltpu.MemorySpace.SEMAPHORE, vmesh)(
            (2,), pltpu.SemaphoreType.DMA.dtype
        ),
    )
    params = dict(
        b=b, n=n, k=k, dcp_size=dcp_size, axis_name=axis_name, row_words=row_words
    )
    out_rows = -(-b // ROWS_PER_PASS) * ROWS_PER_PASS
    out = pl.kernel(
        [
            functools.partial(_vector_body, **params),
            functools.partial(_scalar_body, **params),
        ],
        # Whole passes of rows, so that every row a pass emits has its place.
        out_type=jax.ShapeDtypeStruct((out_rows * k,), jnp.int32),
        mesh=[vmesh, smesh],
        scratch_types=scratch_types,
        # No collective_id: XLA compiles it to a barrier semaphore it also
        # gives the TensorCore collectives this asynchronous call overlaps.
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            needs_layout_passes=False,
        ),
        name=f"sc_topk_dcp_b{b}_n{n}_k{k}_d{dcp_size}",
        # pyrefly: ignore[bad-argument-count]
    )(words.reshape(-1), lengths, bounds, after)
    if scheduling_group_id is not None:
        out = xla_metadata.set_xla_metadata(
            out, _scheduling_group_id=scheduling_group_id
        )
    # pl.kernel and set_xla_metadata are untyped upstream; both return the
    # out_type array here.
    return cast(jax.Array, out)[: b * k].reshape(b, k)
