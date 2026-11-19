# SPDX-License-Identifier: Apache-2.0
# q38 PORT-BL UNION: verbatim from Benliang Wang <benliangw@google.com>, commit 2ce02bb1a on
# benliangw/q38fh-integ-pin-5b44702aea (VLLM_TPU_QSA_DECODE_RPA_UNION metadata kernel).
"""Union page lists and row-mask words of verify requests, built on SparseCore.

In a verify step a request has up to four rows, each with its own top-512
selection of 4-token groups. Under main-KV DCP (decode layout, ownership unit
of one 512-token page) a rank fetches the groups it owns. The union tile
attends one sequence per request over the union of its rows' owned groups,
with a per-token word saying which rows attend each token.

``union_sc_metadata`` returns, per request ``r`` (rows ``starts[r]`` to
``starts[r + 1] - 1``, at most four):

- ``pages`` int32 [R, P]: the union's virtual page ids (``physical * 128 +
  row group``, as ``qsa_rpa._dcp_page_metadata`` builds them). Order is first
  appearance, row by row, except that the request's final partial group (the
  tail of its longest row), when owned, is last. Slots at or past U hold 0.
- ``words`` int32 [R, P]: byte ``o`` of a page's word holds the row bits of
  token ``o`` (bit ``j`` for row ``starts[r] + j``). Words at or past U are 0,
  and so are the bytes of the final page's unwritten tokens.
- ``meta`` int32 [R, 16]: lane 0 U (union pages), lane 1 kv_len (tokens to
  fetch: ``4 U``, less the unwritten tokens of the final page when it is
  owned), lanes 2 to 5 the owned visible tokens of row ``j`` (0 for a missing
  or dead row), the rest 0.

A slot of a row counts when it is ``>= 0``, below the row's complete groups,
owned by the rank, and, in the global form, its index is below ``min(complete,
512)`` (the served per-row rule); ``own_list=True`` (F10's own list, -1 in
dropped slots, any order) drops the index rule. Ids are unique within a row.
A row's partial tail group counts for that row when the rank owns it.

One request per vector subcore (2 SparseCores x 16 subcores), in waves. A
subcore stages the request's rows, its table row and a position table over the
rank's local groups of the request (position in the union plus one, 0 for
unseen). It walks each row in 16-lane chunks: gather the chunk's table
entries, append the unseen groups' pages with a compressed store at a carried
offset, scatter their positions into the table, and add the row's bit to the
word of every valid slot's position. Tail groups follow, one at a time, the
final one last. The table is zeroed before the walk (whole by default, 64 KiB
at a 262,144-token table; ``clear="scatter"`` zeroes only the entries the walk
reads, slower on TPU7x at 100k-token contexts).

``emit_sel=True`` makes the kernel write the token words
(``widen_words(words)``, int32 [R, 4P]) itself: an XLA widening costs about
8 us a call on TPU7x (its minor dimension of 4 relayouts), the kernel about
0.5 us. ``write="tile"`` copies only the 160-slot chunks up to the one holding
U, so slots past it are undefined; a sequence's 640-token tiles below kv_len
read none of them.

``request_body`` is the per-request code for both the kernel and the CPU
emulation in ``tests/kernels/test_qwen4_exp_union_meta.py``, which runs it on
numpy stand-ins for the primitives.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.sparse_core import core_map_helper

LANES = 16
TOPK = 512
MAX_ROWS = 4
RATIO = 4
PAGE_GROUPS = 128  # 4-token groups in one 512-token main-KV page, the DCP unit
META = 16
# Header lanes staged per request.
H_START, H_ROWS, H_LEN, H_RANK = 0, 1, 2, 6
# Per-subcore position table budget, in words (the tile memory is 512 KiB).
POS_WORDS_MAX = 64 * 1024
SELECTED_WORD = 0x01010101
# Output slots per copy under write="live": one 640-token tile of 4-token pages.
WRITE_CHUNK = 160


def pages_per_seq(tile: int = 640) -> int:
    """Union page slots per request: the worst case (4 x 512 selected groups
    and 4 tails) rounded up to whole tiles of ``tile`` tokens."""
    if tile % (RATIO * LANES):
        raise ValueError(f"tile={tile} must be a multiple of {RATIO * LANES} tokens")
    worst = (MAX_ROWS * TOPK + MAX_ROWS) * RATIO
    return -(-worst // tile) * tile // RATIO


def union_reference(
    blocks,
    lengths,
    starts,
    table,
    rank,
    world=4,
    page_groups=128,
    ratio=4,
    own_list=False,
    budget=TOPK,
):
    """Per request ``(pages, sel, kv_len, groups, row_tokens)``: the design's
    reference definition (``sel`` one word per token, bit ``j`` for row ``j``),
    with the served slot rule and each row's owned visible token count."""
    out = []
    for r in range(len(starts) - 1):
        rows = range(starts[r], min(starts[r + 1], starts[r] + MAX_ROWS))
        vis = {}
        row_tokens = [0] * MAX_ROWS
        for j, i in enumerate(rows):
            length = int(lengths[i])
            complete = length // ratio
            count = len(blocks[i]) if own_list else min(complete, budget)
            for g in list(blocks[i])[:count]:
                g = int(g)
                if 0 <= g < complete and (g // page_groups) % world == rank:
                    vis.setdefault(g, [0] * MAX_ROWS)[j] = ratio
                    row_tokens[j] += ratio
            if length % ratio and (complete // page_groups) % world == rank:
                vis.setdefault(complete, [0] * MAX_ROWS)[j] = length % ratio
                row_tokens[j] += length % ratio
        last = max((int(lengths[i]) for i in rows), default=0)
        tail = last // ratio if last % ratio else None
        groups = sorted(g for g in vis if g != tail) + ([tail] if tail in vis else [])
        pages = [
            int(table[starts[r]][g // (world * page_groups)]) * page_groups
            + g % page_groups
            for g in groups
        ]
        sel = [
            sum((vis[g][j] > o) << j for j in range(MAX_ROWS))
            for g in groups
            for o in range(ratio)
        ]
        kv_len = ratio * len(groups) - (ratio - last % ratio if tail in vis else 0)
        out.append((pages, sel, kv_len, groups, row_tokens))
    return out


def _log2(n: int) -> int:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"{n} must be a power of two")
    return n.bit_length() - 1


class ScOps:
    """The SparseCore primitives ``request_body`` uses (the CPU test passes a
    numpy stand-in with the same methods)."""

    xp = jnp

    @staticmethod
    def lane():
        return lax.iota(jnp.int32, LANES)

    @staticmethod
    def full(x):
        return jnp.full((LANES,), x, jnp.int32)

    @staticmethod
    def scalar(ref, at):
        return ref[pl.ds(at, LANES)][0]

    @staticmethod
    def load(ref, at):
        return ref[pl.ds(at, LANES)]

    @staticmethod
    def store(ref, at, x):
        ref[pl.ds(at, LANES)] = x

    @staticmethod
    def store_compressed(ref, at, x, mask):
        plsc.store_compressed(ref.at[pl.ds(at, LANES)], x, mask=mask)

    @staticmethod
    def gather(ref, index, mask):
        return plsc.load_gather(ref, [index], mask=mask)

    @staticmethod
    def store_scatter(ref, index, x, mask):
        plsc.store_scatter(ref, [index], x, mask=mask)

    @staticmethod
    def addupdate_scatter(ref, index, x, mask):
        plsc.addupdate_scatter(ref, [index], x, mask=mask)

    @staticmethod
    def cumsum(x):
        return plsc.cumsum(x)

    @staticmethod
    def popcount(mask):
        return plsc.all_reduce_population_count(mask)[0]

    @staticmethod
    def loop(lower, upper, *, carry=None, unroll=1):
        return plsc.parallel_loop(lower, upper, unroll=unroll, carry=carry)


def request_body(
    ops,
    rows_v,
    head_v,
    tab_v,
    pos_v,
    pages_v,
    words_v,
    meta_v,
    *,
    pages,
    pos_words,
    world,
    own_list,
    clear="scatter",
    insert="store",
    unroll=4,
):
    """One request's union on staged buffers.

    ``rows_v`` [4 * 512] the request's rows (only rows below the header's row
    count are read), ``head_v`` [32] the header (start, rows, 4 lengths, rank),
    ``tab_v`` the request's table row, ``pos_v`` [pos_words] scratch,
    ``pages_v`` / ``words_v`` [pages + 16], ``meta_v`` [16]; the first
    ``pages`` entries of ``pages_v`` / ``words_v`` and ``meta_v`` are the
    outputs.

    ``clear``: "scatter" zeroes the position-table entries the request touches,
    "full" the whole table. ``insert``: "store" writes new positions with
    ``store_scatter``; "add" adds them to the cleared entries with
    ``addupdate_scatter`` (and "scatter" clearing then subtracts the gathered
    value), so the kernel can run without ``store_scatter``.
    """
    xp = ops.xp
    lane = ops.lane()
    zeros = ops.full(0)
    unit_shift = _log2(PAGE_GROUPS)
    entry_shift = _log2(world * PAGE_GROUPS)
    rank = ops.scalar(head_v, H_RANK)
    nrows = ops.scalar(head_v, H_ROWS)
    lengths = [ops.scalar(head_v, H_LEN + j) for j in range(MAX_ROWS)]
    complete = [length >> 2 for length in lengths]
    # Slots of row j to read; 0 for rows the request does not have.
    if own_list:
        limits = [xp.where(nrows > j, TOPK, 0) for j in range(MAX_ROWS)]
    else:
        limits = [
            xp.where(nrows > j, xp.minimum(complete[j], TOPK), 0)
            for j in range(MAX_ROWS)
        ]

    def owned(g):
        return ((g >> unit_shift) & (world - 1)) == rank

    def local(g):
        return ((g >> entry_shift) << unit_shift) | (g & (PAGE_GROUPS - 1))

    def page_of(g):
        return (ops.gather(tab_v, g >> entry_shift, None) << unit_shift) | (
            g & (PAGE_GROUPS - 1)
        )

    def slots(j, i):
        """Chunk ``i`` of row ``j``: (valid lanes, group ids with 0 elsewhere)."""
        g = ops.load(rows_v, j * TOPK + i * LANES)
        valid = (
            ((i * LANES + lane) < limits[j]) & (g >= 0) & (g < complete[j]) & owned(g)
        )
        return valid, xp.where(valid, g, 0)

    # Tails: (condition, group, local group, page, row bits) for each row j,
    # non-final tails first and the final one (the longest row's) last.
    last = lengths[0]
    for j in range(1, MAX_ROWS):
        last = xp.maximum(last, xp.where(nrows > j, lengths[j], 0))
    last = xp.where(nrows > 0, last, 0)
    final_group = xp.where((last & 3) != 0, last >> 2, -1)
    tails = []
    for want_final in (False, True):
        for j in range(MAX_ROWS):
            g = complete[j]
            partial = lengths[j] & 3
            has = (nrows > j) & (partial != 0) & owned(g)
            has = has & ((g == final_group) == want_final)
            word = (SELECTED_WORD & ((1 << (partial << 3)) - 1)) << j
            tails.append((has, g, word))

    def one_lane(cond):
        return lane < xp.where(cond, 1, 0)

    # Clear the position table where the walk will read it.
    if clear == "full":

        def clear_all(i):
            ops.store(pos_v, i * LANES, zeros)

        ops.loop(0, pos_words // LANES, unroll=8)(clear_all)
    else:

        def clear_at(index, mask):
            if insert == "store":
                ops.store_scatter(pos_v, index, zeros, mask)
            else:
                ops.addupdate_scatter(
                    pos_v, index, -ops.gather(pos_v, index, mask), mask
                )

        for j in range(MAX_ROWS):

            def clear_chunk(i, j=j):
                valid, g = slots(j, i)
                clear_at(local(g), valid)

            ops.loop(0, (limits[j] + LANES - 1) >> 4, unroll=unroll)(clear_chunk)
        for has, g, _ in tails:
            clear_at(ops.full(local(xp.where(has, g, 0))), one_lane(has))

    def fill(i):
        ops.store(pages_v, i * LANES, zeros)
        ops.store(words_v, i * LANES, zeros)

    ops.loop(0, pages // LANES + 1, unroll=8)(fill)

    def insert_at(index, x, mask):
        if insert == "store":
            ops.store_scatter(pos_v, index, x, mask)
        else:
            ops.addupdate_scatter(pos_v, index, x, mask)

    # Walk the rows. A row's ids are unique, so a chunk touches table entries
    # and words no other chunk of the row touches; rows run in order.
    woff = 0 if ops.xp is np else jnp.int32(0)
    counts = []
    for j in range(MAX_ROWS):
        bit = SELECTED_WORD << j

        def chunk(i, carry, j=j, bit=bit):
            woff, count = carry
            valid, g = slots(j, i)
            index = local(g)
            seen = ops.gather(pos_v, index, valid)
            new = valid & (seen == 0)
            fresh = xp.where(new, 1, 0)
            at = woff + ops.cumsum(fresh) - fresh
            ops.store_compressed(pages_v, woff, page_of(g), new)
            insert_at(index, at + 1, new)
            ops.addupdate_scatter(
                words_v, xp.where(new, at, seen - 1), ops.full(bit), valid
            )
            return woff + ops.popcount(new), count + ops.popcount(valid)

        woff, count = ops.loop(
            0, (limits[j] + LANES - 1) >> 4, unroll=unroll, carry=(woff, woff * 0)
        )(chunk)
        counts.append(count * RATIO)

    for k, (has, g, word) in enumerate(tails):
        j = k % MAX_ROWS
        active = one_lane(has)
        safe = xp.where(has, g, 0)
        index = ops.full(local(safe))
        seen = ops.gather(pos_v, index, active)
        new = active & (seen == 0)
        ops.store_compressed(pages_v, woff, page_of(ops.full(safe)), new)
        insert_at(index, ops.full(woff + 1), new)
        ops.addupdate_scatter(
            words_v, xp.where(new, woff, seen - 1), ops.full(word), active
        )
        woff = woff + ops.popcount(new)
        counts[j] = counts[j] + xp.where(has, lengths[j] & 3, 0)

    # A compressed store may leave its unselected slots dirty: zero past U.
    ops.store_compressed(pages_v, woff, zeros, lane >= 0)
    final_owned = (final_group >= 0) & owned(xp.maximum(final_group, 0))
    kv_len = woff * RATIO - xp.where(final_owned, RATIO - (last & 3), 0)
    meta = xp.where(lane == 0, woff, xp.where(lane == 1, kv_len, 0))
    for j in range(MAX_ROWS):
        meta = xp.where(lane == 2 + j, counts[j], meta)
    ops.store(meta_v, 0, meta)


def stage_inputs(lengths, starts, table, rank):
    """Per-request header int32 [R, 16] and table rows int32 [R, cols16].

    They depend on the step only (not the layer's selections), so one copy
    serves every layer of a step."""
    num_rows = lengths.shape[0]
    first = starts[:-1].astype(jnp.int32)
    nrows = jnp.clip(starts[1:] - first, 0, MAX_ROWS).astype(jnp.int32)
    j = jnp.arange(MAX_ROWS, dtype=jnp.int32)
    row = jnp.minimum(first[:, None] + j, num_rows - 1)
    lens = jnp.where(j < nrows[:, None], lengths.astype(jnp.int32)[row], 0)
    rank = jnp.asarray(rank, jnp.int32).reshape(-1)[0]
    reqs = first.shape[0]
    header = jnp.concatenate(
        (
            first[:, None],
            nrows[:, None],
            lens,
            jnp.full((reqs, 1), rank, jnp.int32),
            jnp.zeros((reqs, META - 7), jnp.int32),
        ),
        axis=1,
    )
    cols = table.shape[1]
    tab = table.astype(jnp.int32)[jnp.minimum(first, num_rows - 1)]
    tab = jnp.pad(tab, ((0, 0), (0, (-cols) % LANES)))
    return header, tab


def expand_sel(ops, words_v, sel_v, chunks):
    """Token words from page words: ``sel_v[t] = byte t % 4 of words_v[t // 4]``
    for the first ``16 * chunks`` tokens."""
    xp = ops.xp
    lane = ops.lane()
    quad = lane >> 2
    shift = (lane & (RATIO - 1)) << 3

    def body(i):
        w = ops.gather(words_v, i * (LANES // RATIO) + quad, None)
        ops.store(sel_v, i * LANES, (w >> shift) & 0xFF)

    del xp
    ops.loop(0, chunks, unroll=8)(body)


def _kernel(
    blocks_hbm,
    head_hbm,
    tab_hbm,
    pages_hbm,
    second_hbm,
    meta_hbm,
    rows_v,
    head_v,
    tab_v,
    pos_v,
    pages_v,
    words_v,
    meta_v,
    sel_v,
    sem,
    *,
    reqs,
    waves,
    workers,
    cols16,
    out_dma,
    write,
    emit_sel,
    **body_kwargs,
):
    worker = lax.axis_index(("core", "subcore"))
    pages = body_kwargs["pages"]

    def request(wave, carry):
        r = wave * workers + worker

        @pl.when(r < reqs)
        def _():
            pltpu.sync_copy(
                head_hbm.at[pl.ds(pl.multiple_of(r * META, META), META)],
                head_v.at[pl.ds(0, META)],
            )
            start = head_v[pl.ds(H_START, LANES)][0]
            nrows = head_v[pl.ds(H_ROWS, LANES)][0]
            copies = [
                pltpu.make_async_copy(
                    tab_hbm.at[pl.ds(pl.multiple_of(r * cols16, LANES), cols16)],
                    tab_v,
                    sem,
                )
            ]
            for j in range(MAX_ROWS):
                copies.append(
                    pltpu.make_async_copy(
                        blocks_hbm.at[
                            pl.ds(pl.multiple_of((start + j) * TOPK, TOPK), TOPK)
                        ],
                        rows_v.at[pl.ds(j * TOPK, TOPK)],
                        sem,
                    )
                )
            copies[0].start()
            for j in range(MAX_ROWS):
                pl.when(nrows > j)(copies[j + 1].start)
            copies[0].wait()
            for j in range(MAX_ROWS):
                pl.when(nrows > j)(copies[j + 1].wait)
            request_body(
                ScOps,
                rows_v,
                head_v,
                tab_v,
                pos_v,
                pages_v,
                words_v,
                meta_v,
                **body_kwargs,
            )
            meta_out = pltpu.make_async_copy(
                meta_v, meta_hbm.at[pl.ds(pl.multiple_of(r * META, META), META)], sem
            )
            # "live" and "tile" copy the whole chunks below U only.
            chunk = pages if write == "full" else WRITE_CHUNK
            count = jnp.int32(1)
            if write != "full":
                # At least one chunk; no scalar division on SparseCore.
                u = meta_v[pl.ds(0, LANES)][0]
                for c in range(1, pages // chunk):
                    count = count + jnp.where(u > c * chunk, 1, 0)
            if emit_sel:
                expand_sel(ScOps, words_v, sel_v, count * (chunk * RATIO // LANES))
                second_v, width = sel_v, RATIO
            else:
                second_v, width = words_v, 1

            def outs(c):
                at = pl.multiple_of(r * pages + c * chunk, LANES)
                src = pl.ds(pl.multiple_of(c * chunk, LANES), chunk)
                wide_at = pl.multiple_of((r * pages + c * chunk) * width, LANES)
                wide_src = pl.ds(
                    pl.multiple_of(c * chunk * width, LANES), chunk * width
                )
                return [
                    pltpu.make_async_copy(
                        pages_v.at[src], pages_hbm.at[pl.ds(at, chunk)], sem
                    ),
                    pltpu.make_async_copy(
                        second_v.at[wide_src],
                        second_hbm.at[pl.ds(wide_at, chunk * width)],
                        sem,
                    ),
                ]

            if out_dma == "sync":
                meta_out.start()
                meta_out.wait()

                def start_wait(c, carry):
                    for copy in outs(c):
                        copy.start()
                        copy.wait()
                    return carry

                lax.fori_loop(0, count, start_wait, 0)

            else:
                meta_out.start()

                def start(c, carry):
                    for copy in outs(c):
                        copy.start()
                    return carry

                def wait(c, carry):
                    for copy in outs(c):
                        copy.wait()
                    return carry

                lax.fori_loop(0, count, start, 0)
                lax.fori_loop(0, count, wait, 0)

                meta_out.wait()

        return carry

    lax.fori_loop(0, waves, request, None)


def union_sc_metadata_staged(
    blocks,
    header,
    tab,
    *,
    cols,
    world=4,
    pages_per_seq=2080,
    own_list=False,
    clear="full",
    insert="store",
    unroll=4,
    out_dma="async",
    write="full",
    emit_sel=False,
):
    """``union_sc_metadata`` on ``stage_inputs``' header and table rows.

    ``clear``, ``insert``, ``unroll``: see ``request_body``. ``out_dma``:
    "async" issues the output copies together, "sync" one after another.
    ``write``: "full" copies every slot of a row; "live" only the chunks of
    ``WRITE_CHUNK`` slots below U, then zeroes the rest in XLA; "tile" the same
    chunks and nothing else, so slots past the chunk holding U are undefined
    (a 640-token tile reads none of them). ``emit_sel``: the second output is
    ``widen_words(words)``, int32 [R, 4P], written by the kernel.
    """
    num_rows, budget = blocks.shape
    if budget != TOPK:
        raise ValueError(f"blocks must have {TOPK} slots a row, got {budget}")
    if write != "full" and pages_per_seq % WRITE_CHUNK:
        raise ValueError(
            f"write={write!r} needs pages_per_seq a multiple of {WRITE_CHUNK}"
        )
    if pages_per_seq % LANES or pages_per_seq < MAX_ROWS * (TOPK + 1):
        raise ValueError(
            f"pages_per_seq={pages_per_seq} must be a multiple of {LANES} "
            f"holding {MAX_ROWS * (TOPK + 1)} pages"
        )
    _log2(world)
    reqs = header.shape[0]
    cols16 = tab.shape[1]
    pos_words = -(-cols * PAGE_GROUPS // LANES) * LANES
    if pos_words > POS_WORDS_MAX:
        raise ValueError(
            f"the position table needs {pos_words} words, over {POS_WORDS_MAX}"
        )
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available")
    if sc.num_lanes != LANES:
        raise NotImplementedError(f"expected {LANES} lanes, got {sc.num_lanes}")
    workers = sc.num_cores * sc.num_subcores
    mesh = plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )
    width = RATIO if emit_sel else 1
    flat = jax.ShapeDtypeStruct((reqs * pages_per_seq,), jnp.int32)
    wide = jax.ShapeDtypeStruct((reqs * pages_per_seq * width,), jnp.int32)
    pages, second, meta = core_map_helper.kernel(
        functools.partial(
            _kernel,
            reqs=reqs,
            waves=-(-reqs // workers),
            workers=workers,
            cols16=cols16,
            pages=pages_per_seq,
            pos_words=pos_words,
            world=world,
            own_list=own_list,
            clear=clear,
            insert=insert,
            unroll=unroll,
            out_dma=out_dma,
            write=write,
            emit_sel=emit_sel,
        ),
        out_type=(flat, wide, jax.ShapeDtypeStruct((reqs * META,), jnp.int32)),
        mesh=mesh,
        scratch_types=(
            pltpu.VMEM((MAX_ROWS * TOPK,), jnp.int32),  # rows_v
            pltpu.VMEM((2 * LANES,), jnp.int32),  # head_v
            pltpu.VMEM((cols16,), jnp.int32),  # tab_v
            pltpu.VMEM((pos_words,), jnp.int32),  # pos_v
            pltpu.VMEM((pages_per_seq + 2 * LANES,), jnp.int32),  # pages_v
            pltpu.VMEM((pages_per_seq + 2 * LANES,), jnp.int32),  # words_v
            pltpu.VMEM((META,), jnp.int32),  # meta_v
            pltpu.VMEM((RATIO * pages_per_seq if emit_sel else LANES,), jnp.int32),
            pltpu.SemaphoreType.DMA(()),
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        name=f"qsa_union_meta_r{reqs}_p{pages_per_seq}_c{cols}",
    )(blocks.astype(jnp.int32).reshape(-1), header.reshape(-1), tab.reshape(-1))
    pages = pages.reshape(reqs, pages_per_seq)
    second = second.reshape(reqs, pages_per_seq * width)
    meta = meta.reshape(reqs, META)
    if write == "live":
        live = jnp.arange(pages_per_seq, dtype=jnp.int32)[None, :] < meta[:, :1]
        pages = jnp.where(live, pages, 0)
        second = jnp.where(jnp.repeat(live, width, axis=1), second, 0)
    return pages, second, meta


def union_sc_metadata(
    blocks,
    lengths,
    starts,
    table,
    rank,
    *,
    world=4,
    pages_per_seq=2080,
    own_list=False,
    **kwargs,
):
    """Union pages, row-mask words and meta per request (see the module doc).

    ``blocks`` int32 [T, 512], ``lengths`` int32 [T], ``starts`` int32 [R + 1]
    (requests past the batch repeat the last start), ``table`` int32 [T, cols]
    per row (row ``starts[r]`` is read), ``rank`` this DCP rank (int32 scalar
    or [1]). Returns ``(pages [R, P], words [R, P], meta [R, 16])``.
    """
    header, tab = stage_inputs(lengths, starts, table, rank)
    return union_sc_metadata_staged(
        blocks,
        header,
        tab,
        cols=table.shape[1],
        world=world,
        pages_per_seq=pages_per_seq,
        own_list=own_list,
        **kwargs,
    )


def widen_words(words):
    """int32 [R, P] page words -> int32 [R, 4P] token words (byte o of a
    page's word is token o's row bits)."""
    shifts = jnp.arange(RATIO, dtype=jnp.int32) * 8
    sel = (words[..., None] >> shifts) & 0xFF
    return sel.reshape(words.shape[0], -1)


def union_page_metadata(
    blocks,
    lengths,
    starts,
    table,
    rank,
    *,
    world=4,
    pages_per_seq=2080,
    own_list=False,
    emit_sel=True,
    **kwargs,
):
    """``(pages, sel, kv_lens, cu_q_lens, row_live)`` for the union RPA call.

    ``sel`` int32 [R, 4P] one word per union token; ``kv_lens`` [R];
    ``cu_q_lens`` [R + 1] over sequences, a request with U > 0 holding its rows
    and any other none; ``row_live`` bool [T], rows with an owned visible token
    (the others' outputs are 0 and their LSE -inf).
    """
    pages, second, meta = union_sc_metadata(
        blocks,
        lengths,
        starts,
        table,
        rank,
        world=world,
        pages_per_seq=pages_per_seq,
        own_list=own_list,
        emit_sel=emit_sel,
        **kwargs,
    )
    sel = second if emit_sel else widen_words(second)
    return (pages, sel, *union_glue(meta, starts, lengths.shape[0]))


def union_row_live(meta, starts, num_rows):
    """bool [T]: rows with an owned visible token (meta lanes 2 to 5), by
    compare and reduce over [T, R, 4] (no gather)."""
    first = starts[:-1].astype(jnp.int32)
    j = jnp.arange(MAX_ROWS, dtype=jnp.int32)
    row = first[:, None] + j
    live = (j < (starts[1:] - first)[:, None]) & (meta[:, 2 : 2 + MAX_ROWS] > 0)
    rows = jnp.arange(num_rows, dtype=jnp.int32)[:, None, None]
    return jnp.any((row[None] == rows) & live[None], axis=(1, 2))


def union_glue(meta, starts, num_rows):
    """``(kv_lens, cu_q_lens, row_live)`` from ``union_sc_metadata``'s meta."""
    first = starts[:-1].astype(jnp.int32)
    nrows = jnp.clip(starts[1:] - first, 0, MAX_ROWS)
    q_lens = jnp.where(meta[:, 0] > 0, nrows, 0)
    reqs = first.shape[0]
    before = jnp.arange(reqs + 1)[:, None] > jnp.arange(reqs)[None, :]
    cu_q_lens = jnp.sum(jnp.where(before, q_lens[None, :], 0), axis=1, dtype=jnp.int32)
    return meta[:, 1], cu_q_lens, union_row_live(meta, starts, num_rows)


def union_mismatches(pages, words, meta, ref):
    """Host check of one request's outputs against ``union_reference``'s entry:
    the list of differences (empty when equal as sets, final tail last)."""
    ref_pages, ref_sel, ref_kv, _, ref_tokens = ref
    pages, words, meta = (np.asarray(x) for x in (pages, words, meta))
    u = int(meta[0])
    bad = []
    if u != len(ref_pages):
        return [f"U {u} != {len(ref_pages)}"]
    if int(meta[1]) != ref_kv:
        bad.append(f"kv_len {int(meta[1])} != {ref_kv}")
    if list(meta[2 : 2 + MAX_ROWS]) != list(ref_tokens) or meta[2 + MAX_ROWS :].any():
        bad.append(f"row tokens {list(meta[2:])} != {ref_tokens}")
    got = dict(zip(pages[:u].tolist(), words[:u].tolist()))
    want = {
        p: sum(ref_sel[RATIO * k + o] << (8 * o) for o in range(RATIO))
        for k, p in enumerate(ref_pages)
    }
    if len(got) != u:
        bad.append("repeated page")
    if got != want:
        diff = sorted(set(got.items()) ^ set(want.items()))[:4]
        bad.append(f"page words differ, e.g. {diff}")
    if u and ref_kv % RATIO and int(pages[u - 1]) != ref_pages[-1]:
        bad.append("final partial page not last")
    if pages[u:].any() or words[u:].any():
        bad.append("nonzero past U")
    return bad


def reference_arrays(blocks, lengths, starts, table, rank, *, pages_per_seq=2080, **kw):
    """``union_reference`` as padded host arrays: pages int32 [R, P], sel
    int32 [R, 4P] and kv_lens int32 [R] (the union RPA call's inputs)."""
    ref = union_reference(blocks, lengths, starts, table, rank, **kw)
    reqs = len(ref)
    pages = np.zeros((reqs, pages_per_seq), np.int32)
    sel = np.zeros((reqs, RATIO * pages_per_seq), np.int32)
    kv_lens = np.zeros((reqs,), np.int32)
    for r, (p, s, kv, _, _) in enumerate(ref):
        pages[r, : len(p)] = p
        sel[r, : len(s)] = s
        kv_lens[r] = kv
    return pages, sel, kv_lens
