# SPDX-License-Identifier: Apache-2.0
"""SparseCore union gather, step 2: the list and gather steps on v7x SparseCore.

The walking D2 kernel (``qsa_union_dense.union_walk_kernel`` with
``fast_walk=True``) spends about 12 ns of scalar TensorCore time per union
group finding the next set bit of the union bitset and issuing its 4-token DMA
(a TPU run: ``walk_issue``, about 23% of the kernel). The gathered path moves that
walk off the TensorCore. Two SparseCore kernels write each query group's
union, in rank order, to contiguous HBM, and the attention kernel reads each KV
tile as two contiguous copies (``union_walk_kernel(gathered=...)``, already
bitwise the walk).

``union_sc_list`` (list step) returns ``(rows, ids)`` equal to
``qsa_union_dense.union_gather_list_reference``. One query group per vector
subcore (2 SparseCores x 16 subcores), in waves. A subcore stages the group's
union words and per-word pool bases in its VMEM. It then walks the words in
order with a carried running rank. Each 32-bit word is two 16-lane halves: the
word is broadcast to the lanes (``load_gather`` with one repeated index), lane
``l`` of half ``h`` tests bit ``16 h + l``, and ``store_compressed`` writes the
set lanes' group ids and pool rows at the running rank. The rank advances by
the mask's popcount (``all_reduce_population_count``). No per-bit scalar loop.
Stores past ``min(usize, cap_ranks)`` are clamped into a slack region that the
padding fill then overwrites with ``(physw[0], -1)``.

``union_sc_gather`` (gather step) returns ``(gathered_kv, lines)`` equal to
``qsa_union_dense.union_gather_reference``. The flat rank space
``ngroups * cap_ranks`` splits into items of ``part_ranks`` consecutive ranks,
``parts`` (a power of two) per query group. Each subcore owns a contiguous run
of items. Per item, an outer ``emit_pipeline`` step stages the item's rows and
ids, and an inner ``emit_pipeline`` runs ``streams`` concurrent indirect
gathers per step, 16 ranks each:

- KV: the pool viewed in place as int32 rows of one 4-token group (a free
  in-kernel ``ref.bitcast`` of the fp8 pool; an XLA-level bitcast would
  relayout the whole pool). Gather row ``rows // 4`` (4 KB), then one
  contiguous DMA from the gather buffer to the output.
- Lines: gather the plan's bitset rows ``g * words + id // 32`` (512 B), then
  per rank ``(row >> (id % 32)) & 1`` as vector ops, zero for padding,
  repeated ``ratio`` times when ``line_matmul`` is False.

Optional dead-tile skip (``usize`` and ``kv_tile`` given): the KV rows and
lines of tiles at or past ``min(cdiv(usize * ratio, kv_tile), cap_tiles)`` are
left unwritten. The attention kernel never reads them, so its output is
unchanged, but those output rows are not the reference's. The per-item live
step counts are computed in XLA (``item_live_steps``) and staged per item.

Index math is shifts, masks and multiplies only: SparseCore loads scalars only
from SMEM, and the in-tree SparseCore kernels avoid scalar division. The pure
helpers below are shared by the kernels and the CPU emulation in
``tests/kernels/test_qwen4_exp_union_sc_gather.py``.

Status: CPU-traced and lowered to Mosaic under an abstract TPU7x mesh; the
SparseCore compile and the outputs are unverified until a TPU run.

Run-table walk (``union_walk_kernel(run_tables=..., listed_lines=...)``, a TPU run:
0.88x the listed walk, bitwise) needs two more SparseCore steps:

``union_sc_lines`` returns ``qsa_union_dense.union_lines_reference``'s lines:
the gather step's line path without the KV gather (same plan, items, streams
and helpers; optional dead-tile skip).

``union_sc_run_tables`` returns ``qsa_union_dense.union_run_tables_xla``'s
``(counts, entries)``. The flat tile space ``ngroups * cap_tiles`` splits into
pipeline steps of ``tiles_per_step`` whole tiles; each vector subcore owns a
contiguous run of steps. Per tile (``run_table_tile``), two passes over its
16-lane vectors:

- Pass A, in rank order with the run count carried across vectors: start
  flags against each rank's previous rank (a ``load_gather`` one slot back;
  ``union_run_tables_xla``'s ``~join``); the start slots are compacted into a
  run list (carried ``store_compressed``) and each rank's run number is the
  carried count plus a ``cumsum`` of the flags. The tile size is then
  appended to the run list as a sentinel.
- Pass B: each rank's run start and end are ``runs[id]`` and ``runs[id + 1]``
  (two gathers: the spec's ``cummax`` / reverse ``cummin`` scans, without a
  max scan or a reverse pass). From offset and length, the piece flags per
  size (``piece_flags``) and the entry ``(row >> log2(ratio)) << 9 | slot``;
  each size's entries are compacted into its own list in rank order.

The lists are then concatenated in size order (largest first) into a tile
buffer with 16 slots of slack, the rest zero-filled, and copied out aligned.
Every unaligned store is a full-vector ``store_compressed`` whose overflow the
next store overwrites, so lanes past a compressed store's popcount may hold
anything. The tile code takes its primitives from an ``ops`` object so that the
CPU test runs this exact code on numpy stand-ins.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

from vllm_torchtpu.kernels.qwen4_exp import qsa_union_dense as ud
from vllm_torchtpu.kernels.sparse_core import core_map_helper

LANES = 16  # SparseCore vector lanes (v7x); checked against the TPU info.
WORD_BITS = 32
# Gather VMEM budget per vector subcore (512 KiB on v7x), leaving headroom for
# the pipelines' semaphores and the compiler's own scratch.
_GATHER_VMEM_BUDGET = 400 * 1024


def _log2(n: int) -> int:
    if n <= 0 or n & (n - 1):
        raise ValueError(f"{n} is not a power of two")
    return n.bit_length() - 1


def _round_up(a: int, b: int) -> int:
    return -(-a // b) * b


def cap_ranks_for(union_cap: int, ratio: int, kv_tile: int) -> int:
    """Ranks per query group in the gathered layout: the cap in whole KV tiles."""
    return -(-union_cap * ratio // kv_tile) * (kv_tile // ratio)


# Pure index math, shared by the kernels and the CPU emulation.


def expand_half_word(word, base, w, pos, ratio):
    """One 16-bit half of union word ``w`` as lanes.

    ``word``/``base``: int32[L], the word and its pool row broadcast to the
    lanes; ``pos``: int32[L], the bit position each lane tests. Returns the
    set-bit mask, each lane's group id ``32 w + pos`` and its first pool row
    ``base + pos * ratio``.
    """
    mask = (lax.shift_right_logical(word, pos) & 1) != 0
    return mask, w * WORD_BITS + pos, base + pos * ratio


def half_positions(lane):
    """Bit positions the lanes test in a word's two halves (``lane``: 0..L-1)."""
    return lane, lane + LANES


def clamp_rank(rank, limit):
    """Where the list step stores at running rank ``rank``, and where padding starts.

    Ranks at or past ``limit = min(usize, cap_ranks)`` are padding in the
    reference. Their stores land at ``limit`` (in the slack) and the padding
    fill, which starts at the clamped final rank, overwrites them.
    """
    return jnp.minimum(rank, limit)


def list_limit(usize, cap_ranks):
    """Ranks the reference keeps live at most: ``min(usize, cap_ranks)``."""
    return jnp.minimum(usize, cap_ranks)


def padding_rows(first_bases):
    """The padding fill's row vector: word 0's pool base on every lane.

    ``first_bases``: the first 16 per-word bases (a contiguous vector load).
    Lane 0 is extracted and broadcast. a TPU run (M): ``load_gather(physw, [zeros])``
    wrote ``physw[lane]`` on lanes 1-15 on the SparseCore (a contiguous load,
    not a broadcast), so only padding lane 0 matched the reference.
    """
    return jnp.full((LANES,), first_bases[0], jnp.int32)


def pad_vectors(start, cap_ranks):
    """16-lane padding stores that cover ranks ``[start, cap_ranks)``."""
    return lax.shift_right_logical(cap_ranks - start + (LANES - 1), _log2(LANES))


def line_of(row, gid):
    """A rank's mask line: bit ``gid % 32`` of each row's word, 0 for padding."""
    live = (gid >= 0).astype(jnp.int32)
    return jnp.right_shift(row, gid & (WORD_BITS - 1)) & live


def kv_row_index(rows, ratio):
    """Row of the pool's group view (``ratio`` tokens per row) of pool row ``rows``."""
    return lax.shift_right_logical(rows, _log2(ratio))


def bits_row_index(ids, gwords):
    """Row of the flat bitset ``[ngroups * words, width]`` holding id ``ids``.

    Padding ids (-1) read one of the group's first 8 words by lane (their lines are zeroed, so the row is free):
    every padding rank reading word 0 made one 512 B row a hot spot (a TPU run: the padding hot spot dominated the
    gather). ``words`` is a multiple of 8, so the rows exist.
    """
    lane = lax.broadcasted_iota(jnp.int32, jnp.shape(ids), jnp.ndim(ids) - 1) & 7
    return gwords + jnp.where(ids >= 0, lax.shift_right_logical(jnp.maximum(ids, 0), 5), lane)


def live_rank_end(usize, *, cap_ranks, ratio, kv_tile):
    """Ranks the attention kernel reads for a query group (live tiles x tile size)."""
    groups_per_tile = kv_tile // ratio
    tiles = lax.shift_right_logical(usize * ratio + (kv_tile - 1), _log2(kv_tile))
    return jnp.minimum(tiles, cap_ranks // groups_per_tile) * groups_per_tile


@dataclasses.dataclass(frozen=True)
class GatherPlan:
    """Static work split of the gather step.

    ``parts`` items per query group of ``part_ranks`` ranks each; item ``i``
    is query group ``i >> log2(parts)``, part ``i & (parts - 1)``, flat ranks
    ``[i * part_ranks, (i + 1) * part_ranks)``. Worker (vector subcore) ``w``
    owns items ``[w * items_per_worker, (w + 1) * items_per_worker)``. Each
    inner pipeline step moves ``streams * lanes`` ranks.
    """

    ngroups: int
    cap_ranks: int
    workers: int
    lanes: int
    streams: int
    parts: int

    @property
    def chunk(self) -> int:
        return self.lanes * self.streams

    @property
    def part_ranks(self) -> int:
        return self.cap_ranks // self.parts

    @property
    def items(self) -> int:
        return self.ngroups * self.parts

    @property
    def items_per_worker(self) -> int:
        return -(-self.items // self.workers)

    @property
    def steps_per_part(self) -> int:
        return self.part_ranks // self.chunk


def gather_plan(ngroups, cap_ranks, *, workers, lanes, streams, bubble_steps=1.5):
    """Parts per group (a power of two): fewest items-per-worker x (steps + bubble)."""
    chunk = lanes * streams
    if cap_ranks % chunk:
        raise ValueError(f"cap_ranks={cap_ranks} must be a multiple of {chunk}")
    quanta = cap_ranks // chunk
    best, best_cost = None, None
    parts = 1
    while quanta % parts == 0:
        plan = GatherPlan(ngroups, cap_ranks, workers, lanes, streams, parts)
        cost = plan.items_per_worker * (plan.steps_per_part + bubble_steps)
        if best_cost is None or cost < best_cost:
            best, best_cost = plan, cost
        parts *= 2
    return best


def worker_item(worker, k, plan: GatherPlan):
    """Worker ``worker``'s ``k``-th item (items past ``plan.items`` are skipped)."""
    return worker * plan.items_per_worker + k


def staged_block(item, plan: GatherPlan):
    """Block of ``part_ranks`` rows/ids staged for an item (clamped when skipped)."""
    return jnp.minimum(item, plan.items - 1)


def item_coords(item, plan: GatherPlan):
    """(query group, part, first flat rank) of an item: no division."""
    shift = _log2(plan.parts)
    group = lax.shift_right_logical(item, shift)
    return group, item & (plan.parts - 1), item * plan.part_ranks


def window_offset(step, stream, plan: GatherPlan):
    """Offset of an inner step's stream window (``lanes`` ranks) in its item."""
    return (step * plan.streams + stream) * plan.lanes


def window_rank(rank0, step, stream, plan: GatherPlan):
    """First flat rank of an inner step's stream window."""
    return rank0 + window_offset(step, stream, plan)


def line_row(stream, lane, copy, plan: GatherPlan, repeat):
    """Row, in an inner step's lines block, of copy ``copy`` of a lane's line."""
    return (stream * plan.lanes + lane) * repeat + copy


def lines_block(item, step, plan: GatherPlan):
    """Inner step's block index into the lines output (blocks of ``chunk`` ranks)."""
    return item * plan.steps_per_part + step


def part_steps(live_end, part, plan: GatherPlan):
    """Inner steps of an item that cover live ranks (``live_end``: first dead rank)."""
    live = jnp.clip(live_end - part * plan.part_ranks, 0, plan.part_ranks)
    return lax.shift_right_logical(live, _log2(plan.chunk))


def _sc_geometry():
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        raise NotImplementedError("SparseCore is not available on this device")
    if sc.num_lanes != LANES:
        raise NotImplementedError(f"expected {LANES} lanes, got {sc.num_lanes}")
    return sc


def _mesh(sc):
    return plsc.VectorSubcoreMesh(
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,
        core_axis_name="core",
        subcore_axis_name="subcore",
    )


# List step.


def _list_kernel(
    ubits_hbm,
    physw_hbm,
    usize_hbm,
    rows_hbm,
    ids_hbm,
    ubits_v,
    physw_v,
    usize_v,
    rows_v,
    ids_v,
    *,
    ngroups,
    words,
    words_p,
    cap_ranks,
    ratio,
    waves,
    workers,
    shared_physw,
    unroll,
):
    worker = lax.axis_index(("core", "subcore"))
    lane = lax.iota(jnp.int32, LANES)
    every = lane >= 0
    if shared_physw:
        pltpu.sync_copy(physw_hbm, physw_v)

    def group(wave, carry):
        g = wave * workers + worker

        @pl.when(g < ngroups)
        def _():
            row = pl.ds(pl.multiple_of(g * words_p, LANES), words_p)
            pltpu.sync_copy(ubits_hbm.at[row], ubits_v)
            if not shared_physw:
                pltpu.sync_copy(physw_hbm.at[row], physw_v)
            lanes_g = pl.ds(pl.multiple_of(g * LANES, LANES), LANES)
            pltpu.sync_copy(usize_hbm.at[lanes_g], usize_v)
            limit = list_limit(usize_v[...][0], cap_ranks)

            def word_step(w, rank):
                at_word = jnp.full((LANES,), w, jnp.int32)
                word = plsc.load_gather(ubits_v, [at_word])
                base = plsc.load_gather(physw_v, [at_word])
                for pos in half_positions(lane):
                    mask, gid, row = expand_half_word(word, base, at_word, pos, ratio)
                    at = clamp_rank(rank, limit)
                    plsc.store_compressed(ids_v.at[pl.ds(at, LANES)], gid, mask=mask)
                    plsc.store_compressed(rows_v.at[pl.ds(at, LANES)], row, mask=mask)
                    rank = rank + plsc.all_reduce_population_count(mask)[0]
                return rank

            # The carried rank orders the stores (as dcp_sc_compact's scan).
            rank = plsc.parallel_loop(0, words, unroll=unroll, carry=jnp.int32(0))(
                word_step
            )

            start = clamp_rank(rank, limit)
            padding_id = jnp.full((LANES,), -1, jnp.int32)
            padding_row = padding_rows(physw_v[pl.ds(0, LANES)])

            def fill(i):
                at = pl.ds(start + i * LANES, LANES)
                plsc.store_compressed(ids_v.at[at], padding_id, mask=every)
                plsc.store_compressed(rows_v.at[at], padding_row, mask=every)

            plsc.parallel_loop(0, pad_vectors(start, cap_ranks), unroll=4)(fill)
            out = pl.ds(pl.multiple_of(g * cap_ranks, LANES), cap_ranks)
            pltpu.sync_copy(rows_v.at[pl.ds(0, cap_ranks)], rows_hbm.at[out])
            pltpu.sync_copy(ids_v.at[pl.ds(0, cap_ranks)], ids_hbm.at[out])

        return carry

    lax.fori_loop(0, waves, group, None)


def union_sc_list(ubits, usize, physw, *, cap_ranks, ratio=4, unroll=4):
    """SparseCore list step: ``union_gather_list_reference``'s ``(rows, ids)``.

    ``ubits`` int32[ngroups, words]: each query group's union bitset; ``usize``
    int32[ngroups]: its size (ranks at or past it are padding); ``physw``:
    int32[words] or int32[ngroups, words], the pool row of each 32-group word's
    first token (``qsa_union_dense._word_bases``); ``cap_ranks``: ranks per
    group, a multiple of 16 (``cap_ranks_for``). Returns int32[ngroups,
    cap_ranks] twice.
    """
    ngroups, words = ubits.shape
    if cap_ranks % LANES:
        raise ValueError(f"cap_ranks={cap_ranks} must be a multiple of {LANES}")
    if physw.shape not in ((words,), (ngroups, words)):
        raise ValueError(f"physw must be [{words}] or [{ngroups}, {words}]")
    sc = _sc_geometry()
    workers = sc.num_cores * sc.num_subcores
    words_p = _round_up(words, LANES)
    shared = physw.ndim == 1
    pad = words_p - words
    ubits_flat = jnp.pad(ubits.astype(jnp.int32), ((0, 0), (0, pad))).reshape(-1)
    physw_pad = ((0, pad),) if shared else ((0, 0), (0, pad))
    physw_flat = jnp.pad(physw.astype(jnp.int32), physw_pad).reshape(-1)
    usize_lanes = jnp.broadcast_to(usize.astype(jnp.int32)[:, None], (ngroups, LANES))
    flat = jax.ShapeDtypeStruct((ngroups * cap_ranks,), jnp.int32)
    rows, ids = core_map_helper.kernel(
        functools.partial(
            _list_kernel,
            ngroups=ngroups,
            words=words,
            words_p=words_p,
            cap_ranks=cap_ranks,
            ratio=ratio,
            waves=-(-ngroups // workers),
            workers=workers,
            shared_physw=shared,
            unroll=unroll,
        ),
        out_type=(flat, flat),
        mesh=_mesh(sc),
        scratch_types=(
            pltpu.VMEM((words_p,), jnp.int32),
            pltpu.VMEM((words_p,), jnp.int32),
            pltpu.VMEM((LANES,), jnp.int32),
            pltpu.VMEM((cap_ranks + 2 * LANES,), jnp.int32),
            pltpu.VMEM((cap_ranks + 2 * LANES,), jnp.int32),
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        name=f"qsa_union_sc_list_c{cap_ranks}",
    )(ubits_flat, physw_flat, usize_lanes.reshape(-1))
    return rows.reshape(ngroups, cap_ranks), ids.reshape(ngroups, cap_ranks)


def union_sc_list_pages(
    ubits, usize, page_rows, *, ratio, page_size, union_cap, kv_tile, unroll=4
):
    """``union_sc_list`` with ``union_gather_list_reference``'s arguments."""
    physw = ud._word_bases(
        page_rows,
        words=ubits.shape[1],
        page_size=page_size,
        word_tokens=WORD_BITS * ratio,
    )
    cap_ranks = cap_ranks_for(union_cap, ratio, kv_tile)
    return union_sc_list(
        ubits, usize, physw, cap_ranks=cap_ranks, ratio=ratio, unroll=unroll
    )


# Gather step.


def item_live_steps(usize, plan: GatherPlan, *, ratio, kv_tile):
    """int32[items]: each item's inner steps over tiles the attention kernel reads."""
    item = jnp.arange(plan.items, dtype=jnp.int32)
    g, part, _ = item_coords(item, plan)
    end = live_rank_end(
        usize.astype(jnp.int32)[g],
        cap_ranks=plan.cap_ranks,
        ratio=ratio,
        kv_tile=kv_tile,
    )
    return part_steps(end, part, plan)


def _window(ref, step, stream, plan: GatherPlan):
    """An inner step's stream window (``lanes`` ranks) of a staged item block."""
    return ref[pl.ds(window_offset(step, stream, plan), plan.lanes)]


def _bits_specs(ids_v, gwords, width, plan: GatherPlan):
    """One indirect bitset-row gather per stream (the rows of the window's ids)."""

    def bits_index(step, s):
        return bits_row_index(_window(ids_v, step, s, plan), gwords), 0

    return tuple(
        pl.BlockSpec(
            (pl.Indirect(plan.lanes), width), functools.partial(bits_index, s=s)
        )
        for s in range(plan.streams)
    )


def _lines_spec(item, plan: GatherPlan, repeat, width):
    """The inner step's block of ``chunk * repeat`` output lines."""
    return pl.BlockSpec(
        (plan.chunk * repeat, width), lambda step: (lines_block(item, step, plan), 0)
    )


def _store_lines(bit_buf, gids, lines_o, stream, plan: GatherPlan, repeat):
    """A stream window's lines: per rank, ``line_of`` its gathered bitset row."""
    for j in range(plan.lanes):
        line = line_of(bit_buf[pl.ds(j, 1), :], gids[j])
        for t in range(repeat):
            lines_o[pl.ds(line_row(stream, j, t, plan, repeat), 1), :] = line


def _item_steps(steps_v, plan: GatherPlan):
    """An item's inner step count: staged (dead-tile skip) or every step."""
    if steps_v is None:
        return plan.steps_per_part
    # The item's live step count, broadcast to one 16-lane block.
    return steps_v[pl.ds(0, plan.lanes)][0]


def _emit_items(item_step, worker, plan: GatherPlan, staged, steps_hbm):
    """The outer pipeline: per item, stage ``part_ranks`` of each staged array (and its step count)."""

    def item_block(k):
        return (staged_block(worker_item(worker, k, plan), plan),)

    in_specs = [pl.BlockSpec((plan.part_ranks,), item_block)] * len(staged)
    operands = list(staged)
    if steps_hbm is not None:
        in_specs.append(pl.BlockSpec((plan.lanes,), item_block))
        operands.append(steps_hbm)
    pltpu.emit_pipeline(
        item_step, grid=(plan.items_per_worker,), in_specs=tuple(in_specs)
    )(*operands)


def _gather_kernel(
    pool_hbm,
    bits_hbm,
    rows_hbm,
    ids_hbm,
    *refs,
    plan: GatherPlan,
    words,
    ratio,
    line_matmul,
    skip_dead,
):
    if skip_dead:
        steps_hbm, kv_out_hbm, lines_hbm, kv_sem = refs
    else:
        kv_out_hbm, lines_hbm, kv_sem = refs
        steps_hbm = None
    streams, lanes = plan.streams, plan.lanes
    repeat = 1 if line_matmul else ratio
    worker = lax.axis_index(("core", "subcore"))
    # The pool and the gathered KV as int32 rows of one ``ratio``-token group:
    # Mosaic packs the 4 fp8 sublanes of each (4, 256) token into one 32-bit
    # row, which is exactly XLA's T(4,128)(4,1) byte order (a free bitcast).
    pool_w = pool_hbm.bitcast(jnp.int32)
    kv_out_w = kv_out_hbm.bitcast(jnp.int32)
    group_shape = pool_w.shape[1:]

    def item_step(rows_v, ids_v, steps_v=None):
        item = worker_item(worker, pl.program_id(0), plan)
        g, part, rank0 = item_coords(item, plan)
        steps = _item_steps(steps_v, plan)

        def body(*bufs):
            step = pl.program_id(0)
            kv_bufs = bufs[:streams]
            bit_bufs = bufs[streams : 2 * streams]
            lines_o = bufs[2 * streams]
            # KV needs no vector work: the gathered rows are the output rows,
            # so each gather buffer goes straight to HBM (as dsa_gather's nope).
            copies = []
            for s in range(streams):
                out = kv_out_w.at[pl.ds(window_rank(rank0, step, s, plan), lanes)]
                copy = pltpu.make_async_copy(kv_bufs[s], out, kv_sem.at[s])
                copy.start()
                copies.append(copy)
            for s in range(streams):
                gids = _window(ids_v, step, s, plan)
                _store_lines(bit_bufs[s], gids, lines_o, s, plan, repeat)
            for copy in copies:
                copy.wait()

        def kv_index(step, s):
            rows = kv_row_index(_window(rows_v, step, s, plan), ratio)
            return (rows,) + (0,) * len(group_shape)

        kv_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(lanes), *group_shape), functools.partial(kv_index, s=s)
            )
            for s in range(streams)
        )
        bit_specs = _bits_specs(ids_v, g * words, bits_hbm.shape[1], plan)
        lines_spec = _lines_spec(item, plan, repeat, lines_hbm.shape[1])

        @pl.when(jnp.logical_and(item < plan.items, steps > 0))
        def _():
            pltpu.emit_pipeline(
                body,
                grid=(steps,),
                in_specs=kv_specs + bit_specs,
                out_specs=(lines_spec,),
            )(*([pool_w] * streams), *([bits_hbm] * streams), lines_hbm)

    _emit_items(item_step, worker, plan, (rows_hbm, ids_hbm), steps_hbm)


def gather_vmem_bytes(streams, part_ranks, *, group_bytes, width, repeat, lanes=LANES):
    """VMEM per subcore: double-buffered pipeline buffers plus staged rows/ids."""
    kv = 2 * streams * lanes * group_bytes
    bits = 2 * streams * lanes * width * 4
    lines = 2 * streams * lanes * repeat * width * 4
    staged = 2 * 2 * part_ranks * 4
    return kv + bits + lines + staged


def choose_gather_plan(
    ngroups,
    cap_ranks,
    *,
    workers,
    group_bytes,
    width,
    repeat,
    streams=None,
    lanes=LANES,
):
    """The gather's plan: 2 streams when cap_ranks and the VMEM budget allow, else 1."""
    vmem = functools.partial(
        gather_vmem_bytes,
        group_bytes=group_bytes,
        width=width,
        repeat=repeat,
        lanes=lanes,
    )
    plan_for = functools.partial(
        gather_plan, ngroups, cap_ranks, workers=workers, lanes=lanes
    )
    if streams is None:
        streams = 1
        if cap_ranks % (2 * lanes) == 0:
            trial = plan_for(streams=2)
            if vmem(2, trial.part_ranks) <= _GATHER_VMEM_BUDGET:
                streams = 2
    plan = plan_for(streams=streams)
    need = vmem(streams, plan.part_ranks)
    if need > _GATHER_VMEM_BUDGET:
        raise ValueError(f"gather buffers need {need} bytes of VMEM per subcore")
    return plan


def union_sc_gather(
    pool, bits, rows, ids, *, ratio, line_matmul, usize=None, kv_tile=None, streams=None
):
    """SparseCore gather step: ``union_gather_reference``'s ``(gathered_kv, lines)``.

    ``pool`` fp8[pages, page_size, head_axis, packing, dim] (packing x 8 bits
    = 32); ``bits`` int32[ngroups, words, 128]; ``rows``/``ids``
    int32[ngroups, cap_ranks] from the list step. Returns ``gathered_kv``
    [ngroups * cap_ranks * ratio, head_axis, packing, dim] in the pool dtype
    and ``lines`` int32[ngroups * cap_ranks * (1 if line_matmul else ratio),
    128].

    ``usize`` int32[ngroups] with ``kv_tile`` (optional): skip the KV tiles the
    attention kernel never reads (see the module docstring); those KV rows and
    lines are then unspecified. ``streams``: concurrent indirect gathers per
    pipeline step (default: 2 if the VMEM budget allows, else 1).
    """
    pages, page_size, head_axis, packing, dim = pool.shape
    ngroups, cap_ranks = rows.shape
    bgroups, words, width = bits.shape
    if ids.shape != rows.shape or bgroups != ngroups:
        raise ValueError("rows, ids and bits must share the query groups")
    if jnp.dtype(pool.dtype).itemsize * packing != 4:
        raise ValueError("the pool's packed sublanes must make 32-bit words")
    _log2(ratio)
    if page_size % ratio:
        raise ValueError("a page must hold whole groups")
    if (usize is None) != (kv_tile is None):
        raise ValueError("the dead-tile skip needs both usize and kv_tile")
    if kv_tile is not None:
        _log2(kv_tile)
        if cap_ranks % (kv_tile // ratio):
            raise ValueError("cap_ranks must be whole KV tiles")
    sc = _sc_geometry()
    repeat = 1 if line_matmul else ratio
    plan = choose_gather_plan(
        ngroups,
        cap_ranks,
        workers=sc.num_cores * sc.num_subcores,
        group_bytes=ratio * head_axis * dim * 4,
        width=width,
        repeat=repeat,
        streams=streams,
    )
    ranks = ngroups * cap_ranks
    skip = usize is not None
    operands = [
        # Splits the token dim only (major dims): free.
        pool.reshape(pages * page_size // ratio, ratio, head_axis, packing, dim),
        # Free when words % 8 == 0 (union_bits_plan pads words to 8).
        bits.reshape(ngroups * words, width),
        rows.astype(jnp.int32).reshape(-1),
        ids.astype(jnp.int32).reshape(-1),
    ]
    if skip:
        steps = item_live_steps(usize, plan, ratio=ratio, kv_tile=kv_tile)
        steps = jnp.broadcast_to(steps[:, None], (plan.items, LANES))
        operands.append(steps.reshape(-1))
    name = f"qsa_union_sc_gather_s{plan.streams}_p{plan.parts}"
    kv, lines = core_map_helper.kernel(
        functools.partial(
            _gather_kernel,
            plan=plan,
            words=words,
            ratio=ratio,
            line_matmul=line_matmul,
            skip_dead=skip,
        ),
        out_type=(
            jax.ShapeDtypeStruct((ranks, ratio, head_axis, packing, dim), pool.dtype),
            jax.ShapeDtypeStruct((ranks * repeat, width), jnp.int32),
        ),
        mesh=_mesh(sc),
        scratch_types=(pltpu.SemaphoreType.DMA((plan.streams,)),),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        name=name + ("_mm" if line_matmul else "") + ("_live" if skip else ""),
    )(*operands)
    return kv.reshape(ranks * ratio, head_axis, packing, dim), lines


# Lines step.


def _lines_kernel(
    bits_hbm,
    ids_hbm,
    *refs,
    plan: GatherPlan,
    words,
    ratio,
    line_matmul,
    skip_dead,
):
    if skip_dead:
        steps_hbm, lines_hbm = refs
    else:
        (lines_hbm,) = refs
        steps_hbm = None
    streams = plan.streams
    repeat = 1 if line_matmul else ratio
    worker = lax.axis_index(("core", "subcore"))

    def item_step(ids_v, steps_v=None):
        item = worker_item(worker, pl.program_id(0), plan)
        g, _, _ = item_coords(item, plan)
        steps = _item_steps(steps_v, plan)

        def body(*bufs):
            step = pl.program_id(0)
            lines_o = bufs[streams]
            for s in range(streams):
                gids = _window(ids_v, step, s, plan)
                _store_lines(bufs[s], gids, lines_o, s, plan, repeat)

        @pl.when(jnp.logical_and(item < plan.items, steps > 0))
        def _():
            pltpu.emit_pipeline(
                body,
                grid=(steps,),
                in_specs=_bits_specs(ids_v, g * words, bits_hbm.shape[1], plan),
                out_specs=(_lines_spec(item, plan, repeat, lines_hbm.shape[1]),),
            )(*([bits_hbm] * streams), lines_hbm)

    _emit_items(item_step, worker, plan, (ids_hbm,), steps_hbm)


def union_sc_lines(
    bits, ids, *, ratio, line_matmul, usize=None, kv_tile=None, streams=None
):
    """SparseCore lines step: ``union_lines_reference``'s lines, bit for bit.

    ``bits`` int32[ngroups, words, width]: the plan's per-row bitset; ``ids``
    int32[ngroups, cap_ranks]: the list step's group ids (-1 padding). Returns
    int32[ngroups * cap_ranks * (1 if line_matmul else ratio), width]: per
    rank, ``(bits[g, id >> 5, :] >> (id & 31)) & 1``, 0 for padding, once per
    rank (``line_matmul``) or ``ratio`` times.

    ``usize``/``kv_tile`` (optional) skip the tiles the attention kernel never
    reads, as ``union_sc_gather``'s; those lines are then unspecified.
    ``streams``: concurrent bitset-row gathers per pipeline step.
    """
    ngroups, cap_ranks = ids.shape
    bgroups, words, width = bits.shape
    if bgroups != ngroups:
        raise ValueError("ids and bits must share the query groups")
    _log2(ratio)
    if (usize is None) != (kv_tile is None):
        raise ValueError("the dead-tile skip needs both usize and kv_tile")
    if kv_tile is not None:
        _log2(kv_tile)
        if cap_ranks % (kv_tile // ratio):
            raise ValueError("cap_ranks must be whole KV tiles")
    sc = _sc_geometry()
    repeat = 1 if line_matmul else ratio
    plan = choose_gather_plan(
        ngroups,
        cap_ranks,
        workers=sc.num_cores * sc.num_subcores,
        group_bytes=0,
        width=width,
        repeat=repeat,
        streams=streams,
    )
    skip = usize is not None
    operands = [
        # Free when words % 8 == 0 (union_bits_plan pads words to 8).
        bits.reshape(ngroups * words, width),
        ids.astype(jnp.int32).reshape(-1),
    ]
    if skip:
        steps = item_live_steps(usize, plan, ratio=ratio, kv_tile=kv_tile)
        steps = jnp.broadcast_to(steps[:, None], (plan.items, LANES))
        operands.append(steps.reshape(-1))
    name = f"qsa_union_sc_lines_s{plan.streams}_p{plan.parts}"
    return core_map_helper.kernel(
        functools.partial(
            _lines_kernel,
            plan=plan,
            words=words,
            ratio=ratio,
            line_matmul=line_matmul,
            skip_dead=skip,
        ),
        out_type=jax.ShapeDtypeStruct((ngroups * cap_ranks * repeat, width), jnp.int32),
        mesh=_mesh(sc),
        compiler_params=pltpu.CompilerParams(
            use_tc_tiling_on_sc=True,
            needs_layout_passes=True,
            disable_bounds_checks=True,
        ),
        name=name + ("_mm" if line_matmul else "") + ("_live" if skip else ""),
    )(*operands)


# Run tables: pure per-vector math, shared by the kernel and the CPU emulation.


def run_sizes(max_run):
    """Piece sizes, descending: ``max_run``, ..., 2, 1 (the counts' last axis)."""
    return [1 << k for k in reversed(range(_log2(max_run) + 1))]


def prev_index(off, lane):
    """Gather index of each lane's previous rank in the staged block.

    ``off``: the vector's first index in the block. Lane 0 of the block's first
    vector would read index -1; it reads 0 instead, and its join is off anyway
    (slot 0 of a tile always starts a run).
    """
    return jnp.maximum(off - 1 + lane, 0)


def start_flags(rows, ids, prev_rows, prev_ids, slot, ratio):
    """A rank starts a run (``union_run_tables_xla``'s ``~join``).

    It does at tile slot 0 (runs end at tile boundaries), where live and
    padding meet (the ids' sign bits differ), and at a live rank that is not
    pool-contiguous with the previous rank. Padding after padding joins.
    Compares, ``&`` and ``|`` only.
    """
    kind_changes = (ids ^ prev_ids) < 0
    breaks = (ids >= 0) & (rows != prev_rows + ratio)
    return (slot == 0) | kind_changes | breaks


def run_ids(start, runs_before, cumsum):
    """Each rank's run number in its tile: the runs started before the vector plus the inclusive count of starts, less 1."""
    return runs_before + cumsum(start.astype(jnp.int32)) - 1


def piece_flags(slot, run_start, run_end, max_run):
    """Per size (descending), whether a piece of that size starts at the rank.

    A run of length ``L`` is cut into ``L >> log2(max_run)`` pieces of
    ``max_run``, then one piece per set bit of the remainder, largest first
    (``union_run_tables_xla``'s rule, with shifts and masks for the divisions:
    lengths and offsets are never negative).
    """
    log_max = _log2(max_run)
    length = run_end - run_start
    offset = slot - run_start
    full = lax.shift_left(lax.shift_right_logical(length, log_max), log_max)
    rem, past = length - full, offset - full
    flags = [(offset < full) & ((offset & (max_run - 1)) == 0)]
    for size in run_sizes(max_run)[1:]:
        flags.append(((rem & size) != 0) & (past == (rem & ~((size << 1) - 1))))
    return flags


def run_entry(rows, slot, ratio):
    """A piece's entry: its first pool group (``rows >> log2(ratio)``) above its 9-bit tile slot."""
    return (lax.shift_right_logical(rows, _log2(ratio)) << 9) | slot


def list_base(size_index, gpt):
    """Start of a size's piece list in the lists scratch (each list has 16 slots of slack)."""
    return size_index * (gpt + LANES)


def counts_vector(counts, lane):
    """The tile's piece counts on lanes 0..nsizes-1, zeros above."""
    vec = jnp.zeros((LANES,), jnp.int32)
    for c, n in enumerate(counts):
        vec = jnp.where(lane == c, n, vec)
    return vec


class ScOps:
    """The SparseCore primitives ``run_table_tile`` uses.

    The CPU test passes a numpy stand-in with the same methods, so its
    emulation runs ``run_table_tile`` itself.
    """

    @staticmethod
    def lane():
        return lax.iota(jnp.int32, LANES)

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
    def gather(ref, index):
        return plsc.load_gather(ref, [index])

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


def carry_pad(nsizes, even_carry=True):
    """Dummy counts that make pass B's tuple carry even: 1 for an odd carry above 1 (R4: 3 sizes, R16: 5), else 0.

    a TPU run (M, TPU): R4 tables at KV1024 failed and halted the device on mixed-size tiles, while R1/R2/R8/R32 (carries of
    1/2/4/6 counts) passed and the CPU emulation of R4 passed. R4 was the only odd carry above 1 tested. Hypothesis
    (I, unconfirmed): an odd scalar-tuple carry in ``parallel_loop(unroll=2)`` miscompiles on the SparseCore. The
    dummy is carried unchanged and never read. ``even_carry=False`` keeps the old carry (TPU diagnosis only).
    """
    return int(even_carry and nsizes > 1 and nsizes % 2 == 1)


def run_table_tile(ops, rows, ids, counts_out, entries_out, t, scratch, *, gpt, ratio, max_run,
                   even_carry=True, copy_unroll=1):
    """One KV tile's run table: tile ``t`` of the staged ``rows``/``ids`` block.

    Writes the tile's ``gpt`` entries at ``entries_out[t * gpt]`` and its counts
    vector at ``counts_out[t * 16]``. ``scratch``: (run list [gpt + 16], run
    ids [gpt], size lists [nsizes * (gpt + 16)], tile [gpt + 16]), reused
    across tiles (only slots written for this tile are read).
    """
    runs, rid_v, lists, tile = scratch
    lane = ops.lane()
    every = lane >= 0
    base = t * gpt
    vectors = gpt // LANES
    nsizes = len(run_sizes(max_run))
    pad = carry_pad(nsizes, even_carry)

    # Pass A: start flags and run numbers (cumsum), the run count carried
    # across the tile's vectors; start slots compacted into the run list.
    def scan(v, runs_before):
        off = base + v * LANES
        slot = v * LANES + lane
        at = prev_index(off, lane)
        start = start_flags(
            ops.load(rows, off),
            ops.load(ids, off),
            ops.gather(rows, at),
            ops.gather(ids, at),
            slot,
            ratio,
        )
        ops.store_compressed(runs, runs_before, slot, start)
        rid = run_ids(start, runs_before, ops.cumsum)
        ops.store(rid_v, v * LANES, rid)
        return rid[LANES - 1] + 1

    nruns = ops.loop(0, vectors, carry=jnp.int32(0), unroll=2)(scan)
    # Sentinel: the last run ends at the tile's end.
    ops.store_compressed(runs, nruns, jnp.full((LANES,), gpt, jnp.int32), every)

    # Pass B: a rank's run starts at runs[id] and ends at the next run's start;
    # pieces per size, compacted into each size's list in rank order.
    def place(v, counts):
        off = base + v * LANES
        slot = v * LANES + lane
        rid = ops.load(rid_v, v * LANES)
        run_start = ops.gather(runs, rid)
        run_end = ops.gather(runs, rid + 1)
        flags = piece_flags(slot, run_start, run_end, max_run)
        entry = run_entry(ops.load(rows, off), slot, ratio)
        new = []
        for c, (flag, n) in enumerate(zip(flags, counts[:nsizes])):
            ops.store_compressed(lists, list_base(c, gpt) + n, entry, flag)
            new.append(n + ops.popcount(flag))
        return tuple(new) + tuple(counts[nsizes:])  # the dummy, unchanged

    counts = ops.loop(0, vectors, carry=(jnp.int32(0),) * (nsizes + pad), unroll=2)(place)[:nsizes]

    # Concatenate the size lists (largest first), zero the rest, copy out.
    at = jnp.int32(0)
    for c, n in enumerate(counts):

        def copy(i, c=c, at=at):
            vec = ops.load(lists, list_base(c, gpt) + i * LANES)
            ops.store_compressed(tile, at + i * LANES, vec, every)

        ops.loop(0, pad_vectors(0, n), unroll=copy_unroll)(copy)
        at = at + n
    zeros = jnp.zeros((LANES,), jnp.int32)

    def fill(i):
        ops.store_compressed(tile, at + i * LANES, zeros, every)

    ops.loop(0, pad_vectors(at, gpt), unroll=copy_unroll)(fill)

    def out(i):
        ops.store(entries_out, base + i * LANES, ops.load(tile, i * LANES))

    ops.loop(0, vectors, unroll=4)(out)
    ops.store(counts_out, t * LANES, counts_vector(counts, lane))


@dataclasses.dataclass(frozen=True)
class TablePlan:
    """Static work split of the run-table step.

    The flat tile space ``ngroups * tiles`` (tile ``i`` is query group
    ``i // tiles``'s tile ``i % tiles``, ranks ``[i * gpt, (i + 1) * gpt)``)
    splits into ``steps`` pipeline steps of ``tiles_per_step`` tiles; worker
    ``w`` owns steps ``[w * steps_per_worker, (w + 1) * steps_per_worker)``.
    """

    ngroups: int
    tiles: int
    gpt: int
    workers: int
    tiles_per_step: int

    @property
    def total(self) -> int:
        return self.ngroups * self.tiles

    @property
    def steps(self) -> int:
        return self.total // self.tiles_per_step

    @property
    def steps_per_worker(self) -> int:
        return -(-self.steps // self.workers)


def table_scratch_sizes(gpt, nsizes):
    """int32 slots of ``run_table_tile``'s scratch: run list (with its sentinel), run ids, size lists, tile.

    The run list, each size list and the tile take 16 slots of slack for the
    overflow of their last full-vector store.
    """
    return (gpt + LANES, gpt, nsizes * (gpt + LANES), gpt + LANES)


def table_vmem_bytes(gpt, tiles_per_step, nsizes):
    """VMEM per subcore: double-buffered rows/ids/entries/counts blocks plus the tile scratch."""
    blocks = 2 * 4 * tiles_per_step * (3 * gpt + LANES)
    return blocks + 4 * sum(table_scratch_sizes(gpt, nsizes))


def table_plan(
    ngroups, tiles, gpt, *, workers, tiles_per_step=None, max_tiles_per_step=16,
    bubble_tiles=0.5,
):
    """Tiles per step: a divisor of the tile count, fewest steps-per-worker x (tiles + bubble)."""
    total = ngroups * tiles
    if tiles_per_step is not None:
        if total % tiles_per_step:
            raise ValueError(f"tiles_per_step={tiles_per_step} must divide {total} tiles")
        return TablePlan(ngroups, tiles, gpt, workers, tiles_per_step)
    best, best_cost = None, None
    for k in range(1, max_tiles_per_step + 1):
        if total % k:
            continue
        plan = TablePlan(ngroups, tiles, gpt, workers, k)
        cost = plan.steps_per_worker * (k + bubble_tiles)
        if best_cost is None or cost < best_cost:
            best, best_cost = plan, cost
    return best


def worker_steps(worker, plan: TablePlan):
    """(first step, step count) of a worker's contiguous run of steps: no division."""
    first = worker * plan.steps_per_worker
    return first, jnp.clip(plan.steps - first, 0, plan.steps_per_worker)


def _tables_kernel(
    rows_hbm, ids_hbm, counts_hbm, entries_hbm, *scratch, plan: TablePlan, ratio, max_run,
    even_carry=True, copy_unroll=1,
):
    worker = lax.axis_index(("core", "subcore"))
    first, count = worker_steps(worker, plan)
    k, gpt = plan.tiles_per_step, plan.gpt

    def step(rows_b, ids_b, counts_b, entries_b):
        def tile(t, carry):
            run_table_tile(
                ScOps, rows_b, ids_b, counts_b, entries_b, t, scratch,
                gpt=gpt, ratio=ratio, max_run=max_run, even_carry=even_carry, copy_unroll=copy_unroll,
            )
            return carry

        lax.fori_loop(0, k, tile, None)

    def block(i):
        return (first + i,)

    @pl.when(count > 0)
    def _():
        pltpu.emit_pipeline(
            step,
            grid=(count,),
            in_specs=(pl.BlockSpec((k * gpt,), block),) * 2,
            out_specs=(
                pl.BlockSpec((k * LANES,), block),
                pl.BlockSpec((k * gpt,), block),
            ),
        )(rows_hbm, ids_hbm, counts_hbm, entries_hbm)


def check_run_table_args(rows_shape, ids_shape, *, ratio, kv_tile, max_run):
    """``union_run_tables_xla``'s checks, plus 16-lane tiles. Returns (gpt, tiles)."""
    if max_run & (max_run - 1) or not 1 <= max_run <= 32:
        raise ValueError("max_run must be a power of two from 1 to 32")
    if tuple(ids_shape) != tuple(rows_shape):
        raise ValueError("rows and ids must have the same shape")
    _log2(ratio)
    _log2(kv_tile)
    gpt = kv_tile // ratio
    _, cap_ranks = rows_shape
    if cap_ranks % gpt or gpt > 512:
        raise ValueError("the list must hold whole tiles of at most 512 groups")
    if gpt % LANES:
        raise ValueError(f"a tile must hold a multiple of {LANES} groups")
    return gpt, cap_ranks // gpt


def union_sc_run_tables(rows, ids, *, ratio, kv_tile, max_run=8, tiles_per_step=None, even_carry=True,
                        copy_unroll=1):
    """SparseCore run-table step: ``union_run_tables_xla``'s ``(counts, entries)``, bit for bit.

    ``rows``/``ids`` int32[ngroups, cap_ranks]: the list step's. Returns
    ``counts`` int32[ngroups, cap_tiles, log2(max_run) + 1] (pieces per size,
    sizes descending) and ``entries`` int32[ngroups, cap_ranks] (per tile, each
    size's pieces in rank order, sizes descending, ``(pool group << 9) |
    slot``; unused slots 0). ``tiles_per_step``: tiles per pipeline step (a
    divisor of ``ngroups * cap_tiles``; default from ``table_plan``). ``even_carry``: see ``carry_pad`` (keep
    the default; False only to diagnose the old R4/R16 carry on TPU). ``copy_unroll``: unroll of the tile's
    list-concatenation and zero-fill loops (dynamic trip counts). 1, the default since a TPU run (M): at 2, R4 tables at
    KV512 and KV1024 put the size-2 list late (overlapping full-vector stores reordered, I) and a walk on them halted
    the device; at 1 every R4 case matches the host reference. 2 is kept only to reproduce that on TPU.
    """
    gpt, tiles = check_run_table_args(
        rows.shape, ids.shape, ratio=ratio, kv_tile=kv_tile, max_run=max_run
    )
    ngroups, cap_ranks = rows.shape
    nsizes = len(run_sizes(max_run))
    sc = _sc_geometry()
    plan = table_plan(
        ngroups, tiles, gpt, workers=sc.num_cores * sc.num_subcores,
        tiles_per_step=tiles_per_step,
    )
    need = table_vmem_bytes(gpt, plan.tiles_per_step, nsizes)
    if need > _GATHER_VMEM_BUDGET:
        raise ValueError(f"run-table buffers need {need} bytes of VMEM per subcore")
    counts, entries = core_map_helper.kernel(
        functools.partial(_tables_kernel, plan=plan, ratio=ratio, max_run=max_run, even_carry=even_carry,
                          copy_unroll=copy_unroll),
        out_type=(
            jax.ShapeDtypeStruct((plan.total * LANES,), jnp.int32),
            jax.ShapeDtypeStruct((ngroups * cap_ranks,), jnp.int32),
        ),
        mesh=_mesh(sc),
        scratch_types=tuple(
            pltpu.VMEM((n,), jnp.int32) for n in table_scratch_sizes(gpt, nsizes)
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, needs_layout_passes=False
        ),
        name=f"qsa_union_sc_run_tables_g{gpt}_r{max_run}_k{plan.tiles_per_step}"
        + ("_ec" if carry_pad(nsizes, even_carry) else "") + ("" if copy_unroll == 1 else f"_cu{copy_unroll}"),
    )(rows.astype(jnp.int32).reshape(-1), ids.astype(jnp.int32).reshape(-1))
    counts = counts.reshape(plan.total, LANES)[:, :nsizes]
    return counts.reshape(ngroups, tiles, nsizes), entries.reshape(ngroups, cap_ranks)


# SparseCore selection bitset (lane_attn.md option A): union_bits_plan's per-row bitset built on the SparseCore, off
# the TensorCore (a TPU run xprof, M: the TensorCore bitset was 1.15 ms of an 11.85 ms FA call, serial before attention).


def bits_slab(ops, sel, lengths, out, *, k, words, ratio, groups):
    """One 16-row slab's selection bitset, as ``bsp._selection_bitset`` builds it for those rows.

    ``sel`` [16 * k] (the slab's rows of the plan's selections, row-major, invalid -1), ``lengths`` [16] (the plan's
    bit lengths), ``out`` [words * 16] (word-major, a row per lane). Each row adds bit ``g % 32`` of word ``g // 32``
    for every selected group ``g`` below ``groups`` and for its partial tail group (``lengths // ratio`` when
    ``lengths % ratio != 0``). Adds, not ORs, so duplicates give the TensorCore kernel's bits exactly. A scatter's
    16 lanes are 16 different rows, so its addresses never collide.
    """
    lane = ops.lane()
    zeros = jnp.zeros((LANES,), jnp.int32)
    log_ratio = _log2(ratio)

    def clear(i):
        ops.store(out, i * LANES, zeros)

    ops.loop(0, words, unroll=4)(clear)

    def add(value, valid):
        safe = jnp.maximum(value, 0)
        word = lax.shift_right_logical(safe, 5)
        bit = lax.shift_left(jnp.ones_like(safe), jnp.bitwise_and(safe, 31))
        ops.addupdate_scatter(out, word * LANES + lane, bit, valid)

    def column(j):
        value = ops.gather(sel, lane * k + j)
        add(value, (value >= 0) & (value < groups))

    ops.loop(0, k, unroll=2)(column)
    length = ops.load(lengths, 0)
    tail = lax.shift_right_logical(jnp.maximum(length, 0), log_ratio)
    add(tail, (length > 0) & (jnp.bitwise_and(length, ratio - 1) != 0) & (tail < groups))


def bits_vmem_bytes(k, words):
    """VMEM per subcore: double-buffered selection, length and output blocks."""
    return 2 * 4 * (LANES * k + LANES + words * LANES)


def _bits_kernel(sel_hbm, len_hbm, out_hbm, *, items, k, words, ratio, groups, workers):
    worker = lax.axis_index(("core", "subcore"))
    per = -(-items // workers)
    first = worker * per
    count = jnp.clip(items - first, 0, per)

    def step(sel_b, len_b, out_b):
        bits_slab(ScOps, sel_b, len_b, out_b, k=k, words=words, ratio=ratio, groups=groups)

    def block(i):
        return (first + i,)

    @pl.when(count > 0)
    def _():
        pltpu.emit_pipeline(
            step,
            grid=(count,),
            in_specs=(pl.BlockSpec((LANES * k,), block), pl.BlockSpec((LANES,), block)),
            out_specs=pl.BlockSpec((words * LANES,), block),
        )(sel_hbm, len_hbm, out_hbm)


def bits_slabs_to_plan(flat, *, tokens, group, words):
    """[items * words * 16] slab-major SparseCore output -> ``union_bits_plan``'s ``bits`` [T // group, words, 128]."""
    nslab = group // LANES
    bits = flat.reshape(tokens // group, nslab, words, LANES).transpose(0, 2, 1, 3)
    return jnp.pad(bits.reshape(tokens // group, words, group), ((0, 0), (0, 0), (0, 128 - group)))


def union_sc_bits(sel, bit_lengths, *, ratio, groups, words, group):
    """SparseCore build of ``union_bits_plan``'s ``bits``: int32[T // group, words, 128], bit for bit.

    ``sel`` int32[T, k] (the plan's selections: invalid entries -1, ids below ``groups``), ``bit_lengths`` int32[T]
    (the plan's lengths, tail dropped where the tail is already selected). One work item per 16-row slab, a
    contiguous run of items per vector subcore; the slab-major output is regrouped by XLA as the TensorCore plan
    regroups its own.
    """
    tokens, k = sel.shape
    if group % LANES or 128 % group or tokens % group:
        raise ValueError(f"the SparseCore bitset needs groups of whole 16-row slabs that split the {tokens} rows")
    if words % 8 or groups > words * WORD_BITS:
        raise ValueError("words must be a multiple of 8 covering every group")
    _log2(ratio)
    need = bits_vmem_bytes(k, words)
    if need > _GATHER_VMEM_BUDGET:
        raise ValueError(f"the SparseCore bitset needs {need} bytes of VMEM per subcore")
    sc = _sc_geometry()
    workers = sc.num_cores * sc.num_subcores
    items = tokens // LANES
    flat = core_map_helper.kernel(
        functools.partial(_bits_kernel, items=items, k=k, words=words, ratio=ratio, groups=groups, workers=workers),
        out_type=jax.ShapeDtypeStruct((items * words * LANES,), jnp.int32),
        mesh=_mesh(sc),
        scratch_types=(),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True, needs_layout_passes=False),
        name=f"qsa_union_sc_bits_k{k}_w{words}",
    )(sel.astype(jnp.int32).reshape(-1), bit_lengths.astype(jnp.int32))
    return bits_slabs_to_plan(flat, tokens=tokens, group=group, words=words)
