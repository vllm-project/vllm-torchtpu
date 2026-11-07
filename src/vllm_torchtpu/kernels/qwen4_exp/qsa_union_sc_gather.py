# SPDX-License-Identifier: Apache-2.0
"""SparseCore union gather, step 2: the list and gather steps on v7x SparseCore.

P-ATTN. The walking D2 kernel (``qsa_union_dense.union_walk_kernel`` with
``fast_walk=True``) spends about 12 ns of scalar TensorCore time per union
group finding the next set bit of the union bitset and issuing its 4-token DMA
(i1j: ``walk_issue``, about 23% of the kernel). The gathered path moves that
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
    Lane 0 is extracted and broadcast. i1l (M): ``load_gather(physw, [zeros])``
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

    Padding ids (-1) read the group's word 0; their lines are zeroed.
    """
    return gwords + lax.shift_right_logical(jnp.maximum(ids, 0), 5)


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
        gwords = g * words
        if skip_dead:
            # The item's live step count, broadcast to one 16-lane block.
            steps = steps_v[pl.ds(0, lanes)][0]
        else:
            steps = plan.steps_per_part

        def window(ref, step, stream):
            return ref[pl.ds(window_offset(step, stream, plan), lanes)]

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
                gids = window(ids_v, step, s)
                for j in range(lanes):
                    line = line_of(bit_bufs[s][pl.ds(j, 1), :], gids[j])
                    for t in range(repeat):
                        lines_o[pl.ds(line_row(s, j, t, plan, repeat), 1), :] = line
            for copy in copies:
                copy.wait()

        def kv_index(step, s):
            rows = kv_row_index(window(rows_v, step, s), ratio)
            return (rows,) + (0,) * len(group_shape)

        def bits_index(step, s):
            return bits_row_index(window(ids_v, step, s), gwords), 0

        kv_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(lanes), *group_shape), functools.partial(kv_index, s=s)
            )
            for s in range(streams)
        )
        bit_specs = tuple(
            pl.BlockSpec(
                (pl.Indirect(lanes), bits_hbm.shape[1]),
                functools.partial(bits_index, s=s),
            )
            for s in range(streams)
        )
        lines_spec = pl.BlockSpec(
            (plan.chunk * repeat, lines_hbm.shape[1]),
            lambda step: (lines_block(item, step, plan), 0),
        )

        @pl.when(jnp.logical_and(item < plan.items, steps > 0))
        def _():
            pltpu.emit_pipeline(
                body,
                grid=(steps,),
                in_specs=kv_specs + bit_specs,
                out_specs=(lines_spec,),
            )(*([pool_w] * streams), *([bits_hbm] * streams), lines_hbm)

    def item_block(k):
        return (staged_block(worker_item(worker, k, plan), plan),)

    in_specs = [pl.BlockSpec((plan.part_ranks,), item_block)] * 2
    operands = [rows_hbm, ids_hbm]
    if skip_dead:
        in_specs.append(pl.BlockSpec((lanes,), item_block))
        operands.append(steps_hbm)
    pltpu.emit_pipeline(
        item_step, grid=(plan.items_per_worker,), in_specs=tuple(in_specs)
    )(*operands)


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
