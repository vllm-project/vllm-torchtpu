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
"""The push-dispatch fused EP MoE kernel.

v2's kernel (kernels/fused_moe/v2/kernel.py) with the activation all-gather
replaced by in-kernel pushes of each shard's routed rows into the
destination's expert slab; see `_build_push_moe_kernel`. Everything this
module does not redefine -- the expert FFN bodies, the wire epilogue, the
barrier, the table counts -- is v2's, imported unchanged.
"""

import threading
from typing import cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.fused_moe.v2 import host
from vllm_torchtpu.kernels.fused_moe.v2.host import (
    FP4,
    FP8,
    HIDDEN_LANE_BLOCK,
    OUT_PARITIES,
    PACK4,
    QB4,
    ROWBLK,
)
from vllm_torchtpu.kernels.fused_moe.v2.kernel import (
    _REF_BITCAST_JAX_VERSION,
    ACT_FNS,
    COLLECTIVE_ID,
    COMBINE_CHUNK,
    COMBINE_GROUP,
    COMBINE_LEAD,
    COMBINE_PARITIES,
    HEIGHT_RUNGS,
    N_PREFETCH,
    N_PUSH_TABLES,
    N_SCALE_TABLES,
    N_SLAB_TABLES,
    N_TRANSPORT_TABLES,
    N_WEIGHT_SLABS,
    WEIGHT_DMA_PRIORITY,
    _all_pairs_barrier,
    _and_nonempty,
    _rows_wait,
    _slot_copy,
    _tile_row_scales,
    combine_step_tokens,
    expert_ffn_bf16,
    expert_ffn_blockscale,
    expert_ffn_fp8,
    expert_ffn_int8,
    wire_row_and_scale,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Two more scalar-prefetch tables than v2: the dispatch segment table and its
# meta row (host.push_dispatch_segments).
N_DISPATCH_TABLES = 2


def tile_height_rungs(tile_m):
    """v2's equally spaced tile-height ladder, emitted on every build.

    v2 emits the ladder only where a routed expert is expected to fit in one
    tile (rows_alloc <= ep * g_local * tile_m); this build always emits it,
    which is what the push dispatch was measured with.
    """
    rungs = envs.MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE or HEIGHT_RUNGS
    step = tile_m // rungs
    if rungs < 2 or step % ROWBLK or step * rungs != tile_m:
        return (tile_m,)
    return tuple(range(step, tile_m + 1, step))


def _build_push_moe_kernel(
    *,
    g_local,
    capacity,
    hidden,
    inter,
    ep,
    weight_format=host.WeightFormat.FP8,
    rhs_qb=QB4,
    ragged_rows_alloc=None,
    act_fn="silu",
    has_w1_bias=False,
    has_w2_bias=False,
    pad_scale_rows=False,
    logical_intermediate_size=None,
    push_tokens,
    sc_prepush,
):
    """Build the pallas_call and the function that invokes it.

    The v2 kernel with the activation all-gather replaced by push dispatch.
    `push_tokens` is t_local: the lhs operand is only this shard's rows,
    sorted by global expert (host.local_sort_activations), and every shard
    pushes each (destination, expert) run of them as one DMA straight into
    the destination's expert slab (`dispatch_hbm`, aliased to the `slab`
    operand), in the destination's expert visit order, signalling the
    destination's per-local-expert receive semaphore. Visit positions
    [0, sc_prepush) were pushed by the SparseCore before this kernel
    (sc_dispatch.sc_dispatch_chunk_2core), so their tiles start with no wait;
    the TensorCore pushes the rest a visit position ahead of its own compute,
    and waits an expert's semaphore for its pair count before loading its
    first tile. Tiles then read their rows as one contiguous slab copy, so no
    token-gather table exists. Only data movement changes: same rows, same
    math as the all-gather build.

    `pad_scale_rows` takes the four-bit scale tables as [G, rows, N] with
    rows = host.fp4_scale_rows(K, rhs_qb, padded=True): K / rhs_qb blocks
    padded to the f32 sublane tile, the [G * rows, N] row-padded flat operand
    viewed per expert. Each slot then stages an expert's whole padded table
    and the FFN reads only its first K / rhs_qb rows, so the result is the
    unpadded build's bit for bit.
    """
    if pad_scale_rows and weight_format != host.WeightFormat.FP4:
        raise ValueError(
            "pad_scale_rows describes the four-bit block-scale tables; "
            f"weight format {weight_format!r} has none"
        )
    if act_fn not in ACT_FNS:
        raise NotImplementedError(
            f"the fused EP MoE kernel fuses {ACT_FNS}; got {act_fn!r}"
        )
    # The kernel reads its tiles out of the pushed expert slab, and ships
    # results over an all-to-all carrying unweighted rows; it returns
    # (arrival rows, arrival scales f32). The element type of the staging and
    # of the wire is the weight format's, not a literal here.
    form = host.weight_form(weight_format)
    rhs_packed4 = weight_format == host.WeightFormat.FP4
    if (
        logical_intermediate_size is not None
        and int(logical_intermediate_size) != logical_intermediate_size
    ):
        raise ValueError("logical intermediate size must be an integer")
    effective_inter = (
        inter if logical_intermediate_size is None else int(logical_intermediate_size)
    )
    trim_gate_up_padding = effective_inter != inter
    if trim_gate_up_padding and (
        not rhs_packed4
        or rhs_qb != 512
        or has_w1_bias
        or effective_inter <= 0
        or effective_inter >= inter
        or effective_inter % 256
        or inter - effective_inter >= rhs_qb
    ):
        raise ValueError(
            "logical intermediate optimization requires FP4/G512, no W1 bias, "
            "a positive 256-aligned logical size, and less than one trailing G512 block"
        )
    # This is the last N column read, not the end of the physical allocation.
    w1_last_valid_column = inter + effective_inter

    # The weight prefetch distance and the slot count it needs are chosen
    # below, once the shapes they are sized against have been checked: they
    # are a property of THIS build and not of the process, so everything
    # here reads the two locals and nothing reads a module constant.
    #
    # Everything from here to the ragged-stride check below tests a value
    # that came IN from the caller -- a model shape, a mesh width, a serving
    # config -- so each one refuses rather than asserts. An assert is the
    # right tool for this function's own arithmetic, like the slot relation
    # below, and the wrong tool for the caller's operands: `python
    # -O` and PYTHONOPTIMIZE are ordinary things to find in a container
    # image, and under them an assert lets the illegal configuration build
    # and be cached.
    if capacity % ROWBLK:
        raise ValueError(
            f"tile height {capacity} is not a whole number of {ROWBLK}-row "
            "blocks, which is the unit every transport here moves"
        )
    if rhs_packed4:
        # Tested before the modulo below rather than after it: every other
        # guard in this span was converted from an assert so that a caller's
        # operand is refused by name, and a block size of zero divided by
        # the operand first and came back as a ZeroDivisionError.
        if rhs_qb < 1:
            raise ValueError(
                f"the four-bit block size is {rhs_qb}; it is the number of "
                "contraction rows one weight scale covers and there is no "
                "block smaller than one row"
            )
        if hidden % rhs_qb or inter % rhs_qb:
            raise ValueError(
                f"the four-bit block size {rhs_qb} has to divide both "
                f"hidden={hidden} and inter={inter}; the kernel blocks BOTH "
                "matmuls at one block size"
            )
        # Packed rows per k-block must land on the u32 sublane tile.
        if rhs_qb % (host.U32_SUBLANE_TILE * PACK4):
            raise ValueError(
                f"the four-bit block size {rhs_qb} is not a whole number of "
                f"the packed-weight row tile "
                f"({host.U32_SUBLANE_TILE * PACK4} rows): {PACK4} four-bit "
                f"values pack into a 32-bit word and those words tile to "
                f"{host.U32_SUBLANE_TILE} sublanes"
            )
    # The widening chunk has to tile both contractions exactly.
    if weight_format == host.WeightFormat.INT8 and (
        hidden % host.WIDEN_KCHUNK or inter % host.WIDEN_KCHUNK
    ):
        raise ValueError(
            f"integer weights widen one {host.WIDEN_KCHUNK}-row "
            f"contraction chunk at a time, which has to divide both "
            f"hidden={hidden} and inter={inter}"
        )
    # True-length remote pushes: out_vm, contrib and recv take the per-row
    # [rows, lane blocks, 128] geometry, which row-granular offsets require.
    if (
        hidden < HIDDEN_LANE_BLOCK
        or hidden % HIDDEN_LANE_BLOCK
        or hidden > host.HIDDEN_MAX_BLOCKS * HIDDEN_LANE_BLOCK
    ):
        raise ValueError(
            f"hidden {hidden} is not between one and "
            f"{host.HIDDEN_MAX_BLOCKS} whole {HIDDEN_LANE_BLOCK}-lane "
            "blocks, which the per-row transport geometry requires"
        )
    # The intermediate axis had no floor at all, so an expert whose FFN has
    # no intermediate channels built a program and cached it: the estimate
    # for it clears the budget comfortably and every divisibility check
    # passes, because zero is a whole number of everything. The floor is one
    # lane block, the same rule and the same reason as the hidden axis
    # above -- an intermediate narrower than the vector unit's lane count is
    # not an FFN, and the diagnostic an operator eventually receives names a
    # reduction or a reshape rather than the weight that was empty.
    if inter < HIDDEN_LANE_BLOCK:
        raise ValueError(
            f"inter {inter} is narrower than one {HIDDEN_LANE_BLOCK}-lane "
            "block; an expert whose FFN has no intermediate channels to "
            "speak of is not a layer this kernel can serve"
        )
    lane_blocks = host.row_lane_blocks(hidden)
    has_scales = form.has_scales
    has_act_scale = form.quantized_activations
    # The prefetch schedule this build can afford: the deepest runway whose
    # nbuf copies of one expert still leave the rest of the buffers inside
    # the budget, walked down from the ceiling one slot at a time. Every
    # operand it is sized against has been checked above, which is why the
    # choice is made here and not at the head of the function.
    slots = host.fit_weight_slots(
        g_local,
        capacity,
        hidden,
        inter,
        weight_format=weight_format,
        rhs_qb=rhs_qb,
        has_w1_bias=has_w1_bias,
        has_w2_bias=has_w2_bias,
    )
    prefetch_distance = slots.distance
    nbuf = slots.nbuf
    # The relation the whole schedule rests on: a refill writes its slot
    # while the readers of `distance` earlier experts are still live, so
    # distance + 1 consecutive experts have to occupy distinct slots. This
    # function's own arithmetic, hence an assert.
    assert 0 <= prefetch_distance < nbuf, (
        f"weight prefetch distance {prefetch_distance} must satisfy "
        f"0 <= distance < nbuf={nbuf}, so that distance + 1 consecutive "
        "experts occupy distinct weight slots"
    )
    # Distance zero has no runway: one slot, and a visit starts its own
    # weights and then waits on them. Every prefetch site below is switched
    # on this rather than deleted, because the two schedules differ only in
    # WHERE the start happens -- a visit ahead, or at the visit's own head.
    prefetches = prefetch_distance > 0

    # Whether each expert's weights are waited by the block that first reads
    # it rather than waiting at the beginning of each expert's visit.
    # fp4, fp8 expert FFN are running different code paths. This flag can
    # be activated only for fp4 expert FFN now. But the idea should be
    # applicable to fp8 expert FFN as well.
    streams_weights = rhs_packed4
    # Whether issue async copy the NEXT expert's weights as soon as we finish
    # reading the current expert's weights, instead of at the beginning of the
    # next expert's visit.
    #
    # The overlap is free in VMEM, because the next start writes the slot its
    # reader has just finished with: for example gate/up is dead from the
    # last gate/up block of an expert's last tile, which is overlapped with
    # a whole down matmul plus an epilogue.
    starts_next_weights = streams_weights and not prefetches
    if rhs_packed4:
        # Asked here, beside the other device reads, because the packed
        # four-bit layout is the one place a written-down device fact and a
        # queried one have to agree: the block-size check above uses the
        # constant and the buffers the accounting just sized used the
        # record.
        host.check_u32_sublane_tile()
    # The only thing standing between an over-budget model and a kernel that
    # will not fit. It must not be strippable. What it refuses is now only
    # what the descent above could not rescue: the figures it quotes are the
    # ones at distance zero, where the weight slabs are a single expert and
    # there is no runway left to give back.
    if not slots.fits:
        raise ValueError(
            f"the kernel's VMEM buffers need "
            f"{slots.vmem_bytes / 2**20:.1f}MiB, over the "
            f"{slots.limit / 2**20:.1f}MiB budget, for {g_local} local experts "
            f"of {hidden}x{inter} at capacity={capacity} with the weight "
            f"prefetch already given up (nbuf={nbuf}, one expert of weight "
            f"slabs). Nothing further is bought by shortening the runway; "
            f"the remaining levers are the tile height, the expert shape and "
            f"the number of local experts"
        )
    if prefetch_distance < host.MAX_WEIGHT_PREFETCH_DISTANCE:
        logger.info(
            "fused EP MoE: %d local experts of %dx%d at capacity=%d fit at "
            "prefetch distance %d (%d weight slots, %.1fMiB of a %.1fMiB "
            "budget), below the ceiling of %d",
            g_local,
            hidden,
            inter,
            capacity,
            prefetch_distance,
            nbuf,
            slots.vmem_bytes / 2**20,
            slots.limit / 2**20,
            host.MAX_WEIGHT_PREFETCH_DISTANCE,
        )
    # ragged_rows_alloc must cover every tile READ as well as every commit
    # -- the shard's total rows plus tile_m, aligned to tile_m -- because a
    # tail tile reads a full window.
    if (
        ragged_rows_alloc is None
        or ragged_rows_alloc % ROWBLK
        or ragged_rows_alloc % capacity
    ):
        raise ValueError(
            f"ragged_rows_alloc {ragged_rows_alloc} must be a whole number "
            f"of both {ROWBLK}-row blocks and {capacity}-row tiles, because "
            "a tail tile reads a full window past the shard's last row"
        )
    tile_m = capacity  # the tile height IS the capacity
    tile_blocks = tile_m // ROWBLK
    # The mirror's unit is a tile, the run tables' unit is a block, and the
    # kernel converts between them with a shift.
    tile_block_shift = host.pow2_shift(tile_blocks, "tile_m // ROWBLK")
    tile_shift = host.pow2_shift(tile_m, "tile_m")
    height_rungs = tile_height_rungs(tile_m)
    # The contribution slab exists for one reason: to give the push an HBM
    # source. A tile is computed into `out_vm`, copied to `contrib_hbm`, then
    # read straight back out of it by the push that ships it -- two local
    # passes and, at the small buckets, about a quarter of the visit's DMA
    # enqueues, for a buffer nothing else reads. Where a build's experts are
    # expected to fit inside one tile -- the same bound the height ladder
    # above is emitted on -- the tile can be shipped from `out_vm` itself,
    # deleting the commits, the drain that gates them and the slab's traffic.
    # It costs a staging buffer that stays live until the wire has read it,
    # which is why a build whose visits carry several tiles keeps the slab:
    # there the wire is the busy resource and the round trip is what keeps
    # the tile loop off it.
    #
    # This subsumes the deferred commit drain the same bound used to select:
    # deferring a wait one visit is worth nothing once the wait is gone.
    push_from_vmem = ragged_rows_alloc <= ep * g_local * tile_m
    ls_window_rows = host.act_scale_window_rows(tile_m)
    # A segment word packs dest << 24 | local expert << 16 | rows.
    if push_tokens >= 1 << 16 or ep > 128 or g_local > 256:
        raise ValueError("push dispatch packs token/expert/dest into one i32")
    if not 1 <= sc_prepush <= g_local:
        raise ValueError(f"sc_prepush {sc_prepush} must be in [1, g_local={g_local}]")
    n_segs = ep * g_local
    # Whether the TensorCore pushes any visit position at all: with
    # sc_prepush == g_local the SparseCore delivered every one.
    tc_dispatch = sc_prepush < g_local
    n_prefetch = N_PREFETCH + N_DISPATCH_TABLES

    def kernel(*refs):
        it = iter(refs)
        # Block-unit transport tables, from shard_transport_tables_in_blocks.
        # Three of that builder's tables are the same array as another it
        # already ships, so each arrives once and the duplicates are derived
        # here instead of being prefetched again:
        #   push_src == contrib_off  -- a push reads its source at the offset
        #     the commit wrote it to;
        #   push_len == commit_len   -- `region_rows` IS `run_rows_aligned`
        #     reshaped, so slicing either at this shard gives one array;
        #   recv_row_off == push_dst << ROWBLK_SHIFT -- `recv_base` is a cumsum
        #     of ALIGNED run lengths, so the row offset is the block offset in
        #     rows. Asserted over eight shards and several routings by
        #     tests/kernels/fused_moe/test_fused_ep_moe_v2_tables.py; that test
        #     is what keeps this derivation honest if recv_base ever changes to
        #     track true lengths.
        (commit_start_sm, commit_len_sm, contrib_off_sm, push_dst_sm) = (
            next(it) for _ in range(N_TRANSPORT_TABLES)
        )
        # Row-unit push tables, from shard_push_tables_in_rows.
        (true_rows_sm,) = (next(it) for _ in range(N_PUSH_TABLES))
        # (rows, base) i32 [G] row-unit tables from shard_expert_slabs.
        (expert_rows_sm, expert_base_sm) = (next(it) for _ in range(N_SLAB_TABLES))
        # visit_sm[visit_i] is the real expert at compacted step visit_i.
        visit_sm = next(it)
        # The five counts, each still spread over the expert-parallel axis.
        counts_sm = next(it)
        # This shard's expert-parallel index, as data rather than
        # `lax.axis_index` -- see N_RANK_TABLES.
        rank_sm = next(it)
        # Segment table, entry v * ep + d = destination d's v-th visited
        # expert: [dest << 24 | local expert << 16 | rows (S), source row in
        # lhs (S), destination slab row (S)], S = ep * g_local.
        dpairs_sm = next(it)
        # [unused (g_local + 1), receive rows by local expert (g_local),
        #  remote rows this TensorCore pushes (1), rows sent (1)].
        dmeta_sm = next(it)

        def count(row):
            """Close one count's reduction on the scalar core.

            The host stopped a step early and left `ep` lanes to add. `ep`
            is a build-time constant, so this is a fixed run of scalar
            loads and adds: no loop, no dynamic bound, and nothing that
            reaches vector memory.
            """
            total = counts_sm[row, 0]
            for d in range(1, ep):
                total = total + counts_sm[row, d]
            return total

        # Consumed at the head and all through the visit loop, so it is
        # closed once here. The transport totals are closed in the drain
        # instead, where they are read, because that is the end of the
        # kernel and their arithmetic has everything before it to hide in.
        n_visit = count(host.COUNT_VISITS)
        # lhs_hbm is this shard's own rows sorted by global expert, the
        # source of every dispatch push. The wire never ships the topk
        # weight, so it is not an operand.
        lhs_hbm = next(it)
        # The activation row scale exists only where the rows were quantized.
        ls_hbm = next(it) if has_act_scale else None
        w1_hbm, w2_hbm = (next(it) for _ in range(2))
        # The weight scale tables exist only where the weights carry scales.
        w1s_hbm = next(it) if has_scales else None
        w2s_hbm = next(it) if has_scales else None
        # Optional per-expert bias tables, one row per local expert, in the
        # same [G, N] layout as the per-channel weight scales beside them.
        w1b_hbm = next(it) if has_w1_bias else None
        w2b_hbm = next(it) if has_w2_bias else None
        if rhs_packed4:
            # Packed-u32 weight stream: the four-bit [G, K, N] refs viewed as
            # [G, K//8, N] uint32, so the DMA moves half the bytes. This is
            # the ref-level view, and it is a PRIVATE jax method -- Pallas
            # exports pltpu.bitcast for the value-level widening below but
            # has no public equivalent for a ref. It carries no deprecation
            # entry, so if it moves it fails as a bare AttributeError from
            # inside a kernel body. Say what happened instead.
            if not hasattr(w1_hbm, "bitcast"):
                raise NotImplementedError(
                    f"the four-bit weight stream views a memory ref as "
                    f"packed 32-bit words through the private "
                    f"{type(w1_hbm).__name__}.bitcast, which the installed "
                    f"jax {jax.__version__} no longer carries; this kernel "
                    f"was built against jax {_REF_BITCAST_JAX_VERSION}. "
                    "There is no public Pallas equivalent for a ref-level "
                    "bitcast, so this needs a port rather than a rename."
                )
            w1_hbm = w1_hbm.bitcast(jnp.uint32)
            w2_hbm = w2_hbm.bitcast(jnp.uint32)
        next(it)  # slab_in, aliased to dispatch_hbm (output 4)
        recv_hbm = next(it)
        rscl_hbm = next(it)
        # A `push_from_vmem` build writes neither of these: they stay operands
        # for their row geometry, which the send waits count bytes against.
        contrib_hbm = next(it)
        cscl_hbm = next(it)
        # The expert slab [ragged_rows_alloc, lane blocks, 128] every shard
        # pushes its routed rows into; the SparseCore pre-push filled the
        # first sc_prepush visit positions before the kernel started.
        dispatch_hbm = next(it)
        # In vmem_scratch_arrays order, which is what the scratch list is
        # built from: staging, weights, scale tables where they exist, the
        # bias tables each build asked for, the activation row scale where
        # it exists, then the wire buffer and its scale mirror.
        (lhs_vm, w1_vm, w2_vm) = (next(it) for _ in range(3))
        w1s_vm = next(it) if has_scales else None
        w2s_vm = next(it) if has_scales else None
        w1b_vm = next(it) if has_w1_bias else None
        w2b_vm = next(it) if has_w2_bias else None
        ls_vm = next(it) if has_act_scale else None
        out_vm = next(it)
        oscl_vm = next(it)  # [parities, tile blocks, ROWBLK] f32
        lhs_sems, w1_sems, w2_sems = (next(it) for _ in range(3))
        w1s_sems = next(it) if has_scales else None
        w2s_sems = next(it) if has_scales else None
        cp_sem = next(it)
        dsems = next(it)  # [g_local] receive, one per local expert
        dsend_sem = next(it)  # remote dispatch sends
        # One send sem per out_vm parity: with a shared sem the other parity's
        # bytes could satisfy a wait, and order is not promised. Where the
        # tile ships itself, the parity's own sem is what says the wire has
        # finished reading the staging slot, so the same split does double
        # duty -- it guards reuse as well as the final drain.
        if push_from_vmem:
            # Own-destination rows ride the remote sems: for either copy the
            # semaphore fires once the SOURCE has been read, which is exactly
            # what the reuse guard needs, and only this parity signals it.
            send_sems, recv_sem = (next(it) for _ in range(2))
            # Scale-mirror DMAs need their own sems: waits are per-buffer.
            send_scl_sems, recv_scl_sem = (next(it) for _ in range(2))
            commit_sems = commit_scl_sems = None
            mehop_sem = mehop_scl_sem = None
        else:
            commit_sems, send_sem, recv_sem = (next(it) for _ in range(3))
            (commit_scl_sems, send_scl_sem, recv_scl_sem) = (next(it) for _ in range(3))
            # Never cp_sem: its start+wait users must not consume these.
            mehop_sem = next(it)
            mehop_scl_sem = next(it)

        me = rank_sm[0, 0]

        def sync(src, dst):
            """Copy `src` into `dst` and wait for it before returning."""
            copy = pltpu.make_async_copy(src, dst, cp_sem)
            copy.start()
            copy.wait()

        def w1_copy(expert, slot):
            """The gate/up weight slab of `expert`, into weight slot `slot`."""
            return _slot_copy(w1_hbm.at[expert], w1_vm, w1_sems, slot)

        def w2_copy(expert, slot):
            """The down weight slab of `expert`, into weight slot `slot`."""
            if rhs_packed4 and trim_gate_up_padding:
                # C03's down contractions never read the padded K tail.
                # Start and wait use the same shortened descriptor and bytes.
                packed_active = effective_inter // PACK4
                return pltpu.make_async_copy(
                    w2_hbm.at[expert, pl.ds(0, packed_active), :],
                    w2_vm.at[slot, pl.ds(0, packed_active), :],
                    w2_sems.at[slot],
                )
            return _slot_copy(w2_hbm.at[expert], w2_vm, w2_sems, slot)

        def w1s_copy(expert, slot):
            assert w1s_hbm is not None
            return _slot_copy(w1s_hbm.at[expert], w1s_vm, w1s_sems, slot)

        def w2s_copy(expert, slot):
            assert w2s_hbm is not None
            return _slot_copy(w2s_hbm.at[expert], w2s_vm, w2s_sems, slot)

        def weight_starts(expert, slot):
            """Start every buffer slot `slot` holds for `expert`.

            In the order the body consumes them, which is also the order a
            single DMA queue will retire them -- WEIGHT_DMA_PRIORITY selects
            a queue, it does not reorder within one. The scale tables go
            first: they are half a megabyte and the first contraction block
            needs them, so they are all a streamed visit's head waits for.
            """
            if has_scales:
                w1s_copy(expert, slot).start(priority=WEIGHT_DMA_PRIORITY)
                w2s_copy(expert, slot).start(priority=WEIGHT_DMA_PRIORITY)
            w1_copy(expert, slot).start(priority=WEIGHT_DMA_PRIORITY)
            w2_copy(expert, slot).start(priority=WEIGHT_DMA_PRIORITY)

        def weight_waits(expert, slot):
            """Wait for what the expert's first block needs.

            A streamed slab is not waited for here: its wait sits in the block
            that first reads it, where the wire has had longest to deliver, so
            this head waits for the scale tables alone. Every other format
            takes the slab in one matmul and keeps the whole-slab head wait.
            """
            if not streams_weights:
                w1_copy(expert, slot).wait()
                w2_copy(expert, slot).wait()
            if has_scales:
                w1s_copy(expert, slot).wait()
                w2s_copy(expert, slot).wait()

        def prologue():
            """Land the resident tables and start the first weight refills."""
            if has_w1_bias:
                sync(w1b_hbm, w1b_vm)
            if has_w2_bias:
                sync(w2b_hbm, w2b_vm)
            # Guarded on b < n_visit, matching the head waits. The range is
            # clamped to the visit-list length: at one local expert,
            # visit_sm[1] would be a statically out-of-bounds read before
            # the predicate could save it.
            for b in range(min(prefetch_distance, g_local)):

                @pl.when(jnp.int32(b) < n_visit)
                def _(b=b):
                    weight_starts(visit_sm[b], b)

        prologue()

        def dispatch_issue(lo_v, hi_v):
            """Push the (dest, expert) segments of visit positions [lo_v,
            hi_v); never blocks. Destinations are staggered by rank (d = me +
            1, me + 2, ...) like the SparseCore pre-push, so the shards do not
            all hit the same peer first."""

            def one(p, c):
                a = dpairs_sm[p]
                n = jnp.bitwise_and(a, 65535)

                @pl.when(n > 0)
                def _():
                    dest = jnp.right_shift(a, 24)
                    el = jnp.bitwise_and(jnp.right_shift(a, 16), 255)
                    src = lhs_hbm.at[pl.ds(dpairs_sm[n_segs + p], n)]
                    dst = dispatch_hbm.at[pl.ds(dpairs_sm[2 * n_segs + p], n)]

                    # Local destinations are a local HBM copy on the same
                    # receive semaphore.
                    @pl.when(dest == me)
                    def _():
                        pltpu.make_async_copy(src, dst, dsems.at[el]).start(priority=0)

                    @pl.when(dest != me)
                    def _():
                        pltpu.make_async_remote_copy(
                            src_ref=src,
                            dst_ref=dst,
                            send_sem=dsend_sem,
                            recv_sem=dsems.at[el],
                            device_id=(dest,),
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start(priority=0)

                return c

            def one_v(v, c_v):
                base_p = v * ep

                def one_d(d, c_d):
                    return one(base_p + lax.rem(d + me + 1, ep), c_d)

                return lax.fori_loop(0, ep, one_d, c_v)

            lax.fori_loop(lo_v, hi_v, one_v, jnp.int32(0))

        def _start_next_gate_up(visit_i, slot, last_tile):
            """`() -> ()`: start the NEXT expert's gate/up slab, called from
            the reader of the block that retires this expert's.

            The reader is where the start belongs for the same reason the
            wait is: it is the only place that knows which block is being
            read. A later tile of this expert would still be reading the
            slab, so only the last tile starts the next one.
            """

            def start():
                @pl.when(jnp.logical_and(last_tile, visit_i + 1 < n_visit))
                def _():
                    w1_copy(visit_sm[visit_i + 1], slot).start(
                        priority=WEIGHT_DMA_PRIORITY
                    )

            return start

        def _slab_waiter(expert, slot, first_tile):
            """`(copy, block) -> ()`: land the slab that block `block` opens,
            on the expert's first tile.

            A streamed visit owes each slab exactly one wait, and block 0 --
            the first block of its matmul group -- is both where the wire has
            had the longest to deliver it and the only place in the group
            where no accumulation is in flight. Only the first tile owes
            anything: a later tile of the same expert reads a slab that is
            already whole, which is why the wait is predicated rather than
            issued per tile.
            """

            def wait(copy, block):
                if block:
                    return

                @pl.when(first_tile)
                def _():
                    copy(expert, slot).wait()

            return wait

        def _fp4_block_readers(slot, wait_slab=None, slab_retired=None):
            """(w1_block, w2_block) fp4 block readers at weight slot `slot`.

            The one genuinely fp4 place in the kernel: everything around it
            is four-bit generic, and these two lines are where the packed
            words are read as fp4 (e2m1) rather than as some other four-bit
            element type.

            `wait_slab` is where a streamed slab is landed and
            `slab_retired` fires at the last read of the gate/up one: the
            reader is the one place that knows which block is about to be
            contracted, so it is the one place either can sit without
            costing a block of runway.
            """
            # Each widens one k-block: u32 [qb/8, N] -> fp4 [qb, N] -> fp8.
            packed_rows = rhs_qb // PACK4

            def w1_block(b, n_start, n_end):
                # N.B. b is the contraction-block index.
                if wait_slab is not None and n_start == 0:
                    wait_slab(w1_copy, b)
                block = pltpu.bitcast(
                    w1_vm[
                        slot,
                        pl.ds(b * packed_rows, packed_rows),
                        pl.ds(n_start, n_end - n_start),
                    ],
                    FP4,
                ).astype(FP8)
                # slab_retired starts the next expert's gate/up DMA into the
                # same VMEM slot. That is only safe after the last read of
                # the current slab.
                if (
                    slab_retired is not None
                    and b == hidden // rhs_qb - 1
                    and n_end == w1_last_valid_column
                ):
                    slab_retired()
                return block

            def w2_block(b, n_start, n_end, *, block_rows=None):
                active_rows = rhs_qb if block_rows is None else block_rows
                if not 0 < active_rows <= rhs_qb or active_rows % PACK4:
                    raise ValueError("FP4 block rows must be positive and pack-aligned")
                if wait_slab is not None and n_start == 0:
                    wait_slab(w2_copy, b)
                return pltpu.bitcast(
                    w2_vm[
                        slot,
                        pl.ds(b * packed_rows, active_rows // PACK4),
                        pl.ds(n_start, n_end - n_start),
                    ],
                    FP4,
                ).astype(FP8)

            return w1_block, w2_block

        def _int8_chunk_readers(slot):
            """(w1_chunk, w2_chunk) integer chunk readers at slot `slot`.

            Each widens one contraction chunk to bf16, which is what the
            matrix unit takes: TPU v7 has no integer input
            type, so the widening is not an optimization but the only way
            an integer weight reaches a matmul at all.
            """

            def w1_chunk(b):
                return w1_vm[
                    slot, pl.ds(b * host.WIDEN_KCHUNK, host.WIDEN_KCHUNK), :
                ].astype(host.BF16)

            def w2_chunk(b):
                return w2_vm[
                    slot, pl.ds(b * host.WIDEN_KCHUNK, host.WIDEN_KCHUNK), :
                ].astype(host.BF16)

            return w1_chunk, w2_chunk

        # ---- ragged tile machinery ----
        # One-tile lookahead, double-buffered on the global tile parity:
        # a tile's stream rides lhs_sems[parity], waited once at its head.
        def lhs_issue_tile(row_base, live_blocks, slot):
            """Issue one tile's activations: the rows already sit in slab
            order, so one contiguous copy, plus its row-scale window."""
            pltpu.make_async_copy(
                dispatch_hbm.at[pl.ds(row_base, live_blocks * ROWBLK)],
                lhs_vm.at[slot, pl.ds(0, live_blocks * ROWBLK)],
                lhs_sems.at[slot],
            ).start(priority=0)
            if has_act_scale:
                # ls rides along on the same sem. The slab is the dense
                # lane-block view of one f32 per row, so a tile's window is
                # the sublanes its rows fall in, not a row range.
                assert ls_hbm is not None and ls_vm is not None
                pltpu.make_async_copy(
                    ls_hbm.at[pl.ds(row_base // HIDDEN_LANE_BLOCK, ls_window_rows)],
                    ls_vm.at[slot],
                    lhs_sems.at[slot],
                ).start(priority=0)

        def lhs_ready_tile(live_rows, slot):
            """Wait the token rows and the scale window of one tile."""
            # Sems count bytes, so the split wait is exact under any landing
            # order.
            pltpu.make_async_copy(
                lhs_vm.at[slot, pl.ds(0, live_rows)],
                lhs_vm.at[slot, pl.ds(0, live_rows)],
                lhs_sems.at[slot],
            ).wait()
            if has_act_scale:
                assert ls_vm is not None
                pltpu.make_async_copy(
                    ls_vm.at[slot], ls_vm.at[slot], lhs_sems.at[slot]
                ).wait()

        def _mirror_base(e):
            """Sublane expert e's tiles start at in the contribution mirror.

            The tile its slab rows start in, plus two sublanes of slack per
            expert. `host.contrib_mirror_rows` sizes the mirror for that
            slack, and two is enough because ceil(rows / tile_m) is at most
            (rows >> shift) + 1 while the next expert's base is at least
            that much further on.
            """
            return jnp.right_shift(expert_base_sm[e], tile_shift) + 2 * e

        def _run_tiles(e, d, region_blocks):
            """A run's mirror sublanes: (source, destination, count).

            The destination follows the same rule on the run's ARRIVAL rows,
            and the run order there is the global expert order, so the same
            two sublanes of slack keep two runs off one sublane. Nonempty
            runs only -- the callers predicate on the length.
            """
            start = commit_start_sm[e, d]
            lo = jnp.right_shift(start, tile_block_shift)
            hi = jnp.right_shift(start + region_blocks - 1, tile_block_shift)
            dst = jnp.right_shift(push_dst_sm[e, d], tile_block_shift) + 2 * (
                me * g_local + e
            )
            return _mirror_base(e) + lo, dst, hi - lo + 1

        def wait_commits(parity, live_blocks):
            """Wait one tile's commits on `parity`: data rows, then scales."""
            assert commit_sems is not None and commit_scl_sems is not None
            _rows_wait(commit_sems.at[parity], contrib_hbm, live_blocks * ROWBLK)
            # One tile committed one mirror sublane, whatever its row count.
            _rows_wait(commit_scl_sems.at[parity], cscl_hbm, 1)

        def wait_shipped(parity, rows, subs):
            """Wait until the wire has read `parity`'s staging slot.

            `rows` and `subs` are what the tile that last held this parity
            shipped out of it -- rows of payload and mirror sublanes -- and
            nothing else signals these two semaphores, so the counts are
            exact. `contrib_hbm` and `cscl_hbm` are here only for their row
            geometry: a semaphore counts bytes, and those two carry the same
            bytes per row as the buffers actually shipped.
            """

            @pl.when(rows > 0)
            def _():
                _rows_wait(send_sems.at[parity], contrib_hbm, rows)

            @pl.when(subs > 0)
            def _():
                _rows_wait(send_scl_sems.at[parity], cscl_hbm, subs)

        def prime_expert(e, lhs_slot, visit_i):
            """Wait expert ``e``'s dispatched rows and issue its first tile.

            Visit positions below sc_prepush were delivered by the SparseCore
            before the kernel started, so they owe no wait; a static visit
            index (the first visit) skips the predicate altogether.
            """
            rows = expert_rows_sm[e]
            slab_base = expert_base_sm[e]
            if tc_dispatch and not (isinstance(visit_i, int) and visit_i < sc_prepush):
                # Every row of this expert, from every shard, has landed.
                @pl.when(visit_i >= sc_prepush)
                def _():
                    n_in = dmeta_sm[g_local + 1 + e]

                    @pl.when(n_in > 0)
                    def _():
                        _rows_wait(dsems.at[e], dispatch_hbm, n_in)

            lhs_issue_tile(slab_base, jnp.minimum(rows, tile_m) // ROWBLK, lhs_slot)

        def expert_tiles(e, visit_i, carry):
            """One expert step: a fori_loop over its [tile_m, H] tiles."""
            # carry = (tile_count, <per-parity outstanding counts>): the
            # global tile counter and what each out_vm parity still owes --
            # committed blocks, or shipped rows and mirror sublanes where the
            # tile ships itself.
            # The weight slot indexes a contiguous counter so DISTANCE + 1 of
            # them occupy distinct slots; the DMA base stays the real expert.
            slot = lax.rem(visit_i, jnp.int32(nbuf))
            if not prefetches and not starts_next_weights:
                # No runway: nobody started this visit's weights, so it
                # starts them itself and waits on the two lines below.
                # Refilling the one slot cannot race the previous visit's
                # reads of it, because those reads feed accumulators whose
                # results are stored into out_vm inside the tile loop -- so
                # they have all retired by the loop's back edge, which is
                # before this.
                weight_starts(e, slot)
            if starts_next_weights:
                # Every later visit had its weights started by the one before,
                # which is the whole point; the first has nobody to start them
                # for it.

                @pl.when(visit_i == 0)
                def _():
                    weight_starts(e, slot)

            weight_waits(e, slot)

            # Refill the slot for the expert DISTANCE ahead: the previous
            # expert last read it, and DISTANCE < nbuf keeps it off both live
            # readers, so no wait is needed.
            if prefetches:

                @pl.when(visit_i + prefetch_distance < n_visit)
                def _():
                    refill_slot = lax.rem(visit_i + prefetch_distance, jnp.int32(nbuf))
                    ahead = visit_sm[visit_i + prefetch_distance]
                    # Weight refills take their own DMA priority to stay off
                    # the in-order token-gather queue.
                    weight_starts(ahead, refill_slot)

            rows = expert_rows_sm[e]
            slab_base = expert_base_sm[e]
            n_tiles = -(-rows // tile_m)

            def commit_tile(parity, t, tile_block_base, live_blocks):
                """Commit the tile's intersection with each (expert, dest) run."""
                assert commit_sems is not None and commit_scl_sems is not None
                # The runs tile the expert slab contiguously, so the
                # per-dest lengths sum to exactly live_blocks. Rows go per
                # run; the scales go once, below, because a tile's are one
                # sublane wherever its rows went.
                for d in range(ep):
                    run_start = commit_start_sm[e, d]
                    run_len = commit_len_sm[e, d]
                    lo = jnp.maximum(run_start, tile_block_base)
                    hi = jnp.minimum(run_start + run_len, tile_block_base + live_blocks)
                    overlap = jnp.maximum(hi - lo, 0)
                    # An empty intersection leaves lo unclamped, so the
                    # offsets below can point out of range; the copies are
                    # predicated out rather than issued at zero length. The
                    # commit waits count bytes, and a skipped copy
                    # contributes zero bytes either way.
                    lo = jnp.minimum(lo, hi)
                    src_block = lo - tile_block_base
                    dst_block = contrib_off_sm[e, d] + (lo - run_start)

                    @pl.when(overlap > 0)
                    def _(
                        src_block=src_block,
                        dst_block=dst_block,
                        overlap=overlap,
                        parity=parity,
                    ):
                        pltpu.make_async_copy(
                            out_vm.at[
                                parity, pl.ds(src_block * ROWBLK, overlap * ROWBLK)
                            ],
                            contrib_hbm.at[pl.ds(dst_block * ROWBLK, overlap * ROWBLK)],
                            commit_sems.at[parity],
                        ).start()

                # The tile's scales are one sublane wherever its rows
                # went, so they commit once, outside the per-dest loop.
                pltpu.make_async_copy(
                    oscl_vm.at[parity],
                    cscl_hbm.at[pl.ds(_mirror_base(e) + t, 1)],
                    commit_scl_sems.at[parity],
                ).start()

            def ship_tile(parity, t, tile_block_base, live_blocks):
                """Ship the tile straight out of `out_vm`, no slab in between.

                Returns what this tile put on the wire -- payload rows and
                mirror sublanes -- because that is what the next tile on this
                parity has to wait for before it may overwrite the slot.

                Every offset here is the one the two-hop path would have
                used: a commit wrote the tile's intersection with run (e, d)
                at `contrib_off + (lo - run_start)` and the push read the
                run's true rows back from `contrib_off`, so the tile's own
                share of that run starts `lo - run_start` blocks into it.
                """
                rows_out = jnp.int32(0)
                subs_out = jnp.int32(0)
                for d in range(ep):
                    run_start = commit_start_sm[e, d]
                    run_len = commit_len_sm[e, d]
                    lo = jnp.maximum(run_start, tile_block_base)
                    hi = jnp.minimum(run_start + run_len, tile_block_base + live_blocks)
                    overlap = jnp.maximum(hi - lo, 0)
                    # An empty intersection leaves lo unclamped, so the
                    # offsets below can point out of range; every copy is
                    # predicated out rather than issued at zero length.
                    lo = jnp.minimum(lo, hi)
                    into_run = (lo - run_start) * ROWBLK
                    # The run's true rows are its leading rows, so this tile
                    # carries whatever is left of them inside its overlap.
                    true_here = jnp.clip(
                        true_rows_sm[e, d] - into_run, 0, overlap * ROWBLK
                    )
                    is_me = jnp.int32(d) == me

                    @pl.when(overlap > 0)
                    def _(
                        lo=lo,
                        into_run=into_run,
                        overlap=overlap,
                        true_here=true_here,
                        is_me=is_me,
                        run_start=run_start,
                        parity=parity,
                        d=d,
                    ):
                        src_row = (lo - tile_block_base) * ROWBLK
                        dst_row = push_dst_sm[e, d] * ROWBLK + into_run
                        # One sublane per tile a run spans, at the destination
                        # base that run's own arrival rows give it.
                        dst_sub = (
                            jnp.right_shift(push_dst_sm[e, d], tile_block_shift)
                            + 2 * (me * g_local + e)
                            + t
                            - jnp.right_shift(run_start, tile_block_shift)
                        )

                        @pl.when(_and_nonempty(jnp.logical_not(is_me), true_here))
                        def _():
                            pltpu.make_async_remote_copy(
                                src_ref=out_vm.at[parity, pl.ds(src_row, true_here)],
                                dst_ref=recv_hbm.at[pl.ds(dst_row, true_here)],
                                send_sem=send_sems.at[parity],
                                recv_sem=recv_sem,
                                device_id=(jnp.int32(d),),
                                device_id_type=pl.DeviceIdType.MESH,
                            ).start()

                        # The own-destination run needs no fabric, and it moves
                        # the ALIGNED region the two-hop path moved: its trailing
                        # padding rows are never read, but leaving them out would
                        # change bytes the arrival buffer already holds.
                        @pl.when(is_me)
                        def _():
                            pltpu.make_async_copy(
                                out_vm.at[parity, pl.ds(src_row, overlap * ROWBLK)],
                                recv_hbm.at[pl.ds(dst_row, overlap * ROWBLK)],
                                send_sems.at[parity],
                            ).start()

                        @pl.when(jnp.logical_not(is_me))
                        def _():
                            pltpu.make_async_remote_copy(
                                src_ref=oscl_vm.at[parity],
                                dst_ref=rscl_hbm.at[pl.ds(dst_sub, 1)],
                                send_sem=send_scl_sems.at[parity],
                                recv_sem=recv_scl_sem,
                                device_id=(jnp.int32(d),),
                                device_id_type=pl.DeviceIdType.MESH,
                            ).start()

                        @pl.when(is_me)
                        def _():
                            pltpu.make_async_copy(
                                oscl_vm.at[parity],
                                rscl_hbm.at[pl.ds(dst_sub, 1)],
                                send_scl_sems.at[parity],
                            ).start()

                    rows_out = rows_out + jnp.where(is_me, overlap * ROWBLK, true_here)
                    subs_out = subs_out + jnp.where(overlap > 0, 1, 0)
                return rows_out, subs_out

            def tile_body(t, carried):
                """Compute tile t, stage it for the wire and ship it."""
                tile_count = carried[0]
                row_base = slab_base + t * tile_m
                live_rows = jnp.minimum(rows - t * tile_m, tile_m)
                live_blocks = live_rows // ROWBLK
                tile_block_base = t * tile_blocks  # expert-local block base
                parity = lax.rem(tile_count, jnp.int32(OUT_PARITIES))

                # out_vm[parity] reuse guard. Predicated on the outstanding
                # count rather than on the tile count, because the drain
                # zeroes them.
                if push_from_vmem:
                    wait_shipped(
                        parity,
                        jnp.where(parity == 0, carried[1], carried[2]),
                        jnp.where(parity == 0, carried[3], carried[4]),
                    )
                else:
                    pending = jnp.where(parity == 0, carried[1], carried[2])

                    @pl.when(pending > 0)
                    def _():
                        wait_commits(parity, pending)

                # Wait this tile's stream, then issue tile t+1's into the other
                # slot before compute, so the fetch runs under the MXU window.
                lhs_ready_tile(live_rows, parity)
                other_parity = jnp.int32(1) - parity

                @pl.when(t + 1 < n_tiles)
                def _():
                    next_t = t + 1
                    next_rows = jnp.minimum(rows - next_t * tile_m, tile_m)
                    lhs_issue_tile(
                        slab_base + next_t * tile_m,
                        next_rows // ROWBLK,
                        other_parity,
                    )

                # The alternate activation slot has retired its previous tile.
                # Fill the next expert's first tile before this final FFN so
                # its token rows and row scales travel under current compute.
                @pl.when(jnp.logical_and(t + 1 == n_tiles, visit_i + 1 < n_visit))
                def _():
                    nxt = visit_sm[visit_i + 1]
                    prime_expert(nxt, other_parity, visit_i + 1)

                def compute_tile(height):
                    """Run the FFN on the tile's leading `height` rows.

                    Rows past the tile's live rows are never committed, so a
                    height above them is pure waste; `height` is the smallest
                    rung that still covers them.
                    """
                    full = height == tile_m
                    act_rows = (
                        lhs_vm[parity] if full else lhs_vm[parity, pl.ds(0, height)]
                    ).reshape(height, hidden)
                    act_scales = (
                        _tile_row_scales(cast(jax.Ref, ls_vm)[parity], row_base, height)
                        if has_act_scale
                        else None
                    )
                    w1b_row = (
                        cast(jax.Ref, w1b_vm)[pl.ds(e, 1), :] if has_w1_bias else None
                    )
                    if rhs_packed4:
                        # The block scales apply inside
                        # expert_ffn_blockscale, so the epilogues below must
                        # not apply w2s again.
                        assert w1s_vm is not None and w2s_vm is not None
                        if pad_scale_rows:
                            # The slot holds the padded table; the rows
                            # past the last block are padding.
                            w1s_blocks = w1s_vm[slot, pl.ds(0, hidden // rhs_qb), :]
                            w2s_blocks = w2s_vm[slot, pl.ds(0, inter // rhs_qb), :]
                        else:
                            w1s_blocks = w1s_vm[slot]  # [nb1, 2*inter] f32
                            w2s_blocks = w2s_vm[slot]  # [nb2, hidden] f32
                        first_tile, last_tile = t == 0, t == n_tiles - 1
                        w1_block, w2_block = _fp4_block_readers(
                            slot,
                            wait_slab=(
                                _slab_waiter(e, slot, first_tile)
                                if streams_weights
                                else None
                            ),
                            slab_retired=(
                                _start_next_gate_up(visit_i, slot, last_tile)
                                if starts_next_weights
                                else None
                            ),
                        )
                        acc2, mid_scale = expert_ffn_blockscale(
                            act_rows,
                            act_scales,
                            w1_block,
                            w2_block,
                            w1s_blocks,
                            w2s_blocks,
                            qb=rhs_qb,
                            act_fn=act_fn,
                            w1b=w1b_row,
                            logical_intermediate_size=(
                                effective_inter if trim_gate_up_padding else None
                            ),
                        )
                    elif weight_format == host.WeightFormat.INT8:
                        assert w1s_vm is not None
                        w1_chunk, w2_chunk = _int8_chunk_readers(slot)
                        acc2, mid_scale = expert_ffn_int8(
                            act_rows,
                            w1_chunk,
                            w2_chunk,
                            w1s_vm[pl.ds(slot, 1), :],
                            n_chunks1=hidden // host.WIDEN_KCHUNK,
                            n_chunks2=inter // host.WIDEN_KCHUNK,
                            act_fn=act_fn,
                            w1b=w1b_row,
                        )
                    elif weight_format == host.WeightFormat.BF16:
                        acc2, mid_scale = expert_ffn_bf16(
                            act_rows,
                            w1_vm[slot],
                            w2_vm[slot],
                            act_fn=act_fn,
                            w1b=w1b_row,
                        )
                    else:
                        assert w1s_vm is not None
                        acc2, mid_scale = expert_ffn_fp8(
                            act_rows,
                            act_scales,
                            w1_vm[slot],
                            w2_vm[slot],
                            w1s_vm[pl.ds(slot, 1), :],
                            act_fn=act_fn,
                            w1b=w1b_row,
                        )
                    # The destination applies the router weight, not this.
                    # Four-bit weights carry w2s inside the block sums; the
                    # formats whose second matmul took bf16 rows return no row
                    # scale, so the only thing left to apply is the
                    # per-channel weight scale where the format has one. The
                    # epilogue itself is a module-level function, so the
                    # placement of each of these -- and the down bias's
                    # add-once property in particular -- has a host-side
                    # witness rather than only a device-marked one.
                    wire_rows, wire_scales = wire_row_and_scale(
                        acc2,
                        mid_scale,
                        w2s=(
                            cast(jax.Ref, w2s_vm)[pl.ds(slot, 1), :]
                            if (has_scales and not rhs_packed4)
                            else None
                        ),
                        w2b=(
                            cast(jax.Ref, w2b_vm)[pl.ds(e, 1), :]
                            if has_w2_bias
                            else None
                        ),
                        wire_dtype=form.wire_dtype,
                        tile_m=height,
                    )
                    wire_rows_staged = wire_rows.reshape(
                        height, lane_blocks, HIDDEN_LANE_BLOCK
                    )
                    wire_scales_staged = wire_scales.reshape(1, height)

                    def store_parity(static_parity):
                        """Stage the tile's rows and scales in one out_vm
                        slot."""
                        if full:
                            out_vm[static_parity] = wire_rows_staged
                            oscl_vm[static_parity] = wire_scales_staged
                        else:
                            out_vm[static_parity, pl.ds(0, height)] = wire_rows_staged
                            oscl_vm[static_parity, :, pl.ds(0, height)] = (
                                wire_scales_staged
                            )

                    # Store slots must be static, so these branches stay
                    # static.
                    @pl.when(parity == 0)
                    def _():
                        store_parity(0)

                    @pl.when(parity == 1)
                    def _():
                        store_parity(1)

                # Exactly one rung runs: live_rows is a whole number of
                # ROWBLK in [ROWBLK, tile_m], and the rungs partition that.
                if len(height_rungs) == 1:
                    compute_tile(tile_m)
                else:
                    for rung_i, rung in enumerate(height_rungs):
                        below = height_rungs[rung_i - 1] if rung_i else 0

                        @pl.when(jnp.logical_and(live_rows > below, live_rows <= rung))
                        def _(rung=rung):
                            compute_tile(rung)

                if push_from_vmem:
                    rows_out, subs_out = ship_tile(
                        parity, t, tile_block_base, live_blocks
                    )
                    return (
                        tile_count + 1,
                        jnp.where(parity == 0, rows_out, carried[1]),
                        jnp.where(parity == 1, rows_out, carried[2]),
                        jnp.where(parity == 0, subs_out, carried[3]),
                        jnp.where(parity == 1, subs_out, carried[4]),
                    )

                commit_tile(parity, t, tile_block_base, live_blocks)

                return (
                    tile_count + 1,
                    jnp.where(parity == 0, live_blocks, carried[1]),
                    jnp.where(parity == 1, live_blocks, carried[2]),
                )

            carry = lax.fori_loop(0, n_tiles, tile_body, carry)

            if starts_next_weights:
                # What the tile loop was still reading to its last block: the
                # down slab, and the scale tables, which are read a block at a
                # time beside the weights they scale.

                @pl.when(visit_i + 1 < n_visit)
                def _():
                    nxt = visit_sm[visit_i + 1]
                    # Scales first: they are all the next visit's head waits
                    # for, and the down slab behind them is a megabyte-scale
                    # queue they would otherwise sit out.
                    if has_scales:
                        w1s_copy(nxt, slot).start(priority=WEIGHT_DMA_PRIORITY)
                        w2s_copy(nxt, slot).start(priority=WEIGHT_DMA_PRIORITY)
                    w2_copy(nxt, slot).start(priority=WEIGHT_DMA_PRIORITY)

            return carry

        def drain_commits(carried):
            """Drain both parities' pending commits, then zero the counts."""
            # The tile reuse guards test pending > 0, so nothing double-waits.
            tile_count, pending0, pending1 = carried

            @pl.when(pending0 > 0)
            def _():
                wait_commits(0, pending0)

            @pl.when(pending1 > 0)
            def _():
                wait_commits(1, pending1)

            return (tile_count, jnp.int32(0), jnp.int32(0))

        def drain_shipped(carried):
            """Wait out whatever the last tile on each parity put on the wire.

            Every earlier tile was already waited by the reuse guard of the
            tile two later on its parity, so these are the only sends left
            and their counts are the whole of the send semaphores' credit.
            """
            for p in range(host.OUT_PARITIES):
                wait_shipped(p, carried[1 + p], carried[3 + p])

        # Visit 0 is a SparseCore-delivered position, so its first tile is
        # issued before the barrier; the TensorCore's own first pushes go out
        # right after it, once every peer is inside the kernel.
        @pl.when(n_visit > 0)
        def _():
            prime_expert(visit_sm[0], jnp.int32(0), 0)

        _all_pairs_barrier(ep)
        if tc_dispatch:
            dispatch_issue(sc_prepush, min(sc_prepush + 1, g_local))

        def push_expert(e):
            """Push expert e's remote regions and hop its own-dest region."""
            # Remote pushes go per (expert, dest) at true length; run starts
            # stay aligned, so recv positions are unchanged. Each is predicated
            # on its own length: a zero-length REMOTE DMA must never issue.
            for d in range(ep):
                region_blocks = commit_len_sm[e, d]
                src_block = contrib_off_sm[e, d]
                dst_block = push_dst_sm[e, d]
                is_me = jnp.int32(d) == me

                @pl.when(_and_nonempty(jnp.logical_not(is_me), region_blocks))
                def _():
                    # One expert per push, so the region IS expert e's run.
                    true_rows = true_rows_sm[e, d]

                    def push_run():
                        """The rows themselves, at their true length."""
                        pltpu.make_async_remote_copy(
                            src_ref=contrib_hbm.at[
                                pl.ds(src_block * ROWBLK, true_rows)
                            ],
                            dst_ref=recv_hbm.at[
                                pl.ds(push_dst_sm[e, d] * ROWBLK, true_rows)
                            ],
                            send_sem=send_sem,
                            recv_sem=recv_sem,
                            device_id=(jnp.int32(d),),
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start()

                    # A nonempty region can hold an empty (e, d) run.
                    pl.when(true_rows > 0)(push_run)
                    # Scale mirror: the source tiles this run's rows
                    # fall in, landing at the base the destination's own
                    # plan gave the run.
                    src_sub, dst_sub, n_sub = _run_tiles(e, d, region_blocks)
                    pltpu.make_async_remote_copy(
                        src_ref=cscl_hbm.at[pl.ds(src_sub, n_sub)],
                        dst_ref=rscl_hbm.at[pl.ds(dst_sub, n_sub)],
                        send_sem=send_scl_sem,
                        recv_sem=recv_scl_sem,
                        device_id=(jnp.int32(d),),
                        device_id_type=pl.DeviceIdType.MESH,
                    ).start()

                # Safe to defer: recv is read only after the drain.
                @pl.when(_and_nonempty(is_me, region_blocks))
                def _():
                    pltpu.make_async_copy(
                        contrib_hbm.at[
                            pl.ds(src_block * ROWBLK, region_blocks * ROWBLK)
                        ],
                        recv_hbm.at[pl.ds(dst_block * ROWBLK, region_blocks * ROWBLK)],
                        mehop_sem,
                    ).start()
                    src_sub, dst_sub, n_sub = _run_tiles(e, d, region_blocks)
                    pltpu.make_async_copy(
                        cscl_hbm.at[pl.ds(src_sub, n_sub)],
                        rscl_hbm.at[pl.ds(dst_sub, n_sub)],
                        mehop_scl_sem,
                    ).start()

        def visit_step(visit_i, carry):
            """One step of the visit list: compute, then drain and push."""
            e = visit_sm[visit_i]
            if tc_dispatch:
                # Push the position sc_prepush + 1 visits ahead, so a peer's
                # rows for its next expert are in flight while it computes.
                ahead = visit_i + sc_prepush + 1

                @pl.when(ahead < g_local)
                def _():
                    dispatch_issue(ahead, ahead + 1)

            carry = expert_tiles(e, visit_i, carry)
            if push_from_vmem:
                # Each tile shipped itself; there is nothing left to flush.
                return carry
            carry = drain_commits(carry)
            push_expert(e)
            return carry

        carry0 = tuple(jnp.int32(0) for _ in range(5 if push_from_vmem else 3))
        carry_end = lax.fori_loop(0, n_visit, visit_step, carry0)

        def drain_transport():
            """Consume the pushes and the commits the per-tile waits left."""
            if push_from_vmem:
                # Sends are per staging parity and the reuse guards already
                # took all but the last two; arrivals are still a total,
                # because a peer signals them whatever parity it sent from.
                drain_shipped(carry_end)
                _rows_wait(recv_sem, recv_hbm, count(host.COUNT_RECV_ROWS))
                _rows_wait(recv_scl_sem, rscl_hbm, count(host.COUNT_RECV_MIRROR))
                return
            # The per-tile head waits consume every commit but the last one
            # per parity; the drains below consume those.
            _rows_wait(send_sem, contrib_hbm, count(host.COUNT_SEND_ROWS))
            _rows_wait(recv_sem, recv_hbm, count(host.COUNT_RECV_ROWS))
            # The mirror moves whole sublanes, one per tile a run spans,
            # so its counts are their own rows of the count table rather
            # than a division of the row counts.
            _rows_wait(send_scl_sem, cscl_hbm, count(host.COUNT_SEND_MIRROR))
            _rows_wait(recv_scl_sem, rscl_hbm, count(host.COUNT_RECV_MIRROR))
            # Deferred own-destination drain: the total sums my own-dest
            # region lengths, and a skipped pair owes exactly zero rows.
            self_blocks = commit_len_sm[0, me]
            for g in range(1, g_local):
                self_blocks = self_blocks + commit_len_sm[g, me]
            _rows_wait(mehop_sem, recv_hbm, self_blocks * ROWBLK)
            _rows_wait(mehop_scl_sem, rscl_hbm, count(host.COUNT_SELF_MIRROR))

        if tc_dispatch:
            # Whatever visit positions the loop did not reach -- this shard
            # may visit fewer experts than its peers -- then the sends.
            dispatch_issue(jnp.minimum(n_visit + sc_prepush + 1, g_local), g_local)
            n_remote = dmeta_sm[2 * g_local + 1]

            @pl.when(n_remote > 0)
            def _():
                _rows_wait(dsend_sem, dispatch_hbm, n_remote)

        drain_transport()
        _all_pairs_barrier(ep)

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    # The VMEM buffers come from the same list the VMEM accounting sums, in
    # the order the kernel body unpacks them: lhs_vm, w1_vm, w2_vm, the scale
    # tables where the format has them, the bias tables each build asked for,
    # ls_vm where the activations are quantized, out_vm, oscl_vm.
    scratch = (
        [
            pltpu.VMEM(shape, dtype)
            for _, shape, dtype in host.vmem_scratch_arrays(
                g_local,
                capacity,
                hidden,
                inter,
                nbuf=nbuf,
                weight_format=weight_format,
                rhs_qb=rhs_qb,
                has_w1_bias=has_w1_bias,
                has_w2_bias=has_w2_bias,
                pad_scale_rows=pad_scale_rows,
            )
        ]
        + [
            # lhs_sems. Parity-deep, not nbuf-deep: every index into it, and into
            # the lhs_vm and ls_vm buffers it guards, is an out-parity.
            pltpu.SemaphoreType.DMA((OUT_PARITIES,)),
            pltpu.SemaphoreType.DMA((nbuf,)),  # w1_sems
            pltpu.SemaphoreType.DMA((nbuf,)),  # w2_sems
        ]
        + (
            [
                # The scale tables travel with the slabs they scale, so they are
                # slot-deep and need their own sems: a wait is per-buffer.
                pltpu.SemaphoreType.DMA((nbuf,)),  # w1s_sems
                pltpu.SemaphoreType.DMA((nbuf,)),  # w2s_sems
            ]
            if has_scales
            else []
        )
        + [
            pltpu.SemaphoreType.DMA,  # cp_sem
            pltpu.SemaphoreType.DMA((g_local,)),  # dsems
            pltpu.SemaphoreType.DMA,  # dsend_sem
        ]
        + (
            [
                pltpu.SemaphoreType.DMA((OUT_PARITIES,)),  # send_sems
                pltpu.SemaphoreType.DMA,  # recv_sem
                pltpu.SemaphoreType.DMA((OUT_PARITIES,)),  # send_scl_sems
                pltpu.SemaphoreType.DMA,  # recv_scl_sem
            ]
            if push_from_vmem
            else [
                pltpu.SemaphoreType.DMA((OUT_PARITIES,)),  # commit_sems
                pltpu.SemaphoreType.DMA,  # send_sem
                pltpu.SemaphoreType.DMA,  # recv_sem
                pltpu.SemaphoreType.DMA((OUT_PARITIES,)),  # commit_scl_sems
                pltpu.SemaphoreType.DMA,  # send_scl_sem
                pltpu.SemaphoreType.DMA,  # recv_scl_sem
                pltpu.SemaphoreType.DMA,  # mehop_sem
                pltpu.SemaphoreType.DMA,  # mehop_scl_sem
            ]
        )
    )
    # The sorted rows, the two weight slabs and the dispatch slab are always
    # operands; the activation row scale, the two weight scale tables and the
    # two bias tables are there only where the build has them.
    n_inputs = (
        2
        + N_WEIGHT_SLABS
        + int(has_act_scale)
        + N_SCALE_TABLES * int(has_scales)
        + int(has_w1_bias)
        + int(has_w2_bias)
    )

    def make_call(recv_rows):
        """The pallas_call for this build, at this arrival-buffer height."""
        # A caller value, so it refuses rather than asserts: under `python
        # -O` a non-multiple truncates in the two divisions below and the
        # arrival buffer comes out short by up to ROWBLK - 1 rows, while the
        # combine still gathers at positions reaching recv_rows - 1.
        if recv_rows % ROWBLK:
            raise ValueError(
                f"the arrival buffer height {recv_rows} is not a whole "
                f"number of {ROWBLK}-row blocks, which is the unit every "
                "transport into it moves"
            )
        # contrib and cscl are sized for the no-drop worst case.
        contrib_rows = ragged_rows_alloc
        wire = form.wire_dtype
        out_shape = [
            jax.ShapeDtypeStruct((recv_rows, lane_blocks, HIDDEN_LANE_BLOCK), wire),
            jax.ShapeDtypeStruct(
                (host.arrival_mirror_rows(recv_rows, ep * g_local, tile_m), tile_m),
                jnp.float32,
            ),
            jax.ShapeDtypeStruct((contrib_rows, lane_blocks, HIDDEN_LANE_BLOCK), wire),
            jax.ShapeDtypeStruct(
                (host.contrib_mirror_rows(contrib_rows, g_local, tile_m), tile_m),
                jnp.float32,
            ),
            jax.ShapeDtypeStruct(
                (ragged_rows_alloc, lane_blocks, HIDDEN_LANE_BLOCK), form.act_dtype
            ),
        ]
        # Each suffix appears only when its feature is on, so a kernel that
        # fuses silu on eight-bit float weights with no biases keeps the name
        # it had before any of those operands existed, and every fingerprint
        # taken against it still matches.
        format_tag = {
            host.WeightFormat.FP8: "",
            host.WeightFormat.FP4: "_fp4w",
            host.WeightFormat.INT8: "_int8w",
            host.WeightFormat.BF16: "_bf16w",
        }[weight_format]
        bias_tag = ("_w13bias" if has_w1_bias else "") + (
            "_w2bias" if has_w2_bias else ""
        )
        name = (
            f"push_fused_moe"
            f"{format_tag}"
            f"{'' if act_fn == 'silu' else '_' + act_fn}"
            f"{bias_tag}"
            f"_g{g_local}_c{capacity}_nb{nbuf}_s{sc_prepush}"
        )
        return pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=n_prefetch,
                in_specs=[hbm] * n_inputs,
                out_specs=[hbm] * len(out_shape),
                scratch_shapes=tuple(scratch),
                grid=(),
            ),
            out_shape=out_shape,
            # The slab operand (last input) is the dispatch slab (output 4):
            # the SparseCore pre-push's rows stay where it wrote them.
            input_output_aliases={n_prefetch + n_inputs - 1: 4},
            compiler_params=pltpu.CompilerParams(
                collective_id=COLLECTIVE_ID,
                vmem_limit_bytes=host.vmem_limit(),
                # Bounds checks cost time and change no computed value.
                disable_bounds_checks=True,
            ),
            name=name,
        )

    def fn(
        tables,
        expert_rows,
        expert_base,
        act_q_sorted,
        act_scales,
        w1,
        w2,
        w1s,
        w2s,
        w1b=None,
        w2b=None,
        *,
        recv_rows,
        visit,
        counts,
        rank,
        dispatch,
        slab,
    ):
        """The served ragged transport form; returns arrivals and scales.

        `dispatch` = (segment table, meta) from host.push_dispatch_segments,
        `slab` the SparseCore pre-push's output, aliased to the dispatch slab.
        """
        # tables = the block-unit transport tuple and the row-unit push
        # tuple, (rows, base) from shard_expert_slabs, act_scales the
        # activation scale slab in the dense lane-block view
        # host.act_scale_slab_rows names, one f32 per slab row.
        # act_scales, w1s and w2s are the operands the weight format decides
        # on: the activation row scale exists where the rows were quantized
        # and the scale tables where the weights carry scales. w1b
        # [G, 2 * inter] and w2b [G, hidden] are the optional expert biases.
        # Whether each is present is a build-time property, so a mismatch is
        # a caller bug rather than a shape error deeper in. These are caller
        # operands, so they refuse rather than assert, for the reason stated
        # at the top of the builder -- and the failure here is not symmetric.
        # Supplying weight scales to a build whose format has none leaves
        # scale_args empty below and the operand is silently DROPPED, so the
        # kernel runs its matmuls with no weight scale applied and returns a
        # plausible, uniformly wrong result; this is the only guard for that
        # direction. (A build that wants scales and gets None fails loudly on
        # None.astype, and a missing bias fails on the pallas_call operand
        # count.) This is also the check that catches "you named bf16 and
        # supplied scales".
        for operand, present, what in (
            (act_scales, has_act_scale, "activation row scale"),
            (w1s, has_scales, "gate/up weight scale table"),
            (w2s, has_scales, "down weight scale table"),
            (w1b, has_w1_bias, "gate/up bias"),
            (w2b, has_w2_bias, "down bias"),
        ):
            if (operand is not None) != present:
                raise ValueError(
                    f"the kernel was built for weight format "
                    f"{weight_format!r} with has_w1_bias={has_w1_bias} "
                    f"has_w2_bias={has_w2_bias}, which "
                    f"{'wants' if present else 'has no operand for'} the "
                    f"{what}, and it is "
                    f"{'present' if operand is not None else 'absent'}"
                )
        act_scale_arg = (
            (
                act_scales.reshape(
                    host.act_scale_slab_rows(ragged_rows_alloc), HIDDEN_LANE_BLOCK
                ),
            )
            if has_act_scale
            else ()
        )
        scale_args = (
            (w1s.astype(jnp.float32), w2s.astype(jnp.float32)) if has_scales else ()
        )
        biases = tuple(b.astype(jnp.float32) for b in (w1b, w2b) if b is not None)
        call = make_call(recv_rows)
        res = call(
            *tables,
            expert_rows.astype(jnp.int32),
            expert_base.astype(jnp.int32),
            visit.astype(jnp.int32),
            counts.astype(jnp.int32),
            rank.astype(jnp.int32),
            *(d.astype(jnp.int32) for d in dispatch),
            act_q_sorted.reshape(-1, lane_blocks, HIDDEN_LANE_BLOCK),
            *act_scale_arg,
            w1,
            w2,
            *scale_args,
            *biases,
            slab.reshape(-1, lane_blocks, HIDDEN_LANE_BLOCK),
        )
        recv, rscl, _contrib, _contrib_scl, _slab = res
        return recv, rscl

    return fn


_BUILD_CACHE = {}
_BUILD_CACHE_LOCK = threading.Lock()


def build_push_moe_kernel(**kwargs):
    """Memoizing front door for _build_push_moe_kernel (see v2's)."""
    key = tuple(sorted(kwargs.items()))
    fn = _BUILD_CACHE.get(key)
    if fn is None:
        with _BUILD_CACHE_LOCK:
            fn = _BUILD_CACHE.get(key)
            if fn is None:
                fn = _build_push_moe_kernel(**kwargs)
                _BUILD_CACHE[key] = fn
                logger.info(
                    "push fused EP MoE: built program %d for %s",
                    len(_BUILD_CACHE),
                    ", ".join(f"{k}={v}" for k, v in key),
                )
    return fn


def _build_combine_fs_kernel(
    *, t_local, topk, hidden, lane_blocks, wire_dtype, out_dtype
):
    """out[i] = sum_k scales[scl[i, k]] * w[i, k] * arrivals[pos[i, k]].

    v2's combine kernel with the arrival-scale lookup folded in: `scl` (the
    clamped mirror element), the router weight `w` and the flattened arrival
    scale mirror ride scalar prefetch, so `arrival_scales[scl]` is read in
    SMEM rather than by an XLA gather, which the SparseCore offloads and
    serializes behind the other SparseCore work. Same f32 product, same order.

    An arrival row is [lane blocks, 128] -- one register -- while the output
    is [tokens, hidden], which is that layout transposed: a token's hidden
    axis lies along sublanes in the row and along lanes in the output. The sum
    therefore runs in the ROW's layout and only its bf16 result is relaid out,
    so the sublane shuffle sees one bf16 token per group instead of topk
    packed-f8 rows, and the live accumulator is one token wide.
    """
    tb = combine_step_tokens(t_local)
    steps = t_local // tb
    gt = COMBINE_GROUP
    groups = tb // gt
    lanes = HIDDEN_LANE_BLOCK

    def kernel(
        pos_sm,
        scl_sm,
        tw_sm,
        scales_sm,
        arr_hbm,
        out_hbm,
        rows_vm,
        tile_vm,
        out_vm,
        row_sems,
        out_sems,
    ):
        def start_row(base, parity, i, k):
            # Alternate queues by slot: this kernel is one 4 KiB descriptor per
            # arrival row and an in-order queue retires descriptors at a fixed
            # rate, which for a row this small is slower than the row's own
            # bytes. Both queues signal the one semaphore the step waits on,
            # which counts bytes and not descriptors.
            pltpu.make_async_copy(
                arr_hbm.at[pos_sm[(base + i) * topk + k]],
                rows_vm.at[parity, k * tb + i],
                row_sems.at[parity],
            ).start(priority=k % 2)

        def reduce_token(base, parity, i):
            """One token's weighted sum, in the arrival rows' own layout."""
            acc = None
            for k in range(topk):
                idx = (base + i) * topk + k
                c_val = scales_sm[scl_sm[idx]] * tw_sm[idx]
                term = rows_vm[parity, k * tb + i].astype(jnp.float32) * c_val
                acc = term if acc is None else acc + term
            tile_vm[i] = cast(jax.Array, acc).astype(out_dtype)

        def one_step(step, parity):
            # The step's rows were started a step ago; the next step's are
            # started beside this one's reduction, so their descriptor issue
            # rides in slots the vector work leaves empty. The last step
            # re-starts its own rows rather than branching -- the copy is never
            # read and is drained below.
            nxt = jnp.minimum(step + COMBINE_LEAD, steps - 1) * tb
            base = step * tb
            npar = (parity + COMBINE_LEAD) % COMBINE_PARITIES
            pltpu.make_async_copy(
                rows_vm.at[parity], rows_vm.at[parity], row_sems.at[parity]
            ).wait()

            @pl.when(step >= COMBINE_PARITIES)
            def _():
                pltpu.make_async_copy(
                    out_vm.at[parity], out_vm.at[parity], out_sems.at[parity]
                ).wait()

            def one_chunk(chunk, carry):
                for u in range(COMBINE_CHUNK):
                    i = chunk * COMBINE_CHUNK + u
                    for k in range(topk):
                        start_row(nxt, npar, i, k)
                    reduce_token(base, parity, i)
                return carry

            lax.fori_loop(0, tb // COMBINE_CHUNK, one_chunk, jnp.int32(0))
            for g in range(groups):
                out_vm[parity, pl.ds(g * gt, gt)] = tile_vm[pl.ds(g * gt, gt)].reshape(
                    gt, hidden
                )
            pltpu.make_async_copy(
                out_vm.at[parity], out_hbm.at[pl.ds(step * tb, tb)], out_sems.at[parity]
            ).start()

        def step_pair(pair, carry):
            for parity in range(COMBINE_PARITIES):
                one_step(pair * COMBINE_PARITIES + parity, parity)
            return carry

        for p in range(COMBINE_LEAD):
            for i in range(tb):
                for k in range(topk):
                    start_row(p * tb, p, i, k)
        lax.fori_loop(0, steps // COMBINE_PARITIES, step_pair, jnp.int32(0))
        # The last COMBINE_LEAD steps re-start the final step's rows into the
        # buffers the loop no longer reads; drain them, and the stores.
        for p in range(COMBINE_LEAD):
            pltpu.make_async_copy(rows_vm.at[p], rows_vm.at[p], row_sems.at[p]).wait()
        for p in range(COMBINE_PARITIES):
            pltpu.make_async_copy(out_vm.at[p], out_vm.at[p], out_sems.at[p]).wait()

    hbm = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)
    call = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=[hbm],
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM(
                    (COMBINE_PARITIES, topk * tb, lane_blocks, lanes), wire_dtype
                ),
                pltpu.VMEM((tb, lane_blocks, lanes), out_dtype),
                pltpu.VMEM((COMBINE_PARITIES, tb, hidden), out_dtype),
                pltpu.SemaphoreType.DMA((COMBINE_PARITIES,)),
                pltpu.SemaphoreType.DMA((COMBINE_PARITIES,)),
            ),
            grid=(),
        ),
        out_shape=jax.ShapeDtypeStruct((t_local, hidden), out_dtype),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=host.vmem_limit(), disable_bounds_checks=True
        ),
        name=f"moe_v2_combine_t{t_local}_k{topk}_b{tb}_p{COMBINE_PARITIES}_fs",
    )

    def fn(arrivals, arrival_scales, scl, topk_weights, pos):
        # Every table flattens TOKEN-major, like v2's combine.
        return call(
            pos.astype(jnp.int32).reshape(-1),
            scl.astype(jnp.int32).reshape(-1),
            topk_weights.astype(jnp.float32).reshape(-1),
            arrival_scales.astype(jnp.float32).reshape(-1),
            arrivals,
        )

    return fn


_COMBINE_CACHE = {}
_COMBINE_CACHE_LOCK = threading.Lock()


def build_combine_fs_kernel(**kwargs):
    """Memoizing front door for _build_combine_fs_kernel."""
    key = tuple(sorted((k, str(v)) for k, v in kwargs.items()))
    fn = _COMBINE_CACHE.get(key)
    if fn is None:
        with _COMBINE_CACHE_LOCK:
            fn = _COMBINE_CACHE.get(key)
            if fn is None:
                fn = _build_combine_fs_kernel(**kwargs)
                _COMBINE_CACHE[key] = fn
    return fn
