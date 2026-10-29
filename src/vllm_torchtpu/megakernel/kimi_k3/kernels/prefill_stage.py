# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 prefill as pipeline stages: each host's 8 cores run a contiguous
run of whole 4-layer groups (``STAGE_LAYERS``) over the window, sequence
parallel over the stage (SP8).

A stage rank holds 12 of the 96 attention heads as four 3-head groups (each
a TP32 rank's shard and layout) and, EP4 x TP2, its chip's 224 experts over
its core's half of the intermediate (``rank_loader.iter_prefill_stage_rank``).
The layers are the slice program's, over the stage mesh:

* KDA (``make_kda_layer_stage``): the slice layer's pipeline (window
  all-gather and output-projection reduce-scatter under the delta rule)
  over the rank's four head groups, one group after another;
* MLA (``make_mla_layer_stage``): projections down to the query and KV
  latents and through the output gate on the rank's own rows, the latents'
  all-gather, attention at the rank's 12 heads, then an all-to-all back to
  the rows' ranks and the output projection of every head;
* MoE: sequence-parallel routing, the latent all-gather, then the chip's
  routed experts over every token with their reduce-scatter ring in the
  fused expert pipeline (``make_moe_layer_stage``; above 8K the SparseCore
  path, ``make_moe_fused_sp_repacked``).

A stage's ops take and return the residual stream and the residual block
snapshots (``prefix``, ``blocks``); stage boundaries are block boundaries,
so the next stage continues from them.
"""

from __future__ import annotations

import contextlib
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel.kimi_k3.kernels import prefill_megakernel as k
from vllm_torchtpu.megakernel.kimi_k3.kernels.rank_loader import (
    STAGE_HEAD_GROUPS,
    stage_slots,
)

STAGE_LAYERS = ((0, 24), (24, 48), (48, 72), (72, 93))
GROUP_HEADS = 3  # heads per head group: one TP32 rank's shard


def _phase(name):
    """Decorate a kernel phase with an XProf scope (KernelConfig.phase_scopes)."""

    def wrap(function):
        @functools.wraps(function)
        def scoped(*args, **kwargs):
            scope = jax.named_scope(name) if k._config().phase_scopes else contextlib.nullcontext()
            with scope:
                return function(*args, **kwargs)

        return scoped

    return wrap


def _kda_unit(blocks):
    """SP blocks per pipelined KDA unit: the largest divisor of ``blocks``
    up to 4 that leaves at least two units (each unit reloads the head
    groups' weights; more units shorten the pipeline's fill and drain)."""
    return max(u for u in range(1, 5) if blocks % u == 0 and (blocks // u >= 2 or u == 1))


def _kda_layer_kernel(
    slot_ref, segment_ref, active_ref, tables_ref, length_ref, replace_ref,
    prefix_ref, blocks_ref, folded_ref, norm_ref,
    gate_family_ref, fb_family_ref, proj_family_ref, conv_family_ref,
    alog_family_ref, dt_family_ref, kda_norm_family_ref,
    rec_in_ref, hist_in_ref, output_family_ref,
    normalized_ref, gathered_ref, attended_ref, rec_out_ref, hist_out_ref,
    final_ref, rec_at_ref, hist_at_ref, partial_ref,
    pool, *scratch, eps, count, blocks, unit,
):
    """One KDA layer of a stage rank's four head groups, pipelined like the
    slice layer (``k._kda_layer_body``) over units of ``unit`` SP blocks:
    grid step ``s`` runs ring step ``s % ring`` of block ``s // ring``'s
    window all-gather, a share of unit ``s // (ring * unit) - 1``'s KDA
    tiles and the reduce-scatter of block ``s // ring - 2 * unit``. A unit
    runs one head group after another (the order reversed every other unit,
    so a group's weights stay loaded across the boundary), each group
    writing its own output-projection partials (``partial_ref`` ``[4, T,
    hidden]``), which the reduce-scatter sums."""
    slot = slot_ref[0]
    n = tables_ref.shape[-1]
    ring = k.half_steps(n)
    step = pl.program_id(1)
    stripe = k.window_stripe(prefix_ref.shape[0])
    mix_small, gather_sems = scratch[:8], scratch[8:11]
    reduce_sems, projection_scratch = scratch[11:16], scratch[16:19]
    gate_w_vmem, rec_vmem, hist_vmem, kda_scratch = scratch[19], scratch[20], scratch[21], scratch[22:]
    groups = STAGE_HEAD_GROUPS
    width = GROUP_HEADS * 128

    def views(shapes):  # the shared pool: the mixture, then the projection ring
        offset, result = 0, []
        for shape, dtype in shapes:
            size = k.pool_alias.storage_size(shape, jnp.dtype(dtype))
            result.append(k.pool_alias.view(
                pool.at[pl.ds(offset // 128, size // 128), :], shape, dtype))
            offset += size
        return result

    pl.when(step == 0)(lambda: k._ring_enter(tables_ref, jax.lax.axis_index("k3_sp")))

    @pl.when(step < blocks)
    @_phase("kda_normalize")
    def normalize():
        rows = pl.ds(step * stripe, stripe)
        mix = views(k._kda_mix_shapes(stripe))
        k.mixture_norm_tiled_body(
            active_ref, prefix_ref.at[rows], blocks_ref.at[:, rows], folded_ref, norm_ref,
            normalized_ref.at[rows, :7168], *mix, *mix_small, tile=stripe, eps=eps,
        )

        @pl.when(step == 0)
        def load_gate():
            copy = pltpu.make_async_copy(gate_family_ref.at[slot], gate_w_vmem, kda_scratch[-1].at[0])
            copy.start()
            copy.wait()

        # The gate compression of the rank's rows travels with them.
        @functools.partial(pl.run_scoped, features=pltpu.VMEM((stripe, 128), jnp.bfloat16))
        def exchange_features(features):
            features[...] = k._dot(mix[2][0], gate_w_vmem[...]).astype(jnp.bfloat16)
            copy = pltpu.make_async_copy(features, normalized_ref.at[rows, 7168:], gather_sems[2])
            copy.start()
            copy.wait()

    units = blocks // unit
    unit_steps = ring * unit
    unit_tiles = n * stripe * unit // 128  # one head group's tiles in a unit
    phase = step // unit_steps
    current = jnp.maximum(phase - 1, 0)
    attending = (phase >= 1) & (phase <= units)
    share = step % unit_steps
    lo = jnp.where(attending, share * groups * unit_tiles // unit_steps, 0)
    hi = jnp.where(attending, (share + 1) * groups * unit_tiles // unit_steps, 0)

    # One call site for the KDA tile (instruction memory).
    @pl.loop(0, groups)
    @_phase("kda_tiles")
    def head_group(order):
        g = jnp.where(current % 2 == 0, order, groups - 1 - order)
        begin = jnp.clip(lo - order * unit_tiles, 0, unit_tiles)
        end = jnp.clip(hi - order * unit_tiles, 0, unit_tiles)
        starts = (begin == 0) & (end > 0)
        base = current * unit_tiles
        heads = pl.ds(g * width, width)
        tile_scratch = (*kda_scratch[:8], rec_vmem.at[g], hist_vmem.at[g], *kda_scratch[8:])
        k.kda_prefill_tiled_body(
            gathered_ref, fb_family_ref.at[slot, g], proj_family_ref.at[slot, g],
            conv_family_ref.at[slot, g], alog_family_ref.at[slot, g], dt_family_ref.at[slot, g],
            kda_norm_family_ref.at[slot, g], rec_in_ref.at[:, g], hist_in_ref.at[:, g],
            attended_ref.at[:, heads], rec_out_ref.at[:, g], hist_out_ref.at[:, g],
            *tile_scratch,
            tile=128, decay_lower_bound=-5.0, epsilon=eps,
            checkpoint=(rec_at_ref.at[:, g], hist_at_ref.at[:, g]),
            segments=(segment_ref, count),
            tiles=(base + begin, base + end, starts & (current == 0),
                   (end == unit_tiles) & (begin < end) & (current == units - 1),
                   starts & ((current == 0) | (order > 0))),
            block_tiles=unit_tiles,
            projection=(output_family_ref.at[slot, heads], projection_scratch[0],
                        partial_ref.at[g], *projection_scratch[1:]),
        )

    block = step // ring

    @pl.when(block < blocks)
    @_phase("kda_gather")
    def gather():
        k._half_gather_step(tables_ref, normalized_ref, gathered_ref, *gather_sems,
                            step=step % ring, block=block)

    @pl.when(block >= 2 * unit)
    @_phase("kda_reduce")
    def reduce():
        k._half_reduce_step(
            tables_ref, length_ref, partial_ref, prefix_ref, final_ref,
            *reduce_sems, *views(k._kda_reduce_shapes(stripe, (groups,))),
            step=step % ring, block=block - 2 * unit, replace=replace_ref[0] > 0,
        )


def make_kda_layer_stage(mesh, *, seq_len, segments, collective_id=38, eps=1e-5):
    """One KDA layer of a stage over a packed window of ``segments``
    sequences as one Pallas call (``_kda_layer_kernel``). ``run(prefix,
    blocks, attn_res, attn_norm, families, rec, hist, active, length, slot,
    segment_desc, replace)`` returns ``(output, rec, hist, rec_at,
    hist_at)``: the new residual rows and the stage rank's head groups'
    states (``[n, segments, 4, ...]``), ``*_at`` at each sequence's
    checkpoint. ``replace``: the layer starts a residual block, its
    attention output replaces the partial sum."""
    n = k.check_mesh(mesh, "stage KDA")
    k.check_window(seq_len, "stage KDA", n)
    if not segments:
        raise ValueError("the KDA layer runs packed windows")
    groups = STAGE_HEAD_GROUPS
    rows, width = seq_len // n, 7168
    stripe = k.window_stripe(rows)
    blocks = rows // stripe
    unit = _kda_unit(blocks)
    tables = jnp.asarray(k.mesh_ring_tables(mesh), jnp.int32)
    streams = len(k._half_ring_streams(width))
    pool_bytes = max(
        sum(k.pool_alias.storage_size(shape, jnp.dtype(dtype))
            for shape, dtype in shapes)
        for shapes in (k._kda_mix_shapes(stripe), k._kda_reduce_shapes(stripe, (groups,)))
    )
    shapes = (
        jax.ShapeDtypeStruct((128, 384), jnp.bfloat16),
        jax.ShapeDtypeStruct((width, 1664), jnp.bfloat16),
        jax.ShapeDtypeStruct((4, 3, 3, 128), jnp.float32),
        jax.ShapeDtypeStruct((3, 1), jnp.float32),
        jax.ShapeDtypeStruct((3, 128), jnp.float32),
        jax.ShapeDtypeStruct((8, 128), jnp.float32),
    )
    kda_scratch = k.kda_prefill_tiled_scratch_shapes(width + 128, 128, *shapes)
    scratch = (
        pltpu.VMEM((pool_bytes // 128, 128), jnp.uint8),
        *k.mixture_norm_tiled_scratch_shapes(width, stripe)[4:],
        pltpu.SemaphoreType.DMA((streams,)),  # the window all-gather
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA,
        pltpu.SemaphoreType.DMA((2, streams)),  # the reduce-scatter
        pltpu.SemaphoreType.DMA((streams,)),
        pltpu.SemaphoreType.DMA((2, streams)),
        pltpu.SemaphoreType.REGULAR((streams,)),
        pltpu.SemaphoreType.DMA,
        pltpu.VMEM((GROUP_HEADS * 128, width), jnp.bfloat16),  # a group's output projection
        pltpu.VMEM((2, 128, width), jnp.bfloat16),  # its partials
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.VMEM((width, 128), jnp.bfloat16),  # the gate compression weight
        pltpu.VMEM((groups, 3, 128, 128), jnp.float32),  # each group's recurrent state
        pltpu.VMEM((groups, 3, 3, 3, 128), jnp.bfloat16),  # and convolution history
        *kda_scratch[:8], *kda_scratch[10:],
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    state = (segments, groups)
    output_shapes = (
        jax.ShapeDtypeStruct((rows, width + 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq_len, width + 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((seq_len, groups * GROUP_HEADS * 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((*state, 3, 128, 128), jnp.float32),
        jax.ShapeDtypeStruct((*state, 3, 3, 3, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((*state, 3, 128, 128), jnp.float32),
        jax.ShapeDtypeStruct((*state, 3, 3, 3, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((groups, seq_len, width), jnp.bfloat16),  # projection partials
    )
    call = pl.pallas_call(
        functools.partial(_kda_layer_kernel, count=segments, eps=eps, blocks=blocks, unit=unit),
        out_shape=output_shapes,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            grid=(1, k.half_steps(n) * (blocks + 2 * unit)),
            in_specs=(hbm,) * 14,
            out_specs=(hbm,) * len(output_shapes),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id, vmem_limit_bytes=64 * 1024**2,
        ),
        name="kimi_prefill_stage_kda_layer",
    )

    def local(prefix, blocks, folded, norm, families, rec, hist, active, length, slot,
              segment_desc, replace):
        w = {name: value[0] for name, value in families.items()}
        outputs = call(
            slot, segment_desc, active, tables, length, replace,
            prefix[0], blocks[0], folded[0], norm[0],
            w["k_gate"], w["k_fb"], w["k_projection"], w["k_conv"],
            w["k_a_log"], w["k_dt"], w["k_norm"], rec[0], hist[0], w["k_o"],
        )
        _, _, _, new_rec, new_hist, final, rec_at, hist_at, _ = outputs
        return tuple(value[None] for value in (final, new_rec, new_hist, rec_at, hist_at))

    sharded, replicated = P("k3_sp"), P()
    return jax.jit(jax.shard_map(
        local, mesh=mesh,
        in_specs=(sharded,) * 7 + (replicated,) * 5,
        out_specs=(sharded,) * 5, check_vma=False,
    ))


MLA_PACKED = 2176  # gathered lanes: query latent 1536, compressed KV 576, padding


def make_mla_layer_stage(mesh, *, seq_len, segments, prefix_capacity, flash_tile=256, eps=1e-5):
    """One MLA layer of a stage with row-local projections: each rank
    normalizes its rows and projects them to the query and KV latents and
    through the output gate of all 96 heads; an all-gather of the latents
    gives every rank the window, which it attends at its 12 heads over the
    window and the cached prefixes; an all-to-all hands each rank its rows
    of every head, which it gates and projects out. ``run(prefix, blocks,
    folded, norm, active, families, segment_desc, length, slot, arena)``
    returns the new residual rows and the rank-leading ``[n, T, 576]``
    compressed KV (the same on every rank)."""
    n = k.check_mesh(mesh, "stage MLA")
    k.check_window(seq_len, "stage MLA", n)
    if not segments:
        raise ValueError("the stage MLA layer runs packed windows")
    tables = k.mesh_ring_tables(mesh)
    rows = seq_len // n
    stripe = k.window_stripe(rows)
    mixture = k.make_mixture_norm_sp(mesh, seq_len=seq_len, eps=eps)
    fields = dict(zip(k.MLA_FAMILIES, k.MLA_FIELD_NAMES, strict=True))

    def local(hidden, prefix, families, segment_desc, length, slot, pool, ids, desc):
        w = {fields[name]: jax.lax.dynamic_index_in_dim(value[0], slot[0], keepdims=False)
             for name, value in families.items()}
        projections = k.ProjectionWeights(*(w[name] for name in k._PROJECTION_NAMES))
        x = hidden[0]
        query_latent, latent = k.mla_latents(x, projections, epsilon=eps)
        packed = jnp.pad(jnp.concatenate((query_latent, latent), axis=1),
                         ((0, 0), (0, MLA_PACKED - 1536 - 576)))
        gate = k._dot(x, w["output_gate_projection"]).astype(jnp.bfloat16)
        gate = jax.nn.sigmoid(gate.astype(jnp.float32)).astype(jnp.bfloat16)
        # Every rank's rows, in window order (stripes of ``stripe`` rows round the ranks).
        window = jax.lax.all_gather(packed, "k3_sp").reshape(n, rows // stripe, stripe, MLA_PACKED)
        window = window.transpose(1, 0, 2, 3).reshape(seq_len, MLA_PACKED)
        latent = window[:, 1536:1536 + 576]
        query, new_key, new_value = k.mla_heads(
            window[:, :1536], latent, projections, query_scale=k.SCALE_LOG2)
        real = (jnp.arange(seq_len) < length.reshape(()))[None, :, None]
        new_key = jnp.where(real, new_key.transpose(1, 0, 2), 0)
        new_value = jnp.where(real, new_value.transpose(1, 0, 2), 0)
        heads = query.shape[1]
        pages, order = k.gathered_prefix_pages(pool, ids[0], desc[0], tables, "k3_sp",
                                               count=segments, capacity=prefix_capacity)
        arena = k.expand_prefix_pages(pages, order, projections.key_value_up_projection)
        attended = k.mla_flash_prefill_packed(
            query.transpose(1, 0, 2), new_key, new_value, length, segment_desc[:5 * segments],
            count=segments, tile=flash_tile, arena=arena)
        # Each rank's rows of this rank's heads to that rank: [source rank, rows, heads * 128].
        attended = attended.transpose(1, 0, 2).reshape(rows // stripe, n, stripe, heads * 128)
        attended = attended.transpose(1, 0, 2, 3).reshape(n, rows, heads * 128)
        attended = jax.lax.all_to_all(attended, "k3_sp", 0, 0)
        attended = attended.transpose(1, 0, 2).reshape(rows, n * heads * 128)
        projected = k._dot(attended * gate, w["output_projection_repacked"])
        rank = jax.lax.axis_index("k3_sp")
        mine = k.sp_token_index(rank, jnp.arange(rows), stripe, n) < length[0]
        output = prefix[0] + jnp.where(mine[:, None], projected, 0)
        return output.astype(jnp.bfloat16)[None], latent[None]

    sharded, replicated = P("k3_sp"), P()
    mapped = jax.shard_map(
        local, mesh=mesh,
        in_specs=(sharded, sharded, sharded, replicated, replicated, replicated,
                  sharded, sharded, sharded),
        out_specs=(sharded, sharded), check_vma=False,
    )

    def run(prefix, blocks, folded, norm, active, families, segment_desc, length, slot, arena):
        (pool,) = arena.pools
        hidden = mixture(prefix, blocks, folded, norm, active)
        families = {name: families[name] for name in k.MLA_FAMILIES}
        return mapped(hidden, prefix, families, segment_desc, length, slot, pool, arena.ids,
                      arena.local)

    return jax.jit(run)


MOE_CAPACITY = 65536  # routes per chip's experts per window: twice the 8K mean


def _expert_metadata(ids, probs, first_expert, local_experts, capacity):
    """This chip's routes in expert-major, token-major order (as the fused
    slice MoE sorts them): each route's token row and weight
    (``[capacity / 128, 128]`` each, zero past the last route), the routes
    per local expert and their total (more than ``capacity``: dropped)."""
    tokens, top = ids.shape
    local = ids.astype(jnp.int32) - first_expert
    valid = (local >= 0) & (local < local_experts) & (probs != 0)
    keys = jnp.where(valid, local, local_experts).reshape(-1)
    position = jnp.arange(tokens * top, dtype=jnp.int32)
    order = jnp.argsort(keys * (tokens * top) + position)[:capacity]
    sorted_keys = keys[order]
    # One compare-and-sum pass (XLA's searchsorted is a serial scan).
    counts = jnp.sum(
        keys.reshape(tokens, top)[:, :, None] == jnp.arange(local_experts, dtype=jnp.int32),
        axis=(0, 1), dtype=jnp.int32)
    rows = jnp.where(sorted_keys < local_experts, position[order] // top, 0)
    weights = jnp.where(sorted_keys < local_experts, probs.reshape(-1)[order], 0)
    shape = (capacity // 128, 128)
    return (rows.astype(jnp.int32).reshape(shape), weights.astype(jnp.float32).reshape(shape),
            counts, jnp.sum(counts))


def make_moe_layer_stage(mesh, *, seq_len, capacity=MOE_CAPACITY, eps=1e-5, collective_id=45):
    """One MoE layer of a stage: sequence-parallel routing, the latent and
    route all-gathers, then this chip's experts (EP4 x TP2) over every
    token through the fused expert pipeline and its reduce-scatter ring,
    and the latent-up / shared-expert tail. ``run(hidden, prefix,
    families, length, slot)`` returns ``(output, routes)``: the new
    residual rows and the busiest chip's route count. A window whose
    busiest chip has more than ``capacity`` routes (skewed routing) takes
    the SparseCore path instead, which keeps every route."""
    from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_moe import fused
    from vllm_torchtpu.megakernel.kimi_k3.kernels.prefill_moe.routing_topk_kernel import (
        select_top16,
    )

    n = k.check_mesh(mesh, "stage MoE")
    k.check_window(seq_len, "stage MoE", n)
    capacity = min(capacity, seq_len * 16)  # no more routes than the window has
    tables = k.mesh_ring_tables(mesh)
    local_tokens = seq_len // n
    selected_names = ("router", "router_bias", "latent_up", "latent_norm")

    def route(hidden, families, length, slot):
        w = {name: value[0] for name, value in families.items()}
        for name in selected_names:
            w[name] = w[name][slot[0]]
        rank = jax.lax.axis_index("k3_sp")
        x = hidden[0]
        logits = jax.lax.dot_general(
            x, w["router"], (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        raw = jax.nn.sigmoid(logits)
        pad = -local_tokens % 128  # the top-16 kernel takes whole 128-row tiles
        local_ids, local_probs = select_top16(jnp.pad(raw, ((0, pad), (0, 0))), w["router_bias"])
        local_ids, local_probs = local_ids[:local_tokens], local_probs[:local_tokens]
        valid_rows = k.sp_token_index(
            rank, jnp.arange(local_tokens), k.window_stripe(local_tokens), n) < length.reshape(())
        local_probs = jnp.where(valid_rows[:, None], local_probs, 0)
        latent_local = k.gmm_v2(
            x, w["latent_down"][:, None], jnp.full((1,), local_tokens, jnp.int32),
            rhs_layer=slot)
        latent_all = k._moe_latent_all_gather(latent_local[:, :3584], tables)
        routes = jnp.concatenate(
            (local_ids.astype(jnp.int32), jax.lax.bitcast_convert_type(local_probs, jnp.int32)),
            axis=1)
        routes = jax.lax.all_gather(routes, "k3_sp").reshape(seq_len, 32)
        ids_all = routes[:, :16]
        probs_all = jax.lax.bitcast_convert_type(routes[:, 16:], jnp.float32)
        local_experts = w["expert_gu"].shape[-3]
        expert_tp = n * local_experts // 896
        ids, weights, counts, total = _expert_metadata(
            ids_all, probs_all, (rank // expert_tp) * local_experts, local_experts, capacity)
        busiest = jax.lax.pmax(total, "k3_sp")
        return latent_all[None], ids[None], weights[None], counts[None], busiest

    def experts(hidden, prefix, families, length, slot, latent_all, ids, weights, counts):
        w = {name: value[0] for name, value in families.items()}
        for name in selected_names:
            w[name] = w[name][slot[0]]
        w["expert_layer"] = slot
        reduced = fused.routed_experts_ring(
            latent_all[0], ids[0], weights[0], counts[0], slot, w["expert_gu"],
            w["expert_gus"], w["expert_down"], w["expert_ds"], tables, tokens=seq_len,
            capacity=capacity, collective_id=collective_id,
        )
        shared = k._moe_sp_shared_partial(k._moe_sp_shared_rows(hidden[0], "k3_sp", n), w)
        output = k._moe_sp_tail(reduced, shared, prefix[0], length, w, axis_name="k3_sp",
                                eps=eps, n=n)
        return output[None]

    sharded, replicated = P("k3_sp"), P()
    # The busiest count is the same on every rank: replicated, the branch
    # predicate needs no partition id.
    routed = jax.shard_map(
        route, mesh=mesh, in_specs=(sharded, sharded, replicated, replicated),
        out_specs=(sharded,) * 4 + (replicated,), check_vma=False,
    )
    fused_experts = jax.shard_map(
        experts, mesh=mesh, in_specs=(sharded,) * 3 + (replicated,) * 2 + (sharded,) * 4,
        out_specs=sharded, check_vma=False,
    )
    reference = k._make_indexed_moe_sp_ring_reference(mesh, seq_len=seq_len)

    def run(hidden, prefix, families, length, slot):
        families = {name: families[name] for name in k.MOE_FAMILIES}
        latent_all, ids, weights, counts, busiest = routed(hidden, families, length, slot)
        output = jax.lax.cond(
            busiest <= capacity,
            lambda: fused_experts(hidden, prefix, families, length, slot, latent_all, ids,
                                  weights, counts),
            lambda: reference(hidden, prefix, families, length, slot),
        )
        return output, jnp.broadcast_to(busiest, (n,))

    return jax.jit(run)


def _choose(family, slot):
    return jax.lax.dynamic_index_in_dim(family, slot, axis=1, keepdims=False)


def _put_state(family, state, slot):
    return jax.lax.dynamic_update_slice(
        family, state[:, None], (0, slot, *([0] * (family.ndim - 2))))


def _scalar(value):
    return jnp.asarray([value], jnp.int32)


def _stage_moe(mesh, seq_len, moe_path):
    """``run(hidden, prefix, families, length, slot) -> (output, routes)``:
    the fused expert pipeline (``routes``: the busiest chip's route count;
    windows up to 8K, whose token sums fit VMEM) or the SparseCore path
    (``routes`` zero)."""
    if moe_path == "fused" and seq_len <= 8192:
        return make_moe_layer_stage(mesh, seq_len=seq_len)
    reference = k._make_indexed_moe_sp_ring_reference(mesh, seq_len=seq_len)
    n = mesh.size

    def run(hidden, prefix, families, length, slot):
        return reference(hidden, prefix, families, length, slot), jnp.zeros((n,), jnp.int32)

    return run


def make_stage_group4(mesh, *, seq_len, segments, prefix_capacity, slots, first=False,
                      mla_flash_tile=256, moe_path="fused", tap_offset=None):
    """One runtime-indexed group at layers 4g..4g+3 of a stage whose weight
    families start at ``slots`` (``rank_loader.stage_slots``): three KDA
    layers and one MLA layer, each followed by its MoE (layer 0, with
    ``first``: the dense FFN). ``run(prefix, blocks, common, kda_families,
    mla_families, moe_families, kda_state, mla_state, length, group,
    segment_desc, arena)`` as ``make_regular_group4_sp_indexed``, the
    states stage-local (``kda_state``: ``(rec, hist, rec_at, hist_at)``,
    ``[n, stage KDA layers, segments, 4, ...]``), then the busiest chip's
    route count of its MoE layers (``[n]``); with ``tap_offset``, then the
    residual stream entering layer ``4g + tap_offset`` (a DSpark aux layer)."""
    n = k.check_mesh(mesh, "stage group")
    if tap_offset is not None and not 1 <= tap_offset < 4:
        raise ValueError("tap_offset must name one of the group's later layers")
    k.check_window(seq_len, "stage group", n)
    sharding = NamedSharding(mesh, P("k3_sp"))
    mixture = k.make_mixture_norm_sp(mesh, seq_len=seq_len)
    kda = make_kda_layer_stage(mesh, seq_len=seq_len, segments=segments)
    mla = make_mla_layer_stage(mesh, seq_len=seq_len, flash_tile=mla_flash_tile,
                               segments=segments, prefix_capacity=prefix_capacity)
    moe = _stage_moe(mesh, seq_len, moe_path)
    dense = k.make_dense_ffn_sp(mesh, seq_len=seq_len) if first else None
    layer_lo, kda_lo = slots["layer"][0], slots["kda"][0]
    mla_lo, moe_lo = slots["mla"][0], slots["moe"][0]

    def run(prefix, blocks, common, kda_families, mla_families, moe_families,
            kda_state, mla_state, length, group, segment_desc, arena):
        base = group[0] * 4
        kda_states = tuple(kda_state)
        mla_cache = tuple(mla_state)
        routes = jnp.zeros((n,), jnp.int32)
        tap = None
        # An in-place update of one snapshot (a cond would copy all eight).
        snapshot = base // 12
        blocks = blocks.at[:, snapshot].set(
            jnp.where(base % 12 == 0, prefix, blocks[:, snapshot]))
        for offset in range(4):
            index = base + offset
            if offset == tap_offset:
                tap = prefix
            attn_res = _choose(common["attn_res"], index - layer_lo)
            attn_norm = _choose(common["attn_norm"], index - layer_lo)
            attn_active = _scalar((index - 1) // 12 + 1)
            if offset < 3:
                kda_slot = base - group[0] + offset  # the layer's global KDA slot
                slot = kda_slot - kda_lo
                after_attention, *states = kda(
                    prefix, blocks, attn_res, attn_norm, kda_families,
                    _choose(kda_states[0], slot), _choose(kda_states[1], slot),
                    attn_active, length, _scalar(slot), segment_desc,
                    _scalar((kda_slot % 9 == 0).astype(jnp.int32)),
                )
                kda_states = tuple(_put_state(family, state, slot)
                                   for family, state in zip(kda_states, states, strict=True))
            else:
                slot = group[0] - mla_lo
                after_attention, *updates = mla(
                    prefix, blocks, attn_res, attn_norm, attn_active, mla_families,
                    segment_desc, length, _scalar(slot), arena,
                )
                mla_cache = tuple(_put_state(family, update, slot)
                                  for family, update in zip(mla_cache, updates, strict=True))
            ffn_input = mixture(
                after_attention, blocks, _choose(common["mlp_res"], index - layer_lo),
                _choose(common["ffn_norm"], index - layer_lo),
                _scalar(index // 12 + 1),
            )
            if dense is not None and offset == 0:
                prefix = dense(ffn_input, after_attention, common["dense_gu"],
                               common["dense_down"], length)
            else:
                prefix, busiest = moe(ffn_input, after_attention, moe_families, length,
                                      _scalar(index - 1 - moe_lo))
                routes = jnp.maximum(routes, busiest)
        if tap_offset is None:
            return prefix, blocks, kda_states, mla_cache, routes
        return prefix, blocks, kda_states, mla_cache, routes, tap

    return jax.jit(run, out_shardings=(
        sharding, sharding, (sharding,) * 4, (sharding,), sharding)
        + ((sharding,) if tap_offset is not None else ()))


def make_stage_final(mesh, *, seq_len, segments, prefix_capacity, slots, mla_flash_tile=256,
                     moe_path="fused"):
    """The last stage's layer 92 (MLA and its MoE) and each sequence's
    first-token logits: ``run(prefix, blocks, common, mla_families,
    moe_families, mla_state, length, segment_desc, arena, io)`` returns
    ``(logits [segments, V] replicated, mla_state, routes)``."""
    k.check_mesh(mesh, "stage final")
    mixture = k.make_mixture_norm_sp(mesh, seq_len=seq_len)
    mla = make_mla_layer_stage(mesh, seq_len=seq_len, flash_tile=mla_flash_tile,
                               segments=segments, prefix_capacity=prefix_capacity)
    moe = _stage_moe(mesh, seq_len, moe_path)
    logits_op = k.make_prefill_logits_sp(mesh, seq_len=seq_len, layers=93, segments=segments)
    layer = 92 - slots["layer"][0]
    mla_slot, moe_slot = 23 - slots["mla"][0], 91 - slots["moe"][0]

    def run(prefix, blocks, common, mla_families, moe_families, mla_state, length,
            segment_desc, arena, io):
        active = _scalar(8)
        after_attention, *updates = mla(
            prefix, blocks, _choose(common["attn_res"], layer), _choose(common["attn_norm"], layer),
            active, mla_families, segment_desc, length, _scalar(mla_slot), arena,
        )
        mla_state = tuple(_put_state(family, update, mla_slot)
                          for family, update in zip(mla_state, updates, strict=True))
        ffn_input = mixture(after_attention, blocks, _choose(common["mlp_res"], layer),
                            _choose(common["ffn_norm"], layer), active)
        prefix, routes = moe(ffn_input, after_attention, moe_families, length, _scalar(moe_slot))
        logits = logits_op(prefix, blocks, io["output_res"], io["final_norm"], io["lm_head"],
                           segment_desc)
        logits = jax.shard_map(
            lambda part: jax.lax.all_gather(part[0], "k3_sp", axis=1, tiled=True),
            mesh=mesh, in_specs=(P("k3_sp"),), out_specs=P(), check_vma=False,
        )(logits)
        return logits, mla_state, routes

    return jax.jit(run)


def stage_plan(stage: int) -> dict:
    """A stage's layers, family slots and op sequence: ``groups`` are the
    global 4-layer group indices its group ops run (group 0 with the
    embedding and the dense FFN), ``final`` whether it ends with layer 92."""
    first, last = STAGE_LAYERS[stage]
    slots = stage_slots(first, last)
    groups = tuple(range(first // 4, min(last, 92) // 4))
    return dict(first=first, last=last, slots=slots, groups=groups, final=last == 93,
                kda_layers=slots["kda"][1] - slots["kda"][0],
                mla_layers=slots["mla"][1] - slots["mla"][0])
