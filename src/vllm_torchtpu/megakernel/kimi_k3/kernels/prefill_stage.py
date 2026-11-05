# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from vllm_torchtpu.megakernel.kimi_k3.dsa import STAGE_DSA_FAMILIES
from vllm_torchtpu.megakernel.kimi_k3.kernels import (
    dsa_cache,
    dsa_prefill,
    dsa_prefill_select,
    mla_sparse_prefill,
)
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


def _dsa_selection(dsa, w, x, query_latent, window_keys, index_pages, order, segment_desc, length, positions,
                   in_window, *, segments, seq_len, capacity, ranks, stripe, debug=False, own_rows=False):
    """Kimi-K3.1's selection for the stage MLA layer's flash kernel
    (``mla_flash_prefill_packed``'s ``select`` and ``dense``): this rank's
    rows (window ``positions``) score the ``[arena | window]`` candidates,
    the arena's dequantized from the gathered ``index_pages`` (``[slots, P +
    32, 128]``) in the latents' arena ``order``, each page's in its token
    order as ``expand_prefix_pages`` keeps the latents' (arena column ``c``:
    token ``c % P`` of slot ``order[1 + c // P]``), the window's from its
    ``window_keys`` (``[T, 128]``); each takes its top ``index_topk``
    (``dsa_prefill_select.select_words``, bounded by each sequence's real
    context; with ``KernelConfig.dsa_prefill_select`` ``xla``,
    ``dsa_prefill.select_mask`` over the whole arena) and the packed words
    are exchanged. Windows whose rows all hold
    at most ``index_topk`` candidates run dense (nothing scored). With
    ``own_rows`` the words stay this rank's rows' (``[R, W]``, no exchange).
    ``debug``
    also returns the rows' ``q`` and ``w``, every candidate key (``keys``,
    ``[C + T, 128]``) and their ``scores`` (``[R, C + T]``,
    ``dsa_prefill.candidate_scores``, whatever the branch)."""
    topk = dsa.index_topk
    desc = segment_desc[:5 * segments]
    cached = desc[4 * segments:]
    dense = jnp.all(cached + desc[segments:2 * segments] <= topk)
    q, weights = dsa_prefill.index_query(query_latent, w["dsa_wq"], x, w["dsa_wv"])

    def arena_keys():
        page = index_pages.shape[1] - dsa_cache.SCALE_ROWS
        held = index_pages[order[1:]]
        keys = dsa_cache.dequantize_index_keys(held[:, :page].reshape(capacity, 128),
                                               held[:, page:page + page // 128].reshape(capacity))
        return jnp.where((jnp.arange(capacity) < order[0] * page)[:, None], keys, 0)  # unwritten slots

    def select(keys, columns):
        valid = dsa_prefill.candidate_valid(desc, length, count=segments, tokens=seq_len, capacity=columns,
                                            rows=positions)
        mask = dsa_prefill.select_mask(dsa_prefill.candidate_scores(q, weights, keys), valid, topk)
        words = dsa_prefill.pack_bits(mask)
        return words if own_rows else in_window(words)

    def prefixed(_):
        return select(jnp.concatenate((arena_keys(), window_keys)), capacity)

    query_rows = positions.shape[0] if own_rows else seq_len

    def window_only(_):
        words = select(window_keys, 0)
        return jnp.concatenate((jnp.zeros((query_rows, capacity // 32), jnp.uint32), words), axis=1)

    def selected(_):
        words = dsa_prefill_select.select_words(
            q, weights, window_keys, index_pages, order, desc, length, positions, count=segments,
            capacity=capacity, topk=topk)
        if own_rows:
            return words
        return dsa_prefill_select.gather_words(words, desc, count=segments, capacity=capacity, tokens=seq_len,
                                               ranks=ranks, stripe=stripe)

    kernel = k._config().dsa_prefill_select == "kernel"
    transposed = kernel and not own_rows
    if not kernel:
        def skipped(_):
            return jnp.zeros((query_rows, (capacity + seq_len) // 32), jnp.uint32)

        branch = jnp.where(dense, 0, jnp.where(jnp.any(cached > 0), 2, 1))
        words = jax.lax.switch(branch, (skipped, window_only, prefixed), None)
    else:
        def skipped(_):
            shape = (dsa_prefill_select.word_columns(capacity, seq_len), query_rows)
            return jnp.zeros(shape[::-1] if own_rows else shape, jnp.uint32)

        words = jax.lax.cond(dense, skipped, selected, None)
    selection = dict(select=words, dense=dense.astype(jnp.int32).reshape(1), select_transposed=transposed)
    if debug:
        keys = jnp.concatenate((arena_keys(), window_keys))
        selection.update(q=q, w=weights, keys=keys, scores=dsa_prefill.candidate_scores(q, weights, keys))
    return selection


def _sparse_gather_to(table, index, end):
    """The token rows ``index[:end]`` of ``table`` on the SparseCore."""
    return k.ragged_gather_v2(table, index, jnp.int32(0), end)


def _sparse_words(select, desc, length, *, segments, seq_len, capacity, positions=None):
    """``mla_sparse_prefill.selection_words`` of ``_dsa_selection``'s
    ``select`` and whether they are transposed."""
    transposed = select["select_transposed"]
    return mla_sparse_prefill.selection_words(select["select"], select["dense"], desc, length, count=segments,
                                              capacity=capacity, tokens=seq_len, transposed=transposed,
                                              rows=positions), transposed


def _stage_mla_output(attended, gate, w, prefix, length, rows, stripe, n, latent, keys, *extra):
    """The stage MLA layer's outputs from this rank's rows' attention at every
    head (``[rows, 96 * 128]``): the gated output projection added to the
    residual rows, the window's latents, the index keys, then ``extra``."""
    projected = k._dot(attended * gate, w["output_projection_repacked"])
    rank = jax.lax.axis_index("k3_sp")
    mine = k.sp_token_index(rank, jnp.arange(rows), stripe, n) < length[0]
    output = prefix[0] + jnp.where(mine[:, None], projected, 0)
    return output.astype(jnp.bfloat16)[None], latent[None], *keys, *extra


def make_mla_layer_stage(mesh, *, seq_len, segments, prefix_capacity, flash_tile=256, eps=1e-5,
                         dsa=None, rope_probe=False, debug=False):
    """One MLA layer of a stage with row-local projections: each rank
    normalizes its rows and projects them to the query and KV latents and
    through the output gate of all 96 heads; an all-gather of the latents
    gives every rank the window, which it attends at its 12 heads over the
    window and the cached prefixes; an all-to-all hands each rank its rows
    of every head, which it gates and projects out. ``run(prefix, blocks,
    folded, norm, active, families, segment_desc, length, slot, arena)``
    returns the new residual rows and the rank-leading ``[n, T, 576]``
    compressed KV (the same on every rank).

    With ``dsa`` (Kimi-K3.1's ``DsaConfig``) each rank also keys its rows
    (``dsa_prefill.index_keys``; ``families`` hold ``dsa.STAGE_DSA_FAMILIES``
    too), gathered with the latents: two more outputs, the window's key bytes
    ``[n, T, 128]`` and exponents ``[n, T]`` uint8, zero from ``length`` on.
    ``run`` then also takes ``index_cache``: the stage's index-key cache (``[blocks,
    L, split * (P + 32), 128]`` uint8, read only) and the layer's slot in it
    (``[1]``); the attention keeps each row's ``dsa.index_topk`` best keys
    (``_dsa_selection``).

    ``debug`` (with ``dsa``; for checks) adds a last output, a dict of this
    rank's values, each with a leading rank dim: its rows' ``q``, ``w`` and
    ``scores`` with every candidate's ``keys`` (``_dsa_selection``), the
    gathered selection words (``select``, int32 view) and ``dense``, the flash
    kernel's output ``attended`` and its inputs ``query``, ``key``, ``value``,
    ``arena_keys`` and ``arena_values``."""
    if debug and dsa is None:
        raise ValueError("the stage MLA layer's debug outputs are Kimi-K3.1's")
    n = k.check_mesh(mesh, "stage MLA")
    k.check_window(seq_len, "stage MLA", n)
    if not segments:
        raise ValueError("the stage MLA layer runs packed windows")
    tables = k.mesh_ring_tables(mesh)
    rows = seq_len // n
    stripe = k.window_stripe(rows)
    mixture = k.make_mixture_norm_sp(mesh, seq_len=seq_len, eps=eps)
    fields = dict(zip(k.MLA_FAMILIES, k.MLA_FIELD_NAMES, strict=True))
    names = k.MLA_FAMILIES + (() if dsa is None else STAGE_DSA_FAMILIES)

    def in_window(values):
        # Every rank's rows, in window order (stripes of ``stripe`` rows round the ranks).
        gathered = jax.lax.all_gather(values, "k3_sp")
        gathered = gathered.reshape(n, rows // stripe, stripe, *values.shape[1:])
        return gathered.swapaxes(0, 1).reshape(seq_len, *values.shape[1:])

    def local(hidden, prefix, families, segment_desc, length, slot, pool, ids, desc, *index_cache):
        w = {fields.get(name, name): jax.lax.dynamic_index_in_dim(value[0], slot[0], keepdims=False)
             for name, value in families.items()}
        projections = k.ProjectionWeights(*(w[name] for name in k._PROJECTION_NAMES))
        x = hidden[0]
        query_latent, latent = k.mla_latents(x, projections, epsilon=eps)
        keys, window_keys = (), None
        if dsa is not None:
            key_bytes, exponents = (in_window(part) for part in dsa_prefill.index_keys(
                x, latent, w["dsa_wk"], w["dsa_knorm"], rope_probe=rope_probe, eps=dsa.norm_eps))
            window_keys = dsa_cache.dequantize_index_keys(key_bytes, exponents)
            filled = jnp.arange(seq_len) < length.reshape(())
            keys = (jnp.where(filled[:, None], key_bytes, 0)[None],
                    jnp.where(filled, exponents, 0)[None])
        packed = jnp.pad(jnp.concatenate((query_latent, latent), axis=1),
                         ((0, 0), (0, MLA_PACKED - 1536 - 576)))
        gate = k._dot(x, w["output_gate_projection"]).astype(jnp.bfloat16)
        gate = jax.nn.sigmoid(gate.astype(jnp.float32)).astype(jnp.bfloat16)
        window = in_window(packed)
        latent = window[:, 1536:1536 + 576]
        query, new_key, new_value = k.mla_heads(
            window[:, :1536], latent, projections, query_scale=k.SCALE_LOG2)
        real = (jnp.arange(seq_len) < length.reshape(()))[None, :, None]
        new_key = jnp.where(real, new_key.transpose(1, 0, 2), 0)
        new_value = jnp.where(real, new_value.transpose(1, 0, 2), 0)
        heads = query.shape[1]
        mode = "off" if dsa is None else k._config().mla_sparse_prefill
        # The sparse attention gathers token rows: the arena as rows, the window's rows after it.
        flat = mla_sparse_prefill.table_extra(seq_len) if mode != "off" and not debug else None
        pages, order = k.gathered_prefix_pages(pool, ids[0], desc[0], tables, "k3_sp",
                                               count=segments, capacity=prefix_capacity, flat_extra=flat)
        up = projections.key_value_up_projection
        if mode == "rows":  # every head is loaded; the checks and the head-layout tail take this rank's
            heads //= n
            first = jax.lax.axis_index("k3_sp") * heads
            query = jax.lax.dynamic_slice_in_dim(query, first, heads, axis=1)
            new_key = jax.lax.dynamic_slice_in_dim(new_key, first, heads, axis=0)
            new_value = jax.lax.dynamic_slice_in_dim(new_value, first, heads, axis=0)
            up = jax.lax.dynamic_slice_in_dim(up, first * 256, heads * 256, axis=1)
        arena = None if mode != "off" and not debug else k.expand_prefix_pages(pages, order, up)
        select = {}
        if window_keys is not None:
            cache, index_slot = index_cache
            page = pool.shape[-3] * pool.shape[-2]
            rows_per_page = page + dsa_cache.SCALE_ROWS
            cache = jax.lax.dynamic_index_in_dim(cache, index_slot[0], axis=1, keepdims=False)
            index_pages, _ = k.gathered_prefix_pages(
                cache.reshape(-1, rows_per_page // 4, 4, 128), ids[0], desc[0], tables, "k3_sp",
                count=segments, capacity=prefix_capacity, page_tokens=page)
            positions = k.sp_token_index(jax.lax.axis_index("k3_sp"), jnp.arange(rows), stripe, n)
            select = _dsa_selection(
                dsa, w, x, query_latent, window_keys,
                index_pages.reshape(-1, rows_per_page, 128), order, segment_desc, length, positions,
                in_window, segments=segments, seq_len=seq_len, capacity=prefix_capacity, ranks=n, stripe=stripe,
                debug=debug, own_rows=mode == "rows" and not debug)
        checked = {name: select.pop(name) for name in ("q", "w", "keys", "scores") if name in select}
        desc5 = segment_desc[:5 * segments]
        sparse = dict(segments=segments, seq_len=seq_len, capacity=prefix_capacity)
        compact = "sparse_core" if jax.default_backend() == "tpu" else "xla"
        token = functools.partial(
            mla_sparse_prefill.attend_rows, window_latent=jnp.where(real[0], latent, 0),
            key_value_up=projections.key_value_up_projection, pages=pages, order=order,
            capacity=prefix_capacity, gather=_sparse_gather_to, runs=segments * n, compact=compact,
            fp8_gather=k._config().mla_sparse_fp8_gather, groups=k._config().mla_sparse_groups,
            split=k._config().mla_sparse_query_split)
        if mode == "rows":
            # This rank's rows at every head, straight to its output projection.
            positions = k.sp_token_index(jax.lax.axis_index("k3_sp"), jnp.arange(rows), stripe, n)
            if debug:  # the selection was exchanged for the checks: this rank's rows of it
                words, transposed = _sparse_words(select, desc5, length, **sparse)
                words = words[:, positions] if transposed else words[positions]
            else:
                words, transposed = _sparse_words(select, desc5, length, positions=positions, **sparse)

            def sparse_rows(_):
                own = (k._dot(query_latent, projections.query_up_projection) * k.SCALE_LOG2).astype(jnp.bfloat16)
                block = min(k._config().mla_sparse_block, rows)
                if block > 1:
                    return mla_sparse_prefill.attend_blocks(
                        own.reshape(rows, -1, 256), jnp.where(real[0], latent, 0),
                        projections.key_value_up_projection, pages, order, words, capacity=prefix_capacity,
                        gather=_sparse_gather_to, runs=segments * n, transposed=transposed, block=block,
                        compact=compact, fp8_gather=k._config().mla_sparse_fp8_gather).reshape(rows, -1)
                return token(own.reshape(rows, -1, 256), select=words, transposed=transposed).reshape(rows, -1)

            config = k._config()
            dense_below = config.mla_sparse_dense_below if seq_len <= 2048 else config.mla_sparse_dense_below_wide
            if not debug and dense_below:
                # Short contexts: the masked flash at this rank's heads over the whole context costs less than
                # the token gathers (both exact over the same selection); the own-row words exchanged as the
                # default path exchanges them.
                def dense_rows(_):
                    if config.dsa_prefill_select == "kernel":
                        exchanged = dsa_prefill_select.gather_words(
                            select["select"], desc5, count=segments, capacity=prefix_capacity, tokens=seq_len,
                            ranks=n, stripe=stripe)
                    else:
                        exchanged = in_window(select["select"])
                    # The pages gathered again as pages: viewing the flat token rows as pages copies the whole
                    # table, and its layout then slows the sparse path's gathers.
                    held, _ = k.gathered_prefix_pages(pool, ids[0], desc[0], tables, "k3_sp", count=segments,
                                                      capacity=prefix_capacity)
                    attended = k.mla_flash_prefill_packed(
                        query.transpose(1, 0, 2), new_key, new_value, length, desc5, count=segments,
                        tile=flash_tile, arena=k.expand_prefix_pages(held, order, up), select=exchanged,
                        dense=select["dense"], select_transposed=config.dsa_prefill_select == "kernel")
                    attended = attended.transpose(1, 0, 2).reshape(rows // stripe, n, stripe, heads * 128)
                    attended = attended.transpose(1, 0, 2, 3).reshape(n, rows, heads * 128)
                    attended = jax.lax.all_to_all(attended, "k3_sp", 0, 0)
                    return attended.transpose(1, 0, 2).reshape(rows, n * heads * 128)

                context = jnp.max(desc5[4 * segments:] + desc5[segments:2 * segments])
                mine = jax.lax.cond(context <= dense_below, dense_rows, sparse_rows, None)
            else:
                mine = sparse_rows(None)
            if not debug:
                return _stage_mla_output(mine, gate, w, prefix, length, rows, stripe, n, latent, keys)
            # The checks read every window row at this rank's heads: the all-to-all below, reversed.
            mine = jax.lax.all_to_all(mine.reshape(rows, n, heads * 128).transpose(1, 0, 2), "k3_sp", 0, 0)
            mine = mine.reshape(n, rows // stripe, stripe, heads * 128).transpose(1, 0, 2, 3)
            attended = mine.reshape(seq_len, heads, 128).transpose(1, 0, 2)
        elif mode == "heads":
            words, transposed = _sparse_words(select, desc5, length, **sparse)
            attended = token(query, select=words, transposed=transposed).transpose(1, 0, 2)
        else:
            attended = k.mla_flash_prefill_packed(
                query.transpose(1, 0, 2), new_key, new_value, length, segment_desc[:5 * segments],
                count=segments, tile=flash_tile, arena=arena, **select)
        if debug:
            assert arena is not None  # the checks' dense reference
            words = select["select"]
            if select["select_transposed"]:
                # The kernel leaves the words above the window's cached total unwritten.
                desc = segment_desc[:5 * segments]
                total = desc[4 * segments - 1] + desc[5 * segments - 1]
                column = jnp.arange(words.shape[0])[:, None]
                used = (column < (total + 4095) // 4096 * 128) | (column >= prefix_capacity // 32)
                words = jnp.where(used, words, 0).T
            # The window's keys and values from its latents' FP8 rounding (the pool's), as the sparse
            # attention reads them.
            rounded = jnp.where(real[0], latent.astype(jnp.float8_e4m3fn).astype(jnp.bfloat16), 0)
            expanded = k._dot(rounded[:, :512], up).astype(jnp.bfloat16).reshape(seq_len, heads, 256)
            key_fp8 = jnp.concatenate((expanded[:, :, :128],
                                       jnp.broadcast_to(rounded[:, None, 512:576], (seq_len, heads, 64)),
                                       jnp.zeros((seq_len, heads, 64), jnp.bfloat16)), axis=2)
            checked.update(select=jax.lax.bitcast_convert_type(words, jnp.int32),
                           dense=select["dense"], attended=attended, query=query.transpose(1, 0, 2),
                           key=new_key, value=new_value, arena_keys=arena[0], arena_values=arena[1],
                           key_fp8=key_fp8.transpose(1, 0, 2), value_fp8=expanded[:, :, 128:].transpose(1, 0, 2))
            # The latents themselves, for references without rounded keys: the arena's in arena order.
            held = pages[order[1:]].reshape(-1, pages.shape[-1])[:, :576]
            unwritten = jnp.arange(held.shape[0]) >= order[0] * pages.shape[1] * pages.shape[2]
            checked.update(arena_latent=jnp.where(unwritten[:, None], 0, held.astype(jnp.bfloat16)),
                           window_latent=jnp.where(real[0], latent, 0), up=up)
            # The default path (the flash kernel masked by the same selection) beside a sparse mode's.
            checked["attended_masked"] = attended if mode == "off" else k.mla_flash_prefill_packed(
                query.transpose(1, 0, 2), new_key, new_value, length, segment_desc[:5 * segments],
                count=segments, tile=flash_tile, arena=arena, **select)
            checked = {name: value[None] for name, value in checked.items()}
        # Each rank's rows of this rank's heads to that rank: [source rank, rows, heads * 128].
        attended = attended.transpose(1, 0, 2).reshape(rows // stripe, n, stripe, heads * 128)
        attended = attended.transpose(1, 0, 2, 3).reshape(n, rows, heads * 128)
        attended = jax.lax.all_to_all(attended, "k3_sp", 0, 0)
        attended = attended.transpose(1, 0, 2).reshape(rows, n * heads * 128)
        return _stage_mla_output(attended, gate, w, prefix, length, rows, stripe, n, latent, keys,
                                 *((checked,) if debug else ()))

    sharded, replicated = P("k3_sp"), P()
    mapped = jax.shard_map(
        local, mesh=mesh,
        in_specs=(sharded, sharded, sharded, replicated, replicated, replicated,
                  sharded, sharded, sharded) + (() if dsa is None else (sharded, replicated)),
        out_specs=(sharded,) * (2 if dsa is None else 4 + debug), check_vma=False,
    )

    def run(prefix, blocks, folded, norm, active, families, segment_desc, length, slot, arena, index_cache=()):
        (pool,) = arena.pools
        hidden = mixture(prefix, blocks, folded, norm, active)
        families = {name: families[name] for name in names}
        if (dsa is None) != (not index_cache):
            raise ValueError("the stage MLA layer takes an index-key cache exactly with dsa")
        return mapped(hidden, prefix, families, segment_desc, length, slot, pool, arena.ids,
                      arena.local, *index_cache)

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
                      mla_flash_tile=256, moe_path="fused", tap_offset=None, dsa=None,
                      rope_probe=False):
    """One runtime-indexed group at layers 4g..4g+3 of a stage whose weight
    families start at ``slots`` (``rank_loader.stage_slots``): three KDA
    layers and one MLA layer, each followed by its MoE (layer 0, with
    ``first``: the dense FFN). ``run(prefix, blocks, common, kda_families,
    mla_families, moe_families, kda_state, mla_state, length, group,
    segment_desc, arena)`` as ``make_regular_group4_sp_indexed``, the
    states stage-local (``kda_state``: ``(rec, hist, rec_at, hist_at)``,
    ``[n, stage KDA layers, segments, 4, ...]``), then the busiest chip's
    route count of its MoE layers (``[n]``); with ``tap_offset``, then the
    residual stream entering layer ``4g + tap_offset`` (a DSpark aux layer;
    offset 0: the group's input ``prefix``).
    With ``dsa`` (``make_mla_layer_stage``), ``mla_families`` also hold the
    indexer's, ``mla_state`` is ``(latents, key bytes, exponents)`` and
    ``run`` takes ``index_cache``, the index-key cache and the MLA layer's slot."""
    n = k.check_mesh(mesh, "stage group")
    if tap_offset is not None and not 0 <= tap_offset < 4:
        raise ValueError("tap_offset must name one of the group's layers")
    k.check_window(seq_len, "stage group", n)
    sharding = NamedSharding(mesh, P("k3_sp"))
    mixture = k.make_mixture_norm_sp(mesh, seq_len=seq_len)
    kda = make_kda_layer_stage(mesh, seq_len=seq_len, segments=segments)
    mla = make_mla_layer_stage(mesh, seq_len=seq_len, flash_tile=mla_flash_tile,
                               segments=segments, prefix_capacity=prefix_capacity, dsa=dsa,
                               rope_probe=rope_probe)
    moe = _stage_moe(mesh, seq_len, moe_path)
    dense = k.make_dense_ffn_sp(mesh, seq_len=seq_len) if first else None
    mla_states = 1 if dsa is None else 3
    layer_lo, kda_lo = slots["layer"][0], slots["kda"][0]
    mla_lo, moe_lo = slots["mla"][0], slots["moe"][0]

    def run(prefix, blocks, common, kda_families, mla_families, moe_families,
            kda_state, mla_state, length, group, segment_desc, arena, index_cache=()):
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
                    segment_desc, length, _scalar(slot), arena, index_cache,
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
        sharding, sharding, (sharding,) * 4, (sharding,) * mla_states, sharding)
        + ((sharding,) if tap_offset is not None else ()))


def make_stage_final(mesh, *, seq_len, segments, prefix_capacity, slots, mla_flash_tile=256,
                     moe_path="fused", dsa=None, rope_probe=False):
    """The last stage's layer 92 (MLA and its MoE) and each sequence's
    first-token logits: ``run(prefix, blocks, common, mla_families,
    moe_families, mla_state, length, segment_desc, arena, io, index_cache)``
    returns ``(logits [segments, V] replicated, mla_state, routes)``; ``dsa``
    and ``index_cache`` as :func:`make_stage_group4`."""
    k.check_mesh(mesh, "stage final")
    mixture = k.make_mixture_norm_sp(mesh, seq_len=seq_len)
    mla = make_mla_layer_stage(mesh, seq_len=seq_len, flash_tile=mla_flash_tile,
                               segments=segments, prefix_capacity=prefix_capacity, dsa=dsa,
                               rope_probe=rope_probe)
    moe = _stage_moe(mesh, seq_len, moe_path)
    logits_op = k.make_prefill_logits_sp(mesh, seq_len=seq_len, layers=93, segments=segments)
    layer = 92 - slots["layer"][0]
    mla_slot, moe_slot = 23 - slots["mla"][0], 91 - slots["moe"][0]

    def run(prefix, blocks, common, mla_families, moe_families, mla_state, length,
            segment_desc, arena, io, index_cache=()):
        active = _scalar(8)
        after_attention, *updates = mla(
            prefix, blocks, _choose(common["attn_res"], layer), _choose(common["attn_norm"], layer),
            active, mla_families, segment_desc, length, _scalar(mla_slot), arena, index_cache,
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
