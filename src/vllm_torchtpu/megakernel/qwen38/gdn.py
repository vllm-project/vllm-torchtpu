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
"""The GDN mixer of a prefill step as one kernel per core.

Stage ``s`` is flat rows ``[s * cores * chunk, (s + 1) * cores * chunk)``,
i.e. row block ``s`` of every core. The kernel

1. projects its rows (``exchange.project``): each destination core's
   ``[q | k | v | b | a]`` columns are pushed block by block, the gate is
   kept;
2. runs the PCP GDN kernel's per-sequence tiles (``gdn_pcp_fused``) over the
   received rows in flat order, each tile once the stages holding its rows
   have arrived, with the states streamed from vLLM's unified pool and each
   sequence's new states returned for the pool update after the kernel;
3. sends every finished stage's output home (``exchange.send_rows_home``)
   while later tiles compute;
4. applies the output norm and gate to each row block once it is home, and
   quantizes it for the output projection.

A tile's rows start anywhere, which a BF16 HBM row DMA cannot address, so a
tile loads the 16-row-aligned window around its rows (while the previous tile
computes) and shifts it in VMEM.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.gdn_pcp_fused import config, memory_ref
from vllm_torchtpu.kernels.experimental.gdn_pcp_fused import kernel as gdn_kernel
from vllm_torchtpu.kernels.experimental.gdn_pcp_fused.pcp_wrapper import (
    _scatter_state_updates,
)
from vllm_torchtpu.kernels.gdn.v3 import config as v3_config
from vllm_torchtpu.kernels.gdn.v3 import metadata
from vllm_torchtpu.megakernel.qwen38 import exchange
from vllm_torchtpu.megakernel.qwen38.reference import BF16, F32, FP8, FP8_MAX

MIXER_ID = 52  # the kernel's barrier semaphore
TILE = 64  # tokens per GDN tile
ROW_GROUP = 16  # rows of a BF16 (16, 128) tile: a row DMA's granularity
NORM_ROWS = 64  # rows per output-norm step: whole FP8 (32, 128) tiles
NORM_HEADS = 8  # heads per output-norm iteration: independent chains to overlap


def _config(total: int, *, n_kq: int, n_v: int, d_k: int, d_v: int, kernel_size: int):
    """The GDN kernel's configuration for a step of ``total`` flat rows,
    with the unified pool's state layout (FP32 recurrent and conv states)."""
    return config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        batch_size=total,
        kernel_size=kernel_size,
        tile_size=TILE,
        dim_size=2 * n_kq * d_k + n_v * d_v,
        num_kq_heads=n_kq,
        num_v_heads=n_v,
        kq_head_dim=d_k,
        v_head_dim=d_v,
        dtypes=config.Dtypes(
            act_in=F32, act_out=BF16, compute=F32, recurrent_state=F32, conv_state=F32
        ),
        is_shared=True,
    )


def tile_metadata(
    total, seq_lens, query_start_loc, state_indices, distribution, **dims
):
    """The tiles of a step of ``total`` flat rows (``gdn.v3``'s per-sequence
    metadata): the same for every GDN layer whose states ``state_indices``
    name. ``dims`` as :func:`mixer` takes them."""
    cfg = _config(total, **dims)
    tiles = metadata.compute_per_seq_metadata(
        cfg=v3_config.GDNConfig(
            mode=v3_config.GDNMode.PER_SEQ,
            batch_size=cfg.batch_size,
            kernel_size=cfg.kernel_size,
            tile_size=cfg.tile_size,
            dim_size=cfg.dim_size,
            num_kq_heads=cfg.num_kq_heads,
            num_v_heads=cfg.num_v_heads,
            kq_head_dim=cfg.kq_head_dim,
            v_head_dim=cfg.v_head_dim,
            dtypes=v3_config.Dtypes(
                act_in=F32,
                act_out=BF16,
                compute=F32,
                recurrent_state=F32,
                conv_state=F32,
            ),
        ),
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        read_offsets=jnp.zeros((state_indices.size,), jnp.int32),
        start_seq=distribution[0],
        end_seq=distribution[-1],
    )
    return memory_ref.MetadataRef.create(
        cfgs=cfg,
        num_tiles=tiles.num_tiles,
        p_id_to_s_idx=tiles.p_id_to_s_idx.data,
        p_id_to_r_base=tiles.p_id_to_r_base.data,
        p_id_to_r_size=tiles.p_id_to_r_size.data,
        p_id_is_first_tile=tiles.p_id_is_first_tile.data,
        p_id_is_last_tile=tiles.p_id_is_last_tile.data,
        s_idx_has_initial_state=tiles.s_idx_has_initial_state,
        s_idx_to_state_indices=tiles.s_idx_to_state_indices,
        s_idx_to_read_offset=tiles.s_idx_to_read_offset,
    )


def _state_updates_shape(pool, cfg, num_seqs: int):
    """Each sequence's new states as the first slices of a pool block (the
    granules hold the same bytes as in the pool), and how many granules of
    them are state."""
    granule_bytes = 4 * pltpu.get_tpu_info().num_lanes
    slice_bytes = math.prod(pool.shape[2:]) * pool.dtype.itemsize
    conv_bytes = cfg.prev_kernel_size * cfg.dim_size * jnp.dtype(F32).itemsize
    granules = cfg.conv_hbm_offset + pl.cdiv(conv_bytes, granule_bytes)
    slices = pl.cdiv(granules * granule_bytes, slice_bytes)
    if slice_bytes % granule_bytes or slices > pool.shape[1]:
        raise ValueError(f"GDN state of {granules} granules in pool {pool.shape}")
    shape = jax.ShapeDtypeStruct((num_seqs, slices, *pool.shape[2:]), pool.dtype)
    return shape, granules


def mixer(
    x: jax.Array,
    in_proj: jax.Array,
    scale: jax.Array,
    z_proj: jax.Array,
    z_scale: jax.Array,
    pool: jax.Array,
    conv_weight: jax.Array,
    a_log: jax.Array,
    dt_bias: jax.Array,
    norm: jax.Array,
    out_scale: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    seq_lens: jax.Array,
    distribution: jax.Array,
    *,
    axis: str,
    cores: int,
    chunk: int,
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    kernel_size: int,
    pool_block_tokens: int,
    eps: float,
    meta=None,
) -> tuple[jax.Array, jax.Array]:
    """One core's GDN mixer up to the output projection: ``x`` ``[rows, H]``
    FP8 (the core's normed rows, quantized), ``in_proj`` / ``scale`` /
    ``z_proj`` / ``z_scale`` as ``exchange.project`` takes them, this core's
    pool and its shard of the conv weight (``[conv_dim, 1, kernel]``),
    ``a_log`` and ``dt_bias`` (``n_kq`` key and ``n_v`` value heads), the
    output norm's weight ``norm`` ``[d_v]`` (``eps``) and the output
    projection's input scale ``out_scale`` ``[1]``, and the step's
    :func:`tile_metadata` (computed here if None). Returns every head's
    normed and gated output for the core's rows ``[rows, cores * n_v *
    d_v]``, quantized to FP8 by ``out_scale``, and the pool."""
    from vllm_torchtpu.layers.core.gdn_attention import _as_logical_block_pool

    rows, hidden = x.shape
    num_seqs = state_indices.shape[0]
    p = exchange.Projection(
        cores, chunk, rows, hidden, in_proj.shape[0] // cores, z_proj.shape[0]
    )
    total, stage_rows, width = cores * rows, cores * chunk, n_v * d_v
    conv_dim = 2 * n_kq * d_k + n_v * d_v
    dims = dict(n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v, kernel_size=kernel_size)
    cfg = _config(total, **dims)
    if meta is None:
        meta = tile_metadata(
            total, seq_lens, query_start_loc, state_indices, distribution, **dims
        )
    pool_shape = pool.shape
    # Dim 0 indexes state blocks (the kernel addresses them in granules).
    pool = _as_logical_block_pool(pool, pool_block_tokens)
    updates_shape, state_granules = _state_updates_shape(pool, cfg, num_seqs)
    # A sequence's new states are written back iff the kernel computes them:
    # it lies in [0, distribution[-1]), has tokens and a state block.
    seq_ids = jnp.arange(num_seqs, dtype=jnp.int32)
    write_back = (
        (seq_ids < distribution[-1])
        & (query_start_loc[1:] > query_start_loc[:-1])
        & (state_indices >= 0)
        & (state_indices < pool.shape[0])
    )
    state_dst = jnp.where(write_back, state_indices, -1).astype(jnp.int32)
    weights = memory_ref.WeightRefs(
        conv=memory_ref.ConvWeightsRef(weight=conv_weight.swapaxes(0, 2).astype(F32)),
        gdn=memory_ref.GDNWeightsRef(a_log=a_log, dt_bias=dt_bias),
    )
    lanes = cfg.aligned_num_v_heads

    def kernel(
        meta_ref,
        x_hbm,
        w_hbm,
        scale_ref,
        z_hbm,
        z_scale_ref,
        pool_ref,
        state_dst_ref,
        _zeros,
        weights_ref,
        norm_ref,
        out_scale_ref,
        normed_ref,
        back_ref,
        recv_ref,
        z_ref,
        staging,
        rows_staging,
        gdn_out,
        updates_ref,
        *scratch,
    ):
        proj_sems, scratch = scratch[:4], scratch[4:]
        rows_scratch, scratch = scratch[:5], scratch[5:]
        (
            recv_sems,
            rows_recv_sem,
            window,
            window_sem,
            compute_qkv,
            b_scr,
            a_scr,
            counters,
            carry_conv,
            carry_recurrent,
            pad_sem,
        ) = scratch
        me = lax.axis_index(axis)
        exchange.enter(me, cores)
        outputs = (recv_ref, z_ref, staging)
        rows_args = (me, cores, chunk)

        def recv_sem(block):
            return recv_sems.at[block]

        def project(*vmem):
            inputs = (x_hbm, w_hbm, scale_ref, z_hbm, z_scale_ref)
            exchange.project(p, me, inputs, outputs, vmem, proj_sems, recv_sem)

        # The projection's VMEM is free again for the tiles.
        with jax.named_scope("gdn_project"):
            pl.run_scoped(project, *p.vmem_shapes(x.dtype, in_proj.dtype))
        num_tiles = meta_ref.num_tiles[...]
        counters[0] = 0  # stages arrived
        counters[1] = 0  # stages sent home
        b_scr[...] = jnp.zeros_like(b_scr)
        a_scr[...] = jnp.zeros_like(a_scr)

        def arrive_through(stage):
            @pl.loop(counters[0], stage + 1)
            def _(block):
                exchange.wait_projected(p, me, outputs, proj_sems, recv_sem, block)

            counters[0] = jnp.maximum(counters[0], stage + 1)

        def send_home_before(stage):
            """Send home every stage below ``stage`` not yet sent."""

            @pl.loop(counters[1], stage)
            def _(block):
                exchange.send_rows_home(
                    *rows_args,
                    gdn_out,
                    back_ref,
                    rows_staging,
                    rows_scratch,
                    rows_recv_sem.at[block],
                    block,
                )

            counters[1] = jnp.maximum(counters[1], stage)

        def window_load(tile):
            """The copy of the 16-row-aligned window around ``tile``'s rows
            into window slot ``tile % 2``, and where the rows start in it."""
            base = meta_ref.p_id_to_r_base[tile, 0]
            start = pl.multiple_of(base // ROW_GROUP * ROW_GROUP, ROW_GROUP)
            prefix = base - start
            span = pl.multiple_of(
                pl.cdiv(prefix + meta_ref.p_id_to_r_size[tile, 0], ROW_GROUP)
                * ROW_GROUP,
                ROW_GROUP,
            )
            slot = tile % 2
            load = pltpu.make_async_copy(
                recv_ref.at[pl.ds(start, span)],
                window.at[slot, pl.ds(0, span)],
                window_sem.at[slot],
            )
            return load, prefix

        def prefetch(tile):
            last = meta_ref.p_id_to_r_base[tile, 0] + meta_ref.p_id_to_r_size[tile, 0]
            arrive_through((last - 1) // stage_rows)
            window_load(tile)[0].start()

        def split_tile(slot, shift: int):
            """The tile's rows from its window (they start ``shift`` rows in)
            into the GDN kernel's q|k|v, b and a operands."""
            rows32 = window[slot].astype(F32)[shift : shift + TILE]
            compute_qkv[0, :, 0, :] = rows32[:, :conv_dim]
            b_scr[0, :, 0, :n_v] = rows32[:, conv_dim : conv_dim + n_v]
            a_scr[0, :, 0, :n_v] = rows32[:, conv_dim + n_v : conv_dim + 2 * n_v]

        def body(
            conv_in,
            recurrent_in,
            out_slot,
            conv_out,
            recurrent_out,
            meta_s,
            w_s,
            carry_c,
            carry_r,
        ):
            t = pl.program_id(0)
            load, prefix = window_load(t)
            load.wait()
            for shift in range(ROW_GROUP):
                pl.when(prefix == shift)(functools.partial(split_tile, t % 2, shift))

            # The next tile's rows load while this one computes.
            @pl.when(t + 1 < num_tiles)
            def _():
                prefetch(t + 1)

            gdn_kernel.inner_kernel(
                compute_qkv,
                b_scr,
                a_scr,
                conv_in,
                recurrent_in,
                out_slot,
                conv_out,
                recurrent_out,
                meta_s,
                w_s,
                carry_c,
                carry_r,
                cfg=cfg,
                p_id=t,
            )

            # Outputs of tiles up to t - 2 are in HBM (two output buffers).
            @pl.when(t >= 2)
            def _():
                done = meta_s.p_id_to_r_base[t - 2, 0] + meta_s.p_id_to_r_size[t - 2, 0]
                send_home_before(done // stage_rows)

        allocs = memory_ref.create_allocs(
            metadata_ref=meta_ref,
            qkv_ref=compute_qkv,
            b_ref=b_scr,
            a_ref=a_scr,
            out_ref=gdn_out,
            conv_state_ref=pool_ref,
            recurrent_state_ref=pool_ref,
            cfg=cfg,
            state_output_ref=updates_ref,
        )
        conv_in, recurrent_in, out_alloc, conv_out, recurrent_out = allocs[3:]
        pipeline = pltpu.emit_pipeline(
            body=body,
            grid=(num_tiles,),
            in_specs=(conv_in.spec, recurrent_in.spec),
            out_specs=(out_alloc.spec, conv_out.spec, recurrent_out.spec),
        )

        @pl.with_scoped(
            allocations=(conv_in, recurrent_in, out_alloc, conv_out, recurrent_out)
        )
        def run_tiles(allocations):
            # The state outputs go to the updates (``state_output_ref``).
            pipeline(
                pool_ref,
                pool_ref,
                gdn_out,
                pool_ref,
                pool_ref,
                scratches=(meta_ref, weights_ref, carry_conv, carry_recurrent),
                allocations=allocations,
            )

        pool_granules = memory_ref.granule_view(pool_ref, pool_ref.shape[:1])
        update_granules = memory_ref.granule_view(updates_ref, updates_ref.shape[:1])
        pad_granules = update_granules.shape[1] - state_granules

        def pad_copy(seq):
            """A written-back sequence's granules past its states, copied
            from its pool block unchanged (disjoint from the tiles' writes)."""
            block = state_dst_ref[seq]
            size = jnp.where(block >= 0, 1, 0)
            return pltpu.make_async_copy(
                pool_granules.at[
                    pl.ds(jnp.maximum(block, 0), size),
                    pl.ds(state_granules, pad_granules),
                ],
                update_granules.at[
                    pl.ds(seq, size), pl.ds(state_granules, pad_granules)
                ],
                pad_sem,
            )

        def pad_copies(action):
            if pad_granules:

                @pl.loop(0, num_seqs)
                def _(seq):
                    getattr(pad_copy(seq), action)()

        with jax.named_scope("gdn_tiles"):
            pad_copies("start")
            pl.when(num_tiles > 0)(functools.partial(prefetch, 0))
            run_tiles()
            pad_copies("wait")

        def norm_block(block, back_vmem, z_vmem, out_vmem, load_sems, store_sems):
            """Row block ``block`` (home) normed, gated and quantized, in
            steps of ``NORM_ROWS`` rows: each step's rows load while the one
            before computes."""
            steps = chunk // NORM_ROWS

            def rows(step):
                return pl.ds(
                    pl.multiple_of(block * chunk + step * NORM_ROWS, NORM_ROWS),
                    NORM_ROWS,
                )

            def loads(step):
                return [
                    pltpu.make_async_copy(
                        src.at[rows(step)], dst.at[step % 2], load_sems.at[i, step % 2]
                    )
                    for i, (src, dst) in enumerate(
                        ((back_ref, back_vmem), (z_ref, z_vmem))
                    )
                ]

            def store(step):
                return pltpu.make_async_copy(
                    out_vmem.at[step % 2],
                    normed_ref.at[rows(step)],
                    store_sems.at[step % 2],
                )

            for copy in loads(0):
                copy.start()
            for step in range(steps):
                for copy in loads(step):
                    copy.wait()
                if step + 1 < steps:
                    for copy in loads(step + 1):
                        copy.start()
                if step >= 2:
                    store(step - 2).wait()
                norm_rows(
                    back_vmem.at[step % 2], z_vmem.at[step % 2], out_vmem.at[step % 2]
                )
                store(step).start()
            for step in range(max(steps - 2, 0), steps):
                store(step).wait()

        def norm_rows(back, gate, out):
            """``reference.rms_gated`` per head, then the output
            projection's quantization (its rounding points)."""

            @pl.loop(0, cores * n_v // NORM_HEADS)
            def _(step):
                for head in range(NORM_HEADS):
                    first = pl.multiple_of((step * NORM_HEADS + head) * d_v, d_v)
                    lanes = pl.ds(first, d_v)
                    value = back[:, lanes].astype(F32)
                    mean = jnp.mean(value * value, axis=-1, keepdims=True)
                    value *= lax.rsqrt(mean + eps)
                    weighted = (value.astype(BF16) * norm_ref[...]).astype(BF16)
                    silu = jax.nn.silu(gate[:, lanes].astype(F32))
                    gated = (weighted.astype(F32) * silu).astype(BF16)
                    quantized = gated.astype(F32) / out_scale_ref[0]
                    out[:, lanes] = jnp.clip(quantized, -FP8_MAX, FP8_MAX).astype(FP8)

        with jax.named_scope("gdn_finish"):
            arrive_through(p.blocks - 1)
            send_home_before(p.blocks)

            def finish(*norm_scratch):
                @pl.loop(0, p.blocks)
                def _(block):
                    exchange.wait_rows(
                        *rows_args,
                        back_ref,
                        rows_staging,
                        rows_scratch,
                        rows_recv_sem.at[block],
                        block,
                    )
                    norm_block(block, *norm_scratch)

            norm_width = cores * width
            pl.run_scoped(
                finish,
                pltpu.VMEM((2, NORM_ROWS, norm_width), BF16),
                pltpu.VMEM((2, NORM_ROWS, norm_width), BF16),
                pltpu.VMEM((2, NORM_ROWS, norm_width), FP8),
                pltpu.SemaphoreType.DMA((2, 2)),
                pltpu.SemaphoreType.DMA((2,)),
            )

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    received, gate, staging_shape = p.out_shapes()
    carry = cfg.get_scratch_shape_dict()
    n_meta = len(jax.tree.leaves(meta))
    normed, *_, updates = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((rows, cores * width), FP8),
            jax.ShapeDtypeStruct((rows, cores * width), BF16),
            received,
            gate,
            staging_shape,
            jax.ShapeDtypeStruct((total, width), BF16),
            jax.ShapeDtypeStruct((total, n_v, d_v), BF16),
            updates_shape,
        ),
        in_specs=(
            jax.tree.map(lambda _: smem, meta),
            hbm,
            hbm,
            vmem,
            hbm,
            vmem,
            hbm,
            smem,
            hbm,
            jax.tree.map(lambda _: vmem, weights),
            vmem,
            smem,
        ),
        out_specs=(hbm,) * 8,
        scratch_shapes=[
            *p.semaphore_shapes(),
            *exchange.rows_scratch_shapes(chunk, n_v, d_v),
            pltpu.SemaphoreType.DMA((p.blocks,)),
            pltpu.SemaphoreType.DMA((p.blocks,)),  # row pushes home, per block
            pltpu.VMEM((2, TILE + ROW_GROUP, p.width), BF16),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.VMEM((1, TILE, 1, conv_dim), F32),
            pltpu.VMEM((1, TILE, 1, lanes), F32),
            pltpu.VMEM((1, TILE, 1, lanes), F32),
            pltpu.SMEM((2,), jnp.int32),
            carry["carry_conv_scratch_ref"],
            carry["carry_recurrent_scratch_ref"],
            pltpu.SemaphoreType.DMA,
        ],
        # Flat input index: the GDN output starts as zeros, so rows no tile
        # covers (the step's padding) go home as zeros.
        input_output_aliases={n_meta + 7: 6},
        compiler_params=pltpu.CompilerParams(
            collective_id=MIXER_ID,
            # The projection's phase plus this call's other scratch.
            vmem_limit_bytes=max(
                int(0.8 * pltpu.get_tpu_info().vmem_capacity_bytes),
                p.vmem_bytes() + 2**24,
            ),
            disable_bounds_checks=True,
        ),
        name="qwen38_gdn_mixer",
    )(
        meta,
        x,
        in_proj,
        scale,
        z_proj,
        z_scale,
        pool,
        state_dst,
        jnp.zeros((total, n_v, d_v), BF16),
        weights,
        norm.reshape(1, d_v),
        jnp.reshape(out_scale, (1,)).astype(F32),
    )
    # A communicating kernel's in-place update of a donated operand costs a
    # copy of it, so the kernel only reads the pool and returns each
    # sequence's new states, written here in place.
    pool = _scatter_state_updates(
        pool, updates, state_dst, jnp.minimum(distribution[-1], num_seqs)
    )
    return normed, pool.reshape(pool_shape)
