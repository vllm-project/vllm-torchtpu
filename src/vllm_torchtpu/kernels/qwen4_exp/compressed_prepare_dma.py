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
"""Prepare and write compressed BF16 pairs with exact partial-pair ownership."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16
from vllm_torchtpu.kernels.qwen4_exp.tuned_block_sizes import COMPLETED_ROPE_DMA_ROWS


def write_prepared_pairs_dma(
    pool: jax.Array,
    values: jax.Array,
    pair: jax.Array,
    active: jax.Array,
    mask: jax.Array,
    *,
    local: jax.Array,
    trig_cache: jax.Array,
    positions: jax.Array,
    rotary_dim: int,
) -> jax.Array:
    """Deduplicate rotary-block DMA and rotate the token batch before pair selection.

    The indexed path selects raw bits, preserving NaNs and signed zero. Rotary
    arithmetic retains index RoPE's BF16 inner products and final BF16 round.
    Conflicting destination pairs keep the original ordered read/modify/write.
    """
    packing = 4 if pool.dtype == jnp.float8_e4m3fn else 2
    width = pool.shape[-1]
    view = pool.reshape(-1, packing, width)
    if (
        rotary_dim <= 0
        or rotary_dim % 2
        or rotary_dim > min(width, trig_cache.shape[-1])
    ):
        raise ValueError("Rotating pair writes require compatible rotary dimensions")
    half = rotary_dim // 2
    count = len(values)
    angle_rows = COMPLETED_ROPE_DMA_ROWS
    conflict = (
        (pair[:, None] == pair[None, :])
        & active[:, None]
        & active[None, :]
        & ~jnp.eye(count, dtype=jnp.bool_)
    )
    unique = (~jnp.any(conflict)).astype(jnp.int32).reshape(1)

    indexed = local is not None
    rotating = True
    trig_cache = pltpu.with_memory_space_constraint(trig_cache, pltpu.HBM)

    def kernel(indices, live, masks, disjoint, *refs):
        locations, *refs = refs if indexed else (None, *refs)
        (
            pos_ref,
            trig_ref,
            values_ref,
            old,
            updated,
            tiles,
            sem,
            angle_tiles,
            angle_sem,
        ) = refs

        has_write = live[0] != 0
        for token in range(1, count):
            has_write = has_write | (live[token] != 0)

        @pl.when(has_write)
        def completed_groups():
            owners = []
            positions = []
            for t in range(count):
                position = jnp.minimum(pos_ref[t], trig_cache.shape[0] - 1)
                positions.append(position)
                owner = jnp.int32(t)
                for earlier in range(t):
                    owner = jnp.minimum(
                        owner,
                        jnp.where(
                            position // angle_rows == positions[earlier] // angle_rows,
                            earlier,
                            t,
                        ),
                    )
                owners.append(owner)

            def transfer(t):
                return pltpu.make_async_copy(
                    trig_ref.at[
                        pl.ds(positions[t] // angle_rows * angle_rows, angle_rows)
                    ],
                    angle_tiles.at[t],
                    angle_sem.at[t],
                )

            for t in range(count):

                @pl.when(owners[t] == t)
                def start_angle(*, t=t):
                    transfer(t).start()

            def start_read(t):
                @pl.when((live[t] != 0) & ((masks[t, 0] == 0) | (masks[t, 1] == 0)))
                def read():
                    pltpu.make_async_copy(
                        updated.at[indices[t]], tiles.at[t], sem.at[t]
                    ).start()

            @pl.when(disjoint[0] != 0)
            def stage():
                for t in range(count):
                    start_read(t)

            prepared = None if indexed else values_ref[...]
            source = values_ref[...]
            if rotating:
                a, b = (
                    source[:, :half].astype(jnp.float32),
                    source[:, half:rotary_dim].astype(jnp.float32),
                )
                for t in range(count):

                    @pl.when(owners[t] == t)
                    def wait_angle(*, t=t):
                        transfer(t).wait()

                selected_angles = []
                for t in range(count):
                    bits = jax.lax.bitcast_convert_type(
                        angle_tiles[owners[t]], jnp.uint16
                    ).astype(jnp.int32)
                    bits = jnp.max(
                        jnp.where(
                            jnp.arange(angle_rows)[:, None]
                            == positions[t] % angle_rows,
                            bits,
                            0,
                        ),
                        axis=0,
                    )
                    exponent = bits & 0x7F80
                    bits = jnp.where(exponent == 0, bits & 0x8000, bits)
                    bits = jnp.where(
                        (exponent == 0x7F80) & ((bits & 0x7F) != 0), 0x7FC0, bits
                    )
                    selected_angles.append(
                        jax.lax.bitcast_convert_type(
                            bits.astype(jnp.uint16), jnp.bfloat16
                        )
                    )
                angles = jnp.stack(selected_angles).astype(jnp.float32)
                c, sn = angles[:, :half], angles[:, half:rotary_dim]
                parts = (bf16(a * c - bf16(b * sn)), bf16(a * sn + bf16(b * c)))
                if rotary_dim < width:
                    parts += (source[:, rotary_dim:],)
                source = jnp.concatenate(parts, axis=1)
            source_bits = (
                jax.lax.bitcast_convert_type(source, jnp.uint16).astype(jnp.int32)
                if indexed
                else None
            )

            for t in range(count):

                @pl.when(live[t] != 0)
                def store(*, t=t):
                    @pl.when(disjoint[0] == 0)
                    def ordered_read():
                        start_read(t)

                    if indexed:
                        assert locations is not None

                        def select_row(index):
                            bits = jnp.max(
                                jnp.where(
                                    jnp.arange(count)[:, None] == index, source_bits, 0
                                ),
                                axis=0,
                            )
                            return jax.lax.bitcast_convert_type(
                                bits.astype(jnp.uint16), jnp.bfloat16
                            )

                        selected = jnp.stack(
                            (select_row(locations[t, 0]), select_row(locations[t, 1]))
                        )
                    else:
                        assert prepared is not None
                        selected = prepared[t]
                    full = (masks[t, 0] != 0) & (masks[t, 1] != 0)
                    carrier = (
                        tiles.at[t].bitcast(jnp.bfloat16)
                        if pool.dtype != jnp.bfloat16
                        else tiles.at[t]
                    )

                    @pl.when(~full)
                    def merge():
                        pltpu.make_async_copy(
                            updated.at[indices[t]], tiles.at[t], sem.at[t]
                        ).wait()
                        keep = (
                            jnp.where(
                                jnp.arange(2)[:, None] == 0, masks[t, 0], masks[t, 1]
                            )
                            != 0
                        )
                        carrier[...] = jnp.where(keep, selected, carrier[...])

                    @pl.when(full)
                    def initialize():
                        carrier[...] = selected

                    pltpu.make_async_copy(
                        tiles.at[t], updated.at[indices[t]], sem.at[t]
                    ).start()

                    @pl.when(disjoint[0] == 0)
                    def ordered_wait():
                        pltpu.make_async_copy(
                            tiles.at[t], updated.at[indices[t]], sem.at[t]
                        ).wait()

            @pl.when(disjoint[0] != 0)
            def finish():
                for t in range(count):

                    @pl.when(live[t] != 0)
                    def wait(*, t=t):
                        pltpu.make_async_copy(
                            tiles.at[t], updated.at[indices[t]], sem.at[t]
                        ).wait()

    result = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            grid=(1,),
            in_specs=[
                *((pl.BlockSpec(memory_space=pltpu.HBM),) if rotating else ()),
                pl.BlockSpec(values.shape, lambda *_: (0,) * values.ndim),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((count, packing, width), pool.dtype),
                pltpu.SemaphoreType.DMA((count,)),
                pltpu.VMEM((count, angle_rows, trig_cache.shape[1]), trig_cache.dtype),
                pltpu.SemaphoreType.DMA((count,)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(view.shape, view.dtype),
        input_output_aliases={8: 0},
        name="qsa_compressed_angle_dma_write"
        if rotating
        else "qsa_compressed_indexed_write"
        if indexed
        else "qsa_compressed_batched_write",
    )(
        pair.astype(jnp.int32),
        active.astype(jnp.int32),
        mask.astype(jnp.int32),
        unique,
        *((local.astype(jnp.int32),) if indexed else ()),
        positions,
        trig_cache,
        values,
        view,
    )
    return result.reshape(pool.shape)
