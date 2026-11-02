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


def write_prepared_pairs(
    pool, values, pair, active, mask, *, local=None, trig_rows=None
):
    """Accept prepared pairs or indexed keys with optional aligned trig rows.

    The indexed path selects raw bits, preserving NaNs and signed zero. Rotary
    arithmetic retains index RoPE's BF16 inner products and final BF16 round.
    Conflicting destination pairs keep the original ordered read/modify/write.
    """
    packing = 4 if pool.dtype == jnp.float8_e4m3fn else 2
    width = pool.shape[-1]
    view = pool.reshape(-1, packing, width)
    count = len(values)
    conflict = (
        (pair[:, None] == pair[None, :])
        & active[:, None]
        & active[None, :]
        & ~jnp.eye(count, dtype=jnp.bool_)
    )
    unique = (~jnp.any(conflict)).astype(jnp.int32).reshape(1)

    indexed = local is not None
    rotating = trig_rows is not None

    def kernel(indices, live, masks, disjoint, *refs):
        locations, *refs = refs if indexed else (None, *refs)
        trig_ref, *refs = refs if rotating else (None, *refs)
        values_ref, old, updated, tiles, sem = refs

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
            assert trig_ref is not None
            a, b = (
                source[:, :32].astype(jnp.float32),
                source[:, 32:64].astype(jnp.float32),
            )
            angles = trig_ref[...].astype(jnp.float32)
            c, sn = angles[:, :32], angles[:, 32:64]
            source = jnp.concatenate(
                (
                    bf16(a * c - bf16(b * sn)),
                    bf16(a * sn + bf16(b * c)),
                    source[:, 64:],
                ),
                axis=1,
            )
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
                        jnp.where(jnp.arange(2)[:, None] == 0, masks[t, 0], masks[t, 1])
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
            num_scalar_prefetch=5 if indexed else 4,
            grid=(1,),
            in_specs=[
                *(
                    (pl.BlockSpec(trig_rows.shape, lambda *_: (0, 0)),)
                    if rotating
                    else ()
                ),
                pl.BlockSpec(values.shape, lambda *_: (0,) * values.ndim),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((count, packing, width), pool.dtype),
                pltpu.SemaphoreType.DMA((count,)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(view.shape, view.dtype),
        input_output_aliases={5 + int(indexed) + int(rotating): 0},
        name="qsa_compressed_rope_write"
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
        *((trig_rows,) if rotating else ()),
        values,
        view,
    )
    return result.reshape(pool.shape)
