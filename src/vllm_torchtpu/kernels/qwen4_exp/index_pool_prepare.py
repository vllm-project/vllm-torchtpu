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
"""Group-sized pooling and direct BF16 pair writes without token-pair gathers."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16
from vllm_torchtpu.kernels.qwen4_exp.tuned_block_sizes import _COMPILER_SMEM_RESERVE


def pool_index_groups(
    raw: jax.Array,
    ring: jax.Array,
    starts: jax.Array,
    context: jax.Array,
    slots: jax.Array,
    req: jax.Array,
    ratio: int,
) -> jax.Array:
    """Compute each compression group once, retaining the full pooled contract.

    Two spare groups per request cover both context and chunk-tail phases.
    They also leave enough capacity for the final request's padding rows.
    """
    count, width = raw.shape
    tile = 128
    local = jnp.arange(count, dtype=jnp.int32)
    logical = context[req] + local - starts[req]
    first = starts[req] + logical // ratio * ratio - context[req]
    aligned = jnp.all((first == local // ratio * ratio) & (first >= starts[req]))

    def kernel(source, output):
        groups = source[...].reshape(tile // ratio, ratio, width)
        output[...] = jnp.repeat(
            bf16(jnp.mean(groups.astype(jnp.float32), axis=1)), ratio, axis=0
        )

    def contiguous(_):
        return pl.pallas_call(
            kernel,
            grid=(count // tile,),
            in_specs=[pl.BlockSpec((tile, width), lambda i: (i, 0))],
            out_specs=pl.BlockSpec((tile, width), lambda i: (i, 0)),
            out_shape=jax.ShapeDtypeStruct(raw.shape, raw.dtype),
            name="qsa_index_group_mean_tc",
        )(raw)

    def ragged(_):
        base = starts[:-1] // ratio + 2 * jnp.arange(len(context))
        size = (count + ratio - 1) // ratio + 2 * len(context)
        candidates = jnp.arange(size, dtype=jnp.int32)
        owners = jnp.sum(candidates[:, None] >= base[None, 1:], axis=1)
        groups = context[owners] // ratio + candidates - base[owners]
        members = groups[:, None] * ratio + jnp.arange(ratio)[None]
        current_indices = starts[owners, None] + members - context[owners, None]
        current = raw[jnp.clip(current_indices, 0, count - 1)]
        previous = ring[slots[owners, None], members % ratio]
        previous = jnp.where((context[owners] > 0)[:, None, None], previous, 0)
        joined = jnp.where(
            (current_indices >= starts[owners, None])[..., None], current, previous
        )
        pooled = bf16(jnp.mean(joined.astype(jnp.float32), axis=1))
        selected = base[req] + logical // ratio - context[req] // ratio
        return pooled[selected]

    return jax.lax.cond(aligned, contiguous, ragged, None)


def _canonical_bf16_words(words: jax.Array) -> jax.Array:
    """Match the measured TPU7x gather's canonical NaNs and signed zero flush."""
    exponent = words & 0x7F80
    words = jnp.where(exponent == 0, words & 0x8000, words)
    return jnp.where((exponent == 0x7F80) & ((words & 0x7F) != 0), 0x7FC0, words)


def write_indexed_pairs(
    pool: jax.Array,
    keys: jax.Array,
    pair: jax.Array,
    active: jax.Array,
    mask: jax.Array,
    local: jax.Array,
    ratio: int,
) -> jax.Array:
    """Gather only active pairs inside TC; preserve ordered partial-pair RMW."""
    count, width = keys.shape
    tile = 128
    window = 2 * tile
    hardware = pltpu.get_tpu_info()
    sublanes = hardware.num_sublanes
    lanes_per_vector = hardware.num_lanes
    prefetched = (
        6 * count + 2 * pl.cdiv(count // tile, lanes_per_vector) * lanes_per_vector
    )
    if 4 * prefetched + _COMPILER_SMEM_RESERVE > hardware.smem_capacity_bytes:
        raise ValueError("QSA indexed pair metadata exceeds the TPU SMEM budget")
    halo = (ratio + sublanes - 1) // sublanes * sublanes
    packing = 4 if pool.dtype == jnp.float8_e4m3fn else 2
    pairs = tile // (2 * ratio)
    view = pool.reshape(-1, packing, width)
    lanes = tuple(range(2 * ratio - 1, tile, 2 * ratio))
    expected = jnp.zeros((tile,), jnp.bool_).at[jnp.asarray(lanes)].set(True)
    indices = pair.reshape(-1, tile)
    first_pair = indices[:, lanes[0]]
    bulk = (
        jnp.all(active.reshape(-1, tile) == expected[None], axis=1)
        & jnp.all(mask.reshape(-1, tile, 2)[:, lanes], axis=(1, 2))
        & jnp.all(
            indices[:, lanes] == first_pair[:, None] + jnp.arange(pairs)[None], axis=1
        )
    )
    mode = jnp.where(
        bulk, 1, jnp.where(jnp.any(active.reshape(-1, tile), axis=1), 2, 0)
    )

    def kernel(
        first_ref,
        mode_ref,
        pairs_ref,
        live_ref,
        masks_ref,
        local_ref,
        source,
        source_hbm,
        _old,
        updated,
        staged,
        packed_pair,
        semaphores,
    ):
        program = pl.program_id(0)
        first = program * tile

        @pl.when(mode_ref[program] == 1)
        def store_bulk():
            def copy(packed, semaphore):
                words = source.bitcast(jnp.int32)[...]
                rows = words.reshape(tile // ratio, ratio // 2, width)[
                    :, ratio // 2 - 1
                ]
                rows = _canonical_bf16_words((rows >> 16) & 0xFFFF).reshape(
                    pairs, 2, width
                )
                packed.bitcast(jnp.int32)[...] = (rows[:, 0] | (rows[:, 1] << 16))[
                    :, None, :
                ]
                pltpu.async_copy(
                    packed, updated.at[pl.ds(first_ref[program], pairs)], semaphore
                ).wait()

            pl.run_scoped(
                copy,
                pltpu.VMEM((pairs, packing, width), pool.dtype),
                pltpu.SemaphoreType.DMA,
            )

        @pl.when(mode_ref[program] == 2)
        def store_edges():
            start = pl.multiple_of(jnp.clip(first - halo, 0, count - window), sublanes)
            pltpu.async_copy(
                source_hbm.at[pl.ds(start, window)], staged, semaphores.at[0]
            ).wait()
            words = staged.bitcast(jnp.int32)[...]

            def select(index):
                index = jnp.clip(index, 0, count - 1) - start
                selected = jnp.max(
                    jnp.where(
                        jnp.arange(window // 2)[:, None] == index // 2,
                        words,
                        jnp.int32(jnp.iinfo(jnp.int32).min),
                    ),
                    axis=0,
                    keepdims=True,
                )
                return _canonical_bf16_words((selected >> ((index % 2) * 16)) & 0xFFFF)

            def store(lane, _):
                row = first + lane

                @pl.when(live_ref[row] != 0)
                def write():
                    left, right = (
                        select(local_ref[2 * row]),
                        select(local_ref[2 * row + 1]),
                    )
                    value = left | (right << 16)
                    m0, m1 = masks_ref[2 * row], masks_ref[2 * row + 1]
                    full = (m0 != 0) & (m1 != 0)
                    target = updated.at[pairs_ref[row]]
                    carrier = packed_pair.bitcast(jnp.int32)

                    @pl.when(~full)
                    def read():
                        pltpu.async_copy(target, packed_pair, semaphores.at[1]).wait()

                    keep = jnp.where(m0 != 0, 0xFFFF, 0) | jnp.where(
                        m1 != 0, jnp.int32(-65536), 0
                    )

                    @pl.when(~full)
                    def merge():
                        carrier[...] = (carrier[...] & ~keep) | (value & keep)

                    @pl.when(full)
                    def initialize():
                        carrier[...] = value

                    pltpu.async_copy(packed_pair, target, semaphores.at[1]).wait()

            jax.lax.fori_loop(0, tile, store, None)

    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            grid=(count // tile,),
            in_specs=[
                pl.BlockSpec((tile, width), lambda i, *_: (i, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((window, width), keys.dtype),
                pltpu.VMEM((packing, width), pool.dtype),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(view.shape, view.dtype),
        input_output_aliases={8: 0},
        name="qsa_compressed_indexed_write_tc",
    )(
        first_pair.astype(jnp.int32),
        mode.astype(jnp.int32),
        pair.astype(jnp.int32),
        active.astype(jnp.int32),
        mask.astype(jnp.int32).reshape(-1),
        local.astype(jnp.int32).reshape(-1),
        keys,
        keys,
        view,
    ).reshape(pool.shape)
