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
"""Small-window ring updates without materializing old/new state in HBM."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import typed_ldst
from vllm_torchtpu.kernels.qwen4_exp.cache_regions import region_rows
from vllm_torchtpu.kernels.qwen4_exp.qsa import gemma_norm, index_rope
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16


def update_decode_state(
    pool: jax.Array,
    raw: jax.Array,
    positions: jax.Array,
    ids: jax.Array,
    starts: jax.Array,
    context: jax.Array,
    norm: jax.Array,
    trig: jax.Array,
    *,
    ratio: int,
    norm_eps: float,
    rotary_dim: int,
    rotate: bool = True,
    rope_axes: tuple[int, ...] | None = None,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Update small request batches; return normalized/rotated completed keys.

    Keep each request's state in VMEM across its query window. Only completed
    keys are observable by the compressed-cache writer. ID0 remains immutable.
    The caller budgets the resident ring, tags and raw/output planes.
    """
    count, width = raw.shape
    ring_rows = (ratio + 7) // 8 * 8
    axial = positions.ndim == 2
    axes = positions.shape[0] if axial else 1
    tag_fields = axes + 1
    ring_tokens, _ = region_rows(pool, ring_rows * width, jnp.bfloat16)
    tag_tokens, tag_rows = region_rows(pool, ring_rows * tag_fields, jnp.int32)
    tokens = ring_tokens + tag_tokens
    # At most one nonempty request per query token. Its first token is a unique
    # DMA staging slot, even when active metadata rows are not a dense prefix.
    # Resolve this in scalar memory: a JAX nonzero introduces an SC scatter and
    # a synchronization dependency before this small TC state update.
    batch = min(count, len(ids))

    def kernel(
        ids_ref,
        starts_ref,
        context_ref,
        positions_ref,
        raw_ref,
        norm_ref,
        source,
        target,
        keys_ref,
        tags_ref,
        staged,
        semaphores,
    ):
        keys_ref[...] = jnp.zeros(keys_ref.shape, keys_ref.dtype)
        tags_ref[...] = jnp.zeros(tags_ref.shape, tags_ref.dtype)
        for row in range(len(ids)):
            lane = starts_ref[row] if count < len(ids) else row

            @pl.when(starts_ref[row + 1] > starts_ref[row])
            def start(*, row=row, lane=lane):
                pltpu.make_async_copy(
                    source.at[jnp.maximum(ids_ref[row], 0), pl.ds(0, tokens)],
                    staged.at[lane],
                    semaphores.at[lane],
                ).start()

        for row in range(len(ids)):
            lane = starts_ref[row] if count < len(ids) else row

            @pl.when(starts_ref[row + 1] > starts_ref[row])
            def update(*, row=row, lane=lane):
                pltpu.make_async_copy(
                    source.at[jnp.maximum(ids_ref[row], 0), pl.ds(0, tokens)],
                    staged.at[lane],
                    semaphores.at[lane],
                ).wait()
                ring_ref = staged.at[lane, pl.ds(0, ring_tokens)]
                tag_ref = staged.at[lane, pl.ds(ring_tokens, tag_tokens)]
                ring = typed_ldst.load_typed(ring_ref, view_dtype=jnp.bfloat16)
                tags = typed_ldst.load_typed(tag_ref, view_dtype=jnp.int32)
                old_context = context_ref[row]
                ring = jnp.where(old_context > 0, ring, 0)
                words = jnp.arange(tag_rows * width).reshape(tag_rows, width)
                tags = jnp.where(
                    words < ring_rows * tag_fields,
                    jnp.where(old_context > 0, tags, -1),
                    0,
                )
                typed_ldst.store_typed(ring_ref, ring)
                typed_ldst.store_typed(tag_ref, tags)

                def token(i, _):
                    ring = typed_ldst.load_typed(ring_ref, view_dtype=jnp.bfloat16)
                    tags = typed_ldst.load_typed(tag_ref, view_dtype=jnp.int32)
                    slot = (old_context + i - starts_ref[row]) % ratio
                    token_mask = jnp.arange(count)[:, None] == i
                    # Select bits rather than summing floats: a floating sum
                    # loses negative zero and can canonicalize stored NaNs.
                    raw_bits = jax.lax.bitcast_convert_type(
                        raw_ref[...], jnp.uint16
                    ).astype(jnp.int32)
                    raw_token = jax.lax.bitcast_convert_type(
                        jnp.max(
                            jnp.where(token_mask, raw_bits, 0), axis=0, keepdims=True
                        ).astype(jnp.uint16),
                        jnp.bfloat16,
                    )
                    ring = jnp.where(
                        jnp.arange(ring_rows)[:, None] == slot, raw_token, ring
                    )
                    for axis in range(axes):
                        coordinate = (
                            positions_ref[axis, i] if axial else positions_ref[i]
                        )
                        tags = jnp.where(
                            words == tag_fields * slot + axis, coordinate, tags
                        )
                    tags = jnp.where(words == tag_fields * slot + axes, 0, tags)
                    typed_ldst.store_typed(ring_ref, ring)
                    typed_ldst.store_typed(tag_ref, tags)
                    tags_ref[...] = jnp.where(
                        token_mask, tags[0][None, :], tags_ref[...]
                    )

                    @pl.when(slot == ratio - 1)
                    def complete():
                        pooled = bf16(
                            jnp.mean(
                                ring[:ratio].astype(jnp.float32), axis=0, keepdims=True
                            )
                        )
                        keys_ref[...] = jnp.where(
                            token_mask,
                            gemma_norm(pooled, norm_ref[...], norm_eps),
                            keys_ref[...],
                        )

                @pl.when(starts_ref[row + 1] == starts_ref[row] + 1)
                def single_token():
                    token(starts_ref[row], None)

                @pl.when(starts_ref[row + 1] > starts_ref[row] + 1)
                def window():
                    jax.lax.fori_loop(starts_ref[row], starts_ref[row + 1], token, None)

                @pl.when(ids_ref[row] > 0)
                def store():
                    pltpu.make_async_copy(
                        staged.at[lane],
                        target.at[ids_ref[row], pl.ds(0, tokens)],
                        semaphores.at[lane],
                    ).start()

        for row in range(len(ids)):
            lane = starts_ref[row] if count < len(ids) else row

            @pl.when((starts_ref[row + 1] > starts_ref[row]) & (ids_ref[row] > 0))
            def finish(*, row=row, lane=lane):
                pltpu.make_async_copy(
                    staged.at[lane],
                    target.at[ids_ref[row], pl.ds(0, tokens)],
                    semaphores.at[lane],
                ).wait()

    pool, normalized, first_tags = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(raw.shape, lambda i, *_: (0, 0)),
                pl.BlockSpec(norm.shape, lambda i, *_: (0,)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(raw.shape, lambda i, *_: (0, 0)),
                pl.BlockSpec(raw.shape, lambda i, *_: (0, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((batch, tokens, *pool.shape[2:]), pool.dtype),
                pltpu.SemaphoreType.DMA((batch,)),
            ],
        ),
        out_shape=(
            jax.ShapeDtypeStruct(pool.shape, pool.dtype),
            jax.ShapeDtypeStruct(raw.shape, raw.dtype),
            jax.ShapeDtypeStruct(raw.shape, jnp.int32),
        ),
        input_output_aliases={6: 0},
        name="qsa_decode_ring_update",
    )(ids, starts, context, positions, raw, norm, pool)
    first_positions = first_tags[:, :axes].T if axial else first_tags[:, 0]
    first_positions = jnp.maximum(first_positions, 0)
    if not rotate:
        return pool, normalized, first_positions
    keys = index_rope(normalized, first_positions, trig, rotary_dim, rope_axes)
    return pool, keys
