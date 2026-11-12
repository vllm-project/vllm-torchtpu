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
"""Typed QSA ring/tag I/O with one batch of outstanding DMAs per tile."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import typed_ldst
from vllm_torchtpu.kernels.qwen4_exp.cache_regions import region_rows
from vllm_torchtpu.kernels.qwen4_exp.tuned_block_sizes import get_ring_tag_dma_batch


def read_ring_and_tags(
    pool: jax.Array, ids: jax.Array, ring_elements: int, tag_elements: int = 0
) -> tuple[jax.Array, jax.Array | None]:
    """Read adjacent BF16 ring/int32 tags; inactive IDs read the null page."""
    count = len(ids)
    ring_tokens, ring_rows = region_rows(pool, ring_elements, jnp.bfloat16)
    tag_tokens, tag_rows = (
        region_rows(pool, tag_elements, jnp.int32) if tag_elements > 0 else (0, 0)
    )
    tokens = ring_tokens + tag_tokens
    if tokens > pool.shape[1]:
        raise ValueError("QSA state regions exceed the physical page capacity")
    payload = tuple(pool.shape[2:])
    lanes = pool.shape[-1]
    batch = get_ring_tag_dma_batch(
        pool, count, ring_elements=ring_elements, tag_elements=tag_elements
    )

    def kernel(indices, source, ring_out, *rest):
        tags_out, staged, semaphores = rest if tag_elements > 0 else (None, *rest)
        first = pl.program_id(0) * batch
        for lane in range(batch):
            pltpu.make_async_copy(
                source.at[jnp.maximum(indices[first + lane], 0), pl.ds(0, tokens)],
                staged.at[lane],
                semaphores.at[lane],
            ).start()
        for lane in range(batch):
            pltpu.make_async_copy(
                source.at[jnp.maximum(indices[first + lane], 0), pl.ds(0, tokens)],
                staged.at[lane],
                semaphores.at[lane],
            ).wait()
            ring_out[lane] = typed_ldst.load_typed(
                staged.at[lane, pl.ds(0, ring_tokens)], view_dtype=jnp.bfloat16
            )
            if tags_out is not None:
                tags_out[lane] = typed_ldst.load_typed(
                    staged.at[lane, pl.ds(ring_tokens, tag_tokens)],
                    view_dtype=jnp.int32,
                )

    ring_spec = pl.BlockSpec((batch, ring_rows, lanes), lambda i, *_: (i, 0, 0))
    ring_shape = jax.ShapeDtypeStruct((count, ring_rows, lanes), jnp.bfloat16)
    out = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(count // batch,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=(
                [
                    ring_spec,
                    pl.BlockSpec((batch, tag_rows, lanes), lambda i, *_: (i, 0, 0)),
                ]
                if tag_elements > 0
                else ring_spec
            ),
            scratch_shapes=[
                pltpu.VMEM((batch, tokens, *payload), pool.dtype),
                pltpu.SemaphoreType.DMA((batch,)),
            ],
        ),
        out_shape=(
            [ring_shape, jax.ShapeDtypeStruct((count, tag_rows, lanes), jnp.int32)]
            if tag_elements > 0
            else ring_shape
        ),
        name="qwen4_exp_ring_tags_batched_read",
    )(ids, pool)
    if tag_elements == 0:
        return out.reshape(count, -1)[:, :ring_elements], None
    ring, tags = out
    return (
        ring.reshape(count, -1)[:, :ring_elements],
        tags.reshape(count, -1)[:, :tag_elements],
    )


def write_ring_and_tags(
    pool: jax.Array, ring: jax.Array, tags: jax.Array | None, ids: jax.Array
) -> jax.Array:
    """Write unique positive IDs in place, preserving idle/null page bytes."""
    count, ring_elements = ring.shape
    has_tags = tags is not None and tags.shape[-1] > 0
    tag_elements = tags.shape[-1] if has_tags else 0
    ring_tokens, ring_rows = region_rows(pool, ring_elements, ring.dtype)
    tag_tokens, tag_rows = (
        region_rows(pool, tag_elements, tags.dtype) if has_tags else (0, 0)
    )
    tokens = ring_tokens + tag_tokens
    if tokens > pool.shape[1]:
        raise ValueError("QSA state regions exceed the physical page capacity")
    payload = tuple(pool.shape[2:])
    lanes = pool.shape[-1]
    batch = get_ring_tag_dma_batch(
        pool, count, ring_elements=ring_elements, tag_elements=tag_elements
    )
    ring = jnp.pad(ring, ((0, 0), (0, ring_rows * lanes - ring_elements))).reshape(
        count, ring_rows, lanes
    )
    if has_tags:
        assert tags is not None
        tags = jnp.pad(tags, ((0, 0), (0, tag_rows * lanes - tag_elements))).reshape(
            count, tag_rows, lanes
        )

    def kernel(indices, ring_ref, *rest):
        tags_ref, _old, target, staged, semaphores = rest if has_tags else (None, *rest)
        first = pl.program_id(0) * batch
        for lane in range(batch):
            typed_ldst.store_typed(
                staged.at[lane, pl.ds(0, ring_tokens)], ring_ref[lane]
            )
            if tags_ref is not None:
                typed_ldst.store_typed(
                    staged.at[lane, pl.ds(ring_tokens, tag_tokens)], tags_ref[lane]
                )

            @pl.when(indices[first + lane] > 0)
            def start(*, lane=lane):
                pltpu.make_async_copy(
                    staged.at[lane],
                    target.at[indices[first + lane], pl.ds(0, tokens)],
                    semaphores.at[lane],
                ).start()

        for lane in range(batch):

            @pl.when(indices[first + lane] > 0)
            def wait(*, lane=lane):
                pltpu.make_async_copy(
                    staged.at[lane],
                    target.at[indices[first + lane], pl.ds(0, tokens)],
                    semaphores.at[lane],
                ).wait()

    in_specs = [pl.BlockSpec((batch, ring_rows, lanes), lambda i, *_: (i, 0, 0))]
    if has_tags:
        in_specs.append(pl.BlockSpec((batch, tag_rows, lanes), lambda i, *_: (i, 0, 0)))
    in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))
    args = (ids, ring, tags, pool) if has_tags else (ids, ring, pool)
    return pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(count // batch,),
            in_specs=in_specs,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((batch, tokens, *payload), pool.dtype),
                pltpu.SemaphoreType.DMA((batch,)),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={3 if has_tags else 2: 0},
        name="qwen4_exp_ring_tags_batched_write",
    )(*args)
