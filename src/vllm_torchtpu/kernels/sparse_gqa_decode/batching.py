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

"""Memory-bounded decode batching, independent of the benchmark baseline."""

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

# Performance scheduling cap; SMEM capacity is checked separately.
RPA_MAX_QUERY_CHUNK = 64
MAX_SCAN_UNROLL = 4


def rpa_query_batch_size(
    page_columns: int,
    *,
    query: jax.Array | jax.ShapeDtypeStruct,
    cache: jax.Array | jax.ShapeDtypeStruct,
    kv_heads: int,
    ratio: int,
) -> int:
    from vllm_torchtpu.kernels.experimental.batched_rpa import configs, schedule

    if page_columns < 1 or ratio < 1:
        raise ValueError("QSA page width and compression ratio must be positive")
    info = pltpu.get_tpu_info()
    batch = RPA_MAX_QUERY_CHUNK
    while batch:
        cfgs = configs.RpaConfigs(
            block=configs.BlockSizes(bq_sz=1, bq_c_sz=1, bkv_sz=256, batch_size=8),
            model=configs.ModelConfigs(
                num_q_heads=query.shape[1],
                num_kv_heads=kv_heads,
                head_dim=cache.shape[-1],
                mask_value=float(jnp.finfo(jnp.float32).min),
            ),
            serve=configs.ServingConfigs(
                num_seqs=batch,
                page_size=ratio,
                total_q_tokens=batch,
                num_page_indices=batch * page_columns,
                dtype_q=jnp.dtype(jnp.float32),
                dtype_kv=cache.dtype,
                dtype_out=jnp.dtype(jnp.float32),
                skip_kv_update=True,
            ),
            mode=configs.RpaCase.DECODE,
            vmem_limit_bytes=info.vmem_capacity_bytes,
        )
        scratch = schedule.BaseMetadataComputer.get_rpa_schedule(cfgs).scratch_shapes()
        scratch_bytes = sum(
            math.prod(ref.shape) * ref.dtype.itemsize
            for ref in jax.tree.leaves(scratch)
        )
        scalar_shapes = (
            (batch + 1,),
            (batch,),
            (batch,),
            (batch,),
            (batch * page_columns,),
        )
        scalar_bytes = sum(
            math.prod(shape) * jnp.dtype(jnp.int32).itemsize for shape in scalar_shapes
        )
        # Schedule generation has distribution[3] in place of the page table.
        scalar_bytes = max(
            scalar_bytes, (4 * batch + 1 + 3) * jnp.dtype(jnp.int32).itemsize
        )
        if scratch_bytes + scalar_bytes <= info.smem_capacity_bytes:
            return batch
        batch //= 2
    raise ValueError(
        "One QSA page-table row and the RPA schedule exceed SMEM capacity; "
        "reduce the sparse page budget."
    )


def attend_in_chunks(
    attend: Callable[..., tuple[jax.Array, jax.Array]],
    q: jax.Array,
    gate: jax.Array,
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    cache: jax.Array,
    *,
    batch_size: int,
    active_tokens: jax.Array | None = None,
    initial_output: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Bound per-launch SMEM use while preserving cache donation and row order.

    A runtime prefix requires read-only attention. It updates only prefix rows
    of initial_output; an empty prefix launches no attention work.
    """
    if batch_size < 1:
        raise ValueError("Decode batch size must be positive")
    if len(q) == 0:
        return jnp.zeros_like(q), cache
    if initial_output is not None and active_tokens is None:
        raise ValueError("An initial output requires a runtime attention prefix")
    if active_tokens is not None:
        batch = min(batch_size, len(q))

        def step(index, output):
            positions = index * batch + jnp.arange(batch, dtype=jnp.int32)
            live = positions < active_tokens
            rows = jnp.minimum(positions, len(q) - 1)
            result, _ = attend(
                q[rows],
                gate[rows],
                jnp.where(live[:, None], blocks[rows], -1),
                jnp.where(live, lengths[rows], 0),
                table[rows],
                cache,
            )
            destination = jnp.where(live, positions, len(q))
            return output.at[destination].set(result, mode="drop")

        # Segment attention is read-only. Only token output belongs in the
        # loop state; the pooled writer retains the cache owner.
        output = jnp.zeros_like(q) if initial_output is None else initial_output
        output = jax.lax.fori_loop(
            0, (active_tokens + batch - 1) // batch, step, output
        )
        return output, cache
    if len(q) <= batch_size:
        return attend(q, gate, blocks, lengths, table, cache)
    padding = (-len(q)) % batch_size

    def split(x):
        x = jnp.pad(x, ((0, padding),) + ((0, 0),) * (x.ndim - 1))
        return x.reshape(-1, batch_size, *x.shape[1:])

    def step(cache, inputs):
        query, gate, blocks, lengths, table = inputs
        output, cache = attend(query, gate, blocks, lengths, table, cache)
        return cache, output

    # Preserve the measured unroll bound for multi-chunk decode batches.
    num_chunks = (len(q) + batch_size - 1) // batch_size
    cache, output = jax.lax.scan(
        step,
        cache,
        tuple(map(split, (q, gate, blocks, lengths, table))),
        unroll=min(num_chunks, MAX_SCAN_UNROLL),
    )
    return output.reshape(-1, *q.shape[1:])[: len(q)], cache
