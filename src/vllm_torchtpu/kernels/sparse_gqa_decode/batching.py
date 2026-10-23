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

from collections.abc import Callable

import jax
import jax.numpy as jnp

# Inherited RPA page-table/code-size policies, not hardware maxima.
RPA_MAX_QUERY_CHUNK = 64
RPA_PAGE_TABLE_BUDGET_BYTES = 160 * 1024
MAX_SCAN_UNROLL = 4


def rpa_query_batch_size(page_columns: int) -> int:
    # Preserve QSA's existing SMEM budget: the 128-step RPA schedule occupies
    # about 815 KiB on TPU7x. Reserve 160 KiB for page indices and cap each
    # launch at 64 queries, which needs at most 72 decode steps at this budget.
    max_queries = min(
        RPA_MAX_QUERY_CHUNK,
        RPA_PAGE_TABLE_BUDGET_BYTES // (jnp.dtype(jnp.int32).itemsize * page_columns),
    )
    return 1 << (max_queries.bit_length() - 1)


def attend_in_chunks(
    attend: Callable[..., tuple[jax.Array, jax.Array]],
    q: jax.Array,
    gate: jax.Array,
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    cache: jax.Array,
    *,
    batch_size: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Bound per-launch SMEM use while preserving cache donation and row order."""
    if batch_size is None:
        batch_size = rpa_query_batch_size(blocks.shape[1] + 1)
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
