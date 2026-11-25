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

"""Physical micro-page mapping for sparse decode (no KV materialization)."""

import jax
import jax.numpy as jnp


def virtual_page_metadata(
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    *,
    ratio: int = 4,
    page_size: int = 128,
) -> tuple[jax.Array, jax.Array]:
    """Map groups of ratio tokens through physical pages of page_size tokens.

    page_size comes from the original cache shape; the returned RPA page has
    ratio tokens. Valid group/page IDs and bounded lengths are caller contracts.
    """
    batch, budget = blocks.shape
    complete = lengths // ratio
    count = jnp.minimum(complete, budget)
    columns = jnp.arange(budget + 1)[None, :]
    groups = jnp.concatenate((blocks, jnp.zeros((batch, 1), jnp.int32)), axis=1)
    groups = jnp.where(columns == count[:, None], complete[:, None], groups)
    valid = (columns < count[:, None]) | (
        (columns == count[:, None]) & ((lengths % ratio)[:, None] > 0)
    )
    safe = jnp.where(valid, groups, 0)
    # Reuse sparse MLA's vector lookup for small page tables; the generic
    # gather lowers to a slower indirect lookup on this path.
    from vllm_torchtpu.kernels.mla.sparse.kernel import gather_page_ids

    physical = gather_page_ids(
        table.reshape(-1),
        safe // (page_size // ratio),
        jnp.arange(batch, dtype=jnp.int32),
        batch,
    )
    pages = jnp.where(
        valid, physical * (page_size // ratio) + safe % (page_size // ratio), 0
    )
    kv_lens = count * ratio + lengths % ratio
    return pages, kv_lens
