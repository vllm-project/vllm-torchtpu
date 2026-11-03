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
"""The KV write of ragged paged attention v3, without the attention.

`write_kv` leaves the cache exactly as `ragged_paged_attention` leaves it: the
same slots, the same merged K/V rows (`merge_kv`), and no other slot touched.
For callers that need only the write.
"""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.ragged_paged_attention.v3.kernel import (
    get_kv_cache_shape,
    merge_kv,
)


def new_token_slots(
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    num_tokens: int,
    page_size: int,
    num_cache_slots: int,
) -> jax.Array:
    """Flat cache slot RPA v3 writes each new token to.

    Token t of sequence s lands at KV position kv_lens[s] - q_len + (t -
    cu_q_lens[s]), the sequence's last q_len positions, resolved through
    page_indices. Tokens the kernel does not write get a unique slot past the
    cache, which a scatter with mode="drop" skips: padding, sequences past
    distribution[2], and the prefill-only range [distribution[0],
    distribution[1]), which the kernel leaves unprocessed without a chunk
    prefill size.
    """
    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    decode_end, prefill_end, num_seqs = (
        distribution[0],
        distribution[1],
        distribution[2],
    )
    tokens = jnp.arange(num_tokens, dtype=jnp.int32)
    seqs = jnp.arange(max_num_seqs, dtype=jnp.int32)
    q_ends = cu_q_lens[1:]

    ended = (q_ends[None, :] <= tokens[:, None]) & (seqs[None, :] < num_seqs)
    token_seqs = jnp.minimum(jnp.sum(ended, axis=1, dtype=jnp.int32), max_num_seqs - 1)
    in_prefill_range = (token_seqs >= decode_end) & (token_seqs < prefill_end)
    written = (tokens < cu_q_lens[num_seqs]) & ~in_prefill_range

    positions = kv_lens[token_seqs] - q_ends[token_seqs] + tokens
    table_entries = jnp.clip(
        token_seqs * pages_per_seq + positions // page_size,
        0,
        page_indices.shape[0] - 1,
    )
    slots = page_indices[table_entries] * page_size + positions % page_size
    return jnp.where(written, slots, num_cache_slots + tokens)


def _validate_inputs(kv_cache: jax.Array, keys: jax.Array, values: jax.Array) -> None:
    """The shape and dtype checks of RPA v3's static_validate_inputs that
    apply to the write. Shapes and dtypes are static, so this costs nothing
    at run time."""
    if keys.ndim != 3 or keys.shape != values.shape:
        raise ValueError(
            f"Expected 3-D keys and values of one shape, got {keys.shape=}, "
            f"{values.shape=}"
        )
    if not (kv_cache.dtype == keys.dtype == values.dtype):
        raise ValueError(
            f"Expected {kv_cache.dtype=} to equal {keys.dtype=} and {values.dtype=}"
        )
    expected_shape = get_kv_cache_shape(
        kv_cache.shape[0],
        kv_cache.shape[1],
        keys.shape[1],
        keys.shape[2],
        kv_cache.dtype,
    )
    if kv_cache.shape != expected_shape:
        raise ValueError(
            f"Expected {kv_cache.shape=} to equal {expected_shape=} for "
            f"{keys.shape[1]} KV heads of head dim {keys.shape[2]}"
        )


def write_kv(
    kv_cache: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
) -> jax.Array:
    """Returns kv_cache with keys/values written where RPA v3 writes them.

    kv_cache: [total_num_pages, page_size, num_kv_heads_x2 // kv_packing,
    kv_packing, head_dim], as get_kv_cache_shape gives for keys' heads and
    head dim. keys, values: [num_tokens, num_kv_heads, actual_head_dim] in the
    cache dtype. The other arguments are RPA v3's. Raises ValueError on the
    shape and dtype mismatches the kernel rejects.

    Donate kv_cache to update it in place. The scatter goes through a
    [slots, num_kv_heads_x2, head_dim] view, which holds the same bytes in the
    cache's TPU layout; scattering [num_kv_heads_x2 // kv_packing, kv_packing,
    head_dim] rows makes XLA copy the whole cache around the scatter.

    The scatter assumes no two written tokens share a slot (unique_indices):
    distinct sequences must own distinct pages, as the paged KV cache
    allocates them. If two tokens did share a slot, the kernel's sequential
    write would keep the last; here the result is undefined.
    """
    _validate_inputs(kv_cache, keys, values)
    total_num_pages, page_size, heads_per_packing, kv_packing, head_dim = kv_cache.shape
    num_kv_heads_x2 = heads_per_packing * kv_packing
    num_cache_slots = total_num_pages * page_size
    num_tokens = keys.shape[0]
    slots = new_token_slots(
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        num_tokens=num_tokens,
        page_size=page_size,
        num_cache_slots=num_cache_slots,
    )
    rows = merge_kv(keys, values).reshape(num_tokens, num_kv_heads_x2, -1)
    rows = jnp.pad(rows, ((0, 0), (0, 0), (0, head_dim - rows.shape[-1])))
    flat_cache = kv_cache.reshape(num_cache_slots, num_kv_heads_x2, head_dim)
    flat_cache = flat_cache.at[slots].set(rows, mode="drop", unique_indices=True)
    return flat_cache.reshape(kv_cache.shape)
