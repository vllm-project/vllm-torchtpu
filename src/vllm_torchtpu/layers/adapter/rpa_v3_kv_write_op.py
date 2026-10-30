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
"""Torch bridge for the RPA v3 KV write without attention
(`kernels/ragged_paged_attention/v3/kv_write.py`)."""

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.ragged_paged_attention.v3.kv_write import write_kv
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata


def _write_kv_op_fn(
    kv_cache: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
) -> jax.Array:
    return write_kv(
        kv_cache, keys, values, kv_lens, page_indices, cu_q_lens, distribution
    )


def _fake_write_kv_op(
    kv_cache: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    kv_lens: torch.Tensor,
    page_indices: torch.Tensor,
    cu_q_lens: torch.Tensor,
    distribution: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(kv_cache)


# Built at import, not on first use: the first use is inside torch.compile's
# trace, which cannot trace the op's construction. As for the RPA op,
# kernel-iteration mode runs the op eagerly, where a donated cache would be
# dead before the copy_ writeback.
_write_kv_op = pallas.jax_op(
    "pallas::rpa_v3_kv_write",
    _write_kv_op_fn,
    donate_argnums=None if envs.TPU_KERNEL_ITER_MODE else (0,),
)
_write_kv_op.register_fake(_fake_write_kv_op)


def write_new_kv_(
    kv_cache: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> None:
    """Store key/value into kv_cache in place, exactly where RPA v3 would.

    kv_cache is the layer-major 5-D RPA v3 pool; key and value are
    [num_tokens, num_kv_heads, head_size] in the cache dtype.
    """
    new_kv_cache = _write_kv_op(
        kv_cache,
        key,
        value,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        attn_metadata.request_distribution,
    )
    # Donation + copy_ writeback, as in the RPA op: inside the compiled graph
    # XLA aliases the op output onto the donated pool, so the update is in
    # place.
    kv_cache.copy_(new_kv_cache)
