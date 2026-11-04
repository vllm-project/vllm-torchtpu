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
"""Torch entry point for the JAX MSA forward."""

import functools
from collections.abc import Callable

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.minimax_m3.kernels import msa

_KV_CACHE_ARG = 5
_INDEX_CACHE_ARG = 6


@functools.cache
def get_msa_op(
    sm_scale: float, topk: int, init_blocks: int, local_blocks: int
) -> Callable[..., torch.Tensor]:
    """Build (once per config) the op that runs `msa.msa_forward` in place.

    Built eagerly from module `__init__`: `pallas.jax_op` inspects the
    function signature, which Dynamo cannot trace inside a compiled forward.
    """

    def _msa_jax(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        index_q: jax.Array,
        index_k: jax.Array,
        kv_cache: jax.Array,
        index_cache: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return msa.msa_forward(
            q,
            k,
            v,
            index_q,
            index_k,
            kv_cache,
            index_cache,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            sm_scale=sm_scale,
            topk=topk,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )

    op_name = (
        f"pallas::minimax_m3_msa_t{topk}_i{init_blocks}_l{local_blocks}"
        f"_s{sm_scale:.8f}".replace(".", "p")
    )
    jax_op = pallas.jax_op(
        op_name, _msa_jax, donate_argnums=(_KV_CACHE_ARG, _INDEX_CACHE_ARG)
    )

    def _fake(q, k, v, index_q, index_k, kv_cache, index_cache, *args):
        return (
            torch.empty_like(kv_cache),
            torch.empty_like(index_cache),
            (torch.empty_like(q)),
        )

    jax_op.register_fake(_fake)

    def op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        kv_cache: torch.Tensor,
        index_cache: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        new_kv_cache, new_index_cache, out = jax_op(
            q,
            k,
            v,
            index_q,
            index_k,
            kv_cache,
            index_cache,
            metadata.seq_lens,
            metadata.block_tables,
            metadata.query_start_loc,
            metadata.request_distribution,
        )
        kv_cache.copy_(new_kv_cache)
        index_cache.copy_(new_index_cache)
        return out

    return op
