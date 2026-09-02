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
"""Donating slot-to-slot copy for Mamba recurrent state arrays."""

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels import pool_adapters

# --- Unified-pool block seeding (block-table-derived state indexing) ---
# The GDN state lives in the block containing the request's last computed
# token; when that block advances between scheduler steps (chunked-prefill
# boundary, decode crossing, prefix-cache resume) the new block must be
# seeded from the previous state block before the forward runs — the
# counterpart of upstream vLLM's ``preprocess_mamba`` block copies.


def _copy_fn(pool: jax.Array, src: jax.Array,
             dst: jax.Array) -> tuple[jax.Array, jax.Array]:
    new_pool = pool_adapters.copy_blocks(pool, src, dst)
    return new_pool, src[0]


_copy_op = pallas.jax_op("pallas::mamba_state_block_copy",
                         _copy_fn,
                         donate_argnums=(0, ))


def _fake_copy(pool: torch.Tensor, src: torch.Tensor, dst: torch.Tensor):
    return torch.empty_like(pool), torch.empty((),
                                               dtype=src.dtype,
                                               device=src.device)


_copy_op.register_fake(_fake_copy)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def copy_mamba_state_blocks(pools: list[torch.Tensor], src: torch.Tensor,
                            dst: torch.Tensor) -> torch.Tensor:
    """``pool[dst[i]] = pool[src[i]]`` on every pool, as one program.

    A hybrid model's unified pool is split over several raw buffers, and a
    state-block advance must be mirrored on each of them with the same
    pairs; seeding them all from one program costs one dispatch per step
    instead of one per buffer.
    """
    marker = None
    for pool in pools:
        # Plain donation + copy_ writeback (aliased in-place by XLA).
        new_pool, marker = _copy_op(pool, src, dst)
        pool.copy_(new_pool)
    return marker
