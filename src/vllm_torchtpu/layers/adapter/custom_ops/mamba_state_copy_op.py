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

import functools

import jax
import jax.numpy as jnp
import torch
from jax.experimental.pallas import tpu as pltpu
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels import pool_adapters

# --- Unified-pool block seeding (block-table-derived state indexing) ---
# The GDN state lives in the block containing the request's last computed
# token; when that block advances between scheduler steps (chunked-prefill
# boundary, decode crossing, prefix-cache resume) the new block must be
# seeded from the previous state block before the forward runs — the
# counterpart of upstream vLLM's ``preprocess_mamba`` block copies.


def _copy_fn(
    pool: jax.Array, src: jax.Array, dst: jax.Array
) -> tuple[jax.Array, jax.Array]:
    new_pool = pool_adapters.copy_blocks(pool, src, dst)
    return new_pool, src[0]


_copy_op = pallas.jax_op(
    "pallas::mamba_state_block_copy", _copy_fn, donate_argnums=(0,)
)


def _fake_copy(pool: torch.Tensor, src: torch.Tensor, dst: torch.Tensor):
    return torch.empty_like(pool), torch.empty((), dtype=src.dtype, device=src.device)


_copy_op.register_fake(_fake_copy)


@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def copy_mamba_state_blocks(
    pools: list[torch.Tensor], src: torch.Tensor, dst: torch.Tensor
) -> torch.Tensor:
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


def _copy_step_budget_bytes() -> int:
    """Returns the per-step VMEM byte budget for `copy_blocks`.

    Half of the core's VMEM is left to the compiler and the rest is divided
    across four block buffers (double-buffered input and output). Evaluated at
    trace time: `get_tpu_info()` initializes the PJRT runtime.
    """
    return pltpu.get_tpu_info().vmem_capacity_bytes // 2 // 4


def _row_copy_step_rows(rows: int, kernel_block_bytes: int, budget_bytes: int) -> int:
    """Returns the largest divisor of `rows` whose blocks fit in `budget_bytes`."""
    fitting = [
        k
        for k in range(1, rows + 1)
        if rows % k == 0 and k * kernel_block_bytes <= budget_bytes
    ]
    if not fitting:
        raise ValueError(
            f"a kernel block of {kernel_block_bytes} bytes exceeds the "
            f"{budget_bytes}-byte VMEM step budget of the row copy"
        )
    return max(fitting)


@functools.cache
def _row_copy_fn(num_pools: int, split: int, step_rows: int):
    """Compiles the whole-row copy `pool[dst[i]] = pool[src[i]]` for one geometry.

    A manager row of the `(num_blocks, num_pools * split, *page)` pool packs
    `split` kernel blocks for each of the `num_pools` regions. Each
    `(src[i], dst[i])` pair expands into `steps` chunks of `step_rows` kernel
    blocks, copied through zero-copy bitcast views of the donated buffer.
    """
    rows = num_pools * split
    assert rows % step_rows == 0, (rows, step_rows)
    steps = rows // step_rows

    def _rows_fn(
        pool: jax.Array, src: jax.Array, dst: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        assert pool.shape[1] == rows, (pool.shape, rows)
        view = pool.reshape(pool.shape[0] * steps, step_rows, *pool.shape[2:])
        offsets = jnp.arange(steps, dtype=src.dtype)
        src_steps = (src[:, None] * steps + offsets).reshape(-1)
        dst_steps = (dst[:, None] * steps + offsets).reshape(-1)
        new_pool = pool_adapters.copy_blocks(view, src_steps, dst_steps)
        return new_pool.reshape(pool.shape), src[0]

    op = pallas.jax_op(
        f"pallas::mamba_state_row_copy_{num_pools}x{split}s{step_rows}",
        _rows_fn,
        donate_argnums=(0,),
    )

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def run(pool: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        new_pool, marker = op(pool, src, dst)
        pool.copy_(new_pool)
        return marker

    return run


def copy_mamba_state_rows(
    pool: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, num_pools: int, split: int
) -> torch.Tensor:
    """Copies whole manager-block rows in place within the merged block-major pool.

    Each `(src[i], dst[i])` pair moves `num_pools * split` kernel blocks.
    """
    kernel_block_bytes = pool[0, 0].numel() * pool.element_size()
    step_rows = _row_copy_step_rows(
        num_pools * split, kernel_block_bytes, _copy_step_budget_bytes()
    )
    return _row_copy_fn(num_pools, split, step_rows)(pool, src, dst)
