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
"""Shared scalar-prefetch, VMEM scratch-shape, and `pallas_call` launch helpers."""

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.pallas_util import (
    cdiv,
    get_dtype_bitwidth,
    has_bank_conflicts,
)


def make_scalar_prefetches(
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
):
    """Build the 7-tuple scalar-prefetch metadata shared by RPA v3 kernels.

    Slot layout:
      [0] `kv_lens`: `i32[max_num_seqs]`
      [1] `page_indices`: `i32[max_num_seqs * pages_per_seq]`
      [2] `cu_q_lens`: `i32[max_num_seqs + 1]`
      [3] `distribution`: `i32[3]` `(decode_end, prefill_end, mixed_end)`
      [4] `sem_ids`: `i32[3]` `(bq_sem_idx, bkv_sem_idx, bo_sem_idx)`
      [5] `bo_ids`: `i32[4]` `(bo_sem_0_seq_idx, bo_sem_1_seq_idx,
          bo_sem_0_bo_idx, bo_sem_1_bo_idx)`
      [6] `bkv_update_ids`: `i32[6]` `(bkv_sem_0_seq_idx, bkv_sem_1_seq_idx,
          bkv_sem_0_offset, bkv_sem_1_offset, bkv_sem_0_sz, bkv_sem_1_sz)`
    """
    return (
        kv_lens,
        # TODO(jevinjiang): can we use ragged page_indices to save some smem?
        page_indices,
        cu_q_lens,
        distribution,
        jnp.zeros((3,), jnp.int32),
        jnp.full((4,), -1, jnp.int32),
        jnp.full((6,), -1, jnp.int32),
    )


def make_base_rpa_scratch_shapes(
    *,
    q_tail_shape: tuple[int, ...],
    q_dtype,
    kv_tail_shape: tuple[int, ...],
    kv_dtype,
    out_dtype,
    actual_num_kv_heads: int,
    bq_sz: int,
    bkv_sz: int,
    kv_rows_per_token: int,
    pad_bkv_bank_conflicts: bool = True,
) -> list[Any]:
    """Build the 7 shared double-buffered VMEM and DMA scratch specs for RPA v3.

    Scratch slot layout:
      [0] `bkv_x2_ref`: double-buffered KV VMEM tile (padded by +1 on bank conflict
          when `pad_bkv_bank_conflicts` is True)
      [1] `bq_x2_ref`: double-buffered Q VMEM tile
      [2] `bo_x2_ref`: double-buffered O VMEM tile
      [3] `sems`: `DMA((4, 2))` rows `(SEM_BKV_FETCH, SEM_BQ_FETCH, SEM_BO_SEND,
          SEM_BKV_UPDATE)`
      [4] `l_ref`: online softmax row-sum VMEM scratch
      [5] `m_ref`: online softmax row-max VMEM scratch
      [6] `acc_ref`: output accumulator VMEM scratch
    """
    assert len(q_tail_shape) == 3
    num_q_heads_per_kv_head = q_tail_shape[0] * q_tail_shape[1]
    head_dim = q_tail_shape[-1]
    bkv_stride = kv_rows_per_token + int(
        pad_bkv_bank_conflicts and has_bank_conflicts(kv_rows_per_token)
    )
    bkv_double_buf = pltpu.VMEM((2, bkv_sz, bkv_stride, *kv_tail_shape), kv_dtype)
    bq_double_buf = pltpu.VMEM((2, actual_num_kv_heads, bq_sz, *q_tail_shape), q_dtype)
    l_scratch = pltpu.VMEM(
        (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128), out_dtype
    )
    acc_scratch = pltpu.VMEM(
        (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim), out_dtype
    )
    return [
        bkv_double_buf,
        bq_double_buf,
        bq_double_buf,
        pltpu.SemaphoreType.DMA((4, 2)),
        l_scratch,
        l_scratch,
        acc_scratch,
    ]


def get_scratch_vmem_bytes(scratch_shapes: list[Any]) -> int:
    """Calculate total VMEM bytes across all `pltpu.VMEM` specs in `scratch_shapes`."""
    total_bits = sum(
        math.prod(spec.shape) * get_dtype_bitwidth(spec.dtype)
        for spec in scratch_shapes
        if spec.memory_space == pltpu.VMEM
    )
    return cdiv(total_bits, 8)


def invoke_rpa_pallas_kernel(
    *,
    kernel_fn: Callable[..., Any],
    scalar_prefetches: tuple[jax.Array, ...],
    inputs: tuple[Any, ...],
    scratch_shapes: list[Any],
    out_arrays: tuple[jax.Array, jax.Array],
    input_output_aliases: dict[int, int],
    vmem_limit_bytes: int,
    scope_name: str,
    tpu_version: int,
    disable_bounds_checks: bool,
    disable_semaphore_checks: bool,
    in_specs: list[Any] | None = None,
    grid: tuple[Any, ...] = (1,),
    pin_hbm: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Compile and invoke a single-core pipelined RPA v3 `pallas_call`."""
    resolved_in_specs = (
        [pl.BlockSpec(memory_space=pltpu.HBM)] * len(inputs)
        if in_specs is None
        else in_specs
    )
    assert len(resolved_in_specs) == len(inputs)
    use_hbm_pinning = pin_hbm and tpu_version >= 7
    hbm_cls = pltpu.HBM if use_hbm_pinning else jax.ShapeDtypeStruct
    kernel = pl.pallas_call(
        kernel_fn,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=resolved_in_specs,
            out_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * len(out_arrays),
            grid=grid,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            # TODO(jevinjiang): since each sequence depends on the previous
            # one, we need some extra work to support Megacore mode.
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=vmem_limit_bytes,
            # Paged attention invokes multiple small DMAs for each page
            # instead of a single large DMA. Therefore, the overhead of bounds
            # checking becomes too significant so we disable it.
            disable_bounds_checks=disable_bounds_checks,
            # Only set to true if you guarantee there is no race condition.
            disable_semaphore_checks=disable_semaphore_checks,
        ),
        out_shape=[hbm_cls(shape=a.shape, dtype=a.dtype) for a in out_arrays],
        input_output_aliases=input_output_aliases,
        name=scope_name,
    )
    if use_hbm_pinning:
        # jit to color the memory since the q, kv are just preprocessed.
        @jax.jit
        def run(scalars, *args):
            constrained_args = [
                pltpu.with_memory_space_constraint(a, pltpu.HBM)
                if (
                    a is not None
                    and spec is not None
                    and spec.memory_space == pltpu.HBM
                )
                else a
                for a, spec in zip(args, resolved_in_specs, strict=True)
            ]
            return kernel(*scalars, *constrained_args)

        return run(scalar_prefetches, *inputs)

    if not pin_hbm:
        return jax.named_scope(scope_name)(kernel)(*scalar_prefetches, *inputs)

    # TODO: v6 has issues with pinning aliased memory.
    return kernel(*scalar_prefetches, *inputs)
