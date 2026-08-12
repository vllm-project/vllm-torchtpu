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
"""Pallas staging checks for the production PCP runtime schedule ABI."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import kernel
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.reference import \
    build_runtime_schedule_reference
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.schedule import (
    RuntimeScheduleField,
    build_pcp_streaming_schedule_inputs_from_metadata_jax)

pytestmark = pytest.mark.multichip


def _stage_runtime_schedule_kernel(
    tile_plan_ref,
    block_table_ref,
    output_ref,
    dma_sem,
    plan_vmem_ref,
    block_table_vmem_ref,
    schedule_vmem_ref,
    *,
    state_mode,
    group_idx,
    source_rank,
    pcp_size,
    page_size,
    interleave_size,
):
    kernel._load_hbm_row(tile_plan_ref, plan_vmem_ref, dma_sem, 0)
    kernel._load_hbm_row(block_table_ref, block_table_vmem_ref, dma_sem, 0)
    kernel._stage_schedule_step_from_tile_plan(
        plan_vmem_ref[...],
        block_table_vmem_ref[...],
        schedule_vmem_ref,
        jnp.asarray(group_idx, dtype=jnp.int32),
        jnp.asarray(source_rank, dtype=jnp.int32),
        state_mode=state_mode,
        pcp_size=pcp_size,
        page_size=page_size,
        interleave_size=interleave_size,
        lane=0,
    )
    store = pltpu.make_async_copy(
        src_ref=schedule_vmem_ref.at[:],
        dst_ref=output_ref.at[:],
        sem=dma_sem,
    )
    store.start()
    store.wait()


def _stage_runtime_schedule(tile_plan, block_tables, *, state_mode, group_idx,
                            source_rank, pcp_size, page_size, interleave_size):
    output_shape = jax.ShapeDtypeStruct(
        (pcp_size, 1, RuntimeScheduleField.PACKED_NUM_FIELDS),
        jnp.int32,
    )
    return pl.pallas_call(
        functools.partial(
            _stage_runtime_schedule_kernel,
            state_mode=state_mode,
            group_idx=group_idx,
            source_rank=source_rank,
            pcp_size=pcp_size,
            page_size=page_size,
            interleave_size=interleave_size,
        ),
        out_shape=output_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA,
                pltpu.VMEM((1, tile_plan.shape[2]), jnp.int32),
                pltpu.VMEM(block_tables.shape[1:], jnp.int32),
                pltpu.VMEM(
                    (pcp_size, 1, RuntimeScheduleField.PACKED_NUM_FIELDS),
                    jnp.int32,
                ),
            ),
            grid=(1, ),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=8 * 1024 * 1024),
        name=f"pcp_stage_{state_mode}_runtime_schedule",
    )(tile_plan, block_tables)


def _require_tpu():
    devices = jax.local_devices()
    if not devices or devices[0].platform != "tpu":
        pytest.skip("Runtime schedule staging requires a TPU device.")


@pytest.mark.parametrize(
    ("state_mode", "group_idx", "source_rank"),
    [
        ("current", 0, 1),
        ("current", 1, 0),
        ("history", 0, 1),
    ],
)
def test_pallas_staging_matches_runtime_schedule_reference(
        release_jax_backend, state_mode, group_idx, source_rank):
    _require_tpu()
    pcp_size = 2
    page_size = 4
    interleave_size = 2
    tile_plan, block_tables, *_ = \
        build_pcp_streaming_schedule_inputs_from_metadata_jax(
            kv_lens=np.asarray([19], dtype=np.int32),
            page_indices=np.arange(8, dtype=np.int32) + 5,
            cu_q_lens=np.asarray([0, 8], dtype=np.int32),
            distribution=np.asarray([0, 0, 1], dtype=np.int32),
            global_bucket_tokens=8,
            local_kv_cache_num_blocks=8,
            page_size=page_size,
            pcp_size=pcp_size,
            interleave_size=interleave_size,
            q_block_size=4,
        )
    reference = build_runtime_schedule_reference(
        tile_plan,
        block_tables,
        pcp_size=pcp_size,
        page_size=page_size,
        interleave_size=interleave_size,
    )
    active_tile = int(np.flatnonzero(reference.current_num_groups > 0)[0])
    one_tile_plan = tile_plan[active_tile:active_tile + 1]
    staged = jax.jit(
        functools.partial(
            _stage_runtime_schedule,
            state_mode=state_mode,
            group_idx=group_idx,
            source_rank=source_rank,
            pcp_size=pcp_size,
            page_size=page_size,
            interleave_size=interleave_size,
        ))(one_tile_plan, block_tables)
    staged.block_until_ready()

    rows = (reference.current_rows
            if state_mode == "current" else reference.history_rows)
    expected = rows[active_tile, group_idx, source_rank]
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(staged))[:, 0],
        expected,
    )
