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
"""Pallas body of the micro-batched hierarchical reduce-scatter."""

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives.hierrs_tc.config import Config
from vllm_torchtpu.kernels.collectives.hierrs_tc.dma_pipeline import (
    DmaManager,
)
from vllm_torchtpu.kernels.collectives.hierrs_tc.topology import ChunkLocator, Topology


def hier_rs_kernel(
    input_ref,
    output_ref,
    running_sum_ref,
    recv_buf_ref,
    # Separate from recv_buf_ref so a neighbour's phase-2 chunk cannot land on
    # phase-1 data this device has not consumed yet.
    phase2_landing_ref,
    recv_bref,
    run_bref,
    out_bref,
    phase1_send_sems,
    phase1_recv_sems,
    phase2_send_sems,
    phase2_recv_sems,
    *,
    config: Config,
    axis_name: str = "x",
):
    topo = Topology(axis_name)
    locator = ChunkLocator(config, topo)
    dma = DmaManager(
        config,
        topo,
        locator,
        recv_bref,
        run_bref,
        out_bref,
        phase1_send_sems,
        phase1_recv_sems,
        phase2_send_sems,
        phase2_recv_sems,
    )
    num_mb = config.num_micro_batches
    phase1_ops = []
    phase2_ops = []

    def start_phase1(mb):
        phase1_ops.extend(
            dma.start_phase1_d2d_copies(src=input_ref, dst=recv_buf_ref, mb_idx=mb)
        )

    def accumulate_phase1(mb):
        dma.run_phase1_accumulate_pipeline(
            src1=recv_buf_ref,
            src2=input_ref,
            dst=running_sum_ref,
            in1_index_fn=locator.make_phase1_packed_index_fn(mb),
            in2_index_fn=locator.make_phase1_index_fn(mb),
            out_index_fn=locator.make_phase1_packed_index_fn(mb),
            hbm_index_fn=locator.make_phase1_in_index_fn_with_recv_sem(mb),
            block_size=config.mb_size,
            mb_idx=mb,
        )

    def start_phase2(mb, step):
        phase2_ops.extend(
            dma.start_phase2_c2c_copies(
                mb_idx=mb, step_idx=step, src=running_sum_ref, dst=phase2_landing_ref
            )
        )

    def accumulate_phase2(mb, step):
        is_last_step = step == config.num_hcube_dims - 1
        dma.run_phase2_accumulate_pipeline(
            src1=phase2_landing_ref,
            src2=running_sum_ref,
            dst=output_ref if is_last_step else running_sum_ref,
            in_index_fn=locator.make_phase2_index_fn(step, mb),
            out_index_fn=locator.make_phase2_out_index_fn(step, mb)
            if is_last_step
            else locator.make_phase2_index_fn(step, mb),
            hbm_index_fn=locator.make_phase2_in_index_fn_with_recv_sem(step, mb),
            block_size=config.hc_chunk_size,
            mb_idx=mb,
            step_idx=step,
        )

    # Phase 1 of micro-batch m + 1 overlaps hypercube step 0 of micro-batch m.
    start_phase1(0)
    accumulate_phase1(0)
    start_phase2(0, step=0)
    for m in range(num_mb):
        if m < num_mb - 1:
            start_phase1(m + 1)
            accumulate_phase1(m + 1)
            start_phase2(m + 1, step=0)
        accumulate_phase2(m, step=0)

    # Step 1 starts only after every step-0 accumulate has written running_sum.
    start_phase2(0, step=1)
    for m in range(num_mb):
        if m < num_mb - 1:
            start_phase2(m + 1, step=1)
        accumulate_phase2(m, step=1)

    for op in phase1_ops:
        op.wait_send()
    for op in phase2_ops:
        op.wait_send()


def make_unified_scratch_shapes(
    seq_chunk_size,
    mb_size,
    dtype,
    num_chips,
    num_hcube_dims,
    num_micro_batches,
):
    block_spec = pl.BlockSpec(
        block_shape=(seq_chunk_size, mb_size), index_map=lambda *args: (0, 0)
    )
    recv_bref = pltpu.BufferedRef.input(block_spec, dtype, buffer_count=2)
    run_bref = pltpu.BufferedRef.input(block_spec, dtype, buffer_count=2)
    out_bref = pltpu.BufferedRef.output(block_spec, dtype, buffer_count=2)

    p2_sem_shape = (
        num_hcube_dims,
        num_micro_batches,
        num_hcube_dims,
        1 << max(0, num_hcube_dims - 1),
    )

    scratch_shapes = [
        recv_bref,
        run_bref,
        out_bref,
        pltpu.SemaphoreType.DMA((num_chips, num_micro_batches)),
        pltpu.SemaphoreType.DMA((num_chips, num_micro_batches)),
        pltpu.SemaphoreType.DMA(p2_sem_shape),
        pltpu.SemaphoreType.DMA(p2_sem_shape),
    ]
    # In hier_rs_kernel's parameter order.
    return scratch_shapes
