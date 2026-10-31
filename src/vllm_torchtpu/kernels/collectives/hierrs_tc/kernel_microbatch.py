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

import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.collectives.hierrs_tc.config import SCALE_LANE, Config
from vllm_torchtpu.kernels.collectives.hierrs_tc.dma_pipeline import (
    DmaManager,
)
from vllm_torchtpu.kernels.collectives.hierrs_tc.topology import ChunkLocator, Topology


def hier_rs_kernel(
    input_ref,
    output_ref,
    running_sum_ref,
    recv_buf_ref,
    *args,
    config: Config,
    axis_name: str = "x",
    fp8_comm: bool = False,
    fp8_static_scale: float | None = None,
):
    if fp8_comm:
        (
            fp8_send_ref,
            fp8_recv_ref,
            scale_send_ref,
            scale_recv_ref,
            recv_bref,
            run_bref,
            out_bref,
            phase1_send_sems,
            phase1_recv_sems,
            phase2_send_sems,
            phase2_recv_sems,
            scale_p2_send_sems,
            scale_p2_recv_sems,
            fp8_send_bref,
            scale_send_bref,
            fp8_recv_bref,
            scale_bref,
        ) = args
        fp8_dma_refs = dict(
            fp8_send_buf=fp8_send_ref,
            fp8_recv_buf=fp8_recv_ref,
            scale_send_buf=scale_send_ref,
            scale_recv_buf=scale_recv_ref,
            scale_p2_send_sems=scale_p2_send_sems,
            scale_p2_recv_sems=scale_p2_recv_sems,
            fp8_send_bref=fp8_send_bref,
            scale_send_bref=scale_send_bref,
            fp8_recv_bref=fp8_recv_bref,
            scale_bref=scale_bref,
        )
        # The FP8 copies land in fp8_recv_ref and never use this.
        phase2_landing_ref = recv_buf_ref
    else:
        (
            # Separate from recv_buf_ref so a neighbour's phase-2 chunk cannot
            # land on phase-1 data this device has not consumed yet.
            phase2_landing_ref,
            recv_bref,
            run_bref,
            out_bref,
            phase1_send_sems,
            phase1_recv_sems,
            phase2_send_sems,
            phase2_recv_sems,
        ) = args
        fp8_dma_refs = {}

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
        **fp8_dma_refs,
        fp8_static_scale=fp8_static_scale,
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
        if fp8_comm:
            dma.quantize_chunks_to_fp8_staging(running_sum_ref, mb, step_idx=step)
        copies = dma.start_phase2_c2c_copies(
            mb_idx=mb,
            step_idx=step,
            src=running_sum_ref,
            dst=phase2_landing_ref,
            fp8=fp8_comm,
        )
        phase2_ops.extend(data for data, _ in copies)
        phase2_ops.extend(scale for _, scale in copies if scale is not None)

    def accumulate_phase2(mb, step):
        is_last_step = step == config.num_hcube_dims - 1
        if fp8_comm:
            dma.run_phase2_dequant_accumulate_pipeline(
                running_sum_ref,
                output_ref,
                mb,
                step_idx=step,
                is_last_step=is_last_step,
            )
            return
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
    fp8_comm: bool = False,
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
    if fp8_comm:
        fp8_block_spec = pl.BlockSpec(
            block_shape=(seq_chunk_size, mb_size), index_map=lambda *a: (0, 0)
        )
        scale_block_spec = pl.BlockSpec(
            block_shape=(1, SCALE_LANE), index_map=lambda *a: (0, 0)
        )
        fp8 = jnp.float8_e4m3fn
        # The FP8 data copies reuse the phase-2 semaphores; only the scale
        # copies need their own.
        scratch_shapes += [
            pltpu.SemaphoreType.DMA(p2_sem_shape),
            pltpu.SemaphoreType.DMA(p2_sem_shape),
            pltpu.BufferedRef.output(fp8_block_spec, fp8, buffer_count=2),
            pltpu.BufferedRef.output(scale_block_spec, jnp.float32, buffer_count=2),
            pltpu.BufferedRef.input(fp8_block_spec, fp8, buffer_count=2),
            pltpu.BufferedRef.input(scale_block_spec, jnp.float32, buffer_count=2),
        ]
    # In hier_rs_kernel's parameter order.
    return scratch_shapes
