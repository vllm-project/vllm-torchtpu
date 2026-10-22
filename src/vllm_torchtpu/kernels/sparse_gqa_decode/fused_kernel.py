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

"""Decode-only orchestration around unchanged RPA compute and buffering.

Each launch fits in one SMEM schedule window and the derived VMEM budget.
There is no prefill, LSE output, cache update, or multi-window scheduling here.
The decode specialization originated in tokamax/experiments/sparse_gqa_decode_v2.
This adapter directly composes the current upstream RPA manual pipeline; its
compute body, buffering and DMA implementation remain unchanged.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    kernel,
    pipeline,
    schedule,
)
from vllm_torchtpu.kernels.qwen4_exp.rounding import bf16

from .abi import output_aliases, schedule_copy_size
from .tuned_block_sizes import DecodePolicy


def _load_schedule(source, target, semaphore, steps, cfgs, policy):
    """Copy a zero-start live prefix; start/wait share the exact same slice."""
    copies = []
    for src, dst in zip(jax.tree.leaves(source), jax.tree.leaves(target)):
        if jax.typeof(src).memory_space == pltpu.HBM:
            words_per_step = dst.size // cfgs.max_steps_ub
            size = schedule_copy_size(
                words_per_step * steps, dst.size, policy.schedule_copy_words
            )
            copy = pltpu.make_async_copy(
                src.at[pl.ds(0, size)], dst.at[pl.ds(0, size)], semaphore.at[0]
            )
            copy.start()
            copies.append(copy)
    for copy in copies:
        copy.wait()


def _gate_output(output, gate, metadata, step, cfgs):
    """Apply QSA's rounding boundaries only after a request's final KV tile."""
    for lane in range(cfgs.batch_size):

        @pl.when(metadata.is_last_k[step, lane] == 1)
        def apply_gate(lane=lane):
            seq = metadata.s_idx[step, lane]
            g = (
                gate[seq]
                .astype(jnp.float32)
                .reshape(
                    cfgs.model.num_kv_heads,
                    1,
                    cfgs.aligned_num_q_heads_per_kv_head,
                    1,
                    cfgs.aligned_q_head_dim,
                )
            )
            out = output[lane]
            gated = bf16(
                bf16(out).astype(jnp.float32)
                * bf16(jax.nn.sigmoid(g)).astype(jnp.float32)
            )
            output[lane] = gated.astype(jnp.float32)


def decode_kernel(
    *,
    cu_q_lens: jax.Array,
    compressed_query_offsets: jax.Array,
    q_lens: jax.Array,
    pages: jax.Array,
    metadata: schedule.RpaSchedule,
    query: jax.Array,
    unused_new_kv: jax.Array,
    cache: jax.Array,
    gate: jax.Array,
    cfgs: configs.RpaConfigs,
    policy: DecodePolicy,
) -> tuple[jax.Array, jax.Array]:
    """Run one bounded, read-only decode schedule; output aliases query/cache."""
    if (
        cfgs.mode != configs.RpaCase.DECODE
        or cfgs.serve.return_lse
        or not cfgs.serve.skip_kv_update
        or cfgs.bq_sz != 1
    ):
        raise ValueError("QSA decode requires bq=1, no LSE, and no cache update")
    max_tiles = pl.cdiv(cfgs.serve.pages_per_seq * cfgs.serve.page_size, cfgs.bkv_sz)
    if pl.cdiv(cfgs.serve.num_seqs * max_tiles, cfgs.batch_size) > cfgs.max_steps_ub:
        raise ValueError("Split decode requests before exceeding one schedule window")

    def run(
        cu_ref,
        offsets_ref,
        q_lens_ref,
        pages_ref,
        metadata_ref,
        query_ref,
        new_kv_ref,
        cache_ref,
        gate_ref,
        out_ref,
        out_cache_ref,
    ):
        del out_cache_ref  # The aliased cache is read-only for this entry.
        allocs = kernel.create_allocs(cache_ref, query_ref, None, cfgs)

        @pl.with_scoped(
            allocs=allocs,
            schedule_ref=schedule.BaseMetadataComputer.get_rpa_schedule(
                cfgs
            ).scratch_shapes(),
            semaphore=pltpu.SemaphoreType.DMA((1,)),
            scratches=(
                pltpu.VMEM(cfgs.lm_scratch_shape, cfgs.serve.dtype_out),
                pltpu.VMEM(cfgs.lm_scratch_shape, cfgs.serve.dtype_out),
                pltpu.VMEM(cfgs.acc_scratch_shape, cfgs.serve.dtype_out),
            ),
        )
        def execute(allocs, schedule_ref, semaphore, scratches):
            q_alloc, kv_alloc, o_alloc, _ = allocs
            scratches[0][...] = jnp.full_like(scratches[0], -jnp.inf)
            lanes = pltpu.get_tpu_info().num_lanes
            # Padded Q/KV lanes must start at zero: masked multiplication by
            # residual NaNs would otherwise contaminate the softmax accumulator.
            for ref in (
                scratches[1],
                scratches[2],
                q_alloc.window_ref.bitcast(jnp.uint32).reshape(-1, lanes),
                kv_alloc.window_ref.bitcast(jnp.uint32).reshape(-1, lanes),
            ):
                ref[...] = jnp.zeros_like(ref)

            steps = metadata_ref.actual_steps[0]

            @pl.when(steps > 0)
            def attend():
                _load_schedule(
                    metadata_ref, schedule_ref, semaphore, steps, cfgs, policy
                )

                q_bound = q_alloc.bind(src_ref=(query_ref, schedule_ref))
                o_bound = o_alloc.bind(dst_ref=(out_ref, schedule_ref))
                kv_refs = (cache_ref, new_kv_ref, schedule_ref, pages_ref)
                kv_bound = kv_alloc.bind(src_ref=kv_refs, dst_ref=kv_refs)

                def compute(
                    step,
                    slot,
                    total_steps,
                    out_slot,
                    start_next_fetch,
                    start_kv_writeback,
                ):
                    del total_steps
                    output = o_alloc.window_ref.at[out_slot]
                    kernel.rpa_body(
                        step,
                        slot,
                        q_alloc.window_ref.at[slot],
                        kv_alloc.window_ref.at[slot],
                        output,
                        None,
                        schedule_ref,
                        *scratches,
                        cu_q_lens_ref=cu_ref,
                        q_offsets_ref=offsets_ref,
                        kv_cache_lens_ref=offsets_ref,
                        kv_new_lens_ref=q_lens_ref,
                        cfgs=cfgs,
                        start_next_fetch=start_next_fetch,
                        start_kv_writeback=start_kv_writeback,
                    )
                    # LAST_TILE owns final rounding/gating before the pipeline
                    # issues output DMA. Scratch carry is reset inside RPA body.
                    _gate_output(output, gate_ref, schedule_ref, step, cfgs)

                # Three input slots prefetch N+2; two output slots wait N-2.
                # Keep that scheduling visible, and keep gating before O DMA.
                pipeline.run_manual_triple_buffer(
                    steps, compute, q_bound, o_bound, kv_bound
                )

        execute()

    scalars = (cu_q_lens, compressed_query_offsets, q_lens, pages)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        run,
        out_shape=[query, cache],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            in_specs=[
                metadata.in_specs(),
                hbm,
                hbm,
                hbm,
                pl.BlockSpec(gate.shape, lambda *_: (0, 0, 0)),
            ],
            out_specs=[hbm, hbm],
        ),
        input_output_aliases=output_aliases(scalars, metadata),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=cfgs.vmem_limit_bytes,
            disable_bounds_checks=True,
            flags={"XLA_TPU_EVENLY_DISTRIBUTE_CRITICAL_NODES": True},
        ),
        name="fused_qsa_decode_rpa_pipeline",
        # JAX TPU lowering JSON-serializes scalar metadata despite this annotation.
        # pyrefly: ignore[bad-argument-type]
        metadata=kernel.get_kernel_metadata(cfgs),
    )(*scalars, metadata, query, unused_new_kv, cache, gate)
