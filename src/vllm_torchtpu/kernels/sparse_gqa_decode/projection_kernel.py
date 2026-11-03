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

"""Sparse decode with BF16 output-weight prefetch and HC accumulation.

Extends this package's fused_kernel.py pipeline with one weight DMA per
output-column tile. RPA math and schedule loading use the original helpers.
The separate entry retains the original attention-only path for other shapes.
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
from vllm_torchtpu.kernels.sparse_gqa_decode.abi import output_aliases
from vllm_torchtpu.kernels.sparse_gqa_decode.fused_kernel import (
    _gate_output,
    _load_schedule,
)
from vllm_torchtpu.kernels.sparse_gqa_decode.tuned_block_sizes import (
    OUTPUT_PROJECTION_REDUCTION_BLOCKS,
    DecodePolicy,
)


def decode_projected_kernel(
    *,
    cu_q_lens: jax.Array,
    compressed_query_offsets: jax.Array,
    q_lens: jax.Array,
    metadata: schedule.RpaSchedule,
    query: jax.Array,
    unused_new_kv: jax.Array,
    cache: jax.Array,
    gate: jax.Array,
    output_weight: jax.Array,
    residual: jax.Array,
    hc_gate: jax.Array,
    cfgs: configs.RpaConfigs,
    policy: DecodePolicy,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return aliased query/cache and projected output, optionally including HC."""
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
    if output_weight.ndim != 3 or output_weight.dtype != jnp.bfloat16:
        raise ValueError("Projection requires tiled BF16 [N_tiles,K,tile_N] weights")
    output_tiles = output_weight.shape[0]
    output_width = output_weight.shape[-1] * output_tiles
    reduction_key = (cfgs.serve.num_seqs, output_weight.shape[1], output_width)
    if reduction_key not in OUTPUT_PROJECTION_REDUCTION_BLOCKS:
        raise ValueError(f"Unvalidated BF16 projection reduction: {reduction_key}")
    reduction_block = OUTPUT_PROJECTION_REDUCTION_BLOCKS[reduction_key]
    result_width = residual.shape[-1]
    branches = hc_gate.shape[-1]
    if result_width != output_width * branches:
        raise ValueError("HC residual width must equal branches times output width")

    def run(
        cu_ref,
        offsets_ref,
        q_lens_ref,
        metadata_ref,
        query_ref,
        new_kv_ref,
        cache_ref,
        gate_ref,
        weight_ref,
        residual_ref,
        hc_gate_ref,
        out_ref,
        out_cache_ref,
        projected_ref,
        weight_buf,
        attended_buf,
        projection_sems,
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
                kv_refs = (cache_ref, new_kv_ref, schedule_ref)
                kv_bound = kv_alloc.bind(src_ref=kv_refs, dst_ref=kv_refs)

                def compute(
                    step,
                    slot,
                    total_steps,
                    out_slot,
                    start_next_fetch,
                    start_kv_writeback,
                ):
                    for tile in range(output_tiles):

                        @pl.when(step == tile * (total_steps - 1) // output_tiles)
                        def start_weight_tile(tile=tile):
                            pltpu.make_async_copy(
                                weight_ref.at[tile],
                                weight_buf.at[tile],
                                projection_sems.at[tile],
                            ).start()

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

        @pl.when(metadata_ref.actual_steps[0] == 0)
        def load_without_attention():
            for tile in range(output_tiles):
                pltpu.make_async_copy(
                    weight_ref.at[tile], weight_buf.at[tile], projection_sems.at[tile]
                ).start()

        attended_copy = pltpu.make_async_copy(
            out_ref, attended_buf, projection_sems.at[output_tiles]
        )
        attended_copy.start()
        attended_copy.wait()
        attended = (
            attended_buf[...]
            .swapaxes(0, 1)
            .reshape(cfgs.serve.num_seqs, -1)
            .astype(jnp.bfloat16)
        )
        for tile in range(output_tiles):
            pltpu.make_async_copy(
                weight_ref.at[tile], weight_buf.at[tile], projection_sems.at[tile]
            ).wait()
        # Every valid schedule has nonnegative actual_steps, including zero.
        # A runtime all-one mask preserves each FP32 partial-sum boundary;
        # casts alone allow the backend to contract the dot/add expression.
        partial_mask = jnp.where(
            metadata_ref.actual_steps[0] >= 0, jnp.int32(-1), jnp.int32(0)
        )
        for tile in range(output_tiles):
            partials = []
            for block in range(output_weight.shape[1] // reduction_block):
                k_slice = pl.ds(block * reduction_block, reduction_block)
                partial = jax.lax.dot_general(
                    attended[:, k_slice],
                    weight_buf[tile, k_slice, :],
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                partial = jax.lax.bitcast_convert_type(
                    jax.lax.bitcast_convert_type(partial, jnp.int32) & partial_mask,
                    jnp.float32,
                )
                partials.append(partial)
            projected = partials[0]
            for partial in partials[1:]:
                projected = projected + partial
            projected = bf16(projected).astype(jnp.float32)
            for branch in range(branches):
                out_cols = pl.ds(
                    branch * output_width + tile * output_weight.shape[-1],
                    output_weight.shape[-1],
                )
                projected_ref[:, out_cols] = (
                    residual_ref[:, out_cols].astype(jnp.float32)
                    + projected * hc_gate_ref[:, branch : branch + 1]
                ).astype(jnp.bfloat16)

    scalars = (cu_q_lens, compressed_query_offsets, q_lens)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        run,
        out_shape=[
            query,
            cache,
            jax.ShapeDtypeStruct((cfgs.serve.num_seqs, result_width), jnp.bfloat16),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalars),
            in_specs=[
                metadata.in_specs(),
                hbm,
                hbm,
                hbm,
                pl.BlockSpec(gate.shape, lambda *_: (0, 0, 0)),
                hbm,
                pl.BlockSpec(residual.shape, lambda *_: (0, 0)),
                pl.BlockSpec(hc_gate.shape, lambda *_: (0, 0)),
            ],
            out_specs=[
                hbm,
                hbm,
                pl.BlockSpec((cfgs.serve.num_seqs, result_width), lambda *_: (0, 0)),
            ],
            scratch_shapes=(
                pltpu.VMEM(output_weight.shape, output_weight.dtype),
                pltpu.VMEM(query.shape, query.dtype),
                pltpu.SemaphoreType.DMA((output_tiles + 1,)),
            ),
        ),
        input_output_aliases=output_aliases(scalars, metadata),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=cfgs.vmem_limit_bytes,
            disable_bounds_checks=True,
            flags={"XLA_TPU_EVENLY_DISTRIBUTE_CRITICAL_NODES": True},
        ),
        name="qsa_attention_output_prefetch",
        # JAX TPU lowering JSON-serializes scalar metadata despite this annotation.
        # pyrefly: ignore[bad-argument-type]
        metadata=kernel.get_kernel_metadata(cfgs),
    )(
        *scalars,
        metadata,
        query,
        unused_new_kv,
        cache,
        gate,
        output_weight,
        residual,
        hc_gate,
    )
