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
"""Public dispatch wrapper and metadata builder for Batched Ragged Paged Attention.

Validates tensor shapes, selects tuned block sizes, builds DMA schedule tables,
and invokes the Pallas Batched RPA kernel.
"""

import math
import os
import re
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels import static_guards
from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    kernel,
    schedule,
    schedule_cp,
    utils,
)


def hoist_schedule_enabled() -> bool:
    """Whether cross-layer hoisting of `rpa_metadata_schedule_kernel` is enabled."""
    return os.environ.get("TPU_RPA_HOIST_SCHEDULE", "1") not in (
        "0",
        "false",
        "False",
    )


def prepare_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    q_dtype: jnp.dtype,
    kv_dtype: jnp.dtype,
    k_scale: jax.Array | None = None,
    v_scale: jax.Array | None = None,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    page_size: int = 128,
) -> tuple[jax.Array, jax.Array]:
    total_q_tokens, actual_num_q_heads, actual_q_head_dim = q.shape
    _, actual_num_kv_heads, actual_kv_head_dim = k.shape
    num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    q_packing = utils.get_dtype_packing(q_dtype)
    kv_packing = utils.get_dtype_packing(kv_dtype)

    aligned_num_q_heads_per_kv_head = utils.align_to(num_q_heads_per_kv_head, q_packing)
    num_lanes = pltpu.get_tpu_info().num_lanes
    num_sublanes = pltpu.get_tpu_info().num_sublanes
    aligned_q_head_dim = utils.align_to(actual_q_head_dim, num_lanes)

    # Compute aligned kv head dimension, accounting for per-token scale if
    # present.
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        if k_scale is not None and v_scale is not None:
            scale_bits = jax.dtypes.itemsize_bits(k_scale.dtype)
            kv_bits = jax.dtypes.itemsize_bits(kv_dtype)
            num_scale_channels = max(1, scale_bits // kv_bits)
            aligned_kv_head_dim = utils.align_to(
                actual_kv_head_dim + num_scale_channels, num_sublanes * kv_packing
            )
        else:
            aligned_kv_head_dim = utils.align_to(
                actual_kv_head_dim, num_sublanes * kv_packing
            )
    else:
        aligned_kv_head_dim = utils.align_to(actual_kv_head_dim, num_lanes)

    # queries: (T, H, D) -> (T, H_kv, G, D)
    o_hbm_alias_q_hbm = (
        jnp.pad(
            q.reshape(
                total_q_tokens,
                actual_num_kv_heads,
                num_q_heads_per_kv_head,
                actual_q_head_dim,
            ),
            (
                (0, 0),
                (0, 0),
                (0, aligned_num_q_heads_per_kv_head - num_q_heads_per_kv_head),
                (0, aligned_q_head_dim - actual_q_head_dim),
            ),
            constant_values=0,
        )
        .reshape(
            total_q_tokens,
            actual_num_kv_heads,
            aligned_num_q_heads_per_kv_head // q_packing,
            q_packing,
            aligned_q_head_dim,
        )
        .swapaxes(0, 1)
    )

    # Pad keys and values head_dim
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2_aligned = utils.align_to(actual_num_kv_heads_x2, kv_packing)

    if k_scale is not None and v_scale is not None:
        k_scale = k_scale.reshape(total_q_tokens, actual_num_kv_heads)
        v_scale = v_scale.reshape(total_q_tokens, actual_num_kv_heads)

        scale_bits = jax.dtypes.itemsize_bits(k_scale.dtype)
        kv_bits = jax.dtypes.itemsize_bits(k.dtype)
        num_scale_channels = max(1, scale_bits // kv_bits)

        # Bitcast the scale factors to the kv dtype and reshape scale factors to
        # (T, H_kv, num_scale_channels).
        k_scale_split = jax.lax.bitcast_convert_type(k_scale, k.dtype).reshape(
            total_q_tokens, actual_num_kv_heads, num_scale_channels
        )
        v_scale_split = jax.lax.bitcast_convert_type(v_scale, v.dtype).reshape(
            total_q_tokens, actual_num_kv_heads, num_scale_channels
        )

        # k/v have shape (T, H_kv, D) -> (T, H_kv, D + num_scale_channels)
        k = jnp.concatenate([k, k_scale_split], axis=-1)
        v = jnp.concatenate([v, v_scale_split], axis=-1)

        actual_kv_head_dim += num_scale_channels

    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        num_lanes = pltpu.get_tpu_info().num_lanes
        align_tokens = max(num_lanes, page_size)
        padded_total_tokens = utils.align_to(total_q_tokens, align_tokens)
        new_kv_hbm = (
            jnp.pad(
                jnp.concatenate([k, v], axis=-1).reshape(
                    total_q_tokens, actual_num_kv_heads_x2, actual_kv_head_dim
                ),
                (
                    (0, padded_total_tokens - total_q_tokens),
                    (0, 0),
                    (0, aligned_kv_head_dim - actual_kv_head_dim),
                ),
                constant_values=0,
            )
            .reshape(
                padded_total_tokens,
                actual_num_kv_heads_x2,
                aligned_kv_head_dim // kv_packing,
                kv_packing,
            )
            .transpose(1, 2, 3, 0)
        )
    else:
        new_kv_hbm = jnp.pad(
            jnp.concatenate([k, v], axis=-1).reshape(
                total_q_tokens, actual_num_kv_heads_x2, actual_kv_head_dim
            ),
            (
                (0, 0),
                (0, num_kv_heads_x2_aligned - actual_num_kv_heads_x2),
                (0, aligned_kv_head_dim - actual_kv_head_dim),
            ),
            constant_values=0,
        ).reshape(
            total_q_tokens,
            num_kv_heads_x2_aligned // kv_packing,
            kv_packing,
            aligned_kv_head_dim,
        )
    return o_hbm_alias_q_hbm, new_kv_hbm


def prepare_outputs(out: jax.Array) -> jax.Array:
    kv_heads, max_tokens, q_per_kv_packed, q_packing, d = out.shape
    return out.reshape(kv_heads, max_tokens, q_per_kv_packed * q_packing, d)


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    chip_version: pltpu.ChipVersion = pltpu.ChipVersion.TPU_7X,
    use_per_token_scale: bool = False,
    scale_dtype: jnp.dtype | None = None,
):
    chip_info = pltpu.get_tpu_info_for_chip(chip_version, 1)
    num_lanes, num_sublanes = chip_info.num_lanes, chip_info.num_sublanes
    kv_packing = utils.get_dtype_packing(kv_dtype)
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        num_scale_channels = 0
        if use_per_token_scale:
            assert scale_dtype is not None
            kv_bits = jax.dtypes.itemsize_bits(kv_dtype)
            scale_bits = jax.dtypes.itemsize_bits(scale_dtype)
            num_scale_channels = max(1, scale_bits // kv_bits)
        base_dim = actual_head_dim + num_scale_channels
        return (
            total_num_pages,
            actual_num_kv_heads * 2,
            utils.align_to(base_dim, num_sublanes * kv_packing) // kv_packing,
            kv_packing,
            page_size,
        )
    return (
        total_num_pages,
        page_size,
        utils.align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        utils.align_to(actual_head_dim, num_lanes),
    )


def _is_sfs(tpu_info) -> bool:
    """Returns True on Sunfish (TPU v8i).

    Accepts an enum or a string, since the unit tests mock chip_version with a
    plain string and a strict `==` against the enum would never match.
    """
    return tpu_info.generation == 8 and tpu_info.chip_version in (
        pltpu.ChipVersion.TPU_8I,
        "v8i",
        "8i",
    )


def calculate_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
    decode_query_size: int = 1,
    decode_batch_size: int = 8,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
    """Calculate optimal block size for decode and prefill.

    ``decode_batch_size`` is the most sequences one decode step works on
    concurrently; 8 is the experimentally chosen default.
    """

    tpu_info = configs._get_tpu_info()
    num_lanes = tpu_info.num_lanes
    mxu_column_size = tpu_info.mxu_column_size

    # Calculate aligned model dimensions.
    aligned_head_dim = utils.align_to(model_cfgs.head_dim, num_lanes)
    aligned_num_q_heads_per_kv_head = utils.align_to(
        model_cfgs.num_q_heads_per_kv_head, serve_cfgs.packing_q
    )
    aligned_num_q_heads = aligned_num_q_heads_per_kv_head * model_cfgs.num_kv_heads

    if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        aligned_num_kv_heads_x2 = model_cfgs.num_kv_heads * 2
    else:
        bkv_stride = pl.cdiv(model_cfgs.num_kv_heads * 2, serve_cfgs.packing_kv)
        if utils.has_bank_conflicts(bkv_stride):
            bkv_stride += 1
        aligned_num_kv_heads_x2 = bkv_stride * serve_cfgs.packing_kv

    q_bytes = jax.dtypes.itemsize_bits(serve_cfgs.dtype_q) / 8
    kv_bytes = jax.dtypes.itemsize_bits(serve_cfgs.dtype_kv) / 8
    out_bytes = jax.dtypes.itemsize_bits(serve_cfgs.dtype_out) / 8

    is_packed_fp4 = serve_cfgs.dtype_kv == jnp.uint8
    physical_kv_head_dim = model_cfgs.head_dim
    if is_packed_fp4:
        physical_kv_head_dim = physical_kv_head_dim // 2

    num_sublanes = tpu_info.num_sublanes
    packing_kv = serve_cfgs.packing_kv

    if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        num_scale_channels = serve_cfgs.scale_channels
        base_dim = physical_kv_head_dim + num_scale_channels
        aligned_kv_head_dim_val = utils.align_to(base_dim, num_sublanes * packing_kv)
    else:
        aligned_kv_head_dim_val = utils.align_to(model_cfgs.head_dim, num_lanes)

    def calculate_vmem_usage(
        batch_size: int,
        bq_sz: int,
        bkv_sz: int,
        is_decode: bool = False,
    ) -> int:
        """Given tile size, calculate VMEM usage of the kernel."""

        # Step 1: Calculate buffer sizes.

        # Calculate size bq & bkv arrays for a single buffer.
        bq_array_size = bq_sz * aligned_num_q_heads * aligned_head_dim
        vmem_bkv_sz = bkv_sz
        if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            # Allocate extra buffer to perform lane stitching. Must match
            # RpaConfigs.kv_vmem_lanes.
            vmem_bkv_sz += 2 * num_lanes
        bkv_array_size = vmem_bkv_sz * aligned_num_kv_heads_x2 * aligned_kv_head_dim_val

        # Get output buffer size as well - which has same size as query size.
        bo_array_size = bq_array_size

        # Convert to bytes.
        bq_bytes = bq_array_size * q_bytes
        bkv_bytes = bkv_array_size * kv_bytes
        bo_bytes = bo_array_size * out_bytes

        # Account for multiple buffers. These counts are fixed by the pipeline; see
        # `configs.NUM_INPUT_SLOTS` / `configs.NUM_OUTPUT_SLOTS`.
        bq_bytes *= configs.NUM_INPUT_SLOTS
        bkv_bytes *= configs.NUM_INPUT_SLOTS
        bo_bytes *= configs.NUM_OUTPUT_SLOTS

        # Sum up all buffer memory usage (scales are co-located inside bkv_bytes).
        buffer_bytes = bq_bytes + bkv_bytes + bo_bytes

        if serve_cfgs.return_lse:
            lse_array_size = bq_sz * aligned_num_q_heads * num_lanes
            lse_bytes = lse_array_size * out_bytes * 2
            buffer_bytes += lse_bytes

        # Step 2: Calculate worst case memory usage during computation.

        # Calculate the size of loaded bq and bkv size.
        loaded_bq_size = bq_sz * model_cfgs.num_q_heads * aligned_head_dim
        loaded_bkv_size = bkv_sz * model_cfgs.num_kv_heads * aligned_head_dim

        # Calculate peak memory requirement of otuput - which is attention weight.
        qk_size = bq_sz * bkv_sz * model_cfgs.num_q_heads

        # Convert to bytes.
        loaded_bq_bytes = loaded_bq_size * q_bytes
        compute_kv_bytes = kv_bytes
        loaded_bkv_bytes = loaded_bkv_size * compute_kv_bytes
        qk_bytes = qk_size * out_bytes

        # Calculate VMEM contribution of active unpacked scale arrays during
        # compute. Note: since flash_attention consumes quantized k/v directly
        # along with our active k_scale and v_scale tensors (no upfront dequantized
        # intermediate copy), we only need to account for unpacked k_scale + v_scale
        # during compute.
        loaded_scale_bytes = 0
        if serve_cfgs.per_token_scale:
            assert serve_cfgs.per_token_scale_dtype is not None
            scale_bits = jax.dtypes.itemsize_bits(serve_cfgs.per_token_scale_dtype)
            scale_bytes = scale_bits // 8
            # Active k_scale and v_scale each have shape (num_kv_heads, bkv_sz)
            loaded_scale_bytes = 2 * bkv_sz * model_cfgs.num_kv_heads * scale_bytes

        # Sum up all compute memory usage.
        compute_bytes = (
            loaded_bq_bytes + loaded_bkv_bytes + qk_bytes + loaded_scale_bytes
        )

        # Step 3: Sum up all memory usage.
        if is_decode:
            # Pipeline buffers are allocated per batch item, whereas active
            # compute tiles are live for one batch item at a time.
            total_bytes = buffer_bytes * batch_size + compute_bytes
            bq_c_sz = pick_bq_c_sz(batch_size, bq_sz, bkv_sz)
            if bq_c_sz < bq_sz:
                # Every q sub-chunk's QK and PV matmuls read the K and V tiles
                # of all lanes, stacked, so Mosaic keeps them in register
                # allocator spill slots across the sub-chunk loop, with one
                # sub-chunk's f32 scores and its two softmax weight tiles.
                stacked_kv_bytes = 2 * loaded_bkv_bytes
                sub_chunk_bytes = (
                    bq_c_sz * model_cfgs.num_q_heads * bkv_sz * (4 + 2 * out_bytes)
                )
                total_bytes += batch_size * (stacked_kv_bytes + sub_chunk_bytes)
        else:
            # Account for batch size.
            total_bytes = (buffer_bytes + compute_bytes) * batch_size

        # Online-softmax scratch (m, l, and the output accumulator)
        lm_array_size = (
            model_cfgs.num_kv_heads * bq_sz * aligned_num_q_heads_per_kv_head
        )
        acc_array_size = (
            model_cfgs.num_kv_heads * bq_sz * aligned_num_q_heads_per_kv_head
        )
        # m and l scratches share the same shape.
        scratch_bytes = (2 * lm_array_size + acc_array_size) * out_bytes
        total_bytes += scratch_bytes

        return total_bytes

    def calculate_compute_buffer_time(
        batch_size: int, bq_c_sz: int, bkv_sz: int
    ) -> int:
        """Calculate computational complexity of a single compute block."""

        num_k_rows = pl.cdiv(bkv_sz, mxu_column_size)
        num_k_cols = pl.cdiv(model_cfgs.head_dim, mxu_column_size)
        num_k = num_k_rows * num_k_cols
        num_muls = bq_c_sz * num_k * model_cfgs.num_q_heads

        return batch_size * num_muls

    # Threshold value based on hardware generation and precision.
    is_8bit = jnp.dtype(serve_cfgs.dtype_q).itemsize == 1 or (
        kv_bytes == 1 and not serve_cfgs.return_lse
    )

    match tpu_info.generation:
        case 8 if _is_sfs(tpu_info):
            threshold = 800
        case 7:
            threshold = 1500
        case _:
            threshold = 1500

    if is_8bit:
        flops_ratio = round(tpu_info.fp8_ops_per_second / tpu_info.bf16_ops_per_second)
        threshold *= flops_ratio

    def pick_bq_c_sz(batch_size: int, bq_sz: int, bkv_sz: int) -> int:
        """Picks the q sub-chunk size the compute threshold allows for a tile."""
        num_bq_c = 1
        last_valid_bq_c_sz = bq_c_sz = bq_sz
        bq_c_rem = 0

        while (
            calculate_compute_buffer_time(batch_size, bq_c_sz, bkv_sz) > threshold
            or bq_c_rem != 0
        ) and num_bq_c < bq_sz:
            if bq_c_rem == 0:
                last_valid_bq_c_sz = bq_c_sz
            num_bq_c += 1
            bq_c_sz, bq_c_rem = divmod(bq_sz, num_bq_c)
        return last_valid_bq_c_sz

    def find_best_block_sizes(
        max_batch_size: int, fixed_bq_sz: int | None = None
    ) -> configs.BlockSizes:
        """Loop through different block sizes to find the most optimal one."""

        is_decode = fixed_bq_sz is not None
        # Even if we loose some potential performance, we want to avoid OOM at all
        # costs. For decode, active compute tiles are not replicated across
        # batch_size, allowing a 0.85 VMEM cap so bkv_sz reaches 4096 (32 pages).
        decode_vmem_frac = 0.85 if kv_bytes == 1 else 0.75
        capped_vmem_limit_bytes = vmem_limit_bytes * (
            decode_vmem_frac if is_decode else 0.8
        )

        # To maximize MXU, we want bkv_sz to be at least MXU size.
        bkv_default = mxu_column_size
        if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            # SEQ-ALONG-LANE does not support intra-page DMA offset.
            # TODO(kyuyeunk): Reduce granularity to lane size.
            bkv_default = utils.align_to(bkv_default, serve_cfgs.page_size)
        bkv_sz = bkv_stride = bkv_default
        if fixed_bq_sz is None:
            bq_sz = bq_stride = bkv_sz
        else:
            bq_sz = fixed_bq_sz
            bq_stride = 0
        batch_size = max_batch_size

        # Step 1: Lower batch_size if even the smallest bq and bkv size can trigger
        # OOM.

        # If current batch size triggers OOM, decrease batch size until the kernel
        # fits within VMEM limit.
        while (
            batch_size > 1
            and calculate_vmem_usage(batch_size, bq_sz, bkv_sz, is_decode)
            > capped_vmem_limit_bytes
        ):
            batch_size -= 1

        # Indicates OOM was triggered even when batch_size=1.
        # NOTE: If the function does not exit at this point even when the value is
        # zero, it will trigger infinite loop at the next while loop.
        if batch_size == 0:
            raise ValueError("Cannot find batch size that fits within VMEM limit.")

        # Step 2: Increase block sizes until the kernel is unable to fit into VMEM.
        max_seq_len = serve_cfgs.pages_per_seq * serve_cfgs.page_size
        while (
            calculate_vmem_usage(batch_size, bq_sz, bkv_sz, is_decode)
            < capped_vmem_limit_bytes
            and bkv_sz <= max_seq_len
            and bkv_sz <= 4096  # TEMP PATCH
        ):
            # Unless bq is a fixed value, we want to ensure bq size is the same as bkv
            # size. When using causal masking, if bq size is larger than bkv size,
            # entire kv tile can be masked out for some query tokens. Similarly, if
            # bkv size is larger than bq size, entire query tile can be masked out for
            # some kv tokens.
            bkv_sz += bkv_stride
            bq_sz += bq_stride

        # Rollback one step since the last attempted value triggered OOM.
        if bkv_sz > bkv_stride:
            bkv_sz -= bkv_stride
            if fixed_bq_sz is None:
                bq_sz -= bq_stride

        # Indicates OOM was triggered from the starting bkv size.
        if bkv_sz == 0:
            raise ValueError("Cannot find block sizes that fit within VMEM limit.")

        if fixed_bq_sz is None:
            # Clamp prefill/mixed bq_sz to the aligned total_q_tokens bucket so
            # small decode buckets do not allocate 1024-token Q/O VMEM buffers.
            max_useful_bq_sz = max(
                mxu_column_size,
                utils.align_to(serve_cfgs.total_q_tokens, mxu_column_size),
            )
            bq_sz = min(bq_sz, max_useful_bq_sz)

        # Step 3: Given current tile size, calculate compute tile size.
        return configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=pick_bq_c_sz(batch_size, bq_sz, bkv_sz),
            bkv_sz=bkv_sz,
            batch_size=batch_size,
        )

    prefill_batch_size = 2

    decode_block_sizes = decode_block_override(
        find_best_block_sizes(decode_batch_size, decode_query_size), serve_cfgs
    )
    prefill_block_sizes = prefill_block_override(
        find_best_block_sizes(prefill_batch_size), serve_cfgs, mxu_column_size
    )

    return decode_block_sizes, prefill_block_sizes


def decode_block_override(
    default: configs.BlockSizes, serve_cfgs: configs.ServingConfigs
) -> configs.BlockSizes:
    """Applies TPU_RPA_DECODE_BLOCKS ("bkv,batch[,bq_c]") to the decode tiles.

    bkv is the KV block (a multiple of the page size), batch the decode lanes
    and bq_c the query sub-chunk, which must divide the query block (default:
    the heuristic's). Every caller of calculate_block_sizes sees the same
    tiles. Empty: the heuristic.
    """
    raw = os.environ.get("TPU_RPA_DECODE_BLOCKS", "").strip()
    if not raw:
        return default
    try:
        parts = [int(x) for x in raw.split(",")]
        bkv, batch = parts[0], parts[1]
        bq_c = parts[2] if len(parts) > 2 else default.bq_c_sz
    except (ValueError, IndexError) as e:
        raise ValueError(
            f"TPU_RPA_DECODE_BLOCKS must be 'bkv,batch[,bq_c]'; got {raw!r}"
        ) from e
    if min(bkv, batch, bq_c) <= 0 or bkv % serve_cfgs.page_size or default.bq_sz % bq_c:
        raise ValueError(
            f"TPU_RPA_DECODE_BLOCKS={raw!r}: sizes must be positive, bkv a "
            f"multiple of the page size {serve_cfgs.page_size} and bq_c a "
            f"divisor of the query block {default.bq_sz}"
        )
    return configs.BlockSizes(
        bq_sz=default.bq_sz, bq_c_sz=bq_c, bkv_sz=bkv, batch_size=batch
    )


def prefill_block_override(
    default: configs.BlockSizes,
    serve_cfgs: configs.ServingConfigs,
    mxu_column_size: int,
) -> configs.BlockSizes:
    """Applies TPU_RPA_PREFILL_BLOCKS ("bq,bq_c,bkv,batch") to the prefill tiles.

    The heuristic grows bq and bkv together, so with many q heads per KV head
    (32 q / 2 KV heads on a PP=8 stage) VMEM stops both at 256 and every
    256-key block pays a full accumulator rescale. A 1024-key block with a
    small q sub-chunk (256,32,1024,1) fits and runs the prefill kernel ~20%
    faster at 87K context (rpa_prefill_bench.py, j-10897a30). Every caller of
    calculate_block_sizes (the kernel, build_rpa_schedules and the schedule
    shapes) sees the same tiles. bq is clamped to the aligned token count as
    the heuristic does, and bq_c to a divisor of bq. Empty: the heuristic.
    """
    raw = (envs.TPU_RPA_PREFILL_BLOCKS or "").strip()
    if not raw:
        return default
    try:
        bq, bq_c, bkv, batch = (int(x) for x in raw.split(","))
    except ValueError as e:
        raise ValueError(
            f"TPU_RPA_PREFILL_BLOCKS must be 'bq,bq_c,bkv,batch'; got {raw!r}"
        ) from e
    if min(bq, bq_c, bkv, batch) <= 0 or bkv % serve_cfgs.page_size:
        raise ValueError(
            f"TPU_RPA_PREFILL_BLOCKS={raw!r}: sizes must be positive and bkv a "
            f"multiple of the page size {serve_cfgs.page_size}"
        )
    max_useful_bq_sz = max(
        mxu_column_size, utils.align_to(serve_cfgs.total_q_tokens, mxu_column_size)
    )
    bq = min(bq, max_useful_bq_sz)
    bq_c = math.gcd(bq, min(bq_c, bq))
    return configs.BlockSizes(bq_sz=bq, bq_c_sz=bq_c, bkv_sz=bkv, batch_size=batch)


def get_rpa_schedule_shapes(
    total_q_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_num_seqs: int,
    num_page_indices: int,
    dtype_q: jnp.dtype,
    dtype_kv: jnp.dtype,
    *,
    sliding_window: int | None = None,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    decode_query_size: int = 1,
    decode_batch_size: int = 8,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Return the 14 1D int32 array shapes for `(decode_sched, mixed_sched)`."""
    if vmem_limit_bytes is None:
        vmem_limit_bytes = configs._get_tpu_info().vmem_capacity_bytes
    model_cfgs = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        mask_value=float(jnp.finfo(dtype_q).min),
    )
    serve_cfgs = configs.ServingConfigs(
        num_seqs=max_num_seqs,
        num_page_indices=num_page_indices,
        total_q_tokens=total_q_tokens,
        dtype_q=dtype_q,
        dtype_kv=dtype_kv,
        dtype_out=dtype_q,
        page_size=page_size,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
    )
    default_decode, default_prefill = calculate_block_sizes(
        model_cfgs,
        serve_cfgs,
        vmem_limit_bytes,
        decode_query_size=decode_query_size,
        decode_batch_size=decode_batch_size,
    )
    effective_decode = decode_block_sizes or default_decode
    effective_prefill = prefill_block_sizes or default_prefill

    def _leaf_shapes(mode: configs.RpaCase, blk: configs.BlockSizes):
        cfgs = configs.RpaConfigs(
            block=blk,
            model=model_cfgs,
            serve=serve_cfgs,
            vmem_limit_bytes=vmem_limit_bytes,
            mode=mode,
        )
        sched_template = schedule.BaseMetadataComputer.get_rpa_schedule(
            cfgs, multiplier=cfgs.max_schedule_size_multiplier
        )
        return tuple(
            tuple(leaf.shape) for leaf in jax.tree_util.tree_leaves(sched_template)
        )

    decode_shapes = _leaf_shapes(configs.RpaCase.DECODE, effective_decode)
    mixed_shapes = _leaf_shapes(configs.RpaCase.MIXED, effective_prefill)
    return decode_shapes, mixed_shapes


@jax.jit(
    static_argnames=(
        "total_q_tokens",
        "num_q_heads",
        "num_kv_heads",
        "head_dim",
        "page_size",
        "num_page_indices",
        "dtype_q",
        "dtype_kv",
        "sliding_window",
        "kv_layout",
        "decode_query_size",
        "decode_batch_size",
        "decode_block_sizes",
        "prefill_block_sizes",
        "vmem_limit_bytes",
    ),
)
def build_rpa_schedules(
    kv_lens: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    total_q_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    num_page_indices: int,
    dtype_q: jnp.dtype,
    dtype_kv: jnp.dtype,
    sliding_window: int | None = None,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    decode_query_size: int = 1,
    decode_batch_size: int = 8,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
    """Precompute `(decode_schedule_leaves, mixed_schedule_leaves)` once per step."""
    if vmem_limit_bytes is None:
        vmem_limit_bytes = configs._get_tpu_info().vmem_capacity_bytes
    max_num_seqs = kv_lens.shape[0]
    model_cfgs = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        mask_value=float(jnp.finfo(dtype_q).min),
    )
    serve_cfgs = configs.ServingConfigs(
        num_seqs=max_num_seqs,
        num_page_indices=num_page_indices,
        total_q_tokens=total_q_tokens,
        dtype_q=dtype_q,
        dtype_kv=dtype_kv,
        dtype_out=dtype_q,
        page_size=page_size,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
    )
    default_decode, default_prefill = calculate_block_sizes(
        model_cfgs,
        serve_cfgs,
        vmem_limit_bytes,
        decode_query_size=decode_query_size,
        decode_batch_size=decode_batch_size,
    )
    effective_decode = decode_block_sizes or default_decode
    effective_prefill = prefill_block_sizes or default_prefill

    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    kv_cache_lens = kv_lens - q_lens
    kv_new_lens = q_lens
    q_offsets = kv_cache_lens

    def _gen_leaves(
        mode: configs.RpaCase,
        blk: configs.BlockSizes,
        dist: jax.Array,
    ) -> tuple[jax.Array, ...]:
        cfgs = configs.RpaConfigs(
            block=blk,
            model=model_cfgs,
            serve=serve_cfgs,
            vmem_limit_bytes=vmem_limit_bytes,
            mode=mode,
        )
        sched = schedule.generate_rpa_metadata(
            cu_q_lens,
            q_offsets,
            kv_cache_lens,
            kv_new_lens,
            dist,
            cfgs=cfgs,
            computer_cls=schedule.BaseMetadataComputer,
        )
        return tuple(jax.tree_util.tree_leaves(sched))

    if total_q_tokens <= min(decode_query_size, effective_decode.bq_sz):
        decode_dist = jnp.repeat(distribution[2:3], 3)
        decode_leaves = _gen_leaves(
            configs.RpaCase.DECODE, effective_decode, decode_dist
        )
        mixed_cfgs = configs.RpaConfigs(
            block=effective_prefill,
            model=model_cfgs,
            serve=serve_cfgs,
            vmem_limit_bytes=vmem_limit_bytes,
            mode=configs.RpaCase.MIXED,
        )
        mixed_template = schedule.BaseMetadataComputer.get_rpa_schedule(
            mixed_cfgs, multiplier=mixed_cfgs.max_schedule_size_multiplier
        )
        mixed_leaves = tuple(
            jnp.zeros(leaf.shape, dtype=leaf.dtype)
            for leaf in jax.tree_util.tree_leaves(mixed_template)
        )
        return decode_leaves, mixed_leaves

    decode_leaves = _gen_leaves(configs.RpaCase.DECODE, effective_decode, distribution)
    mixed_leaves = _gen_leaves(configs.RpaCase.MIXED, effective_prefill, distribution)
    return decode_leaves, mixed_leaves


@jax.jit(
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "use_per_token_scale",
        "per_token_scale_dtype",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "decode_block_sizes",
        "prefill_block_sizes",
        "vmem_limit_bytes",
        "debug_mode",
        "out_dtype",
        "use_causal_mask",
        "skip_kv_update",
        "update_kv_cache",
        "kv_layout",
        "decode_query_size",
        "decode_batch_size",
        "cp_group_size",
        "attention_scope",
        "return_lse",
        "native_fp8",
        "native_fp8_prefill",
    ),
    # Donation of transient inputs can fail for some runtime buffer layouts in
    # the experimental tuning path. Keep donation only for kv_cache, which is
    # the intended long-lived mutable state.
    donate_argnames=("kv_cache",),
)
def ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = None,
    use_per_token_scale: bool = False,
    per_token_scale_dtype: jnp.dtype | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    dynamic_k_scale: jax.Array | None = None,
    dynamic_v_scale: jax.Array | None = None,
    chunk_prefill_size: int | None = None,
    decode_block_sizes: configs.BlockSizes | None = None,
    prefill_block_sizes: configs.BlockSizes | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
    out_dtype: jnp.dtype | None = None,
    use_causal_mask: bool = True,
    skip_kv_update: bool = False,
    update_kv_cache: bool | None = None,
    kv_layout: configs.KVLayout = configs.KVLayout.HEAD_ALONG_SUBLANE,
    decode_query_size: int = 1,
    decode_batch_size: int = 8,
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: configs.AttentionScope = configs.AttentionScope.FULL,
    return_lse: bool = False,
    native_fp8: str | None = None,
    native_fp8_prefill: str | None = None,
    precomputed_schedules: (
        tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]] | None
    ) = None,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Perform batched ragged paged attention.

    Args:
        queries: [max_num_tokens, num_q_heads, head_dim]. Output of q projection.
        keys: [max_num_tokens, num_kv_heads, head_dim]. Output of k projection.
        values: [max_num_tokens, num_kv_heads, head_dim]. Output of v projection.
        kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
            kv_packing, head_dim]. Stores existing kv cache data where k & vs are
            concatenated along num kv heads dim.
        kv_lens: [max_num_seqs]. Existing kv cache length of each sequence.
        page_indices: [max_num_seqs * pages_per_seqs]. kv cache page table of each
            sequence.
        cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
            length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
            b=cu_q_lens[i+1] represents q/k/v of sequence i.
        distribution: [3]. Cumulative sum of number of decode, prefill, and mixed
            sequences (cumulative end indices for decode, prefill-only, and mixed).
            The prefill-only segment is handled by the mixed kernel call.
            distribution[2] represents total number of sequences.
        sm_scale: Softmax scale value.
        sliding_window: Size of sliding window (also known as local attention). kvs
            outside of the window is not fetched from hbm and masked out during
            computation.
        soft_cap: Cap values of softmax inputs.
        mask_value: Value to use for causal masking. Defaults to smallest
            representable value of the activation dtype.
        use_per_token_scale: Whether to use per-token quantization.
        per_token_scale_dtype: Dtype of per-token scale. Defaults to None.
        q_scale: Quantization scale value of queries.
        k_scale: Per-tensor quantization scale value of keys.
        v_scale: Per-tensor quantization scale value of values.
        dynamic_k_scale: Per-token quantization scale value of keys.
        dynamic_v_scale: Per-token quantization scale value of values.
        chunk_prefill_size: Not used.
        decode_block_sizes: Kernel block size to use during decode.
        prefill_block_sizes: Kernel block size to use during prefill.
        vmem_limit_bytes: VMEM size limit of the kernel. Defaults to maximum VMEM
            size of the hardware.
        debug_mode: Not used.
        out_dtype: Dtype of output. Defaults to dtype of queries.
        use_causal_mask: Not used.
        skip_kv_update: If True, don't write keys/values back to kv_cache
            (KV-share). All kv, including this step's tokens, is read from kv_cache.
        update_kv_cache: If set, overrides skip_kv_update (update_kv_cache=False
            is the same as skip_kv_update=True). None (default) uses skip_kv_update.
        decode_query_size: Number of query tokens in decode (1 by default, can be
            higher in case of speculative decoding).
        decode_batch_size: Most sequences a decode step works on concurrently.
        cp_group_size: Size of the context parallelism (CP) group. KV cache is
            sharded across devices in this group. Defaults to None.
        cp_rank: Rank of the current device within the CP group, which determine
            the token ownership. Defaults to None.
        attention_scope: Which KV positions to attend to. FULL attends all
            positions, CACHE_ONLY skips new tokens, NEW_TOKENS_ONLY skips cached
            tokens. Defaults to FULL.
        return_lse: If True, return log-sum-exp (lse) values along with the
            output. Defaults to False.
        native_fp8: Run the DECODE kernel's QK and/or PV matmuls natively in
            the fp8 KV dtype ("qk", "pv", "qkpv", with hi/lo split operands
            "qk2", "pv2", "qk2pv2", or with static operand scales "qks",
            "pvs"); see `configs.ServingConfigs.native_fp8`. Defaults to the
            TPU_RPA_NATIVE_FP8 environment variable (unset: off).
        native_fp8_prefill: The same for the PREFILL and MIXED kernels ("qk",
            "pv", "qkpv", each operand optionally with "s" for its static
            scale, e.g. "qkpvs"); see `configs.ServingConfigs`. Defaults to
            the TPU_RPA_NATIVE_FP8_PREFILL environment variable (unset: off).
        precomputed_schedules: `(decode_leaves, mixed_leaves)` from
            `build_rpa_schedules`, shared by layers of the same geometry in a
            forward pass. Used only for the configuration `build_rpa_schedules`
            assumes (no CP, FULL scope, KV update, no LSE, no per-token scale,
            output in the query dtype); any other call builds its own schedule.

    Returns:
        out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
        new_kv_cache: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
            kv_packing, head_dim]. Result of new kv cache where k & vs are
            concatenated along num kv heads dim.
        lse (only when return_lse=True): [max_num_tokens, num_q_heads].
            Log-sum-exp values (m + log(l)) for each query token and head,
            needed for merging partial attention results in CP.
    """

    if not use_causal_mask:
        raise ValueError("Only causal attention is supported.")
    if chunk_prefill_size is not None:
        raise ValueError("Specifying chunk prefill size is not supported.")
    if debug_mode:
        raise ValueError("Debug mode is not supported.")

    # update_kv_cache (as passed by tpu-inference) overrides skip_kv_update when
    # set; update_kv_cache=False (KV-share) is the same as skip_kv_update=True.
    if update_kv_cache is not None:
        skip_kv_update = not update_kv_cache

    if out_dtype is None:
        out_dtype = queries.dtype
    if mask_value is None:
        mask_value = cast(float, jnp.finfo(out_dtype).min)
    if vmem_limit_bytes is None:
        vmem_limit_bytes = cast(int, pltpu.get_tpu_info().vmem_capacity_bytes)
    max_num_seqs = kv_lens.shape[0]
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        page_size = kv_cache.shape[4]
    else:
        page_size = kv_cache.shape[1]

    num_q_heads = queries.shape[1]
    head_dim = queries.shape[2]
    num_kv_heads = keys.shape[1]
    num_page_indices = page_indices.shape[0]

    model_cfgs = configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=sliding_window,
        sm_scale=sm_scale,
        soft_cap=soft_cap,
        mask_value=mask_value,
    )

    if k_scale is not None and dynamic_k_scale is not None:
        raise ValueError(
            "Only one of k_scale or dynamic_k_scale can be set. Got"
            f" {k_scale=} and {dynamic_k_scale=}"
        )
    if v_scale is not None and dynamic_v_scale is not None:
        raise ValueError(
            "Only one of v_scale or dynamic_v_scale can be set. Got"
            f" {v_scale=} and {dynamic_v_scale=}"
        )

    k_scale_config = k_scale
    v_scale_config = v_scale
    k_scale_tensor = dynamic_k_scale
    v_scale_tensor = dynamic_v_scale

    if native_fp8 is None:
        native_fp8 = os.environ.get("TPU_RPA_NATIVE_FP8", "")
    if native_fp8 in ("1", "true", "True"):
        native_fp8 = "qkpv"
    if native_fp8 not in ("", "0") and not re.fullmatch(
        r"(qk[2s]?)?(pv[2s]?)?", native_fp8
    ):
        raise ValueError(
            f"native_fp8 must match (qk[2s])(pv[2s]), e.g. qkpv, qk2pv2 or pvs; "
            f"got {native_fp8!r}"
        )
    if native_fp8_prefill is None:
        native_fp8_prefill = envs.TPU_RPA_NATIVE_FP8_PREFILL
    if native_fp8_prefill not in ("", "0") and not re.fullmatch(
        r"(qks?)?(pvs?)?", native_fp8_prefill
    ):
        raise ValueError(
            f"native_fp8_prefill must match (qk[s])(pv[s]), e.g. qkpv or pvs; "
            f"got {native_fp8_prefill!r}"
        )
    serve_cfgs = configs.ServingConfigs(
        native_fp8="" if native_fp8 == "0" else native_fp8,
        native_fp8_prefill="" if native_fp8_prefill == "0" else native_fp8_prefill,
        num_seqs=max_num_seqs,
        num_page_indices=num_page_indices,
        total_q_tokens=queries.shape[0],
        dtype_q=queries.dtype,
        dtype_kv=kv_cache.dtype,
        dtype_out=out_dtype,
        page_size=page_size,
        per_token_scale=use_per_token_scale,
        per_token_scale_dtype=per_token_scale_dtype,
        scale_q=q_scale,
        scale_k=k_scale_config,
        scale_v=v_scale_config,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
        cp_group_size=cp_group_size,
        attention_scope=attention_scope,
        return_lse=return_lse,
        skip_kv_update=skip_kv_update,
    )

    q_hbm, new_kv_hbm = prepare_inputs(
        queries,
        keys,
        values,
        queries.dtype,
        kv_cache.dtype,
        k_scale_tensor,
        v_scale_tensor,
        kv_layout=kv_layout,
        page_size=page_size,
    )

    default_decode, default_prefill = calculate_block_sizes(
        model_cfgs,
        serve_cfgs,
        vmem_limit_bytes,
        decode_query_size=decode_query_size,
        decode_batch_size=decode_batch_size,
    )
    # Pre-allocate LSE buffer.
    lse_hbm_init: jax.Array | None = None
    if return_lse:
        num_lanes = pltpu.get_tpu_info().num_lanes
        q_packing = utils.get_dtype_packing(queries.dtype)
        num_q_heads_per_kv_head = num_q_heads // num_kv_heads
        aligned_num_q_heads_per_kv_head = utils.align_to(
            num_q_heads_per_kv_head, q_packing
        )
        max_tokens = queries.shape[0]
        lse_hbm_init = jnp.full(
            [
                num_kv_heads,
                max_tokens,
                aligned_num_q_heads_per_kv_head // q_packing,
                q_packing,
                num_lanes,
            ],
            -jnp.inf,
            dtype=out_dtype,
        )

    # Compute per-sequence length parameters for the kernel.
    q_lens = cu_q_lens[1:] - cu_q_lens[:-1]
    global_kv_cache_lens = kv_lens - q_lens

    if attention_scope == configs.AttentionScope.CACHE_ONLY:
        if cp_group_size is not None:
            rank = cp_rank[0] if cp_rank is not None else 0
            kv_cache_lens = utils.cp_local_cache_len(
                global_kv_cache_lens, cp_group_size, rank, page_size
            )
        else:
            kv_cache_lens = global_kv_cache_lens
        kv_new_lens = jnp.zeros_like(q_lens)
        q_offsets = kv_cache_lens
    elif attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY:
        kv_cache_lens = global_kv_cache_lens
        kv_new_lens = q_lens
        q_offsets = global_kv_cache_lens
    else:  # FULL
        kv_cache_lens = global_kv_cache_lens
        kv_new_lens = q_lens
        q_offsets = global_kv_cache_lens

    # `build_rpa_schedules` sizes its blocks and DMA waits for this serving
    # configuration only.
    precomputed_schedule_matches = (
        cp_group_size is None
        and attention_scope == configs.AttentionScope.FULL
        and not skip_kv_update
        and not return_lse
        and not use_per_token_scale
        and jnp.dtype(out_dtype) == jnp.dtype(queries.dtype)
    )

    def run_rpa_kernel(
        mode: configs.RpaCase,
        o_hbm_alias_q_hbm: jax.Array,
        kv_cache: jax.Array,
        lse_hbm_in: jax.Array | None,
        dist: jax.Array = distribution,
    ):
        if mode == configs.RpaCase.DECODE:
            effective_blocks = decode_block_sizes or default_decode
        else:
            effective_blocks = prefill_block_sizes or default_prefill

        cfgs = configs.RpaConfigs(
            block=effective_blocks,
            model=model_cfgs,
            serve=serve_cfgs,
            vmem_limit_bytes=vmem_limit_bytes,
            mode=mode,
        )
        cfgs.validate_inputs(
            q=queries,
            k=keys,
            v=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=dist,
        )
        # Select metadata computer class.
        if cp_group_size is not None:
            computer_cls = schedule_cp.CPMetadataComputer
            extra_scalars = (cp_rank,) if cp_rank is not None else ()
        else:
            computer_cls = schedule.BaseMetadataComputer
            extra_scalars = ()

        schedule_hbm: schedule.RpaSchedule | None = None
        if precomputed_schedules is not None and precomputed_schedule_matches:
            precomputed_leaves = (
                precomputed_schedules[0]
                if mode == configs.RpaCase.DECODE
                else precomputed_schedules[1]
            )
            template_leaves, template_def = jax.tree_util.tree_flatten(
                computer_cls.get_rpa_schedule(
                    cfgs, multiplier=cfgs.max_schedule_size_multiplier
                )
            )
            # The kernel runs without bounds checks, so leaves sized for other
            # block sizes must not reach it.
            if [tuple(x.shape) for x in template_leaves] == [
                tuple(x.shape) for x in precomputed_leaves
            ]:
                schedule_hbm = jax.tree_util.tree_unflatten(
                    template_def, list(precomputed_leaves)
                )
        if schedule_hbm is None:
            schedule_hbm = schedule.generate_rpa_metadata(
                cu_q_lens,
                q_offsets,
                kv_cache_lens,
                kv_new_lens,
                dist,
                cfgs=cfgs,
                computer_cls=computer_cls,
                extra_scalars=extra_scalars,
            )
        result = kernel.rpa_kernel(
            cu_q_lens,
            q_offsets,
            kv_cache_lens,
            kv_new_lens,
            page_indices,
            schedule_hbm,
            o_hbm_alias_q_hbm,
            new_kv_hbm,
            kv_cache,
            lse_hbm_in,
            cfgs=cfgs,
            computer_cls=computer_cls,
        )
        if return_lse:
            o_out, kv_out, lse_out = result
        else:
            o_out, kv_out, _ = result
            lse_out = None
        return o_out, kv_out, lse_out

    effective_decode_blocks = decode_block_sizes or default_decode
    if (
        serve_cfgs.total_q_tokens
        <= min(decode_query_size, effective_decode_blocks.bq_sz)
        and cp_group_size is None
    ):
        decode_dist = jnp.repeat(distribution[2:3], 3)
        o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
            configs.RpaCase.DECODE, q_hbm, kv_cache, lse_hbm_init, decode_dist
        )
    else:
        o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
            configs.RpaCase.DECODE, q_hbm, kv_cache, lse_hbm_init
        )
        # "cond" skips the MIXED call when its sequence range
        # [distribution[0], distribution[2]) (`configs.RpaConfigs.seq_range`)
        # is empty.
        # A prefill-only runner always has a MIXED range and calls it
        # directly (kernels/static_guards.py).
        rpa_skip_mode = (
            "never"
            if static_guards.prefill_only()
            else os.environ.get("TPU_RPA_SKIP_MIXED", "cond")
        )
        if rpa_skip_mode == "cond":
            if return_lse:

                def _do_mixed_lse(args):
                    o_in, kv_in, lse_in = args
                    return run_rpa_kernel(configs.RpaCase.MIXED, o_in, kv_in, lse_in)

                o_hbm_alias_q_hbm, kv_cache, lse_hbm = jax.lax.cond(
                    distribution[0] < distribution[2],
                    _do_mixed_lse,
                    lambda args: args,
                    (o_hbm_alias_q_hbm, kv_cache, lse_hbm),
                )
            else:

                def _do_mixed(args):
                    o_in, kv_in = args
                    o_out, kv_out, _ = run_rpa_kernel(
                        configs.RpaCase.MIXED, o_in, kv_in, None
                    )
                    return o_out, kv_out

                o_hbm_alias_q_hbm, kv_cache = jax.lax.cond(
                    distribution[0] < distribution[2],
                    _do_mixed,
                    lambda args: args,
                    (o_hbm_alias_q_hbm, kv_cache),
                )
                lse_hbm = None
        else:
            o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
                configs.RpaCase.MIXED, o_hbm_alias_q_hbm, kv_cache, lse_hbm
            )

    # before: [kv_heads, max_tokens, q_per_kv // q_packing, q_packing, d]
    o_hbm = prepare_outputs(o_hbm_alias_q_hbm)
    # after: [kv_heads, max_tokens, q_per_kv, d]

    # slice back to original shape if padded
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    o_hbm = o_hbm[:, :, :num_q_heads_per_kv_head, :head_dim]
    o_hbm = o_hbm.swapaxes(1, 0).reshape(queries.shape)

    if not return_lse:
        return o_hbm, kv_cache

    # Reshape LSE from [num_kv_heads, max_tokens, q_per_kv // q_packing, q_packing
    # , num_lanes] to [max_tokens, num_q_heads].
    max_tokens = queries.shape[0]
    # Extract first lane (scalar LSE value per token-head pair).
    lse = cast(jax.Array, lse_hbm).reshape(
        num_kv_heads, max_tokens, aligned_num_q_heads_per_kv_head, num_lanes
    )[:, :max_tokens, :num_q_heads_per_kv_head, 0]
    lse = lse.swapaxes(0, 1).reshape(max_tokens, num_q_heads)

    return o_hbm, kv_cache, lse
