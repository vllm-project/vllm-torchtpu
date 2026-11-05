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
"""Wrapper for RPA kernel to match expected interface."""

import dataclasses
import functools
import math
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    configs,
    kernel,
    output_quant,
    schedule,
    schedule_cp,
    utils,
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


def validate_fp8_compute(
    kv_dtype: jnp.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    q_scale: float | None,
    return_lse: bool,
    use_per_token_scale: bool = False,
) -> None:
    """Raises unless `fp8_compute` is supported for the given arguments."""
    kv_dtype = jnp.dtype(kv_dtype)
    if not (
        jnp.issubdtype(kv_dtype, jnp.floating) and utils.get_dtype_bits(kv_dtype) == 8
    ):
        raise ValueError(f"fp8_compute needs an fp8 KV cache, got {kv_dtype}.")
    packing = utils.get_dtype_packing(kv_dtype)
    if (num_q_heads // num_kv_heads) % packing:
        raise ValueError(
            "fp8_compute needs the query heads per KV head "
            f"({num_q_heads // num_kv_heads}) to be a multiple of {packing}."
        )
    if q_scale is not None:
        raise ValueError("fp8_compute quantizes queries at scale 1.0; got q_scale.")
    if return_lse:
        raise ValueError("fp8_compute does not support return_lse.")
    if use_per_token_scale:
        raise ValueError("fp8_compute does not support per-token KV scales.")


def quantize_queries(q_hbm: jax.Array, dtype: jnp.dtype) -> jax.Array:
    """Saturating cast of prepared queries to `dtype` at unit scale.

    Maps `[kv_heads, tokens, G // p, p, d]` to `[kv_heads, tokens, G // p', p', d]`.
    """
    kv_heads, tokens, q_per_kv_packed, q_packing, head_dim = q_hbm.shape
    packing = utils.get_dtype_packing(dtype)
    info = jnp.finfo(dtype)
    q = q_hbm.reshape(kv_heads, tokens, q_per_kv_packed * q_packing, head_dim)
    q = jnp.clip(q, float(info.min), float(info.max)).astype(dtype)
    return q.reshape(kv_heads, tokens, -1, packing, head_dim)


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


_MAX_BKV_SZ = 4096
_FALLBACK_VMEM_BYTES = 1 << 24
_FALLBACK_MXU_COLUMN_SIZE = 128


def _get_tpu_vmem_and_mxu() -> tuple[int, int]:
    """Queries hardware VMEM capacity and MXU column size with a CPU fallback."""
    try:
        info = pltpu.get_tpu_info()
    except Exception:
        info = None
    if info is None:
        return _FALLBACK_VMEM_BYTES, _FALLBACK_MXU_COLUMN_SIZE
    return int(info.vmem_capacity_bytes), int(info.mxu_column_size)


def calculate_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
    decode_query_size: int = 1,
) -> tuple[configs.BlockSizes, configs.BlockSizes]:
    """Calculate optimal block size for decode and prefill."""

    tpu_info = pltpu.get_tpu_info()
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

    def calculate_vmem_usage(batch_size: int, bq_sz: int, bkv_sz: int) -> int:
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
        total_bytes = buffer_bytes + compute_bytes

        # Account for batch size.
        total_bytes *= batch_size

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

    def find_best_block_sizes(
        max_batch_size: int,
        fixed_bq_sz: int | None = None,
        max_bkv_sz: int = _MAX_BKV_SZ,
    ) -> configs.BlockSizes:
        """Loop through different block sizes to find the most optimal one."""

        # Even if we loose some potential performance, we want to avoid OOM at all
        # costs. Therefore, we conservatively only use 80% of the VMEM budget.
        capped_vmem_limit_bytes = vmem_limit_bytes * 0.8

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
            and calculate_vmem_usage(batch_size, bq_sz, bkv_sz)
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
            calculate_vmem_usage(batch_size, bq_sz, bkv_sz) < capped_vmem_limit_bytes
            and bkv_sz <= max_seq_len
            and bkv_sz <= max_bkv_sz
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

        # Step 3: Given current tile size, calculate compute tile size.

        # Threshold value based on hardware generation and precision.
        tpu_info = pltpu.get_tpu_info()

        is_8bit = jnp.dtype(serve_cfgs.dtype_q).itemsize == 1

        match tpu_info.generation:
            case 8 if _is_sfs(tpu_info):
                threshold = 800
            case 7:
                threshold = 1500
            case _:
                threshold = 1500

        if is_8bit:
            flops_ratio = tpu_info.fp8_ops_per_second // tpu_info.bf16_ops_per_second
            threshold *= flops_ratio
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

        return configs.BlockSizes(
            bq_sz=bq_sz,
            bq_c_sz=last_valid_bq_c_sz,
            bkv_sz=bkv_sz,
            batch_size=batch_size,
        )

    # Fixed value based on experimental results.
    decode_batch_size = 8
    prefill_batch_size = 2

    max_bkv_sz = _MAX_BKV_SZ
    if model_cfgs.sliding_window is not None:
        # Bounding bkv_sz <= sliding_window // 2 limits masked-out boundary work
        # in decode and keeps square prefill tiles small enough to contain
        # unmasked interior steps (bq_sz + bkv_sz <= sliding_window). The search
        # never drops below its initial bkv_stride block.
        max_bkv_sz = min(max_bkv_sz, model_cfgs.sliding_window // 2)

    decode_block_sizes = find_best_block_sizes(
        decode_batch_size, decode_query_size, max_bkv_sz=max_bkv_sz
    )
    prefill_block_sizes = find_best_block_sizes(
        prefill_batch_size, max_bkv_sz=max_bkv_sz
    )

    return decode_block_sizes, prefill_block_sizes


FP8_PREFILL_KV_BLOCK = 512
FP8_PREFILL_CHUNK_ROWS = 256
FP8_PREFILL_STEP_ROWS = 8192
FP8_PREFILL_BUFFER_VMEM_FRACTION = 0.8


def fp8_prefill_vmem_bytes(cfgs: configs.RpaConfigs) -> int:
    """VMEM bytes required by the FP8 MIXED kernel's pipeline buffers and scratches."""

    def nbytes(shape: tuple[int, ...], dtype: jnp.dtype) -> int:
        return math.prod(shape) * utils.get_dtype_bits(dtype) // 8

    serve = cfgs.serve
    return (
        configs.NUM_INPUT_SLOTS * nbytes(cfgs.q_vmem_shape, serve.dtype_q)
        + configs.NUM_INPUT_SLOTS * nbytes(cfgs.kv_vmem_shape, serve.dtype_kv)
        + configs.NUM_OUTPUT_SLOTS * nbytes(cfgs.o_vmem_shape, serve.dtype_out)
        + 2 * nbytes(cfgs.lm_scratch_shape, serve.dtype_out)
        + nbytes(cfgs.acc_scratch_shape, serve.dtype_out)
    )


def fp8_prefill_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int | None = None,
) -> configs.BlockSizes:
    """Selects MIXED-kernel block sizes for `fp8_compute` within VMEM budget."""
    hw_vmem, mxu_col = _get_tpu_vmem_and_mxu()
    effective_vmem = hw_vmem if vmem_limit_bytes is None else vmem_limit_bytes

    rows_per_token = utils.align_to(
        model_cfgs.num_q_heads_per_kv_head, serve_cfgs.packing_q
    )
    bq_c = max(1, FP8_PREFILL_CHUNK_ROWS // rows_per_token)
    bkv = FP8_PREFILL_KV_BLOCK
    if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        bkv = utils.align_to(bkv, serve_cfgs.page_size)
    bq = FP8_PREFILL_STEP_ROWS // (rows_per_token * model_cfgs.num_kv_heads)
    bq = utils.align_to(max(bq, bq_c), bq_c)

    if model_cfgs.sliding_window is not None:
        sliding_cap = model_cfgs.sliding_window // 2
        stride = mxu_col
        if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            stride = utils.align_to(stride, serve_cfgs.page_size)
        bkv = max(min(bkv, sliding_cap) // stride, 1) * stride
        bq = max(min(bq, sliding_cap) // bq_c, 1) * bq_c

    while True:
        blocks = configs.BlockSizes(bq_sz=bq, bq_c_sz=bq_c, bkv_sz=bkv, batch_size=1)
        cfgs = configs.RpaConfigs(
            block=blocks,
            model=model_cfgs,
            serve=serve_cfgs,
            mode=configs.RpaCase.MIXED,
            vmem_limit_bytes=effective_vmem,
        )
        if (
            fp8_prefill_vmem_bytes(cfgs)
            <= FP8_PREFILL_BUFFER_VMEM_FRACTION * effective_vmem
        ):
            return blocks
        if bq == bq_c:
            raise ValueError(
                f"fp8_compute: no query block fits {effective_vmem} bytes of VMEM "
                f"for {model_cfgs}."
            )
        bq = utils.align_to(max(bq // 2, bq_c), bq_c)


_RPA_STATIC_ARGNAMES = (
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
    "cp_group_size",
    "attention_scope",
    "return_lse",
    "mask_invalid_v",
    "skip_empty_launch",
    "fence_output",
    "unmasked_interior",
    "fp8_compute",
    "output_quant_dtype",
)


def _ragged_paged_attention_impl(
    q_spec: jax.Array | jax.ShapeDtypeStruct,
    k_spec: jax.Array | jax.ShapeDtypeStruct,
    v_spec: jax.Array | jax.ShapeDtypeStruct,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    prepared_q_fetch: jax.Array | None,
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
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: configs.AttentionScope = configs.AttentionScope.FULL,
    return_lse: bool = False,
    mask_invalid_v: bool | None = None,
    skip_empty_launch: bool = True,
    fence_output: bool = True,
    unmasked_interior: bool = True,
    fp8_compute: bool = False,
    output_quant_dtype: jnp.dtype | None = None,
) -> tuple[Any, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    if not use_causal_mask:
        raise ValueError("Only causal attention is supported.")
    if chunk_prefill_size is not None:
        raise ValueError("Specifying chunk prefill size is not supported.")
    if debug_mode:
        raise ValueError("Debug mode is not supported.")

    if update_kv_cache is not None:
        skip_kv_update = not update_kv_cache

    if out_dtype is None:
        out_dtype = q_spec.dtype
    if mask_value is None:
        mask_value = cast(float, jnp.finfo(out_dtype).min)
    if vmem_limit_bytes is None:
        vmem_limit_bytes, _ = _get_tpu_vmem_and_mxu()
    max_num_seqs = kv_lens.shape[0]
    if kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        page_size = kv_cache.shape[4]
    else:
        page_size = kv_cache.shape[1]

    num_q_heads = q_spec.shape[1]
    head_dim = q_spec.shape[2]
    num_kv_heads = k_spec.shape[1]
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

    serve_cfgs = configs.ServingConfigs(
        num_seqs=max_num_seqs,
        num_page_indices=num_page_indices,
        total_q_tokens=q_spec.shape[0],
        dtype_q=q_spec.dtype,
        dtype_kv=kv_cache.dtype,
        dtype_out=out_dtype,
        page_size=page_size,
        per_token_scale=use_per_token_scale,
        per_token_scale_dtype=per_token_scale_dtype,
        scale_q=q_scale,
        scale_k=k_scale,
        scale_v=v_scale,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
        cp_group_size=cp_group_size,
        attention_scope=attention_scope,
        return_lse=return_lse,
        skip_kv_update=skip_kv_update,
        mask_invalid_v=mask_invalid_v,
        skip_empty_launch=skip_empty_launch,
        fence_output=fence_output,
        unmasked_interior=unmasked_interior,
    )
    if output_quant_dtype is not None and (return_lse or cp_group_size is not None):
        raise ValueError("output_quant_dtype supports neither LSE output nor CP.")

    default_decode, default_prefill = calculate_block_sizes(
        model_cfgs,
        serve_cfgs,
        vmem_limit_bytes,
        decode_query_size=decode_query_size,
    )

    mixed_serve_cfgs = serve_cfgs
    q_fetch_hbm: jax.Array | None = None
    if prepared_q_fetch is not None and not fp8_compute:
        raise ValueError("prepared FP8 queries require fp8_compute=True.")
    if fp8_compute:
        validate_fp8_compute(
            kv_cache.dtype,
            num_q_heads,
            num_kv_heads,
            q_scale,
            return_lse,
            use_per_token_scale,
        )
        mixed_serve_cfgs = dataclasses.replace(
            serve_cfgs, dtype_q=kv_cache.dtype, fp8_compute=True
        )
        if prepared_q_fetch is None:
            q_fetch_hbm = quantize_queries(q_hbm, kv_cache.dtype)
        else:
            want = jax.eval_shape(
                functools.partial(quantize_queries, dtype=kv_cache.dtype), q_hbm
            )
            got = (prepared_q_fetch.shape, prepared_q_fetch.dtype)
            if got != (want.shape, want.dtype):
                raise ValueError(
                    f"prepared FP8 queries {got} != expected {(want.shape, want.dtype)}"
                )
            q_fetch_hbm = prepared_q_fetch
        default_prefill = fp8_prefill_block_sizes(
            model_cfgs,
            mixed_serve_cfgs,
            vmem_limit_bytes,
        )

    lse_hbm_init: jax.Array | None = None
    if return_lse:
        num_lanes = pltpu.get_tpu_info().num_lanes
        q_packing = utils.get_dtype_packing(q_spec.dtype)
        num_q_heads_per_kv_head = num_q_heads // num_kv_heads
        aligned_num_q_heads_per_kv_head = utils.align_to(
            num_q_heads_per_kv_head, q_packing
        )
        max_tokens = q_spec.shape[0]
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

    # XLA copies the output buffer around lax.cond, so host-level empty-segment
    # branching is bounded to buckets no larger than a full decode batch.
    branch_on_empty_segment = q_spec.shape[0] <= max_num_seqs * decode_query_size

    def run_rpa_kernel(
        mode: configs.RpaCase,
        o_hbm_alias_q_hbm: jax.Array,
        kv_cache: jax.Array,
        lse_hbm_in: jax.Array | None,
    ):
        if mode == configs.RpaCase.DECODE:
            effective_blocks = decode_block_sizes or default_decode
            mode_serve_cfgs = serve_cfgs
        else:
            effective_blocks = prefill_block_sizes or default_prefill
            mode_serve_cfgs = mixed_serve_cfgs

        cfgs = configs.RpaConfigs(
            block=effective_blocks,
            model=model_cfgs,
            serve=mode_serve_cfgs,
            vmem_limit_bytes=vmem_limit_bytes,
            mode=mode,
        )
        cfgs.validate_inputs(
            q=cast(jax.Array, q_spec),
            k=cast(jax.Array, k_spec),
            v=cast(jax.Array, v_spec),
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
        )
        if cp_group_size is not None:
            computer_cls = schedule_cp.CPMetadataComputer
            extra_scalars = (cp_rank,) if cp_rank is not None else ()
        else:
            computer_cls = schedule.BaseMetadataComputer
            extra_scalars = ()

        schedule_hbm = schedule.generate_rpa_metadata(
            cu_q_lens,
            q_offsets,
            kv_cache_lens,
            kv_new_lens,
            distribution,
            page_indices,
            cfgs=cfgs,
            computer_cls=computer_cls,
            extra_scalars=extra_scalars,
        )

        def launch(operands):
            o_in, kv_in, lse_in = operands
            result = kernel.rpa_kernel(
                cu_q_lens,
                q_offsets,
                kv_cache_lens,
                kv_new_lens,
                schedule_hbm,
                o_in,
                new_kv_hbm,
                kv_in,
                lse_in,
                cfgs=cfgs,
                computer_cls=computer_cls,
                q_fetch_hbm=q_fetch_hbm if mode == configs.RpaCase.MIXED else None,
            )
            o_out, kv_out, lse_out = result
            return o_out, kv_out, lse_out if return_lse else None

        operands = (o_hbm_alias_q_hbm, kv_cache, lse_hbm_in)
        if not (skip_empty_launch and branch_on_empty_segment):
            return launch(operands)
        # Keep schedule generation outside the conditional so identical layer
        # geometries remain eligible for schedule CSE across layers.
        seq_start, seq_end = cfgs.seq_range(distribution)
        return jax.lax.cond(seq_end > seq_start, launch, lambda ops: ops, operands)

    o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
        configs.RpaCase.DECODE, q_hbm, kv_cache, lse_hbm_init
    )
    o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
        configs.RpaCase.MIXED, o_hbm_alias_q_hbm, kv_cache, lse_hbm
    )

    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    if output_quant_dtype is not None:
        quantized = output_quant.quantize_head_major_output(
            o_hbm_alias_q_hbm,
            quant_dtype=output_quant_dtype,
            num_q_heads_per_kv_head=num_q_heads_per_kv_head,
            head_dim=head_dim,
        )
        return quantized, kv_cache

    o_hbm = prepare_outputs(o_hbm_alias_q_hbm)
    o_hbm = o_hbm[:, :, :num_q_heads_per_kv_head, :head_dim]
    o_hbm = o_hbm.swapaxes(1, 0).reshape(q_spec.shape)
    if fence_output:
        # Materialize the token-major transpose in output precision rather than
        # fusing it into downstream f32 quantization or projection math.
        o_hbm = jax.lax.optimization_barrier(o_hbm)

    if not return_lse:
        return o_hbm, kv_cache

    max_tokens = q_spec.shape[0]
    lse = cast(jax.Array, lse_hbm).reshape(
        num_kv_heads, max_tokens, aligned_num_q_heads_per_kv_head, num_lanes
    )[:, :max_tokens, :num_q_heads_per_kv_head, 0]
    lse = lse.swapaxes(0, 1).reshape(max_tokens, num_q_heads)

    return o_hbm, kv_cache, lse


@jax.jit(
    static_argnames=_RPA_STATIC_ARGNAMES,
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
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: configs.AttentionScope = configs.AttentionScope.FULL,
    return_lse: bool = False,
    mask_invalid_v: bool | None = None,
    skip_empty_launch: bool = True,
    fence_output: bool = True,
    unmasked_interior: bool = True,
    fp8_compute: bool = False,
    output_quant_dtype: jnp.dtype | None = None,
    prepared_inputs: tuple[jax.Array, ...] | None = None,
) -> tuple[Any, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Perform batched ragged paged attention."""
    page_size = (
        kv_cache.shape[4]
        if kv_layout == configs.KVLayout.SEQ_ALONG_LANE
        else kv_cache.shape[1]
    )
    prepare = functools.partial(
        prepare_inputs,
        q_dtype=queries.dtype,
        kv_dtype=kv_cache.dtype,
        k_scale=dynamic_k_scale,
        v_scale=dynamic_v_scale,
        kv_layout=kv_layout,
        page_size=page_size,
    )
    prepared_q_fetch: jax.Array | None = None
    if prepared_inputs is None:
        q_hbm, new_kv_hbm = prepare(queries, keys, values)
    else:
        expected = [
            (a.shape, a.dtype) for a in jax.eval_shape(prepare, queries, keys, values)
        ]
        got = [(a.shape, a.dtype) for a in prepared_inputs[:2]]
        if got != expected or len(prepared_inputs) > 3:
            raise ValueError(f"prepared_inputs {got} != prepare_inputs {expected}")
        q_hbm, new_kv_hbm = prepared_inputs[:2]
        if len(prepared_inputs) == 3:
            prepared_q_fetch = prepared_inputs[2]

    return _ragged_paged_attention_impl(
        queries,
        keys,
        values,
        q_hbm,
        new_kv_hbm,
        prepared_q_fetch,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        use_per_token_scale=use_per_token_scale,
        per_token_scale_dtype=per_token_scale_dtype,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        dynamic_k_scale=dynamic_k_scale,
        dynamic_v_scale=dynamic_v_scale,
        chunk_prefill_size=chunk_prefill_size,
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=prefill_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
        out_dtype=out_dtype,
        use_causal_mask=use_causal_mask,
        skip_kv_update=skip_kv_update,
        update_kv_cache=update_kv_cache,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
        cp_group_size=cp_group_size,
        cp_rank=cp_rank,
        attention_scope=attention_scope,
        return_lse=return_lse,
        mask_invalid_v=mask_invalid_v,
        skip_empty_launch=skip_empty_launch,
        fence_output=fence_output,
        unmasked_interior=unmasked_interior,
        fp8_compute=fp8_compute,
        output_quant_dtype=output_quant_dtype,
    )


@jax.jit(
    static_argnames=("num_q_heads", "num_kv_heads", "head_dim", *_RPA_STATIC_ARGNAMES),
    donate_argnames=("kv_cache",),
)
def ragged_paged_attention_prepared(
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_fetch_hbm: jax.Array | None = None,
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
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: configs.AttentionScope = configs.AttentionScope.FULL,
    return_lse: bool = False,
    mask_invalid_v: bool | None = None,
    skip_empty_launch: bool = True,
    fence_output: bool = True,
    unmasked_interior: bool = True,
    fp8_compute: bool = False,
    output_quant_dtype: jnp.dtype | None = None,
) -> tuple[Any, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Run batched ragged paged attention directly from pre-packed prologue outputs."""
    tokens = q_hbm.shape[1]
    kv_head_dim = head_dim // 2 if kv_cache.dtype == jnp.uint8 else head_dim
    q_spec = jax.ShapeDtypeStruct((tokens, num_q_heads, head_dim), q_hbm.dtype)
    kv_spec = jax.ShapeDtypeStruct((tokens, num_kv_heads, kv_head_dim), kv_cache.dtype)
    page_size = (
        kv_cache.shape[4]
        if kv_layout == configs.KVLayout.SEQ_ALONG_LANE
        else kv_cache.shape[1]
    )
    expected = [
        (a.shape, a.dtype)
        for a in jax.eval_shape(
            functools.partial(
                prepare_inputs,
                q_dtype=q_spec.dtype,
                kv_dtype=kv_cache.dtype,
                k_scale=dynamic_k_scale,
                v_scale=dynamic_v_scale,
                kv_layout=kv_layout,
                page_size=page_size,
            ),
            q_spec,
            kv_spec,
            kv_spec,
        )
    ]
    got = [(q_hbm.shape, q_hbm.dtype), (new_kv_hbm.shape, new_kv_hbm.dtype)]
    if got != expected:
        raise ValueError(f"prepared inputs {got} != prepare_inputs {expected}")

    return _ragged_paged_attention_impl(
        q_spec,
        kv_spec,
        kv_spec,
        q_hbm,
        new_kv_hbm,
        q_fetch_hbm,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        use_per_token_scale=use_per_token_scale,
        per_token_scale_dtype=per_token_scale_dtype,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        dynamic_k_scale=dynamic_k_scale,
        dynamic_v_scale=dynamic_v_scale,
        chunk_prefill_size=chunk_prefill_size,
        decode_block_sizes=decode_block_sizes,
        prefill_block_sizes=prefill_block_sizes,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
        out_dtype=out_dtype,
        use_causal_mask=use_causal_mask,
        skip_kv_update=skip_kv_update,
        update_kv_cache=update_kv_cache,
        kv_layout=kv_layout,
        decode_query_size=decode_query_size,
        cp_group_size=cp_group_size,
        cp_rank=cp_rank,
        attention_scope=attention_scope,
        return_lse=return_lse,
        mask_invalid_v=mask_invalid_v,
        skip_empty_launch=skip_empty_launch,
        fence_output=fence_output,
        unmasked_interior=unmasked_interior,
        fp8_compute=fp8_compute,
        output_quant_dtype=output_quant_dtype,
    )
