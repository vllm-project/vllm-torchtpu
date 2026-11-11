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
    shared_kv_rotary_dims: int = 0,
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

    if shared_kv_rotary_dims:
        validate_shared_kv(
            shared_kv_rotary_dims,
            actual_num_kv_heads,
            actual_kv_head_dim,
            kv_layout,
            use_per_token_scale=k_scale is not None,
        )
        half, r = actual_kv_head_dim // 2, shared_kv_rotary_dims
        rotated = jnp.concatenate([k[..., :r], k[..., half : half + r]], axis=-1)
        rotated = jnp.pad(rotated, ((0, 0), (0, 0), (0, actual_kv_head_dim - 2 * r)))
        k, v = v, rotated

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


def validate_shared_kv(
    rotary_dims: int,
    num_kv_heads: int,
    head_dim: int,
    kv_layout: configs.KVLayout,
    use_per_token_scale: bool = False,
) -> None:
    """Raises unless shared-KV pages (`shared_kv_rotary_dims`) can hold
    this layer's KV."""
    packing_rows = 4 * pltpu.get_tpu_info().num_sublanes  # fp8 rows per tile
    if kv_layout != configs.KVLayout.SEQ_ALONG_LANE or num_kv_heads != 1:
        raise ValueError(
            "shared_kv_rotary_dims needs SEQ_ALONG_LANE pages of one KV head, "
            f"got {kv_layout} with {num_kv_heads} KV heads."
        )
    if use_per_token_scale:
        raise ValueError("shared_kv_rotary_dims does not support per-token scales.")
    if rotary_dims % packing_rows or (head_dim // 2) % packing_rows:
        raise ValueError(
            f"shared_kv_rotary_dims ({rotary_dims}) and half of head_dim "
            f"({head_dim // 2}) must be whole {packing_rows}-row tiles."
        )
    if not 0 < rotary_dims <= head_dim // 2:
        raise ValueError(
            f"shared_kv_rotary_dims {rotary_dims} is not in (0, {head_dim // 2}]."
        )


def validate_fp8_compute(
    kv_dtype: jnp.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    q_scale: float | None,
    return_lse: bool,
    use_per_token_scale: bool = False,
) -> None:
    """Raises unless `fp8_compute` can run for this call."""
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
    """Casts the prepared queries to `dtype`, saturating, at a scale of 1.0."""
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


def calculate_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
    decode_query_size: int = 1,
    cap_sliding_decode_block: bool = False,
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
        bq_array_size = bq_sz * aligned_num_q_heads * aligned_head_dim
        vmem_bkv_sz = bkv_sz + (
            2 * num_lanes
            if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE
            else 0
        )
        bkv_array_size = vmem_bkv_sz * aligned_num_kv_heads_x2 * aligned_kv_head_dim_val
        if serve_cfgs.shared_kv_rotary_dims:
            bkv_array_size = vmem_bkv_sz * (
                model_cfgs.head_dim + 2 * serve_cfgs.shared_kv_rotary_dims
            )
        buffer_bytes = (
            bq_array_size * q_bytes * configs.NUM_INPUT_SLOTS
            + bkv_array_size * kv_bytes * configs.NUM_INPUT_SLOTS
            + bq_array_size * out_bytes * configs.NUM_OUTPUT_SLOTS
        )
        if serve_cfgs.return_lse:
            buffer_bytes += bq_sz * aligned_num_q_heads * num_lanes * out_bytes * 2

        loaded_bq_bytes = bq_sz * model_cfgs.num_q_heads * aligned_head_dim * q_bytes
        loaded_bkv_bytes = (
            bkv_sz * model_cfgs.num_kv_heads * aligned_head_dim * kv_bytes
        )
        qk_bytes = bq_sz * bkv_sz * model_cfgs.num_q_heads * out_bytes
        loaded_scale_bytes = 0
        if serve_cfgs.per_token_scale:
            assert serve_cfgs.per_token_scale_dtype is not None
            scale_bytes = (
                jax.dtypes.itemsize_bits(serve_cfgs.per_token_scale_dtype) // 8
            )
            loaded_scale_bytes = 2 * bkv_sz * model_cfgs.num_kv_heads * scale_bytes

        # Sum up all compute memory usage.
        compute_bytes = (
            loaded_bq_bytes + loaded_bkv_bytes + qk_bytes + loaded_scale_bytes
        )
        total_bytes = (buffer_bytes + compute_bytes) * batch_size
        lm_array_size = (
            model_cfgs.num_kv_heads * bq_sz * aligned_num_q_heads_per_kv_head
        )
        return total_bytes + 3 * lm_array_size * out_bytes

    def calculate_compute_buffer_time(
        batch_size: int, bq_c_sz: int, bkv_sz: int
    ) -> int:
        num_k = pl.cdiv(bkv_sz, mxu_column_size) * pl.cdiv(
            model_cfgs.head_dim, mxu_column_size
        )
        return batch_size * bq_c_sz * num_k * model_cfgs.num_q_heads

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
        bq_sz = bkv_sz if fixed_bq_sz is None else fixed_bq_sz
        bq_stride = bkv_sz if fixed_bq_sz is None else 0
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
        threshold = 800 if (tpu_info.generation == 8 and _is_sfs(tpu_info)) else 1500
        if jnp.dtype(serve_cfgs.dtype_q).itemsize == 1:
            threshold *= tpu_info.fp8_ops_per_second // tpu_info.bf16_ops_per_second
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

    decode_max_bkv_sz = _MAX_BKV_SZ
    if cap_sliding_decode_block and model_cfgs.sliding_window is not None:
        decode_max_bkv_sz = min(decode_max_bkv_sz, model_cfgs.sliding_window // 2)
    return (
        find_best_block_sizes(8, decode_query_size, decode_max_bkv_sz),
        find_best_block_sizes(2),
    )


FP8_PREFILL_KV_BLOCK = 512
FP8_PREFILL_CHUNK_ROWS = 256
FP8_PREFILL_STEP_ROWS = 8192
FP8_PREFILL_BUFFER_VMEM_FRACTION = 0.8


def fp8_prefill_block_sizes(
    model_cfgs: configs.ModelConfigs,
    serve_cfgs: configs.ServingConfigs,
    vmem_limit_bytes: int,
) -> configs.BlockSizes:
    """Mixed-kernel block sizes for `fp8_compute`."""
    if model_cfgs.head_dim < 256:
        return calculate_block_sizes(model_cfgs, serve_cfgs, vmem_limit_bytes)[1]
    rows_per_token = utils.align_to(
        model_cfgs.num_q_heads_per_kv_head, serve_cfgs.packing_q
    )
    bq_c = max(1, FP8_PREFILL_CHUNK_ROWS // rows_per_token)
    bkv = FP8_PREFILL_KV_BLOCK
    if serve_cfgs.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        bkv = utils.align_to(bkv, serve_cfgs.page_size)
    bq = FP8_PREFILL_STEP_ROWS // (rows_per_token * model_cfgs.num_kv_heads)
    bq = utils.align_to(max(bq, bq_c), bq_c)
    while True:
        blocks = configs.BlockSizes(bq_sz=bq, bq_c_sz=bq_c, bkv_sz=bkv, batch_size=1)
        cfgs = configs.RpaConfigs(
            block=blocks,
            model=model_cfgs,
            serve=serve_cfgs,
            mode=configs.RpaCase.MIXED,
            vmem_limit_bytes=vmem_limit_bytes,
        )
        if (
            fp8_prefill_vmem_bytes(cfgs)
            <= FP8_PREFILL_BUFFER_VMEM_FRACTION * vmem_limit_bytes
        ):
            return blocks
        if bq == bq_c:
            raise ValueError(
                f"fp8_compute: no query block fits {vmem_limit_bytes} bytes of VMEM "
                f"for {model_cfgs}."
            )
        bq = utils.align_to(max(bq // 2, bq_c), bq_c)


def schedule_size_multiplier(cfgs: configs.RpaConfigs, num_q_tokens: int) -> int:
    """Return HBM schedule multiplier covering all steps of one call."""
    kv_blocks = pl.cdiv(cfgs.serve.pages_per_seq * cfgs.serve.page_size, cfgs.bkv_sz)
    if cfgs.model.sliding_window is not None:
        kv_blocks = min(
            kv_blocks, pl.cdiv(cfgs.model.sliding_window + cfgs.bq_sz, cfgs.bkv_sz) + 1
        )
    q_blocks = pl.cdiv(num_q_tokens, cfgs.bq_sz) + cfgs.serve.num_seqs
    if cfgs.mode == configs.RpaCase.DECODE:
        q_blocks = min(
            q_blocks,
            cfgs.serve.num_seqs * pl.cdiv(cfgs.serve.decode_query_size, cfgs.bq_sz),
        )
    steps = pl.cdiv(q_blocks * kv_blocks, cfgs.batch_size) + cfgs.max_steps_ub
    return max(cfgs.max_schedule_size_multiplier, pl.cdiv(steps, cfgs.max_steps_ub))


def fp8_prefill_vmem_bytes(cfgs: configs.RpaConfigs) -> int:
    """VMEM of the mixed kernel's pipeline buffers and softmax scratch."""

    def nbytes(shape, dtype):
        return math.prod(shape) * utils.get_dtype_bits(dtype) // 8

    serve = cfgs.serve
    return (
        configs.NUM_INPUT_SLOTS * nbytes(cfgs.q_vmem_shape, serve.dtype_q)
        + configs.NUM_INPUT_SLOTS * nbytes(cfgs.kv_vmem_shape, serve.dtype_kv)
        + configs.NUM_OUTPUT_SLOTS * nbytes(cfgs.o_vmem_shape, serve.dtype_out)
        + 2 * nbytes(cfgs.lm_scratch_shape, serve.dtype_out)
        + nbytes(cfgs.acc_scratch_shape, serve.dtype_out)
    )


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
        "cp_group_size",
        "attention_scope",
        "return_lse",
        "mask_invalid_v",
        "skip_empty_launch",
        "cap_sliding_decode_block",
        "fp8_compute",
        "fence_output",
        "unmasked_interior",
        "output_quant_dtype",
        "shared_kv_rotary_dims",
        "skip_masked_kv_zero_fill",
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
    cp_group_size: int | None = None,
    cp_rank: jax.Array | None = None,
    attention_scope: configs.AttentionScope = configs.AttentionScope.FULL,
    return_lse: bool = False,
    mask_invalid_v: bool | None = None,
    skip_empty_launch: bool = False,
    cap_sliding_decode_block: bool = False,
    fp8_compute: bool = False,
    fence_output: bool = False,
    unmasked_interior: bool = False,
    output_quant_dtype: jnp.dtype | None = None,
    shared_kv_rotary_dims: int = 0,
    skip_masked_kv_zero_fill: bool = False,
    prepared_inputs: tuple[jax.Array, ...] | None = None,
) -> tuple[Any, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Run batched ragged paged attention and update ``kv_cache``."""

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

    serve_cfgs = configs.ServingConfigs(
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
        mask_invalid_v=mask_invalid_v,
        skip_empty_launch=skip_empty_launch,
        unmasked_interior=unmasked_interior,
        shared_kv_rotary_dims=shared_kv_rotary_dims,
        skip_masked_kv_zero_fill=skip_masked_kv_zero_fill,
    )
    if shared_kv_rotary_dims:
        validate_shared_kv(
            shared_kv_rotary_dims,
            num_kv_heads,
            head_dim,
            kv_layout,
            use_per_token_scale,
        )
        if cp_group_size is not None or return_lse:
            raise ValueError(
                "shared_kv_rotary_dims supports neither CP nor LSE output."
            )
    if output_quant_dtype is not None and (return_lse or cp_group_size is not None):
        raise ValueError("output_quant_dtype supports neither LSE output nor CP.")

    prepare = functools.partial(
        prepare_inputs,
        q_dtype=queries.dtype,
        kv_dtype=kv_cache.dtype,
        k_scale=k_scale_tensor,
        v_scale=v_scale_tensor,
        kv_layout=kv_layout,
        page_size=page_size,
        shared_kv_rotary_dims=shared_kv_rotary_dims,
    )
    prepared_q_fetch = None
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

    default_decode, default_prefill = calculate_block_sizes(
        model_cfgs,
        serve_cfgs,
        vmem_limit_bytes,
        decode_query_size=decode_query_size,
        cap_sliding_decode_block=cap_sliding_decode_block,
    )

    mixed_serve_cfgs = serve_cfgs
    q_fetch_hbm = None
    if prepared_q_fetch is not None and not fp8_compute:
        raise ValueError("prepared FP8 queries are for the fp8_compute kernel.")
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
                    f"prepared FP8 queries {got} are not quantize_queries' "
                    f"{(want.shape, want.dtype)}"
                )
            q_fetch_hbm = prepared_q_fetch
        default_prefill = fp8_prefill_block_sizes(
            model_cfgs,
            mixed_serve_cfgs,
            vmem_limit_bytes,
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
    else:
        kv_cache_lens = global_kv_cache_lens
        kv_new_lens = q_lens
        q_offsets = global_kv_cache_lens

    branch_on_empty_segment = queries.shape[0] <= max_num_seqs * decode_query_size

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
        multiplier = schedule_size_multiplier(cfgs, queries.shape[0])
        if multiplier != cfgs.max_schedule_size_multiplier:
            cfgs = dataclasses.replace(
                cfgs,
                serve=dataclasses.replace(
                    cfgs.serve, max_schedule_size_multiplier=multiplier
                ),
            )
        cfgs.validate_inputs(
            q=queries,
            k=keys,
            v=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            page_indices=page_indices,
            cu_q_lens=cu_q_lens,
            distribution=distribution,
        )
        # Select metadata computer class.
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
        seq_start, seq_end = cfgs.seq_range(distribution)
        return jax.lax.cond(
            jnp.greater(seq_end, seq_start), launch, lambda ops: ops, operands
        )

    o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
        configs.RpaCase.DECODE, q_hbm, kv_cache, lse_hbm_init
    )
    o_hbm_alias_q_hbm, kv_cache, lse_hbm = run_rpa_kernel(
        configs.RpaCase.MIXED, o_hbm_alias_q_hbm, kv_cache, lse_hbm
    )
    if output_quant_dtype is not None:
        quantized = output_quant.quantize_head_major_output(
            o_hbm_alias_q_hbm,
            num_q_heads_per_kv_head=num_q_heads // num_kv_heads,
            head_dim=head_dim,
            quant_dtype=output_quant_dtype,
        )
        return quantized, kv_cache

    # before: [kv_heads, max_tokens, q_per_kv // q_packing, q_packing, d]
    o_hbm = prepare_outputs(o_hbm_alias_q_hbm)
    # after: [kv_heads, max_tokens, q_per_kv, d]

    # slice back to original shape if padded
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    o_hbm = o_hbm[:, :, :num_q_heads_per_kv_head, :head_dim]
    o_hbm = o_hbm.swapaxes(1, 0).reshape(queries.shape)
    if fence_output:
        # Prevent XLA from fusing downstream ops into the RPA output buffer.
        o_hbm = jax.lax.optimization_barrier(o_hbm)

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
