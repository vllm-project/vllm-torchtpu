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

import functools
import os

import jax
from jax import numpy as jnp
from jax.experimental.pallas import tpu as pltpu

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.experimental.moe_dense_expert import (
    can_use_dense_expert,
    dense_expert_moe,
)
from vllm_torchtpu.kernels.megablox.gmm_v2 import (
    Dimensions,
    InputConfigs,
    TileSizes,
    calculate_tiling,
    get_packing_factor,
    gmm_v2,
    quantize_lhs_rows,
)
from vllm_torchtpu.kernels.megablox.moe_onehot_unpermute import (
    blockwise_onehot_unpermute,
    can_use_blockwise_onehot_unpermute,
)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce import (
    ragged_gather_reduce as ragged_gather_reduce_v1,
)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2.wrapper import (
    ragged_gather_reduce_v2,
)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import (
    ragged_gather_reduce as ragged_gather_reduce_v3,
)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import (
    token_block_alignment,
)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_v2 import ragged_gather_v2
from vllm_torchtpu.kernels.sparse_core.support import get_sparse_core_info
from vllm_torchtpu.logger import init_logger


def _get_topk_weights_dtype(default_dtype: jnp.dtype) -> jnp.dtype:
    """Returns jnp.float32 if TPU_MOE_TOPK_WEIGHTS_FLOAT32 is enabled,
    else default_dtype.
    """
    return jnp.float32 if envs.TPU_MOE_TOPK_WEIGHTS_FLOAT32 else default_dtype


logger = init_logger(__name__)


def _select_ragged_gather_reduce(version: str):
    return {
        "v1": ragged_gather_reduce_v1,
        "v2": ragged_gather_reduce_v2,
        "v3": ragged_gather_reduce_v3,
    }[version]


# The server sets its environment before importing model code, so the selected
# EP combine remains fixed for the process lifetime. The default is v2; set
# RAGGED_GATHER_REDUCE_VERSION=v3 to select the destination-major kernel for EP.
# The non-EP combine in moe_gmm selects that kernel directly.
ragged_gather_reduce = _select_ragged_gather_reduce(envs.RAGGED_GATHER_REDUCE_VERSION)

_ONEHOT_AUTO_CAP = 1200
_FAST_ONEHOT_AUTO_THRESHOLD = 5120
# Fixed at import, like the other MoE algorithm selectors.
_DENSE_EXPERT_THRESHOLD = envs.TPU_MOE_DENSE_EXPERT_THRESHOLD
# Fixed at import: build the MoE permutation from one packed sort.
_SINGLE_SORT = envs.TPU_MOE_SINGLE_SORT
# Fixed at import: let the counting sort replace the comparison sorts on the
# non-ragged paths (EP one-hot permute and non-EP TP). Off, those paths run
# the packed single sort or the three-sort form; ragged permute is unaffected.
_COUNTING_SORT = envs.TPU_MOE_COUNTING_SORT
# Routed-row cap for the one-hot EP permute and non-EP paths. Up to here the
# counting sort (or, where it declines, the packed single sort) replaces the
# three-sort form; above it the three sorts stay. Past this point the single
# sort's inverse-permutation scatter costs more than the sorts it replaces,
# and the counting sort's rank contraction does too at 256+ local experts.
_SINGLE_SORT_MAX_ROWS = 2048
# Rows per block of the counting sort's rank contraction: the triangular
# matmul is B x B, so this trades that against the number of blocks.
_COUNTING_SORT_BLOCK = 256
# Up to one counting-sort block on non-ragged paths, scattering token ids
# through the counting sort's positions is cheaper than the packed value sort
# it replaces and leaves the decode routing prep with no comparison sort.
# Ragged permute keeps the packed sort so SparseCore gather overlap is
# unchanged, and past one block the scatter costs more than that sort.
_COUNTING_SORT_SCATTER_MAX_ROWS = _COUNTING_SORT_BLOCK


_ONEHOT_MOE_PERMUTE_THRESHOLD = envs.ONEHOT_MOE_PERMUTE_THRESHOLD
_TPU_MOE_OWNER_OUTPUT_ON: bool = envs.TPU_MOE_OWNER_OUTPUT_MODE.lower() == "on"


def _prefill_moe_numerics() -> bool:
    """P/D merge role selector for the routed-GMM numerics in this file.

    True only on the prefill node (VLLM_DSV41_PURE_20L_PREFILL=1): selects the
    prefill tree's GMM tiling, W4A8/BF16-LHS fp4 handling (the
    VLLM_DSV41_MOE_SUBLANE_TILING / VLLM_DSV41_MOE_FP4_GMM1_FP8 /
    VLLM_DSV41_MOE_FP4_GMM2_FP8 / VLLM_DSV41_MOE_FP4_BF16_LHS knobs), fp8
    ragged-gather and dense-expert prequantized input. False (decode) keeps
    the deployed decode stack's behaviour exactly; in particular the decode
    launch's VLLM_DSV41_MOE_FP4_GMM1_FP8=0 must NOT switch decode onto the
    BF16-LHS path. Read at trace time; the env is fixed for the process.
    """
    return bool(envs.VLLM_DSV41_PURE_20L_PREFILL)


def resolve_onehot_permute_threshold() -> int:
    """Routed-row count at/below which MoE permute+combine run as TensorCore
    one-hot matmuls instead of the SparseCore gather/reduce kernels.

    Auto (env unset/empty/negative) resolves to ``min(SC block rows - 1,
    _ONEHOT_AUTO_CAP)`` on SparseCore TPUs and 0 elsewhere. An explicit
    ``ONEHOT_MOE_PERMUTE_THRESHOLD`` wins: 0 disables the one-hot path, a
    positive value forces that threshold.
    """
    explicit = _ONEHOT_MOE_PERMUTE_THRESHOLD
    if explicit is not None and explicit >= 0:
        return explicit
    sc_info = get_sparse_core_info()
    if sc_info is None:
        return 0
    block_rows = sc_info.num_lanes * sc_info.num_cores * sc_info.num_subcores
    threshold = min(block_rows - 1, _ONEHOT_AUTO_CAP)

    return threshold


def unpack_fp4_to_e2m1(w_packed: jax.Array) -> jax.Array:
    """Unpack a uint8-packed E2M1 weight (2 fp4 per byte along the last axis)
    into ``float4_e2m1fn`` and move the contracting axis into GMM layout. Input
    is the checkpoint layout ``[..., N, K/2]`` (output-major, packed contracting
    dim last); output is ``[..., K, N]`` to match the ``[size_group, size_k,
    size_n]`` rhs gmm_v2 expects. Run this once at weight load so the forward
    pass hands native fp4 straight to the kernel with no per-forward unpack."""
    fp4 = jax.lax.bitcast_convert_type(w_packed, jnp.float4_e2m1fn)
    fp4 = fp4.reshape(*w_packed.shape[:-1], -1)  # [..., N, K]
    return jnp.swapaxes(fp4, -1, -2)  # [..., K, N]


def requant_unpack_kmajor(
    w_packed: jax.Array, scale_f: jax.Array, block: int
) -> tuple[jax.Array, jax.Array]:
    """W4A8 requantization in JAX: unpack the checkpoint block-16 fp4 weight,
    dequantize with its fused scale, requantize to ``block`` fp4, and lay out
    K-major for gmm_v2. Input is checkpoint layout ``[..., N, K/2]`` packed
    uint8 with fused fp32 block-16 ``scale_f`` ``[..., N, K/16]``; output is
    native fp4 ``[..., K, N]`` plus the fp32 kernel scale ``[..., K/block, 1,
    N]``. Requantizing in JAX (not torch) keeps the dequantized weight off the
    host and rounds to fp4 with the cast the kernel expects."""
    fp4 = jax.lax.bitcast_convert_type(w_packed, jnp.float4_e2m1fn)
    fp4 = fp4.reshape(*w_packed.shape[:-1], -1)  # [..., N, K]
    size_k = fp4.shape[-1]
    num_in_blocks = scale_f.shape[-1]
    dequant = (
        fp4.astype(jnp.float32).reshape(
            *fp4.shape[:-1], num_in_blocks, size_k // num_in_blocks
        )
        * scale_f[..., None]
    ).reshape(*fp4.shape[:-1], size_k)
    fp4_max = float(jnp.finfo(jnp.float4_e2m1fn).max)
    blocked = dequant.reshape(*dequant.shape[:-1], size_k // block, block)
    scale = jnp.max(jnp.abs(blocked), axis=-1, keepdims=True) * (1.0 / fp4_max)
    scale_inv = jnp.where(scale == 0, 0.0, 1.0 / scale)
    requant = jnp.clip(blocked * scale_inv, -fp4_max, fp4_max).astype(jnp.float4_e2m1fn)
    requant = requant.reshape(*dequant.shape[:-1], size_k)  # [..., N, K]
    scale = scale.squeeze(-1).astype(jnp.float32)  # [..., N, K/block]
    return (
        jnp.swapaxes(requant, -1, -2),
        jnp.expand_dims(jnp.swapaxes(scale, -1, -2), -2),
    )


def quantize_to_native_fp4_kmajor(
    w: jax.Array, block: int, pack: bool = True
) -> tuple[jax.Array, jax.Array]:
    """Quantize K-major float weight ([..., K, N]) to packed uint8 e2m1 and
    per-block FP32 scale.
    Returns uint8 to safely cross the PyTorch/JAX bridge before unpacking in gmm_v2.

    ``pack=False`` returns the same values as ``float4_e2m1fn`` [..., K, N]
    instead -- what the fused EP MoE kernel's FP4 form reads, and what torch
    receives as a native ``torch.float4_e2m1fn_x2`` tensor. Only the transport
    differs: the quantization above is one cast either way, so the two forms
    carry bit-identical weights and a caller may choose per layer.

    TODO: Make a more generic version of this function;
    possibly combine with quantize_tensor_to_fp4
    """

    size_k = w.shape[-2]
    num_blocks = size_k // block
    blocked = w.astype(jnp.float32).reshape(
        *w.shape[:-2], num_blocks, block, w.shape[-1]
    )
    fp4_max = float(jnp.finfo(jnp.float4_e2m1fn).max)
    abs_max = jnp.max(jnp.abs(blocked), axis=-2, keepdims=True)
    scale = abs_max / fp4_max
    scale_inv = jnp.where(scale == 0, 0.0, 1.0 / scale)
    quantized = jnp.clip(blocked * scale_inv, -fp4_max, fp4_max).astype(
        jnp.float4_e2m1fn
    )
    quantized = quantized.reshape(*w.shape[:-2], size_k, w.shape[-1])  # [..., K, N]

    scale = scale.astype(jnp.float32)  # [..., num_blocks, 1, N]
    if not pack:
        return quantized, scale

    # Real e2m1 bit-packing (2 values/byte along K), matching gmm_v2's own
    # should_unpack/bitcast(quant_dtype) unpacking convention.
    pairs = quantized.reshape(*w.shape[:-2], size_k // 2, 2, w.shape[-1])
    pairs = jnp.swapaxes(pairs, -1, -2)  # [..., K/2, N, 2] -- pair axis last
    packed = jax.lax.bitcast_convert_type(pairs, jnp.uint8)  # [..., K/2, N]

    return packed, scale


def _moe_gmm_tiling(
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    fuse_act: str | None = None,
) -> TileSizes:
    tiles = calculate_tiling(
        dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act=fuse_act
    )
    if (
        dims.size_group > 1
        and rhs_cfgs.should_dequantize_before_matmul
        and dims.size_m >= 128
        and 128 % dims.size_lhs_sublane == 0
    ):
        rows_per_group = dims.size_m // dims.size_group
        if rows_per_group <= 64:
            tile_m = 128 if dims.size_m < 256 else 256
            return TileSizes(
                tile_m=tile_m,
                tile_k=tiles.tile_k,
                tile_n=tiles.tile_n,
                bucket_base=128,
            )
        if rows_per_group <= 160:
            return TileSizes(
                tile_m=min(256, tiles.tile_m),
                tile_k=tiles.tile_k,
                tile_n=tiles.tile_n,
                bucket_base=64,
            )
        if (
            rows_per_group <= (600 if dims.size_k <= 512 else 384)
            and dims.size_m >= 512
            and tiles.tile_m >= 256
        ):
            return TileSizes(
                tile_m=512,
                tile_k=tiles.tile_k,
                tile_n=tiles.tile_n,
                bucket_base=128,
            )
        if rows_per_group <= 768 and tiles.tile_m >= 256:
            return TileSizes(
                tile_m=256,
                tile_k=tiles.tile_k,
                tile_n=tiles.tile_n,
                bucket_base=128,
            )
        return tiles
    if dims.size_k <= 512 and dims.size_group > 1:
        if 512 <= dims.size_m <= 1024 and 128 % dims.size_lhs_sublane == 0:
            return TileSizes(
                tile_m=128,
                tile_k=tiles.tile_k,
                tile_n=tiles.tile_n,
                bucket_base=64,
            )
    if not _prefill_moe_numerics():
        # Decode stack tiling (unchanged from the deployed decode tree).
        if (
            dims.size_group > 1
            and 32 <= dims.size_m <= dims.size_group * 64
            and 32 % dims.size_lhs_sublane == 0
        ):
            return TileSizes(
                tile_m=32,
                tile_k=tiles.tile_k,
                tile_n=tiles.tile_n,
                bucket_base=32,
            )
        if (
            dims.size_group > 1
            and 64 * dims.size_group < dims.size_m <= 165 * dims.size_group
            and 64 % dims.size_lhs_sublane == 0
        ):
            return TileSizes(
                tile_m=128 if dims.size_m <= 104 * dims.size_group else 256,
                tile_k=tiles.tile_k,
                tile_n=tiles.tile_n,
                bucket_base=64,
            )
        return tiles
    _sub_tile = bool(envs.VLLM_DSV41_MOE_SUBLANE_TILING)
    if (
        dims.size_group > 1
        and 32
        <= dims.size_m
        <= (max(dims.size_group * 100, 1024) if _sub_tile else dims.size_group * 40)
        and 32 % dims.size_lhs_sublane == 0
    ):
        return TileSizes(
            tile_m=32,
            tile_k=tiles.tile_k,
            tile_n=tiles.tile_n,
            bucket_base=32,
        )
    if (
        _sub_tile
        and dims.size_group > 1
        and max(dims.size_group * 100, 1024) < dims.size_m <= 256 * dims.size_group
        and 64 % dims.size_lhs_sublane == 0
    ):
        return TileSizes(
            tile_m=64,
            tile_k=tiles.tile_k,
            tile_n=tiles.tile_n,
            bucket_base=64,
        )
    return tiles


def gmm_wrapper(
    lhs,
    rhs,
    rhs_scale,
    rhs_bias,
    group_sizes,
    group_offset,
    zero_initialize=False,
    fuse_act=None,
    preferred_element_type=None,
    rhs_quant_dtype=None,
    lhs_scale=None,
    out_row_scale=None,
    emit_quantized_out=False,
):
    # fp4 weights: keep bf16 activations. Quantizing activations to fp8 collapses
    # fp4 accuracy (error compounds across MoE layers) with no decode speedup
    # (decode is weight-HBM-bound). fp8/int4 weights keep the default fp8 act.
    if _prefill_moe_numerics():
        is_fp4_weight = (
            jnp.issubdtype(rhs.dtype, jnp.floating)
            and jax.dtypes.itemsize_bits(rhs.dtype) == 4
        ) or rhs_quant_dtype == jnp.float4_e2m1fn
        use_fp8_lhs = (not is_fp4_weight) or (lhs_scale is not None)
    else:
        # Decode stack: only block < 128 fp4 (W4A16) keeps bf16 activations;
        # block-512 requantized fp4 runs W4A8 with in-kernel lhs quantization.
        _rhs_blk = (
            (rhs.shape[1] * get_packing_factor(rhs.dtype, rhs_quant_dtype))
            // rhs_scale.shape[1]
            if rhs_scale is not None
            else 0
        )
        is_fp4_w4a16 = (
            jnp.issubdtype(rhs.dtype, jnp.floating)
            and jax.dtypes.itemsize_bits(rhs.dtype) == 4
            and _rhs_blk < 128
        )
        use_fp8_lhs = not is_fp4_w4a16
    return gmm_v2(
        lhs=lhs,
        rhs=rhs,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_sizes=group_sizes,
        group_offset=group_offset[0],
        tile_info=_moe_gmm_tiling,
        zero_initialize=zero_initialize,
        fuse_act=fuse_act,
        preferred_element_type=preferred_element_type,
        maybe_quantize_lhs=use_fp8_lhs,
        rhs_quant_dtype=rhs_quant_dtype,
        lhs_scale=lhs_scale,
        out_row_scale=out_row_scale,
        emit_quantized_out=emit_quantized_out,
    )


def _counting_sort_positions(
    sort_keys: jax.Array, num_buckets: int
) -> tuple[jax.Array | None, jax.Array | None]:
    """Stable counting sort: row -> its position in the sorted order.

    `jnp.argsort(jnp.argsort(keys))` spends a second comparison sort over M
    distinct values on what is really the inverse of the first permutation.
    The keys take only `num_buckets` values, so the rank within a bucket comes
    from a triangular matmul on the one-hot block -- MXU work proportional to
    M*block*num_buckets instead of a length-M sort -- and the bucket totals
    fall out of the same histogram, which is exactly `group_sizes`.
    """
    m = sort_keys.shape[0]
    block = min(128 if m > 8192 else _COUNTING_SORT_BLOCK, m)
    if num_buckets > block:
        # XLA miscompiles the rank matmul here on jaxlib 0.10.2 + libtpu
        # 0.0.44.1: under jit the positions come back not a permutation. Only
        # num_buckets > block is affected, so that is the excluded region.
        return None, None
    pad = -m % block
    if pad:
        # The sentinel bucket sorts after every real one, so padding into it
        # leaves the real bucket bases untouched; the caller slices the
        # sentinel off `counts` anyway.
        sort_keys = jnp.pad(sort_keys, (0, pad), constant_values=num_buckets - 1)
    onehot = jax.nn.one_hot(
        sort_keys.reshape(-1, block), num_buckets, dtype=jnp.bfloat16
    )
    # Compute the block histogram with its own contraction and fuse
    # `(rank + base[:, None, :]) * onehot` into a single reduction so XLA fuses
    # it directly into the rank einsum's kOutput fusion without materializing
    # the 3D tensor to HBM. Exact: 0/1 summands, at most `block` of them.
    ones = jnp.ones((block,), jnp.bfloat16)
    hist = jnp.einsum("j,bje->be", ones, onehot, preferred_element_type=jnp.float32)
    # Prefix sums via MXU triangular contractions avoid VPU `reduce-window`
    # scans (`jnp.cumsum`), and single-block inputs skip the block prefix.
    upper_buckets = jnp.triu(jnp.ones((num_buckets, num_buckets), jnp.bfloat16), 1)
    if hist.shape[0] == 1:
        counts = hist[0]
        base = jnp.einsum(
            "e,ef->f",
            counts,
            upper_buckets,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )[None, :]
    else:
        counts = hist.sum(axis=0)
        bucket_offset = jnp.einsum(
            "e,ef->f",
            counts,
            upper_buckets,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        if hist.shape[0] <= _COUNTING_SORT_BLOCK:
            lower_blocks = jnp.tril(
                jnp.ones((hist.shape[0], hist.shape[0]), jnp.bfloat16), -1
            )
            block_offset = jnp.einsum(
                "ab,be->ae",
                lower_blocks,
                hist,
                precision=jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
        else:
            block_offset = jnp.cumsum(hist, axis=0) - hist
        base = block_offset + bucket_offset
    lower = jnp.tril(jnp.ones((block, block), jnp.bfloat16), -1)
    rank = jnp.einsum("ij,bje->bie", lower, onehot, preferred_element_type=jnp.float32)
    pos = ((rank + base[:, None, :]) * onehot).sum(axis=2).astype(jnp.int32)
    return pos.reshape(-1)[:m].astype(jnp.int32), counts.astype(jnp.int32)


def _single_sort_permutation(
    sort_keys: jax.Array, topk: int, flat_bits: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Returns the stable argsort of `sort_keys`, its inverse, and token ids.

    Packing (key, flat position) into one int32 makes a single unstable sort
    order entries like a stable argsort. The inverse is a scatter, and flat
    position p belongs to token p // topk.
    """
    pos = jnp.arange(sort_keys.shape[0], dtype=jnp.int32)
    packed = jax.lax.sort(jnp.left_shift(sort_keys, flat_bits) | pos, is_stable=False)
    sorted_indices = jnp.bitwise_and(packed, (1 << flat_bits) - 1)
    revert = jnp.zeros_like(pos).at[sorted_indices].set(pos, unique_indices=True)
    return sorted_indices, revert, sorted_indices // topk


def prepare_routed_gmm_inputs(
    hidden_states_local: jax.Array,
    topk_indices_local: jax.Array,
    topk_weights_local: jax.Array,
    *,
    local_num_experts: int,
    topk: int,
    use_ep: bool,
    use_sparse_core: bool,
    onehot_moe_permute_threshold: int = 0,
    skip_padded_tokens: bool = False,
    return_sorted_indices: bool = False,
    return_onehot_combine: bool = False,
):
    """Prepare the local-only routing layout using already-local expert ids.

    For non-EP runs `topk_indices_local` is always non-negative, so the
    valid-mask path computed below is a no-op (jnp.where on an all-True
    mask). For EP, the caller applies the expert-map remap upstream
    (inside this JIT) and marks non-local entries with a negative id which
    the mask filters out. Always running the masked branch keeps a single
    code path.
    """
    num_tokens_local = hidden_states_local.shape[0]
    topk_indices_flat = topk_indices_local.flatten()
    topk_weights_flat = topk_weights_local.flatten()
    token_indices_flat = jnp.arange(num_tokens_local, dtype=jnp.int32).repeat(topk)

    if skip_padded_tokens:
        # Padded tokens keep their selected expert ids, but carry zero
        # expert routing weights. Treat them as invalid to remove them
        # from GMM work.
        valid_mask = (topk_indices_flat >= 0) & (topk_weights_flat != 0)
    else:
        valid_mask = topk_indices_flat >= 0
    sort_keys = jnp.where(valid_mask, topk_indices_flat, local_num_experts)
    m = sort_keys.shape[0]
    token_bits = max(1, (num_tokens_local - 1).bit_length())

    if (
        not use_ep
        and m <= max(onehot_moe_permute_threshold, 768)
        and hidden_states_local.dtype == jnp.bfloat16
        and not skip_padded_tokens
    ):
        # Single uint32 packed radix sort + MXU one-hot permute avoids the
        # second inverse sort and dynamic 2D gathers while preserving exact
        # route-order BF16 combine bit-for-bit.
        idx_bits = max(1, (m - 1).bit_length())
        idx_mask = (1 << idx_bits) - 1
        iota = jnp.arange(m, dtype=jnp.int32)
        packed = jax.lax.sort(
            jnp.left_shift(topk_indices_flat.astype(jnp.uint32), idx_bits)
            | iota.astype(jnp.uint32),
            is_stable=False,
        )
        sorted_indices = jnp.bitwise_and(packed, idx_mask).astype(jnp.int32)
        token_indices_sorted = sorted_indices // topk
        group_sizes_local = (
            jax.nn.one_hot(topk_indices_flat, local_num_experts, dtype=jnp.float32)
            .sum(axis=0)
            .astype(jnp.int32)
        )
        onehot = jax.nn.one_hot(
            token_indices_sorted, num_tokens_local, dtype=hidden_states_local.dtype
        )
        x = onehot @ hidden_states_local
        if return_onehot_combine:
            w_sorted_non_ep = jnp.where(
                valid_mask[sorted_indices],
                topk_weights_flat[sorted_indices].astype(hidden_states_local.dtype),
                0,
            )
            return (
                x,
                group_sizes_local,
                sorted_indices,
                topk_weights_flat,
                valid_mask,
                token_indices_sorted,
                sorted_indices,
                onehot,
                w_sorted_non_ep,
            )
        if return_sorted_indices:
            return (
                x,
                group_sizes_local,
                sorted_indices,
                topk_weights_flat,
                valid_mask,
                token_indices_sorted,
                sorted_indices,
            )
        return (
            x,
            group_sizes_local,
            sorted_indices,
            topk_weights_flat,
            valid_mask,
            token_indices_sorted,
        )

    fast_onehot = (
        return_onehot_combine
        and use_ep
        and use_sparse_core
        and m <= onehot_moe_permute_threshold
        and hidden_states_local.dtype in (jnp.bfloat16, jnp.int8)
        and topk_weights_flat.dtype == jnp.bfloat16
        and (local_num_experts + 1).bit_length() + token_bits <= 17
        and not _TPU_MOE_OWNER_OUTPUT_ON
    )
    if fast_onehot:
        # Pack (sort_key, token_idx, 15-bit non-negative bf16_weight) into a
        # single uint32 so one radix sort yields both `token_indices_sorted`
        # and `w_sorted` up to 17 key+token bits without dynamic gathers.
        group_sizes_local = (
            jax.nn.one_hot(sort_keys, local_num_experts, dtype=jnp.float32)
            .sum(axis=0)
            .astype(jnp.int32)
        )
        w_masked = jnp.where(
            valid_mask & (topk_weights_flat > 0),
            topk_weights_flat,
            jnp.zeros_like(topk_weights_flat),
        )
        w_u15 = jnp.bitwise_and(
            jax.lax.bitcast_convert_type(w_masked, jnp.uint16).astype(jnp.uint32),
            jnp.uint32(0x7FFF),
        )
        packed = jax.lax.sort(
            jnp.left_shift(sort_keys.astype(jnp.uint32), 15 + token_bits)
            | jnp.left_shift(token_indices_flat.astype(jnp.uint32), 15)
            | w_u15,
            is_stable=False,
        )
        token_indices_sorted = jnp.bitwise_and(
            jnp.right_shift(packed, 15), (1 << token_bits) - 1
        ).astype(jnp.int32)
        w_sorted = jax.lax.bitcast_convert_type(
            jnp.bitwise_and(packed, jnp.uint32(0x7FFF)).astype(jnp.uint16),
            jnp.bfloat16,
        )
        onehot = jax.nn.one_hot(
            token_indices_sorted, num_tokens_local, dtype=jnp.bfloat16
        )
        if hidden_states_local.dtype == jnp.int8:
            x = (onehot.astype(jnp.int8) @ hidden_states_local).astype(jnp.int8)
        else:
            if skip_padded_tokens:
                active_rows = (topk_weights_local != 0).any(axis=-1, keepdims=True)
                hidden_states_clean = jnp.where(active_rows, hidden_states_local, 0)
            else:
                hidden_states_clean = jnp.where(
                    jnp.isfinite(hidden_states_local), hidden_states_local, 0
                )
            x = onehot @ hidden_states_clean
        return (
            x,
            group_sizes_local,
            token_indices_sorted,
            topk_weights_flat,
            valid_mask,
            token_indices_sorted,
            None,
            onehot,
            w_sorted,
        )

    # The keys take only `local_num_experts + 1` values, so a counting sort
    # gives the positions and `group_sizes` without a second comparison sort:
    # on the ragged permute at any row count (as before), and -- unless
    # TPU_MOE_COUNTING_SORT stands it down -- on the one-hot and non-EP paths
    # within the single sort's envelope, where it replaces that sort. On
    # non-ragged paths within `_COUNTING_SORT_SCATTER_MAX_ROWS`, scattering
    # `token_indices_flat` by those positions also removes the packed value
    # sort. The non-EP path past `_COUNTING_SORT_BLOCK` rows takes the
    # counting sort as well (decode-stack tuning) instead of the inverse sort.
    ragged_permute = (
        use_ep
        and use_sparse_core
        and token_indices_flat.shape[0] > onehot_moe_permute_threshold
    )
    counting_sort = (
        ragged_permute
        or (_COUNTING_SORT and token_indices_flat.shape[0] <= _SINGLE_SORT_MAX_ROWS)
        or (
            not (use_ep and use_sparse_core)
            and token_indices_flat.shape[0] > _COUNTING_SORT_BLOCK
        )
    )
    argsort_revert_indices, bucket_counts = (
        _counting_sort_positions(sort_keys, local_num_experts + 1)
        if counting_sort
        else (None, None)
    )
    packable = (local_num_experts + 1).bit_length() + token_bits < 31
    flat_bits = max(1, (token_indices_flat.shape[0] - 1).bit_length())
    single_sort = (
        _SINGLE_SORT
        and argsort_revert_indices is None
        and not ragged_permute
        and token_indices_flat.shape[0] <= _SINGLE_SORT_MAX_ROWS
        and (local_num_experts + 1).bit_length() + flat_bits < 31
    )
    # Outside the single sort's envelope, two packed sorts replace the
    # argsort + inverse-argsort pair. The second packed sort shifts indices
    # < 2**flat_bits by flat_bits.
    idx_packable = (
        local_num_experts + 1
    ).bit_length() + flat_bits < 31 and 2 * flat_bits < 31
    token_indices_sorted: jax.Array | None = None
    if (
        argsort_revert_indices is not None
        and not ragged_permute
        and token_indices_flat.shape[0] <= _COUNTING_SORT_SCATTER_MAX_ROWS
    ):
        token_indices_sorted = (
            jnp.zeros_like(token_indices_flat)
            .at[argsort_revert_indices]
            .set(token_indices_flat, unique_indices=True)
        )
    sorted_indices = (
        jnp.argsort(sort_keys)
        if token_indices_sorted is None
        and not single_sort
        and not (argsort_revert_indices is None and idx_packable)
        and (argsort_revert_indices is None or not packable)
        else None
    )

    if argsort_revert_indices is not None:
        group_sizes_local = bucket_counts[:local_num_experts]
    else:
        if single_sort:
            logger.info_once("MoE permutation runs as one packed sort.")
            sorted_indices, argsort_revert_indices, token_indices_sorted = (
                _single_sort_permutation(sort_keys, topk, flat_bits)
            )
        elif idx_packable:
            idx_mask = (1 << flat_bits) - 1
            iota = jnp.arange(m, dtype=jnp.int32)
            packed1 = jax.lax.sort(
                jnp.left_shift(sort_keys, flat_bits) | iota, is_stable=False
            )
            sorted_indices = jnp.bitwise_and(packed1, idx_mask)
            packed2 = jax.lax.sort(
                jnp.left_shift(sorted_indices, flat_bits) | iota, is_stable=False
            )
            argsort_revert_indices = jnp.bitwise_and(packed2, idx_mask)
        else:
            argsort_revert_indices = jnp.argsort(sorted_indices)
        topk_indices_for_count = jnp.where(valid_mask, topk_indices_flat, 0)
        group_sizes_local = jax.nn.one_hot(
            topk_indices_for_count, local_num_experts, dtype=jnp.int32
        ) * valid_mask[:, None].astype(jnp.int32)
        group_sizes_local = group_sizes_local.sum(axis=0)

    if token_indices_sorted is None:
        if sorted_indices is not None:
            # Flat position p belongs to token p // topk: no gather needed.
            token_indices_sorted = sorted_indices // topk
        elif packable:
            # Packing (key, token) into one int32 keeps the payload out of the sort
            # and drops the follow-up gather. Equal packed values carry the same
            # token id, so their order cannot matter.
            packed = jax.lax.sort(
                jnp.left_shift(sort_keys, token_bits) | token_indices_flat,
                is_stable=False,
            )
            token_indices_sorted = jnp.bitwise_and(packed, (1 << token_bits) - 1)
        else:
            token_indices_sorted = token_indices_flat[sorted_indices]

    onehot = None
    if use_ep and use_sparse_core:
        if token_indices_sorted.shape[0] <= onehot_moe_permute_threshold:
            # Use one-hot matmul for permutation, which can be faster for small
            # batch size. When `argsort_revert_indices` came from the counting
            # sort, build the `[M, T]` permutation directly from it so
            # `fused_moe_gmm` does not depend on `token_indices_sorted` (letting
            # XLA DCE its scatter unless owner-output combine uses it).
            if bucket_counts is not None:
                onehot = (
                    jax.nn.one_hot(
                        argsort_revert_indices.reshape(num_tokens_local, topk),
                        argsort_revert_indices.size,
                        dtype=hidden_states_local.dtype,
                    )
                    .sum(axis=1)
                    .T
                )
            else:
                onehot = jax.nn.one_hot(
                    token_indices_sorted,
                    num_tokens_local,
                    dtype=hidden_states_local.dtype,
                )
            if skip_padded_tokens:
                active_rows = (topk_weights_local != 0).any(axis=-1, keepdims=True)
                hidden_states_clean = jnp.where(active_rows, hidden_states_local, 0)
            else:
                hidden_states_clean = jnp.where(
                    jnp.isfinite(hidden_states_local), hidden_states_local, 0
                )
            x = onehot @ hidden_states_clean
        else:
            # Match uLLM/reference EP routing: materialize the valid local
            # expert prefix through SparseCore ragged_gather. Invalid
            # non-local rows are sorted after the valid prefix.
            valid_count = group_sizes_local.sum(dtype=jnp.int32)
            if (
                hidden_states_local.dtype == jnp.float8_e4m3fn
                and _prefill_moe_numerics()
            ):
                x = jax.lax.bitcast_convert_type(
                    ragged_gather_v2(
                        jax.lax.bitcast_convert_type(hidden_states_local, jnp.int8),
                        token_indices_sorted,
                        jnp.array([0], dtype=jnp.int32),
                        valid_count[None],
                    ),
                    jnp.float8_e4m3fn,
                )
            else:
                x = ragged_gather_v2(
                    hidden_states_local,
                    token_indices_sorted,
                    jnp.array([0], dtype=jnp.int32),
                    valid_count[None],
                )
    elif (
        use_sparse_core
        and 3072 < m <= (32768 if _prefill_moe_numerics() else 16384)
        and hidden_states_local.dtype
        in (
            (jnp.bfloat16, jnp.float32, jnp.int8, jnp.int4, jnp.float8_e4m3fn)
            if _prefill_moe_numerics()
            else (jnp.bfloat16, jnp.float32, jnp.int8, jnp.int4)
        )
        and hidden_states_local.shape[-1] >= 128
        and hidden_states_local.shape[-1] % 128 == 0
        and hidden_states_local.shape[0]
        % (32 // jax.dtypes.itemsize_bits(hidden_states_local.dtype))
        == 0
    ):
        end_count = (
            jnp.array([m], dtype=jnp.int32)
            if (not use_ep and not skip_padded_tokens)
            else group_sizes_local.sum(dtype=jnp.int32)[None]
        )
        if hidden_states_local.dtype == jnp.float8_e4m3fn:
            x = jax.lax.bitcast_convert_type(
                ragged_gather_v2(
                    jax.lax.bitcast_convert_type(hidden_states_local, jnp.int8),
                    token_indices_sorted,
                    jnp.array([0], dtype=jnp.int32),
                    end_count,
                ),
                jnp.float8_e4m3fn,
            )
        else:
            x = ragged_gather_v2(
                hidden_states_local,
                token_indices_sorted,
                jnp.array([0], dtype=jnp.int32),
                end_count,
            )
    else:
        x = hidden_states_local[token_indices_sorted]
    if return_onehot_combine:
        return (
            x,
            group_sizes_local,
            argsort_revert_indices,
            topk_weights_flat,
            valid_mask,
            token_indices_sorted,
            sorted_indices,
            onehot,
            None,
        )
    if return_sorted_indices:
        return (
            x,
            group_sizes_local,
            argsort_revert_indices,
            topk_weights_flat,
            valid_mask,
            token_indices_sorted,
            sorted_indices,
        )
    return (
        x,
        group_sizes_local,
        argsort_revert_indices,
        topk_weights_flat,
        valid_mask,
        token_indices_sorted,
    )


def moe_gmm(
    x: jax.Array,
    w1: jax.Array,
    w1_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2: jax.Array,
    w2_scale: jax.Array | None,
    w2_bias: jax.Array | None,
    group_sizes: jax.Array,
    argsort_revert_indices: jax.Array,
    topk_weights_flat: jax.Array,
    valid_mask_flat: jax.Array,
    token_indices_sorted: jax.Array,
    *,
    activation: str,
    num_tokens: int,
    topk: int,
    use_ep: bool,
    use_sparse_core: bool,
    onehot_moe_permute_threshold: int = 0,
    rhs_quant_dtype: jnp.dtype | None = None,
    sorted_indices: jax.Array | None = None,
    lhs_scale: jax.Array | None = None,
    onehot_permute: jax.Array | None = None,
    w_sorted: jax.Array | None = None,
) -> jax.Array:
    """Run grouped GEMM for routed tokens and reduce back to tokens.

    The valid-mask gating below is a no-op for non-EP runs (mask is all
    True) and keeps the single code path uniform for both EP and non-EP.
    EP uses SparseCore ragged_gather + ragged_gather_reduce for token movement.
    With `lhs_scale`, `x` holds rows pre-quantized by quantize_lhs_rows and the
    output keeps the unquantized dtype, `lhs_scale.dtype`.
    """
    group_offset = jnp.array([0], dtype=jnp.int32)
    out_dtype = x.dtype if lhs_scale is None else lhs_scale.dtype

    packing_factor = get_packing_factor(w2.dtype, rhs_quant_dtype)
    inter_dim = w2.shape[1] * packing_factor
    _prefill = _prefill_moe_numerics()
    if not _prefill:
        # Decode stack: block >= 128 fp4 is treated like fp8 (W4A8 both GMMs);
        # only block < 128 fp4 dequantizes the pre-quantized lhs to bf16.
        _w2_blk = (
            ((w2.shape[1] * packing_factor) // w2_scale.shape[1])
            if w2_scale is not None
            else 0
        )
        is_w2_fp4 = (
            rhs_quant_dtype == jnp.float4_e2m1fn
            or (
                jnp.issubdtype(w2.dtype, jnp.floating)
                and jax.dtypes.itemsize_bits(w2.dtype) == 4
            )
        ) and (_w2_blk < 128)
        use_w4a8_gmm2 = False
        if is_w2_fp4 and lhs_scale is not None:
            x = (
                (x.reshape(x.shape[0], lhs_scale.shape[1], -1) * lhs_scale[:, :, None])
                .reshape(x.shape)
                .astype(jnp.bfloat16)
            )
            lhs_scale = None
    else:
        is_w2_fp4 = (
            jnp.issubdtype(w2.dtype, jnp.floating)
            and jax.dtypes.itemsize_bits(w2.dtype) == 4
        ) or rhs_quant_dtype == jnp.float4_e2m1fn
        use_w4a8 = bool(envs.VLLM_DSV41_MOE_FP4_GMM1_FP8)
        use_w4a8_gmm2 = bool(envs.VLLM_DSV41_MOE_FP4_GMM2_FP8)
    if (
        _prefill
        and (is_w2_fp4 or w1.dtype == jnp.float4_e2m1fn)
        and lhs_scale is not None
    ):
        n_blocks_k1 = x.shape[1] // 512
        if (
            use_w4a8
            and x.shape[1] % 512 == 0
            and lhs_scale.shape[1] in (1, n_blocks_k1)
        ):
            lhs_scale = jnp.broadcast_to(
                lhs_scale.astype(jnp.bfloat16), (x.shape[0], n_blocks_k1)
            )
        else:
            if lhs_scale.shape[1] == 1:
                x = (x.astype(jnp.float32) * lhs_scale).astype(lhs_scale.dtype)
            else:
                nb = lhs_scale.shape[1]
                x = (
                    (
                        x.reshape(x.shape[0], nb, x.shape[1] // nb).astype(jnp.float32)
                        * lhs_scale[:, :, None]
                    )
                    .reshape(x.shape)
                    .astype(lhs_scale.dtype)
                )
            lhs_scale = None
    fuse_gmm1_fp8 = not is_w2_fp4 and inter_dim == 512
    gmm1_res = gmm_wrapper(
        x,
        w1,
        w1_scale,
        w1_bias,
        group_sizes,
        group_offset,
        zero_initialize=False,
        fuse_act=activation,
        rhs_quant_dtype=rhs_quant_dtype,
        lhs_scale=lhs_scale,
        emit_quantized_out=fuse_gmm1_fp8,
    )
    if fuse_gmm1_fp8:
        u8_pair = jax.lax.bitcast_convert_type(
            gmm1_res[:, : inter_dim // 2], jnp.float8_e4m3fn
        )
        gmm1_in = jnp.concatenate([u8_pair[..., 0], u8_pair[..., 1]], axis=1)
        gmm1_scale = gmm1_res[:, inter_dim // 2 : inter_dim // 2 + 1]
    elif (
        w2_scale is not None
        and inter_dim % 128 == 0
        and (
            (not is_w2_fp4 and gmm1_res.shape[0] <= 4608)
            or (is_w2_fp4 and use_w4a8_gmm2 and inter_dim % 512 == 0)
        )
    ):
        gmm1_in, gmm1_scale = quantize_lhs_rows(gmm1_res[:, :inter_dim])
    else:
        gmm1_in = gmm1_res[:, :inter_dim]
        gmm1_scale = None
    if (
        gmm1_scale is not None
        and is_w2_fp4
        and w2_scale is not None
        and gmm1_scale.shape != (gmm1_in.shape[0], w2_scale.shape[1])
    ):
        gmm1_scale = jnp.broadcast_to(
            gmm1_scale.reshape(-1, 1), (gmm1_in.shape[0], w2_scale.shape[1])
        )

    use_direct_non_ep_combine = (
        not use_ep
        and onehot_permute is not None
        and w_sorted is not None
        and sorted_indices is not None
        and argsort_revert_indices is sorted_indices
        and sorted_indices.size <= max(onehot_moe_permute_threshold, 768)
    )
    gmm2_out_scale = w_sorted[:, None] if use_direct_non_ep_combine else None

    gmm2_res = gmm_wrapper(
        gmm1_in,
        w2,
        w2_scale,
        w2_bias,
        group_sizes,
        group_offset,
        zero_initialize=False,
        preferred_element_type=out_dtype,
        rhs_quant_dtype=rhs_quant_dtype,
        lhs_scale=gmm1_scale,
        out_row_scale=gmm2_out_scale,
    )
    # The destination-major kernel packs source indices into 20 bits and
    # processes 16-lane SparseCore tiles.
    use_local_ragged_combine = (
        # Local SparseCore path; route-pair metadata needs at least two routes.
        use_sparse_core
        and not use_ep
        and topk >= 2
        # TensorCore combine handles top-8 batches.
        and topk != 8
        # Honor the configured SparseCore dispatch cutoff.
        and argsort_revert_indices.size > onehot_moe_permute_threshold
        # BF16 kernel input.
        and gmm2_res.dtype == jnp.bfloat16
        # Packed source-index capacity.
        and gmm2_res.shape[0] <= 1 << 20
    )
    if use_local_ragged_combine:
        try:
            tpu_info = pltpu.get_tpu_info()
        except ValueError:
            # get_tpu_info raises for unsupported device kinds (e.g. CPU hosts).
            tpu_info = None
        sc_info = get_sparse_core_info(tpu_info)
        # Lane-aligned width, 16-lane SparseCore, and v3's own cutoff (two
        # BF16 buffers use at least 60% of VMEM).
        if (
            sc_info is not None
            and sc_info.num_lanes == 16
            and gmm2_res.shape[-1] % tpu_info.num_lanes == 0
            and gmm2_res.size * 4 >= tpu_info.vmem_capacity_bytes * 0.6
        ):
            # The kernel pads destination tokens to one 64-token block per
            # SparseCore row partition: require at least one full block and
            # padding of at most half the batch.
            alignment = token_block_alignment(gmm2_res.shape[-1], tpu_info)
            padded_tokens = -(-num_tokens // alignment) * alignment
            if num_tokens >= alignment and padded_tokens * 2 <= num_tokens * 3:
                return ragged_gather_reduce_v3(
                    gmm2_res,
                    argsort_revert_indices,
                    topk_weights_flat,
                    valid_mask_flat,
                    reduce_group_size=topk,
                ).astype(out_dtype)
    if use_ep and use_sparse_core:
        if argsort_revert_indices.size <= onehot_moe_permute_threshold:
            if onehot_permute is not None and w_sorted is not None:
                computed = jnp.arange(
                    gmm2_res.shape[0], dtype=jnp.int32
                ) < group_sizes.sum(dtype=jnp.int32)
                gmm2_res = jnp.where(computed[:, None], gmm2_res, 0)
                combine = onehot_permute.T * w_sorted.astype(gmm2_res.dtype)[None, :]
                return (combine @ gmm2_res).astype(out_dtype)
            # Use onehot + matmul for unpermutation, which can be faster
            # for small batch size.
            # `argsort_revert_indices` is the inverse of the sort, so scattering
            # through it is the gather by the (unmaterialised) forward one.
            if sorted_indices is not None:
                topk_weights_sorted = topk_weights_flat[sorted_indices]
                valid_mask_sorted = valid_mask_flat[sorted_indices]
            else:
                topk_weights_sorted = (
                    jnp.zeros_like(topk_weights_flat)
                    .at[argsort_revert_indices]
                    .set(topk_weights_flat)
                )
                valid_mask_sorted = (
                    jnp.zeros_like(valid_mask_flat)
                    .at[argsort_revert_indices]
                    .set(valid_mask_flat)
                )
            valid_count = group_sizes.sum(dtype=jnp.int32)[None]
            if _TPU_MOE_OWNER_OUTPUT_ON and can_use_blockwise_onehot_unpermute(
                gmm2_res,
                token_indices_sorted,
                topk_weights_sorted,
                valid_count,
                num_tokens=num_tokens,
            ):
                logger.info_once(
                    "Selected owner_output blockwise one-hot unpermute: "
                    "routes=%d hidden=%d tokens=%d",
                    gmm2_res.shape[0],
                    gmm2_res.shape[1],
                    num_tokens,
                )
                return blockwise_onehot_unpermute(
                    gmm2_res,
                    token_indices_sorted,
                    topk_weights_sorted,
                    valid_count,
                    num_tokens=num_tokens,
                ).astype(out_dtype)
            # Rows past the computed prefix (non-local experts under EP,
            # skipped padded tokens) are never written by the GMM; zero them
            # so stale NaNs cannot spread through the matmul (0 * NaN == NaN).
            computed = jnp.arange(gmm2_res.shape[0], dtype=jnp.int32) < group_sizes.sum(
                dtype=jnp.int32
            )
            gmm2_res = jnp.where(computed[:, None], gmm2_res, 0)
            w_sorted = jnp.where(
                valid_mask_sorted,
                topk_weights_sorted.astype(gmm2_res.dtype),
                0,
            )
            onehot = (
                onehot_permute
                if onehot_permute is not None and onehot_permute.dtype == gmm2_res.dtype
                else jax.nn.one_hot(
                    token_indices_sorted, num_tokens, dtype=gmm2_res.dtype
                )
            )
            combine = onehot.T * w_sorted[None, :]
            return (combine @ gmm2_res).astype(out_dtype)
        return ragged_gather_reduce(
            gmm2_res,
            argsort_revert_indices,
            topk_weights_flat,
            valid_mask_flat,
            reduce_group_size=topk,
        ).astype(out_dtype)

    if use_direct_non_ep_combine:
        return (onehot_permute.T @ gmm2_res).astype(out_dtype)

    topk_weights = topk_weights_flat.reshape((num_tokens, topk))
    valid_mask = valid_mask_flat.reshape((num_tokens, topk))
    if (
        sorted_indices is not None
        and argsort_revert_indices is sorted_indices
        and sorted_indices.size <= max(onehot_moe_permute_threshold, 768)
    ):
        inv_onehot = jax.nn.one_hot(
            sorted_indices, sorted_indices.size, dtype=gmm2_res.dtype
        )
        token_hidden = inv_onehot.T @ gmm2_res
    else:
        token_hidden = gmm2_res[argsort_revert_indices]
    token_topk_hidden = token_hidden.reshape((num_tokens, topk, gmm2_res.shape[-1]))
    token_topk_hidden = token_topk_hidden * jnp.expand_dims(topk_weights, axis=-1)
    token_topk_hidden = jnp.where(valid_mask[:, :, None], token_topk_hidden, 0)
    # FP32 top-k weights can promote BF16 outputs; keep the custom-op dtype.
    return token_topk_hidden.sum(axis=1).astype(out_dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "activation",
        "use_ep",
        "use_sparse_core",
        "onehot_moe_permute_threshold",
        "rhs_quant_dtype",
        "skip_padded_tokens",
    ),
)
def fused_moe_func(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    experts_start: jax.Array | None = None,
    topk: int = 1,
    activation: str = "silu",
    use_ep: bool = False,
    use_sparse_core: bool = True,
    onehot_moe_permute_threshold: int = 0,
    rhs_quant_dtype: jnp.dtype | None = None,
    skip_padded_tokens: bool = False,
) -> jax.Array:
    """Run MoE with precomputed expert ids and weights.

    For linear EP placement, ``experts_start`` is the first global expert id
    owned by this shard, passed as a 0-d int32 array (real traced data, not a
    Python int/compile-time constant -- every EP rank owns a different value,
    and binding it as a constant would make each rank JIT-compile a
    structurally different program under the same custom-op name, which
    desyncs the in-graph EP collectives). The kernel remaps global ids to
    local ids with an elementwise subtract and masks non-local experts.
    ``use_ep`` is a static flag enabling EP routing (identical across ranks,
    safe to bind as a compile-time constant); ``use_sparse_core`` is a static
    flag selecting the #193 SparseCore ragged gather/gather-reduce (vs the
    pre-#193 plain-JAX path).

    For packed weights, `rhs_quant_dtype` is used to specify the logical dtype of the
    weights. Currently, only INT4 logical weights packed inside INT32 or UINT32 carrier
    containers are supported. Weights must be packed along the contracting dimension
    (K-axis) in LSB-first order.
    """

    # Convert 3D scale [experts, blocks, N] to 4D [experts, blocks, 1, N]
    # expected by gmm_v2.
    if w1_scale is not None and w1_scale.ndim == 3:
        w1_scale = jnp.expand_dims(w1_scale, 2)
    if w2_scale is not None and w2_scale.ndim == 3:
        w2_scale = jnp.expand_dims(w2_scale, 2)

    orig_num_tokens, hidden_size = hidden_states.shape
    pad_t = -orig_num_tokens % 16 if (orig_num_tokens * topk) % 16 != 0 else 0
    if pad_t:
        hidden_states = jnp.pad(hidden_states, ((0, pad_t), (0, 0)))
        topk_weights = jnp.pad(topk_weights, ((0, pad_t), (0, 0)), constant_values=0)
        topk_ids = jnp.pad(topk_ids, ((0, pad_t), (0, 0)), constant_values=-1)
    num_tokens = hidden_states.shape[0]

    # NVFP4 weights arrive as native float4_e2m1fn in gmm_v2's K-major layout
    # (unpacked once at load; see fused_moe.load_kmajor_fp4). gmm_v2 picks the
    # regime from the block size: block-16 -> W4A16, block >= MXU -> W4A8.
    _, padded_hidden_size, _ = w1.shape
    packing_factor = get_packing_factor(w1.dtype, rhs_quant_dtype)
    padded_hidden_size *= packing_factor

    assert topk_weights.shape == (num_tokens, topk)
    assert topk_ids.shape == (num_tokens, topk)

    # Pin topk_weights to hidden_states.dtype (or float32 if
    # TPU_MOE_TOPK_WEIGHTS_FLOAT32 is enabled) so the EP mask, GMM reduce,
    # and `where` propagate appropriately.
    topk_weights = topk_weights.astype(_get_topk_weights_dtype(hidden_states.dtype))

    if use_ep and use_sparse_core and num_tokens * topk <= onehot_moe_permute_threshold:
        # The one-hot permute and unpermute are matmuls across all tokens, and
        # 0 * NaN == NaN: one token with a non-finite hidden state or routing
        # weight would turn every token's output into NaN. Zero those values
        # so the damage stays with that token.
        hidden_states = jnp.where(jnp.isfinite(hidden_states), hidden_states, 0)
        topk_weights = jnp.where(jnp.isfinite(topk_weights), topk_weights, 0)

    # Decode services also compile large prefill buckets. Apply the token
    # threshold before kernel admission so those buckets retain routed GMM.
    # Under EP, auto-enable up to 128 tokens when TPU_MOE_DENSE_EXPERT_THRESHOLD
    # is unset (0) across all supported quantization & intermediate-size regimes.
    effective_dense_threshold = (
        (
            192
            if packing_factor == 2 and w2.shape[1] * packing_factor <= 512
            else (
                160
                if w2.shape[1] * packing_factor <= 512
                else (32 if _prefill_moe_numerics() else 128)
            )
        )
        if (_DENSE_EXPERT_THRESHOLD == 0 and use_ep)
        else _DENSE_EXPERT_THRESHOLD
    )
    use_dense_expert = (
        0 < num_tokens <= effective_dense_threshold
        and can_use_dense_expert(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            w1_bias,
            w2_bias,
            activation=activation,
            rhs_quant_dtype=rhs_quant_dtype,
            skip_padded_tokens=skip_padded_tokens,
        )
    )
    if not use_dense_expert:
        # Flatten `(num_tokens, topk)` -> `(num_tokens * topk,)` before the EP
        # range filter so it executes in 1D `{0:T(1024)}` without 2D minor-dim=6
        # lane padding (`95.3%` waste) and 2D-to-1D slice/concat relayouts.
        topk_ids = topk_ids.reshape(-1)
        topk_weights = topk_weights.reshape(-1)

    if use_ep and experts_start is not None:
        local_ids = topk_ids - experts_start
        valid = (local_ids >= 0) & (local_ids < w1.shape[0])
        topk_weights = jnp.where(valid, topk_weights, jnp.zeros_like(topk_weights))
        topk_ids = jnp.where(valid, local_ids, jnp.full_like(local_ids, -1))

    if use_dense_expert:
        logger.info_once("Selected experimental dense-expert decode MoE")
        return dense_expert_moe(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            activation=activation,
            rhs_quant_dtype=rhs_quant_dtype,
        )[:orig_num_tokens, :hidden_size]

    effective_onehot_threshold = (
        _FAST_ONEHOT_AUTO_THRESHOLD
        if (
            onehot_moe_permute_threshold >= 511
            and (
                _ONEHOT_MOE_PERMUTE_THRESHOLD is None
                or _ONEHOT_MOE_PERMUTE_THRESHOLD < 0
            )
            and hidden_states.dtype == jnp.bfloat16
        )
        else onehot_moe_permute_threshold
    )

    (
        x,
        group_sizes,
        argsort_revert_indices,
        topk_weights_flat,
        valid_mask,
        token_indices_sorted,
        sorted_indices,
        onehot_permute,
        w_sorted,
    ) = prepare_routed_gmm_inputs(
        hidden_states,
        topk_ids,
        topk_weights,
        local_num_experts=w1.shape[0],
        topk=topk,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=effective_onehot_threshold,
        skip_padded_tokens=skip_padded_tokens,
        return_sorted_indices=True,
        return_onehot_combine=True,
    )
    x = jnp.pad(x, ((0, 0), (0, padded_hidden_size - hidden_size)))
    x = moe_gmm(
        x,
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        group_sizes,
        argsort_revert_indices,
        topk_weights_flat,
        valid_mask,
        token_indices_sorted,
        activation=activation,
        num_tokens=num_tokens,
        topk=topk,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=effective_onehot_threshold,
        rhs_quant_dtype=rhs_quant_dtype,
        sorted_indices=sorted_indices,
        onehot_permute=onehot_permute,
        w_sorted=w_sorted,
    )
    return x[:orig_num_tokens, :hidden_size]


# Column block of gmm_v2's lhs quantization (make_gmm_configs); gmm_v2 rejects
# pre-quantized lhs whose blocks differ.
MOE_FP8_DISPATCH_BLOCK = 512


def quantize_moe_dispatch_rows(
    hidden_states: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Quantizes MoE inputs before dispatch exactly as the routed gmm1 would.

    Returns int8 rows [T, H] holding the float8_e4m3fn bits and the flat
    per-block scales [T * H // MOE_FP8_DISPATCH_BLOCK] in hidden_states.dtype:
    collective-friendly dtypes and layouts for a dim-0 all-gather. Consumed by
    fused_moe_func_fp8_dispatch.
    """
    hidden_q, hidden_scale = quantize_lhs_rows(
        hidden_states, block_size=MOE_FP8_DISPATCH_BLOCK
    )
    return (
        jax.lax.bitcast_convert_type(hidden_q, jnp.int8),
        hidden_scale.reshape(-1),
    )


def quantize_moe_dispatch_packed_rows(
    hidden_states: jax.Array,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
) -> jax.Array:
    """Quantizes MoE inputs and packs `(q_i8, scale, topk_weights, topk_ids)`
    into a single 128-lane-aligned `[T, H + trailer]` `int8` buffer so a single
    2D AllGather replaces 4 separate collectives per chunk.
    """
    num_tokens, hidden_size = hidden_states.shape
    q_i8, scale_flat = quantize_moe_dispatch_rows(hidden_states)
    scale_bytes = jax.lax.bitcast_convert_type(
        scale_flat.reshape(num_tokens, hidden_size // MOE_FP8_DISPATCH_BLOCK),
        jnp.int8,
    ).reshape(num_tokens, -1)
    tw_bytes = jax.lax.bitcast_convert_type(
        topk_weights.astype(hidden_states.dtype), jnp.int8
    ).reshape(num_tokens, -1)
    ti_bytes = jax.lax.bitcast_convert_type(
        topk_ids.astype(jnp.int32), jnp.int8
    ).reshape(num_tokens, -1)
    used_bytes = scale_bytes.shape[1] + tw_bytes.shape[1] + ti_bytes.shape[1]
    trailer_bytes = ((used_bytes + 127) // 128) * 128
    pad_bytes = trailer_bytes - used_bytes
    parts = [q_i8, scale_bytes, tw_bytes, ti_bytes]
    if pad_bytes > 0:
        parts.append(jnp.zeros((num_tokens, pad_bytes), dtype=jnp.int8))
    return jnp.concatenate(parts, axis=1)


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "activation",
        "use_ep",
        "use_sparse_core",
        "onehot_moe_permute_threshold",
        "rhs_quant_dtype",
        "skip_padded_tokens",
    ),
)
def fused_moe_func_fp8_dispatch(
    hidden_q: jax.Array,
    hidden_scale: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    experts_start: jax.Array | None = None,
    topk: int = 1,
    activation: str = "silu",
    use_ep: bool = False,
    use_sparse_core: bool = True,
    onehot_moe_permute_threshold: int = 0,
    rhs_quant_dtype: jnp.dtype | None = None,
    skip_padded_tokens: bool = False,
) -> jax.Array:
    """fused_moe_func on rows pre-quantized by quantize_moe_dispatch_rows
    or packed by quantize_moe_dispatch_packed_rows.

    Bit-identical to fused_moe_func on the unquantized rows: routing is
    unchanged and gmm1 consumes exactly the fp8 rows and scales it would
    otherwise compute in-kernel. Returns [T, H] in hidden_scale.dtype.
    """
    if w1_scale is not None and w1_scale.ndim == 3:
        w1_scale = jnp.expand_dims(w1_scale, 2)
    if w2_scale is not None and w2_scale.ndim == 3:
        w2_scale = jnp.expand_dims(w2_scale, 2)

    orig_num_tokens = hidden_q.shape[0]
    act_dtype = hidden_scale.dtype
    padded_hidden_size = w1.shape[1] * get_packing_factor(w1.dtype, rhs_quant_dtype)
    if hidden_q.shape[1] > padded_hidden_size:
        packed = hidden_q
        hidden_size = padded_hidden_size
        hidden_q = packed[:, :hidden_size]
        num_blocks = hidden_size // MOE_FP8_DISPATCH_BLOCK
        itemsize = jnp.dtype(act_dtype).itemsize
        scale_b = num_blocks * itemsize
        tw_b = topk * itemsize
        ti_b = topk * 4
        off0 = hidden_size
        off1 = off0 + scale_b
        off2 = off1 + tw_b
        off3 = off2 + ti_b
        scale_i8 = packed[:, off0:off1].reshape(orig_num_tokens, num_blocks, itemsize)
        hidden_scale = jax.lax.bitcast_convert_type(scale_i8, act_dtype)
        tw_i8 = packed[:, off1:off2].reshape(orig_num_tokens, topk, itemsize)
        topk_weights = jax.lax.bitcast_convert_type(tw_i8, act_dtype)
        ti_i8 = packed[:, off2:off3].reshape(orig_num_tokens, topk, 4)
        topk_ids = jax.lax.bitcast_convert_type(ti_i8, jnp.int32)
    else:
        hidden_size = hidden_q.shape[1]
        hidden_scale = hidden_scale.reshape(orig_num_tokens, -1)
    if padded_hidden_size != hidden_size:
        raise NotImplementedError("fp8 MoE dispatch needs an unpadded hidden size.")
    pad_t = -orig_num_tokens % 32
    if pad_t:
        hidden_q = jnp.pad(hidden_q, ((0, pad_t), (0, 0)))
        hidden_scale = jnp.pad(hidden_scale, ((0, pad_t), (0, 0)))
        topk_weights = jnp.pad(topk_weights, ((0, pad_t), (0, 0)), constant_values=0)
        topk_ids = jnp.pad(topk_ids, ((0, pad_t), (0, 0)), constant_values=-1)
    num_tokens = hidden_q.shape[0]
    packing_factor = get_packing_factor(w1.dtype, rhs_quant_dtype)
    is_fp4_w4a8 = rhs_quant_dtype == jnp.float4_e2m1fn and w1.dtype in (
        jnp.float4_e2m1fn,
        jnp.uint8,
    )
    effective_dense_threshold = (
        (
            192
            if packing_factor == 2 and w2.shape[1] * packing_factor <= 512
            else (
                160
                if w2.shape[1] * packing_factor <= 512
                else (32 if _prefill_moe_numerics() else 128)
            )
        )
        if (_DENSE_EXPERT_THRESHOLD == 0 and use_ep)
        else _DENSE_EXPERT_THRESHOLD
    )
    use_dense_expert = (
        0 < num_tokens <= effective_dense_threshold
        and can_use_dense_expert(
            jax.ShapeDtypeStruct((num_tokens, hidden_size), act_dtype),
            w1,
            w2,
            w1_scale,
            w2_scale,
            w1_bias,
            w2_bias,
            activation=activation,
            rhs_quant_dtype=rhs_quant_dtype,
            skip_padded_tokens=skip_padded_tokens,
        )
    )
    assert topk_weights.shape == (num_tokens, topk)
    assert topk_ids.shape == (num_tokens, topk)

    # Same routing as fused_moe_func (whose EP remap is elementwise).
    topk_weights = topk_weights.astype(act_dtype)
    if not use_dense_expert:
        topk_weights = topk_weights.reshape(-1)
        topk_ids = topk_ids.reshape(-1)
    if use_ep and experts_start is not None:
        local_ids = topk_ids - experts_start
        valid = (local_ids >= 0) & (local_ids < w1.shape[0])
        topk_weights = jnp.where(valid, topk_weights, jnp.zeros_like(topk_weights))
        topk_ids = jnp.where(valid, local_ids, jnp.full_like(local_ids, -1))

    if use_dense_expert:
        is_w4a16 = (
            rhs_quant_dtype is None
            and w1.dtype == jnp.float4_e2m1fn
            and (w1_scale is None or w1.shape[1] // w1_scale.shape[1] < 128)
        )
        if is_w4a16 or not _prefill_moe_numerics():
            hidden_bf16 = (
                jax.lax.bitcast_convert_type(hidden_q, jnp.float8_e4m3fn)
                .astype(act_dtype)
                .reshape(num_tokens, -1, MOE_FP8_DISPATCH_BLOCK)
                * jnp.expand_dims(hidden_scale, -1)
            ).reshape(num_tokens, hidden_size)
            return dense_expert_moe(
                hidden_bf16,
                w1,
                w2,
                w1_scale,
                w2_scale,
                topk_weights,
                topk_ids,
                activation=activation,
                rhs_quant_dtype=rhs_quant_dtype,
            )[:orig_num_tokens, :hidden_size]
        return dense_expert_moe(
            jnp.empty((num_tokens, hidden_size), act_dtype),
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            activation=activation,
            rhs_quant_dtype=rhs_quant_dtype,
            hidden_q=hidden_q,
            hidden_scale=hidden_scale,
        )[:orig_num_tokens, :hidden_size]

    is_fp4_rhs = rhs_quant_dtype == jnp.float4_e2m1fn or (
        jnp.issubdtype(w1.dtype, jnp.floating)
        and jax.dtypes.itemsize_bits(w1.dtype) == 4
    )
    if _prefill_moe_numerics():
        default_bf16_lhs = not bool(envs.VLLM_DSV41_MOE_FP4_GMM1_FP8)
        use_fp4_bf16_lhs = is_fp4_rhs and (
            bool(envs.VLLM_DSV41_MOE_FP4_BF16_LHS)
            if envs.VLLM_DSV41_MOE_FP4_BF16_LHS is not None
            else default_bf16_lhs
        )
    else:
        # Decode stack: always hand gmm1 the fp8 rows + scales.
        use_fp4_bf16_lhs = False
    if use_fp4_bf16_lhs:
        hidden_in = (
            jax.lax.bitcast_convert_type(hidden_q, jnp.float8_e4m3fn)
            .astype(act_dtype)
            .reshape(num_tokens, -1, MOE_FP8_DISPATCH_BLOCK)
            * jnp.expand_dims(hidden_scale, -1)
        ).reshape(num_tokens, hidden_size)
    else:
        hidden_in = hidden_q

    effective_onehot_threshold = (
        _FAST_ONEHOT_AUTO_THRESHOLD
        if (
            onehot_moe_permute_threshold >= 511
            and (
                _ONEHOT_MOE_PERMUTE_THRESHOLD is None
                or _ONEHOT_MOE_PERMUTE_THRESHOLD < 0
            )
            and act_dtype == jnp.bfloat16
        )
        else onehot_moe_permute_threshold
    )

    (
        x,
        group_sizes,
        argsort_revert_indices,
        topk_weights_flat,
        valid_mask,
        token_indices_sorted,
        sorted_indices,
        onehot_permute,
        w_sorted,
    ) = prepare_routed_gmm_inputs(
        hidden_in,
        topk_ids,
        topk_weights,
        local_num_experts=w1.shape[0],
        topk=topk,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=effective_onehot_threshold,
        skip_padded_tokens=skip_padded_tokens,
        return_sorted_indices=True,
        return_onehot_combine=True,
    )
    x = moe_gmm(
        x if use_fp4_bf16_lhs else jax.lax.bitcast_convert_type(x, jnp.float8_e4m3fn),
        w1,
        w1_scale,
        w1_bias,
        w2,
        w2_scale,
        w2_bias,
        group_sizes,
        argsort_revert_indices,
        topk_weights_flat,
        valid_mask,
        token_indices_sorted,
        activation=activation,
        num_tokens=num_tokens,
        topk=topk,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=effective_onehot_threshold,
        rhs_quant_dtype=rhs_quant_dtype,
        sorted_indices=sorted_indices,
        onehot_permute=onehot_permute,
        w_sorted=w_sorted,
        lhs_scale=None if use_fp4_bf16_lhs else hidden_scale[token_indices_sorted],
    )
    return x[:orig_num_tokens, :hidden_size]
