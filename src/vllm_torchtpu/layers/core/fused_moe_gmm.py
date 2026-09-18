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

import jax
from jax import numpy as jnp
from jax.experimental.pallas import tpu as pltpu

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.megablox.gmm_v2 import get_packing_factor, gmm_v2
from vllm_torchtpu.kernels.megablox.moe_onehot_unpermute import (
    blockwise_onehot_unpermute, can_use_blockwise_onehot_unpermute)
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce import \
    ragged_gather_reduce as ragged_gather_reduce_v1
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v2.wrapper import \
    ragged_gather_reduce_v2
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import \
    ragged_gather_reduce as ragged_gather_reduce_v3
from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import \
    token_block_alignment
from vllm_torchtpu.kernels.sparse_core.ragged_gather_v2 import ragged_gather_v2
from vllm_torchtpu.logger import init_logger

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
ragged_gather_reduce = _select_ragged_gather_reduce(
    envs.RAGGED_GATHER_REDUCE_VERSION)

_ONEHOT_AUTO_CAP = 512


def resolve_onehot_permute_threshold() -> int:
    """Routed-row count at/below which MoE permute+combine run as TensorCore
    one-hot matmuls instead of the SparseCore gather/reduce kernels.

    Auto (env unset/empty/negative) resolves to ``min(SC block rows - 1,
    _ONEHOT_AUTO_CAP)`` on SparseCore TPUs and 0 elsewhere. An explicit
    ``ONEHOT_MOE_PERMUTE_THRESHOLD`` wins: 0 disables the one-hot path, a
    positive value forces that threshold.
    """
    explicit = envs.ONEHOT_MOE_PERMUTE_THRESHOLD
    if explicit is not None and explicit >= 0:
        return explicit
    try:
        sc_info = pltpu.get_tpu_info().sparse_core
    except ValueError:
        # get_tpu_info raises for unsupported device kinds (e.g. CPU hosts).
        return 0
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


def requant_unpack_kmajor(w_packed: jax.Array, scale_f: jax.Array,
                          block: int) -> tuple[jax.Array, jax.Array]:
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
    dequant = (fp4.astype(jnp.float32).reshape(*fp4.shape[:-1], num_in_blocks,
                                               size_k // num_in_blocks) *
               scale_f[..., None]).reshape(*fp4.shape[:-1], size_k)
    fp4_max = float(jnp.finfo(jnp.float4_e2m1fn).max)
    blocked = dequant.reshape(*dequant.shape[:-1], size_k // block, block)
    scale = jnp.max(jnp.abs(blocked), axis=-1, keepdims=True) * (1.0 / fp4_max)
    scale_inv = jnp.where(scale == 0, 0.0, 1.0 / scale)
    requant = jnp.clip(blocked * scale_inv, -fp4_max,
                       fp4_max).astype(jnp.float4_e2m1fn)
    requant = requant.reshape(*dequant.shape[:-1], size_k)  # [..., N, K]
    scale = scale.squeeze(-1).astype(jnp.float32)  # [..., N, K/block]
    return (jnp.swapaxes(requant, -1,
                         -2), jnp.expand_dims(jnp.swapaxes(scale, -1, -2), -2))


def quantize_to_native_fp4_kmajor(
        w: jax.Array,
        block: int,
        pack: bool = True) -> tuple[jax.Array, jax.Array]:
    """Quantize K-major float weight ([..., K, N]) to packed uint8 e2m1 and per-block FP32 scale.
    Returns uint8 to safely cross the PyTorch/JAX bridge before unpacking in gmm_v2.

    ``pack=False`` returns the same values as ``float4_e2m1fn`` [..., K, N]
    instead -- what the fused EP MoE kernel's FP4 form reads, and what torch
    receives as a native ``torch.float4_e2m1fn_x2`` tensor. Only the transport
    differs: the quantization above is one cast either way, so the two forms
    carry bit-identical weights and a caller may choose per layer.

    TODO: Make a more generic version of this function; possibly combine with quantize_tensor_to_fp4
    """

    size_k = w.shape[-2]
    num_blocks = size_k // block
    blocked = w.astype(jnp.float32).reshape(*w.shape[:-2], num_blocks, block,
                                            w.shape[-1])
    fp4_max = float(jnp.finfo(jnp.float4_e2m1fn).max)
    abs_max = jnp.max(jnp.abs(blocked), axis=-2, keepdims=True)
    scale = abs_max / fp4_max
    scale_inv = jnp.where(scale == 0, 0.0, 1.0 / scale)
    quantized = jnp.clip(blocked * scale_inv, -fp4_max,
                         fp4_max).astype(jnp.float4_e2m1fn)
    quantized = quantized.reshape(*w.shape[:-2], size_k,
                                  w.shape[-1])  # [..., K, N]

    scale = scale.astype(jnp.float32)  # [..., num_blocks, 1, N]
    if not pack:
        return quantized, scale

    # Real e2m1 bit-packing (2 values/byte along K), matching gmm_v2's own
    # should_unpack/bitcast(quant_dtype) unpacking convention.
    pairs = quantized.reshape(*w.shape[:-2], size_k // 2, 2, w.shape[-1])
    pairs = jnp.swapaxes(pairs, -1, -2)  # [..., K/2, N, 2] -- pair axis last
    packed = jax.lax.bitcast_convert_type(pairs, jnp.uint8)  # [..., K/2, N]

    return packed, scale


def gmm_wrapper(lhs,
                rhs,
                rhs_scale,
                rhs_bias,
                group_sizes,
                group_offset,
                zero_initialize=False,
                fuse_act=None,
                preferred_element_type=None,
                rhs_quant_dtype=None):
    # fp4 weights: keep bf16 activations. Quantizing activations to fp8 collapses
    # fp4 accuracy (error compounds across MoE layers) with no decode speedup
    # (decode is weight-HBM-bound). fp8/int4 weights keep the default fp8 act.
    is_fp4_weight = (jnp.issubdtype(rhs.dtype, jnp.floating)
                     and jax.dtypes.itemsize_bits(rhs.dtype) == 4)
    return gmm_v2(
        lhs=lhs,
        rhs=rhs,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_sizes=group_sizes,
        group_offset=group_offset[0],
        zero_initialize=zero_initialize,
        fuse_act=fuse_act,
        preferred_element_type=preferred_element_type,
        maybe_quantize_lhs=not is_fp4_weight,
        rhs_quant_dtype=rhs_quant_dtype,
    )


# Rows per block of the rank contraction: the triangular matmul is B x B, so this
# trades that against the number of blocks.
_COUNTING_SORT_BLOCK = 512


def _counting_sort_positions(
        sort_keys: jax.Array,
        num_buckets: int) -> tuple[jax.Array | None, jax.Array | None]:
    """Stable counting sort: row -> its position in the sorted order.

    `jnp.argsort(jnp.argsort(keys))` spends a second comparison sort over M
    distinct values on what is really the inverse of the first permutation.
    The keys take only `num_buckets` values, so the rank within a bucket comes
    from a triangular matmul on the one-hot block -- MXU work proportional to
    M*block*num_buckets instead of a length-M sort -- and the bucket totals
    fall out of the same histogram, which is exactly `group_sizes`.
    """
    m = sort_keys.shape[0]
    block = min(_COUNTING_SORT_BLOCK, m)
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
        sort_keys = jnp.pad(sort_keys, (0, pad),
                            constant_values=num_buckets - 1)
    onehot = jax.nn.one_hot(sort_keys.reshape(-1, block),
                            num_buckets,
                            dtype=jnp.bfloat16)
    # The histogram rides along as one more row of the contraction the rank
    # matmul already performs. Exact: 0/1 summands, at most `block` of them.
    lower = jnp.concatenate(
        [
            jnp.ones((1, block), jnp.bfloat16),
            jnp.tril(jnp.ones((block, block), jnp.bfloat16), -1),
        ],
        axis=0,
    )
    both = jnp.einsum("ij,bje->bie",
                      lower,
                      onehot,
                      preferred_element_type=jnp.float32)
    hist = both[:, 0, :]
    rank = both[:, 1:, :]
    counts = hist.sum(axis=0)
    base = (jnp.cumsum(hist, axis=0) - hist) + (jnp.cumsum(counts) - counts)
    pos = ((base[:, None, :] + rank) * onehot).sum(axis=2)
    return pos.reshape(-1)[:m].astype(jnp.int32), counts.astype(jnp.int32)


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
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
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
    token_indices_flat = jnp.arange(num_tokens_local,
                                    dtype=jnp.int32).repeat(topk)

    if skip_padded_tokens:
        # Padded tokens keep their selected expert ids, but carry zero
        # expert routing weights. Treat them as invalid to remove them
        # from GMM work.
        valid_mask = (topk_indices_flat >= 0) & (topk_weights_flat != 0)
    else:
        valid_mask = topk_indices_flat >= 0
    sort_keys = jnp.where(valid_mask, topk_indices_flat, local_num_experts)
    # The keys take only `local_num_experts + 1` values, so a counting sort
    # gives the positions and `group_sizes` without a second comparison sort.
    # Ragged permute only: under the one-hot permute these positions break MTP.
    ragged_permute = (use_ep and use_sparse_core
                      and token_indices_flat.shape[0]
                      > onehot_moe_permute_threshold)
    argsort_revert_indices, bucket_counts = (_counting_sort_positions(
        sort_keys, local_num_experts + 1) if ragged_permute else (None, None))
    token_bits = max(1, (num_tokens_local - 1).bit_length())
    packable = ((local_num_experts + 1).bit_length() + token_bits < 31)
    sorted_indices = (jnp.argsort(sort_keys) if argsort_revert_indices is None
                      or not packable else None)

    if argsort_revert_indices is not None:
        group_sizes_local = bucket_counts[:local_num_experts]
    else:
        argsort_revert_indices = jnp.argsort(sorted_indices)
        topk_indices_for_count = jnp.where(valid_mask, topk_indices_flat, 0)
        group_sizes_local = (jax.nn.one_hot(
            topk_indices_for_count, local_num_experts, dtype=jnp.int32) *
                             valid_mask[:, None].astype(jnp.int32))
        group_sizes_local = group_sizes_local.sum(axis=0)

    if packable:
        # Packing (key, token) into one int32 keeps the payload out of the sort
        # and drops the follow-up gather. Equal packed values carry the same
        # token id, so their order cannot matter.
        packed = jax.lax.sort(jnp.left_shift(sort_keys, token_bits)
                              | token_indices_flat,
                              is_stable=False)
        token_indices_sorted = jnp.bitwise_and(packed, (1 << token_bits) - 1)
    else:
        token_indices_sorted = token_indices_flat[sorted_indices]

    if use_ep and use_sparse_core:
        if token_indices_sorted.shape[0] <= onehot_moe_permute_threshold:
            # Use one-hot matmul for permutation, which can be faster
            # for small batch size
            onehot = jax.nn.one_hot(token_indices_sorted,
                                    num_tokens_local,
                                    dtype=hidden_states_local.dtype)
            x = onehot @ hidden_states_local
        else:
            # Match uLLM/reference EP routing: materialize the valid local
            # expert prefix through SparseCore ragged_gather. Invalid
            # non-local rows are sorted after the valid prefix.
            valid_count = group_sizes_local.sum(dtype=jnp.int32)
            x = ragged_gather_v2(
                hidden_states_local,
                token_indices_sorted,
                jnp.array([0], dtype=jnp.int32),
                valid_count[None],
            )
    else:
        x = hidden_states_local[token_indices_sorted]
    return (x, group_sizes_local, argsort_revert_indices, topk_weights_flat,
            valid_mask, token_indices_sorted)


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
) -> jax.Array:
    """Run grouped GEMM for routed tokens and reduce back to tokens.

    The valid-mask gating below is a no-op for non-EP runs (mask is all
    True) and keeps the single code path uniform for both EP and non-EP.
    EP uses SparseCore ragged_gather + ragged_gather_reduce for token movement.
    """
    group_offset = jnp.array([0], dtype=jnp.int32)

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
    )
    packing_factor = get_packing_factor(w2.dtype, rhs_quant_dtype)
    gmm1_res = gmm1_res[:, :w2.shape[1] * packing_factor]

    topk_weights = topk_weights_flat.reshape((num_tokens, topk))
    valid_mask = valid_mask_flat.reshape((num_tokens, topk))

    gmm2_res = gmm_wrapper(gmm1_res,
                           w2,
                           w2_scale,
                           w2_bias,
                           group_sizes,
                           group_offset,
                           zero_initialize=False,
                           preferred_element_type=x.dtype,
                           rhs_quant_dtype=rhs_quant_dtype)
    # The destination-major kernel packs source indices into 20 bits and
    # processes 16-lane SparseCore tiles.
    use_local_ragged_combine = (
        # Local SparseCore path; route-pair metadata needs at least two routes.
        use_sparse_core and not use_ep and topk >= 2
        # TensorCore combine handles top-8 batches.
        and topk != 8
        # Honor the configured SparseCore dispatch cutoff.
        and argsort_revert_indices.size > onehot_moe_permute_threshold
        # BF16 kernel input.
        and gmm2_res.dtype == jnp.bfloat16
        # Packed source-index capacity.
        and gmm2_res.shape[0] <= 1 << 20)
    if use_local_ragged_combine:
        try:
            tpu_info = pltpu.get_tpu_info()
        except ValueError:
            # get_tpu_info raises for unsupported device kinds (e.g. CPU hosts).
            tpu_info = None
        sc_info = tpu_info.sparse_core if tpu_info is not None else None
        # Lane-aligned width, 16-lane SparseCore, and v3's own cutoff (two
        # BF16 buffers use at least 60% of VMEM).
        if (sc_info is not None and sc_info.num_lanes == 16
                and gmm2_res.shape[-1] % tpu_info.num_lanes == 0
                and gmm2_res.size * 4 >= tpu_info.vmem_capacity_bytes * 0.6):
            # The kernel pads destination tokens to one 64-token block per
            # SparseCore row partition: require at least one full block and
            # padding of at most half the batch.
            alignment = token_block_alignment(gmm2_res.shape[-1], tpu_info)
            padded_tokens = -(-num_tokens // alignment) * alignment
            if (num_tokens >= alignment
                    and padded_tokens * 2 <= num_tokens * 3):
                return ragged_gather_reduce_v3(
                    gmm2_res,
                    argsort_revert_indices,
                    topk_weights_flat,
                    valid_mask_flat,
                    reduce_group_size=topk,
                ).astype(x.dtype)
    if use_ep and use_sparse_core:
        if argsort_revert_indices.size <= onehot_moe_permute_threshold:
            # Use onehot + matmul for unpermutation, which can be faster
            # for small batch size.
            # `argsort_revert_indices` is the inverse of the sort, so scattering
            # through it is the gather by the (unmaterialised) forward one.
            topk_weights_sorted = jnp.zeros_like(topk_weights_flat).at[
                argsort_revert_indices].set(topk_weights_flat)
            valid_count = group_sizes.sum(dtype=jnp.int32)[None]
            if (envs.TPU_MOE_OWNER_OUTPUT_MODE.lower() == "on" and
                    can_use_blockwise_onehot_unpermute(gmm2_res,
                                                       token_indices_sorted,
                                                       topk_weights_sorted,
                                                       valid_count,
                                                       num_tokens=num_tokens)):
                logger.info_once(
                    "Selected owner_output blockwise one-hot unpermute: "
                    "routes=%d hidden=%d tokens=%d", gmm2_res.shape[0],
                    gmm2_res.shape[1], num_tokens)
                return blockwise_onehot_unpermute(
                    gmm2_res,
                    token_indices_sorted,
                    topk_weights_sorted,
                    valid_count,
                    num_tokens=num_tokens,
                ).astype(x.dtype)
            # Rows past the computed prefix (non-local experts under EP,
            # skipped padded tokens) are never written by the GMM; zero them
            # so stale NaNs cannot spread through the matmul (0 * NaN == NaN).
            computed = jnp.arange(
                gmm2_res.shape[0],
                dtype=jnp.int32) < group_sizes.sum(dtype=jnp.int32)
            gmm2_res = jnp.where(computed[:, None], gmm2_res, 0)
            revert_indices = argsort_revert_indices.reshape(num_tokens, topk)
            onehot = jax.nn.one_hot(revert_indices,
                                    argsort_revert_indices.size,
                                    dtype=gmm2_res.dtype)
            combine = (onehot * topk_weights[..., None] *
                       valid_mask[..., None]).sum(axis=1)
            return (combine @ gmm2_res).astype(x.dtype)
        return ragged_gather_reduce(
            gmm2_res,
            argsort_revert_indices,
            topk_weights_flat,
            valid_mask_flat,
            reduce_group_size=topk,
        ).astype(x.dtype)

    token_hidden = gmm2_res[argsort_revert_indices]
    token_topk_hidden = token_hidden.reshape(
        (num_tokens, topk, gmm2_res.shape[-1]))
    token_topk_hidden = token_topk_hidden * jnp.expand_dims(topk_weights,
                                                            axis=-1)
    token_topk_hidden = jnp.where(valid_mask[:, :, None], token_topk_hidden, 0)
    # FP32 top-k weights can promote BF16 outputs; keep the custom-op dtype.
    return token_topk_hidden.sum(axis=1).astype(x.dtype)


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

    For packed weights, `rhs_quant_dtype` is used to specify the logical dtype of the weights.
    Currently, only INT4 logical weights packed inside INT32 or UINT32 carrier containers are
    supported. Weights must be packed along the contracting dimension (K-axis) in LSB-first order.
    """

    # Convert 3D scale [experts, blocks, N] to 4D [experts, blocks, 1, N] expected by gmm_v2.
    if w1_scale is not None and w1_scale.ndim == 3:
        w1_scale = jnp.expand_dims(w1_scale, 2)
    if w2_scale is not None and w2_scale.ndim == 3:
        w2_scale = jnp.expand_dims(w2_scale, 2)

    num_tokens, hidden_size = hidden_states.shape

    # NVFP4 weights arrive as native float4_e2m1fn in gmm_v2's K-major layout
    # (unpacked once at load; see fused_moe.load_kmajor_fp4). gmm_v2 picks the
    # regime from the block size: block-16 -> W4A16, block >= MXU -> W4A8.
    _, padded_hidden_size, _ = w1.shape
    packing_factor = get_packing_factor(w1.dtype, rhs_quant_dtype)
    padded_hidden_size *= packing_factor

    assert topk_weights.shape == (num_tokens, topk)
    assert topk_ids.shape == (num_tokens, topk)

    # Pin topk_weights to hidden_states.dtype so the EP mask, GMM
    # reduce, and `where` propagate in BF16 — some custom_routing_fn
    # callers return FP32 weights and would otherwise promote downstream.
    topk_weights = topk_weights.astype(hidden_states.dtype)

    if use_ep and experts_start is not None:
        local_ids = topk_ids - experts_start
        valid = (local_ids >= 0) & (local_ids < w1.shape[0])
        topk_weights = jnp.where(valid, topk_weights,
                                 jnp.zeros_like(topk_weights))
        topk_ids = jnp.where(valid, local_ids, jnp.full_like(local_ids, -1))

    (x, group_sizes, argsort_revert_indices, topk_weights_flat, valid_mask,
     token_indices_sorted) = prepare_routed_gmm_inputs(
         hidden_states,
         topk_ids,
         topk_weights,
         local_num_experts=w1.shape[0],
         topk=topk,
         use_ep=use_ep,
         use_sparse_core=use_sparse_core,
         onehot_moe_permute_threshold=onehot_moe_permute_threshold,
         skip_padded_tokens=skip_padded_tokens,
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
        onehot_moe_permute_threshold=onehot_moe_permute_threshold,
        rhs_quant_dtype=rhs_quant_dtype,
    )
    return x[:num_tokens, :hidden_size]
