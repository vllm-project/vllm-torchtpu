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
from vllm_torchtpu.distributed.ep_combine_layout import (
    combine_piece_rows,
    split_pieces,
)
from vllm_torchtpu.kernels.experimental.moe_dense_expert import (
    can_use_dense_expert,
    dense_expert_moe,
)
from vllm_torchtpu.kernels.megablox.gmm_v2 import (
    LHS_QUANT_BLOCK_SIZE,
    can_fuse_mlp,
    get_packing_factor,
    gmm_v2,
    gmm_v2_fused_mlp,
    token_rows_fit_smem,
)
from vllm_torchtpu.kernels.megablox.moe_gather_unpermute import (
    can_use_gather_unpermute,
    gather_unpermute,
)
from vllm_torchtpu.kernels.megablox.moe_onehot_unpermute import (
    blockwise_onehot_unpermute,
    can_use_blockwise_onehot_unpermute,
)
from vllm_torchtpu.kernels.megablox.quantized_rows import (
    QUANTIZE_ROWS_ALIGN,
    ROWS_PER_WORD,
    SOURCE_DTYPE,
    quantize_rows,
    quantized_row_width,
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
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _select_ragged_gather_reduce(version: str):
    return {
        "v1": ragged_gather_reduce_v1,
        "v2": ragged_gather_reduce_v2,
        "v3": ragged_gather_reduce_v3,
    }[version]


# The server sets its environment before importing model code, so the selected
# EP combine remains fixed for the process lifetime. The default is v2 (run on
# the TensorCore at TC_COMBINE_MIN_TOKENS tokens and above); set
# RAGGED_GATHER_REDUCE_VERSION=v3 to select the destination-major kernel for EP.
# The non-EP combine in moe_gmm selects that kernel directly.
ragged_gather_reduce = _select_ragged_gather_reduce(envs.RAGGED_GATHER_REDUCE_VERSION)

_ONEHOT_AUTO_CAP = 512
# Fixed at import, like the other MoE algorithm selectors.
_DENSE_EXPERT_THRESHOLD = envs.TPU_MOE_DENSE_EXPERT_THRESHOLD
# Smallest EP token count whose v2 combine runs on the TensorCore (0: never).
# On DSv4-Flash DP8 the static estimate at 2048 tokens is ~45 us (kernel plus
# routing tables) against ~90 us of SparseCore v2 plus convert; decode at the
# 1024-token bucket keeps v2. A decode step of 2048 tokens takes it too.
TC_COMBINE_MIN_TOKENS = envs.TPU_MOE_TC_COMBINE_MIN_TOKENS
# From this EP token count, the expert MLP reads the routed rows straight
# from the tokens' rows (gmm_v2_fused_mlp's token_rows), past the first
# DISPATCH_HEAD_ROWS that the SparseCore gathers ahead. Below it the full
# SparseCore gather is short and the TensorCore has other work to hide it
# behind.
DISPATCH_SPLIT_MIN_TOKENS = 4096
# Whole 2048-row ragged_gather_v2 blocks, and whole 16-row sublane blocks as
# gmm_v2_fused_mlp's head must be, or 0 (the default): no SparseCore gather,
# the MLP fetches every row. A head can only start once the dispatch
# all-gather lands, so it sits between the two with nothing to hide it.
DISPATCH_HEAD_ROWS = 0
# The dtype an EP batch's rows may arrive quantized to, as int32 words
# (quantize_dispatch_rows), as the fused MLP's gate/up leg quantizes them.
DISPATCH_QUANT_DTYPE = jnp.float8_e4m3fn
# Fixed at import: build the MoE permutation from one packed sort.
_SINGLE_SORT = envs.TPU_MOE_SINGLE_SORT
# Above 2,048 routed rows, the inverse-permutation scatter costs more than
# the two sorts it replaces when `ragged_permute` is off (e.g. non-EP prefill).
_SINGLE_SORT_MAX_ROWS = 2048


def quantize_dispatch_rows(hidden_states: jax.Array) -> jax.Array:
    """This rank's rows, quantized as fused_moe_func's expert MLP would.

    For an EP dispatch to move, in about half the bytes, when
    dispatches_quantized_rows holds for the gathered batch; fused_moe_func
    computes the same output from them as from the bf16 rows.
    """
    return quantize_rows(
        hidden_states, block_size=LHS_QUANT_BLOCK_SIZE, q_dtype=DISPATCH_QUANT_DTYPE
    )


def quantizes_dispatch_rows(
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    *,
    topk: int,
    activation: str,
    rhs_quant_dtype: jnp.dtype | None,
) -> bool:
    """Whether fused_moe_func reads these EP experts' input as
    quantize_dispatch_rows words at every batch dispatches_quantized_rows
    admits (with the SparseCore on): the shape checks of its head-gather
    path that do not depend on the batch. Shapes only, and it asks the TPU,
    so call it once, outside traced code.
    """
    if w1_scale is not None and w1_scale.ndim == 3:
        w1_scale = jax.ShapeDtypeStruct(
            (*w1_scale.shape[:2], 1, w1_scale.shape[2]), w1_scale.dtype
        )
    if w2_scale is not None and w2_scale.ndim == 3:
        w2_scale = jax.ShapeDtypeStruct(
            (*w2_scale.shape[:2], 1, w2_scale.shape[2]), w2_scale.dtype
        )
    hidden_size = w2.shape[-1]
    padded_hidden_size = w1.shape[1] * get_packing_factor(w1.dtype, rhs_quant_dtype)
    # The smallest batch admitted: the fused MLP's tiles are the same above it.
    num_routes = DISPATCH_SPLIT_MIN_TOKENS * topk
    try:
        return (
            padded_hidden_size == hidden_size
            and _is_fp4_weight(w1) == _is_fp4_weight(w2)
            and can_fuse_mlp(
                jax.ShapeDtypeStruct(
                    (num_routes, quantized_row_width(hidden_size)),
                    DISPATCH_QUANT_DTYPE,
                ),
                w1,
                w2,
                jax.ShapeDtypeStruct((w1.shape[0],), jnp.int32),
                w1_scale,
                w1_bias,
                w2_scale,
                w2_bias,
                token_rows=True,
                quantized_lhs_dtype=DISPATCH_QUANT_DTYPE,
                **_fused_mlp_kwargs(activation, SOURCE_DTYPE, w1, rhs_quant_dtype),
            )
        )
    except ValueError:
        # get_tpu_info raises for unsupported device kinds (e.g. CPU hosts).
        return False


def dispatches_quantized_rows(
    num_local_tokens: int, world_size: int, topk: int, onehot_moe_permute_threshold: int
) -> bool:
    """Whether to EP-dispatch num_local_tokens rows per rank, over world_size
    ranks, as quantize_dispatch_rows words, to experts for which
    quantizes_dispatch_rows holds.

    The rest of fused_moe_func's head-gather path, which depends on the
    batch: it alone reads the words, and fused_moe_func refuses them
    anywhere else. Integer arithmetic only: callers ask it of a traced
    token count.
    """
    num_tokens = num_local_tokens * world_size
    num_routes = num_tokens * topk
    return (
        num_local_tokens % QUANTIZE_ROWS_ALIGN == 0
        and num_tokens >= DISPATCH_SPLIT_MIN_TOKENS
        and num_routes > onehot_moe_permute_threshold
        and token_rows_fit_smem(num_routes, QUANTIZE_ROWS_ALIGN, ROWS_PER_WORD)
    )


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


def _is_fp4_weight(rhs: jax.Array) -> bool:
    # fp4 weights: keep bf16 activations. Quantizing activations to fp8 collapses
    # fp4 accuracy (error compounds across MoE layers) with no decode speedup
    # (decode is weight-HBM-bound). fp8/int4 weights keep the default fp8 act.
    return (
        jnp.issubdtype(rhs.dtype, jnp.floating)
        and jax.dtypes.itemsize_bits(rhs.dtype) == 4
    )


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
):
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
        maybe_quantize_lhs=not _is_fp4_weight(rhs),
        rhs_quant_dtype=rhs_quant_dtype,
    )


# Rows per block of the rank contraction: the triangular matmul is B x B, so this
# trades that against the number of blocks.
_COUNTING_SORT_BLOCK = 512


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
        sort_keys = jnp.pad(sort_keys, (0, pad), constant_values=num_buckets - 1)
    onehot = jax.nn.one_hot(
        sort_keys.reshape(-1, block), num_buckets, dtype=jnp.bfloat16
    )
    # The histogram rides along as one more row of the contraction the rank
    # matmul already performs. Exact: 0/1 summands, at most `block` of them.
    lower = jnp.concatenate(
        [
            jnp.ones((1, block), jnp.bfloat16),
            jnp.tril(jnp.ones((block, block), jnp.bfloat16), -1),
        ],
        axis=0,
    )
    both = jnp.einsum("ij,bje->bie", lower, onehot, preferred_element_type=jnp.float32)
    hist = both[:, 0, :]
    rank = both[:, 1:, :]
    counts = hist.sum(axis=0)
    base = (jnp.cumsum(hist, axis=0) - hist) + (jnp.cumsum(counts) - counts)
    pos = ((base[:, None, :] + rank) * onehot).sum(axis=2)
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
    gather_rows: bool = True,
) -> tuple[jax.Array | None, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Prepare the local-only routing layout using already-local expert ids.

    For non-EP runs `topk_indices_local` is always non-negative, so the
    valid-mask path computed below is a no-op (jnp.where on an all-True
    mask). For EP, the caller applies the expert-map remap upstream
    (inside this JIT) and marks non-local entries with a negative id which
    the mask filters out. Always running the masked branch keeps a single
    code path. Without ``gather_rows`` the routed rows are left to the
    caller and None is returned for them.
    """
    num_tokens_local = topk_indices_local.shape[0]
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
    # The keys take only `local_num_experts + 1` values, so a counting sort
    # gives the positions and `group_sizes` without a second comparison sort.
    # Ragged permute only: under the one-hot permute these positions break MTP.
    ragged_permute = (
        use_ep
        and use_sparse_core
        and token_indices_flat.shape[0] > onehot_moe_permute_threshold
    )
    argsort_revert_indices, bucket_counts = (
        _counting_sort_positions(sort_keys, local_num_experts + 1)
        if ragged_permute
        else (None, None)
    )
    token_bits = max(1, (num_tokens_local - 1).bit_length())
    packable = (local_num_experts + 1).bit_length() + token_bits < 31
    flat_bits = max(1, (token_indices_flat.shape[0] - 1).bit_length())
    single_sort = (
        _SINGLE_SORT
        and not ragged_permute
        and token_indices_flat.shape[0] <= _SINGLE_SORT_MAX_ROWS
        and (local_num_experts + 1).bit_length() + flat_bits < 31
    )
    sorted_indices = (
        jnp.argsort(sort_keys)
        if not single_sort and (argsort_revert_indices is None or not packable)
        else None
    )
    token_indices_sorted: jax.Array | None = None

    if argsort_revert_indices is not None:
        group_sizes_local = bucket_counts[:local_num_experts]
    else:
        if single_sort:
            logger.info_once("MoE permutation runs as one packed sort.")
            sorted_indices, argsort_revert_indices, token_indices_sorted = (
                _single_sort_permutation(sort_keys, topk, flat_bits)
            )
        else:
            argsort_revert_indices = jnp.argsort(sorted_indices)
        topk_indices_for_count = jnp.where(valid_mask, topk_indices_flat, 0)
        group_sizes_local = jax.nn.one_hot(
            topk_indices_for_count, local_num_experts, dtype=jnp.int32
        ) * valid_mask[:, None].astype(jnp.int32)
        group_sizes_local = group_sizes_local.sum(axis=0)

    if token_indices_sorted is None:
        if packable:
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

    if not gather_rows:
        x = None
    elif use_ep and use_sparse_core:
        if token_indices_sorted.shape[0] <= onehot_moe_permute_threshold:
            # Use one-hot matmul for permutation, which can be faster
            # for small batch size
            onehot = jax.nn.one_hot(
                token_indices_sorted, num_tokens_local, dtype=hidden_states_local.dtype
            )
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
    return (
        x,
        group_sizes_local,
        argsort_revert_indices,
        topk_weights_flat,
        valid_mask,
        token_indices_sorted,
    )


def _piece_major(x: jax.Array, piece_rows: tuple[int, ...]) -> jax.Array:
    """Token-ordered rows of ``x`` reordered piece by piece (ep_combine_layout)."""
    return jnp.concatenate(split_pieces(x, piece_rows))


def _piece_major_routes(
    piece_rows: tuple[int, ...], topk: int, *routes: jax.Array
) -> tuple[jax.Array, ...]:
    """Reorders flat per-route arrays so the unpermute writes piece-major rows.

    The arrays are token-major, so reordering their elements in runs of whole
    tokens moves the routes of a token together and in order, and every output
    row is reduced exactly as before. Kept flat (each sub-block's routes are
    one contiguous run) so the reorder needs no relayout.
    """
    route_rows = tuple(rows * topk for rows in piece_rows)
    return tuple(_piece_major(r, route_rows) for r in routes)


def _runs_tc_combine(
    gmm2_res: jax.Array,
    topk_weights_flat: jax.Array,
    num_tokens: int,
    topk: int,
    chunk_pieces: tuple[int, ...] = (),
) -> bool:
    """Whether the EP v2 combine runs as the TensorCore kernel."""
    return (
        ragged_gather_reduce is ragged_gather_reduce_v2
        and 0 < TC_COMBINE_MIN_TOKENS <= num_tokens
        and can_use_gather_unpermute(
            gmm2_res, topk_weights_flat, num_tokens, topk, chunk_pieces
        )
    )


def _fused_mlp_kwargs(
    activation: str, dtype: jnp.dtype, w1: jax.Array, rhs_quant_dtype
) -> dict:
    return dict(
        fuse_act=activation,
        preferred_element_type=dtype,
        maybe_quantize_lhs=not _is_fp4_weight(w1),
        rhs_quant_dtype=rhs_quant_dtype,
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
    combine_in_pieces: bool = False,
    token_rows: jax.Array | None = None,
    lhs_quantized: bool = False,
) -> jax.Array:
    """Run grouped GEMM for routed tokens and reduce back to tokens.

    The valid-mask gating below is a no-op for non-EP runs (mask is all
    True) and keeps the single code path uniform for both EP and non-EP.
    EP uses SparseCore ragged_gather + ragged_gather_reduce for token movement.
    ``combine_in_pieces`` says the output is reduce-scattered by the EP
    combine, which may split it (ep_combine_layout); its rows are then
    returned piece-major, as the split combine reads them.
    With ``token_rows`` (the tokens' rows), x holds only the first routed
    rows and the fused MLP reads the rest from token_rows by
    token_indices_sorted; the caller has checked that it can. With
    ``lhs_quantized`` both are int32 words of quantize_dispatch_rows rows.
    """
    group_offset = jnp.array([0], dtype=jnp.int32)
    dtype = SOURCE_DTYPE if lhs_quantized else x.dtype

    # One kernel for both legs keeps the gate/up output in VMEM instead of
    # writing it to HBM for the down leg to read back.
    fused_kwargs = _fused_mlp_kwargs(activation, dtype, w1, rhs_quant_dtype)
    if token_rows is not None:
        gmm2_res = gmm_v2_fused_mlp(
            x,
            w1,
            w2,
            group_sizes,
            w1_scale,
            w1_bias,
            w2_scale,
            w2_bias,
            group_offset[0],
            token_rows=token_rows,
            token_row_ids=token_indices_sorted,
            quantized_lhs_dtype=DISPATCH_QUANT_DTYPE if lhs_quantized else None,
            **fused_kwargs,
        )
    elif _is_fp4_weight(w1) == _is_fp4_weight(w2) and can_fuse_mlp(
        x, w1, w2, group_sizes, w1_scale, w1_bias, w2_scale, w2_bias, **fused_kwargs
    ):
        gmm2_res = gmm_v2_fused_mlp(
            x,
            w1,
            w2,
            group_sizes,
            w1_scale,
            w1_bias,
            w2_scale,
            w2_bias,
            group_offset[0],
            **fused_kwargs,
        )
    else:
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
        gmm1_res = gmm1_res[:, : w2.shape[1] * packing_factor]

        gmm2_res = gmm_wrapper(
            gmm1_res,
            w2,
            w2_scale,
            w2_bias,
            group_sizes,
            group_offset,
            zero_initialize=False,
            preferred_element_type=dtype,
            rhs_quant_dtype=rhs_quant_dtype,
        )

    piece_rows = combine_piece_rows(num_tokens) if combine_in_pieces else ()
    # The TensorCore combine writes the pieces from token-ordered routes, one
    # call per piece; any other combine is given the routes piece-major.
    tc_pieces = (
        piece_rows
        if piece_rows
        and use_ep
        and use_sparse_core
        and argsort_revert_indices.size > onehot_moe_permute_threshold
        and _runs_tc_combine(gmm2_res, topk_weights_flat, num_tokens, topk, piece_rows)
        else ()
    )
    if piece_rows and not tc_pieces:
        argsort_revert_indices, topk_weights_flat, valid_mask_flat = (
            _piece_major_routes(
                piece_rows,
                topk,
                argsort_revert_indices,
                topk_weights_flat,
                valid_mask_flat,
            )
        )
    topk_weights = topk_weights_flat.reshape((num_tokens, topk))
    valid_mask = valid_mask_flat.reshape((num_tokens, topk))

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
        sc_info = tpu_info.sparse_core if tpu_info is not None else None
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
                ).astype(dtype)
    if use_ep and use_sparse_core:
        if argsort_revert_indices.size <= onehot_moe_permute_threshold:
            # Use onehot + matmul for unpermutation, which can be faster
            # for small batch size.
            # `argsort_revert_indices` is the inverse of the sort, so scattering
            # through it is the gather by the (unmaterialised) forward one.
            topk_weights_sorted = (
                jnp.zeros_like(topk_weights_flat)
                .at[argsort_revert_indices]
                .set(topk_weights_flat)
            )
            valid_count = group_sizes.sum(dtype=jnp.int32)[None]
            # The blockwise unpermute writes rows in token order.
            if (
                envs.TPU_MOE_OWNER_OUTPUT_MODE.lower() == "on"
                and not piece_rows
                and can_use_blockwise_onehot_unpermute(
                    gmm2_res,
                    token_indices_sorted,
                    topk_weights_sorted,
                    valid_count,
                    num_tokens=num_tokens,
                )
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
                ).astype(dtype)
            # Rows past the computed prefix (non-local experts under EP,
            # skipped padded tokens) are never written by the GMM; zero them
            # so stale NaNs cannot spread through the matmul (0 * NaN == NaN).
            computed = jnp.arange(gmm2_res.shape[0], dtype=jnp.int32) < group_sizes.sum(
                dtype=jnp.int32
            )
            gmm2_res = jnp.where(computed[:, None], gmm2_res, 0)
            revert_indices = argsort_revert_indices.reshape(num_tokens, topk)
            onehot = jax.nn.one_hot(
                revert_indices, argsort_revert_indices.size, dtype=gmm2_res.dtype
            )
            combine = (onehot * topk_weights[..., None] * valid_mask[..., None]).sum(
                axis=1
            )
            return (combine @ gmm2_res).astype(dtype)
        if _runs_tc_combine(gmm2_res, topk_weights_flat, num_tokens, topk):
            # Same result as v2's SparseCore kernel (v2 falls back to XLA only
            # for inputs far smaller than this). The TensorCore would otherwise
            # idle while the SparseCore runs v2, and it sums the rows sooner.
            return gather_unpermute(
                gmm2_res,
                argsort_revert_indices,
                topk_weights_flat,
                valid_mask_flat,
                num_tokens=num_tokens,
                topk=topk,
                chunk_pieces=tc_pieces,
            ).astype(dtype)
        return ragged_gather_reduce(
            gmm2_res,
            argsort_revert_indices,
            topk_weights_flat,
            valid_mask_flat,
            reduce_group_size=topk,
        ).astype(dtype)

    token_hidden = gmm2_res[argsort_revert_indices]
    token_topk_hidden = token_hidden.reshape((num_tokens, topk, gmm2_res.shape[-1]))
    token_topk_hidden = token_topk_hidden * jnp.expand_dims(topk_weights, axis=-1)
    token_topk_hidden = jnp.where(valid_mask[:, :, None], token_topk_hidden, 0)
    # FP32 top-k weights can promote BF16 outputs; keep the custom-op dtype.
    return token_topk_hidden.sum(axis=1).astype(dtype)


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
        "combine_in_pieces",
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
    combine_in_pieces: bool = False,
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

    With ``combine_in_pieces`` the rows come back piece-major whenever the EP
    combine splits a batch of this size (see ``moe_gmm``).

    hidden_states may also be the int32 words of ``quantize_dispatch_rows``
    rows, for an EP batch for which ``dispatches_quantized_rows`` holds; the
    output is then the bf16 one the unquantized rows give.
    """

    # Convert 3D scale [experts, blocks, N] to 4D [experts, blocks, 1, N]
    # expected by gmm_v2.
    if w1_scale is not None and w1_scale.ndim == 3:
        w1_scale = jnp.expand_dims(w1_scale, 2)
    if w2_scale is not None and w2_scale.ndim == 3:
        w2_scale = jnp.expand_dims(w2_scale, 2)

    num_tokens, hidden_size = hidden_states.shape
    # Rows quantized ahead arrive as int32 words, ROWS_PER_WORD rows each.
    lhs_quantized = hidden_states.dtype == jnp.int32
    row_dtype = DISPATCH_QUANT_DTYPE if lhs_quantized else hidden_states.dtype
    dtype = hidden_states.dtype
    if lhs_quantized:
        # The rows end in their block scales; w2 writes the hidden width.
        num_tokens, hidden_size, dtype = topk_ids.shape[0], w2.shape[-1], SOURCE_DTYPE
        if hidden_states.shape[0] * ROWS_PER_WORD != num_tokens:
            raise ValueError(f"{hidden_states.shape} words for {num_tokens} tokens")
    if combine_in_pieces and not use_ep:
        raise ValueError("combine_in_pieces needs use_ep: only EP combines split")

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
    topk_weights = topk_weights.astype(dtype)

    if use_ep and use_sparse_core and num_tokens * topk <= onehot_moe_permute_threshold:
        # The one-hot permute and unpermute are matmuls across all tokens, and
        # 0 * NaN == NaN: one token with a non-finite hidden state or routing
        # weight would turn every token's output into NaN. Zero those values
        # so the damage stays with that token.
        hidden_states = jnp.where(jnp.isfinite(hidden_states), hidden_states, 0)
        topk_weights = jnp.where(jnp.isfinite(topk_weights), topk_weights, 0)

    if use_ep and experts_start is not None:
        local_ids = topk_ids - experts_start
        valid = (local_ids >= 0) & (local_ids < w1.shape[0])
        topk_weights = jnp.where(valid, topk_weights, jnp.zeros_like(topk_weights))
        topk_ids = jnp.where(valid, local_ids, jnp.full_like(local_ids, -1))

    # Decode services also compile large prefill buckets. Apply the token
    # threshold before kernel admission so those buckets retain routed GMM.
    if 0 < num_tokens <= _DENSE_EXPERT_THRESHOLD and can_use_dense_expert(
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
    ):
        logger.info_once("Selected experimental dense-expert decode MoE")
        out = dense_expert_moe(
            hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids
        )
        piece_rows = combine_piece_rows(num_tokens) if combine_in_pieces else ()
        return _piece_major(out, piece_rows) if piece_rows else out

    num_routes = num_tokens * topk
    # Large batches (prefill chunks; the count is of tokens gathered across
    # EP ranks): the fused MLP fetches the routed rows itself, past a head of
    # DISPATCH_HEAD_ROWS the SparseCore gathers first if that is not 0.
    dispatch_head_only = (
        use_ep
        and use_sparse_core
        and num_tokens >= DISPATCH_SPLIT_MIN_TOKENS
        and num_routes > onehot_moe_permute_threshold
        and padded_hidden_size == hidden_size
        # Whole 32-bit words of rows (two bf16 or four fp8 rows each).
        and num_tokens % (4 // jnp.dtype(row_dtype).itemsize) == 0
        and _is_fp4_weight(w1) == _is_fp4_weight(w2)
        and can_fuse_mlp(
            jax.ShapeDtypeStruct((num_routes, hidden_states.shape[1]), row_dtype),
            w1,
            w2,
            jax.ShapeDtypeStruct((w1.shape[0],), jnp.int32),
            w1_scale,
            w1_bias,
            w2_scale,
            w2_bias,
            token_rows=True,
            quantized_lhs_dtype=DISPATCH_QUANT_DTYPE if lhs_quantized else None,
            **_fused_mlp_kwargs(activation, dtype, w1, rhs_quant_dtype),
        )
    )
    if lhs_quantized and not dispatch_head_only:
        raise ValueError(
            f"{num_tokens} quantized dispatch rows of {hidden_states.shape[1]} "
            "columns reach no fused MLP that reads them"
        )
    (
        x,
        group_sizes,
        argsort_revert_indices,
        topk_weights_flat,
        valid_mask,
        token_indices_sorted,
    ) = prepare_routed_gmm_inputs(
        hidden_states,
        topk_ids,
        topk_weights,
        local_num_experts=w1.shape[0],
        topk=topk,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=onehot_moe_permute_threshold,
        skip_padded_tokens=skip_padded_tokens,
        gather_rows=not dispatch_head_only,
    )
    if dispatch_head_only and DISPATCH_HEAD_ROWS == 0:
        # No head rows: gmm_v2_fused_mlp then takes no lhs operand.
        x = hidden_states[:0]
    elif dispatch_head_only:
        valid_count = group_sizes.sum(dtype=jnp.int32)
        x = ragged_gather_v2(
            hidden_states,
            token_indices_sorted[:DISPATCH_HEAD_ROWS],
            jnp.array([0], dtype=jnp.int32),
            jnp.minimum(valid_count, DISPATCH_HEAD_ROWS)[None],
            rows_per_word=ROWS_PER_WORD if lhs_quantized else 1,
        )
    else:
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
        combine_in_pieces=combine_in_pieces,
        token_rows=hidden_states if dispatch_head_only else None,
        lhs_quantized=lhs_quantized,
    )
    return x[:num_tokens, :hidden_size]
