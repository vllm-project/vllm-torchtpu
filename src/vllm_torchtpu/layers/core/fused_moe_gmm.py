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
from typing import NamedTuple

import jax
from jax import numpy as jnp
from jax.experimental.pallas import tpu as pltpu

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.experimental.moe_dense_expert import (
    can_use_dense_expert,
    dense_expert_moe,
)
from vllm_torchtpu.kernels.experimental.moe_dense_expert_fp4 import (
    Route,
    SharedExpert,
    can_use_dense_expert_fp4,
    dense_expert_moe_fp4,
    shared_activation,
)
from vllm_torchtpu.kernels.megablox import pair_combine, tc_combine
from vllm_torchtpu.kernels.megablox.gmm_v2 import (
    LHS_AMAX_LANES,
    get_packing_factor,
    gmm_v2,
    gmm_v2_mlp,
    takes_prequantized_lhs,
)
from vllm_torchtpu.kernels.megablox.moe_onehot_bounded import (
    bounded_onehot_combine,
    bounded_onehot_permute,
    can_use_bounded_onehot,
)
from vllm_torchtpu.kernels.megablox.moe_onehot_unpermute import (
    blockwise_onehot_unpermute,
    can_use_blockwise_onehot_unpermute,
)
from vllm_torchtpu.kernels.router_topk import (
    select_values_t as router_topk_select_values_t,
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

_ONEHOT_AUTO_CAP = 512
# Fixed at import, like the other MoE algorithm selectors.
_DENSE_EXPERT_THRESHOLD = envs.TPU_MOE_DENSE_EXPERT_THRESHOLD
_DENSE_FP4_MAX_TOKENS = envs.TPU_MOE_DENSE_FP4_MAX_TOKENS
_DENSE_FP4_SPARSE_MAX_TOKENS = envs.TPU_MOE_DENSE_FP4_SPARSE_MAX_TOKENS
_DUAL_LAYOUT_MAX_TOKENS = envs.TPU_MOE_DUAL_LAYOUT_MAX_TOKENS
# The pair combine's measured envelope (TPU_MOE_PAIR_COMBINE): MiniMax-M3's
# top-4 routing at prefill chunks.
_PAIR_COMBINE_TOPK = 4
_PAIR_COMBINE_MIN_TOKENS = 1024
# Fixed at import: one-hot permute/combine over the valid prefix only.
_BOUNDED_ONEHOT = envs.TPU_MOE_BOUNDED_ONEHOT
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
_COUNTING_SORT_BLOCK = 512
# Up to one counting-sort block on non-ragged paths, scattering token ids
# through the counting sort's positions is cheaper than the packed value sort
# it replaces and leaves the decode routing prep with no comparison sort.
# Ragged permute keeps the packed sort so SparseCore gather overlap is
# unchanged, and past one block the scatter costs more than that sort.
_COUNTING_SORT_SCATTER_MAX_ROWS = _COUNTING_SORT_BLOCK


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


class TpSlicedExperts(NamedTuple):
    """Rank r's TP slice of every routed expert: intermediate columns
    [r * i / R, (r + 1) * i / R) of each expert's gate, up and down, in the
    K-major fp4 layout and per-128-row f32 block scales of the EP weights
    (`fused_moe_func`'s dual layout)."""

    w1: jax.Array  # [E, h, 2 * i / R] fp4: [gate | up]
    w2: jax.Array  # [E, i / R, h] fp4
    s1: jax.Array  # [E, h / 128, 1, 2 * i / R] f32
    s2: jax.Array  # [E, i / 128 / R, 1, h] f32


def _fp4_to_words(w: jax.Array) -> jax.Array:
    """fp4 [..., n] -> int32 [..., n / 8]: collectives take no 4-bit type."""
    pairs = w.reshape(*w.shape[:-1], -1, 2)
    octets = jax.lax.bitcast_convert_type(pairs, jnp.uint8)
    quads = octets.reshape(*w.shape[:-1], -1, 4)
    return jax.lax.bitcast_convert_type(quads, jnp.int32)


def _words_to_fp4(words: jax.Array) -> jax.Array:
    """Inverse of `_fp4_to_words`."""
    octets = jax.lax.bitcast_convert_type(words, jnp.uint8)
    fp4 = jax.lax.bitcast_convert_type(
        octets.reshape(*words.shape[:-1], -1), jnp.float4_e2m1fn
    )
    return fp4.reshape(*words.shape[:-1], -1)


def tp_slice_sends(
    w1: jax.Array, w2: jax.Array, s1: jax.Array, s2: jax.Array, num_ranks: int
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """An EP rank's routed experts ([E_local, h, 2i] fp4 [gate | up], [E_local,
    i, h] fp4 and their [.., K/128, 1, N] f32 scales) cut into `num_ranks`
    TP slices of the intermediate dim, destination-major: rows [q * E_local,
    (q + 1) * E_local) of each result are rank q's slice of this rank's
    experts, ready for an all-to-all; fp4 travels as int32 words."""
    e, h, two_i = w1.shape
    i_r = two_i // 2 // num_ranks
    blocks = i_r // 128

    def by_rank(v, axis, width):
        # [.., i, ..] -> [R, .., i / R, ..] with the rank axis leading.
        v = v.reshape(*v.shape[:axis], num_ranks, width, *v.shape[axis + 1 :])
        return jnp.moveaxis(v, axis, 0)

    def gate_up(v):  # [E, .., 2i] -> [R, E, .., 2i / R], [gate | up] per rank
        v = v.reshape(*v.shape[:-1], 2, num_ranks, i_r)
        v = jnp.moveaxis(v, -2, 0)
        return v.reshape(*v.shape[:-2], 2 * i_r)

    def flat(v):
        return v.reshape(num_ranks * e, *v.shape[2:])

    return (
        flat(_fp4_to_words(gate_up(w1))),
        flat(_fp4_to_words(by_rank(w2, 1, i_r))),
        flat(gate_up(s1)),
        flat(by_rank(s2, 1, blocks)),
    )


def tp_slices_from_received(
    w1: jax.Array, w2: jax.Array, s1: jax.Array, s2: jax.Array
) -> TpSlicedExperts:
    """The all-to-all of `tp_slice_sends`' results (source-major, i.e. global
    expert order under linear EP placement) as this rank's TpSlicedExperts."""
    return TpSlicedExperts(_words_to_fp4(w1), _words_to_fp4(w2), s1, s2)


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
    """Native fp4 (not a packed carrier) rhs."""
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
    # fp4 weights: keep bf16 activations. Quantizing activations to fp8 collapses
    # fp4 accuracy (error compounds across MoE layers) with no decode speedup
    # (decode is weight-HBM-bound). fp8/int4 weights keep the default fp8 act.
    is_fp4_weight = _is_fp4_weight(rhs)
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


def _counting_sort_positions(
    sort_keys: jax.Array, num_buckets: int
) -> tuple[jax.Array | None, jax.Array | None]:
    """Stable counting sort of slot-major keys ``[topk, tokens]``: each
    route's position in the order a stable sort of the token-major flat keys
    (route ``t * topk + j``) gives, laid out like the keys, and the bucket
    totals (which are exactly `group_sizes`).

    `jnp.argsort(jnp.argsort(keys))` spends a second comparison sort over M
    distinct values on what is really the inverse of the first permutation.
    The keys take only `num_buckets` values, so a route's position is its
    bucket's offset plus the routes of earlier tokens in that bucket plus
    the earlier slots of its own token there. The per-token histogram's
    prefix over tokens is a triangular matmul per token block (MXU work
    proportional to tokens*block*num_buckets), and everything stays with
    tokens on lanes: a token-major [tokens * topk] layout of the router's
    slot-major output costs lane-padded relayouts per array.
    """
    topk, num_tokens = sort_keys.shape
    if num_buckets > min(_COUNTING_SORT_BLOCK, topk * num_tokens):
        # XLA miscompiled the token-major form's rank matmul here on jaxlib
        # 0.10.2 + libtpu 0.0.44.1 (positions not a permutation under jit);
        # that region keeps the comparison sorts.
        return None, None
    block = min(_COUNTING_SORT_BLOCK, num_tokens)
    pad = -num_tokens % block
    if pad:
        # Padded tokens sort after every real one (last token, sentinel
        # bucket), so the real positions are unchanged; the caller slices
        # the sentinel off `counts`.
        sort_keys = jnp.pad(
            sort_keys, ((0, 0), (0, pad)), constant_values=num_buckets - 1
        )
    buckets = jnp.arange(num_buckets, dtype=jnp.int32)[:, None]
    onehot = [sort_keys[j][None, :] == buckets for j in range(topk)]  # [B, T]
    per_token = sum(o.astype(jnp.bfloat16) for o in onehot)
    blocks = per_token.reshape(num_buckets, -1, block)
    # Routes of earlier tokens in the block, per bucket: exact (small
    # integer summands, f32 accumulation).
    earlier = jnp.triu(jnp.ones((block, block), jnp.bfloat16), 1)
    rank = jnp.einsum(
        "ebs,st->ebt", blocks, earlier, preferred_element_type=jnp.float32
    )
    hist = blocks.astype(jnp.float32).sum(axis=2)  # [B, blocks]
    counts = hist.sum(axis=1)
    lower_buckets = jnp.tril(jnp.ones((num_buckets, num_buckets), jnp.float32), -1)
    base = jnp.einsum(
        "fe,e->f",
        lower_buckets,
        counts,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )[:, None]
    if hist.shape[1] > 1:
        earlier_blocks = jnp.triu(jnp.ones((hist.shape[1],) * 2, jnp.float32), 1)
        base = base + jnp.einsum(
            "eb,bc->ec",
            hist,
            earlier_blocks,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
    start = (rank + base[:, :, None]).reshape(num_buckets, -1)
    positions = []
    for j in range(topk):
        # Earlier slots of the same token in the same bucket (the sentinel
        # can hold several).
        same = sum((sort_keys[i] == sort_keys[j]).astype(jnp.int32) for i in range(j))
        pos = jnp.sum(jnp.where(onehot[j], start, 0.0), axis=0).astype(jnp.int32)
        positions.append(pos + same)
    return jnp.stack(positions)[:, :num_tokens], counts.astype(jnp.int32)


def _zero_non_finite(x: jax.Array) -> jax.Array:
    return jnp.where(jnp.isfinite(x), x, 0)


def _use_bounded_onehot(num_tokens: int, num_slots: int, rows: jax.Array) -> bool:
    return (
        _BOUNDED_ONEHOT
        and rows.dtype == jnp.bfloat16
        and can_use_bounded_onehot(num_tokens, num_slots, rows.shape[1])
    )


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
    token_indices_flat = jnp.arange(num_tokens_local, dtype=jnp.int32).repeat(topk)

    # The masks and keys are built slot-major ([topk, tokens], how the router
    # lays them out, see route_from_scores) for the counting sort and the
    # packed sort, which feed the permute; the token-major flat forms are
    # only consumed after it.
    indices_t = topk_indices_local.T
    if skip_padded_tokens:
        # Padded tokens keep their selected expert ids, but carry zero
        # expert routing weights. Treat them as invalid to remove them
        # from GMM work.
        valid_t = (indices_t >= 0) & (topk_weights_local.T != 0)
    else:
        valid_t = indices_t >= 0
    sort_keys_t = jnp.where(valid_t, indices_t, local_num_experts)
    valid_mask = valid_t.T.reshape(-1)
    sort_keys = sort_keys_t.T.reshape(-1)
    # The keys take only `local_num_experts + 1` values, so a counting sort
    # gives the positions and `group_sizes` without a second comparison sort:
    # on the ragged permute at any row count (as before), and -- unless
    # TPU_MOE_COUNTING_SORT stands it down -- on the one-hot and non-EP paths
    # within the single sort's envelope, where it replaces that sort. On
    # non-ragged paths within `_COUNTING_SORT_SCATTER_MAX_ROWS`, scattering
    # `token_indices_flat` by those positions also removes the packed value
    # sort.
    ragged_permute = (
        use_ep
        and use_sparse_core
        and token_indices_flat.shape[0] > onehot_moe_permute_threshold
    )
    counting_sort = ragged_permute or (
        _COUNTING_SORT and token_indices_flat.shape[0] <= _SINGLE_SORT_MAX_ROWS
    )
    positions_t, bucket_counts = (
        _counting_sort_positions(sort_keys_t, local_num_experts + 1)
        if counting_sort
        else (None, None)
    )
    argsort_revert_indices = None if positions_t is None else positions_t.T.reshape(-1)
    token_bits = max(1, (num_tokens_local - 1).bit_length())
    packable = (local_num_experts + 1).bit_length() + token_bits < 31
    flat_bits = max(1, (token_indices_flat.shape[0] - 1).bit_length())
    single_sort = (
        _SINGLE_SORT
        and argsort_revert_indices is None
        and not ragged_permute
        and token_indices_flat.shape[0] <= _SINGLE_SORT_MAX_ROWS
        and (local_num_experts + 1).bit_length() + flat_bits < 31
    )
    token_indices_sorted: jax.Array | None = None
    if (
        argsort_revert_indices is not None
        and not ragged_permute
        and token_indices_flat.shape[0] <= _COUNTING_SORT_SCATTER_MAX_ROWS
    ):
        token_indices_sorted = (
            jnp.zeros_like(token_indices_flat)
            .at[positions_t.reshape(-1)]
            .set(
                jnp.tile(jnp.arange(num_tokens_local, dtype=jnp.int32), topk),
                unique_indices=True,
            )
        )
    sorted_indices = (
        jnp.argsort(sort_keys)
        if token_indices_sorted is None
        and not single_sort
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
            # token id, so their order (here slot-major) cannot matter.
            packed = jax.lax.sort(
                jnp.left_shift(sort_keys_t, token_bits).reshape(-1)
                | jnp.tile(jnp.arange(num_tokens_local, dtype=jnp.int32), topk),
                is_stable=False,
            )
            token_indices_sorted = jnp.bitwise_and(packed, (1 << token_bits) - 1)
        else:
            token_indices_sorted = token_indices_flat[sorted_indices]

    if use_ep and use_sparse_core:
        if token_indices_sorted.shape[0] <= onehot_moe_permute_threshold:
            # Use one-hot matmul for permutation, which can be faster for small
            # batch size. Non-finite hidden states are zeroed first (every
            # branch, the bounded kernel in VMEM): the matmul would spread them
            # to every slot (0 * NaN == NaN). When `argsort_revert_indices` came
            # from the counting sort, build the `[M, T]` permutation directly
            # from it so `fused_moe_gmm` does not depend on `token_indices_sorted`
            # (letting XLA DCE its scatter unless owner-output combine uses it).
            # The bounded kernel runs only the valid local prefix of slots.
            if bucket_counts is not None and _use_bounded_onehot(
                num_tokens_local, argsort_revert_indices.size, hidden_states_local
            ):
                x = bounded_onehot_permute(
                    hidden_states_local,
                    argsort_revert_indices,
                    group_sizes_local.sum(dtype=jnp.int32)[None],
                )
            elif bucket_counts is not None:
                onehot = (
                    jax.nn.one_hot(
                        argsort_revert_indices.reshape(num_tokens_local, topk),
                        argsort_revert_indices.size,
                        dtype=hidden_states_local.dtype,
                    )
                    .sum(axis=1)
                    .T
                )
                x = onehot @ _zero_non_finite(hidden_states_local)
            else:
                onehot = jax.nn.one_hot(
                    token_indices_sorted,
                    num_tokens_local,
                    dtype=hidden_states_local.dtype,
                )
                x = onehot @ _zero_non_finite(hidden_states_local)
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


# A shared expert's down projection still to apply: (activation [T, i_s]
# bf16, w2 [i_s, H] e4m3, s2 [1, H] f32).
SharedDown = tuple[jax.Array, jax.Array, jax.Array]


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
    shared_down: SharedDown | None = None,
    x_amax: jax.Array | None = None,
) -> jax.Array:
    """Run grouped GEMM for routed tokens and reduce back to tokens, plus
    the shared expert's down projection when ``shared_down`` is given (the
    TensorCore combine evaluates it in the same pass).

    ``x_amax``: ``x`` holds e4m3 rows of bf16 rows whose amax it gives, as
    gmm_v2's ``lhs_amax`` takes them; the output is bf16.

    The valid-mask gating below is a no-op for non-EP runs (mask is all
    True) and keeps the single code path uniform for both EP and non-EP.
    EP uses SparseCore ragged_gather + ragged_gather_reduce for token movement.
    """
    group_offset = jnp.array([0], dtype=jnp.int32)
    out_dtype = x.dtype if x_amax is None else jnp.dtype(jnp.bfloat16)

    topk_weights = topk_weights_flat.reshape((num_tokens, topk))
    valid_mask = valid_mask_flat.reshape((num_tokens, topk))

    if (
        _is_fp4_weight(w1)
        and _is_fp4_weight(w2)
        and get_packing_factor(w1.dtype, rhs_quant_dtype) == 1
        and get_packing_factor(w2.dtype, rhs_quant_dtype) == 1
        and w1_bias is None
        and w2_bias is None
    ):
        # One kernel for both gmms (gmm_v2_mlp): down's first weight block
        # loads under gate_up's last step instead of after it. Packing
        # factor 1 means rhs_quant_dtype carries nothing gmm_v2_mlp needs.
        gmm2_res = gmm_v2_mlp(
            x,
            w1,
            w2,
            group_sizes,
            rhs1_scale=w1_scale,
            rhs2_scale=w2_scale,
            group_offset=group_offset,
            fuse_act=activation,
            preferred_element_type=out_dtype,
            maybe_quantize_lhs=False,
            lhs_amax=x_amax,
        )
    else:
        assert x_amax is None
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
            preferred_element_type=out_dtype,
            rhs_quant_dtype=rhs_quant_dtype,
        )
    if (
        envs.TPU_MOE_PAIR_COMBINE
        and not use_ep
        and topk == _PAIR_COMBINE_TOPK
        and num_tokens >= _PAIR_COMBINE_MIN_TOKENS
        and pair_combine.supported(
            num_tokens,
            gmm2_res.shape[0],
            gmm2_res.shape[1],
            topk,
            gmm2_res.dtype,
            topk_weights_flat.dtype,
        )
    ):
        # TP-sliced experts: every expert is local, so a token's rows lie in
        # every expert's range; the pair kernel DMAs each route's own row pair.
        # Invalid routes read pair 0, and a valid row's pair can hold the first
        # row past the GMM's valid prefix: zero the rows the GMM did not write
        # among those so a zero weight cannot carry stale NaN.
        rows = gmm2_res.shape[0]
        computed = group_sizes.sum(dtype=jnp.int32)
        for edge in (computed, jnp.maximum(computed, 1)):
            row = jnp.minimum(edge, rows - 1)
            unwritten = (edge >= computed) & (edge < rows)
            gmm2_res = gmm2_res.at[row].set(
                jnp.where(unwritten, 0, gmm2_res[row]).astype(gmm2_res.dtype)
            )
        return _add_shared_down(
            pair_combine.pair_combine(
                gmm2_res,
                argsort_revert_indices,
                topk_weights_flat,
                valid_mask_flat,
                topk=topk,
            ).astype(out_dtype),
            shared_down,
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
                return _add_shared_down(
                    ragged_gather_reduce_v3(
                        gmm2_res,
                        argsort_revert_indices,
                        topk_weights_flat,
                        valid_mask_flat,
                        reduce_group_size=topk,
                    ).astype(out_dtype),
                    shared_down,
                )
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
            if (
                envs.TPU_MOE_OWNER_OUTPUT_MODE.lower() == "on"
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
                return _add_shared_down(
                    blockwise_onehot_unpermute(
                        gmm2_res,
                        token_indices_sorted,
                        topk_weights_sorted,
                        valid_count,
                        num_tokens=num_tokens,
                    ).astype(out_dtype),
                    shared_down,
                )
            # The dense combine one-hot takes the weights' dtype; the kernel
            # multiplies in the routed rows' dtype.
            if topk_weights.dtype == gmm2_res.dtype and _use_bounded_onehot(
                num_tokens, argsort_revert_indices.size, gmm2_res
            ):
                # The kernel adds the shared expert's down projection, so the
                # routed sum does not round-trip HBM through a separate add.
                fuse = (
                    shared_down is not None
                    and gmm2_res.dtype == out_dtype
                    and can_use_bounded_onehot(
                        num_tokens,
                        argsort_revert_indices.size,
                        gmm2_res.shape[1],
                        shared_inter=shared_down[0].shape[1],
                    )
                )
                return _add_shared_down(
                    bounded_onehot_combine(
                        gmm2_res,
                        argsort_revert_indices,
                        jnp.where(valid_mask, topk_weights, 0),
                        valid_count,
                        shared_down if fuse else None,
                    ).astype(out_dtype),
                    None if fuse else shared_down,
                )
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
            return _add_shared_down((combine @ gmm2_res).astype(out_dtype), shared_down)
        tc_args = (num_tokens, gmm2_res.shape[-1], topk, group_sizes.shape[0])
        if envs.TPU_MOE_TC_COMBINE and tc_combine.supported(*tc_args, gmm2_res.dtype):
            fuse = shared_down is not None and tc_combine.supported(
                *tc_args, gmm2_res.dtype, shared_inter=shared_down[0].shape[1]
            )
            pieces = _combine_pieces(num_tokens, tc_args[1:], gmm2_res.dtype)
            per = num_tokens // pieces
            # Slot-major [topk, tokens]: when the counting sort ran, the
            # routing prep built them so (prepare_routed_gmm_inputs) and these
            # transposes cancel its; on its argsort path they are relayouts.
            pos_t, weights_t, valid_t = (
                a.reshape(num_tokens, topk).T
                for a in (argsort_revert_indices, topk_weights_flat, valid_mask_flat)
            )
            out = []
            for i in range(pieces):
                tokens = slice(i * per, (i + 1) * per)
                piece_shared = None
                if shared_down is not None:
                    act, w2, s2 = shared_down
                    piece_shared = (act[tokens], w2, s2)
                routed = tc_combine.tc_combine(
                    gmm2_res,
                    pos_t[:, tokens],
                    weights_t[:, tokens],
                    valid_t[:, tokens],
                    group_sizes,
                    topk=topk,
                    shared_down=piece_shared if fuse else None,
                    # The routes are sorted stably by expert (above), so
                    # each expert's rows are in token order.
                    fallback=pieces == 1,
                ).astype(out_dtype)
                # Per piece, so a piece's consumer need not wait for the rest.
                out.append(_add_shared_down(routed, None if fuse else piece_shared))
            return out[0] if pieces == 1 else jnp.concatenate(out)
        return _add_shared_down(
            ragged_gather_reduce(
                gmm2_res,
                argsort_revert_indices,
                topk_weights_flat,
                valid_mask_flat,
                reduce_group_size=topk,
            ).astype(out_dtype),
            shared_down,
        )

    token_hidden = gmm2_res[argsort_revert_indices]
    token_topk_hidden = token_hidden.reshape((num_tokens, topk, gmm2_res.shape[-1]))
    token_topk_hidden = token_topk_hidden * jnp.expand_dims(topk_weights, axis=-1)
    token_topk_hidden = jnp.where(valid_mask[:, :, None], token_topk_hidden, 0)
    # FP32 top-k weights can promote BF16 outputs; keep the custom-op dtype.
    return _add_shared_down(
        token_topk_hidden.sum(axis=1).astype(out_dtype), shared_down
    )


# Mixed / prefill-sized steps only: decode and drafter buckets (<= 512 rows)
# keep one combine call, and every piece keeps >= 2048 tokens (with 2 pieces)
# like the SP pieces it feeds (TPU_MINIMAX_M3_SP_FFN_PIECES_MIN_TOKENS).
_COMBINE_PIECES_MIN_TOKENS = 4096


def _combine_pieces(num_tokens: int, args: tuple, dtype) -> int:
    """TPU_MOE_TC_COMBINE_PIECES on steps of at least 4096 tokens whose
    pieces the kernel takes; else 1."""
    pieces = envs.TPU_MOE_TC_COMBINE_PIECES
    if pieces <= 1 or num_tokens < _COMBINE_PIECES_MIN_TOKENS or num_tokens % pieces:
        return 1
    return pieces if tc_combine.supported(num_tokens // pieces, *args, dtype) else 1


# The largest e4m3 magnitude, the amax every quantized row is scaled to.
_E4M3_MAX = float(jnp.finfo(jnp.float8_e4m3fn).max)

# MiniMaxM3Mxfp8LinearMethod materializes the bf16 weight from this many rows.
_SHARED_UPCAST_MIN_ROWS = 4096


def _shared_projection(v: jax.Array, w: jax.Array, scale: jax.Array) -> jax.Array:
    w = w.astype(jnp.bfloat16)
    if v.shape[0] >= _SHARED_UPCAST_MIN_ROWS:
        w = jax.lax.optimization_barrier(w)
    y = jnp.matmul(v, w).astype(jnp.float32) * scale
    return y.astype(jnp.bfloat16)


def shared_expert_down(x: jax.Array, shared: SharedExpert) -> SharedDown:
    """`shared`'s gate_up and activation on the bf16 tokens ``x``."""
    gate, up = jnp.split(_shared_projection(x, shared.w1, shared.s1), 2, axis=-1)
    act = shared_activation(gate.astype(jnp.float32), up.astype(jnp.float32))
    return act.astype(jnp.bfloat16), shared.w2, shared.s2


def shared_activation_e4m3(
    q: jax.Array, row_scale: jax.Array, w1: jax.Array, s1: jax.Array
) -> jax.Array:
    """`shared_expert_down`'s activation on e4m3 rows ``q`` with f32 row
    scales (an SP step's gathered rows): gate_up contracts e4m3 x e4m3 on the
    MXU's fp8 rate, rounded where `_shared_projection` rounds, from the rows
    before their bf16 dequantization."""
    acc = jax.lax.dot(q, w1, preferred_element_type=jnp.float32)
    y = (acc * row_scale).astype(jnp.bfloat16).astype(jnp.float32) * s1
    gate, up = jnp.split(y.astype(jnp.bfloat16), 2, axis=-1)
    return shared_activation(gate.astype(jnp.float32), up.astype(jnp.float32)).astype(
        jnp.bfloat16
    )


def _shared_down(
    x: jax.Array,
    shared: SharedExpert,
    rows: tuple[tuple[jax.Array, jax.Array], ...] | None,
) -> SharedDown:
    """`shared_expert_down` on the bf16 tokens ``x``, or on their gathered e4m3
    row pieces ``rows`` when given (`shared_activation_e4m3`)."""
    if rows is None:
        return shared_expert_down(x, shared)
    act = jnp.concatenate(
        [shared_activation_e4m3(q, scale, shared.w1, shared.s1) for q, scale in rows]
    )
    return act, shared.w2, shared.s2


def _add_shared_down(routed: jax.Array, shared_down: SharedDown | None) -> jax.Array:
    if shared_down is None:
        return routed
    return _shared_projection(*shared_down) + routed


def shared_expert_mlp(
    x: jax.Array, shared: SharedExpert, routed: jax.Array
) -> jax.Array:
    """``routed`` plus `shared` on the bf16 tokens ``x``, rounded as
    SharedExpert describes."""
    return _add_shared_down(routed, shared_expert_down(x, shared))


def route_from_scores(
    scores: jax.Array, choice: jax.Array, route: Route
) -> tuple[jax.Array, jax.Array]:
    """``(topk_weights, topk_ids)`` of the [tokens, E] f32 sigmoid ``scores``
    and ``choice`` (scores + correction bias), as moe_routing.select_experts
    computes them with the rowmax top-k."""
    # Slot-major ([topk, tokens], tokens on lanes) as the kernel writes it:
    # the transposes below are layout-free views for the routing prep, which
    # builds its sort keys slot-major (prepare_routed_gmm_inputs); a
    # token-major [tokens, topk] layout costs a lane-padded relayout per array.
    weights, ids = router_topk_select_values_t(choice, scores, route.topk)
    weights = weights / jnp.maximum(weights.sum(axis=0, keepdims=True), 1e-20)
    if route.scale != 1.0:
        weights = weights * route.scale
    return weights.astype(route.weight_dtype).T, ids.T


def _e4m3_routed_rows(
    rows: tuple[tuple[jax.Array, jax.Array], ...] | None,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w1_bias: jax.Array | None,
    w2_bias: jax.Array | None,
    *,
    rhs_quant_dtype: jnp.dtype | None,
    num_tokens: int,
    topk: int,
    activation: str,
    ragged_permute: bool,
) -> tuple[jax.Array, jax.Array] | None:
    """The tokens' e4m3 rows ``[T, H]`` and each row's amax as a bf16 row
    (f32, broadcast over ``[T, LHS_AMAX_LANES]`` as gmm_v2's lhs_amax takes
    it), for gate_up to take in place of the bf16 rows, or None.

    The rows come from an SP step's e4m3 all-gather pieces (``rows``, e4m3
    rows with f32 scales), which the routed path otherwise takes as
    ``bf16(q * scale)``. Their quantizers (sp_norm, model._quantize_rows)
    scale each row's amax to e4m3's 448 (a row with amax under their 1e-12
    floor excepted), so the W8A8 gate_up's per-row quantization of those
    bf16 rows returns ``q`` itself (its two bf16 roundings move a value by
    under 2^-7, below half an e4m3 step), with the scale of their amax,
    ``bf16(448 * scale)``: the gate_up output is unchanged, without the
    dequantized rows. Taken where the SparseCore gathers the routed rows and
    gate_up runs W8A8 on every row bucket.
    """
    if (
        rows is None
        or not ragged_permute
        or w1_bias is not None
        or w2_bias is not None
        or not (_is_fp4_weight(w1) and _is_fp4_weight(w2))
        or get_packing_factor(w1.dtype, rhs_quant_dtype) != 1
        or get_packing_factor(w2.dtype, rhs_quant_dtype) != 1
        or w1_scale is None
        or w2_scale is None
        or not takes_prequantized_lhs(
            num_tokens * topk, w1, w1_scale, w1.shape[0], fuse_act=activation
        )
    ):
        return None
    q = jnp.concatenate([q for q, _ in rows])
    scale = jnp.concatenate([s for _, s in rows]).reshape(-1)
    # The bf16 rounding of the dequantized amax, as reduce_precision: XLA may
    # elide a convert pair (f32 -> bf16 -> f32) inside a fusion (excess
    # precision), which here changed 3291 of 50M outputs on tpu7x.
    amax = jax.lax.reduce_precision(
        jnp.float32(_E4M3_MAX) * scale.astype(jnp.float32),
        exponent_bits=8,
        mantissa_bits=7,
    )
    return q, jnp.broadcast_to(amax[:, None], (q.shape[0], LHS_AMAX_LANES))


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
        "route",
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
    shared: SharedExpert | None = None,
    tp_experts: TpSlicedExperts | None = None,
    shared_down: SharedDown | None = None,
    route: Route | None = None,
    shared_rows: tuple[tuple[jax.Array, jax.Array], ...] | None = None,
    routed_rows: tuple[tuple[jax.Array, jax.Array], ...] | None = None,
) -> jax.Array:
    """Run MoE with precomputed expert ids and weights.

    With ``route``, ``topk_weights`` and ``topk_ids`` are instead the [tokens,
    E] f32 sigmoid routing scores and the scores plus correction bias: the
    dense fp4 kernel's steps (all rows or sparse rows) route in the same
    call, every other path routes first (`route_from_scores`).

    ``tp_experts`` is the dual layout of EP weights: this rank's TP slice of
    every routed expert. Steps of up to TPU_MOE_DUAL_LAYOUT_MAX_TOKENS tokens
    run the dense fp4 kernel on it with the global ids (every rank streams
    the same share of the touched experts, where EP leaves the busiest rank
    to set the step time); the result is a per-rank partial like the EP one.

    With ``shared``, an always-on expert's output (rounded to bf16) is added to
    the routed output, as vLLM's MoERunner adds a separately computed shared
    expert; the dense fp4 kernel evaluates it in the same call where it serves
    the step. ``shared_rows`` are the e4m3 rows (with f32 row scales) the
    tokens were gathered as, in pieces: on steps the dense fp4 kernel does not
    serve, the shared expert's gate_up then contracts them as e4m3
    (`shared_activation_e4m3`). With ``shared_down`` the same rows come as
    ``routed_rows``, for the routed experts (`_e4m3_routed_rows`).

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

    num_tokens, hidden_size = hidden_states.shape

    dual = (
        tp_experts is not None
        and use_ep
        and 0 < num_tokens <= _DUAL_LAYOUT_MAX_TOKENS
        and rhs_quant_dtype is None
        and w1_bias is None
        and w2_bias is None
        and can_use_dense_expert_fp4(
            hidden_states, *tp_experts, activation=activation, shared=shared
        )
    )
    # Above the dense cap and up to the sparse one, the kernel evaluates each
    # expert on its own routes' rows only (16-row tiles).
    sparse_rows = num_tokens > _DENSE_FP4_MAX_TOKENS
    dense_fp4 = (
        not dual
        and 0 < num_tokens <= max(_DENSE_FP4_MAX_TOKENS, _DENSE_FP4_SPARSE_MAX_TOKENS)
        and rhs_quant_dtype is None
        and w1_bias is None
        and w2_bias is None
        and can_use_dense_expert_fp4(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            activation=activation,
            shared=shared,
            sparse_rows=sparse_rows,
        )
    )
    if route is not None and not dense_fp4:
        topk_weights, topk_ids = route_from_scores(topk_weights, topk_ids, route)
        route = None
    if shared is not None and activation != "swigluoai":
        raise NotImplementedError(f"shared experts take swigluoai, not {activation}")
    if shared is not None and not (dense_fp4 or dual):
        # The routed op adds the shared expert's down projection (the
        # TensorCore combine fuses it).
        return fused_moe_func(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            w1_bias,
            w2_bias,
            topk_weights,
            topk_ids,
            experts_start,
            topk=topk,
            activation=activation,
            use_ep=use_ep,
            use_sparse_core=use_sparse_core,
            onehot_moe_permute_threshold=onehot_moe_permute_threshold,
            rhs_quant_dtype=rhs_quant_dtype,
            skip_padded_tokens=skip_padded_tokens,
            shared_down=_shared_down(hidden_states, shared, shared_rows),
            routed_rows=shared_rows,
        )

    # NVFP4 weights arrive as native float4_e2m1fn in gmm_v2's K-major layout
    # (unpacked once at load; see fused_moe.load_kmajor_fp4). gmm_v2 picks the
    # regime from the block size: block-16 -> W4A16, block >= MXU -> W4A8.
    _, padded_hidden_size, _ = w1.shape
    packing_factor = get_packing_factor(w1.dtype, rhs_quant_dtype)
    padded_hidden_size *= packing_factor

    if route is not None:
        logger.info_once("Selected dense-expert fp4 decode MoE with fused routing")
        return _add_shared_down(
            dense_expert_moe_fp4(
                hidden_states,
                w1,
                w2,
                w1_scale,
                w2_scale,
                topk_weights,
                topk_ids,
                activation=activation,
                shared=shared,
                sparse_rows=sparse_rows,
                route=route,
                experts_start=experts_start if use_ep else None,
            ),
            shared_down,
        )

    assert topk_weights.shape == (num_tokens, topk)
    assert topk_ids.shape == (num_tokens, topk)

    # Pin topk_weights to hidden_states.dtype (or float32 if
    # TPU_MOE_TOPK_WEIGHTS_FLOAT32 is enabled) so the EP mask, GMM reduce,
    # and `where` propagate appropriately.
    topk_weights = topk_weights.astype(_get_topk_weights_dtype(hidden_states.dtype))

    if dual:
        logger.info_once("Selected dense fp4 decode MoE on TP-sliced experts")
        return _add_shared_down(
            dense_expert_moe_fp4(
                hidden_states,
                *tp_experts,
                topk_weights,
                topk_ids,
                activation=activation,
                shared=shared,
            ),
            shared_down,
        )

    onehot_permute = (
        use_ep and use_sparse_core and num_tokens * topk <= onehot_moe_permute_threshold
    )
    if onehot_permute:
        # The one-hot permute and unpermute are matmuls across all tokens, and
        # 0 * NaN == NaN: one token with a non-finite hidden state or routing
        # weight would turn every token's output into NaN. Zero those values
        # so the damage stays with that token (the permute zeroes the hidden
        # states, see prepare_routed_gmm_inputs).
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
        if onehot_permute:
            hidden_states = _zero_non_finite(hidden_states)
        return _add_shared_down(
            dense_expert_moe(
                hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids
            ),
            shared_down,
        )

    # Decode: a core holds few routed rows per expert and weight streaming
    # binds, so the permutation-free kernel skips the routing sort, permute
    # and combine around the GMM (EP ranks, or TP ranks holding every expert's
    # slice).
    if dense_fp4:
        logger.info_once("Selected dense-expert fp4 decode MoE")
        return _add_shared_down(
            dense_expert_moe_fp4(
                hidden_states,
                w1,
                w2,
                w1_scale,
                w2_scale,
                topk_weights,
                topk_ids,
                activation=activation,
                shared=shared,
                sparse_rows=sparse_rows,
            ),
            shared_down,
        )

    ragged_permute = (
        use_ep and use_sparse_core and num_tokens * topk > onehot_moe_permute_threshold
    )
    e4m3_rows = _e4m3_routed_rows(
        routed_rows,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        rhs_quant_dtype=rhs_quant_dtype,
        num_tokens=num_tokens,
        topk=topk,
        activation=activation,
        ragged_permute=ragged_permute,
    )
    (
        x,
        group_sizes,
        argsort_revert_indices,
        topk_weights_flat,
        valid_mask,
        token_indices_sorted,
    ) = prepare_routed_gmm_inputs(
        hidden_states if e4m3_rows is None else e4m3_rows[0],
        topk_ids,
        topk_weights,
        local_num_experts=w1.shape[0],
        topk=topk,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=onehot_moe_permute_threshold,
        skip_padded_tokens=skip_padded_tokens,
    )
    x_amax = None
    if e4m3_rows is not None:
        # The rows' amax gathered beside them: a TensorCore gather of 32-bit
        # scalars runs at ~5 ns a row (178 us for 32768 rows, tpu7x).
        x_amax = ragged_gather_v2(
            e4m3_rows[1],
            token_indices_sorted,
            jnp.array([0], dtype=jnp.int32),
            group_sizes.sum(dtype=jnp.int32)[None],
        )
    if shared_down is not None and ragged_permute:
        # The whole shared activation before gate_up: it fills the wait on
        # the SparseCore gather instead of running between combine pieces.
        x, act = jax.lax.optimization_barrier((x, shared_down[0]))
        shared_down = (act, *shared_down[1:])
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
        shared_down=shared_down if padded_hidden_size == hidden_size else None,
        x_amax=x_amax,
    )
    if padded_hidden_size != hidden_size:
        return _add_shared_down(x[:num_tokens, :hidden_size], shared_down)
    return x[:num_tokens, :hidden_size]
