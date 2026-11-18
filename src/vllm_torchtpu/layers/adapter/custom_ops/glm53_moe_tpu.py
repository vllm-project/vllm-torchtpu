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
"""One-dispatch TPU MoE block for GLM 5.3 Flash.

Decode on TPU is host-dispatch bound: the vLLM `FusedMoE` path runs the router
GEMM, the sigmoid/bias/top-k/renorm routing, the routed-expert GMM, the shared
expert (two FP8 matmuls and a clamped SwiGLU) and the combine as ~20 separate
host dispatches per layer.

`get_glm53_fused_moe_op` runs the same math as ONE `jax_op` per layer: router
GEMM -> fused Pallas `noaux_tc` routing -> sort-free routed GMMs -> shared
expert -> `shared + routed`. The TP all-reduce stays outside, exactly where
vLLM's `MoERunner` puts it (on the sum). `VLLM_TPU_GLM53_MOE_IMPL=vllm` keeps
the vLLM `FusedMoE` path instead.

Numerics follow the vLLM path op for op:
  * router logits are a bf16 GEMM rounded to bf16, then cast to fp32
    (`GateLinear` on TPU falls through to `F.linear` + `.to(float32)`);
  * `select_experts`: fp32 sigmoid, `+ e_score_correction_bias` for the
    choice, un-biased scores gathered, renormalized (clamp 1e-20), scaled by
    `routed_scaling_factor`, cast to the activation dtype;
  * the shared expert reuses `_quantized_matmul_jax` (the FP8 linear bridge).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

MOE_IMPL_FUSED = "fused"
MOE_IMPL_VLLM = "vllm"

# Clamp floor below any real routing score and strictly above finfo.min.
_ROUTER_NEG = -3.0e38
_ROUTER_MAX_BLOCK_ROWS = 512


def glm53_moe_impl() -> str:
    """The MoE implementation selected by `VLLM_TPU_GLM53_MOE_IMPL`."""
    from vllm_torchtpu import envs

    raw = envs.VLLM_TPU_GLM53_MOE_IMPL
    impl = (raw or MOE_IMPL_FUSED).strip().lower()
    if impl not in (MOE_IMPL_FUSED, MOE_IMPL_VLLM):
        raise ValueError(
            f"VLLM_TPU_GLM53_MOE_IMPL={raw!r}; expected "
            f"{MOE_IMPL_FUSED!r} or {MOE_IMPL_VLLM!r}."
        )
    return impl


def clamped_swiglu_jax(gate_up: jax.Array, *, limit: float) -> jax.Array:
    """`silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)` in JAX.

    Same op order and dtypes as the torch expression
    `F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)` with
    `gate, up = gate_up.chunk(2, dim=-1)`.
    """
    gate, up = jnp.split(gate_up, 2, axis=-1)
    gate = jnp.minimum(gate, limit)
    up = jnp.clip(up, -limit, limit)
    # torch's bf16 silu computes in fp32 and rounds once.
    act = jax.nn.silu(gate.astype(jnp.float32)).astype(gate.dtype)
    return act * up


def _glm53_router_kernel(
    logits_ref: Any,
    bias_ref: Any,
    w_ref: Any,
    i_ref: Any,
    *,
    topk: int,
    n: int,
    routed_scale: float,
    renormalize: bool,
) -> None:
    """Fused Pallas kernel for GLM 5.3 `noaux_tc` routing.

    Computes fp32 sigmoid(logits), adds `score_bias` in VREGs, selects top
    `topk` experts by repeated max and mask while extracting the unbiased
    sigmoid scores on the fly, renormalizes and scales by `routed_scale`, and
    stores `(weights, ids)` directly from VMEM.
    """
    logits = logits_ref[...].astype(jnp.float32)
    scores = jax.nn.sigmoid(logits)
    biased = scores + bias_ref[...]
    x = jnp.where(biased > _ROUTER_NEG, biased, _ROUTER_NEG)
    masked = jnp.finfo(jnp.float32).min
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    unbiased_cols = []
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
        sel = iota == idx[:, None]
        w_j = jnp.max(jnp.where(sel, scores, -1.0), axis=1, keepdims=True)
        unbiased_cols.append(w_j)
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(sel, masked, x)

    if renormalize:
        w_sum = unbiased_cols[0]
        for j in range(1, topk):
            w_sum = w_sum + unbiased_cols[j]
        inv_sum = jnp.float32(routed_scale) / jnp.maximum(w_sum, 1e-20)
        for j in range(topk):
            w_ref[:, j] = (unbiased_cols[j][:, 0] * inv_sum[:, 0]).astype(w_ref.dtype)
    else:
        scale_f32 = jnp.float32(routed_scale)
        for j in range(topk):
            w_ref[:, j] = (unbiased_cols[j][:, 0] * scale_f32).astype(w_ref.dtype)


def glm53_router_topk(
    logits: jax.Array,
    score_bias: jax.Array,
    *,
    topk: int,
    routed_scale: float,
    renormalize: bool,
    out_dtype: Any,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Run the fused Pallas `noaux_tc` router over `[rows, experts]` logits."""
    rows, n = logits.shape
    block = min(_ROUTER_MAX_BLOCK_ROWS, rows)
    bias_2d = score_bias.astype(jnp.float32).reshape(1, n)
    return pl.pallas_call(
        functools.partial(
            _glm53_router_kernel,
            topk=topk,
            n=n,
            routed_scale=routed_scale,
            renormalize=renormalize,
        ),
        grid=(pl.cdiv(rows, block),),
        in_specs=[
            pl.BlockSpec((block, n), lambda i: (i, 0)),
            pl.BlockSpec((1, n), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, topk), out_dtype),
            jax.ShapeDtypeStruct((rows, topk), jnp.int32),
        ],
        interpret=interpret,
    )(logits, bias_2d)


def glm53_route_fast_jax(
    x: jax.Array,
    gate_w: jax.Array,
    score_bias: jax.Array,
    *,
    topk: int,
    routed_scale: float,
    renormalize: bool,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Fused GLM 5.3 `noaux_tc` routing without a post-router gather.

    Off TPU (and without `interpret`) this falls back to `glm53_route_jax`
    with `jax.lax.top_k`.
    """
    if jax.default_backend() != "tpu" and not interpret:
        return glm53_route_jax(
            x,
            gate_w,
            score_bias,
            topk=topk,
            routed_scale=routed_scale,
            renormalize=renormalize,
            topk_fn=_lax_topk,
        )
    logits = jnp.dot(x, gate_w.T, preferred_element_type=jnp.float32).astype(
        gate_w.dtype
    )
    return glm53_router_topk(
        logits,
        score_bias,
        topk=topk,
        routed_scale=routed_scale,
        renormalize=renormalize,
        out_dtype=x.dtype,
        interpret=interpret,
    )


def glm53_route_jax(
    x: jax.Array,
    gate_w: jax.Array,
    score_bias: jax.Array,
    *,
    topk: int,
    routed_scale: float,
    renormalize: bool,
    topk_fn: Callable[[jax.Array, int], tuple[jax.Array, jax.Array]],
) -> tuple[jax.Array, jax.Array]:
    """GLM 5.3 `noaux_tc` routing; returns `(weights[x.dtype], ids[int32])`."""
    logits = jnp.dot(x, gate_w.T, preferred_element_type=jnp.float32)
    logits = logits.astype(gate_w.dtype).astype(jnp.float32)
    scores = jax.nn.sigmoid(logits)
    _, ids = topk_fn(scores + score_bias.astype(jnp.float32)[None, :], topk)
    ids = ids.astype(jnp.int32)
    weights = jnp.take_along_axis(scores, ids, axis=-1)
    if renormalize:
        weights = weights / jnp.maximum(weights.sum(axis=-1, keepdims=True), 1e-20)
    if routed_scale != 1.0:
        weights = weights * jnp.float32(routed_scale)
    return weights.astype(x.dtype), ids


def _lax_topk(scores: jax.Array, k: int) -> tuple[jax.Array, jax.Array]:
    return jax.lax.top_k(scores, k)


def _sort_activations_no_revert(
    topk_expert_indices: jax.Array,
    topk_weights: jax.Array,
    *,
    local_num_experts: int,
    use_ep: bool,
    experts_start: jax.Array | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sort tokens and weights by expert in one pass without revert indices.

    When `(sort_key, token_id, bf16_weight)` fits in 31 bits (all `T <= 512`
    under EP=8 and `T <= 64` under TP=8), packs all three fields into a single
    non negative `int32` and runs one 1-array `jax.lax.sort` with zero gathers
    and zero `_counting_sort_positions` overhead.
    """
    num_tokens, topk = topk_expert_indices.shape
    m = num_tokens * topk
    experts_flat = topk_expert_indices.reshape(-1).astype(jnp.int32)
    weights_flat = topk_weights.reshape(-1)
    if use_ep and experts_start is not None:
        experts_flat = experts_flat - experts_start.astype(jnp.int32)
        valid_mask = jnp.logical_and(
            experts_flat >= 0, experts_flat < local_num_experts
        )
        sort_keys = jnp.where(valid_mask, experts_flat, local_num_experts)
        weights_flat = jnp.where(valid_mask, weights_flat, 0)
    else:
        sort_keys = experts_flat

    token_indices_flat = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), topk)
    sort_bits = max(1, int(local_num_experts).bit_length())
    token_bits = max(1, (int(num_tokens) - 1).bit_length())
    token_mask = (1 << token_bits) - 1

    if weights_flat.dtype == jnp.bfloat16 and (sort_bits + token_bits + 16 <= 31):
        w_u16 = jax.lax.bitcast_convert_type(weights_flat, jnp.uint16).astype(jnp.int32)
        packed = (
            jax.lax.shift_left(sort_keys, jnp.int32(16 + token_bits))
            | jax.lax.shift_left(token_indices_flat, jnp.int32(16))
            | w_u16
        )
        packed_sorted = jax.lax.sort(packed, is_stable=False)
        token_indices_sorted = jax.lax.shift_right_logical(
            packed_sorted, jnp.int32(16)
        ) & jnp.int32(token_mask)
        w_u16_sorted = (packed_sorted & jnp.int32(0xFFFF)).astype(jnp.uint16)
        topk_weights_sorted = jax.lax.bitcast_convert_type(w_u16_sorted, jnp.bfloat16)
    else:
        key_tok = (
            jax.lax.shift_left(sort_keys, jnp.int32(token_bits)) | token_indices_flat
        )
        key_tok_sorted, topk_weights_sorted = jax.lax.sort(
            [key_tok, weights_flat], is_stable=False, num_keys=1
        )
        token_indices_sorted = key_tok_sorted & jnp.int32(token_mask)

    group_sizes = jax.nn.one_hot(sort_keys, local_num_experts, dtype=jnp.int32).sum(
        axis=0, dtype=jnp.int32
    )
    valid_count = (
        jnp.sum(group_sizes, dtype=jnp.int32)[None]
        if use_ep
        else jnp.array([m], dtype=jnp.int32)
    )
    return token_indices_sorted, topk_weights_sorted, group_sizes, valid_count


def _can_use_sc_v3_combine(
    route_capacity: int,
    num_tokens: int,
    hidden_size: int,
    topk: int,
    dtype: Any,
) -> bool:
    """Return whether `ragged_gather_reduce_v3` should run for this batch."""
    from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import (
        token_block_alignment,
    )
    from vllm_torchtpu.kernels.sparse_core.support import get_sparse_core_info

    if (
        jax.default_backend() != "tpu"
        or dtype != jnp.bfloat16
        or topk < 2
        or route_capacity > (1 << 20)
    ):
        return False
    try:
        tpu_info = pltpu.get_tpu_info()
    except ValueError:
        return False
    sc_info = get_sparse_core_info(tpu_info)
    if (
        sc_info is None
        or sc_info.num_lanes != 16
        or hidden_size % tpu_info.num_lanes != 0
        or route_capacity * hidden_size * 4 < tpu_info.vmem_capacity_bytes * 0.6
    ):
        return False
    alignment = token_block_alignment(hidden_size, tpu_info)
    padded_tokens = -(-num_tokens // alignment) * alignment
    return num_tokens >= alignment and padded_tokens * 2 <= num_tokens * 3


def _glm53_routed_moe_fast_jax(
    hidden_states: jax.Array,
    w1: jax.Array,
    w2: jax.Array,
    w1_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    topk_weights: jax.Array,
    topk_ids: jax.Array,
    experts_start: jax.Array | None,
    *,
    topk: int,
    activation: str,
    use_ep: bool,
    use_sparse_core: bool,
    onehot_moe_permute_threshold: int,
    layer_base: jax.Array | None = None,
) -> jax.Array:
    """Fast routed MoE path for GLM 5.3 Flash.

    With `layer_base` (int32[1], the layer index), `w1`/`w2` and their scales
    stack every MoE layer as `[L, E, ...]` (one expert copy that another
    consumer of the same weights shares); the GMMs read groups
    `layer_base * E + [0, E)` of the flattened `[L * E, ...]` view in place,
    so no per-layer slice is materialized.

    Uses `blockwise_onehot_unpermute` (or direct 2D one hot matmul for
    `M < 128`) paired with `_sort_activations_no_revert` whenever the owner
    buffer fits in VMEM (`T <= 512`), and dispatches `ragged_gather_reduce_v3`
    on SparseCore for larger prefill batches (`T >= 512`).
    """
    from vllm_torchtpu.kernels.megablox.gmm_v2 import get_packing_factor
    from vllm_torchtpu.kernels.megablox.moe_onehot_unpermute import (
        blockwise_onehot_unpermute,
        can_use_blockwise_onehot_unpermute,
    )
    from vllm_torchtpu.kernels.sparse_core.ragged_gather_reduce_v3 import (
        ragged_gather_reduce as ragged_gather_reduce_v3,
    )
    from vllm_torchtpu.kernels.sparse_core.ragged_gather_v2 import ragged_gather_v2
    from vllm_torchtpu.layers.core.fused_moe_gmm import (
        gmm_wrapper,
        prepare_routed_gmm_inputs,
        ragged_gather_reduce,
    )

    rhs_group_base = None
    num_layer_experts = w1.shape[0]
    if layer_base is not None:
        n_layers, num_layer_experts = w1.shape[0], w1.shape[1]
        flat_groups = n_layers * num_layer_experts

        def flat(a: jax.Array | None) -> jax.Array | None:
            return None if a is None else a.reshape(flat_groups, *a.shape[2:])

        w1 = w1.reshape(flat_groups, *w1.shape[2:])
        w2 = w2.reshape(flat_groups, *w2.shape[2:])
        w1_scale, w2_scale = flat(w1_scale), flat(w2_scale)
        rhs_group_base = layer_base.reshape(1).astype(jnp.int32) * num_layer_experts
    if w1_scale is not None and w1_scale.ndim == 3:
        w1_scale = jnp.expand_dims(w1_scale, 2)
    if w2_scale is not None and w2_scale.ndim == 3:
        w2_scale = jnp.expand_dims(w2_scale, 2)

    num_tokens, hidden_size = hidden_states.shape
    _, padded_hidden_size, _ = w1.shape
    packing_factor = get_packing_factor(w1.dtype, None)
    padded_hidden_size *= packing_factor
    hidden_states = jnp.where(jnp.isfinite(hidden_states), hidden_states, 0)
    topk_weights = jnp.where(jnp.isfinite(topk_weights), topk_weights, 0).astype(
        hidden_states.dtype
    )

    m = num_tokens * topk
    local_num_experts = w1.shape[0] if layer_base is None else num_layer_experts
    on_tpu = jax.default_backend() == "tpu"

    # For T <= 512 (M <= 4096), SparseCore ragged_gather_reduce falls back to
    # TensorCore gather_fusion because 2 * size(gmm2_res) < 0.6 * VMEM (38.4 MiB
    # on v7x). Use blockwise_onehot_unpermute (M >= 128) or direct 2D one-hot
    # matmul (M < 128, e.g. b8 where T=8, M=64) across that entire regime so
    # every bucket has a native MXU or SparseCore combine with no revert sort.
    if on_tpu and num_tokens <= 512:
        dummy_route = jax.ShapeDtypeStruct((m, padded_hidden_size), hidden_states.dtype)
        dummy_idx = jax.ShapeDtypeStruct((m,), jnp.int32)
        dummy_w = jax.ShapeDtypeStruct((m,), topk_weights.dtype)
        dummy_vc = jax.ShapeDtypeStruct((1,), jnp.int32)
        try:
            use_blockwise_onehot = can_use_blockwise_onehot_unpermute(
                dummy_route,  # type: ignore[arg-type]
                dummy_idx,  # type: ignore[arg-type]
                dummy_w,  # type: ignore[arg-type]
                dummy_vc,  # type: ignore[arg-type]
                num_tokens=num_tokens,
            )
        except ValueError:
            use_blockwise_onehot = False
    else:
        use_blockwise_onehot = False

    use_small_2d_onehot = (
        m < 128 and hidden_states.dtype == jnp.bfloat16
    ) or not on_tpu
    need_revert = not (use_blockwise_onehot or use_small_2d_onehot)

    if not need_revert:
        (
            token_indices_sorted,
            topk_weights_sorted,
            group_sizes,
            valid_count,
        ) = _sort_activations_no_revert(
            topk_ids,
            topk_weights,
            local_num_experts=local_num_experts,
            use_ep=use_ep,
            experts_start=experts_start,
        )
        if use_ep and use_sparse_core and on_tpu:
            if m <= onehot_moe_permute_threshold:
                onehot = jax.nn.one_hot(
                    token_indices_sorted, num_tokens, dtype=hidden_states.dtype
                )
                safe_hidden = jnp.where(jnp.isfinite(hidden_states), hidden_states, 0)
                x = onehot @ safe_hidden
            else:
                x = ragged_gather_v2(
                    hidden_states,
                    token_indices_sorted,
                    jnp.array([0], dtype=jnp.int32),
                    valid_count,
                )
        else:
            x = hidden_states[token_indices_sorted]
        argsort_revert_indices = None
        topk_weights_flat = None
        valid_mask_flat = None
    else:
        if use_ep and experts_start is not None:
            local_ids = topk_ids - experts_start
            valid = (local_ids >= 0) & (local_ids < local_num_experts)
            topk_weights_ep = jnp.where(
                valid, topk_weights, jnp.zeros_like(topk_weights)
            )
            topk_ids_ep = jnp.where(valid, local_ids, jnp.full_like(local_ids, -1))
        else:
            topk_weights_ep = topk_weights
            topk_ids_ep = topk_ids
        (
            x,
            group_sizes,
            argsort_revert_indices,
            topk_weights_flat,
            valid_mask_flat,
            token_indices_sorted,
        ) = prepare_routed_gmm_inputs(
            hidden_states,
            topk_ids_ep,
            topk_weights_ep,
            local_num_experts=local_num_experts,
            topk=topk,
            use_ep=use_ep,
            use_sparse_core=use_sparse_core,
            onehot_moe_permute_threshold=onehot_moe_permute_threshold,
            skip_padded_tokens=False,
        )
        topk_weights_sorted = None
        valid_count = None

    if padded_hidden_size > hidden_size:
        x = jnp.pad(x, ((0, 0), (0, padded_hidden_size - hidden_size)))

    group_offset = jnp.array([0], dtype=jnp.int32)
    gmm1_res = gmm_wrapper(
        x,
        w1,
        w1_scale,
        None,
        group_sizes,
        group_offset,
        zero_initialize=False,
        fuse_act=activation,
        rhs_quant_dtype=None,
        rhs_group_base=rhs_group_base,
    )
    w2_packing_factor = get_packing_factor(w2.dtype, None)
    gmm1_res = gmm1_res[:, : w2.shape[1] * w2_packing_factor]
    gmm2_res = gmm_wrapper(
        gmm1_res,
        w2,
        w2_scale,
        None,
        group_sizes,
        group_offset,
        zero_initialize=False,
        preferred_element_type=x.dtype,
        rhs_quant_dtype=None,
        rhs_group_base=rhs_group_base,
    )

    if use_blockwise_onehot:
        assert topk_weights_sorted is not None and valid_count is not None
        out = blockwise_onehot_unpermute(
            gmm2_res,
            token_indices_sorted,
            topk_weights_sorted,
            valid_count,
            num_tokens=num_tokens,
        ).astype(hidden_states.dtype)
    elif use_small_2d_onehot:
        assert topk_weights_sorted is not None and valid_count is not None
        route_idx = jnp.arange(m, dtype=jnp.int32)
        computed = route_idx < valid_count[0]
        gmm2_valid = jnp.where(computed[:, None] & jnp.isfinite(gmm2_res), gmm2_res, 0)
        owner_ids = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
        valid_weights = jnp.where(computed, topk_weights_sorted, 0).astype(
            gmm2_res.dtype
        )
        combine = (owner_ids == token_indices_sorted[None, :]).astype(
            gmm2_res.dtype
        ) * valid_weights[None, :]
        out = (combine @ gmm2_valid).astype(hidden_states.dtype)
    else:
        assert (
            argsort_revert_indices is not None
            and topk_weights_flat is not None
            and valid_mask_flat is not None
        )
        if use_sparse_core and _can_use_sc_v3_combine(
            m, num_tokens, gmm2_res.shape[-1], topk, gmm2_res.dtype
        ):
            out = ragged_gather_reduce_v3(
                gmm2_res,
                argsort_revert_indices,
                topk_weights_flat,
                valid_mask_flat,
                reduce_group_size=topk,
            ).astype(hidden_states.dtype)
        elif use_ep and use_sparse_core:
            out = ragged_gather_reduce(
                gmm2_res,
                argsort_revert_indices,
                topk_weights_flat,
                valid_mask_flat,
                reduce_group_size=topk,
            ).astype(hidden_states.dtype)
        else:
            token_hidden = gmm2_res[argsort_revert_indices].reshape(
                (num_tokens, topk, gmm2_res.shape[-1])
            )
            token_hidden = token_hidden * topk_weights[:, :, None]
            token_hidden = jnp.where(
                valid_mask_flat.reshape(num_tokens, topk)[:, :, None],
                token_hidden,
                0,
            )
            out = token_hidden.sum(axis=1).astype(hidden_states.dtype)

    return out[:num_tokens, :hidden_size]


def _glm53_moe_impl(
    x: jax.Array,
    gate_w: jax.Array,
    score_bias: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    w13_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    experts_start: jax.Array,
    sh_w13: jax.Array,
    sh_s13: jax.Array,
    sh_w2: jax.Array,
    sh_s2: jax.Array,
    layer_base: jax.Array | None,
    *,
    topk: int,
    routed_scale: float,
    renormalize: bool,
    swiglu_limit: float,
    activation: str,
    use_ep: bool,
    use_sparse_core: bool,
    onehot_threshold: int,
) -> jax.Array:
    """Router + routed experts + shared expert, un-reduced across TP/EP.

    `layer_base` (int32[1]) selects this layer inside stacked `[L, E, ...]`
    expert tensors (see `Glm53FusedMoeArgs.use_expert_store`).
    """
    from vllm_torchtpu.layers.adapter.linear_common import _quantized_matmul_jax

    x = jnp.where(jnp.isfinite(x), x, 0)
    weights, ids = glm53_route_fast_jax(
        x,
        gate_w,
        score_bias,
        topk=topk,
        routed_scale=routed_scale,
        renormalize=renormalize,
    )
    routed = _glm53_routed_moe_fast_jax(
        x,
        w13,
        w2,
        w13_scale,
        w2_scale,
        weights,
        ids,
        experts_start,
        topk=topk,
        activation=activation,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=onehot_threshold,
        layer_base=layer_base,
    )
    gate_up = _quantized_matmul_jax(x, sh_w13, sh_s13)
    shared = _quantized_matmul_jax(
        clamped_swiglu_jax(gate_up, limit=swiglu_limit), sh_w2, sh_s2
    )
    # vLLM `MoERunner`: `result = shared_output + fused_output`.
    return shared + routed.astype(shared.dtype)


def _glm53_moe_jax(
    x: jax.Array,
    gate_w: jax.Array,
    score_bias: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    w13_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    experts_start: jax.Array,
    sh_w13: jax.Array,
    sh_s13: jax.Array,
    sh_w2: jax.Array,
    sh_s2: jax.Array,
    *,
    topk: int,
    routed_scale: float,
    renormalize: bool,
    swiglu_limit: float,
    activation: str,
    use_ep: bool,
    use_sparse_core: bool,
    onehot_threshold: int,
) -> jax.Array:
    """The fused MoE op body for per-layer expert tensors."""
    return _glm53_moe_impl(
        x,
        gate_w,
        score_bias,
        w13,
        w2,
        w13_scale,
        w2_scale,
        experts_start,
        sh_w13,
        sh_s13,
        sh_w2,
        sh_s2,
        None,
        topk=topk,
        routed_scale=routed_scale,
        renormalize=renormalize,
        swiglu_limit=swiglu_limit,
        activation=activation,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
        onehot_threshold=onehot_threshold,
    )


def _glm53_moe_stacked_jax(
    x: jax.Array,
    gate_w: jax.Array,
    score_bias: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    w13_scale: jax.Array,
    w2_scale: jax.Array,
    experts_start: jax.Array,
    sh_w13: jax.Array,
    sh_s13: jax.Array,
    sh_w2: jax.Array,
    sh_s2: jax.Array,
    layer_base: jax.Array,
    *,
    topk: int,
    routed_scale: float,
    renormalize: bool,
    swiglu_limit: float,
    activation: str,
    use_ep: bool,
    use_sparse_core: bool,
    onehot_threshold: int,
) -> jax.Array:
    """The fused MoE op body for a stacked expert store.

    `w13`/`w2` (and their scales) are `[L, E, ...]` and `layer_base`
    (int32[1]) is this layer's index into them.
    """
    return _glm53_moe_impl(
        x,
        gate_w,
        score_bias,
        w13,
        w2,
        w13_scale,
        w2_scale,
        experts_start,
        sh_w13,
        sh_s13,
        sh_w2,
        sh_s2,
        layer_base,
        topk=topk,
        routed_scale=routed_scale,
        renormalize=renormalize,
        swiglu_limit=swiglu_limit,
        activation=activation,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
        onehot_threshold=onehot_threshold,
    )


_FUSED_MOE_OPS: dict[tuple[Any, ...], Callable] = {}


def _fake_fused_moe(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.empty_like(x)


def get_glm53_fused_moe_op(
    *,
    topk: int,
    routed_scale: float,
    renormalize: bool,
    swiglu_limit: float,
    activation: str,
    use_ep: bool = False,
    stacked: bool = False,
) -> Callable:
    """One cached `jax_op` per static configuration (shared by all layers).

    `stacked` builds the shared expert store variant: the expert operands are
    the whole `[L, E, ...]` store and a trailing int32[1] operand is the layer
    index.
    """
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.core.fused_moe_gmm import (
        resolve_onehot_permute_threshold,
    )

    use_sparse_core = bool(envs.USE_MOE_SPARSE_CORE)
    onehot = int(resolve_onehot_permute_threshold())
    key = (
        topk,
        routed_scale,
        renormalize,
        swiglu_limit,
        activation,
        use_ep,
        use_sparse_core,
        onehot,
        stacked,
    )
    op = _FUSED_MOE_OPS.get(key)
    if op is not None:
        return op
    from torch_tpu._internal import pallas

    name = (
        f"glm53::fused_moe_k{topk}_s{routed_scale:g}_r{int(renormalize)}"
        f"_l{swiglu_limit:g}_{activation}_ep{int(use_ep)}"
        f"_sc{int(use_sparse_core)}_oh{onehot}" + ("_stacked" if stacked else "")
    ).replace(".", "p")
    op = pallas.jax_op(
        name,
        functools.partial(
            _glm53_moe_stacked_jax if stacked else _glm53_moe_jax,
            topk=topk,
            routed_scale=routed_scale,
            renormalize=renormalize,
            swiglu_limit=swiglu_limit,
            activation=activation,
            use_ep=use_ep,
            use_sparse_core=use_sparse_core,
            onehot_threshold=onehot,
        ),
    )
    op.register_fake(_fake_fused_moe)
    _FUSED_MOE_OPS[key] = op
    return op


class Glm53FusedMoeArgs:
    """Resolved operands of one MoE layer for the fused op (built once).

    `unsupported` lists why the fused op cannot serve this layer (empty when
    it can); `op` is None in that case and the caller keeps the vLLM path.
    """

    def __init__(self, moe: Any) -> None:
        from vllm_torchtpu import envs
        from vllm_torchtpu.layers.adapter import moe_routing
        from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
        from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
            enable_pipelined_collective_and_compute,
        )
        from vllm_torchtpu.layers.adapter.quantization.fp8 import (
            VllmFp8LinearMethodTPU,
            VllmFp8MoEMethodTPU,
        )

        experts = moe.experts
        routed = getattr(experts, "routed_experts", experts)
        method = getattr(routed, "quant_method", None)
        shared = moe.shared_experts
        reasons = []
        if not isinstance(method, VllmFp8MoEMethodTPU):
            reasons.append(f"routed method {type(method).__name__}")
        elif getattr(method, "_use_adaptive_fused_moe", False):
            reasons.append("adaptive fused MoE")
        else:
            from vllm_torchtpu.layers.adapter.fused_moe_ep import (
                fused_moe_ep_supported,
            )

            if fused_moe_ep_supported(method):
                reasons.append("fused EP MoE")
        moe_cfg = getattr(routed, "moe_config", None)
        par_cfg = getattr(moe_cfg, "moe_parallel_config", None)
        use_ep = bool(getattr(par_cfg, "use_ep", False))
        if getattr(par_cfg, "enable_eplb", False):
            reasons.append("EPLB")
        experts_start = getattr(routed, "_experts_start", None)
        if use_ep and experts_start is None:
            reasons.append("EP missing _experts_start")
        if enable_pipelined_collective_and_compute():
            reasons.append("pipelined collective MoE")
        if envs.TPU_MOE_SKIP_PADDED_TOKENS:
            reasons.append("TPU_MOE_SKIP_PADDED_TOKENS")
        if moe_routing._SIMULATION_STRATEGY is not None:
            reasons.append("routing simulation")
        if moe_routing._ROUTER_TOPK != "rowmax":
            reasons.append(f"TPU_MOE_ROUTER_TOPK={moe_routing._ROUTER_TOPK}")
        if getattr(routed, "custom_routing_function", None) is not None:
            reasons.append("custom routing function")
        if (
            getattr(routed, "w13_bias", None) is not None
            or getattr(routed, "w2_bias", None) is not None
        ):
            reasons.append("expert biases")
        if shared is None:
            reasons.append("no shared expert")
        else:
            for lin in (shared.gate_up_proj, shared.down_proj):
                if not isinstance(lin.quant_method, VllmFp8LinearMethodTPU):
                    reasons.append(f"shared {type(lin.quant_method).__name__}")
                elif getattr(lin, KEEP_VLLM_LAYOUT_ATTR, False):
                    reasons.append("shared expert keeps vLLM layout")
                elif getattr(lin.quant_method, "w8a16_max_rows", 0):
                    # The fused shared expert always quantizes activations.
                    reasons.append("TPU_FP8_LINEAR_W8A16_MAX_ROWS")
        activation = getattr(method, "_tpu_activation_str", None)
        if activation is None and not reasons:
            reasons.append("activation unresolved")
        self.unsupported = "; ".join(reasons)
        self.op: Callable | None = None
        if self.unsupported:
            return

        self._op_kwargs = dict(
            topk=int(moe.top_k),
            routed_scale=float(moe.routed_scaling_factor),
            renormalize=bool(getattr(routed, "renormalize", True)),
            swiglu_limit=float(moe.swiglu_limit),
            activation=str(activation),
            use_ep=use_ep,
        )
        self.op = get_glm53_fused_moe_op(**self._op_kwargs)
        device = routed.w13_weight.device
        if experts_start is None:
            experts_start = torch.zeros((), dtype=torch.int32, device=device)
        self.tensors: tuple[Any, ...] = (
            moe.gate.weight,
            moe.gate.e_score_correction_bias,
            routed.w13_weight,
            routed.w2_weight,
            routed.w13_weight_scale_inv,
            routed.w2_weight_scale_inv,
            experts_start,
            shared.gate_up_proj.weight,
            shared.gate_up_proj.weight_scale,
            shared.down_proj.weight,
            shared.down_proj.weight_scale,
        )

    def use_expert_store(self, store: Any, layer_index: int) -> None:
        """Read this layer's routed experts from a stacked `[L, E, ...]` store.

        `store` stacks every MoE layer's routed experts so that one copy of
        the weights can be shared with another consumer: `w13` [L, E, H, 2I/W]
        fp8, `w13_s` [L, E, 1, 1, 2I/W] f32, `w2` [L, E, I/W, H] fp8 and `w2_s`
        [L, E, 1, 1, H] f32 (W is the weight packing factor). It holds the
        same tensors the per-layer parameters held (moved, not changed), so
        the outputs are unchanged.
        """
        assert self.op is not None, self.unsupported
        if self._op_kwargs["use_ep"]:
            raise ValueError(
                "The shared expert store needs TP experts (serve without "
                "--enable-expert-parallel)."
            )
        self.op = get_glm53_fused_moe_op(**self._op_kwargs, stacked=True)
        t = list(self.tensors)
        t[2], t[3], t[4], t[5] = store.w13, store.w2, store.w13_s, store.w2_s
        device = store.w13.device
        layer = torch.tensor([layer_index], dtype=torch.int32, device=device)
        self.tensors = (*t, layer)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert self.op is not None, self.unsupported
        return self.op(x, *self.tensors)
