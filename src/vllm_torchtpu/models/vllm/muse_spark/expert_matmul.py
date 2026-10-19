# SPDX-License-Identifier: Apache-2.0
"""Grouped Muse expert matmuls with tiled BF16/NVFP4 weights."""

from functools import cache, partial

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2
from vllm_torchtpu.layers.core.fused_moe_gmm import (
    prepare_routed_gmm_inputs,
    resolve_onehot_permute_threshold,
)


def _grouped_experts(
    x: jax.Array,
    ids: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    w13_scale: jax.Array | None,
    w2_scale: jax.Array | None,
    w13_global: jax.Array | None,
    w2_global: jax.Array | None,
    experts_start: jax.Array | None,
    *,
    num_experts: int | None,
) -> jax.Array:
    tokens, top_k = ids.shape
    if num_experts is not None:
        assert experts_start is not None
        ids = ids - experts_start
        ids = jnp.where((ids >= 0) & (ids < w13.shape[0]), ids, -1)
    inputs, sizes, inverse, _, valid, _ = prepare_routed_gmm_inputs(
        x,
        ids,
        jnp.ones(ids.shape, x.dtype),
        local_num_experts=w13.shape[0],
        topk=top_k,
        use_ep=num_experts is not None,
        use_sparse_core=envs.USE_MOE_SPARSE_CORE,
        onehot_moe_permute_threshold=resolve_onehot_permute_threshold(),
    )

    def matmul(lhs, weight, scale, global_scale):
        quant_dtype = None
        if scale is not None:
            # Gate and up retain separate per-expert global scales.
            global_scale = jnp.repeat(
                global_scale.reshape(weight.shape[0], -1),
                weight.shape[-1]
                // (global_scale.shape[-1] if global_scale.ndim == 2 else 1),
                axis=-1,
            )[:, None, :]
            quant_dtype = jnp.float4_e2m1fn
        return gmm_v2(
            lhs=lhs,
            rhs=weight,
            rhs_scale=scale,
            rhs_global_scale=global_scale,
            group_sizes=sizes,
            group_offset=None,
            zero_initialize=False,
            rhs_quant_dtype=quant_dtype,
            rhs_dequantize_in_fp32=True,
            maybe_quantize_lhs=False,
            preferred_element_type=x.dtype,
            acc_dtype=jnp.float32,
        )

    # Preserve BF16 rounding at both GEMM boundaries around FP32 SiLU/multiply.
    gate, up = jnp.split(matmul(inputs, w13, w13_scale, w13_global), 2, axis=-1)
    activated = (jax.nn.silu(gate.astype(jnp.float32)) * up.astype(jnp.float32)).astype(
        x.dtype
    )
    rows = matmul(activated, w2, w2_scale, w2_global)
    return jnp.where(valid[:, None], rows[inverse], 0).reshape(
        tokens, top_k, x.shape[-1]
    )


@cache
def get_grouped_experts_op(num_experts: int | None = None):
    # Build once outside torch.compile; tensor EP offsets keep rank graphs identical.
    suffix = "" if num_experts is None else f"_e{num_experts}"
    op = pallas.jax_op(
        "pallas::muse_grouped_experts" + suffix,
        partial(_grouped_experts, num_experts=num_experts),
    )

    def fake(
        x, ids, w13, w2, w13_scale, w2_scale, w13_global, w2_global, experts_start
    ):
        return torch.empty(*ids.shape, x.shape[-1], device=x.device, dtype=x.dtype)

    op.register_fake(fake)

    def grouped_experts(
        x,
        ids,
        w13,
        w2,
        w13_scale,
        w2_scale,
        w13_global=None,
        w2_global=None,
        experts_start=None,
    ):
        # Match the int32 JAX indexing schema at the custom-op boundary.
        return op(
            x,
            ids.to(torch.int32),
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_global,
            w2_global,
            experts_start,
        )

    return grouped_experts
