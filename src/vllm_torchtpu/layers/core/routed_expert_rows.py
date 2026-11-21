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
"""Routed expert GMMs that return one row per token and selected expert.

Unlike ``fused_moe_gmm``, the rows are returned before the weighted sum, as
``[tokens, top_k, hidden]``, so a model can normalize each expert's output
before combining them. Weights are BF16, or packed NVFP4 with compact uint8
E4M3 block scales and FP32 per-expert global scales laid out as
``[experts, 1, out_features]``.
"""

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.megablox.gmm_v2 import gmm_v2
from vllm_torchtpu.layers.core.fused_moe_gmm import prepare_routed_gmm_inputs


def routed_expert_rows(
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
    use_ep: bool,
    use_sparse_core: bool,
    onehot_moe_permute_threshold: int,
) -> jax.Array:
    tokens, top_k = ids.shape
    if use_ep:
        assert experts_start is not None
        ids = ids - experts_start
        ids = jnp.where((ids >= 0) & (ids < w13.shape[0]), ids, -1)
        if use_sparse_core and tokens * top_k <= onehot_moe_permute_threshold:
            # The one-hot permute is a matmul across all tokens, and
            # 0 * NaN == NaN: zero non-finite values, as fused_moe_func does,
            # so the damage stays with that token.
            x = jnp.where(jnp.isfinite(x), x, 0)
    inputs, sizes, inverse, _, valid, _ = prepare_routed_gmm_inputs(
        x,
        ids,
        jnp.ones(ids.shape, x.dtype),
        local_num_experts=w13.shape[0],
        topk=top_k,
        use_ep=use_ep,
        use_sparse_core=use_sparse_core,
        onehot_moe_permute_threshold=onehot_moe_permute_threshold,
    )

    def matmul(lhs, weight, scale, global_scale):
        # Packed experts carry global scales already expanded to [E, 1, N].
        quant_dtype = None if scale is None else jnp.float4_e2m1fn
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
