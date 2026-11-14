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
"""Fused q/gate split, Q/K RMSNorm, RoPE and gate for Qwen3-Next on TPU.

TPU_FUSED_QK_NORM_ROPE=1 runs these as one Pallas call per attention layer
(kernels/qk_norm_rope_gate). The entry point follows the contract of
vllm.model_executor.layers.fused_qk_norm_rope.fused_qk_rmsnorm_rope_gate,
except that the norm weights are passed with the + 1 already applied.
"""

import functools

import jax
import torch

from vllm_torchtpu.kernels.qk_norm_rope_gate import qk_norm_rope_gate_rows

_ops_by_geometry: dict = {}


def _run_kernel(
    q_gate: jax.Array,
    k: jax.Array,
    q_weight: jax.Array,
    k_weight: jax.Array,
    cos_sin: jax.Array,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    # cos_sin holds the rope rows for these tokens, gathered by position in
    # torch before the call.
    return qk_norm_rope_gate_rows(
        q_gate,
        k,
        q_weight,
        k_weight,
        cos_sin,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
    )


def _get_op(num_heads, num_kv_heads, head_dim, rotary_dim, eps):
    """Returns the torch op for one geometry. prebuild() builds it at model
    construction; the traced forward only looks it up, because Dynamo cannot
    register an op while tracing."""
    key = (num_heads, num_kv_heads, head_dim, rotary_dim, float(eps))
    op = _ops_by_geometry.get(key)
    if op is not None:
        return op
    from torch_tpu._internal import pallas

    kernel_fn = functools.partial(
        _run_kernel,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        eps=float(eps),
    )
    op = pallas.jax_op(
        f"pallas::qk_norm_rope_gate_h{num_heads}_kv{num_kv_heads}"
        f"_d{head_dim}_r{rotary_dim}",
        kernel_fn,
    )

    def _fake(q_gate, k, q_weight, k_weight, cos_sin):
        rows = q_gate.shape[0]
        return (
            q_gate.new_empty(rows, num_heads * head_dim),
            k.new_empty(rows, num_kv_heads * head_dim),
            q_gate.new_empty(rows, num_heads * head_dim),
        )

    op.register_fake(_fake)
    _ops_by_geometry[key] = op
    return op


def prebuild(num_heads, num_kv_heads, head_dim, rotary_dim, eps):
    """Registers the op for this geometry outside any traced region."""
    _get_op(num_heads, num_kv_heads, head_dim, rotary_dim, eps)


def fused_qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight_plus_one: torch.Tensor,
    k_weight_plus_one: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns post-norm, post-RoPE q and k, and the pre-sigmoid gate.

    positions is [rows] or, for text-only M-RoPE, [3, rows] with equal rows.
    """
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(
            "rotary_dim must be a positive even integer <= head_dim, "
            f"got {rotary_dim} / {head_dim}"
        )
    op = _get_op(num_q_heads, num_kv_heads, head_dim, rotary_dim, eps)
    token_positions = positions[0] if positions.dim() == 2 else positions
    cos_sin = cos_sin_cache.index_select(0, token_positions.to(torch.int64))
    return op(q_gate, k, q_weight_plus_one, k_weight_plus_one, cos_sin[:, :rotary_dim])
