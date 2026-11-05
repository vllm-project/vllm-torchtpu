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
"""Route Gemma-4 vision attention Q/K norm + RoPE through a fused Pallas kernel.

HF ``Gemma4VisionAttention.forward`` runs ``q_norm`` and
``apply_multidimensional_rope`` as separate ops for both Q and K. This patch
replaces just those two steps with ``kernels.gemma4_vision.qk_norm_rope``; the
projections (including clipping), V path and attention are HF's own code.
"""

from __future__ import annotations

import functools
import math
import sys
from typing import Any

import torch

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_HF_MODULE = "transformers.models.gemma4.modeling_gemma4"
_ROPE_NDIM = 2
_ops: dict[str, Any] = {}
_orig_forward = None


def _qk_norm_rope_jax(x, cos, sin, scale, *, num_heads, head_dim, eps):
    """Module-level so ``pallas.jax_op`` can trace and register it."""
    from vllm_torchtpu.kernels.gemma4_vision import qk_norm_rope as lib

    return lib.qk_norm_rope(
        x,
        cos,
        sin,
        scale,
        num_heads=num_heads,
        head_dim=head_dim,
        ndim=_ROPE_NDIM,
        eps=eps,
    )


def _fake_qk_norm_rope(x, cos, sin, scale, *args, **kwargs):
    """Abstract implementation for Dynamo tracing: same shape/dtype as ``x``."""
    return torch.empty_like(x)


def _get_op(num_heads: int, head_dim: int, eps: float):
    name = f"pallas::gemma4_vision_qk_norm_rope_h{num_heads}_d{head_dim}_e{eps:.0e}"
    name = name.replace("-", "m").replace("+", "p").replace(".", "_")
    op = _ops.get(name)
    if op is None:
        from jax.sharding import PartitionSpec as P
        from torch_tpu._internal import pallas

        # mesh=None: the vision tower runs per rank on local images, so the
        # kernel runs on this rank's device.
        op = pallas.jax_op(
            name,
            functools.partial(
                _qk_norm_rope_jax, num_heads=num_heads, head_dim=head_dim, eps=eps
            ),
            mesh=None,
            input_partition_specs=(P(), P(), P(), P()),
        )
        op.register_fake(_fake_qk_norm_rope)
        _ops[name] = op
    return op


def _fused_eligible(attn, states: torch.Tensor, cos, position_ids) -> bool:
    head_dim = attn.head_dim
    if position_ids is None or position_ids.shape[-1] != _ROPE_NDIM:
        return False
    if cos.shape[-1] != head_dim or head_dim % (2 * _ROPE_NDIM) != 0:
        return False
    if states.shape[-1] % 128 != 0 or states.shape[-1] % head_dim != 0:
        return False
    return all(getattr(n, "with_scale", False) for n in (attn.q_norm, attn.k_norm))


def _norm_rope(attn, norm, states, cos, sin, tokens, hidden_shape):
    num_heads = states.shape[-1] // attn.head_dim
    op = _get_op(num_heads, attn.head_dim, float(norm.eps))
    out = op(
        states.reshape(tokens, -1),
        cos.reshape(tokens, attn.head_dim),
        sin.reshape(tokens, attn.head_dim),
        norm.weight,
    )
    return out.view(hidden_shape)


def _patched_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    **kwargs,
):
    hf = sys.modules[_HF_MODULE]
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    cos, sin = position_embeddings

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    if _fused_eligible(self, query_states, cos, position_ids) and (
        key_states.shape == query_states.shape
    ):
        tokens = math.prod(input_shape)
        query_states = _norm_rope(
            self, self.q_norm, query_states, cos, sin, tokens, hidden_shape
        )
        key_states = _norm_rope(
            self, self.k_norm, key_states, cos, sin, tokens, hidden_shape
        )
    else:
        query_states = self.q_norm(query_states.view(hidden_shape))
        query_states = hf.apply_multidimensional_rope(
            query_states, cos, sin, position_ids
        )
        key_states = self.k_norm(key_states.view(hidden_shape))
        key_states = hf.apply_multidimensional_rope(key_states, cos, sin, position_ids)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    value_states = self.v_proj(hidden_states).view(hidden_shape)
    value_states = self.v_norm(value_states)
    value_states = value_states.transpose(1, 2)

    attention_interface = hf.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, hf.eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.scaling,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


setattr(_patched_forward, "_tpu_qk_norm_rope_patched", True)  # noqa: B010


def patch_gemma4_vision_attention() -> None:
    """Install the fused Q/K norm + RoPE forward on ``Gemma4VisionAttention``."""
    global _orig_forward
    import vllm_torchtpu.envs as envs

    if not envs.TPU_GEMMA4_VISION_FUSED_QK_NORM_ROPE:
        return
    try:
        __import__(_HF_MODULE)
    except ImportError:
        return
    cls = getattr(sys.modules[_HF_MODULE], "Gemma4VisionAttention", None)
    if cls is None or getattr(cls.forward, "_tpu_qk_norm_rope_patched", False):
        return
    _orig_forward = cls.forward
    cls.forward = _patched_forward
    logger.info(
        "Applied TPU patch: fused Q/K RMSNorm + RoPE kernel for Gemma-4 vision."
    )
