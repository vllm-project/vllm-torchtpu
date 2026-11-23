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
"""Fused QKV prologue, FP8 output quantization, and shared K=V pages for Gemma 4."""

from __future__ import annotations

from typing import Any

import torch
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import is_forward_context_available
from vllm.model_executor.layers.attention.attention import get_attention_context

from vllm_torchtpu.layers.adapter.attention import PallasBatchedRPAAttentionBackendImpl
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.gemma4.layers import (
    apply_o_proj,
    model_supports_multimodal,
    rotary_cos_sin,
    runtime_fp8_weight_dtype,
    split_norm_rotary_qkv,
    supports_qkv_prologue,
)

logger = init_logger(__name__)


def _nonempty_mm_range(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, torch.Tensor):
        numel = val.shape[0] if val.dim() > 0 else val.numel()
        return not (isinstance(numel, int) and numel == 0)
    return not (isinstance(val, (list, tuple, dict)) and len(val) == 0)


def has_active_mm_prefix(
    attn: Any = None, attn_metadata: Any = None, kwargs: dict[str, Any] | None = None
) -> bool:
    if kwargs:
        return True
    if attn_metadata is None and attn is not None and is_forward_context_available():
        layer_name = getattr(getattr(attn, "attn", None), "layer_name", None)
        if layer_name is not None:
            attn_metadata = get_attention_context(layer_name)[0]
    return attn_metadata is not None and any(
        _nonempty_mm_range(getattr(attn_metadata, a, None))
        for a in ("mm_prefix_range", "mm_prefix_range_tensor")
    )


def should_enable_shared_kv_pages(
    attn: Any, impl: PallasBatchedRPAAttentionBackendImpl, vllm_config: Any
) -> bool:
    """Return True when ``attn`` can safely store compact shared K=V pages."""
    if vllm_config is None:
        return False
    if (
        model_supports_multimodal(vllm_config.model_config)
        or vllm_config.kv_transfer_config is not None
        or vllm_config.speculative_config is not None
        or impl.sliding_window is not None
        or not attn.config.attention_k_eq_v
        or getattr(attn.config, "num_kv_shared_layers", 0)
        or not attn.rotary_emb.is_neox_style
        or getattr(attn.rotary_emb, "rope_angles", None) is None
    ):
        return False
    return not impl.shared_kv_unsupported()


def _fused_attention_from_qkv(
    attn: Any, positions: torch.Tensor, qkv: torch.Tensor
) -> Any:
    meta, layer, kv_cache, _ = get_attention_context(attn.attn.layer_name)
    impl: Any = layer.impl
    if (
        has_active_mm_prefix(attn, meta)
        or impl.kv_sharing_target_layer_name is not None
        or impl._qkv_kernel is None
    ):
        return attn.attn(*split_norm_rotary_qkv(attn, positions, qkv))
    cos, sin = rotary_cos_sin(attn.rotary_emb, qkv, positions)
    wq, wk, eps = attn.q_norm.weight, attn.k_norm.weight, attn.q_norm.variance_epsilon
    return impl.forward_from_qkv(layer, qkv, cos, sin, wq, wk, eps, kv_cache, meta)


def attention_from_qkv(attn: Any, positions: torch.Tensor, qkv: torch.Tensor) -> Any:
    if getattr(attn, "_tpu_qkv_prologue_ready", False) and not has_active_mm_prefix(
        attn
    ):
        return _fused_attention_from_qkv(attn, positions, qkv)
    return attn.attn(*split_norm_rotary_qkv(attn, positions, qkv))


def o_projection(attn: Any, out: Any) -> torch.Tensor:
    return apply_o_proj(attn.o_proj, out)


def _configure_attention_impl(self: Any, vllm_config: Any) -> None:
    self._tpu_qkv_prologue_ready = False
    impl = self.attn.impl
    if (
        self.is_kv_shared_layer
        or vllm_config is None
        or model_supports_multimodal(vllm_config.model_config)
        or vllm_config.speculative_config is not None
        or not isinstance(impl, PallasBatchedRPAAttentionBackendImpl)
    ):
        return
    if should_enable_shared_kv_pages(self, impl, vllm_config):
        rotary_dims = self.rotary_emb.rope_angles
        impl.set_shared_kv(rotary_dims, self.k_norm)
        logger.info_once(
            "Gemma 4 shared K=V pages enabled (%d/%d rotated dims).",
            rotary_dims,
            self.head_dim,
        )
    if not supports_qkv_prologue(self, impl):
        return
    impl.qkv_prologue_eps = self.q_norm.variance_epsilon
    o_proj = self.o_proj
    impl.qkv_output_quant_dtype = (
        runtime_fp8_weight_dtype(o_proj) if o_proj.input_is_parallel else None
    )
    self._tpu_qkv_prologue_ready = True


def _patched_attention_init(self: Any, *args: Any, **kwargs: Any) -> None:
    type(self)._tpu_orig_init(self, *args, **kwargs)
    _configure_attention_impl(self, get_current_vllm_config_or_none())


def _patched_attention_forward(
    self: Any, positions: torch.Tensor, hidden_states: torch.Tensor, **kwargs: Any
) -> torch.Tensor:
    if (
        self.is_kv_shared_layer
        or not getattr(self, "_tpu_qkv_prologue_ready", False)
        or has_active_mm_prefix(self, kwargs=kwargs)
    ):
        return type(self)._tpu_orig_forward(self, positions, hidden_states, **kwargs)
    qkv, _ = self.qkv_proj(hidden_states)
    return o_projection(self, _fused_attention_from_qkv(self, positions, qkv))


def apply_gemma4_attention_patch(gemma4_mod: Any) -> bool:
    attn_cls = gemma4_mod.Gemma4Attention
    if getattr(attn_cls, "_TPU_GEMMA4_ATTN_PATCHED", False):
        return False
    attn_cls._tpu_orig_init = attn_cls.__init__
    attn_cls._tpu_orig_forward = attn_cls.forward
    attn_cls.__init__ = _patched_attention_init
    attn_cls.forward = _patched_attention_forward
    attn_cls._TPU_GEMMA4_ATTN_PATCHED = True
    logger.info("Applied TPU patch: Gemma 4 attention prologue/FP8/shared-KV.")
    return True
