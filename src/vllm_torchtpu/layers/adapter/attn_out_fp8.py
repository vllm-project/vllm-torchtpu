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
"""Post-attention-init tagging for fused attention-output FP8 quantization.

Reads ``--additional-config '{"tpu_fused_attn_out_fp8": true}'`` (off by
default) and tags eligible per-channel FP8 ``o_proj`` layers on allowlisted
attention classes whose ``forward`` feeds ``self.attn(...)`` directly to
``self.o_proj(...)`` without an intervening token-major op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import RowParallelLinear

from vllm_torchtpu.kernels.quantized_matmul.attn_out_fp8 import (
    MIN_ROW_TILE,
    select_row_tile,
)
from vllm_torchtpu.layers.adapter.attention import (
    PallasBatchedRPAAttentionBackendImpl,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    ATTN_OUT_HEAD_LAYOUT_ATTR,
    KEEP_VLLM_LAYOUT_ATTR,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# `--additional-config` switch gating the fused attention-output FP8 path.
FUSED_ATTN_OUT_FP8_KEY = "tpu_fused_attn_out_fp8"

# Exact `<module>.<qualname>` allowlist of attention modules whose `forward`
# passes `self.attn(...)` directly to `self.o_proj(...)` with no intervening
# token-major op (such as a sigmoid output gate or head-dim slice). Modules
# outside this set (e.g., `AfmoeAttention`, `Qwen3NextAttention`,
# `DeepseekV2Attention`) stay untagged so the fused kernel never runs on an
# input whose relayout cannot cancel.
_DIRECT_ATTN_OUT_CLASSES: frozenset[str] = frozenset(
    {
        "vllm.model_executor.models.gemma.GemmaAttention",
        "vllm.model_executor.models.gemma2.Gemma2Attention",
        "vllm.model_executor.models.gemma3.Gemma3Attention",
        "vllm.model_executor.models.gemma4.Gemma4Attention",
        "vllm.model_executor.models.glm4.Glm4Attention",
        "vllm.model_executor.models.glm4_moe.Glm4MoeAttention",
        "vllm.model_executor.models.gpt_oss.OAIAttention",
        "vllm.model_executor.models.llama.LlamaAttention",
        "vllm.model_executor.models.llama4.Llama4Attention",
        "vllm.model_executor.models.mistral.MistralAttention",
        "vllm.model_executor.models.mixtral.MixtralAttention",
        "vllm.model_executor.models.qwen2.Qwen2Attention",
        "vllm.model_executor.models.qwen3.Qwen3Attention",
        "vllm.model_executor.models.qwen3_moe.Qwen3MoeAttention",
        "vllm_torchtpu.models.vllm.qwen2.TPUTPPaddedQwen2Attention",
    }
)


def fused_attn_out_fp8_enabled(vllm_config: VllmConfig | None) -> bool:
    """Parse ``additional_config['tpu_fused_attn_out_fp8']``; non-bool raises."""
    if vllm_config is None:
        return False
    additional_config = vllm_config.additional_config
    if not isinstance(additional_config, dict):
        return False
    value = additional_config.get(FUSED_ATTN_OUT_FP8_KEY, False)
    if not isinstance(value, bool):
        raise ValueError(
            f"additional_config[{FUSED_ATTN_OUT_FP8_KEY!r}] must be a bool, "
            f"got {value!r}"
        )
    return value


def _torch_to_jax_dtype(dtype: torch.dtype) -> jnp.dtype | None:
    return pallas.pallas.TORCH_TO_JAX_DTYPE_MAP.get(dtype)


def _attn_out_head_layout(
    module: torch.nn.Module,
    attn: Attention,
    o_proj: RowParallelLinear,
    *,
    act_dtype: jnp.dtype,
) -> tuple[int, int, int] | None:
    """Per-rank ``(num_kv_heads, q_per_kv, head_dim)`` if eligible, else None."""
    cls = type(module)
    if f"{cls.__module__}.{cls.__qualname__}" not in _DIRECT_ATTN_OUT_CLASSES:
        return None
    if not (
        isinstance(attn.impl, PallasBatchedRPAAttentionBackendImpl)
        and attn.impl.emits_head_major_batched_rpa_output()
    ):
        return None
    if not isinstance(o_proj.quant_method, VllmFp8LinearMethodTPU):
        return None
    if getattr(o_proj, KEEP_VLLM_LAYOUT_ATTR, False):
        return None
    weight = o_proj.weight
    assert isinstance(weight, torch.Tensor)
    if (
        not weight.dtype.is_floating_point
        or weight.dtype.itemsize != 1
        or o_proj.weight_scale.ndim != 1
    ):
        return None
    quant_dtype = _torch_to_jax_dtype(weight.dtype)
    if quant_dtype is None:
        return None
    q_per_kv, remainder = divmod(attn.num_heads, attn.num_kv_heads)
    if remainder or o_proj.input_size_per_partition != (
        attn.num_heads * attn.head_size
    ):
        return None
    if (
        select_row_tile(
            MIN_ROW_TILE,
            attn.num_kv_heads,
            q_per_kv,
            attn.head_size,
            act_dtype=act_dtype,
            quant_dtype=quant_dtype,
        )
        is None
    ):
        return None
    return attn.num_kv_heads, q_per_kv, attn.head_size


def tag_attn_out_projections(
    *models: torch.nn.Module,
    enabled: bool,
    act_dtype: jnp.dtype = jnp.bfloat16,
) -> int:
    """Record the attention head layout on eligible FP8 ``o_proj`` layers.

    Inspects allowlisted attention modules owning both ``attn`` (vLLM
    ``Attention``) and ``o_proj`` (``RowParallelLinear``) after weight loading
    and ``_initialize_attention_kernels`` have finalized weight layouts and
    attention backend modes. Returns ``0`` immediately when ``enabled`` is
    False.
    """
    if not enabled or not models:
        return 0
    tagged = 0
    seen: set[int] = set()
    for root in models:
        for module in root.modules():
            attn = getattr(module, "attn", None)
            o_proj = getattr(module, "o_proj", None)
            if not isinstance(attn, Attention) or not isinstance(
                o_proj, RowParallelLinear
            ):
                continue
            if id(o_proj) in seen:
                continue
            seen.add(id(o_proj))
            if hasattr(o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR):
                delattr(o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR)
            head_layout = _attn_out_head_layout(
                module, attn, o_proj, act_dtype=act_dtype
            )
            if head_layout is not None:
                setattr(o_proj, ATTN_OUT_HEAD_LAYOUT_ATTR, head_layout)
                tagged += 1
    return tagged


def arm_runner_models(model_runner: Any) -> int:
    """Tag eligible ``o_proj`` layers on the target model and separate drafter.

    Called at the end of ``TPUModelRunner._initialize_attention_kernels``, once
    weight layouts, PCP/DCP modes, and ``tp=1`` draft rebinding are settled.
    """
    vllm_config = getattr(model_runner, "vllm_config", None)
    enabled = fused_attn_out_fp8_enabled(vllm_config)
    if not enabled:
        return 0

    model_dtype = getattr(getattr(vllm_config, "model_config", None), "dtype", None)
    act_dtype = (
        _torch_to_jax_dtype(model_dtype)
        if isinstance(model_dtype, torch.dtype)
        else None
    )
    if act_dtype is None:
        act_dtype = jnp.bfloat16

    roots: list[torch.nn.Module] = []
    model = getattr(model_runner, "model", None)
    if isinstance(model, torch.nn.Module):
        roots.append(model)
    # `drafter` only (not `_async_drafter`, whose property asserts).
    draft = getattr(getattr(model_runner, "drafter", None), "draft_model", None)
    if isinstance(draft, torch.nn.Module) and draft is not model:
        roots.append(draft)

    tagged = tag_attn_out_projections(*roots, enabled=enabled, act_dtype=act_dtype)
    if tagged:
        logger.info(
            "Tagged %d attention o_proj layers for the fused "
            "attention-output FP8 quantization.",
            tagged,
        )
    return tagged
