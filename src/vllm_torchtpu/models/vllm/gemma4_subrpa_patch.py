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
"""Subspace-RPA for Gemma 4 global attention (``TPU_SUBSPACE_RPA=1``).

Gemma 4 global-attention layers with ``attention_k_eq_v`` use one projection
for K and V, ``v_norm`` has no weight, and RoPE rotates only 128 of the 512
head channels (``[0:64)`` and ``[256:320)``). On the other 384 channels K is
just ``k_norm.weight * V``.

After the weights are loaded, each eligible layer is rewritten once:

1. Head channels are permuted so the rotated ones sit at ``[0:128)``. The same
   permutation is applied to ``qkv_proj``, ``o_proj``, ``q_norm`` and
   ``k_norm``, so the layer output does not change.
2. ``k_norm.weight[128:]`` is folded into ``q_norm.weight[128:]``. The
   unrotated part of K then equals V, so the V columns of ``qkv_proj`` are
   dropped.
3. The RoPE table is cut down to the 128 rotated channels.

The forward pass hands the attention kernel a 128-dim rotated K and the full
V, and BatchedRPA stores only those two per token.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from vllm.model_executor.layers.attention import attention as vllm_attn_layer
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.models.gemma4 import Gemma4Attention
from vllm.model_executor.utils import set_weight_attrs

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.attention import (
    SUBSPACE_RPA_HEAD_DIM,
    SUBSPACE_RPA_ROT_DIM,
)
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_HALF_HEAD_DIM = SUBSPACE_RPA_HEAD_DIM // 2
_HALF_ROT_DIM = SUBSPACE_RPA_ROT_DIM // 2
_PATCHED_ATTN_CLS_ATTR = "_tpu_subspace_rpa_cls_patched"
_LOADER_HOOKED_ATTR = "_tpu_gemma4_subspace_rpa_hooked"


def _replace_param_or_buf(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    old = getattr(module, name, None)
    weight_attrs = getattr(old, "__dict__", {}) if old is not None else {}
    if isinstance(old, nn.Parameter):
        new_param = nn.Parameter(tensor, requires_grad=False)
        set_weight_attrs(new_param, weight_attrs)
        setattr(module, name, new_param)
    elif name in getattr(module, "_buffers", {}):
        module.register_buffer(name, tensor)
    else:
        setattr(module, name, tensor)


def build_gemma4_subspace_perm(device: torch.device | str = "cpu") -> torch.Tensor:
    """Return the head-channel permutation that moves RoPE channels to the front."""
    rot_end = _HALF_HEAD_DIM + _HALF_ROT_DIM
    bounds = (
        (0, _HALF_ROT_DIM),
        (_HALF_HEAD_DIM, rot_end),
        (_HALF_ROT_DIM, _HALF_HEAD_DIM),
        (rot_end, SUBSPACE_RPA_HEAD_DIM),
    )
    return torch.cat(
        [torch.arange(lo, hi, dtype=torch.long, device=device) for lo, hi in bounds]
    )


def is_subspace_eligible_gemma4_attn(attn: Any) -> bool:
    """Return True if ``attn`` is a Gemma 4 512-dim ``k_eq_v`` global layer."""
    if attn.is_sliding or attn.is_kv_shared_layer:
        return False
    if not getattr(attn.config, "attention_k_eq_v", False):
        return False
    if attn.head_dim != SUBSPACE_RPA_HEAD_DIM:
        return False
    # Only proportional RoPE defines these; other RoPE types are ineligible.
    rope = attn.rotary_emb
    if (
        getattr(rope, "rope_angles", None) != _HALF_ROT_DIM
        or getattr(rope, "nope_angles", None) != _HALF_HEAD_DIM - _HALF_ROT_DIM
    ):
        return False
    return attn.attn.impl.kv_cache_quantized_dtype is not None


def conjugate_gemma4_subspace_attn_module(attn: Any) -> bool:
    """Permute head channels, drop the V columns and shrink the RoPE table."""
    if getattr(attn, "_tpu_subrpa_active", False):
        return False
    if not is_subspace_eligible_gemma4_attn(attn):
        return False

    dev = attn.q_norm.weight.device
    perm = build_gemma4_subspace_perm(dev)
    qk_heads = attn.num_heads + attn.num_kv_heads
    qk_idx = (
        torch.arange(qk_heads, dtype=torch.long, device=dev).unsqueeze(1)
        * SUBSPACE_RPA_HEAD_DIM
        + perm.unsqueeze(0)
    ).reshape(-1)
    q_idx = qk_idx[: attn.q_size]
    orig_qkv_cols = attn.q_size + 2 * attn.kv_size
    new_qkv_cols = attn.q_size + attn.kv_size

    to_sync: list[torch.Tensor] = []

    # 1. qkv_proj: permute head channels and drop the V columns.
    qkv = attn.qkv_proj
    w = qkv.weight.data
    qkv_flipped = bool(getattr(qkv, WEIGHT_FLIPPED_ATTR, False))
    if qkv_flipped:
        if w.shape[1] != orig_qkv_cols:
            raise ValueError(
                f"Unexpected flipped qkv_proj.weight shape {tuple(w.shape)}, "
                f"expected (*, {orig_qkv_cols})"
            )
        new_w = fresh_buffer(w[:, qk_idx].contiguous())
    else:
        if w.shape[0] != orig_qkv_cols:
            raise ValueError(
                f"Unexpected unflipped qkv_proj.weight shape {tuple(w.shape)}, "
                f"expected ({orig_qkv_cols}, *)"
            )
        new_w = fresh_buffer(w[qk_idx, :].contiguous())
    _replace_param_or_buf(qkv, "weight", new_w)
    to_sync.append(qkv.weight)

    ws: Any = getattr(qkv, "weight_scale", None)
    if isinstance(ws, torch.Tensor) and ws.numel() != 1:
        if ws.ndim in (1, 2) and ws.shape[0] == orig_qkv_cols:
            permuted_ws = ws.data[qk_idx]
        elif ws.ndim == 4 and ws.shape[-1] == orig_qkv_cols:
            permuted_ws = ws.data[..., qk_idx]
        else:
            raise ValueError(
                f"Unexpected qkv_proj.weight_scale shape {tuple(ws.shape)} for "
                f"{orig_qkv_cols} output channels"
            )
        _replace_param_or_buf(
            qkv, "weight_scale", fresh_buffer(permuted_ws.contiguous())
        )
        to_sync.append(qkv.weight_scale)
    if hasattr(qkv, "output_size_per_partition"):
        qkv.output_size_per_partition = new_qkv_cols
    if (
        getattr(qkv, "weight_block_size", None) is not None
        and isinstance(ws, torch.Tensor)
        and ws.ndim == 1
        and getattr(qkv, "weight_scale", ws).shape[0] == new_qkv_cols
    ):
        in_dim = new_w.shape[0] if qkv_flipped else new_w.shape[1]
        qkv.weight_block_size = (1, in_dim)

    # 2. o_proj: apply the same permutation to its input channels.
    oproj = attn.o_proj
    ow = oproj.weight.data
    o_flipped = bool(getattr(oproj, WEIGHT_FLIPPED_ATTR, False))
    if o_flipped:
        new_ow = fresh_buffer(ow[q_idx, :].contiguous())
    else:
        new_ow = fresh_buffer(ow[:, q_idx].contiguous())
    _replace_param_or_buf(oproj, "weight", new_ow)
    to_sync.append(oproj.weight)

    # 3. q_norm and k_norm: permute, then fold the unrotated k_norm weights
    #    into q_norm. k_norm keeps only its rotated channels.
    wq_f32 = attn.q_norm.weight.data[perm].float()
    wk_f32 = attn.k_norm.weight.data[perm].float()
    wq_f32[SUBSPACE_RPA_ROT_DIM:] *= wk_f32[SUBSPACE_RPA_ROT_DIM:]
    new_wq = fresh_buffer(wq_f32.to(attn.q_norm.weight.dtype).contiguous())
    new_wk_rot = fresh_buffer(
        wk_f32[:SUBSPACE_RPA_ROT_DIM].to(attn.k_norm.weight.dtype).contiguous()
    )
    _replace_param_or_buf(attn.q_norm, "weight", new_wq)
    _replace_param_or_buf(attn.k_norm, "weight", new_wk_rot)
    to_sync.extend([attn.q_norm.weight, attn.k_norm.weight])

    # 4. Keep only the rotated columns of the RoPE table. The table is shared
    #    between layers, so do it once.
    rope = attn.rotary_emb
    if not getattr(rope, "_tpu_subrpa_compacted", False):
        for buf_name in ("cos_sin_cache", "cos_sin_cache_bf16"):
            buf: Any = getattr(rope, buf_name, None)
            if (
                isinstance(buf, torch.Tensor)
                and buf.ndim == 2
                and buf.shape[1] == SUBSPACE_RPA_HEAD_DIM
            ):
                rot_cols = (
                    buf[:, :_HALF_ROT_DIM],
                    buf[:, _HALF_HEAD_DIM : _HALF_HEAD_DIM + _HALF_ROT_DIM],
                )
                compacted = fresh_buffer(torch.cat(rot_cols, dim=-1).contiguous())
                _replace_param_or_buf(rope, buf_name, compacted)
                to_sync.append(getattr(rope, buf_name))
        rope.rotary_dim = SUBSPACE_RPA_ROT_DIM
        rope._tpu_subrpa_compacted = True

    if dev.type == "tpu" and to_sync:
        synchronize_tensors(to_sync)

    attn._tpu_subrpa_active = True
    return True


def conjugate_gemma4_subspace_model(model: nn.Module) -> int:
    """Apply ``conjugate_gemma4_subspace_attn_module`` across ``model``."""
    count = 0
    for mod in model.modules():
        if isinstance(mod, Gemma4Attention) and conjugate_gemma4_subspace_attn_module(
            mod
        ):
            count += 1
    if count:
        logger.info(
            "Rewrote %d Gemma 4 global attention layers for Subspace-RPA",
            count,
        )
    return count


def apply_subspace_rope_128(
    rotary_emb: Any,
    positions: torch.Tensor,
    query_3d: torch.Tensor,
    k_rot_3d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the first 128 channels of ``query_3d`` and all of ``k_rot_3d``."""
    cos_sin_cache = rotary_emb._match_cos_sin_cache_dtype(query_3d)
    pos_flat = positions.flatten()
    cos_sin = cos_sin_cache.index_select(0, pos_flat)
    cos, sin = cos_sin.chunk(2, dim=-1)

    q_rot = ApplyRotaryEmb.forward_static(
        query_3d[..., :SUBSPACE_RPA_ROT_DIM],
        cos,
        sin,
        True,
        enable_fp32_compute=True,
    )
    q_out = torch.cat((q_rot, query_3d[..., SUBSPACE_RPA_ROT_DIM:]), dim=-1)
    k_rot_out = ApplyRotaryEmb.forward_static(
        k_rot_3d,
        cos,
        sin,
        True,
        enable_fp32_compute=True,
    )
    return q_out, k_rot_out


def call_subspace_unified_attention(
    attn_layer: Any,
    query_3d: torch.Tensor,
    k_rot_3d: torch.Tensor,
    value_3d: torch.Tensor,
) -> torch.Tensor:
    """Invoke ``unified_attention_with_output`` with 128-dim ``k_rot``."""
    num_tokens = query_3d.shape[0]
    head_size_v = attn_layer.head_size_v
    hidden_out = attn_layer.num_heads * head_size_v
    output = torch.empty(
        (num_tokens, attn_layer.num_heads, head_size_v),
        dtype=query_3d.dtype,
        device=query_3d.device,
    )
    if attn_layer.use_direct_call:
        vllm_attn_layer.unified_attention_with_output(
            query_3d,
            k_rot_3d,
            value_3d,
            output,
            attn_layer.layer_name,
            kv_cache_dummy_dep=None,
        )
    else:
        encoded = vllm_attn_layer._encode_layer_name(attn_layer.layer_name)
        torch.ops.vllm.unified_attention_with_output(
            query_3d,
            k_rot_3d,
            value_3d,
            output,
            encoded,
            kv_cache_dummy_dep=None,
        )
    return output.view(-1, hidden_out)


def subspace_attention_partial(
    attn: Any,
    positions: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """``Gemma4Attention.forward`` for a layer rewritten for Subspace-RPA."""
    qk, _ = attn.qkv_proj(x)
    q, k_raw = qk.split([attn.q_size, attn.kv_size], dim=-1)

    q_3d = attn.q_norm(q.unflatten(-1, (attn.num_heads, attn.head_dim)))
    v_3d = attn.v_norm(k_raw.unflatten(-1, (attn.num_kv_heads, attn.head_dim)))
    k_rot_3d = (
        v_3d[..., :SUBSPACE_RPA_ROT_DIM].float() * attn.k_norm.weight.float()
    ).to(v_3d.dtype)

    q_3d, k_rot_3d = apply_subspace_rope_128(attn.rotary_emb, positions, q_3d, k_rot_3d)
    attn_output = call_subspace_unified_attention(attn.attn, q_3d, k_rot_3d, v_3d)
    output, _ = attn.o_proj(attn_output)
    return output


def _install_loader_post_hook() -> None:
    from vllm.model_executor.model_loader import base_loader

    orig_process = base_loader.process_weights_after_loading
    if getattr(orig_process, _LOADER_HOOKED_ATTR, False):
        return

    @functools.wraps(orig_process)
    def _wrapped_process_weights_after_loading(model, model_config, target_device):
        res = orig_process(model, model_config, target_device)
        if envs.TPU_SUBSPACE_RPA:
            conjugate_gemma4_subspace_model(model)
        return res

    setattr(_wrapped_process_weights_after_loading, _LOADER_HOOKED_ATTR, True)
    base_loader.process_weights_after_loading = _wrapped_process_weights_after_loading


def maybe_patch_gemma4_subspace_rpa(
    model_config: ModelConfig | None = None,
) -> bool | None:
    """Install the Subspace-RPA forward and load hook when ``TPU_SUBSPACE_RPA=1``."""
    if not envs.TPU_SUBSPACE_RPA:
        return None
    if model_config is None:
        return False
    text_config = model_config.hf_text_config
    if "gemma4" not in str(getattr(text_config, "model_type", "")).lower():
        return False
    if not getattr(text_config, "attention_k_eq_v", False):
        return False

    _install_loader_post_hook()

    attn_cls = Gemma4Attention
    if not getattr(attn_cls, _PATCHED_ATTN_CLS_ATTR, False):
        orig_forward = attn_cls.forward

        def _patched_gemma4_attn_forward(
            self: Any,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            **kwargs: Any,
        ) -> torch.Tensor:
            if getattr(self, "_tpu_subrpa_active", False):
                return subspace_attention_partial(self, positions, hidden_states)
            return orig_forward(self, positions, hidden_states, **kwargs)

        attn_cls.forward = _patched_gemma4_attn_forward
        setattr(attn_cls, _PATCHED_ATTN_CLS_ATTR, True)

    logger.info("Enabled Gemma 4 Subspace-RPA (TPU_SUBSPACE_RPA=1)")
    return True
