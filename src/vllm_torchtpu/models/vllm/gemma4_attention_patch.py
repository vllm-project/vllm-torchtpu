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
"""Gemma4 attention from its fused qkv rows (TPU_GEMMA4_FUSED_ATTN_PROLOGUE).

Between its qkv and o projections, upstream ``Gemma4Attention.forward`` runs
per-head RMSNorms of q, k and v, neox rotary on q and k, and the attention
layer, whose TPU backend then quantizes K/V to FP8 and lays q and K/V out for
the batched RPA kernel. XLA runs that as a dozen passes with f32
intermediates. With the flag, ``attention_from_qkv`` hands the projection's
rows to ``PallasBatchedRPAAttentionBackendImpl.forward_from_qkv``, which does
all of it in one Pallas pass (``kernels/experimental/batched_rpa/prologue``).
It is a model patch because upstream exposes nothing between the qkv
projection and ``self.attn``: a backend or custom-op hook only ever sees the
normed, rotated q/k/v, never the projection's rows.

With TPU_GEMMA4_QUANTIZED_ATTN_OUTPUT the backend also returns the output rows
quantized per token for the FP8 o_proj (``batched_rpa/output_quant``), and
``o_projection`` completes the projection on them.

With TPU_GEMMA4_SHARED_KV_PAGES, each ``attention_k_eq_v`` full-attention
layer stores shared-KV pages (``set_shared_kv``): its value is
``rmsnorm(k_raw)`` and its key ``rope(k_norm.weight * rmsnorm(k_raw))``, so the
key equals the weight times the value outside the dims the proportional rope
rotates, and only those rotated dims are stored besides the value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.models.gemma4 import Gemma4Attention

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig

logger = init_logger(__name__)

_orig_attention_forward = Gemma4Attention.forward
_orig_attention_init = Gemma4Attention.__init__


def _upstream_attention(
    attn: Any, positions: torch.Tensor, qkv: torch.Tensor
) -> torch.Tensor:
    q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    q = attn.q_norm(q.unflatten(-1, (attn.num_heads, attn.head_dim))).flatten(-2, -1)
    k = attn.k_norm(k.unflatten(-1, (attn.num_kv_heads, attn.head_dim)))
    q, k = attn.rotary_emb(positions, q, k.flatten(-2, -1))
    v = attn.v_norm(v.unflatten(-1, (attn.num_kv_heads, attn.head_dim)))
    return attn.attn(q, k, v.flatten(-2, -1))


def _fused_attention(
    attn: Any, positions: torch.Tensor, qkv: torch.Tensor
) -> torch.Tensor:
    rope = attn.rotary_emb
    if not rope.is_neox_style:
        raise NotImplementedError("TPU_GEMMA4_FUSED_ATTN_PROLOGUE needs neox rotary")
    # The rotary cache rows of the tokens, in the activation dtype as the
    # upstream rotary reads them.
    cos_sin = rope._match_cos_sin_cache_dtype(qkv).index_select(0, positions.flatten())
    # Contiguous: the op's operands are dense buffers, not strided views.
    cos, sin = (half.contiguous() for half in cos_sin.chunk(2, dim=-1))
    attn_metadata, layer, kv_cache, _ = get_attention_context(attn.attn.layer_name)
    return layer.impl.forward_from_qkv(
        layer,
        qkv,
        cos,
        sin,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.q_norm.variance_epsilon,
        kv_cache,
        attn_metadata,
    )


def attention_from_qkv(attn: Any, positions: torch.Tensor, qkv: torch.Tensor) -> Any:
    """``Gemma4Attention.forward`` between the qkv and the o projection: the
    output rows, or under TPU_GEMMA4_QUANTIZED_ATTN_OUTPUT ``(rows quantized per
    token, their scales)``, for ``o_projection``."""
    if envs.TPU_GEMMA4_FUSED_ATTN_PROLOGUE:
        return _fused_attention(attn, positions, qkv)
    return _upstream_attention(attn, positions, qkv)


def o_projection(attn: Any, out: Any) -> torch.Tensor:
    """``attn.o_proj`` on ``attention_from_qkv``'s output, reduced as the
    projection's ``reduce_results`` says."""
    o_proj = attn.o_proj
    if not isinstance(out, tuple):
        return o_proj(out)[0]
    x_q, x_scale = out
    method = o_proj.quant_method
    if x_q.dtype != o_proj.weight.dtype or not method.supports_quantized_input(
        o_proj, x_q.shape[0]
    ):
        raise NotImplementedError(
            "TPU_GEMMA4_QUANTIZED_ATTN_OUTPUT needs a per-channel FP8 o_proj that "
            f"takes {x_q.dtype} rows quantized per token"
        )
    y = method.apply_quantized_input(o_proj, x_q, x_scale)
    if o_proj.reduce_results and o_proj.tp_size > 1:
        y = tensor_model_parallel_all_reduce(y)
    return y


def _patched_attention_forward(
    self: Gemma4Attention,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    if self.is_kv_shared_layer:  # only q is projected; K/V come from the cache
        return _orig_attention_forward(self, positions, hidden_states, **kwargs)
    qkv, _ = self.qkv_proj(hidden_states)
    return o_projection(self, _fused_attention(self, positions, qkv))


def _patched_attention_init(self: Gemma4Attention, *args: Any, **kwargs: Any) -> None:
    """Upstream's init, then a refusal of what forward_from_qkv cannot run,
    so it surfaces at model build rather than at the first trace."""
    from vllm_torchtpu.layers.adapter.attention import (
        PallasBatchedRPAAttentionBackendImpl,
    )

    _orig_attention_init(self, *args, **kwargs)
    if self.is_kv_shared_layer:
        return
    impl = self.attn.impl
    if not isinstance(impl, PallasBatchedRPAAttentionBackendImpl):
        raise NotImplementedError(
            "TPU_GEMMA4_FUSED_ATTN_PROLOGUE needs the batched RPA backend "
            f"(--attention-backend CUSTOM); {self.attn.layer_name} uses "
            f"{type(impl).__name__}"
        )
    if envs.TPU_GEMMA4_SHARED_KV_PAGES:
        _set_shared_kv(self, impl)
    unsupported = impl.qkv_prologue_unsupported()
    if not self.rotary_emb.is_neox_style:
        unsupported.append("non-neox rotary")
    if unsupported:
        raise NotImplementedError(
            f"TPU_GEMMA4_FUSED_ATTN_PROLOGUE does not support {', '.join(unsupported)}"
            f" ({self.attn.layer_name}); unset it for this configuration"
        )
    # The backend builds forward_from_qkv's op in initialize_kernel, before
    # the forward is traced.
    impl.qkv_prologue_eps = self.q_norm.variance_epsilon
    if envs.TPU_GEMMA4_QUANTIZED_ATTN_OUTPUT:
        impl.qkv_output_quant_dtype = _o_proj_quant_dtype(self.o_proj)


def _shared_kv_attention_init(self: Gemma4Attention, *args: Any, **kwargs: Any) -> None:
    """Upstream's init, then shared-KV pages for an attention_k_eq_v layer
    (TPU_GEMMA4_SHARED_KV_PAGES without the fused prologue)."""
    from vllm_torchtpu.layers.adapter.attention import (
        PallasBatchedRPAAttentionBackendImpl,
    )

    _orig_attention_init(self, *args, **kwargs)
    if self.is_kv_shared_layer:
        return
    impl = self.attn.impl
    if not isinstance(impl, PallasBatchedRPAAttentionBackendImpl):
        raise NotImplementedError(
            "TPU_GEMMA4_SHARED_KV_PAGES needs the batched RPA backend "
            f"(--attention-backend CUSTOM); {self.attn.layer_name} uses "
            f"{type(impl).__name__}"
        )
    _set_shared_kv(self, impl)


def _set_shared_kv(attn: Gemma4Attention, impl: Any) -> None:
    """Shared-KV pages for an ``attention_k_eq_v`` full-attention layer."""
    if impl.sliding_window is not None or not getattr(
        attn.config, "attention_k_eq_v", False
    ):
        return  # its K and V are separate projections
    rope = attn.rotary_emb
    rotary_dims = getattr(rope, "rope_angles", None)
    if rotary_dims is None or not rope.is_neox_style:
        raise NotImplementedError(
            "TPU_GEMMA4_SHARED_KV_PAGES needs Gemma4's proportional neox rotary "
            f"(leading dims of each half rotated); {attn.attn.layer_name} has "
            f"{type(rope).__name__}"
        )
    impl.set_shared_kv(rotary_dims, attn.k_norm)
    logger.info_once(
        "Gemma4 k_eq_v layers store shared-KV pages: the value and the key's %d "
        "rotated dims of each half of head_dim %d.",
        rotary_dims,
        attn.head_dim,
    )


def _o_proj_quant_dtype(o_proj: Any) -> torch.dtype:
    """The FP8 dtype of ``o_proj``'s runtime weight, which its rows are
    quantized to per token; a projection that quantizes otherwise is refused."""
    from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8LinearMethodTPU

    method = o_proj.quant_method
    dtype = block_size = None
    if isinstance(method, VllmFp8LinearMethodTPU) and not method.online_fp8:
        # The weight process_weights_after_loading makes, not the checkpoint's.
        _, dtype, block_size, _ = method._linear_quant_config
    if (
        dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
        or block_size is not None
        or method.block_quant
        or method.w8a16_max_rows
        or o_proj.bias is not None
        or not o_proj.input_is_parallel
    ):
        raise NotImplementedError(
            "TPU_GEMMA4_QUANTIZED_ATTN_OUTPUT needs a per-channel W8A8 FP8 o_proj "
            "(not online FP8, no block scales, bias or "
            f"TPU_FP8_LINEAR_W8A16_MAX_ROWS); unset it for {o_proj.prefix}"
        )
    return dtype


def maybe_patch_gemma4_attention(model_config: ModelConfig | None) -> bool | None:
    """Returns False until a Gemma4 text model config is available."""
    if (
        envs.TPU_GEMMA4_QUANTIZED_ATTN_OUTPUT
        and not envs.TPU_GEMMA4_FUSED_ATTN_PROLOGUE
    ):
        raise ValueError(
            "TPU_GEMMA4_QUANTIZED_ATTN_OUTPUT needs TPU_GEMMA4_FUSED_ATTN_PROLOGUE"
        )
    if not (envs.TPU_GEMMA4_FUSED_ATTN_PROLOGUE or envs.TPU_GEMMA4_SHARED_KV_PAGES):
        return None
    if model_config is None or model_config.hf_text_config.model_type != "gemma4_text":
        return False
    text_config = model_config.hf_text_config
    if envs.TPU_GEMMA4_SHARED_KV_PAGES and (
        not getattr(text_config, "attention_k_eq_v", False)
        or getattr(text_config, "num_kv_shared_layers", 0)
    ):
        # A KV-shared layer would read its target's shared-KV pages as K/V.
        raise ValueError(
            "TPU_GEMMA4_SHARED_KV_PAGES needs a Gemma4 model with attention_k_eq_v "
            "and no KV-shared layers"
        )
    if not envs.TPU_GEMMA4_FUSED_ATTN_PROLOGUE:
        Gemma4Attention.__init__ = _shared_kv_attention_init
        return None
    Gemma4Attention.__init__ = _patched_attention_init
    Gemma4Attention.forward = _patched_attention_forward
    logger.info(
        "Applied TPU patch: Gemma4 attention takes its qkv rows straight to the "
        "batched RPA kernel (q/k/v norms, rotary and FP8 KV in one Pallas pass)."
    )
    return None
