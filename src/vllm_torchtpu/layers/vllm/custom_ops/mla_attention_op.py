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
"""Out-of-tree (OOT) custom operator layers and wrappers for Multi-Head Latent Attention (MLA)."""

from typing import Any

import torch
from torch.nn import Parameter
from torch_tpu._internal import sync
from vllm.config import CacheConfig
from vllm.model_executor.layers.attention import mla_attention
from vllm.model_executor.layers.attention.attention import \
    get_attention_context
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.mla import (MLAModules,
                                            MultiHeadLatentAttentionWrapper)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import (rotate_gptj,
                                                                rotate_neox)
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import \
    DeepseekScalingRotaryEmbedding
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.mla.prefill import selector

from vllm_torchtpu.layers.vllm.attention import TPU_STR_DTYPE_TO_TORCH_DTYPE


class TPUDummyMLAPrefillBackend:

    def __init__(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass


class VllmTPUMLAAttention(MLAAttention):
    """TPU-optimized out-of-tree wrapper for Multi-Head Latent Attention."""

    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_backend: Any | None = None,
        use_sparse: bool = False,
        indexer: object | None = None,
        **extra_impl_args,
    ):
        torch.nn.Module.__init__(self)

        original_mla_get_backend = getattr(mla_attention,
                                           "get_mla_prefill_backend", None)
        original_selector_get_backend = getattr(selector,
                                                "get_mla_prefill_backend",
                                                None)

        if original_mla_get_backend is not None:
            mla_attention.get_mla_prefill_backend = lambda config: TPUDummyMLAPrefillBackend
        if original_selector_get_backend is not None:
            selector.get_mla_prefill_backend = lambda config: TPUDummyMLAPrefillBackend

        try:
            # Keyword-only: vLLM inserts new positional params into
            # MLAAttention.__init__ (e.g. dcp_q_replicate in v0.26.1rc0),
            # which silently shifts positional args into the wrong slots.
            super().__init__(num_heads=num_heads,
                             scale=scale,
                             qk_nope_head_dim=qk_nope_head_dim,
                             qk_rope_head_dim=qk_rope_head_dim,
                             v_head_dim=v_head_dim,
                             q_lora_rank=q_lora_rank,
                             kv_lora_rank=kv_lora_rank,
                             kv_b_proj=kv_b_proj,
                             cache_config=cache_config,
                             quant_config=quant_config,
                             prefix=prefix,
                             attn_backend=attn_backend,
                             use_sparse=use_sparse,
                             indexer=indexer,
                             **extra_impl_args)
        finally:
            if original_mla_get_backend is not None:
                mla_attention.get_mla_prefill_backend = original_mla_get_backend
            if original_selector_get_backend is not None:
                selector.get_mla_prefill_backend = original_selector_get_backend

        # For compatibility reasons.
        self.kv_sharing_target_layer_name = None
        self.attn_type = AttentionType.DECODER
        self.sliding_window = None

        self.kv_cache_quantized_dtype = None
        if self.kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE.get(
                self.kv_cache_dtype.lower().strip())

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)

        device = torch.device("tpu")

        if self.kv_cache_quantized_dtype is not None:
            from vllm_torchtpu.layers.common.quantization import \
                quantize_tensor
            W_UK_T, W_UK_T_scale = quantize_tensor(
                self.W_UK_T, self.kv_cache_quantized_dtype, axis=1)
            self.W_UK_T = Parameter(W_UK_T.to(device), requires_grad=False)
            self.W_UK_T_scale = Parameter(W_UK_T_scale.to(device),
                                          requires_grad=False)

            W_UV, W_UV_scale = quantize_tensor(self.W_UV,
                                               self.kv_cache_quantized_dtype,
                                               axis=1)
            self.W_UV = Parameter(W_UV.to(device), requires_grad=False)
            self.W_UV_scale = Parameter(W_UV_scale.to(device),
                                        requires_grad=False)
        else:
            self.W_UK_T = Parameter(self.W_UK_T.to(device),
                                    requires_grad=False)
            self.W_UV = Parameter(self.W_UV.to(device), requires_grad=False)

        # Safely detach and clear kv_b_proj parameter buffers without breaking PyTorch attribute integrity
        kv_b_proj_params = dict(self.kv_b_proj.named_parameters())
        for key in kv_b_proj_params.keys():
            if key in self.kv_b_proj._parameters:
                self.kv_b_proj._parameters[key] = None
            elif hasattr(self.kv_b_proj, key):
                delattr(self.kv_b_proj, key)

        if self.W_UK_T.device.type == "tpu":
            sync.synchronize(self.W_UK_T, wait=True)
            if hasattr(self, "W_UK_T_scale"):
                sync.synchronize(self.W_UK_T_scale, wait=True)
            sync.synchronize(self.W_UV, wait=True)
            if hasattr(self, "W_UV_scale"):
                sync.synchronize(self.W_UV_scale, wait=True)

        q_scale, k_scale, v_scale = self.impl._get_kv_scales(self)
        self.mla_op = self.impl._build_mla_op(self,
                                              q_scale=q_scale,
                                              k_scale=k_scale,
                                              v_scale=v_scale)

    def forward(self,
                q: tuple[torch.Tensor, torch.Tensor],
                kv_c_normed: torch.Tensor,
                k_pe: torch.Tensor,
                output: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        if getattr(self, "calculate_kv_scales", False):
            torch.ops.vllm.maybe_calc_kv_scales(q, kv_c_normed, k_pe,
                                                self.layer_name)

        attn_metadata, _, kv_cache, _ = get_attention_context(self.layer_name)

        return self.impl.forward(
            layer=self,
            q=q,
            kv_c_normed=kv_c_normed,
            k_pe=k_pe,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
        )


# Backward compatibility alias right in case legacy external imports reference old name
VllmMLAAttention = VllmTPUMLAAttention


@MultiHeadLatentAttentionWrapper.register_oot
class VllmTPUMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_topk: bool = False,
    ) -> None:
        torch.nn.Module.__init__(self)

        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.g_proj = getattr(mla_modules, "g_proj", None)
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.skip_topk = skip_topk

        if self.indexer is not None and not self.skip_topk:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = VllmTPUMLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
        )

        self.prefix = prefix

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if hasattr(super(), "process_weights_after_loading"):
            super().process_weights_after_loading(act_dtype)
        if self.rotary_emb is not None and hasattr(self.rotary_emb,
                                                   "cos_sin_cache"):
            device = torch.device("tpu")
            self.rotary_emb.cos_sin_cache = self.rotary_emb.cos_sin_cache.to(
                device)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None")
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None")
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None")

            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            q_c = self.q_a_layernorm(q_c)
            q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None")
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None")
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim],
                                   dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim],
                               dim=-1)

        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        if self.indexer and self.is_sparse:
            _topk_indices = self.indexer(hidden_states, q_c, positions,
                                         self.indexer_rope_emb)

        if llama_4_scaling is not None:
            q_nope *= llama_4_scaling
            q_pe *= llama_4_scaling

        attn_out = self.mla_attn(
            (q_nope, q_pe),
            kv_c_normed,
            k_pe,
            output_shape=(hidden_states.shape[0],
                          self.num_heads * self.v_head_dim),
        )

        if self.g_proj is not None:
            attn_out *= self.g_proj(hidden_states)[0].sigmoid()
        return self.o_proj(attn_out)[0]


VllmMultiHeadLatentAttentionWrapper = VllmTPUMultiHeadLatentAttentionWrapper


@DeepseekScalingRotaryEmbedding.register_oot
class VllmDeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbedding):
    """TPU-compatible DeepSeek Scaling RoPE implementation."""

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert key is not None
        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        pos = torch.add(positions,
                        offsets) if offsets is not None else positions
        cos_sin = self.cos_sin_cache.index_select(0, pos)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
            sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key
