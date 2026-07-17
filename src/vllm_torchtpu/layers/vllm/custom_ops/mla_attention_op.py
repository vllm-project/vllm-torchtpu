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

import functools
from typing import Any

import jax
import torch
from torch.nn import Parameter
from torch_tpu._internal import pallas, sync
from vllm.config import CacheConfig
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

from vllm_torchtpu.layers.common.attention_interface import mla_attention
from vllm_torchtpu.layers.common.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.vllm.attention import TPU_STR_DTYPE_TO_TORCH_DTYPE
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context


def mla_attention_core_tpu(
    kv_cache: jax.Array,
    q_nope: jax.Array,
    q_pe: jax.Array,
    kv_c_normed: jax.Array,
    k_pe: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    request_distribution: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    num_attention_heads: int,
    qk_nope_head_dim: int,
    sm_scale: float,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    metadata = AttentionMetadata(
        input_positions=None,
        block_tables=block_tables,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
    )
    new_kv_cache, outputs = mla_attention(
        q_nope,
        q_pe,
        kv_c_normed,
        k_pe,
        kv_cache,
        metadata,
        mesh,
        num_attention_heads,
        qk_nope_head_dim,
        sm_scale=sm_scale,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return new_kv_cache, outputs


class VllmMLAAttention(MLAAttention):

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
        super().__init__(num_heads, scale, qk_nope_head_dim, qk_rope_head_dim,
                         v_head_dim, q_lora_rank, kv_lora_rank, kv_b_proj,
                         cache_config, quant_config, prefix, attn_backend,
                         use_sparse, indexer, **extra_impl_args)

        # For compatibility reasons.
        self.kv_sharing_target_layer_name = None
        self.attn_type = AttentionType.DECODER
        self.sliding_window = None

        self.kv_cache_quantized_dtype = None
        if self.kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE.get(
                self.kv_cache_dtype.lower().strip())

    def _build_mla_op(self,
                      q_scale: float | None = None,
                      k_scale: float | None = None,
                      v_scale: float | None = None):
        vllm_context = get_vllm_model_wrapper_context()
        wrapped_fn = functools.partial(
            mla_attention_core_tpu,
            mesh=vllm_context.mesh,
            num_attention_heads=self.num_heads,
            qk_nope_head_dim=self.qk_nope_head_dim,
            sm_scale=self.scale,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        op_name = f"pallas::mla_attention_{self.layer_name.replace('.', '_')}"
        mla_jax_op = pallas.jax_op(op_name, wrapped_fn, donate_argnums=(0, ))

        def _fake_mla(kv_cache, q_nope, q_pe, kv_c_normed, k_pe, *args,
                      **kwargs):
            num_tokens = q_nope.size(0)
            out_shape = (num_tokens, self.num_heads, self.kv_lora_rank)
            return torch.empty_like(kv_cache), torch.empty(
                out_shape, dtype=q_nope.dtype, device=q_nope.device)

        mla_jax_op.register_fake(_fake_mla)

        def mla_impl(kv_cache: torch.Tensor, q_nope: torch.Tensor,
                     q_pe: torch.Tensor, kv_c_normed: torch.Tensor,
                     k_pe: torch.Tensor, seq_lens: torch.Tensor,
                     block_tables: torch.Tensor, query_start_loc: torch.Tensor,
                     request_distribution: torch.Tensor) -> torch.Tensor:
            new_kv, outputs = mla_jax_op(kv_cache, q_nope, q_pe, kv_c_normed,
                                         k_pe, seq_lens, block_tables,
                                         query_start_loc, request_distribution)

            kv_cache.copy_(new_kv)
            return outputs

        return mla_impl

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

        # Delete kv_b_proj_params as the dequantized weights are now stored
        # in self.W_UK_T and self.W_UV.
        kv_b_proj_params = dict(self.kv_b_proj.named_parameters())
        for key in kv_b_proj_params.keys():
            delattr(self.kv_b_proj, key)

        if self.W_UK_T.device.type == "tpu":
            sync.synchronize(self.W_UK_T, wait=True)
            if hasattr(self, "W_UK_T_scale"):
                sync.synchronize(self.W_UK_T_scale, wait=True)
            sync.synchronize(self.W_UV, wait=True)
            if hasattr(self, "W_UV_scale"):
                sync.synchronize(self.W_UV_scale, wait=True)

        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype is not None:
            q_scale = getattr(self, "_q_scale_float", None)
            if q_scale is None and hasattr(self, "_q_scale"):
                q_scale = self._q_scale.item() if not isinstance(
                    self._q_scale, torch.Tensor
                ) or self._q_scale.ndim == 0 else self._q_scale.tolist()
            k_scale = getattr(self, "_k_scale_float", None)
            if k_scale is None and hasattr(self, "_k_scale"):
                k_scale = self._k_scale.item() if not isinstance(
                    self._k_scale, torch.Tensor
                ) or self._k_scale.ndim == 0 else self._k_scale.tolist()
            v_scale = getattr(self, "_v_scale_float", None)
            if v_scale is None and hasattr(self, "_v_scale"):
                v_scale = self._v_scale.item() if not isinstance(
                    self._v_scale, torch.Tensor
                ) or self._v_scale.ndim == 0 else self._v_scale.tolist()
            if v_scale is None:
                v_scale = k_scale

        self.mla_op = self._build_mla_op(q_scale=q_scale,
                                         k_scale=k_scale,
                                         v_scale=v_scale)

    def forward(self,
                q: tuple[torch.Tensor, torch.Tensor],
                kv_c_normed: torch.Tensor,
                k_pe: torch.Tensor,
                output: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(q, kv_c_normed, k_pe,
                                                self.layer_name)

        # Get the attention metadata and kv cache
        attn_metadata, _, kv_cache, _ = get_attention_context(self.layer_name)

        q_nope, q_pe = q
        input_dtype = q_nope.dtype

        # For determine_available_memory case.
        if kv_cache.numel() == 0:
            out_shape = (q_nope.shape[0], self.num_heads * self.v_head_dim)
            if output is None:
                output = torch.ones(out_shape,
                                    dtype=input_dtype,
                                    device=q_nope.device)
            else:
                output.fill_(1)
            return output

        # (B, N, P) x (N, P, L) -> (B, N, L)
        q_nope_t = q_nope.transpose(0, 1)
        ql_nope = torch.bmm(q_nope_t.to(torch.float32),
                            self.W_UK_T.to(torch.float32))
        if hasattr(self, "W_UK_T_scale"):
            ql_nope = ql_nope * self.W_UK_T_scale.to(torch.float32)
        ql_nope = ql_nope.transpose(0, 1).to(input_dtype)

        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype is not None:
            from vllm_torchtpu.layers.common.quantization import quantize_kv
            q_scale = getattr(self, "_q_scale_float", None)
            if q_scale is None and hasattr(self, "_q_scale"):
                q_scale = self._q_scale.item() if not isinstance(
                    self._q_scale, torch.Tensor
                ) or self._q_scale.ndim == 0 else self._q_scale.tolist()
            k_scale = getattr(self, "_k_scale_float", None)
            if k_scale is None and hasattr(self, "_k_scale"):
                k_scale = self._k_scale.item() if not isinstance(
                    self._k_scale, torch.Tensor
                ) or self._k_scale.ndim == 0 else self._k_scale.tolist()
            v_scale = getattr(self, "_v_scale_float", None)
            if v_scale is None and hasattr(self, "_v_scale"):
                v_scale = self._v_scale.item() if not isinstance(
                    self._v_scale, torch.Tensor
                ) or self._v_scale.ndim == 0 else self._v_scale.tolist()
            if v_scale is None:
                v_scale = k_scale

            kv_c_normed, _ = quantize_kv(self.kv_cache_quantized_dtype,
                                         kv_c_normed,
                                         value=None,
                                         k_scale=k_scale)
            k_pe, _ = quantize_kv(self.kv_cache_quantized_dtype,
                                  k_pe,
                                  value=None,
                                  k_scale=k_scale)

        ql_nope = ql_nope.view(-1, self.num_heads, self.kv_lora_rank)
        q_pe = q_pe.view(-1, self.num_heads, self.qk_rope_head_dim)
        kv_c_normed = kv_c_normed.view(-1, self.kv_lora_rank)
        k_pe = k_pe.view(-1, self.qk_rope_head_dim)

        # Call mla_op
        outputs = self.mla_op(kv_cache, ql_nope, q_pe, kv_c_normed, k_pe,
                              attn_metadata.seq_lens,
                              attn_metadata.block_tables,
                              attn_metadata.query_start_loc,
                              attn_metadata.request_distribution)

        outputs_t = outputs.reshape(-1, self.num_heads,
                                    self.kv_lora_rank).transpose(0, 1)
        out_proj = torch.bmm(outputs_t.to(torch.float32),
                             self.W_UV.to(torch.float32))
        if hasattr(self, "W_UV_scale"):
            out_proj = out_proj * self.W_UV_scale.to(torch.float32)
        outputs = out_proj.transpose(0, 1).to(input_dtype).reshape(
            -1, self.num_heads * self.v_head_dim)

        if outputs is not output and output is not None:
            output.copy_(outputs)

        return outputs


@MultiHeadLatentAttentionWrapper.register_oot
class VllmMultiHeadLatentAttentionWrapper(MultiHeadLatentAttentionWrapper):

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
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.skip_topk = skip_topk

        if self.indexer is not None and not self.skip_topk:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
            self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = VllmMLAAttention(
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

        return self.o_proj(attn_out)[0]


@DeepseekScalingRotaryEmbedding.register_oot
class VllmDeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbedding):

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
