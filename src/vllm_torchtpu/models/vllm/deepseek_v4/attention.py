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
"""DeepSeek-V4 sparse MLA attention layer."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.deepseek_v4 import attention as dsv4_attention
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionBackend

from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4.deepseek_v4_attention_op import (
    BATCH_AXIS, VllmDeepseekV4SWACache, _attention_jax,
    get_packed_mla_head_size)
from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4.deepseek_v4_compressor import \
    VllmDeepseekCompressor
from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4.deepseek_v4_indexer import \
    VllmDeepseekV4Indexer
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)

# Cache compiled Pallas JAX attention ops by name to avoid duplicate graph tracing across layers.
_attn_op_cache: dict[str, Any] = {}


class VllmDeepseekV4MLAAttention(dsv4_attention.DeepseekV4Attention,
                                 AttentionLayerBase):
    """Sparse MLA attention on TPU, over the SWA and compressed-KV caches."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list | None = None,
    ) -> None:
        # Route submodule instantiation to TorchTPU-native implementations. The
        # base constructors allocate torch.cuda.Event for an aux-stream fan-out
        # this backend replaces, so the events are stubbed out for the call.
        orig_indexer = dsv4_attention.DeepseekV4Indexer
        orig_compressor = dsv4_attention.DeepseekCompressor
        orig_swa_cache = dsv4_attention.DeepseekV4SWACache
        orig_cuda_event = torch.cuda.Event
        dsv4_attention.DeepseekV4Indexer = VllmDeepseekV4Indexer
        dsv4_attention.DeepseekCompressor = VllmDeepseekCompressor
        dsv4_attention.DeepseekV4SWACache = VllmDeepseekV4SWACache
        torch.cuda.Event = lambda *args, **kwargs: None

        try:
            super().__init__(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            )
            self.custom_prefix = prefix
            object.__setattr__(self, "mla_attn", self)
            for module in self.modules():
                if module is not self and hasattr(module, "get_kv_cache_spec"):
                    if module.__class__.__name__ == "DeepseekV4Attention":
                        module.get_kv_cache_spec = self.get_kv_cache_spec
        finally:
            dsv4_attention.DeepseekV4Indexer = orig_indexer
            dsv4_attention.DeepseekCompressor = orig_compressor
            dsv4_attention.DeepseekV4SWACache = orig_swa_cache
            torch.cuda.Event = orig_cuda_event

        # Bind compressor key-cache reference to SWA or main layer depending on compression ratio.
        if hasattr(self, "compressor") and self.compressor is not None:
            if self.compress_ratio <= 1:
                object.__setattr__(self.compressor, "k_cache",
                                   self.swa_cache_layer)
            else:
                object.__setattr__(self.compressor, "k_cache", self)

        # Set up TP-aware weight loader for attention sink parameters.
        if hasattr(self, "attn_sink") and self.attn_sink is not None:

            def _attn_sink_loader(param, loaded_weight):
                tp_size = get_tensor_model_parallel_world_size()
                tp_rank = get_tensor_model_parallel_rank()
                n_local = loaded_weight.shape[0] // tp_size
                narrow = loaded_weight[tp_rank * n_local:(tp_rank + 1) *
                                       n_local]
                param[:narrow.shape[0]].copy_(narrow)

            self.attn_sink.weight_loader = _attn_sink_loader

        hf_config = vllm_config.model_config.hf_config
        object.__setattr__(self, "attn_out_dim",
                           hf_config.num_attention_heads * hf_config.head_dim)
        self.num_layers = hf_config.num_hidden_layers

        # Register submodules into static forward context for torch.compile tracking.
        compilation_config = vllm_config.compilation_config
        if compilation_config is not None:
            compilation_config.static_forward_context[prefix] = self
            if hasattr(self.mla_attn, "swa_cache_layer"):
                compilation_config.static_forward_context[
                    self.mla_attn.swa_cache_layer.
                    prefix] = self.mla_attn.swa_cache_layer
            if hasattr(self.mla_attn, "indexer") and hasattr(
                    self.mla_attn.indexer, "k_cache"):
                compilation_config.static_forward_context[
                    self.mla_attn.indexer.k_cache.
                    prefix] = self.mla_attn.indexer.k_cache
            if hasattr(self.mla_attn, "compressor") and hasattr(
                    self.mla_attn.compressor, "state_cache"):
                compilation_config.static_forward_context[
                    self.mla_attn.compressor.state_cache.
                    prefix] = self.mla_attn.compressor.state_cache

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def get_attn_backend(self) -> type[AttentionBackend]:
        from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackend
        return PallasAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """Derive the main compressed-KV cache spec, or None for SWA-only layers."""
        if self.compress_ratio <= 1:
            return None

        block_size = vllm_config.cache_config.block_size
        comp_ratio = max(1, self.compress_ratio)
        hf_config = vllm_config.model_config.hf_config
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=get_packed_mla_head_size(hf_config),
            dtype=torch.uint8,
            compress_ratio=min(comp_ratio, block_size),
            alignment=None,
        )

    @property
    def attn_op(self):
        """The Pallas attention op, built lazily on first forward with final runtime geometry."""
        op = self.__dict__.get("_attn_op_instance")
        if op is not None:
            return op

        swa_tensor = getattr(getattr(self.mla_attn, "swa_cache_layer", None),
                             "kv_cache", None)
        caches_real = swa_tensor is not None and swa_tensor.numel() > 0
        if not caches_real:
            prof = self.__dict__.get("_attn_op_prof_instance")
            if prof is not None:
                return prof

        op, built_from_real_caches = self._build_attn_op()
        object.__setattr__(
            self, "_attn_op_instance"
            if built_from_real_caches else "_attn_op_prof_instance", op)
        return op

    def _build_attn_op(self):
        vllm_context = get_vllm_model_wrapper_context()
        mesh = vllm_context.mesh

        swa_only = self.compress_ratio <= 1
        is_csa = self.compress_ratio == 4
        swa_cache = getattr(self.mla_attn, "swa_cache_layer", None)
        if swa_cache is None:
            raise RuntimeError(
                f"{self.custom_prefix}: swa_cache_layer unavailable at "
                "attention-op build time.")

        logical_page_size = swa_cache.block_size
        if logical_page_size <= 0:
            raise RuntimeError(
                f"{self.custom_prefix}: SWA logical page size is "
                f"{logical_page_size}; expected positive block size.")

        # Detect whether SWA and main KV caches share the same underlying memory buffer.
        main_cache = getattr(self.mla_attn, "kv_cache", None)
        swa_tensor = getattr(swa_cache, "kv_cache", None)
        two_caches_same_buffer = bool(
            not swa_only and main_cache is not None and swa_tensor is not None
            and main_cache.numel() > 0 and swa_tensor.numel() > 0
            and main_cache.data_ptr() == swa_tensor.data_ptr())

        built_from_real_caches = bool(swa_tensor is not None
                                      and swa_tensor.numel() > 0)

        cfg_lps = VllmDeepseekV4SWACache._swa_block_size(
            vllm_context.vllm_config.cache_config.block_size, self.window_size)
        if cfg_lps != logical_page_size:
            logical_page_size = cfg_lps

        hf_config = vllm_context.vllm_config.model_config.hf_config
        sm_scale = (getattr(self.mla_attn, "scale", None)
                    or getattr(self.mla_attn, "softmax_scale", None)
                    or getattr(self, "softmax_scale", None)
                    or getattr(self, "scale", None)
                    or (hf_config.head_dim**-0.5 if hasattr(
                        hf_config, "head_dim") else None))
        if sm_scale is None:
            raise RuntimeError(
                f"{self.custom_prefix}: attention scale unavailable at "
                "attention-op build time.")

        wrapped_fn = functools.partial(
            _attention_jax,
            sm_scale=sm_scale,
            sliding_window=self.window_size,
            logical_page_size=logical_page_size,
            swa_only=swa_only,
            is_csa=is_csa,
            two_caches_same_buffer=two_caches_same_buffer,
        )

        op_type = "swa" if swa_only else ("csa" if is_csa else "hca")
        # Build unique cache key encoding geometry, overlay status, and profiling stage.
        op_name = (f"pallas::deepseek_v4_attention_{op_type}_v2"
                   f"_p{logical_page_size}"
                   f"{'_aliased' if two_caches_same_buffer else ''}"
                   f"{'' if built_from_real_caches else '_prof'}")

        if op_name in _attn_op_cache:
            return _attn_op_cache[op_name], built_from_real_caches

        logger.info(
            "[ATTN_BUILD] %s: building %s logical_page_size=%s window=%s "
            "two_caches_same_buffer=%s real_caches=%s", self.custom_prefix,
            op_name, logical_page_size, self.window_size,
            two_caches_same_buffer, built_from_real_caches)

        attn_data_axis = None
        attn_head_axis = "model"
        batch_axis = BATCH_AXIS
        extra_spec = P(batch_axis) if not is_csa else P(batch_axis, None)

        # Donate sw_cache to permit in-place updates by the Pallas sliding-window kernel.
        attn_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            mesh=mesh,
            donate_argnums=(2, ),
            input_partition_specs=(
                P(attn_data_axis, attn_head_axis, None),  # q
                P(),  # new_kv
                P(),  # sw_cache
                P(batch_axis),  # swa_kv_lens
                P(batch_axis),  # swa_page_indices
                P(),  # swa_cu_q_lens
                P(),  # swa_distribution
                P(),  # main_cache_kv
                P(batch_axis),  # main_kv_lens
                extra_spec,  # extra
                P(batch_axis),  # main_page_indices
                P(),  # main_cu_q_lens
                P(),  # main_distribution
                P(attn_head_axis),  # attention_sinks
            ),
        )

        def _fake_attn(q, new_kv, sw_cache, *args, **kwargs):
            """Abstract fake tensor implementation for PyTorch Dynamo graph tracing."""
            num_tokens = q.shape[0]
            out_tensor = torch.empty((num_tokens, self.mla_attn.n_local_heads,
                                      self.mla_attn.head_dim),
                                     dtype=q.dtype,
                                     device=q.device)
            return out_tensor, torch.empty_like(sw_cache)

        attn_jax_op.register_fake(_fake_attn)
        _attn_op_cache[op_name] = attn_jax_op
        return attn_jax_op, built_from_real_caches

    def attn_gemm(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None,
               torch.Tensor | None]:
        """Project input states to query/key latents and compute indexing/compression scores."""
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)

        compressor = getattr(self.mla_attn, "compressor", None)
        if compressor is not None:
            kv_score = hidden_states @ compressor.fused_wkv_wgate.weight.T
        else:
            kv_score = None

        if self.indexer is not None:
            indexer = self.indexer
            indexer_weights, _ = indexer.weights_proj(hidden_states)
            indexer_kv_score = (
                hidden_states @ indexer.compressor.fused_wkv_wgate.weight.T)
        else:
            indexer_weights = None
            indexer_kv_score = None

        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    def qnorm_rope(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RMSNorm and scaling RoPE to query head states."""
        orig_dtype = q.dtype
        qf = q.to(torch.float32)

        rms = torch.rsqrt(
            qf.pow(2).mean(dim=-1, keepdim=True) + self.mla_attn.eps)
        qf = qf * rms

        q_rotated, _ = self.mla_attn.rotary_emb(positions, qf)
        return q_rotated.to(orig_dtype)

    def kv_rope(
        self,
        kv: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply scaling RoPE to decoupled key states."""
        orig_dtype = kv.dtype
        kv_rotated, _ = self.mla_attn.rotary_emb(
            positions,
            kv.unsqueeze(1).to(torch.float32))
        return kv_rotated.squeeze(1).to(orig_dtype)

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor | None,
        indexer_kv_score: torch.Tensor | None,
        indexer_weights: torch.Tensor | None,
        positions: torch.Tensor,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        """Execute indexing, state compression, and kernel attention dispatch."""
        q = self.wq_b(qr).view(qr.shape[0], self.mla_attn.n_local_heads,
                               self.mla_attn.head_dim)
        q = self.qnorm_rope(q, positions)
        kv = self.kv_rope(kv, positions)

        topk_indices = None
        compressor = getattr(self.mla_attn, "compressor", None)
        if self.indexer is not None:
            indexer_emb = getattr(self.mla_attn, "indexer_rotary_emb",
                                  getattr(self.mla_attn, "rotary_emb", None))
            topk_indices = self.indexer(hidden_states, qr, indexer_kv_score,
                                        indexer_weights, positions,
                                        indexer_emb)
        if compressor is not None:
            compressor(kv_score, positions, self.mla_attn.rotary_emb)

        res = self.forward_mqa(
            q,
            kv,
            positions,
            None,
            topk_indices=topk_indices,
        )
        return res

    @staticmethod
    def _as_kernel_cache_view(cache: torch.Tensor) -> torch.Tensor:
        """Bitcast 1-byte cache allocations to uint8 required by Pallas kernels."""
        if cache.dtype != torch.uint8:
            return cache.view(torch.uint8)
        return cache

    def _get_active_sw_cache(self) -> torch.Tensor | None:
        """Retrieve active sliding-window cache buffer if allocated."""
        swa_cache_layer = getattr(self.mla_attn, "swa_cache_layer", None)
        if swa_cache_layer is None:
            raise RuntimeError(
                f"{self.custom_prefix}: swa_cache_layer unavailable.")
        cache = getattr(swa_cache_layer, "kv_cache", None)
        if cache is not None and cache.numel() > 0:
            return self._as_kernel_cache_view(cache)
        return None

    def _get_active_main_cache(self) -> torch.Tensor | None:
        """Retrieve active main compressed-KV cache buffer if allocated."""
        cache = getattr(self, "kv_cache", None)
        if cache is not None and cache.numel() > 0:
            return self._as_kernel_cache_view(cache)
        return None

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor | None,
        *,
        topk_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Prepare metadata operands and dispatch the Pallas attention op."""
        attn_ctx = get_forward_context().attn_metadata
        main_prefix = getattr(self.mla_attn, "prefix",
                              getattr(self, "custom_prefix", ""))

        def _get_field(meta, field_name):
            if meta is None:
                return None
            if isinstance(meta, dict):
                return meta.get(field_name)
            return getattr(meta, field_name, None)

        if isinstance(attn_ctx, dict):
            swa_cache_layer = getattr(self.mla_attn, "swa_cache_layer", None)
            swa_layer_name = getattr(swa_cache_layer, "prefix", None)
            if swa_layer_name not in attn_ctx:
                raise RuntimeError(
                    f"{self.custom_prefix}: no attention metadata for SWA "
                    f"layer {swa_layer_name!r}.")
            swa_attn_metadata = attn_ctx[swa_layer_name]
            main_attn_metadata = attn_ctx.get(main_prefix)
        else:
            swa_attn_metadata = attn_ctx
            main_attn_metadata = attn_ctx

        orig_sw_cache = self._get_active_sw_cache()
        sw_cache = orig_sw_cache
        if sw_cache is None:
            # 0-token placeholder cache used during profiling forward passes.
            sw_cache = torch.zeros((0, 1, 1, self.mla_attn.head_dim),
                                   dtype=torch.uint8,
                                   device=q.device)

        swa_only = self.compress_ratio <= 1
        is_csa = self.compress_ratio == 4

        if swa_attn_metadata is not None:
            swa_seq_lens = _get_field(swa_attn_metadata, "seq_lens")
            swa_block_tables = _get_field(swa_attn_metadata, "block_tables")
            swa_query_start_loc = _get_field(swa_attn_metadata,
                                             "query_start_loc")
            swa_request_distribution = _get_field(swa_attn_metadata,
                                                  "request_distribution")
        else:
            swa_seq_lens = None
            swa_block_tables = None
            swa_query_start_loc = None
            swa_request_distribution = None

        if not swa_only and main_attn_metadata is not None:
            main_cache_kv = self._get_active_main_cache()
            if main_cache_kv is None:
                # Clone tensor handle so XLA traces distinct operands for signature matching.
                main_cache_kv = sw_cache.clone()
            m_seq_lens = _get_field(main_attn_metadata, "seq_lens")
            main_kv_lens = m_seq_lens // self.compress_ratio if m_seq_lens is not None else None
            main_page_indices = _get_field(main_attn_metadata, "block_tables")
            main_cu_q_lens = _get_field(main_attn_metadata, "query_start_loc")
            main_distribution = _get_field(main_attn_metadata,
                                           "request_distribution")
        else:
            main_cache_kv = sw_cache.clone()
            main_kv_lens = swa_seq_lens.clone(
            ) if swa_seq_lens is not None else None
            main_page_indices = swa_block_tables.clone(
            ) if swa_block_tables is not None else None
            main_cu_q_lens = swa_query_start_loc.clone(
            ) if swa_query_start_loc is not None else None
            main_distribution = swa_request_distribution.clone(
            ) if swa_request_distribution is not None else None

        if is_csa:
            assert topk_indices is not None
            extra = topk_indices
        else:
            extra = (positions + 1).to(torch.int32) // self.compress_ratio

        output, new_sw_cache = self.attn_op(
            q,
            kv,
            sw_cache,
            swa_seq_lens,
            swa_block_tables,
            swa_query_start_loc,
            swa_request_distribution,
            main_cache_kv,
            main_kv_lens,
            extra,
            main_page_indices,
            main_cu_q_lens,
            main_distribution,
            self.attn_sink,
        )

        if orig_sw_cache is not None and new_sw_cache is not None and new_sw_cache.numel(
        ) > 0:
            orig_sw_cache.copy_(new_sw_cache)

        return output

    def _o_proj(self, o: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """Apply inverse RoPE and low-rank output projections wo_a and wo_b."""
        t = o.shape[0]
        o_f = o.to(torch.float32).view(t, self.n_local_heads, self.head_dim)
        o_ref, _ = self.rotary_emb(positions, o_f, inverse=True)
        o_ref = o_ref.to(torch.bfloat16)

        n_local_groups = self.n_local_groups
        o_lora_rank = self.o_lora_rank

        o_ref_grouped = o_ref.view(t, n_local_groups, -1)
        wo_a_grouped = self.wo_a.weight.view(n_local_groups, o_lora_rank, -1)
        wo_a_scale = self.wo_a.weight_scale.view(n_local_groups, o_lora_rank)

        z = torch.einsum("tgd,grd->tgr", o_ref_grouped,
                         wo_a_grouped.to(torch.bfloat16))
        z = z * wo_a_scale.unsqueeze(0).to(torch.bfloat16)
        z = z.to(torch.bfloat16)
        z_flat = z.flatten(1)

        out = self.wo_b(z_flat)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for DeepSeek-V4 attention layer."""
        if positions.ndim > 1:
            positions = positions.flatten()

        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self.attn_gemm(hidden_states))

        qr, kv = qr_kv.split(
            [self.mla_attn.q_lora_rank, self.mla_attn.head_dim], dim=-1)
        qr = self.q_norm(qr)
        kv = self.kv_norm(kv)

        attn_output = self.attention_impl(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            None,
        )

        out = self._o_proj(attn_output, positions)
        return out
