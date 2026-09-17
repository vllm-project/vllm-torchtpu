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

import jax
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
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackend

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionBackend

from vllm_torchtpu.kernels.deepseek_v4 import rope as rope_kernel
from vllm_torchtpu.kernels.deepseek_v4.o_projection import \
    fused_reverse_rope_wo_a_projection
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_attention_op import (
    BATCH_AXIS, VllmDeepseekV4SWACache, _attention_csa, _attention_hca,
    get_packed_mla_head_size)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_compressor import \
    VllmDeepseekCompressor
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_indexer import \
    VllmDeepseekV4Indexer
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

logger = init_logger(__name__)

_pallas_op_cache: dict[str, Any] = {}


class DeepseekV4TPUAttentionBackend(PallasAttentionBackend):

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.BLHNC, )


# Module-level so `pallas.jax_op` can trace and register them as torch ops.
def _qnorm_rope_jax(
    x: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    *,
    eps: float,
) -> jax.Array:
    """Per-head RMSNorm (no weight) fused with RoPE on the query heads."""
    return rope_kernel.qnorm_rope(x, positions, cos_sin_cache, eps=eps)


def _rope_jax(
    x: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
) -> jax.Array:
    """RoPE over the trailing `rotary_dim` channels of `x`."""
    return rope_kernel.rope(x, positions, cos_sin_cache)


def _o_proj_jax(
    x: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    wo_a: jax.Array,
    wo_a_scale: jax.Array,
    *,
    head_dim: int,
) -> jax.Array:
    """Inverse RoPE fused into the per-group `wo_a` projection."""
    return fused_reverse_rope_wo_a_projection(
        x,
        positions,
        cos_sin_cache,
        wo_a,
        wo_a_scale.reshape(-1),
        head_dim=head_dim,
        inverse=True,
        quantize_activations=True,
    )


def _fake_rope(x, positions, cos_sin_cache, *args, **kwargs):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    return torch.empty_like(x)


def _fake_o_proj(x, positions, cos_sin_cache, wo_a, wo_a_scale, *args,
                 **kwargs):
    """Abstract implementation for PyTorch Dynamo graph tracing."""
    return torch.empty((x.shape[0], wo_a.shape[-1]),
                       dtype=x.dtype,
                       device=x.device)


def _name_float(value: float) -> str:
    """A float rendered so it is safe inside a torch op name."""
    return f"{value:.6g}".replace(".", "p").replace("-", "m").replace("+", "")


def _live_cache(entry: object) -> torch.Tensor | None:
    """The array bound to a layer, or None while it has none.

    vLLM binds a numel==0 placeholder until `initialize_kv_cache` runs, so an
    empty array is reported as absent rather than handed to a kernel.
    """
    return entry if isinstance(entry,
                               torch.Tensor) and entry.numel() > 0 else None


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
        return DeepseekV4TPUAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """Derive the main compressed-KV cache spec, or None for SWA-only layers."""
        if self.compress_ratio <= 1:
            return None

        block_size = vllm_config.cache_config.block_size
        comp_ratio = max(1, self.compress_ratio)
        hf_config = vllm_config.model_config.hf_config
        if comp_ratio == 4:
            # CSA (`sparse_mla`) reads the NoPE record from this array and the
            # RoPE channels from the companion `{prefix}_rope` array; the
            # packed width here budgets for both.
            head_size = get_packed_mla_head_size(hf_config)
        else:
            # HCA (`mla`) stores raw bf16 latents rather than pay for DSv4 FP8
            # quantization and dequantization on every access. Its cache is a
            # small share of the total, so the extra width is cheap.
            head_size = 512 * 2
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=head_size,
            dtype=torch.uint8,
            tokens_per_state=min(comp_ratio, block_size),
            alignment=None,
        )

    @property
    def attn_op(self):
        """The Pallas attention op, built lazily on first forward with final runtime geometry."""
        op = self.__dict__.get("_attn_op_instance")
        if op is not None:
            return op

        swa_tensor = _live_cache(
            getattr(getattr(self.mla_attn, "swa_cache_layer", None),
                    "kv_cache", None))
        if swa_tensor is None:
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
        # A CSA layer binds a `(nope, rope)` pair; the SWA cache overlays the
        # NoPE array, so that is the one to compare.
        main_entry = getattr(self.mla_attn, "kv_cache", None)
        main_cache = _live_cache(
            main_entry[0] if isinstance(main_entry, tuple) else main_entry)
        swa_tensor = _live_cache(getattr(swa_cache, "kv_cache", None))
        # `_live_cache` already reports numel==0 placeholders as absent, so
        # a non-None tensor here is a real allocation.
        two_caches_same_buffer = bool(
            not swa_only and main_cache is not None and swa_tensor is not None
            and main_cache.data_ptr() == swa_tensor.data_ptr())

        built_from_real_caches = swa_tensor is not None

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
            _attention_csa if is_csa else _attention_hca,
            sm_scale=sm_scale,
            sliding_window=self.window_size,
            logical_page_size=logical_page_size,
            swa_only=swa_only,
            two_caches_same_buffer=two_caches_same_buffer,
        )

        op_type = "swa" if swa_only else ("csa" if is_csa else "hca")
        # Build unique cache key encoding geometry, overlay status, and profiling stage.
        op_name = (f"pallas::deepseek_v4_attention_{op_type}"
                   f"_p{logical_page_size}"
                   f"{'_aliased' if two_caches_same_buffer else ''}"
                   f"{'' if built_from_real_caches else '_prof'}")

        if op_name in _pallas_op_cache:
            return _pallas_op_cache[op_name], built_from_real_caches

        logger.info(
            "[ATTN_BUILD] %s: building %s logical_page_size=%s window=%s "
            "two_caches_same_buffer=%s real_caches=%s", self.custom_prefix,
            op_name, logical_page_size, self.window_size,
            two_caches_same_buffer, built_from_real_caches)

        attn_data_axis = None
        attn_head_axis = "model"
        batch_axis = BATCH_AXIS
        extra_spec = P(batch_axis) if not is_csa else P(batch_axis, None)

        input_partition_specs = (
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
        )
        if is_csa:
            input_partition_specs += (P(), )  # main_cache_rope

        # Donate sw_cache to permit in-place updates by the Pallas sliding-window kernel.
        attn_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            mesh=mesh,
            donate_argnums=(2, ),
            input_partition_specs=input_partition_specs,
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
        _pallas_op_cache[op_name] = attn_jax_op
        return attn_jax_op, built_from_real_caches

    def attn_gemm(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        qr_kv, _ = self.fused_wqa_wkv(hidden_states)

        if self.indexer is not None:
            indexer_weights, _ = self.indexer.weights_proj(hidden_states)
        else:
            indexer_weights = None

        return qr_kv, indexer_weights

    @property
    def qnorm_rope_op(self):
        """The fused q-RMSNorm + RoPE op, built lazily on first forward."""
        op = self.__dict__.get("_qnorm_rope_op_instance")
        if op is not None:
            return op

        eps = float(self.mla_attn.eps)
        op_name = f"pallas::deepseek_v4_qnorm_rope_e{_name_float(eps)}"
        op = _pallas_op_cache.get(op_name)
        if op is None:
            op = pallas.jax_op(
                op_name,
                functools.partial(_qnorm_rope_jax, eps=eps),
                mesh=get_vllm_model_wrapper_context().mesh,
                input_partition_specs=(
                    P(None, "model", None),  # q
                    P(),  # positions
                    P(),  # cos_sin_cache
                ),
            )
            op.register_fake(_fake_rope)
            _pallas_op_cache[op_name] = op
        object.__setattr__(self, "_qnorm_rope_op_instance", op)
        return op

    @property
    def kv_rope_op(self):
        """The RoPE op for the decoupled key states, built lazily."""
        op = self.__dict__.get("_kv_rope_op_instance")
        if op is not None:
            return op

        op_name = "pallas::deepseek_v4_kv_rope"
        op = _pallas_op_cache.get(op_name)
        if op is None:
            op = pallas.jax_op(
                op_name,
                _rope_jax,
                mesh=get_vllm_model_wrapper_context().mesh,
                input_partition_specs=(
                    P(),  # kv
                    P(),  # positions
                    P(),  # cos_sin_cache
                ),
            )
            op.register_fake(_fake_rope)
            _pallas_op_cache[op_name] = op
        object.__setattr__(self, "_kv_rope_op_instance", op)
        return op

    def qnorm_rope(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RMSNorm and scaling RoPE to query head states."""
        return self.qnorm_rope_op(q, positions,
                                  self.mla_attn.rotary_emb.cos_sin_cache)

    def kv_rope(
        self,
        kv: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply scaling RoPE to decoupled key states."""
        return self.kv_rope_op(kv, positions,
                               self.mla_attn.rotary_emb.cos_sin_cache)

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
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
            topk_indices = self.indexer(hidden_states, qr, indexer_weights,
                                        positions, indexer_emb)
        if compressor is not None:
            compressor(hidden_states, positions, self.mla_attn.rotary_emb)

        res = self.forward_mqa(
            q,
            kv,
            positions,
            None,
            topk_indices=topk_indices,
        )
        return res

    @staticmethod
    def _as_kernel_cache_view(
            cache: torch.Tensor | None) -> torch.Tensor | None:
        """Bitcast 1-byte cache allocations to uint8 required by Pallas kernels."""
        if cache is None:
            return None
        if cache.dtype != torch.uint8:
            return cache.view(torch.uint8)
        return cache

    def _get_active_sw_cache(self) -> torch.Tensor | None:
        """Retrieve active sliding-window cache buffer if allocated."""
        swa_cache_layer = getattr(self.mla_attn, "swa_cache_layer", None)
        if swa_cache_layer is None:
            raise RuntimeError(
                f"{self.custom_prefix}: swa_cache_layer unavailable.")
        return self._as_kernel_cache_view(
            _live_cache(getattr(swa_cache_layer, "kv_cache", None)))

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

        # A CSA layer's KV entry is a `(nope, rope)` pair.
        kv_entry = getattr(self, "kv_cache", None)
        nope_entry, rope_entry = (kv_entry if isinstance(kv_entry, tuple) else
                                  (kv_entry, None))

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
            main_cache_kv = self._as_kernel_cache_view(_live_cache(nope_entry))
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

        operands = (
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
        if is_csa:
            main_cache_rope = self._as_kernel_cache_view(
                _live_cache(rope_entry))
            if main_cache_rope is None:
                if _live_cache(nope_entry) is not None:
                    # Not the profiling pass -- the compressed-KV array is
                    # live, so the companion must be too. Falling through
                    # would hand the kernel the NoPE array as its RoPE
                    # operand and silently attend to the wrong channels.
                    raise RuntimeError(
                        f"{self.custom_prefix}: CSA layer has a live "
                        "compressed-KV array but no companion RoPE array. "
                        "The runner binds them as a (nope, rope) pair on this "
                        "layer's `kv_cache`; got "
                        f"{type(kv_entry).__name__}.")
                # Profiling pass: shapes are placeholders and the kernel is
                # skipped. Clone rather than alias -- XLA dedupes identical
                # inputs into one operand and desyncs the declared signature.
                main_cache_rope = main_cache_kv.clone()
            operands += (main_cache_rope, )

        output, new_sw_cache = self.attn_op(*operands)

        if orig_sw_cache is not None and new_sw_cache is not None and new_sw_cache.numel(
        ) > 0:
            orig_sw_cache.copy_(new_sw_cache)

        return output

    @property
    def o_proj_op(self):
        """The fused inverse-RoPE + `wo_a` projection op, built lazily."""
        op = self.__dict__.get("_o_proj_op_instance")
        if op is not None:
            return op

        self._assert_o_proj_supported()
        head_dim = int(self.head_dim)
        op_name = f"pallas::deepseek_v4_o_proj_d{head_dim}"
        op = _pallas_op_cache.get(op_name)
        if op is None:
            op = pallas.jax_op(
                op_name,
                functools.partial(_o_proj_jax, head_dim=head_dim),
                mesh=get_vllm_model_wrapper_context().mesh,
                input_partition_specs=(
                    P(None, "model", None),  # o
                    P(),  # positions
                    P(),  # cos_sin_cache
                    P(),  # wo_a
                    P(),  # wo_a_scale
                ),
            )
            op.register_fake(_fake_o_proj)
            _pallas_op_cache[op_name] = op
        object.__setattr__(self, "_o_proj_op_instance", op)
        return op

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        """Store `wo_a` in the layout the fused o-projection kernel wants,
        which is (self.n_local_heads * self.head_dim // self.n_local_groups,
                  self.n_local_groups * self.o_lora_rank).
        """
        del act_dtype
        weight = self.wo_a.weight
        transposed = (self.n_local_heads * self.head_dim //
                      self.n_local_groups,
                      self.n_local_groups * self.o_lora_rank)
        if tuple(weight.shape) == transposed:
            # Already done -- a reload path can run this hook twice.
            return
        assert tuple(weight.shape) == transposed[::-1], (
            f"{self.prefix}.wo_a: expected the linear to hold "
            f"{list(transposed[::-1])}, got {list(weight.shape)}.")

        out = torch.empty(transposed, dtype=weight.dtype, device=weight.device)
        out.copy_(weight.data.t())
        self.wo_a.weight = torch.nn.Parameter(out, requires_grad=False)

    def _assert_o_proj_supported(self) -> None:
        """Check what the fused `wo_a` kernel requires.

        The kernel is the only `wo_a` path, so an unmet condition is a
        configuration error.
        """
        weight = self.wo_a.weight
        scale = getattr(self.wo_a, "weight_scale", None)

        # Both MXU operands are fp8: the kernel quantizes the activations
        # itself and multiplies them against the fp8 `wo_a`. A different
        # runtime weight dtype (REQUANTIZE_WEIGHT_DTYPE) has nothing to run on.
        assert weight.dtype == torch.float8_e4m3fn, (
            f"{self.prefix}.wo_a: the fused o-projection kernel needs an "
            f"fp8_e4m3fn weight, got {weight.dtype}. Check "
            "REQUANTIZE_WEIGHT_DTYPE and that the checkpoint is fp8.")

        # Per-output-channel scales only. The blockwise requantization path
        # (ENABLE_QUANTIZED_MATMUL_KERNEL + REQUANTIZE_BLOCK_SIZE) reshapes
        # this to [n_in_blocks, 1, n_out], which the kernel cannot consume.
        assert scale is not None and scale.ndim == 1 and scale.shape[0] == (
            weight.shape[-1]), (
                f"{self.prefix}.wo_a: the fused o-projection kernel needs a "
                f"per-channel weight_scale of shape [{weight.shape[-1]}], got "
                f"{None if scale is None else list(scale.shape)}. Unset "
                "REQUANTIZE_BLOCK_SIZE.")

        # `wo_a` is applied per group, and the kernel assumes one group's
        # heads fill exactly one sublane.
        assert self.n_local_heads % self.n_local_groups == 0
        heads_per_group = self.n_local_heads // self.n_local_groups
        assert heads_per_group == 8, (
            f"{self.prefix}.wo_a: the fused o-projection kernel assumes 8 "
            f"heads per group, got {heads_per_group}.")

        # Lane/rotation constraints of the kernel and its cos/sin gather.
        assert self.head_dim % 128 == 0, (
            f"{self.prefix}.wo_a: head_dim {self.head_dim} is not a multiple "
            "of the 128-lane width.")
        rotary_dim = self.rotary_emb.cos_sin_cache.shape[-1]
        assert rotary_dim % 2 == 0 and rotary_dim <= 128, (
            f"{self.prefix}.wo_a: rotary_dim {rotary_dim} must be even and at "
            "most 128.")

    def _o_proj(self, o: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        """Apply inverse RoPE and low-rank output projections wo_a and wo_b."""
        # The kernel folds the inverse RoPE, the activation quantization and
        # the per-group `wo_a` matmul into one pass.
        assert o.dtype == torch.bfloat16, (
            f"{self.prefix}: the fused o-projection kernel needs bf16 "
            f"activations, got {o.dtype}.")
        assert o.shape[1:] == (self.n_local_heads, self.head_dim), (
            f"{self.prefix}: the fused o-projection kernel consumes the heads "
            f"in place and needs [t, {self.n_local_heads}, {self.head_dim}], "
            f"got {list(o.shape)}.")
        z = self.o_proj_op(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a.weight,
            self.wo_a.weight_scale,
        )

        out = self.wo_b(z)
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

        qr_kv, indexer_weights = self.attn_gemm(hidden_states)

        qr, kv = qr_kv.split(
            [self.mla_attn.q_lora_rank, self.mla_attn.head_dim], dim=-1)
        qr = self.q_norm(qr)
        kv = self.kv_norm(kv)

        attn_output = self.attention_impl(
            hidden_states,
            qr,
            kv,
            indexer_weights,
            positions,
            None,
        )

        out = self._o_proj(attn_output, positions)
        return out
