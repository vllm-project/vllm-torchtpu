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
from typing import Any, cast

import jax
import jax.numpy as jnp
import torch
import vllm.models.deepseek_v4.amd.model as ds_v4_amd_model
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4 import attention as dsv4_attention
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend, DeepseekV4SWACache)
from vllm.v1.kv_cache_interface import (KVCacheSpec, MLAAttentionSpec,
                                        SlidingWindowMLASpec)

from vllm_torchtpu.kernels.deepseek_v4.mla import mla_ragged_paged_attention
from vllm_torchtpu.kernels.deepseek_v4.mla_swa import \
    mla_sliding_window_ragged_paged_attention
from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4.deepseek_v4_compressor import \
    VllmDeepseekCompressor
from vllm_torchtpu.layers.vllm.custom_ops.deepseek_v4.deepseek_v4_indexer import \
    VllmDeepseekV4Indexer
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context
from vllm_torchtpu.utils import align_to

from vllm.model_executor.layers.attention_layer_base import (  # isort: skip
    AttentionBackend, AttentionLayerBase)

logger = init_logger(__name__)

_attn_op_cache = {}

BATCH_AXIS = None


# Module-level so `pallas.jax_op` can trace and register it as a torch op.
def _attention_jax(
    q: jax.Array,
    new_kv: jax.Array,
    sw_cache: jax.Array,
    swa_kv_lens: jax.Array,
    swa_page_indices: jax.Array,
    swa_cu_q_lens: jax.Array,
    swa_distribution: jax.Array,
    main_cache_kv: jax.Array,
    main_kv_lens: jax.Array,
    extra: jax.Array,
    main_page_indices: jax.Array,
    main_cu_q_lens: jax.Array,
    main_distribution: jax.Array,
    attention_sinks: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int,
    logical_page_size: int,
    swa_only: bool,
    is_csa: bool,
    two_caches_same_buffer: bool,
) -> tuple[jax.Array, jax.Array]:
    if main_cache_kv.shape[0] == 0 or (swa_only and sw_cache.shape[0] == 0):
        # Profiling-shape trace: the kernel is skipped. Unused inputs stay in
        # the compiled signature because torch_tpu jits with keep_unused=True.
        return jnp.zeros(q.shape, dtype=q.dtype), sw_cache

    def _attention_local(q, new_kv, sw_cache, swa_kv_lens, swa_page_indices,
                         swa_cu_q_lens, swa_distribution, main_cache_kv,
                         main_kv_lens, extra, main_page_indices,
                         main_cu_q_lens, main_distribution, attention_sinks):
        if swa_page_indices is not None:
            swa_page_indices = swa_page_indices.flatten()
        if main_page_indices is not None:
            main_page_indices = main_page_indices.flatten()

        if sw_cache.shape[0] == 0:
            swa_output = jnp.zeros_like(q)
            updated_sw_cache = sw_cache
            swa_l = jnp.zeros((q.shape[0], q.shape[1], 1), dtype=jnp.float32)
            swa_m = jnp.full((q.shape[0], q.shape[1], 1),
                             -1e9,
                             dtype=jnp.float32)
        else:
            swa_output, updated_sw_cache, swa_l, swa_m = (
                mla_sliding_window_ragged_paged_attention(
                    q=q,
                    new_kv=new_kv,
                    cache_kv=sw_cache,
                    kv_lens=swa_kv_lens,
                    page_indices=swa_page_indices,
                    cu_q_lens=swa_cu_q_lens,
                    distribution=swa_distribution,
                    attention_sinks=attention_sinks,
                    sm_scale=sm_scale,
                    sliding_window=sliding_window,
                    logical_page_size=logical_page_size,
                    num_kv_pages_per_block=1,
                    num_queries_per_block=1,
                    unnormalized_output=False if swa_only else True,
                ))

        if swa_only:
            return swa_output, updated_sw_cache

        # When both caches are one buffer, the compressed-KV read must see
        # the SWA kernel's write from this pass, not the pre-write snapshot.
        if two_caches_same_buffer:
            main_cache_kv = updated_sw_cache

        if swa_l is not None and swa_l.ndim == 3:
            swa_l = jnp.squeeze(swa_l, axis=-1)
        if swa_m is not None and swa_m.ndim == 3:
            swa_m = jnp.squeeze(swa_m, axis=-1)

        output = mla_ragged_paged_attention(
            q=q,
            cache_kv=main_cache_kv,
            kv_lens=main_kv_lens,
            kv_lens_to_attend=None if is_csa else extra,
            topk_indices=extra if is_csa else None,
            page_indices=main_page_indices,
            cu_q_lens=main_cu_q_lens,
            distribution=main_distribution,
            attention_sinks=attention_sinks,
            swa_accumulation=swa_output,
            swa_l=swa_l,
            swa_m=swa_m,
            sm_scale=sm_scale,
            num_kv_pages_per_block=1,
            num_queries_per_block=1,
        )

        return output, updated_sw_cache

    return _attention_local(q, new_kv, sw_cache, swa_kv_lens, swa_page_indices,
                            swa_cu_q_lens, swa_distribution, main_cache_kv,
                            main_kv_lens, extra, main_page_indices,
                            main_cu_q_lens, main_distribution, attention_sinks)


class VllmDeepseekSparseSWABackend(DeepseekSparseSWABackend):
    """TPU attention backend for DeepSeek-V4 SWA cache."""

    @classmethod
    def get_min_page_size(cls, vllm_config: VllmConfig) -> int:
        return 1


class VllmDeepseekV4SWACache(DeepseekV4SWACache):
    """Sliding-window KV cache, paged to fit inside a CSA main page."""

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config,
    ) -> None:
        super().__init__(head_dim, window_size, dtype, prefix, cache_config)
        # Provisional: `cache_config.block_size` is still vLLM's default at
        # construction time. `_swa_block_size` recomputes from the final config.
        self.block_size = self._swa_block_size(cache_config.block_size,
                                               window_size)

    @staticmethod
    def _swa_block_size(compressed_kv_cache_bz: int,
                        window_size: int | None) -> int:
        """SWA page size, sized to fit inside a CSA main cache page.

        A CSA page holds `block_size // 4` compressed rows, so the SWA cache
        uses the same divisor and then clamps to the window it must hold. The
        host page only has to be large enough, not equal.
        """
        csa_compression_ratio = 4
        block_size = compressed_kv_cache_bz // csa_compression_ratio
        if window_size is None:
            return block_size
        return min(block_size, window_size)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # `_build_attn_op` reads this attribute as the kernel's
        # `logical_page_size`, so spec and kernel must not disagree.
        self.block_size = self._swa_block_size(
            vllm_config.cache_config.block_size, self.window_size)
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=align_to(448 + 64 * 2 + 7, 128),
            dtype=torch.uint8,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            alignment=None,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return VllmDeepseekSparseSWABackend


# The AMD model constructs this class directly (amd/model.py:250);
# `patch_deepseek_v4_mla_cls` swaps the name to our TPU subclass.
# TODO(patemotter): drop this dependency once bringup is done, by subclassing
# the generic DeepseekV4Attention ABC and implementing its platform hooks.
_orig_ds_v4_attention_cls = ds_v4_amd_model.DeepseekV4ROCMAiterMLAAttention


# DeepseekV4Attention must come first: it may already inherit
# AttentionLayerBase, and the reverse order is an unresolvable MRO.
class VllmDeepseekV4MLAAttention(_orig_ds_v4_attention_cls,
                                 AttentionLayerBase):
    """Sparse MLA attention on TPU, over the SWA and compressed-KV caches."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list | None = None,
    ) -> None:
        orig_indexer = dsv4_attention.DeepseekV4Indexer
        orig_amd_indexer = getattr(ds_v4_amd_model, "DeepseekV4Indexer", None)
        orig_compressor = dsv4_attention.DeepseekCompressor
        orig_swa_cache = dsv4_attention.DeepseekV4SWACache
        dsv4_attention.DeepseekV4Indexer = VllmDeepseekV4Indexer
        if orig_amd_indexer is not None:
            ds_v4_amd_model.DeepseekV4Indexer = VllmDeepseekV4Indexer
        dsv4_attention.DeepseekCompressor = VllmDeepseekCompressor
        dsv4_attention.DeepseekV4SWACache = VllmDeepseekV4SWACache
        try:
            cast(Any, _orig_ds_v4_attention_cls).__init__(
                self,
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
                aux_stream_list=aux_stream_list,
            )
            self.custom_prefix = prefix
            _real_mla_attn = getattr(self, "mla_attn", self)
            # `self.mla_attn` aliases `self` below, so swa_cache_layer /
            # compressor / indexer must be read from the wrapper.
            for _attr in ("swa_cache_layer", "compressor", "indexer"):
                if hasattr(_real_mla_attn, _attr) and not hasattr(self, _attr):
                    object.__setattr__(self, _attr,
                                       getattr(_real_mla_attn, _attr))
            self.__dict__['mla_attn'] = self
            for module in self.modules():
                if module is not self and hasattr(module, "get_kv_cache_spec"):
                    if module.__class__.__name__ in (
                            "DeepseekV4MultiHeadLatentAttentionWrapper",
                            "DeepseekV4MLAAttention", "DeepseekV4Attention",
                            "DeepseekV4ROCMAiterMLAAttention"):
                        module.get_kv_cache_spec = self.get_kv_cache_spec
        finally:
            dsv4_attention.DeepseekV4Indexer = orig_indexer
            if orig_amd_indexer is not None:
                ds_v4_amd_model.DeepseekV4Indexer = orig_amd_indexer
            dsv4_attention.DeepseekCompressor = orig_compressor
            dsv4_attention.DeepseekV4SWACache = orig_swa_cache
        if hasattr(self.mla_attn,
                   "compressor") and self.mla_attn.compressor is not None:
            if self.compress_ratio <= 1:
                object.__setattr__(self.mla_attn.compressor, "k_cache",
                                   self.mla_attn.swa_cache_layer)
            else:
                object.__setattr__(self.mla_attn.compressor, "k_cache",
                                   self.mla_attn.mla_attn)
        hf_config = vllm_config.model_config.hf_config
        object.__setattr__(self, "attn_out_dim",
                           hf_config.num_attention_heads * hf_config.head_dim)
        # Not built here: construction is too early to read the geometry the
        # kernel specializes on. See the `attn_op` property.
        self.num_layers = vllm_config.model_config.hf_config.num_hidden_layers
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
        if self.compress_ratio <= 1:
            # SWA-only layer: its KV lives in the separately allocated
            # DeepseekV4SWACache, and `forward_mqa` never reads a main cache.
            return None

        block_size = vllm_config.cache_config.block_size
        comp_ratio = max(1, self.compress_ratio)
        spec = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=align_to(448 + 64 * 2 + 7, 128),
            dtype=torch.uint8,
            compress_ratio=min(comp_ratio, block_size),
            alignment=None,
        )
        return spec

    @property
    def attn_op(self):
        """The Pallas attention op, built on first use rather than in `__init__`.

        Building at construction time reads state that is not final yet:

        - `cache_config.block_size` is still vLLM's default 16 during model
          construction (the platform's `update_block_size_for_backend` override
          is not visible on any config object reachable from here), so the SWA
          logical page size came out 4 instead of 256.
        - the KV caches are still `torch.tensor([])` placeholders, so nothing
          about real tensor identity or geometry can be observed.

        By first forward both are settled, and the model-wrapper context is
        active (`tpu_runner.py:2713`), so `_build_attn_op` can read what it
        needs. The op is memoized per instance, and `_attn_op_cache` still
        dedupes the underlying jax op across layers.
        """
        op = self.__dict__.get("_attn_op_instance")
        if op is not None:
            return op
        # Ops built during the profiling forward see numel==0 placeholder
        # caches; they get their own memo slot and never serve real traffic.
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
            # A wrong logical page size silently misaddresses every SWA slot.
            raise RuntimeError(
                f"{self.custom_prefix}: swa_cache_layer unavailable at "
                "attention-op build time; refusing to guess the SWA cache "
                "logical page size.")
        # Safe to read directly: the op is built lazily on first forward, by
        # which point `get_kv_cache_spec` has set the final block size.
        logical_page_size = swa_cache.block_size
        if logical_page_size <= 0:
            raise RuntimeError(
                f"{self.custom_prefix}: SWA logical page size is "
                f"{logical_page_size}; expected `get_kv_cache_spec` to have "
                "set it before the first forward.")

        # Real tensor identity, so it must be read after the caches are
        # allocated -- at construction both are empty placeholders.
        main_cache = getattr(self.mla_attn, "kv_cache", None)
        swa_tensor = getattr(swa_cache, "kv_cache", None)
        two_caches_same_buffer = bool(
            not swa_only and main_cache is not None and swa_tensor is not None
            and main_cache.numel() > 0 and swa_tensor.numel() > 0
            and main_cache.data_ptr() == swa_tensor.data_ptr())

        # Placeholder caches mean provisional geometry; ops built from them
        # are throwaways (see the `attn_op` property).
        built_from_real_caches = bool(swa_tensor is not None
                                      and swa_tensor.numel() > 0)

        # A mismatch here means the op would bake a page size the runner's
        # block tables were not built for.
        cfg_lps = VllmDeepseekV4SWACache._swa_block_size(
            vllm_context.vllm_config.cache_config.block_size, self.window_size)
        if cfg_lps != logical_page_size:
            logger.warning(
                "[ATTN_BUILD] %s: swa_cache.block_size=%s disagrees with "
                "config-derived SWA page size %s "
                "(cache_config.block_size=%s, window=%s). Using the "
                "config-derived value.", self.custom_prefix, logical_page_size,
                cfg_lps, vllm_context.vllm_config.cache_config.block_size,
                self.window_size)
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
                "attention-op build time (expected head_dim**-0.5 from "
                "DeepseekV4Attention.__init__).")

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
        # Every value baked into the traced op belongs in the key; `_prof` /
        # `_p{page}` keep provisional-geometry ops out of the real registry.
        op_name = (f"pallas::deepseek_v4_attention_{op_type}_v2"
                   f"_p{logical_page_size}"
                   f"{'_aliased' if two_caches_same_buffer else ''}"
                   f"{'' if built_from_real_caches else '_prof'}")

        global _attn_op_cache
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
        # Donate `sw_cache` so XLA writes in place instead of allocating a
        # second full-size buffer; `forward_mqa` copies the result back.
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
            num_tokens = q.shape[0]
            out_tensor = torch.empty((num_tokens, self.mla_attn.n_local_heads,
                                      self.mla_attn.head_dim),
                                     dtype=q.dtype,
                                     device=q.device)
            return out_tensor, torch.empty_like(sw_cache)

        attn_jax_op.register_fake(_fake_attn)

        # `_prof` ops are cached under their own key, so the real-cache build
        # always misses them and re-derives geometry.
        _attn_op_cache[op_name] = attn_jax_op
        return attn_jax_op, built_from_real_caches

    def attn_gemm(self, hidden_states):
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
        orig_dtype = q.dtype
        qf = q.to(torch.float32)

        # Per-head RMSNorm (no weight) over the full head_dim.
        rms = torch.rsqrt(
            qf.pow(2).mean(dim=-1, keepdim=True) + self.mla_attn.eps)
        qf = qf * rms

        # DeepseekV4ScalingRotaryEmbedding rotates the trailing rotary_dim
        # channels internally, so pass the full head.
        q_rotated, _ = self.mla_attn.rotary_emb(positions, qf)
        return q_rotated.to(orig_dtype)

    def kv_rope(
        self,
        kv: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = kv.dtype
        # DeepseekV4ScalingRotaryEmbedding rotates the trailing rotary_dim
        # channels internally; unsqueeze a head dim so cos/sin broadcast.
        kv_rotated, _ = self.mla_attn.rotary_emb(
            positions,
            kv.unsqueeze(1).to(torch.float32))
        return kv_rotated.squeeze(1).to(orig_dtype)

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        kv: torch.Tensor,
        kv_score: torch.Tensor,
        indexer_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
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
        # The kernels require uint8; vLLM allocates as `kv_cache_dtype`.
        # Every candidate dtype is 1 byte, so this is a lossless bitcast.
        if cache.dtype != torch.uint8:
            return cache.view(torch.uint8)
        return cache

    def _get_active_sw_cache(self) -> torch.Tensor | None:
        # `swa_cache_layer` owns the SWA buffer, in the page geometry the SWA
        # kernel addresses.
        swa_cache_layer = getattr(self.mla_attn, "swa_cache_layer", None)
        if swa_cache_layer is None:
            raise RuntimeError(
                f"{self.custom_prefix}: swa_cache_layer unavailable; refusing "
                "to drive the SWA kernel with another cache's geometry.")
        # numel==0 placeholders are skipped like None: XLA folds an empty
        # array away and desyncs the operand list.
        cache = getattr(swa_cache_layer, "kv_cache", None)
        if cache is not None and cache.numel() > 0:
            return self._as_kernel_cache_view(cache)
        return None

    def _get_active_main_cache(self) -> torch.Tensor | None:
        # `self.mla_attn` aliases `self`, so this is the only owner. The
        # numel==0 pre-`initialize_kv_cache` placeholder is skipped like None.
        cache = getattr(self, "kv_cache", None)
        if cache is not None and cache.numel() > 0:
            return self._as_kernel_cache_view(cache)
        return None

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
        *,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
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
            # The SWA layer is an AttentionLayerBase, so it always has its own
            # entry in this dict.
            if swa_layer_name not in attn_ctx:
                raise RuntimeError(
                    f"{self.custom_prefix}: no attention metadata for SWA "
                    f"layer {swa_layer_name!r}; refusing to substitute "
                    "another layer's.")
            swa_attn_metadata = attn_ctx[swa_layer_name]
            # Only the main prefix addresses the compressed-KV buffer that
            # `main_page_indices` reads; every other buffer is disjoint.
            main_attn_metadata = attn_ctx.get(main_prefix)
        else:
            swa_attn_metadata = attn_ctx
            main_attn_metadata = attn_ctx

        orig_sw_cache = self._get_active_sw_cache()
        sw_cache = orig_sw_cache
        if sw_cache is None:
            sw_cache = torch.zeros((0, 16, 1, self.mla_attn.head_dim),
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
                # `.clone()`, not an alias: XLA dedupes identical inputs into
                # one operand and desyncs the declared 14-operand signature.
                main_cache_kv = sw_cache.clone()
            m_seq_lens = _get_field(main_attn_metadata, "seq_lens")
            # Both CSA and HCA divide by compress_ratio; skipping HCA leaves
            # main_kv_lens undivided and overruns the valid compressed rows.
            main_kv_lens = m_seq_lens // self.compress_ratio if m_seq_lens is not None else None
            main_page_indices = _get_field(main_attn_metadata, "block_tables")
            main_cu_q_lens = _get_field(main_attn_metadata, "query_start_loc")
            main_distribution = _get_field(main_attn_metadata,
                                           "request_distribution")
        else:
            # `.clone()`, not an alias: XLA dedupes identical inputs into one
            # operand and desyncs the declared 14-operand signature.
            main_cache_kv = sw_cache.clone()
            # Reuse SWA metadata rather than `None`, which is an empty pytree
            # leaf and drops out of the operand list. Cloned as above.
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
        t = o.shape[0]
        o_f = o.to(torch.float32).view(t, self.mla_attn.n_local_heads,
                                       self.mla_attn.head_dim)
        # Inverse GPT-J RoPE on the trailing rotary_dim channels of each head.
        o_ref, _ = self.mla_attn.rotary_emb(positions, o_f, inverse=True)
        o_ref = o_ref.to(torch.bfloat16)

        # Requant + TP sharding leave `wo_a.weight_scale` as a 1-D per-output-
        # channel vector matching the local weight's rows.
        w = self.wo_a.weight
        s = self.wo_a.weight_scale
        assert s.ndim == 1 and s.shape[0] == w.shape[0], (w.shape, s.shape)
        wo_a_dequant = w.to(o_ref.dtype) * s.view(-1, 1).to(o_ref.dtype)

        # Heads are chunked into groups; each group projects with its own
        # [o_lora_rank, d] block of wo_a.
        n_local_groups = self.n_local_groups
        o_lora_rank = self.o_lora_rank

        o_ref_grouped = o_ref.view(t, n_local_groups, -1)
        wo_a_grouped = wo_a_dequant.view(n_local_groups, o_lora_rank, -1)

        z = torch.einsum("tgd,grd->tgr", o_ref_grouped, wo_a_grouped)
        z_flat = z.flatten(1)

        # wo_b: RowParallelLinear back to hidden_size (returns (out, bias)).
        out = self.wo_b(z_flat)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
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


def patch_deepseek_v4_mla_cls() -> None:
    import vllm.models.deepseek_v4.amd.model as ds_v4_amd_model
    ds_v4_amd_model.DeepseekV4ROCMAiterMLAAttention = VllmDeepseekV4MLAAttention
    dsv4_attention.DeepseekV4SWACache = VllmDeepseekV4SWACache
    if hasattr(ds_v4_amd_model, "DeepseekV4SWACache"):
        ds_v4_amd_model.DeepseekV4SWACache = VllmDeepseekV4SWACache
    if not hasattr(DeepseekSparseSWABackend, "get_min_page_size"):
        DeepseekSparseSWABackend.get_min_page_size = classmethod(
            lambda cls, vllm_config: 1)
    logger.info(
        "Patched DeepseekV4ROCMAiterMLAAttention and DeepseekV4SWACache for TPU."
    )
