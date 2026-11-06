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
"""DeepSeek-V4.1 sparse MLA + sliding-window attention layer for TPU."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers import linear as vllm_lin
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.common.ops.sequence_parallel import sp_all_gather
from vllm.models.deepseek_v41 import attention as upstream
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_torchtpu.layers.adapter.custom_ops import deepseek_v41 as ops
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.models.vllm.deepseek_v4 import attention as v4_attn

from .cache import ROPE_WORDS, ROW_WORDS, NativeCacheView, PagedCacheRows, write_cache
from .compressor import (
    AttentionBackend,
    CompressorStateCache,
    DeepseekCompressor,
    save_compress_norm,
)
from .indexer import (
    INDEX_HEAD_DIM,
    DeepseekV41Indexer,
    IndexerCache,
    TpuSparseAttnIndexer,
    publish_candidate_blocks,
)
from .metadata import (
    active_cache,
    block_table,
    cache_slots,
    layer_meta,
    not_replayed,
    replay_starts,
    rotate_rope,
    token_requests,
)

PACKED_ROW_BYTES: int = 584
KV_PAGE_ALIGNMENT: int = 512
_ROPE_TOKENS_PER_ROW: int = ROW_WORDS // ROPE_WORDS
_EMPTY_SPARSE_TOPK: int = 128
_EMPTY_SPARSE_ROPE_ROWS: int = _EMPTY_SPARSE_TOPK // _ROPE_TOKENS_PER_ROW
_DEFAULT_SWA_BLOCK_SIZE: int = 64
_SUPPORTED_COMPRESS_RATIOS: tuple[int, ...] = (0, 1, 2)
_NUM_LN_EVENTS: int = 4


# pyrefly: ignore[inconsistent-inheritance]
class SWACache(NativeCacheView, DeepseekV4SWACache):
    """Unpacked ``bfloat16`` sliding-window KV cache backed by ``NativeCacheView``.

    Data lifecycle:
        - Created: ``SlidingWindowMLASpec`` descriptor in ``get_kv_cache_spec``
          (trailing block-drafter SWA retention is applied canonically in
          ``kv_cache_utils.get_kv_cache_groups``).
        - Changed: ``self.kv_cache`` / ``self.paged_cache_rows`` via
          ``NativeCacheView.bind_kv_cache``.
    """

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        block_size: int = _DEFAULT_SWA_BLOCK_SIZE,
    ) -> None:
        """Initialize the sliding-window KV cache layer.

        Args:
            head_dim: Attention head dimension.
            window_size: Sliding-window token capacity.
            dtype: Storage data type (must be ``torch.bfloat16``).
            prefix: Layer parameter prefix.
            cache_config: vLLM KV cache configuration.
            block_size: KV cache page block size in tokens.

        Raises:
            ValueError: If ``dtype`` is not ``torch.bfloat16``.
        """
        if dtype != torch.bfloat16:
            raise ValueError(f"SWACache requires dtype=torch.bfloat16, got {dtype}")
        DeepseekV4SWACache.__init__(
            self,
            head_dim=head_dim,
            window_size=window_size,
            dtype=dtype,
            prefix=prefix,
            cache_config=cache_config,
            backend_cls=AttentionBackend,
            block_size=block_size,
            bounded_replay=cache_config.swa_bounded_replay,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """Return an unpacked ``bfloat16`` sliding-window MLA KV cache spec.

        Args:
            vllm_config: Global vLLM engine configuration.

        Returns:
            Unpadded ``bfloat16`` ``MLAAttentionSpec`` instance.
        """
        spec = cast(MLAAttentionSpec, super().get_kv_cache_spec(vllm_config))
        return replace(spec, dtype=torch.bfloat16, page_size_padded=None)

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Return the TPU sparse MLA attention backend class.

        Returns:
            The ``AttentionBackend`` class.
        """
        return AttentionBackend


# pyrefly: ignore[inconsistent-inheritance]
class DeepseekV41Attention(NativeCacheView, upstream.DeepseekV4Attention):
    """DeepSeek-V4.1 sparse MLA + sliding-window attention layer for TPU.

    Orchestrates local unpacked ``bfloat16`` SWA reads/writes, optional 1x/2x
    ``DeepseekCompressor`` updates into packed FP8 + BF16-RoPE ``PagedCacheRows``,
    optional ``DeepseekV41Indexer`` top-K selection, and fused
    ``dsv41_sparse_mla_swa_attention``.

    Stage flags toggled by parent ``DeepseekV41ForCausalLM`` / ``DSpark`` models:
        - ``ced_global_from_encoder``: True on decoder layers whose global
          compressed KV and indexer K were already published by the encoder.
        - ``ced_decoder``: True on CED decoder layers so ``replay_starts``
          honors decoder replay lower bounds alongside ``window_starts``.
        - ``non_causal_multi_token_decode`` / ``dspark_block_size``: Configured
          on DSpark draft layers for non-causal block-parallel SWA splicing
          without mutating committed target SWA or compressed KV pools.
    """

    backend_cls = AttentionBackend
    swa_backend_cls = AttentionBackend
    use_fp8_ds_mla_layout = False
    # pyrefly: ignore[bad-override-mutable-attribute]
    kv_rope_op = cast(
        Callable[[Tensor, Tensor, Tensor], Tensor],
        v4_attn.VllmDeepseekV4MLAAttention.kv_rope_op,
    )
    o_proj_op = v4_attn.VllmDeepseekV4MLAAttention.o_proj_op
    # pyrefly: ignore[bad-override-param-name]
    _o_proj = v4_attn.VllmDeepseekV4MLAAttention._o_proj
    _assert_o_proj_supported = (
        v4_attn.VllmDeepseekV4MLAAttention._assert_o_proj_supported
    )
    process_weights_after_loading = (
        v4_attn.VllmDeepseekV4MLAAttention.process_weights_after_loading
    )

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        """Return unpadded local query head count on TPU.

        Args:
            num_heads: Local query head count.

        Returns:
            Unpadded local query head count.
        """
        return num_heads

    # pyrefly: ignore[bad-override]
    def _indexer_cls(self, k_cache: IndexerCache | None) -> type[DeepseekV41Indexer]:
        """Return the TPU sparse attention indexer class.

        Args:
            k_cache: Optional bound indexer key cache.

        Returns:
            The ``DeepseekV41Indexer`` class.
        """
        del k_cache
        return DeepseekV41Indexer

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: Tensor | None = None,
        aux_stream_list: Sequence[object] | None = None,
        candidate_block_buffer: Tensor | None = None,
    ) -> None:
        """Initialize the DeepSeek-V4.1 sparse MLA + SWA attention layer.

        Args:
            vllm_config: Global vLLM engine configuration.
            prefix: Layer parameter prefix.
            topk_indices_buffer: Shared per-token top-K index buffer.
            aux_stream_list: Auxiliary stream list (unused on TPU).
            candidate_block_buffer: Shared coarse candidate block buffer.

        Raises:
            ValueError: If ``vllm_config.cache_config`` is missing or
                ``compress_ratio`` is unsupported.
        """
        # TODO(upstream): vllm/models/deepseek_v41/attention.py:438,495,557,577
        # hardcode CUDA streams/events, FP8 DS-MLA layout, and DeepseekCompressor
        # / DeepseekV4SWACache classes without platform hooks.
        nn.Module.__init__(self)
        hf_config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        if cache_config is None:
            raise ValueError("DeepseekV41Attention requires vllm_config.cache_config")

        tp_size = get_tensor_model_parallel_world_size()
        layer_id = extract_layer_index(prefix)
        self.layer_id = layer_id
        self.prefix = prefix
        self.window_size: int = hf_config.sliding_window
        self.hidden_size: int = hf_config.hidden_size
        self.n_heads: int = hf_config.num_attention_heads
        self.n_local_heads: int = self.n_heads // tp_size
        self.q_lora_rank: int = hf_config.q_lora_rank
        self.o_lora_rank: int = hf_config.o_lora_rank
        self.head_dim: int = hf_config.head_dim
        self.rope_head_dim: int = hf_config.qk_rope_head_dim
        self.nope_head_dim: int = self.head_dim - self.rope_head_dim
        self.n_groups: int = hf_config.o_groups
        self.n_local_groups: int = self.n_groups // tp_size

        ratios = hf_config.compress_ratios
        compress_ratio = int(ratios[layer_id]) if layer_id < len(ratios) else 0
        if compress_ratio not in _SUPPORTED_COMPRESS_RATIOS:
            raise ValueError(f"Unsupported compress_ratio={compress_ratio}")
        self.compress_ratio = compress_ratio

        self.kv_source_layers = tuple(hf_config.kv_source_layer_ids)
        self.index_source_layers = tuple(hf_config.index_source_layer_ids)
        self.candidate_source_layer: int = hf_config.candidate_source_layer_id
        self.candidate_block_size: int = hf_config.candidate_block_size
        self.candidate_topk_blocks: int = hf_config.candidate_topk_blocks

        is_backbone_layer = layer_id < hf_config.num_hidden_layers
        self.is_kv_source = is_backbone_layer and layer_id in self.kv_source_layers
        self.is_index_source = (
            is_backbone_layer and layer_id in self.index_source_layers
        )
        kv_sources = [
            src_layer
            for src_layer in self.kv_source_layers
            if src_layer <= layer_id and compress_ratio > 0
        ]
        index_sources = [
            src_layer
            for src_layer in self.index_source_layers
            if src_layer <= layer_id and compress_ratio > 0
        ]
        self.kv_source_layer_id = max(kv_sources, default=None)
        self.index_source_layer_id = max(index_sources, default=None)

        self.eps: float = hf_config.rms_norm_eps
        self.scale: float = self.head_dim**-0.5
        self.padded_heads = self.get_padded_num_q_heads(self.n_local_heads)
        sink_init = torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32)
        self.attn_sink = nn.Parameter(sink_init, requires_grad=False)

        self.fused_wqa_wkv = vllm_lin.MergedColumnParallelLinear(
            self.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fused_wqa_wkv",
            disable_tp=True,
        )
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)

        q_out_dim = self.n_heads * self.head_dim
        o_mid_dim = self.n_groups * self.o_lora_rank
        self.wq_b = vllm_lin.ColumnParallelLinear(
            self.q_lora_rank,
            q_out_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )
        self.wo_a = vllm_lin.ColumnParallelLinear(
            q_out_dim // self.n_groups,
            o_mid_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        # pyrefly: ignore[bad-argument-type]
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = vllm_lin.RowParallelLinear(
            o_mid_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )

        rope = upstream.build_deepseek_v4_rope(
            hf_config,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            max_position_embeddings=hf_config.max_position_embeddings,
            compress_ratio=compress_ratio,
        )
        max_model_len = vllm_config.model_config.max_model_len
        if rope.cos_sin_cache.shape[0] > max_model_len:
            rope.cos_sin_cache = rope.cos_sin_cache[:max_model_len].clone()
        self.rotary_emb = rope
        self.indexer_rotary_emb = rope

        self.topk_indices_buffer = topk_indices_buffer
        self.candidate_block_buffer = candidate_block_buffer
        static_ctx = vllm_config.compilation_config.static_forward_context
        self._static_forward_context = static_ctx
        static_ctx[prefix] = self

        self.kv_cache = torch.tensor([])
        # pyrefly: ignore[bad-override-mutable-attribute]
        self.indexer: DeepseekV41Indexer | None = None
        # pyrefly: ignore[bad-override-mutable-attribute]
        self.compressor: DeepseekCompressor | None = None

        source_prefix = (
            None
            if self.kv_source_layer_id is None
            else upstream._replace_layer_index(prefix, self.kv_source_layer_id)
        )
        if self.is_index_source:
            owner_prefix = prefix if self.is_kv_source else source_prefix
            indexer_k_prefix = f"{owner_prefix}.indexer.k_cache"
            if self.is_kv_source:
                indexer_k_cache = IndexerCache(
                    head_dim=INDEX_HEAD_DIM,
                    dtype=torch.bfloat16,
                    prefix=indexer_k_prefix,
                    cache_config=cache_config,
                    compress_ratio=compress_ratio,
                )
            else:
                indexer_k_cache = cast(IndexerCache, static_ctx.get(indexer_k_prefix))
            is_candidate_source = layer_id == self.candidate_source_layer
            uses_candidates = 0 <= self.candidate_source_layer < layer_id
            active_candidate_buffer = (
                candidate_block_buffer
                if (is_candidate_source or uses_candidates)
                else None
            )
            indexer_cls = self._indexer_cls(indexer_k_cache)
            self.indexer = indexer_cls(
                vllm_config=vllm_config,
                config=hf_config,
                hidden_size=self.hidden_size,
                q_lora_rank=self.q_lora_rank,
                quant_config=quant_config,
                cache_config=cache_config,
                topk_indices_buffer=topk_indices_buffer,
                compress_ratio=compress_ratio,
                prefix=f"{prefix}.indexer",
                owns_k=self.is_kv_source,
                k_cache=indexer_k_cache,
                main_head_dim=self.head_dim,
                candidate_block_buffer=active_candidate_buffer,
                candidate_block_size=self.candidate_block_size,
                candidate_write=is_candidate_source,
            )

        # pyrefly: ignore[bad-assignment]
        self._prepare_and_attn_fn = self._prepare_and_attn
        self.aux_stream_list = cast(list[torch.cuda.Stream] | None, aux_stream_list)
        # pyrefly: ignore[bad-override-mutable-attribute]
        self.ln_events: list[object | None] = [None] * _NUM_LN_EVENTS
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = max_model_len
        self.gemm_rs = None
        self.kv_mxfp8 = False
        self.kv_cache_dtype = "fp8"
        self.kv_cache_torch_dtype = torch.float8_e4m3fn
        self.swa_bytes_per_token = PACKED_ROW_BYTES
        self.compressed_bytes_per_token = PACKED_ROW_BYTES
        self.kv_page_alignment = KV_PAGE_ALIGNMENT
        self.swa_cache_layer = SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=torch.bfloat16,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
            block_size=cache_config.block_size,
        )
        if self.is_kv_source:
            self.compressor = DeepseekCompressor(
                vllm_config=vllm_config,
                compress_ratio=compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=prefix,
            )
        self.compressed_cache_prefix = source_prefix
        self.dspark_block_size = 0
        self.ced_global_from_encoder = False
        self.ced_decoder = False
        self.non_causal_multi_token_decode = False

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """Return ``uint8`` unpadded ``MLAAttentionSpec`` on KV-source layers.

        Args:
            vllm_config: Global vLLM engine configuration.

        Returns:
            Unpadded ``uint8`` ``MLAAttentionSpec`` on KV-source layers, or
            ``None`` on non-source layers.
        """
        spec = super().get_kv_cache_spec(vllm_config)
        if spec is None:
            return None
        mla_spec = cast(MLAAttentionSpec, spec)
        return replace(mla_spec, dtype=torch.uint8, page_size_padded=None)

    def _run_parallel_input_projections(
        self, hidden_states: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """Project ``hidden_states`` into ``(qr_kv, kv_score, indexer_weights)``.

        Args:
            hidden_states: Token hidden states ``[num_tokens, hidden_size]``.

        Returns:
            Tuple ``(qr_kv, kv_score, indexer_weights)``.
        """
        qr_kv = self._fused_wqa_wkv_gemm(hidden_states)
        kv_score = (
            self.compressor.project(hidden_states)
            if self.compressor is not None
            else None
        )
        indexer_weights = (
            self.indexer.weights_proj(hidden_states)[0]
            if self.indexer is not None
            else None
        )
        return qr_kv, kv_score, indexer_weights

    def _split_qkv_and_norm(self, qr_kv: Tensor) -> tuple[Tensor, None, Tensor]:
        """Split fused ``[q_lora, kv]`` projection and apply RMSNorm to each.

        Args:
            qr_kv: Fused projection ``[num_tokens, q_lora_rank + head_dim]``.

        Returns:
            Tuple ``(qr_proj, None, kv_latent)``.
        """
        qr_proj, kv_latent = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        return self.q_norm(qr_proj), None, self.kv_norm(kv_latent)

    def project_context_kv(self, hidden_states: Tensor) -> Tensor:
        """Project ``hidden_states`` into RMS-normalized SWA KV latents.

        Args:
            hidden_states: Token hidden states ``[num_tokens, hidden_size]``.

        Returns:
            RMS-normalized SWA KV latents ``[num_tokens, head_dim]``.
        """
        _, _, kv_latent = self._split_qkv_and_norm(
            self._fused_wqa_wkv_gemm(hidden_states)
        )
        return kv_latent

    def rotate(self, inputs: Tensor, positions: Tensor) -> Tensor:
        """Apply layer rotary position embeddings to ``inputs`` at ``positions``.

        Args:
            inputs: Query or KV tensor with trailing dimension ``head_dim``.
            positions: Token positions ``int32[num_tokens]``.

        Returns:
            Tensor with rotary embeddings applied to the trailing RoPE slice.
        """
        return rotate_rope(inputs, positions, self.rotary_emb, self.kv_rope_op)

    def _swa_state(self) -> tuple[PagedCacheRows | Tensor, AttentionMetadata | None]:
        """Return the active SWA cache view and its forward-context metadata.

        Returns:
            Tuple ``(swa_cache, swa_meta)``.
        """
        swa_layer = self.swa_cache_layer
        return active_cache(swa_layer), layer_meta(swa_layer.prefix)

    def _write_swa_kv(
        self,
        kv_rotated: Tensor,
        positions: Tensor,
        after_read: bool = False,
    ) -> None:
        """Scatter trailing window tokens into ``swa_cache_layer`` in place.

        Args:
            kv_rotated: RoPE-rotated SWA KV latents ``[num_tokens, head_dim]``.
            positions: Token positions ``int32[num_tokens]``.
            after_read: Whether to order the write after prior SWA cache reads.
        """
        swa_cache, swa_meta = self._swa_state()
        if (
            swa_meta is None
            or not swa_cache.numel()
            or not cast(Tensor, swa_meta.seq_lens).numel()
        ):
            return
        slots = cache_slots(swa_cache, swa_meta, positions)
        query_ends = cast(Tensor, swa_meta.query_start_loc)[1:].to(positions.device)
        req_indices = token_requests(swa_meta, positions).clamp_max(
            query_ends.shape[0] - 1
        )
        table_capacity = block_table(swa_meta).shape[1] * swa_cache.shape[1]
        window_cap = min(self.window_size + self.dspark_block_size, table_capacity)
        token_rows = torch.arange(positions.shape[0], device=positions.device)
        keep_mask = (slots >= 0) & (token_rows >= query_ends[req_indices] - window_cap)
        write_cache(swa_cache, slots, kv_rotated, keep_mask, after_read=after_read)

    def _gather_seq_swa(self, live_kv: Tensor | None = None) -> Tensor | None:
        """Gather per-sequence ``[num_seqs, eff_window, head_dim]`` SWA history.

        Under ``non_causal_multi_token_decode``, splices ``live_kv`` draft block
        tokens directly after committed prefix tokens without mutating the SWA
        cache (preserving target KV pools across speculative steps and idle DP
        dummy forwards). Masks ``null_block=0`` and positions ``< replay_starts``
        with ``torch.where`` / ``masked_fill`` before ``bfloat16`` reads so
        unzeroed recycled bit patterns never produce ``NaN``.

        Args:
            live_kv: Optional live KV latents ``[num_tokens, head_dim]`` for
                non-causal draft splicing.

        Returns:
            Per-sequence SWA window tensor ``bfloat16[num_seqs, eff_window, head_dim]``,
            or ``None`` if the SWA cache is empty or unscheduled.
        """
        swa_cache, swa_meta = self._swa_state()
        non_causal = self.non_causal_multi_token_decode
        block_tokens = max(0, self.dspark_block_size) if non_causal else 0
        eff_window = self.window_size + block_tokens
        if (
            swa_meta is None
            or not swa_cache.numel()
            or not cast(Tensor, swa_meta.seq_lens).numel()
            or eff_window <= 0
        ):
            return None

        seq_lens = cast(Tensor, swa_meta.seq_lens).to(torch.int64)
        cu_query = cast(Tensor, swa_meta.query_start_loc).to(torch.int64)
        query_lens = cu_query[1:] - cu_query[:-1]
        live_count = 0 if live_kv is None else live_kv.shape[0]
        live_per_seq = torch.minimum(
            query_lens, (live_count - cu_query[:-1]).clamp_min(0)
        ).clamp(0, block_tokens)

        if non_causal:
            seq_end = seq_lens - 1
            committed_end = seq_lens - live_per_seq
            window_base = (seq_end - eff_window + 1).clamp_min(0)
        else:
            seq_end = seq_lens - query_lens - 1
            committed_end = seq_lens - query_lens
            window_base = seq_end - eff_window + 1

        window_offsets = torch.arange(
            eff_window, dtype=torch.int64, device=seq_lens.device
        )
        kv_positions = window_base[:, None] + window_offsets
        lower_bounds = replay_starts(
            seq_lens.shape[0], seq_lens.device, ced_decoder=self.ced_decoder
        )
        min_pos = (
            lower_bounds.to(torch.int64).clamp_min(0).unsqueeze(1)
            if lower_bounds is not None
            else 0
        )

        swa_table = block_table(swa_meta).to(torch.int64)
        clamped_pos = kv_positions.clamp_min(0)
        rows_per_page = swa_cache.shape[1]
        page_cols = (clamped_pos // rows_per_page).clamp_max(swa_table.shape[1] - 1)
        blocks = swa_table.gather(1, page_cols)
        min_block = (
            1 if isinstance(swa_cache, PagedCacheRows) and swa_cache.null_block else 0
        )
        valid_committed = (
            (kv_positions >= min_pos)
            & (kv_positions < committed_end.unsqueeze(1))
            & (blocks >= min_block)
        )
        gathered_rows = swa_cache[blocks.clamp_min(0), clamped_pos % rows_per_page]
        swa_out = gathered_rows.bfloat16().masked_fill(
            ~valid_committed.unsqueeze(-1), 0
        )

        if not (
            non_causal and live_kv is not None and block_tokens > 0 and live_count > 0
        ):
            return swa_out

        live_offset = kv_positions - committed_end.unsqueeze(1)
        in_live_block = (live_offset >= 0) & (live_offset < live_per_seq.unsqueeze(1))
        live_row_idx = (
            cu_query[:-1].unsqueeze(1) + live_offset.clamp(0, block_tokens - 1)
        ).clamp(0, live_count - 1)
        return torch.where(
            in_live_block.unsqueeze(-1), live_kv.bfloat16()[live_row_idx], swa_out
        )

    def _compress_kv(
        self,
        kv_score: Tensor,
        positions: Tensor,
        produce_indexer_k: bool = False,
    ) -> Tensor:
        """Compress ``kv_score``, write global KV cache, and optionally write K.

        Args:
            kv_score: Unnormalized KV + gate projection ``[num_tokens, 2 * head_dim]``.
            positions: Token positions ``int32[num_tokens]``.
            produce_indexer_k: Whether to also project and write indexer K.

        Returns:
            Compressed KV latents ``[num_tokens, head_dim]``.
        """
        assert self.compressor is not None
        latent = self.compressor(kv_score, positions)
        self.compressor.insert_cache(latent, positions, self.rotary_emb)
        if produce_indexer_k and self.indexer is not None and self.indexer.owns_k:
            self.indexer._produce_k(latent, positions, self.indexer_rotary_emb)
        return latent

    def write_encoder_global_kv(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        *,
        sp_sharded: bool = False,
    ) -> None:
        """Project and publish global compressed KV and indexer K from the encoder.

        Args:
            hidden_states: Encoder hidden states ``[num_tokens, hidden_size]``.
            positions: Token positions ``int32[num_tokens]``.
            sp_sharded: Whether ``hidden_states`` are sequence-parallel sharded.
        """
        if self.compressor is None:
            return
        kv_score = self.compressor.project(hidden_states)
        if sp_sharded:
            kv_score = sp_all_gather(kv_score)[: positions.shape[0]]
        self._compress_kv(kv_score, positions, produce_indexer_k=True)

    def forward_kv(self, positions: Tensor, hidden_states: Tensor) -> None:
        """Write SWA KV, compressed KV, and indexer K without running attention.

        Args:
            positions: Token positions ``int32[num_tokens]``.
            hidden_states: Token hidden states ``[num_tokens, hidden_size]``.
        """
        qr_kv, kv_score, _ = self._run_parallel_input_projections(hidden_states)
        _, _, kv_latent = self._split_qkv_and_norm(qr_kv)
        self._write_swa_kv(self.rotate(kv_latent, positions), positions)
        if self.compressor is not None and kv_score is not None:
            self._compress_kv(kv_score, positions, produce_indexer_k=True)

    # pyrefly: ignore[bad-override-param-name]
    def _prepare_and_attn(
        self,
        hidden_states: Tensor,
        qr_proj: Tensor,
        kv_latent: Tensor,
        qr_scale: Tensor | None,
        kv_score: Tensor | None,
        indexer_weights: Tensor | None,
        positions: Tensor,
        attn_output: Tensor,
    ) -> None:
        """Project Q, update caches (ordered after SWA prefix read), and run MQA.

        Args:
            hidden_states: Token hidden states ``[num_tokens, hidden_size]``.
            qr_proj: Normalized query LoRA latents ``[num_tokens, q_lora_rank]``.
            kv_latent: Normalized SWA KV latents ``[num_tokens, head_dim]``.
            qr_scale: Optional query LoRA quantization scale.
            kv_score: Optional compressor ``[kv, gate]`` projection.
            indexer_weights: Optional indexer head importance weights.
            positions: Token positions ``int32[num_tokens]``.
            attn_output: Output buffer ``[num_tokens, n_local_heads, head_dim]``
                populated in place.
        """
        q_heads = self._wq_b_proj(qr_proj, qr_scale).view(
            -1, self.n_local_heads, self.head_dim
        )
        q_rotated = self.rotate(q_heads, positions)
        kv_rotated = self.rotate(kv_latent, positions)

        if self.non_causal_multi_token_decode:
            pre_write_swa: Tensor | None = None
            compressed_latent: Tensor | None = None
        else:
            pre_write_swa = self._gather_seq_swa()
            self._write_swa_kv(
                kv_rotated, positions, after_read=pre_write_swa is not None
            )
            should_compress = (
                self.compressor is not None
                and not self.ced_global_from_encoder
                and kv_score is not None
            )
            compressed_latent = (
                self._compress_kv(kv_score, positions)
                if should_compress and kv_score is not None
                else None
            )

        if self.indexer is not None and indexer_weights is not None:
            idx_q, idx_q_scale, idx_weights = self.indexer(
                qr_proj,
                compressed_latent,
                indexer_weights,
                positions,
                self.indexer_rotary_emb,
                qr_scale,
            )
            quant_q = idx_q if idx_q_scale is None else (idx_q, idx_q_scale)
            self.indexer.indexer_op(hidden_states, quant_q, positions, idx_weights)

        self.forward_mqa(q_rotated, kv_rotated, positions, attn_output, pre_write_swa)

    def forward_mqa(
        self,
        q: Tensor,
        kv: Tensor,
        positions: Tensor,
        output: Tensor,
        pre_write_swa: Tensor | None = None,
    ) -> None:
        """Gather sparse FP8 + SWA KV and write fused attention into ``output``.

        Args:
            q: RoPE-rotated query heads ``[num_tokens, n_local_heads, head_dim]``.
            kv: RoPE-rotated SWA KV latents ``[num_tokens, head_dim]``.
            positions: Token positions ``int32[num_tokens]``.
            output: Attention output buffer ``[num_tokens, n_local_heads, head_dim]``
                written in place.
            pre_write_swa: Optional SWA window gathered before the current step's
                SWA cache write.
        """
        num_tokens = q.shape[0]
        device = q.device
        swa_cache, swa_meta = self._swa_state()
        if (
            swa_meta is None
            or not swa_cache.numel()
            or not cast(Tensor, swa_meta.seq_lens).numel()
        ):
            output.zero_()
            return

        seq_lens = cast(Tensor, swa_meta.seq_lens).int()
        cu_q_lens = cast(Tensor, swa_meta.query_start_loc).int()
        seq_swa = self._gather_seq_swa(kv) if pre_write_swa is None else pre_write_swa
        lower_bounds = replay_starts(
            seq_lens.shape[0], device, ced_decoder=self.ced_decoder
        )

        if self.non_causal_multi_token_decode:
            req_indices = (
                token_requests(swa_meta, positions)
                .clamp(0, seq_lens.shape[0] - 1)
                .long()
            )
            seq_lens = (
                (seq_lens.to(torch.int64)[req_indices] - 1).clamp_min(0).add(1).int()
            )
            cu_q_lens = torch.arange(
                num_tokens + 1, dtype=torch.int32, device=device
            ).clamp_max(cu_q_lens[-1])
            eff_window = self.window_size + self.dspark_block_size
            swa_kv = None if seq_swa is None else seq_swa[req_indices]
            swa_min_pos = None if lower_bounds is None else lower_bounds[req_indices]
        else:
            eff_window = self.window_size
            flat_swa = None if seq_swa is None else seq_swa.reshape(-1, self.head_dim)
            swa_kv = (
                None if flat_swa is None else torch.cat((flat_swa, kv.bfloat16()), 0)
            )
            swa_min_pos = lower_bounds

        comp_prefix = self.compressed_cache_prefix
        comp_layer = (
            cast(NativeCacheView | None, self._static_forward_context.get(comp_prefix))
            if comp_prefix is not None
            else None
        )
        comp_cache = active_cache(comp_layer) if comp_layer is not None else None
        comp_meta = layer_meta(comp_prefix) if comp_prefix is not None else None
        has_compressed = (
            self.compress_ratio > 0
            and comp_cache is not None
            and bool(comp_cache.numel())
        )
        if (
            has_compressed
            and comp_cache is not None
            and comp_meta is not None
            and self.topk_indices_buffer is not None
        ):
            topk_indices = self.topk_indices_buffer[:num_tokens]
            comp_bt = block_table(comp_meta).int()
            page_size = comp_cache.shape[1]
            if isinstance(comp_cache, PagedCacheRows):
                assert comp_cache.rope is not None
                kv_pool_lo = comp_cache.raw.view(-1, page_size, ROW_WORDS)
                rope_pool_lo = comp_cache.rope.view(
                    -1, page_size // _ROPE_TOKENS_PER_ROW, ROW_WORDS
                )
                kv_pool_hi = (
                    None
                    if comp_cache.raw_hi is None
                    else comp_cache.raw_hi.view(-1, page_size, ROW_WORDS)
                )
                rope_pool_hi = (
                    None
                    if comp_cache.rope_hi is None
                    else comp_cache.rope_hi.view(
                        -1, page_size // _ROPE_TOKENS_PER_ROW, ROW_WORDS
                    )
                )
                mapped_bt = comp_cache.pages(comp_bt)
                gather_bt = comp_bt if mapped_bt is None else mapped_bt
            else:
                kv_pool_lo = comp_cache[0]
                rope_pool_lo = comp_cache[1]
                kv_pool_hi = None
                rope_pool_hi = None
                gather_bt = comp_bt
            gathered_nope, gathered_rope = ops.attention_op.dsv41_gather_sparse_fp8_kv(
                kv_pool_lo,
                rope_pool_lo,
                topk_indices,
                gather_bt,
                cast(Tensor, comp_meta.query_start_loc).int(),
                page_size,
                kv_pool_hi,
                rope_pool_hi,
            )
            kv_ratio = self.compress_ratio
        else:
            topk_indices = seq_lens.new_full((num_tokens, _EMPTY_SPARSE_TOPK), -1)
            gathered_nope = topk_indices.new_zeros(
                (num_tokens, _EMPTY_SPARSE_TOPK, ROW_WORDS)
            )
            gathered_rope = topk_indices.new_zeros(
                (num_tokens, _EMPTY_SPARSE_ROPE_ROWS, ROW_WORDS)
            )
            kv_ratio = max(1, self.compress_ratio)

        attn_res = ops.attention_op.dsv41_sparse_mla_swa_attention(
            q.bfloat16(),
            gathered_nope,
            gathered_rope,
            topk_indices,
            self.attn_sink,
            seq_lens,
            cu_q_lens,
            self.scale,
            eff_window,
            swa_kv,
            kv_ratio,
            swa_min_pos,
        )
        output.copy_(attn_res.to(output.dtype))


__all__ = [
    "INDEX_HEAD_DIM",
    "KV_PAGE_ALIGNMENT",
    "PACKED_ROW_BYTES",
    "AttentionBackend",
    "CompressorStateCache",
    "DeepseekCompressor",
    "DeepseekV41Attention",
    "DeepseekV41Indexer",
    "IndexerCache",
    "SWACache",
    "TpuSparseAttnIndexer",
    "active_cache",
    "block_table",
    "cache_slots",
    "layer_meta",
    "not_replayed",
    "publish_candidate_blocks",
    "replay_starts",
    "rotate_rope",
    "save_compress_norm",
    "token_requests",
]
