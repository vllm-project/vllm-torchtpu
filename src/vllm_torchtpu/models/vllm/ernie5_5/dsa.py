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
"""ERNIE 5.5 MLA-DSA attention on TPU (layers with compress_ratios[i] <= 0).

vLLM-ERNIE ``Ernie5_5DSAAttention`` (nvidia/attention.py:762-926) on the GLM-5.2
sparse-MLA path, unchanged kernels:
- keys = the DSA indexer's top-2048 of the tokens OLDER than the 128-token window, plus
  the window itself (merge_swa_topk). StreamIndex top-k returns scores, so asking it for
  top-(2048 + 128) over every causal token and dropping the window rows leaves exactly
  the top-2048 of the older tokens (an older token outranked by at most 2047 older + 128
  window tokens is inside the top-2176); the window is then appended and the row
  left-packed.
- per-head sink (``softmax_offset``): the sparse-MLA wrapper rescales by
  sigmoid(lse - sink) from the kernel's log-sum-exp (``attn_sinks`` on the attention
  layer).
- the indexer's q / k get ERNIE's orthonormal 128-point Hadamard rotation before the fp8
  quantization (scores are unchanged in exact arithmetic; it spreads outliers before
  e4m3).
- VHA on the per-head output after W_UV, then ``o * sigmoid(o_gate(q_c))`` and ``wo_b``.
"""

import copy
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.deepseek_v2 import Indexer
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_torchtpu.layers.adapter.attention import (
    PallasMLAttentionBackend,
    VllmTPUDeepseekV32IndexerBackend,
)
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op
from vllm_torchtpu.models.vllm.ernie5_5.layers import ErnieVHA


def _hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def merge_window_topk(
    idx: jax.Array, scores: jax.Array, pos: jax.Array, k: int, window: int
) -> jax.Array:
    """[T, k + window] candidates (``-1`` / ``-inf`` unfilled) over every causal token
    -> the top-k of the tokens older than the window followed by the window
    ``[pos - window + 1, pos]``, left-packed, ``-1`` tail (width k + window)."""
    older = (idx >= 0) & (idx <= (pos - window)[:, None])
    s = jnp.where(older, scores, -jnp.inf)
    top_s, top_j = jax.lax.top_k(s, k)
    sel = jnp.where(jnp.isfinite(top_s), jnp.take_along_axis(idx, top_j, axis=1), -1)
    win = pos[:, None] - (window - 1) + jnp.arange(window, dtype=pos.dtype)[None, :]
    win = jnp.where(win >= 0, win, -1)
    merged = jnp.concatenate([sel, win], axis=1)
    order = jnp.argsort((merged < 0).astype(jnp.int32), axis=1, stable=True)
    return jnp.take_along_axis(merged, order, axis=1).astype(jnp.int32)


class ErnieTPUSparseAttnIndexer(mla_op.VllmTPUSparseAttnIndexer):
    """StreamIndex top-(k + window) with scores, then the window exclusion + merge
    (merge_window_topk) in-op."""

    ernie_window: int = 128
    ernie_topk: int = 2048

    def _build_streamidx_op(self):
        k, window = self.ernie_topk, self.ernie_window
        width = k + window
        kv_cache_mode = (
            mla_op.envs.VLLM_TPU_INDEXER_KV_CACHE_MODE
            if mla_op.envs.is_indexer_kv_cache_mode_explicit()
            else None
        )
        small_bucket = (
            mla_op.envs.TPU_STREAMIDX_SMALL_BUCKET_BQ,
            mla_op.envs.TPU_STREAMIDX_SMALL_BUCKET_BKVP,
            mla_op.envs.TPU_STREAMIDX_SMALL_BUCKET_TOKENS,
        )
        # as the GLM op: None = one unchunked pass, whose int32 scores are [bucket
        # tokens, max_model_len] (8 GiB at 8192 x 262,144); a chunk keeps at most two
        # [chunk, max_model_len] buffers live
        chunk_tokens = mla_op.envs.TPU_STREAMIDX_CHUNK_TOKENS

        def _ernie_streamidx_topk_jax(
            q_bytes: jax.Array,
            weights: jax.Array,
            cache_kv: jax.Array,
            k_packed: jax.Array,
            seq_lens: jax.Array,
            page_indices: jax.Array,
            cu_q_lens: jax.Array,
            distribution: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            writer, kv_layout = mla_op._resolve_indexer_kv_cache_mode(
                cache_kv, k_packed, mode=kv_cache_mode
            )
            if writer == "xla":
                raise NotImplementedError(
                    "ERNIE DSA indexer: XLA K-cache writer not wired "
                    "(use the pallas layout)"
                )
            cache_kv = mla_op.coalesced_kv_append(
                cache_kv,
                k_packed,
                seq_lens,
                page_indices,
                cu_q_lens,
                kv_layout=kv_layout,
            )
            q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
            kv_pages_per_block, queries_per_block = mla_op._streamidx_block_sizes(
                q.shape[0], *small_bucket
            )
            idx, scores = mla_op.streamindex_topk(
                q,
                weights,
                cache_kv,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
                k=width,
                compression_ratio=1,
                num_kv_pages_per_block=kv_pages_per_block,
                num_queries_per_block=queries_per_block,
                enable_early_exit=False,
                kv_layout=kv_layout,
                chunk_tokens=chunk_tokens,
                return_scores=True,
            )
            # absolute position of every token row (padding rows get -1 and keep only -1
            # entries)
            tok = jnp.arange(q.shape[0], dtype=jnp.int32)
            seq_id = jnp.searchsorted(cu_q_lens[1:], tok, side="right").astype(
                jnp.int32
            )
            n_seq = seq_lens.shape[0]
            seq_c = jnp.minimum(seq_id, n_seq - 1)
            q_len = cu_q_lens[seq_c + 1] - cu_q_lens[seq_c]
            pos = jnp.where(
                tok < cu_q_lens[-1],
                seq_lens[seq_c] - q_len + (tok - cu_q_lens[seq_c]),
                -1,
            )
            return cache_kv, merge_window_topk(idx, scores, pos, k, window)

        # the op's body, for abstract tracing on CPU
        self.streamidx_jax_fn = _ernie_streamidx_topk_jax
        op_name = (
            f"pallas::ernie_streamidx_topk_{self.k_cache.prefix.replace('.', '_')}"
        )
        jax_op = mla_op.pallas.jax_op(
            op_name, _ernie_streamidx_topk_jax, donate_argnums=(2,)
        )

        def _fake(q_bytes, weights, cache_kv, *args, **kwargs):
            return torch.empty_like(cache_kv), torch.empty(
                (q_bytes.shape[0], width), dtype=torch.int32, device=q_bytes.device
            )

        jax_op.register_fake(_fake)

        def op(*args):
            new_cache, topk_indices = jax_op(*args)
            args[2].copy_(new_cache)
            return topk_indices

        self._streamidx_op = op
        return op


class ErnieTPUIndexer(mla_op.VllmTPUIndexer):
    """VllmTPUIndexer + ERNIE's Hadamard on q / k and the window-merged top-k rows
    (width topk + window)."""

    @classmethod
    def rebind_ernie(cls, indexer: Indexer, window: int) -> "ErnieTPUIndexer":
        ind = mla_op.VllmTPUIndexer.rebind(indexer)
        ind.__class__ = cls
        op = ind.indexer_op
        op.__class__ = ErnieTPUSparseAttnIndexer
        op.ernie_window, op.ernie_topk = window, ind.topk_tokens
        op.topk_op = (
            op._build_streamidx_op()
        )  # rebuilt with the ERNIE body (built eagerly in __init__)
        # the shared row table holds topk + window columns
        ind.topk_tokens = op.topk_tokens = ind.topk_tokens + window
        ind.register_buffer(
            "hadamard", _hadamard(ind.head_dim).to(torch.bfloat16), persistent=False
        )
        return ind

    def forward(self, hidden_states, qr, positions, rotary_emb):
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        kw, _ = self.wk_weights_proj(hidden_states)
        k = self.k_norm(kw[:, : self.head_dim])
        weights = kw[:, self.head_dim :]
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        q = torch.cat([q_pe.reshape(-1, self.n_head, self.rope_dim), q_nope], dim=-1)
        k = torch.cat([k_pe.reshape(-1, self.rope_dim), k_nope], dim=-1)
        h = self.hadamard.to(q.dtype)
        q = (q.reshape(-1, self.head_dim) @ h).to(q.dtype)
        k = (k @ h.to(k.dtype)).to(k.dtype)
        q_fp8, q_scale = mla_op.quantize_tensor(
            q, torch.float8_e4m3fn, axis=-1, use_ue8m0=False
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)
        wd = weights.dtype
        weights = (
            (weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5)
            .squeeze(-1)
            .to(wd)
        )
        return self.indexer_op(hidden_states, q_fp8, k, weights)


# ERNIE mixes page sizes (DSv4 HCA window / compressed / state caches, the DSA latent
# and indexer caches). vLLM packs mixed pages only under a block-outermost layout, so
# ERNIE runs DeepseekV4ForCausalLM's BLHNC (tpu_platform.py) and the GLM-family DSA
# caches, which declare only LBNHC, accept it here too.
class ErnieMLABackend(PallasMLAttentionBackend):
    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.BLHNC,)


class ErnieIndexerBackend(VllmTPUDeepseekV32IndexerBackend):
    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.BLHNC,)


def make_topk_buffer(vllm_config: VllmConfig) -> torch.Tensor:
    cfg = vllm_config.model_config.hf_config
    return torch.empty(
        vllm_config.scheduler_config.max_num_batched_tokens,
        cfg.index_topk + cfg.sliding_window,
        dtype=torch.int32,
    )


class ErnieDSAAttention(nn.Module):
    """MLA over the window-merged DSA top-k with a per-head sink, VHA and a sigmoid
    output gate."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        # the shared config, as deepseek_v2 passes it: the platform sets the KV block
        # size after model init, and the indexer cache's spec reads
        # cache_config.block_size (a copy froze it at 16 vs the 1024 everyone else uses)
        cache_config = vllm_config.cache_config
        tp = get_tensor_model_parallel_world_size()
        self.num_heads = config.hybrid_mla_num_attention_heads
        if self.num_heads % tp:
            raise ValueError(
                f"attention TP {tp} must divide {self.num_heads} MLA heads"
            )
        self.num_local_heads = self.num_heads // tp
        self.q_lora_rank, self.kv_lora_rank = (
            config.hybrid_mla_q_lora_rank,
            config.hybrid_mla_kv_lora_rank,
        )
        self.nope, self.rope = (
            config.hybrid_mla_qk_nope_head_dim,
            config.hybrid_mla_qk_rope_head_dim,
        )
        self.v_head_dim = config.hybrid_mla_v_head_dim
        H = config.hidden_size
        self.fused_qkv_a_proj = MergedColumnParallelLinear(
            H,
            [self.q_lora_rank, self.kv_lora_rank + self.rope],
            bias=False,
            quant_config=quant_config,
            disable_tp=True,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * (self.nope + self.rope),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.nope + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.wo_b = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            H,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )
        self.o_gate = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.v_head_dim,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.o_gate",
        )
        self.vha = ErnieVHA(self.num_heads)
        mla_rope = config.mla_rope_parameters
        self.rotary_emb = get_rope(
            self.rope,
            max_position=config.max_position_embeddings,
            is_neox_style=False,
            rope_parameters=mla_rope,
        )
        self.indexer_rope_emb = get_rope(
            self.rope,
            max_position=config.max_position_embeddings,
            is_neox_style=True,
            rope_parameters=mla_rope,
        )
        self.attn_sink = nn.Parameter(
            torch.empty(self.num_local_heads, dtype=torch.float32), requires_grad=False
        )
        if topk_indices_buffer is None:
            topk_indices_buffer = make_topk_buffer(vllm_config)
        # The HCA layers' DeepSeek-V4 attention rewrites the shared cache_dtype "fp8" to
        # "fp8_ds_mla", its uint8 packed format; this GLM-5.2 sparse-MLA latent is plain
        # fp8 e4m3 (update_sparse_mla_kv_cache asserts it). MLAAttention reads
        # cache_config only in __init__; the indexer keeps the shared one (block size).
        mla_cache_config = cache_config
        if cache_config.cache_dtype == "fp8_ds_mla":
            mla_cache_config = copy.copy(cache_config)
            mla_cache_config.cache_dtype = "fp8"
        # deepseek_v2.Indexer reads index_* and qk_rope_head_dim (64 for both ERNIE
        # attention kinds)
        icfg = SimpleNamespace(
            index_topk=config.index_topk,
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            qk_rope_head_dim=self.rope,
        )
        indexer = Indexer(
            vllm_config,
            icfg,
            H,
            self.q_lora_rank,
            quant_config,
            cache_config,
            topk_indices_buffer,
            prefix=f"{prefix}.indexer",
        )
        self.indexer = ErnieTPUIndexer.rebind_ernie(indexer, config.sliding_window)
        self.topk_indices_buffer = topk_indices_buffer
        self.mla_attn = mla_op.VllmTPUMLAAttention(
            num_heads=self.num_local_heads,
            scale=(self.nope + self.rope) ** -0.5,
            qk_nope_head_dim=self.nope,
            qk_rope_head_dim=self.rope,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            kv_b_proj=self.kv_b_proj,
            cache_config=mla_cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_sparse=True,
            indexer=self.indexer,
            topk_indices_buffer=topk_indices_buffer,
        )
        # read by PallasMLAttentionBackendImpl._build_sparse_mla_op; not a second
        # registration of the parameter
        object.__setattr__(self.mla_attn, "attn_sinks", self.attn_sink)
        # same impls, BLHNC declared (see ErnieMLABackend)
        self.mla_attn.attn_backend = ErnieMLABackend
        self.indexer.k_cache.get_attn_backend = lambda: ErnieIndexerBackend

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if positions.ndim > 1:
            positions = positions.flatten()
        q_c, kv_lora = self.fused_qkv_a_proj(hidden_states)[0].split(
            [self.q_lora_rank, self.kv_lora_rank + self.rope], dim=-1
        )
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.nope + self.rope)
        q_nope, q_pe = q.split([self.nope, self.rope], dim=-1)
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.rope], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)
        q_dtype, k_dtype = q_pe.dtype, k_pe.dtype
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        q_pe, k_pe = q_pe.to(q_dtype), k_pe.to(k_dtype)
        topk = self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)[
            : hidden_states.shape[0]
        ]
        n = hidden_states.shape[0]
        o = self.mla_attn(
            (q_nope, q_pe),
            kv_c_normed,
            k_pe,
            output_shape=(n, self.num_local_heads * self.v_head_dim),
            topk_indices=topk,
        )
        o = self.vha(o.view(n, self.num_local_heads, self.v_head_dim)).reshape(n, -1)
        return self.wo_b(o * torch.sigmoid(self.o_gate(q_c)))
