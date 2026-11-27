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
"""MiniMax Sparse Attention layer on TPU."""

import functools
from collections.abc import Callable

import torch
from torch import nn
from transformers import PreTrainedConfig
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MinimaxM3QKVParallelLinearWithIndexer,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_torchtpu.layers.adapter.attention import PallasMLAttentionBackend
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.owned_kv_cache import TpuOwnedKVCache
from vllm_torchtpu.models.minimax_m3 import dcp_local_kv, sp_o_proj
from vllm_torchtpu.models.minimax_m3.kernels import msa_fp4
from vllm_torchtpu.models.minimax_m3.ops.msa import (
    fp4_dcp_rotates_queries,
    get_msa_dcp_op,
    get_msa_op,
    get_msa_sharded_op,
)
from vllm_torchtpu.models.minimax_m3.ops.qk_norm_rope import get_qk_norm_rope_op
from vllm_torchtpu.models.minimax_m3.step_halves import split_metadata


class MiniMaxM3Cache(nn.Module, TpuOwnedKVCache):
    """One row per token in the packed MLA page layout.

    Holds ``[K | V]`` rows for the attention cache and index keys for the
    indexer cache; only the MSA kernels read it, so no attention backend's
    layout is involved. With ``token_shards > 1`` a page holds only every
    ``token_shards``-th token of its block (the index-cache shard of one
    tensor-parallel rank), packed as ``[pages, S / shards / p, p, W]``.
    Under DCP a page holds whole blocks: vLLM's logical block ``j`` of
    ``dcp * S`` tokens is block ``j * dcp + r`` on DCP rank ``r``
    (`kernels.msa_dcp`); the spec is per rank and DCP-sharded, so the
    scheduler counts ``dcp`` times the tokens per page. A DCP group wider
    than a KV head's replicas stores all of its ``G`` KV heads per row,
    ``[K_0 | V_0 | ... | K_G-1 | V_G-1]``.
    """

    def __init__(
        self,
        row_width: int,
        dtype: torch.dtype,
        prefix: str,
        token_shards: int = 1,
        scales_in_page: bool = False,
    ) -> None:
        super().__init__()
        self.row_width = row_width
        # 4-bit pages that carry their per-slot f32 scales in one extra word
        # row (`kernels.msa_fp4.scales_in_page`); needs row_width == block size.
        self.scales_in_page = scales_in_page and dtype == torch.uint8
        self.dtype = dtype
        self.prefix = prefix
        self.token_shards = token_shards
        self.kv_cache = torch.tensor([])
        vllm_config = get_current_vllm_config()
        self.block_size = vllm_config.cache_config.block_size
        if self.block_size % token_shards:
            raise ValueError(
                f"block size {self.block_size} is not a multiple of the "
                f"{token_shards} index-cache shards"
            )
        static_forward_context = vllm_config.compilation_config
        static_forward_context = static_forward_context.static_forward_context
        if prefix in static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # MLAAttentionSpec budgets one row of row_width per token; a shard
        # stores 1 / token_shards of the rows, budgeted as a narrower row.
        # A 4-bit (uint8) row takes row_width / 2 bytes.
        width = self.row_width // 2 if self.dtype == torch.uint8 else self.row_width
        if self.scales_in_page:  # + one f32 scale per token
            width += 4
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=width // self.token_shards,
            dtype=self.dtype,
        )

    def kv_cache_shape(self, num_blocks: int) -> tuple[int, ...] | None:
        tokens = self.block_size // self.token_shards
        if self.dtype == torch.uint8:
            # 4-bit: 8 tokens per 32-bit word of 4 bytes (`kernels.msa_fp4`),
            # plus the scale row when the page carries its scales.
            if self.scales_in_page and self.row_width != tokens:
                raise NotImplementedError(
                    f"folded scales need a {tokens}-wide row, got {self.row_width}"
                )
            return (num_blocks, tokens // 8 + self.scales_in_page, 4, self.row_width)
        if self.token_shards == 1:
            return None
        packing = 32 // (8 * self.dtype.itemsize)
        return (num_blocks, tokens // packing, packing, self.row_width)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return PallasMLAttentionBackend


class MiniMaxM3ScaleCache(MiniMaxM3Cache):
    """Scale pages of a layer's 4-bit caches: ``[pages, R, 128]`` f32, one
    page per KV block, lanes = the block's tokens (`kernels.msa_fp4`)."""

    def __init__(self, kv_heads: int, prefix: str) -> None:
        super().__init__(msa_fp4.scale_rows(kv_heads), torch.float32, prefix)
        if self.block_size != msa_fp4.SCALE_LANES:
            raise NotImplementedError(
                f"4-bit MSA caches need block size {msa_fp4.SCALE_LANES}, "
                f"got {self.block_size}."
            )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # R f32 per token = one [R, 128] page per block.
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.row_width,
            dtype=torch.float32,
        )

    def kv_cache_shape(self, num_blocks: int) -> tuple[int, ...] | None:
        return (num_blocks, self.row_width, msa_fp4.SCALE_LANES)


# --additional-config key: shard the MSA index-K cache over the TP group
# (`kernels.msa` index-cache sharding). Part of the compile-cache key.
SHARD_INDEX_CACHE_KEY = "minimax_shard_index_cache"


def shard_index_cache_enabled(vllm_config: VllmConfig) -> bool:
    config = vllm_config.additional_config
    if not isinstance(config, dict):
        return False
    enabled = config.get(SHARD_INDEX_CACHE_KEY, False)
    if not isinstance(enabled, bool):
        raise ValueError(f"additional_config.{SHARD_INDEX_CACHE_KEY} must be a bool")
    return enabled


@functools.cache
def _dcp_rank_tensor(rank: int, device: str) -> torch.Tensor:
    """One ``int32[1]`` rank tensor shared by every MSA layer's buffer.

    Dynamo lifts a tensor shared by several modules as one graph input, so
    XLA can CSE the rank-dependent step metadata (KV-write runs, indexer work
    list) across layers instead of recomputing it in every layer. The sharing
    holds because the model is built on the target device: a later
    ``Module.to`` would give each layer its own copy again.
    """
    return torch.tensor([rank], dtype=torch.int32, device=device)


@functools.cache
def _hadamard_tensor(n: int, device: str) -> torch.Tensor:
    """The bf16 Sylvester Hadamard matrix of `msa_fp4`, shared by every MSA
    layer as `_dcp_rank_tensor` is (a module buffer: built at load time,
    outside the traced forward)."""
    return torch.tensor(msa_fp4.hadamard(n), dtype=torch.bfloat16, device=device)


def cache_dtype(cache_config: CacheConfig | None) -> torch.dtype:
    """Storage dtype of both MSA caches for ``--kv-cache-dtype``."""
    name = cache_config.cache_dtype if cache_config is not None else "auto"
    if name in ("auto", "bfloat16"):
        return torch.bfloat16
    if name in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    if name == "int4_per_token_head":
        # 4-bit caches with per-token-head scales (`kernels.msa_fp4`).
        return torch.uint8
    raise NotImplementedError(
        f"MiniMax M3 on TPU supports --kv-cache-dtype auto, bfloat16, fp8 or "
        f"int4_per_token_head, got {name!r}."
    )


def norm_rope_heads(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    norms: tuple[GemmaRMSNorm, GemmaRMSNorm, GemmaRMSNorm, GemmaRMSNorm],
    rotary_emb: nn.Module,
    op: Callable[..., tuple[torch.Tensor, ...]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``[q | k | v | index_q | index_k]`` and norm + rotate all but v
    in one kernel, `op` (`ops.qk_norm_rope.get_qk_norm_rope_op` for these
    head counts; it also fixes q's layout).

    The four Gemma RMSNorms differ only in their weights and RoPE acts on each
    head alone, so the kernel norms every head with its row of a per-head
    ``1 + w`` table, then rotates it, with the numerics of the TPU
    ``rms_norm`` provider and the FP32 rotary embedding.
    """
    if not rotary_emb.is_neox_style or rotary_emb.head_size != head_dim:
        raise NotImplementedError("MiniMax M3 rotates NeoX-style within each head.")
    q_norm, k_norm, index_q_norm, index_k_norm = norms
    weight = (
        torch.cat(
            [
                q_norm.weight.expand(num_heads, head_dim),
                k_norm.weight.expand(num_kv_heads, head_dim),
                index_q_norm.weight.expand(num_kv_heads, head_dim),
                index_k_norm.weight.expand(1, head_dim),
            ]
        ).float()
        + 1.0
    )
    assert all(n.variance_epsilon == q_norm.variance_epsilon for n in norms)
    cos_sin = rotary_emb._match_cos_sin_cache_dtype(qkv)
    cos_sin = cos_sin.index_select(0, positions.flatten())
    return op(qkv, cos_sin, weight)


class StepHalves:
    """The two token halves of a step (TPU_MINIMAX_M3_SP_HALVES): their
    positions and the attention metadata of their queries, derived from the
    step's once per forward (half None: the step's own).

    Every trace of the model derives them first thing, so the traces of all
    token buckets read their inputs in one order, as
    compilation/shape_variants needs to hand a bucket between traces. The
    second half attends after the first (whose keys it reads from the cache).
    """

    def __init__(self, positions: torch.Tensor, num_tokens: int) -> None:
        # positions' length is its own symbol: tie it to the token count the
        # halves split, or narrowing it specializes the trace.
        torch._check(positions.shape[0] == num_tokens)
        self.split = num_tokens // 2
        self.num_tokens = num_tokens
        self.positions = (self.take(positions, 0), self.take(positions, 1))
        self.whole_positions = positions
        self._metadata: dict[int, tuple[AttentionMetadata, ...]] = {}
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, dict):
            for md in metadata.values():
                if isinstance(md, AttentionMetadata) and id(md) not in self._metadata:
                    self._metadata[id(md)] = split_metadata(md, self.split)

    def take(self, x: torch.Tensor, half: int) -> torch.Tensor:
        """Half ``half``'s rows of step rows ``x`` (narrow: a symbolic slice
        would specialize the token count)."""
        if half == 0:
            return x.narrow(0, 0, self.split)
        return x.narrow(0, self.split, self.num_tokens - self.split)

    def positions_of(self, half: int | None) -> torch.Tensor:
        return self.whole_positions if half is None else self.positions[half]

    def metadata(self, md: AttentionMetadata, half: int | None) -> AttentionMetadata:
        """Half ``half``'s metadata (None: the whole step's)."""
        return self._metadata[id(md)][2 if half is None else half]


def _zero_of(dep: torch.Tensor) -> torch.Tensor:
    """An exact f32 zero [1, 1] computed from ``dep`` (a data dependence)."""
    x = dep[:1, :1].float()
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) * 0


def _step_rows(pieces: list[torch.Tensor]) -> torch.Tensor:
    """Rows of the all-gathered pieces (piece i: every rank's i-th row block
    of its shard) back in step order."""
    tp = get_tensor_model_parallel_world_size()
    x = torch.stack(pieces)
    n, rows = x.shape[:2]
    return x.view(n, tp, rows // tp, -1).transpose(0, 1).reshape(n * rows, -1)


class MiniMaxM3SparseAttention(nn.Module):
    """GQA attention over the KV blocks the lightning indexer selects.

    One fused GEMM emits ``[q | k | v | index_q | index_k]``. All four
    projections get a per-head Gemma RMSNorm and partial NeoX RoPE. K/V rows
    live in `kv_cache`, index keys in `index_cache`.
    """

    def __init__(
        self,
        config: PreTrainedConfig,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        sparse_cfg = config.sparse_attention_config
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.head_dim = config.head_dim
        self.index_head_dim = sparse_cfg["sparse_index_dim"]
        assert sparse_cfg["sparse_num_index_heads"] == config.num_key_value_heads
        assert self.index_head_dim == self.head_dim
        assert sparse_cfg["sparse_score_type"] == "max", sparse_cfg
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        scaling = self.head_dim**-0.5

        self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            config.num_key_value_heads,
            sparse_cfg["sparse_num_index_heads"],
            self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        sp_o_proj.mark(self.o_proj)
        eps = config.rms_norm_eps
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=eps)
        self.index_q_norm = GemmaRMSNorm(self.index_head_dim, eps=eps)
        self.index_k_norm = GemmaRMSNorm(self.index_head_dim, eps=eps)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters={
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": config.partial_rotary_factor,
            },
        )
        # With one index head per rank, each rank can keep a 1 / tp share of
        # the index keys (`kernels.msa` index-cache sharding).
        shard = shard_index_cache_enabled(get_current_vllm_config())
        self.index_shards = tp_size if shard else 1
        if self.index_shards > 1 and tp_size != config.num_key_value_heads:
            raise NotImplementedError(
                f"additional_config.{SHARD_INDEX_CACHE_KEY} needs one index head "
                f"per rank (tensor parallel size {config.num_key_value_heads}), "
                f"got {tp_size}."
            )
        # Decode context parallelism: the ranks sharing this KV head split its
        # context by whole sparse blocks (`kernels.msa_dcp`).
        vllm_config = get_current_vllm_config()
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        if self.dcp_size > 1 and self.index_shards > 1:
            raise NotImplementedError(
                f"additional_config.{SHARD_INDEX_CACHE_KEY} does not combine with "
                "--decode-context-parallel-size."
            )
        # TP ranks per KV head; a DCP group wider than that spans several KV
        # heads, and each rank stores all of their rows for its blocks.
        self.kv_head_replicas = max(1, tp_size // config.num_key_value_heads)
        self.dcp_kv_heads = max(1, self.dcp_size // self.kv_head_replicas)
        # Group KV heads decode-sized steps project themselves (dcp_local_kv;
        # 1: none, they gather them).
        self.local_kv_heads = 1
        if self.dcp_kv_heads > 1 and dcp_local_kv.enabled():
            self.local_kv_heads = self.dcp_kv_heads
            dcp_local_kv.mark(
                self.qkv_proj,
                self.num_heads,
                self.head_dim,
                self.dcp_size,
                self.kv_head_replicas,
            )
        dtype = cache_dtype(cache_config)
        self.kv_cache = MiniMaxM3Cache(
            2 * self.head_dim * self.dcp_kv_heads, dtype, prefix=f"{prefix}.attn"
        )
        self.index_cache = MiniMaxM3Cache(
            self.index_head_dim,
            dtype,
            prefix=f"{prefix}.attn.index_cache",
            token_shards=self.index_shards,
            # One DMA per page in the indexer instead of a page plus a scale page.
            scales_in_page=True,
        )
        self.fp4 = dtype == torch.uint8
        # Cached keys' norm over the exact ones' (`score_bound`): e4m3 at most
        # 1/16; int4 rows of a Hadamard-rotated vector carry ~12 % rounding,
        # taken as 1.25 (a heuristic margin: the worst case, every other
        # element half a step off, is 1.476).
        self.key_rounding = 1.25 if self.fp4 else 1.0625 if dtype.itemsize == 1 else 1.0
        self.scale_cache = None
        if self.fp4:
            if self.index_shards > 1:
                raise NotImplementedError(
                    "--kv-cache-dtype int4_per_token_head needs an unsharded "
                    f"index cache; got {self.index_shards} index shards."
                )
            self.scale_cache = MiniMaxM3ScaleCache(
                self.dcp_kv_heads, prefix=f"{prefix}.attn.scale_cache"
            )
        msa_config = (
            scaling,
            sparse_cfg["sparse_topk_blocks"],
            sparse_cfg.get("sparse_init_block", 0),
            sparse_cfg.get("sparse_local_block", 0),
        )
        if self.dcp_size > 1:
            from vllm_torchtpu.distributed.dcp import get_dcp_rank, get_dcp_world_size

            if get_dcp_world_size() != self.dcp_size:
                raise RuntimeError(
                    f"DCP group of {get_dcp_world_size()} ranks, expected "
                    f"{self.dcp_size}."
                )
            spec = vllm_config.speculative_config
            self.msa_op = get_msa_dcp_op(
                *msa_config,
                self.dcp_size,
                self.kv_head_replicas,
                fp4=self.fp4,
                decode_rows=1 + (spec.num_speculative_tokens if spec else 0),
            )
            # A buffer, not a Python int, so the rank is a graph input and
            # every DCP rank compiles the same program (see get_msa_dcp_op).
            self.register_buffer(
                "dcp_rank",
                _dcp_rank_tensor(get_dcp_rank(), str(torch.get_default_device())),
                persistent=False,
            )
            if self.fp4:  # each rank (un)rotates its own query heads (get_msa_dcp_op)
                self.register_buffer(
                    "hadamard",
                    _hadamard_tensor(self.head_dim, str(torch.get_default_device())),
                    persistent=False,
                )
        elif self.index_shards > 1:
            self.msa_op = get_msa_sharded_op(
                *msa_config, get_tensor_model_parallel_rank(), self.index_shards
            )
        else:
            self.msa_op = get_msa_op(*msa_config, fp4=self.fp4)
        # A DCP group over one KV head reads q head-major (`get_msa_dcp_op`).
        self.q_head_major = self.dcp_size > 1 and self.dcp_kv_heads == 1
        self.norm_rope_op = get_qk_norm_rope_op(
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            eps,
            self.q_head_major,
            rotate=self.fp4
            and self.dcp_size > 1
            and fp4_dcp_rotates_queries(self.dcp_size, self.kv_head_replicas),
        )
        if self.local_kv_heads > 1:
            # Several KV heads per group: q is token-major, nothing rotates.
            self.local_norm_rope_op = get_qk_norm_rope_op(
                self.num_heads, self.local_kv_heads, self.head_dim, eps, False
            )

    def score_bound(self) -> float:
        """``|q . k| * head_dim**-0.5`` bound from the loaded norms: per-head
        RMS norms (weight ``1 + w``; RoPE and the int4 rotation keep norms)
        give ``sqrt(D) * max|1 + w_q| * max|1 + w_k|`` for exact keys, times
        the cache's key rounding (`key_rounding`, a margin for int4)."""
        q_gain = (1.0 + self.q_norm.weight.float()).abs().max().item()
        k_gain = (1.0 + self.k_norm.weight.float()).abs().max().item()
        return self.head_dim**0.5 * q_gain * k_gain * self.key_rounding

    def _qkv(
        self, hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        e4m3_apply = getattr(self.qkv_proj.quant_method, "apply_e4m3_rows", None)
        if isinstance(hidden_states, tuple) and e4m3_apply is None:
            q8, row_scale = hidden_states
            hidden_states = (q8.float() * row_scale).to(torch.bfloat16)
        if isinstance(hidden_states, tuple):
            # qkv_proj is bias-free and keeps the linear's forward = apply.
            return e4m3_apply(self.qkv_proj, *hidden_states)
        qkv, _ = self.qkv_proj(hidden_states)
        return qkv

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list | None,
        project: bool = True,
        qkv: torch.Tensor | None = None,
        half: tuple[StepHalves, int | None] | None = None,
        after: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``hidden_states`` may come as e4m3 rows with per-row f32 scales
        (an SP step's e4m3 all-gather), which qkv_proj contracts as e4m3, or
        as a list of such all-gathered row pieces (model.py `_ffn_pieces`).
        ``project=False`` returns this rank's heads before o_proj. ``qkv``
        (with hidden_states None) is qkv_proj's output, already computed.
        ``half`` (step, i): the rows are token half i of the step (i None: all
        of them), attended with the step's derived metadata (StepHalves)."""
        if half is not None:
            positions = half[0].positions_of(half[1])
        # dcp_local_kv: decode-sized steps of bf16 rows or e4m3 row tuples also
        # project the group's other KV heads; larger steps, gathered row pieces
        # and a qkv precomputed by residual_ar gather them.
        local = False
        if self.local_kv_heads > 1:
            order = dcp_local_kv.group_order(
                self.qkv_proj, self.dcp_rank, self.local_kv_heads, self.kv_head_replicas
            )
            local = (
                qkv is None
                and isinstance(hidden_states, (torch.Tensor, tuple))
                and positions.shape[0] < dcp_local_kv.MAX_TOKENS
            )
        if qkv is None and isinstance(hidden_states, list):
            # An SP step's gathered row pieces: project each as it lands.
            qkv = _step_rows([self._qkv(p) for p in hidden_states])
        elif qkv is None:
            qkv = self._qkv(hidden_states)
        if after is not None:
            # The caches are written in place: this call's writes must follow
            # the reads of the call that produced `after` (else XLA copies
            # the caches).
            qkv = qkv + _zero_of(after).to(qkv.dtype)
        kv_heads, norm_rope_op = self.num_kv_heads, self.norm_rope_op
        if local:
            kv_heads, norm_rope_op = self.local_kv_heads, self.local_norm_rope_op
            qkv = dcp_local_kv.group_heads(
                qkv,
                dcp_local_kv.project(self.qkv_proj, hidden_states),
                self.num_heads,
                self.head_dim,
                kv_heads,
                order,
            )
        q, k, v, index_q, index_k = norm_rope_heads(
            qkv,
            positions,
            self.num_heads,
            kv_heads,
            self.head_dim,
            (self.q_norm, self.k_norm, self.index_q_norm, self.index_k_norm),
            self.rotary_emb,
            norm_rope_op,
        )
        num_tokens = qkv.shape[0]
        if not self.q_head_major:
            q = q.view(num_tokens, self.num_heads, self.head_dim)
        metadata = get_forward_context().attn_metadata
        kv_cache = self.kv_cache.kv_cache
        if not isinstance(metadata, dict) or kv_cache.numel() == 0:
            # Memory profiling runs before the caches are bound.
            attn_output = (q.transpose(0, 1) if self.q_head_major else q).reshape(
                num_tokens, self.q_size
            )
        else:
            layer_metadata = metadata[self.kv_cache.prefix]
            assert isinstance(layer_metadata, AttentionMetadata)
            index_q = index_q.view(num_tokens, kv_heads, self.index_head_dim)
            if self.index_shards > 1:
                # Every rank scores all index heads on its share of the keys.
                index_q = get_tp_group().all_gather(index_q.contiguous(), dim=1)
            extra = {"dcp_rank": self.dcp_rank} if self.dcp_size > 1 else {}
            if self.scale_cache is not None:
                extra["scale_cache"] = self.scale_cache.kv_cache
                if self.dcp_size > 1:
                    extra["hadamard"] = self.hadamard
            if half is not None:
                layer_metadata = half[0].metadata(layer_metadata, half[1])
            attn_output = self.msa_op(
                q,
                k.view(num_tokens, kv_heads, self.head_dim),
                v.view(num_tokens, kv_heads, self.head_dim),
                index_q,
                index_k,
                kv_cache,
                self.index_cache.kv_cache,
                layer_metadata,
                **extra,
            ).reshape(num_tokens, self.q_size)
        if not project:
            return attn_output
        output, _ = self.o_proj(attn_output)
        return output
