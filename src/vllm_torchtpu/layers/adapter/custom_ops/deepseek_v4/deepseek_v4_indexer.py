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
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
from jax.sharding import PartitionSpec as P
from torch.distributed import _functional_collectives as funcol
from torch_tpu._internal import pallas
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4 import attention as dsv4_attention
from vllm.models.deepseek_v4.attention import DeepseekV4Indexer, DeepseekV4IndexerCache
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    config as compressor_config,
)
from vllm_torchtpu.kernels.deepseek_v4.rope import rope_quant, rope_quant_flat
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    KVLayout,
    get_kv_cache_shape,
    merge_kv_split_topk,
    streamindex_topk,
    token_window_metadata,
)
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_compressor import (
    VllmDeepseekCompressor,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    get_pallas_op_mesh,
)

if TYPE_CHECKING:
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

logger = init_logger(__name__)

_indexer_op_cache = {}

BATCH_AXIS = None


def _indexer_impl(
    q: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    compression_ratio: int,
    softmax_scale: float,
    n_head: int,
    num_kv_pages_per_block: tuple[int, int, int],
    num_queries_per_block: tuple[int, int, int],
    verify_q_len: int | None = None,
    flat_head_dim: int = 0,
    split_rank: jax.Array | None = None,
    kv_split: int = 1,
    row_window: tuple | None = None,
    row_window_tokens: int = 0,
    window_pieces: int = 1,
    group_max: bool = False,
) -> jax.Array:
    out_width = k if kv_split == 1 else 2 * k
    if cache_kv.shape[0] == 0:
        # Profiling-shape trace: the kernel is skipped. Unused inputs stay in
        # the compiled signature because torch_tpu jits with keep_unused=True.
        return jnp.zeros((q.shape[0], out_width), dtype=jnp.int32)
    if cache_kv.shape[1] != compressor_config.SAL_SUBLANE_GROUPS:
        raise ValueError(
            f"DeepSeek-V4 indexer cache {cache_kv.shape} does not match the "
            "SEQ_ALONG_LANE layout "
            f"(expected {compressor_config.SAL_SUBLANE_GROUPS} rows)."
        )

    def _indexer_local(
        q,
        positions,
        cos_sin_cache,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
    ):
        # One kernel rotates the queries and quantizes each row on the way out.
        # Note: vLLM's implementation rounds the scale factors up to the next
        # power of 2, but the plain `abs_max / dtype_max` the kernel returns is
        # sufficient here.
        if flat_head_dim:
            # TPU_DSV4_ROPE_QUANT_FLAT: ``q`` is the flat projection; the kernel
            # reads its (8, 128) tiles and returns [tokens, heads, dim].
            q_quant, q_scales = rope_quant_flat(
                q,
                positions,
                cos_sin_cache,
                num_heads=q.shape[1] // flat_head_dim,
                quant_dtype=jnp.float8_e4m3fn,
            )
        else:
            q_quant, q_scales = rope_quant(
                q, positions, cos_sin_cache, quant_dtype=jnp.float8_e4m3fn
            )

        # Fold the query quantization scales into the weights.
        weights = indexer_weights * softmax_scale * (n_head**-0.5) * q_scales

        out = streamindex_topk(
            q=q_quant,
            indexer_weights=weights,
            cache_kv=cache_kv,
            seq_lens=seq_lens,
            page_indices=page_indices.flatten(),
            cu_q_lens=cu_q_lens,
            distribution=distribution,
            k=k,
            compression_ratio=compression_ratio,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            decode_req_batch_size=4,
            chunk_tokens=(
                q.shape[0] // window_pieces
                if window_pieces > 1
                else envs.TPU_STREAMIDX_CHUNK_TOKENS
            ),
            chunk_sched_groups=envs.TPU_STREAMIDX_CHUNK_SCHED_GROUPS,
            group_max=group_max,
            group_max_min_kv=envs.TPU_DSV4_INDEXER_GROUP_MAX_MIN_KV if group_max else 0,
            kv_layout=KVLayout.SEQ_ALONG_LANE,
            uniform_q_len=verify_q_len,
            sc_flat_scores=envs.TPU_DSV4_SC_TOPK_FLAT_SCORES,
            sc_topk_coop_max_rows=envs.TPU_DSV4_SC_TOPK_COOP_MAX_ROWS,
            kv_split=kv_split,
            kv_split_rank=0 if split_rank is None else split_rank[0],
            row_window=row_window,
            row_window_tokens=row_window_tokens,
            strict_causal=envs.TPU_STREAMIDX_STRICT_CAUSAL,
        )
        return out if kv_split == 1 else jnp.concatenate(out, axis=1)

    return _indexer_local(
        q,
        positions,
        cos_sin_cache,
        indexer_weights,
        cache_kv,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
    )


# Module-level so `pallas.jax_op` can trace and register it as a torch op.
# The op signatures must hold only arrays and plain static args (no optional
# arrays), so the split variant is a separate function.
def _indexer_jax(
    q: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    compression_ratio: int,
    softmax_scale: float,
    n_head: int,
    num_kv_pages_per_block: tuple[int, int, int],
    num_queries_per_block: tuple[int, int, int],
    verify_q_len: int | None = None,
    flat_head_dim: int = 0,
) -> jax.Array:
    return _indexer_impl(
        q, positions, cos_sin_cache, indexer_weights, cache_kv, seq_lens,
        page_indices, cu_q_lens, distribution,
        k=k, compression_ratio=compression_ratio, softmax_scale=softmax_scale,
        n_head=n_head, num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block, verify_q_len=verify_q_len,
        flat_head_dim=flat_head_dim,
    )


def _indexer_split_jax(
    q: jax.Array,
    positions: jax.Array,
    cos_sin_cache: jax.Array,
    indexer_weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    split_rank: jax.Array,  # i32[1]
    *,
    k: int,
    compression_ratio: int,
    softmax_scale: float,
    n_head: int,
    num_kv_pages_per_block: tuple[int, int, int],
    num_queries_per_block: tuple[int, int, int],
    kv_split: int,
    verify_q_len: int | None = None,
    flat_head_dim: int = 0,
) -> jax.Array:
    """This rank's kv_split share: i32[T, 2k] (global columns, score bits)."""
    return _indexer_impl(
        q, positions, cos_sin_cache, indexer_weights, cache_kv, seq_lens,
        page_indices, cu_q_lens, distribution,
        k=k, compression_ratio=compression_ratio, softmax_scale=softmax_scale,
        n_head=n_head, num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block, verify_q_len=verify_q_len,
        flat_head_dim=flat_head_dim, split_rank=split_rank, kv_split=kv_split,
    )


def _indexer_token_split_jax(
    q: jax.Array,  # [T / tp, H, D] (or [T / tp, H * D] flat): this rank's rows
    positions: jax.Array,  # [T / tp]
    cos_sin_cache: jax.Array,
    indexer_weights: jax.Array,  # [T / tp, H]
    cache_kv: jax.Array,
    seq_lens: jax.Array,  # whole batch metadata
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    rank: jax.Array,  # i32[1] TP rank
    *,
    tp: int,
    k: int,
    compression_ratio: int,
    softmax_scale: float,
    n_head: int,
    num_kv_pages_per_block: tuple[int, int, int],
    num_queries_per_block: tuple[int, int, int],
    verify_q_len: int | None = None,
    flat_head_dim: int = 0,
    window_pieces: int = 1,
) -> jax.Array:
    """TPU_DSV4_INDEXER_TOKEN_SPLIT_ABOVE: the top-k of rows
    [rank * T/tp, (rank + 1) * T/tp) only, as the whole batch selects them."""
    window = q.shape[0]
    offset = (rank[0] * window).astype(jnp.int32)
    seq_w, pages_w, cu_w, dist_w = token_window_metadata(
        seq_lens, page_indices.flatten(), cu_q_lens, distribution, offset,
        window=window, compression_ratio=compression_ratio,
    )  # fmt: skip
    return _indexer_impl(
        q, positions, cos_sin_cache, indexer_weights, cache_kv, seq_w, pages_w,
        cu_w, dist_w,
        k=k, compression_ratio=compression_ratio, softmax_scale=softmax_scale,
        n_head=n_head, num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block, verify_q_len=verify_q_len,
        flat_head_dim=flat_head_dim,
        row_window=(seq_lens, cu_q_lens, distribution, offset),
        row_window_tokens=window * tp,
        window_pieces=window_pieces,
        group_max=envs.TPU_DSV4_INDEXER_GROUP_MAX,
    )  # fmt: skip


def _merge_kv_split_a2a_jax(exchanged: jax.Array, *, k: int, kv_split: int) -> jax.Array:
    """i32[T, 2k] after the all-to-all (row block r = rank r's share of this
    rank's token block) -> the unsplit top-k of this rank's block,
    i32[T / kv_split, k]."""
    return merge_kv_split_topk(exchanged.reshape(kv_split, -1, exchanged.shape[1]), k)


def tp_rank_as_data(group, device: torch.device) -> torch.Tensor:
    """This rank's index in `group` as an i32[1] device value, so that every
    rank compiles the same program (cccc1c5b7: a Python-constant rank gave
    wrong rows under torch_tpu)."""
    ids = torch.arange(group.world_size, dtype=torch.int32, device=device)
    return _all_to_all(ids, group)[:1]


def _all_to_all(x: torch.Tensor, group) -> torch.Tensor:
    """Block d of x's leading axis goes to rank d (as cp_mla_attention's: wait
    only on an AsyncCollectiveTensor; a second wait_tensor fails compile)."""
    out = funcol.all_to_all_single(x.contiguous(), None, None, group.device_group)
    if isinstance(out, funcol.AsyncCollectiveTensor):
        return funcol.wait_tensor(out)
    return out


class VllmDeepseekV4IndexerCache(DeepseekV4IndexerCache):
    """Indexer K cache as a packed uint8 record on TPU."""

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = self.cache_config.block_size
        # The per-token record does not depend on which page size is used.
        _, sublane_groups, packing, _ = get_kv_cache_shape(
            1,
            compressor_config.SAL_PAGE_SIZES[0],
            128,
            KVLayout.SEQ_ALONG_LANE,
        )
        return MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=sublane_groups * packing,
            dtype=torch.uint8,
            tokens_per_state=min(self.compress_ratio, block_size),
            alignment=None,
        )


class VllmDeepseekV4Indexer(DeepseekV4Indexer):
    """TPU indexer: ranks compressed KV rows via the streamindex top-k kernel."""

    def __init__(self, *args, **kwargs) -> None:
        orig_compressor = dsv4_attention.DeepseekCompressor
        orig_cache = dsv4_attention.DeepseekV4IndexerCache
        dsv4_attention.DeepseekCompressor = VllmDeepseekCompressor
        dsv4_attention.DeepseekV4IndexerCache = VllmDeepseekV4IndexerCache
        try:
            super().__init__(*args, **kwargs)
        finally:
            dsv4_attention.DeepseekCompressor = orig_compressor
            dsv4_attention.DeepseekV4IndexerCache = orig_cache
        object.__setattr__(self, "indexer_op", self._build_indexer_op())
        tp = get_tensor_model_parallel_world_size()
        split_min = envs.TPU_DSV4_INDEXER_KV_SPLIT_MIN_TOKENS
        object.__setattr__(self, "kv_split", tp if (split_min > 0 and tp > 1) else 1)
        object.__setattr__(self, "kv_split_min_tokens", split_min)
        if self.kv_split > 1:
            object.__setattr__(
                self, "indexer_split_ops", self._build_indexer_op(self.kv_split)
            )
        object.__setattr__(self.compressor, "k_cache", self.k_cache)
        tok_above = envs.TPU_DSV4_INDEXER_TOKEN_SPLIT_ABOVE
        object.__setattr__(self, "token_split_above", 0)
        if tok_above > 0 and envs.TPU_DSV4_SEQ_PARALLEL and tp > 1:
            assert tok_above & (tok_above - 1) == 0, "power of 2"
            vcfg = get_current_vllm_config()
            spec = vcfg.speculative_config
            rows = vcfg.scheduler_config.max_num_seqs * (
                1 if spec is None else spec.num_speculative_tokens + 1
            )
            # The smallest split bucket is 2 x tok_above; its window must hold
            # every decode/verify row (they lead the batch), or a verify window
            # would be cut between ranks.
            assert rows <= 2 * tok_above // tp, (
                f"TPU_DSV4_INDEXER_TOKEN_SPLIT_ABOVE={tok_above}: {rows} decode/"
                f"verify rows do not fit a {2 * tok_above // tp}-row window"
            )
            object.__setattr__(self, "token_split_above", tok_above)
            object.__setattr__(self, "token_split_op", self._build_token_split_op(tp))
            if envs.TPU_DSV4_CSA_TOPK_FENCE_Q:
                # Built here, outside any trace, like the other jax ops.
                from vllm_torchtpu.layers.adapter.custom_ops.optimization_barrier_op import (  # noqa: E501
                    optimization_barrier2_op,
                )

                object.__setattr__(self, "fence_op", optimization_barrier2_op())

    def _build_token_split_op(self, tp: int):
        mesh = get_pallas_op_mesh()
        spec = get_current_vllm_config().speculative_config
        verify_q_len = None if spec is None else spec.num_speculative_tokens + 1
        flat = envs.TPU_DSV4_ROPE_QUANT_FLAT
        pieces = envs.TPU_DSV4_INDEXER_TOKEN_SPLIT_PIECES
        assert pieces >= 1 and pieces & (pieces - 1) == 0, "power of 2"
        op_name = (
            f"pallas::deepseek_v4_indexer_tok_k{self.topk_tokens}"
            f"_c{self.compress_ratio}_h{self.n_head}_tp{tp}"
            f"{'' if verify_q_len is None else f'_vq{verify_q_len}'}"
            f"{'_flat' if flat else ''}"
            f"{f'_pc{pieces}' if pieces > 1 else ''}"
            f"{'' if envs.TPU_STREAMIDX_CHUNK_SCHED_GROUPS else '_ng'}"
            f"{'_gm' if envs.TPU_DSV4_INDEXER_GROUP_MAX else ''}"
        )
        if op_name in _indexer_op_cache:
            return _indexer_op_cache[op_name]
        op = pallas.jax_op(
            op_name,
            functools.partial(
                _indexer_token_split_jax,
                tp=tp,
                flat_head_dim=self.head_dim if flat else 0,
                k=self.topk_tokens,
                compression_ratio=self.compress_ratio,
                softmax_scale=self.softmax_scale,
                n_head=self.n_head,
                num_kv_pages_per_block=envs.TPU_DSV4_INDEXER_NUM_KV_PAGES_PER_BLOCK,
                num_queries_per_block=envs.TPU_DSV4_INDEXER_NUM_QUERIES_PER_BLOCK,
                verify_q_len=verify_q_len,
                window_pieces=pieces,
            ),
            mesh=mesh,
            input_partition_specs=(
                P(None, "model") if flat else P(None, "model", None),  # q
                P(None),  # positions
                P(),  # cos_sin_cache
                P(None, "model"),  # indexer_weights
                P(),  # cache_kv
                P(BATCH_AXIS),  # seq_lens
                P(BATCH_AXIS),  # page_indices
                P(),  # cu_q_lens
                P(),  # distribution
                P(),  # rank
            ),
        )

        def _fake(q, *args, **kwargs):
            return torch.empty(
                (q.shape[0], self.topk_tokens), dtype=torch.int32, device=q.device
            )

        op.register_fake(_fake)
        _indexer_op_cache[op_name] = op
        return op

    def _token_split_active(self, num_tokens: int) -> bool:
        if not self.token_split_above or num_tokens <= self.token_split_above:
            return False
        from vllm_torchtpu.layers.adapter.fused_moe_ep import SP_GATE

        return not SP_GATE["replicated"] and num_tokens % get_tp_group().world_size == 0

    def _build_indexer_op(self, kv_split: int = 1):
        mesh = get_pallas_op_mesh()
        # Speculative verify windows all hold K + 1 tokens; the runner puts
        # them in `[distribution[0], distribution[1])`.
        spec = get_current_vllm_config().speculative_config
        verify_q_len = None if spec is None else spec.num_speculative_tokens + 1
        flat = envs.TPU_DSV4_ROPE_QUANT_FLAT

        wrapped_fn = functools.partial(
            _indexer_jax if kv_split == 1 else _indexer_split_jax,
            k=self.topk_tokens,
            compression_ratio=self.compress_ratio,
            softmax_scale=self.softmax_scale,
            n_head=self.n_head,
            num_kv_pages_per_block=envs.TPU_DSV4_INDEXER_NUM_KV_PAGES_PER_BLOCK,
            num_queries_per_block=envs.TPU_DSV4_INDEXER_NUM_QUERIES_PER_BLOCK,
            verify_q_len=verify_q_len,
            flat_head_dim=self.head_dim if flat else 0,
            **({} if kv_split == 1 else {"kv_split": kv_split}),
        )

        _scale = (
            f"{self.softmax_scale:.6g}".replace(".", "p")
            .replace("-", "m")
            .replace("+", "")
        )
        op_name = (
            f"pallas::deepseek_v4_indexer_k{self.topk_tokens}"
            f"_c{self.compress_ratio}_h{self.n_head}_s{_scale}"
            f"{'' if verify_q_len is None else f'_vq{verify_q_len}'}"
            f"{'' if kv_split == 1 else f'_split{kv_split}'}"
            f"{'_flat' if flat else ''}"
        )
        global _indexer_op_cache
        if op_name in _indexer_op_cache:
            return _indexer_op_cache[op_name]

        attn_data_axis = None
        attn_head_axis = "model"
        batch_axis = BATCH_AXIS
        indexer_jax_op = pallas.jax_op(
            op_name,
            wrapped_fn,
            mesh=mesh,
            input_partition_specs=(
                # q: flat [T, heads * dim] under TPU_DSV4_ROPE_QUANT_FLAT
                P(attn_data_axis, attn_head_axis)
                if flat
                else P(attn_data_axis, attn_head_axis, None),
                P(attn_data_axis),  # positions
                P(),  # cos_sin_cache
                P(attn_data_axis, attn_head_axis),  # indexer_weights
                P(),  # cache_kv
                P(batch_axis),  # seq_lens
                P(batch_axis),  # page_indices
                P(),  # cu_q_lens
                P(),  # distribution
                *((P(),) if kv_split > 1 else ()),  # split rank
            ),
        )
        out_width = self.topk_tokens * (1 if kv_split == 1 else 2)

        def _fake_indexer(q, *args, **kwargs):
            return torch.empty(
                (q.shape[0], out_width), dtype=torch.int32, device=q.device
            )

        indexer_jax_op.register_fake(_fake_indexer)
        if kv_split > 1:
            a2a_merge_op = pallas.jax_op(
                f"pallas::deepseek_v4_indexer_merge_a2a_k{self.topk_tokens}_s{kv_split}",
                functools.partial(
                    _merge_kv_split_a2a_jax, k=self.topk_tokens, kv_split=kv_split
                ),
                mesh=mesh,
                input_partition_specs=(P(),),
            )

            def _fake_merge_a2a(exchanged):
                return torch.empty(
                    (exchanged.shape[0] // kv_split, self.topk_tokens),
                    dtype=torch.int32,
                    device=exchanged.device,
                )

            a2a_merge_op.register_fake(_fake_merge_a2a)
            indexer_jax_op = (indexer_jax_op, a2a_merge_op)

        _indexer_op_cache[op_name] = indexer_jax_op
        return indexer_jax_op

    def forward(
        self,
        hidden_states: torch.Tensor,
        query: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
        slot_mapping: torch.Tensor | None = None,
        fence: torch.Tensor | None = None,
        compressor_skip_proj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """fence (TPU_DSV4_CSA_TOPK_FENCE_Q, token-split path only): a tensor
        this rank's top-k is fenced with before its all-gather; returns
        (top-k, fence) so the caller uses the fenced copy. Scheduling only.
        compressor_skip_proj (TPU_DSV4_COMPRESSOR_PAIR): the pair op already
        projected this compressor's tokens."""
        num_tokens = query.shape[0]
        token_split = self._token_split_active(num_tokens)
        if token_split:
            # This rank's contiguous 1/tp of the rows (rank as device data, so
            # every TP rank traces one graph); the projection shrinks with them.
            group = get_tp_group()
            tp = group.world_size
            rank = tp_rank_as_data(group, query.device)
            query, indexer_weights, q_positions = (
                t.reshape(tp, -1, *t.shape[1:]).index_select(0, rank)[0]
                for t in (query, indexer_weights, positions)
            )
        q, _ = self.wq_b(query)
        if not envs.TPU_DSV4_ROPE_QUANT_FLAT:
            q = q.view(-1, self.n_head, self.head_dim)

        self.compressor(
            hidden_states, positions, rotary_emb, skip_proj=compressor_skip_proj
        )

        # No KV cache is bound during the profiling forward; the values are
        # unused, so return correctly-shaped dummies. vLLM binds a numel==0
        # placeholder rather than None, so test emptiness.
        kv_cache = getattr(self.k_cache, "kv_cache", None)
        if kv_cache is None or kv_cache.numel() == 0:
            dummy = torch.zeros(
                (num_tokens, self.topk_tokens), dtype=torch.int32, device=q.device
            )
            return dummy if fence is None else (dummy, fence)

        # The TPU runner stores its own AttentionMetadata, not the vLLM base class.
        attn_ctx = cast(
            "dict[str, AttentionMetadata] | AttentionMetadata",
            get_forward_context().attn_metadata,
        )
        if isinstance(attn_ctx, dict):
            prefix_key = getattr(getattr(self, "k_cache", None), "prefix", None)
            if prefix_key not in attn_ctx:
                raise KeyError(
                    f"DeepSeek-V4 indexer prefix {prefix_key!r} has no "
                    f"attention metadata. Known: {sorted(attn_ctx)}"
                )
            attn_metadata = attn_ctx[prefix_key]
        else:
            attn_metadata = attn_ctx

        idx_seq_lens = attn_metadata.seq_lens
        idx_block_tables = attn_metadata.block_tables
        idx_q_start_loc = attn_metadata.query_start_loc
        idx_req_dist = attn_metadata.request_distribution

        args = (
            q,
            positions,
            rotary_emb.cos_sin_cache,
            indexer_weights,
            kv_cache,
            idx_seq_lens,
            idx_block_tables,
            idx_q_start_loc,
            idx_req_dist,
        )
        if token_split:
            from vllm_torchtpu.models.vllm.deepseek_v4.seq_parallel import (
                build_sp_ops,
            )

            local = self.token_split_op(q, q_positions, *args[2:], rank)
            if fence is not None:
                # The SparseCore top-k's done otherwise sits right behind its
                # start with the TensorCore idle; joining it with the q
                # projection (which does not need it) lets the scheduler put
                # that projection between the two. The op is built at init
                # (Dynamo cannot trace its lru_cache'd builder).
                local, fence = self.fence_op(local, fence)
                return build_sp_ops()["ag"](local).view(num_tokens, -1), fence
            return build_sp_ops()["ag"](local).view(num_tokens, -1)
        if fence is not None:
            raise ValueError("fence is only taken on the token-split path")
        if (
            self.kv_split == 1
            or q.shape[0] < self.kv_split_min_tokens
            or q.shape[0] % self.kv_split
        ):
            return self.indexer_op(*args)
        share_op, a2a_merge_op = self.indexer_split_ops
        group = get_tp_group()
        rank = tp_rank_as_data(group, q.device)
        share = share_op(*args, rank)
        exchanged = _all_to_all(share, group)
        return tensor_model_parallel_all_gather(a2a_merge_op(exchanged), 0)
