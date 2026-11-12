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
"""Out-of-tree (OOT) custom operator layers and wrappers for
Multi-Head Latent Attention (MLA)."""

from typing import Any, cast

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from torch.nn import Parameter
from torch_tpu._internal import pallas
from vllm.config import CacheConfig, get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import mla_attention
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    DCPGroupColumnParallelLinear,
)
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import rotate_gptj, rotate_neox
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache, Indexer
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.mla.prefill import selector

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dcp import get_dcp_group, get_or_create_dcp_mesh
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    KVLayout,
    coalesced_kv_append,
    coalesced_kv_append_dcp,
    cp_rank_as_data,
    get_kv_cache_shape,
    indexer_kv_layout,
    kv_cache_append,
    streamindex_topk,
    streamindex_topk_dcp,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
    pcp_streaming_jax_op,
)
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    OOB_PAGE,
    _row_positions,
    update_sparse_mla_kv_cache,
    update_sparse_mla_kv_cache_dcp,
)
from vllm_torchtpu.layers.adapter.attention import (
    TPU_STR_DTYPE_TO_TORCH_DTYPE,
    PallasMLAttentionBackendImpl,
    VllmTPUDeepseekV32IndexerBackend,
)
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization.configs import should_quantize_on_load
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.quantization import quantize_kv, quantize_tensor
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

logger = init_logger(__name__)

# Skip indexer scoring when every sequence in the batch is shorter than
# `index_topk`, where the top-k is the identity and the scores decide nothing.
# The guard is batch-wide, so this fires during warmup and on short-context
# traffic and not at all once any request in the batch is past `index_topk`
# tokens.
STREAMIDX_ENABLE_EARLY_EXIT = True
# StreamIdx block sizes as (decode, prefill, MIXED), tuned by microbenchmark.
STREAMIDX_KV_PAGES_PER_BLOCK = (2, 2, 2)
STREAMIDX_QUERIES_PER_BLOCK = (1, 128, 128)
# MIXED-pass KV pages per block for small buckets when only the bq knob is set.
STREAMIDX_SMALL_BUCKET_DEFAULT_BKVP = 8


def _streamidx_block_sizes(
    num_tokens: int,
    small_bucket_bq: int | None,
    small_bucket_bkvp: int | None,
    small_bucket_tokens: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """(num_kv_pages_per_block, num_queries_per_block) for a padded token bucket.

    MIXED-pass query blocks hold one sequence each, so an MTP request
    (q_len = gamma + 1) in a 128-row block scores ~124 padding rows. Buckets up
    to `small_bucket_tokens` take the small-bucket sizes instead.
    """
    if small_bucket_bq is None or num_tokens > small_bucket_tokens:
        return STREAMIDX_KV_PAGES_PER_BLOCK, STREAMIDX_QUERIES_PER_BLOCK
    bkvp = small_bucket_bkvp or STREAMIDX_SMALL_BUCKET_DEFAULT_BKVP
    return (
        STREAMIDX_KV_PAGES_PER_BLOCK[:2] + (bkvp,),
        STREAMIDX_QUERIES_PER_BLOCK[:2] + (small_bucket_bq,),
    )


_INDEXER_KV_CACHE_FALLBACK_LOGGED: set[tuple[str, str]] = set()
_INDEXER_KV_CACHE_INFO_LOGGED: set[tuple[str, str, str, tuple[int, ...]]] = set()


def _warn_indexer_kv_cache_fallback_once(mode: str, reason: str) -> None:
    key = (mode, reason)
    if key in _INDEXER_KV_CACHE_FALLBACK_LOGGED:
        return
    _INDEXER_KV_CACHE_FALLBACK_LOGGED.add(key)
    logger.warning(
        "VLLM_TPU_INDEXER_KV_CACHE_MODE=%r falling back to 'xla': %s",
        mode,
        reason,
    )


def _log_indexer_kv_cache_once(
    mode: str,
    writer: str,
    kv_layout: KVLayout,
    cache_shape: tuple[int, ...],
) -> None:
    layout_tag = "SAL" if kv_layout == KVLayout.SEQ_ALONG_LANE else "HAS"
    shape_tuple = tuple(int(d) for d in cache_shape)
    key = (mode, writer, layout_tag, shape_tuple)
    if key in _INDEXER_KV_CACHE_INFO_LOGGED:
        return
    _INDEXER_KV_CACHE_INFO_LOGGED.add(key)
    logger.info(
        "indexer kv cache: mode=%s writer=%s layout=%s cache_shape=%s",
        mode,
        writer,
        layout_tag,
        shape_tuple,
    )


def _unsupported_indexer_kv_cache_reason(
    cache_kv: jax.Array,
    k_packed: jax.Array,
    kv_layout: KVLayout,
) -> str | None:
    """Returns a human-readable reason if `cache_kv`/`k_packed` cannot use Pallas."""
    backend = jax.default_backend()
    if backend != "tpu":
        return f"backend={backend!r} is not 'tpu'"
    try:
        kv_cache_append.check_scatter_rows_supported(cache_kv, k_packed, kv_layout)
    except ValueError as exc:
        return str(exc)
    return None


def _resolve_indexer_kv_cache_mode(
    cache_kv: jax.Array,
    k_packed: jax.Array,
    *,
    mode: str | None = None,
) -> tuple[str, KVLayout]:
    """Resolves the indexer K-cache `(writer, kv_layout)` for `cache_kv` and `k_packed`.

    Layout is decided once by `indexer_kv_layout(block_size, mode=...)` and
    validated against `get_kv_cache_shape`. `writer` is `'pallas'` for `'has'`
    and `'sal'`, or `'xla'` for `'xla'`. XLA fallback is only permitted when
    `kv_layout == KVLayout.HEAD_ALONG_SUBLANE`; `SEQ_ALONG_LANE` on a non-TPU
    backend or unsupported geometry raises `ValueError`.
    """
    raw_mode = envs.VLLM_TPU_INDEXER_KV_CACHE_MODE if mode is None else mode
    resolved = raw_mode.strip().lower()
    if resolved not in ("xla", "has", "sal"):
        raise ValueError(
            f"Invalid VLLM_TPU_INDEXER_KV_CACHE_MODE={raw_mode!r}; "
            "expected one of ('xla', 'has', 'sal')."
        )
    if cache_kv.ndim != 4:
        raise ValueError(
            f"Expected 4-D cache_kv, got {cache_kv.ndim}-D shape {cache_kv.shape}."
        )
    if k_packed.ndim != 2:
        raise ValueError(
            f"Expected 2-D k_packed, got {k_packed.ndim}-D shape {k_packed.shape}."
        )

    raw_width = int(k_packed.shape[-1])
    actual_head_dim = (
        raw_width - max(1, raw_width // 128) if raw_width % 64 != 0 else raw_width
    )
    sal_sublane_groups = ((actual_head_dim + 127) // 128) * 128 // 4 + 1
    block_size = (
        int(cache_kv.shape[3])
        if int(cache_kv.shape[1]) == sal_sublane_groups
        else int(cache_kv.shape[1]) * 4
    )
    kv_layout = indexer_kv_layout(block_size, mode=mode)
    effective_mode = (
        "has"
        if resolved == "sal" and kv_layout == KVLayout.HEAD_ALONG_SUBLANE
        else resolved
    )
    expected_shape = get_kv_cache_shape(
        int(cache_kv.shape[0]),
        block_size,
        actual_head_dim,
        kv_layout,
    )
    if tuple(int(d) for d in cache_kv.shape) != expected_shape:
        raise ValueError(
            f"cache_kv shape {tuple(cache_kv.shape)} does not match expected "
            f"{kv_layout.name} shape {expected_shape} for "
            f"VLLM_TPU_INDEXER_KV_CACHE_MODE={resolved!r}."
        )

    if effective_mode == "xla":
        _log_indexer_kv_cache_once(effective_mode, "xla", kv_layout, cache_kv.shape)
        return "xla", kv_layout

    reason = _unsupported_indexer_kv_cache_reason(cache_kv, k_packed, kv_layout)
    if reason is not None:
        if kv_layout == KVLayout.SEQ_ALONG_LANE:
            raise ValueError(
                f"VLLM_TPU_INDEXER_KV_CACHE_MODE={resolved!r} with layout "
                f"{kv_layout.name} cannot fall back to 'xla': {reason}"
            )
        _warn_indexer_kv_cache_fallback_once(effective_mode, reason)
        _log_indexer_kv_cache_once(effective_mode, "xla", kv_layout, cache_kv.shape)
        return "xla", kv_layout

    _log_indexer_kv_cache_once(effective_mode, "pallas", kv_layout, cache_kv.shape)
    return "pallas", kv_layout


def _dcp_indexer_a2a_for_bucket(
    num_tokens: int, enabled: bool, min_tokens: int
) -> bool:
    """Whether the DCP indexer merge for this token bucket uses the all-to-all.

    `num_tokens` is the static padded bucket, so this picks one merge per
    compiled variant: decode-sized buckets keep the all-gather merge and
    prefill-sized ones (>= `min_tokens`) take the all-to-all.
    """
    return enabled and num_tokens >= min_tokens


def _topk_rows_update_jax(buf: jax.Array, rows: jax.Array) -> jax.Array:
    """`buf[:rows.shape[0]] = rows` as one row update of a donated buffer."""
    return jax.lax.dynamic_update_slice(buf, rows.astype(buf.dtype), (0, 0))


# The shared top-k table's row update, donating the table.
#
# `buf[:n] = rows` on the module buffer reaches XLA as a graph input that is
# mutated but not donated -- aot_autograd returns the new value and writes it
# back after the graph -- plus a view write lowered as two row updates of the
# same base. Both force a full `[max_num_batched_tokens, width]` copy of the
# table in every indexed layer. Routing the write through a donating op with a
# `copy_` writeback (the KV-cache pattern) marks the table a buffer donor, so
# XLA updates it in place. Built at import, not lazily: dynamo cannot trace the
# op's construction from inside a compiled forward.
_TOPK_ROWS_UPDATE_OP = pallas.jax_op(
    "pallas::indexer_topk_rows_update", _topk_rows_update_jax, donate_argnums=(0,)
)


# torch.compile traces the forward with fake tensors (shape and dtype, no data),
# so it can't run the JAX function to learn what this op returns. This fake
# tells it: a tensor shaped like `buf`, i.e. the whole updated table, which the
# caller copies back into `buf`. The real `_topk_rows_update_jax` runs only
# when the compiled graph executes.
def _fake_topk_rows_update(buf: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    del rows
    return torch.empty_like(buf)


_TOPK_ROWS_UPDATE_OP.register_fake(_fake_topk_rows_update)


def _topk_rows(
    buf: torch.Tensor, topk_tokens: int, num_tokens: int, topk_indices: torch.Tensor
) -> torch.Tensor:
    """The `[num_tokens, buf.shape[1]]` rows the -1 fill plus partial write
    would leave at the top of `buf`: the top-k, then -1, clamped to `buf`.

    The top-k kernels return one row per query token, so only the columns
    ever need -1 padding.
    """
    num_tokens = min(num_tokens, buf.shape[0])
    rows = topk_indices[:num_tokens, :topk_tokens].to(buf.dtype)
    assert rows.shape[0] == num_tokens, (rows.shape, num_tokens)
    pad_cols = buf.shape[1] - rows.shape[1]
    if pad_cols:
        rows = torch.nn.functional.pad(rows, (0, pad_cols), value=-1)
    return rows


def indexer_k_append(
    cache_kv: jax.Array,
    k_packed: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    *,
    writer: str,
    kv_layout: Any,
) -> jax.Array:
    """Append this step's packed K rows to an unsharded indexer cache.

    `writer` and `kv_layout` come from `_resolve_indexer_kv_cache_mode`. The
    XLA writer places rows with the latent-cache writers' `_row_positions`;
    padded rows go to `OOB_PAGE`, which the scatter drops.
    """
    if writer != "xla":
        return coalesced_kv_append(
            cache_kv, k_packed, seq_lens, page_indices, cu_q_lens, kv_layout=kv_layout
        )
    _, block_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape
    block_size = block_size_per_kv_packing * kv_packing
    # Cache rows are padded to the 128-lane HBM width; pad the packed fp8+scale
    # rows to match before scattering.
    k_packed = jnp.pad(k_packed, ((0, 0), (0, lkv_dim - k_packed.shape[-1])))
    _, seq_id, pos, valid = _row_positions(k_packed.shape[0], seq_lens, cu_q_lens)
    pages = page_indices.reshape(seq_lens.shape[0], -1)
    # A padded row's `pos` can run off the page table: clip it in bounds and
    # let `valid` drop the row.
    page_idx = jnp.clip(pos // block_size, 0, pages.shape[1] - 1)
    page = jnp.where(valid, pages[seq_id, page_idx], jnp.int32(OOB_PAGE))
    slot = pos % block_size
    return cache_kv.at[page, slot // kv_packing, slot % kv_packing].set(k_packed)


def _cp_scatter_indexer_k(
    cache_kv: jax.Array,
    k_packed: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    *,
    mesh,
    cp_size: int,
    interleave_size: int,
    axis_name: str = DCP_AXIS_NAME,
    writer: str = "xla",
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE,
) -> jax.Array:
    """Scatter this step's indexer K rows into the DCP-sharded K cache.

    Each rank writes only the rows it owns, and ownership
    follows the same interleave the reader assumes -- a virtual page spans
    `page_size * cp_size` positions and rank `r` holds the `interleave_size`
    -sized chunks at offset `r * interleave_size` within each
    `cp_size * interleave_size` cycle. Rows owned by another rank are dropped
    by aiming them at an out-of-bounds page, the same trick the unsharded
    scatter uses for padding.

    `k_packed` must already be in request-major order, since the positions
    come from `cu_q_lens`/`seq_lens`.

    `axis_name`/`cp_size` name the axis the cache is split over.
    """
    if writer not in ("xla", "pallas"):
        raise ValueError(f"Expected writer in ('xla', 'pallas'), got {writer!r}.")
    if writer == "xla" and kv_layout != KVLayout.HEAD_ALONG_SUBLANE:
        raise ValueError(
            f"writer='xla' only supports HEAD_ALONG_SUBLANE, got {kv_layout}."
        )

    def _local(cache_kv, k_packed, seq_lens, page_indices, cu_q_lens):
        rank = cp_rank_as_data(axis_name, cp_size)
        if writer == "pallas":
            return coalesced_kv_append_dcp(
                cache_kv,
                k_packed,
                seq_lens,
                page_indices,
                cu_q_lens,
                rank,
                dcp_size=cp_size,
                interleave_size=interleave_size,
                kv_layout=kv_layout,
            )

        _, block_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape
        page_size = block_size_per_kv_packing * kv_packing
        # The latent cache is guarded on its own page size in
        # `update_sparse_mla_kv_cache_dcp`; this cache is a separate
        # allocation, so it needs the check on its own number.
        if page_size % interleave_size:
            raise ValueError(
                f"interleave_size={interleave_size} must divide the indexer "
                f"K cache page_size={page_size}. Otherwise an interleave "
                "chunk straddles a page boundary and the local index "
                "`streamindex_topk_dcp` derives from a global position stops "
                "agreeing with the (page, slot) written here."
            )
        k_packed = jnp.pad(k_packed, ((0, 0), (0, lkv_dim - k_packed.shape[-1])))

        # Same notion of "where does this row belong" as the latent-cache
        # writers, so the two cannot drift apart.
        _, seq_id, pos, in_batch = _row_positions(
            k_packed.shape[0], seq_lens, cu_q_lens
        )

        virtual_page_size = jnp.int32(page_size * cp_size)
        cycle = jnp.int32(cp_size * interleave_size)
        interleave = jnp.int32(interleave_size)
        offset = pos % virtual_page_size
        owner = (offset % cycle) // interleave
        slot = (offset // cycle) * interleave + (offset % interleave)

        # Virtual ordinal -> physical block in this rank's own shard, exactly
        # as `streamindex_topk_dcp` resolves it on the read side.
        local_page_indices = jnp.mod(
            page_indices, jnp.int32(cache_kv.shape[0])
        ).reshape(seq_lens.shape[0], -1)
        # A padded token's `pos` is meaningless and can run off the end of the
        # page table; clip so the gather stays in bounds and let `valid` drop
        # the row, the same way the sparse MLA writer does.
        max_pages_per_seq = local_page_indices.shape[1]
        virtual_page = jnp.clip(pos // virtual_page_size, 0, max_pages_per_seq - 1)

        valid = jnp.logical_and(in_batch, owner == rank)
        page = jnp.where(
            valid, local_page_indices[seq_id, virtual_page], jnp.int32(OOB_PAGE)
        )
        return cache_kv.at[page, slot // kv_packing, slot % kv_packing].set(k_packed)

    replicated = jax.sharding.PartitionSpec()
    sharded = jax.sharding.PartitionSpec(axis_name)
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(sharded, replicated, replicated, replicated, replicated),
        out_specs=sharded,
        check_vma=False,
    )(cache_kv, k_packed, seq_lens, page_indices, cu_q_lens)


@SparseAttnIndexer.register_oot
class VllmTPUSparseAttnIndexer(SparseAttnIndexer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Build eagerly: pallas.jax_op runs inspect.signature(), which Dynamo
        # cannot trace if the op is first built inside a compiled forward.
        self.topk_op = self._build_streamidx_op()

        parallel_config = getattr(
            get_current_vllm_config_or_none(), "parallel_config", None
        )
        pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
        pcp_size = pcp_size if isinstance(pcp_size, int) else 1
        dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
        dcp_size = dcp_size if isinstance(dcp_size, int) else 1
        self.topk_op_dcp = None
        self.dcp_size = dcp_size
        if dcp_size > 1:
            if pcp_size > 1:
                raise NotImplementedError(
                    f"pcp_size={pcp_size} and dcp_size={dcp_size} are both >1; "
                    "the sparse indexer shards its K cache by position on "
                    "whichever axis it is given and has no layout for two at "
                    "once."
                )
            assert parallel_config is not None
            self.topk_op_dcp = self._build_streamidx_dcp_op(
                dcp_size, parallel_config.cp_kv_cache_interleave_size
            )

    def _build_streamidx_op(self):
        topk = self.topk_tokens
        # None means unchunked. The kernel falls back to a single unpipelined
        # pass when `max_num_tokens` is not a multiple of the chunk.
        chunk_tokens = envs.TPU_STREAMIDX_CHUNK_TOKENS
        small_bucket = (
            envs.TPU_STREAMIDX_SMALL_BUCKET_BQ,
            envs.TPU_STREAMIDX_SMALL_BUCKET_BKVP,
            envs.TPU_STREAMIDX_SMALL_BUCKET_TOKENS,
        )
        kv_cache_mode = (
            envs.VLLM_TPU_INDEXER_KV_CACHE_MODE
            if envs.is_indexer_kv_cache_mode_explicit()
            else None
        )

        def _streamidx_topk_jax(
            q_bytes: jax.Array,  # uint8 view of fp8 q [num_tokens, H, D]
            weights: jax.Array,  # [num_tokens, H]
            # uint8 [num_blocks, block_size_per_kv_packing, kv_packing, lkv_dim]
            # where lkv_dim = D + 1 padded to a 128 multiple
            cache_kv: jax.Array,
            k_packed: jax.Array,  # uint8 [num_tokens, D + 1] fp8 k + e8m0 scale
            seq_lens: jax.Array,  # i32 [num_seqs]
            page_indices: jax.Array,  # i32 [num_seqs * pages_per_seq]
            cu_q_lens: jax.Array,  # i32 [num_seqs + 1]
            distribution: jax.Array,  # i32 [3]
        ) -> tuple[jax.Array, jax.Array]:
            writer, kv_layout = _resolve_indexer_kv_cache_mode(
                cache_kv, k_packed, mode=kv_cache_mode
            )
            cache_kv = indexer_k_append(
                cache_kv,
                k_packed,
                seq_lens,
                page_indices,
                cu_q_lens,
                writer=writer,
                kv_layout=kv_layout,
            )

            q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
            kv_pages_per_block, queries_per_block = _streamidx_block_sizes(
                q.shape[0], *small_bucket
            )
            topk_indices = streamindex_topk(
                q,
                weights,
                cache_kv,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
                k=topk,
                compression_ratio=1,
                num_kv_pages_per_block=kv_pages_per_block,
                num_queries_per_block=queries_per_block,
                enable_early_exit=STREAMIDX_ENABLE_EARLY_EXIT,
                kv_layout=kv_layout,
                chunk_tokens=chunk_tokens,
            )
            return cache_kv, topk_indices

        op_name = f"pallas::streamidx_topk_{self.k_cache.prefix.replace('.', '_')}"
        jax_op = pallas.jax_op(op_name, _streamidx_topk_jax, donate_argnums=(2,))

        def _fake(q_bytes, weights, cache_kv, *args, **kwargs):
            return torch.empty_like(cache_kv), torch.empty(
                (q_bytes.shape[0], topk), dtype=torch.int32, device=q_bytes.device
            )

        jax_op.register_fake(_fake)

        def op(*args):
            new_cache, topk_indices = jax_op(*args)
            args[2].copy_(new_cache)
            return topk_indices

        self._streamidx_op = op
        return op

    def _build_streamidx_dcp_op(self, dcp_size: int, interleave_size: int):
        """The DCP twin of `_build_streamidx_op`.

        DCP shards the KV cache and leaves the *queries*
        replicated, so there is no rank-major token packing to undo. Every
        token-major input arrives already in request-major order and stays
        replicated across the op boundary. Only the cache is split.
        """
        topk = self.topk_tokens
        mesh = get_or_create_dcp_mesh(axis_name=DCP_AXIS_NAME)
        # Chunking cuts the replicated query-token axis, which is orthogonal to
        # the KV axis DCP shards, so the same knob applies here unchanged.
        chunk_tokens = envs.TPU_STREAMIDX_CHUNK_TOKENS
        a2a_merge = envs.TPU_DCP_INDEXER_A2A_MERGE
        a2a_min_tokens = envs.TPU_DCP_INDEXER_A2A_MIN_TOKENS
        small_bucket = (
            envs.TPU_STREAMIDX_SMALL_BUCKET_BQ,
            envs.TPU_STREAMIDX_SMALL_BUCKET_BKVP,
            envs.TPU_STREAMIDX_SMALL_BUCKET_TOKENS,
        )
        kv_cache_mode = (
            envs.VLLM_TPU_INDEXER_KV_CACHE_MODE
            if envs.is_indexer_kv_cache_mode_explicit()
            else None
        )

        def _streamidx_topk_dcp_jax(
            q_bytes: jax.Array,  # replicated, uint8 view of fp8 q
            weights: jax.Array,  # replicated
            cache_kv: jax.Array,  # this rank's position shard
            k_packed: jax.Array,  # replicated
            seq_lens: jax.Array,
            page_indices: jax.Array,  # virtual page ordinals
            cu_q_lens: jax.Array,
            distribution: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            writer, kv_layout = _resolve_indexer_kv_cache_mode(
                cache_kv, k_packed, mode=kv_cache_mode
            )
            # `_cp_scatter_indexer_k` pads the packed fp8+scale rows out to
            # the cache's record width itself.
            cache_kv = _cp_scatter_indexer_k(
                cache_kv,
                k_packed,
                seq_lens,
                page_indices,
                cu_q_lens,
                mesh=mesh,
                cp_size=dcp_size,
                interleave_size=interleave_size,
                axis_name=DCP_AXIS_NAME,
                writer=writer,
                kv_layout=kv_layout,
            )

            q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
            kv_pages_per_block, queries_per_block = _streamidx_block_sizes(
                q.shape[0], *small_bucket
            )
            local_topk = streamindex_topk_dcp(
                q,
                weights,
                cache_kv,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
                mesh=mesh,
                k=topk,
                compression_ratio=1,
                dcp_size=dcp_size,
                interleave_size=interleave_size,
                num_kv_pages_per_block=kv_pages_per_block,
                num_queries_per_block=queries_per_block,
                kv_layout=kv_layout,
                chunk_tokens=chunk_tokens,
                a2a_merge=_dcp_indexer_a2a_for_bucket(
                    q.shape[0], a2a_merge, a2a_min_tokens
                ),
            )
            return cache_kv, local_topk

        replicated = jax.sharding.PartitionSpec()
        sharded = jax.sharding.PartitionSpec(DCP_AXIS_NAME)
        op_name = f"pallas::streamidx_topk_dcp_{self.k_cache.prefix.replace('.', '_')}"
        # Reuses the PCP export wrapper verbatim: nothing in it is PCP-specific
        # once the mesh and the specs are arguments, and the reason it exists --
        # `jax_placeholders` specializing the dynamic token dim while scaling
        # sharded dims to global size -- applies identically here.
        jax_op = pcp_streaming_jax_op(
            op_name,
            _streamidx_topk_dcp_jax,
            donate_argnums=(2,),
            mesh=mesh,
            input_partition_specs=(
                replicated,  # q_bytes
                replicated,  # weights
                sharded,  # cache_kv
                replicated,  # k_packed
                replicated,  # seq_lens
                replicated,  # page_indices
                replicated,  # cu_q_lens
                replicated,  # distribution
            ),
            output_partition_specs=(
                sharded,  # new cache_kv
                sharded,  # local_topk: [num_tokens, topk] per rank
            ),
        )

        # We must overwrite the default fake implementation as vLLM uses
        # dynamic dimensions for the query.
        def _fake_streamidx_topk_dcp(
            q_bytes,
            weights,
            cache_kv,
            k_packed,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            **kwargs,
        ):
            # pyrefly: ignore[unsupported-delete]
            del (
                weights,
                k_packed,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
                kwargs,
            )
            return (
                torch.empty_like(cache_kv),
                q_bytes.new_empty((q_bytes.shape[0], topk), dtype=torch.int32),
            )

        jax_op.register_fake(_fake_streamidx_topk_dcp)

        def op(*args):
            new_cache, local_topk = jax_op(*args)
            args[2].copy_(new_cache)
            return local_topk

        return op

    @staticmethod
    def pack_k(k: torch.Tensor) -> torch.Tensor:
        """K quantized to fp8 with a per-token e8m0 scale, packed into the
        uint8 row layout the TPU kernel unpacks (head_dim value bytes + 1
        scale byte)."""
        k_quant, k_scale = quantize_tensor(
            k,
            torch.float8_e4m3fn,
            axis=-1,
            use_ue8m0=True,
        )
        # cat requires a single dtype, so view the fp8 values as raw bytes.
        return torch.cat([k_quant.view(torch.uint8), k_scale], dim=-1)

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q_values: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        assert q_values.dtype == torch.float8_e4m3fn, (
            f"streamindex_topk expects fp8 q, got {q_values.dtype}"
        )

        k_quant_packed = VllmTPUSparseAttnIndexer.pack_k(k)

        attn_metadata = get_forward_context().attn_metadata
        metadata = None
        if isinstance(attn_metadata, dict):
            metadata = attn_metadata.get(self.k_cache.prefix)
        if not isinstance(metadata, AttentionMetadata):
            return self.topk_indices_buffer

        # uint8 [num_blocks, block_size_per_kv_packing, kv_packing, lkv_dim],
        # lkv_dim = head_dim + 1 padded to a 128 multiple
        kv_cache = self.k_cache.kv_cache
        if kv_cache.numel() == 0:
            return self.topk_indices_buffer

        seq_lens = metadata.seq_lens
        page_indices = metadata.block_tables
        cu_q_lens = metadata.query_start_loc
        rd = metadata.request_distribution
        distribution = torch.stack([rd[0], rd[0], rd[2]])

        if self.topk_op_dcp is not None:
            # DCP replicates the queries, so the token
            # inputs are already the whole batch in request-major order. What
            # comes back is not global positions but this rank's own local
            # cache indices, left-packed and `-1` padded -- the attention side
            # consumes them directly and merges across ranks by log-sum-exp.
            topk_indices = self.topk_op_dcp(
                q_values.view(torch.uint8),
                weights,
                kv_cache,
                k_quant_packed,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
            )
        else:
            topk_indices = self.topk_op(
                q_values.view(torch.uint8),
                weights,
                kv_cache,
                k_quant_packed,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
            )

        return self._write_topk_rows(hidden_states.shape[0], topk_indices)

    def _write_topk_rows(
        self, num_tokens: int, topk_indices: torch.Tensor
    ) -> torch.Tensor:
        """The rows the -1 fill plus partial write would leave, in one update.

        Writes the `_topk_rows` with a single row update through a donating op
        (see `_TOPK_ROWS_UPDATE_OP`), so XLA updates the table in place: same
        contents, no full-table copy per indexed layer. Returns those rows
        rather than the buffer, so attention reads them instead of a whole
        `[max_num_batched_tokens, width]` buffer version. The buffer still gets
        the same contents for the layers that reuse it.
        """
        buf = self.topk_indices_buffer
        rows = _topk_rows(buf, self.topk_tokens, num_tokens, topk_indices)
        buf.copy_(_TOPK_ROWS_UPDATE_OP(buf, rows))
        return rows


class VllmTPUIndexerCache(DeepseekV32IndexerCache):
    """Indexer K cache retyped onto the TPU backend. See `VllmTPUIndexer`."""

    def get_attn_backend(self) -> type[DeepseekV32IndexerBackend]:
        return VllmTPUDeepseekV32IndexerBackend


class VllmTPUIndexer(Indexer):
    """TPU-optimized out-of-tree wrapper for Indexer.

    Never constructed directly. `Indexer` is neither a `CustomOp` nor a
    `PluggableLayer`, so it has no `register_oot` hook, and
    `DeepseekV2MLAAttention` builds it before this plugin sees it. `rebind`
    retypes the finished instance in place: state is preserved, method lookup
    moves to this class.
    """

    @classmethod
    def rebind(cls, indexer: Indexer) -> "VllmTPUIndexer":
        """Retype an in-tree `Indexer`, and its K cache, onto the TPU path."""
        if isinstance(indexer, cls):
            return indexer
        assert indexer.quant_block_size == indexer.head_dim, (
            "streamindex_topk requires quant_block_size == head_dim, got "
            f"{indexer.quant_block_size} != {indexer.head_dim}"
        )
        indexer.__class__ = cls
        indexer.k_cache.__class__ = VllmTPUIndexerCache
        # ue8m0 keeps one scale byte per quant block; upstream sizes the row for
        # a 4-byte fp32 scale. Resize in place rather than rebuilding: the cache
        # is already registered in `static_forward_context` under its prefix and
        # a second instance would be rejected as a duplicate. `get_kv_cache_spec`
        # reads `head_dim` lazily, so this lands before the runner collects specs.
        indexer.k_cache.head_dim = (
            indexer.head_dim + indexer.head_dim // indexer.quant_block_size
        )
        return cast("VllmTPUIndexer", indexer)

    def _project_k(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(k_pe, k_nope, weights)`: K normed and split, before the rope."""
        # Fused wk + weights_proj: one GEMM, then split
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        return k_pe, k_nope, weights

    def k_packed(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, rotary_emb
    ) -> torch.Tensor:
        """The packed fp8 + e8m0 K rows `forward` hands the indexer op to
        write into the indexer cache, without the q path (the MTP KV-only
        pass). k_pe is rotated alone, as in
        `VllmTPUMultiHeadLatentAttentionWrapper.kv_latents`."""
        k_pe, k_nope, _ = self._project_k(hidden_states)
        k_pe = k_pe.unsqueeze(1)
        _, k_pe = rotary_emb(positions, k_pe.clone(), k_pe)
        assert k_pe is not None
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)
        return VllmTPUSparseAttnIndexer.pack_k(k)

    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        k_pe, k_nope, weights = self._project_k(hidden_states)

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # Note: RoPE (NeoX) can introduce extra leading dimensions during
        # compilation so we need to reshape back to token-flattened shapes
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        # `rotary_emb` is shape-preserving; `q_pe` is already
        # [num_tokens, n_head, rope_dim].
        q = torch.cat([q_pe, q_nope], dim=-1)
        # `k_pe` is [num_tokens, 1, rope_dim] (MQA).
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

        # we only quant q here since k quant is fused with cache insertion
        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = quantize_tensor(
            q, torch.float8_e4m3fn, axis=-1, use_ue8m0=False
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        # q_scale is fp32, so fold it in and cast back: the kernel expects the
        # weights in the activation dtype.
        weights_dtype = weights.dtype
        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
        )
        weights = weights.squeeze(-1).to(weights_dtype)

        return self.indexer_op(hidden_states, q_fp8, k, weights)


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
        dcp_q_replicate: bool = False,
        use_sparse: bool = False,
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        **extra_impl_args,
    ):
        torch.nn.Module.__init__(self)

        original_mla_get_backend = getattr(
            mla_attention, "get_mla_prefill_backend", None
        )
        original_selector_get_backend = getattr(
            selector, "get_mla_prefill_backend", None
        )

        if original_mla_get_backend is not None:
            mla_attention.get_mla_prefill_backend = (
                lambda config: TPUDummyMLAPrefillBackend
            )
        if original_selector_get_backend is not None:
            selector.get_mla_prefill_backend = lambda config: TPUDummyMLAPrefillBackend

        try:
            # Keyword-only: vLLM inserts new positional params into
            # MLAAttention.__init__ (e.g. dcp_q_replicate in v0.26.1rc0),
            # which silently shifts positional args into the wrong slots.
            super().__init__(
                num_heads=num_heads,
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
                **extra_impl_args,
            )
        finally:
            if original_mla_get_backend is not None:
                mla_attention.get_mla_prefill_backend = original_mla_get_backend
            if original_selector_get_backend is not None:
                selector.get_mla_prefill_backend = original_selector_get_backend

        # process_weights_after_loading masks this around upstream's copy.
        self.dcp_q_replicate = dcp_q_replicate
        # The shared top-k table. A layer that reuses an earlier layer's
        # indices has no indexer, so this is where its top-k width is stated.
        self.topk_indices_buffer = topk_indices_buffer

        # For compatibility reasons.
        self.prefix = prefix
        self.kv_sharing_target_layer_name = None
        self.attn_type = AttentionType.DECODER
        self.sliding_window = None

        self.kv_cache_quantized_dtype = None
        if self.kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = TPU_STR_DTYPE_TO_TORCH_DTYPE.get(
                self.kv_cache_dtype.lower().strip()
            )

    def _gather_dcp_qrep_weights(self) -> None:
        """Query replication attends with the whole DCP group's heads, so the
        absorbed W_UK_T has to span them too. Gathered once here rather than
        gathering q every step -- that trade is the point of qrep."""
        self.W_UK_T_dcp_qrep = None
        self.W_UK_T_dcp_qrep_scale = None
        if not self.dcp_q_replicate:
            return
        dcp_group = get_dcp_group()
        if dcp_group is None or int(dcp_group.world_size) <= 1:
            raise RuntimeError(
                "dcp_q_replicate is set but no DCP group is initialized, "
                "so W_UK_T cannot be gathered across the group."
            )
        # Upstream MLAAttention registers W_UK_T_dcp_qrep as a Parameter.
        self.W_UK_T_dcp_qrep = Parameter(
            fresh_buffer(dcp_group.all_gather(self.W_UK_T.contiguous(), dim=0)),
            requires_grad=False,
        )
        if hasattr(self, "W_UK_T_scale"):
            self.W_UK_T_dcp_qrep_scale = Parameter(
                fresh_buffer(
                    dcp_group.all_gather(self.W_UK_T_scale.contiguous(), dim=0)
                ),
                requires_grad=False,
            )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # Upstream `MLAAttention.process_weights_after_loading` reads this
        # weight raw -- `get_and_maybe_dequant_weights(kv_b_proj).T` -- and
        # asserts the [n_out, n_in] shape. The TPU linear method has already
        # flipped it to (k, n), so present the layout upstream expects for the
        # duration of that call. `kv_b_proj`'s parameters are cleared a few
        # lines below once `W_UK_T`/`W_UV` are extracted, and it never runs a
        # forward pass, so this view is the only consumer either way.
        flipped = getattr(self.kv_b_proj, WEIGHT_FLIPPED_ATTR, False)
        kv_b_weight = None
        if flipped:
            kv_b_weight = cast(Parameter, self.kv_b_proj.weight)
            self.kv_b_proj.weight = Parameter(
                kv_b_weight.transpose(0, 1).contiguous(), requires_grad=False
            )
        # Upstream would gather W_UK_T across DCP too; _gather_dcp_qrep_weights
        # does it once, on device.
        dcp_q_replicate, self.dcp_q_replicate = self.dcp_q_replicate, False
        try:
            super().process_weights_after_loading(act_dtype)
        finally:
            self.dcp_q_replicate = dcp_q_replicate
            if flipped:
                assert kv_b_weight is not None
                self.kv_b_proj.weight = kv_b_weight

        device = torch.device("tpu")

        if should_quantize_on_load(self.prefix):
            W_UK_T, w_uk_scale = quantize_tensor(
                self.W_UK_T.to(torch.float32), torch.float8_e4m3fn, axis=1
            )
            W_UV, w_uv_scale = quantize_tensor(
                self.W_UV.to(torch.float32), torch.float8_e4m3fn, axis=1
            )
            self.W_UK_T = Parameter(W_UK_T, requires_grad=False)
            self.W_UV = Parameter(W_UV, requires_grad=False)
            self.W_UK_T_scale = Parameter(
                fresh_buffer(w_uk_scale.to(act_dtype)).to(device), requires_grad=False
            )
            self.W_UV_scale = Parameter(
                fresh_buffer(w_uv_scale.to(act_dtype)).to(device), requires_grad=False
            )

        # move W_UK_T and W_UV into fresh contiguous buffers
        self.W_UK_T = Parameter(fresh_buffer(self.W_UK_T), requires_grad=False)
        self.W_UV = Parameter(fresh_buffer(self.W_UV), requires_grad=False)

        # Keep W_UK_T/W_UV in the activation dtype unless quantized on load via
        # QUANTIZE_ON_LOAD_PREFIXES. There is no quantized bmm for these two, so
        # forward_mla upcasts them back to the activation dtype before torch.bmm
        # anyway; quantizing them on load is strictly a memory-saving technique
        # `kv_cache_quantized_dtype` still governs the KV cache itself.
        self.W_UK_T = Parameter(self.W_UK_T.to(device), requires_grad=False)
        self.W_UV = Parameter(self.W_UV.to(device), requires_grad=False)

        self._gather_dcp_qrep_weights()

        # Safely detach and clear kv_b_proj parameter buffers without breaking
        # PyTorch attribute integrity
        kv_b_proj_params = dict(self.kv_b_proj.named_parameters())
        for key in kv_b_proj_params:
            if key in self.kv_b_proj._parameters:
                self.kv_b_proj._parameters[key] = None
            elif hasattr(self.kv_b_proj, key):
                delattr(self.kv_b_proj, key)

        if self.W_UK_T.device.type == "tpu":
            tensors: list[torch.Tensor] = [self.W_UK_T, self.W_UV]
            if hasattr(self, "W_UK_T_scale"):
                tensors.extend([self.W_UK_T_scale, self.W_UV_scale])
            tensors.extend(
                t
                for t in (self.W_UK_T_dcp_qrep, self.W_UK_T_dcp_qrep_scale)
                if t is not None
            )
            synchronize_tensors(tensors)

        impl = cast(PallasMLAttentionBackendImpl, self.impl)
        q_scale, k_scale, v_scale = impl._get_kv_scales(self)
        self.mla_op = impl._build_mla_op(
            self, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale
        )
        self.sparse_mla_op = impl._build_sparse_mla_op(self)

        dcp_size = getattr(impl, "dcp_size", 1)
        self.sparse_mla_dcp_op = None
        if dcp_size > 1:
            self.sparse_mla_dcp_op = impl._build_sparse_mla_dcp_op(
                self, dcp_size, impl.dcp_interleave_size
            )

    def forward(
        self,
        q: tuple[torch.Tensor, torch.Tensor],
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_metadata, _, kv_cache, _ = get_attention_context(self.layer_name)

        return cast(PallasMLAttentionBackendImpl, self.impl).forward(
            layer=self,
            q=q,
            kv_c_normed=kv_c_normed,
            k_pe=k_pe,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            topk_indices=kwargs.get("topk_indices"),
            q_dcp_replicated=kwargs.get("q_dcp_replicated"),
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
        non_causal_multi_token_decode: bool = False,
        allow_short_prefill_indexer_scoring_skip: bool = False,
        gate_is_fused: bool = False,
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
        self.g_proj = mla_modules.g_proj
        self.gate_is_fused = gate_is_fused
        if self.gate_is_fused:
            assert self.q_lora_rank is not None, "Fused MLA gating requires q_lora_rank"
            assert self.fused_qkv_a_proj is not None, (
                "Fused MLA gating requires fused_qkv_a_proj"
            )
            assert self.g_proj is None, "Fused MLA gating cannot also use g_proj"
        self.kimi_preprocess = None
        self.register_buffer("kimi_gate_weight", None, persistent=False)
        # `Indexer` has no `register_oot` hook and is fully built by
        # `DeepseekV2MLAAttention` before this wrapper runs, so retype it in
        # place. The mutation is visible through every reference to the object,
        # including `DeepseekV2MLAAttention.indexer`.
        indexer = mla_modules.indexer
        if indexer is not None:
            indexer = VllmTPUIndexer.rebind(cast(Indexer, indexer))
        self.indexer = indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        self.skip_topk = skip_topk
        self.topk_indices_buffer = mla_modules.topk_indices_buffer

        if self.indexer is not None and not self.skip_topk:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens

        # DCP query replication: `q_proj` is sharded across DCP *groups*, so it
        # returns the group's whole head set and decode needs no q all-gather.
        q_proj_layer = self._q_proj_layer()
        self.dcp_q_replicate = bool(getattr(q_proj_layer, "qrep_active", False))
        self.dcp_group_size = (
            int(getattr(q_proj_layer, "group_size", 1)) if self.dcp_q_replicate else 1
        )

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
            kv_b_proj=cast(ColumnParallelLinear, self.kv_b_proj),
            dcp_q_replicate=self.dcp_q_replicate and self.is_sparse,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=self.topk_indices_buffer,
            non_causal_multi_token_decode=non_causal_multi_token_decode,
        )

        self.prefix = prefix

    def _project_qkv_a(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """`(q_c, kv_lora, output_gate)`: the latent projection, before any
        norm. `q_c` is None without q_lora, `output_gate` unless fused."""
        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            if self.gate_is_fused:
                q_c, kv_lora, output_gate = qkv_lora.split(
                    [
                        self.q_lora_rank,
                        self.kv_lora_rank + self.qk_rope_head_dim,
                        self.num_heads * self.v_head_dim,
                    ],
                    dim=-1,
                )
                return q_c, kv_lora, output_gate
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            return q_c, kv_lora, None
        assert self.kv_a_proj_with_mqa is not None, (
            "kv_a_proj_with_mqa is required when q_lora_rank is None"
        )
        return None, self.kv_a_proj_with_mqa(hidden_states)[0], None

    def _split_kv_lora(
        self, kv_lora: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(kv_c_normed, k_pe)` of the latent projection, k_pe not rotated."""
        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        return self.kv_a_layernorm(kv_c), k_pe

    def kv_latents(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(kv_c_normed, k_pe)` exactly as `forward` hands them to the MLA
        attention, without the q path (the MTP KV-only pass).

        The q columns of `fused_qkv_a_proj` are computed and dropped, as in
        `forward`. The rope is applied to k_pe alone: it doubles as the query
        argument through a copy whose result is dropped, and a rope's key
        output does not depend on the query.
        """
        assert self.kimi_preprocess is None, "the Kimi fused preprocess has no k path"
        _, kv_lora, _ = self._project_qkv_a(hidden_states)
        kv_c_normed, k_pe = self._split_kv_lora(kv_lora)
        k_pe = k_pe.unsqueeze(1)
        if self.rotary_emb is not None:
            k_pe_dtype = k_pe.dtype
            _, k_pe = self.rotary_emb(positions, k_pe.clone(), k_pe)
            assert k_pe is not None
            k_pe = k_pe.to(dtype=k_pe_dtype)
        return kv_c_normed, k_pe

    def _q_proj_layer(self):
        """The projection that produces `q`, whichever of the two is wired."""
        return self.q_b_proj if self.q_lora_rank is not None else self.q_proj

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if hasattr(super(), "process_weights_after_loading"):
            # torch's stubs resolve this hasattr-guarded method as a Tensor.
            # pyrefly: ignore[not-callable]
            super().process_weights_after_loading(act_dtype)
        if self.rotary_emb is not None and hasattr(self.rotary_emb, "cos_sin_cache"):
            device = torch.device("tpu")
            self.rotary_emb.cos_sin_cache = self.rotary_emb.cos_sin_cache.to(device)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        prefill_group=None,
    ) -> torch.Tensor:
        if self.kimi_preprocess is not None:
            assert self.fused_qkv_a_proj is not None
            assert self.q_a_layernorm is not None
            assert self.q_b_proj is not None
            q_c, kv_c_normed, k_pe, output_gate = self.kimi_preprocess(
                hidden_states,
                self.fused_qkv_a_proj.weight,
                self.kimi_gate_weight,
                self.q_a_layernorm.weight,
                self.kv_a_layernorm.weight,
                prefill_group is not None,
            )
            q = self.q_b_proj(q_c)[0]
        else:
            q_c, kv_lora, output_gate = self._project_qkv_a(hidden_states)
            if q_c is not None:
                assert self.q_a_layernorm is not None, (
                    "q_a_layernorm is required when q_lora_rank is not None"
                )
                assert self.q_b_proj is not None, (
                    "q_b_proj is required when q_lora_rank is not None"
                )
                q_c = self.q_a_layernorm(q_c)
                q = self.q_b_proj(q_c)[0]
            else:
                assert self.q_proj is not None, (
                    "q_proj is required when q_lora_rank is None"
                )
                q = self.q_proj(hidden_states)[0]
            kv_c_normed, k_pe = self._split_kv_lora(kv_lora)

        # `dcp_group_size` is 1 unless query replication is on, in which case
        # the projection already returned the whole DCP group's heads.
        q = q.view(-1, self.num_heads * self.dcp_group_size, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q_pe_dtype = q_pe.dtype
            k_pe_dtype = k_pe.dtype
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            # RoPE caches/frequencies can be fp32 and promote the rotated
            # tensors.  Dense MLA requires new KV values to exactly match the
            # cache dtype, and queries should remain in the activation dtype.
            q_pe = q_pe.to(dtype=q_pe_dtype)
            k_pe = k_pe.to(dtype=k_pe_dtype)

        topk_indices = None
        if self.is_sparse:
            if self.indexer is not None and not self.skip_topk:
                topk_index = self.indexer(
                    hidden_states, q_c, positions, self.indexer_rope_emb
                )
            else:
                # reuse the indices an earlier layer wrote into the shared buffer.
                topk_index = self.topk_indices_buffer
            assert topk_index is not None
            topk_indices = topk_index[: hidden_states.shape[0]]

            # print(f"top_k_indices.shape: {topk_indices.shape}, topk_indices: {topk_indices[0]}")  # noqa: E501

        if llama_4_scaling is not None:
            q_nope *= llama_4_scaling
            q_pe *= llama_4_scaling

        # Hand the group-wide q to the sparse DCP path, which would otherwise
        # all-gather the heads itself; everything else takes this rank's shard.
        q_dcp_replicated = None
        if self.dcp_q_replicate:
            if self.is_sparse:
                q_dcp_replicated = (q_nope, q_pe)
            q_proj_layer = cast(DCPGroupColumnParallelLinear, self._q_proj_layer())
            if q_dcp_replicated is not None and topk_indices is not None:
                # The sparse DCP path reads only q_dcp_replicated, so this
                # rank's shard is dead. Slice the same heads on every rank: a
                # rank-dependent slice makes every rank trace a different graph,
                # and precompile peer replay then refuses every peer artifact.
                n = q_nope.shape[-2] // q_proj_layer.group_size
                q_nope, q_pe = q_nope[..., :n, :], q_pe[..., :n, :]
            else:
                local_view = q_proj_layer._local_view
                q_nope, q_pe = local_view(q_nope), local_view(q_pe)

        attn_out = self.mla_attn(
            (q_nope, q_pe),
            kv_c_normed,
            k_pe,
            output_shape=(q.shape[0], self.num_heads * self.v_head_dim),
            topk_indices=topk_indices,
            q_dcp_replicated=q_dcp_replicated,
        )

        if output_gate is not None:
            attn_out *= output_gate.sigmoid()
        elif self.g_proj is not None:
            attn_out *= self.g_proj(hidden_states)[0].sigmoid()
        if prefill_group is not None:
            return prefill_group.project_reduce_scatter(attn_out, self.o_proj)
        return self.o_proj(attn_out)[0]


VllmMultiHeadLatentAttentionWrapper = VllmTPUMultiHeadLatentAttentionWrapper


def mla_kv_write_caches_dcp(
    nope: jax.Array,
    rope: jax.Array,
    idx_cache: jax.Array,
    kv_c: jax.Array,
    k_pe: jax.Array,
    k_packed: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    qsl: jax.Array,
    idx_seq_lens: jax.Array,
    idx_pages: jax.Array,
    idx_cu: jax.Array,
    *,
    mla_kv_spec: tuple[Any, Any],
    mesh: Any,
    dcp_size: int,
    interleave_size: int,
    kv_cache_mode: str | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Write one step's MLA latents and indexer-K rows into DCP position-
    sharded caches (global arrays: caches sharded over the DCP axis, rows and
    metadata replicated), without the attention.

    The MLA half is the shard_map `sparse_mla_attention_dcp` runs around
    `update_sparse_mla_kv_cache_dcp`, each rank's id taken from
    `cp_rank_as_data` as there; the indexer half is the `_cp_scatter_indexer_k`
    call of `_build_streamidx_dcp_op`, with the same writer / layout
    resolution.
    """
    ax = DCP_AXIS_NAME
    nope_spec, rope_spec = mla_kv_spec

    def _local(nope_, rope_, kv_c_, k_pe_, sl_, bt_, qsl_):
        rank = cp_rank_as_data(ax, dcp_size)
        return update_sparse_mla_kv_cache_dcp(
            nope_,
            rope_,
            kv_c_,
            k_pe_,
            sl_,
            bt_,
            qsl_,
            rank,
            nope_spec=nope_spec,
            rope_spec=rope_spec,
            dcp_size=dcp_size,
            interleave_size=interleave_size,
        )

    in_specs = (
        P(ax),
        P(ax),
        P(None, None),
        P(None, None),
        P(None),
        P(None),
        P(None),
    )
    nope, rope = jax.jit(
        jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=(P(ax), P(ax)),
            check_vma=False,
        )
    )(nope, rope, kv_c, k_pe, seq_lens, block_tables, qsl)
    writer, kv_layout = _resolve_indexer_kv_cache_mode(
        idx_cache, k_packed, mode=kv_cache_mode
    )
    idx_cache = _cp_scatter_indexer_k(
        idx_cache,
        k_packed,
        idx_seq_lens,
        idx_pages,
        idx_cu,
        mesh=mesh,
        cp_size=dcp_size,
        interleave_size=interleave_size,
        axis_name=ax,
        writer=writer,
        kv_layout=kv_layout,
    )
    return nope, rope, idx_cache


def mla_kv_write_caches(
    nope: jax.Array,
    rope: jax.Array,
    idx_cache: jax.Array,
    kv_c: jax.Array,
    k_pe: jax.Array,
    k_packed: jax.Array,
    seq_lens: jax.Array,
    block_tables: jax.Array,
    qsl: jax.Array,
    idx_seq_lens: jax.Array,
    idx_pages: jax.Array,
    idx_cu: jax.Array,
    *,
    mla_kv_spec: tuple[Any, Any],
    mesh: Any,
    kv_cache_mode: str | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The unsharded twin of `mla_kv_write_caches_dcp`: the replicated
    shard_map `sparse_mla_attention` runs around `update_sparse_mla_kv_cache`,
    and the indexer append of `_build_streamidx_op` (`indexer_k_append`)."""
    nope_spec, rope_spec = mla_kv_spec

    def _local(nope_, rope_, kv_c_, k_pe_, sl_, bt_, qsl_):
        return update_sparse_mla_kv_cache(
            nope_,
            rope_,
            kv_c_,
            k_pe_,
            sl_,
            bt_,
            qsl_,
            nope_spec=nope_spec,
            rope_spec=rope_spec,
        )

    nope, rope = jax.jit(
        jax.shard_map(
            _local,
            mesh=mesh,
            in_specs=(
                P(None),
                P(None),
                P(None, None),
                P(None, None),
                P(None),
                P(None),
                P(None),
            ),
            out_specs=(P(None), P(None)),
            check_vma=False,
        )
    )(nope, rope, kv_c, k_pe, seq_lens, block_tables, qsl)
    writer, kv_layout = _resolve_indexer_kv_cache_mode(
        idx_cache, k_packed, mode=kv_cache_mode
    )
    idx_cache = indexer_k_append(
        idx_cache,
        k_packed,
        idx_seq_lens,
        idx_pages,
        idx_cu,
        writer=writer,
        kv_layout=kv_layout,
    )
    return nope, rope, idx_cache


class MlaKvWriteOp:
    """Writes one sparse MLA layer's KV and indexer-K rows, nothing else.

    The MTP KV-only pass (`spec_decode/mtp_kv_only.py`) runs it on rows whose
    attention output nobody reads. The latents come from the wrapper's and
    indexer's own projections (`kv_latents`, `k_packed`), and both writers
    are the ones the full pass uses, inside the same partition specs: under
    DCP the caches are sharded over the DCP axis and the rows are replicated,
    and each rank keeps only the rows it owns.

    Built eagerly at load (Dynamo cannot trace an op's construction).
    """

    def __init__(self, wrapper: VllmTPUMultiHeadLatentAttentionWrapper, mesh: Any):
        self.wrapper = wrapper
        # `mla_kv_spec` is attached when the caches are allocated, so the MLA
        # layer is read as `Any`, as `_build_sparse_mla_op` reads it.
        mla_layer: Any = wrapper.mla_attn
        self.mla_layer = mla_layer
        assert isinstance(wrapper.indexer, VllmTPUIndexer)
        self.indexer = wrapper.indexer
        self.indexer_op = self.indexer.indexer_op
        impl = cast(PallasMLAttentionBackendImpl, self.mla_layer.impl)
        self.impl = impl
        dcp_size = int(getattr(impl, "dcp_size", 1))
        if int(getattr(self.indexer_op, "dcp_size", 1)) != dcp_size:
            raise ValueError(
                "the MLA and indexer caches disagree on the DCP size: "
                f"{dcp_size} vs {self.indexer_op.dcp_size}"
            )
        self.dcp_size = dcp_size
        kv_cache_mode = (
            envs.VLLM_TPU_INDEXER_KV_CACHE_MODE
            if envs.is_indexer_kv_cache_mode_explicit()
            else None
        )
        name = mla_layer.layer_name.replace(".", "_")

        def _fake(nope, rope, idx_cache, *rest):
            del rest
            return (
                torch.empty_like(nope),
                torch.empty_like(rope),
                torch.empty_like(idx_cache),
            )

        if dcp_size > 1:
            interleave = int(impl.dcp_interleave_size)
            dcp_mesh = get_or_create_dcp_mesh(axis_name=DCP_AXIS_NAME)
            ax = DCP_AXIS_NAME

            # Fixed arity: torch custom-op schemas take no varargs.
            def _write_dcp_jax(
                nope: jax.Array,
                rope: jax.Array,
                idx_cache: jax.Array,
                kv_c: jax.Array,
                k_pe: jax.Array,
                k_packed: jax.Array,
                seq_lens: jax.Array,
                block_tables: jax.Array,
                qsl: jax.Array,
                idx_seq_lens: jax.Array,
                idx_pages: jax.Array,
                idx_cu: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return mla_kv_write_caches_dcp(
                    nope,
                    rope,
                    idx_cache,
                    kv_c,
                    k_pe,
                    k_packed,
                    seq_lens,
                    block_tables,
                    qsl,
                    idx_seq_lens,
                    idx_pages,
                    idx_cu,
                    # Recorded when the caches are allocated, after this op is
                    # built, so it is read at trace time.
                    mla_kv_spec=mla_layer.mla_kv_spec,
                    mesh=dcp_mesh,
                    dcp_size=dcp_size,
                    interleave_size=interleave,
                    kv_cache_mode=kv_cache_mode,
                )

            self.jax_op = pcp_streaming_jax_op(
                f"pallas::mla_kv_write_dcp_{name}",
                _write_dcp_jax,
                donate_argnums=(0, 1, 2),
                mesh=dcp_mesh,
                input_partition_specs=(P(ax), P(ax), P(ax)) + (P(),) * 9,
                output_partition_specs=(P(ax), P(ax), P(ax)),
            )
        else:

            def _write_jax(
                nope: jax.Array,
                rope: jax.Array,
                idx_cache: jax.Array,
                kv_c: jax.Array,
                k_pe: jax.Array,
                k_packed: jax.Array,
                seq_lens: jax.Array,
                block_tables: jax.Array,
                qsl: jax.Array,
                idx_seq_lens: jax.Array,
                idx_pages: jax.Array,
                idx_cu: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return mla_kv_write_caches(
                    nope,
                    rope,
                    idx_cache,
                    kv_c,
                    k_pe,
                    k_packed,
                    seq_lens,
                    block_tables,
                    qsl,
                    idx_seq_lens,
                    idx_pages,
                    idx_cu,
                    mla_kv_spec=mla_layer.mla_kv_spec,
                    mesh=mesh,
                    kv_cache_mode=kv_cache_mode,
                )

            self.jax_op = pallas.jax_op(
                f"pallas::mla_kv_write_{name}",
                _write_jax,
                donate_argnums=(0, 1, 2),
            )
        self.jax_op.register_fake(_fake)

    def __call__(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        """Write this step's rows; returns k_pe (the program needs an output)."""
        wrapper, layer = self.wrapper, self.mla_layer
        kv_c_normed, k_pe = wrapper.kv_latents(positions, hidden_states)
        k_packed = self.indexer.k_packed(
            hidden_states, positions, wrapper.indexer_rope_emb
        )

        md, _, kv_cache, _ = get_attention_context(layer.layer_name)
        fwd_md = get_forward_context().attn_metadata
        idx_prefix = self.indexer_op.k_cache.prefix
        idx_md = fwd_md.get(idx_prefix) if isinstance(fwd_md, dict) else None
        idx_cache = self.indexer_op.k_cache.kv_cache
        # The memory probe runs before the caches are bound; write nothing.
        if not isinstance(kv_cache, (tuple, list)) or idx_cache.numel() == 0:
            return k_pe
        # Bound caches without metadata would leave readers of these rows
        # with stale entries, i.e. silently wrong results: refuse instead.
        if not isinstance(md, AttentionMetadata) or not isinstance(
            idx_md, AttentionMetadata
        ):
            raise RuntimeError(
                "MLA KV write: no attention metadata for "
                f"{layer.layer_name!r} or {idx_prefix!r}"
            )

        _, k_scale, _ = self.impl._get_kv_scales(layer)
        if layer.kv_cache_quantized_dtype is not None:
            assert k_scale is not None
            kv_c_normed, _ = quantize_kv(
                layer.kv_cache_quantized_dtype, kv_c_normed, value=None, k_scale=k_scale
            )
            k_pe_q, _ = quantize_kv(
                layer.kv_cache_quantized_dtype, k_pe, value=None, k_scale=k_scale
            )
        else:
            k_pe_q = k_pe
        nope, rope = kv_cache
        new_nope, new_rope, new_idx = self.jax_op(
            nope,
            rope,
            idx_cache,
            kv_c_normed.view(-1, layer.kv_lora_rank),
            k_pe_q.view(-1, layer.qk_rope_head_dim),
            k_packed,
            md.seq_lens,
            md.block_tables,
            md.query_start_loc,
            idx_md.seq_lens,
            idx_md.block_tables,
            idx_md.query_start_loc,
        )
        nope.copy_(new_nope)
        rope.copy_(new_rope)
        idx_cache.copy_(new_idx)
        return k_pe


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
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]

        pos = torch.add(positions, offsets) if offsets is not None else positions
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
            query = torch.cat((query_rot, query[..., self.rotary_dim :]), dim=-1)
            key = torch.cat((key_rot, key[..., self.rotary_dim :]), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key
