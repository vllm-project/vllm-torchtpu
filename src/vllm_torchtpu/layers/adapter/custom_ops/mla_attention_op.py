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
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import rotate_gptj, rotate_neox
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache, Indexer
from vllm.model_executor.utils import replace_parameter
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
from vllm_torchtpu.kernels.mla.kv_cache_utils import OOB_PAGE, _row_positions
from vllm_torchtpu.layers.adapter.attention import (
    TPU_STR_DTYPE_TO_TORCH_DTYPE,
    PallasMLAttentionBackendImpl,
    VllmTPUDeepseekV32IndexerBackend,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    FP8_PALLAS_SCALE,
    FP8_PALLAS_WEIGHT,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.adapter.quantization.configs import should_quantize_on_load
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.quantization import quantize_tensor
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

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
        skip_empty_decode = envs.TPU_STREAMIDX_SKIP_EMPTY_DECODE

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
            if writer == "xla":
                _, block_size_per_kv_packing, kv_packing, lkv_dim = cache_kv.shape
                block_size = block_size_per_kv_packing * kv_packing
                # Cache rows are padded to the 128-lane HBM width; pad the packed
                # fp8+scale rows to match before scattering.
                k_packed = jnp.pad(
                    k_packed, ((0, 0), (0, lkv_dim - k_packed.shape[-1]))
                )

                # Insert this step's K rows into the KV cache.
                tok = jnp.arange(k_packed.shape[0], dtype=jnp.int32)
                seq_id = jnp.searchsorted(cu_q_lens[1:], tok, side="right").astype(
                    jnp.int32
                )
                q_len = cu_q_lens[seq_id + 1] - cu_q_lens[seq_id]
                pos = seq_lens[seq_id] - q_len + (tok - cu_q_lens[seq_id])
                page = jnp.where(
                    tok < cu_q_lens[-1],
                    page_indices.reshape(seq_lens.shape[0], -1)[
                        seq_id, pos // block_size
                    ],
                    jnp.int32(2**30),
                )
                slot = pos % block_size
                cache_kv = cache_kv.at[page, slot // kv_packing, slot % kv_packing].set(
                    k_packed
                )
            else:
                cache_kv = coalesced_kv_append(
                    cache_kv,
                    k_packed,
                    seq_lens,
                    page_indices,
                    cu_q_lens,
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
                skip_empty_decode=skip_empty_decode,
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
        skip_empty_decode = envs.TPU_STREAMIDX_SKIP_EMPTY_DECODE

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
                skip_empty_decode=skip_empty_decode,
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

        k_quant, k_scale = quantize_tensor(
            k,
            torch.float8_e4m3fn,
            axis=-1,
            use_ue8m0=True,
        )
        # Pack per-token fp8 values + 1-byte e8m0 scale into the uint8 row
        # layout the TPU kernel unpacks (head_dim value bytes + 1 scale byte).
        # cat requires a single dtype, so view the fp8 values as raw bytes.
        k_quant_packed = torch.cat([k_quant.view(torch.uint8), k_scale], dim=-1)

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

        rows = self._write_topk_rows(hidden_states.shape[0], topk_indices)
        if envs.TPU_INDEXER_TOPK_SHARED_READ:
            # Read back what the skip_topk layers will read, so the attention
            # index prep is one CSE class across them.
            return self.topk_indices_buffer
        return rows

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

    def permute_rope_to_half_split(self) -> None:
        """TPU_INDEXER_ROPE_NEOX_PERMUTE: put each head's rope columns of
        `wq_b`, the k rope columns of `wk_weights_proj` and the matching
        `k_norm` affine entries in half-split order (see
        `mla_rope_half_split_perm`); `forward` then runs the rope NeoX-style.

        The rope part is the first `rope_dim` channels of each `head_dim` head
        (q) and of k. `k_norm` is a LayerNorm over all `head_dim` k channels:
        its mean / variance do not depend on channel order, so permuting its
        per-channel weight and bias with the same index keeps it exact. q and
        k are fp8-quantized per `head_dim` row (one block), whose amax is order
        independent, and the cache stores P k consistently. Runs on the
        processed runtime weights; idempotent.
        """
        if getattr(self, "_rope_neox_permuted", False):
            return
        rope, hd, nh = self.rope_dim, self.head_dim, self.n_head
        permute_linear_output_channels(
            self.wq_b, mla_rope_output_index(nh * hd, rope, hd, 0, nh)
        )
        kw = self.wk_weights_proj
        flipped = getattr(kw, WEIGHT_FLIPPED_ATTR, False)
        kw_n_out = cast(torch.Tensor, kw.weight).shape[1 if flipped else 0]
        assert kw_n_out == hd + nh, (kw_n_out, hd, nh)
        permute_linear_output_channels(
            kw, mla_rope_output_index(kw_n_out, rope, 0, 0, 1)
        )
        norm_index = mla_rope_output_index(hd, rope, 0, 0, 1)
        updated = []
        for name, param in list(self.k_norm.named_parameters(recurse=False)):
            if param.ndim != 1 or param.shape[0] != hd:
                raise NotImplementedError(
                    f"indexer rope permutation: k_norm.{name} "
                    f"{tuple(param.shape)} is not per k channel"
                )
            new = _index_select_bits(param.data, 0, norm_index)
            replace_parameter(self.k_norm, name, Parameter(new, requires_grad=False))
            updated.append(getattr(self.k_norm, name))
        if updated and updated[0].device.type == "tpu":
            synchronize_tensors(updated)
        self._rope_neox_permuted = True

    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        # Fused wk + weights_proj: one GEMM, then split
        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        permuted = getattr(self, "_rope_neox_permuted", False)
        if permuted and rope_half_split_leading_supported(rotary_emb):
            # Half-split columns: rope the whole head in place, no split/cat.
            q = rope_half_split_leading(rotary_emb, positions, q, self.rope_dim)
            k = rope_half_split_leading(rotary_emb, positions, k, self.rope_dim)
        else:
            q_pe, q_nope = torch.split(
                q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )
            k_pe, k_nope = torch.split(
                k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
            )
            if permuted:
                q_pe, k_pe = rope_half_split(
                    rotary_emb, positions, q_pe, k_pe.unsqueeze(1)
                )
            else:
                q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
            # A key was passed in, so one comes back.
            assert k_pe is not None
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


def _fresh(t: torch.Tensor) -> torch.Tensor:
    """Allocate a fresh contiguous device buffer (breaks view/stride chains so
    the torch_tpu pallas boundary ships a plain row-major buffer)."""
    return torch.empty(t.shape, dtype=t.dtype, device=t.device).copy_(t)


# --- MLA rope in half-split order (TPU_MLA_ROPE_NEOX_PERMUTE) ---------------
#
# GPT-J (interleaved) rope pairs channels (2i, 2i+1); NeoX rope pairs
# (i, i + d/2). With P = [0, 2, ..., d-2, 1, 3, ..., d-1] and new[j] = old[P[j]],
# neox(P x) = P gptj(x) elementwise-exactly, and (P q).(P k) = q.k. So permuting
# the rope output columns of the q projection and of the k_pe projection by
# the same P, and running the rope NeoX-style, leaves every attention score
# unchanged while the KV cache holds P k_pe consistently. The win is layout:
# the `x[..., ::2]` split forces a tokens-minor relayout of q on TPU.


def mla_rope_half_split_perm(rope_dim: int) -> torch.Tensor:
    """`P` above: the old channel each new channel reads (evens, then odds)."""
    assert rope_dim % 2 == 0, f"rope dim {rope_dim} must be even"
    return torch.cat([torch.arange(0, rope_dim, 2), torch.arange(1, rope_dim, 2)])


def mla_rope_output_index(
    n_out: int, rope_dim: int, period: int, offset: int, count: int
) -> torch.Tensor:
    """A gather index over `n_out` output channels that applies `P` to `count`
    rope segments at `offset + i * period`, identity elsewhere."""
    index = torch.arange(n_out)
    perm = mla_rope_half_split_perm(rope_dim)
    for i in range(count):
        start = offset + i * period
        assert start + rope_dim <= n_out, (start, rope_dim, n_out)
        index[start : start + rope_dim] = start + perm
    return index


_SAME_WIDTH_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _index_select_bits(t: torch.Tensor, axis: int, index: torch.Tensor):
    """`index_select` that moves raw bits, so fp8 / bf16 / fp32 are all exact
    and no dtype-specific gather support is needed on the device."""
    as_int = t.view(_SAME_WIDTH_INT[t.element_size()])
    out = torch.index_select(as_int, axis, index.to(t.device))
    return _fresh(out.view(t.dtype))


def permute_linear_output_channels(layer: torch.nn.Module, index: torch.Tensor):
    """Reorder a processed linear layer's output channels: `out'[j] = out[index[j]]`.

    Runs after the quant method's `process_weights_after_loading`, so it reads
    the runtime layout: `weight` is `[n_in, n_out]` when flipped
    (`WEIGHT_FLIPPED_ATTR`) and `[n_out, n_in]` otherwise; any other parameter
    (FP8 `weight_scale` `[1, n_blocks, 1, n_out]` / `[n_out]`, a bias) is
    per output channel on its last axis, else on axis 0, else a per-tensor
    scalar. Anything else is a layout this does not know and raises.
    """
    n_out = index.numel()
    flipped = getattr(layer, WEIGHT_FLIPPED_ATTR, False)
    updated = []
    for name, param in list(layer.named_parameters(recurse=False)):
        t = param.data
        if name in (FP8_PALLAS_WEIGHT, FP8_PALLAS_SCALE):
            # TPU_FP8_LINEAR_PALLAS's zero-padded [.., n_pad] kernel copy: permute
            # the real channels, leave the pad channels where they are.
            pad = torch.arange(n_out, t.shape[-1], dtype=index.dtype)
            new = _index_select_bits(t, t.ndim - 1, torch.cat([index, pad]))
            replace_parameter(layer, name, Parameter(new, requires_grad=False))
            updated.append(getattr(layer, name))
            continue
        if name == "weight":
            axis = 1 if flipped else 0
            if t.ndim != 2 or t.shape[axis] != n_out:
                raise NotImplementedError(
                    f"rope permutation: weight {tuple(t.shape)} (flipped="
                    f"{flipped}) has no {n_out}-wide output axis"
                )
        elif t.ndim >= 1 and t.shape[-1] == n_out:
            axis = t.ndim - 1
        elif t.ndim >= 1 and t.shape[0] == n_out:
            axis = 0
        elif t.numel() == 1:
            continue
        else:
            raise NotImplementedError(
                f"rope permutation: parameter {name} {tuple(t.shape)} is not "
                f"per output channel of a {n_out}-wide layer"
            )
        new = _index_select_bits(t, axis, index)
        replace_parameter(layer, name, Parameter(new, requires_grad=False))
        updated.append(getattr(layer, name))
    if updated and updated[0].device.type == "tpu":
        synchronize_tensors(updated)


def rope_half_split(rotary_emb, positions, query, key):
    """`rotary_emb`'s own arithmetic, NeoX-style, without touching the module
    (`get_rope` shares one instance across every layer that asks for it)."""
    if isinstance(rotary_emb, DeepseekScalingRotaryEmbedding):
        return VllmDeepseekScalingRotaryEmbedding.apply_rope(
            rotary_emb, positions, query, key, is_neox_style=True
        )
    return RotaryEmbedding.forward_static(
        positions,
        query,
        key,
        rotary_emb.head_size,
        rotary_emb.rotary_dim,
        rotary_emb._match_cos_sin_cache_dtype(query),
        True,
    )


def rope_half_split_leading(
    rotary_emb, positions: torch.Tensor, x: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    """NeoX rope on the leading `rope_dim` channels of `x` `[T, ..., D]`, the
    other `D - rope_dim` channels passed through, as ONE full-width
    elementwise expression: `x * C + rot(x) * S` with
    `C = [cos, cos, 1...]`, `S = [-sin, sin, 0...]` and
    `rot(x) = [x[h:2h], x[:h], x[2h:]]` (h = rope_dim / 2).

    Same products and the same add per channel as `rope_half_split`
    (`x1 * cos - x2 * sin == x1 * cos + x2 * (-sin)`, pass-through is
    `x * 1 + y * 0`), but no split / cat of the head: XLA sees a single
    D-wide fusion instead of half-width rope pieces it has to lay out and
    copy back. Only for a plain `RotaryEmbedding` (cache in `x`'s dtype)."""
    t, d, h = x.shape[0], x.shape[-1], rope_dim // 2
    cos_sin = rotary_emb._match_cos_sin_cache_dtype(x).index_select(
        0, positions.flatten()
    )
    cos, sin = cos_sin.chunk(2, dim=-1)
    parts_c, parts_s = [cos, cos], [-sin, sin]
    if d > rope_dim:
        parts_c.append(cos.new_ones(t, d - rope_dim))
        parts_s.append(sin.new_zeros(t, d - rope_dim))
    bcast = (t,) + (1,) * (x.ndim - 2) + (d,)
    c = torch.cat(parts_c, dim=-1).view(bcast)
    s_ = torch.cat(parts_s, dim=-1).view(bcast)
    rot = torch.cat([x[..., h:rope_dim], x[..., :h], x[..., rope_dim:]], dim=-1)
    return x * c + rot * s_


def rope_half_split_leading_supported(rotary_emb) -> bool:
    return not isinstance(rotary_emb, DeepseekScalingRotaryEmbedding) and isinstance(
        rotary_emb, RotaryEmbedding
    )


def rope_half_split_supported(rotary_emb, rope_dim: int) -> bool:
    if rotary_emb is None or getattr(rotary_emb, "is_neox_style", True):
        return False
    if rotary_emb.rotary_dim != rope_dim or rotary_emb.head_size != rope_dim:
        return False
    return isinstance(rotary_emb, DeepseekScalingRotaryEmbedding) or (
        isinstance(rotary_emb, RotaryEmbedding)
        and type(rotary_emb).forward_native is RotaryEmbedding.forward_native
    )


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

        # Set after super(), not through it: upstream's
        # process_weights_after_loading gathers W_UK_T on the flag too, and
        # this class does that itself once the weight is on device.
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
            kv_b_weight = cast(torch.Tensor, self.kv_b_proj.weight)
            self.kv_b_proj.weight = Parameter(
                kv_b_weight.transpose(0, 1).contiguous(), requires_grad=False
            )
        try:
            super().process_weights_after_loading(act_dtype)
        finally:
            if flipped:
                assert kv_b_weight is not None
                self.kv_b_proj.weight = Parameter(kv_b_weight, requires_grad=False)

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
                _fresh(w_uk_scale.to(act_dtype)).to(device), requires_grad=False
            )
            self.W_UV_scale = Parameter(
                _fresh(w_uv_scale.to(act_dtype)).to(device), requires_grad=False
            )

        # move W_UK_T and W_UV into fresh contiguous buffers
        self.W_UK_T = Parameter(_fresh(self.W_UK_T), requires_grad=False)
        self.W_UV = Parameter(_fresh(self.W_UV), requires_grad=False)

        # Keep W_UK_T/W_UV in the activation dtype unless quantized on load via
        # QUANTIZE_ON_LOAD_PREFIXES. There is no quantized bmm for these two, so
        # forward_mla upcasts them back to the activation dtype before torch.bmm
        # anyway; quantizing them on load is strictly a memory-saving technique
        # `kv_cache_quantized_dtype` still governs the KV cache itself.
        self.W_UK_T = Parameter(self.W_UK_T.to(device), requires_grad=False)
        self.W_UV = Parameter(self.W_UV.to(device), requires_grad=False)

        # Query replication attends with the whole DCP group's heads, so the
        # absorbed W_UK_T has to span them too. Gathered once here rather than
        # gathering q every step -- that trade is the point of qrep.
        self.W_UK_T_dcp_qrep = None
        self.W_UK_T_dcp_qrep_scale = None
        self.W_UV_dcp_qrep = None
        self.W_UV_dcp_qrep_scale = None
        if self.dcp_q_replicate:
            dcp_group = get_dcp_group()
            if dcp_group is None or int(dcp_group.world_size) <= 1:
                raise RuntimeError(
                    "dcp_q_replicate is set but no DCP group is initialized, "
                    "so W_UK_T cannot be gathered across the group."
                )
            self.W_UK_T_dcp_qrep = _fresh(
                dcp_group.all_gather(self.W_UK_T.contiguous(), dim=0)
            )
            if hasattr(self, "W_UK_T_scale"):
                self.W_UK_T_dcp_qrep_scale = _fresh(
                    dcp_group.all_gather(self.W_UK_T_scale.contiguous(), dim=0)
                )
            # The group's W_UV too, so W_UV can run before the merge.
            if envs.TPU_DCP_MLA_UV_BEFORE_MERGE:
                self.W_UV_dcp_qrep = _fresh(
                    dcp_group.all_gather(self.W_UV.contiguous(), dim=0)
                )
                if hasattr(self, "W_UV_scale"):
                    self.W_UV_dcp_qrep_scale = _fresh(
                        dcp_group.all_gather(self.W_UV_scale.contiguous(), dim=0)
                    )

        # Safely detach and clear kv_b_proj parameter buffers without breaking
        # PyTorch attribute integrity
        kv_b_proj_params = dict(self.kv_b_proj.named_parameters())
        for key in kv_b_proj_params:
            if key in self.kv_b_proj._parameters:
                self.kv_b_proj._parameters[key] = None
            elif hasattr(self.kv_b_proj, key):
                delattr(self.kv_b_proj, key)

        if self.W_UK_T.device.type == "tpu":
            tensors = [self.W_UK_T, self.W_UV]
            if hasattr(self, "W_UK_T_scale"):
                tensors.extend([self.W_UK_T_scale, self.W_UV_scale])
            tensors.extend(
                t
                for t in (
                    self.W_UK_T_dcp_qrep,
                    self.W_UK_T_dcp_qrep_scale,
                    self.W_UV_dcp_qrep,
                    self.W_UV_dcp_qrep_scale,
                )
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

        # Post-load work the owning wrapper registered: it holds the q / kv_a
        # projections, but is not itself visited by the loader's attention
        # pass, which runs after every linear has been quantized and flipped.
        for hook in getattr(self, "post_load_hooks", ()):
            hook()

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

        # TPU_MLA_ROPE_NEOX_PERMUTE: set once the q / k_pe projection columns
        # are in half-split order, which is what switches `forward` to NeoX
        # rope -- the two can never disagree.
        self._rope_neox_permuted = False
        # Both permutes are on by default; a rope they cannot reorder keeps
        # the interleaved layout unless the env was set explicitly.
        if envs.TPU_MLA_ROPE_NEOX_PERMUTE:
            if rope_half_split_supported(self.rotary_emb, self.qk_rope_head_dim):
                hooks = self.mla_attn.__dict__.setdefault("post_load_hooks", [])
                hooks.append(self.permute_rope_to_half_split)
            elif envs.is_set("TPU_MLA_ROPE_NEOX_PERMUTE"):
                raise NotImplementedError(
                    "TPU_MLA_ROPE_NEOX_PERMUTE needs a GPT-J-style "
                    f"RotaryEmbedding over the whole {self.qk_rope_head_dim}-dim "
                    f"rope head; got {type(self.rotary_emb).__name__}"
                )
        if envs.TPU_INDEXER_ROPE_NEOX_PERMUTE and self.indexer is not None:
            if rope_half_split_supported(self.indexer_rope_emb, self.indexer.rope_dim):
                hooks = self.mla_attn.__dict__.setdefault("post_load_hooks", [])
                hooks.append(self.indexer.permute_rope_to_half_split)
            elif envs.is_set("TPU_INDEXER_ROPE_NEOX_PERMUTE"):
                neox = getattr(self.indexer_rope_emb, "is_neox_style", None)
                raise NotImplementedError(
                    "TPU_INDEXER_ROPE_NEOX_PERMUTE needs a GPT-J-style "
                    f"RotaryEmbedding over the whole {self.indexer.rope_dim}-dim "
                    "indexer rope head (indexer_rope_interleave); got "
                    f"{type(self.indexer_rope_emb).__name__} is_neox_style={neox}"
                )

    def permute_rope_to_half_split(self) -> None:
        """Put the q and k_pe rope columns in half-split order (see
        `mla_rope_half_split_perm`). Runs on the processed runtime weights."""
        if self._rope_neox_permuted:
            return
        if self.kimi_preprocess is not None:
            raise NotImplementedError(
                "TPU_MLA_ROPE_NEOX_PERMUTE: the Kimi fused preprocess reads "
                "fused_qkv_a_proj raw"
            )
        rope = self.qk_rope_head_dim
        q_layer = self._q_proj_layer()
        q_heads = self.num_heads * self.dcp_group_size
        q_index = mla_rope_output_index(
            q_heads * self.qk_head_dim,
            rope,
            period=self.qk_head_dim,
            offset=self.qk_nope_head_dim,
            count=q_heads,
        )
        if self.q_lora_rank is not None:
            kv_layer, kv_offset = (
                self.fused_qkv_a_proj,
                self.q_lora_rank + self.kv_lora_rank,
            )
        else:
            kv_layer, kv_offset = self.kv_a_proj_with_mqa, self.kv_lora_rank
        # q_lora_rank picks which projections exist; the chosen ones are set.
        assert q_layer is not None and kv_layer is not None
        weight = cast(torch.Tensor, kv_layer.weight)
        kv_n_out = weight.shape[
            1 if getattr(kv_layer, WEIGHT_FLIPPED_ATTR, False) else 0
        ]
        kv_index = mla_rope_output_index(kv_n_out, rope, 0, kv_offset, 1)
        permute_linear_output_channels(q_layer, q_index)
        permute_linear_output_channels(kv_layer, kv_index)
        self._rope_neox_permuted = True

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
        q_c = None
        kv_lora = None
        output_gate = None

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
            if self.q_lora_rank is not None:
                assert self.fused_qkv_a_proj is not None, (
                    "fused_qkv_a_proj is required when q_lora_rank is not None"
                )
                assert self.q_a_layernorm is not None, (
                    "q_a_layernorm is required when q_lora_rank is not None"
                )
                assert self.q_b_proj is not None, (
                    "q_b_proj is required when q_lora_rank is not None"
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
                else:
                    q_c, kv_lora = qkv_lora.split(
                        [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                        dim=-1,
                    )
                q_c = self.q_a_layernorm(q_c)
                q = self.q_b_proj(q_c)[0]
            else:
                assert self.kv_a_proj_with_mqa is not None, (
                    "kv_a_proj_with_mqa is required when q_lora_rank is None"
                )
                assert self.q_proj is not None, (
                    "q_proj is required when q_lora_rank is None"
                )
                kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
                q = self.q_proj(hidden_states)[0]

            assert kv_lora is not None
            kv_c, k_pe = kv_lora.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_c_normed = self.kv_a_layernorm(kv_c)

        # `dcp_group_size` is 1 unless query replication is on, in which case
        # the projection already returned the whole DCP group's heads.
        q = q.view(-1, self.num_heads * self.dcp_group_size, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        if self.rotary_emb is not None:
            q_pe_dtype = q_pe.dtype
            k_pe_dtype = k_pe.dtype
            if getattr(self, "_rope_neox_permuted", False):
                q_pe, k_pe = rope_half_split(self.rotary_emb, positions, q_pe, k_pe)
            else:
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
            local_view = cast(
                DCPGroupColumnParallelLinear, self._q_proj_layer()
            )._local_view
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
        return self.apply_rope(
            self, positions, query, key, offsets, is_neox_style=self.is_neox_style
        )

    @staticmethod
    def apply_rope(
        rope: DeepseekScalingRotaryEmbedding,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
        *,
        is_neox_style: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert key is not None
        query_rot = query[..., : rope.rotary_dim]
        key_rot = key[..., : rope.rotary_dim]

        pos = torch.add(positions, offsets) if offsets is not None else positions
        cos_sin = rope.cos_sin_cache.index_select(0, pos)
        cos, sin = cos_sin.chunk(2, dim=-1)
        if is_neox_style:
            cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
            sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = rotate_neox if is_neox_style else rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if rope.rotary_dim < rope.head_size:
            query = torch.cat((query_rot, query[..., rope.rotary_dim :]), dim=-1)
            key = torch.cat((key_rot, key[..., rope.rotary_dim :]), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key
