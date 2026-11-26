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

import functools
from typing import Any, cast

import jax
import jax.numpy as jnp
import torch
from torch.nn import Parameter
from torch_tpu._internal import pallas
from vllm.compilation import backends as vllm_compilation_backends
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
from vllm.model_executor.layers.rotary_embedding.common import (
    ApplyRotaryEmb,
    rotate_gptj,
    rotate_neox,
)
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
from vllm_torchtpu.distributed.dcp import (
    dcp_lane_operand,
    dcp_rank_operand,
    get_dcp_group,
    get_dcp_rank,
    get_dcp_world_size,
    get_or_create_dcp_mesh,
)
from vllm_torchtpu.kernels import indexer_qprep
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    DCP_AXIS_NAME,
    KVLayout,
    coalesced_kv_append,
    coalesced_kv_append_dcp,
    cp_rank_as_data,
    get_kv_cache_shape,
    kv_cache_append,
    streamindex_topk,
    streamindex_topk_dcp,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
    pcp_streaming_jax_op,
)
from vllm_torchtpu.kernels.mla.fused_qb_absorb import (
    HEAD_BLOCK as _FUSED_QB_HEAD_BLOCK,
)
from vllm_torchtpu.kernels.mla.fused_qb_absorb import (
    fused_qb_rope_absorb_split,
    fused_qb_rope_absorb_split_halves,
    supports_fused_qb_rope_absorb_split,
)
from vllm_torchtpu.kernels.mla.kv_cache_utils import OOB_PAGE, _row_positions
from vllm_torchtpu.kernels.quantized_matmul import fp8_skinny
from vllm_torchtpu.kernels.quantized_matmul import (
    fp8_window_all_gather as fp8_window_all_gather_mod,
)
from vllm_torchtpu.layers.adapter import (
    dcp_nonidx_qb_mla_op,
    dcp_token_parallel_op,
    dcp_unified_idx_qb_mla_op,
    dcp_unified_indexer_mla_op,
    small_all_reduce_op,
)
from vllm_torchtpu.layers.adapter.attention import (
    TPU_STR_DTYPE_TO_TORCH_DTYPE,
    PallasMLAttentionBackendImpl,
    VllmTPUDeepseekV32IndexerBackend,
    resolve_indexer_kv_layout,
)
from vllm_torchtpu.layers.adapter.cp_mla_attention import (
    local_q_all_gather_op,
    local_q_norm_all_gather_op,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    DCP_ROWS_ATTR,
    FP8_PALLAS_N_ATTR,
    FP8_PALLAS_SCALE,
    FP8_PALLAS_WEIGHT,
    KEEP_VLLM_LAYOUT_ATTR,
    PREQUANT_ATTR,
    WEIGHT_FLIPPED_ATTR,
    _fp8_linear_pallas_hbm_pinned_jax,
    _fp8_linear_pallas_jax,
    retain_input,
)
from vllm_torchtpu.layers.adapter.quantization.configs import should_quantize_on_load
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8LinearMethodTPU,
    fp8_skinny_hbm_pin_selected,
    split_fp8_linear_columns,
)
from vllm_torchtpu.layers.adapter.rotary_embedding import TpuRotaryEmbedding
from vllm_torchtpu.layers.adapter.small_all_reduce_op import row_parallel_or
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.layers.core.quantization import quantize_tensor
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.glm_dsa_dispatch import (
    INDEXER_QPREP_MIN_TOKENS,
    MLA_SPLIT_QKV_A_CHUNK,
    STREAMIDX_DEFAULT_MIXED_KV_PAGES,
    bucket_gate_edge,
    glm_dsa_v7x_active,
    in_shape_variants_warmup,
    set_active_model_config,
)
from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

logger = init_logger(__name__)

_GLM_STREAMIDX_DEFAULT_SMALL_BQ = 4


def _retain_buffers(out: torch.Tensor, *buffers: torch.Tensor | None) -> torch.Tensor:
    if not in_shape_variants_warmup():
        return out
    for buf in buffers:
        if buf is not None:
            out = retain_input(out, buf)
    return out


def _row_parallel_or_default(attn_out: torch.Tensor, o_proj: Any) -> torch.Tensor:
    if glm_dsa_v7x_active():
        return cast(torch.Tensor, row_parallel_or(attn_out, o_proj))
    return o_proj(attn_out)[0]


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
STREAMIDX_PACK_BKVP_CAP_TOKENS = 8
STREAMIDX_PACK_BKVP = (32, 16)


def _streamidx_small_bucket(
    num_tokens: int, small_bucket_bq: int | None, small_bucket_tokens: int
) -> bool:
    return small_bucket_bq is not None and num_tokens <= small_bucket_tokens


def _streamidx_block_sizes(
    num_tokens: int,
    small_bucket_bq: int | None,
    small_bucket_bkvp: int | None,
    small_bucket_tokens: int,
    mixed_bkvp: int | None = None,
    decode_pack: bool = False,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """(num_kv_pages_per_block, num_queries_per_block) for a padded token bucket.

    MIXED-pass query blocks hold one sequence each, so an MTP request
    (q_len = gamma + 1) in a 128-row block scores ~124 padding rows. Buckets up
    to `small_bucket_tokens` take the small-bucket sizes instead.
    """
    if not _streamidx_small_bucket(num_tokens, small_bucket_bq, small_bucket_tokens):
        if mixed_bkvp is None:
            return STREAMIDX_KV_PAGES_PER_BLOCK, STREAMIDX_QUERIES_PER_BLOCK
        return (
            STREAMIDX_KV_PAGES_PER_BLOCK[:2] + (mixed_bkvp,),
            STREAMIDX_QUERIES_PER_BLOCK,
        )
    if small_bucket_bkvp:
        bkvp = small_bucket_bkvp
    elif decode_pack:
        bkvp = STREAMIDX_PACK_BKVP[num_tokens > STREAMIDX_PACK_BKVP_CAP_TOKENS]
    else:
        bkvp = STREAMIDX_SMALL_BUCKET_DEFAULT_BKVP
    return (
        STREAMIDX_KV_PAGES_PER_BLOCK[:2] + (bkvp,),
        STREAMIDX_QUERIES_PER_BLOCK[:2] + (cast(int, small_bucket_bq),),
    )


def _qb_window_gather_jax(
    x,
    w_q,
    w_s,
    rank,
    lane,
    *,
    num_cores: int,
    axis: str,
    lane_axis: str | None,
    replicated,
):
    m, k = x.shape
    if not fp8_window_all_gather_mod.supports(m, k, w_q.shape[1], num_cores):
        return replicated(x, w_q, w_s)
    return fp8_window_all_gather_mod.fp8_window_all_gather(
        x,
        w_q,
        w_s,
        rank,
        num_cores=num_cores,
        axis=axis,
        lane_axis=lane_axis,
        lane=None if lane_axis is None else lane,
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
    kv_layout = resolve_indexer_kv_layout(block_size, mode=mode)
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


def _dcp_indexer_a2a_for_bucket(
    num_tokens: int, enabled: bool, min_tokens: int
) -> bool:
    return enabled and num_tokens > bucket_gate_edge(min_tokens)


def _qb_overlap_operands(layer: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not isinstance(getattr(layer, "quant_method", None), VllmFp8LinearMethodTPU):
        return None
    if getattr(layer, FP8_PALLAS_N_ATTR, None) is None:
        return None
    if getattr(layer, KEEP_VLLM_LAYOUT_ATTR, False):
        return None
    if getattr(layer, "bias", None) is not None:
        return None
    if hasattr(layer, FP8_PALLAS_WEIGHT):
        return None
    if getattr(layer, "gather_output", False):
        return None
    return layer.weight, layer.weight_scale


def _fused_qb_operands(layer: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return per-channel FP8 `(weight, weight_scale)` for fused q_b_proj, or None."""
    weight = getattr(layer, "weight", None)
    scale = getattr(layer, "weight_scale", None)
    if (
        weight is None
        or scale is None
        or not isinstance(weight, torch.Tensor)
        or not isinstance(scale, torch.Tensor)
    ):
        return None
    if (
        weight.dtype != torch.float8_e4m3fn
        or scale.ndim != 1
        or scale.shape[0] != weight.shape[1]
        or not getattr(layer, WEIGHT_FLIPPED_ATTR, False)
        or getattr(layer, KEEP_VLLM_LAYOUT_ATTR, False)
        or getattr(layer, "bias", None) is not None
        or getattr(layer, "gather_output", False)
        or hasattr(layer, FP8_PALLAS_WEIGHT)
        or getattr(layer, PREQUANT_ATTR, None) is not None
    ):
        return None
    return weight, scale


def _fake_fused_qb_rope_absorb_split(
    q_c: torch.Tensor,
    w_qb: torch.Tensor,
    w_s: torch.Tensor,
    w_uk_t: torch.Tensor,
    cos_sin: torch.Tensor,
    split_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del w_qb, w_s
    heads, _, rank = w_uk_t.shape
    total_tokens = q_c.shape[0]
    d_pe = cos_sin.shape[1]
    return (
        q_c.new_empty((split_rows, heads, rank)),
        q_c.new_empty((total_tokens - split_rows, heads, rank)),
        q_c.new_empty((total_tokens, heads, d_pe)),
    )


_FUSED_QB_ROPE_ABSORB_SPLIT_OP: Any = pallas.jax_op(
    "pallas::mla_fused_qb_rope_absorb_split", fused_qb_rope_absorb_split
)
_FUSED_QB_ROPE_ABSORB_SPLIT_OP.register_fake(_fake_fused_qb_rope_absorb_split)


def _fake_fused_qb_rope_absorb_split_halves(
    q_c: torch.Tensor,
    w_qb: torch.Tensor,
    w_s: torch.Tensor,
    w_uk_t_lo: torch.Tensor,
    w_uk_t_hi: torch.Tensor,
    cos_sin: torch.Tensor,
    split_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del w_qb, w_s
    heads = w_uk_t_lo.shape[0] + w_uk_t_hi.shape[0]
    rank = w_uk_t_lo.shape[2]
    total_tokens = q_c.shape[0]
    d_pe = cos_sin.shape[1]
    return (
        q_c.new_empty((split_rows, heads, rank)),
        q_c.new_empty((total_tokens - split_rows, heads, rank)),
        q_c.new_empty((total_tokens, heads, d_pe)),
    )


_FUSED_QB_ROPE_ABSORB_SPLIT_HALVES_OP: Any = pallas.jax_op(
    "pallas::mla_fused_qb_rope_absorb_split_halves",
    fused_qb_rope_absorb_split_halves,
)
_FUSED_QB_ROPE_ABSORB_SPLIT_HALVES_OP.register_fake(
    _fake_fused_qb_rope_absorb_split_halves
)


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
    rank_ids: jax.Array | None = None,
    compact: bool = False,
    k_sharded: bool = False,
    step_tokens: int | None = None,
    zigzag_bands: int = 1,
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

    if (
        k_sharded
        and writer != "pallas"
        and (step_tokens is None or k_packed.shape[0] == step_tokens)
    ):
        raise ValueError(f"k_sharded requires writer='pallas', got {writer!r}.")

    def _local(cache_kv, k_packed, seq_lens, page_indices, cu_q_lens, *rank_arg):
        rank = rank_arg[0][0] if rank_arg else cp_rank_as_data(axis_name, cp_size)
        if k_sharded:
            if step_tokens is not None and k_packed.shape[0] * cp_size != step_tokens:
                return kv_cache_append.coalesced_owned_kv_append_dcp(
                    cache_kv,
                    k_packed,
                    seq_lens,
                    page_indices,
                    cu_q_lens,
                    rank,
                    num_tokens=step_tokens,
                    dcp_size=cp_size,
                    interleave_size=interleave_size,
                    kv_layout=kv_layout,
                    writer=writer,
                )
            return kv_cache_append.sharded_kv_append_dcp(
                cache_kv,
                k_packed,
                seq_lens,
                page_indices,
                cu_q_lens,
                rank,
                dcp_size=cp_size,
                interleave_size=interleave_size,
                axis_name=axis_name,
                kv_layout=kv_layout,
                zigzag_bands=zigzag_bands,
            )
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
                compact=compact,
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
    k_spec = sharded if k_sharded else replicated
    rank_args = () if rank_ids is None else (rank_ids,)
    return jax.shard_map(
        _local,
        mesh=mesh,
        in_specs=(sharded, k_spec, replicated, replicated, replicated)
        + (sharded,) * len(rank_args),
        out_specs=sharded,
        check_vma=False,
    )(cache_kv, k_packed, seq_lens, page_indices, cu_q_lens, *rank_args)


@SparseAttnIndexer.register_oot
class VllmTPUSparseAttnIndexer(SparseAttnIndexer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vllm_config = get_current_vllm_config_or_none()
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is not None:
            set_active_model_config(model_config)
        self._glm_v7x = glm_dsa_v7x_active()
        self._shared_meta = self._glm_v7x
        # Build eagerly: pallas.jax_op runs inspect.signature(), which Dynamo
        # cannot trace if the op is first built inside a compiled forward.
        self.topk_op = self._build_streamidx_op()

        parallel_config = getattr(vllm_config, "parallel_config", None)
        pcp_size = getattr(parallel_config, "prefill_context_parallel_size", 1)
        pcp_size = pcp_size if isinstance(pcp_size, int) else 1
        dcp_size = getattr(parallel_config, "decode_context_parallel_size", 1)
        dcp_size = dcp_size if isinstance(dcp_size, int) else 1
        self.topk_op_dcp = None
        self.topk_op_tp = None
        self.topk_op_dcp_qb = None
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
            if (
                self._glm_v7x
                and vllm_config is not None
                and dcp_token_parallel_op.configure_for(vllm_config)
            ):
                self.topk_op_tp = dcp_token_parallel_op.build_indexer_op(
                    self, dcp_token_parallel_op.layout_for(vllm_config)
                )
            if self._glm_v7x:
                self.topk_op_dcp_qb = self._build_streamidx_dcp_op(
                    dcp_size,
                    parallel_config.cp_kv_cache_interleave_size,
                    qb_overlap=True,
                )

    def _build_streamidx_op(self):
        topk = self.topk_tokens
        glm_v7x = bool(self.__dict__.get("_glm_v7x", False))
        # None means unchunked. The kernel falls back to a single unpipelined
        # pass when `max_num_tokens` is not a multiple of the chunk.
        chunk_tokens = envs.TPU_STREAMIDX_CHUNK_TOKENS
        small_bq = envs.TPU_STREAMIDX_SMALL_BUCKET_BQ
        if small_bq is None and glm_v7x:
            small_bq = _GLM_STREAMIDX_DEFAULT_SMALL_BQ
        small_bucket = (
            small_bq,
            envs.TPU_STREAMIDX_SMALL_BUCKET_BKVP,
            envs.TPU_STREAMIDX_SMALL_BUCKET_TOKENS,
        )
        kv_cache_mode = (
            envs.VLLM_TPU_INDEXER_KV_CACHE_MODE
            if envs.is_indexer_kv_cache_mode_explicit()
            else None
        )
        skip_empty_decode = glm_v7x
        skip_empty_decode_chunked = glm_v7x
        mixed_bkvp = STREAMIDX_DEFAULT_MIXED_KV_PAGES if glm_v7x else None
        decode_pack = glm_v7x
        shared_meta = bool(self.__dict__.get("_shared_meta", False))

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
                q.shape[0], *small_bucket, mixed_bkvp, decode_pack
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
                skip_empty_decode=skip_empty_decode,
                skip_empty_decode_chunked=skip_empty_decode_chunked,
                chunk_tokens=chunk_tokens,
                decode_pack=decode_pack,
                shared_meta=shared_meta,
                coop_v2=bool(glm_v7x),
                double_slice_budget=bool(glm_v7x),
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

    def _build_streamidx_dcp_op(
        self, dcp_size: int, interleave_size: int, qb_overlap: bool = False
    ):
        topk = self.topk_tokens
        glm_v7x = bool(self.__dict__.get("_glm_v7x", False))
        mesh = get_or_create_dcp_mesh(axis_name=DCP_AXIS_NAME)
        # Chunking cuts the replicated query-token axis, which is orthogonal to
        # the KV axis DCP shards, so the same knob applies here unchanged.
        chunk_tokens = envs.TPU_STREAMIDX_CHUNK_TOKENS
        a2a_merge = bool(envs.TPU_DCP_INDEXER_A2A_MERGE or glm_v7x)
        a2a_min_tokens = envs.TPU_DCP_INDEXER_A2A_MIN_TOKENS
        small_bq = envs.TPU_STREAMIDX_SMALL_BUCKET_BQ
        if small_bq is None and glm_v7x:
            small_bq = _GLM_STREAMIDX_DEFAULT_SMALL_BQ
        small_bucket = (
            small_bq,
            envs.TPU_STREAMIDX_SMALL_BUCKET_BKVP,
            envs.TPU_STREAMIDX_SMALL_BUCKET_TOKENS,
        )
        kv_cache_mode = (
            envs.VLLM_TPU_INDEXER_KV_CACHE_MODE
            if envs.is_indexer_kv_cache_mode_explicit()
            else None
        )
        skip_empty_decode = glm_v7x
        skip_empty_decode_chunked = glm_v7x
        mixed_bkvp = STREAMIDX_DEFAULT_MIXED_KV_PAGES if glm_v7x else None
        decode_pack = glm_v7x
        decode_as_mixed = glm_v7x
        shared_meta = bool(self.__dict__.get("_shared_meta", False))
        hist_merge = bool(glm_v7x and dcp_size <= 16 and 16 % dcp_size == 0)
        rank_operand = glm_v7x
        rank_tensor = dcp_rank_operand() if rank_operand else None
        small_ag = bool(glm_v7x and rank_operand)
        lane_tensor = dcp_lane_operand() if small_ag else None
        k_write_compact = glm_v7x

        qb_overlap_fn = None
        qb_pin_max_tokens = 0
        qb_window_gather = False
        if qb_overlap:
            qb_pin_max_tokens = int(fp8_skinny.MAX_M)
            layer_prefix = self.k_cache.prefix.split(".self_attn")[0]
            qb_overlap_fn = (
                _fp8_linear_pallas_hbm_pinned_jax
                if fp8_skinny_hbm_pin_selected(f"{layer_prefix}.self_attn.q_b_proj")
                else _fp8_linear_pallas_jax
            )
            lane_axes = tuple(a for a in mesh.axis_names if a != DCP_AXIS_NAME)
            lane_axis = lane_axes[0] if lane_axes else None
            if lane_axis is None or mesh.shape[lane_axis] <= 1 or small_ag:
                qb_window_gather = True
                qb_overlap_fn = functools.partial(
                    _qb_window_gather_jax,
                    num_cores=dcp_size,
                    axis=DCP_AXIS_NAME,
                    lane_axis=lane_axis,
                    replicated=qb_overlap_fn,
                )

        def _streamidx_topk_dcp_impl(
            q_bytes: jax.Array,
            weights: jax.Array,
            cache_kv: jax.Array,
            k_packed: jax.Array,
            seq_lens: jax.Array,
            page_indices: jax.Array,  # virtual page ordinals
            cu_q_lens: jax.Array,
            distribution: jax.Array,
            qb_args: tuple[jax.Array, ...],
            rank_ids_arg: jax.Array | None,
            lane_ids_arg: jax.Array | None = None,
        ) -> tuple[jax.Array, ...]:
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
                rank_ids=rank_ids_arg,
                compact=k_write_compact,
            )

            q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)
            kv_pages_per_block, queries_per_block = _streamidx_block_sizes(
                q.shape[0], *small_bucket, mixed_bkvp, decode_pack
            )
            overlap: dict[str, Any] = {}
            if qb_overlap:
                overlap = dict(
                    overlap_fn=qb_overlap_fn,
                    overlap_args=tuple(qb_args),
                    overlap_pin_max_tokens=qb_pin_max_tokens,
                    overlap_rank=qb_window_gather,
                )
            large = _dcp_indexer_a2a_for_bucket(q.shape[0], True, a2a_min_tokens)
            out = streamindex_topk_dcp(
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
                decode_req_batch_size=2 if glm_v7x else 4,
                kv_layout=kv_layout,
                skip_empty_decode=skip_empty_decode,
                skip_empty_decode_chunked=skip_empty_decode_chunked,
                chunk_tokens=chunk_tokens,
                a2a_merge=large and a2a_merge and not hist_merge,
                hist_merge=large and hist_merge,
                dcp_rank_ids=rank_ids_arg,
                dcp_lane_ids=lane_ids_arg,
                flat_merge=bool(glm_v7x),
                small_all_gather=small_ag,
                decode_pack=decode_pack,
                shared_meta=shared_meta,
                decode_as_mixed=decode_as_mixed
                and _streamidx_small_bucket(
                    q.shape[0], small_bucket[0], small_bucket[2]
                ),
                coop_v2=bool(glm_v7x),
                double_slice_budget=bool(glm_v7x),
                coop_merge=bool(glm_v7x),
                **overlap,
            )
            if qb_overlap:
                assert isinstance(out, tuple)
                local_topk, q_out = out
                return cache_kv, local_topk, q_out
            assert not isinstance(out, tuple)
            return cache_kv, out

        def _dispatch(*args: jax.Array) -> tuple[jax.Array, ...]:
            base, extra = args[:8], args[8:]
            idx = 0
            qb_args = extra[idx : idx + 3] if qb_overlap else ()
            idx += 3 * int(bool(qb_overlap))
            rk = extra[idx] if rank_operand else None
            idx += int(bool(rank_operand))
            ln = extra[idx] if small_ag else None
            return cast(Any, _streamidx_topk_dcp_impl)(*base, qb_args, rk, ln)

        arg_names = (
            (
                "q_bytes",
                "weights",
                "cache_kv",
                "k_packed",
                "seq_lens",
                "page_indices",
                "cu_q_lens",
                "distribution",
            )
            + (("qb_x", "qb_w", "qb_s") if qb_overlap else ())
            + (("rank_ids",) if rank_operand else ())
            + (("lane_ids",) if small_ag else ())
        )
        jax_fn = dcp_token_parallel_op.make_typed_jax_wrapper(
            _dispatch, arg_names, 3 if qb_overlap else 2
        )

        replicated = jax.sharding.PartitionSpec()
        sharded = jax.sharding.PartitionSpec(DCP_AXIS_NAME)
        op_name = f"pallas::streamidx_topk_dcp_{self.k_cache.prefix.replace('.', '_')}"
        if qb_overlap:
            op_name += "_qb"
        jax_op = pcp_streaming_jax_op(
            op_name,
            jax_fn,
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
                *((replicated,) * 3 if qb_overlap else ()),
                *((sharded,) if rank_operand else ()),
                *((sharded,) if small_ag else ()),
            ),
            output_partition_specs=(
                sharded,  # new cache_kv
                sharded,  # local_topk: [num_tokens, topk] per rank
                *((replicated,) if qb_overlap else ()),
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
            *rest,
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
            outs: tuple[torch.Tensor, ...] = (
                torch.empty_like(cache_kv),
                q_bytes.new_empty((q_bytes.shape[0], topk), dtype=torch.int32),
            )
            if qb_overlap:
                qb_x, qb_w = rest[0], rest[1]
                outs += (qb_x.new_empty((qb_x.shape[0], qb_w.shape[1])),)
            return outs

        jax_op.register_fake(_fake_streamidx_topk_dcp)

        def op(*args):
            if rank_tensor is not None:
                args = (*args, rank_tensor)
            if lane_tensor is not None:
                args = (*args, lane_tensor)
            new_cache, local_topk, *rest = jax_op(*args)
            args[2].copy_(new_cache)
            if qb_overlap:
                return local_topk, rest[0]
            return local_topk

        return op

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q_values: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        qb_overlap: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        k_sharded: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        rows, q = self._forward_impl(
            hidden_states, q_values, k, weights, qb_overlap, k_sharded=k_sharded
        )
        return rows if qb_overlap is None else (rows, q)

    def _forward_impl(
        self,
        hidden_states: torch.Tensor,
        q_values: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        qb_overlap: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        *,
        k_sharded: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
            return self.topk_indices_buffer, None

        # uint8 [num_blocks, block_size_per_kv_packing, kv_packing, lkv_dim],
        # lkv_dim = head_dim + 1 padded to a 128 multiple
        kv_cache = self.k_cache.kv_cache
        if kv_cache.numel() == 0:
            return self.topk_indices_buffer, None

        seq_lens = metadata.seq_lens
        page_indices = metadata.block_tables
        cu_q_lens = metadata.query_start_loc
        rd = metadata.request_distribution
        assert isinstance(seq_lens, torch.Tensor)
        assert isinstance(page_indices, torch.Tensor)
        assert isinstance(cu_q_lens, torch.Tensor)
        assert isinstance(rd, torch.Tensor)
        distribution = torch.stack([rd[0], rd[0], rd[2]])

        q_b_out = None
        num_rows = hidden_states.shape[0]
        topk_op_tp = getattr(self, "topk_op_tp", None)
        step_tokens = (
            hidden_states.shape[0] * topk_op_tp.layout.dcp_size
            if (k_sharded and topk_op_tp is not None)
            else hidden_states.shape[0]
        )
        topk_op_dcp_qb = getattr(self, "topk_op_dcp_qb", None)
        if qb_overlap is not None and topk_op_dcp_qb is not None:
            topk_indices, q_b_out = topk_op_dcp_qb(
                q_values.view(torch.uint8),
                weights,
                kv_cache,
                k_quant_packed,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
                *qb_overlap,
            )
        elif topk_op_tp is not None and dcp_token_parallel_op.applies(step_tokens):
            extra_idx = {"k_sharded": True} if k_sharded else {}
            topk_indices = dcp_token_parallel_op.indexer_rows(
                self,
                q_values.view(torch.uint8),
                weights,
                kv_cache,
                k_quant_packed,
                seq_lens,
                page_indices,
                cu_q_lens,
                distribution,
                **extra_idx,
            )
            num_rows = topk_indices.shape[0]
        elif k_sharded:
            raise ValueError("k_sharded requires token-parallel topk_op_tp")
        elif self.topk_op_dcp is not None:
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

        rows = self._write_topk_rows(
            num_rows,
            topk_indices,  # pyrefly: ignore[bad-argument-type]
        )
        if self.__dict__.get("_glm_v7x", False):
            return self.topk_indices_buffer, q_b_out
        return rows, q_b_out

    def has_live_unified_cache(self) -> bool:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return False
        metadata = attn_metadata.get(self.k_cache.prefix)
        return (
            isinstance(metadata, AttentionMetadata)
            and self.k_cache.kv_cache.numel() > 0
        )

    def live_cache_step_metadata(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None
        metadata = attn_metadata.get(self.k_cache.prefix)
        if (
            not isinstance(metadata, AttentionMetadata)
            or self.k_cache.kv_cache.numel() == 0
        ):
            return None
        return metadata.seq_lens, metadata.block_tables, metadata.query_start_loc

    def prepare_unified_inputs(
        self,
        q_values: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        k_sharded: bool = False,
    ) -> dcp_unified_indexer_mla_op.UnifiedIndexerInputs | None:
        assert q_values.dtype == torch.float8_e4m3fn, (
            f"streamindex_topk expects fp8 q, got {q_values.dtype}"
        )
        attn_metadata = get_forward_context().attn_metadata
        metadata = None
        if isinstance(attn_metadata, dict):
            metadata = attn_metadata.get(self.k_cache.prefix)
        if not isinstance(metadata, AttentionMetadata):
            return None
        kv_cache = self.k_cache.kv_cache
        if kv_cache.numel() == 0:
            return None
        k_quant, k_scale = quantize_tensor(
            k,
            torch.float8_e4m3fn,
            axis=-1,
            use_ue8m0=True,
        )
        k_quant_packed = torch.cat([k_quant.view(torch.uint8), k_scale], dim=-1)
        return dcp_unified_indexer_mla_op.UnifiedIndexerInputs(
            indexer_op=self,
            q_bytes=q_values.view(torch.uint8),
            weights=weights,
            k_packed=k_quant_packed,
            idx_metadata=metadata,
            k_sharded=k_sharded,
        )

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


_qprep_ops: dict[tuple[int, int, int], Any] = {}


def _get_indexer_qprep_op(n_head: int, rope_dim: int, min_tokens: int):
    key = (n_head, rope_dim, min_tokens)
    op = _qprep_ops.get(key)
    if op is not None:
        return op

    def fn(q: jax.Array, c: jax.Array, s: jax.Array) -> tuple[jax.Array, jax.Array]:
        if q.shape[0] < min_tokens:
            return indexer_qprep.reference(q, c, s, n_head=n_head, rope_dim=rope_dim)
        return indexer_qprep.indexer_q_prep(q, c, s, n_head=n_head, rope_dim=rope_dim)

    def fake(q: torch.Tensor, c: torch.Tensor, s: torch.Tensor):
        del s
        return (
            torch.empty(
                (q.shape[0], n_head, c.shape[-1]),
                dtype=torch.float8_e4m3fn,
                device=q.device,
            ),
            torch.empty((q.shape[0], n_head), dtype=torch.float32, device=q.device),
        )

    op = pallas.jax_op(f"pallas::indexer_qprep_h{n_head}_r{rope_dim}_m{min_tokens}", fn)
    op.register_fake(fake)
    _qprep_ops[key] = op
    return op


def rope_half_split_leading_cs(
    rotary_emb,
    positions: torch.Tensor,
    like: torch.Tensor,
    d: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `[T, d]` `(C, S)` tables matching `rope_half_split_leading`."""
    t = positions.flatten().shape[0]
    cos_sin = rotary_emb._match_cos_sin_cache_dtype(like).index_select(
        0, positions.flatten()
    )
    cos, sin = cos_sin.chunk(2, dim=-1)
    parts_c, parts_s = [cos, cos], [-sin, sin]
    if d > rope_dim:
        parts_c.append(cos.new_ones(t, d - rope_dim))
        parts_s.append(sin.new_zeros(t, d - rope_dim))
    return torch.cat(parts_c, dim=-1), torch.cat(parts_s, dim=-1)


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
        # pyrefly: ignore[bad-argument-type]
        indexer._qprep_op = None
        if glm_dsa_v7x_active():
            # pyrefly: ignore[bad-argument-type]
            indexer._qprep_op = _get_indexer_qprep_op(
                indexer.n_head,
                indexer.rope_dim,
                INDEXER_QPREP_MIN_TOKENS,
            )
        return cast("VllmTPUIndexer", indexer)

    def permute_rope_to_half_split(self) -> None:
        """Reorder `wq_b`, `wk_weights_proj`, and `k_norm` rope channels to NeoX."""
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

    def prepare_inputs(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions,
        rotary_emb,
        *,
        kw: (
            torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None,
        q_rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        if isinstance(kw, tuple):
            k, weights, k_positions = kw
            k_sharded = True
        else:
            if kw is None:
                kw, _ = self.wk_weights_proj(hidden_states)
            k = kw[:, : self.head_dim]
            weights = kw[:, self.head_dim :]
            if q_rows is not None:
                weights = weights.index_select(0, q_rows)
            k_positions = positions
            k_sharded = False
        q_positions = positions if q_rows is None else positions.index_select(0, q_rows)

        k = self.k_norm(k)
        permuted = getattr(self, "_rope_neox_permuted", False)
        qprep_op = getattr(self, "_qprep_op", None)
        if (
            qprep_op is not None
            and permuted
            and rope_half_split_leading_supported(rotary_emb)
        ):
            k = rope_half_split_leading(rotary_emb, k_positions, k, self.rope_dim)
            c, s_ = rope_half_split_leading_cs(
                rotary_emb, q_positions, q, self.head_dim, self.rope_dim
            )
            q_fp8, q_scale = qprep_op(q.reshape(-1, self.n_head * self.head_dim), c, s_)
            q_scale = q_scale.view(-1, self.n_head, 1)
        else:
            if permuted and rope_half_split_leading_supported(rotary_emb):
                q = rope_half_split_leading(rotary_emb, q_positions, q, self.rope_dim)
                k = rope_half_split_leading(rotary_emb, k_positions, k, self.rope_dim)
            else:
                q_pe, q_nope = torch.split(
                    q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
                )
                k_pe, k_nope = torch.split(
                    k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
                )
                rope_fn = (
                    (lambda pos, q_, k_: rope_half_split(rotary_emb, pos, q_, k_))
                    if permuted
                    else rotary_emb
                )
                if q_rows is None and not k_sharded:
                    q_pe, k_pe = rope_fn(positions, q_pe, k_pe.unsqueeze(1))
                else:
                    k_pe_u = k_pe.unsqueeze(1)
                    q_pe = rope_fn(
                        q_positions,
                        q_pe,
                        q_pe[:, :1, :]
                        if k_sharded
                        else k_pe_u.index_select(0, cast(torch.Tensor, q_rows)),
                    )[0]
                    k_pe = rope_fn(k_positions, k_pe_u, k_pe_u)[1]
                assert k_pe is not None
                q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
                k_pe = k_pe.reshape(-1, 1, self.rope_dim)
                q = torch.cat([q_pe, q_nope], dim=-1)
                k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

            q = q.view(-1, self.head_dim)
            q_fp8, q_scale = quantize_tensor(
                q, torch.float8_e4m3fn, axis=-1, use_ue8m0=False
            )
            q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
            q_scale = q_scale.view(-1, self.n_head, 1)

        weights_dtype = weights.dtype
        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
        )
        weights = weights.squeeze(-1).to(weights_dtype)
        return q_fp8, k, weights

    def forward(  # pyrefly: ignore[bad-override]
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions,
        rotary_emb,
        q_b_layer: torch.nn.Module | None = None,
        kw: (
            torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None,
        q_rows: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        q_fp8, k, weights = self.prepare_inputs(
            hidden_states, qr, positions, rotary_emb, kw=kw, q_rows=q_rows
        )
        return self._finish_forward(
            hidden_states,
            qr,
            q_fp8,
            k,
            weights,
            q_b_layer,
            k_sharded=isinstance(kw, tuple),
        )

    def _finish_forward(
        self,
        hidden_states,
        qr,
        q_fp8,
        k,
        weights,
        q_b_layer,
        *,
        k_sharded: bool = False,
    ):
        extra = {"k_sharded": True} if k_sharded else {}
        if q_b_layer is None:
            return self.indexer_op(hidden_states, q_fp8, k, weights, **extra)
        operands = _qb_overlap_operands(q_b_layer)
        if operands is None or not getattr(q_b_layer, "qrep_active", False):
            return self.indexer_op(hidden_states, q_fp8, k, weights, **extra), None
        qb_x = qr.reshape(-1, qr.shape[-1])
        return self.indexer_op(
            hidden_states, q_fp8, k, weights, qb_overlap=(qb_x, *operands), **extra
        )


def mla_rope_half_split_perm(rope_dim: int) -> torch.Tensor:
    """Gather index reordering GPT-J interleaved channels to NeoX (evens, odds)."""
    assert rope_dim % 2 == 0, f"rope dim {rope_dim} must be even"
    return torch.cat([torch.arange(0, rope_dim, 2), torch.arange(1, rope_dim, 2)])


def mla_rope_output_index(
    n_out: int, rope_dim: int, period: int, offset: int, count: int
) -> torch.Tensor:
    """Gather index over `n_out` channels applying `mla_rope_half_split_perm`."""
    index = torch.arange(n_out)
    perm = mla_rope_half_split_perm(rope_dim)
    for i in range(count):
        start = offset + i * period
        assert start + rope_dim <= n_out, (start, rope_dim, n_out)
        index[start : start + rope_dim] = start + perm
    return index


_SAME_WIDTH_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _index_select_bits(t: torch.Tensor, axis: int, index: torch.Tensor):
    as_int = t.view(_SAME_WIDTH_INT[t.element_size()])
    out = torch.index_select(as_int, axis, index.to(t.device))
    return fresh_buffer(out.view(t.dtype))


def permute_linear_output_channels(layer: torch.nn.Module, index: torch.Tensor):
    """Reorder processed runtime output channels of `layer` by `index`."""
    n_out = index.numel()
    flipped = getattr(layer, WEIGHT_FLIPPED_ATTR, False)
    updated = []
    for name, param in list(layer.named_parameters(recurse=False)):
        t = param.data
        if name in (FP8_PALLAS_WEIGHT, FP8_PALLAS_SCALE):
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
    """Run `rotary_emb` arithmetic in NeoX half-split order without mutation."""
    if isinstance(rotary_emb, DeepseekScalingRotaryEmbedding):
        return VllmDeepseekScalingRotaryEmbedding.apply_rope(
            rotary_emb, positions, query, key, is_neox_style=True
        )
    cache = rotary_emb._match_cos_sin_cache_dtype(query)
    cos, sin = cache.index_select(0, positions.flatten()).chunk(2, dim=-1)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        rows = x.view(cos.shape[0], -1, rotary_emb.head_size)
        rows = ApplyRotaryEmb.forward_static(rows, cos, sin, True, True)
        return rows.reshape(x.shape)

    return rotate(query), None if key is None else rotate(key)


def mla_q_rope_table(
    rotary_emb, positions: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    """Build per-token `f32[T, 2, 128]` `(C, S)` table for in-kernel `q_pe` RoPE."""
    if isinstance(rotary_emb, DeepseekScalingRotaryEmbedding):
        cache = rotary_emb.cos_sin_cache
    else:
        cache = rotary_emb._match_cos_sin_cache_dtype(query)
    cos, sin = cache.index_select(0, positions.flatten()).chunk(2, dim=-1)
    assert cos.shape[-1] == 32, f"rope half {cos.shape[-1]} != 32"
    c = torch.cat([cos, cos, cos, cos], dim=-1).float()
    s_ = torch.cat([-sin, sin, -sin, sin], dim=-1).float()
    return torch.stack([c, s_], dim=1)


def rope_half_split_leading(
    rotary_emb, positions: torch.Tensor, x: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    """Apply NeoX RoPE to leading `rope_dim` channels in one full-width op."""
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
    return (x.float() * c.float() + rot.float() * s_.float()).to(x.dtype)


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
        and type(rotary_emb).forward_native
        in (RotaryEmbedding.forward_native, TpuRotaryEmbedding.forward_native)
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

        # process_weights_after_loading masks this around upstream's copy.
        self.dcp_q_replicate = dcp_q_replicate
        # The shared top-k table. A layer that reuses an earlier layer's
        # indices has no indexer, so this is where its top-k width is stated.
        self.topk_indices_buffer = topk_indices_buffer
        self.token_parallel_layout = None
        vllm_config = get_current_vllm_config_or_none()
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is not None:
            set_active_model_config(model_config)
        if (
            use_sparse
            and vllm_config is not None
            and glm_dsa_v7x_active()
            and dcp_token_parallel_op.configure_for(vllm_config)
        ):
            self.token_parallel_layout = dcp_token_parallel_op.layout_for(vllm_config)

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
        self.W_UK_T_dcp_qrep_lo = None
        self.W_UK_T_dcp_qrep_hi = None
        self.W_UK_T_dcp_qrep_scale = None
        if not self.dcp_q_replicate:
            return
        dcp_group = get_dcp_group()
        if dcp_group is None or int(dcp_group.world_size) <= 1:
            raise RuntimeError(
                "dcp_q_replicate is set but no DCP group is initialized, "
                "so W_UK_T cannot be gathered across the group."
            )
        gathered_w_uk = dcp_group.all_gather(self.W_UK_T.contiguous(), dim=0)
        if (
            glm_dsa_v7x_active()
            and not hasattr(self, "W_UK_T_scale")
            and gathered_w_uk.shape[0] >= 2 * _FUSED_QB_HEAD_BLOCK
            and gathered_w_uk.shape[0] % (2 * _FUSED_QB_HEAD_BLOCK) == 0
        ):
            half = gathered_w_uk.shape[0] // 2
            self.W_UK_T_dcp_qrep_lo = Parameter(
                fresh_buffer(gathered_w_uk[:half]), requires_grad=False
            )
            self.W_UK_T_dcp_qrep_hi = Parameter(
                fresh_buffer(gathered_w_uk[half:]), requires_grad=False
            )
        else:
            self.W_UK_T_dcp_qrep = Parameter(
                fresh_buffer(gathered_w_uk), requires_grad=False
            )
        del gathered_w_uk
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
                for t in (
                    self.W_UK_T_dcp_qrep,
                    self.W_UK_T_dcp_qrep_lo,
                    self.W_UK_T_dcp_qrep_hi,
                    self.W_UK_T_dcp_qrep_scale,
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
        self.sparse_mla_dcp_rope_op = None
        self.sparse_mla_dcp_send_op = None
        self.sparse_mla_dcp_fused_op = None
        self.sparse_mla_tp_op = None
        self.sparse_mla_unified_tp_op = None
        self.sparse_mla_unified_idx_qb_op = None
        self.sparse_mla_in_branch_qb_op = None
        if dcp_size > 1:
            self.sparse_mla_dcp_op = impl._build_sparse_mla_dcp_op(
                self, dcp_size, impl.dcp_interleave_size
            )
            if self.token_parallel_layout is not None:
                self.sparse_mla_tp_op = dcp_token_parallel_op.build_attention_op(
                    self,
                    impl,
                    self.token_parallel_layout,
                    q_replicated=bool(self.dcp_q_replicate),
                )
                if getattr(self.sparse_mla_tp_op, "serves_oproj", False):
                    dcp_group = get_dcp_group()
                    assert dcp_group is not None
                    self.W_UV_tp_all = fresh_buffer(
                        dcp_group.all_gather(self.W_UV.contiguous(), dim=0)
                    )
                    if self.W_UV_tp_all.device.type == "tpu":
                        synchronize_tensors([self.W_UV_tp_all])
                    self._build_o_proj_pair_half(dcp_group)
                    if self.dcp_q_replicate:
                        self.sparse_mla_in_branch_qb_op = (
                            dcp_nonidx_qb_mla_op.build_in_branch_qb_attention_op(
                                self,
                                impl,
                                self.token_parallel_layout,
                                q_replicated=True,
                            )
                        )
                indexer_op = getattr(self.indexer, "indexer_op", None)
                if (
                    self.dcp_q_replicate
                    and indexer_op is not None
                    and getattr(indexer_op, "topk_op_tp", None) is not None
                ):
                    self.indexer_k_cache = indexer_op.k_cache
                    indexer_op.topk_op_tp = dcp_token_parallel_op.build_indexer_op(
                        indexer_op,
                        self.token_parallel_layout,
                        q_own_rows=True,
                    )
                    if getattr(self.sparse_mla_tp_op, "serves_oproj", False):
                        callbacks = dcp_unified_indexer_mla_op.UnifiedIndexerCallbacks(
                            resolve_kv_cache_mode=_resolve_indexer_kv_cache_mode,
                            scatter_indexer_k=_cp_scatter_indexer_k,
                            streamidx_block_sizes=_streamidx_block_sizes,
                            dcp_indexer_a2a_for_bucket=_dcp_indexer_a2a_for_bucket,
                        )
                        u_mod = dcp_unified_indexer_mla_op
                        self.sparse_mla_unified_tp_op = (
                            u_mod.build_unified_indexer_attention_op(
                                self,
                                impl,
                                indexer_op,
                                self.token_parallel_layout,
                                q_replicated=True,
                                callbacks=callbacks,
                            )
                        )
                        uqb_mod = dcp_unified_idx_qb_mla_op
                        self.sparse_mla_unified_idx_qb_op = (
                            uqb_mod.build_unified_indexer_in_branch_qb_op(
                                self,
                                impl,
                                indexer_op,
                                self.token_parallel_layout,
                                q_replicated=True,
                                callbacks=callbacks,
                            )
                        )
            if (
                glm_dsa_v7x_active()
                and self.dcp_q_replicate
                and (envs.TPU_MLA_SPARSE_SPLIT_Q or glm_dsa_v7x_active())
                and (self.qk_nope_head_dim, self.qk_rope_head_dim) == (192, 64)
                and isinstance(impl, PallasMLAttentionBackendImpl)
            ):
                build_dcp = cast(Any, impl._build_sparse_mla_dcp_op)
                self.sparse_mla_dcp_rope_op = build_dcp(
                    self, dcp_size, impl.dcp_interleave_size, q_rope=True
                )
                self.sparse_mla_dcp_send_op = build_dcp(
                    self,
                    dcp_size,
                    impl.dcp_interleave_size,
                    q_rope=True,
                    send_layout=True,
                )
                self.sparse_mla_dcp_fused_op = build_dcp(
                    self,
                    dcp_size,
                    impl.dcp_interleave_size,
                    q_rope=True,
                    send_layout=True,
                    fused_exchange=True,
                )

        source = getattr(self, "sp_o_proj_source", None)
        if source and getattr(source[0], "weight_dcp", None) is not None:
            del source[0].weight_dcp, source[0].weight_scale_dcp

        for hook in getattr(self, "post_load_hooks", ()):
            hook()

    def _build_o_proj_pair_half(self, dcp_group: Any) -> None:
        source = getattr(self, "sp_o_proj_source", None)
        o_proj = source[0] if source else None
        world = int(dcp_group.world_size)
        if (
            o_proj is None
            or world % 2
            or not dcp_token_parallel_op.serves_sequence_row_oproj(o_proj)
        ):
            return
        parity = int(dcp_group.rank_in_group) % 2
        whole_k = getattr(o_proj, "weight_dcp", None)
        if whole_k is not None and world > 2:
            half_rows = whole_k.shape[0] // 2
            self.W_O_tp_half = fresh_buffer(
                whole_k[parity * half_rows : (parity + 1) * half_rows]
            )
            self.W_O_tp_half_scale = fresh_buffer(o_proj.weight_scale_dcp)
            del o_proj.weight_dcp, o_proj.weight_scale_dcp
        else:
            shards = dcp_group.all_gather(o_proj.weight.contiguous(), dim=0)
            scales = dcp_group.all_gather(o_proj.weight_scale.reshape(1, -1), dim=0)
            half_rows = shards.shape[0] // 2
            self.W_O_tp_half = fresh_buffer(
                shards[parity * half_rows : (parity + 1) * half_rows]
            )
            self.W_O_tp_half_scale = fresh_buffer(
                scales[parity * (world // 2) : (parity + 1) * (world // 2)]
            )
            del shards, scales
        if self.W_O_tp_half.device.type == "tpu":
            synchronize_tensors([self.W_O_tp_half, self.W_O_tp_half_scale])

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
            sp_o_proj=kwargs.get("sp_o_proj"),
            kv_sharded=kwargs.get("kv_sharded", False),
            unified_indexer_inputs=kwargs.get("unified_indexer_inputs"),
            in_branch_qb_inputs=kwargs.get("in_branch_qb_inputs"),
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
        vllm_config = get_current_vllm_config_or_none()
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is not None:
            set_active_model_config(model_config)
        glm_v7x = glm_dsa_v7x_active()

        has_tp_layout = (
            getattr(self.mla_attn, "token_parallel_layout", None) is not None
        )
        if glm_v7x or has_tp_layout:
            mla_attn_any: Any = self.mla_attn
            mla_attn_any.sp_o_proj_source = (self.o_proj,)
            if (
                glm_v7x
                and self.q_a_layernorm is not None
                and (self.indexer is None or self.skip_topk)
                and getattr(self.q_a_layernorm, "has_weight", True)
                and getattr(self.q_a_layernorm, "variance_size_override", None) is None
            ):
                mla_attn_any.in_branch_q_ln_eps = float(
                    getattr(self.q_a_layernorm, "variance_epsilon", 1e-6)
                )
            if (
                has_tp_layout
                and not getattr(self.o_proj, DCP_ROWS_ATTR, False)
                and self.mla_attn.dcp_q_replicate
                and self.g_proj is None
                and not self.gate_is_fused
            ):
                setattr(self.o_proj, DCP_ROWS_ATTR, True)

        self.decode_prologue = bool(
            glm_v7x
            and self.is_sparse
            and self.dcp_q_replicate
            and self.rotary_emb is not None
            and (envs.TPU_MLA_SPARSE_SPLIT_Q or glm_v7x)
            and (self.qk_nope_head_dim, self.qk_rope_head_dim) == (192, 64)
            and isinstance(self.mla_attn.impl, PallasMLAttentionBackendImpl)
        )
        self.qb_overlap = bool(
            glm_v7x
            and self.is_sparse
            and self.indexer is not None
            and not self.skip_topk
            and self.q_lora_rank is not None
            and getattr(self.indexer.indexer_op, "topk_op_dcp_qb", None) is not None
        )
        dcp_indexer = (
            getattr(getattr(self.indexer, "indexer_op", None), "dcp_size", 1) > 1
        )
        self._dcp_rank_keep = (
            dcp_rank_operand()
            if glm_v7x
            and self.is_sparse
            and self.indexer is not None
            and not self.skip_topk
            and (self.dcp_group_size > 1 or dcp_indexer)
            else None
        )
        self._dcp_lane_keep = (
            dcp_lane_operand() if self._dcp_rank_keep is not None else None
        )

        self.local_q_fused_op = None
        self.local_q_norm_fused_op = None
        self.local_q_window = None
        self.local_q_lane = None
        if (
            glm_v7x
            and self.dcp_q_replicate
            and self.is_sparse
            and self.q_lora_rank is not None
            and (get_dcp_rank(), get_dcp_world_size())
            == (
                getattr(q_proj_layer, "rank_in_group", None),
                self.dcp_group_size,
            )
        ):
            self.local_q_fused_op = local_q_all_gather_op()
            if self.local_q_fused_op is not None:
                if (
                    self.q_a_layernorm is not None
                    and (self.indexer is None or self.skip_topk)
                    and getattr(self.q_a_layernorm, "has_weight", True)
                    and getattr(self.q_a_layernorm, "variance_size_override", None)
                    is None
                ):
                    q_eps = float(getattr(self.q_a_layernorm, "variance_epsilon", 1e-6))
                    self.local_q_norm_fused_op = local_q_norm_all_gather_op(q_eps)
                self.local_q_window = dcp_rank_operand()
                self.local_q_lane = dcp_lane_operand()

        self._rope_neox_permuted = False
        has_lora = getattr(vllm_config, "lora_config", None) is not None
        if (
            glm_v7x
            and not has_lora
            and rope_half_split_supported(self.rotary_emb, self.qk_rope_head_dim)
        ):
            hooks = self.mla_attn.__dict__.setdefault("post_load_hooks", [])
            hooks.append(self.permute_rope_to_half_split)
        if (
            glm_v7x
            and not has_lora
            and self.indexer is not None
            and rope_half_split_supported(self.indexer_rope_emb, self.indexer.rope_dim)
        ):
            hooks = self.mla_attn.__dict__.setdefault("post_load_hooks", [])
            hooks.append(self.indexer.permute_rope_to_half_split)

    def permute_rope_to_half_split(self) -> None:
        """Reorder `q` and `k_pe` rope output channels to NeoX half-split order."""
        if self._rope_neox_permuted or self.kimi_preprocess is not None:
            return
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

    def _local_q_serves(self, m: int) -> bool:
        operands = _qb_overlap_operands(self.q_b_proj)
        return operands is not None and bool(
            fp8_window_all_gather_mod.supports(
                m, operands[0].shape[0], operands[0].shape[1], self.dcp_group_size
            )
        )

    def _project_local_q(self, q_c: torch.Tensor) -> torch.Tensor:
        operands = _qb_overlap_operands(self.q_b_proj)
        assert operands is not None and self.local_q_fused_op is not None
        return self.local_q_fused_op(
            q_c, *operands, self.local_q_window, self.local_q_lane
        )

    def _fused_qb_rope_absorb_operands(
        self,
        q_c: torch.Tensor,
        *,
        q_rows: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None,
        num_kv_tokens: int,
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | tuple[torch.Tensor, torch.Tensor],
            int,
            torch.Tensor,
        ]
        | None
    ):
        if (
            not glm_dsa_v7x_active()
            or q_rows is None
            or llama_4_scaling is not None
            or not self.dcp_q_replicate
            or not self.is_sparse
            or self.rotary_emb is None
            or not getattr(self, "_rope_neox_permuted", False)
            or self.rotary_emb.rotary_dim != self.qk_rope_head_dim
            or self.rotary_emb.head_size != self.qk_rope_head_dim
            or q_c.device.type != "tpu"
        ):
            return None
        tp_op = getattr(self.mla_attn, "sparse_mla_tp_op", None)
        if (
            tp_op is None
            or not getattr(tp_op, "split_q_inputs", False)
            or not dcp_token_parallel_op.applies(num_kv_tokens)
        ):
            return None
        w_uk_lo = getattr(self.mla_attn, "W_UK_T_dcp_qrep_lo", None)
        w_uk_hi = getattr(self.mla_attn, "W_UK_T_dcp_qrep_hi", None)
        w_uk_t: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        if (
            w_uk_lo is not None
            and w_uk_hi is not None
            and isinstance(w_uk_lo, torch.Tensor)
            and isinstance(w_uk_hi, torch.Tensor)
        ):
            lo_t = cast(torch.Tensor, w_uk_lo)
            hi_t = cast(torch.Tensor, w_uk_hi)
            w_uk_t = (lo_t, hi_t)
            w_uk_shape = (
                lo_t.shape[0] + hi_t.shape[0],
                lo_t.shape[1],
                lo_t.shape[2],
            )
        else:
            w_uk_raw = getattr(self.mla_attn, "W_UK_T_dcp_qrep", None)
            if w_uk_raw is None or not isinstance(w_uk_raw, torch.Tensor):
                return None
            raw_t = cast(torch.Tensor, w_uk_raw)
            w_uk_t = raw_t
            w_uk_shape = tuple(raw_t.shape)
        if getattr(self.mla_attn, "W_UK_T_dcp_qrep_scale", None) is not None:
            return None
        qb_ops = _fused_qb_operands(self.q_b_proj)
        if qb_ops is None:
            return None
        w_qb, s_qb = qb_ops
        rotary_any: Any = self.rotary_emb
        cos_sin_cache = (
            self.rotary_emb.cos_sin_cache
            if isinstance(self.rotary_emb, DeepseekScalingRotaryEmbedding)
            else rotary_any._match_cos_sin_cache_dtype(q_c)
        )
        split_rows = tp_op.layout.split_rows
        if not supports_fused_qb_rope_absorb_split(
            tuple(q_c.shape),
            q_c.dtype,
            tuple(w_qb.shape),
            w_qb.dtype,
            tuple(s_qb.shape),
            w_uk_shape,
            (q_c.shape[0], cos_sin_cache.shape[-1]),
            split_rows,
            bfloat16_dtype=torch.bfloat16,
            fp8_dtype=torch.float8_e4m3fn,
        ):
            return None
        return w_qb, s_qb, w_uk_t, split_rows, cos_sin_cache

    def _local_q_norm_serves(self, q_c: torch.Tensor) -> bool:
        if (
            getattr(self, "local_q_norm_fused_op", None) is None
            or not (self.indexer is None or self.skip_topk)
            or self.q_a_layernorm is None
            or not getattr(self.q_a_layernorm, "has_weight", True)
            or getattr(self.q_a_layernorm, "variance_size_override", None) is not None
        ):
            return False
        norm_w = getattr(self.q_a_layernorm, "weight", None)
        operands = _qb_overlap_operands(self.q_b_proj)
        return (
            norm_w is not None
            and norm_w.dim() == 1
            and operands is not None
            and norm_w.shape[0] == operands[0].shape[0]
            and fp8_window_all_gather_mod.supports_qnorm(
                q_c.shape[0],
                operands[0].shape[0],
                operands[0].shape[1],
                self.dcp_group_size,
            )
        )

    def _project_local_q_norm(self, q_c: torch.Tensor) -> torch.Tensor:
        operands = _qb_overlap_operands(self.q_b_proj)
        assert (
            operands is not None
            and self.local_q_norm_fused_op is not None
            and self.q_a_layernorm is not None
        )
        return self.local_q_norm_fused_op(
            q_c,
            *operands,
            self.local_q_window,
            self.local_q_lane,
            self.q_a_layernorm.weight,
        )

    def _resolve_sp_o_proj(
        self,
        num_tokens: int,
        prefill_group: Any,
        output_gate: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not (
            prefill_group is not None
            and self.is_sparse
            and output_gate is None
            and self.g_proj is None
            and getattr(
                getattr(self.mla_attn, "sparse_mla_tp_op", None), "serves_oproj", False
            )
            and dcp_token_parallel_op.applies(num_tokens)
            and dcp_token_parallel_op.serves_sequence_row_oproj(self.o_proj)
        ):
            return None
        w_half = getattr(self.mla_attn, "W_O_tp_half", None)
        o_proj_any: Any = self.o_proj
        mla_any: Any = self.mla_attn
        return (
            (o_proj_any.weight, o_proj_any.weight_scale)
            if w_half is None
            else (w_half, mla_any.W_O_tp_half_scale)
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if hasattr(super(), "process_weights_after_loading"):
            # torch's stubs resolve this hasattr-guarded method as a Tensor.
            # pyrefly: ignore[not-callable]
            super().process_weights_after_loading(act_dtype)
        if self.rotary_emb is not None and hasattr(self.rotary_emb, "cos_sin_cache"):
            device = torch.device("tpu")
            self.rotary_emb.cos_sin_cache = self.rotary_emb.cos_sin_cache.to(device)

    def forward(  # pyrefly: ignore[bad-override]
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        prefill_group=None,
        post_attn_norm: tuple[torch.Tensor, torch.Tensor, float] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        q_c = None
        kv_lora = None
        output_gate = None
        early_topk = None
        if (
            self.is_sparse
            and self.indexer is not None
            and not self.skip_topk
            and (
                getattr(self, "qb_overlap", False)
                or getattr(self, "_dcp_rank_keep", None) is not None
            )
        ):
            cos_sin_keep = (
                getattr(self.rotary_emb, "cos_sin_cache", None)
                if self.rotary_emb is not None
                else None
            )
            hidden_states = _retain_buffers(
                hidden_states,
                cos_sin_keep,
                getattr(self, "_dcp_rank_keep", None),
                getattr(self, "_dcp_lane_keep", None),
            )

        pre_qkv = pre_kw = pre_qc_qrows = None
        kv_positions = positions
        kv_sharded = False
        num_tokens = positions.shape[0]
        project_gather = getattr(prefill_group, "project_gather", None)
        if project_gather is not None:
            pre_qkv, pre_kw = project_gather(self, hidden_states, num_tokens)
            if isinstance(pre_qkv, tuple):
                if len(pre_qkv) == 3:
                    pre_qc_qrows, kv_lora, kv_positions = pre_qkv
                    kv_sharded = True
                else:
                    pre_qc_qrows, kv_lora = pre_qkv
                hidden_states = (
                    pre_kw
                    if (kv_sharded and isinstance(pre_kw, torch.Tensor))
                    else kv_lora
                )
            else:
                hidden_states = pre_qkv
        else:
            num_tokens = hidden_states.shape[0]
        if getattr(self, "local_q_fused_op", None) is not None:
            operands = (self.local_q_window, self.local_q_lane)
            if kv_lora is not None:
                kv_lora = _retain_buffers(
                    kv_lora, *(t for t in operands if t is not None)
                )
                if not kv_sharded:
                    hidden_states = kv_lora
            else:
                hidden_states = _retain_buffers(
                    hidden_states, *(t for t in operands if t is not None)
                )

        q_rows = None
        tp_op = getattr(self.mla_attn, "sparse_mla_tp_op", None)
        if tp_op is not None and tp_op.q_replicated:
            if not dcp_token_parallel_op.applies(num_tokens):
                hidden_states = _retain_buffers(hidden_states, tp_op.rank)
            elif self.kimi_preprocess is None:
                q_rows = dcp_token_parallel_op.q_rows(tp_op, num_tokens)

        indexer_qrows = None
        if q_rows is not None:
            idx = self.indexer if not self.skip_topk else None
            if idx is None or getattr(
                getattr(idx.indexer_op, "topk_op_tp", None), "q_own_rows", False
            ):
                indexer_qrows = q_rows

        q: torch.Tensor | None = None
        fused_qb_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        in_branch_qb_inputs: dcp_nonidx_qb_mla_op.InBranchQbInputs | None = None
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
            num_tokens = q_c.shape[0]
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

                if pre_qc_qrows is not None:
                    q_c = pre_qc_qrows
                else:
                    qkv_lora = (
                        pre_qkv
                        if pre_qkv is not None
                        else self.fused_qkv_a_proj(hidden_states)[0]
                    )
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
                            [
                                self.q_lora_rank,
                                self.kv_lora_rank + self.qk_rope_head_dim,
                            ],
                            dim=-1,
                        )
                    if indexer_qrows is not None:
                        q_c = q_c.index_select(0, indexer_qrows)
                fuse_local_q_norm = (
                    not getattr(self, "qb_overlap", False)
                    and q_rows is None
                    and pre_qkv is None
                    and pre_qc_qrows is None
                    and not dcp_token_parallel_op.applies(num_tokens)
                    and vllm_compilation_backends.model_tag == "backbone"
                    and self._local_q_norm_serves(q_c)
                )
                if fuse_local_q_norm:
                    q = self._project_local_q_norm(q_c)
                elif (
                    getattr(self, "qb_overlap", False)
                    and q_rows is None
                    and pre_qkv is None
                    and pre_qc_qrows is None
                    and not dcp_token_parallel_op.applies(num_tokens)
                ):
                    q_c = self.q_a_layernorm(q_c)
                    assert self.indexer is not None
                    ov_out = self.indexer(
                        hidden_states,
                        q_c,
                        positions,
                        self.indexer_rope_emb,
                        q_b_layer=self.q_b_proj,
                    )
                    assert isinstance(ov_out, tuple)
                    early_topk, q = ov_out
                    if q is None:
                        q = self.q_b_proj(q_c)[0]
                elif (
                    getattr(self, "local_q_fused_op", None) is not None
                    and q_rows is None
                    and pre_qkv is None
                    and pre_qc_qrows is None
                    and not dcp_token_parallel_op.applies(num_tokens)
                    and vllm_compilation_backends.model_tag == "backbone"
                    and self._local_q_serves(q_c.shape[0])
                ):
                    q_c = self.q_a_layernorm(q_c)
                    q = self._project_local_q(q_c)
                else:
                    q_c_raw_for_q = (
                        q_c
                        if (q_rows is None or q_c.shape[0] == q_rows.shape[0])
                        else q_c.index_select(0, q_rows)
                    )
                    fused_ops = self._fused_qb_rope_absorb_operands(
                        q_c_raw_for_q,
                        q_rows=q_rows,
                        llama_4_scaling=llama_4_scaling,
                        num_kv_tokens=positions.shape[0],
                    )
                    if fused_ops is not None:
                        assert q_rows is not None
                        w_qb, s_qb, w_uk_t, split_rows, cos_sin_cache = fused_ops
                        cos_sin = cos_sin_cache.index_select(
                            0, positions.index_select(0, q_rows)
                        )
                        in_branch_qb_op = getattr(
                            self.mla_attn, "sparse_mla_in_branch_qb_op", None
                        )
                        unified_qb_op = getattr(
                            self.mla_attn, "sparse_mla_unified_idx_qb_op", None
                        )
                        sp_o_proj_ready = (
                            self._resolve_sp_o_proj(
                                num_tokens, prefill_group, output_gate
                            )
                            is not None
                        )
                        can_defer_nonidx_qb = (
                            (self.indexer is None or self.skip_topk)
                            and in_branch_qb_op is not None
                            and (
                                not kv_sharded
                                or getattr(in_branch_qb_op, "serves_sharded_kv", False)
                            )
                            and sp_o_proj_ready
                        )
                        can_defer_idx_qb = (
                            self.is_sparse
                            and early_topk is None
                            and self.indexer is not None
                            and not self.skip_topk
                            and llama_4_scaling is None
                            and indexer_qrows is not None
                            and unified_qb_op is not None
                            and (
                                not kv_sharded
                                or getattr(unified_qb_op, "serves_sharded_kv", False)
                            )
                            and (
                                not isinstance(pre_kw, tuple)
                                or getattr(unified_qb_op, "serves_sharded_k", False)
                            )
                            and sp_o_proj_ready
                            and cast(
                                Any, self.indexer
                            ).indexer_op.has_live_unified_cache()
                        )
                        q_ln_w: Any = getattr(self.q_a_layernorm, "weight", None)
                        q_ln_eps_val = getattr(
                            self.q_a_layernorm, "variance_epsilon", None
                        )
                        has_q_ln_fn = getattr(in_branch_qb_op, "has_q_ln", None)
                        defer_nonidx_q_ln = (
                            can_defer_nonidx_qb
                            and glm_dsa_v7x_active()
                            and isinstance(q_ln_w, torch.Tensor)
                            and q_ln_w.ndim == 1
                            and q_ln_w.shape[0] == q_c_raw_for_q.shape[-1]
                            and getattr(self.q_a_layernorm, "has_weight", True)
                            and getattr(
                                self.q_a_layernorm, "variance_size_override", None
                            )
                            is None
                            and not (
                                hasattr(self.q_a_layernorm, "is_prequant_armed")
                                and cast(Any, self.q_a_layernorm).is_prequant_armed()
                            )
                            and q_ln_eps_val is not None
                            and callable(has_q_ln_fn)
                            and bool(has_q_ln_fn(float(q_ln_eps_val), kv_sharded))
                        )
                        if defer_nonidx_q_ln:
                            assert isinstance(q_ln_w, torch.Tensor)
                            assert q_ln_eps_val is not None
                            q_c_for_q = q_c_raw_for_q
                            q_ln_weight_arg: torch.Tensor | None = q_ln_w
                            q_ln_eps_arg: float | None = float(q_ln_eps_val)
                        else:
                            q_c = self.q_a_layernorm(q_c)
                            q_c_for_q = (
                                q_c
                                if (q_rows is None or q_c.shape[0] == q_rows.shape[0])
                                else q_c.index_select(0, q_rows)
                            )
                            q_ln_weight_arg = None
                            q_ln_eps_arg = None
                        if can_defer_nonidx_qb or can_defer_idx_qb:
                            if isinstance(w_uk_t, tuple):
                                w_uk_lo, w_uk_hi = w_uk_t
                                in_branch_qb_inputs = (
                                    dcp_nonidx_qb_mla_op.InBranchQbInputs(
                                        q_c=q_c_for_q,
                                        w_qb=w_qb,
                                        s_qb=s_qb,
                                        w_uk_t=w_uk_lo.to(q_c_for_q.dtype),
                                        cos_sin=cos_sin,
                                        w_uk_t_hi=w_uk_hi.to(q_c_for_q.dtype),
                                        q_ln_weight=q_ln_weight_arg,
                                        q_ln_eps=q_ln_eps_arg,
                                    )
                                )
                            else:
                                in_branch_qb_inputs = (
                                    dcp_nonidx_qb_mla_op.InBranchQbInputs(
                                        q_c=q_c_for_q,
                                        w_qb=w_qb,
                                        s_qb=s_qb,
                                        w_uk_t=w_uk_t.to(q_c_for_q.dtype),
                                        cos_sin=cos_sin,
                                        q_ln_weight=q_ln_weight_arg,
                                        q_ln_eps=q_ln_eps_arg,
                                    )
                                )
                        elif isinstance(w_uk_t, tuple):
                            w_uk_lo, w_uk_hi = w_uk_t
                            fused_qb_out = _FUSED_QB_ROPE_ABSORB_SPLIT_HALVES_OP(
                                q_c_for_q,
                                w_qb,
                                s_qb,
                                w_uk_lo.to(q_c_for_q.dtype),
                                w_uk_hi.to(q_c_for_q.dtype),
                                cos_sin,
                                split_rows,
                            )
                        else:
                            fused_qb_out = _FUSED_QB_ROPE_ABSORB_SPLIT_OP(
                                q_c_for_q,
                                w_qb,
                                s_qb,
                                w_uk_t.to(q_c_for_q.dtype),
                                cos_sin,
                                split_rows,
                            )
                        q = None
                    else:
                        q_c = self.q_a_layernorm(q_c)
                        q_c_for_q = (
                            q_c
                            if (q_rows is None or q_c.shape[0] == q_rows.shape[0])
                            else q_c.index_select(0, q_rows)
                        )
                        q = self.q_b_proj(q_c_for_q)[0]
            else:
                assert q_rows is None, "token-parallel rows need the q_lora path"
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

        k_pe = k_pe.unsqueeze(1)

        q_rope_table = None
        if fused_qb_out is not None:
            k_pe_dtype = k_pe.dtype
            _, k_pe_rot = rope_half_split(self.rotary_emb, kv_positions, k_pe, k_pe)
            assert k_pe_rot is not None
            k_pe = k_pe_rot.to(dtype=k_pe_dtype)
            ql_head, ql_tail, q_pe_all = fused_qb_out
            q_nope = ql_head[:, : self.num_heads, : self.qk_nope_head_dim]
            q_pe = q_pe_all[:, : self.num_heads, :]
        elif in_branch_qb_inputs is not None:
            k_pe_dtype = k_pe.dtype
            _, k_pe_rot = rope_half_split(self.rotary_emb, kv_positions, k_pe, k_pe)
            assert k_pe_rot is not None
            k_pe = k_pe_rot.to(dtype=k_pe_dtype)
            q_c_ref = in_branch_qb_inputs.q_c[:1, :1].unsqueeze(1)
            q_nope = q_c_ref.expand(1, self.num_heads, self.qk_nope_head_dim)
            q_pe = q_c_ref.expand(1, self.num_heads, self.qk_rope_head_dim)
        else:
            assert q is not None
            q = q.view(-1, self.num_heads * self.dcp_group_size, self.qk_head_dim)
            q_nope, q_pe = q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )

            fold_q_rope = (
                getattr(self, "decode_prologue", False)
                and getattr(self, "_rope_neox_permuted", False)
                and llama_4_scaling is None
                and q_rows is None
                and not dcp_token_parallel_op.applies(num_tokens)
            )
            if fold_q_rope:
                k_pe_dtype = k_pe.dtype
                _, k_pe = rope_half_split(self.rotary_emb, positions, q_pe, k_pe)
                assert k_pe is not None
                k_pe = k_pe.to(dtype=k_pe_dtype)
                q_rope_table = mla_q_rope_table(self.rotary_emb, positions, q)
            elif self.rotary_emb is not None:
                q_pe_dtype = q_pe.dtype
                k_pe_dtype = k_pe.dtype
                if getattr(self, "_rope_neox_permuted", False):

                    def rope(pos, q_, k_):
                        return rope_half_split(self.rotary_emb, pos, q_, k_)
                else:
                    rope = self.rotary_emb
                if q_rows is None:
                    q_pe, k_pe = rope(positions, q_pe, k_pe)
                else:
                    q_pe = rope(
                        positions.index_select(0, q_rows),
                        q_pe,
                        q_pe[:, :1, :] if kv_sharded else k_pe.index_select(0, q_rows),
                    )[0]
                    k_pe = rope(kv_positions, k_pe, k_pe)[1]
                assert k_pe is not None
                q_pe = q_pe.to(dtype=q_pe_dtype)
                k_pe = k_pe.to(dtype=k_pe_dtype)

        sp_o_proj = self._resolve_sp_o_proj(num_tokens, prefill_group, output_gate)

        unified_op = getattr(
            self.mla_attn,
            "sparse_mla_unified_idx_qb_op"
            if in_branch_qb_inputs is not None
            else "sparse_mla_unified_tp_op",
            None,
        )
        k_sharded = isinstance(pre_kw, tuple)
        unified_idx_inputs = None
        if (
            self.is_sparse
            and early_topk is None
            and self.indexer is not None
            and not self.skip_topk
            and sp_o_proj is not None
            and llama_4_scaling is None
            and unified_op is not None
            and (not kv_sharded or getattr(unified_op, "serves_sharded_kv", False))
            and (not k_sharded or getattr(unified_op, "serves_sharded_k", False))
            and indexer_qrows is not None
            and (
                fused_qb_out is not None
                or in_branch_qb_inputs is not None
                or self.dcp_q_replicate
            )
            and cast(Any, self.indexer).indexer_op.has_live_unified_cache()
        ):
            assert q_c is not None
            indexer_mod: Any = self.indexer
            q_idx_fp8, k_idx, w_idx = indexer_mod.prepare_inputs(
                hidden_states,
                q_c,
                positions,
                self.indexer_rope_emb,
                kw=pre_kw,
                q_rows=indexer_qrows,
            )
            unified_idx_inputs = indexer_mod.indexer_op.prepare_unified_inputs(
                q_idx_fp8, k_idx, w_idx, k_sharded=k_sharded
            )

        topk_indices = None
        if self.is_sparse and unified_idx_inputs is None:
            if early_topk is not None:
                topk_index = early_topk
            elif self.indexer is not None and not self.skip_topk:
                assert q_c is not None
                idx_kwargs: dict[str, Any] = {}
                if pre_kw is not None:
                    idx_kwargs["kw"] = pre_kw
                if indexer_qrows is not None:
                    idx_kwargs["q_rows"] = indexer_qrows
                topk_index = self.indexer(
                    hidden_states,
                    q_c,
                    positions,
                    self.indexer_rope_emb,
                    **idx_kwargs,
                )
            else:
                topk_index = self.topk_indices_buffer
            assert isinstance(topk_index, torch.Tensor)
            topk_indices = topk_index[:num_tokens]

        if llama_4_scaling is not None:
            q_nope *= llama_4_scaling
            q_pe *= llama_4_scaling

        has_topk = topk_indices is not None or unified_idx_inputs is not None
        q_dcp_replicated: Any = None
        if fused_qb_out is not None:
            ql_head, ql_tail, q_pe_all = fused_qb_out
            q_dcp_replicated = ((ql_head, ql_tail), q_pe_all)
        elif in_branch_qb_inputs is not None:
            q_dcp_replicated = None
        elif self.dcp_q_replicate:
            if q_rope_table is not None:
                q_dcp_replicated = (q_nope, q, q_rope_table)
            elif self.is_sparse:
                q_dcp_replicated = (q_nope, q_pe)
            q_proj_layer = cast(DCPGroupColumnParallelLinear, self._q_proj_layer())
            if q_dcp_replicated is not None and has_topk:
                n = q_nope.shape[-2] // q_proj_layer.group_size
                q_nope, q_pe = q_nope[..., :n, :], q_pe[..., :n, :]
            else:
                local_view = q_proj_layer._local_view
                q_nope, q_pe = local_view(q_nope), local_view(q_pe)

        if not has_topk:
            sp_o_proj = None

        attn_out = self.mla_attn(
            (q_nope, q_pe),
            kv_c_normed,
            k_pe,
            output_shape=(num_tokens, self.num_heads * self.v_head_dim),
            topk_indices=topk_indices,
            q_dcp_replicated=q_dcp_replicated,
            sp_o_proj=sp_o_proj,
            kv_sharded=kv_sharded,
            unified_indexer_inputs=unified_idx_inputs,
            in_branch_qb_inputs=in_branch_qb_inputs,
        )
        if sp_o_proj is not None:
            return _retain_buffers(
                attn_out,
                cast(torch.Tensor, self.mla_attn.W_UV),
                cast(torch.Tensor, self.o_proj.weight),
                cast(torch.Tensor, self.o_proj.weight_scale),
                getattr(self.o_proj, FP8_PALLAS_WEIGHT, None),
                getattr(self.o_proj, FP8_PALLAS_SCALE, None),
            )
        if isinstance(attn_out, torch.Tensor):
            attn_out = _retain_buffers(
                attn_out,
                getattr(self.mla_attn, "W_UV_tp_all", None),
                getattr(self.mla_attn, "W_O_tp_half", None),
                getattr(self.mla_attn, "W_O_tp_half_scale", None),
            )

        if output_gate is not None:
            attn_out *= output_gate.sigmoid()
        elif self.g_proj is not None:
            attn_out *= self.g_proj(hidden_states)[0].sigmoid()
        if prefill_group is not None:
            return prefill_group.project_reduce_scatter(attn_out, self.o_proj)
        if post_attn_norm is not None:
            res_in, norm_w, norm_eps = post_attn_norm
            return small_all_reduce_op.row_parallel_or_rmsnorm(
                attn_out, self.o_proj, res_in, norm_w, norm_eps
            )
        return _row_parallel_or_default(attn_out, self.o_proj)


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


def split_qkv_a_proj_widths(
    q_lora_rank: int, kv_dim: int, chunk_cols: int
) -> tuple[int, ...]:
    """Partition `[q_lora_rank | kv_dim]` into `(chunk_cols..., rem, kv_dim)`."""
    if q_lora_rank <= 0 or kv_dim <= 0 or chunk_cols <= 0:
        return ()
    parts: list[int] = []
    remaining = q_lora_rank
    while remaining > chunk_cols:
        parts.append(chunk_cols)
        remaining -= chunk_cols
    parts.append(remaining)
    parts.append(kv_dim)
    return tuple(parts)


def arm_split_qkv_a_proj(model: torch.nn.Module, chunk_cols: int | None = None) -> int:
    """Split each MLA wrapper's `fused_qkv_a_proj` weight into disjoint columns."""
    chunk = MLA_SPLIT_QKV_A_CHUNK if chunk_cols is None else int(chunk_cols)
    if chunk <= 0:
        return 0
    count = 0
    for module in model.modules():
        if not isinstance(module, VllmTPUMultiHeadLatentAttentionWrapper):
            continue
        if (
            module.kimi_preprocess is not None
            or module.gate_is_fused
            or module.q_lora_rank is None
            or module.fused_qkv_a_proj is None
        ):
            continue
        kv_dim = int(module.kv_lora_rank) + int(module.qk_rope_head_dim)
        widths = split_qkv_a_proj_widths(int(module.q_lora_rank), kv_dim, chunk)
        if len(widths) >= 2 and split_fp8_linear_columns(
            module.fused_qkv_a_proj, widths
        ):
            count += 1
    return count
