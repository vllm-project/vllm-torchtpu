"""TPU MLA and compressor adapters with upstream cache ownership."""

import contextlib
import dataclasses
import math
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.utils import replace_parameter
from vllm.models.common.ops.sequence_parallel import sp_all_gather
from vllm.models.deepseek_v41 import attention as upstream
from vllm.models.deepseek_v41 import compressor as compressor_upstream
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

from vllm_torchtpu.kernels.experimental import decode_preparation
from vllm_torchtpu.layers.adapter.linear_common import (
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.core import table_lookup
from vllm_torchtpu.utils import synchronize_tensors

from . import csa2_adapter, preprocess_ops
from .attention_kernels import (
    gathered_kv_cache_scope,
    paged_attention,
)
from .cache import (
    BLOCK_STRIDE_ALIGNMENT,
    SWA_PAGE_TOKENS,
    IndexCacheRows,
    NativeCacheView,
    PagedCacheRows,
    write_cache,
)
from .ced import prefix_replay_enabled
from .indexer_cache import IndexerCache, PackedFP8IndexCacheRows, indexer_uses_fp8
from .ops import (
    device_hooks,
    quantize_fp8_activation,
    rms_norm,
    rotary,
    round_fp8_activation,
)
from .projection_rope import initialize_projection_rope
from .sparse_mqa_indexer import SparseMQAIndexer, quantize_query

logger = init_logger(__name__)

_active_stage_attn_cache: dict[Any, Any] | None = None


@contextlib.contextmanager
def stage_attention_cache_scope():
    """Share addressing and selected global KV payloads across stage layers."""
    global _active_stage_attn_cache
    prev = _active_stage_attn_cache
    _active_stage_attn_cache = {}
    try:
        with gathered_kv_cache_scope():
            yield
    finally:
        _active_stage_attn_cache = prev


def select_candidate_blocks(logits, lengths, topk_blocks, block_size):
    if block_size < 1 or topk_blocks < 0:
        raise ValueError("invalid candidate block parameters")
    width = logits.shape[-1]
    if width == 0:
        return torch.zeros_like(logits, dtype=torch.bool)
    lengths = torch.as_tensor(lengths, device=logits.device)
    visible = torch.arange(width, device=logits.device) < lengths
    scores = F.pad(
        logits.masked_fill(~visible, -torch.inf),
        (0, -width % block_size),
        value=-torch.inf,
    )
    scores = scores.unflatten(-1, (-1, block_size)).amax(-1)
    last = (lengths - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(scores.shape[-1], device=scores.device) == last, torch.inf
    )
    selected = scores.topk(min(topk_blocks, scores.shape[-1]), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, selected.indices, selected.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def select_compressed_indices(scores, lengths, topk, *, candidates=None):
    """Return logical compressed positions; unavailable picks are always -1.

    Do not carry a source layer's SWA concatenation offset into reuse layers.
    Equal scores retain PyTorch topk's unspecified tie ordering.
    """
    if topk < 0:
        raise ValueError("topk must be nonnegative")
    valid = torch.arange(scores.shape[-1], device=scores.device) < torch.as_tensor(
        lengths, device=scores.device
    )
    if candidates is not None:
        valid = valid & candidates
    masked = scores.masked_fill(~valid, -torch.inf)
    indices = masked.topk(min(topk, scores.shape[-1]), dim=-1).indices.sort(-1).values
    return indices.masked_fill(masked.gather(-1, indices) == -torch.inf, -1).int()


def sparse_attention(q, kv, sink, indices, scale):
    """Shared K/V latent with a per-query-head zero-value attention sink."""
    if kv.shape[1] == 0 or indices.shape[-1] == 0:
        return torch.zeros_like(q)
    valid = (indices >= 0) & (indices < kv.shape[1])
    batch = torch.arange(kv.shape[0], device=kv.device)[:, None, None]
    selected = kv[batch, indices.long().clamp(0, kv.shape[1] - 1)]
    selected = selected.masked_fill(~valid.unsqueeze(-1), 0).float()
    logits = torch.einsum("bshd,bskd->bshk", q.float(), selected) * scale
    logits = logits.masked_fill(~valid.unsqueeze(-2), -torch.inf)
    if q.dtype == torch.bfloat16:
        # Match the released kernel: online blocks of 64, BF16 unnormalized
        # exponentials for the value GEMM, FP32 denominator, sink added last.
        maximum = torch.full_like(logits[..., 0], -1e30)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(q, dtype=torch.float32)
        for start in range(0, logits.shape[-1], 64):
            scores = logits[..., start : start + 64]
            new_maximum = torch.maximum(maximum, scores.amax(-1))
            correction = torch.exp(maximum - new_maximum)
            exponentials = torch.exp(scores - new_maximum.unsqueeze(-1))
            denominator = denominator * correction + exponentials.sum(-1)
            numerator = numerator * correction.unsqueeze(-1) + torch.einsum(
                "bshk,bskd->bshd",
                exponentials.to(torch.bfloat16).float(),
                selected[:, :, start : start + 64],
            )
            maximum = new_maximum
        denominator = denominator + torch.exp(sink.float() - maximum)
        return (numerator / denominator.unsqueeze(-1)).to(q.dtype)
    sinks = sink.float().expand(*logits.shape[:-1]).unsqueeze(-1)
    probs = torch.cat((logits, sinks), dim=-1).softmax(-1)[..., :-1]
    return torch.einsum("bshk,bskd->bshd", probs, selected).to(q.dtype)


def _block_table(metadata):
    return metadata.block_tables.reshape(metadata.seq_lens.shape[0], -1)


def _token_requests(metadata, positions, out_int32=False):
    dtype = torch.int32 if out_int32 else torch.int64
    rows = torch.arange(positions.shape[0], device=positions.device, dtype=dtype)
    return table_lookup.bucketize_right(
        rows, metadata.query_start_loc[1:], out_int32=out_int32
    )


def _window_starts():
    """Per-request first position holding SWA KV, or None (SWA bounded replay)."""
    return get_forward_context().additional_kwargs.get("deepseek_v41_window_starts")


def _shift_swa_metadata(metadata, records, shift, page_size):
    """Move CSA2 SWA metadata and write records by ``shift`` rows.

    ``prepare_swa_metadata`` adds the row offset to every visible physical
    page (``offset // page_size``) and to every valid record; hidden pages are
    0 and padding records -1. A visible page reads 0 only for the null block
    at offset 0, which holds no KV for any layer.
    """
    if not shift:
        return metadata, records
    pages = metadata[:, 2:]
    pages = torch.where(pages > 0, pages + shift // page_size, pages)
    metadata = torch.cat((metadata[:, :2], pages), dim=1)
    records = torch.where(records >= 0, records + shift, records)
    return metadata, records


def _swa_group_tables(metadata, shape):
    """Distinct SWA block tables of this forward, in first-use order.

    Block-outer packing deals SWA layers to several groups, each with its own
    table; layers of one group share the table object. Tables of another
    shape (another request layout) are left to their own preparation.
    """
    tables = []
    for name, value in metadata.items():
        table = getattr(value, "block_tables", None)
        if (
            name.endswith(".swa_cache")
            and table is not None
            and table.numel() == shape[0] * shape[1]
            and not any(table is seen for seen in tables)
        ):
            tables.append(table)
    return tables


def _replayed(metadata, positions, starts, window):
    """Tokens a prefix hit replays: their prefix-cached KV must stay as is."""
    requests = _token_requests(metadata, positions).clamp(max=starts.shape[0] - 1)
    start = table_lookup.take(starts, requests)
    return (start > 0) & (positions >= start) & (positions < start + window)


def _cache_slots(
    cache,
    metadata,
    positions,
    *,
    compress_ratio=1,
    ring=False,
    wide=True,
    offset=0,
):
    """Addresses of ``positions + offset`` from the shared TPU metadata contract.

    ``wide=False`` computes in int32, for a cache whose slots fit in it.
    Layers of one cache group read the same tables, so a stage computes each
    address set once (~4 us of lookups per call).
    """
    stage_cache = _active_stage_attn_cache
    key = ("cache_slots", cache.shape[1], compress_ratio, ring, wide, offset)
    if stage_cache is not None:
        for tables, starts, cached_positions, slots in stage_cache.get(key, ()):
            if (
                tables is metadata.block_tables
                and starts is metadata.query_start_loc
                and cached_positions is positions
            ):
                return slots
    slots = _compute_cache_slots(
        cache,
        metadata,
        positions + offset if offset else positions,
        compress_ratio=compress_ratio,
        ring=ring,
        wide=wide,
    )
    if stage_cache is not None:
        entry = (metadata.block_tables, metadata.query_start_loc, positions, slots)
        stage_cache.setdefault(key, []).append(entry)
    return slots


def _compute_cache_slots(cache, metadata, positions, *, compress_ratio, ring, wide):
    if metadata.seq_lens.numel() == 0:
        return torch.full_like(positions, -1)
    table = _block_table(metadata)
    requests = _token_requests(metadata, positions, out_int32=not wide).clamp(
        max=table.shape[0] - 1
    )
    rows_per_page = cache.shape[1]
    logical = positions // compress_ratio
    columns = torch.zeros_like(logical) if ring else logical // rows_per_page
    columns = columns.clamp(max=table.shape[1] - 1)
    blocks = table[requests, columns.long() if wide else columns]
    if wide:
        blocks = blocks.long()
    slots = blocks * rows_per_page + logical % rows_per_page
    # Padded request boundaries repeat the last real token offset. Keep the
    # mask on device, including dummy runs and batches with no active tokens.
    rows = torch.arange(positions.shape[0], device=positions.device)
    if not wide:
        rows = rows.int()
    return slots.masked_fill(rows >= metadata.query_start_loc[-1], -1)


class CompressorStateCache(NativeCacheView, compressor_upstream.CompressorStateCache):
    """Upstream CircularBufferSpec, allocation lifecycle and cache binding."""

    def get_attn_backend(self):
        return AttentionBackend

    def get_kv_cache_spec(self, vllm_config):
        return dataclasses.replace(
            super().get_kv_cache_spec(vllm_config),
            block_stride_alignment=BLOCK_STRIDE_ALIGNMENT,
        )


def save_compress_norm(
    kv_score,
    positions,
    state_cache,
    slot_mapping,
    query_start_loc,
    token_to_req_indices,
    norm_weight,
    eps,
    ratio,
    latent,
):
    """Tensor implementation of the compressor_upstream fused compression operation.

    Read the previous row before publishing ring writes. For a long prefill,
    previous rows come directly from this chunk and only the final ring-sized
    suffix is stored, so writes never race aliases from earlier ring wraps.
    """
    count = kv_score.shape[0]
    if slot_mapping is not None:
        slot_mapping = slot_mapping[:count]
    if token_to_req_indices is not None:
        token_to_req_indices = token_to_req_indices[:count]
    dim = norm_weight.numel()
    values = kv_score[:, :dim].float()
    valid = slot_mapping >= 0
    if ratio == 2:
        capacity = state_cache.shape[1]
        cache = state_cache
        rows = torch.arange(kv_score.shape[0], device=kv_score.device)
        requests = token_to_req_indices.long().clamp(max=query_start_loc.numel() - 2)
        starts = table_lookup.take(query_start_loc, requests)
        ends = table_lookup.take(query_start_loc, requests + 1)
        previous_slots = (
            slot_mapping.clamp(min=0) // capacity * capacity
            + (positions - 1) % capacity
        )
        stored = cache[previous_slots // capacity, previous_slots % capacity]
        previous = torch.where(
            (rows > starts)[:, None], kv_score[(rows - 1).clamp(min=0)].float(), stored
        )
        scores = torch.stack((previous[:, dim:], kv_score[:, dim:].float()), 1)
        pair = torch.stack((previous[:, :dim], values), 1)
        values = (pair * scores.softmax(1)).sum(1)
        # All reads above refer to the old ring. Keep unique physical writes.
        publish = valid & (rows >= ends - capacity)
        write_cache(cache, slot_mapping, kv_score.float(), publish, after=stored)
        valid = valid & (positions % 2 == 1)
    elif ratio != 1:
        raise ValueError(f"V4.1 compression ratio must be 1 or 2, got {ratio}")
    result = rms_norm(values, norm_weight, eps)
    latent.copy_(result.masked_fill(~valid[:, None], 0).to(latent.dtype))


class DeepseekCompressor(compressor_upstream.DeepseekCompressor):
    """Inherit weights and state ownership; consume the shared TPU metadata."""

    def __init__(self, *args, **kwargs):
        with device_hooks(
            compressor_upstream, CompressorStateCache=CompressorStateCache
        ):
            super().__init__(*args, **kwargs)

    def project(self, hidden_states):
        from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR

        layer = self.fused_wkv_wgate
        weight = layer.weight.float()
        if not getattr(layer, WEIGHT_FLIPPED_ATTR, False):
            weight = weight.t()
        # Upstream accumulates the unquantized KV/score projection in FP32.
        # Keep that precision through pooling and normalization.
        return torch.mm(hidden_states.float(), weight)

    def forward(self, kv_score, positions):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None
        if self.state_cache is None:
            state_cache = query_start_loc = requests = None
            cache = self._static_forward_context[self.k_cache_prefix].kv_cache
            slots = _cache_slots(cache, metadata[self.k_cache_prefix], positions)
        else:
            state_metadata = metadata[self.state_cache.prefix]
            state_cache = self.state_cache.kv_cache
            stage_cache = _active_stage_attn_cache
            state_rows = (
                state_cache.shape[1]
                if isinstance(getattr(state_cache, "shape", None), tuple)
                else None
            )
            if (
                stage_cache is not None
                and state_rows is not None
                and state_metadata.seq_lens.numel() > 0
            ):
                table = _block_table(state_metadata)
                state_key = ("compressor_state_addr", table.shape, state_rows)
                cached_state = stage_cache.get(state_key)
                if cached_state is None:
                    query_start_loc = state_metadata.query_start_loc
                    requests = _token_requests(state_metadata, positions)
                    clamped_requests = requests.clamp(max=table.shape[0] - 1)
                    ring_rem = positions % state_rows
                    token_rows = torch.arange(
                        positions.shape[0], device=positions.device
                    )
                    valid_token = token_rows < query_start_loc[-1]
                    cached_state = (
                        query_start_loc,
                        requests,
                        clamped_requests,
                        ring_rem,
                        valid_token,
                    )
                    stage_cache[state_key] = cached_state
                else:
                    (
                        query_start_loc,
                        requests,
                        clamped_requests,
                        ring_rem,
                        valid_token,
                    ) = cached_state
                blocks = table[clamped_requests, 0].long()
                slots = (blocks * state_rows + ring_rem).masked_fill(~valid_token, -1)
            else:
                slots = _cache_slots(state_cache, state_metadata, positions, ring=True)
                query_start_loc = state_metadata.query_start_loc
                requests = _token_requests(state_metadata, positions)
        latent = (
            torch.zeros_like(positions[:, None], dtype=torch.bfloat16)
            .expand(-1, self.head_dim)
            .contiguous()
        )
        save_compress_norm(
            kv_score,
            positions,
            state_cache,
            slots,
            query_start_loc,
            requests,
            self.norm.weight,
            self.rms_norm_eps,
            self.compress_ratio,
            latent,
        )
        return latent


def indexer_uses_fp4(vllm_config):
    """Resolve the upstream format without imposing a CUDA capability check."""
    indexer_uses_fp8(vllm_config)
    return False


class AttentionBackend(compressor_upstream.CompressorBackend):
    @staticmethod
    def get_name():
        return "TORCH_TPU_DEEPSEEK_V41"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, *args, **kwargs
    ):
        return num_blocks, num_kv_heads, block_size, head_size


class SWACache(NativeCacheView, DeepseekV4SWACache):
    def __init__(self, *args, **kwargs):
        # SWA pages keep CSA2's 64-token word-pool geometry, independent of the
        # compressed caches' `--block-size`: a larger compressed block then
        # packs several SWA layers into one block-outer cache group.
        kwargs["block_size"] = SWA_PAGE_TOKENS
        # Upstream ties bounded replay to its V2 model runner; the TPU model
        # implements the replay itself (see `prefix_replay_enabled`).
        kwargs["bounded_replay"] = prefix_replay_enabled(get_current_vllm_config())
        # FP8 applies to the compressed main KV only; SWA rows stay BF16.
        kwargs["dtype"] = torch.bfloat16
        kwargs["backend_cls"] = AttentionBackend
        super().__init__(*args, **kwargs)

    def get_kv_cache_spec(self, vllm_config):
        return dataclasses.replace(
            super().get_kv_cache_spec(vllm_config),
            block_stride_alignment=BLOCK_STRIDE_ALIGNMENT,
        )


def _value(output):
    return output[0] if isinstance(output, tuple) else output


def _insert(cache, slots, values, valid):
    write_cache(cache, slots, values, valid)


def _gather_query_rows(rows):
    """All-gather per-rank query rows (dim 0), padded to 128 lanes."""
    width = rows.shape[-1]
    padding = (-width) % 128
    if padding:
        rows = F.pad(rows, (0, padding))
    rows = tensor_model_parallel_all_gather(rows.contiguous(), dim=0)
    return rows[:, :width].contiguous() if padding else rows


def _row_slots(cache, block_table, requests, logical):
    """Gather fixed-width token rows without inspecting device values on host."""
    safe = logical.clamp(min=0, max=block_table.shape[1] * cache.shape[1] - 1)
    # Gather each token's table row, then index within it 128 columns at a
    # time. A [tokens, width] elementwise 2-D gather is 16-40x slower on TPU.
    rows = block_table[requests]
    pages = (safe // cache.shape[1]).long()
    blocks = torch.cat(
        [
            rows.gather(1, pages[:, start : start + 128])
            for start in range(0, pages.shape[1], 128)
        ],
        dim=1,
    ).long()
    return blocks * cache.shape[1] + safe % cache.shape[1]


def _bound_geometry(module) -> tuple[int, int, int] | None:
    """``(records per page, page stride, row offset)`` of a bound native cache.

    Read from the allocator layout in record units, like `PagedCacheRows`, so
    it is available before the first forward builds a cache view.
    """
    layout = getattr(module, "_tpu_cache_layout", None)
    if layout is None:
        return None
    shape, dtype, stride_bytes, offset_bytes = layout
    record_bytes = math.prod(shape[2:]) * dtype.itemsize
    return shape[1], stride_bytes // record_bytes, offset_bytes // record_bytes


def _index_page_records(index_cache):
    """Records per page, or zero when the physical layout breaks page alignment."""
    if not isinstance(index_cache, IndexCacheRows):
        return 0
    per_page = index_cache.shape[1]
    if index_cache.row_offset % per_page or index_cache.page_stride % per_page:
        return 0
    if index_cache.raw_hi is not None and index_cache.bank_records % per_page:
        return 0
    return per_page


def _query_split(count, ranks):
    """``(part, padding)``: rows per TP rank and the static pad that covers them.

    ``part`` is ``ceil(count / ranks)`` rounded up to even, so never 1: a part
    of 1 row (``ceil`` at T=4/8 on 8 ranks) makes Dynamo 0/1-specialize and
    trace a second encoder/decoder graph for those buckets.
    ``ranks * part - count`` is at most ``2 * ranks - 1``, so
    ``padding = 2 * ranks`` rows always cover it.
    """
    return (count + 2 * ranks - 1) // (2 * ranks) * 2, 2 * ranks


def _request_row_slots(cache, block_table, width):
    """Slots of logical rows ``0..width-1`` for every request: ``[requests, width]``.

    Every token of a request scans the same capacity-wide rows, so address
    them once per request instead of once per token.
    """
    rows_per_page = cache.shape[1]
    # Logical row r lives in table column r // rows_per_page, so repeating each
    # column rows_per_page times lines blocks up with rows 0..width-1 without
    # a full-width gather (16-40x slower on TPU, see _row_slots).
    blocks = (
        block_table[:, :, None]
        .expand(-1, -1, rows_per_page)
        .reshape(block_table.shape[0], -1)
        .long()
    )
    offsets = (
        torch.arange(width, device=block_table.device).clamp(max=blocks.shape[1] - 1)
        % rows_per_page
    )
    if width > blocks.shape[1]:
        # Match _row_slots' clamp to the table's last row.
        blocks = torch.cat(
            (blocks, blocks[:, -1:].expand(-1, width - blocks.shape[1])), dim=1
        )
    return blocks[:, :width] * rows_per_page + offsets


def _read_rows(cache, block_table, requests, logical):
    slots = _row_slots(cache, block_table, requests, logical)
    return cache[slots // cache.shape[1], slots % cache.shape[1]]


class DeepseekV41Attention(NativeCacheView, upstream.DeepseekV4Attention):
    backend_cls = AttentionBackend
    swa_backend_cls = AttentionBackend
    use_fp8_ds_mla_layout = False
    _ced_decoder = False
    _ced_global_from_encoder = False
    _sp_sharded_input = False
    _q_projection_rope_op: Any = None
    # TP1 FP8-KV layers run the fused Q-A/KV, Q-B and compressor kernels.
    _fused_preprocess = False
    _packed_compressor_op: Any = None
    _address_compressor_op: Any = None
    # `CompressorLayout` fields the fused compressor ops were built for, set
    # once the KV cache is bound (`bind_compressor_layout`).
    _compressor_layout_fields: tuple[int, ...] | None = None
    _swa_query_group_size = 1

    @classmethod
    def get_padded_num_q_heads(cls, num_heads):
        return num_heads

    def __init__(self, vllm_config, *args, **kwargs):
        with (
            device_hooks(
                upstream,
                DeepseekCompressor=DeepseekCompressor,
                dsa_indexer_uses_fp4=indexer_uses_fp4,
                DeepseekV4IndexerCache=IndexerCache,
                DeepseekV4SWACache=SWACache,
            ),
            device_hooks(torch.cuda, Event=lambda: None),
        ):
            super().__init__(vllm_config, *args, **kwargs)
        if self.kv_cache_torch_dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise NotImplementedError(
                "TPU V4.1 supports --kv-cache-dtype auto, bfloat16 or fp8"
            )
        # FP8 main KV uses CSA2 records: the whole post-RoPE latent in e4m3,
        # with one FP32 scale per record in a companion (csa2.cache).
        self._fp8_kv = self.kv_cache_torch_dtype == torch.float8_e4m3fn
        speculative = vllm_config.speculative_config
        if speculative is not None and speculative.num_speculative_tokens is not None:
            self._swa_query_group_size = speculative.num_speculative_tokens + 1
        # Every FP8 layer, at any TP size, reads and writes CSA2 records.
        self._csa2 = self._fp8_kv
        if self._csa2 and not (
            self.head_dim == 512
            and self.window_size == 128
            and self.n_local_heads % 8 == 0
            and 8 <= self.n_local_heads <= 128
        ):
            raise NotImplementedError(
                "FP8 V4.1 KV (CSA2) needs head_dim 512, window 128 and 8-128 "
                f"local heads divisible by 8, got {self.n_local_heads}"
            )
        self._fused_wo_b = False
        # The fused Q-A/KV, Q-B and compressor kernels are TP1, 64-head kernels;
        # other TP sizes write the same CSA2 records through the unfused path.
        self._fused_preprocess = (
            self._csa2
            and get_tensor_model_parallel_world_size() == 1
            and self.n_local_heads == 64
        )
        self.swa_cache_layer._tpu_swa_words = self._csa2
        if self.compress_ratio:
            logger.info_once(
                "DeepSeek-V4.1 attention backend: %s",
                "csa2_paged" if self._csa2 else "native",
            )

        self.register_buffer(
            "_tpu_rope_cache", self.rotary_emb.cos_sin_cache, persistent=False
        )
        # The TP rank as data, not a Python constant: every rank must compile
        # the same program, or XLA may order its collectives differently per
        # rank and deadlock (see `_select_paged_indices`).
        self.register_buffer(
            "_tp_rank",
            torch.tensor(
                get_tensor_model_parallel_rank(),
                dtype=torch.int32,
                device=self.rotary_emb.cos_sin_cache.device,
            ),
            persistent=False,
        )
        self._config = vllm_config.model_config.hf_config.to_dict()
        if self.indexer is not None:
            indexer = cast(Any, self.indexer)
            op = indexer.indexer_op
            indexer.indexer_op = SparseMQAIndexer(
                indexer.k_cache,
                self._config["index_topk"],
                op.candidate_blocks,
                self.candidate_block_size,
                candidate_source_layer=(
                    self.candidate_source_layer
                    if op.candidate_blocks is not None
                    else None
                ),
                candidate_write=getattr(op, "candidate_write", False),
            )
        self._max_model_len = vllm_config.model_config.max_model_len
        object.__setattr__(self, "mla_attn", self)
        self._prepare_and_attn_fn = self._prepare_tpu

    def get_kv_cache_spec(self, vllm_config):
        spec = super().get_kv_cache_spec(vllm_config)
        if spec is None:
            return spec
        spec = dataclasses.replace(spec, block_stride_alignment=BLOCK_STRIDE_ALIGNMENT)
        if not self._fp8_kv:
            return spec
        return dataclasses.replace(spec, dtype=torch.uint8, page_size_padded=None)

    # Mirrors upstream vllm/models/deepseek_v41/attention.py:561-596
    # (DeepseekV41Attention.forward) with SP-sharded input projection support
    # and SymInt-safe o_padded allocation.
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del llama_4_scaling
        num_tokens = positions.shape[0]
        # Allocate o_padded from positions via zeros_like + expand so TorchTPU's
        # lowering does not call torch.empty on a compound SymInt expression.
        o_padded = (
            torch.zeros_like(
                positions[:, None, None],
                dtype=hidden_states.dtype,
            )
            .expand(-1, self.padded_heads, self.head_dim)
            .contiguous()
        )
        if self._fused_preprocess:
            # One kernel projects, normalizes and rotates qr/kv; the
            # compressor projects inside its own kernel.
            assert not self._sp_sharded_input  # TP1; SP needs TP > 1
            qr, kv = self._qa_kv_norm_rope(hidden_states, positions)
            qr_scale = kv_score = None
            indexer_weights = (
                _value(self.indexer.weights_proj(hidden_states))
                if self.indexer is not None and not self._all_indices_fit
                else None
            )
        else:
            qr_kv, kv_score, indexer_weights = self._run_parallel_input_projections(
                hidden_states
            )
            qr, qr_scale, kv = self._split_qkv_and_norm(qr_kv)
        if self._sp_sharded_input:
            if indexer_weights is not None:
                assert qr.dtype == kv.dtype == indexer_weights.dtype
                packed = sp_all_gather(torch.cat((qr, kv, indexer_weights), dim=-1))[
                    :num_tokens
                ]
                qr, kv, indexer_weights = (
                    t.contiguous()
                    for t in packed.split(
                        (self.q_lora_rank, self.head_dim, self.indexer.n_head),
                        dim=-1,
                    )
                )
            else:
                assert qr.dtype == kv.dtype
                packed = sp_all_gather(torch.cat((qr, kv), dim=-1))[:num_tokens]
                qr, kv = (
                    t.contiguous()
                    for t in packed.split(
                        (self.q_lora_rank, self.head_dim),
                        dim=-1,
                    )
                )
            if kv_score is not None:
                kv_score = sp_all_gather(kv_score)[:num_tokens]
        args = (
            hidden_states,
            qr,
            kv,
            qr_scale,
            kv_score,
            indexer_weights,
            positions,
            o_padded,
        )
        self._prepare_and_attn_fn(*args)
        o = o_padded[:, : self.n_local_heads, :]
        return self._o_proj(o, positions)

    def _run_parallel_input_projections(self, hidden_states):
        qr_kv = self._fused_wqa_wkv_gemm(hidden_states)
        kv_score = (
            self.compressor.project(hidden_states)
            if self.compressor is not None and not self._ced_global_from_encoder
            else None
        )
        index_weights = (
            _value(self.indexer.weights_proj(hidden_states))
            if self.indexer is not None and not self._all_indices_fit
            else None
        )
        return qr_kv, kv_score, index_weights

    @property
    def _all_indices_fit(self):
        return bool(self.compress_ratio) and (
            self._max_model_len // self.compress_ratio <= self._config["index_topk"]
        )

    def _quantize_indexer(self, values):
        # Values are kept dequantized in BF16 storage, not silently changed to
        # another quantizer. Power-of-two FP8 scales remain with Q/K, so weights
        # must NOT fold the Q scale a second time.
        return round_fp8_activation(values, self.indexer.head_dim)

    def _project_index_q(self, qr, positions):
        return self._quantize_indexer(self._project_index_q_unquantized(qr, positions))

    def _project_index_q_unquantized(self, qr, positions):
        index_q = _value(self.indexer.wq_b(qr)).reshape(
            -1, self.indexer.n_head, self.indexer.head_dim
        )
        return self._rotate(index_q, positions)

    def _split_qkv_and_norm(self, qr_kv):
        return (
            self.q_norm(qr_kv[:, : self.q_lora_rank]),
            None,
            self.kv_norm(qr_kv[:, self.q_lora_rank :]),
        )

    def _qa_kv_norm_rope(self, hidden_states, positions, *, emit_qr=True):
        """Fused ``_fused_wqa_wkv_gemm`` -> ``_split_qkv_and_norm`` -> SWA ``_rotate``.

        Returns ``(qr, kv)``, or the rotated ``kv`` alone without ``emit_qr``.
        """
        layer = self.fused_wqa_wkv
        return preprocess_ops.qa_rope(emit_qr, self.kv_norm.variance_epsilon)(
            hidden_states,
            layer.weight,
            layer.weight_scale,
            self.q_norm.weight,
            self.kv_norm.weight,
            self._rope_rows(positions),
        )

    def _rope_rows(self, positions, ratio=1):
        """``_tpu_rope_cache`` rows at ``positions``, gathered once per stage.

        ``ratio`` > 1 marks ``_group_positions(positions)``; ratio-1 groups
        share the token rows of the Q-A/KV and Q-B kernels.
        """
        assert positions.dtype == torch.int32
        table = self._tpu_rope_cache
        stage_cache = _active_stage_attn_cache
        key = ("rope_rows", id(table), ratio)
        cached = stage_cache.get(key) if stage_cache is not None else None
        if cached is not None and cached[0] is positions and cached[1] is table:
            return cached[2]
        # Set at load (runner.utils.relayout_rope_tables) for 64-wide tables.
        pairs = getattr(self, "_tpu_rope_pairs", None)
        if pairs is not None:
            rows = preprocess_ops.rope_paired_rows_sc()(pairs, positions)
        else:
            rows = preprocess_ops.rope_rows()(table, positions)
        if stage_cache is not None:
            stage_cache[key] = (positions, table, rows)
        return rows

    def _group_positions(self, positions):
        if self.compress_ratio == 1:
            return positions
        cache = _active_stage_attn_cache
        key = ("compress_group_positions", self.compress_ratio)
        previous = cache.get(key) if cache is not None else None
        if previous is not None and previous[0] is positions:
            return previous[1]
        grouped = positions // self.compress_ratio * self.compress_ratio
        if cache is not None:
            cache[key] = (positions, grouped)
        return grouped

    @property
    def kv_rope_op(self):
        op = self.__dict__.get("_kv_rope_op_instance")
        if op is not None:
            return op
        from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
            VllmDeepseekV4MLAAttention,
        )

        return cast(Callable, VllmDeepseekV4MLAAttention.kv_rope_op.fget)(self)

    def _rotate(self, values, positions, inverse=False):
        if values.device.type != "cpu" and not inverse:
            return self.kv_rope_op(values, positions, self._tpu_rope_cache)
        return rotary(
            values.unsqueeze(0),
            positions,
            self._config,
            compressed=bool(self.compress_ratio),
            inverse=inverse,
        ).squeeze(0)

    def _produce_index_k(self, latent, positions, cache, slots, valid) -> None:
        """Publish index keys from the compressor's pre-RoPE latent."""
        indexer = self.indexer
        keys = indexer.k_norm(_value(indexer.wk(latent)))
        keys = self._rotate(keys, positions)
        if not isinstance(cache, PackedFP8IndexCacheRows):
            keys = self._quantize_indexer(keys)
        _insert(cache, slots, keys, valid)
        stage_cache = _active_stage_attn_cache
        if stage_cache is not None:
            version_key = ("candidate_index_version", indexer.k_cache.prefix)
            stage_cache[version_key] = stage_cache.get(version_key, 0) + 1

    def _compressor_layout_admitted(
        self, region, state_cache, main_cache=None, index_cache=None
    ) -> bool:
        """Compare the bound cache geometry with the registered kernel layout."""
        fields = dict(
            state_capacity=state_cache.shape[1],
            state_stride=state_cache.page_stride,
            state_offset=state_cache.row_offset,
            window=self.window_size,
        )
        if main_cache is not None and index_cache is not None:
            fields.update(
                main_capacity=main_cache.shape[1],
                main_stride=main_cache.page_stride,
                main_offset=main_cache.row_offset,
                index_capacity=index_cache.shape[1],
            )
        return preprocess_ops.compressor_layout_admitted(
            region, self._compressor_layout_fields, **fields
        )

    def bind_compressor_layout(self) -> None:
        """Rebuild the fused ratio-2 compressor ops for the bound cache geometry.

        Weights load before the KV cache is bound, so the ops built in
        `process_weights_after_loading` carry the default layout. Block-outer
        groups give every layer its own state/main offsets, so each layer's
        kernels are specialized for its own pages here, before any forward.
        """
        if self._packed_compressor_op is None or self.compressor is None:
            return
        state = _bound_geometry(self.compressor.state_cache)
        main = _bound_geometry(self)
        index = _bound_geometry(self.indexer.k_cache)
        if state is None or main is None or index is None:
            return
        layout = decode_preparation.CompressorLayout(
            state_capacity=state[0],
            state_stride=state[1],
            state_offset=state[2],
            window=self.window_size,
            main_capacity=main[0],
            main_stride=main[1],
            main_offset=main[2],
            index_capacity=index[0],
        )
        eps = self.compressor.rms_norm_eps
        self._packed_compressor_op = preprocess_ops.compress_rope_packed(eps, layout)
        self._address_compressor_op = preprocess_ops.compress_rope_addresses(
            eps, layout
        )
        self._compressor_layout_fields = dataclasses.astuple(layout)

    def _write_csa2_global_kv(self, hidden_states, positions):
        """Fused compressor: one kernel projects, pools (reading and writing the
        ratio-2 state ring), normalizes, rotates and CSA2-encodes the records."""
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return
        compressor = self.compressor
        ratio = self.compress_ratio
        kv_meta = metadata[self.prefix]
        index_k = self.indexer.k_cache
        starts = _window_starts()
        group_positions = self._group_positions(positions)
        args = [
            hidden_states,
            compressor.fused_wkv_wgate.weight,
            compressor.norm.weight,
            self._rope_rows(group_positions, ratio),
        ]
        state_meta = metadata[compressor.state_cache.prefix] if ratio == 2 else None
        state_cache = compressor.state_cache.kv_cache if ratio == 2 else None
        # Only structural, non-token conditions are checked here. Token count,
        # request count and table widths are dispatched inside the JAX bridge
        # (decode_preparation.admit), so Dynamo keeps the token dim dynamic.
        packed = (
            self._packed_compressor_op is not None
            and state_meta is not None
            and isinstance(state_cache, PagedCacheRows)
            and state_cache.raw_hi is not None
            and starts is not None
            and state_meta.seq_lens.numel() > 0
            and self._compressor_layout_admitted("compressor_state", state_cache)
        )
        index_meta = metadata[index_k.prefix]
        addresses = (
            packed
            and self._address_compressor_op is not None
            and isinstance(self.kv_cache, PagedCacheRows)
            and isinstance(index_k.kv_cache, PackedFP8IndexCacheRows)
            and index_k.compress_ratio == 2
            # One shared request mapping: the kernel derives both caches'
            # requests from kv query_start_loc.
            and index_meta.query_start_loc is kv_meta.query_start_loc
            and self._compressor_layout_admitted(
                "compressor_addresses", state_cache, self.kv_cache, index_k.kv_cache
            )
        )
        compressed_slots = index_slots = records = None
        if not addresses:
            compressed_slots = _cache_slots(
                self.kv_cache, kv_meta, positions, compress_ratio=ratio
            )
            index_slots = _cache_slots(
                index_k.kv_cache,
                metadata[index_k.prefix],
                positions,
                compress_ratio=index_k.compress_ratio,
            )
        if addresses:
            assert state_meta is not None and state_cache is not None
            assert state_cache.raw_hi is not None
            latent, words, scales, low, high, valid, records, index_slots = (
                self._address_compressor_op(
                    _block_table(state_meta),
                    state_meta.query_start_loc,
                    kv_meta.query_start_loc,
                    positions,
                    starts,
                    *args,
                    state_cache.raw,
                    state_cache.raw_hi,
                    _block_table(kv_meta),
                    _block_table(index_meta),
                )
            )
            state_cache.raw.copy_(low)
            state_cache.raw_hi.copy_(high)
        elif packed:
            assert state_meta is not None and state_cache is not None
            assert state_cache.raw_hi is not None
            latent, words, scales, low, high, valid = self._packed_compressor_op(
                _block_table(state_meta),
                state_meta.query_start_loc,
                kv_meta.query_start_loc,
                positions,
                starts,
                *args,
                state_cache.raw,
                state_cache.raw_hi,
            )
            state_cache.raw.copy_(low)
            state_cache.raw_hi.copy_(high)
        else:
            rows = torch.arange(positions.shape[0], device=positions.device)
            valid = (rows < kv_meta.query_start_loc[-1]) & (
                (positions + 1) % ratio == 0
            )
            if starts is not None:
                valid = valid & ~_replayed(kv_meta, positions, starts, self.window_size)
            pools = ()
            if ratio == 1:
                args += [torch.ones_like(positions[:, None]), valid[:, None].int()]
            else:
                state_meta = metadata[compressor.state_cache.prefix]
                state_cache = compressor.state_cache.kv_cache
                pools = tuple(
                    p for p in (state_cache.raw, state_cache.raw_hi) if p is not None
                )
                starts_loc = state_meta.query_start_loc
                requests = _token_requests(state_meta, positions).clamp(
                    max=starts_loc.numel() - 2
                )
                # A request's first row pools with the ring's stored predecessor;
                # only its last ring-capacity rows are stored.
                has_prev = rows != table_lookup.take(starts_loc, requests)
                slots = _cache_slots(state_cache, state_meta, positions, ring=True)
                previous = _cache_slots(
                    state_cache, state_meta, positions, ring=True, offset=-1
                )
                stored = (slots >= 0) & (
                    rows
                    >= table_lookup.take(starts_loc, requests + 1)
                    - state_cache.shape[1]
                )
                args += [
                    has_prev[:, None].int(),
                    valid[:, None].int(),
                    *pools,
                    state_cache.row_indices(slots).masked_fill(~stored, -1).int(),
                    state_cache.row_indices(previous)
                    .masked_fill(has_prev | (slots < 0), -1)
                    .int(),
                ]
            op = preprocess_ops.compress_rope(
                ratio, len(pools), compressor.rms_norm_eps
            )
            latent, words, scales, *states = op(*args)
            for pool, state in zip(pools, states, strict=True):
                pool.copy_(state)
        if not addresses:
            assert compressed_slots is not None
            records = self.kv_cache.row_indices(compressed_slots.clamp(min=0))
            records = records.masked_fill(~valid | (compressed_slots < 0), -1)
        assert records is not None and index_slots is not None
        csa2_adapter.publish_main(self.kv_cache, records, words, scales)
        self._produce_index_k(
            latent, group_positions, index_k.kv_cache, index_slots, valid
        )

    def _write_global_kv(self, kv_score, positions):
        metadata = get_forward_context().attn_metadata
        latent = self.compressor(kv_score, positions)
        indexer = self.indexer
        index_cache = indexer.k_cache.kv_cache
        kv_meta = metadata[self.prefix]
        idx_meta = metadata[indexer.k_cache.prefix]
        stage_cache = _active_stage_attn_cache
        kv_rows = (
            self.kv_cache.shape[1]
            if isinstance(getattr(self.kv_cache, "shape", None), tuple)
            else None
        )
        idx_rows = (
            index_cache.shape[1]
            if isinstance(getattr(index_cache, "shape", None), tuple)
            else None
        )
        idx_ratio = indexer.k_cache.compress_ratio
        if (
            stage_cache is not None
            and kv_rows is not None
            and idx_rows is not None
            and kv_meta.seq_lens.numel() > 0
            and idx_meta.seq_lens.numel() > 0
        ):
            kv_table = _block_table(kv_meta)
            idx_table = _block_table(idx_meta)
            slot_key = (
                "global_kv_addr",
                kv_table.shape,
                self.compress_ratio,
                kv_rows,
                idx_table.shape,
                idx_ratio,
                idx_rows,
            )
            cached_addr = stage_cache.get(slot_key)
            if cached_addr is None:
                requests = _token_requests(kv_meta, positions).clamp(
                    max=kv_table.shape[0] - 1
                )
                logical = positions // self.compress_ratio
                kv_col = (logical // kv_rows).clamp(max=kv_table.shape[1] - 1).long()
                kv_rem = logical % kv_rows
                if (
                    idx_ratio == self.compress_ratio
                    and idx_rows == kv_rows
                    and idx_table.shape == kv_table.shape
                ):
                    idx_col = kv_col
                    idx_rem = kv_rem
                else:
                    idx_logical = positions // idx_ratio
                    idx_col = (
                        (idx_logical // idx_rows)
                        .clamp(max=idx_table.shape[1] - 1)
                        .long()
                    )
                    idx_rem = idx_logical % idx_rows
                token_rows = torch.arange(positions.shape[0], device=positions.device)
                valid_token = token_rows < kv_meta.query_start_loc[-1]
                valid_group = valid_token & ((positions + 1) % self.compress_ratio == 0)
                group_positions = logical * self.compress_ratio
                cached_addr = (
                    requests,
                    kv_col,
                    kv_rem,
                    idx_col,
                    idx_rem,
                    valid_token,
                    valid_group,
                    group_positions,
                )
                stage_cache[slot_key] = cached_addr
            else:
                (
                    requests,
                    kv_col,
                    kv_rem,
                    idx_col,
                    idx_rem,
                    valid_token,
                    valid_group,
                    group_positions,
                ) = cached_addr
            kv_blocks = kv_table[requests, kv_col].long()
            compressed_slots = (kv_blocks * kv_rows + kv_rem).masked_fill(
                ~valid_token, -1
            )
            valid = (compressed_slots >= 0) & valid_group
            idx_blocks = idx_table[requests, idx_col].long()
            index_slots = (idx_blocks * idx_rows + idx_rem).masked_fill(
                ~valid_token, -1
            )
        else:
            compressed_slots = _cache_slots(
                self.kv_cache,
                kv_meta,
                positions,
                compress_ratio=self.compress_ratio,
            )
            valid = (compressed_slots >= 0) & (
                (positions + 1) % self.compress_ratio == 0
            )
            group_positions = positions // self.compress_ratio * self.compress_ratio
            index_slots = _cache_slots(
                index_cache,
                idx_meta,
                positions,
                compress_ratio=idx_ratio,
            )
        starts = _window_starts()
        if starts is not None:
            # Shared prefix rows must survive replay unchanged in both paths.
            valid = valid & ~_replayed(kv_meta, positions, starts, self.window_size)
        _insert(
            self.kv_cache,
            compressed_slots,
            self._rotate(latent, group_positions),
            valid,
        )
        self._produce_index_k(latent, group_positions, index_cache, index_slots, valid)

    def write_encoder_global_kv(
        self, hidden_states, positions, *, sp_sharded: bool = False
    ):
        """Publish decoder global KV for all encoder tokens before replay."""
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict) or not self.kv_cache.numel():
            return
        if self._fused_preprocess:
            assert not sp_sharded  # fused preprocessing is TP1; SP needs TP > 1
            self._write_csa2_global_kv(hidden_states, positions)
            return
        kv_score = self.compressor.project(hidden_states)
        if sp_sharded:
            kv_score = sp_all_gather(kv_score)[: positions.shape[0]]
        self._write_global_kv(kv_score, positions)

    def _prepare_tpu(
        self,
        hidden_states,
        qr,
        kv,
        qr_scale,
        kv_score,
        index_weights,
        positions,
        output,
    ):
        """Attend into ``output``.

        With ``_fused_preprocess``, ``kv`` already carries its SWA RoPE."""
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict) or not self.swa_cache_layer.kv_cache.numel():
            output.zero_()
            return None
        swa = metadata[self.swa_cache_layer.prefix]
        swa_cache = self.swa_cache_layer.kv_cache
        swa_table = _block_table(swa)
        tiled = isinstance(swa_cache, PagedCacheRows) and self.window_size % 64 == 0
        has_fast_swa_offset = isinstance(getattr(swa_cache, "row_offset", None), int)
        stage_cache = _active_stage_attn_cache
        rows_per_page = (
            swa_cache.shape[1]
            if isinstance(getattr(swa_cache, "shape", None), tuple)
            else None
        )
        non_causal = bool(getattr(self, "non_causal_multi_token_decode", False))
        swa_width = (
            -(-(self.window_size + int(getattr(self, "dspark_block_size", 5))) // 64)
            * 64
            if non_causal
            else self.window_size
        )
        csa2_metadata = csa2_bounds = csa2_write_records = None
        swa_published = False
        if self._csa2:
            if not isinstance(swa_cache, PagedCacheRows):
                raise ValueError("CSA2 requires a native shared word-row cache")
            minimum = torch.zeros_like(swa.query_start_loc[:-1])
            if self._ced_decoder:
                minimum = get_forward_context().additional_kwargs[
                    "deepseek_v41_replay_starts"
                ]
            elif (window_starts := _window_starts()) is not None:
                minimum = window_starts
            minimum = minimum[: swa_table.shape[0]]
            # Block-outer groups give each SWA layer its own row offset within
            # the block; the metadata is otherwise identical, so the key omits
            # the offset and _shift_swa_metadata moves the cached result.
            # Upstream deals SWA layers to groups round-robin, so consecutive
            # layers use different block tables: keep one entry per table, or
            # each layer evicts the previous group's entry and always misses.
            meta_key = (
                "csa2_swa",
                rows_per_page,
                swa_cache.page_stride,
                non_causal,
                self._ced_decoder,
            )
            entries = (
                stage_cache.setdefault(meta_key, []) if stage_cache is not None else []
            )
            cached = next((e for e in entries if e[0] is swa.block_tables), None)
            if cached is None and stage_cache is not None and not entries:
                # First CSA2 layer of the stage: one program prepares every SWA
                # group's table, so later layers of other groups also hit.
                tables = _swa_group_tables(metadata, swa_table.shape)
                if len(tables) > 1 and any(t is swa.block_tables for t in tables):
                    batched, bounds, records = csa2_adapter.prepare_swa_tables(
                        torch.stack([t.reshape(swa_table.shape) for t in tables]),
                        swa.query_start_loc,
                        positions,
                        minimum,
                        swa_cache,
                        non_causal,
                    )
                    entries.extend(
                        (table, 0, batched[g], bounds, records[g])
                        for g, table in enumerate(tables)
                    )
                    cached = next(e for e in entries if e[0] is swa.block_tables)
            if cached is None:
                if (
                    self._fused_preprocess
                    and swa_cache.swa_words
                    # Stride and offset are whole pages (`PagedCacheRows`
                    # checks this for word-pool owners); block-outer groups
                    # place each SWA layer at its own page of the block.
                    and rows_per_page == SWA_PAGE_TOKENS
                    and not non_causal
                ):
                    csa2_metadata, csa2_bounds, csa2_write_records = (
                        csa2_adapter.prepare_publish_swa(
                            swa_table,
                            swa.query_start_loc,
                            positions,
                            minimum,
                            swa_cache,
                            kv.reshape(-1, 512),
                        )
                    )
                    swa_published = True
                else:
                    csa2_metadata, csa2_bounds, csa2_write_records = (
                        csa2_adapter.prepare_swa(
                            swa_table,
                            swa.query_start_loc,
                            positions,
                            minimum,
                            swa_cache,
                            non_causal,
                        )
                    )
                entries.append(
                    (
                        swa.block_tables,
                        swa_cache.row_offset,
                        csa2_metadata,
                        csa2_bounds,
                        csa2_write_records,
                    )
                )
            else:
                _, offset, csa2_metadata, csa2_bounds, csa2_write_records = cached
                csa2_metadata, csa2_write_records = _shift_swa_metadata(
                    csa2_metadata,
                    csa2_write_records,
                    swa_cache.row_offset - offset,
                    rows_per_page,
                )
            swa_valid = csa2_write_records >= 0
            swa_invalid_3d = (~swa_valid)[:, None, None]
            requests = _token_requests(swa, positions).clamp(max=swa_table.shape[0] - 1)
            swa_slots = records = valid_window = logical = None
        else:
            if (
                stage_cache is not None
                and rows_per_page is not None
                and swa.seq_lens.numel() > 0
            ):
                use_3page_swa = (
                    not non_causal
                    and tiled
                    and swa_table.shape[1] > 0
                    and self.window_size <= 2 * rows_per_page
                )
                swa_key = (
                    "swa_token_addr",
                    swa_table.shape,
                    rows_per_page,
                    swa_cache.page_stride if has_fast_swa_offset else None,
                    self.window_size,
                    self._ced_decoder,
                    use_3page_swa,
                    non_causal,
                    swa_width,
                    tiled,
                )
                cached_swa = stage_cache.get(swa_key)
                if cached_swa is None:
                    token_rows = torch.arange(
                        positions.shape[0], device=positions.device
                    )
                    valid_token = token_rows < swa.query_start_loc[-1]
                    requests = _token_requests(swa, positions).clamp(
                        max=swa_table.shape[0] - 1
                    )
                    pos_col = (
                        (positions // rows_per_page)
                        .clamp(max=swa_table.shape[1] - 1)
                        .long()
                    )
                    pos_rem = positions % rows_per_page
                    if non_causal:
                        req_idx = requests.clamp(max=swa.seq_lens.shape[0] - 1)
                        end_pos = table_lookup.take(swa.seq_lens, req_idx)
                        query_len = table_lookup.take(
                            swa.query_start_loc, req_idx + 1
                        ) - table_lookup.take(swa.query_start_loc, req_idx)
                        prefix_len = (end_pos - query_len).clamp(min=0)
                        start_pos = (prefix_len - self.window_size).clamp(min=0)
                        logical = (
                            start_pos[:, None]
                            + torch.arange(swa_width, device=positions.device)[None, :]
                        )
                        base_valid_window = (
                            (logical >= 0)
                            & (logical < end_pos[:, None])
                            & valid_token[:, None]
                        )
                    else:
                        logical = (
                            positions[:, None]
                            - self.window_size
                            + 1
                            + torch.arange(self.window_size, device=positions.device)[
                                None, :
                            ]
                        )
                        base_valid_window = (logical >= 0) & valid_token[:, None]
                    if self._ced_decoder:
                        replay_starts = get_forward_context().additional_kwargs[
                            "deepseek_v41_replay_starts"
                        ]
                        base_valid_window = base_valid_window & (
                            logical
                            >= table_lookup.take(replay_starts, requests)[:, None]
                        )
                    if use_3page_swa:
                        safe = logical.clamp(
                            min=0, max=swa_table.shape[1] * rows_per_page - 1
                        )
                        swa_pages = (safe // rows_per_page).long()
                        swa_page_rem = safe % rows_per_page
                        c2 = swa_pages[:, -1]
                        c1 = (c2 - 1).clamp(min=0)
                        c0 = (c2 - 2).clamp(min=0)
                        is_c2 = swa_pages == c2[:, None]
                        is_c1 = swa_pages == c1[:, None]
                        swa_3page = (c0, c1, is_c2, is_c1, swa_page_rem)
                    else:
                        swa_3page = None
                    cached_swa = (
                        valid_token,
                        requests,
                        pos_col,
                        pos_rem,
                        logical,
                        base_valid_window,
                        swa_3page,
                    )
                    stage_cache[swa_key] = cached_swa
                else:
                    (
                        valid_token,
                        requests,
                        pos_col,
                        pos_rem,
                        logical,
                        base_valid_window,
                        swa_3page,
                    ) = cached_swa
                swa_slots_key = ("swa_token_slots", swa_key)
                cached_swa_slots = stage_cache.get(swa_slots_key)
                if (
                    cached_swa_slots is None
                    or cached_swa_slots[0] is not swa.block_tables
                ):
                    b2 = swa_table[requests, pos_col].long()
                    swa_slots = (b2 * rows_per_page + pos_rem).masked_fill(
                        ~valid_token, -1
                    )
                    swa_valid = swa_slots >= 0
                    swa_invalid_3d = (~swa_valid)[:, None, None]
                    valid_window = base_valid_window & swa_valid[:, None]
                    if swa_3page is not None:
                        c0, c1, is_c2, is_c1, swa_page_rem = swa_3page
                        b1 = swa_table[requests, c1].long()
                        b0 = swa_table[requests, c0].long()
                        swa_blocks = torch.where(
                            is_c2,
                            b2[:, None],
                            torch.where(is_c1, b1[:, None], b0[:, None]),
                        )
                        swa_row_slots = (
                            swa_blocks * swa_cache.page_stride + swa_page_rem
                            if has_fast_swa_offset
                            else swa_blocks * rows_per_page + swa_page_rem
                        )
                    else:
                        raw_row_slots = (
                            _row_slots(swa_cache, swa_table, requests, logical)
                            if tiled
                            else None
                        )
                        swa_row_slots = (
                            raw_row_slots // rows_per_page * swa_cache.page_stride
                            + raw_row_slots % rows_per_page
                            if raw_row_slots is not None and has_fast_swa_offset
                            else raw_row_slots
                        )
                    stage_cache[swa_slots_key] = (
                        swa.block_tables,
                        swa_slots,
                        swa_valid,
                        swa_invalid_3d,
                        valid_window,
                        swa_row_slots,
                    )
                else:
                    (
                        _,
                        swa_slots,
                        swa_valid,
                        swa_invalid_3d,
                        valid_window,
                        swa_row_slots,
                    ) = cached_swa_slots
            else:
                swa_slots = _cache_slots(swa_cache, swa, positions)
                swa_valid = swa_slots >= 0
                swa_invalid_3d = (~swa_valid)[:, None, None]
                requests = _token_requests(swa, positions).clamp(
                    max=swa_table.shape[0] - 1
                )
                if non_causal and swa.seq_lens.numel() > 0:
                    req_idx = requests.clamp(max=swa.seq_lens.shape[0] - 1)
                    end_pos = table_lookup.take(swa.seq_lens, req_idx)
                    query_len = table_lookup.take(
                        swa.query_start_loc, req_idx + 1
                    ) - table_lookup.take(swa.query_start_loc, req_idx)
                    prefix_len = (end_pos - query_len).clamp(min=0)
                    start_pos = (prefix_len - self.window_size).clamp(min=0)
                    logical = (
                        start_pos[:, None]
                        + torch.arange(swa_width, device=positions.device)[None, :]
                    )
                    valid_window = (
                        (logical >= 0)
                        & (logical < end_pos[:, None])
                        & swa_valid[:, None]
                    )
                else:
                    logical = (
                        positions[:, None]
                        - self.window_size
                        + 1
                        + torch.arange(self.window_size, device=positions.device)[
                            None, :
                        ]
                    )
                    valid_window = (logical >= 0) & swa_valid[:, None]
                if self._ced_decoder:
                    replay_starts = get_forward_context().additional_kwargs[
                        "deepseek_v41_replay_starts"
                    ]
                    valid_window = valid_window & (
                        logical >= table_lookup.take(replay_starts, requests)[:, None]
                    )
                raw_row_slots = (
                    _row_slots(swa_cache, swa_table, requests, logical)
                    if tiled
                    else None
                )
                swa_row_slots = (
                    raw_row_slots // rows_per_page * swa_cache.page_stride
                    + raw_row_slots % rows_per_page
                    if raw_row_slots is not None
                    and has_fast_swa_offset
                    and rows_per_page is not None
                    else raw_row_slots
                )
            if not self._ced_decoder and (starts := _window_starts()) is not None:
                # Apply before records are built, including the tiled/local-Q path.
                valid_window = valid_window & (
                    logical
                    >= table_lookup.take(
                        starts, requests.clamp(max=starts.shape[0] - 1)
                    )[:, None]
                )
            records = (
                (
                    swa_cache.row_offset + swa_row_slots
                    if has_fast_swa_offset and rows_per_page is not None
                    else swa_cache.row_indices(swa_row_slots)
                ).masked_fill(~valid_window, -1)
                if swa_row_slots is not None
                else None
            )
        if self._fused_preprocess:
            # Shares the step's gathered rows with the Q-A/KV kernel.
            q = preprocess_ops.qb_rope()(
                qr, self.wq_b.weight, self.wq_b.weight_scale, self._rope_rows(positions)
            )
        else:
            # V4.1 normalizes qr before projection; neither route renormalizes q.
            if (
                self._q_projection_rope_op is not None
                and qr.dtype == torch.bfloat16
                and positions.dtype == torch.int32
                and self._tpu_rope_cache.dtype == torch.float32
            ):
                q = self._q_projection_rope_op(
                    qr,
                    self.wq_b.weight,
                    self.wq_b.weight_scale,
                    positions,
                    self._tpu_rope_cache,
                )
            else:
                # Other weight/rotary layouts retain the original model path.
                q = _value(self.wq_b(qr)).reshape(-1, self.n_local_heads, self.head_dim)
                q = self._rotate(q, positions)
        # Match upstream's plain BF16 cache selected by this backend. Packed
        # FP8/NVFP4 cache numerics must not be imposed on BF16 cache rows.
        if not self._fused_preprocess:
            kv = self._rotate(kv, positions)
        if self._csa2:
            assert csa2_write_records is not None
            if not swa_published:
                csa2_adapter.publish_swa(
                    swa_cache, csa2_write_records, kv.reshape(-1, 512)
                )
        else:
            _insert(swa_cache, swa_slots, kv, swa_valid)
        weights = None
        if self.compressor is not None and not self._ced_global_from_encoder:
            if self._fused_preprocess:
                self._write_csa2_global_kv(hidden_states, positions)
            else:
                self._write_global_kv(kv_score, positions)
        if self.indexer is not None and not self._all_indices_fit:
            weights = (
                index_weights.float()
                * self.indexer.head_dim**-0.5
                * self.indexer.n_head**-0.5
            )
        count = positions.shape[0]
        window = None if tiled else _read_rows(swa_cache, swa_table, requests, logical)
        rows = window
        valid_rows = valid_window
        main_cache = swa_cache
        main_records = positions.new_empty((count, 0))
        pages = starts = None
        if self.compress_ratio:
            comp_key = (
                "compressed_attn",
                self.compressed_cache_prefix,
                getattr(self, "index_source_layer_id", None),
                tiled,
                self._fp8_kv,
            )
            cached_comp = (
                stage_cache.get(comp_key)
                if stage_cache is not None and self.indexer is None
                else None
            )
            if cached_comp is not None and tiled:
                main_cache, pages, starts, main_records = cached_comp
            else:
                source = self._static_forward_context[self.compressed_cache_prefix]
                source_meta = metadata[self.compressed_cache_prefix]
                capacity = max(1, self._max_model_len // self.compress_ratio)
                lengths = ((positions + 1) // self.compress_ratio)[:, None]
                if self.indexer is not None:
                    if self._all_indices_fit:
                        logical_compressed = torch.arange(
                            capacity, device=positions.device
                        )[None, :].expand(count, -1)
                        selected = logical_compressed.masked_fill(
                            (logical_compressed >= lengths) | (~swa_valid[:, None]),
                            -1,
                        ).int()
                    else:
                        indexer = self.indexer
                        weights = cast(torch.Tensor, weights)
                        keys_meta = metadata[indexer.k_cache.prefix]
                        index_cache = indexer.k_cache.kv_cache
                        index_records = None
                        if isinstance(index_cache, PackedFP8IndexCacheRows):
                            index_records = _block_table(keys_meta)
                        elif tiled:
                            # Rows are row_indices of _row_slots(..., logical),
                            # computed per request: [requests, capacity]. The
                            # indexer shares each request's key tiles.
                            has_fast_idx_offset = isinstance(
                                getattr(index_cache, "row_offset", None), int
                            )
                            idx_req_key = (
                                "index_req_slots",
                                indexer.k_cache.prefix,
                                index_cache.shape[1] if has_fast_idx_offset else None,
                                (
                                    index_cache.page_stride
                                    if has_fast_idx_offset
                                    else None
                                ),
                                capacity,
                            )
                            cached_idx_req = (
                                stage_cache.get(idx_req_key)
                                if stage_cache is not None
                                else None
                            )
                            if cached_idx_req is None:
                                req_slots = _request_row_slots(
                                    index_cache, _block_table(keys_meta), capacity
                                )
                                cached_idx_req = (
                                    req_slots
                                    // index_cache.shape[1]
                                    * index_cache.page_stride
                                    + req_slots % index_cache.shape[1]
                                    if has_fast_idx_offset
                                    else req_slots
                                )
                                if stage_cache is not None:
                                    stage_cache[idx_req_key] = cached_idx_req
                            index_records = (
                                index_cache.row_offset + cached_idx_req
                                if has_fast_idx_offset
                                else index_cache.row_indices(cached_idx_req)
                            )
                        if tiled or isinstance(index_cache, PackedFP8IndexCacheRows):
                            assert index_records is not None
                            selected = self._select_paged_indices(
                                qr,
                                positions,
                                index_cache,
                                index_records,
                                requests,
                                weights,
                                lengths,
                                swa_valid,
                                query_start_loc=keys_meta.query_start_loc,
                            )
                        else:
                            index_q = self._project_index_q(qr, positions)
                            logical_compressed = torch.arange(
                                capacity, device=positions.device
                            )[None, :].expand(count, -1)
                            keys = _read_rows(
                                index_cache,
                                _block_table(keys_meta),
                                requests,
                                logical_compressed,
                            )
                            scores = (
                                torch.einsum(
                                    "thd,tcd->thc", index_q.float(), keys.float()
                                ).relu()
                                * weights[:, :, None]
                            ).sum(1)[None]
                            selected = self._select_indices(
                                scores, lengths, logical_compressed, count
                            )
                    if stage_cache is None or not tiled:
                        # Keep the persistent buffer's physical shape across compile
                        # buckets when not sharing stage-scoped main_records directly.
                        self.topk_indices_buffer.copy_(
                            torch.nn.functional.pad(
                                selected,
                                (
                                    0,
                                    self.topk_indices_buffer.shape[-1]
                                    - selected.shape[-1],
                                    0,
                                    self.topk_indices_buffer.shape[0] - count,
                                ),
                                value=-1,
                            )
                        )
                        selected = self.topk_indices_buffer[
                            :count,
                            : min(capacity, self.topk_indices_buffer.shape[-1]),
                        ]
                    else:
                        target_width = min(capacity, self.topk_indices_buffer.shape[-1])
                        if selected.shape[-1] < target_width:
                            selected = F.pad(
                                selected,
                                (0, target_width - selected.shape[-1]),
                                value=-1,
                            )
                        else:
                            selected = selected[:, :target_width]
                else:
                    selected = self.topk_indices_buffer[
                        :count, : min(capacity, self.topk_indices_buffer.shape[-1])
                    ]
                main_cache = source.kv_cache
                valid_compressed = (
                    (selected >= 0) & (selected < lengths) & swa_valid[:, None]
                )
                if tiled:
                    if self._fp8_kv:
                        pages_key = (
                            "compressed_pages",
                            self.compressed_cache_prefix,
                        )
                        pages = (
                            stage_cache.get(pages_key)
                            if stage_cache is not None
                            else None
                        )
                        if pages is None:
                            pages = main_cache.pages(_block_table(source_meta))
                            if stage_cache is not None and pages is not None:
                                stage_cache[pages_key] = pages
                    else:
                        pages = None
                    if pages is not None:
                        # SparseMLA resolves pages itself (Pallas), avoiding a
                        # [tokens, topk] XLA gather of the block table here.
                        main_records = selected.masked_fill(~valid_compressed, -1)
                        starts = source_meta.query_start_loc
                    else:
                        slots = _row_slots(
                            main_cache,
                            _block_table(source_meta),
                            requests,
                            selected.long(),
                        )
                        main_records = main_cache.row_indices(slots).masked_fill(
                            ~valid_compressed, -1
                        )
                    if stage_cache is not None:
                        stage_cache[comp_key] = (
                            main_cache,
                            pages,
                            starts,
                            main_records,
                        )
                else:
                    compressed_rows = _read_rows(
                        source.kv_cache,
                        _block_table(source_meta),
                        requests,
                        selected.long(),
                    )
                    rows = torch.cat((window, compressed_rows), 1)
                    valid_rows = torch.cat((valid_window, valid_compressed), 1)
        if tiled:
            if self._csa2:
                assert q is not None and csa2_metadata is not None
                assert csa2_bounds is not None
                result = csa2_adapter.attention(
                    q,
                    swa_cache,
                    main_cache,
                    main_records,
                    pages,
                    starts,
                    swa.query_start_loc,
                    positions,
                    self.attn_sink,
                    self._tpu_rope_cache,
                    csa2_metadata,
                    csa2_bounds,
                    self.compress_ratio,
                    self.scale,
                    non_causal,
                )
                output.copy_(result)
                return None
            assert records is not None and q is not None
            # The op zeroes the invalid rows itself (in the shared DSpark
            # draft kernel's write-back).
            result = paged_attention(
                q,
                swa_cache,
                main_cache,
                records,
                main_records,
                self.attn_sink,
                self.scale,
                self._swa_query_group_size,
                row_valid=swa_valid,
            )
            output.copy_(result)
            return None
        if self._fp8_kv and self.compress_ratio:
            raise NotImplementedError("FP8 V4.1 KV cache requires the tiled path")
        assert q is not None
        assert valid_rows is not None
        rows = cast(torch.Tensor, rows)
        q = cast(torch.Tensor, q)
        indices = (
            torch.arange(rows.shape[1], device=rows.device)[None, :]
            .expand(count, -1)
            .masked_fill(~valid_rows, -1)
        )
        # Treat every query as its own batch: each has a fixed-width paged gather.
        result = sparse_attention(
            q[:, None], rows, self.attn_sink, indices[:, None], self.scale
        )[:, 0]
        output.copy_(result.masked_fill(swa_invalid_3d, 0))
        return None

    def _select_paged_indices(
        self,
        qr,
        positions,
        index_cache,
        records,
        requests,
        weights,
        lengths,
        swa_valid,
        query_start_loc=None,
    ):
        """Select logical indices from the already-published Index-K cache.

        ``query_start_loc``: request row boundaries of ``qr``, which spare the
        FP8 indexer from deriving them from ``requests``.
        """
        indexer = self.indexer
        count = qr.shape[0]
        packed_fp8 = isinstance(index_cache, PackedFP8IndexCacheRows)
        has_candidates = indexer.indexer_op.candidate_blocks is not None
        candidate_write = has_candidates and getattr(
            indexer.indexer_op, "candidate_write", False
        )
        candidates = (
            self.candidate_block_buffer[:count]
            if has_candidates
            else positions.new_empty((count, 0), dtype=torch.int32)
        )
        # Scoring and selection are independent per query, but every TP rank
        # holds the whole batch: split the queries over ranks and gather the
        # selections. Each rank already holds the updated replicated K cache.
        # Contiguous parts of `_query_split` rows, computed rather than branched
        # on the token count, which would freeze the traced bucket's choice
        # (compilation/shape_variants.py). A static pad (never empty: a
        # zero-size pad guards on the count since torch.cat skips empty
        # inputs) covers the last part; padding rows have zero length and
        # select nothing.
        # A 16-row-aligned part (SparseCore top-k row tile) put a whole decode
        # batch of ``count <= 16`` on rank 0 while the other ranks scored only
        # padding (split change alone at C=16: L20 3.20 -> 1.48 ms,
        # L24 0.75 -> 0.31 ms).
        ranks = get_tensor_model_parallel_world_size()
        split = ranks > 1
        stage_cache = _active_stage_attn_cache
        query_split_key = (
            "indexer_query_split",
            getattr(self, "compress_ratio", None),
            ranks,
        )
        cached_query_split = (
            stage_cache.get(query_split_key) if stage_cache is not None else None
        )
        if cached_query_split is None:
            # Padding queries neither select nor publish candidates.
            active_lengths = lengths.masked_fill(~swa_valid[:, None], 0)
            batch_max = active_lengths.max()
            if split:
                part, padding = _query_split(count, ranks)
                mine = self._tp_rank * part + torch.arange(
                    part, dtype=torch.int32, device=qr.device
                )
                positions_part, lengths_part, requests_part = (
                    torch.cat(
                        (x, x.new_full((padding, *x.shape[1:]), fill))
                    ).index_select(0, mine)
                    for x, fill in (
                        (positions, 0),
                        (active_lengths, 0),
                        # FP8 candidate grouping must exclude padding requests.
                        (requests, -1 if packed_fp8 else 0),
                    )
                )
            else:
                padding = 0
                mine = None
                positions_part = positions
                lengths_part = active_lengths
                requests_part = requests
            cached_query_split = (
                active_lengths,
                batch_max,
                padding,
                mine,
                positions_part,
                lengths_part,
                requests_part,
            )
            if stage_cache is not None:
                stage_cache[query_split_key] = cached_query_split
        else:
            (
                active_lengths,
                batch_max,
                padding,
                mine,
                positions_part,
                lengths_part,
                requests_part,
            ) = cached_query_split
        offsets_part = None
        if packed_fp8 and query_start_loc is not None:
            # This rank's rows [rank * part, rank * part + part) of each request.
            offsets_part = query_start_loc.int()
            if split:
                part = positions_part.shape[0]
                offsets_part = (offsets_part - self._tp_rank * part).clamp(0, part)
        if split:
            qr_part, weights_part, candidates_part = (
                torch.cat((x, x.new_full((padding, *x.shape[1:]), fill))).index_select(
                    0, mine
                )
                for x, fill in ((qr, 0), (weights, 0), (candidates, -1))
            )
        else:
            qr_part, weights_part, candidates_part = qr, weights, candidates

        if packed_fp8 and qr_part.device.type != "cpu":
            projected = _value(indexer.wq_b(qr_part)).reshape(
                -1, indexer.n_head, indexer.head_dim
            )
            q_part, weights_part = quantize_query(
                projected, positions_part, self._tpu_rope_cache, weights_part
            )
        elif packed_fp8:
            q_part, q_scale = quantize_fp8_activation(
                self._project_index_q_unquantized(qr_part, positions_part),
                indexer.head_dim,
            )
            weights_part = weights_part.float() * q_scale.squeeze(-1)
        else:
            q_part = self._project_index_q(qr_part, positions_part)

        selected, published = indexer.indexer_op(
            q_part,
            weights_part,
            index_cache,
            records,
            lengths_part,
            requests_part,
            candidates=candidates_part,
            batch_max=batch_max,
            page_records=_index_page_records(index_cache),
            unordered=self._fp8_kv,
            score_width=max(1, self._max_model_len // self.compress_ratio),
            stage_cache=stage_cache if not split else None,
            query_start_loc=offsets_part,
        )
        if split:
            selected = _gather_query_rows(selected)[:count]
            if candidate_write:
                published = _gather_query_rows(published)[:count]
        if candidate_write:
            if stage_cache is not None:
                version_key = ("candidate_publication", self.candidate_source_layer)
                stage_cache[version_key] = stage_cache.get(version_key, 0) + 1
            self.candidate_block_buffer.copy_(
                torch.nn.functional.pad(
                    published,
                    (0, 0, 0, self.candidate_block_buffer.shape[0] - count),
                    value=-1,
                )
            )
        return selected

    def _select_indices(self, scores, lengths, logical_compressed, count):
        indexer = self.indexer
        candidates = None
        # If every block fits, publication and filtering are identities. The
        # Flash config has 2048 candidate slots but only 512 blocks at 4K.
        capacity = logical_compressed.shape[-1]
        if (
            indexer.indexer_op.candidate_blocks is not None
            and self.candidate_block_buffer.shape[-1] * self.candidate_block_size
            < capacity
        ):
            if getattr(indexer.indexer_op, "candidate_write", False):
                published = select_candidate_blocks(
                    scores,
                    lengths,
                    self._config["candidate_topk_blocks"],
                    self.candidate_block_size,
                )
                blocks = published[..., :: self.candidate_block_size]
                ids = (
                    torch.arange(blocks.shape[-1], device=blocks.device)
                    .expand_as(blocks)
                    .masked_fill(~blocks, -1)
                )
                selected = ids.topk(
                    min(ids.shape[-1], self.candidate_block_buffer.shape[-1]), dim=-1
                ).values
                self.candidate_block_buffer.copy_(
                    torch.nn.functional.pad(
                        selected[0].int(),
                        (
                            0,
                            self.candidate_block_buffer.shape[-1] - selected.shape[-1],
                            0,
                            self.candidate_block_buffer.shape[0] - count,
                        ),
                        value=-1,
                    )
                )
            else:
                ids = self.candidate_block_buffer[:count]
                # Membership is a 2-D block table, not [tokens, context, topk].
                # The extra column absorbs all invalid/padded candidate ids.
                num_blocks = (
                    capacity + self.candidate_block_size - 1
                ) // self.candidate_block_size
                valid_ids = (ids >= 0) & (ids < num_blocks)
                block_mask = torch.zeros(
                    (count, num_blocks + 1), dtype=torch.bool, device=ids.device
                )
                block_mask.scatter_(
                    1, ids.masked_fill(~valid_ids, num_blocks).long(), True
                )
                candidates = block_mask.gather(
                    1, logical_compressed // self.candidate_block_size
                )[None]
        if candidates is not None:
            candidates = candidates | (lengths.max() <= self._config["index_topk"])
        return select_compressed_indices(
            scores, lengths, self._config["index_topk"], candidates=candidates
        )[0]

    def forward_mqa(self, q, kv, positions, output):
        raise RuntimeError("TPU sparse attention is dispatched by _prepare_tpu")

    @property
    def o_proj_op(self):
        from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
            VllmDeepseekV4MLAAttention,
        )

        getter = cast(Callable, VllmDeepseekV4MLAAttention.o_proj_op.fget)
        return getter(self)

    def _assert_o_proj_supported(self):
        from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
            VllmDeepseekV4MLAAttention,
        )

        VllmDeepseekV4MLAAttention._assert_o_proj_supported(self)

    def process_weights_after_loading(self, act_dtype):
        from vllm_torchtpu.models.vllm.deepseek_v4.attention import (
            VllmDeepseekV4MLAAttention,
        )

        VllmDeepseekV4MLAAttention.process_weights_after_loading(self, act_dtype)
        self._assert_o_proj_supported()
        tp_size = get_tensor_model_parallel_world_size()
        self._q_projection_rope_op = None
        if self._fused_preprocess:
            self._flip_compressor_weight()
            self._check_fused_weights()
            # Register before TorchDynamo captures the first model forward.
            eps = self.kv_norm.variance_epsilon
            preprocess_ops.rope_rows()
            preprocess_ops.rope_paired_rows_sc()
            preprocess_ops.qa_rope(True, eps)
            preprocess_ops.qa_rope(False, eps)  # DSpark context KV
            preprocess_ops.qb_rope()
            if self.compressor is not None:
                ratio = self.compress_ratio
                if ratio == 2:
                    self._packed_compressor_op = preprocess_ops.compress_rope_packed(
                        self.compressor.rms_norm_eps
                    )
                    self._address_compressor_op = (
                        preprocess_ops.compress_rope_addresses(
                            self.compressor.rms_norm_eps
                        )
                    )
                for banks in (0,) if ratio == 1 else (1, 2):
                    preprocess_ops.compress_rope(
                        ratio, banks, self.compressor.rms_norm_eps
                    )
        elif (
            tp_size == 1
            and self.wo_a.weight.device.type == "tpu"
            and self.wq_b.weight.shape == (1280, 32768)
            and self.wq_b.weight.dtype == torch.float8_e4m3fn
            and self.wq_b.weight_scale.shape == (32768,)
            and self.wq_b.weight_scale.dtype == torch.float32
            and self.n_local_heads == 64
            and self.head_dim == 512
            and self._config["qk_rope_head_dim"] == 64
        ):
            self._q_projection_rope_op = initialize_projection_rope()
        if self.wo_a.weight.device.type == "tpu":
            _ = self.o_proj_op
        # wo_b joins wo_a in one call when it is a plain TP1 per-channel FP8
        # linear on the canonical (k, n) layout, which is what wo_b's own
        # quantized_matmul would consume.
        wo_b_scale = getattr(self.wo_b, "weight_scale", None)
        self._fused_wo_b = (
            self._csa2
            and tp_size == 1
            and getattr(self.wo_b, "bias", None) is None
            and getattr(self.wo_b, WEIGHT_FLIPPED_ATTR, False)
            and self.wo_b.weight.dtype == torch.float8_e4m3fn
            and self.wo_b.weight.shape
            == (self.n_groups * self.o_lora_rank, self.hidden_size)
            and wo_b_scale is not None
            and wo_b_scale.dtype == torch.float32
            and wo_b_scale.shape == (self.hidden_size,)
        )

    def _flip_compressor_weight(self):
        """Store the compressor projection as (k, n) for the fused kernel.

        It is built with ``quant_config=None``, so no TPU linear method flips
        it; ``DeepseekCompressor.project`` accepts either layout."""
        layer = getattr(self.compressor, "fused_wkv_wgate", None)
        if layer is None or getattr(layer, WEIGHT_FLIPPED_ATTR, False):
            return
        replace_parameter(layer, "weight", layer.weight.data.t().contiguous())
        setattr(layer, WEIGHT_FLIPPED_ATTR, True)
        if layer.weight.device.type == "tpu":
            synchronize_tensors([layer.weight])

    def _check_fused_weights(self):
        """The fused kernels' weight layouts; anything else is a load error."""

        def per_channel_fp8(layer, k, n):
            scale = getattr(layer, "weight_scale", None)
            return (
                getattr(layer, WEIGHT_FLIPPED_ATTR, False)
                and layer.weight.dtype == torch.float8_e4m3fn
                and layer.weight.shape == (k, n)
                and scale is not None
                and scale.dtype == torch.float32
                and scale.shape == (n,)
            )

        rotary_dim = self._config["qk_rope_head_dim"]
        problems = []
        if not per_channel_fp8(
            self.fused_wqa_wkv, self.hidden_size, self.q_lora_rank + self.head_dim
        ):
            problems.append("fused_wqa_wkv must be per-channel FP8 [hidden, rank+512]")
        if not per_channel_fp8(
            self.wq_b, self.q_lora_rank, self.n_local_heads * self.head_dim
        ):
            problems.append("wq_b must be per-channel FP8 [rank, heads*512]")
        if (
            self.q_norm.weight.dtype != self.kv_norm.weight.dtype
            or self.q_norm.weight.dtype not in (torch.bfloat16, torch.float32)
            or self.q_norm.variance_epsilon != self.kv_norm.variance_epsilon
        ):
            problems.append("q_norm/kv_norm must share a BF16/FP32 dtype and epsilon")
        if (
            self.q_lora_rank % 128
            or self._tpu_rope_cache.dtype != torch.float32
            or self._tpu_rope_cache.shape[1] != rotary_dim
            or not 0 < rotary_dim <= 128
            or rotary_dim % 2
        ):
            problems.append("needs a 128-lane q_lora_rank and FP32 rotary table")
        if self.compressor is not None:
            layer = self.compressor.fused_wkv_wgate
            norm = self.compressor.norm.weight
            if not (
                getattr(layer, WEIGHT_FLIPPED_ATTR, False)
                and layer.weight.dtype == torch.bfloat16
                and layer.weight.shape == (self.hidden_size, self.compress_ratio * 512)
                and norm.dtype in (torch.bfloat16, torch.float32)
                and norm.numel() == 512
            ):
                problems.append(
                    "compressor must be BF16 [hidden, ratio*512] with a 512 norm"
                )
        if problems:
            raise NotImplementedError(
                f"{self.prefix}: fused DeepSeek-V4.1 attention preprocessing "
                "(TP1, FP8 KV) " + "; ".join(problems)
            )

    def _o_proj(self, output, positions):
        assert output.dtype == torch.bfloat16
        assert output.shape[1:] == (self.n_local_heads, self.head_dim)
        if self._csa2:
            if self._fused_wo_b:
                return csa2_adapter.project_out(
                    output,
                    self.wo_a.weight,
                    self.wo_a.weight_scale,
                    self.wo_b.weight,
                    self.wo_b.weight_scale,
                )
            projected = csa2_adapter.project(
                output, self.wo_a.weight, self.wo_a.weight_scale
            )
            return _value(self.wo_b(projected))
        # Keep V4.1's bounded rotary table while using V4's FP8 projection kernel.
        projected = self.o_proj_op(
            output,
            positions,
            self._tpu_rope_cache,
            self.wo_a.weight,
            self.wo_a.weight_scale,
        )
        return _value(self.wo_b(projected))
