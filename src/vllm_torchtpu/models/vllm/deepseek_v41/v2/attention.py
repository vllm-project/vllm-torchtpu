"""TPU MLA and compressor adapters with upstream cache ownership."""

import dataclasses
from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v41 import attention as upstream
from vllm.models.deepseek_v41 import compressor as compressor_upstream
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

from .attention_kernels import paged_attention, paged_indexer, sparse_fp8_attention
from .cache import NativeCacheView, PagedCacheRows, write_cache
from .ops import device_hooks, rms_norm, rotary, round_fp8_activation


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


def _token_requests(metadata, positions):
    rows = torch.arange(positions.shape[0], device=positions.device)
    return torch.bucketize(rows, metadata.query_start_loc[1:], right=True)


def _cache_slots(cache, metadata, positions, *, compress_ratio=1, ring=False):
    """Derive this cache's addresses from the shared TPU metadata contract."""
    if metadata.seq_lens.numel() == 0:
        return torch.full_like(positions, -1)
    table = _block_table(metadata)
    requests = _token_requests(metadata, positions).clamp(max=table.shape[0] - 1)
    rows_per_page = cache.shape[1]
    logical = positions // compress_ratio
    columns = torch.zeros_like(logical) if ring else logical // rows_per_page
    blocks = table[requests, columns.clamp(max=table.shape[1] - 1).long()].long()
    slots = blocks * rows_per_page + logical % rows_per_page
    # Padded request boundaries repeat the last real token offset. Keep the
    # mask on device, including dummy runs and batches with no active tokens.
    rows = torch.arange(positions.shape[0], device=positions.device)
    return slots.masked_fill(rows >= metadata.query_start_loc[-1], -1)


class CompressorStateCache(NativeCacheView, compressor_upstream.CompressorStateCache):
    """Upstream CircularBufferSpec, allocation lifecycle and cache binding."""

    def get_attn_backend(self):
        return AttentionBackend


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
        starts = query_start_loc[requests]
        ends = query_start_loc[requests + 1]
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
            slots = _cache_slots(state_cache, state_metadata, positions, ring=True)
            query_start_loc = state_metadata.query_start_loc
            requests = _token_requests(state_metadata, positions)
        latent = torch.empty(
            (kv_score.shape[0], self.head_dim),
            dtype=torch.bfloat16,
            device=kv_score.device,
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
    dtype = vllm_config.attention_config.resolve_indexer_kv_dtype("fp8")
    if dtype != "fp8":
        raise ValueError(f"TPU V4.1 indexer requires fp8, got {dtype}")
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
        kwargs["block_size"] = kwargs["cache_config"].block_size
        # FP8 applies to the compressed main KV only; SWA rows stay BF16.
        kwargs["dtype"] = torch.bfloat16
        kwargs["backend_cls"] = AttentionBackend
        super().__init__(*args, **kwargs)


class IndexerCache(NativeCacheView, upstream.DeepseekV4IndexerCache):
    def __init__(self, *args, **kwargs):
        # The first TPU path stores rounded values in BF16; ownership and
        # compression ratio still come from upstream. Packed storage follows
        # when the corresponding TPU kernel is connected.
        kwargs["head_dim"] = (
            get_current_vllm_config().model_config.hf_config.index_head_dim
        )
        kwargs["dtype"] = torch.bfloat16
        super().__init__(*args, **kwargs)

    def get_attn_backend(self):
        return AttentionBackend


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
        # FP8 main KV uses DSV4's packed record (448 e4m3 + 64 e8m0 bytes)
        # plus a BF16 RoPE companion; see cache.encode_fp8_kv.
        self._fp8_kv = self.kv_cache_torch_dtype == torch.float8_e4m3fn
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
        self._max_model_len = vllm_config.model_config.max_model_len
        # Branch-free SWA patch (`_attend_window`) only while its static
        # capacity, 63 tokens per extra request, stays small. Measured
        # (tpu7x-2): max_num_seqs 1 +4.6%, 4 +1.6%, 8 -4.3% prefill.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self._swa_max_requests = max_num_seqs if max_num_seqs <= 4 else None
        object.__setattr__(self, "mla_attn", self)
        self._prepare_and_attn_fn = self._prepare_tpu

    def get_kv_cache_spec(self, vllm_config):
        spec = super().get_kv_cache_spec(vllm_config)
        if spec is None or not self._fp8_kv:
            return spec
        return dataclasses.replace(spec, dtype=torch.uint8, page_size_padded=None)

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

    def _split_qkv_and_norm(self, qr_kv):
        return (
            self.q_norm(qr_kv[:, : self.q_lora_rank]),
            None,
            self.kv_norm(qr_kv[:, self.q_lora_rank :]),
        )

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

    def _write_global_kv(self, kv_score, positions, *, defer_index_write=True):
        metadata = get_forward_context().attn_metadata
        latent = self.compressor(kv_score, positions)
        compressed_slots = _cache_slots(
            self.kv_cache,
            metadata[self.prefix],
            positions,
            compress_ratio=self.compress_ratio,
        )
        valid = (compressed_slots >= 0) & ((positions + 1) % self.compress_ratio == 0)
        group_positions = positions // self.compress_ratio * self.compress_ratio
        _insert(
            self.kv_cache,
            compressed_slots,
            self._rotate(latent, group_positions),
            valid,
        )
        indexer = self.indexer
        keys = indexer.k_norm(_value(indexer.wk(latent)))
        keys = self._quantize_indexer(self._rotate(keys, group_positions))
        index_cache = indexer.k_cache.kv_cache
        index_slots = _cache_slots(
            index_cache,
            metadata[indexer.k_cache.prefix],
            positions,
            compress_ratio=indexer.k_cache.compress_ratio,
        )
        if (
            defer_index_write
            and isinstance(index_cache, PagedCacheRows)
            and self.window_size % 64 == 0
            and not self._all_indices_fit
        ):
            records = index_cache.row_indices(index_slots.clamp(min=0))
            return records.masked_fill(~valid | (index_slots < 0), -1), keys
        _insert(index_cache, index_slots, keys, valid)
        return None

    def write_encoder_global_kv(self, hidden_states, positions):
        """Publish decoder global KV for all encoder tokens before replay."""
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict) or not self.kv_cache.numel():
            return
        self._write_global_kv(
            self.compressor.project(hidden_states), positions, defer_index_write=False
        )

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
        metadata = get_forward_context().attn_metadata
        output.zero_()
        if not isinstance(metadata, dict) or not self.swa_cache_layer.kv_cache.numel():
            return
        swa = metadata[self.swa_cache_layer.prefix]
        swa_cache = self.swa_cache_layer.kv_cache
        swa_table = _block_table(swa)
        swa_slots = _cache_slots(swa_cache, swa, positions)
        q = _value(self.wq_b(qr)).reshape(-1, self.n_local_heads, self.head_dim)
        # V4.1 normalizes the low-rank qr before wq_b, but does not normalize
        # the projected query heads. Upstream's fused insert passes
        # apply_q_norm=False; V4's qnorm_rope kernel changes attention scores.
        q = self._rotate(q, positions)
        # Match upstream's plain BF16 cache selected by this backend. Packed
        # FP8/NVFP4 cache numerics must not be imposed on BF16 cache rows.
        kv = self._rotate(kv, positions)
        _insert(self.swa_cache_layer.kv_cache, swa_slots, kv, swa_slots >= 0)
        index_q = weights = None
        pending_index_write = None
        if self.compressor is not None and not self._ced_global_from_encoder:
            pending_index_write = self._write_global_kv(kv_score, positions)
        if self.indexer is not None and not self._all_indices_fit:
            index_q = _value(self.indexer.wq_b(qr)).reshape(
                -1, self.indexer.n_head, self.indexer.head_dim
            )
            index_q = self._quantize_indexer(self._rotate(index_q, positions))
            weights = (
                index_weights.float()
                * self.indexer.head_dim**-0.5
                * self.indexer.n_head**-0.5
            )
        count = positions.shape[0]
        requests = _token_requests(swa, positions).clamp(max=swa_table.shape[0] - 1)
        logical = (
            positions[:, None]
            - self.window_size
            + 1
            + torch.arange(self.window_size, device=positions.device)[None, :]
        )
        tiled = isinstance(swa_cache, PagedCacheRows) and self.window_size % 64 == 0
        window = None if tiled else _read_rows(swa_cache, swa_table, requests, logical)
        valid_window = (logical >= 0) & (swa_slots[:, None] >= 0)
        if self._ced_decoder:
            replay_starts = get_forward_context().additional_kwargs[
                "deepseek_v41_replay_starts"
            ]
            valid_window &= logical >= replay_starts[requests, None]
        rows = window
        valid_rows = valid_window
        main_cache = swa_cache
        main_records = positions.new_empty((count, 0))
        pages = starts = None
        if self.compress_ratio:
            source = self._static_forward_context[self.compressed_cache_prefix]
            source_meta = metadata[self.compressed_cache_prefix]
            capacity = max(1, self._max_model_len // self.compress_ratio)
            logical_compressed = torch.arange(capacity, device=positions.device)
            logical_compressed = logical_compressed[None, :].expand(count, -1)
            lengths = ((positions + 1) // self.compress_ratio)[:, None]
            if self.indexer is not None:
                if self._all_indices_fit:
                    selected = logical_compressed.masked_fill(
                        logical_compressed >= lengths, -1
                    ).int()
                else:
                    indexer = self.indexer
                    index_q = cast(torch.Tensor, index_q)
                    weights = cast(torch.Tensor, weights)
                    keys_meta = metadata[indexer.k_cache.prefix]
                    index_cache = indexer.k_cache.kv_cache
                    if tiled:
                        # Rows are row_indices of _row_slots(..., logical_compressed),
                        # computed per request: [requests, capacity]. The indexer
                        # shares each request's key tiles across its tokens.
                        index_records = index_cache.row_indices(
                            _request_row_slots(
                                index_cache, _block_table(keys_meta), capacity
                            )
                        )
                        selected = self._select_paged_indices(
                            index_q,
                            index_cache,
                            index_records,
                            requests,
                            weights,
                            lengths,
                            swa_slots,
                            pending_index_write,
                        )
                    else:
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
                # Keep the persistent buffer's physical shape across compile
                # buckets. TorchTPU must not replace it with a shorter donated
                # slice when prefill switches to decode.
                self.topk_indices_buffer.copy_(
                    torch.nn.functional.pad(
                        selected,
                        (
                            0,
                            self.topk_indices_buffer.shape[-1] - selected.shape[-1],
                            0,
                            self.topk_indices_buffer.shape[0] - count,
                        ),
                        value=-1,
                    )
                )
            selected = self.topk_indices_buffer[
                :count, : min(capacity, self.topk_indices_buffer.shape[-1])
            ]
            main_cache = source.kv_cache
            valid_compressed = (
                (selected >= 0) & (selected < lengths) & (swa_slots[:, None] >= 0)
            )
            if tiled:
                pages = (
                    main_cache.pages(_block_table(source_meta))
                    if self._fp8_kv
                    else None
                )
                if pages is not None:
                    # SparseMLA resolves pages itself (Pallas), avoiding a
                    # [tokens, topk] XLA gather of the block table here.
                    main_records = selected.masked_fill(~valid_compressed, -1)
                    starts = source_meta.query_start_loc
                else:
                    slots = _row_slots(
                        main_cache, _block_table(source_meta), requests, selected.long()
                    )
                    main_records = main_cache.row_indices(slots).masked_fill(
                        ~valid_compressed, -1
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
            slots = _row_slots(swa_cache, swa_table, requests, logical)
            records = swa_cache.row_indices(slots).masked_fill(~valid_window, -1)
            if self._fp8_kv and self.compress_ratio:
                result = sparse_fp8_attention(
                    q,
                    swa_cache,
                    main_cache,
                    records,
                    main_records,
                    self.attn_sink,
                    self.scale,
                    pages,
                    starts,
                    self._swa_max_requests,
                )
            else:
                result = paged_attention(
                    q,
                    swa_cache,
                    main_cache,
                    records,
                    main_records,
                    self.attn_sink,
                    self.scale,
                )
            output.copy_(result.masked_fill((swa_slots < 0)[:, None, None], 0))
            return
        if self._fp8_kv and self.compress_ratio:
            raise NotImplementedError("FP8 V4.1 KV cache requires the tiled path")
        rows = cast(torch.Tensor, rows)
        indices = (
            torch.arange(rows.shape[1], device=rows.device)[None, :]
            .expand(count, -1)
            .masked_fill(~valid_rows, -1)
        )
        # Treat every query as its own batch: each has a fixed-width paged gather.
        result = sparse_attention(
            q[:, None], rows, self.attn_sink, indices[:, None], self.scale
        )[:, 0]
        output.copy_(result.masked_fill((swa_slots < 0)[:, None, None], 0))

    def _select_paged_indices(
        self,
        index_q,
        index_cache,
        records,
        requests,
        weights,
        lengths,
        swa_slots,
        pending_index_write,
    ):
        """Publish explicit index-cache and logical-selection outputs."""
        indexer = self.indexer
        count = index_q.shape[0]
        has_candidates = indexer.indexer_op.candidate_blocks is not None
        candidate_write = has_candidates and getattr(
            indexer.indexer_op, "candidate_write", False
        )
        candidates = (
            self.candidate_block_buffer[:count]
            if has_candidates
            else swa_slots.new_empty((count, 0), dtype=torch.int32)
        )
        # Padding queries neither select nor publish candidates.
        active_lengths = lengths.masked_fill(swa_slots[:, None] < 0, 0)
        write_records, write_keys = (
            pending_index_write
            if pending_index_write is not None
            else (
                swa_slots.new_empty((0,), dtype=torch.int32),
                index_q.new_empty((0, index_q.shape[-1])),
            )
        )
        # Scoring and selection are independent per query, but every TP rank
        # holds the whole batch: split the queries over ranks and gather the
        # selections. Every rank still writes all index keys (replicated cache).
        # Pad the queries to 16 rows per rank (SparseCore top-k row tiles)
        # rather than branching on the token count, which would freeze the
        # traced bucket's choice (compilation/shape_variants.py). Padding rows
        # have zero length and select nothing. Always pad at least one row: a
        # zero-size pad guards on the count (torch.cat skips empty inputs).
        ranks = get_tensor_model_parallel_world_size()
        split = ranks > 1
        queries = (index_q, weights, active_lengths, candidates, requests)
        if split:
            rows = 16 * ranks
            padding = (count // rows + 1) * rows - count
            part = (count + padding) // ranks
            mine = self._tp_rank * part + torch.arange(
                part, dtype=torch.int32, device=index_q.device
            )
            fills = (0, 0, 0, -1, 0)
            queries = tuple(
                torch.cat((x, x.new_full((padding, *x.shape[1:]), fill))).index_select(
                    0, mine
                )
                for x, fill in zip(queries, fills)
            )
        q_part, weights_part, lengths_part, candidates_part, requests_part = queries
        updated_cache, selected, published = paged_indexer(
            q_part,
            index_cache,
            records,
            weights_part,
            lengths_part,
            candidates_part,
            write_records,
            write_keys,
            requests_part,
            topk=self._config["index_topk"],
            block_size=max(1, self.candidate_block_size),
            candidate_write=candidate_write,
            candidate_filter=has_candidates and not candidate_write,
            # FP8 attention re-sorts the selection (compact_records).
            unordered=self._fp8_kv,
            batch_max=active_lengths.max(),
        )
        if split:
            selected = _gather_query_rows(selected)[:count]
            if candidate_write:
                published = _gather_query_rows(published)[:count]
        index_cache.raw.copy_(updated_cache)
        if candidate_write:
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
        # Register before TorchDynamo captures the first model forward.
        if self.wo_a.weight.device.type == "tpu":
            _ = self.o_proj_op

    def _o_proj(self, output, positions):
        assert output.dtype == torch.bfloat16
        assert output.shape[1:] == (self.n_local_heads, self.head_dim)
        # Keep V4.1's bounded rotary table while using V4's FP8 projection kernel.
        projected = self.o_proj_op(
            output,
            positions,
            self._tpu_rope_cache,
            self.wo_a.weight,
            self.wo_a.weight_scale,
        )
        return _value(self.wo_b(projected))
