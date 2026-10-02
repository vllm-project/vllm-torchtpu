"""TPU MLA and compressor adapters with upstream cache ownership."""

from typing import cast

import torch
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v41 import attention as upstream
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

from vllm_torchtpu.models.vllm.deepseek_v4.attention import DeepseekTPUAttentionMixin
from vllm_torchtpu.runner.kv_cache_dsv4 import deepseek_v41_cache_spec

from .cache import (
    NativeCacheView,
    PagedCacheRows,
    _block_table,
    _cache_slots,
    _row_slots,
    _token_requests,
    write_cache,
)
from .ced import ced_attention
from .compressor import AttentionBackend, DeepseekCompressor, write_compressed_cache
from .indexer import paged_indexer
from .ops import device_hooks, rotary, round_fp8_activation


def indexer_uses_fp4(vllm_config):
    """Resolve the upstream format without imposing a CUDA capability check."""
    dtype = vllm_config.attention_config.resolve_indexer_kv_dtype("fp8")
    if dtype != "fp8":
        raise ValueError(f"TPU V4.1 indexer requires fp8, got {dtype}")
    return False


class SWACache(NativeCacheView, DeepseekV4SWACache):
    def __init__(self, *args, **kwargs):
        kwargs["block_size"] = kwargs["cache_config"].block_size
        kwargs["backend_cls"] = AttentionBackend
        super().__init__(*args, **kwargs)

    @property
    def kv_cache(self):
        return self._bound_cache

    @kv_cache.setter
    def kv_cache(self, cache):
        self._bound_cache = cache

    @property
    def paged_cache(self):
        raw = getattr(self, "_tpu_raw_cache", None)
        return (
            PagedCacheRows(raw, self._tpu_cache_layout)
            if raw is not None
            else self._bound_cache
        )

    def bind_kv_cache(self, cache, *, raw_cache=None, layout=None):
        if layout is None:
            raise ValueError("V4.1 SWA binding requires the allocator's logical layout")
        self.kv_cache = cache
        self._tpu_raw_cache = raw_cache
        self._tpu_cache_layout = layout
        self.block_size = layout[0][1]


class IndexerCache(NativeCacheView, upstream.DeepseekV4IndexerCache):
    def __init__(self, *args, **kwargs):
        # Indexer keys retain FP8-rounded BF16 values for exact scoring;
        # compressed attention KV is packed separately.
        kwargs["head_dim"] = (
            get_current_vllm_config().model_config.hf_config.index_head_dim
        )
        kwargs["dtype"] = torch.bfloat16
        super().__init__(*args, **kwargs)

    def get_attn_backend(self):
        return AttentionBackend


def _value(output):
    return output[0] if isinstance(output, tuple) else output


class DeepseekV41Attention(
    DeepseekTPUAttentionMixin, NativeCacheView, upstream.DeepseekV4Attention
):
    backend_cls = AttentionBackend
    swa_backend_cls = AttentionBackend
    use_fp8_ds_mla_layout = False
    _ced_decoder = False
    _ced_global_from_encoder = False
    csa_kernel_batch_size = 1

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
        if self.kv_cache_torch_dtype != torch.bfloat16:
            raise NotImplementedError(
                "TPU V4.1 uses BF16 SWA and packed FP8 compressed KV; use "
                "--kv-cache-dtype auto or bfloat16"
            )
        if self.compress_ratio and (
            self.head_dim != 512
            or vllm_config.model_config.hf_config.qk_rope_head_dim != 64
        ):
            raise ValueError("Native CSA requires 448 NoPE and 64 RoPE dimensions")
        self.register_buffer(
            "_tpu_rope_cache", self.rotary_emb.cos_sin_cache, persistent=False
        )
        self._config = vllm_config.model_config.hf_config.to_dict()
        self._max_model_len = vllm_config.model_config.max_model_len
        self.custom_prefix = self.prefix
        object.__setattr__(self, "mla_attn", self)
        self._prepare_and_attn_fn = self._prepare_tpu

    @property
    def attention_kind(self):
        return "csa" if self.compress_ratio else "swa"

    @property
    def main_cache_layer(self):
        return (
            self._static_forward_context[self.compressed_cache_prefix]
            if self.compress_ratio
            else self
        )

    @property
    def kv_cache(self):
        return self._bound_cache

    @kv_cache.setter
    def kv_cache(self, cache):
        self._bound_cache = cache

    def bind_kv_cache(self, cache, *, raw_cache=None, layout=None):
        if not isinstance(cache, tuple) or len(cache) != 2:
            raise ValueError("V4.1 compressed KV requires a (NoPE, RoPE) pair")
        if layout is None:
            raise ValueError(
                "V4.1 compressed KV requires the allocator's logical layout"
            )
        self.kv_cache = cache
        self.cache_num_states = layout[0][1]

    def get_kv_cache_spec(self, vllm_config):
        return deepseek_v41_cache_spec(
            super().get_kv_cache_spec(vllm_config),
            vllm_config,
            compressed=True,
        )

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
            self.kv_cache[0],
            metadata[self.prefix],
            positions,
            compress_ratio=self.compress_ratio,
            rows_per_page=self.cache_num_states,
        )
        valid = (compressed_slots >= 0) & ((positions + 1) % self.compress_ratio == 0)
        group_positions = positions // self.compress_ratio * self.compress_ratio
        write_compressed_cache(
            self.kv_cache,
            compressed_slots,
            self._rotate(latent, group_positions),
            valid,
            self.cache_num_states,
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
        if defer_index_write and not self._all_indices_fit:
            records = index_cache.row_indices(index_slots.clamp(min=0))
            return records.masked_fill(~valid | (index_slots < 0), -1), keys
        write_cache(index_cache, index_slots, keys, valid)
        return None

    def write_encoder_global_kv(self, hidden_states, positions):
        """Publish decoder global KV for all encoder tokens before replay."""
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict) or not isinstance(self.kv_cache, tuple):
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
        swa_cache = self.swa_cache_layer.paged_cache
        swa_table = _block_table(swa)
        swa_slots = _cache_slots(swa_cache, swa, positions)
        q = _value(self.wq_b(qr)).reshape(-1, self.n_local_heads, self.head_dim)
        # V4.1 normalizes the low-rank qr before wq_b, but does not normalize
        # the projected query heads. Upstream's fused insert passes
        # apply_q_norm=False; V4's qnorm_rope kernel changes attention scores.
        q = self._rotate(q, positions)
        # SWA retains BF16 values; only compressed KV is packed as FP8.
        kv = self._rotate(kv, positions)
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
        selected = None
        if self.compress_ratio:
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
                    slots = _row_slots(
                        index_cache,
                        _block_table(keys_meta),
                        requests,
                        logical_compressed,
                    )
                    selected = self._select_paged_indices(
                        index_q,
                        index_cache,
                        index_cache.row_indices(slots),
                        weights,
                        lengths,
                        swa_slots,
                        pending_index_write,
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
            selected = selected.masked_fill(
                (selected < 0) | (selected >= lengths) | (swa_slots[:, None] < 0), -1
            )
        self.forward_mqa(q, kv, positions, output, topk_indices=selected)
        output.masked_fill_((swa_slots < 0)[:, None, None], 0)

    def _select_paged_indices(
        self,
        index_q,
        index_cache,
        records,
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
        updated_cache, selected, published = paged_indexer(
            index_q,
            index_cache,
            records,
            weights,
            active_lengths,
            candidates,
            write_records,
            write_keys,
            topk=self._config["index_topk"],
            block_size=max(1, self.candidate_block_size),
            candidate_write=candidate_write,
            candidate_filter=has_candidates and not candidate_write,
        )
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

    def forward_mqa(self, q, kv, positions, output, *, topk_indices=None):
        if not self._ced_decoder:
            output.copy_(
                self._run_mqa(q, kv, positions, output, topk_indices=topk_indices)
            )
            return
        # Decoder replay begins at the whole prompt's tail. Its first query
        # must not see older keys even when those pages remain in SWA storage.
        metadata = get_forward_context().attn_metadata
        swa = metadata[self.swa_cache_layer.prefix]
        cache = self.swa_cache_layer.paged_cache
        slots = _cache_slots(cache, swa, positions)
        write_cache(cache, slots, kv, slots >= 0)
        table = _block_table(swa)
        requests = _token_requests(swa, positions).clamp(max=table.shape[0] - 1)
        logical = (
            positions[:, None]
            - self.window_size
            + 1
            + torch.arange(self.window_size, device=positions.device)[None]
        )
        replay_starts = get_forward_context().additional_kwargs[
            "deepseek_v41_replay_starts"
        ]
        valid = (
            (logical >= 0)
            & (slots[:, None] >= 0)
            & (logical >= replay_starts[requests, None])
        )
        records = cache.row_indices(
            _row_slots(cache, table, requests, logical)
        ).masked_fill(~valid, -1)
        source = self.main_cache_layer
        if topk_indices is None:
            topk_indices = positions.new_empty(
                (positions.shape[0], 0), dtype=torch.int32
            )
            main = (self.swa_cache_layer.kv_cache,) * 2
            source_meta, num_states = swa, 1
        else:
            main = source.kv_cache
            source_meta, num_states = metadata[source.prefix], source.cache_num_states
        output.copy_(
            ced_attention(
                q,
                cache,
                main,
                records,
                topk_indices,
                source_meta,
                self.attn_sink,
                self.scale,
                num_states,
            )
        )

    def process_weights_after_loading(self, act_dtype):
        super().process_weights_after_loading(act_dtype)
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
