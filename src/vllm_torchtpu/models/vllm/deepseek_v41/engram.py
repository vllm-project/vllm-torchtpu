"""Upstream Engram hashing, request history and TPU embedding lookup."""

import weakref
from types import SimpleNamespace

import torch
from vllm.models.deepseek_v41.common import engram as upstream

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_cache import (
    write_cache,
)
from vllm_torchtpu.layers.adapter.upstream_bindings import (
    bind_device_ops,
    device_hooks,
)
from vllm_torchtpu.model_loader_patches import register_row_slice


class NgramHashState(upstream.NgramHashState):
    """Keep upstream tokenizer/layout/cache lifetime; replace Triton hashing."""

    def ensure_cache(self):
        if torch.compiler.is_compiling():
            # The outer CausalLM initializes the cache before tracing.
            # Layer views inside the graph are temporary Python tensor objects;
            # comparing their weakrefs must not reset persistent token history.
            return (
                self._cache is not None
                if self.use_slot_cache
                else self.swa_cache_module.kv_cache.numel() > 0
            )
        # The kernel property derives a view of the canonical pool. Lifetime
        # tracking must use the stable allocator object, not that temporary view.
        kv_cache = self.swa_cache_module.bound_cache
        if not kv_cache.numel():
            self._cache = None
            self._kv_cache_ref = None
            return False
        if not self.use_slot_cache:
            return True
        # Hybrid allocation can grow the physical SWA page beyond the
        # block size copied at model construction. Match the bound view
        # used by slot_mapping and block_table before upstream sizes its
        # generated-token history cache.
        block_size = self.swa_cache_module.block_size
        if self.block_size != block_size:
            self.block_size = block_size
            self._kv_cache_ref = None
        if self._kv_cache_ref is not None and self._kv_cache_ref() is kv_cache:
            return True
        self._cache = torch.zeros(
            kv_cache.shape[0] * self.block_size,
            dtype=torch.int32,
            device=kv_cache.device,
        )
        self._kv_cache_ref = weakref.ref(kv_cache)
        return True

    def forward(
        self,
        input_ids,
        positions,
        query_start_loc,
        dead_mask,
        lookback_token_ids,
        lookback_dead_mask,
        slot_mapping,
        block_table,
    ):
        count = input_ids.numel()
        layers, ngram = self.multipliers.shape
        if count == 0:
            return input_ids.new_empty(
                (0, layers, (ngram - 1) * self.primes.shape[-1]), dtype=torch.int32
            )
        if slot_mapping is not None:
            slot_mapping = slot_mapping[:count]
        if self.use_slot_cache:
            assert self._cache is not None and slot_mapping is not None
            mapped = self.token_map[input_ids.long()].masked_fill(
                dead_mask, upstream.DEAD_ID
            )
            valid = slot_mapping >= 0
            write_cache(self._cache, slot_mapping, mapped, valid)
        rows = torch.arange(count, device=input_ids.device)
        reqs = torch.bucketize(rows, query_start_loc[1:], right=True).clamp(
            max=lookback_token_ids.shape[0] - 1
        )
        starts = query_start_loc[reqs].long().clamp(max=count - 1)
        chunk_positions = positions[starts]
        blocked = torch.zeros(count, dtype=torch.bool, device=input_ids.device)
        rolling = torch.zeros(
            (count, layers), dtype=torch.int64, device=input_ids.device
        )
        hashes = []
        for shift in range(ngram):
            past = positions - shift
            in_chunk = past >= chunk_positions
            indices = (rows - shift).clamp(min=0)
            batch = self.token_map[input_ids[indices].long()].masked_fill(
                dead_mask[indices], upstream.DEAD_ID
            )
            cols = chunk_positions - 1 - past
            in_window = ~in_chunk & (cols >= 0) & (cols < self.lookback_depth)
            cols = cols.clamp(min=0, max=self.lookback_depth - 1).long()
            window_ids = lookback_token_ids[reqs, cols]
            known = in_window & (window_ids >= 0)
            window = self.token_map[window_ids.clamp(min=0).long()].masked_fill(
                lookback_dead_mask[reqs, cols], upstream.DEAD_ID
            )
            if self.use_slot_cache:
                assert self._cache is not None
                logical = past.clamp(
                    min=0, max=block_table.shape[1] * self.block_size - 1
                ).long()
                blocks = block_table[reqs, logical // self.block_size].long()
                slots = (blocks * self.block_size + logical % self.block_size).clamp(
                    min=0, max=self._cache.numel() - 1
                )
                fallback = self._cache[slots]
            else:
                fallback = torch.full_like(window, self.pad_id)
            source = torch.where(
                in_chunk, batch, torch.where(known, window, fallback)
            ).long()
            blocked = blocked | (past < 0) | (source == upstream.DEAD_ID)
            source = source.masked_fill(blocked, self.pad_id)
            rolling = rolling ^ (source[:, None] * self.multipliers[:, shift])
            if shift:
                hashes.append(
                    rolling[:, :, None] % self.primes[:, shift - 1]
                    + self.offsets[
                        :,
                        (shift - 1) * self.primes.shape[-1] : shift
                        * self.primes.shape[-1],
                    ]
                )
        return torch.cat(hashes, -1).to(torch.int32)


class ParallelEngramEmbedding(upstream.ParallelEngramEmbedding):
    """Reuse upstream head sharding and checkpoint loaders; gather rows on TPU."""

    @staticmethod
    def _weight_loader(param, loaded_weight):
        loader = upstream.ParallelEngramEmbedding._weight_loader
        if loaded_weight.shape[0] == param.shape[0]:
            # Already this rank's rows: the sharded Run:AI stream fetches only
            # them (see Engram._create_embedding). A full table has as many
            # rows only with a single shard, where both loaders agree.
            loader = _load_local_rows
        if param.device.type != "tpu":
            return loader(param, loaded_weight)
        from vllm_torchtpu.model_loader_patches import ExpertParamStager

        # Reuse upstream's shard selection and scale-byte interpretation.
        # A host stand-in detaches the shard from the giant mapped table;
        # the existing TPU stager bounds uploads and synchronizes publication.
        stager = ExpertParamStager()
        loader(stager.host_param(param), loaded_weight)
        stager.flush(param)

    def __init__(self, *args, **kwargs):
        # Upstream asks for the SM count solely to size its Triton lookup grid.
        # The TPU lookup below does not use that launch property.
        with device_hooks(
            torch.cuda,
            get_device_properties=lambda _: SimpleNamespace(multi_processor_count=0),
        ):
            super().__init__(*args, **kwargs)

    def lookup(self, indices, out, background=False):
        heads = (
            torch.arange(self.part_n_hash_cols, device=indices.device) + self.head_start
        )
        selected = indices[:, heads.clamp(max=self.n_hash_cols - 1)].long()
        valid = (
            (heads < self.n_hash_cols)[None, :]
            & (selected >= self.vocab_start_idx)
            & (selected < self.vocab_end_idx)
        )
        rows = (selected - self.vocab_start_idx).clamp(
            min=0, max=self.part_num_embeddings - 1
        )
        weight, scales = self._storage()
        values = weight[rows].float()
        scale = (
            (scales[rows].float() - 127).exp2().repeat_interleave(self.block_size, -1)
        )
        out.copy_((values * scale).masked_fill(~valid[:, :, None], 0).to(out.dtype))


def _load_local_rows(param, loaded_weight):
    """Upstream's head-shard loader for a tensor holding only this rank's rows."""
    if loaded_weight.dtype == torch.float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)
    assert loaded_weight.shape == param.shape, (
        f"engram shard {tuple(loaded_weight.shape)} does not fit param "
        f"{tuple(param.shape)}"
    )
    param.data.copy_(loaded_weight)


def _gather_engram_rows(rows, dim):
    # Retain the existing 128-lane width alignment for narrow fixtures.
    # This does not fix the native Engram NaN: it also reproduces at width
    # 128, while an isolated 3-D BF16 gather passes. Released heads are 256.
    width = rows.shape[-1]
    padding = (-width) % 128
    if padding:
        rows = torch.nn.functional.pad(rows, (0, padding))
    result = upstream.tensor_model_parallel_all_gather(rows, dim=dim)
    return result[..., :width].contiguous() if padding else result


class Engram(upstream.Engram):
    """Inherit projections, staging and collective head gather from upstream."""

    embed = bind_device_ops(
        upstream.Engram.embed, tensor_model_parallel_all_gather=_gather_engram_rows
    )

    def _create_embedding(self, layout, layer_hash_index):
        embedding = ParallelEngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
        )
        # Each table is ~98 GB in one tensor, over the Run:AI streamer's
        # per-tensor memory limit, and a rank keeps only its heads' rows.
        layer_id = layout.layer_ids[layer_hash_index]
        for leaf in ("weight", "scale"):
            register_row_slice(
                f"layers.{layer_id}.engram.embed.{leaf}",
                embedding.vocab_start_idx,
                embedding.part_num_embeddings,
            )
        return embedding

    def forward(self, hidden_states, hash_ids, token_mask=None):
        if self.use_sequence_parallel:
            raise NotImplementedError(
                "TPU Engram sequence parallel staging is not enabled"
            )
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        keys = (
            kv[:, : self.hc_mult * self.dim]
            .unflatten(-1, (self.hc_mult, self.dim))
            .float()
        )
        values = kv[:, self.hc_mult * self.dim :].float()
        hidden = hidden_states.float()
        dot = (hidden * self.q_weight * self.k_weight * keys).sum(-1)
        dot *= torch.rsqrt(hidden.square().mean(-1) + self.eps)
        dot *= torch.rsqrt(keys.square().mean(-1) + self.eps) * self.dim**-0.5
        gate_input = dot.abs().clamp(min=self.clamp_value).sqrt()
        gate = torch.where(dot < 0, -gate_input, gate_input).sigmoid()
        if token_mask is not None:
            gate = gate * token_mask[:, None]
        return (hidden + gate[:, :, None] * values[:, None, :]).to(hidden_states.dtype)
