"""Upstream Engram hashing, request history and TPU embedding lookup.

Shared by the v1 and v2 DeepSeek V4.1 implementations. Each implementation
subclasses ``NgramHashState`` to bind its own slot-cache write op and, when
its SWA cache module differs, the cache lifetime in ``ensure_cache``.
"""

from types import SimpleNamespace

import torch
from vllm.models.deepseek_v41.common import engram as upstream

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.upstream_bindings import (
    bind_device_ops,
    device_hooks,
)
from vllm_torchtpu.model_loader_patches import register_row_slice


class NgramHashState(upstream.NgramHashState):
    """Keep upstream tokenizer/layout/cache lifetime; replace Triton hashing."""

    def write_cache(
        self,
        cache: torch.Tensor,
        slots: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Write mapped token ids into the slot cache; bound per implementation."""
        raise NotImplementedError

    def ensure_cache(self) -> bool:
        if torch.compiler.is_compiling():
            # The outer CausalLM initializes the cache before tracing.
            # Layer views inside the graph are temporary Python tensor objects;
            # comparing their weakrefs must not reset persistent token history.
            return (
                self._cache is not None
                if self.use_slot_cache
                else self.swa_cache_module.kv_cache.numel() > 0
            )
        kv_cache = self.swa_cache_module.kv_cache
        if self.use_slot_cache and kv_cache.numel():
            # Hybrid allocation can grow the physical SWA page beyond the
            # block size copied at model construction. Match the bound view
            # used by slot_mapping and block_table before upstream sizes its
            # generated-token history cache.
            block_size = kv_cache.shape[1]
            if self.block_size != block_size:
                self.block_size = block_size
                self._kv_cache_ref = None
        return super().ensure_cache()

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
            self.write_cache(self._cache, slot_mapping, mapped, valid)

        def fallback(past, reqs, window):
            if not self.use_slot_cache:
                return torch.full_like(window, self.pad_id)
            assert self._cache is not None
            logical = past.clamp(
                min=0, max=block_table.shape[1] * self.block_size - 1
            ).long()
            blocks = block_table[reqs, logical // self.block_size].long()
            slots = (blocks * self.block_size + logical % self.block_size).clamp(
                min=0, max=self._cache.numel() - 1
            )
            return self._cache[slots]

        tables = (self.token_map, self.multipliers, self.primes, self.offsets)
        return self._hash(
            tables,
            input_ids,
            positions,
            query_start_loc,
            dead_mask,
            lookback_token_ids,
            lookback_dead_mask,
            fallback,
        )

    def hash_on_host(
        self,
        input_ids,
        positions,
        query_start_loc,
        dead_mask,
        lookback_token_ids,
        lookback_dead_mask,
    ):
        """``forward`` on CPU tensors, for steps whose tokens are all prompt.

        Every n-gram source of a real prompt row is then either in the chunk
        or in ``lookback_token_ids`` (the runner fills it from the prompt), so
        the generated-token slot cache is never consulted and is not written:
        prompt history is always re-supplied by lookback, and decode steps
        (which do need the cache) keep the device path. Padded rows read
        ``pad_id`` here instead of an arbitrary cache slot; their outputs are
        never consumed. Integer math, so hashes equal the device path's.
        """
        count = input_ids.numel()
        layers, ngram = self.multipliers.shape
        if count == 0:
            return input_ids.new_empty(
                (0, layers, (ngram - 1) * self.primes.shape[-1]), dtype=torch.int32
            )
        tables = getattr(self, "_host_tables", None)
        if tables is None:
            tables = tuple(
                t.to("cpu")
                for t in (self.token_map, self.multipliers, self.primes, self.offsets)
            )
            self._host_tables = tables
        return self._hash(
            tables,
            input_ids,
            positions,
            query_start_loc,
            dead_mask,
            lookback_token_ids,
            lookback_dead_mask,
            lambda past, reqs, window: torch.full_like(window, self.pad_id),
        )

    def _hash(
        self,
        tables,
        input_ids,
        positions,
        query_start_loc,
        dead_mask,
        lookback_token_ids,
        lookback_dead_mask,
        fallback,
    ):
        token_map, multipliers, primes, offsets = tables
        count = input_ids.numel()
        layers, ngram = multipliers.shape
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
            batch = token_map[input_ids[indices].long()].masked_fill(
                dead_mask[indices], upstream.DEAD_ID
            )
            cols = chunk_positions - 1 - past
            in_window = ~in_chunk & (cols >= 0) & (cols < self.lookback_depth)
            cols = cols.clamp(min=0, max=self.lookback_depth - 1).long()
            window_ids = lookback_token_ids[reqs, cols]
            known = in_window & (window_ids >= 0)
            window = token_map[window_ids.clamp(min=0).long()].masked_fill(
                lookback_dead_mask[reqs, cols], upstream.DEAD_ID
            )
            source = torch.where(
                in_chunk,
                batch,
                torch.where(known, window, fallback(past, reqs, window)),
            ).long()
            blocked = blocked | (past < 0) | (source == upstream.DEAD_ID)
            source = source.masked_fill(blocked, self.pad_id)
            rolling = rolling ^ (source[:, None] * multipliers[:, shift])
            if shift:
                hashes.append(
                    rolling[:, :, None] % primes[:, shift - 1]
                    + offsets[
                        :,
                        (shift - 1) * primes.shape[-1] : shift * primes.shape[-1],
                    ]
                )
        return torch.cat(hashes, -1).to(torch.int32)


def _load_local_rows(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Upstream's head-shard loader for a tensor holding only this rank's rows."""
    if param.dtype == torch.uint8 or loaded_weight.dtype == torch.float8_e8m0fnu:
        loaded_weight = loaded_weight.view(torch.uint8)
    else:
        loaded_weight = loaded_weight.to(param.dtype)
    assert loaded_weight.shape == param.shape, (
        f"engram shard {tuple(loaded_weight.shape)} does not fit param "
        f"{tuple(param.shape)}"
    )
    param.data.copy_(loaded_weight)


class ParallelEngramEmbedding(upstream.ParallelEngramEmbedding):
    """Reuse upstream head sharding and checkpoint loaders; support Host DRAM."""

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

    def __init__(self, *args, cpu_offload: bool | None = None, **kwargs):
        self.cpu_offload = (
            bool(envs.TPU_DEEPSEEK_V41_ENGRAM_CPU_OFFLOAD)
            if cpu_offload is None
            else bool(cpu_offload)
        )
        # Upstream asks for the SM count solely to size its Triton lookup grid.
        # The TPU lookup below does not use that launch property.
        with device_hooks(
            torch.cuda,
            get_device_properties=lambda _: SimpleNamespace(multi_processor_count=0),
        ):
            super().__init__(*args, **kwargs)
        upstream.set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
        if self.weight_scale_inv is not None:
            upstream.set_weight_attrs(
                self.weight_scale_inv, {"dummy_weight_value": 127}
            )

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cpu_offload:
            return super()._allocate_weights()
        return (
            torch.empty(
                self.part_num_embeddings,
                self.dim,
                dtype=torch.float8_e4m3fn,
                device="cpu",
            ),
            torch.empty(
                self.part_num_embeddings,
                self.dim // self.block_size,
                dtype=torch.uint8,
                device="cpu",
            ),
        )

    def lookup(self, indices, out, background=False):
        weight, scales = self._storage()
        work_device = weight.device
        if indices.device != work_device:
            indices = indices.to(work_device)
        heads = (
            torch.arange(self.part_n_hash_cols, device=work_device) + self.head_start
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
        values = weight[rows].float()
        scale = (
            (scales[rows].float() - 127).exp2().repeat_interleave(self.block_size, -1)
        )
        out.copy_((values * scale).masked_fill(~valid[:, :, None], 0).to(out.dtype))

    def gather_raw(self, indices, raw_out, scale_out):
        """Gather local heads' raw FP8 bytes and E8M0 scale bytes (no dequant).

        ``raw_out`` is ``[T, local_heads, dim]`` uint8 and ``scale_out`` is
        ``[T, local_heads, dim // block_size]`` uint8, both on the table's
        device. Rows outside this shard (or padded heads) become FP8 zero with
        a unit scale, so ``dequant_fp8_rows`` yields exactly the zero rows that
        ``lookup`` writes for them.
        """
        weight, scales = self._storage()
        work_device = weight.device
        if indices.device != work_device:
            indices = indices.to(work_device)
        heads = (
            torch.arange(self.part_n_hash_cols, device=work_device) + self.head_start
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
        flat = rows.flatten()
        torch.index_select(
            weight.view(torch.uint8), 0, flat, out=raw_out.view(-1, self.dim)
        )
        torch.index_select(
            scales, 0, flat, out=scale_out.view(-1, self.dim // self.block_size)
        )
        invalid = ~valid[:, :, None]
        raw_out.masked_fill_(invalid, 0)
        scale_out.masked_fill_(invalid, _UNIT_SCALE)


# E8M0 scale byte for 2**0. Scale bytes encode 2**(byte - 127).
_UNIT_SCALE = 127


def dequant_fp8_rows(raw, scale, block_size):
    """``[..., dim]`` FP8 bytes x ``[..., dim // block_size]`` E8M0 bytes -> bf16.

    Matches ``ParallelEngramEmbedding.lookup``'s ``values.float() *
    exp2(scale - 127)`` rounded once to bf16. The power of two is built from
    its float32 exponent bits (``byte << 23``), which is exact for bytes
    1..254 and does not depend on the backend's ``exp2`` accuracy; the product
    of a float32 and a power of two is then exact, so only the final bf16
    rounding remains. Byte 0 (2**-127, a float32 subnormal) maps to 0.
    """
    values = raw.view(torch.float8_e4m3fn).float()
    power = (scale.to(torch.int32) << 23).view(torch.float32)
    power = power.repeat_interleave(block_size, -1)
    return (values * power).to(torch.bfloat16)


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

    def _init_staging(self, max_tokens: int, head_dim: int) -> None:
        super()._init_staging(max_tokens, head_dim)
        self._staged_raw: torch.Tensor | None = None
        if getattr(self.embed_tokens, "cpu_offload", False):
            # Host DRAM tables: the host only gathers raw FP8 bytes and scale
            # bytes (a memcpy-bound index_select); dequantization runs inside
            # the compiled graph (`_ready_rows`). Dequantizing on the host cost
            # ~86 ms/layer per 8192-token chunk single-threaded vs ~16 ms for the
            # raw gather.
            cols = self.embed_tokens.part_n_hash_cols
            blocks = head_dim // self.embed_tokens.block_size
            self._host_raw = torch.zeros(
                max_tokens, cols, head_dim, dtype=torch.uint8, device="cpu"
            )
            self._host_scale = torch.full(
                (max_tokens, cols, blocks), _UNIT_SCALE, dtype=torch.uint8, device="cpu"
            )
            self._staged_raw = torch.zeros(
                max_tokens, cols, head_dim, dtype=torch.uint8
            )
            self._staged_scale = torch.full(
                (max_tokens, cols, blocks), _UNIT_SCALE, dtype=torch.uint8
            )

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        if (
            self._staged_raw is not None
            and hash_ids.shape[0] <= self._host_raw.shape[0]
        ):
            num_tokens = hash_ids.shape[0]
            self.embed_tokens.gather_raw(
                hash_ids, self._host_raw[:num_tokens], self._host_scale[:num_tokens]
            )
            if num_tokens < self._host_raw.shape[0]:
                self._host_raw[num_tokens:].zero_()
                self._host_scale[num_tokens:].fill_(_UNIT_SCALE)
            # On TorchTPU, `copy_()` into a slice/view of a materialized device
            # tensor allocates a fresh buffer per call instead of overwriting in
            # place, whereas a whole-tensor `copy_()` overwrites in place (see
            # `deepseek_v4/model.py:_HostStagedParams`).
            self._staged_raw.copy_(self._host_raw)
            self._staged_scale.copy_(self._host_scale)
            return
        super().prepare_embeddings(hash_ids)

    def _ready_rows(self, num_tokens: int) -> torch.Tensor:
        if self._staged_raw is None:
            return super()._ready_rows(num_tokens)
        return dequant_fp8_rows(
            self._staged_raw[:num_tokens],
            self._staged_scale[:num_tokens],
            self.embed_tokens.block_size,
        )

    def _sp_embed(self, hash_ids):
        """Rows for this rank's SP token chunk with all hash heads.

        Each rank stages ``[T, local_heads, dim]`` rows for every token. A
        token-axis all-gather yields ``[tp, T_pad, local_heads, dim]`` whose
        rank-major head order matches upstream's non-SP ``dim=1`` gather.
        """
        rows = self._ready_rows(hash_ids.shape[0])
        cols = self.embed_tokens.n_hash_cols
        tp = self.embed_tokens.tp_size
        num_tokens = rows.shape[0]
        chunk = -(-num_tokens // tp)
        rank = upstream.get_tensor_model_parallel_rank()
        if tp == 1:
            return rows[:, :cols]
        pad = chunk * tp - num_tokens
        if pad:
            rows = torch.nn.functional.pad(rows, (0, 0, 0, 0, 0, pad))
        gathered = _gather_engram_rows(rows, dim=0).unflatten(0, (tp, chunk * tp))
        local = gathered[:, rank * chunk : (rank + 1) * chunk]
        return local.transpose(0, 1).flatten(1, 2)[:, :cols]

    def forward(self, hidden_states, hash_ids, token_mask=None):
        if self.use_sequence_parallel:
            tp = self.embed_tokens.tp_size
            chunk = -(-hash_ids.shape[0] // tp)
            assert hidden_states.shape[0] == chunk, (hidden_states.shape, chunk)
            if token_mask is not None:
                # Padded SP rows keep the gate shut.
                pad = chunk * tp - token_mask.shape[0]
                token_mask = torch.cat([token_mask, token_mask.new_zeros(pad)])
                rank = upstream.get_tensor_model_parallel_rank()
                token_mask = token_mask[rank * chunk : (rank + 1) * chunk]
            embedded = self._sp_embed(hash_ids)
        else:
            embedded = self.embed(hash_ids)
        kv = self.wkv(embedded.flatten(-2))
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
