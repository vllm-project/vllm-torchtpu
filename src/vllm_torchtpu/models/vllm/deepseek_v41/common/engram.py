"""Upstream Engram hashing, request history and TPU embedding lookup.

Shared by the v1 and v2 DeepSeek V4.1 implementations. Each implementation
subclasses ``NgramHashState`` to bind its own slot-cache write op and, when
its SWA cache module differs, the cache lifetime in ``ensure_cache``.
"""

from types import SimpleNamespace

import numpy as np
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
                np.asarray(t.to("cpu"), dtype=np.int64)
                for t in (self.token_map, self.multipliers, self.primes, self.offsets)
            )
            self._host_tables = tables
        token_map_np, mult_np, primes_np, offsets_np = tables
        ids_np = np.asarray(input_ids, dtype=np.int64)
        pos_np = np.asarray(positions, dtype=np.int64)
        qsl_np = np.asarray(query_start_loc, dtype=np.int64)
        dead_np = np.asarray(dead_mask, dtype=np.bool_)
        lb_np = np.asarray(lookback_token_ids, dtype=np.int64)
        lb_dead_np = np.asarray(lookback_dead_mask, dtype=np.bool_)
        rows = np.arange(count, dtype=np.int64)
        reqs = np.minimum(
            np.searchsorted(qsl_np[1:], rows, side="right"),
            lb_np.shape[0] - 1,
        )
        starts = np.minimum(qsl_np[reqs], count - 1)
        chunk_positions = pos_np[starts]
        shifts = np.arange(ngram, dtype=np.int64)
        past = pos_np[:, None] - shifts
        in_chunk = past >= chunk_positions[:, None]
        indices = np.maximum(rows[:, None] - shifts, 0)
        mapped_batch = token_map_np[ids_np].copy()
        mapped_batch[dead_np] = upstream.DEAD_ID
        batch = mapped_batch[indices]
        cols = chunk_positions[:, None] - 1 - past
        in_window = (~in_chunk) & (cols >= 0) & (cols < self.lookback_depth)
        cols_clamped = np.clip(cols, 0, self.lookback_depth - 1)
        reqs_col = reqs[:, None]
        window_ids = lb_np[reqs_col, cols_clamped]
        known = in_window & (window_ids >= 0)
        mapped_lb = token_map_np[np.maximum(lb_np, 0)].copy()
        mapped_lb[lb_dead_np] = upstream.DEAD_ID
        window = mapped_lb[reqs_col, cols_clamped]
        source = np.where(in_chunk, batch, np.where(known, window, int(self.pad_id)))
        blocked = np.cumsum((past < 0) | (source == upstream.DEAD_ID), axis=1) > 0
        source = np.where(blocked, int(self.pad_id), source)
        terms = source[:, None, :] * mult_np[None, :, :]
        rolled = np.bitwise_xor.accumulate(terms, axis=2)[:, :, 1:]
        hashed = (rolled[:, :, :, None] % primes_np[None, :, :, :]).reshape(
            count, layers, -1
        ) + offsets_np[None, :, :]
        return torch.from_numpy(np.ascontiguousarray(hashed, dtype=np.int32))

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
        _, ngram = multipliers.shape
        dev = input_ids.device
        rows = torch.arange(count, device=dev)
        reqs = torch.bucketize(rows, query_start_loc[1:], right=True).clamp(
            max=lookback_token_ids.shape[0] - 1
        )
        starts = query_start_loc[reqs].long().clamp(max=count - 1)
        chunk_positions = positions[starts]
        shifts = torch.arange(ngram, device=dev)
        past = positions[:, None] - shifts
        in_chunk = past >= chunk_positions[:, None]
        indices = (rows[:, None] - shifts).clamp(min=0)
        mapped_batch = token_map[input_ids.long()].masked_fill(
            dead_mask, upstream.DEAD_ID
        )
        batch = mapped_batch[indices]
        cols = chunk_positions[:, None] - 1 - past
        in_window = ~in_chunk & (cols >= 0) & (cols < self.lookback_depth)
        cols = cols.clamp(min=0, max=self.lookback_depth - 1).long()
        reqs_col = reqs[:, None]
        window_ids = lookback_token_ids[reqs_col, cols]
        known = in_window & (window_ids >= 0)
        mapped_lookback = token_map[lookback_token_ids.clamp(min=0).long()].masked_fill(
            lookback_dead_mask, upstream.DEAD_ID
        )
        window = mapped_lookback[reqs_col, cols]
        source = torch.where(
            in_chunk,
            batch,
            torch.where(known, window, fallback(past, reqs_col, window)),
        ).long()
        blocked = ((past < 0) | (source == upstream.DEAD_ID)).cumsum(dim=1) > 0
        source = source.masked_fill(blocked, self.pad_id)
        terms = source[:, None, :] * multipliers[None, :, :]
        rolling = terms[:, :, 0]
        rolling_steps = []
        for shift in range(1, ngram):
            rolling = rolling ^ terms[:, :, shift]
            rolling_steps.append(rolling)
        rolled = torch.stack(rolling_steps, dim=2)
        hashed = (rolled[:, :, :, None] % primes[None, :, :, :]).flatten(2) + offsets[
            None, :, :
        ]
        return hashed.to(torch.int32)


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
        device. Rows outside this shard (or padded heads) get the
        ``_INVALID_SCALE`` sentinel in every scale block; their raw bytes are
        left as gathered (masking them too cost ~11 ms per 8192-token step on
        the host). ``dequant_fp8_rows`` maps sentinel blocks to +0,
        exactly the zero rows that ``lookup`` writes for them.
        """
        weight, scales = self._storage()
        work_device = weight.device
        if indices.device != work_device:
            indices = indices.to(work_device)
        heads_cache = getattr(self, "_gather_heads_cache", None)
        if heads_cache is None or heads_cache[0].device != work_device:
            heads = (
                torch.arange(self.part_n_hash_cols, device=work_device)
                + self.head_start
            )
            heads_cache = (
                heads.clamp(max=self.n_hash_cols - 1),
                (heads < self.n_hash_cols)[None, :],
            )
            self._gather_heads_cache = heads_cache
        heads_clamped, heads_valid = heads_cache
        selected = indices[:, heads_clamped].long()
        valid = (
            heads_valid
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
        if not bool(valid.all()):
            scale_out.masked_fill_(~valid[:, :, None], _INVALID_SCALE)


# E8M0 scale byte for 2**0. Scale bytes encode 2**(byte - 127).
_UNIT_SCALE = 127
# E8M0 byte 255 is NaN and never appears in a real table; staging uses it to
# mark rows outside the shard, padded heads and the unused tail.
_INVALID_SCALE = 255


def dequant_fp8_rows(raw, scale, block_size):
    """``[..., dim]`` FP8 bytes x ``[..., dim // block_size]`` E8M0 bytes -> bf16.

    Matches ``ParallelEngramEmbedding.lookup``'s ``values.float() *
    exp2(scale - 127)`` rounded once to bf16. The power of two is built from
    its float32 exponent bits (``byte << 23``), which is exact for bytes
    1..254 and does not depend on the backend's ``exp2`` accuracy; the product
    of a float32 and a power of two is then exact, so only the final bf16
    rounding remains. Byte 0 (2**-127, a float32 subnormal) maps to 0.
    Byte 255 (``_INVALID_SCALE``) yields +0 whatever the raw bytes are,
    including FP8 NaN.
    """
    values = raw.view(torch.float8_e4m3fn).float()
    power = (scale.to(torch.int32) << 23).view(torch.float32)
    power = power.repeat_interleave(block_size, -1)
    invalid = (scale == _INVALID_SCALE).repeat_interleave(block_size, -1)
    return torch.where(invalid, 0.0, values * power).to(torch.bfloat16)


def _unpack_bytes(words, num_bytes):
    """Little-endian bytes of int32 ``words`` (1-D) -> first ``num_bytes`` uint8.

    Inverse of the host's ``uint8_buffer.view(torch.int32)`` on a
    little-endian host. Staging crosses H2D as 1-D int32 because TorchTPU only
    takes the fast linearizer path for 1-D 32-bit tensors.
    """
    shifts = torch.arange(0, 32, 8, dtype=torch.int32, device=words.device)
    unpacked = (words.unsqueeze(-1) >> shifts) & 0xFF
    return unpacked.to(torch.uint8).reshape(-1)[:num_bytes]


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
        self._active_raw: torch.Tensor | None = None
        self._staged_by_tokens: dict[int, torch.Tensor] = {}
        self._staged_views_by_tokens: dict[int, torch.Tensor] = {}
        if getattr(self.embed_tokens, "cpu_offload", False):
            # Host DRAM tables: the host only gathers raw FP8 bytes and scale
            # bytes (a memcpy-bound index_select); dequantization runs inside
            # the compiled graph (`_ready_rows`). Dequantizing on the host cost
            # ~86 ms/layer per 8192-token chunk single-threaded vs ~16 ms for the
            # raw gather.
            # Both buffers cross H2D packed into a single 1-D int32 tensor per
            # layer (`_unpack_bytes`): a 3-D uint8 `copy_()` takes TorchTPU's
            # slow XlaLinearize path (~9.75 vs ~0.59 ms/layer),
            # and packing raw + scale halves PJRT H2D transfers per step.
            cols = self.embed_tokens.part_n_hash_cols
            blocks = head_dim // self.embed_tokens.block_size
            assert head_dim % 4 == 0, head_dim
            raw_words_per_tok = (cols * head_dim) // 4
            scale_words_per_tok = -(-(cols * blocks) // 4)
            words_per_tok = raw_words_per_tok + scale_words_per_tok
            self._host_packed_buf = torch.zeros(
                max_tokens * words_per_tok * 4, dtype=torch.uint8, device="cpu"
            )
            self._host_raw = self._host_packed_buf[
                : max_tokens * raw_words_per_tok * 4
            ].view(max_tokens, cols, head_dim)
            scale_bytes = max_tokens * cols * blocks
            self._host_scale_buf = torch.full(
                (max_tokens * scale_words_per_tok * 4,),
                _INVALID_SCALE,
                dtype=torch.uint8,
                device="cpu",
            )
            self._host_scale = self._host_scale_buf[:scale_bytes].view(
                max_tokens, cols, blocks
            )
            self._staged_raw = torch.zeros(
                max_tokens * words_per_tok, dtype=torch.int32
            )
            # -1 is four `_INVALID_SCALE` bytes, matching `_host_scale_buf`.
            self._staged_raw[max_tokens * raw_words_per_tok :].fill_(-1)
            self._staged_by_tokens[max_tokens] = self._staged_raw
            self._active_raw = self._staged_views(max_tokens)

    def _staged_tensors(self, num_tokens: int) -> torch.Tensor:
        assert self._staged_raw is not None
        staged = self._staged_by_tokens.get(num_tokens)
        if staged is None:
            _, cols, head_dim = self._host_raw.shape
            blocks = self._host_scale.shape[-1]
            raw_words = num_tokens * ((cols * head_dim) // 4)
            scale_words = num_tokens * (-(-(cols * blocks) // 4))
            dev = self._staged_raw.device
            staged = torch.zeros(raw_words + scale_words, dtype=torch.int32, device=dev)
            staged[raw_words:].fill_(-1)
            self._staged_by_tokens[num_tokens] = staged
        return staged

    def _staged_views(self, num_tokens: int) -> torch.Tensor:
        view = self._staged_views_by_tokens.get(num_tokens)
        if view is None:
            _, cols, head_dim = self._host_raw.shape
            blocks = self._host_scale.shape[-1]
            raw_words_per_tok = (cols * head_dim) // 4
            scale_words_per_tok = -(-(cols * blocks) // 4)
            staged = self._staged_tensors(num_tokens)
            view = staged.view(num_tokens, raw_words_per_tok + scale_words_per_tok)
            if num_tokens >= 2:
                torch._dynamo.mark_dynamic(view, 0)
            self._staged_views_by_tokens[num_tokens] = view
        return view

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        if (
            self._staged_raw is not None
            and hash_ids.shape[0] <= self._host_raw.shape[0]
        ):
            num_tokens = hash_ids.shape[0]
            self.embed_tokens.gather_raw(
                hash_ids, self._host_raw[:num_tokens], self._host_scale[:num_tokens]
            )
            _, cols, head_dim = self._host_raw.shape
            blocks = self._host_scale.shape[-1]
            raw_words = num_tokens * ((cols * head_dim) // 4)
            scale_bytes = num_tokens * cols * blocks
            scale_words = num_tokens * (-(-(cols * blocks) // 4))
            if scale_words * 4 > scale_bytes:
                self._host_scale_buf[scale_bytes : scale_words * 4].fill_(
                    _INVALID_SCALE
                )
            # On TorchTPU, `copy_()` into a slice/view of a materialized device
            # tensor allocates a fresh buffer per call instead of overwriting in
            # place, whereas a whole-tensor `copy_()` overwrites in place (see
            # `deepseek_v4/model.py:_HostStagedParams`). Allocate and reuse one
            # exact-size whole 1-D packed device tensor per `num_tokens` bucket
            # so each Engram layer issues a single fast 1-D H2D transfer, while
            # exposing a 2-D `[num_tokens, words_per_tok]` dynamic-dim0 view to
            # `torch.compile` (`create_concrete_args`).
            host_words = self._host_packed_buf.view(torch.int32)
            host_words[raw_words : raw_words + scale_words].copy_(
                self._host_scale_buf[: scale_words * 4].view(torch.int32)
            )
            staged = self._staged_tensors(num_tokens)
            staged.copy_(host_words[: raw_words + scale_words])
            self._active_raw = self._staged_views(num_tokens)
            return
        super().prepare_embeddings(hash_ids)

    def _ready_rows(self, num_tokens: int) -> torch.Tensor:
        if self._staged_raw is None:
            return super()._ready_rows(num_tokens)
        _, cols, head_dim = self._host_raw.shape
        blocks = self._host_scale.shape[-1]
        raw_words_per_tok = (cols * head_dim) // 4
        scale_words_per_tok = -(-(cols * blocks) // 4)
        raw_bytes = num_tokens * cols * head_dim
        scale_bytes = num_tokens * cols * blocks
        assert self._active_raw is not None
        by_word = self._active_raw.view(
            raw_words_per_tok + scale_words_per_tok, num_tokens
        )
        raw = _unpack_bytes(by_word[:raw_words_per_tok].view(-1), raw_bytes)
        scale = _unpack_bytes(by_word[raw_words_per_tok:].view(-1), scale_bytes)
        return dequant_fp8_rows(
            raw.reshape(num_tokens, cols, head_dim),
            scale.reshape(num_tokens, cols, blocks),
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
