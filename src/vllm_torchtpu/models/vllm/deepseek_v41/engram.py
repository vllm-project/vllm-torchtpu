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
"""TPU Engram n-gram hashing, POSIX shared-memory storage, and 1-D int32 staging."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from vllm.config import get_current_vllm_config
from vllm.distributed import (
    get_engram_dp_group,
    get_engram_dp_size,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.deepseek_v41.common import engram as upstream
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

_INVALID_SCALE: int = 255
_FP32_MANTISSA_BITS: int = 23
_TPU_LANE_ALIGNMENT: int = 128
_WORD_BYTES: int = 4
_DEFAULT_ENGRAM_BLOCK_SIZE: int = 32
_DUMMY_SCALE_INV: int = 127
# vLLM's BlockPool reserves physical block 0 as the null block: it never holds
# a live request's tokens, but padded slot_mapping / block_table entries may
# point into it.
_NULL_BLOCK_ID: int = 0


def _write_slot_cache(
    cache: Tensor, slots: Tensor, values: Tensor, valid: Tensor
) -> None:
    """Write ``values`` into 1-D ``cache`` at ``slots`` where ``valid`` is True.

    Keeps the write shape static to avoid dynamic-shape recompiles, following
    the JAX KV cache writers (``kernels/mla/kv_cache_utils.py``), which point
    invalid rows at an out-of-bounds page that the scatter drops. Torch indexing
    cannot go out of bounds, so invalid rows go to a dummy slot ``cache[-1]``
    instead, which is never read.
    """
    dummy_slot = cache.numel() - 1
    index = torch.where(valid, slots, dummy_slot).long()
    cache[index] = values


def dequant_fp8_rows(
    raw: torch.Tensor,
    scale: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Dequantize ``uint8[..., dim]`` FP8 E4M3FN rows and E8M0 ``scale`` to BF16.

    Blocks marked with ``scale == 255`` (out-of-vocabulary or padded heads) map
    deterministically to ``0.0`` without propagating ``NaN`` from raw byte patterns.

    Args:
        raw: ``uint8[..., dim]`` raw FP8 E4M3FN byte tensor (Used).
        scale: ``uint8[..., dim // block_size]`` E8M0 exponent scale tensor (Used).
        block_size: Number of elements sharing each E8M0 block scale.

    Returns:
        ``bfloat16[..., dim]`` dequantized embedding rows (Created).
    """
    fp8_blocks = raw.view(torch.float8_e4m3fn).unflatten(-1, (-1, block_size)).float()
    scale_i32 = scale.to(torch.int32)
    invalid = (scale_i32 == _INVALID_SCALE).unsqueeze(-1)
    safe_scale = torch.where(scale_i32 == _INVALID_SCALE, 0, scale_i32)
    power = (safe_scale << _FP32_MANTISSA_BITS).view(torch.float32).unsqueeze(-1)
    zero = torch.zeros((), dtype=torch.bfloat16, device=raw.device)
    dequantized = (fp8_blocks * power).to(torch.bfloat16)
    return torch.where(invalid, zero, dequantized).flatten(-2)


def _unpack_bytes(words: torch.Tensor, num_bytes: int) -> torch.Tensor:
    """View 1-D ``int32`` staging ``words`` as ``num_bytes`` little-endian ``uint8``.

    Args:
        words: 1-D ``int32`` staging word tensor (Used).
        num_bytes: Number of leading ``uint8`` bytes to slice.

    Returns:
        1-D ``uint8[num_bytes]`` view (Created).
    """
    return words.contiguous().view(torch.uint8).reshape(-1)[:num_bytes]


def _gather_engram_rows(rows: torch.Tensor, dim: int) -> torch.Tensor:
    """All-gather ``rows`` across TP ranks with 128-lane BF16 alignment padding.

    Args:
        rows: Local shard tensor ``[..., width]`` (Used).
        dim: Dimension along which to all-gather across TP ranks.

    Returns:
        TP all-gathered tensor with trailing alignment padding stripped (Created).
    """
    width = rows.shape[-1]
    pad = (-width) % _TPU_LANE_ALIGNMENT
    padded = F.pad(rows, (0, pad)) if pad else rows
    gathered = tensor_model_parallel_all_gather(padded, dim=dim)
    if pad:
        return gathered[..., :width].contiguous()
    return gathered


def _dp_gather_engram_rows(staged: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """All-gather head-sharded ``staged`` rows across the Engram DP group.

    Args:
        staged: ``[dp_size * tokens_per_rank, local_heads, width]`` tensor (Used).
        num_tokens: Active local token count on this rank.

    Returns:
        ``[num_tokens, dp_size * local_heads, width]`` gathered rows (Created).
    """
    dp_group = get_engram_dp_group()
    if dp_group is None or dp_group.world_size <= 1:
        return staged[:num_tokens]
    dp_size = dp_group.world_size
    tokens_per_rank = staged.shape[0] // dp_size
    num_heads = staged.shape[1]
    width = staged.shape[2]
    pad = (-width) % _TPU_LANE_ALIGNMENT
    padded = F.pad(staged, (0, pad)) if pad else staged
    gathered = dp_group.all_gather(padded, dim=0)
    if pad:
        gathered = gathered[..., :width]
    start = dp_group.rank_in_group * tokens_per_rank
    selected = gathered.view(dp_size, dp_size * tokens_per_rank, num_heads, width)[
        :, start : start + num_tokens
    ]
    return (
        selected.transpose(0, 1)
        .reshape(num_tokens, dp_size * num_heads, width)
        .contiguous()
    )


class TpuDPSharedEngramStorage(upstream.DPSharedEngramStorage):
    """POSIX ``/dev/shm`` host weights shared across a node-local Engram DP group.

    # TODO(upstream): Drop this override once upstream ``DPSharedEngramStorage``
    # stops requiring CUDA ``cudaHostRegister``/``cudaHostUnregister`` and
    # ``get_accelerator_view``, or torch-tpu can register a ``MAP_SHARED``
    # ``/dev/shm`` mapping in place. ``torch-tpu <= 0.2.0`` lacks
    # ``TpuPinnedAllocator``; post-``0.2.0`` ``GetTpuPinnedAllocator()`` only
    # allocates from the ``16 GiB`` ``TPU_PREMAPPED_BUFFER_SIZE`` pool.
    """

    def _allocate(self, num_bytes: int) -> torch.Tensor:
        """Create a shared ``/dev/shm`` backing file on rank 0 and mmap on all ranks."""
        group = self.group
        with ExitStack() as stack:
            path: str | None = None
            error: str | None = None
            if group.rank_in_group == 0:
                try:
                    upstream.check_shm_free_space(num_bytes)
                    shm_file = stack.enter_context(
                        tempfile.NamedTemporaryFile(
                            prefix="vllm_tpu_engram_", dir=upstream.SHM_PATH
                        )
                    )
                    shm_file.truncate(num_bytes)
                    path = shm_file.name
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            path, error = cast(
                tuple[str | None, str | None],
                group.broadcast_object((path, error)),
            )
            if error is not None:
                raise RuntimeError(
                    "Engram shared-memory creation failed on EDP rank 0: " + error
                )

            tensor: torch.Tensor | None = None
            try:
                assert path is not None
                untyped = torch.UntypedStorage.from_file(
                    path, shared=True, nbytes=num_bytes
                )
                # pyrefly: ignore[bad-argument-type]
                tensor = torch.empty(0, dtype=torch.uint8, device="cpu").set_(untyped)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            errors: list[str | None] = [None] * group.world_size
            dist.all_gather_object(errors, error, group=group.cpu_group)
            failures = "; ".join(
                f"EDP rank {rank}: {rank_error}"
                for rank, rank_error in enumerate(errors)
                if rank_error is not None
            )
            if failures:
                raise RuntimeError(
                    "Engram shared-memory initialization failed: " + failures
                )
            assert tensor is not None
            return tensor

    def get_views(
        self, weight: torch.Tensor, scales: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Verify parameter storage pointers match the shared ``/dev/shm`` mapping.

        Args:
            weight: Current ``weight`` parameter tensor (Used).
            scales: Current ``weight_scale_inv`` parameter tensor (Used).

        Returns:
            Tuple ``(self.weight, self.weight_scale_inv)`` backed by ``/dev/shm``.

        Raises:
            RuntimeError: If either parameter storage pointer was replaced.
        """
        if (weight.data_ptr(), scales.data_ptr()) != (
            self.weight.data_ptr(),
            self.weight_scale_inv.data_ptr(),
        ):
            raise RuntimeError("Shared Engram parameter storage must not be replaced")
        return self.weight, self.weight_scale_inv


class NgramHashState(upstream.NgramHashState):
    """Subclass upstream ``NgramHashState`` replacing Triton kernels with PyTorch.

    Data lifecycle:
        - Created: ``forward`` return ``int32[tokens, layers,
          n_hash_cols]`` hash index tensors.
        - Used: ``input_ids``, ``positions``, ``query_start_loc``, ``dead_mask``,
          ``lookback_token_ids``, ``lookback_dead_mask``, and ``block_table``.
        - Changed: ``ensure_cache`` updates ``block_size`` and ``_cache``;
          ``forward`` updates ``_cache`` at active ``slot_mapping`` slots outside
          the null block ``0``.
    """

    def ensure_cache(self) -> bool:
        """Synchronize ``block_size`` with ``swa_cache_module`` and allocate slots.

        ``VllmDeepseekV4SWACache.get_kv_cache_spec`` finalizes ``block_size``
        after construction, and the bound ``kv_cache`` may be the overlaid CSA
        page tensor, so the logical block size comes from the module and never
        from ``kv_cache.shape``. The slot cache gets one extra trailing dummy
        slot for masked writes (see ``_write_slot_cache``).

        Returns:
            ``True`` when the slot cache (or SWA KV cache) is ready for hashing.
        """
        # We shouldn't invoke this method within a compiled function.
        assert not torch.compiler.is_compiling()
        swa = cast(DeepseekV4SWACache, self.swa_cache_module)
        block_size = int(swa.block_size)
        if self.use_slot_cache and self.block_size != block_size:
            self.block_size = block_size
            self._kv_cache_ref = None
        kv_cache = cast(Tensor, swa.kv_cache)
        # super() reallocates `_cache` only when a new KV cache is bound (and
        # then updates `_kv_cache_ref`), so check for that before calling it.
        rebind = self._kv_cache_ref is None or self._kv_cache_ref() is not kv_cache
        ready = super().ensure_cache()
        if ready and self.use_slot_cache and rebind:
            assert self._cache is not None
            # Append the dummy slot for `_write_slot_cache`, once per allocation.
            self._cache = F.pad(self._cache, (0, 1))
        return ready

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        query_start_loc: Tensor,
        dead_mask: Tensor,
        lookback_token_ids: Tensor,
        lookback_dead_mask: Tensor,
        slot_mapping: Tensor | None,
        block_table: Tensor | None,
    ) -> Tensor:
        """Update slot cache (skipping the null block) and compute n-gram hashes."""
        token_count = input_ids.numel()
        mapped_batch: Tensor | None = None
        if token_count > 0 and self.use_slot_cache:
            assert self._cache is not None and slot_mapping is not None
            slots = slot_mapping[:token_count]
            mapped_raw = self.token_map.index_select(0, input_ids.int())
            mapped_batch = mapped_raw.masked_fill(dead_mask, upstream.DEAD_ID)
            # Also drops padded slots (-1), like upstream's `slot >= 0`.
            first_live_slot = (_NULL_BLOCK_ID + 1) * self.block_size
            _write_slot_cache(
                self._cache, slots, mapped_batch, slots >= first_live_slot
            )

        def fallback_slot_lookup(past: Tensor, reqs: Tensor, window: Tensor) -> Tensor:
            if not self.use_slot_cache:
                return torch.full_like(window, self.pad_id)
            assert self._cache is not None and block_table is not None
            block_size = self.block_size
            num_rows, num_cols = block_table.shape
            # Real slots only; the trailing dummy slot is never read.
            num_slots = self._cache.numel() - 1
            logical_pos = past.int().clamp(0, num_cols * block_size - 1)
            table_rows = reqs.int().clamp(max=num_rows - 1)
            table_idx = (table_rows * num_cols + logical_pos // block_size).reshape(-1)
            blocks = (
                block_table.int()
                .reshape(-1)
                .index_select(0, table_idx)
                .view_as(logical_pos)
            )
            cache_slots = (blocks * block_size + logical_pos % block_size).clamp(
                0, num_slots - 1
            )
            gathered = self._cache.index_select(0, cache_slots.reshape(-1)).view_as(
                cache_slots
            )
            # Null-block (and unmapped -1) entries hold no history.
            return gathered.masked_fill(blocks <= _NULL_BLOCK_ID, upstream.DEAD_ID)

        tables = (self.token_map, self.multipliers, self.primes, self.offsets)
        return self._hash(
            tables,
            input_ids,
            positions,
            query_start_loc,
            dead_mask,
            lookback_token_ids,
            lookback_dead_mask,
            fallback_slot_lookup,
            mapped_batch,
        )

    def _hash(
        self,
        tables: tuple[Tensor, Tensor, Tensor, Tensor],
        input_ids: Tensor,
        positions: Tensor,
        query_start_loc: Tensor,
        dead_mask: Tensor,
        lookback_ids: Tensor,
        lookback_dead: Tensor,
        fallback: Callable[[Tensor, Tensor, Tensor], Tensor],
        mapped_batch: Tensor | None = None,
    ) -> Tensor:
        """Evaluate rolling XOR-modular n-gram hashes across batch and lookback IDs."""
        token_map, multipliers, primes, offsets = tables
        token_count = input_ids.numel()
        num_layers, ngram_size = multipliers.shape
        if token_count == 0:
            num_cols = (ngram_size - 1) * primes.shape[-1]
            return input_ids.new_empty((0, num_layers, num_cols), dtype=torch.int32)

        device = input_ids.device
        rows = torch.arange(token_count, dtype=torch.int32, device=device)
        query_bounds = query_start_loc.int()
        pos_i32 = positions.int()
        # Each token's request: mark interior request starts and cumsum, as in
        # qwen4_exp/ple.py (cheaper on TPU than a [tokens, reqs] comparison).
        # Starts at or past token_count land in a dropped overflow slot, and
        # padding rows past the last query boundary map to the last request.
        num_reqs = query_bounds.numel() - 1
        starts = query_bounds[1:num_reqs].clamp(max=token_count)
        marks = torch.zeros(token_count + 1, dtype=torch.int32, device=device)
        marks.index_add_(0, starts, torch.ones_like(starts))
        reqs = marks[:token_count].cumsum(0, dtype=torch.int32).clamp(max=num_reqs - 1)
        first_token_idx = query_bounds.index_select(0, reqs).clamp(max=token_count - 1)
        chunk_start_pos = pos_i32.index_select(0, first_token_idx)

        shifts = torch.arange(ngram_size, dtype=torch.int32, device=device)
        past = pos_i32[:, None] - shifts
        in_chunk = past >= chunk_start_pos[:, None]
        if mapped_batch is None:
            mapped_raw = token_map.index_select(0, input_ids.int())
            mapped_batch = mapped_raw.masked_fill(dead_mask, upstream.DEAD_ID)
        batch_indices = (rows[:, None] - shifts).clamp(min=0).reshape(-1)
        batch_tokens = mapped_batch.index_select(0, batch_indices).view(
            token_count, ngram_size
        )

        lookback_cols = chunk_start_pos[:, None] - 1 - past
        in_window = (
            ~in_chunk & (lookback_cols >= 0) & (lookback_cols < self.lookback_depth)
        )
        reqs_col = reqs[:, None]
        lookback_idx = (
            reqs_col * lookback_ids.shape[1]
            + lookback_cols.clamp(0, self.lookback_depth - 1)
        ).reshape(-1)
        window_raw_ids = (
            lookback_ids.reshape(-1)
            .index_select(0, lookback_idx)
            .view(token_count, ngram_size)
        )
        lookback_flat = lookback_ids.int().clamp(min=0).reshape(-1)
        mapped_lookback = (
            token_map.index_select(0, lookback_flat)
            .view_as(lookback_ids)
            .masked_fill(lookback_dead, upstream.DEAD_ID)
        )
        window_tokens = (
            mapped_lookback.reshape(-1)
            .index_select(0, lookback_idx)
            .view(token_count, ngram_size)
        )
        source_tokens = torch.where(
            in_chunk,
            batch_tokens,
            torch.where(
                in_window & (window_raw_ids >= 0),
                window_tokens,
                fallback(past, reqs_col, window_tokens),
            ),
        )
        blocked = ((past < 0) | (source_tokens == upstream.DEAD_ID)).cumsum(
            1, dtype=torch.int32
        ) > 0
        terms = (
            source_tokens.masked_fill(blocked, self.pad_id).long()[:, None, :]
            * multipliers[None, :, :]
        )
        rolling = terms[:, :, 0]
        rolled_orders: list[Tensor] = []
        for order_idx in range(1, ngram_size):
            rolling = rolling ^ terms[:, :, order_idx]
            rolled_orders.append(rolling)
        rolled = torch.stack(rolled_orders, dim=2)
        hashed = (rolled[:, :, :, None] % primes[None]).flatten(2) + offsets[None]
        return hashed.to(torch.int32)


class ParallelEngramEmbedding(upstream.ParallelEngramEmbedding):
    """TPU-adapted ``ParallelEngramEmbedding`` supporting HBM and ``/dev/shm`` offload.

    # TODO(upstream): Call ``super().__init__`` and drop ``gather_raw``/``lookup``
    # once upstream ``ParallelEngramEmbedding`` no longer requires CUDA
    # ``is_uva_available()``, ``torch.cuda.get_device_properties`` and the Triton
    # ``_engram_lookup_kernel``.
    """

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        block_size: int = _DEFAULT_ENGRAM_BLOCK_SIZE,
        cpu_offload: bool = False,
        dp_shared_memory: bool = False,
        use_thp: bool = False,
    ) -> None:
        # Call nn.Module.__init__ directly: upstream's __init__ queries CUDA
        # (is_uva_available, device properties), which fails on TPU. The
        # argument checks below mirror upstream's.
        nn.Module.__init__(self)
        assert head_sizes and all(size > 0 for size in head_sizes)
        assert sum(head_sizes) <= num_embeddings
        # TODO(Wenqin): we could add a new flag for TPU to keep only the first
        # layer of Engram on HBM.
        self.cpu_offload = bool(cpu_offload)
        self.dp_shared_memory = bool(dp_shared_memory)
        # Kept for upstream signature compatibility; transparent huge pages are
        # not implemented on TPU, so this flag is currently a no-op.
        self.use_thp = use_thp
        self.dp_size = get_engram_dp_size()
        if self.dp_shared_memory:
            if not self.cpu_offload:
                raise ValueError("dp_shared_memory requires cpu_offload=True")
            if self.dp_size <= 1:
                raise ValueError(
                    "dp_shared_memory requires a node-local Engram DP group with "
                    f"size > 1; effective Engram DP size is {self.dp_size}."
                )
            # DP replicas share one host table, so heads are only TP-sharded.
            self.dp_size = 1
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        self.n_hash_cols = len(head_sizes)
        self.tp_size = get_tensor_model_parallel_world_size()

        num_shards, head_rank = super()._get_shard_info()
        self.part_n_hash_cols = -(-self.n_hash_cols // num_shards)
        assert (num_shards - 1) * self.part_n_hash_cols < self.n_hash_cols, (
            f"Engram sharding leaves ranks without hash heads: "
            f"{self.n_hash_cols} heads over {num_shards} shards"
        )
        self.head_start = head_rank * self.part_n_hash_cols
        head_end = self.head_start + self.part_n_hash_cols
        vocab_start = sum(head_sizes[: self.head_start])
        self.vocab_start_idx = vocab_start
        self.vocab_end_idx = sum(head_sizes[:head_end])
        self.part_num_embeddings = self.vocab_end_idx - vocab_start
        self.peer_rows = 0

        weight, scales = self._allocate_weights()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale_inv = nn.Parameter(scales, requires_grad=False)
        weight_attrs = {
            "weight_loader": self._weight_loader,
            "engram_vocab_start": vocab_start,
        }
        for param in (self.weight, self.weight_scale_inv):
            upstream.set_weight_attrs(param, weight_attrs)
        upstream.set_weight_attrs(self.weight, {"dummy_weight_value": 1.0})
        upstream.set_weight_attrs(
            self.weight_scale_inv, {"dummy_weight_value": _DUMMY_SCALE_INV}
        )

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate weights in HBM, shared ``/dev/shm``, or private CPU RAM."""
        if not self.cpu_offload:
            return super()._allocate_weights()
        if self.dp_shared_memory:
            group = get_engram_dp_group()
            assert group is not None
            storage = TpuDPSharedEngramStorage(
                self.part_num_embeddings, self.dim, self.block_size, group
            )
            self._shared_memory = storage
            self._weight_loader = storage.load_weight
            return storage.weight, storage.weight_scale_inv
        num_rows = self.part_num_embeddings
        scale_dim = self.dim // self.block_size
        weight_cpu = torch.empty(
            num_rows, self.dim, dtype=torch.float8_e4m3fn, device="cpu"
        )
        scale_cpu = torch.empty(num_rows, scale_dim, dtype=torch.uint8, device="cpu")
        return weight_cpu, scale_cpu

    def _storage(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return backing ``(weight, weight_scale_inv)`` tensors."""
        if self._shared_memory is not None:
            return self._shared_memory.get_views(self.weight, self.weight_scale_inv)
        return self.weight.data, self.weight_scale_inv.data

    def _select_local_heads(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice local TP/EDP hash columns and compute row indices plus validity."""
        start = min(self.head_start, self.n_hash_cols)
        end = min(self.head_start + self.part_n_hash_cols, self.n_hash_cols)
        selected = indices.int()[:, start:end]
        valid = (selected >= self.vocab_start_idx) & (selected < self.vocab_end_idx)
        pad = self.part_n_hash_cols - (end - start)
        if pad:
            selected = F.pad(selected, (0, pad))
            valid = F.pad(valid, (0, pad), value=False)
        rows = (selected - self.vocab_start_idx).clamp(0, self.part_num_embeddings - 1)
        return rows, valid

    def gather_raw(
        self,
        indices: torch.Tensor,
        raw_out: torch.Tensor,
        scale_out: torch.Tensor,
    ) -> None:
        """Gather local heads' raw FP8 and E8M0 scale bytes into host staging.

        Args:
            indices: ``int32``/``int64`` hash indices ``[num_tokens, n_hash_cols]``
                (Used).
            raw_out: ``uint8[num_tokens, part_n_hash_cols, dim]`` destination
                (Changed).
            scale_out: ``uint8[num_tokens, part_n_hash_cols, blocks]`` destination
                (Changed).
        """
        weight, scales = self._storage()
        idx_on_storage = (
            indices.to(weight.device) if indices.device != weight.device else indices
        )
        rows, valid = self._select_local_heads(idx_on_storage)
        flat_rows = rows.reshape(-1)
        torch.index_select(
            weight.view(torch.uint8), 0, flat_rows, out=raw_out.view(-1, self.dim)
        )
        torch.index_select(
            scales, 0, flat_rows, out=scale_out.view(-1, scales.shape[1])
        )
        scale_out.masked_fill_(~valid.unsqueeze(-1), _INVALID_SCALE)

    def lookup(
        self,
        indices: torch.Tensor,
        out: torch.Tensor,
        background: bool = False,
    ) -> None:
        """Gather and dequantize local Engram rows into ``out``.

        Args:
            indices: ``int32``/``int64`` hash indices ``[num_tokens, n_hash_cols]``
                (Used).
            out: ``bfloat16[num_tokens, part_n_hash_cols, dim]`` output buffer
                (Changed).
            background: Ignored compatibility flag from upstream signature.
        """
        del background
        if indices.shape[0] * self.part_n_hash_cols == 0:
            return
        weight, scales = self._storage()
        out_shape = (indices.shape[0], self.part_n_hash_cols)
        raw = torch.empty(
            (*out_shape, self.dim), dtype=torch.uint8, device=weight.device
        )
        scale = torch.empty(
            (*out_shape, scales.shape[1]), dtype=torch.uint8, device=weight.device
        )
        self.gather_raw(indices, raw, scale)
        out.copy_(dequant_fp8_rows(raw, scale, self.block_size).to(out.dtype))


class Engram(upstream.Engram):
    """TPU-adapted ``Engram`` with 1-D ``int32`` host staging and PyTorch gating.

    # TODO(upstream): Drop the ``forward``/staging overrides once upstream
    # ``Engram`` no longer constructs ``torch.cuda.Stream``/``Event`` or calls
    # the Triton ``_fused_engram_post_wkv_kernel``.
    """

    def __init__(
        self,
        config: Any,
        quant_config: QuantizationConfig | None,
        layout: upstream.EngramLayout,
        layer_hash_index: int,
        use_sequence_parallel: bool,
        prefix: str,
        prefetch_stream: torch.cuda.Stream | None = None,
    ) -> None:
        if use_sequence_parallel:
            raise NotImplementedError(
                "TPU Engram does not support sequence parallelism yet"
            )
        super().__init__(
            config,
            quant_config,
            layout,
            layer_hash_index,
            use_sequence_parallel,
            prefix,
            prefetch_stream,
        )

    def _create_embedding(
        self,
        layout: upstream.EngramLayout,
        layer_hash_index: int,
    ) -> ParallelEngramEmbedding:
        cfg = get_current_vllm_config().engram_config
        assert cfg is not None
        return ParallelEngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            tuple(size for order in layout.primes[layer_hash_index] for size in order),
            cpu_offload=cfg.cpu_offload,
            dp_shared_memory=bool(cfg.dp_shared_memory),
            use_thp=cfg.use_thp,
        )

    def _init_staging(self, max_tokens: int, head_dim: int) -> None:
        """Initialize per-bucket 1-D ``int32`` device and host staging state."""
        dp_tokens = max_tokens * self.embed_tokens.dp_size
        cols = self.embed_tokens.part_n_hash_cols
        self.staged_rows = torch.empty(dp_tokens, cols, head_dim, dtype=torch.bfloat16)
        self._prefetch_stream = None
        self._prefetch_done = None
        self._staged_raw = None
        self._active_staged: torch.Tensor | None = None
        self._active_raw: torch.Tensor | None = None
        self._staged_by_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._host_buf_by_tokens: dict[int, torch.Tensor] = {}
        if not self.embed_tokens.cpu_offload:
            return

        blocks = head_dim // self.embed_tokens.block_size
        raw_words = (cols * head_dim) // _WORD_BYTES
        scale_words = -(-(cols * blocks) // _WORD_BYTES)
        words_per_token = raw_words + scale_words
        self._scale_blocks = blocks
        self._words_per_tok = words_per_token
        self._staging_shape = (dp_tokens, cols, head_dim)
        init = torch.zeros(dp_tokens * words_per_token, dtype=torch.int32, device="cpu")
        init[dp_tokens * raw_words :].fill_(-1)
        self._staged_raw = init.to(self.staged_rows.device)

    def _staged_pair(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(flat_root_i32, view_2d)`` device staging buffers for a bucket."""
        assert self._staged_raw is not None
        pair = self._staged_by_tokens.get(num_tokens)
        if pair is None:
            max_tokens, cols, head_dim = self._staging_shape
            words_per_token = self._words_per_tok
            raw_words = num_tokens * ((cols * head_dim) // _WORD_BYTES)
            if num_tokens == max_tokens:
                staged = self._staged_raw
            else:
                # Init the staged buffer on CPU to avoid creating a new graph on TPU.
                init = torch.zeros(
                    num_tokens * words_per_token, dtype=torch.int32, device="cpu"
                )
                init[raw_words:].fill_(-1)
                staged = init.to(self._staged_raw.device)
            view = staged.view(num_tokens, words_per_token)
            if num_tokens >= 2:
                torch._dynamo.mark_dynamic(view, 0)
            pair = (staged, view)
            self._staged_by_tokens[num_tokens] = pair
        return pair

    def is_staged(self) -> bool:
        """Whether 1-D ``int32`` host-offload staging is active."""
        return self._staged_raw is not None

    def staged_device_tensors(self) -> tuple[torch.Tensor, ...]:
        """Active staged device buffers prepared on the current step."""
        return tuple(
            tensor
            for tensor in (self._active_staged, self._active_raw)
            if tensor is not None and tensor.numel() > 0
        )

    def _host_buf(self, num_tokens: int, raw_bytes: int, total_bytes: int) -> Tensor:
        """Return the CPU staging buffer for ``num_tokens``.

        One buffer per bucket is enough because the H2D ``copy_`` is synchronous,
        so the buffer is free again once ``prepare_embeddings`` returns. Bytes
        past ``raw_bytes`` start as ``_INVALID_SCALE``; ``gather_raw`` only ever
        rewrites the scale bytes, so the padding tail stays invalid.

        Note: ``torch-tpu <= 0.2.0`` does not register ``TpuPinnedAllocator``
        (``is_pin_memory_available() -> False``, so ``Tensor.pin_memory()``
        raises ``RuntimeError``); post-``0.2.0`` ``GetTpuPinnedAllocator()``
        allocates from the ``16 GiB`` ``TPU_PREMAPPED_BUFFER_SIZE`` pool, while
        PJRT transfers standard CPU buffers directly during synchronous ``copy_``.
        """
        buf = self._host_buf_by_tokens.get(num_tokens)
        if buf is None:
            buf = torch.zeros(total_bytes, dtype=torch.uint8, device="cpu")
            buf[raw_bytes:].fill_(_INVALID_SCALE)
            self._host_buf_by_tokens[num_tokens] = buf
        return buf

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Gather host FP8/E8M0 rows and copy into the 1-D ``int32`` device buffer.

        Args:
            hash_ids: ``int32[num_tokens, n_hash_cols]`` hash indices (Used).

        Raises:
            ValueError: If ``num_tokens`` exceeds the host-offload staging size.
        """
        num_tokens = hash_ids.shape[0]
        if self._staged_raw is None:
            super().prepare_embeddings(hash_ids)
            return
        max_tokens = self._staging_shape[0]
        if num_tokens > max_tokens:
            raise ValueError(
                f"Engram staging holds at most {max_tokens} tokens, "
                f"got num_tokens={num_tokens}"
            )
        staged, view = self._staged_pair(num_tokens)
        if num_tokens > 0:
            _, cols, head_dim = self._staging_shape
            blocks = self._scale_blocks
            words_per_token = self._words_per_tok
            raw_bytes = num_tokens * cols * head_dim
            scale_bytes = num_tokens * cols * blocks
            host_buf = self._host_buf(
                num_tokens,
                raw_bytes,
                num_tokens * words_per_token * _WORD_BYTES,
            )
            host_raw = host_buf[:raw_bytes].view(num_tokens, cols, head_dim)
            host_scale = host_buf[raw_bytes : raw_bytes + scale_bytes].view(
                num_tokens, cols, blocks
            )
            cast(ParallelEngramEmbedding, self.embed_tokens).gather_raw(
                hash_ids, host_raw, host_scale
            )
            # Whole-tensor 1-D int32 copy updates the captured root buffer in place.
            staged.copy_(host_buf.view(torch.int32), non_blocking=False)
        self._active_staged = staged
        self._active_raw = view

    def _ready_rows(self, num_tokens: int) -> torch.Tensor:
        """Unpack, dequantize, and DP-gather staged rows for ``num_tokens``."""
        dp_size = self.embed_tokens.dp_size
        if self._staged_raw is None:
            if dp_size > 1:
                slot = upstream.engram_gathered_num_tokens()
                return _dp_gather_engram_rows(
                    self.staged_rows[: slot * dp_size], num_tokens
                )
            return self.staged_rows[:num_tokens]

        _, cols, head_dim = self._staging_shape
        blocks = self._scale_blocks
        raw_words_per_tok = (cols * head_dim) // _WORD_BYTES
        scale_words_per_tok = -(-(cols * blocks) // _WORD_BYTES)
        active = self._active_raw
        if active is None:
            fallback_tokens = (
                num_tokens
                if dp_size == 1
                else upstream.engram_gathered_num_tokens() * dp_size
            )
            _, active = self._staged_pair(fallback_tokens)
        staged_tokens = active.shape[0]
        if dp_size == 1 and staged_tokens < num_tokens:
            raise RuntimeError(
                f"Engram staged {staged_tokens} tokens in prepare_embeddings, "
                f"but forward needs num_tokens={num_tokens}"
            )
        by_word = active.view(raw_words_per_tok + scale_words_per_tok, staged_tokens)
        raw = _unpack_bytes(
            by_word[:raw_words_per_tok].reshape(-1),
            staged_tokens * cols * head_dim,
        ).reshape(staged_tokens, cols, head_dim)
        scale = _unpack_bytes(
            by_word[raw_words_per_tok:].reshape(-1),
            staged_tokens * cols * blocks,
        ).reshape(staged_tokens, cols, blocks)
        rows = dequant_fp8_rows(raw, scale, self.embed_tokens.block_size)
        if dp_size > 1:
            return _dp_gather_engram_rows(rows, num_tokens)
        return rows[:num_tokens]

    def embed(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """Return dequantized ``[tokens, n_hash_cols, head_dim]`` Engram rows.

        Args:
            hash_ids: ``int32[num_tokens, n_hash_cols]`` hash indices (Used).

        Returns:
            ``bfloat16[num_tokens, n_hash_cols, head_dim]`` embedding tensor (Created).
        """
        rows = self._ready_rows(hash_ids.shape[0])
        if self.embed_tokens.tp_size > 1:
            rows = _gather_engram_rows(rows, dim=1)
        return rows[:, : self.embed_tokens.n_hash_cols]

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply RMS-normalized signed-sqrt sigmoid gating over Engram embeddings.

        Args:
            hidden_states: ``[num_tokens, hc_mult, dim]`` residual streams (Used).
            hash_ids: ``int32[num_tokens, n_hash_cols]`` hash indices (Used).
            token_mask: Optional ``bool[num_tokens]`` mask disabling gated updates
                on dead/sentinel tokens (Used).

        Returns:
            Updated ``[num_tokens, hc_mult, dim]`` residual stream tensor (Created).
        """
        if hidden_states.shape[0] == 0:
            return torch.empty_like(hidden_states)
        projected_kv = (
            self.wkv(self.embed(hash_ids).flatten(-2))
            .unflatten(-1, (self.hc_mult + 1, self.dim))
            .float()
        )
        hidden_f32 = hidden_states.float()
        keys = projected_kv[:, : self.hc_mult]
        values = projected_kv[:, self.hc_mult :]
        q_weight = self.q_weight.float()
        k_weight = self.k_weight.float()
        hidden_rms = torch.rsqrt(hidden_f32.square().mean(-1) + self.eps)
        key_rms = torch.rsqrt(keys.square().mean(-1) + self.eps)
        dot = (
            (hidden_f32 * q_weight * k_weight * keys).sum(-1)
            * hidden_rms
            * key_rms
            * (self.dim**-0.5)
        )
        gated_mag = dot.abs().clamp(min=self.clamp_value).sqrt()
        gate = torch.where(dot < 0, -gated_mag, gated_mag).sigmoid()
        if token_mask is not None:
            gate = gate * token_mask[:, None]
        return (hidden_f32 + gate.unsqueeze(-1) * values).to(hidden_states.dtype)
