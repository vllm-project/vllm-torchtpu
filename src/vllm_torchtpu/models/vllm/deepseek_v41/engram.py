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
from typing import cast

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
from vllm.models.deepseek_v41.common import engram as upstream
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

from .cache import NativeCacheView, write_cache

_INVALID_SCALE: int = 255
_FP32_MANTISSA_BITS: int = 23
_TPU_LANE_ALIGNMENT: int = 128
_WORD_BYTES: int = 4
_DEFAULT_ENGRAM_BLOCK_SIZE: int = 32
_DUMMY_SCALE_INV: int = 127


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
    pad = 0 if rows.dtype == torch.int32 else (-width) % _TPU_LANE_ALIGNMENT
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

    # TODO(upstream): ``vllm/models/deepseek_v41/common/engram.py:813,847`` calls
    # CUDA ``cudaHostRegister``/``cudaHostUnregister`` and ``get_accelerator_view``.
    # ``torch-tpu <= 0.2.0`` lacks ``TpuPinnedAllocator``; post-``0.2.0``
    # ``GetTpuPinnedAllocator()`` allocates from the ``16 GiB``
    # ``TPU_PREMAPPED_BUFFER_SIZE`` pool and lacks in-place ``MAP_SHARED``
    # ``/dev/shm`` host-pointer registration.
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
        - Created: ``hash_on_host`` and ``forward`` return ``int32[tokens, layers,
          n_hash_cols]`` hash index tensors.
        - Used: ``input_ids``, ``positions``, ``query_start_loc``, ``dead_mask``,
          ``lookback_ids``, ``lookback_dead``, and ``block_table``.
        - Changed: ``ensure_cache`` updates ``block_size`` and ``_cache``;
          ``forward`` updates ``_cache`` at active ``slot_mapping`` slots outside
          ``null_block=0``.
    """

    _cached_cpu_tables: tuple[Tensor, Tensor, Tensor, Tensor] | None = None

    def ensure_cache(self) -> bool:
        """Synchronize ``block_size`` with ``swa_cache_module`` and allocate slots.

        Returns:
            ``True`` when the slot cache (or SWA KV cache) is ready for hashing.
        """
        swa = cast(DeepseekV4SWACache, self.swa_cache_module)
        kv_cache = cast(Tensor, swa.kv_cache)
        if torch.compiler.is_compiling():
            if self.use_slot_cache:
                return self._cache is not None
            return kv_cache.numel() > 0

        if kv_cache.numel():
            resolved_block_size: int | None = int(
                kv_cache.shape[2] if kv_cache.ndim == 4 else kv_cache.shape[1]
            )
        else:
            resolved_block_size = cast(int | None, swa.block_size)
        if (
            self.use_slot_cache
            and resolved_block_size is not None
            and self.block_size != resolved_block_size
        ):
            self.block_size = int(resolved_block_size)
            self._kv_cache_ref = None
        return super().ensure_cache()

    def hash_on_host(
        self,
        input_ids: Tensor,
        positions: Tensor,
        query_start_loc: Tensor,
        dead_mask: Tensor,
        lookback_ids: Tensor,
        lookback_dead: Tensor,
    ) -> Tensor:
        """Compute ``int32[tokens, layers, n_hash_cols]`` hashes on CPU tensors."""
        tables = self._cached_cpu_tables
        if tables is None:
            token_map_cpu = self.token_map.cpu()
            multipliers_cpu = self.multipliers.cpu()
            primes_cpu = self.primes.cpu()
            offsets_cpu = self.offsets.cpu()
            tables = (token_map_cpu, multipliers_cpu, primes_cpu, offsets_cpu)
            self._cached_cpu_tables = tables

        def fallback_pad(_past: Tensor, _reqs: Tensor, window: Tensor) -> Tensor:
            return torch.full_like(window, self.pad_id)

        return self._hash(
            tables,
            input_ids,
            positions,
            query_start_loc,
            dead_mask,
            lookback_ids,
            lookback_dead,
            fallback_pad,
        )

    # pyrefly: ignore[bad-override-param-name]
    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        query_start_loc: Tensor,
        dead_mask: Tensor,
        lookback_ids: Tensor,
        lookback_dead: Tensor,
        slot_mapping: Tensor | None,
        block_table: Tensor | None,
    ) -> Tensor:
        """Update slot cache (skipping ``null_block=0``) and compute n-gram hashes."""
        swa = self.swa_cache_module
        paged_swa = swa.paged_cache_rows if isinstance(swa, NativeCacheView) else None
        has_null_block = paged_swa.null_block if paged_swa is not None else False
        min_block = 1 if has_null_block else 0

        token_count = input_ids.numel()
        mapped_batch: Tensor | None = None
        if token_count > 0 and self.use_slot_cache:
            assert self._cache is not None and slot_mapping is not None
            slots = slot_mapping[:token_count]
            mapped_raw = self.token_map.index_select(0, input_ids.int())
            mapped_batch = mapped_raw.masked_fill(dead_mask, upstream.DEAD_ID)
            write_cache(
                self._cache, slots, mapped_batch, slots >= min_block * self.block_size
            )

        def fallback_slot_lookup(past: Tensor, reqs: Tensor, window: Tensor) -> Tensor:
            if not self.use_slot_cache:
                return torch.full_like(window, self.pad_id)
            assert self._cache is not None and block_table is not None
            block_size = self.block_size
            num_cols = block_table.shape[1]
            logical_pos = past.int().clamp(0, num_cols * block_size - 1)
            table_idx = (reqs.int() * num_cols + logical_pos // block_size).reshape(-1)
            blocks = (
                block_table.int()
                .reshape(-1)
                .index_select(0, table_idx)
                .view_as(logical_pos)
            )
            cache_slots = (blocks * block_size + logical_pos % block_size).clamp(
                0, self._cache.numel() - 1
            )
            gathered = self._cache.index_select(0, cache_slots.reshape(-1)).view_as(
                cache_slots
            )
            return gathered.masked_fill(blocks < min_block, upstream.DEAD_ID)

        tables = (self.token_map, self.multipliers, self.primes, self.offsets)
        return self._hash(
            tables,
            input_ids,
            positions,
            query_start_loc,
            dead_mask,
            lookback_ids,
            lookback_dead,
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
        reqs = (rows.unsqueeze(1) >= query_bounds[1:].unsqueeze(0)).sum(
            dim=-1, dtype=torch.int32
        )
        reqs = reqs.clamp(max=lookback_ids.shape[0] - 1)
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

    # TODO(upstream): ``vllm/models/deepseek_v41/common/engram.py:981,1002,1143``
    # requires CUDA ``is_uva_available()``, ``torch.cuda.get_device_properties``,
    # and Triton ``_engram_lookup_kernel``.
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
        # NOTE(cross-boundary-contract): Call nn.Module.__init__(self) directly
        # instead of super().__init__ to bypass upstream common/engram.py:981 CUDA
        # is_uva_available() queries on TPU. Assumes positive head_sizes from
        # EngramLayout.primes; dp_shared_memory is gated on cpu_offload and sets
        # self.dp_size = 1, while use_thp is retained for upstream constructor
        # signature compatibility.
        nn.Module.__init__(self)
        host_offload = bool(cpu_offload)
        self.cpu_offload = host_offload
        self.host_offload = host_offload
        shared_mem = bool(dp_shared_memory) and host_offload
        self.dp_shared_memory = shared_mem
        self._dp_shared_memory = shared_mem
        self.use_thp = use_thp
        self.dp_size = 1 if self.dp_shared_memory else get_engram_dp_size()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        self.n_hash_cols = len(head_sizes)
        self.tp_size = get_tensor_model_parallel_world_size()

        num_shards, head_rank = super()._get_shard_info()
        self.part_n_hash_cols = -(-self.n_hash_cols // num_shards)
        self.head_start = head_rank * self.part_n_hash_cols
        head_end = self.head_start + self.part_n_hash_cols
        vocab_start = sum(head_sizes[: self.head_start])
        self.vocab_start_idx = vocab_start
        self.vocab_end_idx = sum(head_sizes[:head_end])
        self.part_num_embeddings = self.vocab_end_idx - vocab_start

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
        if not self.host_offload:
            return super()._allocate_weights()
        if self._dp_shared_memory:
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
        if not bool(valid.all()):
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

    # TODO(upstream): ``vllm/models/deepseek_v41/common/engram.py:1603,1711``
    # constructs ``torch.cuda.Stream``/``Event`` and invokes Triton
    # ``_fused_engram_post_wkv_kernel``.
    """

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
        """Initialize per-bucket 1-D ``int32`` device staging and host ring state."""
        dp_tokens = max_tokens * self.embed_tokens.dp_size
        cols = self.embed_tokens.part_n_hash_cols
        self.staged_rows = torch.empty(dp_tokens, cols, head_dim, dtype=torch.bfloat16)
        self._prefetch_stream = None
        self._prefetch_done = None
        self._staged_raw = None
        self._active_staged: torch.Tensor | None = None
        self._active_raw: torch.Tensor | None = None
        self._staged_by_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._host_ring_by_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._host_ring_idx: dict[int, int] = {}
        if not self.embed_tokens.cpu_offload:
            return

        blocks = head_dim // self.embed_tokens.block_size
        raw_words = (cols * head_dim) // _WORD_BYTES
        scale_words = -(-(cols * blocks) // _WORD_BYTES)
        words_per_token = raw_words + scale_words
        self._scale_blocks = blocks
        self._words_per_tok = words_per_token
        self._staging_shape = (dp_tokens, cols, head_dim)
        init = torch.zeros(dp_tokens * words_per_token, dtype=torch.int32)
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
                init = torch.zeros(num_tokens * words_per_token, dtype=torch.int32)
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

    def _host_slot(self, num_tokens: int, raw_bytes: int, total_bytes: int) -> Tensor:
        """Return the next alternating CPU staging buffer for ``num_tokens``.

        Note: ``torch-tpu <= 0.2.0`` does not register ``TpuPinnedAllocator``
        (``is_pin_memory_available() -> False``, so ``Tensor.pin_memory()``
        raises ``RuntimeError``); post-``0.2.0`` ``GetTpuPinnedAllocator()``
        allocates from the ``16 GiB`` ``TPU_PREMAPPED_BUFFER_SIZE`` pool, while
        PJRT transfers standard CPU buffers directly during synchronous ``copy_``.
        """
        ring = self._host_ring_by_tokens.get(num_tokens)
        if ring is None:
            buf_a = torch.zeros(total_bytes, dtype=torch.uint8, device="cpu")
            buf_b = torch.zeros(total_bytes, dtype=torch.uint8, device="cpu")
            buf_a[raw_bytes:].fill_(_INVALID_SCALE)
            buf_b[raw_bytes:].fill_(_INVALID_SCALE)
            ring = (buf_a, buf_b)
            self._host_ring_by_tokens[num_tokens] = ring
        slot_idx = self._host_ring_idx.get(num_tokens, 0)
        self._host_ring_idx[num_tokens] = slot_idx ^ 1
        return ring[slot_idx]

    def prepare_embeddings(self, hash_ids: torch.Tensor) -> None:
        """Gather host FP8/E8M0 rows and copy into the 1-D ``int32`` device buffer.

        Args:
            hash_ids: ``int32[num_tokens, n_hash_cols]`` hash indices (Used).
        """
        num_tokens = hash_ids.shape[0]
        if self._staged_raw is not None and num_tokens <= self._staging_shape[0]:
            staged, view = self._staged_pair(num_tokens)
            if num_tokens > 0:
                _, cols, head_dim = self._staging_shape
                blocks = self._scale_blocks
                words_per_token = self._words_per_tok
                raw_bytes = num_tokens * cols * head_dim
                scale_bytes = num_tokens * cols * blocks
                host_buf = self._host_slot(
                    num_tokens,
                    raw_bytes,
                    num_tokens * words_per_token * _WORD_BYTES,
                )
                host_buf[raw_bytes + scale_bytes :] = _INVALID_SCALE
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
            return
        super().prepare_embeddings(hash_ids)

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
