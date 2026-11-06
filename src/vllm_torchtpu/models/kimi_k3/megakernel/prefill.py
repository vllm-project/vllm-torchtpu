# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/kimi_k3/prefill.py
"""Kimi-K3 prefill: the sequence-parallel prefill stack over vLLM's unified pools.

:class:`KimiPrefiller` (the prefill role's :class:`MegakernelProgram`) computes
every scheduled prompt chunk into vLLM's unified pools for the KV connector:

* **Windows.** A step's rows are cut into pieces of at most the largest window
  (:func:`plan_pieces`); the i-th pieces of all rows are packed, up to
  :data:`SLOTS` sequences (at most one per page of the window) at
  page-aligned offsets, into windows run in order
  (:func:`pack_windows`) through the ``first`` (embedding + layers 0-3),
  ``group`` (eight layers per call) and ``final`` (layer 92 + logits) ops.
* **Context parallel over the pool** (DCP 32): rank ``r`` holds a sequence's
  global pages ``k`` with ``k % 32 == r``; MLA layers attend each rank's share
  of the cached prefixes, merged across ranks, and each rank writes the new
  pages it owns.
* **KDA state** stays in memory per request (:class:`_Request`), or comes from
  the block vLLM's ``align`` mode keeps at a prefix-cache hit (:func:`skips_step`
  covers ``align``-mode splits).
* **First tokens** are chosen in an op from the finished prompts' last-token
  logits, with the decoder's in-op sampling.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.logger import init_logger

from . import sampling
from .draft_mla_prefill import PrefillDraftLatents
from .draft_ring import PrefillRing
from .layout import NUM_LAYERS, cache_layout
from .pools import (
    CONV_ROWS,
    KDA_LAYERS,
    MLA_LAYERS,
    SSM_ROWS,
    bind_raw_pools,
    draft_raw_pools,
    kda_blocks_at_column,
    kda_blocks_at_length,
    read_draft_rings,
    tp_rank_permutation,
    write_draft_rings,
)
from .program import (
    DCP_SIZE,
    HIDDEN_SIZE,
    TP_SIZE,
    VOCAB,
    kernel_modules,
)
from .runtime_shim.bridge import (
    OpRegistrationSpec,
    hbm,
    kernel_mesh,
    materialize,
    register_global_op,
    register_op,
    report_load_stats,
    stream_to_device,
)
from .runtime_shim.compilation import (
    CompileCall,
    compile_ops,
)
from .runtime_shim.export_cache import prepare_distributed
from .runtime_shim.program import (
    KVGroup,
    RequestInput,
    StepInput,
    StepOutput,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from jax.sharding import Mesh

    from .runtime_shim.config import MegakernelEngineConfig

    type CustomOp = Callable[..., tuple[torch.Tensor, ...]]
    type RingPair = tuple[torch.Tensor, torch.Tensor]

logger = init_logger(__name__)

AXIS = "k3_sp"
WINDOWS = (512, 1024, 2048, 4096, 8192)
PAGE_TOKENS = 256
SLOTS = 16  # sequences per window, at most (and at most one per page)
GROUP_PAIRS = 11  # regular 8-layer groups: layers 4..91
COPY_PAIRS = 8  # copy-on-write block copies per PoolWriter.copy_blocks call

_MIN_SPLITS_FOR_EARLY_STOP = 2
_FIRST_STACK_DEPTH = 4
_FIRST_TAP_LAYER = 3
_MLA_FLASH_TILE = 256
_MLA_LATENT_DIM = 576
_MLA_PAD_COLS = 64
_KDA_HEADS = 3
_KDA_HEAD_DIM = 128
_KDA_CONV_TAPS = 3
_GROUP_RUN_HEADER_ARGS = 10
_FINAL_EXPERT_BANK = 12
_PREFIX_ALIGN_TOKENS = 4096
_CP_ALIGN_TOKENS = 1024
_SAMPLING_PARAM_COLS = 5
_POSITIVE_INT32_MASK = 0x7FFFFFFF
_SEGMENT_DESC_FIELDS = 5
_RING_META_ELEMENTS = 2
_STATE_SPEC_COUNT = 7


def live_slots(used: int, slots: int) -> int:
    """Return ``used`` rounded up to a power of two, capped at ``slots``."""
    return min(slots, 1 << max(used - 1, 0).bit_length())


def write_sizes(slots: int) -> tuple[int, ...]:
    """Return every distinct :func:`live_slots` count for ``slots`` sequences."""
    return tuple(sorted({live_slots(n, slots) for n in range(1, slots + 1)}))


def _device_i32(values: Sequence[int] | Sequence[Sequence[int]]) -> torch.Tensor:
    """Create an int32 tensor from ``values`` and place it on the TPU device."""
    return torch.tensor(values, dtype=torch.int32).to("tpu")


def _zeros_i32(n: int) -> torch.Tensor:
    """Allocate a 1D int32 zero tensor of length ``n`` on the TPU device."""
    return torch.zeros(n, dtype=torch.int32).to("tpu")


def _rank_zeros_i32(n: int) -> torch.Tensor:
    """Allocate a rank-leading ``[1, n]`` int32 zero tensor on the TPU device."""
    return torch.zeros((1, n), dtype=torch.int32).to("tpu")


def _as_float(bits: jax.Array) -> jax.Array:
    """Bitcast int32 parameter words ``bits`` back to float32 values."""
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def prefill_windows(max_num_batched_tokens: int, max_model_len: int) -> tuple[int, ...]:
    """Return compiled prefill windows covering step budget and model length."""
    windows = [w for w in WINDOWS if w <= max(WINDOWS[0], max_num_batched_tokens)]
    holding = [w for w in windows if w >= max_model_len]
    return tuple(w for w in windows if w < max_model_len) + (
        (min(holding),) if holding else ()
    )


@dataclasses.dataclass(frozen=True)
class Piece:
    """Tokens ``[start, end)`` of one request computed in one packed window.

    ``stop`` (a skipped step's end in ``(start, end]``) is where the window
    also checkpoints the KDA state, or ``None`` when no checkpoint is needed.
    """

    row: int
    start: int
    end: int
    stop: int | None = None
    page_tokens: int = PAGE_TOKENS

    @property
    def length(self) -> int:
        """Return the unpadded token count ``end - start`` in this piece."""
        return self.end - self.start

    @property
    def rows(self) -> int:
        """Return the page-padded window row count occupied by this piece."""
        return -(-self.length // self.page_tokens) * self.page_tokens


def skips_step(
    computed: int,
    end: int,
    prompt: int,
    start: int,
    *window_and_block: int,
) -> bool:
    """Return True when a partial prefill step ``[computed, end)`` can be deferred.

    A step is deferred when ``end < prompt``, the remaining suffix from ``start``
    fits the largest window, and ``[computed, end)`` does not cross a logical
    offloading block boundary.
    """
    largest, block = window_and_block
    if end >= prompt or prompt - start > largest:
        return False
    return end // block == computed // block


def plan_pieces(
    row: int,
    start: int,
    end: int,
    stops: Sequence[int],
    *window_and_page: int,
) -> list[Piece]:
    """Split a request's token range ``[start, end)`` into window-sized pieces.

    Each piece holds at most ``largest`` tokens and at most one checkpoint from
    ``stops`` so consecutive pieces run in consecutive prefill waves.
    """
    largest = window_and_page[0]
    page_tokens = (
        window_and_page[1]
        if len(window_and_page) >= _MIN_SPLITS_FOR_EARLY_STOP
        else PAGE_TOKENS
    )
    pieces: list[Piece] = []
    ordered_stops = sorted(e for e in stops if start < e < end)
    while start < end:
        stop = min(start + largest, end)
        inside = [e for e in ordered_stops if start < e <= stop]
        if len(inside) >= _MIN_SPLITS_FOR_EARLY_STOP:
            stop = inside[0]
        checkpoint = next((e for e in inside if e <= stop), None)
        pieces.append(Piece(row, start, stop, checkpoint, page_tokens))
        start = stop
    return pieces


def pack_windows(
    pieces: list[Piece],
    windows: tuple[int, ...],
    slots: int,
    capacity: int,
) -> list[tuple[int, list[tuple[Piece, int]]]]:
    """Greedily pack ``pieces`` in order into windows with page-aligned offsets.

    Returns ``(window_size, [(piece, offset)])`` per packed window, choosing the
    smallest compiled window in ``windows`` that fits the last member's extent.
    """
    largest = windows[-1]
    packed: list[tuple[int, list[tuple[Piece, int]]]] = []
    members: list[tuple[Piece, int]] = []
    offset = 0
    prefix = 0

    def close() -> None:
        """Append the current packed window using the smallest fitting window size."""
        extent = members[-1][1] + members[-1][0].length
        packed.append((min(w for w in windows if w >= extent), list(members)))

    for piece in pieces:
        if piece.length > largest:
            msg = (
                f"piece of {piece.length} tokens is longer than "
                f"the {largest}-token window"
            )
            raise ValueError(msg)
        cached = -(-piece.start // piece.page_tokens) * piece.page_tokens
        if members and (
            len(members) == slots
            or offset + piece.length > largest
            or prefix + cached > capacity
        ):
            close()
            members, offset, prefix = [], 0, 0
        members.append((piece, offset))
        offset += piece.rows
        prefix += cached
    if members:
        close()
    return packed


# Expert families follow the first (layers 1..3), eleven regular groups, and
# final layer. Keeping those resident banks separate avoids addressing a 92-layer
# expert operand in every fused MoE call. No padding or device-side copies.
_EXPERT_FAMILIES = ("expert_gu", "expert_gus", "expert_down", "expert_ds")
_EXPERT_BANK_SIZES = (3,) + (8,) * 11 + (1,)


@dataclasses.dataclass
class _ExpertBankCursor:
    """Streaming accumulator state for one MoE expert weight family across banks."""

    expected_part: int = 0
    bank: int = 0
    used: int = 0
    buffer: np.ndarray | None = None


def _bank_expert_weights(
    iterator: Iterator[tuple[str, str, np.ndarray]],
) -> Iterator[tuple[str, str, np.ndarray]]:
    """Stream checkpoint parts into resident per-bank expert weight arrays."""
    state: dict[str, _ExpertBankCursor] = {}
    for group, name, value in iterator:
        family, marker, suffix = name.partition("@")
        if group != "moe" or family not in _EXPERT_FAMILIES:
            yield group, name, value
            continue
        part, parts = map(int, suffix.split("/")) if marker else (0, 1)
        cursor = state.setdefault(family, _ExpertBankCursor())
        if part != cursor.expected_part:
            msg = f"Out-of-order expert checkpoint part: {name}"
            raise ValueError(msg)
        offset = 0
        while offset < value.shape[0]:
            if cursor.bank == len(_EXPERT_BANK_SIZES):
                msg = f"Too many expert layers for {family}"
                raise ValueError(msg)
            capacity = _EXPERT_BANK_SIZES[cursor.bank]
            if cursor.buffer is None:
                cursor.buffer = np.empty(
                    (capacity, *value.shape[1:]),
                    dtype=value.dtype,
                )
            count = min(capacity - cursor.used, value.shape[0] - offset)
            cursor.buffer[cursor.used : cursor.used + count] = value[
                offset : offset + count
            ]
            cursor.used += count
            offset += count
            if cursor.used == capacity:
                yield group, f"{family}#bank{cursor.bank}", cursor.buffer
                cursor.bank += 1
                cursor.used = 0
                cursor.buffer = None
        cursor.expected_part = part + 1
        if part + 1 == parts and (
            cursor.bank != len(_EXPERT_BANK_SIZES) or cursor.used != 0
        ):
            msg = f"Incomplete expert checkpoint: {family}"
            raise ValueError(msg)
    if set(state) != set(_EXPERT_FAMILIES) or any(
        c.bank != len(_EXPERT_BANK_SIZES) or c.used != 0 for c in state.values()
    ):
        msg = "Missing expert checkpoint parts"
        raise ValueError(msg)


@dataclasses.dataclass(frozen=True, slots=True)
class _PackedWindowInputs:
    """Device operands for a single packed prefill window execution."""

    shard: torch.Tensor
    extent: torch.Tensor
    segments: torch.Tensor
    recurrent: torch.Tensor
    conv: torch.Tensor
    pools: list[torch.Tensor]
    ids: torch.Tensor
    local: torch.Tensor
    ring: Callable[..., None] | None = None


class PrefillProgram:
    """The prefill stack's device weights and packed-window custom ops.

    :meth:`run_packed` prefills a window of up to :meth:`window_slots`
    sequences, each continuing a prefix whose pages are spread over the
    ranks' pools (context parallel attention); ``capacity`` is the most
    tokens a window's prefixes may hold in all.
    """

    def __init__(
        self,
        windows: tuple[int, ...],
        capacity: int,
        slots: int,
        page_tokens: int = PAGE_TOKENS,
    ) -> None:
        """Initialize prefill program for ``windows``, ``capacity``, and ``slots``."""
        self.windows = tuple(sorted(int(w) for w in windows))
        self.capacity, self.slots = int(capacity), int(slots)
        self.page_tokens = int(page_tokens)
        self._kernels, self._rank_loader = kernel_modules(
            "prefill_megakernel",
            "rank_loader",
        )
        self.mesh, self.rank = kernel_mesh(AXIS, TP_SIZE)
        loader = self._rank_loader
        self.names: dict[str, tuple[str, ...]] = {
            "io": tuple(loader.PREFILL_IO),
            "common": tuple(loader.PREFILL_COMMON),
            "kda": tuple(loader.PREFILL_KDA),
            "mla": tuple(loader.PREFILL_MLA),
            "moe": tuple(self._kernels.MOE_FAMILIES),
        }
        self.weights: dict[str, dict[str, torch.Tensor]] = {}
        self._expert_banks: list[dict[str, torch.Tensor]] = []
        self._ops: dict[int, dict[str, CustomOp]] = {}
        self._pairs: list[torch.Tensor] = []
        self._state_ops: dict[tuple[str, int], CustomOp] = {}

    def load_weights(self, presharded: Path) -> None:
        """Stream this rank's presharded weights and expert banks onto the TPU."""
        cache = presharded / "prefill-sp-cache"
        raw_iter = self._rank_loader.iter_prefill_rank(presharded, cache, self.rank)
        self._expert_banks = [{} for _ in _EXPERT_BANK_SIZES]
        iterator = _bank_expert_weights(raw_iter)

        def store(group: str, name: str, tensor: torch.Tensor) -> None:
            """Store one streamed weight tensor into its bank or group table."""
            family, marker, index = name.partition("#bank")
            if marker:
                bank = int(index)
                self._expert_banks[bank][family] = tensor
                if bank != 0:
                    return
                name = family
            self.weights.setdefault(group, {})[name] = tensor

        stats = stream_to_device(iterator, store)
        if self._expert_banks and any(
            set(bank) != set(_EXPERT_FAMILIES) for bank in self._expert_banks
        ):
            msg = "K3 prefill: incomplete resident expert banks"
            raise RuntimeError(msg)
        for group, names in self.names.items():
            missing = [n for n in names if n not in self.weights.get(group, {})]
            if missing:
                msg = f"K3 prefill: missing {group} weights {missing}"
                raise RuntimeError(msg)
        materialize(
            self._expert_banks[-1]["expert_ds"]
            if self._expert_banks
            else self.weights["moe"]["expert_ds"]
        )
        if self.rank == 0:
            logger.info(
                "K3 prefill: rank 0 weights resident (%.1f GiB) in %.1f s "
                "from %s; HBM %s",
                stats["gib"],
                stats["wall"],
                presharded,
                hbm(self.rank),
            )
        report_load_stats("prefill", self.rank, stats)

    def flat_weights(self, *groups: str, expert_bank: int = 0) -> list[torch.Tensor]:
        """Flatten the named weight ``groups`` into a positional operand list."""
        experts = self._expert_banks[expert_bank] if self._expert_banks else {}
        return [
            experts[n] if g == "moe" and n in experts else self.weights[g][n]
            for g in groups
            for n in self.names[g]
        ]

    def _unflat(
        self,
        args: Sequence[jax.Array],
        *groups: str,
    ) -> tuple[dict[str, dict[str, jax.Array]], list[jax.Array]]:
        """Reconstruct nested weight dictionaries for ``groups`` from ``args``."""
        out: dict[str, dict[str, jax.Array]] = {}
        index = 0
        for group in groups:
            out[group] = {n: args[index + i] for i, n in enumerate(self.names[group])}
            index += len(self.names[group])
        return out, list(args[index:])

    def _count(self, *groups: str) -> int:
        """Return the total number of weight tensors across ``groups``."""
        return sum(len(self.names[g]) for g in groups)

    def window_slots(self, window: int) -> int:
        """Return the compiled sequence slot count for ``window``."""
        return max(1, min(self.slots, window // self.page_tokens))

    def _zeros_sharded(
        self,
        shape: tuple[int, ...],
        dtype: jax.typing.DTypeLike,
    ) -> jax.Array:
        """Allocate a rank-leading zero state array constrained to ``AXIS``."""
        state_sharding = jax.sharding.NamedSharding(self.mesh, P(AXIS))
        return jax.lax.with_sharding_constraint(
            jnp.zeros((TP_SIZE, *shape), dtype),
            state_sharding,
        )

    def _split_cp_prefix(
        self,
        args: Sequence[jax.Array],
        pools: int,
    ) -> tuple[object, Sequence[jax.Array]]:
        """Split ``args`` into a ``CpPrefix`` arena over ``pools`` and weights."""
        return (
            self._kernels.CpPrefix(
                tuple(args[:pools]),
                args[pools],
                args[pools + 1],
            ),
            args[pools + 2 :],
        )

    def _register_first_op(self, window: int, count: int, name: str) -> CustomOp:
        """Build and register the embedding + layers 0..3 custom op for ``window``."""
        sharded, replicated = P(AXIS), P()
        embedding = self._kernels.make_embedding_sp(self.mesh, seq_len=window)
        first = self._kernels.make_prefill_stack_sp_indexed(
            self.mesh,
            seq_len=window,
            depth=_FIRST_STACK_DEPTH,
            mla_flash_tile=_MLA_FLASH_TILE,
            unroll=True,
            segments=count,
            prefix_capacity=self.capacity,
            tap_layer=_FIRST_TAP_LAYER,
        )
        weight_groups = ("common", "kda", "mla", "moe")

        def run_first(
            tokens: jax.Array,
            length: jax.Array,
            segments: jax.Array,
            rec: jax.Array,
            *hist_and_weights: jax.Array,
        ) -> tuple[jax.Array, ...]:
            """Embed ``tokens`` and run the first four layers with layer-3 tap."""
            hist = hist_and_weights[0]
            arena, weight_args = self._split_cp_prefix(hist_and_weights[1:], 1)
            w, _ = self._unflat(weight_args, "io", *weight_groups)
            hidden = embedding(tokens, w["io"]["embedding"], length)
            latents = self._zeros_sharded(
                (MLA_LAYERS, window, _MLA_LATENT_DIM),
                jnp.bfloat16,
            )
            rec_at = self._zeros_sharded(
                (KDA_LAYERS, count, _KDA_HEADS, _KDA_HEAD_DIM, _KDA_HEAD_DIM),
                jnp.float32,
            )
            hist_at = self._zeros_sharded(
                (
                    KDA_LAYERS,
                    count,
                    _KDA_CONV_TAPS,
                    _KDA_HEADS,
                    _KDA_HEADS,
                    _KDA_HEAD_DIM,
                ),
                jnp.bfloat16,
            )
            prefix, blocks, kda_state, mla_state, tap = first(
                hidden,
                w["common"],
                w["kda"],
                w["mla"],
                w["moe"],
                (rec, hist, rec_at, hist_at),
                (latents,),
                length,
                segment_desc=segments,
                arena=arena,
            )
            return (prefix, blocks, *kda_state, *mla_state, tap)

        state_specs = (sharded,) * _STATE_SPEC_COUNT
        return register_global_op(
            self._op_name(f"{name}_first", window),
            self.mesh,
            run_first,
            OpRegistrationSpec(
                in_specs=(sharded,)
                + (replicated,) * 2
                + (sharded,) * 2
                + (sharded,) * 3
                + (sharded,) * self._count("io", *weight_groups),
                out_specs=(*state_specs, sharded),
                donate_argnums=(3, 4),
                export_key=self._export_key(f"{name}_first", window),
            ),
        )

    def _register_group_op(self, window: int, count: int, name: str) -> CustomOp:
        """Build and register the 8-layer regular group custom op for ``window``."""
        sharded, replicated = P(AXIS), P()
        regular = self._kernels.make_regular_group8_sp_indexed(
            self.mesh,
            seq_len=window,
            mla_flash_tile=_MLA_FLASH_TILE,
            segments=count,
            prefix_capacity=self.capacity,
        )
        weight_groups = ("common", "kda", "mla", "moe")

        def run_group(*state_and_weights: jax.Array) -> tuple[jax.Array, ...]:
            """Advance packed residual streams and states across an 8-layer group."""
            (
                prefix,
                blocks,
                rec,
                hist,
                rec_at,
                hist_at,
                latents,
                length,
                pair,
                segments,
            ) = state_and_weights[:_GROUP_RUN_HEADER_ARGS]
            arena, weight_args = self._split_cp_prefix(
                state_and_weights[_GROUP_RUN_HEADER_ARGS:],
                2,
            )
            w, _ = self._unflat(weight_args, *weight_groups)
            prefix, blocks, kda_state, mla_state, middle, tap = regular(
                prefix,
                blocks,
                w["common"],
                w["kda"],
                w["mla"],
                w["moe"],
                (rec, hist, rec_at, hist_at),
                (latents,),
                length,
                pair,
                segment_desc=segments,
                arena=arena,
            )
            return (prefix, blocks, *kda_state, *mla_state, middle, tap)

        state_specs = (sharded,) * _STATE_SPEC_COUNT
        return register_global_op(
            self._op_name(f"{name}_group8", window),
            self.mesh,
            run_group,
            OpRegistrationSpec(
                in_specs=state_specs
                + (replicated,) * 3
                + (sharded,) * 4
                + (sharded,) * self._count(*weight_groups),
                out_specs=state_specs + (sharded,) * 2,
                donate_argnums=tuple(range(_STATE_SPEC_COUNT)),
                export_key=self._export_key(f"{name}_group", window),
            ),
        )

    def _register_final_op(self, window: int, count: int, name: str) -> CustomOp:
        """Build and register the final MLA layer 92 + logits op for ``window``."""
        sharded, replicated = P(AXIS), P()
        final = self._kernels.make_final_mla_sp_indexed(
            self.mesh,
            seq_len=window,
            mla_flash_tile=_MLA_FLASH_TILE,
            segments=count,
            prefix_capacity=self.capacity,
        )
        logits_op = self._kernels.make_prefill_logits_sp(
            self.mesh,
            seq_len=window,
            layers=NUM_LAYERS,
            segments=count,
        )

        def run_final(
            prefix: jax.Array,
            blocks: jax.Array,
            latents: jax.Array,
            length: jax.Array,
            *segments_and_weights: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            """Compute final MLA layer 92 and all-gather vocabulary logits."""
            segments = segments_and_weights[0]
            arena, weight_args = self._split_cp_prefix(segments_and_weights[1:], 1)
            w, _ = self._unflat(weight_args, "io", "common", "mla", "moe")
            prefix, blocks, (latents,) = final(
                prefix,
                blocks,
                w["common"],
                w["mla"],
                w["moe"],
                (latents,),
                length,
                segment_desc=segments,
                arena=arena,
            )
            logits = logits_op(
                prefix,
                blocks,
                w["io"]["output_res"],
                w["io"]["final_norm"],
                w["io"]["lm_head"],
                segments,
            )
            logits = jax.shard_map(
                lambda part: jax.lax.all_gather(part[0], AXIS, axis=1, tiled=True),
                mesh=self.mesh,
                in_specs=(sharded,),
                out_specs=replicated,
                check_vma=False,
            )(logits)
            return logits, latents

        return register_global_op(
            self._op_name(f"{name}_final", window),
            self.mesh,
            run_final,
            OpRegistrationSpec(
                in_specs=(sharded,) * 3
                + (replicated,) * 2
                + (sharded,) * 3
                + (sharded,) * self._count("io", "common", "mla", "moe"),
                out_specs=(replicated, sharded),
                donate_argnums=(0, 1, 2),
                export_key=self._export_key(f"{name}_final", window),
            ),
        )

    def build_window_ops(self, window: int) -> dict[str, CustomOp]:
        """Build or return cached ``first``, ``group``, and ``final`` window ops."""
        if window in self._ops:
            return self._ops[window]
        if window not in self.windows:
            msg = f"K3 prefill: window {window} is not among {self.windows}"
            raise ValueError(msg)
        count = self.window_slots(window)
        name = f"k3_prefill_packed{count}c"
        ops: dict[str, CustomOp] = {
            "first": self._register_first_op(window, count, name),
            "group": self._register_group_op(window, count, name),
            "final": self._register_final_op(window, count, name),
        }
        if not self._pairs:
            self._pairs = [
                torch.tensor([pair], dtype=torch.int32).to("tpu")
                for pair in range(GROUP_PAIRS)
            ]
        self._ops[window] = ops
        return ops

    def _op_name(self, base: str, window: int) -> str:
        """Format the custom-call op name for ``base`` at ``window`` and capacity."""
        return f"pallas::{base}_w{window}_c{self.capacity}"

    def _export_key(self, op: str, window: int) -> str:
        """Return the export-cache key baking in the static configuration of ``op``."""
        return repr(
            (
                "prefill",
                "moe-bank8" if self._expert_banks else "moe-global",
                op,
                window,
                self.capacity,
                sorted((g, tuple(n)) for g, n in self.names.items()),
            )
        )

    def run_packed(
        self,
        inputs: _PackedWindowInputs,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Prefill one packed window and return logits, MLA latents, and KDA states."""
        ops = self.build_window_ops(int(inputs.shard.shape[-1]) * TP_SIZE)
        tail = (inputs.ids, inputs.local)
        *state, tap3 = ops["first"](
            inputs.shard,
            inputs.extent,
            inputs.segments,
            inputs.recurrent,
            inputs.conv,
            inputs.pools[0],
            *tail,
            *self.flat_weights("io", "common", "kda", "mla", "moe"),
        )
        middle = tap90 = inputs.shard
        for p, pair in enumerate(self._pairs):
            *state, middle, tap90 = ops["group"](
                *state,
                inputs.extent,
                pair,
                inputs.segments,
                inputs.pools[2 * p + 1],
                inputs.pools[2 * p + 2],
                *tail,
                *self.flat_weights("common", "kda", "mla", "moe", expert_bank=p + 1),
            )
        prefix, blocks, rec, hist, rec_at, hist_at, latents = state
        if inputs.ring is not None:
            inputs.ring(blocks, prefix, middle, tap3, tap90)
        logits, latents = ops["final"](
            prefix,
            blocks,
            latents,
            inputs.extent,
            inputs.segments,
            inputs.pools[MLA_LAYERS - 1],
            *tail,
            *self.flat_weights(
                "io",
                "common",
                "mla",
                "moe",
                expert_bank=_FINAL_EXPERT_BANK,
            ),
        )
        return logits, latents, rec, hist, rec_at, hist_at

    def zero_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Allocate one sequence's zero KDA recurrent and conv states on TPU."""
        return (
            torch.zeros(
                (1, KDA_LAYERS, _KDA_HEADS, _KDA_HEAD_DIM, _KDA_HEAD_DIM),
                dtype=torch.float32,
                device="tpu",
            ),
            torch.zeros(
                (
                    1,
                    KDA_LAYERS,
                    _KDA_CONV_TAPS,
                    _KDA_HEADS,
                    _KDA_HEADS,
                    _KDA_HEAD_DIM,
                ),
                dtype=torch.bfloat16,
                device="tpu",
            ),
        )

    def state_op(self, kind: str, count: int) -> CustomOp:
        """Build or return the cached KDA state ``stack`` or ``split`` custom op."""
        key = (kind, count)
        if key not in self._state_ops:
            if kind == "stack":

                def stack_body(*states: jax.Array) -> tuple[jax.Array, jax.Array]:
                    """Stack ``count`` per-sequence recurrent and conv states."""
                    return (
                        jnp.stack(states[:count], axis=2),
                        jnp.stack(states[count:], axis=2),
                    )

                body_fn: Callable[..., tuple[jax.Array, ...]] = stack_body
                in_specs = (P(AXIS),) * (2 * count)
                out_specs = (P(AXIS),) * 2
            else:

                def split_body(
                    recurrent: jax.Array,
                    conv: jax.Array,
                ) -> tuple[jax.Array, ...]:
                    """Split packed recurrent and conv tensors into ``count`` pairs."""
                    return (
                        *(recurrent[:, :, s] for s in range(count)),
                        *(conv[:, :, s] for s in range(count)),
                    )

                body_fn = split_body
                in_specs = (P(AXIS),) * 2
                out_specs = (P(AXIS),) * (2 * count)

            self._state_ops[key] = register_op(
                f"pallas::k3_prefill_{kind}_states_{count}",
                self.mesh,
                body_fn,
                OpRegistrationSpec(
                    in_specs=in_specs,
                    out_specs=out_specs,
                    export_key=repr(("states", kind, count)),
                ),
            )
        return self._state_ops[key]

    def sp_order(self, window: int) -> np.ndarray:
        """Return ``window`` token indices in rank-major sequence-parallel order."""
        order: np.ndarray = np.asarray(self._kernels.sp_order(window), dtype=np.int32)
        return order

    def stack_states(
        self,
        states: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack per-sequence KDA state pairs into ``[1, 69, n, ...]`` tensors."""
        out = self.state_op("stack", len(states))(
            *(r for r, _ in states),
            *(c for _, c in states),
        )
        return out[0], out[1]

    def split_states(
        self,
        recurrent: torch.Tensor,
        conv: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Split packed ``[1, 69, n, ...]`` KDA states into per-sequence pairs."""
        count = int(recurrent.shape[2])
        out = self.state_op("split", count)(recurrent, conv)
        return list(zip(out[:count], out[count:], strict=True))


@dataclasses.dataclass(frozen=True, slots=True)
class _BoundPoolIndices:
    """Layer-to-pool index mappings and TP-rank KDA head permutation."""

    mla_raw: Sequence[int]
    kda_raw: Sequence[int]
    state_perm: Sequence[tuple[int, int]] | None


@dataclasses.dataclass(frozen=True, slots=True)
class _PackedPoolWrite:
    """Window outputs and target block tables for a packed pool write."""

    latents: torch.Tensor
    recurrent: torch.Tensor
    conv: torch.Tensor
    recurrent_at: torch.Tensor
    conv_at: torch.Tensor
    table: torch.Tensor
    kda_blocks: torch.Tensor
    blocks_at: torch.Tensor
    rings: tuple[list[RingPair], list[RingPair]] | None = None


class PoolWriter:
    """Writes packed windows' latent pages and KDA states into vLLM's unified pools.

    ``raw_pools`` are the distinct pool tensors (``layout.cache_layout()``);
    ``mla_raw[m]`` and ``kda_raw[k]`` index them for MLA layer ``m`` and KDA
    layer ``k`` (``pools.bind_raw_pools``). KDA heads are written in
    vLLM TP-rank order (``state_perm``, from the kernel's mesh order) and
    read back in mesh order.
    """

    def __init__(
        self,
        mesh: Mesh,
        rank: int,
        raw_pools: list[torch.Tensor],
        indices: _BoundPoolIndices,
    ) -> None:
        """Initialize pool writer over ``raw_pools`` with ``indices`` mappings."""
        self.mesh, self.rank = mesh, rank
        self.raw_pools = raw_pools
        self.layout = cache_layout(str(raw_pools[0].dtype))
        self.mla_raw = tuple(indices.mla_raw)
        self._kda_raw = tuple(indices.kda_raw)
        self._perm = (
            None
            if indices.state_perm is None
            else tuple((int(a), int(b)) for a, b in indices.state_perm)
        )
        self._ops: dict[tuple[object, ...], CustomOp] = {}

    def mla_pools(self) -> list[torch.Tensor]:
        """Return the 24 MLA layers' pool tensors in layer order ``0..23``."""
        return [self.raw_pools[self.mla_raw[m]] for m in range(MLA_LAYERS)]

    def _adopt(
        self,
        new: torch.Tensor | Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Update ``raw_pools`` storage in place from donated op outputs ``new``."""
        updated = [new] if isinstance(new, torch.Tensor) else list(new)
        attr_name = "set_"
        for old, fresh in zip(self.raw_pools, updated, strict=True):
            if fresh is not old:
                getattr(old, attr_name)(fresh)
        return self.raw_pools

    def copy_blocks(self, pairs: Sequence[tuple[int, int]]) -> list[torch.Tensor]:
        """Copy source blocks to destination blocks across all raw pools."""
        for first in range(0, len(pairs), COPY_PAIRS):
            chunk = list(pairs[first : first + COPY_PAIRS])
            chunk += [(0, 0)] * (COPY_PAIRS - len(chunk))
            source = torch.tensor([a for a, _ in chunk], dtype=torch.int32).to("tpu")
            destination = torch.tensor([b for _, b in chunk], dtype=torch.int32).to(
                "tpu"
            )
            self._adopt(self.copy_op()(*self.raw_pools, source, destination))
        return self.raw_pools

    def copy_op(self) -> CustomOp:
        """Build or return the cached copy-on-write block-copy custom op."""
        key = ("copy",)
        if key not in self._ops:
            num_raw = len(self.raw_pools)

            def body(*args: jax.Array) -> tuple[jax.Array, ...]:
                """Scatter ``source`` blocks into ``destination`` across raw pools."""
                raws, source, destination = (
                    args[:num_raw],
                    args[num_raw],
                    args[num_raw + 1],
                )
                return tuple(raw.at[destination].set(raw[source]) for raw in raws)

            self._ops[key] = register_op(
                f"pallas::k3_prefill_copy_blocks_r{num_raw}",
                self.mesh,
                body,
                OpRegistrationSpec(
                    in_specs=(P(AXIS),) * num_raw + (P(), P()),
                    out_specs=(P(AXIS),) * num_raw,
                    donate_argnums=tuple(range(num_raw)),
                    export_key=repr(("copy_blocks", num_raw, COPY_PAIRS)),
                ),
            )
        return self._ops[key]

    def _write_kda(
        self,
        raws: list[jax.Array],
        states: tuple[jax.Array, jax.Array],
        kda_blocks: jax.Array,
        count: int,
    ) -> list[jax.Array]:
        """Scatter ``count`` sequences' KDA states into ``kda_blocks`` across layers."""
        recurrent, conv = states
        blocks = kda_blocks.reshape(count, KDA_LAYERS)
        for k in range(KDA_LAYERS):
            index, ids = self._kda_raw[k], blocks[:, k]
            ssm_shape = jax.eval_shape(
                lambda raw=raws[index], block_ids=ids: pool_adapters.gather_region(
                    raw,
                    block_ids,
                    tok0=0,
                    ntok=SSM_ROWS,
                    out_dtype=jnp.float32,
                    out_lanes=_KDA_HEAD_DIM,
                )
            ).shape
            ssm = recurrent[k].reshape(count, -1, _KDA_HEAD_DIM)
            ssm = jnp.pad(ssm, ((0, 0), (0, ssm_shape[1] - ssm.shape[1]), (0, 0)))
            raws[index] = pool_adapters.scatter_region(
                raws[index],
                ssm,
                ids,
                tok0=0,
                ntok=SSM_ROWS,
            )
            conv_shape = jax.eval_shape(
                lambda raw=raws[index], block_ids=ids: pool_adapters.gather_region(
                    raw,
                    block_ids,
                    tok0=SSM_ROWS,
                    ntok=CONV_ROWS,
                    out_dtype=jnp.bfloat16,
                )
            ).shape
            taps = conv[k].reshape(count, -1).astype(jnp.bfloat16)
            taps = jnp.pad(
                taps,
                ((0, 0), (0, int(np.prod(conv_shape[1:])) - taps.shape[1])),
            )
            raws[index] = pool_adapters.scatter_region(
                raws[index],
                taps.reshape(conv_shape),
                ids,
                tok0=SSM_ROWS,
                ntok=CONV_ROWS,
            )
        return raws

    def write_packed(self, batch: _PackedPoolWrite) -> list[torch.Tensor]:
        """Write a packed window's MLA latent pages, KDA states, and DSpark rings."""
        window = int(batch.latents.shape[2])
        slots = int(batch.recurrent.shape[2])
        count = int(batch.kda_blocks.shape[0]) // KDA_LAYERS
        self._adopt(
            self.write_op(window, slots, count)(
                batch.latents,
                batch.recurrent,
                batch.conv,
                batch.recurrent_at,
                batch.conv_at,
                *self.raw_pools,
                batch.table,
                batch.kda_blocks,
                batch.blocks_at,
            )
        )
        if batch.rings is not None:
            after, at = batch.rings
            self._write_rings(at[:count], batch.blocks_at)
            self._write_rings(after[:count], batch.kda_blocks)
        return self.raw_pools

    def write_op(self, window: int, slots: int, count: int) -> CustomOp:
        """Build or return cached packed-window pool write op for ``count`` slots."""
        key = ("write", window, slots, count)
        if key not in self._ops:
            num_raw, mla_raw, perm, pages = (
                len(self.raw_pools),
                self.mla_raw,
                self._perm,
                window // self.layout.tokens_per_block,
            )

            def body(
                latents: jax.Array,
                recurrent: jax.Array,
                conv: jax.Array,
                *states_and_pools: jax.Array,
            ) -> tuple[jax.Array, ...]:
                """Write MLA latent pages and checkpoint/final KDA states into pools."""
                recurrent_at, conv_at = states_and_pools[0], states_and_pools[1]
                raws = list(states_and_pools[2 : 2 + num_raw])
                table, kda_blocks, blocks_at = states_and_pools[2 + num_raw :]
                table_row = table[0]
                latents_row = latents[0]
                rec_live, conv_live, rec_at_live, conv_at_live = (
                    x[0, :, :count] for x in (recurrent, conv, recurrent_at, conv_at)
                )
                if perm is not None:
                    rec_live, conv_live, rec_at_live, conv_at_live = (
                        jax.lax.ppermute(x, AXIS, perm)
                        for x in (rec_live, conv_live, rec_at_live, conv_at_live)
                    )
                for m in range(MLA_LAYERS):
                    index = mla_raw[m]
                    rows = (
                        jnp.pad(latents_row[m], ((0, 0), (0, _MLA_PAD_COLS)))
                        .astype(raws[index].dtype)
                        .reshape(pages, *self.layout.block_shape)
                    )

                    def write_page(
                        page_idx: int,
                        pool: jax.Array,
                        *,
                        layer_rows: jax.Array = rows,
                    ) -> jax.Array:
                        """Write ``page_idx`` of ``layer_rows`` into ``pool``."""
                        return jax.lax.dynamic_update_index_in_dim(
                            pool,
                            jax.lax.dynamic_index_in_dim(
                                layer_rows,
                                page_idx,
                                keepdims=False,
                            ),
                            table_row[page_idx],
                            axis=0,
                        )

                    raws[index] = jax.lax.fori_loop(0, pages, write_page, raws[index])
                raws = self._write_kda(
                    raws,
                    (rec_at_live, conv_at_live),
                    blocks_at,
                    count,
                )
                raws = self._write_kda(
                    raws,
                    (rec_live, conv_live),
                    kda_blocks,
                    count,
                )
                return tuple(raws)

            op_name = (
                f"pallas::k3_prefill_write_pool_packed{slots}_{count}"
                f"_r{num_raw}_w{window}_ranked"
            )
            self._ops[key] = register_op(
                op_name,
                self.mesh,
                body,
                OpRegistrationSpec(
                    in_specs=(P(AXIS),) * (6 + num_raw) + (P(),) * 2,
                    out_specs=(P(AXIS),) * num_raw,
                    donate_argnums=tuple(range(5, 5 + num_raw)),
                    export_key=repr(
                        (
                            "write_pool_packed",
                            self.layout,
                            window,
                            slots,
                            count,
                            self.mla_raw,
                            self._kda_raw,
                            self._perm,
                        )
                    ),
                ),
            )
        return self._ops[key]

    def _write_rings(
        self,
        rings: Sequence[RingPair],
        blocks: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Write each sequence's DSpark ring ``(cache, meta)`` into ``blocks``."""
        return self._adopt(
            self.rings_op(len(rings))(
                *self.raw_pools,
                blocks,
                *(cache for cache, _ in rings),
                *(meta for _, meta in rings),
            )
        )

    def rings_op(self, count: int) -> CustomOp:
        """Build or return cached DSpark ring-write custom op for ``count`` slots."""
        key = ("write_rings", count)
        if key not in self._ops:
            num_raw, perm = len(self.raw_pools), self._perm
            (dspark,) = kernel_modules("dspark")

            def body(*args: jax.Array) -> tuple[jax.Array, ...]:
                """Pack and scatter ``count`` sequences' DSpark rings into raw pools."""
                raws = list(args[:num_raw])
                blocks = args[num_raw]
                caches = args[num_raw + 1 : num_raw + 1 + count]
                metas = args[num_raw + 1 + count :]
                stacked_caches = jnp.stack([cache[0] for cache in caches])
                if perm is not None:
                    stacked_caches = jax.lax.ppermute(stacked_caches, AXIS, perm)
                packed = dspark.pack_ring(stacked_caches, jnp.stack(metas), KDA_LAYERS)
                return tuple(
                    write_draft_rings(
                        raws,
                        self._kda_raw,
                        packed,
                        blocks.reshape(count, KDA_LAYERS),
                    )
                )

            self._ops[key] = register_op(
                f"pallas::k3_prefill_write_rings{count}_r{num_raw}_ranked",
                self.mesh,
                body,
                OpRegistrationSpec(
                    in_specs=(P(AXIS),) * num_raw
                    + (P(),)
                    + (P(AXIS),) * count
                    + (P(),) * count,
                    out_specs=(P(AXIS),) * num_raw,
                    donate_argnums=tuple(range(num_raw)),
                    export_key=repr(
                        ("write_rings", self.layout, count, self._kda_raw, self._perm)
                    ),
                ),
            )
        return self._ops[key]

    def read_rings(
        self,
        draft_config: object,
        kda_blocks: torch.Tensor,
        lengths: Sequence[int],
    ) -> list[RingPair]:
        """Read DSpark rings from ``kda_blocks`` for ``lengths`` in mesh order."""
        sequences = len(lengths)
        out = self.read_rings_op(draft_config, sequences)(
            *self.raw_pools,
            kda_blocks,
            torch.tensor(list(lengths), dtype=torch.int32).to("tpu"),
        )
        return list(zip(out[:sequences], out[sequences:], strict=True))

    def read_rings_op(self, draft_config: object, sequences: int) -> CustomOp:
        """Build or return cached DSpark ring-read custom op for ``sequences``."""
        key = ("read_rings", sequences)
        if key not in self._ops:
            (dspark,) = kernel_modules("dspark")
            num_raw, kda_raw, cfg = len(self.raw_pools), self._kda_raw, draft_config
            perm = (
                None
                if self._perm is None
                else tuple((dst, src) for src, dst in self._perm)
            )

            def body(*args: jax.Array) -> tuple[jax.Array, ...]:
                """Gather and unpack ``sequences`` DSpark rings from raw pools."""
                raws, blocks, lengths = (
                    args[:num_raw],
                    args[num_raw],
                    args[num_raw + 1],
                )
                packed = read_draft_rings(
                    raws,
                    kda_raw,
                    blocks.reshape(sequences, KDA_LAYERS),
                    dspark.RING_ROW_LANES,
                )
                out = [
                    dspark.unpack_ring(cfg, packed[s], lengths[s])
                    for s in range(sequences)
                ]
                caches = jnp.stack([cache for cache, _ in out])
                if perm is not None:
                    caches = jax.lax.ppermute(caches, AXIS, perm)
                return (
                    *(caches[s][None] for s in range(sequences)),
                    *(meta for _, meta in out),
                )

            cfg_tuple = (
                dataclasses.astuple(cfg)
                if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type)
                else repr(cfg)
            )
            self._ops[key] = register_op(
                f"pallas::k3_prefill_read_rings_r{num_raw}_s{sequences}",
                self.mesh,
                body,
                OpRegistrationSpec(
                    in_specs=(P(AXIS),) * num_raw + (P(), P()),
                    out_specs=(P(AXIS),) * sequences + (P(),) * sequences,
                    export_key=repr(
                        ("read_rings", sequences, kda_raw, perm, cfg_tuple)
                    ),
                ),
            )
        return self._ops[key]

    def read_states(
        self,
        kda_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read ``kda_blocks`` into ``[1, 69, n, ...]`` recurrent and conv states."""
        sequences = int(kda_blocks.shape[0]) // KDA_LAYERS
        out = self.read_states_op(sequences)(*self.raw_pools, kda_blocks)
        return out[0], out[1]

    def read_states_op(self, sequences: int) -> CustomOp:
        """Build or return cached KDA state-read custom op for ``sequences``."""
        key = ("read", sequences)
        if key not in self._ops:
            num_raw, kda_raw = len(self.raw_pools), self._kda_raw
            perm = (
                None
                if self._perm is None
                else tuple((dst, src) for src, dst in self._perm)
            )
            ssm_elements = _KDA_HEADS * _KDA_HEAD_DIM
            conv_elements = _KDA_CONV_TAPS * _KDA_HEADS * _KDA_HEADS * _KDA_HEAD_DIM

            def body(*args: jax.Array) -> tuple[jax.Array, jax.Array]:
                """Gather recurrent and conv states for ``sequences`` from raw pools."""
                raws, blocks = (
                    args[:num_raw],
                    args[num_raw].reshape(sequences, KDA_LAYERS),
                )
                recurrent_list: list[jax.Array] = []
                conv_list: list[jax.Array] = []
                for k in range(KDA_LAYERS):
                    raw, ids = raws[kda_raw[k]], blocks[:, k]
                    ssm = pool_adapters.gather_region(
                        raw,
                        ids,
                        tok0=0,
                        ntok=SSM_ROWS,
                        out_dtype=jnp.float32,
                        out_lanes=_KDA_HEAD_DIM,
                    )
                    recurrent_list.append(
                        ssm[:, :ssm_elements].reshape(
                            -1,
                            _KDA_HEADS,
                            _KDA_HEAD_DIM,
                            _KDA_HEAD_DIM,
                        )
                    )
                    taps = pool_adapters.gather_region(
                        raw,
                        ids,
                        tok0=SSM_ROWS,
                        ntok=CONV_ROWS,
                        out_dtype=jnp.bfloat16,
                    )
                    conv_list.append(
                        taps.reshape(sequences, -1)[:, :conv_elements].reshape(
                            -1,
                            _KDA_CONV_TAPS,
                            _KDA_HEADS,
                            _KDA_HEADS,
                            _KDA_HEAD_DIM,
                        )
                    )
                recurrent = jnp.stack(recurrent_list)
                conv = jnp.stack(conv_list)
                if perm is not None:
                    recurrent = jax.lax.ppermute(recurrent, AXIS, perm)
                    conv = jax.lax.ppermute(conv, AXIS, perm)
                return recurrent[None], conv[None]

            self._ops[key] = register_op(
                f"pallas::k3_prefill_read_states_r{num_raw}_s{sequences}",
                self.mesh,
                body,
                OpRegistrationSpec(
                    in_specs=(P(AXIS),) * num_raw + (P(),),
                    out_specs=(P(AXIS),) * 2,
                    export_key=repr(("read_states", sequences, self._kda_raw, perm)),
                ),
            )
        return self._ops[key]


@dataclasses.dataclass
class _Request:
    """In-flight prefill request state retained across scheduler steps."""

    req_id: str
    first_cached: int
    start: int
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    skipped: list[tuple[int, torch.Tensor]] = dataclasses.field(default_factory=list)
    ring: RingPair | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _ScheduledWork:
    """Per-request piece plan and checkpoint map scheduled in the current step."""

    row_index: int
    state: _Request
    pieces: list[Piece]
    stops: dict[int, torch.Tensor]
    finishes: bool


@dataclasses.dataclass(frozen=True, slots=True)
class _WindowPackPlan:
    """Prepared token array, segment descriptors, and state blocks for a window."""

    tokens: np.ndarray
    descriptor: list[int]
    states: list[tuple[torch.Tensor, torch.Tensor]]
    finals: list[torch.Tensor]
    ats: list[torch.Tensor]
    unused: int


class KimiPrefiller:
    """Sequence-parallel prefill program (:class:`MegakernelProgram`) over pools."""

    def __init__(
        self,
        config: MegakernelEngineConfig,
        presharded: Path,
        draft_checkpoint: Path | None = None,
        *,
        tp_rank: int,
    ) -> None:
        """Initialize the Kimi-K3 prefiller for ``config`` and ``tp_rank``."""
        self.config = config
        self.page_tokens = config.block_size
        self.presharded = Path(presharded)
        self.cp_rank = tp_rank % config.decode_context_parallel_size
        budget = config.max_num_batched_tokens
        windows = prefill_windows(budget, config.max_model_len)
        capacity = max(
            windows[-1],
            -(-config.max_model_len // _PREFIX_ALIGN_TOKENS) * _PREFIX_ALIGN_TOKENS,
        )
        self.cp_capacity = (
            -(-(capacity // DCP_SIZE + SLOTS * self.page_tokens) // _CP_ALIGN_TOKENS)
            * _CP_ALIGN_TOKENS
        )
        self.slots = min(config.max_num_seqs, SLOTS)
        self.program = PrefillProgram(windows, capacity, self.slots, self.page_tokens)
        self.rank = self.program.rank
        self._requests: dict[str, _Request] = {}
        self._zero_pair: tuple[torch.Tensor, torch.Tensor] | None = None
        self._writer: PoolWriter | None = None
        perm = tp_rank_permutation(tp_rank, self.rank)
        self._state_perm = None if perm is None else [(dst, src) for src, dst in perm]
        self._mla_group = 0
        self._kda_groups: list[tuple[int, int]] = []
        self._zero_logits: torch.Tensor | None = None
        self._sampler_ops: dict[bool, CustomOp] = {}
        self._loaded = False
        self.ring: PrefillRing | None = None
        self.draft_latents: PrefillDraftLatents | None = None
        if draft_checkpoint is not None:
            (mla,) = kernel_modules("dspark_mla")
            if mla.is_mla_draft(Path(draft_checkpoint)):
                self.draft_latents = PrefillDraftLatents(
                    self.program.mesh,
                    self.rank,
                    AXIS,
                    Path(draft_checkpoint),
                    self.presharded,
                    config.max_model_len,
                )
            else:
                self.ring = PrefillRing(
                    self.program.mesh,
                    self.rank,
                    AXIS,
                    draft_checkpoint,
                    self.presharded,
                )

    def load_weights(self) -> None:
        """Load target and draft prefill weights onto the TPU if not yet resident."""
        if self._loaded:
            return
        started = time.perf_counter()
        self.program.load_weights(self.presharded)
        if self.ring is not None:
            self.ring.load_weights()
        if self.draft_latents is not None:
            self.draft_latents.load_weights()
        self._loaded = True
        if self.rank == 0:
            logger.info(
                "K3 prefill: weights resident in %.1f s",
                time.perf_counter() - started,
            )

    def bind_kv_caches(
        self,
        caches: Mapping[str, torch.Tensor],
        groups: Sequence[KVGroup],
    ) -> None:
        """Bind vLLM's unified KV pool tensors and record MLA/KDA group indices."""
        self.load_weights()
        raw_pools, mla_raw, kda_raw, mla_names, kda_names = bind_raw_pools(
            caches,
            self.config.kv_cache_dtype,
        )
        self._writer = PoolWriter(
            self.program.mesh,
            self.rank,
            raw_pools,
            _BoundPoolIndices(mla_raw, kda_raw, self._state_perm),
        )
        if self.draft_latents is not None:
            self.draft_latents.bind(draft_raw_pools(caches))
        of = {name: g for g, group in enumerate(groups) for name in group.layer_names}
        mla_groups = {of[n] for n in mla_names if n in of}
        if len(mla_groups) != 1:
            msg = (
                "K3 prefill: the MLA layers are in "
                f"KV cache groups {sorted(mla_groups)}"
            )
            raise RuntimeError(msg)
        self._mla_group = mla_groups.pop()
        missing = [n for n in kda_names if n not in of]
        if missing:
            msg = f"K3 prefill: KDA layers {missing[:3]}... are in no KV cache group"
            raise RuntimeError(msg)
        self._kda_groups = [(of[n], int(groups[of[n]].block_size)) for n in kda_names]

    def _zeros(self) -> torch.Tensor:
        """Return a cached ``[1, VOCAB]`` float32 zero logit row on the TPU."""
        if self._zero_logits is None:
            self._zero_logits = torch.zeros((1, VOCAB), dtype=torch.float32).to("tpu")
        return self._zero_logits

    def _collect_step_work(
        self,
        rows: Sequence[RequestInput],
        largest: int,
    ) -> tuple[list[_ScheduledWork], list[tuple[int, _Request]]]:
        """Classify scheduled requests into runnable piece plans and cache-hit reads."""
        work: list[_ScheduledWork] = []
        hits: list[tuple[int, _Request]] = []
        for i, row in enumerate(rows):
            if row.scheduled <= 0:
                continue
            req, total, cached, prompt = (
                row.req_id,
                row.end,
                row.computed,
                row.num_prompt_tokens,
            )
            state = self._requests.get(req)
            expected = (
                None
                if state is None
                else (state.skipped[-1][0] if state.skipped else state.start)
            )
            if state is None or cached != expected:
                state = _Request(
                    req,
                    cached,
                    cached if cached % self.page_tokens == 0 else 0,
                )
                self._requests[req] = state
                if state.start:
                    hits.append((i, state))
            if skips_step(
                cached,
                total,
                prompt,
                state.start,
                largest,
                self.page_tokens * DCP_SIZE,
            ):
                state.skipped.append(
                    (
                        total,
                        self._blocks(
                            kda_blocks_at_length(
                                row.block_tables,
                                self._kda_groups,
                                total,
                            )
                        ),
                    )
                )
                continue
            stops = dict(state.skipped)
            state.skipped = []
            work.append(
                _ScheduledWork(
                    row_index=i,
                    state=state,
                    pieces=plan_pieces(
                        i,
                        state.start,
                        total,
                        list(stops),
                        largest,
                        self.page_tokens,
                    ),
                    stops=stops,
                    finishes=total >= prompt,
                )
            )
        return work, hits

    def _execute_waves(
        self,
        rows: Sequence[RequestInput],
        work: Sequence[_ScheduledWork],
    ) -> dict[int, torch.Tensor]:
        """Pack and run prefill window waves across ``work`` and collect logits."""
        logits: dict[int, torch.Tensor] = {}
        by_row = {item.row_index: (item.state, item.stops) for item in work}
        waves = max((len(item.pieces) for item in work), default=0)
        for wave in range(waves):
            pieces = [item.pieces[wave] for item in work if len(item.pieces) > wave]
            last = {item.row_index for item in work if len(item.pieces) == wave + 1}
            for window, members in pack_windows(
                pieces,
                self.program.windows,
                self.slots,
                self.program.capacity,
            ):
                out = self._run_window(rows, by_row, window, members)
                for s, (piece, _) in enumerate(members):
                    if piece.row in last:
                        logits[piece.row] = out[s : s + 1]
        return logits

    def step(self, step_input: StepInput) -> StepOutput:
        """Run one prefill step and return sampled first tokens for finished prompts."""
        if self._writer is None:
            msg = "K3 prefill: a request arrived before the KV pools were bound"
            raise RuntimeError(msg)
        rows, copies = step_input.requests, step_input.copies
        if copies:
            pairs = [(int(src), int(dst)) for src, dst in copies]
            self._writer.copy_blocks(pairs)
            if self.draft_latents is not None:
                self.draft_latents.copy_blocks(pairs, COPY_PAIRS)
        for req in step_input.finished:
            self._requests.pop(req, None)
        largest = self.program.windows[-1]
        work, hits = self._collect_step_work(rows, largest)
        self._read_hit_states(rows, hits)
        logits = self._execute_waves(rows, work)
        done = {item.row_index for item in work if item.finishes}
        for item in work:
            if item.finishes:
                del self._requests[item.state.req_id]
        return StepOutput(self._first_tokens(rows, logits, done))

    def _blocks(self, values: np.ndarray) -> torch.Tensor:
        """Upload int32 block-index array ``values`` onto the TPU device."""
        return torch.from_numpy(np.ascontiguousarray(values, np.int32)).to("tpu")

    def _first_tokens(
        self,
        rows: Sequence[RequestInput],
        logits: dict[int, torch.Tensor],
        done: set[int],
    ) -> torch.Tensor:
        """Sample first tokens for rows in ``done`` into ``[rows, 1]`` int32 IDs."""
        count = self.config.max_num_seqs
        if len(rows) > count:
            msg = f"K3 prefill: {len(rows)} rows exceed max_num_seqs {count}"
            raise RuntimeError(msg)
        stacked = torch.cat(
            [logits.get(i, self._zeros()) for i in range(len(rows))]
            + [self._zeros()] * (count - len(rows))
        )
        params = np.zeros((count, _SAMPLING_PARAM_COLS), np.int32)
        params[:, 1] = np.float32(1).view(np.int32)
        for i, row in enumerate(rows):
            temperature, top_p, top_k, seed = row.sampling.as_tuple()
            params[i] = (
                np.float32(temperature).view(np.int32),
                np.float32(top_p).view(np.int32),
                top_k,
                seed & _POSITIVE_INT32_MASK,
                max(row.end - 1, 0),
            )
        sampled = any(row.sampling.temperature > 0 for row in rows)
        (tokens,) = self._sampler_op(sampled=sampled)(
            stacked,
            torch.from_numpy(params).to("tpu"),
        )
        keep = torch.tensor([i in done for i in range(count)], dtype=torch.bool).to(
            "tpu"
        )
        return torch.where(keep, tokens, torch.full_like(tokens, -1))[
            : max(len(rows), 1),
            None,
        ]

    def _sampler_op(self, *, sampled: bool) -> CustomOp:
        """Build or return the first-token greedy or stochastic sampling custom op."""
        if sampled in self._sampler_ops:
            return self._sampler_ops[sampled]
        width = VOCAB // TP_SIZE

        def body(logits: jax.Array, params: jax.Array) -> tuple[jax.Array]:
            """Choose first tokens across rows on the TP vocabulary mesh."""
            rank = jax.lax.axis_index(sampling.AXIS)
            local = jax.lax.dynamic_slice_in_dim(logits, rank * width, width, axis=1)
            chosen = sampling.greedy(local)
            if sampled:
                temperature = _as_float(params[:, 0])
                top_p = _as_float(params[:, 1])
                draws = sampling.sample_rows(
                    local,
                    jnp.where(temperature > 0, temperature, 1.0),
                    top_p,
                    params[:, 2],
                    params[:, 3],
                    params[:, 4],
                )
                chosen = jnp.where(temperature > 0, draws, chosen)
            return (chosen.astype(jnp.int32),)

        key = ("sampler", sampled)
        name = f"pallas::k3_prefill_first_token{'_sampled' if sampled else ''}"
        self._sampler_ops[sampled] = register_op(
            name,
            kernel_mesh(sampling.AXIS, TP_SIZE)[0],
            body,
            OpRegistrationSpec(
                in_specs=(P(), P()), out_specs=(P(),), export_key=repr(key)
            ),
        )
        return self._sampler_ops[sampled]

    def _read_hit_states(
        self,
        rows: Sequence[RequestInput],
        hits: Sequence[tuple[int, _Request]],
    ) -> None:
        """Populate KDA states and DSpark rings for new prefix-cache hit requests."""
        if not hits:
            return
        if self._writer is None:
            msg = "K3 prefill: pool writer must be bound before reading hit states"
            raise RuntimeError(msg)
        for first in range(0, len(hits), self.slots):
            chunk = hits[first : first + self.slots]
            blocks = [
                self._blocks(
                    kda_blocks_at_column(
                        rows[i].block_tables,
                        self._kda_groups,
                        state.start // self.page_tokens - 1,
                    )
                )
                for i, state in chunk
            ]
            blocks += [self._zero_blocks()] * (self.slots - len(chunk))
            recurrent, conv = self._writer.read_states(torch.cat(blocks))
            for (_, state), pair in zip(
                chunk,
                self.program.split_states(recurrent, conv),
                strict=False,
            ):
                state.state = pair
            if self.ring is not None:
                lengths = [state.start for _, state in chunk]
                lengths += [0] * (self.slots - len(chunk))
                rings = self._writer.read_rings(
                    self.ring.config,
                    torch.cat(blocks),
                    lengths,
                )
                for (_, state), ring in zip(chunk, rings, strict=False):
                    state.ring = ring

    def _zero_blocks(self) -> torch.Tensor:
        """Allocate a ``[KDA_LAYERS]`` int32 zero block-index vector on the TPU."""
        return torch.zeros(KDA_LAYERS, dtype=torch.int32).to("tpu")

    def _zero_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the shared single-sequence zero KDA state pair on the TPU."""
        if self._zero_pair is None:
            self._zero_pair = self.program.zero_state()
        return self._zero_pair

    def _pack_window_members(
        self,
        rows: Sequence[RequestInput],
        by_row: Mapping[int, tuple[_Request, dict[int, torch.Tensor]]],
        window: int,
        members: Sequence[tuple[Piece, int]],
    ) -> _WindowPackPlan:
        """Assemble token IDs, segment descriptors, and KDA blocks for ``members``."""
        count = self.program.window_slots(window)
        tokens = np.full(window, -1, np.int32)
        offsets: list[int] = []
        lengths: list[int] = []
        checkpoints: list[int] = []
        bases: list[int] = []
        cached: list[int] = []
        states: list[tuple[torch.Tensor, torch.Tensor]] = []
        finals: list[torch.Tensor] = []
        ats: list[torch.Tensor] = []
        base = 0
        for piece, offset in members:
            state, stops = by_row[piece.row]
            if piece.start % self.page_tokens:
                msg = (
                    f"K3 prefill: request {state.req_id} continues mid-page at "
                    f"token {piece.start}; windows need page-aligned chunks "
                    "(vllm_torchtpu's Mamba split patch aligns them)"
                )
                raise RuntimeError(msg)
            row = rows[piece.row]
            tokens[offset : offset + piece.length] = np.asarray(
                row.token_ids[piece.start : piece.end],
                np.int32,
            )
            offsets.append(offset)
            lengths.append(piece.length)
            checkpoints.append(0 if piece.stop is None else piece.stop - piece.start)
            bases.append(base)
            cached.append(piece.start)
            base += piece.start // self.page_tokens * self.page_tokens
            states.append(
                state.state if state.state is not None else self._zero_state()
            )
            finals.append(
                self._blocks(
                    kda_blocks_at_length(row.block_tables, self._kda_groups, row.end)
                )
            )
            ats.append(self._zero_blocks() if piece.stop is None else stops[piece.stop])
        unused = count - len(members)
        offsets += [window] * unused
        lengths += [0] * unused
        checkpoints += [0] * unused
        bases += [base] * unused
        cached += [0] * unused
        states += [self._zero_state()] * unused
        live = live_slots(len(members), count)
        finals += [self._zero_blocks()] * (live - len(members))
        ats += [self._zero_blocks()] * (live - len(members))
        return _WindowPackPlan(
            tokens=tokens,
            descriptor=offsets + lengths + checkpoints + bases + cached,
            states=states,
            finals=finals,
            ats=ats,
            unused=unused,
        )

    def _build_window_ring_hook(
        self,
        window: int,
        members: Sequence[tuple[Piece, int]],
        by_row: Mapping[int, tuple[_Request, dict[int, torch.Tensor]]],
        order: np.ndarray,
    ) -> tuple[
        Callable[..., None] | None,
        dict[str, tuple[list[RingPair], list[RingPair]]],
        dict[str, torch.Tensor],
    ]:
        """Build the optional draft-latent or DSpark-ring hook for ``_run_window``."""
        ring_out: dict[str, tuple[list[RingPair], list[RingPair]]] = {}
        draft_out: dict[str, torch.Tensor] = {}
        ring_hook: Callable[..., None] | None = None
        if self.draft_latents is not None:
            draft_latents = self.draft_latents
            draft_positions = np.full(window, -1, np.int32)
            for piece, offset in members:
                draft_positions[offset : offset + piece.length] = np.arange(
                    piece.start,
                    piece.end,
                )

            def draft_hook(
                blocks: torch.Tensor,
                _prefix: torch.Tensor,
                _middle: torch.Tensor,
                tap3: torch.Tensor,
                tap90: torch.Tensor,
            ) -> None:
                """Project MLA draft latents from auxiliary residual stream taps."""
                draft_out["draft"] = draft_latents.latents(
                    window,
                    blocks,
                    tap3,
                    tap90,
                    draft_positions,
                    order,
                )

            ring_hook = draft_hook

        if self.ring is not None:
            ring_model = self.ring
            unused = self.program.window_slots(window) - len(members)
            positions = np.full(window, -1, np.int32)
            sequences = np.full(window, -1, np.int32)
            ends: list[int] = []
            stops: list[int] = []
            rings: list[RingPair] = []
            for s, (piece, offset) in enumerate(members):
                state, _ = by_row[piece.row]
                positions[offset : offset + piece.length] = np.arange(
                    piece.start,
                    piece.end,
                )
                sequences[offset : offset + piece.length] = s
                ends.append(piece.end)
                stops.append(-1 if piece.stop is None else piece.stop)
                if state.ring is None:
                    state.ring = ring_model.empty(piece.start)
                rings.append(state.ring)
            ends += [-1] * unused
            stops += [-1] * unused
            rings += [ring_model.zero()] * unused
            ordered_pos = positions[order]
            ordered_seq = sequences[order]

            def dspark_hook(
                blocks: torch.Tensor,
                prefix: torch.Tensor,
                middle: torch.Tensor,
                _tap3: torch.Tensor,
                _tap90: torch.Tensor,
            ) -> None:
                """Update per-sequence DSpark rings from window residual streams."""
                ring_out["rings"] = ring_model.update(
                    blocks,
                    prefix,
                    middle,
                    ordered_pos,
                    ordered_seq,
                    ends,
                    stops,
                    rings,
                )

            ring_hook = dspark_hook

        return ring_hook, ring_out, draft_out

    def _run_window(
        self,
        rows: Sequence[RequestInput],
        by_row: Mapping[int, tuple[_Request, dict[int, torch.Tensor]]],
        window: int,
        members: Sequence[tuple[Piece, int]],
    ) -> torch.Tensor:
        """Run one packed window, write pools and KDA states, and return logits."""
        if self._writer is None:
            msg = "K3 prefill: pool writer must be bound before running windows"
            raise RuntimeError(msg)
        local = window // TP_SIZE
        pack_plan = self._pack_window_members(rows, by_row, window, members)
        ids, local_bases, write_table = self._cp_operands(
            rows,
            by_row,
            window,
            members,
        )
        order = self.program.sp_order(window)
        ring_hook, ring_out, draft_out = self._build_window_ring_hook(
            window,
            members,
            by_row,
            order,
        )
        recurrent, conv = self.program.stack_states(pack_plan.states)
        shard = (
            torch.from_numpy(
                pack_plan.tokens[order[self.rank * local : (self.rank + 1) * local]]
            )
            .unsqueeze(0)
            .to("tpu")
        )
        extent = members[-1][1] + members[-1][0].length
        logits, latents, recurrent, conv, recurrent_at, conv_at = (
            self.program.run_packed(
                _PackedWindowInputs(
                    shard=shard,
                    extent=_device_i32([extent]),
                    segments=_device_i32(pack_plan.descriptor),
                    recurrent=recurrent,
                    conv=conv,
                    pools=self._writer.mla_pools(),
                    ids=ids,
                    local=local_bases,
                    ring=ring_hook,
                )
            )
        )
        self._writer.write_packed(
            _PackedPoolWrite(
                latents=latents,
                recurrent=recurrent,
                conv=conv,
                recurrent_at=recurrent_at,
                conv_at=conv_at,
                table=write_table,
                kda_blocks=torch.cat(pack_plan.finals),
                blocks_at=torch.cat(pack_plan.ats),
                rings=ring_out.get("rings"),
            )
        )
        if "draft" in draft_out and self.draft_latents is not None:
            self.draft_latents.write(
                draft_out["draft"],
                write_table,
                self._writer.layout.block_shape,
            )
        for s, ((piece, _), pair) in enumerate(
            zip(members, self.program.split_states(recurrent, conv), strict=False)
        ):
            state, _ = by_row[piece.row]
            state.state, state.start = pair, piece.end
            if "rings" in ring_out:
                state.ring = ring_out["rings"][0][s]
        if self.rank == 0:
            logger.info(
                "K3 prefill: packed %d-token window: %s",
                window,
                ", ".join(
                    f"{by_row[p.row][0].req_id} {p.start}..{p.end - 1}@{o}"
                    + ("" if p.stop is None else f" ckpt {p.stop}")
                    for p, o in members
                ),
            )
        return logits

    def _cp_operands(
        self,
        rows: Sequence[RequestInput],
        by_row: Mapping[int, tuple[_Request, dict[int, torch.Tensor]]],
        window: int,
        members: Sequence[tuple[Piece, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build this rank's context-parallel prefix page IDs and write table."""
        cp, rank = DCP_SIZE, self.cp_rank
        own: list[int] = []
        bases: list[int] = []
        counts: list[int] = []
        cursor = 0
        table = [0] * (window // self.page_tokens)
        for piece, offset in members:
            state, _ = by_row[piece.row]
            row = rows[piece.row].block_tables[self._mla_group]
            first = piece.start // self.page_tokens
            owned = [int(row[k // cp]) for k in range(rank, first, cp)]
            bases.append(cursor)
            counts.append(len(owned) * self.page_tokens)
            own += owned
            cursor += len(owned) * self.page_tokens
            skip = max(state.first_cached - piece.start, 0) // self.page_tokens
            for p in range(skip, piece.rows // self.page_tokens):
                if (first + p) % cp == rank:
                    table[offset // self.page_tokens + p] = int(row[(first + p) // cp])
        unused = self.program.window_slots(window) - len(members)
        if cursor > self.cp_capacity:
            msg = (
                f"K3 prefill: this rank's prefix share {cursor} "
                f"exceeds {self.cp_capacity}"
            )
            raise RuntimeError(msg)
        ids = [len(own)] + own + [0] * (self.cp_capacity // self.page_tokens - len(own))
        local = bases + [cursor] * unused + counts + [0] * unused
        return _device_i32([ids]), _device_i32([local]), _device_i32([table])

    def _compile_window_calls(
        self,
        window: int,
        calls: list[CompileCall],
        ring_writes: set[int],
        cache: torch.Tensor | None,
    ) -> None:
        """Append ``CompileCall`` entries for one prefill ``window`` onto ``calls``."""
        if self._writer is None:
            msg = "K3 prefill: pool writer must be bound before compilation_calls"
            raise RuntimeError(msg)
        program, writer, ring = self.program, self._writer, self.ring
        pools, mla = tuple(writer.raw_pools), writer.mla_pools()
        count, rows = program.window_slots(window), window // TP_SIZE
        ops = program.build_window_ops(window)
        stream = torch.empty((1, rows, HIDDEN_SIZE), dtype=torch.bfloat16)
        blocks = torch.empty((1, 8, rows, HIDDEN_SIZE), dtype=torch.bfloat16)
        recurrent = torch.empty(
            (1, KDA_LAYERS, count, _KDA_HEADS, _KDA_HEAD_DIM, _KDA_HEAD_DIM),
            dtype=torch.float32,
        )
        conv = torch.empty(
            (
                1,
                KDA_LAYERS,
                count,
                _KDA_CONV_TAPS,
                _KDA_HEADS,
                _KDA_HEADS,
                _KDA_HEAD_DIM,
            ),
            dtype=torch.bfloat16,
        )
        latents = torch.empty(
            (1, MLA_LAYERS, window, _MLA_LATENT_DIM),
            dtype=torch.bfloat16,
        )
        state = (stream, blocks, recurrent, conv, recurrent, conv, latents)
        extent = torch.empty((1,), dtype=torch.int32)
        segments = torch.empty((_SEGMENT_DESC_FIELDS * count,), dtype=torch.int32)
        tail = (
            torch.empty(
                (1, 1 + self.cp_capacity // self.page_tokens), dtype=torch.int32
            ),
            torch.empty((1, 2 * count), dtype=torch.int32),
        )
        calls.extend(
            (
                CompileCall(
                    ops["first"],
                    (
                        torch.empty((1, rows), dtype=torch.int32),
                        extent,
                        segments,
                        recurrent,
                        conv,
                        mla[0],
                        *tail,
                        *program.flat_weights("io", "common", "kda", "mla", "moe"),
                    ),
                    {},
                    priority=window,
                ),
                CompileCall(
                    ops["group"],
                    (
                        *state,
                        extent,
                        torch.empty((1,), dtype=torch.int32),
                        segments,
                        mla[1],
                        mla[2],
                        *tail,
                        *program.flat_weights(
                            "common",
                            "kda",
                            "mla",
                            "moe",
                            expert_bank=1,
                        ),
                    ),
                    {},
                    priority=window,
                ),
                CompileCall(
                    ops["final"],
                    (
                        stream,
                        blocks,
                        latents,
                        extent,
                        segments,
                        mla[MLA_LAYERS - 1],
                        *tail,
                        *program.flat_weights(
                            "io",
                            "common",
                            "mla",
                            "moe",
                            expert_bank=_FINAL_EXPERT_BANK,
                        ),
                    ),
                    {},
                    priority=window,
                ),
            )
        )
        one = (recurrent[:, :, 0], conv[:, :, 0])
        calls.append(
            CompileCall(
                program.state_op("stack", count),
                (one[0],) * count + (one[1],) * count,
                {},
            )
        )
        calls.append(
            CompileCall(
                program.state_op("split", count),
                (recurrent, conv),
                {},
            )
        )
        if ring is not None and cache is not None:
            calls.append(
                CompileCall(
                    ring.update_op(window, count),
                    (
                        blocks,
                        stream,
                        stream,
                        torch.empty((window,), dtype=torch.int32),
                        torch.empty((window,), dtype=torch.int32),
                        torch.empty((count,), dtype=torch.int32),
                        torch.empty((count,), dtype=torch.int32),
                        *ring.weights,
                        *(cache,) * count,
                        *(torch.empty((_RING_META_ELEMENTS,), dtype=torch.int32),)
                        * count,
                    ),
                    {},
                )
            )
        for live in write_sizes(count):
            calls.append(
                CompileCall(
                    writer.write_op(window, count, live),
                    (
                        latents,
                        recurrent,
                        conv,
                        recurrent,
                        conv,
                        *pools,
                        torch.empty((1, window // self.page_tokens), dtype=torch.int32),
                        torch.empty((live * KDA_LAYERS,), dtype=torch.int32),
                        torch.empty((live * KDA_LAYERS,), dtype=torch.int32),
                    ),
                    {},
                )
            )
            if ring is not None and cache is not None and live not in ring_writes:
                ring_writes.add(live)
                calls.append(
                    CompileCall(
                        writer.rings_op(live),
                        (
                            *pools,
                            torch.empty((live * KDA_LAYERS,), dtype=torch.int32),
                            *(cache,) * live,
                            *(torch.empty((_RING_META_ELEMENTS,), dtype=torch.int32),)
                            * live,
                        ),
                        {},
                    )
                )

    def compilation_calls(self) -> list[CompileCall]:
        """Build shape-only ``CompileCall`` descriptors for every warmup custom op."""
        writer, ring = self._writer, self.ring
        if writer is None:
            msg = "K3 prefill: pool writer must be bound before compilation_calls"
            raise RuntimeError(msg)
        pools = tuple(writer.raw_pools)
        calls: list[CompileCall] = []
        ring_writes: set[int] = set()
        cache: torch.Tensor | None = None
        if ring is not None:
            c = ring.config
            cache = torch.empty(
                (1, c.layers, c.window, 2 * c.head_dim),
                dtype=torch.bfloat16,
            )
        for window in self.program.windows:
            self._compile_window_calls(window, calls, ring_writes, cache)
        hit = torch.empty((self.slots * KDA_LAYERS,), dtype=torch.int32)
        calls.append(CompileCall(writer.read_states_op(self.slots), (*pools, hit), {}))
        if ring is not None:
            calls.append(
                CompileCall(
                    writer.read_rings_op(ring.config, self.slots),
                    (*pools, hit, torch.empty((self.slots,), dtype=torch.int32)),
                    {},
                )
            )
        calls.append(
            CompileCall(
                writer.copy_op(),
                (
                    *pools,
                    torch.empty((COPY_PAIRS,), dtype=torch.int32),
                    torch.empty((COPY_PAIRS,), dtype=torch.int32),
                ),
                {},
            )
        )
        logits = torch.empty((self.config.max_num_seqs, VOCAB), dtype=torch.float32)
        calls.extend(
            CompileCall(
                self._sampler_op(sampled=sampled),
                (
                    logits,
                    torch.empty(
                        (self.config.max_num_seqs, _SAMPLING_PARAM_COLS),
                        dtype=torch.int32,
                    ),
                ),
                {},
            )
            for sampled in (False, True)
        )
        return calls

    def _precompile(self) -> None:
        """Trace and compile :meth:`compilation_calls` concurrently across workers."""
        started = time.perf_counter()
        calls = self.compilation_calls()
        calls.sort(key=lambda call: call.priority, reverse=True)
        prepare_distributed(calls)
        compile_ops(calls)
        if self.rank == 0:
            logger.info(
                "K3 prefill: precompiled %d programs in %.1f s",
                len(calls),
                time.perf_counter() - started,
            )

    def _warm_window(self, window: int, writer: PoolWriter) -> list[torch.Tensor]:
        """Run one warmup pass for ``window`` and its pool writes on device."""
        count = self.program.window_slots(window)
        desc = [0] + [window] * (count - 1) + [1] + [0] * (count - 1) + [0] * 3 * count
        recurrent, conv = self.program.stack_states(
            [self.program.zero_state() for _ in range(count)]
        )
        ring_hook: Callable[..., None] | None = None
        ring_out: dict[str, tuple[list[RingPair], list[RingPair]]] = {}
        draft_out: dict[str, torch.Tensor] = {}
        if self.ring is not None:
            ring_model = self.ring
            none = np.full(window, -1, np.int32)

            def warm_ring(
                blocks: torch.Tensor,
                prefix: torch.Tensor,
                middle: torch.Tensor,
                _tap3: torch.Tensor,
                _tap90: torch.Tensor,
            ) -> None:
                """Warm up DSpark ring update for ``window``."""
                ring_out["rings"] = ring_model.update(
                    blocks,
                    prefix,
                    middle,
                    none,
                    none,
                    [-1] * count,
                    [-1] * count,
                    [ring_model.zero()] * count,
                )

            ring_hook = warm_ring

        if self.draft_latents is not None:
            draft_latents = self.draft_latents
            order = self.program.sp_order(window)

            def warm_draft(
                blocks: torch.Tensor,
                _prefix: torch.Tensor,
                _middle: torch.Tensor,
                tap3: torch.Tensor,
                tap90: torch.Tensor,
            ) -> None:
                """Warm up MLA DSpark draft latent projection for ``window``."""
                draft_out["draft"] = draft_latents.latents(
                    window,
                    blocks,
                    tap3,
                    tap90,
                    np.full(window, -1, np.int32),
                    order,
                )

            ring_hook = warm_draft

        logits, latents, recurrent, conv, recurrent_at, conv_at = (
            self.program.run_packed(
                _PackedWindowInputs(
                    shard=torch.zeros(
                        (1, window // TP_SIZE),
                        dtype=torch.int32,
                    ).to("tpu"),
                    extent=_device_i32([1]),
                    segments=_device_i32(desc),
                    recurrent=recurrent,
                    conv=conv,
                    pools=writer.mla_pools(),
                    ids=_rank_zeros_i32(1 + self.cp_capacity // self.page_tokens),
                    local=_rank_zeros_i32(2 * count),
                    ring=ring_hook,
                )
            )
        )
        outputs = [logits, self.program.split_states(recurrent, conv)[0][0]]
        for live in write_sizes(count):
            writer.write_packed(
                _PackedPoolWrite(
                    latents=latents,
                    recurrent=recurrent,
                    conv=conv,
                    recurrent_at=recurrent_at,
                    conv_at=recurrent_at if conv_at is None else conv_at,
                    table=_rank_zeros_i32(window // self.page_tokens),
                    kda_blocks=_zeros_i32(live * KDA_LAYERS),
                    blocks_at=_zeros_i32(live * KDA_LAYERS),
                    rings=ring_out.get("rings"),
                )
            )
        if "draft" in draft_out and self.draft_latents is not None:
            self.draft_latents.write(
                draft_out["draft"],
                _rank_zeros_i32(window // self.page_tokens),
                writer.layout.block_shape,
            )
        return outputs

    def warmup(self) -> None:
        """Precompile and execute every prefill window, pool op, and sampler once."""
        writer = self._writer
        if writer is None:
            msg = "K3 prefill: bind the KV caches before warming up"
            raise RuntimeError(msg)
        started = time.perf_counter()
        self._precompile()
        outputs: list[torch.Tensor] = []
        for window in self.program.windows:
            outputs.extend(self._warm_window(window, writer))
        count = self.slots
        outputs.append(writer.read_states(_zeros_i32(count * KDA_LAYERS))[0])
        if self.ring is not None:
            outputs.append(
                writer.read_rings(
                    self.ring.config,
                    _zeros_i32(count * KDA_LAYERS),
                    [0] * count,
                )[0][0]
            )
        writer.copy_blocks([(0, 0)])
        if self.draft_latents is not None:
            self.draft_latents.copy_blocks([(0, 0)], COPY_PAIRS)
        stacked = torch.zeros(
            (self.config.max_num_seqs, VOCAB),
            dtype=torch.float32,
        ).to("tpu")
        params = torch.zeros(
            (self.config.max_num_seqs, _SAMPLING_PARAM_COLS),
            dtype=torch.int32,
        ).to("tpu")
        outputs.extend(
            self._sampler_op(sampled=sampled)(stacked, params)[0]
            for sampled in (False, True)
        )
        for tensor in outputs:
            materialize(tensor)
        materialize(writer.raw_pools[0])
        if self.rank == 0:
            logger.info(
                "K3 prefill: windows %s warm in %.1f s; HBM %s",
                list(self.program.windows),
                time.perf_counter() - started,
                hbm(self.rank),
            )
