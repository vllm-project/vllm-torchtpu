# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.bridge import (
    hbm,
    kernel_mesh,
    materialize,
    register_global_op,
    register_op,
    report_load_stats,
    stream_to_device,
)
from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.megakernel.kimi_k3.layout import NUM_LAYERS
from vllm_torchtpu.megakernel.kimi_k3.pools import (
    CONV_ROWS,
    KDA_LAYERS,
    MLA_LAYERS,
    SSM_ROWS,
    bind_raw_pools,
    kda_blocks_at_column,
    kda_blocks_at_length,
    read_draft_rings,
    tp_rank_permutation,
    write_draft_rings,
)
from vllm_torchtpu.megakernel.kimi_k3.program import (
    DCP_SIZE,
    TP_SIZE,
    VOCAB,
    kernel_modules,
)
from vllm_torchtpu.megakernel.program import (
    KVGroup,
    RequestInput,
    StepInput,
    StepOutput,
)

logger = init_logger(__name__)

AXIS = "k3_sp"
WINDOWS = (512, 1024, 2048, 4096, 8192)
PAGE_TOKENS = 256
SLOTS = 16  # sequences per window, at most (and at most one per page)
GROUP_PAIRS = 11  # regular 8-layer groups: layers 4..91
COPY_PAIRS = 8  # copy-on-write block copies per PoolWriter.copy_blocks call


def live_slots(used: int, slots: int) -> int:
    """The sequence slots a window's pool writes cover: ``used`` rounded up
    to a power of two, at most the window's ``slots`` (one compiled write
    per size; slots past it are unused and their states are not written)."""
    return min(slots, 1 << max(used - 1, 0).bit_length())


def write_sizes(slots: int) -> tuple[int, ...]:
    """Every :func:`live_slots` size of a window with ``slots`` slots."""
    return tuple(sorted({live_slots(n, slots) for n in range(1, slots + 1)}))


def _device_i32(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32).to("tpu")


def _zeros_i32(n: int) -> torch.Tensor:
    return torch.zeros(n, dtype=torch.int32).to("tpu")


def _rank_zeros_i32(n: int) -> torch.Tensor:
    return torch.zeros((1, n), dtype=torch.int32).to("tpu")


def _as_float(bits):
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def prefill_windows(max_num_batched_tokens: int, max_model_len: int) -> tuple[int, ...]:
    """The windows to compile: those up to the step budget (at least the
    smallest), up to the smallest one holding ``--max-model-len``."""
    windows = [w for w in WINDOWS if w <= max(WINDOWS[0], max_num_batched_tokens)]
    holding = [w for w in windows if w >= max_model_len]
    return tuple(w for w in windows if w < max_model_len) + (
        (min(holding),) if holding else ()
    )


@dataclasses.dataclass(frozen=True)
class Piece:
    """Tokens ``[start, end)`` of one request computed in one packed window;
    ``stop`` (a skipped step's end in ``(start, end]``) is where the window
    also checkpoints the KDA state, ``None`` for no checkpoint."""

    row: int
    start: int
    end: int
    stop: int | None = None
    page_tokens: int = PAGE_TOKENS

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def rows(self) -> int:  # window rows taken, padding to the next sequence included
        return -(-self.length // self.page_tokens) * self.page_tokens


def skips_step(
    computed: int, end: int, prompt: int, start: int, largest: int, block: int
) -> bool:
    """Whether a step scheduling a request's tokens ``[computed, end)`` of
    ``prompt`` is skipped (no token): its state is at ``start`` and the rest
    of the prompt fits the ``largest`` window. Never when the step completes
    a ``block``-token logical block: vLLM's KV offloading stores that block
    (its pages and the boundary KDA state) behind this step's device work,
    so a later computing step would write them after the host copy was
    taken, and a hit restores the stale bytes."""
    if end >= prompt or prompt - start > largest:
        return False
    return end // block == computed // block


def plan_pieces(
    row: int,
    start: int,
    end: int,
    stops: list[int],
    largest: int,
    page_tokens: int = PAGE_TOKENS,
) -> list[Piece]:
    """Split a request's tokens ``[start, end)`` (``start`` block-aligned)
    into pieces of at most ``largest`` tokens, each with at most one
    checkpoint: a piece holding more than one of ``stops`` ends at the first.
    Consecutive pieces must run in consecutive windows (each reads the pages
    and state the previous one wrote)."""
    pieces, stops = [], sorted(e for e in stops if start < e < end)
    while start < end:
        stop = min(start + largest, end)
        inside = [e for e in stops if start < e <= stop]
        if len(inside) > 1:
            stop = inside[0]
        checkpoint = next((e for e in inside if e <= stop), None)
        pieces.append(Piece(row, start, stop, checkpoint, page_tokens))
        start = stop
    return pieces


def pack_windows(
    pieces: list[Piece], windows: tuple[int, ...], slots: int, capacity: int
) -> list[tuple[int, list[tuple[Piece, int]]]]:
    """Greedy, in order: ``pieces`` (at most one per request, each at most
    the largest window) into windows of at most ``slots`` sequences, each at
    a page-aligned offset, whose cached prefixes (the tokens before each
    piece) fit ``capacity``. Returns ``(window size, [(piece, offset)])`` per
    window, the size the smallest compiled one holding its last sequence."""
    largest, packed, members, offset, prefix = windows[-1], [], [], 0, 0

    def close():
        extent = members[-1][1] + members[-1][0].length
        packed.append((min(w for w in windows if w >= extent), list(members)))

    for piece in pieces:
        if piece.length > largest:
            raise ValueError(
                f"piece of {piece.length} tokens is longer than "
                f"the {largest}-token window"
            )
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


def _bank_expert_weights(iterator):
    """Stream checkpoint parts into exactly one resident copy of each expert."""
    state: dict[str, list[Any]] = {}
    for group, name, value in iterator:
        family, marker, suffix = name.partition("@")
        if group != "moe" or family not in _EXPERT_FAMILIES:
            yield group, name, value
            continue
        part, parts = map(int, suffix.split("/")) if marker else (0, 1)
        entry = state.setdefault(family, [0, 0, 0, None])
        expected_part, bank, used, buffer = entry
        if part != expected_part:
            raise ValueError(f"Out-of-order expert checkpoint part: {name}")
        offset = 0
        while offset < value.shape[0]:
            if bank == len(_EXPERT_BANK_SIZES):
                raise ValueError(f"Too many expert layers for {family}")
            capacity = _EXPERT_BANK_SIZES[bank]
            if buffer is None:
                buffer = np.empty((capacity, *value.shape[1:]), dtype=value.dtype)
            count = min(capacity - used, value.shape[0] - offset)
            buffer[used : used + count] = value[offset : offset + count]
            used += count
            offset += count
            if used == capacity:
                yield group, f"{family}#bank{bank}", buffer
                bank, used, buffer = bank + 1, 0, None
        state[family] = [part + 1, bank, used, buffer]
        if part + 1 == parts and (bank != len(_EXPERT_BANK_SIZES) or used):
            raise ValueError(f"Incomplete expert checkpoint: {family}")
    if set(state) != set(_EXPERT_FAMILIES) or any(
        bank != len(_EXPERT_BANK_SIZES) or used for _, bank, used, _ in state.values()
    ):
        raise ValueError("Missing expert checkpoint parts")


class PrefillProgram:
    """The prefill stack's device weights and packed-window ops.

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
        self.windows = tuple(sorted(int(w) for w in windows))
        self.capacity, self.slots = int(capacity), int(slots)
        self.page_tokens = int(page_tokens)
        self._kernels, self._rank_loader = kernel_modules(
            "prefill_megakernel", "rank_loader"
        )
        self.mesh, self.rank = kernel_mesh(AXIS, TP_SIZE)
        loader = self._rank_loader
        self.names = {
            "io": loader.PREFILL_IO,
            "common": loader.PREFILL_COMMON,
            "kda": loader.PREFILL_KDA,
            "mla": loader.PREFILL_MLA,
            "moe": tuple(self._kernels.MOE_FAMILIES),
        }
        self.weights: dict[str, dict[str, torch.Tensor]] = {}
        self._expert_banks: list[dict[str, torch.Tensor]] = []
        self._ops: dict[Any, dict[str, Any]] = {}
        self._pairs: list[torch.Tensor] = []
        self._state_ops: dict[tuple[str, int], Any] = {}

    # ------------------------------------------------------------ weights
    def load_weights(self, presharded) -> None:
        cache = presharded / "prefill-sp-cache"
        from .config import active

        iterator = self._rank_loader.iter_prefill_rank(
            presharded, cache, self.rank, split_experts=True
        )
        self._expert_banks = []
        if active().prefill_fused_moe and active().prefill_expert_parallel == 16:
            self._expert_banks = [{} for _ in _EXPERT_BANK_SIZES]
            iterator = _bank_expert_weights(iterator)

        def store(group, name, tensor):
            family, marker, index = name.partition("#bank")
            if marker:
                bank = int(index)
                self._expert_banks[bank][family] = tensor
                # The default dictionary retains handles to bank zero so common
                # shape/introspection code still sees every named family.
                if bank != 0:
                    return
                name = family
            self.weights.setdefault(group, {})[name] = tensor

        stats = stream_to_device(iterator, store)
        if self._expert_banks and any(
            set(bank) != set(_EXPERT_FAMILIES) for bank in self._expert_banks
        ):
            raise RuntimeError("K3 prefill: incomplete resident expert banks")
        for group, names in self.names.items():
            missing = [n for n in names if n not in self.weights.get(group, {})]
            if missing:
                raise RuntimeError(f"K3 prefill: missing {group} weights {missing}")
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

    def _flat(self, *groups: str, expert_bank: int = 0) -> list[torch.Tensor]:
        experts = self._expert_banks[expert_bank] if self._expert_banks else {}
        return [
            experts[n] if g == "moe" and n in experts else self.weights[g][n]
            for g in groups
            for n in self.names[g]
        ]

    def _unflat(self, args, *groups: str) -> tuple[dict, list]:
        out, index = {}, 0
        for group in groups:
            out[group] = {n: args[index + i] for i, n in enumerate(self.names[group])}
            index += len(self.names[group])
        return out, list(args[index:])

    def _count(self, *groups: str) -> int:
        return sum(len(self.names[g]) for g in groups)

    def window_slots(self, window: int) -> int:
        """The sequence slots a window's ops are compiled for: ``slots``, or
        fewer when the window holds fewer pages (each sequence starts a
        page), so a small window carries no state for slots it never fills."""
        return max(1, min(self.slots, window // self.page_tokens))

    # ------------------------------------------------------------ ops
    #
    # Sequence s of a window occupies rows [offset_s, offset_s + length_s),
    # offsets ascending multiples of the page size; other rows carry token
    # id -1. The int32 [5 * slots] descriptor is offsets | lengths |
    # checkpoint positions (relative to each sequence's start, multiples of
    # 128) | prefix bases | prefix lengths (the whole prefixes laid end to
    # end; each rank's share travels in the CP operands, see
    # prefill_megakernel.CpPrefix).

    def _build(self, window: int) -> dict[str, Any]:
        if window in self._ops:
            return self._ops[window]
        if window not in self.windows:
            raise ValueError(f"K3 prefill: window {window} is not among {self.windows}")
        mesh, ops, kernels = self.mesh, {}, self._kernels
        count = self.window_slots(window)
        sharded, replicated = P(AXIS), P()
        embedding = self._kernels.make_embedding_sp(mesh, seq_len=window)
        first = self._kernels.make_prefill_stack_sp_indexed(
            mesh,
            seq_len=window,
            depth=4,
            mla_flash_tile=256,
            unroll=True,
            segments=count,
            prefix_capacity=self.capacity,
        )
        regular = self._kernels.make_regular_group8_sp_indexed(
            mesh,
            seq_len=window,
            mla_flash_tile=256,
            segments=count,
            prefix_capacity=self.capacity,
        )
        final = self._kernels.make_final_mla_sp_indexed(
            mesh,
            seq_len=window,
            mla_flash_tile=256,
            segments=count,
            prefix_capacity=self.capacity,
        )
        logits_op = self._kernels.make_prefill_logits_sp(
            mesh, seq_len=window, layers=NUM_LAYERS, segments=count
        )
        weight_groups = ("common", "kda", "mla", "moe")
        state_sharding = jax.sharding.NamedSharding(mesh, sharded)

        def zeros(shape, dtype):
            # Rank-leading states: shard them, or XLA may replicate them.
            return jax.lax.with_sharding_constraint(
                jnp.zeros((32, *shape), dtype), state_sharding
            )

        def prefix_specs(pools: int):
            """Specs of an op's prefix operands: its MLA layers' pools, then
            this rank's prefix ids and the local bases/counts."""
            return (sharded,) * (pools + 2)

        def split(args, pools: int):
            return (
                kernels.CpPrefix(tuple(args[:pools]), args[pools], args[pools + 1]),
                args[pools + 2 :],
            )

        def run_first(tokens, length, segments, rec, hist, *args):
            arena, args = split(args, 1)
            w, _ = self._unflat(args, "io", *weight_groups)
            hidden = embedding(tokens, w["io"]["embedding"], length)
            latents = zeros((MLA_LAYERS, window, 576), jnp.bfloat16)
            rec_at = zeros((KDA_LAYERS, count, 3, 128, 128), jnp.float32)
            hist_at = zeros((KDA_LAYERS, count, 3, 3, 3, 128), jnp.bfloat16)
            prefix, blocks, kda_state, mla_state = first(
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
            return (prefix, blocks, *kda_state, *mla_state)

        # prefix, blocks, rec, hist, rec_at, hist_at, latents
        state_specs = (sharded,) * 7
        name = f"k3_prefill_packed{count}c"
        ops["first"] = register_global_op(
            self._op_name(f"{name}_first", window),
            mesh,
            run_first,
            (sharded,)
            + (replicated,) * 2
            + (sharded,) * 2
            + prefix_specs(1)
            + (sharded,) * self._count("io", *weight_groups),
            state_specs,
            donate_argnums=(3, 4),
            export_key=self._export_key(f"{name}_first", window),
        )

        def run_group(
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
            *args,
        ):
            arena, args = split(args, 2)
            w, _ = self._unflat(args, *weight_groups)
            prefix, blocks, kda_state, mla_state, middle = regular(
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
            return (prefix, blocks, *kda_state, *mla_state, middle)

        ops["group"] = register_global_op(
            self._op_name(f"{name}_group8", window),
            mesh,
            run_group,
            state_specs
            + (replicated,) * 3
            + prefix_specs(2)
            + (sharded,) * self._count(*weight_groups),
            state_specs + (sharded,),  # then the stream between its two groups
            donate_argnums=tuple(range(7)),
            export_key=self._export_key(f"{name}_group", window),
        )

        def run_final(prefix, blocks, latents, length, segments, *args):
            arena, args = split(args, 1)
            w, _ = self._unflat(args, "io", "common", "mla", "moe")
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
            # Full vocabulary on every rank, gathered in a manual region.
            logits = jax.shard_map(
                lambda part: jax.lax.all_gather(part[0], AXIS, axis=1, tiled=True),
                mesh=mesh,
                in_specs=(sharded,),
                out_specs=replicated,
                check_vma=False,
            )(logits)
            return logits, latents

        ops["final"] = register_global_op(
            self._op_name(f"{name}_final", window),
            mesh,
            run_final,
            (sharded,) * 3
            + (replicated,) * 2
            + prefix_specs(1)
            + (sharded,) * self._count("io", "common", "mla", "moe"),
            (replicated, sharded),
            donate_argnums=(0, 1, 2),
            export_key=self._export_key(f"{name}_final", window),
        )
        if not self._pairs:
            self._pairs = [
                torch.tensor([pair], dtype=torch.int32).to("tpu")
                for pair in range(GROUP_PAIRS)
            ]
        self._ops[window] = ops
        return ops

    def _op_name(self, base: str, window: int) -> str:
        return f"pallas::{base}_w{window}_c{self.capacity}"

    def _export_key(self, op: str, window: int) -> str:
        """The static configuration a window op's trace bakes in (export cache)."""
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
        shard: torch.Tensor,
        extent: torch.Tensor,
        segments: torch.Tensor,
        recurrent: torch.Tensor,
        conv: torch.Tensor,
        pools: list[torch.Tensor],
        ids: torch.Tensor,
        local: torch.Tensor,
        ring: Any = None,
    ):
        """Prefill a packed window: ``shard`` [1, W/32] this rank's token
        ids (its SP stripes, :meth:`sp_order`), ``extent`` [1] the end of
        the last sequence, ``segments`` the descriptor,
        ``recurrent``/``conv`` [1, 69, slots, ...] the incoming KDA states
        (donated; ``slots`` the window's :meth:`window_slots`), ``pools`` the
        24 MLA layers' unified pools (MLA layer m's at m), ``ids`` [1, 1 +
        ...] this rank's prefix page count then its prefix pages end to end
        and ``local`` [1, 2 * slots] each sequence's base and token count in
        them. Returns ``(logits [slots, VOCAB], latents [1, 24, W, 576],
        recurrent, conv, recurrent_at, conv_at)`` (the states after each
        sequence and at its checkpoint).
        ``ring(blocks, prefix, middle)`` (the DSpark ring's update, if any)
        runs on the residual streams before the final op consumes them."""
        ops = self._build(int(shard.shape[-1]) * 32)
        tail = (ids, local)
        state = ops["first"](
            shard,
            extent,
            segments,
            recurrent,
            conv,
            pools[0],
            *tail,
            *self._flat("io", "common", "kda", "mla", "moe"),
        )
        middle = None
        for p, pair in enumerate(self._pairs):
            *state, middle = ops["group"](
                *state,
                extent,
                pair,
                segments,
                pools[2 * p + 1],
                pools[2 * p + 2],
                *tail,
                *self._flat("common", "kda", "mla", "moe", expert_bank=p + 1),
            )
        prefix, blocks, rec, hist, rec_at, hist_at, latents = state
        if ring is not None:
            ring(blocks, prefix, middle)  # the last pair's middle: entering layer 88
        logits, latents = ops["final"](
            prefix,
            blocks,
            latents,
            extent,
            segments,
            pools[MLA_LAYERS - 1],
            *tail,
            *self._flat("io", "common", "mla", "moe", expert_bank=12),
        )
        return logits, latents, rec, hist, rec_at, hist_at

    def zero_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """One sequence's zero KDA states, ``[1, 69, ...]``."""
        return (
            torch.zeros(
                (1, KDA_LAYERS, 3, 128, 128), dtype=torch.float32, device="tpu"
            ),
            torch.zeros(
                (1, KDA_LAYERS, 3, 3, 3, 128), dtype=torch.bfloat16, device="tpu"
            ),
        )

    def _state_op(self, kind: str, count: int):
        key = (kind, count)
        if key not in self._state_ops:
            if kind == "stack":

                def body(*states):
                    return (
                        jnp.stack(states[:count], axis=2),
                        jnp.stack(states[count:], axis=2),
                    )

                specs = ((P(AXIS),) * 2 * count, (P(AXIS),) * 2)
            else:

                def body(recurrent, conv):
                    return (
                        *(recurrent[:, :, s] for s in range(count)),
                        *(conv[:, :, s] for s in range(count)),
                    )

                specs = ((P(AXIS),) * 2, (P(AXIS),) * 2 * count)
            self._state_ops[key] = register_op(
                f"pallas::k3_prefill_{kind}_states_{count}",
                self.mesh,
                body,
                *specs,
                export_key=repr(("states", kind, count)),
            )
        return self._state_ops[key]

    def sp_order(self, window: int) -> np.ndarray:
        """The window's token indices in rank-major SP ownership order (each
        rank holds 64-row stripes of every 2048-token block; one window / 32-row
        stripe under 2048 tokens)."""
        return self._kernels.sp_order(window)

    def stack_states(
        self, states: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequences' ``[1, 69, ...]`` KDA states -> the packed ``[1, 69, n,
        ...]`` incoming states of :meth:`run_packed` (new tensors: only op
        outputs reach an op, as torch views would compile new executables)."""
        return tuple(
            self._state_op("stack", len(states))(
                *(r for r, _ in states), *(c for _, c in states)
            )
        )

    def split_states(
        self, recurrent: torch.Tensor, conv: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """The inverse of :meth:`stack_states`."""
        count = int(recurrent.shape[2])
        out = self._state_op("split", count)(recurrent, conv)
        return list(zip(out[:count], out[count:]))


class PoolWriter:
    """Writes packed windows' latent pages and KDA states into vLLM's unified
    pools, and reads KDA states back (a prefix hit's).

    ``raw_pools`` are the distinct pool tensors (``layout.cache_layout()``);
    ``mla_raw[m]`` / ``kda_raw[k]`` index them for MLA layer ``m`` and KDA
    layer ``k`` (``pools.bind_raw_pools``). KDA heads are written in
    vLLM TP-rank order (``state_perm``, from the kernel's mesh order) and
    read back in mesh order.
    """

    def __init__(
        self,
        mesh,
        rank: int,
        raw_pools: list[torch.Tensor],
        mla_raw: list[int],
        kda_raw: list[int],
        state_perm: list[tuple[int, int]] | None,
    ) -> None:
        self.mesh, self.rank = mesh, rank
        self.raw_pools = raw_pools
        from vllm_torchtpu.megakernel.kimi_k3.layout import cache_layout

        self.layout = cache_layout(str(raw_pools[0].dtype))
        self.mla_raw, self._kda_raw = tuple(mla_raw), tuple(kda_raw)
        self._perm = (
            None
            if state_perm is None
            else tuple((int(a), int(b)) for a, b in state_perm)
        )
        self._ops: dict[Any, Any] = {}

    def mla_pools(self) -> list[torch.Tensor]:
        """The 24 MLA layers' pools, MLA layer m's at m."""
        return [self.raw_pools[self.mla_raw[m]] for m in range(MLA_LAYERS)]

    def _adopt(self, new) -> list[torch.Tensor]:
        new = [new] if isinstance(new, torch.Tensor) else list(new)
        for old, fresh in zip(self.raw_pools, new, strict=True):
            if fresh is not old:
                # Donated in place: keep the connector's tensor identity.
                old.set_(fresh)
        return self.raw_pools

    def copy_blocks(self, pairs: list[tuple[int, int]]) -> list[torch.Tensor]:
        """Block ``dst`` of every pool becomes a copy of block ``src``: vLLM's
        copy-on-write for a prefix hit ending inside a cached block (each rank
        copies its own slice of the shared block id). ``COPY_PAIRS`` pairs
        per call; spare slots copy the null block onto itself."""
        if "copy" not in self._ops:
            num_raw = len(self.raw_pools)

            def body(*args):
                raws, source, destination = (
                    args[:num_raw],
                    args[num_raw],
                    args[num_raw + 1],
                )
                return tuple(raw.at[destination].set(raw[source]) for raw in raws)

            self._ops["copy"] = register_op(
                f"pallas::k3_prefill_copy_blocks_r{num_raw}",
                self.mesh,
                body,
                (P(AXIS),) * num_raw + (P(), P()),
                (P(AXIS),) * num_raw,
                donate_argnums=tuple(range(num_raw)),
                export_key=repr(("copy_blocks", num_raw, COPY_PAIRS)),
            )
        for first in range(0, len(pairs), COPY_PAIRS):
            chunk = list(pairs[first : first + COPY_PAIRS])
            chunk += [(0, 0)] * (COPY_PAIRS - len(chunk))
            source = torch.tensor([a for a, _ in chunk], dtype=torch.int32).to("tpu")
            destination = torch.tensor([b for _, b in chunk], dtype=torch.int32).to(
                "tpu"
            )
            self._adopt(self._ops["copy"](*self.raw_pools, source, destination))
        return self.raw_pools

    def _write_kda(self, raws: list, recurrent, conv, kda_blocks, count: int) -> list:
        """(inside an op) ``count`` sequences' KDA states, ``[69, count,
        ...]``, into ``kda_blocks`` ``[count * 69]`` (each sequence's 69
        blocks in turn; unused sequences aim at the null block)."""
        from vllm_torchtpu.kernels import pool_adapters

        blocks = kda_blocks.reshape(count, KDA_LAYERS)
        for k in range(KDA_LAYERS):
            index, ids = self._kda_raw[k], blocks[:, k]
            ssm_shape = jax.eval_shape(
                lambda raw=raws[index], ids=ids: pool_adapters.gather_region(
                    raw,
                    ids,
                    tok0=0,
                    ntok=SSM_ROWS,
                    out_dtype=jnp.float32,
                    out_lanes=128,
                )
            ).shape
            ssm = recurrent[k].reshape(count, -1, 128)
            ssm = jnp.pad(ssm, ((0, 0), (0, ssm_shape[1] - ssm.shape[1]), (0, 0)))
            raws[index] = pool_adapters.scatter_region(
                raws[index], ssm, ids, tok0=0, ntok=SSM_ROWS
            )
            conv_shape = jax.eval_shape(
                lambda raw=raws[index], ids=ids: pool_adapters.gather_region(
                    raw, ids, tok0=SSM_ROWS, ntok=CONV_ROWS, out_dtype=jnp.bfloat16
                )
            ).shape
            taps = conv[k].reshape(count, -1).astype(jnp.bfloat16)
            taps = jnp.pad(
                taps, ((0, 0), (0, int(np.prod(conv_shape[1:])) - taps.shape[1]))
            )
            raws[index] = pool_adapters.scatter_region(
                raws[index],
                taps.reshape(conv_shape),
                ids,
                tok0=SSM_ROWS,
                ntok=CONV_ROWS,
            )
        return raws

    def write_packed(
        self,
        latents,
        recurrent,
        conv,
        recurrent_at,
        conv_at,
        table: torch.Tensor,
        kda_blocks: torch.Tensor,
        blocks_at: torch.Tensor,
        rings: tuple[list, list] | None = None,
    ) -> list[torch.Tensor]:
        """A packed window's writes (:meth:`PrefillProgram.run_packed`'s
        outputs): window page ``p``'s latent rows into block ``table[0, p]``
        (``[1, W / page]``, this rank's own table: the null block for pages it
        does not own or not to write), then sequence ``s``'s KDA states at its
        checkpoint into ``blocks_at[s * 69:(s + 1) * 69]`` and its final ones
        into ``kda_blocks[s * 69:...]``. Only the first ``len(kda_blocks) //
        69`` slots' states are written (:func:`live_slots`: the window's used
        ones, rounded up); the rest of the window's slots are unused. ``rings``:
        each of those sequences' DSpark ring ``(cache, meta)`` after the
        window and at its checkpoint (:meth:`PrefillRing.update`), written
        with the states."""
        window, slots = int(latents.shape[2]), int(recurrent.shape[2])
        count = int(kda_blocks.shape[0]) // KDA_LAYERS
        key = ("write", window, slots, count)
        if key not in self._ops:
            num_raw, mla_raw, perm, pages = (
                len(self.raw_pools),
                self.mla_raw,
                self._perm,
                window // self.layout.tokens_per_block,
            )

            def body(latents, recurrent, conv, recurrent_at, conv_at, *rest):
                raws = list(rest[:num_raw])
                table, kda_blocks, blocks_at = rest[num_raw:]
                table = table[0]
                latents = latents[0]
                # The live slots only: unused ones would go to the null block.
                recurrent, conv, recurrent_at, conv_at = (
                    x[0, :, :count] for x in (recurrent, conv, recurrent_at, conv_at)
                )
                if perm is not None:
                    recurrent, conv, recurrent_at, conv_at = (
                        jax.lax.ppermute(x, AXIS, perm)
                        for x in (recurrent, conv, recurrent_at, conv_at)
                    )
                for m in range(MLA_LAYERS):
                    index = mla_raw[m]
                    rows = (
                        jnp.pad(latents[m], ((0, 0), (0, 64)))
                        .astype(raws[index].dtype)
                        .reshape(pages, *self.layout.block_shape)
                    )

                    def write_page(p, pool, rows=rows):
                        return jax.lax.dynamic_update_index_in_dim(
                            pool,
                            jax.lax.dynamic_index_in_dim(rows, p, keepdims=False),
                            table[p],
                            axis=0,
                        )

                    raws[index] = jax.lax.fori_loop(0, pages, write_page, raws[index])
                # Checkpoints first (a sequence without one aims it at the null block).
                raws = self._write_kda(raws, recurrent_at, conv_at, blocks_at, count)
                raws = self._write_kda(raws, recurrent, conv, kda_blocks, count)
                return tuple(raws)

            self._ops[key] = register_op(
                f"pallas::k3_prefill_write_pool_packed{slots}_{count}_r{num_raw}_w{window}_ranked",
                self.mesh,
                body,
                (P(AXIS),) * (6 + num_raw) + (P(),) * 2,
                (P(AXIS),) * num_raw,
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
            )
        self._adopt(
            self._ops[key](
                latents,
                recurrent,
                conv,
                recurrent_at,
                conv_at,
                *self.raw_pools,
                table,
                kda_blocks,
                blocks_at,
            )
        )
        if rings is not None:
            after, at = rings
            self._write_rings(at[:count], blocks_at)
            self._write_rings(after[:count], kda_blocks)
        return self.raw_pools

    def _write_rings(self, rings: list, blocks: torch.Tensor) -> list[torch.Tensor]:
        """Each sequence's DSpark ring ``(cache, meta)`` into the draft rows of
        its 69 state blocks (``blocks`` ``[n * 69]``). Its own op: with the
        window's other writes, 16 sequences' rings would exceed the 64
        arguments a torch op takes."""
        count = len(rings)
        key = ("write_rings", count)
        if key not in self._ops:
            num_raw, perm = len(self.raw_pools), self._perm
            (dspark,) = kernel_modules("dspark")

            def body(*args):
                raws = list(args[:num_raw])
                blocks = args[num_raw]
                caches = args[num_raw + 1 : num_raw + 1 + count]
                metas = args[num_raw + 1 + count :]
                caches = jnp.stack([cache[0] for cache in caches])
                if perm is not None:
                    caches = jax.lax.ppermute(caches, AXIS, perm)
                packed = dspark.pack_ring(caches, jnp.stack(metas), KDA_LAYERS)
                return tuple(
                    write_draft_rings(
                        raws, self._kda_raw, packed, blocks.reshape(count, KDA_LAYERS)
                    )
                )

            self._ops[key] = register_op(
                f"pallas::k3_prefill_write_rings{count}_r{num_raw}_ranked",
                self.mesh,
                body,
                (P(AXIS),) * num_raw + (P(),) + (P(AXIS),) * count + (P(),) * count,
                (P(AXIS),) * num_raw,
                donate_argnums=tuple(range(num_raw)),
                export_key=repr(
                    ("write_rings", self.layout, count, self._kda_raw, self._perm)
                ),
            )
        return self._adopt(
            self._ops[key](
                *self.raw_pools,
                blocks,
                *(cache for cache, _ in rings),
                *(meta for _, meta in rings),
            )
        )

    def read_rings(
        self, draft_config: Any, kda_blocks: torch.Tensor, lengths: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """The DSpark rings in the draft rows of ``kda_blocks`` (``[n *
        69]``, a prefix hit's state blocks per sequence) for ``lengths``
        tokens, in mesh order: ``(cache, meta)`` per sequence -- an empty ring
        covering the length where the blocks hold none for it."""
        sequences = len(lengths)
        key = ("read_rings", sequences)
        if key not in self._ops:
            (dspark,) = kernel_modules("dspark")
            num_raw, kda_raw, c = len(self.raw_pools), self._kda_raw, draft_config
            perm = (
                None
                if self._perm is None
                else tuple((dst, src) for src, dst in self._perm)
            )

            def body(*args):
                raws, blocks, lengths = args[:num_raw], args[num_raw], args[num_raw + 1]
                packed = read_draft_rings(
                    raws,
                    kda_raw,
                    blocks.reshape(sequences, KDA_LAYERS),
                    dspark.RING_ROW_LANES,
                )
                out = [
                    dspark.unpack_ring(c, packed[s], lengths[s])
                    for s in range(sequences)
                ]
                caches = jnp.stack([cache for cache, _ in out])
                if perm is not None:
                    caches = jax.lax.ppermute(caches, AXIS, perm)
                return (
                    *(caches[s][None] for s in range(sequences)),
                    *(meta for _, meta in out),
                )

            self._ops[key] = register_op(
                f"pallas::k3_prefill_read_rings_r{num_raw}_s{sequences}",
                self.mesh,
                body,
                (P(AXIS),) * num_raw + (P(), P()),
                (P(AXIS),) * sequences + (P(),) * sequences,
                export_key=repr(
                    ("read_rings", sequences, kda_raw, perm, dataclasses.astuple(c))
                ),
            )
        out = self._ops[key](
            *self.raw_pools,
            kda_blocks,
            torch.tensor(lengths, dtype=torch.int32).to("tpu"),
        )
        return list(zip(out[:sequences], out[sequences:], strict=True))

    def read_states(
        self, kda_blocks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """KDA states: ``kda_blocks`` ``[n * 69]`` (each sequence's 69 blocks
        in turn) -> ``[1, 69, n, ...]`` recurrent and conv states in mesh
        order."""
        from vllm_torchtpu.kernels import pool_adapters

        sequences = int(kda_blocks.shape[0]) // KDA_LAYERS
        key = ("read", sequences)
        if key not in self._ops:
            num_raw, kda_raw = len(self.raw_pools), self._kda_raw
            perm = (
                None
                if self._perm is None
                else tuple((dst, src) for src, dst in self._perm)
            )

            def body(*args):
                raws, blocks = (
                    args[:num_raw],
                    args[num_raw].reshape(sequences, KDA_LAYERS),
                )
                recurrent, conv = [], []
                for k in range(KDA_LAYERS):
                    raw, ids = raws[kda_raw[k]], blocks[:, k]
                    ssm = pool_adapters.gather_region(
                        raw,
                        ids,
                        tok0=0,
                        ntok=SSM_ROWS,
                        out_dtype=jnp.float32,
                        out_lanes=128,
                    )
                    recurrent.append(ssm[:, : 3 * 128].reshape(-1, 3, 128, 128))
                    taps = pool_adapters.gather_region(
                        raw, ids, tok0=SSM_ROWS, ntok=CONV_ROWS, out_dtype=jnp.bfloat16
                    )
                    conv.append(
                        taps.reshape(sequences, -1)[:, : 3 * 3 * 3 * 128].reshape(
                            -1, 3, 3, 3, 128
                        )
                    )
                recurrent, conv = jnp.stack(recurrent), jnp.stack(conv)
                if perm is not None:
                    recurrent = jax.lax.ppermute(recurrent, AXIS, perm)
                    conv = jax.lax.ppermute(conv, AXIS, perm)
                return recurrent[None], conv[None]

            self._ops[key] = register_op(
                f"pallas::k3_prefill_read_states_r{num_raw}_s{sequences}",
                self.mesh,
                body,
                (P(AXIS),) * num_raw + (P(),),
                (P(AXIS),) * 2,
                export_key=repr(("read_states", sequences, self._kda_raw, perm)),
            )
        recurrent, conv = self._ops[key](*self.raw_pools, kda_blocks)
        return recurrent, conv


@dataclasses.dataclass
class _Request:
    """A request between its scheduler steps. Its earlier tokens' MLA pages
    are in the pool (the request owns them for its life); its KDA state is
    kept here -- vLLM's ``align`` mode frees a step's state block two steps
    later, before a skipped step's successor would read it."""

    req_id: str
    # tokens computed before its first step: their pages are in the pool already
    first_cached: int
    start: int  # tokens computed so far (block-aligned until the prompt is done)
    # KDA state after ``start``, [1, 69, ...]; None: zero.
    state: tuple[torch.Tensor, torch.Tensor] | None = None
    # Skipped partial steps: (end, KDA state blocks vLLM assigned the state after
    # ``end``).
    skipped: list = dataclasses.field(default_factory=list)
    # The DSpark ring after ``start`` (draft_ring.PrefillRing): (cache, meta).
    ring: tuple[torch.Tensor, torch.Tensor] | None = None


class KimiPrefiller:
    """The prefill role's program (:class:`MegakernelProgram`; see the module
    docstring)."""

    def __init__(
        self,
        config: MegakernelEngineConfig,
        presharded: Path,
        draft_checkpoint: Path | None = None,
        *,
        tp_rank: int,
    ) -> None:
        self.config = config
        self.page_tokens = config.block_size
        self.presharded = Path(presharded)
        self.cp_rank = tp_rank % config.decode_context_parallel_size
        budget = config.max_num_batched_tokens
        windows = prefill_windows(budget, config.max_model_len)
        # The most prefix tokens a window's sequences may hold in all (a
        # multiple of 4096), and a rank's share of them: 1/32 plus a page of
        # rounding slack per sequence, in multiples of 1024.
        capacity = max(windows[-1], -(-config.max_model_len // 4096) * 4096)
        self.cp_capacity = (
            -(-(capacity // DCP_SIZE + SLOTS * self.page_tokens) // 1024) * 1024
        )
        self.slots = min(config.max_num_seqs, SLOTS)
        self.program = PrefillProgram(windows, capacity, self.slots, self.page_tokens)
        self.rank = self.program.rank
        self._requests: dict[str, _Request] = {}
        self._zero_pair: tuple[torch.Tensor, torch.Tensor] | None = None
        self._writer: PoolWriter | None = None
        # The pool on the device of TP rank q must hold kernel rank q's KDA
        # heads: the inverse of the decode importer's permutation.
        perm = tp_rank_permutation(tp_rank, self.rank)
        self._state_perm = None if perm is None else [(dst, src) for src, dst in perm]
        self._mla_group = 0
        self._kda_groups: list[tuple[int, int]] = []
        self._zero_logits: torch.Tensor | None = None
        self._sampler: Any = None
        self._loaded = False
        # With the DSpark speculator: each request's draft ring, computed from
        # the windows' residual streams and written with its KDA state.
        self.ring = None
        if draft_checkpoint is not None:
            from vllm_torchtpu.megakernel.kimi_k3.draft_ring import PrefillRing

            self.ring = PrefillRing(
                self.program.mesh, self.rank, AXIS, draft_checkpoint, self.presharded
            )

    def load_weights(self) -> None:
        if self._loaded:
            return
        started = time.perf_counter()
        self.program.load_weights(self.presharded)
        if self.ring is not None:
            self.ring.load_weights()
        self._loaded = True
        if self.rank == 0:
            logger.info(
                "K3 prefill: weights resident in %.1f s",
                time.perf_counter() - started,
            )

    def bind_kv_caches(
        self, caches: Mapping[str, torch.Tensor], groups: Sequence[KVGroup]
    ) -> None:
        """Write vLLM's unified pools; ``groups`` say which block-table row
        holds a request's MLA pages and where each KDA layer's state blocks
        are."""
        self.load_weights()
        raw_pools, mla_raw, kda_raw, mla_names, kda_names = bind_raw_pools(
            caches, self.config.kv_cache_dtype
        )
        self._writer = PoolWriter(
            self.program.mesh, self.rank, raw_pools, mla_raw, kda_raw, self._state_perm
        )
        of = {name: g for g, group in enumerate(groups) for name in group.layer_names}
        mla_groups = {of[n] for n in mla_names if n in of}
        if len(mla_groups) != 1:
            raise RuntimeError(
                "K3 prefill: the MLA layers are in "
                f"KV cache groups {sorted(mla_groups)}"
            )
        self._mla_group = mla_groups.pop()
        missing = [n for n in kda_names if n not in of]
        if missing:
            raise RuntimeError(
                f"K3 prefill: KDA layers {missing[:3]}... are in no KV cache group"
            )
        self._kda_groups = [(of[n], int(groups[of[n]].block_size)) for n in kda_names]

    def _zeros(self) -> torch.Tensor:
        if self._zero_logits is None:
            self._zero_logits = torch.zeros((1, VOCAB), dtype=torch.float32).to("tpu")
        return self._zero_logits

    # ------------------------------------------------------------ steps
    def step(self, step_input: StepInput) -> StepOutput:
        """One prefill step. Requests that finish their prompt get their first
        token, chosen in the op (``[requests, 1]``; -1 for the others)."""
        if self._writer is None:
            raise RuntimeError(
                "K3 prefill: a request arrived before the KV pools were bound"
            )
        rows, copies = step_input.requests, step_input.copies
        if copies:
            # vLLM's copy-on-write block copies (a prefix hit ending inside a
            # cached block gets a private copy of it), before the windows that
            # read the copies as their prefix.
            self._writer.copy_blocks([(int(src), int(dst)) for src, dst in copies])
        for req in step_input.finished:
            self._requests.pop(req, None)
        largest = self.program.windows[-1]
        work, hits = [], []
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
                # New (or resumed after preemption): from a block-aligned hit, else
                # from token 0.
                state = _Request(
                    req, cached, cached if cached % self.page_tokens == 0 else 0
                )
                self._requests[req] = state
                if state.start:
                    hits.append((i, state))
            if skips_step(
                cached, total, prompt, state.start, largest, self.page_tokens * DCP_SIZE
            ):
                state.skipped.append(
                    (
                        total,
                        self._blocks(
                            kda_blocks_at_length(
                                row.block_tables, self._kda_groups, total
                            )
                        ),
                    )
                )
                continue
            stops = dict(state.skipped)
            state.skipped = []
            work.append(
                (
                    i,
                    state,
                    plan_pieces(
                        i, state.start, total, list(stops), largest, self.page_tokens
                    ),
                    stops,
                    total >= prompt,
                )
            )
        self._read_hit_states(rows, hits)
        logits: dict[int, torch.Tensor] = {}
        by_row = {i: (state, stops) for i, state, _, stops, _ in work}
        for wave in range(max((len(pieces) for _, _, pieces, _, _ in work), default=0)):
            pieces = [p[wave] for _, _, p, _, _ in work if len(p) > wave]
            last = {i for i, _, p, _, _ in work if len(p) == wave + 1}
            for window, members in pack_windows(
                pieces, self.program.windows, self.slots, self.program.capacity
            ):
                out = self._run_window(rows, by_row, window, members)
                for s, (piece, _) in enumerate(members):
                    if piece.row in last:
                        logits[piece.row] = out[s : s + 1]
        done = {i for i, _, _, _, finishes in work if finishes}
        for i, state, _, _, finishes in work:
            if finishes:
                del self._requests[state.req_id]
        return StepOutput(self._first_tokens(rows, logits, done))

    def _blocks(self, values: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(values, np.int32)).to("tpu")

    def _first_tokens(
        self,
        rows: Sequence[RequestInput],
        logits: dict[int, torch.Tensor],
        done: set[int],
    ) -> torch.Tensor:
        """The first token of each row that finished its prompt (``[rows,
        1]``, -1 elsewhere), chosen by :meth:`_sampler_op` on the rows'
        logits (padded to ``max_num_seqs`` rows: one compiled op)."""
        count = self.config.max_num_seqs
        if len(rows) > count:
            raise RuntimeError(
                f"K3 prefill: {len(rows)} rows exceed max_num_seqs {count}"
            )
        stacked = torch.cat(
            [logits.get(i, self._zeros()) for i in range(len(rows))]
            + [self._zeros()] * (count - len(rows))
        )
        params = np.zeros((count, 5), np.int32)
        params[:, 1] = np.float32(1).view(np.int32)
        for i, row in enumerate(rows):
            temperature, top_p, top_k, seed = row.sampling.as_tuple()
            params[i] = (
                np.float32(temperature).view(np.int32),
                np.float32(top_p).view(np.int32),
                top_k,
                seed & 0x7FFFFFFF,
                max(row.end - 1, 0),
            )
        sampled = any(row.sampling.temperature > 0 for row in rows)
        (tokens,) = self._sampler_op(sampled)(
            stacked, torch.from_numpy(params).to("tpu")
        )
        keep = torch.tensor([i in done for i in range(count)], dtype=torch.bool).to(
            "tpu"
        )
        return torch.where(keep, tokens, torch.full_like(tokens, -1))[
            : max(len(rows), 1), None
        ]

    def _sampler_op(self, sampled: bool) -> Any:
        """``(logits [n, VOCAB] replicated, params [n, 5]) -> tokens [n]``:
        greedy, or with ``sampled`` each row with a temperature drawn from
        its distribution (the decoder's in-op sampling, :mod:`sampling`, on
        its ``tp`` mesh: the same devices in the same order)."""
        key = ("sampler", sampled)
        if key in self.program._ops:
            return self.program._ops[key]
        from vllm_torchtpu.megakernel.kimi_k3 import sampling

        width = VOCAB // TP_SIZE

        def body(logits, params):
            rank = jax.lax.axis_index(sampling.AXIS)
            local = jax.lax.dynamic_slice_in_dim(logits, rank * width, width, axis=1)
            chosen = sampling.greedy(local)
            if sampled:
                temperature, top_p = _as_float(params[:, 0]), _as_float(params[:, 1])
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

        name = f"pallas::k3_prefill_first_token{'_sampled' if sampled else ''}"
        self.program._ops[key] = register_op(
            name,
            kernel_mesh(sampling.AXIS, TP_SIZE)[0],
            body,
            (P(), P()),
            (P(),),
            export_key=repr(key),
        )
        return self.program._ops[key]

    def _read_hit_states(self, rows: Sequence[RequestInput], hits: list) -> None:
        """The KDA states at new requests' prefix-cache hits, from the blocks
        vLLM's ``align`` mode keeps for them (up to ``slots`` per read)."""
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
                chunk, self.program.split_states(recurrent, conv)
            ):
                state.state = pair
            if self.ring is not None:
                assert self._writer is not None
                lengths = [state.start for _, state in chunk]
                lengths += [0] * (self.slots - len(chunk))
                rings = self._writer.read_rings(
                    self.ring.config, torch.cat(blocks), lengths
                )
                for (_, state), ring in zip(chunk, rings, strict=False):
                    state.ring = ring

    def _zero_blocks(self) -> torch.Tensor:
        return torch.zeros(KDA_LAYERS, dtype=torch.int32).to("tpu")

    def _zero_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Only ever an input of stack_states (not donated): one pair serves all.
        if self._zero_pair is None:
            self._zero_pair = self.program.zero_state()
        return self._zero_pair

    def _run_window(
        self, rows: Sequence[RequestInput], by_row, window: int, members
    ) -> torch.Tensor:
        """One packed window: run it on the members' prefixes, write each
        member's pages (from its first uncached one) and KDA states (final,
        and at its checkpoint), keep its state. Returns the ``[slots,
        VOCAB]`` logits (the window's slots)."""
        count, local = self.program.window_slots(window), window // 32
        tokens = np.full(window, -1, np.int32)
        offsets, lengths, checkpoints, bases, cached = [], [], [], [], []
        states, finals, ats = [], [], []
        base = 0
        for piece, offset in members:
            state, stops = by_row[piece.row]
            if piece.start % self.page_tokens:
                # Its window pages would land shifted in the pool: corrupt caches.
                raise RuntimeError(
                    f"K3 prefill: request {state.req_id} continues mid-page at "
                    f"token {piece.start}; windows need page-aligned chunks "
                    "(vllm_torchtpu's Mamba split patch aligns them)"
                )
            row = rows[piece.row]
            tokens[offset : offset + piece.length] = np.asarray(
                row.token_ids[piece.start : piece.end], np.int32
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
        # The pool writes cover the live slots only (live_slots).
        live = live_slots(len(members), count)
        finals += [self._zero_blocks()] * (live - len(members))
        ats += [self._zero_blocks()] * (live - len(members))
        ids, local_bases, write_table = self._cp_operands(rows, by_row, window, members)
        order = self.program.sp_order(window)  # the ranks' rows, rank-major
        ring_hook, ring_out = None, {}
        if self.ring is not None:
            positions = np.full(window, -1, np.int32)
            sequences = np.full(window, -1, np.int32)
            ends, stops, rings = [], [], []
            for s, (piece, offset) in enumerate(members):
                state, _ = by_row[piece.row]
                positions[offset : offset + piece.length] = np.arange(
                    piece.start, piece.end
                )
                sequences[offset : offset + piece.length] = s
                ends.append(piece.end)
                stops.append(-1 if piece.stop is None else piece.stop)
                if state.ring is None:
                    state.ring = self.ring.empty(piece.start)
                rings.append(state.ring)
            ends += [-1] * unused
            stops += [-1] * unused
            rings += [self.ring.zero()] * unused
            # The ring update takes rows in ownership order.
            positions, sequences = positions[order], sequences[order]

            def ring_hook(blocks, prefix, middle):
                ring_out["after"], ring_out["at"] = self.ring.update(
                    blocks, prefix, middle, positions, sequences, ends, stops, rings
                )

        recurrent, conv = self.program.stack_states(states)
        shard = (
            torch.from_numpy(tokens[order[self.rank * local : (self.rank + 1) * local]])
            .unsqueeze(0)
            .to("tpu")
        )
        extent = members[-1][1] + members[-1][0].length
        logits, latents, recurrent, conv, recurrent_at, conv_at = (
            self.program.run_packed(
                shard,
                _device_i32([extent]),
                _device_i32(offsets + lengths + checkpoints + bases + cached),
                recurrent,
                conv,
                self._writer.mla_pools(),
                ids,
                local_bases,
                ring_hook,
            )
        )
        self._writer.write_packed(
            latents,
            recurrent,
            conv,
            recurrent_at,
            conv_at,
            write_table,
            torch.cat(finals),
            torch.cat(ats),
            (ring_out["after"], ring_out["at"]) if ring_out else None,
        )
        for s, ((piece, _), pair) in enumerate(
            zip(members, self.program.split_states(recurrent, conv))
        ):
            state, _ = by_row[piece.row]
            state.state, state.start = pair, piece.end
            if ring_out:
                state.ring = ring_out["after"][s]
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
        self, rows: Sequence[RequestInput], by_row, window: int, members
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """This rank's context parallel operands for a packed window: its own
        prefix pages end to end (``[1, 1 + cp_capacity / page]``: count, then
        the block ids of global pages k with ``k % 32 == cp_rank``, at logical
        block ``k // 32``), each sequence's base and token count in that list
        (``[1, 2 * slots]``, the window's slots), and its page table for the
        window's writes (``[1, W / page]``: the pages it owns, the null block
        elsewhere)."""
        cp, rank = DCP_SIZE, self.cp_rank
        own, bases, counts, cursor = [], [], [], 0
        table = [0] * (window // self.page_tokens)
        for piece, offset in members:
            state, _ = by_row[piece.row]
            row = rows[piece.row].block_tables[self._mla_group]
            first = piece.start // self.page_tokens
            # Prefix pages: all whole.
            owned = [int(row[k // cp]) for k in range(rank, first, cp)]
            bases.append(cursor)
            counts.append(len(owned) * self.page_tokens)
            own += owned
            cursor += len(owned) * self.page_tokens
            # Cached pages: not rewritten.
            skip = max(state.first_cached - piece.start, 0) // self.page_tokens
            for p in range(skip, piece.rows // self.page_tokens):
                if (first + p) % cp == rank:
                    table[offset // self.page_tokens + p] = int(row[(first + p) // cp])
        unused = self.program.window_slots(window) - len(members)
        if cursor > self.cp_capacity:
            raise RuntimeError(
                f"K3 prefill: this rank's prefix share {cursor} "
                f"exceeds {self.cp_capacity}"
            )
        ids = [len(own)] + own + [0] * (self.cp_capacity // self.page_tokens - len(own))
        local = bases + [cursor] * unused + counts + [0] * unused
        # [1, n]: the rank axis the op bodies strip
        return _device_i32([ids]), _device_i32([local]), _device_i32([table])

    def warmup(self) -> None:
        """Compile every window's program and write, the hit-state read, the
        state stack/split, the block copy and the first-token samplers,
        issued together (torch_tpu compiles on its runtime threads, so they
        compile concurrently; the issue order is the same on every rank,
        which the communicating kernels need)."""
        writer = self._writer
        if writer is None:
            raise RuntimeError("K3 prefill: bind the KV caches before warming up")
        started = time.perf_counter()
        outputs = []
        for window in self.program.windows:
            count = self.program.window_slots(window)
            desc = (
                [0] + [window] * (count - 1) + [1] + [0] * (count - 1) + [0] * 3 * count
            )
            recurrent, conv = self.program.stack_states(
                [self.program.zero_state() for _ in range(count)]
            )
            ring_hook, ring_out = None, {}
            if self.ring is not None:
                none = np.full(window, -1, np.int32)

                def ring_hook(
                    blocks, prefix, middle, none=none, ring_out=ring_out, count=count
                ):
                    ring_out["after"], ring_out["at"] = self.ring.update(
                        blocks,
                        prefix,
                        middle,
                        none,
                        none,
                        [-1] * count,
                        [-1] * count,
                        [self.ring.zero()] * count,
                    )

            logits, latents, recurrent, conv, recurrent_at, conv_at = (
                self.program.run_packed(
                    torch.zeros((1, window // 32), dtype=torch.int32).to("tpu"),
                    _device_i32([1]),
                    _device_i32(desc),
                    recurrent,
                    conv,
                    writer.mla_pools(),
                    _rank_zeros_i32(1 + self.cp_capacity // self.page_tokens),
                    _rank_zeros_i32(2 * count),
                    ring_hook,
                )
            )
            outputs += [logits, self.program.split_states(recurrent, conv)[0][0]]
            for live in write_sizes(count):
                writer.write_packed(
                    latents,
                    recurrent,
                    conv,
                    recurrent_at,
                    conv_at,
                    _rank_zeros_i32(window // self.page_tokens),
                    _zeros_i32(live * KDA_LAYERS),
                    _zeros_i32(live * KDA_LAYERS),
                    (ring_out["after"], ring_out["at"]) if ring_out else None,
                )
        count = self.slots  # a hit-state read's sequences (_read_hit_states)
        outputs.append(writer.read_states(_zeros_i32(count * KDA_LAYERS))[0])
        if self.ring is not None:
            outputs.append(
                writer.read_rings(
                    self.ring.config, _zeros_i32(count * KDA_LAYERS), [0] * count
                )[0][0]
            )
        # Copy-on-write prefix hits (the null block onto itself).
        writer.copy_blocks([(0, 0)])
        stacked = torch.zeros(
            (self.config.max_num_seqs, VOCAB), dtype=torch.float32
        ).to("tpu")
        params = torch.zeros((self.config.max_num_seqs, 5), dtype=torch.int32).to("tpu")
        for sampled in (False, True):
            outputs.append(self._sampler_op(sampled)(stacked, params)[0])
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
