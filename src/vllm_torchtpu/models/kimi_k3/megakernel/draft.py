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
# src/vllm_torchtpu/megakernel/kimi_k3/draft.py
"""DSpark speculative decoding for the K3 decode megakernel.

The draft is ``kernels/dspark.py``'s fused Pallas kernel
(``RedHatAI/Kimi-K3-speculator.dspark``, a DFlash-style 5-layer block
drafter with a Markov head) on the decoder's 32-rank mesh. After a verify
(acceptance and the KDA state commit done), :meth:`DsparkDraft.step_single`
(seven drafts alone) or :meth:`DsparkDraft.step_batched` (three alone or
several requests) writes the verified rows' context into the draft's ring
and proposes the next drafts after the verifier's bonus token. The emitted
tokens are the target's, whatever the draft proposes, so a cold ring is
correct, only slower until it fills; after a P/D hand-off the decoder starts
from the ring the prefill computed (:meth:`DsparkDraft.from_ring`,
``draft_ring``).

State per speculating sequence (:class:`DraftState`): the ring ``[1, 5,
2048, 128]`` bf16 per rank, its positions ``[1, 2048]`` and the last output
vector ``[1, 128]`` (``kernels/dspark.py`` ``OUT_*``: drafts, targets, count,
bonus). The draft's kernel-layout shards
(``kernels/dspark.py`` ``save_kernel_shards``) are in the pre-shard directory,
``dspark-<the speculator revision's first 8 characters>``.
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger

from . import draft_mla
from .common.constants import AXIS_TP, GIB_BYTES, REV_PREFIX_LEN, ROPE_POS_SLACK
from .common.paths import require_local_path
from .draft_mla import BANK_FIELDS, DECISION_COLS, DRAFT_VECTOR_LEN
from .program import TP_SIZE, kernel_modules
from .runtime_shim.bridge import (
    OpRegistrationSpec,
    hbm,
    kernel_mesh,
    materialize,
    op_key_suffix,
    register_global_op,
    synchronize_tpu,
    upload,
)
from .runtime_shim.compilation import CompileCall

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path
    from types import ModuleType

    from .common.protocols import ControlLike, DsparkKernelProtocol
    from .decode_ops import BatchedDraftStepInputs, DraftStepInputs


logger = init_logger(__name__)

_SMALL_PART_ROWS = 4
_SMALL_PART_REQUESTS = 16
_LARGE_PART_REQUESTS = 8


@dataclasses.dataclass
class DraftState:
    """One speculating sequence's draft state on device."""

    cache: torch.Tensor  # [1, layers, window, 128] bf16, this rank's KV head
    positions: torch.Tensor  # [1, window] int32 (replicated), -1 = empty
    vector: torch.Tensor  # [1, 128] int32 (replicated): the last step's output


def _settled(state: DraftState) -> DraftState:
    """Materialize ``state`` tensors asynchronously on TPU devices.

    Ensures batched draft ops compile once rather than per mix of new and
    continuing states (torch_tpu otherwise compiles a tensor's deferred work
    into the graph of the next op taking it).
    """
    synchronize_tpu([state.cache, state.positions, state.vector])
    return state


def _batched_fn(
    dspark: ModuleType,
    fused: Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    control: ControlLike,
    names: tuple[str, ...],
    requests: int,
    rows_per: int,
    first: int = 0,
    verified: int | None = None,
) -> Callable[..., tuple[jax.Array, ...]]:
    """Build the batched draft op's global-view function.

    One ``fused`` call runs over the request dimension so draft weights stream
    once across all requests, followed by emitted token extraction and control
    updates. Wider verifies split across ``first .. first + requests - 1``.

    The states arrive stacked on a request axis (``caches`` ``[1, R, layers,
    window, 128]`` per rank, ``cache_positions`` ``[1, R, window]``,
    ``vectors`` ``[1, R, 128]``) and leave the same way. Module-level so tests
    can drive it with a stub ``fused``.
    """
    typed_dspark = cast("DsparkKernelProtocol", dspark)
    width, out_count, out_bonus = (
        int(typed_dspark.ROWS),
        int(typed_dspark.OUT_COUNT),
        int(typed_dspark.OUT_BONUS),
    )
    req_count, k = requests, rows_per
    total = (requests if verified is None else verified) * k
    rows, count = slice(first * k, (first + req_count) * k), len(names)

    def fn(*args: jax.Array) -> tuple[jax.Array, ...]:
        """Execute one fused batched draft call and advance control state."""
        weights = {name: args[i] for i, name in enumerate(names)}
        caches, cache_positions, vectors = (
            args[count],
            args[count + 1],
            args[count + 2],
        )
        aux, logits, ids, state, decisions = args[count + 3 :]
        tokens, positions = ids[:total][rows], ids[total : 2 * total][rows]
        indices = ids[3 * total : 4 * total][rows]
        aux, logits = aux[:, rows], logits[:, rows]
        decisions = decisions[first : first + req_count]
        plan, _ = control.fields(state, first + req_count)
        plan = plan[first:]
        bases = positions.reshape(req_count, k)[:, 0]
        forced_count, forced_anchor = decisions[:, 0], decisions[:, 1]
        # Each request's group padded to the kernel's full block width.
        aux_padded = aux.reshape(aux.shape[0], req_count, k, aux.shape[-1])
        aux_padded = jnp.pad(
            aux_padded,
            ((0, 0), (0, 0), (0, width - k), (0, 0)),
        )
        aux_padded = aux_padded.reshape(
            aux.shape[0],
            req_count * width,
            aux.shape[-1],
        )
        logits_padded = logits.reshape(
            logits.shape[0],
            req_count,
            k,
            logits.shape[-1],
        )
        logits_padded = jnp.pad(
            logits_padded,
            ((0, 0), (0, 0), (0, width - k), (0, 0)),
        )
        # make_fused_draft's singleton ABI omits the request axis. Keep
        # packing/splitting inside this op so returned Torch state buffers
        # retain the canonical shapes across request joins and departures.
        if req_count == 1:
            caches = caches[:, 0]
            cache_positions = cache_positions[:, 0]
            vectors = vectors[:, 0]
            logits_padded = logits_padded[:, 0]
        vectors_out, caches_out, positions_out = fused(
            weights,
            dspark.FusedDraftStepInputs(
                cache=caches,
                positions=cache_positions,
                aux=aux_padded,
                logits=logits_padded,
                previous=vectors,
                position=bases,
                forced_count=forced_count,
                forced_anchor=forced_anchor,
            ),
        )
        if req_count == 1:
            caches_out = caches_out[:, None]
            positions_out = positions_out[:, None]
        accepted_all = vectors_out[:, out_count]
        bonus_all = vectors_out[:, out_bonus]
        tokens_r = tokens.reshape(req_count, k)
        indices_r = indices.reshape(req_count, k)
        row = jnp.arange(width, dtype=jnp.int32)[None, :]
        emitted = jnp.where(
            row < accepted_all[:, None],
            tokens_r[:, jnp.minimum(jnp.arange(width) + 1, k - 1)],
            jnp.where(row == accepted_all[:, None], bonus_all[:, None], -1),
        ).astype(jnp.int32)
        slots = plan[:, control.SLOT]
        # Padding requests share slot zero. Match the sequential updates' last-writer
        # semantics explicitly; a scatter with duplicate indices would not guarantee
        # which request wins.
        writer = jnp.max(
            jnp.where(
                jnp.arange(control.slots)[:, None] == slots[None, :],
                jnp.arange(req_count)[None, :],
                -1,
            ),
            axis=1,
        )
        updates = jnp.stack(
            (
                bonus_all,
                bases + accepted_all + 1,
                jnp.take_along_axis(indices_r, accepted_all[:, None], axis=1)[:, 0],
            )
        )
        banks = jnp.where(
            writer[None, :] >= 0,
            updates[:, jnp.maximum(writer, 0)],
            state[: control.plan_start].reshape(BANK_FIELDS, control.slots),
        )
        plan = jnp.where(
            (jnp.arange(control.FIELDS)[None, :] == control.RESET)
            & (slots[:, None] != 0),
            0,
            plan,
        )
        state = jnp.concatenate(
            (
                banks.reshape(-1),
                state[control.plan_start : control.plan_start + first * control.FIELDS],
                plan.reshape(-1),
                state[control.plan_start + (first + req_count) * control.FIELDS :],
            )
        )
        return vectors_out[None], caches_out, positions_out, emitted, state

    return fn


def draft_class(
    checkpoint: str | Path,
) -> type[DsparkDraft | draft_mla.MlaDsparkDraft]:
    """Return the draft class matching the speculator ``checkpoint``.

    The draft implementation for the speculator checkpoint: the MLA draft
    (``Inferact/Kimi-K3-DSpark``, :mod:`draft_mla`) or :class:`DsparkDraft`
    (``RedHatAI/Kimi-K3-speculator.dspark``).
    """
    checkpoint_path = require_local_path(checkpoint, label="checkpoint")
    (mla,) = kernel_modules("dspark_mla")
    if mla.is_mla_draft(checkpoint_path):
        return draft_mla.MlaDsparkDraft
    return DsparkDraft


class DsparkDraft:
    """The fused DSpark draft: weights, per-sequence rings, and step ops."""

    uses_unified_pools: bool = False

    def __init__(
        self,
        max_position: int,
        checkpoint: str | Path,
        presharded: Path | str,
        synthetic_rates: tuple[float, ...] | None = None,
        synthetic_rates_by_width: Mapping[int, Sequence[float]] | None = None,
    ) -> None:
        """Initialize DSpark drafter from ``checkpoint`` and ``presharded``."""
        checkpoint_path = require_local_path(checkpoint, label="checkpoint")
        presharded_path = require_local_path(presharded, label="presharded")
        (self.dspark,) = kernel_modules("dspark")
        # vLLM's synthetic_acceptance_rates: the op then accepts a drawn number of
        # drafts instead of the leading drafts that match the target.
        # synthetic_rates_by_width gives each draft width its own rates.
        self.synthetic_rates = synthetic_rates
        self.synthetic_rates_by_width = synthetic_rates_by_width
        self.config = self.dspark.DraftConfig.from_checkpoint(checkpoint_path)
        self.mesh, self.rank = kernel_mesh(AXIS_TP, TP_SIZE)
        if self.rank == 0:
            for width, rates in (synthetic_rates_by_width or {}).items():
                logger.info(
                    "K3 DSpark: synthetic acceptance with %d drafts, rates %s "
                    "(mean length %.2f)",
                    width,
                    list(rates),
                    1 + sum(rates),
                )
            if synthetic_rates is not None and synthetic_rates_by_width is None:
                logger.info(
                    "K3 DSpark: synthetic acceptance, rates %s (mean length %.2f)",
                    list(synthetic_rates),
                    1 + sum(synthetic_rates),
                )
        self.names: tuple[str, ...] = tuple(self.dspark.WEIGHT_NAMES)
        self.max_position = int(max_position)
        self.weights: dict[str, torch.Tensor] = {}
        rev = checkpoint_path.resolve().name[:REV_PREFIX_LEN]
        self._presharded = presharded_path / f"dspark-{rev}"
        self._ops: dict[
            tuple[object, ...], Callable[..., tuple[torch.Tensor, ...]]
        ] = {}
        self._aux_layers = tuple(int(layer) for layer in self.config.aux_layers)

    def _device(self, anchor: int | torch.Tensor | None = None) -> torch.device | str:
        """Return resident device of ``anchor`` or loaded weights, or ``"tpu"``."""
        if isinstance(anchor, torch.Tensor):
            return anchor.device
        if self.weights:
            return next(iter(self.weights.values())).device
        return "tpu"

    @property
    def aux_layers(self) -> tuple[int, ...]:
        """Return target model layer indices tapped for draft context."""
        return self._aux_layers

    def load_weights(self) -> float:
        """Load this rank's kernel-layout shard and regenerate RoPE tables.

        This rank's kernel-layout shard (``kernels/dspark.py``
        ``save_kernel_shards``) with RoPE tables regenerated for
        ``max_position`` (the stored ones cover 16k positions).
        """
        started = time.perf_counter()
        directory = self._presharded
        index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        if (
            index.get("format") != self.dspark.KERNEL_SHARD_FORMAT
            or int(index.get("ranks", 0)) != TP_SIZE
        ):
            msg = f"{directory}: not a 32-rank DSpark kernel shard set"
            raise ValueError(msg)
        raw = np.fromfile(directory / f"rank{self.rank:02d}.bin", dtype=np.uint8)
        rope_q, rope_kv = self.dspark.rope_tables(
            self.config,
            self.max_position + ROPE_POS_SLACK,
        )
        for name in self.names:
            if name == "rope_q":
                value = rope_q
            elif name == "rope_kv":
                value = rope_kv
            else:
                spec = index["arrays"][name]
                dtype = (
                    np.dtype(spec["dtype"])
                    if spec["dtype"] != "bfloat16"
                    else np.dtype(jnp.bfloat16)
                )
                count = int(np.prod(spec["shape"]))
                offset = int(spec["offset"])
                value = (
                    raw[offset : offset + count * dtype.itemsize]
                    .view(dtype)
                    .reshape(spec["shape"])
                )
            self.weights[name] = upload(value)
        materialize(self.weights[self.names[-1]])
        wall = time.perf_counter() - started
        if self.rank == 0:
            gib = (
                sum(t.numel() * t.element_size() for t in self.weights.values())
                / GIB_BYTES
            )
            logger.info(
                "K3 DSpark: draft weights resident (%.2f GiB per rank, RoPE to %d) "
                "in %.1f s; HBM %s",
                gib,
                self.max_position,
                wall,
                hbm(self.rank),
            )
        return wall

    def new_state(self, anchor: int | torch.Tensor) -> DraftState:
        """Return a cold draft state seeded with ``anchor``.

        The next verify starts with ``anchor`` followed by ``anchor`` repeated
        as placeholder drafts that are rejected on the first verify.
        """
        cfg = self.config
        dev = self._device(anchor)
        cache = torch.zeros(
            (1, cfg.layers, cfg.window, 2 * cfg.head_dim),
            dtype=torch.bfloat16,
            device=dev,
        )
        positions = torch.full((1, cfg.window), -1, dtype=torch.int32, device=dev)
        return _settled(DraftState(cache, positions, self._vector(anchor)))

    def from_ring(
        self,
        cache: torch.Tensor,
        positions: torch.Tensor,
        anchor: int | torch.Tensor,
    ) -> DraftState:
        """Construct a draft state over a prefill-computed ring and ``anchor``.

        ``cache`` (``[1, layers, window, 128]``) and ``positions``
        (``[1, window]``) come from ``KdaImporter.import_ring``; the next
        verify starts with ``anchor`` and placeholder drafts.
        """
        return _settled(DraftState(cache, positions, self._vector(anchor)))

    def _vector(self, anchor: int | torch.Tensor) -> torch.Tensor:
        """Allocate and seed a ``[1, 128]`` draft output vector with ``anchor``."""
        # Built on the device either way, then settled with its state: the
        # draft ops compile per operand history (see _settled).
        dev = self._device(anchor)
        vector = torch.full((1, DRAFT_VECTOR_LEN), -1, dtype=torch.int32, device=dev)
        anchor_t = (
            anchor
            if isinstance(anchor, torch.Tensor)
            else torch.tensor([int(anchor)], dtype=torch.int32, device=dev)
        )
        anchor_t = anchor_t.reshape(1).to(torch.int32)
        vector[0, : self.dspark.ROWS - 1] = anchor_t
        vector[0, self.dspark.OUT_BONUS] = anchor_t[0]
        # Indexed writes leave a flat backing buffer even after _settled; the clone has
        # draft outputs' [1, 128] backing, so mixes of new and continuing states share
        # an executable.
        return vector.clone()

    def synthetic_rates_for(self, drafts: int) -> tuple[float, ...] | None:
        """Return synthetic acceptance rates configured for ``drafts`` width."""
        if self.synthetic_rates_by_width is not None:
            return tuple(self.synthetic_rates_by_width[drafts])
        return self.synthetic_rates

    def _build_single(
        self,
        control: ControlLike,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build or return the single-request fused draft custom op."""
        key = ("owned", control.key())
        if key in self._ops:
            return self._ops[key]

        dspark, rows = self.dspark, self.dspark.ROWS
        fused = dspark.make_fused_draft(self.config, self.mesh, rows, requests=1)
        names = self.names
        count = len(names)

        def fn(*args: jax.Array) -> tuple[jax.Array, ...]:
            """Execute single-request fused draft and advance control state."""
            weights = {name: args[i] for i, name in enumerate(names)}
            (
                cache,
                positions,
                aux,
                logits,
                previous,
                ids,
                state,
                decisions,
            ) = args[count:]
            tokens, base, slot = ids[:rows], ids[rows], ids[2 * rows]
            forced_count, forced_anchor = decisions[0, 0], decisions[0, 1]
            vector, cache, positions = fused(
                weights,
                dspark.FusedDraftStepInputs(
                    cache=cache,
                    positions=positions,
                    aux=aux,
                    logits=logits,
                    previous=previous,
                    position=base[None],
                    forced_count=forced_count[None],
                    forced_anchor=forced_anchor[None],
                ),
            )
            accepted = vector[0, dspark.OUT_COUNT]
            bonus = vector[0, dspark.OUT_BONUS]
            row = jnp.arange(rows, dtype=jnp.int32)
            emitted = jnp.where(
                row < accepted,
                tokens[jnp.minimum(row + 1, rows - 1)],
                jnp.where(row == accepted, bonus, -1),
            ).astype(jnp.int32)
            following = jnp.concatenate(
                (
                    bonus[None],
                    vector[0, dspark.OUT_DRAFTS : dspark.OUT_DRAFTS + rows - 1],
                )
            )
            following_ids = jnp.concatenate(
                (
                    following,
                    base + accepted + 1 + row,
                    ids[2 * rows : 3 * rows],
                    accepted[None],
                    ids[3 * rows + 1 :],
                )
            ).astype(jnp.int32)
            state = state.at[slot].set(bonus)
            state = state.at[control.slots + slot].set(base + accepted + 1)
            state = state.at[2 * control.slots + slot].set(accepted)
            reset = control.plan_start + control.RESET
            state = state.at[reset].set(jnp.where(slot != 0, 0, state[reset]))
            return vector, cache, positions, emitted[None], following_ids, state

        in_specs = (P(AXIS_TP),) * count + (
            P(AXIS_TP),
            P(),
            P(),
            P(AXIS_TP),
            P(),
            P(),
            P(),
            P(),
        )
        out_specs = (P(), P(AXIS_TP), P(), P(), P(), P())
        export_key_tuple = (
            "dspark_draft_owned_rq",
            control.key(),
            names,
            self.max_position,
            dataclasses.astuple(self.config),
        )
        self._ops[key] = register_global_op(
            f"pallas::k3_dspark_draft_owned_{op_key_suffix(export_key_tuple)}",
            self.mesh,
            fn,
            OpRegistrationSpec(
                in_specs=in_specs,
                out_specs=out_specs,
                donate_argnums=(count,),
                export_key=repr(export_key_tuple),
            ),
        )
        return self._ops[key]

    def _build_metadata(
        self,
        requests: int,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build or return the metadata-packing op for large draft batches.

        Pack replicated metadata before the large-batch draft boundary.

        Sixteen caches/positions/vectors plus weights exceed Torch's
        64-argument limit; two packed metadata tensors keep that op below it,
        while its outputs stay canonical per-request buffers.
        """
        key = ("metadata", requests, self.config.window)
        if key not in self._ops:

            def fn(*args: jax.Array) -> tuple[jax.Array, jax.Array]:
                """Stack per-request positions and vectors along axis 1."""
                return (
                    jnp.stack(args[:requests], axis=1),
                    jnp.stack(args[requests:], axis=1),
                )

            export_key_tuple = ("dspark_metadata", requests, self.config.window)
            self._ops[key] = register_global_op(
                f"pallas::k3_dspark_metadata_r{requests}_{op_key_suffix(export_key_tuple)}",
                self.mesh,
                fn,
                OpRegistrationSpec(
                    in_specs=(P(),) * (2 * requests),
                    out_specs=(P(), P()),
                    export_key=repr(export_key_tuple),
                ),
            )
        return self._ops[key]

    @staticmethod
    def parts(requests: int, rows_per: int) -> list[tuple[int, int]]:
        """Return ``(first, count)`` slices for each fused draft call.

        The fused kernel drafts at most 16 requests of up to four rows, or
        8 requests of more rows, per call.
        """
        size = (
            _SMALL_PART_REQUESTS
            if rows_per <= _SMALL_PART_ROWS
            else _LARGE_PART_REQUESTS
        )
        return [
            (first, min(size, requests - first)) for first in range(0, requests, size)
        ]

    def _build_batched(
        self,
        control: ControlLike,
        requests: int,
        rows_per: int,
        first: int = 0,
        verified: int | None = None,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build or return the batched fused draft custom op.

        The draft step after a batched verify (``DecodeOps.
        verify_batched``): one fused kernel call over the request dimension
        (the weights stream once for all requests; ``_batched_fn``), then each
        request's control entry advanced. ``first`` / ``verified``: this
        call's part of a wider verify (``parts``).
        """
        verified_count = requests if verified is None else verified
        key = ("batched", control.key(), requests, rows_per, first, verified_count)
        if key in self._ops:
            return self._ops[key]
        fused = self.dspark.make_fused_draft(
            self.config,
            self.mesh,
            rows_per,
            requests=requests,
        )
        batched_fn = _batched_fn(
            self.dspark,
            fused,
            control,
            self.names,
            requests,
            rows_per,
            first,
            verified_count,
        )
        count = len(self.names)

        def fn(*args: jax.Array) -> tuple[jax.Array, ...]:
            """Unpack per-request buffers, invoke ``batched_fn``, and split outputs."""
            req_count = requests
            if req_count > _LARGE_PART_REQUESTS:
                positions, previous = (
                    args[count + req_count],
                    args[count + req_count + 1],
                )
                tail = args[count + req_count + 2 :]
            else:
                positions = jnp.stack(
                    args[count + req_count : count + 2 * req_count],
                    axis=1,
                )
                previous = jnp.stack(
                    args[count + 2 * req_count : count + 3 * req_count],
                    axis=1,
                )
                tail = args[count + 3 * req_count :]
            vectors, caches, positions, emitted, state = batched_fn(
                *args[:count],
                jnp.stack(args[count : count + req_count], axis=1),
                positions,
                previous,
                *tail,
            )
            return (
                *(vectors[:, j] for j in range(req_count)),
                *(caches[:, j] for j in range(req_count)),
                *(positions[:, j] for j in range(req_count)),
                emitted,
                state,
            )

        part = "" if verified_count == requests else f"_part{first}of{verified_count}"
        export_key_tuple = (
            "dspark_draft_rbatched",
            control.key(),
            requests,
            rows_per,
            first,
            verified_count,
            self.names,
            self.max_position,
            dataclasses.astuple(self.config),
        )
        self._ops[key] = register_global_op(
            f"pallas::k3_dspark_draft_rbatch{requests}x{rows_per}{part}_{op_key_suffix(export_key_tuple)}",
            self.mesh,
            fn,
            OpRegistrationSpec(
                in_specs=(P(AXIS_TP),) * count
                + (P(AXIS_TP),) * requests
                + (P(),) * (2 if requests > _LARGE_PART_REQUESTS else 2 * requests)
                + (P(), P(AXIS_TP), P(), P(), P()),
                out_specs=(P(),) * requests
                + (P(AXIS_TP),) * requests
                + (P(),) * requests
                + (P(), P()),
                donate_argnums=tuple(range(count, count + requests)),
                export_key=repr(export_key_tuple),
            ),
        )
        return self._ops[key]

    def compilation_calls(
        self,
        control: ControlLike,
        spec_shapes: Sequence[tuple[int, int]],
        *,
        spec_alone: bool = False,
    ) -> list[CompileCall]:
        """Return precompilation calls for configured draft shapes.

        Fresh vectors and returned vectors have the same backing shape, so
        these graphs also cover new/continuing state mixtures. The ordinary
        warmup still validates every transition with real donated buffers.
        """
        cfg = self.config
        weights = tuple(self.weights[n] for n in self.names)
        cache = torch.empty(
            (1, cfg.layers, cfg.window, 2 * cfg.head_dim),
            dtype=torch.bfloat16,
        )
        positions = torch.empty((1, cfg.window), dtype=torch.int32)
        vector = torch.empty((1, DRAFT_VECTOR_LEN), dtype=torch.int32)
        state = torch.empty(control.size, dtype=torch.int32)
        drafted = {
            count
            for r, k in spec_shapes
            for _, count in self.parts(r, k)
            if count > _LARGE_PART_REQUESTS
        }
        calls: list[CompileCall] = [
            CompileCall(
                self._build_metadata(requests),
                ((positions,) * requests + (vector,) * requests),
                {},
                priority=0,
            )
            for requests in sorted(drafted)
        ]

        def tail(
            rows: int,
            ids_size: int,
            requests: int,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            """Build dummy tail operands for precompilation."""
            return (
                torch.empty(
                    (len(self.aux_layers), rows, cfg.hidden),
                    dtype=torch.bfloat16,
                ),
                torch.empty((1, rows, cfg.vocab // TP_SIZE), dtype=torch.float32),
                torch.empty(ids_size, dtype=torch.int32),
                state,
                torch.empty((requests, DECISION_COLS), dtype=torch.int32),
            )

        if spec_alone:
            rows = self.dspark.ROWS
            aux, logits, ids, state_t, decisions = tail(
                rows,
                3 * rows + 1 + rows * control.pages,
                1,
            )
            calls.append(
                CompileCall(
                    self._build_single(control),
                    (
                        *weights,
                        cache,
                        positions,
                        aux,
                        logits,
                        vector,
                        ids,
                        state_t,
                        decisions,
                    ),
                    {},
                    priority=2 * rows,
                )
            )
        for verified, spec_rows in spec_shapes:
            rows = verified * spec_rows
            for first, requests in self.parts(verified, spec_rows):
                calls.append(
                    CompileCall(
                        self._build_batched(
                            control,
                            requests,
                            spec_rows,
                            first,
                            verified,
                        ),
                        (
                            *weights,
                            *((cache,) * requests),
                            *(
                                (
                                    torch.empty(
                                        (1, requests, cfg.window),
                                        dtype=torch.int32,
                                    ),
                                    torch.empty(
                                        (1, requests, DRAFT_VECTOR_LEN),
                                        dtype=torch.int32,
                                    ),
                                )
                                if requests > _LARGE_PART_REQUESTS
                                else (positions,) * requests + (vector,) * requests
                            ),
                            *tail(
                                rows,
                                4 * rows + verified * control.pages,
                                verified,
                            ),
                        ),
                        {},
                        priority=2 * rows,
                    )
                )
        return calls

    def step_batched(
        self,
        states: Sequence[DraftState],
        inputs: BatchedDraftStepInputs,
    ) -> tuple[list[DraftState], torch.Tensor, torch.Tensor]:
        """Run batched draft step after a batched verify.

        Uses packed verify operand ``inputs.ids`` and accepted-count/bonus
        ``inputs.decisions`` (``[R, 2]``). Packing and splitting happen inside
        the JAX op so each returned Torch state owns a canonical per-request
        buffer. Returns ``(states, tokens [R, 8], control_state)`` with emitted
        tokens padded by ``-1``.
        """
        verified = len(states)
        new: list[DraftState] = []
        emitted: list[torch.Tensor] = []
        control_state = inputs.control_state
        for first, req_count in self.parts(verified, inputs.rows_per):
            part = states[first : first + req_count]
            op = self._build_batched(
                inputs.control,
                req_count,
                inputs.rows_per,
                first,
                verified,
            )
            metadata: tuple[torch.Tensor, ...] = tuple(
                s.positions for s in part
            ) + tuple(s.vector for s in part)
            if req_count > _LARGE_PART_REQUESTS:
                metadata = self._build_metadata(req_count)(*metadata)
            out = op(
                *(self.weights[n] for n in self.names),
                *(s.cache for s in part),
                *metadata,
                inputs.aux,
                inputs.logits,
                inputs.ids,
                control_state,
                inputs.decisions,
            )
            new += [
                _settled(DraftState(out[req_count + j], out[2 * req_count + j], out[j]))
                for j in range(req_count)
            ]
            emitted.append(out[-2])
            control_state = out[-1]
        emitted_cat = emitted[0] if len(emitted) == 1 else torch.cat(emitted)
        return new, emitted_cat, control_state

    def step_single(
        self,
        state: DraftState,
        inputs: DraftStepInputs,
    ) -> tuple[DraftState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run single-request draft step after a single verify.

        Drafts after the verify whose packed operand was ``inputs.ids`` with
        accepted count and bonus in ``inputs.decisions`` (``[1, 2]``). Returns
        ``(state, tokens [1, 8], following_ids, control_state)`` where emitted
        tokens are padded with ``-1`` and ``control_state`` records the slot's
        next token, position, and state row.
        """
        op = self._build_single(inputs.control)
        vector, cache, positions, emitted, following, next_control = op(
            *(self.weights[n] for n in self.names),
            state.cache,
            state.positions,
            inputs.aux,
            inputs.logits,
            state.vector,
            inputs.ids,
            inputs.control_state,
            inputs.decisions,
        )
        settled = _settled(DraftState(cache, positions, vector))
        return settled, emitted, following, next_control
