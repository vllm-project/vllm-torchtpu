# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculative decoding for the K3 decode megakernel.

The draft is ``kernels.dspark``'s fused Pallas kernel
(``RedHatAI/Kimi-K3-speculator.dspark``, a DFlash-style 5-layer block drafter
with a Markov head) on the decoder's 32-rank mesh. After a verify (acceptance
and the KDA state commit done), :meth:`DsparkDraft.step_single` (seven drafts
alone) or :meth:`DsparkDraft.step_batched` (three alone or several requests)
writes the verified rows' context into the draft's ring and proposes the next
drafts after the verifier's bonus token. The emitted tokens are the target's,
whatever the draft proposes, so a cold ring is correct, only slower until it
fills; after a P/D hand-off the decoder starts from the ring the prefill
computed (:meth:`DsparkDraft.from_ring`, ``draft_ring``).

State per speculating sequence (:class:`DraftState`): the ring ``[1, 5,
2048, 128]`` bf16 per rank, its positions ``[1, 2048]`` and the last output
vector ``[1, 128]`` (``kernels.dspark.OUT_*``: drafts, targets, count, bonus).
The draft's kernel-layout shards (``kernels.dspark.save_kernel_shards``) are in
the pre-shard directory, ``dspark-<the speculator revision's first 8
characters>``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.bridge import (
    hbm,
    kernel_mesh,
    materialize,
    register_global_op,
    upload,
)
from vllm_torchtpu.megakernel.kimi_k3.program import TP_SIZE, kernel_modules

logger = init_logger(__name__)

AXIS = "tp"


@dataclasses.dataclass
class DraftState:
    """One speculating sequence's draft state (device tensors)."""

    cache: torch.Tensor  # [1, layers, window, 128] bf16, this rank's KV head
    positions: torch.Tensor  # [1, window] int32 (replicated), -1 = empty
    vector: torch.Tensor  # [1, 128] int32 (replicated): the last step's output


def _settled(state: DraftState) -> DraftState:
    """``state`` with its tensors materialized, so batched draft ops compile
    once rather than per mix of new and continuing states (torch_tpu otherwise
    compiles a tensor's deferred work into the graph of the next op taking it)."""
    from torch_tpu._internal.sync import synchronize

    synchronize([state.cache, state.positions, state.vector], wait=False)
    return state


def _batched_fn(
    dspark: Any,
    fused: Any,
    control: Any,
    names: tuple[str, ...],
    requests: int,
    rows_per: int,
    first: int = 0,
    verified: int | None = None,
) -> Any:
    """The batched draft op's global-view function: one ``fused`` call over
    the request dimension (the draft weights stream once for all requests),
    then each request's emitted tokens and control entry. The final operand
    holds the verifier's decisions ``[R, 2]`` (accepted count, bonus token).
    A verify of ``verified`` requests wider than one fused call drafts in
    parts: this one takes its requests ``first .. first + requests - 1``
    (their verify rows, decisions and plan rows).

    The states arrive stacked on a request axis (``caches`` ``[1, R, layers,
    window, 128]`` per rank, ``cache_positions`` ``[1, R, window]``,
    ``vectors`` ``[1, R, 128]``) and leave the same way. Module-level so tests
    can drive it with a stub ``fused``."""
    width, R, k = dspark.ROWS, requests, rows_per
    total = (requests if verified is None else verified) * k
    rows = slice(first * k, (first + R) * k)
    count = len(names)

    def fn(*args):
        import jax.numpy as jnp

        weights = {name: args[i] for i, name in enumerate(names)}
        caches, cache_positions, vectors = args[count], args[count + 1], args[count + 2]
        aux, logits, ids, state, decisions = args[count + 3 :]
        tokens, positions = ids[:total][rows], ids[total : 2 * total][rows]
        indices = ids[3 * total : 4 * total][rows]
        aux, logits = aux[:, rows], logits[:, rows]
        decisions = decisions[first : first + R]
        plan, _ = control.fields(state, first + R)
        plan = plan[first:]
        bases = positions.reshape(R, k)[:, 0]
        forced_count, forced_anchor = decisions[:, 0], decisions[:, 1]
        # Each request's group padded to the kernel's full block width.
        aux_padded = aux.reshape(aux.shape[0], R, k, aux.shape[-1])
        aux_padded = jnp.pad(aux_padded, ((0, 0), (0, 0), (0, width - k), (0, 0)))
        aux_padded = aux_padded.reshape(aux.shape[0], R * width, aux.shape[-1])
        logits_padded = logits.reshape(logits.shape[0], R, k, logits.shape[-1])
        logits_padded = jnp.pad(logits_padded, ((0, 0), (0, 0), (0, width - k), (0, 0)))
        # make_fused_draft's singleton ABI omits the request axis. Keep
        # packing/splitting inside this op so returned Torch state buffers
        # retain the canonical shapes across request joins and departures.
        if R == 1:
            caches = caches[:, 0]
            cache_positions = cache_positions[:, 0]
            vectors = vectors[:, 0]
            logits_padded = logits_padded[:, 0]
        vectors_out, caches_out, positions_out = fused(
            weights,
            caches,
            cache_positions,
            aux_padded,
            logits_padded,
            vectors,
            bases,
            forced_count,
            forced_anchor,
        )
        if R == 1:
            caches_out = caches_out[:, None]
            positions_out = positions_out[:, None]
        accepted_all = vectors_out[:, dspark.OUT_COUNT]  # [R]
        bonus_all = vectors_out[:, dspark.OUT_BONUS]  # [R]
        tokens_r = tokens.reshape(R, k)
        indices_r = indices.reshape(R, k)
        row = jnp.arange(width, dtype=jnp.int32)[None, :]
        emitted = jnp.where(
            row < accepted_all[:, None],
            tokens_r[:, jnp.minimum(jnp.arange(width) + 1, k - 1)],
            jnp.where(row == accepted_all[:, None], bonus_all[:, None], -1),
        ).astype(jnp.int32)
        slots = plan[:, control.SLOT]
        # Padding requests share slot zero. Match the sequential updates'
        # last-writer semantics explicitly; a scatter with duplicate indices
        # would not guarantee which request wins.
        writer = jnp.max(
            jnp.where(
                jnp.arange(control.slots)[:, None] == slots[None, :],
                jnp.arange(R)[None, :],
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
            state[: control.plan_start].reshape(3, control.slots),
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
                state[control.plan_start + (first + R) * control.FIELDS :],
            )
        )
        return vectors_out[None], caches_out, positions_out, emitted, state

    return fn


def draft_class(checkpoint) -> type:
    """The draft implementation for the speculator checkpoint: the MLA draft
    (``Inferact/Kimi-K3-DSpark``, :mod:`draft_mla`) or :class:`DsparkDraft`
    (``RedHatAI/Kimi-K3-speculator.dspark``)."""
    (mla,) = kernel_modules("dspark_mla")
    if mla.is_mla_draft(Path(checkpoint)):
        from vllm_torchtpu.megakernel.kimi_k3.draft_mla import MlaDsparkDraft

        return MlaDsparkDraft
    return DsparkDraft


class DsparkDraft:
    """The fused DSpark draft: weights, per-sequence rings and the step op."""

    def __init__(
        self,
        max_position: int,
        checkpoint: str,
        presharded: Path,
        synthetic_rates: tuple[float, ...] | None = None,
        synthetic_rates_by_width: dict[int, tuple[float, ...]] | None = None,
    ):
        (self.dspark,) = kernel_modules("dspark")
        # vLLM's synthetic_acceptance_rates: the op then accepts a drawn number
        # of drafts instead of the leading drafts that match the target.
        # synthetic_rates_by_width gives each draft width its own rates.
        self.synthetic_rates = synthetic_rates
        self.synthetic_rates_by_width = synthetic_rates_by_width
        self.config = self.dspark.DraftConfig.from_checkpoint(checkpoint)
        self.mesh, self.rank = kernel_mesh(AXIS, TP_SIZE)
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
        self.names = tuple(self.dspark.WEIGHT_NAMES)
        self.max_position = int(max_position)
        self.weights: dict[str, torch.Tensor] = {}
        self._presharded = (
            Path(presharded) / f"dspark-{Path(checkpoint).resolve().name[:8]}"
        )
        self._ops: dict[Any, Any] = {}
        self._aux_layers = tuple(int(layer) for layer in self.config.aux_layers)

    @property
    def aux_layers(self) -> tuple[int, ...]:
        return self._aux_layers

    def load_weights(self) -> float:
        """This rank's kernel-layout shard (``kernels.dspark.save_kernel_shards``)
        with RoPE tables regenerated for ``max_position`` (the stored ones
        cover 16k positions)."""
        import time

        started = time.perf_counter()
        directory = self._presharded
        index = json.loads((directory / "index.json").read_text())
        if (
            index.get("format") != self.dspark.KERNEL_SHARD_FORMAT
            or int(index.get("ranks", 0)) != 32
        ):
            raise ValueError(f"{directory}: not a 32-rank DSpark kernel shard set")
        raw = np.fromfile(directory / f"rank{self.rank:02d}.bin", dtype=np.uint8)
        rope_q, rope_kv = self.dspark.rope_tables(self.config, self.max_position + 96)
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
                    else np.dtype(jax.numpy.bfloat16)
                )
                count = int(np.prod(spec["shape"]))
                value = (
                    raw[spec["offset"] : spec["offset"] + count * dtype.itemsize]
                    .view(dtype)
                    .reshape(spec["shape"])
                )
            self.weights[name] = upload(value)
        materialize(self.weights[self.names[-1]])
        wall = time.perf_counter() - started
        if self.rank == 0:
            gib = (
                sum(t.numel() * t.element_size() for t in self.weights.values()) / 2**30
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
        """A cold draft state whose next verify is ``anchor`` followed by
        ``anchor`` repeated as placeholder drafts (they are simply rejected)."""
        c = self.config
        cache = torch.zeros(
            (1, c.layers, c.window, 2 * c.head_dim), dtype=torch.bfloat16
        ).to("tpu")
        positions = torch.full((1, c.window), -1, dtype=torch.int32).to("tpu")
        return _settled(DraftState(cache, positions, self._vector(anchor)))

    def from_ring(
        self, cache: torch.Tensor, positions: torch.Tensor, anchor: int | torch.Tensor
    ) -> DraftState:
        """A draft state over a ring the prefill computed
        (``KdaImporter.import_ring``: ``cache [1, layers, window, 128]``,
        ``positions [1, window]``); its next verify is ``anchor`` with
        placeholder drafts, like :meth:`new_state`'s."""
        return _settled(DraftState(cache, positions, self._vector(anchor)))

    def _vector(self, anchor: int | torch.Tensor) -> torch.Tensor:
        # Built on the device either way, then settled with its state: the
        # draft ops compile per operand history (see _settled).
        vector = torch.full((1, 128), -1, dtype=torch.int32).to("tpu")
        if not isinstance(anchor, torch.Tensor):
            anchor = torch.tensor([int(anchor)], dtype=torch.int32).to("tpu")
        anchor = anchor.reshape(1).to(torch.int32)
        vector[0, : self.dspark.ROWS - 1] = anchor
        vector[0, self.dspark.OUT_BONUS] = anchor[0]
        # Indexed writes leave a flat backing buffer even after _settled; the
        # clone has draft outputs' [1, 128] backing, so mixes of new and
        # continuing states share an executable.
        return vector.clone()

    def _synthetic_rates(self, drafts: int) -> tuple[float, ...] | None:
        """The synthetic rates a verify of ``drafts`` drafts draws from."""
        if self.synthetic_rates_by_width is not None:
            return self.synthetic_rates_by_width[drafts]
        return self.synthetic_rates

    def _build_single(self, control: Any) -> Any:
        key = ("owned", control.key())
        if key in self._ops:
            return self._ops[key]

        dspark, rows = self.dspark, self.dspark.ROWS
        fused = dspark.make_fused_draft(self.config, self.mesh, rows, requests=1)
        names = self.names
        count = len(names)

        def fn(*args):
            import jax.numpy as jnp

            weights = {name: args[i] for i, name in enumerate(names)}
            cache, positions, aux, logits, previous, ids, state, decisions = args[
                count:
            ]
            tokens, base, slot = ids[:rows], ids[rows], ids[2 * rows]
            forced_count, forced_anchor = decisions[0, 0], decisions[0, 1]
            vector, cache, positions = fused(
                weights,
                cache,
                positions,
                aux,
                logits,
                previous,
                base[None],
                forced_count[None],
                forced_anchor[None],
            )
            accepted, bonus = vector[0, dspark.OUT_COUNT], vector[0, dspark.OUT_BONUS]
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

        in_specs = (P(AXIS),) * count + (P(AXIS), P(), P(), P(AXIS), P(), P(), P(), P())
        out_specs = (P(), P(AXIS), P(), P(), P(), P())
        self._ops[key] = register_global_op(
            "pallas::k3_dspark_draft_owned",
            self.mesh,
            fn,
            in_specs,
            out_specs,
            donate_argnums=(count,),
            export_key=repr(
                (
                    "dspark_draft_owned_rq",
                    control.key(),
                    names,
                    self.max_position,
                    dataclasses.astuple(self.config),
                )
            ),
        )
        return self._ops[key]

    def _build_metadata(self, requests: int) -> Any:
        """Pack replicated metadata before the large-batch draft boundary.

        Sixteen caches/positions/vectors plus weights exceed Torch's
        64-argument limit; two packed metadata tensors keep that op below it,
        while its outputs stay canonical per-request buffers.
        """
        key = ("metadata", requests)
        if key not in self._ops:

            def fn(*args):
                import jax.numpy as jnp

                return (
                    jnp.stack(args[:requests], axis=1),
                    jnp.stack(args[requests:], axis=1),
                )

            self._ops[key] = register_global_op(
                f"pallas::k3_dspark_metadata_r{requests}",
                self.mesh,
                fn,
                (P(),) * (2 * requests),
                (P(), P()),
                export_key=repr(("dspark_metadata", requests, self.config.window)),
            )
        return self._ops[key]

    @staticmethod
    def parts(requests: int, rows_per: int) -> list[tuple[int, int]]:
        """``(first, count)`` of each fused draft call after a verify of
        ``requests`` requests: the fused kernel drafts at most 16 requests of
        up to four rows, or 8 of more."""
        size = 16 if rows_per <= 4 else 8
        return [
            (first, min(size, requests - first)) for first in range(0, requests, size)
        ]

    def _build_batched(
        self,
        control: Any,
        requests: int,
        rows_per: int,
        first: int = 0,
        verified: int | None = None,
    ) -> Any:
        """The draft step after a batched verify (``DecodeOps.
        verify_batched``): one fused kernel call over the request dimension
        (the weights stream once for all requests; ``_batched_fn``), then each
        request's control entry advanced. ``first`` / ``verified``: this
        call's part of a wider verify (``parts``)."""
        verified = requests if verified is None else verified
        key = ("batched", control.key(), requests, rows_per, first, verified)
        if key in self._ops:
            return self._ops[key]
        fused = self.dspark.make_fused_draft(
            self.config, self.mesh, rows_per, requests=requests
        )
        batched_fn = _batched_fn(
            self.dspark,
            fused,
            control,
            self.names,
            requests,
            rows_per,
            first,
            verified,
        )
        count = len(self.names)

        def fn(*args):
            import jax.numpy as jnp

            R = requests
            if R > 8:
                positions, previous = args[count + R : count + R + 2]
                tail = args[count + R + 2 :]
            else:
                positions = jnp.stack(args[count + R : count + 2 * R], axis=1)
                previous = jnp.stack(args[count + 2 * R : count + 3 * R], axis=1)
                tail = args[count + 3 * R :]
            vectors, caches, positions, emitted, state = batched_fn(
                *args[:count],
                jnp.stack(args[count : count + R], axis=1),
                positions,
                previous,
                *tail,
            )
            return (
                *(vectors[:, j] for j in range(R)),
                *(caches[:, j] for j in range(R)),
                *(positions[:, j] for j in range(R)),
                emitted,
                state,
            )

        part = "" if verified == requests else f"_part{first}of{verified}"
        self._ops[key] = register_global_op(
            f"pallas::k3_dspark_draft_rbatch{requests}x{rows_per}{part}",
            self.mesh,
            fn,
            (P(AXIS),) * count
            + (P(AXIS),) * requests
            + (P(),) * (2 if requests > 8 else 2 * requests)
            + (P(), P(AXIS), P(), P(), P()),
            (P(),) * requests + (P(AXIS),) * requests + (P(),) * requests + (P(), P()),
            donate_argnums=tuple(range(count, count + requests)),
            export_key=repr(
                (
                    "dspark_draft_rbatched",
                    control.key(),
                    requests,
                    rows_per,
                    first,
                    verified,
                    self.names,
                    self.max_position,
                    dataclasses.astuple(self.config),
                )
            ),
        )
        return self._ops[key]

    def compilation_calls(
        self,
        control: Any,
        spec_alone: bool,
        spec_shapes: tuple[tuple[int, int], ...],
    ) -> list:
        """Configured draft graphs with canonical state shapes.

        Fresh vectors and returned vectors have the same backing shape, so
        these graphs also cover new/continuing state mixtures. The ordinary
        warmup still validates every transition with real donated buffers.
        """
        from vllm_torchtpu.megakernel.compilation import CompileCall

        c = self.config
        weights = tuple(self.weights[n] for n in self.names)
        cache = torch.empty(
            (1, c.layers, c.window, 2 * c.head_dim), dtype=torch.bfloat16
        )
        positions = torch.empty((1, c.window), dtype=torch.int32)
        vector = torch.empty((1, 128), dtype=torch.int32)
        state = torch.empty(control.size, dtype=torch.int32)
        calls = []
        drafted = {
            count for r, k in spec_shapes for _, count in self.parts(r, k) if count > 8
        }
        for requests in sorted(drafted):
            calls.append(
                CompileCall(
                    self._build_metadata(requests),
                    ((positions,) * requests + (vector,) * requests),
                    {},
                    priority=0,
                )
            )

        def tail(rows, ids_size, requests):
            return (
                torch.empty(
                    (len(self.aux_layers), rows, c.hidden), dtype=torch.bfloat16
                ),
                torch.empty((1, rows, c.vocab // TP_SIZE), dtype=torch.float32),
                torch.empty(ids_size, dtype=torch.int32),
                state,
                torch.empty((requests, 2), dtype=torch.int32),
            )

        if spec_alone:
            rows = self.dspark.ROWS
            aux, logits, ids, state, decisions = tail(
                rows, 3 * rows + 1 + rows * control.pages, 1
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
                        state,
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
                            control, requests, spec_rows, first, verified
                        ),
                        (
                            *weights,
                            *((cache,) * requests),
                            *(
                                (
                                    torch.empty(
                                        (1, requests, c.window), dtype=torch.int32
                                    ),
                                    torch.empty((1, requests, 128), dtype=torch.int32),
                                )
                                if requests > 8
                                else (positions,) * requests + (vector,) * requests
                            ),
                            *tail(rows, 4 * rows + verified * control.pages, verified),
                        ),
                        {},
                        priority=2 * rows,
                    )
                )
        return calls

    def step_batched(
        self,
        states: list[DraftState],
        aux: torch.Tensor,
        logits: torch.Tensor,
        ids: torch.Tensor,
        control_state: torch.Tensor,
        decisions: torch.Tensor,
        control: Any,
        rows_per: int,
    ) -> tuple[list[DraftState], torch.Tensor, torch.Tensor]:
        """Draft after a batched verify whose packed operand was ``ids``.
        ``decisions`` [R, 2] contains its accepted counts and bonus tokens.
        Packing and splitting happen inside the JAX op; each returned
        Torch state owns a canonical per-request buffer, including after
        requests join or leave. Returns ``(states, tokens
        [R, 8] -- each request's accepted drafts then its bonus, -1 padded --
        the control state)``."""
        verified = len(states)
        new, emitted = [], []
        for first, R in self.parts(verified, rows_per):
            part = states[first : first + R]
            op = self._build_batched(control, R, rows_per, first, verified)
            metadata = tuple(s.positions for s in part) + tuple(s.vector for s in part)
            if R > 8:
                metadata = self._build_metadata(R)(*metadata)
            out = op(
                *(self.weights[n] for n in self.names),
                *(s.cache for s in part),
                *metadata,
                aux,
                logits,
                ids,
                control_state,
                decisions,
            )
            new += [
                _settled(DraftState(out[R + j], out[2 * R + j], out[j]))
                for j in range(R)
            ]
            emitted.append(out[-2])
            control_state = out[-1]
        emitted = emitted[0] if len(emitted) == 1 else torch.cat(emitted)
        return new, emitted, control_state

    def step_single(
        self,
        state: DraftState,
        aux: torch.Tensor,
        logits: torch.Tensor,
        ids: torch.Tensor,
        control_state: torch.Tensor,
        decisions: torch.Tensor,
        control: Any,
    ):
        """Draft after the verify whose packed operand was ``ids``.
        ``decisions`` [1, 2] contains its accepted count and bonus token.
        Returns ``(state, tokens [1, 8] -- the accepted
        drafts then the bonus, -1 padded -- the next verify's packed operand,
        the control state)``, the control recording the slot's next token,
        position and state row (the accepted count)."""
        op = self._build_single(control)
        vector, cache, positions, emitted, following, control_state = op(
            *(self.weights[n] for n in self.names),
            state.cache,
            state.positions,
            aux,
            logits,
            state.vector,
            ids,
            control_state,
            decisions,
        )
        state = _settled(DraftState(cache, positions, vector))
        return state, emitted, following, control_state
