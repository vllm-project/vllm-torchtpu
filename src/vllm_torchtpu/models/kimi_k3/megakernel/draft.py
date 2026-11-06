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

The draft is ``kernels.dspark``'s fused Pallas kernel on the decoder's 32-rank
mesh. After a verify, :meth:`DsparkDraft.step_single` or
:meth:`DsparkDraft.step_batched` writes the verified rows' context into the
draft's ring and proposes the next drafts after the verifier's bonus token.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger

from . import draft_mla
from .program import TP_SIZE, kernel_modules
from .runtime_shim.bridge import (
    OpRegistrationSpec,
    hbm,
    kernel_mesh,
    materialize,
    register_global_op,
    upload,
)
from .runtime_shim.compilation import CompileCall

logger = init_logger(__name__)

AXIS = "tp"
_VECTOR_LEN = 128
_REV_PREFIX_LEN = 8
_ROPE_POS_SLACK = 96
_GIB_BYTES = 2**30
_SMALL_PART_ROWS = 4
_SMALL_PART_REQUESTS = 16
_LARGE_PART_REQUESTS = 8
_BANK_FIELDS = 3
_DECISION_COLS = 2


class _ControlLike(Protocol):
    """Structural protocol for decoder control layouts used by draft ops."""

    SLOT: int
    RESET: int
    FIELDS: int
    slots: int
    pages: int
    plan_start: int
    size: int

    def key(self) -> tuple[int, int, int]:
        """Return the ``(slots, rows, pages)`` layout key."""
        ...

    def fields(
        self,
        control: jax.Array,
        batch: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Return ``(plan, tables)`` slices for ``batch`` rows."""
        ...


@dataclasses.dataclass
class DraftState:
    """One speculating sequence's draft state on device."""

    cache: torch.Tensor
    positions: torch.Tensor
    vector: torch.Tensor


def _settled(state: DraftState) -> DraftState:
    """Materialize ``state`` tensors asynchronously on TPU devices."""
    if state.cache.device.type == "tpu":
        sync_mod = importlib.import_module("torch_tpu._internal.sync")
        sync_mod.synchronize(
            [state.cache, state.positions, state.vector],
            wait=False,
        )
    return state


def _batched_fn(
    dspark: object,
    fused: Callable[..., tuple[jax.Array, jax.Array, jax.Array]],
    control: _ControlLike,
    names: tuple[str, ...],
    *batch_spec: int | None,
) -> Callable[..., tuple[jax.Array, ...]]:
    """Build the batched draft op's global-view function."""
    requests_raw, rows_per_raw, first_raw, verified_raw = (*batch_spec, 0, None)[:4]
    requests, rows_per, first = (
        int(requests_raw or 0),
        int(rows_per_raw or 0),
        int(first_raw or 0),
    )
    dspark_attr = vars(dspark)
    width, out_count, out_bonus = (
        int(dspark_attr["ROWS"]),
        int(dspark_attr["OUT_COUNT"]),
        int(dspark_attr["OUT_BONUS"]),
    )
    req_count, k = requests, rows_per
    total = (requests if verified_raw is None else int(verified_raw)) * k
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
        if req_count == 1:
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
            state[: control.plan_start].reshape(_BANK_FIELDS, control.slots),
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
    """Return the draft class matching the speculator ``checkpoint``."""
    (mla,) = kernel_modules("dspark_mla")
    if mla.is_mla_draft(Path(checkpoint)):
        return draft_mla.MlaDsparkDraft
    return DsparkDraft


class DsparkDraft:
    """The fused DSpark draft: weights, per-sequence rings, and step ops."""

    def __init__(
        self,
        max_position: int,
        checkpoint: str,
        presharded: Path,
        synthetic_rates: tuple[float, ...] | None = None,
        synthetic_rates_by_width: Mapping[int, Sequence[float]] | None = None,
    ) -> None:
        """Initialize DSpark drafter from ``checkpoint`` and ``presharded``."""
        (self.dspark,) = kernel_modules("dspark")
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
        self.names: tuple[str, ...] = tuple(self.dspark.WEIGHT_NAMES)
        self.max_position = int(max_position)
        self.weights: dict[str, torch.Tensor] = {}
        rev = Path(checkpoint).resolve().name[:_REV_PREFIX_LEN]
        self._presharded = Path(presharded) / f"dspark-{rev}"
        self._ops: dict[
            tuple[object, ...], Callable[..., tuple[torch.Tensor, ...]]
        ] = {}
        self._aux_layers = tuple(int(layer) for layer in self.config.aux_layers)

    @property
    def aux_layers(self) -> tuple[int, ...]:
        """Return target model layer indices tapped for draft context."""
        return self._aux_layers

    def load_weights(self) -> float:
        """Load this rank's kernel-layout shard and regenerate RoPE tables."""
        started = time.perf_counter()
        directory = self._presharded
        index = json.loads((directory / "index.json").read_text())
        if (
            index.get("format") != self.dspark.KERNEL_SHARD_FORMAT
            or int(index.get("ranks", 0)) != TP_SIZE
        ):
            msg = f"{directory}: not a 32-rank DSpark kernel shard set"
            raise ValueError(msg)
        raw = np.fromfile(directory / f"rank{self.rank:02d}.bin", dtype=np.uint8)
        rope_q, rope_kv = self.dspark.rope_tables(
            self.config,
            self.max_position + _ROPE_POS_SLACK,
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
                / _GIB_BYTES
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
        """Return a cold draft state seeded with ``anchor``."""
        cfg = self.config
        cache = torch.zeros(
            (1, cfg.layers, cfg.window, 2 * cfg.head_dim),
            dtype=torch.bfloat16,
        ).to("tpu")
        positions = torch.full((1, cfg.window), -1, dtype=torch.int32).to("tpu")
        return _settled(DraftState(cache, positions, self._vector(anchor)))

    def from_ring(
        self,
        cache: torch.Tensor,
        positions: torch.Tensor,
        anchor: int | torch.Tensor,
    ) -> DraftState:
        """Construct a draft state over a prefill-computed ring and ``anchor``."""
        return _settled(DraftState(cache, positions, self._vector(anchor)))

    def _vector(self, anchor: int | torch.Tensor) -> torch.Tensor:
        """Allocate and seed a ``[1, 128]`` draft output vector with ``anchor``."""
        vector = torch.full((1, _VECTOR_LEN), -1, dtype=torch.int32).to("tpu")
        anchor_t = (
            anchor.reshape(1).to(torch.int32)
            if isinstance(anchor, torch.Tensor)
            else torch.tensor([int(anchor)], dtype=torch.int32).to("tpu")
        )
        vector[0, : self.dspark.ROWS - 1] = anchor_t
        vector[0, self.dspark.OUT_BONUS] = anchor_t[0]
        return vector.clone()

    def synthetic_rates_for(self, drafts: int) -> tuple[float, ...] | None:
        """Return synthetic acceptance rates configured for ``drafts`` width."""
        if self.synthetic_rates_by_width is not None:
            return tuple(self.synthetic_rates_by_width[drafts])
        return self.synthetic_rates

    def _build_single(
        self,
        control: _ControlLike,
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
                cache,
                positions,
                aux,
                logits,
                previous,
                base[None],
                forced_count[None],
                forced_anchor[None],
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

        in_specs = (P(AXIS),) * count + (
            P(AXIS),
            P(),
            P(),
            P(AXIS),
            P(),
            P(),
            P(),
            P(),
        )
        out_specs = (P(), P(AXIS), P(), P(), P(), P())
        self._ops[key] = register_global_op(
            "pallas::k3_dspark_draft_owned",
            self.mesh,
            fn,
            OpRegistrationSpec(
                in_specs=in_specs,
                out_specs=out_specs,
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
            ),
        )
        return self._ops[key]

    def _build_metadata(
        self,
        requests: int,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build or return the metadata-packing op for large draft batches."""
        key = ("metadata", requests)
        if key not in self._ops:

            def fn(*args: jax.Array) -> tuple[jax.Array, jax.Array]:
                """Stack per-request positions and vectors along axis 1."""
                return (
                    jnp.stack(args[:requests], axis=1),
                    jnp.stack(args[requests:], axis=1),
                )

            self._ops[key] = register_global_op(
                f"pallas::k3_dspark_metadata_r{requests}",
                self.mesh,
                fn,
                OpRegistrationSpec(
                    in_specs=(P(),) * (2 * requests),
                    out_specs=(P(), P()),
                    export_key=repr(("dspark_metadata", requests, self.config.window)),
                ),
            )
        return self._ops[key]

    @staticmethod
    def parts(requests: int, rows_per: int) -> list[tuple[int, int]]:
        """Return ``(first, count)`` slices for each fused draft call."""
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
        control: _ControlLike,
        requests: int,
        rows_per: int,
        first: int = 0,
        verified: int | None = None,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build or return the batched fused draft custom op."""
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
        self._ops[key] = register_global_op(
            f"pallas::k3_dspark_draft_rbatch{requests}x{rows_per}{part}",
            self.mesh,
            fn,
            OpRegistrationSpec(
                in_specs=(P(AXIS),) * count
                + (P(AXIS),) * requests
                + (P(),) * (2 if requests > _LARGE_PART_REQUESTS else 2 * requests)
                + (P(), P(AXIS), P(), P(), P()),
                out_specs=(P(),) * requests
                + (P(AXIS),) * requests
                + (P(),) * requests
                + (P(), P()),
                donate_argnums=tuple(range(count, count + requests)),
                export_key=repr(
                    (
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
                ),
            ),
        )
        return self._ops[key]

    def compilation_calls(
        self,
        control: _ControlLike,
        spec_shapes: tuple[tuple[int, int], ...],
        *,
        spec_alone: bool = False,
    ) -> list[CompileCall]:
        """Return precompilation calls for configured draft shapes."""
        cfg = self.config
        weights = tuple(self.weights[n] for n in self.names)
        cache = torch.empty(
            (1, cfg.layers, cfg.window, 2 * cfg.head_dim),
            dtype=torch.bfloat16,
        )
        positions = torch.empty((1, cfg.window), dtype=torch.int32)
        vector = torch.empty((1, _VECTOR_LEN), dtype=torch.int32)
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
                torch.empty((requests, _DECISION_COLS), dtype=torch.int32),
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
                                        (1, requests, _VECTOR_LEN),
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
        *step_inputs: object,
    ) -> tuple[list[DraftState], torch.Tensor, torch.Tensor]:
        """Run batched draft step after a batched verify."""
        (
            aux,
            logits,
            ids,
            control_state_obj,
            decisions,
            control_obj,
            rows_per_obj,
        ) = step_inputs
        control_state = cast("torch.Tensor", control_state_obj)
        control = cast("_ControlLike", control_obj)
        rows_per = int(cast("int", rows_per_obj))
        verified = len(states)
        new: list[DraftState] = []
        emitted: list[torch.Tensor] = []
        for first, req_count in self.parts(verified, rows_per):
            part = states[first : first + req_count]
            op = self._build_batched(control, req_count, rows_per, first, verified)
            metadata: tuple[torch.Tensor, ...] = tuple(
                s.positions for s in part
            ) + tuple(s.vector for s in part)
            if req_count > _LARGE_PART_REQUESTS:
                metadata = self._build_metadata(req_count)(*metadata)
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
        *step_inputs: object,
    ) -> tuple[DraftState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run single-request draft step after a single verify."""
        aux, logits, ids, control_state, decisions, control_obj = step_inputs
        control = cast("_ControlLike", control_obj)
        op = self._build_single(control)
        vector, cache, positions, emitted, following, next_control = op(
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
        settled = _settled(DraftState(cache, positions, vector))
        return settled, emitted, following, next_control
