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
# src/vllm_torchtpu/megakernel/kimi_k3/draft_mla.py
"""MLA DSpark speculative decoding for the K3 decode megakernel.

:class:`MlaDsparkDraft` shares :class:`draft.DsparkDraft`'s interface: after a
verify, :meth:`step_single` / :meth:`step_batched` run the fused draft kernel
and commit the verify rows' context latents to the draft's five pools, then
advance the control state.
"""

from __future__ import annotations

import dataclasses
import importlib
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

from . import config as kernel_config
from .program import TP_SIZE, kernel_modules
from .runtime_shim.compilation import CompileCall

logger = init_logger(__name__)

AXIS = "tp"
TARGET_WEIGHTS = {"lm_head": "lm_head", "embed": "embedding"}
MAX_ROWS = 64
DRAFT_VECTOR_LEN = 128
_REV_PREFIX_LEN = 8
_ROPE_POS_SLACK = 96
_GIB_BYTES = 2**30
_BANK_FIELDS = 3
_DECISION_COLS = 2
_MAX_DP_GROUPS = 2


class _ControlLike(Protocol):
    """Structural protocol for decoder control layouts used by MLA draft ops."""

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


def _set_tensor_in_place(target: torch.Tensor, source: torch.Tensor) -> None:
    """Replace ``target`` storage with ``source`` in place."""
    attr_name = "set_"
    getattr(target, attr_name)(source)


@dataclasses.dataclass
class MlaDraftState:
    """One speculating sequence's draft state holding its output vector."""

    vector: torch.Tensor


def _settled(state: MlaDraftState) -> MlaDraftState:
    """Synchronize ``state.vector`` asynchronously on TPU devices."""
    if state.vector.device.type == "tpu":
        sync_mod = importlib.import_module("torch_tpu._internal.sync")
        sync_mod.synchronize([state.vector], wait=False)
    return state


def make_draft_state(
    anchor: int | torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> MlaDraftState:
    """Construct an initial ``MlaDraftState`` seeded with ``anchor``."""
    (dspark,) = kernel_modules("dspark")
    dev = torch.device(device)
    vector = torch.full((1, DRAFT_VECTOR_LEN), -1, dtype=torch.int32, device=dev)
    anchor_t = (
        anchor.reshape(1).to(dtype=torch.int32, device=dev)
        if isinstance(anchor, torch.Tensor)
        else torch.tensor([int(anchor)], dtype=torch.int32, device=dev)
    )
    vector[0, : dspark.ROWS - 1] = anchor_t
    vector[0, dspark.OUT_BONUS] = anchor_t[0]
    return _settled(MlaDraftState(vector.clone()))


def _batched_fn(
    dspark: object,
    fused: Callable[..., tuple[jax.Array, jax.Array]],
    commit: Callable[..., Sequence[jax.Array]],
    control: _ControlLike,
    *batch_spec: object,
) -> Callable[..., tuple[jax.Array, ...]]:
    """Build the batched MLA draft op's global-view function."""
    names_obj, layers_obj, index_obj, verified_obj, rows_per_obj = batch_spec
    names = cast("tuple[str, ...]", names_obj)
    layers = int(cast("int", layers_obj))
    index = cast("tuple[int, ...]", index_obj)
    verified = int(cast("int", verified_obj))
    rows_per = int(cast("int", rows_per_obj))
    dspark_rows = int(vars(dspark)["ROWS"])
    out_count = int(vars(dspark)["OUT_COUNT"])
    out_bonus = int(vars(dspark)["OUT_BONUS"])
    width, pages = dspark_rows, control.pages
    count = len(names)
    req_count, k = len(index), rows_per
    total = req_count * k
    whole = index == tuple(range(verified))
    picked = np.asarray(index, np.int32)
    rows = (picked[:, None] * k + np.arange(k, dtype=np.int32)[None, :]).reshape(-1)

    def fn(*args: jax.Array) -> tuple[jax.Array, ...]:
        """Run one fused MLA draft part, commit latents, and advance control."""
        weights = {name: args[i][0] for i, name in enumerate(names)}
        pools: Sequence[jax.Array] = args[count : count + layers]
        previous = jnp.concatenate(
            args[count + layers : count + layers + req_count],
            axis=0,
        )
        aux, logits, ids, state, decisions = args[count + layers + req_count :]
        n = verified * k
        tokens, positions = ids[:n], ids[n : 2 * n]
        indices = ids[3 * n : 4 * n]
        tables = ids[4 * n :].reshape(verified, pages)
        plan_all, _ = control.fields(state, verified)
        plan = plan_all
        if not whole:
            tokens, positions, indices = (x[rows] for x in (tokens, positions, indices))
            aux, logits = aux[:, rows], logits[:, rows]
            tables = tables[picked]
            plan, decisions = plan[picked], decisions[picked]
        bases = positions.reshape(req_count, k)[:, 0]
        vectors, latents = fused(
            weights,
            pools,
            aux,
            logits[0].reshape(req_count, k, logits.shape[-1]),
            previous,
            bases,
            decisions[:, 0],
            decisions[:, 1],
            tables,
        )
        row_positions = (
            bases[:, None] + jnp.arange(k, dtype=jnp.int32)[None, :]
        ).reshape(total)
        pools = commit(
            pools,
            latents,
            jnp.repeat(tables, k, axis=0),
            row_positions,
        )
        accepted_all, bonus_all = (
            vectors[:, out_count],
            vectors[:, out_bonus],
        )
        tokens_r, indices_r = (
            tokens.reshape(req_count, k),
            indices.reshape(req_count, k),
        )
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
        if not whole:
            plan = plan_all.at[picked].set(plan)
        state = jnp.concatenate(
            (
                banks.reshape(-1),
                plan.reshape(-1),
                state[control.plan_start + verified * control.FIELDS :],
            )
        )
        return (
            *(vectors[j : j + 1] for j in range(req_count)),
            *pools,
            emitted,
            state,
        )

    return fn


class MlaDsparkDraft:
    """The fused MLA DSpark draft: weights, pools, and step ops."""

    def __init__(
        self,
        max_position: int,
        checkpoint: str,
        presharded: Path,
        synthetic_rates: tuple[float, ...] | None = None,
        synthetic_rates_by_width: Mapping[int, Sequence[float]] | None = None,
    ) -> None:
        """Initialize MLA DSpark drafter from ``checkpoint`` and ``presharded``."""
        (self.dspark, self.mla, self.kernel, self.paged) = kernel_modules(
            "dspark",
            "dspark_mla",
            "dspark_mla_kernel",
            "paged",
        )
        bridge = importlib.import_module(".runtime_shim.bridge", __package__)
        self._bridge = bridge
        self.synthetic_rates = synthetic_rates
        self.synthetic_rates_by_width = synthetic_rates_by_width
        self.config = self.mla.MlaDraftConfig.from_checkpoint(Path(checkpoint))
        self.mesh, self.rank = bridge.kernel_mesh(AXIS, TP_SIZE)
        self.attention_dp = kernel_config.active().attention_dp
        if self.attention_dp not in (1, _MAX_DP_GROUPS):
            msg = "the MLA DSpark draft runs on TP32 or under attention DP2"
            raise ValueError(msg)
        self.max_position = int(max_position)
        self.names: tuple[str, ...] = tuple(self.kernel.WEIGHT_NAMES)
        self.weights: dict[str, torch.Tensor] = {}
        self.pools: list[torch.Tensor] = []
        self.owner_table: tuple[int, ...] | None = None
        rev = Path(checkpoint).resolve().name[:_REV_PREFIX_LEN]
        self._presharded = Path(presharded) / f"dspark-{rev}"
        self._ops: dict[tuple[object, ...], object] = {}
        self._aux_layers = tuple(int(layer) for layer in self.config.aux_layers)
        if self.rank == 0:
            logger.info(
                "K3 DSpark (MLA): aux layers %s, block %d, softmax scale %.5f",
                self._aux_layers,
                self.config.block,
                self.config.softmax_scale,
            )

    @property
    def aux_layers(self) -> tuple[int, ...]:
        """Return target model layer indices tapped for draft context."""
        return self._aux_layers

    def synthetic_rates_for(self, drafts: int) -> tuple[float, ...] | None:
        """Return synthetic acceptance rates configured for ``drafts`` width."""
        if self.synthetic_rates_by_width is not None:
            return tuple(self.synthetic_rates_by_width[drafts])
        return self.synthetic_rates

    def load_weights(self) -> float:
        """Load this rank's kernel shards and regenerate the RoPE table."""
        started = time.perf_counter()
        shards = dict(self.mla.load_rank_shards(self._presharded, self.rank))
        if self.attention_dp > 1:
            shards.update(
                self.mla.held_shards(
                    self._presharded,
                    self.rank,
                    self.attention_dp,
                    shards,
                )
            )
        shards["qkv_tile"], shards["context_tile"] = self.mla.qkv_tiles(
            shards.pop("qkv_a"),
            self.rank,
        )
        for name, value in shards.items():
            self.weights[name] = self._bridge.upload(np.ascontiguousarray(value))
        self.weights["rope"] = self._bridge.upload(
            self.mla.rope_table(self.config, self.max_position + _ROPE_POS_SLACK)
        )
        self._bridge.materialize(self.weights["rope"])
        wall = time.perf_counter() - started
        if self.rank == 0:
            gib = (
                sum(t.numel() * t.element_size() for t in self.weights.values())
                / _GIB_BYTES
            )
            logger.info(
                "K3 DSpark (MLA): draft weights resident (%.2f GiB per rank, "
                "RoPE to %d) in %.1f s; HBM %s",
                gib,
                self.max_position,
                wall,
                self._bridge.hbm(self.rank),
            )
        return wall

    def attach_target(self, target_weights: dict[str, torch.Tensor]) -> None:
        """Attach the target's LM head and embedding and extract mask embedding."""
        for name, target in TARGET_WEIGHTS.items():
            self.weights[name] = target_weights[target]
        mask, vocab = self.config.mask_token, self.config.vocab // TP_SIZE

        def body(embedding: jax.Array) -> tuple[jax.Array]:
            """Extract and all-reduce the mask token embedding row."""
            rank = jax.lax.axis_index(AXIS)
            local = mask - rank * vocab
            row = embedding[0][jnp.clip(local, 0, vocab - 1)].astype(jnp.float32)
            row = jnp.where((local >= 0) & (local < vocab), row, 0.0)
            return (jax.lax.psum(row, AXIS).astype(jnp.bfloat16)[None, None],)

        op = self._bridge.register_op(
            "pallas::k3_dspark_mla_mask_embed",
            self.mesh,
            body,
            self._bridge.OpRegistrationSpec(
                in_specs=(P(AXIS),),
                out_specs=(P(AXIS),),
                export_key=repr(("dspark_mla_mask_embed", mask, vocab)),
            ),
        )
        (self.weights["mask_embed"],) = op(self.weights["embed"])

    def bind_pools(
        self,
        template: torch.Tensor,
        owner_table: Sequence[int] | None,
        pools: Sequence[torch.Tensor] = (),
    ) -> None:
        """Bind or allocate the five draft layer KV cache pools."""
        self.owner_table = (
            None if owner_table is None else tuple(int(i) for i in owner_table)
        )
        if pools:
            if len(pools) != self.config.layers or any(
                p.shape != template.shape or p.dtype != template.dtype for p in pools
            ):
                msg = (
                    f"K3 DSpark (MLA): expected {self.config.layers} draft pools "
                    f"like {tuple(template.shape)} {template.dtype}"
                )
                raise RuntimeError(msg)
            self.pools = list(pools)
        elif not (self.pools and self.pools[0].shape == template.shape):
            self.pools = [torch.zeros_like(template) for _ in range(self.config.layers)]
        if self.rank == 0:
            logger.info(
                "K3 DSpark (MLA): %d draft pools %s %s; HBM %s",
                len(self.pools),
                tuple(template.shape),
                template.dtype,
                self._bridge.hbm(self.rank),
            )

    def new_state(self, anchor: int | torch.Tensor) -> MlaDraftState:
        """Return an initial draft state seeded with ``anchor``."""
        dev = self.pools[0].device if self.pools else torch.device("tpu")
        return make_draft_state(anchor, device=dev)

    def from_ring(
        self,
        *_args: object,
    ) -> MlaDraftState:
        """Raise RuntimeError because MLA DSpark stores context in pools."""
        msg = "the MLA DSpark draft keeps its context in pools, not a ring"
        raise RuntimeError(msg)

    def _operands(self) -> tuple[torch.Tensor, ...]:
        """Return ordered weight tensors for draft op invocation."""
        return tuple(self.weights[n] for n in self.names)

    def _commit(
        self,
        pools: Sequence[jax.Array],
        latents: jax.Array,
        tables: jax.Array,
        positions: jax.Array,
    ) -> Sequence[jax.Array]:
        """Commit ``latents`` to the draft pools on their owning ranks."""
        dp = self.attention_dp
        if dp > 1:
            group = jax.lax.axis_index(AXIS) // (TP_SIZE // dp)
            size = positions.shape[0] // dp
            latents = jax.lax.dynamic_slice_in_dim(latents, group * size, size, 1)
            tables = jax.lax.dynamic_slice_in_dim(tables, group * size, size)
            positions = jax.lax.dynamic_slice_in_dim(positions, group * size, size)
        rows = latents.transpose(1, 0, 2)
        committed: Sequence[jax.Array] = self.paged.commit_latent_rows(
            pools,
            rows,
            tables,
            positions,
            tuple(range(self.config.layers)),
            owner_table=self.owner_table,
        )
        return committed

    def _build_single(
        self,
        control: _ControlLike,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build or return the single-request MLA draft custom op."""
        key = ("single", control.key())
        if key in self._ops:
            return cast("Callable[..., tuple[torch.Tensor, ...]]", self._ops[key])
        dspark, rows, names = self.dspark, self.dspark.ROWS, self.names
        count, layers, pages = len(names), self.config.layers, control.pages
        fused = self.kernel.fused_draft_local(
            self.config,
            requests=1,
            pages=pages,
            owner_table=self.owner_table,
        )

        def body(*args: jax.Array) -> tuple[jax.Array, ...]:
            """Execute single-request MLA draft, commit latents, and advance control."""
            weights = {name: args[i][0] for i, name in enumerate(names)}
            pools: Sequence[jax.Array] = args[count : count + layers]
            aux, logits, previous, ids, state, decisions = args[count + layers :]
            tokens, base, slot = ids[:rows], ids[rows], ids[2 * rows]
            table = ids[3 * rows + 1 : 3 * rows + 1 + pages]
            vector, latents = fused(
                weights,
                pools,
                aux,
                logits,
                previous,
                base[None],
                decisions[:, 0],
                decisions[:, 1],
                table[None],
            )
            positions = base + jnp.arange(rows, dtype=jnp.int32)
            pools = self._commit(
                pools,
                latents,
                jnp.broadcast_to(table[None], (rows, pages)),
                positions,
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
            return (vector, *pools, emitted[None], following_ids, state)

        in_specs = (P(AXIS),) * (count + layers) + (
            P(),
            P(AXIS),
            P(),
            P(),
            P(),
            P(),
        )
        out_specs = (P(),) + (P(AXIS),) * layers + (P(), P(), P())
        self._ops[key] = self._bridge.register_op(
            "pallas::k3_dspark_mla_draft_single",
            self.mesh,
            body,
            self._bridge.OpRegistrationSpec(
                in_specs=in_specs,
                out_specs=out_specs,
                donate_argnums=tuple(range(count, count + layers)),
                export_key=repr(
                    (
                        "dspark_mla_single",
                        control.key(),
                        names,
                        self.max_position,
                        dataclasses.astuple(self.config),
                        self.owner_table,
                    )
                ),
            ),
        )
        return cast("Callable[..., tuple[torch.Tensor, ...]]", self._ops[key])

    @staticmethod
    def parts(
        requests: int,
        dp: int = 1,
        rows_per: int = 8,
    ) -> list[tuple[int, ...]]:
        """Return request index tuples for each fused draft call part."""
        share = requests // dp
        per = min(requests, MAX_ROWS // rows_per) // dp
        return [
            tuple(g * share + first + i for g in range(dp) for i in range(per))
            for first in range(0, share, per)
        ]

    def _build_batched(
        self,
        control: _ControlLike,
        index: tuple[int, ...],
        verified: int,
        rows_per: int,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Build or return the batched MLA draft custom op for ``index``."""
        key = ("batched", control.key(), index, verified, rows_per)
        if key in self._ops:
            return cast("Callable[..., tuple[torch.Tensor, ...]]", self._ops[key])
        names = self.names
        count, layers = len(names), self.config.layers
        dp = self.attention_dp
        req_count, k = len(index), rows_per
        whole = index == tuple(range(verified))
        fused = self.kernel.fused_draft_local(
            self.config,
            requests=req_count,
            pages=control.pages,
            block=min(self.config.block, k - 1),
            rows=k,
            owner_table=self.owner_table,
            attention_dp=dp,
        )
        body = _batched_fn(
            self.dspark,
            fused,
            self._commit,
            control,
            names,
            layers,
            index,
            verified,
            k,
        )
        in_specs = (
            (P(AXIS),) * (count + layers)
            + (P(),) * req_count
            + (P(), P(AXIS), P(), P(), P())
        )
        out_specs = (P(),) * req_count + (P(AXIS),) * layers + (P(), P())
        part = "" if whole else f"_part{index[0]}of{verified}"
        self._ops[key] = self._bridge.register_op(
            f"pallas::k3_dspark_mla_draft_r{req_count}x{k}{part}",
            self.mesh,
            body,
            self._bridge.OpRegistrationSpec(
                in_specs=in_specs,
                out_specs=out_specs,
                donate_argnums=tuple(range(count, count + layers)),
                export_key=repr(
                    (
                        "dspark_mla_batched",
                        control.key(),
                        index,
                        verified,
                        k,
                        dp,
                        names,
                        self.max_position,
                        dataclasses.astuple(self.config),
                        self.owner_table,
                    )
                ),
            ),
        )
        return cast("Callable[..., tuple[torch.Tensor, ...]]", self._ops[key])

    def _adopt(self, pools: Sequence[torch.Tensor]) -> None:
        """Update bound draft pool tensors in place from donated outputs."""
        for old, fresh in zip(self.pools, pools, strict=True):
            if fresh is not old:
                _set_tensor_in_place(old, fresh)

    def step_single(
        self,
        state: MlaDraftState,
        *step_inputs: object,
    ) -> tuple[MlaDraftState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run single-request MLA draft step after a single verify."""
        aux, logits, ids, control_state, decisions, control_obj = step_inputs
        control = cast("_ControlLike", control_obj)
        op = self._build_single(control)
        out = op(
            *self._operands(),
            *self.pools,
            aux,
            logits,
            state.vector,
            ids,
            control_state,
            decisions,
        )
        layers = self.config.layers
        vector, pools = out[0], out[1 : 1 + layers]
        emitted, following, next_control = out[1 + layers :]
        self._adopt(pools)
        return _settled(MlaDraftState(vector)), emitted, following, next_control

    def step_batched(
        self,
        states: Sequence[MlaDraftState],
        *step_inputs: object,
    ) -> tuple[list[MlaDraftState], torch.Tensor, torch.Tensor]:
        """Run batched MLA draft step after a batched verify."""
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
        verified, layers = len(states), self.config.layers
        new: list[MlaDraftState | None] = [None] * verified
        emitted: list[torch.Tensor] = []
        order: list[int] = []
        for index in self.parts(verified, self.attention_dp, rows_per):
            req_count = len(index)
            op = self._build_batched(control, index, verified, rows_per)
            out = op(
                *self._operands(),
                *self.pools,
                *(states[j].vector for j in index),
                aux,
                logits,
                ids,
                control_state,
                decisions,
            )
            vectors, pools = out[:req_count], out[req_count : req_count + layers]
            part_emitted, control_state = out[req_count + layers :]
            self._adopt(pools)
            for j, vector in zip(index, vectors, strict=True):
                new[j] = _settled(MlaDraftState(vector))
            emitted.append(part_emitted)
            order.extend(index)
        tokens = emitted[0] if len(emitted) == 1 else torch.cat(emitted)
        if order != list(range(verified)):
            tokens = tokens.index_select(0, self._request_rows(tuple(order)))
        final_states = [s for s in new if s is not None]
        return final_states, tokens, control_state

    def _request_rows(self, order: tuple[int, ...]) -> torch.Tensor:
        """Return the row index of each request in the concatenated part tokens."""
        key = ("rows", order)
        if key not in self._ops:
            dev = self.pools[0].device if self.pools else torch.device("tpu")
            self._ops[key] = torch.tensor(
                np.argsort(np.asarray(order)),
                dtype=torch.int64,
                device=dev,
            )
        return cast("torch.Tensor", self._ops[key])

    def compilation_calls(
        self,
        control: _ControlLike,
        spec_shapes: Sequence[tuple[int, int]],
        *,
        spec_alone: bool = False,
    ) -> list[CompileCall]:
        """Return precompilation calls for configured MLA draft shapes."""
        cfg = self.config
        operands = self._operands()
        vector = torch.empty((1, DRAFT_VECTOR_LEN), dtype=torch.int32)
        state = torch.empty(control.size, dtype=torch.int32)

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
            """Build dummy tail operands for MLA draft precompilation."""
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

        calls: list[CompileCall] = []
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
                        *operands,
                        *self.pools,
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
        for requests, spec_rows in spec_shapes:
            rows = requests * spec_rows
            calls.extend(
                CompileCall(
                    self._build_batched(control, index, requests, spec_rows),
                    (
                        *operands,
                        *self.pools,
                        *((vector,) * len(index)),
                        *tail(
                            rows,
                            4 * rows + requests * control.pages,
                            requests,
                        ),
                    ),
                    {},
                    priority=2 * rows,
                )
                for index in self.parts(requests, self.attention_dp, spec_rows)
            )
        return calls
