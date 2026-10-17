# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA DSpark speculative decoding for the K3 decode megakernel
(``Inferact/Kimi-K3-DSpark``, ``kernels.dspark_mla``).

:class:`MlaDsparkDraft` has :class:`draft.DsparkDraft`'s interface: after a
verify, :meth:`step_single` / :meth:`step_batched` run the fused draft kernel
(``kernels.dspark_mla_kernel``) and commit the verify rows' context latents
to the draft's five pools, then advance the control. The pools are shaped
like the target's MLA pools and read through the target's block tables
(decode context parallel pages), so a request's draft context lives beside
its target pages; a request without them (no prefill-written context) drafts
from what it has, which is only slower (the emitted tokens are the
target's). The draft's per-request state is its last output vector.

A fused call drafts at most 64 rows; a wider verify drafts in parts
(:meth:`MlaDsparkDraft.parts`). Under attention DP2 a request's pages are on
its DP group's ranks, which attend its rows with their two held TP32 shards'
heads (``dspark_mla.held_shards``) and alone commit its latents; each part
takes as many requests of each group.

The kernel-layout shards (``kernels.dspark_mla.save_kernel_shards``) are in
the pre-shard directory, ``dspark-<the draft revision's first 8 characters>``;
the target's LM head and embedding are the decoder's own.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.bridge import (
    hbm,
    kernel_mesh,
    materialize,
    register_op,
    upload,
)
from vllm_torchtpu.megakernel.kimi_k3 import config as kernel_config
from vllm_torchtpu.megakernel.kimi_k3.program import TP_SIZE, kernel_modules

logger = init_logger(__name__)

AXIS = "tp"
TARGET_WEIGHTS = {"lm_head": "lm_head", "embed": "embedding"}
MAX_ROWS = 64


@dataclasses.dataclass
class MlaDraftState:
    """One speculating sequence's draft state: its last output vector
    ``[1, 128]`` int32 (replicated; ``kernels.dspark.OUT_*``)."""

    vector: torch.Tensor


def _settled(state: MlaDraftState) -> MlaDraftState:
    from torch_tpu._internal.sync import synchronize

    synchronize([state.vector], wait=False)
    return state


def _batched_fn(
    dspark: Any,
    fused: Any,
    commit: Any,
    control: Any,
    names: tuple[str, ...],
    layers: int,
    index: tuple[int, ...],
    verified: int,
    rows_per: int,
) -> Any:
    """The batched draft op's global-view function for the requests
    ``index`` of a verify of ``verified`` requests (``MlaDsparkDraft.parts``):
    their rows of the verify's operands, one ``fused`` call, ``commit`` of
    their context latents, then their control entries. Module-level so tests
    can drive it with stubs."""
    width, pages = dspark.ROWS, control.pages
    count = len(names)
    R, k = len(index), rows_per
    total = R * k
    whole = index == tuple(range(verified))
    picked = np.asarray(index, np.int32)
    rows = (picked[:, None] * k + np.arange(k, dtype=np.int32)[None, :]).reshape(-1)

    def fn(*args):
        import jax.numpy as jnp

        weights = {name: args[i][0] for i, name in enumerate(names)}
        pools = args[count : count + layers]
        previous = jnp.concatenate(
            args[count + layers : count + layers + R], axis=0
        )  # [R, 128]
        aux, logits, ids, state, decisions = args[count + layers + R :]
        n = verified * k  # the verify's rows
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
        bases = positions.reshape(R, k)[:, 0]
        vectors, latents = fused(
            weights,
            pools,
            aux,
            logits[0].reshape(R, k, logits.shape[-1]),
            previous,
            bases,
            decisions[:, 0],
            decisions[:, 1],
            tables,
        )
        row_positions = (
            bases[:, None] + jnp.arange(k, dtype=jnp.int32)[None, :]
        ).reshape(total)
        pools = commit(pools, latents, jnp.repeat(tables, k, axis=0), row_positions)
        accepted_all, bonus_all = (
            vectors[:, dspark.OUT_COUNT],
            vectors[:, dspark.OUT_BONUS],
        )
        tokens_r, indices_r = tokens.reshape(R, k), indices.reshape(R, k)
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
        if not whole:
            plan = plan_all.at[picked].set(plan)
        state = jnp.concatenate(
            (
                banks.reshape(-1),
                plan.reshape(-1),
                state[control.plan_start + verified * control.FIELDS :],
            )
        )
        return (*(vectors[j : j + 1] for j in range(R)), *pools, emitted, state)

    return fn


class MlaDsparkDraft:
    """The fused MLA DSpark draft: weights, pools and the step ops."""

    imports_ring = False  # the context is in the draft pools, not a KDA-row ring

    def __init__(
        self,
        max_position: int,
        checkpoint: str,
        presharded: Path,
        synthetic_rates: tuple[float, ...] | None = None,
        synthetic_rates_by_width: dict[int, tuple[float, ...]] | None = None,
    ):
        (self.dspark, self.mla, self.kernel, self.paged) = kernel_modules(
            "dspark", "dspark_mla", "dspark_mla_kernel", "paged"
        )
        self.synthetic_rates = synthetic_rates
        self.synthetic_rates_by_width = synthetic_rates_by_width
        self.config = self.mla.MlaDraftConfig.from_checkpoint(Path(checkpoint))
        self.mesh, self.rank = kernel_mesh(AXIS, TP_SIZE)
        self.attention_dp = kernel_config.active().attention_dp
        if self.attention_dp not in (1, 2):
            raise ValueError("the MLA DSpark draft runs on TP32 or under attention DP2")
        self.max_position = int(max_position)
        self.names = tuple(self.kernel.WEIGHT_NAMES)
        self.weights: dict[str, torch.Tensor] = {}
        self.pools: list[torch.Tensor] = []
        self.owner_table: tuple[int, ...] | None = None
        self._presharded = (
            Path(presharded) / f"dspark-{Path(checkpoint).resolve().name[:8]}"
        )
        self._ops: dict[Any, Any] = {}
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
        return self._aux_layers

    def _synthetic_rates(self, drafts: int) -> tuple[float, ...] | None:
        if self.synthetic_rates_by_width is not None:
            return self.synthetic_rates_by_width[drafts]
        return self.synthetic_rates

    # ------------------------------------------------------------ weights and pools
    def load_weights(self) -> float:
        """This rank's kernel shards and the RoPE table (to ``max_position``)."""
        started = time.perf_counter()
        shards = dict(self.mla.load_rank_shards(self._presharded, self.rank))
        if self.attention_dp > 1:
            shards.update(
                self.mla.held_shards(
                    self._presharded, self.rank, self.attention_dp, shards
                )
            )
        # The kernel computes one 128-lane tile of the fused q_a | kv_a
        # projection per rank and gathers the rest.
        shards["qkv_tile"], shards["context_tile"] = self.mla.qkv_tiles(
            shards.pop("qkv_a"), self.rank
        )
        for name, value in shards.items():
            self.weights[name] = upload(np.ascontiguousarray(value))
        self.weights["rope"] = upload(
            self.mla.rope_table(self.config, self.max_position + 96)
        )
        materialize(self.weights["rope"])
        wall = time.perf_counter() - started
        if self.rank == 0:
            gib = (
                sum(t.numel() * t.element_size() for t in self.weights.values()) / 2**30
            )
            logger.info(
                "K3 DSpark (MLA): draft weights resident (%.2f GiB per rank, "
                "RoPE to %d) in %.1f s; HBM %s",
                gib,
                self.max_position,
                wall,
                hbm(self.rank),
            )
        return wall

    def attach_target(self, target_weights: dict[str, torch.Tensor]) -> None:
        """The target's LM head and embedding (the decoder's), and the mask
        token's embedding row from it."""
        for name, target in TARGET_WEIGHTS.items():
            self.weights[name] = target_weights[target]
        mask, vocab = self.config.mask_token, self.config.vocab // TP_SIZE

        def body(embedding):
            import jax
            import jax.numpy as jnp

            rank = jax.lax.axis_index(AXIS)
            local = mask - rank * vocab
            row = embedding[0][jnp.clip(local, 0, vocab - 1)].astype(jnp.float32)
            row = jnp.where((local >= 0) & (local < vocab), row, 0.0)
            return (jax.lax.psum(row, AXIS).astype(jnp.bfloat16)[None, None],)

        op = register_op(
            "pallas::k3_dspark_mla_mask_embed",
            self.mesh,
            body,
            (P(AXIS),),
            (P(AXIS),),
            export_key=repr(("dspark_mla_mask_embed", mask, vocab)),
        )
        (self.weights["mask_embed"],) = op(self.weights["embed"])

    def bind_pools(self, template: torch.Tensor, owner_table, pools=()) -> None:
        """The draft layers' pools: vLLM's (``pools``: the draft layers' KV
        caches, in the target's MLA group, so the connector transfers them),
        else five allocated like the target's MLA pool ``template`` (synthetic
        fixtures). Read through the target's block tables and page owners."""
        self.owner_table = (
            None if owner_table is None else tuple(int(i) for i in owner_table)
        )
        if pools:
            if len(pools) != self.config.layers or any(
                p.shape != template.shape or p.dtype != template.dtype for p in pools
            ):
                raise RuntimeError(
                    f"K3 DSpark (MLA): expected {self.config.layers} draft pools "
                    f"like {tuple(template.shape)} {template.dtype}"
                )
            self.pools = list(pools)
        elif not (self.pools and self.pools[0].shape == template.shape):
            self.pools = [torch.zeros_like(template) for _ in range(self.config.layers)]
        if self.rank == 0:
            logger.info(
                "K3 DSpark (MLA): %d draft pools %s %s; HBM %s",
                len(self.pools),
                tuple(template.shape),
                template.dtype,
                hbm(self.rank),
            )

    # ------------------------------------------------------------ state
    def new_state(self, anchor: int | torch.Tensor) -> MlaDraftState:
        """A state whose next verify is ``anchor`` with placeholder drafts."""
        vector = torch.full((1, 128), -1, dtype=torch.int32).to("tpu")
        if not isinstance(anchor, torch.Tensor):
            anchor = torch.tensor([int(anchor)], dtype=torch.int32).to("tpu")
        anchor = anchor.reshape(1).to(torch.int32)
        vector[0, : self.dspark.ROWS - 1] = anchor
        vector[0, self.dspark.OUT_BONUS] = anchor[0]
        return _settled(MlaDraftState(vector.clone()))

    def from_ring(
        self, *args
    ) -> MlaDraftState:  # pragma: no cover - the MLA draft has no ring
        raise RuntimeError(
            "the MLA DSpark draft keeps its context in pools, not a ring"
        )

    # ------------------------------------------------------------ ops
    def _operands(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.weights[n] for n in self.names)

    def _commit(self, pools, latents, tables, positions):
        """Commit ``latents`` ``[layers, T, 640]`` (T rows at ``positions``,
        ``tables`` ``[T, pages]``) to the draft pools on their owners. Under
        attention DP the rows are group-major and each group commits only its
        own: its ranks' pool rows hold other requests' pages for the other
        groups' block ids."""
        dp = self.attention_dp
        if dp > 1:
            import jax

            group = jax.lax.axis_index(AXIS) // (TP_SIZE // dp)
            size = positions.shape[0] // dp
            latents = jax.lax.dynamic_slice_in_dim(latents, group * size, size, 1)
            tables = jax.lax.dynamic_slice_in_dim(tables, group * size, size)
            positions = jax.lax.dynamic_slice_in_dim(positions, group * size, size)
        rows = latents.transpose(1, 0, 2)
        return self.paged.commit_latent_rows(
            pools,
            rows,
            tables,
            positions,
            tuple(range(self.config.layers)),
            owner_table=self.owner_table,
        )

    def _build_single(self, control: Any) -> Any:
        key = ("single", control.key())
        if key in self._ops:
            return self._ops[key]
        dspark, rows, names = self.dspark, self.dspark.ROWS, self.names
        count, layers, pages = len(names), self.config.layers, control.pages
        fused = self.kernel.fused_draft_local(
            self.config, requests=1, pages=pages, owner_table=self.owner_table
        )

        def body(*args):
            import jax.numpy as jnp

            weights = {name: args[i][0] for i, name in enumerate(names)}
            pools = args[count : count + layers]
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
                pools, latents, jnp.broadcast_to(table[None], (rows, pages)), positions
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
            return (vector, *pools, emitted[None], following_ids, state)

        in_specs = (P(AXIS),) * (count + layers) + (P(), P(AXIS), P(), P(), P(), P())
        out_specs = (P(),) + (P(AXIS),) * layers + (P(), P(), P())
        self._ops[key] = register_op(
            "pallas::k3_dspark_mla_draft_single",
            self.mesh,
            body,
            in_specs,
            out_specs,
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
        )
        return self._ops[key]

    @staticmethod
    def parts(requests: int, dp: int = 1, rows_per: int = 8) -> list[tuple[int, ...]]:
        """The verify's requests each fused call drafts: at most
        ``MAX_ROWS // rows_per``; under attention DP (``dp``) as many of each group's
        (group ``g``'s are the verify's ``g``-th share), group-major as the
        kernel attends them."""
        share = requests // dp
        per = min(requests, MAX_ROWS // rows_per) // dp
        return [
            tuple(g * share + first + i for g in range(dp) for i in range(per))
            for first in range(0, share, per)
        ]

    def _build_batched(
        self, control: Any, index: tuple[int, ...], verified: int, rows_per: int
    ) -> Any:
        """The draft after a batched verify of ``verified`` requests for its
        requests ``index`` (``parts``)."""
        key = ("batched", control.key(), index, verified, rows_per)
        if key in self._ops:
            return self._ops[key]
        names = self.names
        count, layers = len(names), self.config.layers
        dp = self.attention_dp
        R, k = len(index), rows_per
        whole = index == tuple(range(verified))
        fused = self.kernel.fused_draft_local(
            self.config,
            requests=R,
            pages=control.pages,
            block=min(self.config.block, k - 1),
            rows=k,
            owner_table=self.owner_table,
            attention_dp=dp,
        )
        body = _batched_fn(
            self.dspark, fused, self._commit, control, names, layers, index, verified, k
        )
        in_specs = (
            (P(AXIS),) * (count + layers) + (P(),) * R + (P(), P(AXIS), P(), P(), P())
        )
        out_specs = (P(),) * R + (P(AXIS),) * layers + (P(), P())
        part = "" if whole else f"_part{index[0]}of{verified}"
        self._ops[key] = register_op(
            f"pallas::k3_dspark_mla_draft_r{R}x{k}{part}",
            self.mesh,
            body,
            in_specs,
            out_specs,
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
        )
        return self._ops[key]

    def _adopt(self, pools) -> None:
        for old, fresh in zip(self.pools, pools, strict=True):
            if fresh is not old:
                old.set_(fresh)

    def step_single(self, state, aux, logits, ids, control_state, decisions, control):
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
        emitted, following, control_state = out[1 + layers :]
        self._adopt(pools)
        return _settled(MlaDraftState(vector)), emitted, following, control_state

    def step_batched(
        self, states, aux, logits, ids, control_state, decisions, control, rows_per
    ):
        verified, layers = len(states), self.config.layers
        new: list[Any] = [None] * verified
        emitted, order = [], []
        for index in self.parts(verified, self.attention_dp, rows_per):
            R = len(index)
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
            vectors, pools = out[:R], out[R : R + layers]
            part_emitted, control_state = out[R + layers :]
            self._adopt(pools)
            for j, vector in zip(index, vectors, strict=True):
                new[j] = _settled(MlaDraftState(vector))
            emitted.append(part_emitted)
            order.extend(index)
        tokens = emitted[0] if len(emitted) == 1 else torch.cat(emitted)
        if order != list(range(verified)):
            tokens = tokens.index_select(0, self._request_rows(tuple(order)))
        return new, tokens, control_state

    def _request_rows(self, order: tuple[int, ...]) -> torch.Tensor:
        """The row of each request in the parts' concatenated tokens."""
        key = ("rows", order)
        if key not in self._ops:
            self._ops[key] = torch.tensor(
                np.argsort(np.asarray(order)), dtype=torch.int64
            ).to("tpu")
        return self._ops[key]

    def compilation_calls(self, control: Any, spec_alone: bool, spec_shapes) -> list:
        from vllm_torchtpu.megakernel.compilation import CompileCall

        c = self.config
        operands = self._operands()
        vector = torch.empty((1, 128), dtype=torch.int32)
        state = torch.empty(control.size, dtype=torch.int32)

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

        calls = []
        if spec_alone:
            rows = self.dspark.ROWS
            aux, logits, ids, state_, decisions = tail(
                rows, 3 * rows + 1 + rows * control.pages, 1
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
                        state_,
                        decisions,
                    ),
                    {},
                    priority=2 * rows,
                )
            )
        for requests, spec_rows in spec_shapes:
            rows = requests * spec_rows
            for index in self.parts(requests, self.attention_dp, spec_rows):
                calls.append(
                    CompileCall(
                        self._build_batched(control, index, requests, spec_rows),
                        (
                            *operands,
                            *self.pools,
                            *((vector,) * len(index)),
                            *tail(rows, 4 * rows + requests * control.pages, requests),
                        ),
                        {},
                        priority=2 * rows,
                    )
                )
        return calls
