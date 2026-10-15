# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 decode: the decode role's program over vLLM's unified pools.

:class:`KimiDecoder` runs on one TP32 kernel mesh (all devices in ascending id;
a worker's kernel rank is its mesh position, not its TP rank) over
:class:`DecodeOps` and, when speculating, a :class:`DsparkDraft`. MLA is decode
context parallel, reading and committing latent pages in place where the KV
connector landed them; KDA state lives in private slots (``pools.SlotMap``),
imported with the prefill's DSpark ring on a request's first step. Every op
reads and returns a replicated control vector (:class:`Control`) holding each
continuing request's next input, so nothing is uploaded per step while the
host's tokens lag (async scheduling). A step is one of:

* one request, seven drafts: :meth:`DecodeOps.verify_packed`, then
  :meth:`DsparkDraft.step_single`;
* three drafts alone, or several requests: :meth:`DecodeOps.verify_batched`,
  then :meth:`DsparkDraft.step_batched` (each request keeps its own ring);
* otherwise a plain step (:meth:`DecodeOps.plain_step`), also whenever a
  structured-output request decodes (grammar masks: the verify ops do not
  mask, so the whole batch steps plain, without drafts).

The verify ops decide acceptance and commit only the selected KDA post-state.
Under attention DP a request's pages and KDA state live on the DP group its MLA
block ids name (``megakernel.dp_blocks``), and it takes one of that group's
batch rows (``_group_rows``).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.bridge import hbm, materialize, register_global_op
from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.megakernel.kimi_k3.decode_ops import (
    GREEDY,
    Control,
    DecodeOps,
    dp_row_owners,
)
from vllm_torchtpu.megakernel.kimi_k3.pools import (
    KDA_LAYERS,
    KdaImporter,
    SlotMap,
    kda_blocks_at_length,
    tp_rank_permutation,
)
from vllm_torchtpu.megakernel.kimi_k3.program import (
    SPEC_BATCHES,
    VOCAB,
    kernel_modules,
    spec_tokens_per_batch,
)
from vllm_torchtpu.megakernel.program import (
    KVGroup,
    RequestInput,
    StepInput,
    StepOutput,
)

logger = init_logger(__name__)


class KimiDecoder:
    """The decode role's :class:`MegakernelProgram` over ``ops`` and, when
    ``config`` speculates, the DSpark ``draft`` (None: plain steps only). Slot 0
    is the null slot padding rows use; slots below ``sub_base`` are the regular
    ones (live requests plus those whose release is in flight). ``spec_shapes``
    holds the compiled (padded request count, verify rows per request) pairs."""

    WIDTH = DecodeOps.VERIFY_ROWS  # tokens per row of a step's result

    def __init__(
        self,
        config: MegakernelEngineConfig,
        ops: DecodeOps,
        draft: Any = None,
        *,
        tp_rank: int,
    ) -> None:
        self.config, self.ops, self.draft = config, ops, draft
        self.tp_rank, self.rank = tp_rank, ops.rank
        self.sub_base = 2 * config.max_num_seqs + 1
        needed = Control.total_slots(self.sub_base) if draft else self.sub_base
        if ops.num_slots < needed:
            raise ValueError(
                f"K3 decode: the ops hold {ops.num_slots} KDA slots; "
                f"--max-num-seqs {config.max_num_seqs} needs {needed}"
            )
        self.slot_map = SlotMap(self.sub_base, int(ops.attention_dp))
        self.layout: Control | None = None
        self.control: torch.Tensor | None = None
        self.device_plan: np.ndarray | None = None
        self.tracked: set[str] = set()  # requests whose next input the control holds
        self.spec_req: str | None = None
        self.draft_state = None
        self.next_ids: torch.Tensor | None = None
        # Rings the prefill computed, per request, for its first speculation.
        self._offered: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # With a draft, the schedule (vLLM's per-batch-size draft count,
        # validated in kimi_k3.program) says which batch sizes speculate; every
        # compiled verifier/drafter shape is derived from it.
        self.spec_tokens = (spec_tokens_per_batch(config) or ()) if draft else ()
        shapes = set()
        for b, k in enumerate(self.spec_tokens):
            if k and (b >= 2 or k == 3):
                compiled = next((s for s in SPEC_BATCHES if s >= b), 0)
                if not compiled or compiled > max(ops.batch_sizes):
                    raise ValueError(
                        f"K3 decode: no compiled speculation batch holds {b} "
                        f"requests (batches {SPEC_BATCHES}, up to "
                        f"{max(ops.batch_sizes)})"
                    )
                shapes.add((compiled, k + 1))
        self.spec_shapes = tuple(sorted(shapes))
        self.spec_batches = tuple(sorted({b for b, _ in shapes}))
        self.spec_alone = (
            len(self.spec_tokens) > 1 and self.spec_tokens[1] == self.WIDTH - 1
        )
        # Batched speculation: each request's draft state between its steps,
        # and the verify rows its draft vector's drafts were made for (only
        # while they still follow its next token).
        self._drafts: dict[str, Any] = {}
        self._fresh: dict[str, int] = {}
        self._mode_rows: dict[tuple[int, ...], torch.Tensor] = {}
        self._pads: list[Any] = []  # draft states of padding groups
        self._compiled_layout: tuple | None = None
        self._zero: torch.Tensor | None = None
        self._modes: dict[int, torch.Tensor] = {}
        self._live_batches: set[int] = set()
        self.stats = {"plans": 0, "enters": 0, "chained": 0, "plain": 0, "batched": 0}
        self.groups = int(ops.attention_dp)
        self._group: dict[str, tuple[int, int]] = {}  # request -> (DP group, context)
        if self.groups > 1 and draft is not None:
            raise ValueError("K3 decode: attention DP decodes without speculation")
        # Import each transferred request's DSpark ring: with a DSpark
        # speculative config the transfer carries one (backend.draft_ring_rows).
        self._ring = draft is not None
        self._ring_reports = 4
        self._importer: KdaImporter | None = None
        self.destination_ranks: tuple[int, ...] | None = None
        self._saw_request = False
        self._pages: int | None = None
        self._num_blocks = 0
        self._mla_group = 0
        self._kda_groups: list[tuple[int, int]] = []

    def load_weights(self) -> None:
        started = time.perf_counter()
        self.ops.load_weights()
        if self.draft is not None and not self.draft.weights:
            self.draft.load_weights()
        if self.rank == 0:
            logger.info(
                "K3 decode: weights resident in %.1f s",
                time.perf_counter() - started,
            )

    def bind_kv_caches(
        self, caches: Mapping[str, torch.Tensor], groups: Sequence[KVGroup]
    ) -> None:
        """Decode from vLLM's unified pools; ``groups`` say which block-table
        row the decoder reads (the MLA layers') and where each KDA layer's
        state blocks are (for a request's import)."""
        self.load_weights()
        # A device's KV transfer ordinal is its mesh position: the connector
        # lists workers in mesh order (``destination_ranks``: TP rank per position).
        perm = tp_rank_permutation(self.tp_rank, self.rank)
        self.destination_ranks = (
            None if perm is None else tuple(q for _, q in sorted(perm))
        )
        # One importer per ops (it registers torch ops, whose names are
        # process-global), shared by the decoders over them.
        self._importer = self.ops.cached_op(
            ("kda_importer", None),
            lambda: KdaImporter(
                self.ops.mesh, None, self.ops.attention_dp, self.ops.num_slots
            ),
        )
        num_blocks = self._importer.bind(caches, self.config.kv_cache_dtype)
        self._num_blocks = num_blocks
        self.ops.use_unified_pools(self._importer.raw_pools, self._importer.mla_raw)
        of = {name: g for g, group in enumerate(groups) for name in group.layer_names}
        mla_groups = {of[n] for n in self._importer.mla_layer_names if n in of}
        if len(mla_groups) != 1:
            raise RuntimeError(
                f"K3 decode: the MLA layers are in KV cache groups {sorted(mla_groups)}"
            )
        self._mla_group = mla_groups.pop()
        missing = [n for n in self._importer.kda_layer_names if n not in of]
        if missing:
            raise RuntimeError(
                f"K3 decode: KDA layers {missing[:3]}... are in no KV cache group"
            )
        self._kda_groups = [
            (of[n], int(groups[of[n]].block_size))
            for n in self._importer.kda_layer_names
        ]
        self._pages = int(groups[self._mla_group].table_width)
        if self.rank == 0:
            logger.info(
                "K3 decode: DCP decoder on vLLM's %d unified pools (%d blocks); HBM %s",
                len(self._importer.raw_pools),
                num_blocks,
                hbm(self.rank),
            )

    def warmup(self) -> None:
        if self._pages is None:
            raise RuntimeError("K3 decode: bind the KV caches before warming up")
        started = time.perf_counter()
        self._warm_importer()
        self._warm_steps()
        self._warm_importer()  # an import after the steps (another operand history)
        self._warm_steps()
        if self.rank == 0:
            logger.info("K3 decode: ops warm in %.1f s", time.perf_counter() - started)

    def step(self, step_input: StepInput) -> StepOutput:
        """One decode step (see the module docstring)."""
        if self._pages is None:
            raise RuntimeError(
                "K3 decode: a request arrived before the KV pools were bound"
            )
        if step_input.copies:
            raise NotImplementedError(
                "K3 decode: copy-on-write block copies (prefix hits inside a block)"
            )
        mask = step_input.mask
        if mask is not None and mask.shape != (len(step_input.requests), VOCAB // 32):
            raise ValueError(
                f"K3 decode: grammar masks {mask.shape} do not cover "
                f"{len(step_input.requests)} requests of the {VOCAB}-token "
                "vocabulary"
            )
        self.forget(step_input.finished)
        return StepOutput(self._decode(step_input.requests, mask))

    def _admit(self, request: RequestInput, slot: int) -> None:
        """A request's first step: import its KDA state after the tokens
        computed before this step into ``slot``, and its DSpark ring. A
        request with no state block to import (none computed, or only the
        null block) starts from a clean state."""
        # Each KDA layer's state after the computed tokens: the block holding
        # position computed - 1, where the transfer committed it. (When the
        # computed tokens end on a block boundary the request already holds
        # the next block too; vLLM's runners would copy the state into it.)
        kda = kda_blocks_at_length(
            request.block_tables, self._kda_groups, request.computed
        )
        if self.groups > 1 and int(kda.max()) >= self._num_blocks:
            raise RuntimeError(
                f"K3 decode: KDA state blocks {kda.tolist()} are not shared blocks"
            )
        # The decoder has no position-0 reset. Under attention DP the importer
        # clears the slot (from the null blocks) on the request's group only.
        if not kda.any() and self.groups == 1:
            for state in self.ops.slots:
                state[0, slot].zero_()
            return
        assert self._importer is not None
        started = time.perf_counter()
        kda_blocks = torch.from_numpy(kda).to("tpu")
        conv, recurrent = self._importer.import_state(*self.ops.slots, kda_blocks, slot)
        self.ops.slots = [conv, recurrent]
        if self._ring:
            # The DSpark ring the prefill wrote beside the state (cold unless
            # it wrote one for exactly these tokens): the request's first
            # speculation starts from it instead of a cold ring.
            cache, positions = self._importer.import_ring(
                self.draft.config, kda_blocks, request.computed
            )
            self._offered[request.req_id] = (cache, positions)
            if self._ring_reports:  # the first few imports only (a host read each)
                self._ring_reports -= 1
                held = int((positions >= 0).sum().cpu())
                if self.rank == 0:
                    logger.info(
                        "K3 decode: request %s's draft ring holds %d positions",
                        request.req_id,
                        held,
                    )
        if self.rank == 0:
            logger.info(
                "K3 decode: imported request %s (%d tokens) into slot %d in %.1f ms",
                request.req_id,
                request.computed,
                slot,
                1e3 * (time.perf_counter() - started),
            )
            if not self._saw_request:
                self._saw_request = True
                logger.info("K3 decode: first real request; HBM %s", hbm(self.rank))

    def _null_ring(self) -> tuple[torch.Tensor, torch.Tensor]:
        """An imported ring from the null block (for warming up)."""
        assert self.draft is not None and self._importer is not None
        return self._importer.import_ring(
            self.draft.config, torch.zeros(KDA_LAYERS, dtype=torch.int32).to("tpu"), 0
        )

    def _warm_importer(self) -> None:
        """Compile the KDA import on the null slot and block."""
        assert self._importer is not None
        started = time.perf_counter()
        conv, recurrent = self._importer.import_state(
            *self.ops.slots, torch.zeros(KDA_LAYERS, dtype=torch.int32).to("tpu"), 0
        )
        self.ops.slots = [conv, recurrent]
        materialize(recurrent)
        if self.rank == 0:
            logger.info(
                "K3 decode: warmed the KDA import in %.1f s",
                time.perf_counter() - started,
            )

    # ------------------------------------------------------------ state and helper ops
    def _prepare(self, pages: int) -> Control:
        if self.layout is None or self.layout.pages != pages:
            self.layout = Control(self.sub_base, max(self.ops.batch_sizes), pages)
            self.control = None
        if self.control is None:
            self.control = torch.zeros(self.layout.size, dtype=torch.int32).to("tpu")
            self.device_plan = None
        if self._zero is None:
            self._zero = torch.zeros(1, dtype=torch.int32).to("tpu")
            self._modes = {
                0: torch.zeros(1, dtype=torch.int32).to("tpu"),
                1: torch.ones(1, dtype=torch.int32).to("tpu"),
            }
        return self.layout

    def _splice(self) -> Any:
        """``(control, plan) -> control`` with the plan part replaced."""
        layout = self.layout
        assert layout is not None
        key = ("splice", layout.key())

        def build():
            import jax.numpy as jnp

            def fn(control, plan):
                return (jnp.concatenate((control[: layout.plan_start], plan)),)

            return register_global_op(
                "pallas::k3_decode_control_splice",
                self.ops.mesh,
                fn,
                (P(), P()),
                (P(),),
                export_key=repr(key),
            )

        return self.ops.cached_op(key, build)

    def _enter(self) -> Any:
        """``(control, vector, keep) -> ids``: the verify operand of plan row
        0's request from the control: its next token and after it placeholder
        drafts (``keep`` 0) or the last draft vector's drafts (``keep`` 1:
        the same request, a new plan), at its next positions; every row its
        slot; its state row; the plan's table."""
        layout, sub_base = self.layout, self.sub_base
        assert layout is not None
        key = ("enter", layout.key(), sub_base)

        def build():
            import jax.numpy as jnp

            rows = self.WIDTH
            (dspark,) = kernel_modules("dspark")

            def fn(control, vector, keep):
                slot, token, position, row, _, tables = layout.inputs(control, 1)
                drafts = vector[0, dspark.OUT_DRAFTS : dspark.OUT_DRAFTS + rows - 1]
                tokens = jnp.where(
                    keep[0] != 0,
                    jnp.concatenate((token, drafts)),
                    jnp.full((rows,), token[0]),
                )
                sub = jnp.concatenate(
                    (
                        slot,
                        layout.state_slot(slot, jnp.arange(1, rows, dtype=jnp.int32)),
                    )
                )
                ids = jnp.concatenate(
                    (
                        tokens,
                        position[0] + jnp.arange(rows, dtype=jnp.int32),
                        sub,
                        row,
                        jnp.tile(tables[0], rows),
                    )
                )
                return (ids.astype(jnp.int32),)

            return register_global_op(
                "pallas::k3_decode_dspark_enter",
                self.ops.mesh,
                fn,
                (P(), P(), P()),
                (P(),),
                export_key=repr(key),
            )

        return self.ops.cached_op(key, build)

    def _enter_batched(self, requests: int, rows_per: int) -> Any:
        """``(control, vector_0.., modes) -> ids``: the batched verify operand
        (``DecodeOps.verify_batched``) of plan rows 0..requests-1: per
        request its next token, then the first drafts of its draft vector
        (``modes`` 1) or placeholders (0), at its next positions; every row
        its request's slot (``indices``: each row's logical entry); one table
        per request."""
        layout, k = self.layout, rows_per
        assert layout is not None
        key = ("enter_batched", layout.key(), requests, k)

        def build():
            import jax.numpy as jnp

            (dspark,) = kernel_modules("dspark")
            R = requests

            def fn(control, *rest):
                vectors, modes = rest[:R], rest[R]
                slot, token, position, row, _, tables = layout.inputs(control, R)
                others = jnp.arange(1, layout.BANK + 1, dtype=jnp.int32)
                tokens, positions, slots, indices = [], [], [], []
                for j in range(R):
                    drafts = vectors[j][
                        0, dspark.OUT_DRAFTS : dspark.OUT_DRAFTS + k - 1
                    ]
                    tokens.append(
                        jnp.where(
                            modes[j] != 0,
                            jnp.concatenate((token[j : j + 1], drafts)),
                            jnp.full((k,), token[j]),
                        )
                    )
                    positions.append(position[j] + jnp.arange(k, dtype=jnp.int32))
                    rest_entries = jnp.where(others <= row[j], others - 1, others)
                    entry = jnp.concatenate((row[j : j + 1], rest_entries[: k - 1]))
                    indices.append(entry)
                    slots.append(layout.state_slot(slot[j], entry))
                ids = jnp.concatenate(
                    (*tokens, *positions, *slots, *indices, tables.reshape(-1))
                )
                return (ids.astype(jnp.int32),)

            return register_global_op(
                f"pallas::k3_decode_dspark_enter{requests}x{k}",
                self.ops.mesh,
                fn,
                (P(),) * (R + 2),
                (P(),),
                export_key=repr(key),
            )

        return self.ops.cached_op(key, build)

    def _spec_batch(self, requests: int) -> int:
        """The compiled batched-speculation size holding ``requests``
        decoding requests (padded with null ones), or 0: none."""
        if requests < 1 or (requests == 1 and 1 not in self.spec_batches):
            return 0
        return next((b for b in self.spec_batches if b >= requests), 0)

    def _modes_for(self, modes: tuple[int, ...]) -> torch.Tensor:
        if modes not in self._mode_rows:
            self._mode_rows[modes] = torch.tensor(modes, dtype=torch.int32).to("tpu")
        return self._mode_rows[modes]

    def _take_state(self, req_id: str) -> Any:
        """``req_id``'s draft state: kept from its last speculation, else the
        ring the prefill computed, else a cold one."""
        assert self.draft is not None
        state = self._drafts.pop(req_id, None)
        if state is not None:
            return state
        offered = self._offered.pop(req_id, None)
        if offered is not None:
            return self.draft.from_ring(*offered, self._zero)
        return self.draft.new_state(self._zero)

    def _park_single(self) -> None:
        """The single-request speculation's state back with its request."""
        if self.spec_req is not None and self.draft_state is not None:
            self._drafts[self.spec_req] = self.draft_state
        self.spec_req, self.draft_state, self.next_ids = None, None, None

    def _sync_plan(self, entries, tables) -> bool:
        """Upload the plan for ``entries`` unless the device holds it; returns
        whether it uploaded (the verify operand's table may have changed)."""
        layout = self.layout
        assert layout is not None
        plan = layout.plan(entries, tables)
        if self.device_plan is not None and np.array_equal(plan, self.device_plan):
            return False
        (self.control,) = self._splice()(self.control, torch.from_numpy(plan).to("tpu"))
        self.stats["plans"] += 1
        self.device_plan = layout.consumed(plan, len(entries))
        return True

    def _replay(
        self,
        slot: int,
        tokens: np.ndarray,
        start: int,
        table: np.ndarray,
        group: int = 0,
    ) -> None:
        """Tokens computed on this role (a recompute after a preemption):
        one batch-one plain step per token (on a row of its DP ``group``),
        their outputs discarded."""
        batch = self.ops.batch_for(1)
        for offset, token in enumerate(tokens):
            entries = [(slot, True, int(token), start + offset, GREEDY)]
            if self.groups > 1:
                _, entries, tables, _ = self._place(
                    [group], entries, [table], len(table)
                )
                self._sync_plan(entries, tables)
            else:
                self._sync_plan(entries, [table])
            _, self.control = self.ops.plain_step(
                self.control, batch, self.layout, self.sub_base, False
            )

    def forget(self, finished: Iterable[str]) -> None:
        finished = set(finished or ())
        if not finished:
            return
        self.slot_map.release(finished)
        self.tracked -= finished
        for req in finished:
            self._group.pop(req, None)
            self._offered.pop(req, None)
            self._drafts.pop(req, None)
            self._fresh.pop(req, None)
        if self.spec_req in finished:
            self.spec_req, self.draft_state, self.next_ids = None, None, None

    # ------------------------------------------------------------ attention DP
    def _mla_table(self, request: RequestInput):
        """``request``'s MLA block table as pool rows. Under attention DP
        the scheduler's ids are ``g N + r``: DP group ``g``'s row ``r``
        (``megakernel.dp_blocks``); the request keeps that group."""
        table = request.block_tables[self._mla_group]
        if self.groups > 1:
            group = int(table.max()) // self._num_blocks
            self._group[request.req_id] = (group, request.end)
            table = table % self._num_blocks
        return table

    def _group_of(self, request: RequestInput) -> int:
        """``request``'s DP group: the scheduler's, else chosen on its
        first step: the group with the fewest requests, then the least
        context."""
        if request.req_id not in self._group:
            count = [0] * self.groups
            context = [0] * self.groups
            for group, length in self._group.values():
                count[group] += 1
                context[group] += length
            group = min(range(self.groups), key=lambda g: (count[g], context[g]))
            self._group[request.req_id] = (group, request.end)
        return self._group[request.req_id][0]

    def _group_rows(self, batch: int, group: int) -> list[int]:
        """The batch rows DP group ``group`` attends: ``p * B / O + group * r + j``
        (``O`` local row owners, ``dp_row_owners``; ``r = B / (O dp)``)."""
        owners = dp_row_owners(batch)
        per = batch // self.groups // owners
        return [
            p * (batch // owners) + group * per + j
            for p in range(owners)
            for j in range(per)
        ]

    def _place(self, groups: list[int], entries, tables, pages: int):
        """Plan rows for requests of DP ``groups`` (in order): ``(batch,
        entries, tables, plan row of each request)``, the other rows null."""
        load = [groups.count(g) for g in range(self.groups)]
        batch = self.ops.batch_for(self.groups * max(load))
        free = [self._group_rows(batch, g) for g in range(self.groups)]
        null_entry, null_table = (0, True, 0, 0, GREEDY), np.zeros(pages, np.int32)
        placed_entries, placed_tables = [null_entry] * batch, [null_table] * batch
        rows = []
        for group, entry, table in zip(groups, entries, tables, strict=True):
            row = free[group].pop(0)
            placed_entries[row], placed_tables[row] = entry, table
            rows.append(row)
        return batch, placed_entries, placed_tables, rows

    # ------------------------------------------------------------ the step
    def _decode(
        self, requests: Sequence[RequestInput], mask: np.ndarray | None = None
    ) -> torch.Tensor:
        """One decode step for ``requests`` (the runner's batch order): the
        tokens ``[>= len(requests), 8]`` int32, row i the request's accepted
        drafts then its bonus token, -1 padded (all -1 for a request that
        takes no token this step: a partial prefill chunk or an unscheduled
        one). ``mask``: the requests' grammar bitmasks (row ``i`` request
        ``i``'s; all-ones unconstrained), applied to the plain step's logits."""
        pages = self._pages
        assert pages is not None
        layout = self._prepare(pages)
        entries, tables, decoding = [], [], []
        for i, request in enumerate(requests):
            if request.scheduled <= 0:
                continue
            table = self._mla_table(request)
            group = self._group_of(request) if self.groups > 1 else 0
            if self.slot_map.group(request.req_id) not in (None, group):
                # Only its old group held its state: placed in another group
                # (after a preemption), it starts there as a new request.
                self.slot_map.release([request.req_id])
            fresh = request.req_id not in self.slot_map
            slot = self.slot_map.allocate(request.req_id, group)
            if fresh:
                self.tracked.discard(request.req_id)
                self._admit(request, slot)
            sampling = request.sampling.as_tuple()
            if request.req_id in self.tracked:
                entries.append((slot, False, 0, 0, sampling))
            else:
                tokens = request.token_ids[request.computed : request.end]
                replay = request.scheduled - (0 if request.partial else 1)
                if replay:
                    self._replay(
                        slot,
                        tokens[:replay],
                        request.computed,
                        table,
                        self._group_of(request) if self.groups > 1 else 0,
                    )
                if request.partial:
                    continue
                entries.append(
                    (
                        slot,
                        True,
                        int(tokens[request.scheduled - 1]),
                        request.end - 1,
                        sampling,
                    )
                )
                self.tracked.add(request.req_id)
            tables.append(table)
            decoding.append((i, request))
        sampled = any(entry[4][0] > 0 for entry in entries)
        emitted = None
        placed = None  # attention DP: the plan row of each decoding request
        # The drafts the schedule gives this step's batch (validated at
        # startup: the schedule never asks for a shape the kernels lack).
        k = (
            self.spec_tokens[len(decoding)]
            if len(decoding) < len(self.spec_tokens)
            else 0
        )
        if mask is not None:
            # A structured request decodes: the whole batch steps plain and
            # masked (the verify ops do not mask), its drafts unused.
            k = 0
        if len(decoding) == 1 and k == self.WIDTH - 1:
            emitted = self._speculate(decoding[0][1], entries, tables, sampled)
        elif k > 0 and (
            len(decoding) == 1 or any(r.scheduled > 1 for _, r in decoding)
        ):
            # A lone request speculates from its anchor-only first step;
            # ``_fresh`` keeps its placeholder draft rows from being accepted.
            emitted = self._speculate_batched(decoding, entries, tables, sampled)
        elif decoding:
            # A request leaving speculation keeps its KDA state in its slot
            # (nothing moves) and its draft state, whose drafts no longer
            # follow its next token.
            self._park_single()
            for _, request in decoding:
                self._fresh.pop(request.req_id, None)
            if self.groups > 1:
                batch, entries, tables, placed = self._place(
                    [self._group_of(r) for _, r in decoding], entries, tables, pages
                )
            else:
                batch = self.ops.batch_for(len(decoding))
            if batch not in self._live_batches:
                self._live_batches.add(batch)
                if self.rank == 0:
                    logger.info(
                        "K3 decode: first live step at batch size %d (%d rows)",
                        batch,
                        len(decoding),
                    )
            self._sync_plan(entries, tables)
            self.stats["plain"] += 1
            mask_operand = None
            if mask is not None:
                # One row per plan row (the decoding requests' rows, in their
                # placement order); padding rows unconstrained.
                mask_operand = np.full((batch, VOCAB // 32), -1, np.int32)
                rows = placed if placed is not None else range(len(decoding))
                for plan_row, (i, _) in zip(rows, decoding, strict=True):
                    mask_operand[plan_row] = mask[i]
                mask_operand = torch.from_numpy(mask_operand).to("tpu")
            emitted, self.control = self.ops.plain_step(
                self.control, batch, layout, self.sub_base, sampled, mask_operand
            )
        decoded = [i for i, _ in decoding]
        if placed is not None:
            # Plan rows back to the runner's rows.
            position_of = dict(zip(decoded, placed, strict=True))
        elif decoded == list(range(len(requests))):
            return emitted
        else:
            position_of = {i: k for k, i in enumerate(decoded)}
        # Rows without a token this step: -1 rows around the decoded ones.
        if emitted is None:
            return torch.full(
                (max(len(requests), 1), self.WIDTH), -1, dtype=torch.int32
            ).to("tpu")
        blank = int(emitted.shape[0])
        index = torch.tensor(
            [position_of.get(r, blank) for r in range(len(requests))], dtype=torch.int64
        )
        padded = torch.cat(
            (emitted, torch.full((1, self.WIDTH), -1, dtype=torch.int32).to("tpu"))
        )
        return padded.index_select(0, index.to("tpu"))

    def _speculate(
        self, request: RequestInput, entries, tables, sampled: bool
    ) -> torch.Tensor:
        changed = self._sync_plan(entries, tables)
        reset = entries[0][1]
        continuing = (
            self.spec_req == request.req_id and self.next_ids is not None and not reset
        )
        if not continuing:
            self._park_single()
            self.spec_req = request.req_id
            self.draft_state = self._take_state(request.req_id)
        assert self.draft_state is not None and self.layout is not None
        if continuing and not changed:
            ids = self.next_ids
            self.stats["chained"] += 1
        else:
            self.stats["enters"] += 1
            (ids,) = self._enter()(
                self.control,
                self.draft_state.vector,
                self._modes[1 if continuing else 0],
            )
        # Accept drafts only when the rows carry this request's real drafts
        # and the scheduler scheduled them all.
        mode = self._modes[1 if continuing and request.scheduled == self.WIDTH else 0]
        sharded, aux, decisions = self.ops.verify_packed(
            ids,
            self.draft.aux_layers,
            self.layout,
            self.control,
            mode,
            self.draft._synthetic_rates(self.WIDTH - 1),
        )
        self.draft_state, emitted, self.next_ids, self.control = self.draft.step_single(
            self.draft_state,
            aux,
            sharded,
            ids,
            self.control,
            decisions,
            self.layout,
        )
        if self.device_plan is not None:
            self.device_plan = self.layout.consumed(self.device_plan, 1)
        self._fresh[request.req_id] = self.WIDTH
        return emitted

    def _speculate_batched(
        self, decoding, entries, tables, sampled: bool
    ) -> torch.Tensor:
        """One batched DSpark step for the decoding rows (module docstring).
        A request's drafts are verified (mode 1) only when they were made for
        at least the scheduled width after its current next token and the
        scheduler scheduled as many tokens; otherwise its rows are
        placeholders."""
        assert self.draft is not None
        self._park_single()
        requests = len(decoding)
        k = self.spec_tokens[requests] + 1
        compiled = self._spec_batch(requests)
        self._sync_plan(entries, tables)  # plan rows past ``requests``: the null slot
        states = [self._take_state(r.req_id) for _, r in decoding]
        while len(states) < compiled:  # padding groups (null slot): reused states
            states.append(
                self._pads.pop() if self._pads else self.draft.new_state(self._zero)
            )
        modes = tuple(
            int(self._fresh.get(r.req_id, 0) >= k and r.scheduled >= k)
            for _, r in decoding
        ) + (0,) * (compiled - requests)
        modes_t = self._modes_for(modes)
        (ids,) = self._enter_batched(compiled, k)(
            self.control, *(s.vector for s in states), modes_t
        )
        sharded, aux, decisions = self.ops.verify_batched(
            ids,
            self.draft.aux_layers,
            compiled,
            k,
            self.layout,
            self.control,
            modes_t,
            self.draft._synthetic_rates(k - 1),
        )
        states, emitted, self.control = self.draft.step_batched(
            states, aux, sharded, ids, self.control, decisions, self.layout, k
        )
        for (_, request), state in zip(decoding, states[:requests], strict=True):
            self._drafts[request.req_id] = state
            self._fresh[request.req_id] = k
        self._pads.extend(states[requests:])
        if self.device_plan is not None and self.layout is not None:
            self.device_plan = self.layout.consumed(self.device_plan, requests)
        self.stats["batched"] += 1
        return emitted

    # ------------------------------------------------------------ warmup
    def _warm_steps(self) -> None:
        """Compile the step ops on the null slot and page, in the orders
        serving runs them (torch_tpu compiles an op per operand buffer
        history): every batch size, a DSpark request entering, continuing,
        changing its table, then batches again; greedy, then sampled; with a
        DSpark ring import, entering from an imported ring once too."""
        started = time.perf_counter()
        pages = self._pages
        assert pages is not None
        ring = self._null_ring if self._ring else None
        self._prepare(pages)
        assert self.layout is not None
        null = np.zeros(pages, np.int32)

        def plain_step(batch: int, sampled: bool, masked: bool = False) -> torch.Tensor:
            params = (0.7, 0.9, 20, 1) if sampled else GREEDY
            self._sync_plan([(0, True, 0, 0, params)] * batch, [null] * batch)
            mask = None
            if masked:
                mask = torch.full(
                    (batch, VOCAB // 32), -1, dtype=torch.int32, device="tpu"
                )
            emitted, self.control = self.ops.plain_step(
                self.control, batch, self.layout, self.sub_base, sampled, mask
            )
            return emitted

        if self._compiled_layout != self.layout.key():
            from torch_tpu._internal.sync import synchronize

            from vllm_torchtpu.megakernel.compilation import compile_ops

            # The first step compiles its own variant (its state buffers come
            # from allocation, not from an op), which the precompiled graphs
            # do not cover: start it first so that compile overlaps theirs.
            first = plain_step(self.ops.batch_sizes[0], False)
            synchronize(
                [first, self.control, *self.ops.slots, *self.ops.unified_pools],
                wait=False,
            )
            calls = self.ops.compilation_calls(
                self.layout,
                self.sub_base,
                self.draft.aux_layers if self.draft is not None else (),
                self.spec_alone,
                self.spec_shapes,
                self.draft._synthetic_rates if self.draft is not None else None,
                masked_ops=True,
            )
            if self.draft is not None:
                calls.extend(
                    self.draft.compilation_calls(
                        self.layout, self.spec_alone, self.spec_shapes
                    )
                )
            calls.sort(key=lambda call: call.priority, reverse=True)
            # These builders capture the same static kernel/configuration
            # on every rank; weights and state are dynamic operands.
            from vllm_torchtpu.megakernel.export_cache import prepare_distributed

            prepare_distributed(calls)
            compile_ops(calls)
            materialize(first)
            self._compiled_layout = self.layout.key()
            if self.rank == 0:
                logger.info(
                    "K3 decode: precompiled %d decode/draft programs in %.1f s",
                    len(calls),
                    time.perf_counter() - started,
                )

        def plain(sampled: bool, masked: bool = False) -> None:
            outs = [
                plain_step(batch, sampled, masked) for batch in self.ops.batch_sizes
            ]
            for emitted in outs:
                materialize(emitted)

        def speculate(sampled: bool, offered=None, state=None) -> None:
            params = (0.7, 0.9, 20, 1) if sampled else GREEDY
            self.device_plan = None
            self._sync_plan([(0, True, 0, 0, params)], [null])
            if state is not None:
                self.draft_state = state
            else:
                self.draft_state = (
                    self.draft.new_state(self._zero)
                    if offered is None
                    else self.draft.from_ring(*offered, self._zero)
                )
            (ids,) = self._enter()(
                self.control, self.draft_state.vector, self._modes[0]
            )
            emitted = None
            for step in range(4):
                # A new plan (a table grew): the operand rebuilt from the control.
                if step == 2:
                    self.device_plan = None
                    self._sync_plan([(0, True, 0, 0, params)], [null])
                    (ids,) = self._enter()(
                        self.control, self.draft_state.vector, self._modes[1]
                    )
                sharded, aux, decisions = self.ops.verify_packed(
                    ids,
                    self.draft.aux_layers,
                    self.layout,
                    self.control,
                    self._modes[min(step, 1)],
                    self.draft._synthetic_rates(self.WIDTH - 1),
                )
                self.draft_state, emitted, ids, self.control = self.draft.step_single(
                    self.draft_state,
                    aux,
                    sharded,
                    ids,
                    self.control,
                    decisions,
                    self.layout,
                )
            materialize(emitted)

        def speculate_batched(
            sampled: bool, requests: int, rows_per: int, offered=None, states=None
        ) -> list:
            params = (0.7, 0.9, 20, 1) if sampled else GREEDY
            assert self.draft is not None
            if states is None:
                states = [self.draft.new_state(self._zero) for _ in range(requests)]
            if offered is not None:
                states[0] = self.draft.from_ring(*offered, self._zero)
            emitted = None
            for step in range(3):
                self.device_plan = None
                self._sync_plan(
                    [(0, step == 0, 0, 0, params)] * requests, [null] * requests
                )
                modes = self._modes_for((min(step, 1),) * requests)
                (ids,) = self._enter_batched(requests, rows_per)(
                    self.control, *(s.vector for s in states), modes
                )
                sharded, aux, decisions = self.ops.verify_batched(
                    ids,
                    self.draft.aux_layers,
                    requests,
                    rows_per,
                    self.layout,
                    self.control,
                    modes,
                    self.draft._synthetic_rates(rows_per - 1),
                )
                states, emitted, self.control = self.draft.step_batched(
                    states,
                    aux,
                    sharded,
                    ids,
                    self.control,
                    decisions,
                    self.layout,
                    rows_per,
                )
            materialize(emitted)
            return states

        def transitions(sampled: bool) -> None:
            """Draft states crossing between the single and batched paths
            (each op compiles per operand history): single -> batched ->
            single."""
            assert self.draft is not None
            for requests, rows_per in self.spec_shapes:
                speculate(sampled)  # leaves its state in self.draft_state
                states = [self.draft_state] + [
                    self.draft.new_state(self._zero) for _ in range(requests - 1)
                ]
                states = speculate_batched(sampled, requests, rows_per, states=states)
                speculate(sampled, state=states[0])

        speculating = self.draft is not None and (self.spec_alone or self.spec_batches)
        for sampled in (False, True):
            plain(sampled)
            if speculating:
                if self.spec_alone:
                    speculate(sampled)
                    if ring is not None:
                        speculate(sampled, ring())
                for requests, rows_per in self.spec_shapes:
                    speculate_batched(sampled, requests, rows_per)
                    if ring is not None:
                        speculate_batched(sampled, requests, rows_per, ring())
                if self.spec_alone and self.spec_batches:
                    transitions(sampled)
                elif (1, 4) in self.spec_shapes:
                    # One-request, four-row (1x4) states share the other
                    # shapes' canonical buffers: warm the transitions between
                    # 1x4 and every larger batched shape too.
                    for requests, rows_per in self.spec_shapes:
                        if requests == 1:
                            continue
                        states = speculate_batched(sampled, 1, 4)
                        states.extend(
                            self.draft.new_state(self._zero)
                            for _ in range(requests - 1)
                        )
                        states = speculate_batched(
                            sampled, requests, rows_per, states=states
                        )
                        speculate_batched(sampled, 1, 4, states=states[:1])
                plain(sampled)
            # Structured outputs: the grammar-masked variants (all-ones masks
            # here) run the same op history serving does; with DSpark a masked
            # step also follows and precedes speculative ones.
            plain(sampled, masked=True)
            if speculating:
                if self.spec_alone:
                    speculate(sampled)
                    plain(sampled, masked=True)
                    speculate(sampled)
                for requests, rows_per in self.spec_shapes:
                    speculate_batched(sampled, requests, rows_per)
                    plain(sampled, masked=True)
                    speculate_batched(sampled, requests, rows_per)
        self.spec_req, self.draft_state, self.next_ids, self.device_plan = (
            None,
            None,
            None,
            None,
        )
        self._drafts.clear()
        self._fresh.clear()
        if self.rank == 0:
            logger.info(
                "K3 decode: warmed the decode steps (batch sizes %s%s; "
                "%d-page tables) in %.1f s",
                list(self.ops.batch_sizes),
                ", DSpark" if self.draft is not None else "",
                pages,
                time.perf_counter() - started,
            )
