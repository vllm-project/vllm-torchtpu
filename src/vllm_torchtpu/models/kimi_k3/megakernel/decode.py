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
# src/vllm_torchtpu/megakernel/kimi_k3/decode.py
"""Kimi-K3 decode: the decode role's program over vLLM's unified pools.

:class:`KimiDecoder` runs on one TP32 kernel mesh (all devices in ascending
id; a worker's kernel rank is its mesh position, not its TP rank) over
:class:`DecodeOps` and, when speculating, a :class:`DsparkDraft`. MLA is
decode context parallel, reading and committing latent pages in place where
the KV connector landed them; KDA state lives in private slots
(``pools.SlotMap``), imported with the prefill's DSpark ring on a request's
first step. Every op reads and returns a replicated control vector
(:class:`Control`) holding each continuing request's next input, so nothing
is uploaded per step while the host's tokens lag (async scheduling). A step
is one of:

* one request, seven drafts: :meth:`DecodeOps.verify_packed`, then
  :meth:`DsparkDraft.step_single`;
* three drafts alone, or several requests: :meth:`DecodeOps.verify_batched`,
  then :meth:`DsparkDraft.step_batched` (each request keeps its own ring);
* otherwise a plain step (:meth:`DecodeOps.plain_step`), also whenever a
  structured-output request decodes (grammar masks: the verify ops do not
  mask, so the whole batch steps plain, without drafts).

The verify ops decide acceptance and commit only the selected KDA
post-state. Under attention DP a request's pages and KDA state live on the
DP group its MLA block ids name (``megakernel.dp_blocks``), and it takes one
of that group's batch rows (``_group_rows``); a speculating request takes
one of its group's half of the batched verify's requests (a single request
too).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol, cast

import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal.sync import synchronize

from vllm_torchtpu.logger import init_logger

from .common.constants import BITS_PER_WORD
from .decode_ops import (
    GREEDY,
    Control,
    DecodeOps,
    dp_row_owners,
)
from .pools import (
    KDA_LAYERS,
    KdaImporter,
    SlotMap,
    draft_raw_pools,
    kda_blocks_at_length,
    tp_rank_permutation,
)
from .program import (
    DP_SPEC_BATCHES,
    SPEC_BATCHES,
    VOCAB,
    kernel_modules,
    spec_tokens_per_batch,
    supports_spec_tokens,
)
from .runtime_shim.bridge import (
    OpRegistrationSpec,
    hbm,
    materialize,
    register_global_op,
)
from .runtime_shim.compilation import CompileCall, compile_ops
from .runtime_shim.export_cache import prepare_distributed
from .runtime_shim.program import (
    KVGroup,
    RequestInput,
    StepInput,
    StepOutput,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    import jax

    from .runtime_shim.config import MegakernelEngineConfig

logger = init_logger(__name__)

_SHORT_DRAFTS = 3
_MIN_MULTI_BATCH = 2
_RING_REPORTS_INIT = 4
_MS_PER_SEC = 1e3
_WARM_SINGLE_STEPS = 4
_WARM_REPLAN_STEP = 2
_WARM_BATCHED_STEPS = 3
_SAMPLE_WARM_PARAMS = (0.7, 0.9, 20, 1)

PlanEntry = tuple[int, bool, int, int, tuple[float, float, int, int]]


class _DraftStateLike(Protocol):
    """Protocol for DSpark single or MLA draft state objects."""

    vector: torch.Tensor


class _DraftConfigLike(Protocol):
    """Protocol for DSpark draft config instances."""

    window: int


class _DraftProgramLike(Protocol):
    """Protocol for DSpark draft program instances attached to ``KimiDecoder``."""

    weights: Mapping[str, torch.Tensor] | Sequence[torch.Tensor]
    config: _DraftConfigLike
    aux_layers: tuple[int, ...]
    uses_unified_pools: bool

    def load_weights(self) -> None:
        """Load draft weights onto TPU."""
        ...

    def attach_target(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Attach target decoder weights when needed by MLA draft."""
        ...

    def bind_pools(
        self,
        target_pool: torch.Tensor,
        owner_table: tuple[int, ...] | None,
        draft_pools: list[torch.Tensor],
    ) -> None:
        """Bind unified pools for MLA draft attention."""
        ...

    def new_state(self, zero: torch.Tensor) -> _DraftStateLike:
        """Allocate a fresh cold draft state."""
        ...

    def from_ring(
        self,
        cache: torch.Tensor,
        positions: torch.Tensor,
        zero: torch.Tensor,
    ) -> _DraftStateLike:
        """Build draft state from an imported prefill ring."""
        ...

    def synthetic_rates_for(self, drafts: int) -> tuple[float, ...] | None:
        """Return synthetic acceptance rates for ``drafts`` width, if configured."""
        ...

    def step_single(
        self,
        state: _DraftStateLike,
        aux: torch.Tensor,
        sharded: torch.Tensor,
        ids: torch.Tensor,
        control_state: torch.Tensor,
        decisions: torch.Tensor,
        control: Control,
    ) -> tuple[_DraftStateLike, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance single-request draft state across verified tokens and emit drafts."""
        ...

    def step_batched(
        self,
        states: Sequence[_DraftStateLike],
        aux: torch.Tensor,
        sharded: torch.Tensor,
        ids: torch.Tensor,
        control_state: torch.Tensor,
        decisions: torch.Tensor,
        control: Control,
        rows_per: int,
    ) -> tuple[list[_DraftStateLike], torch.Tensor, torch.Tensor]:
        """Advance batched draft states across verified rows and emit draft tokens."""
        ...

    def compilation_calls(
        self,
        layout: Control,
        *,
        spec_alone: bool,
        spec_shapes: tuple[tuple[int, int], ...],
    ) -> list[CompileCall]:
        """Return precompilation calls for draft ops."""
        ...


def _compute_spec_shapes(
    spec_tokens: tuple[int, ...],
    groups: int,
    max_num_seqs: int,
    batch_sizes: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    """Compute sorted ``(compiled_batch, verify_rows)`` speculative shapes."""
    shapes: set[tuple[int, int]] = set()
    max_batch = max(batch_sizes)
    for b, k in enumerate(spec_tokens):
        if k and groups > 1:
            fuller = (-(-b // 2), min(b, max_num_seqs // groups))
            for compiled in DP_SPEC_BATCHES:
                if (
                    2 * fuller[0] <= compiled
                    and compiled // 4 < fuller[1]
                    and compiled <= max_batch
                    and supports_spec_tokens(compiled, k, groups)
                ):
                    shapes.add((compiled, k + 1))
        elif k and (b >= _MIN_MULTI_BATCH or k == _SHORT_DRAFTS):
            compiled = next((s for s in SPEC_BATCHES if s >= b), 0)
            if not compiled or compiled > max_batch:
                msg = (
                    f"K3 decode: no compiled speculation batch holds {b} "
                    f"requests (batches {SPEC_BATCHES}, up to {max_batch})"
                )
                raise ValueError(msg)
            shapes.add((compiled, k + 1))
    return tuple(sorted(shapes))


class KimiDecoder:
    """The decode role's ``MegakernelProgram`` over ``ops`` and optional ``draft``.

    The decode role's :class:`MegakernelProgram` over ``ops`` and, when
    ``config`` speculates, the DSpark ``draft`` (None: plain steps only). Slot 0
    is the null slot padding rows use; slots below ``sub_base`` are the regular
    ones (live requests plus those whose release is in flight). ``spec_shapes``
    holds the compiled (padded request count, verify rows per request) pairs.
    """

    WIDTH = DecodeOps.VERIFY_ROWS  # tokens per row of a step's result

    def __init__(
        self,
        config: MegakernelEngineConfig,
        ops: DecodeOps,
        draft: object = None,
        *,
        tp_rank: int,
    ) -> None:
        """Initialize decoder state, slot allocator, and speculative schedule."""
        self.config = config
        self.ops = ops
        self.draft = cast("_DraftProgramLike | None", draft)
        self.tp_rank, self.rank = tp_rank, ops.rank
        self.sub_base = 2 * config.max_num_seqs + 1
        needed = (
            Control.total_slots(self.sub_base)
            if self.draft is not None
            else self.sub_base
        )
        # Under attention DP each group's share of the banks follows the ops' slot count
        # (pools.group_slots), so it must be this decoder's.
        if ops.num_slots < needed or (ops.attention_dp > 1 and ops.num_slots != needed):
            msg = (
                f"K3 decode: the ops hold {ops.num_slots} KDA slots; "
                f"--max-num-seqs {config.max_num_seqs} needs {needed}"
            )
            raise ValueError(msg)
        self.slot_map = SlotMap(self.sub_base, int(ops.attention_dp))
        self.layout: Control | None = None
        self.control: torch.Tensor | None = None
        self.device_plan: np.ndarray | None = None
        self.tracked: set[str] = set()  # requests whose next input the control holds
        self.spec_req: str | None = None
        self.draft_state: _DraftStateLike | None = None
        self.next_ids: torch.Tensor | None = None
        # Rings the prefill computed, per request, for its first speculation.
        self._offered: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        # With a draft, the schedule (vLLM's per-batch-size draft count, validated in
        # kimi_k3.program) says which batch sizes speculate; every compiled
        # verifier/drafter shape is derived from it.
        self.spec_tokens: tuple[int, ...] = (
            (spec_tokens_per_batch(config) or ()) if self.draft is not None else ()
        )
        self.groups = int(ops.attention_dp)
        self.spec_shapes = _compute_spec_shapes(
            self.spec_tokens,
            self.groups,
            config.max_num_seqs,
            ops.batch_sizes,
        )
        self.spec_batches = tuple(sorted({b for b, _ in self.spec_shapes}))
        # One sequence's 1x8 verify cannot span the DP groups.
        self.spec_alone = (
            self.groups == 1
            and len(self.spec_tokens) > 1
            and self.spec_tokens[1] == self.WIDTH - 1
        )
        # Batched speculation: each request's draft state between its steps, and the
        # verify rows its draft vector's drafts were made for (only while they still
        # follow its next token).
        self._drafts: dict[str, _DraftStateLike] = {}
        self._fresh: dict[str, int] = {}
        self._mode_rows: dict[tuple[int, ...], torch.Tensor] = {}
        self._pads: list[_DraftStateLike] = []  # draft states of padding groups
        self._compiled_layout: tuple[int, int, int] | None = None
        self._zero: torch.Tensor | None = None
        self._zero_kda_blocks: torch.Tensor | None = None
        self._modes: dict[int, torch.Tensor] = {}
        self._live_batches: set[int] = set()
        self.stats = {"plans": 0, "enters": 0, "chained": 0, "plain": 0, "batched": 0}
        self._group: dict[str, tuple[int, int]] = {}  # request -> (DP group, context)
        # Import each transferred request's DSpark ring: with a DSpark speculative
        # config the transfer carries one (backend.draft_ring_rows). (The MLA draft
        # keeps its context in its own pools: no ring.)
        self._ring = self.draft is not None and not self.draft.uses_unified_pools
        self._ring_reports = _RING_REPORTS_INIT
        # One importer per ops (it registers torch ops, whose names are process-global),
        # shared by the decoders over them.
        self._importer: KdaImporter | None = None
        self.destination_ranks: tuple[int, ...] | None = None
        self._saw_request = False
        self._pages: int | None = None
        self._num_blocks = 0
        self._mla_group = 0
        self._kda_groups: list[tuple[int, int]] = []

    def load_weights(self) -> None:
        """Load target decoder and optional draft weights onto TPU."""
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
        self,
        caches: Mapping[str, torch.Tensor],
        groups: Sequence[KVGroup],
    ) -> None:
        """Bind unified KV/KDA pools and configure MLA/KDA block table groups.

        Decode from vLLM's unified pools; ``groups`` say which block-table
        row the decoder reads (the MLA layers') and where each KDA layer's
        state blocks are (for a request's import).
        """
        self.load_weights()
        # A device's KV transfer ordinal is its mesh position: the connector lists
        # workers in mesh order (``destination_ranks``: TP rank per position).
        perm = tp_rank_permutation(self.tp_rank, self.rank)
        self.destination_ranks = (
            None if perm is None else tuple(q for _, q in sorted(perm))
        )
        importer_obj = self.ops.cached_op(
            ("kda_importer", None),
            lambda: KdaImporter(
                self.ops.mesh,
                None,
                self.ops.attention_dp,
                self.ops.num_slots,
            ),
        )
        importer = cast("KdaImporter", importer_obj)
        self._importer = importer
        num_blocks = importer.bind(caches, self.config.kv_cache_dtype)
        self._num_blocks = num_blocks
        self.ops.use_unified_pools(importer.raw_pools, importer.mla_raw)
        if self.draft is not None and self.draft.uses_unified_pools:
            if self.ops.pool_map is None:
                msg = (
                    "K3 decode: unified pool map must be initialized before draft bind"
                )
                raise RuntimeError(msg)
            # The MLA draft: the target's LM head and embedding, and pools shaped like
            # the target's MLA pools (same tables and owners).
            self.draft.attach_target(self.ops.weights)
            self.draft.bind_pools(
                self.ops.unified_pools[self.ops.pool_map[0]],
                self.ops.owner_table,
                draft_raw_pools(caches),
            )
        of = {name: g for g, grp in enumerate(groups) for name in grp.layer_names}
        mla_groups = {of[n] for n in importer.mla_layer_names if n in of}
        if len(mla_groups) != 1:
            msg = (
                f"K3 decode: the MLA layers are in KV cache groups {sorted(mla_groups)}"
            )
            raise RuntimeError(msg)
        self._mla_group = mla_groups.pop()
        missing = [n for n in importer.kda_layer_names if n not in of]
        if missing:
            msg = f"K3 decode: KDA layers {missing[:3]}... are in no KV cache group"
            raise RuntimeError(msg)
        self._kda_groups = [
            (of[n], int(groups[of[n]].block_size)) for n in importer.kda_layer_names
        ]
        self._pages = int(groups[self._mla_group].table_width)
        if self.rank == 0:
            logger.info(
                "K3 decode: DCP decoder on vLLM's %d unified pools (%d blocks); HBM %s",
                len(importer.raw_pools),
                num_blocks,
                hbm(self.rank),
            )

    def warmup(self) -> None:
        """Warm up KDA importer and decode step executables."""
        if self._pages is None:
            msg = "K3 decode: bind the KV caches before warming up"
            raise RuntimeError(msg)
        started = time.perf_counter()
        self._warm_importer()  # an import after the steps (another operand history)
        self._warm_steps()
        self._warm_importer()
        self._warm_steps()
        if self.rank == 0:
            logger.info(
                "K3 decode: ops warm in %.1f s",
                time.perf_counter() - started,
            )

    def step(self, step_input: StepInput) -> StepOutput:
        """Execute one decode step for ``step_input`` and return emitted tokens.

        One decode step (see the module docstring).
        """
        if self._pages is None:
            msg = "K3 decode: a request arrived before the KV pools were bound"
            raise RuntimeError(msg)
        if step_input.copies:
            msg = "K3 decode: copy-on-write block copies (prefix hits inside a block)"
            raise NotImplementedError(msg)
        mask = step_input.mask
        words = VOCAB // BITS_PER_WORD
        if mask is not None and mask.shape != (len(step_input.requests), words):
            msg = (
                f"K3 decode: grammar masks {mask.shape} do not cover "
                f"{len(step_input.requests)} requests of the {VOCAB}-token "
                "vocabulary"
            )
            raise ValueError(msg)
        self.forget(step_input.finished)
        return StepOutput(self._decode(step_input.requests, mask))

    def _ensure_zero_kda_blocks(self) -> torch.Tensor:
        """Return a settled all-zero TPU KDA block-id vector."""
        if self._zero_kda_blocks is None:
            self._zero_kda_blocks = torch.zeros(
                KDA_LAYERS,
                dtype=torch.int32,
            ).to("tpu")
            synchronize([self._zero_kda_blocks], wait=False)
        return self._zero_kda_blocks

    def _admit(self, request: RequestInput, slot: int) -> None:
        """Import ``request``'s KDA state and optional DSpark ring into ``slot``.

        A request's first step: import its KDA state after the tokens
        computed before this step into ``slot``, and its DSpark ring. A
        request with no state block to import (none computed, or only the
        null block) starts from a clean state.
        """
        # Each KDA layer's state after the computed tokens: the block holding position
        # computed - 1, where the transfer committed it. (When the computed tokens end
        # on a block boundary the request already holds the next block too; vLLM's
        # runners would copy the state into it.)
        kda = kda_blocks_at_length(
            request.block_tables,
            self._kda_groups,
            request.computed,
        )
        if self.groups > 1 and int(kda.max()) >= self._num_blocks:
            msg = f"K3 decode: KDA state blocks {kda.tolist()} are not shared blocks"
            raise RuntimeError(msg)
        # The decoder has no position-0 reset. Under attention DP the importer clears
        # the slot (from the null blocks) on the request's group only.
        if not kda.any() and self.groups == 1:
            if (
                self._importer is not None
                and self.ops.slots
                and self.ops.slots[0].device.type == "tpu"
            ):
                conv, recurrent = self._importer.import_state(
                    self.ops.slots[0],
                    self.ops.slots[1],
                    self._ensure_zero_kda_blocks(),
                    slot,
                )
                self.ops.slots = [conv, recurrent]
                return
            for state in self.ops.slots:
                state[0, slot].zero_()
            if self.ops.slots and self.ops.slots[0].device.type == "tpu":
                synchronize(list(self.ops.slots), wait=False)
            return
        if self._importer is None:
            msg = "K3 decode: KDA importer is not bound"
            raise RuntimeError(msg)
        started = time.perf_counter()
        kda_blocks = torch.from_numpy(kda).to("tpu")
        synchronize([kda_blocks], wait=False)
        conv, recurrent = self._importer.import_state(
            self.ops.slots[0],
            self.ops.slots[1],
            kda_blocks,
            slot,
        )
        self.ops.slots = [conv, recurrent]
        if self._ring and self.draft is not None:
            # The DSpark ring the prefill wrote beside the state (cold unless it wrote
            # one for exactly these tokens): the request's first speculation starts from
            # it instead of a cold ring.
            cache, positions = self._importer.import_ring(
                self.draft.config,
                kda_blocks,
                request.computed,
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
                _MS_PER_SEC * (time.perf_counter() - started),
            )
            if not self._saw_request:
                self._saw_request = True
                logger.info("K3 decode: first real request; HBM %s", hbm(self.rank))

    def _null_ring(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Import a draft ring from the null block for warmup.

        An imported ring from the null block (for warming up).
        """
        if self.draft is None or self._importer is None:
            msg = "K3 decode: draft ring warmup requires bound draft and importer"
            raise RuntimeError(msg)
        return self._importer.import_ring(
            self.draft.config,
            self._ensure_zero_kda_blocks(),
            0,
        )

    def _warm_importer(self) -> None:
        """Compile and run the KDA import op on the null slot and block."""
        if self._importer is None:
            msg = "K3 decode: importer warmup requires bound importer"
            raise RuntimeError(msg)
        started = time.perf_counter()
        conv, recurrent = self._importer.import_state(
            self.ops.slots[0],
            self.ops.slots[1],
            self._ensure_zero_kda_blocks(),
            0,
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
        """Ensure control layout, device control tensor, and mode constants exist."""
        if self.layout is None or self.layout.pages != pages:
            self.layout = Control(self.sub_base, max(self.ops.batch_sizes), pages)
            self.control = None
        if self.control is None:
            self.control = torch.zeros(self.layout.size, dtype=torch.int32).to("tpu")
            synchronize([self.control], wait=False)
            self.device_plan = None
        if self._zero is None:
            self._zero = torch.zeros(1, dtype=torch.int32).to("tpu")
            self._modes = {
                0: torch.zeros(1, dtype=torch.int32).to("tpu"),
                1: torch.ones(1, dtype=torch.int32).to("tpu"),
            }
            synchronize(
                [self._zero, self._modes[0], self._modes[1]],
                wait=False,
            )
        return self.layout

    def _splice(self) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Return op replacing the plan slice of the replicated control vector.

        ``(control, plan) -> control`` with the plan part replaced.
        """
        layout = self.layout
        if layout is None:
            msg = "K3 decode: control layout is not initialized"
            raise RuntimeError(msg)
        key = ("splice", layout.key())

        def build() -> Callable[..., tuple[torch.Tensor, ...]]:
            """Register control-plan splice custom op."""

            def fn(control: jax.Array, plan: jax.Array) -> tuple[jax.Array]:
                """Concatenate control header with updated ``plan``."""
                return (jnp.concatenate((control[: layout.plan_start], plan)),)

            return register_global_op(
                "pallas::k3_decode_control_splice",
                self.ops.mesh,
                fn,
                OpRegistrationSpec(
                    in_specs=(P(), P()), out_specs=(P(),), export_key=repr(key)
                ),
            )

        return cast(
            "Callable[..., tuple[torch.Tensor, ...]]",
            self.ops.cached_op(key, build),
        )

    def _enter(self) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Return op building the single-request verify operand from control state.

        ``(control, vector, keep) -> ids``: the verify operand of plan row
        0's request from the control: its next token and after it placeholder
        drafts (``keep`` 0) or the last draft vector's drafts (``keep`` 1:
        the same request, a new plan), at its next positions; every row its
        slot; its state row; the plan's table.
        """
        layout, sub_base = self.layout, self.sub_base
        if layout is None:
            msg = "K3 decode: control layout is not initialized"
            raise RuntimeError(msg)
        key = ("enter", layout.key(), sub_base)

        def build() -> Callable[..., tuple[torch.Tensor, ...]]:
            """Register single-request DSpark enter op."""
            rows = self.WIDTH
            (dspark,) = kernel_modules("dspark")

            def fn(
                control: jax.Array,
                vector: jax.Array,
                keep: jax.Array,
            ) -> tuple[jax.Array]:
                """Assemble single-request verify packet."""
                slot, tok, position, row, _, tables = layout.inputs(control, 1)
                drafts = vector[0, dspark.OUT_DRAFTS : dspark.OUT_DRAFTS + rows - 1]
                tokens = jnp.where(
                    keep[0] != 0,
                    jnp.concatenate((tok, drafts)),
                    jnp.full((rows,), tok[0]),
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
                OpRegistrationSpec(
                    in_specs=(P(), P(), P()), out_specs=(P(),), export_key=repr(key)
                ),
            )

        return cast(
            "Callable[..., tuple[torch.Tensor, ...]]",
            self.ops.cached_op(key, build),
        )

    def _enter_batched(
        self,
        requests: int,
        rows_per: int,
    ) -> Callable[..., tuple[torch.Tensor, ...]]:
        """Return op building the batched verify operand for ``requests x rows_per``.

        ``(control, vector_0.., modes) -> ids``: the batched verify operand
        (``DecodeOps.verify_batched``) of plan rows 0..requests-1: per
        request its next token, then the first drafts of its draft vector
        (``modes`` 1) or placeholders (0), at its next positions; every row
        its request's slot (``indices``: each row's logical entry); one table
        per request.
        """
        layout, k = self.layout, rows_per
        if layout is None:
            msg = "K3 decode: control layout is not initialized"
            raise RuntimeError(msg)
        key = ("enter_batched", layout.key(), requests, k)

        def build() -> Callable[..., tuple[torch.Tensor, ...]]:
            """Register batched DSpark enter op."""
            (dspark,) = kernel_modules("dspark")

            def fn(
                control: jax.Array,
                *vectors_and_modes: jax.Array,
            ) -> tuple[jax.Array]:
                """Assemble batched verify packet across ``requests`` sequences."""
                vectors = vectors_and_modes[:requests]
                modes = vectors_and_modes[requests]
                slot, tok, position, row, _, tables = layout.inputs(control, requests)
                others = jnp.arange(1, layout.BANK + 1, dtype=jnp.int32)
                tokens, positions, slots, indices = [], [], [], []
                for j in range(requests):
                    drafts = vectors[j][
                        0,
                        dspark.OUT_DRAFTS : dspark.OUT_DRAFTS + k - 1,
                    ]
                    tokens.append(
                        jnp.where(
                            modes[j] != 0,
                            jnp.concatenate((tok[j : j + 1], drafts)),
                            jnp.full((k,), tok[j]),
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
                OpRegistrationSpec(
                    in_specs=(P(),) * (requests + 2),
                    out_specs=(P(),),
                    export_key=repr(key),
                ),
            )

        return cast(
            "Callable[..., tuple[torch.Tensor, ...]]",
            self.ops.cached_op(key, build),
        )

    def _spec_batch(self, requests: int, rows_per: int, groups: list[int]) -> int:
        """Return compiled batched-speculation size holding ``requests``, or 0.

        The compiled batched-speculation size holding ``requests``
        decoding requests of ``rows_per`` verify rows (padded with null ones;
        under attention DP each of ``groups``' requests in its group's half),
        or 0: none.
        """
        if self.groups > 1:
            # Each group's requests fill half of a compiled batch: from b split evenly
            # to all of them in one group (up to its cap).
            fuller = max(groups.count(g) for g in range(self.groups))
            return next(
                (
                    b
                    for b, rows in self.spec_shapes
                    if rows == rows_per and b // self.groups >= fuller
                ),
                0,
            )
        if requests < 1 or (requests == 1 and 1 not in self.spec_batches):
            return 0
        return next((b for b in self.spec_batches if b >= requests), 0)

    def _modes_for(self, modes: tuple[int, ...]) -> torch.Tensor:
        """Return cached int32 TPU tensor for ``modes`` tuple."""
        if modes not in self._mode_rows:
            mode_row = torch.tensor(modes, dtype=torch.int32).to("tpu")
            synchronize([mode_row], wait=False)
            self._mode_rows[modes] = mode_row
        return self._mode_rows[modes]

    def _take_state(self, req_id: str) -> _DraftStateLike:
        """Return existing, ring-offered, or fresh cold draft state for ``req_id``.

        ``req_id``'s draft state: kept from its last speculation, else the
        ring the prefill computed, else a cold one.
        """
        if self.draft is None or self._zero is None:
            msg = "K3 decode: draft state requested without initialized draft"
            raise RuntimeError(msg)
        state = self._drafts.pop(req_id, None)
        if state is not None:
            return state
        offered = self._offered.pop(req_id, None)
        if offered is not None:
            return self.draft.from_ring(offered[0], offered[1], self._zero)
        return self.draft.new_state(self._zero)

    def _park_single(self) -> None:
        """Park active single-request speculation state back into ``self._drafts``.

        The single-request speculation's state back with its request.
        """
        if self.spec_req is not None and self.draft_state is not None:
            self._drafts[self.spec_req] = self.draft_state
        self.spec_req, self.draft_state, self.next_ids = None, None, None

    def _sync_plan(
        self,
        entries: Sequence[PlanEntry],
        tables: Sequence[np.ndarray],
    ) -> bool:
        """Upload plan for ``entries`` and ``tables`` when changed.

        Upload the plan for ``entries`` unless the device holds it; returns
        whether it uploaded (the verify operand's table may have changed).
        """
        layout = self.layout
        if layout is None or self.control is None:
            msg = "K3 decode: control state is not prepared"
            raise RuntimeError(msg)
        plan = layout.plan(entries, tables)
        if self.device_plan is not None and np.array_equal(plan, self.device_plan):
            return False
        (self.control,) = self._splice()(
            self.control,
            torch.from_numpy(plan).to("tpu"),
        )
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
        """Replay recomputed ``tokens`` one batch-1 plain step at a time.

        Tokens computed on this role (a recompute after a preemption):
        one batch-one plain step per token (on a row of its DP ``group``),
        their outputs discarded.
        """
        if self.layout is None or self.control is None:
            msg = "K3 decode: control state is not prepared"
            raise RuntimeError(msg)
        batch = self.ops.batch_for(1)
        for offset, tok in enumerate(tokens):
            entries: list[PlanEntry] = [(slot, True, int(tok), start + offset, GREEDY)]
            if self.groups > 1:
                _, entries, tables, _ = self._place(
                    [group],
                    entries,
                    [table],
                    len(table),
                )
                self._sync_plan(entries, tables)  # other plan rows: the null slot
            else:
                self._sync_plan(entries, [table])
            _, self.control = self.ops.plain_step(
                self.control,
                (batch, self.layout, self.sub_base),
                sampled=False,
            )

    def forget(self, finished: Iterable[str]) -> None:
        """Release KDA slots and draft states for ``finished`` request IDs."""
        done = set(finished or ())
        if not done:
            return
        self.slot_map.release(done)
        self.tracked -= done
        for req in done:
            self._group.pop(req, None)
            self._offered.pop(req, None)
            self._drafts.pop(req, None)
            self._fresh.pop(req, None)
        if self.spec_req in done:
            self.spec_req, self.draft_state, self.next_ids = None, None, None

    def _mla_table(self, request: RequestInput) -> np.ndarray:
        """Return ``request``'s MLA block table normalized to local pool rows.

        ``request``'s MLA block table as pool rows. Under attention DP
        the scheduler's ids are ``g N + r``: DP group ``g``'s row ``r``
        (``megakernel.dp_blocks``); the request keeps that group.
        """
        table = request.block_tables[self._mla_group]
        if self.groups > 1:
            group = int(table.max()) // self._num_blocks
            self._group[request.req_id] = (group, request.end)
            table = table % self._num_blocks
        return table

    def _group_of(self, request: RequestInput) -> int:
        """Return or assign attention-DP group index for ``request``.

        ``request``'s DP group: the scheduler's, else chosen on its
        first step: the group with the fewest requests, then the least
        context.
        """
        if request.req_id not in self._group:
            count = [0] * self.groups
            context = [0] * self.groups
            for grp, length in self._group.values():
                count[grp] += 1
                context[grp] += length
            chosen = min(range(self.groups), key=lambda g: (count[g], context[g]))
            self._group[request.req_id] = (chosen, request.end)
        return self._group[request.req_id][0]

    def _group_rows(self, batch: int, group: int) -> list[int]:
        """Return batch row indices attended by attention-DP ``group``.

        The batch rows DP group ``group`` attends: ``p * B / O + group * r + j``
        (``O`` local row owners, ``dp_row_owners``; ``r = B / (O dp)``).
        """
        owners = dp_row_owners(batch)
        per = batch // self.groups // owners
        return [
            p * (batch // owners) + group * per + j
            for p in range(owners)
            for j in range(per)
        ]

    def _place(
        self,
        groups: list[int],
        entries: Sequence[PlanEntry],
        tables: Sequence[np.ndarray],
        pages: int,
    ) -> tuple[int, list[PlanEntry], list[np.ndarray], list[int]]:
        """Place ``entries`` and ``tables`` onto their DP groups' batch rows.

        Plan rows for requests of DP ``groups`` (in order): ``(batch,
        entries, tables, plan row of each request)``, the other rows null.
        """
        load = [groups.count(g) for g in range(self.groups)]
        batch = self.ops.batch_for(self.groups * max(load))
        free = [self._group_rows(batch, g) for g in range(self.groups)]
        null_entry: PlanEntry = (0, True, 0, 0, GREEDY)
        null_table = np.zeros(pages, np.int32)
        placed_entries = [null_entry] * batch
        placed_tables = [null_table] * batch
        rows: list[int] = []
        for group, entry, table in zip(groups, entries, tables, strict=True):
            row = free[group].pop(0)
            placed_entries[row], placed_tables[row] = entry, table
            rows.append(row)
        return batch, placed_entries, placed_tables, rows

    def _collect_decoding_requests(
        self,
        requests: Sequence[RequestInput],
    ) -> tuple[list[PlanEntry], list[np.ndarray], list[tuple[int, RequestInput]]]:
        """Admit/replay scheduled requests and return plan entries, tables, and rows."""
        entries: list[PlanEntry] = []
        tables: list[np.ndarray] = []
        decoding: list[tuple[int, RequestInput]] = []
        for i, request in enumerate(requests):
            if request.scheduled <= 0:
                continue
            table = self._mla_table(request)
            group = self._group_of(request) if self.groups > 1 else 0
            if self.slot_map.group(request.req_id) not in (None, group):
                # Only its old group held its state: placed in another group (after a
                # preemption), it starts there as a new request.
                self.slot_map.release([request.req_id])
            fresh = request.req_id not in self.slot_map
            slot = self.slot_map.allocate(request.req_id, group)
            if fresh:
                self.tracked.discard(request.req_id)
                self._admit(request, slot)
            sampling_tuple = request.sampling.as_tuple()
            if request.req_id in self.tracked:
                entries.append((slot, False, 0, 0, sampling_tuple))
            else:
                # Scheduled slots past the known tokens are draft slots (vLLM pads a
                # request starting to decode to the speculative width).
                known = min(request.end, request.num_tokens)
                tokens = request.token_ids[request.computed : known]
                replay = len(tokens) - (0 if request.partial else 1)
                if replay:
                    self._replay(
                        slot,
                        tokens[:replay],
                        request.computed,
                        table,
                        group,
                    )
                if request.partial:
                    continue
                entries.append((slot, True, int(tokens[-1]), known - 1, sampling_tuple))
                self.tracked.add(request.req_id)
            tables.append(table)
            decoding.append((i, request))
        return entries, tables, decoding

    def _run_plain_decode(
        self,
        layout: Control,
        decoding: list[tuple[int, RequestInput]],
        entries: list[PlanEntry],
        tables: list[np.ndarray],
        *,
        mask: np.ndarray | None,
    ) -> tuple[torch.Tensor, list[int] | None]:
        """Run one plain decode step and return ``(emitted, placed)``."""
        if self._pages is None or self.control is None:
            msg = "K3 decode: plain step requires bound pages and control tensor"
            raise RuntimeError(msg)
        # A request leaving speculation keeps its KDA state in its slot (nothing moves)
        # and its draft state, whose drafts no longer follow its next token.
        self._park_single()
        for _, request in decoding:
            self._fresh.pop(request.req_id, None)
        # attention DP: the plan row of each decoding request
        placed: list[int] | None = None
        groups = [self._group_of(r) for _, r in decoding] if self.groups > 1 else []
        if self.groups > 1:
            batch, entries, tables, placed = self._place(
                groups,
                entries,
                tables,
                self._pages,
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
        sampled = any(entry[4][0] > 0 for entry in entries)
        self._sync_plan(entries, tables)
        self.stats["plain"] += 1
        # One row per plan row (the decoding requests' rows, in their placement order);
        # padding rows unconstrained.
        mask_operand: torch.Tensor | None = None
        if mask is not None:
            mask_arr = np.full((batch, VOCAB // BITS_PER_WORD), -1, np.int32)
            rows = placed if placed is not None else range(len(decoding))
            for plan_row, (i, _) in zip(rows, decoding, strict=True):
                mask_arr[plan_row] = mask[i]
            mask_operand = torch.from_numpy(mask_arr).to("tpu")
        emitted, self.control = self.ops.plain_step(
            self.control,
            (batch, layout, self.sub_base),
            sampled=sampled,
            mask=mask_operand,
        )
        return emitted, placed

    def _decode(
        self,
        requests: Sequence[RequestInput],
        mask: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Execute one decode step across ``requests`` in runner batch order.

        One decode step for ``requests`` (the runner's batch order): the
        tokens ``[>= len(requests), 8]`` int32, row i the request's accepted
        drafts then its bonus token, -1 padded (all -1 for a request that
        takes no token this step: a partial prefill chunk or an unscheduled
        one). ``mask``: the requests' grammar bitmasks (row ``i`` request
        ``i``'s; all-ones unconstrained), applied to the plain step's logits.
        """
        pages = self._pages
        if pages is None:
            msg = "K3 decode: pages not bound"
            raise RuntimeError(msg)
        layout = self._prepare(pages)
        entries, tables, decoding = self._collect_decoding_requests(requests)
        emitted: torch.Tensor | None = None
        placed: list[int] | None = None
        groups = [self._group_of(r) for _, r in decoding] if self.groups > 1 else []
        # The drafts the schedule gives this step's batch (validated at startup: the
        # schedule never asks for a shape the kernels lack; under attention DP a step
        # whose fuller group exceeds every compiled verify decodes plainly).
        k = (
            self.spec_tokens[len(decoding)]
            if len(decoding) < len(self.spec_tokens) and mask is None
            else 0
        )
        compiled = self._spec_batch(len(decoding), k + 1, groups) if k else 0
        if self.spec_alone and len(decoding) == 1 and k == self.WIDTH - 1:
            emitted = self._speculate(decoding[0][1], entries, tables)
        elif compiled and (
            len(decoding) == 1 or any(r.scheduled > 1 for _, r in decoding)
        ):
            # A lone request speculates from its anchor-only first step; ``_fresh``
            # keeps its placeholder draft rows from being accepted.
            emitted, placed = self._speculate_batched(
                decoding,
                entries,
                tables,
                compiled,
                groups,
            )
        elif decoding:
            emitted, placed = self._run_plain_decode(
                layout,
                decoding,
                entries,
                tables,
                mask=mask,
            )
        decoded = [i for i, _ in decoding]
        # Rows without a token this step: -1 rows around the decoded ones.
        if emitted is None:
            return torch.full(
                (max(len(requests), 1), self.WIDTH),
                -1,
                dtype=torch.int32,
            ).to("tpu")
        if placed is not None:
            # Plan rows back to the runner's rows.
            position_of = dict(zip(decoded, placed, strict=True))
        elif decoded == list(range(len(requests))):
            return emitted
        else:
            position_of = {i: idx for idx, i in enumerate(decoded)}
        blank = int(emitted.shape[0])
        index = torch.tensor(
            [position_of.get(r, blank) for r in range(len(requests))],
            dtype=torch.int64,
        )
        padded = torch.cat(
            (emitted, torch.full((1, self.WIDTH), -1, dtype=torch.int32).to("tpu"))
        )
        return padded.index_select(0, index.to("tpu"))

    def _speculate(
        self,
        request: RequestInput,
        entries: Sequence[PlanEntry],
        tables: Sequence[np.ndarray],
    ) -> torch.Tensor:
        """Run one single-request speculative verify + draft step."""
        changed = self._sync_plan(entries, tables)
        reset = entries[0][1]
        continuing = (
            self.spec_req == request.req_id and self.next_ids is not None and not reset
        )
        if not continuing:
            self._park_single()
            self.spec_req = request.req_id
            self.draft_state = self._take_state(request.req_id)
        if (
            self.draft is None
            or self.draft_state is None
            or self.layout is None
            or self.control is None
        ):
            msg = "K3 decode: single-request speculation state is incomplete"
            raise RuntimeError(msg)
        if continuing and not changed and self.next_ids is not None:
            ids = self.next_ids
            self.stats["chained"] += 1
        else:
            self.stats["enters"] += 1
            (ids,) = self._enter()(
                self.control,
                self.draft_state.vector,
                self._modes[1 if continuing else 0],
            )
        # Accept drafts only when the rows carry this request's real drafts and the
        # scheduler scheduled them all.
        mode = self._modes[1 if continuing and request.scheduled == self.WIDTH else 0]
        sharded, aux, decisions = self.ops.verify_packed(
            (ids, self.draft.aux_layers),
            self.layout,
            (self.control, mode),
            synthetic_rates=self.draft.synthetic_rates_for(self.WIDTH - 1),
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
        self,
        decoding: Sequence[tuple[int, RequestInput]],
        entries: Sequence[PlanEntry],
        tables: Sequence[np.ndarray],
        compiled: int,
        groups: list[int],
    ) -> tuple[torch.Tensor, list[int] | None]:
        """Run one batched DSpark verify + draft step across ``decoding`` rows.

        One batched DSpark step for the decoding rows (module docstring)
        through the ``compiled`` verify; returns the tokens and, under
        attention DP, each request's plan row (group ``g``'s requests fill
        rows ``g compiled / dp ..``). A request's drafts are verified (mode 1)
        only when they were made for at least the scheduled width after its
        current next token and the scheduler scheduled as many tokens;
        otherwise its rows are placeholders.
        """
        if (
            self.draft is None
            or self.layout is None
            or self.control is None
            or self._zero is None
        ):
            msg = "K3 decode: batched speculation state is incomplete"
            raise RuntimeError(msg)
        self._park_single()
        requests = len(decoding)
        # A structured request decodes: the whole batch steps plain and masked (the
        # verify ops do not mask), its drafts unused.
        k = self.spec_tokens[requests] + 1
        rows = list(range(requests))
        plan_entries = list(entries)
        plan_tables = list(tables)
        if self.groups > 1:
            share, filled = compiled // self.groups, [0] * self.groups
            for i, group in enumerate(groups):
                rows[i] = group * share + filled[group]
                filled[group] += 1
            placed_entries: list[PlanEntry] = [(0, True, 0, 0, GREEDY)] * compiled
            placed_tables = [np.zeros(self.layout.pages, np.int32)] * compiled
            for row, entry, table in zip(rows, plan_entries, plan_tables, strict=True):
                placed_entries[row], placed_tables[row] = entry, table
            plan_entries, plan_tables = placed_entries, placed_tables
        self._sync_plan(plan_entries, plan_tables)
        states: list[_DraftStateLike | None] = [None] * compiled
        modes = [0] * compiled
        for row, (_, req) in zip(rows, decoding, strict=True):
            states[row] = self._take_state(req.req_id)
            modes[row] = int(self._fresh.get(req.req_id, 0) >= k and req.scheduled >= k)
        padding = [row for row in range(compiled) if states[row] is None]
        for row in padding:  # padding groups (null slot): reused states
            states[row] = (
                self._pads.pop() if self._pads else self.draft.new_state(self._zero)
            )
        active_states = [cast("_DraftStateLike", s) for s in states]
        modes_t = self._modes_for(tuple(modes))
        (ids,) = self._enter_batched(compiled, k)(
            self.control,
            *(s.vector for s in active_states),
            modes_t,
        )
        sharded, aux, decisions = self.ops.verify_batched(
            (ids, self.draft.aux_layers, (compiled, k)),
            self.layout,
            (self.control, modes_t),
            synthetic_rates=self.draft.synthetic_rates_for(k - 1),
        )
        next_states, emitted, self.control = self.draft.step_batched(
            active_states,
            aux,
            sharded,
            ids,
            self.control,
            decisions,
            self.layout,
            k,
        )
        for row, (_, request) in zip(rows, decoding, strict=True):
            self._drafts[request.req_id] = next_states[row]
            self._fresh[request.req_id] = k
        self._pads.extend(next_states[row] for row in padding)
        if self.device_plan is not None:
            self.device_plan = self.layout.consumed(self.device_plan, compiled)
        self.stats["batched"] += 1
        return emitted, rows if self.groups > 1 else None

    def _precompile_decode_programs(
        self,
        layout: Control,
        null_table: np.ndarray,
    ) -> None:
        """Precompile all configured decode and draft programs in parallel."""
        if self._compiled_layout == layout.key():
            return
        started = time.perf_counter()
        # The first step compiles its own variant (its state buffers come from
        # allocation, not from an op), which the precompiled graphs do not cover: start
        # it first so that compile overlaps theirs.
        first = self._warm_plain_step(
            self.ops.batch_sizes[0],
            null_table,
            sampled=False,
            masked=False,
        )
        if self.control is None:
            msg = "K3 decode: control tensor missing after first plain step"
            raise RuntimeError(msg)
        synchronize(
            [first, self.control, *self.ops.slots, *self.ops.unified_pools],
            wait=False,
        )
        calls = self.ops.compilation_calls(
            layout,
            self.sub_base,
            self.draft.aux_layers if self.draft is not None else (),
            spec_alone=self.spec_alone,
            spec_shapes=self.spec_shapes,
            synthetic_rates=(
                self.draft.synthetic_rates_for if self.draft is not None else None
            ),
            masked_ops=True,
        )
        if self.draft is not None:
            calls.extend(
                self.draft.compilation_calls(
                    layout,
                    spec_alone=self.spec_alone,
                    spec_shapes=self.spec_shapes,
                )
            )
        calls.sort(key=lambda call: call.priority, reverse=True)
        prepare_distributed(calls)
        compile_ops(calls)
        materialize(first)
        self._compiled_layout = layout.key()
        if self.rank == 0:
            logger.info(
                "K3 decode: precompiled %d decode/draft programs in %.1f s",
                len(calls),
                time.perf_counter() - started,
            )

    def _warm_plain_step(
        self,
        batch: int,
        null_table: np.ndarray,
        *,
        sampled: bool,
        masked: bool = False,
    ) -> torch.Tensor:
        """Run one warmup plain step at ``batch``."""
        if self.layout is None or self.control is None:
            msg = "K3 decode: layout or control not prepared for warmup"
            raise RuntimeError(msg)
        params = _SAMPLE_WARM_PARAMS if sampled else GREEDY
        self._sync_plan([(0, True, 0, 0, params)] * batch, [null_table] * batch)
        mask: torch.Tensor | None = None
        if masked:
            mask = torch.full(
                (batch, VOCAB // BITS_PER_WORD),
                -1,
                dtype=torch.int32,
                device="tpu",
            )
        emitted, self.control = self.ops.plain_step(
            self.control,
            (batch, self.layout, self.sub_base),
            sampled=sampled,
            mask=mask,
        )
        return emitted

    def _warm_single_spec(
        self,
        null_table: np.ndarray,
        *,
        sampled: bool,
        offered: tuple[torch.Tensor, torch.Tensor] | None = None,
        state: _DraftStateLike | None = None,
    ) -> None:
        """Run warmup single-request speculative steps."""
        if (
            self.draft is None
            or self.layout is None
            or self.control is None
            or self._zero is None
        ):
            msg = "K3 decode: single-request warmup requires draft and control"
            raise RuntimeError(msg)
        params = _SAMPLE_WARM_PARAMS if sampled else GREEDY
        self.device_plan = None
        self._sync_plan([(0, True, 0, 0, params)], [null_table])
        if state is not None:
            self.draft_state = state
        else:
            self.draft_state = (
                self.draft.new_state(self._zero)
                if offered is None
                else self.draft.from_ring(offered[0], offered[1], self._zero)
            )
        (ids,) = self._enter()(
            self.control,
            self.draft_state.vector,
            self._modes[0],
        )
        emitted: torch.Tensor | None = None
        for step_idx in range(_WARM_SINGLE_STEPS):
            if step_idx == _WARM_REPLAN_STEP:
                self.device_plan = None
                self._sync_plan([(0, True, 0, 0, params)], [null_table])
                (ids,) = self._enter()(
                    self.control,
                    self.draft_state.vector,
                    self._modes[1],
                )
            sharded, aux, decisions = self.ops.verify_packed(
                (ids, self.draft.aux_layers),
                self.layout,
                (self.control, self._modes[min(step_idx, 1)]),
                synthetic_rates=self.draft.synthetic_rates_for(self.WIDTH - 1),
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
        if emitted is not None:
            materialize(emitted)

    def _warm_batched_spec(
        self,
        null_table: np.ndarray,
        batch_shape: tuple[int, int],
        *,
        sampled: bool,
        offered: tuple[torch.Tensor, torch.Tensor] | None = None,
        states: list[_DraftStateLike] | None = None,
    ) -> list[_DraftStateLike]:
        """Run warmup batched speculative steps for ``batch_shape``."""
        if (
            self.draft is None
            or self.layout is None
            or self.control is None
            or self._zero is None
        ):
            msg = "K3 decode: batched warmup requires draft and control"
            raise RuntimeError(msg)
        requests, rows_per = batch_shape
        params = _SAMPLE_WARM_PARAMS if sampled else GREEDY
        active = (
            [self.draft.new_state(self._zero) for _ in range(requests)]
            if states is None
            else list(states)
        )
        if offered is not None:
            active[0] = self.draft.from_ring(offered[0], offered[1], self._zero)
        emitted: torch.Tensor | None = None
        for step_idx in range(_WARM_BATCHED_STEPS):
            self.device_plan = None
            self._sync_plan(
                [(0, step_idx == 0, 0, 0, params)] * requests,
                [null_table] * requests,
            )
            modes = self._modes_for((min(step_idx, 1),) * requests)
            (ids,) = self._enter_batched(requests, rows_per)(
                self.control,
                *(s.vector for s in active),
                modes,
            )
            sharded, aux, decisions = self.ops.verify_batched(
                (ids, self.draft.aux_layers, (requests, rows_per)),
                self.layout,
                (self.control, modes),
                synthetic_rates=self.draft.synthetic_rates_for(rows_per - 1),
            )
            active, emitted, self.control = self.draft.step_batched(
                active,
                aux,
                sharded,
                ids,
                self.control,
                decisions,
                self.layout,
                rows_per,
            )
        if emitted is not None:
            materialize(emitted)
        return active

    def _warm_spec_transitions(
        self,
        null_table: np.ndarray,
        *,
        sampled: bool,
    ) -> None:
        """Warm draft state transitions across single and batched shapes."""
        if self.draft is None or self._zero is None:
            return
        if self.spec_alone and self.spec_batches:
            # One-request, four-row (1x4) states share the other shapes' canonical
            # buffers: warm the transitions between 1x4 and every larger batched shape
            # too.
            for requests, rows_per in self.spec_shapes:
                self._warm_single_spec(null_table, sampled=sampled)
                if self.draft_state is None:
                    continue
                states = [self.draft_state] + [
                    self.draft.new_state(self._zero) for _ in range(requests - 1)
                ]
                states = self._warm_batched_spec(
                    null_table,
                    (requests, rows_per),
                    sampled=sampled,
                    states=states,
                )
                self._warm_single_spec(null_table, sampled=sampled, state=states[0])
        elif (1, _SHORT_DRAFTS + 1) in self.spec_shapes:
            for requests, rows_per in self.spec_shapes:
                if requests == 1:
                    continue
                states = self._warm_batched_spec(
                    null_table,
                    (1, _SHORT_DRAFTS + 1),
                    sampled=sampled,
                )
                states.extend(
                    self.draft.new_state(self._zero) for _ in range(requests - 1)
                )
                states = self._warm_batched_spec(
                    null_table,
                    (requests, rows_per),
                    sampled=sampled,
                    states=states,
                )
                self._warm_batched_spec(
                    null_table,
                    (1, _SHORT_DRAFTS + 1),
                    sampled=sampled,
                    states=states[:1],
                )

    def _warm_plain_pass(
        self,
        null_table: np.ndarray,
        *,
        sampled: bool,
        masked: bool = False,
    ) -> None:
        """Run all configured plain batch sizes before materializing."""
        emitted_batches = [
            self._warm_plain_step(batch, null_table, sampled=sampled, masked=masked)
            for batch in self.ops.batch_sizes
        ]
        for emitted in emitted_batches:
            materialize(emitted)

    def _warm_spec_pass(
        self,
        null_table: np.ndarray,
        ring_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]] | None,
        *,
        sampled: bool,
    ) -> None:
        """Warm unmasked and masked speculative decode paths for ``sampled``."""
        if self.spec_alone:
            self._warm_single_spec(null_table, sampled=sampled)
            if ring_fn is not None:
                self._warm_single_spec(null_table, sampled=sampled, offered=ring_fn())
        for requests, rows_per in self.spec_shapes:
            self._warm_batched_spec(null_table, (requests, rows_per), sampled=sampled)
            if ring_fn is not None:
                self._warm_batched_spec(
                    null_table,
                    (requests, rows_per),
                    sampled=sampled,
                    offered=ring_fn(),
                )
        self._warm_spec_transitions(null_table, sampled=sampled)
        self._warm_plain_pass(null_table, sampled=sampled)

    def _warm_masked_spec_pass(
        self,
        null_table: np.ndarray,
        *,
        sampled: bool,
    ) -> None:
        """Warm grammar-masked decode interleaved with speculative steps."""
        if self.spec_alone:
            self._warm_single_spec(null_table, sampled=sampled)
            self._warm_plain_pass(null_table, sampled=sampled, masked=True)
            self._warm_single_spec(null_table, sampled=sampled)
        for requests, rows_per in self.spec_shapes:
            self._warm_batched_spec(null_table, (requests, rows_per), sampled=sampled)
            self._warm_plain_pass(null_table, sampled=sampled, masked=True)
            self._warm_batched_spec(null_table, (requests, rows_per), sampled=sampled)

    def _warm_steps(self) -> None:
        """Warm plain, speculative, and grammar-masked decode step programs.

        Compile the step ops on the null slot and page, in the orders
        serving runs them (torch_tpu compiles an op per operand buffer
        history): every batch size, a DSpark request entering, continuing,
        changing its table, then batches again; greedy, then sampled; with a
        DSpark ring import, entering from an imported ring once too.
        """
        pages = self._pages
        if pages is None:
            msg = "K3 decode: bind the KV caches before warming up"
            raise RuntimeError(msg)
        started = time.perf_counter()
        layout = self._prepare(pages)
        null_table = np.zeros(pages, dtype=np.int32)
        ring_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]] | None = (
            self._null_ring if (self._ring and self._importer is not None) else None
        )
        self._precompile_decode_programs(layout, null_table)
        speculating = self.draft is not None and (
            self.spec_alone or bool(self.spec_batches)
        )
        for sampled in (False, True):
            self._warm_plain_pass(null_table, sampled=sampled)
            if speculating:
                self._warm_spec_pass(null_table, ring_fn, sampled=sampled)
            self._warm_plain_pass(null_table, sampled=sampled, masked=True)
            if speculating:
                self._warm_masked_spec_pass(null_table, sampled=sampled)
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
                "K3 decode: warmed decode steps (batches %s%s; %d pages) in %.1f s",
                list(self.ops.batch_sizes),
                ", DSpark" if self.draft is not None else "",
                pages,
                time.perf_counter() - started,
            )
