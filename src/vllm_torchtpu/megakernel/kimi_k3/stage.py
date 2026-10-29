# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 prefill pipeline stage: one host's 8 cores and a quarter of the
layers (``kernels/prefill_stage.py``).

:class:`StagePrefillProgram` holds a stage's weights and runs a packed
window through its ops: the first stage embeds the window's tokens, the
others continue from the previous stage's residual stream and block
snapshots (``prefix``, ``blocks``), and the last returns each sequence's
first-token logits. MLA prefixes are context parallel over the stage (each
core attends its own pages, merged across the stage).

:class:`KimiPrefillStage` serves it as a vLLM pipeline stage (the prefill
role under PP4 x PCP8): vLLM's scheduler steps in, the stage's pools and
the next stage's hand-off out.
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

from vllm_torchtpu.distributed.pp_wave import HIDDEN_STATES
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel.bridge import (
    hbm,
    materialize,
    register_global_op,
    register_op,
    report_load_stats,
    stream_to_device,
)
from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.megakernel.kimi_k3.layout import (
    LAYOUT,
    STAGE_SPLIT,
    stage_cache_layout,
)
from vllm_torchtpu.megakernel.kimi_k3.pools import (
    CONV_ROWS,
    DRAFT_LAYER0,
    SSM_ROWS,
    is_mla_layer,
    kda_blocks_at_column,
    kda_blocks_at_length,
    layer_index,
)
from vllm_torchtpu.megakernel.kimi_k3.prefill import (
    COPY_PAIRS,
    SLOTS,
    Piece,
    _device_i32,
    pack_windows,
)
from vllm_torchtpu.megakernel.kimi_k3.program import VOCAB, kernel_modules
from vllm_torchtpu.megakernel.program import (
    KVGroup,
    RequestInput,
    StepInput,
    StepOutput,
)

logger = init_logger(__name__)

AXIS = "k3_sp"
STAGES = 4
CORES = 8
HIDDEN = 7168
DECODE_PAGE_ROWS = LAYOUT.block_shape[0]
# The MLA DSpark draft's aux streams the group ops tap (the others, entering
# layers 24, 48 and 72, are block snapshots): entering layer 3 (the first
# stage's group 0) and layer 90 (the last stage's last group, group 22).
TAP_FIRST, TAP_LAST = 3, 90


class StagePrefillProgram:
    """A stage's device weights and packed-window ops over ``mesh`` (the
    stage's cores in rank order, axis ``k3_sp``); ``core`` is this worker's
    position in it."""

    def __init__(
        self,
        stage: int,
        mesh,
        core: int,
        capacity: int,
        slots: int,
        moe_path: str = "fused",
        taps: bool = False,
    ) -> None:
        """``taps``: the group ops also return the DSpark aux streams on
        this stage (:data:`TAP_FIRST`, :data:`TAP_LAST`)."""
        self._kernels, self._stage_kernels, self._rank_loader = kernel_modules(
            "prefill_megakernel", "prefill_stage", "rank_loader"
        )
        self.stage, self.mesh, self.rank = stage, mesh, core
        self.plan = self._stage_kernels.stage_plan(stage)
        self.capacity, self.slots = int(capacity), int(slots)
        self.moe_path = moe_path
        self.taps = taps
        loader = self._rank_loader
        io = (("embedding",) if stage == 0 else ()) + (
            ("lm_head", "output_res", "final_norm") if self.plan["final"] else ()
        )
        common = ("attn_res", "attn_norm", "mlp_res", "ffn_norm") + (
            ("dense_gu", "dense_down") if stage == 0 else ()
        )
        self.names = {
            "io": io,
            "common": common,
            "kda": loader.PREFILL_KDA,
            "mla": loader.PREFILL_MLA,
            "moe": tuple(self._kernels.MOE_FAMILIES),
        }
        self.weights: dict[str, dict[str, torch.Tensor]] = {}
        self._ops: dict[int, dict[str, Any]] = {}
        self._groups: dict[int, torch.Tensor] = {}

    def load_weights(self, presharded) -> None:
        presharded = Path(presharded)
        iterator = self._rank_loader.iter_prefill_stage_rank(
            presharded,
            presharded / "prefill-sp-cache",
            self.rank,
            self.plan["first"],
            self.plan["last"],
        )

        def store(group, name, tensor):
            self.weights.setdefault(group, {})[name] = tensor

        stats = stream_to_device(iterator, store)
        for group, names in self.names.items():
            missing = [n for n in names if n not in self.weights.get(group, {})]
            if missing:
                raise RuntimeError(
                    f"K3 prefill stage: missing {group} weights {missing}"
                )
        materialize(self.weights["moe"]["expert_ds"])
        if self.rank == 0:
            logger.info(
                "K3 prefill stage %d: core 0 weights resident (%.1f GiB) in %.1f s; "
                "HBM %s",
                self.stage,
                stats["gib"],
                stats["wall"],
                hbm(self.rank),
            )
        report_load_stats(f"prefill stage {self.stage}", self.rank, stats)

    def _flat(self, *groups: str) -> list[torch.Tensor]:
        return [self.weights[g][n] for g in groups for n in self.names[g]]

    def _unflat(self, args, *groups: str) -> tuple[dict, list]:
        out, index = {}, 0
        for group in groups:
            out[group] = {n: args[index + i] for i, n in enumerate(self.names[group])}
            index += len(self.names[group])
        return out, list(args[index:])

    def _count(self, *groups: str) -> int:
        return sum(len(self.names[g]) for g in groups)

    def window_slots(self, window: int) -> int:
        return max(1, min(self.slots, window // 256))

    def build(self, window: int, page_tokens: int) -> dict[str, Any]:
        if window in self._ops:
            return self._ops[window]
        kernels, stage_kernels, mesh = self._kernels, self._stage_kernels, self.mesh
        count = self.window_slots(window)
        sharded, replicated = P(AXIS), P()
        plan = self.plan
        weight_groups = ("common", "kda", "mla", "moe")
        ops = {}
        name = f"k3_prefill_stage{self.stage}_c{count}"

        def arena(pool, ids, local):
            # Prefix pages are the decoder's: a stage block is STAGE_SPLIT of them.
            pages = pool.reshape(-1, DECODE_PAGE_ROWS, *pool.shape[2:])
            return kernels.CpPrefix((pages,), ids, local)

        def group_body(builder):
            def run(
                prefix,
                blocks,
                rec,
                hist,
                rec_at,
                hist_at,
                latents,
                length,
                group,
                segments,
                pool,
                ids,
                local,
                *args,
            ):
                w, _ = self._unflat(args, *weight_groups)
                prefix, blocks, kda_state, mla_state, routes, *tap = builder(
                    prefix,
                    blocks,
                    w["common"],
                    w["kda"],
                    w["mla"],
                    w["moe"],
                    (rec, hist, rec_at, hist_at),
                    (latents,),
                    length,
                    group,
                    segments,
                    arena(pool, ids, local),
                )
                return (prefix, blocks, *kda_state, *mla_state, routes, *tap)

            return run

        state_specs = (
            sharded,
        ) * 7  # prefix, blocks, rec, hist, rec_at, hist_at, latents
        group_in = (
            state_specs
            + (replicated,) * 3
            + (sharded,) * 3
            + (sharded,) * self._count(*weight_groups)
        )
        for key, first in (("group", False), ("group_first", True)):
            if first and plan["groups"][0] != 0:
                continue
            tap_offset = self.tap_offset(first)
            builder = stage_kernels.make_stage_group4(
                mesh,
                seq_len=window,
                segments=count,
                prefix_capacity=self.capacity,
                slots=plan["slots"],
                first=first,
                moe_path=self.moe_path,
                tap_offset=tap_offset,
            )
            tapped = tap_offset is not None
            op_key = f"{key}_tap" if tapped else key
            ops[key] = register_global_op(
                f"pallas::{name}_{op_key}_{self.moe_path}_w{window}",
                mesh,
                group_body(builder),
                group_in,
                state_specs + (sharded,) * (2 if tapped else 1),
                donate_argnums=tuple(range(7)),
            )
        if self.stage == 0:
            embedding = kernels.make_embedding_sp(mesh, seq_len=window)

            def run_embed(tokens, length, table):
                hidden = embedding(tokens, table, length)
                blocks = jnp.zeros((CORES, 8, window // CORES, HIDDEN), jnp.bfloat16)
                return hidden, blocks

            ops["embed"] = register_global_op(
                f"pallas::{name}_embed_w{window}",
                mesh,
                run_embed,
                (sharded, replicated, sharded),
                (sharded, sharded),
            )
        if plan["final"]:
            final = stage_kernels.make_stage_final(
                mesh,
                seq_len=window,
                segments=count,
                prefix_capacity=self.capacity,
                slots=plan["slots"],
                moe_path=self.moe_path,
            )

            def run_final(
                prefix, blocks, latents, length, segments, pool, ids, local, *args
            ):
                w, _ = self._unflat(args, "io", "common", "mla", "moe")
                logits, (latents,), routes = final(
                    prefix,
                    blocks,
                    w["common"],
                    w["mla"],
                    w["moe"],
                    (latents,),
                    length,
                    segments,
                    arena(pool, ids, local),
                    w["io"],
                )
                return logits, latents, routes

            ops["final"] = register_global_op(
                f"pallas::{name}_final_{self.moe_path}_w{window}",
                mesh,
                run_final,
                (sharded,) * 3
                + (replicated,) * 2
                + (sharded,) * 3
                + (sharded,) * self._count("io", "common", "mla", "moe"),
                (replicated, sharded, sharded),
                donate_argnums=(2,),
            )
        self._ops[window] = ops
        return ops

    def tap_offset(self, first: bool) -> int | None:
        """The layer in its group whose entering stream a group op taps:
        :data:`TAP_FIRST` in the first stage's group 0, :data:`TAP_LAST` in
        the last stage's groups (only group 22's is used)."""
        if not self.taps:
            return None
        if first and self.stage == 0:
            return TAP_FIRST % 4
        if not first and self.plan["final"]:
            return TAP_LAST % 4
        return None

    def zero_states(self, window: int) -> list[torch.Tensor]:
        """The stage's KDA states (rec, hist, rec_at, hist_at) and MLA
        latents for a window, rank-leading as the ops take them."""
        count = self.window_slots(window)
        kda, mla = self.plan["kda_layers"], self.plan["mla_layers"]
        groups = self._rank_loader.STAGE_HEAD_GROUPS
        rec = (1, kda, count, groups, 3, 128, 128)
        hist = (1, kda, count, groups, 3, 3, 3, 128)
        return [
            torch.zeros(rec, dtype=torch.float32, device="tpu"),
            torch.zeros(hist, dtype=torch.bfloat16, device="tpu"),
            torch.zeros(rec, dtype=torch.float32, device="tpu"),
            torch.zeros(hist, dtype=torch.bfloat16, device="tpu"),
            torch.zeros((1, mla, window, 576), dtype=torch.bfloat16, device="tpu"),
        ]

    def run(
        self,
        window: int,
        page_tokens: int,
        length,
        segments,
        pools,
        ids,
        local,
        *,
        tokens=None,
        prefix=None,
        blocks=None,
        states=None,
    ):
        """One packed window through the stage: ``tokens`` (the first
        stage: this core's token ids in SP order, ``[1, W/8]``) or the
        previous stage's ``prefix``/``blocks``; ``pools`` the stage's MLA
        layers' pools, in order, whose pages ``ids``/``local`` name
        (``kernels.CpPrefix``, in decode pages). Returns ``(prefix, blocks,
        states, logits, routes, taps)``; ``logits`` from the last stage only,
        ``routes`` the busiest chip's route count per op (fused MoE: above
        ``prefill_stage.MOE_CAPACITY``, routes were dropped), ``taps`` the
        DSpark aux streams this stage computes, by layer (with ``taps``)."""
        ops = self.build(window, page_tokens)
        if states is None:
            states = self.zero_states(window)
        if self.stage == 0:
            prefix, blocks = ops["embed"](
                tokens, length, self.weights["io"]["embedding"]
            )
        rec, hist, rec_at, hist_at, latents = states
        routes, taps = [], {}
        for mla, group in enumerate(self.plan["groups"]):
            op = ops["group_first"] if group == 0 else ops["group"]
            if group not in self._groups:
                self._groups[group] = torch.tensor([group], dtype=torch.int32).to("tpu")
            prefix, blocks, rec, hist, rec_at, hist_at, latents, busiest, *tap = op(
                prefix,
                blocks,
                rec,
                hist,
                rec_at,
                hist_at,
                latents,
                length,
                self._groups[group],
                segments,
                pools[mla],
                ids,
                local,
                *self._flat("common", "kda", "mla", "moe"),
            )
            routes.append(busiest)
            if tap:
                taps[4 * group + self.tap_offset(group == 0)] = tap[0]
        logits = None
        if self.plan["final"]:
            logits, latents, busiest = ops["final"](
                prefix,
                blocks,
                latents,
                length,
                segments,
                pools[-1],
                ids,
                local,
                *self._flat("io", "common", "mla", "moe"),
            )
            routes.append(busiest)
        return (
            prefix,
            blocks,
            [rec, hist, rec_at, hist_at, latents],
            logits,
            routes,
            taps,
        )


# The windows a serving step runs in: a step's tokens plus up to a decode page
# of padding per sequence.
STAGE_WINDOWS = (512, 1024, 2048, 4096, 8192, 16384)
# Residual snapshots a stage hands on: stage s adds snapshots 2s and 2s + 1,
# and the last stage reads the first six.
HANDOFF_SNAPSHOTS = 6
SNAPSHOT = "residual_snapshot_{}"
# With the MLA DSpark draft: the stream entering layer 3, from the first stage
# to the last (which computes the draft's context latents).
DRAFT_TAP = f"draft_tap_{TAP_FIRST}"


@dataclasses.dataclass
class _StageRequest:
    """A request between its steps: the tokens computed so far and its KDA
    state after them (``[1, kda, 1, ...]`` recurrent and conv; None: zero)."""

    req_id: str
    start: int
    state: tuple[torch.Tensor, torch.Tensor] | None = None


class KimiPrefillStage:
    """The K3 prefill's pipeline stage ``stage`` on one host's cores
    (:class:`PipelineStage`): a step's scheduled tokens are packed into one
    window (sequences at decode-page offsets), run through the stage's
    layers, their pages and KDA states written into vLLM's pools for the KV
    connector, and the residual stream with its snapshots handed on; the
    last stage chooses the first token of each finished prompt.

    Pools (``layout.stage_cache_layout``): a block is ``STAGE_SPLIT`` decode
    blocks. With 2048-token blocks under PCP8, chunk ``k`` (2048 tokens) of a
    request is on core ``k % 8`` at block-table column ``k // 8``: the
    decoder's pages ``4 k .. 4 k + 3``, of its DCP ranks ``4 (k % 8) + i`` in
    the same column. A KDA state block holds the core's head shards, shard
    ``g`` in decode block ``g`` as TP32 rank ``4 core + g`` keeps it.
    """

    def __init__(
        self,
        config: MegakernelEngineConfig,
        presharded: Path,
        *,
        stage: int,
        mesh: Any,
        core: int,
        draft: Path | None = None,
    ) -> None:
        """``draft``: the MLA DSpark draft (``Inferact/Kimi-K3-DSpark``)
        whose context latents the last stage computes and writes into the
        draft's KV cache layers, for the decoder to draft with."""
        self.config = config
        self.presharded = Path(presharded)
        self.stage, self.mesh, self.core = stage, mesh, core
        self.page_tokens = config.block_size  # one core's chunk
        self.decode_page = self.page_tokens // STAGE_SPLIT
        self.slots = min(config.max_num_seqs, SLOTS)
        needed = config.max_num_batched_tokens + self.slots * (self.decode_page - 1)
        if needed > STAGE_WINDOWS[-1]:
            raise NotImplementedError(
                f"K3 prefill stage: a step of {config.max_num_batched_tokens} tokens "
                f"over {self.slots} sequences needs a window above {STAGE_WINDOWS[-1]}"
            )
        self.windows = tuple(w for w in STAGE_WINDOWS if w < needed) + (
            min(w for w in STAGE_WINDOWS if w >= needed),
        )
        # The most cached prefix tokens a window's sequences hold in all, and
        # a core's share in decode pages (a chunk of rounding per sequence).
        self.capacity = max(self.windows[-1], -(-config.max_model_len // 4096) * 4096)
        self.cp_pages = (
            -(-(self.capacity // CORES + self.slots * self.page_tokens) // 1024)
            * 1024
            // self.decode_page
        )
        if draft is not None:
            (mla,) = kernel_modules("dspark_mla")
            if not mla.is_mla_draft(Path(draft)):
                raise NotImplementedError(
                    "the K3 prefill stages compute the MLA DSpark draft's context "
                    f"latents only, not {draft}'s ring"
                )
        self.program = StagePrefillProgram(
            stage, mesh, core, self.capacity, self.slots, taps=draft is not None
        )
        self.last = bool(self.program.plan["final"])
        self.tapped = draft is not None
        self.draft = None
        if draft is not None and self.last:
            from vllm_torchtpu.megakernel.kimi_k3.draft_mla_prefill import (
                PrefillDraftLatents,
            )

            held = 32 // CORES  # the TP32 ranks a core stands for
            self.draft = PrefillDraftLatents(
                mesh,
                core,
                AXIS,
                Path(draft),
                self.presharded,
                config.max_model_len,
                fc_ranks=tuple(range(held * core, held * (core + 1))),
            )
        self._kernels = self.program._kernels
        self._requests: dict[str, _StageRequest] = {}
        self._null: set[str] = set()  # timing and warm-up requests
        self._raw: list[torch.Tensor] = []
        self._mla_raw: tuple[int, ...] = ()
        self._kda_raw: tuple[int, ...] = ()
        self._draft_raw: tuple[int, ...] = ()
        self._mla_group = 0
        self._kda_groups: list[tuple[int, int]] = []
        self._table_widths: tuple[int, ...] = ()
        self._ops: dict[Any, Any] = {}
        self._loaded = False

    # ------------------------------------------------------------ PipelineStage
    def handoff_template(self) -> Mapping[str, tuple[tuple[int, ...], torch.dtype]]:
        row = ((HIDDEN,), torch.bfloat16)
        return {
            HIDDEN_STATES: row,
            **{SNAPSHOT.format(i): row for i in range(HANDOFF_SNAPSHOTS)},
            **({DRAFT_TAP: row} if self.tapped else {}),
        }

    def max_rows(self) -> int:
        return self.windows[-1] // CORES

    def rows(self, step_input: StepInput) -> int:
        return sum(window for window, _ in self._plan(step_input.requests)) // CORES

    def time_steps(self, points: Sequence[tuple[int, int]]) -> list[float]:
        """Each point's step time on this stage: the median of three null
        steps after a compiling one."""
        times = []
        for tokens, prefix in points:
            runs = []
            for _ in range(4):
                step = self._null_step(tokens, prefix)
                started = time.perf_counter()
                self._run_to_end(step)
                runs.append((time.perf_counter() - started) * 1e3)
            times.append(float(np.median(runs[1:])))
        return times

    # ------------------------------------------------------------ setup
    def load_weights(self) -> None:
        if not self._loaded:
            self.program.load_weights(self.presharded)
            if self.draft is not None:
                self.draft.load_weights()
            self._loaded = True

    def bind_kv_caches(
        self, caches: Mapping[str, torch.Tensor], groups: Sequence[KVGroup]
    ) -> None:
        """The stage's layers' pools; ``groups`` say which block-table row
        holds a request's MLA pages and where each KDA layer's state blocks
        are."""
        self.load_weights()
        plan = self.program.plan
        layers = sorted(
            (
                layer_index(name),
                name,
                cache[0] if isinstance(cache, (list, tuple)) else cache,
            )
            for name, cache in caches.items()
            if plan["first"] <= layer_index(name) < plan["last"]
        )
        mla = [entry for entry in layers if is_mla_layer(entry[0])]
        kda = [entry for entry in layers if not is_mla_layer(entry[0])]
        if len(mla) != plan["mla_layers"] or len(kda) != plan["kda_layers"]:
            raise RuntimeError(
                f"K3 prefill stage {self.stage}: expected {plan['mla_layers']} MLA and "
                f"{plan['kda_layers']} KDA layers, found {len(mla)} and {len(kda)}"
            )
        # The MLA DSpark draft's layers (vLLM numbers them after the target's)
        # live on the last stage, which writes them.
        drafts = sorted(
            (
                layer_index(name),
                name,
                cache[0] if isinstance(cache, (list, tuple)) else cache,
            )
            for name, cache in caches.items()
            if layer_index(name) >= DRAFT_LAYER0
        )
        expected = 0 if self.draft is None else self.draft.config.layers
        if len(drafts) != expected:
            raise RuntimeError(
                f"K3 prefill stage {self.stage}: expected {expected} MLA DSpark draft "
                f"layers, found {len(drafts)}"
            )
        index_of: dict[int, int] = {}
        for _, _, tensor in layers + drafts:
            if tensor.data_ptr() not in index_of:
                index_of[tensor.data_ptr()] = len(self._raw)
                self._raw.append(tensor)
        self._mla_raw = tuple(index_of[t.data_ptr()] for _, _, t in mla)
        self._kda_raw = tuple(index_of[t.data_ptr()] for _, _, t in kda)
        self._draft_raw = tuple(index_of[t.data_ptr()] for _, _, t in drafts)
        layout = stage_cache_layout(
            self.config.kv_cache_dtype,
            tuple(layer for layer, _, _ in mla),
            tuple(layer for layer, _, _ in kda),
        )
        shapes = {tuple(t.shape[1:]) for t in self._raw}
        dtypes = {str(t.dtype).removeprefix("torch.") for t in self._raw}
        if shapes != {layout.block_shape} or dtypes != {layout.dtype}:
            raise NotImplementedError(
                f"K3 prefill stage: expected pools [blocks, {layout.block_shape}] "
                f"{layout.dtype}; vLLM allocated {sorted(shapes)} {sorted(dtypes)}"
            )
        of = {name: g for g, group in enumerate(groups) for name in group.layer_names}
        mla_groups = {of[name] for _, name, _ in mla + drafts}
        if len(mla_groups) != 1:
            raise RuntimeError(
                "K3 prefill stage: the MLA layers are in KV cache groups "
                f"{sorted(mla_groups)}"
            )
        self._mla_group = mla_groups.pop()
        self._kda_groups = [
            (of[name], int(groups[of[name]].block_size)) for _, name, _ in kda
        ]
        self._table_widths = tuple(int(group.table_width) for group in groups)

    def warmup(self) -> None:
        """Compile every window's ops: a step filling each."""
        for window in self.windows:
            started = time.perf_counter()
            self._run_to_end(self._null_step(window, 0))
            if self.core == 0:
                logger.info(
                    "K3 prefill stage %d: %d-token window ready in %.1f s",
                    self.stage,
                    window,
                    time.perf_counter() - started,
                )

    # ------------------------------------------------------------ steps
    def _plan(self, rows: Sequence[RequestInput]):
        """The step's windows, ``[(size, [(piece, offset)])]``: several when
        the sequences' cached prefixes overflow one window's arena. They run
        one after another and hand on their rows end to end, at most the
        largest window's (one sequence per window if packing exceeds it)."""
        pieces = [
            Piece(i, row.computed, row.end, None, self.decode_page)
            for i, row in enumerate(rows)
            if row.scheduled > 0
        ]
        if not pieces:
            return []
        largest = self.windows[-1]
        packed = pack_windows(pieces, self.windows, self.slots, self.capacity)
        if sum(window for window, _ in packed) > largest:
            packed = pack_windows(pieces, self.windows, 1, self.capacity)
        if sum(window for window, _ in packed) > largest:
            raise RuntimeError(
                "K3 prefill stage: a step's windows "
                f"{[window for window, _ in packed]} exceed the {largest}-row hand-off"
            )
        return packed

    def step(self, step_input: StepInput) -> StepOutput:
        if not self._raw:
            raise RuntimeError(
                "K3 prefill stage: a request arrived before the KV pools were bound"
            )
        if step_input.copies:
            self._copy_blocks([(int(a), int(b)) for a, b in step_input.copies])
        for req in step_input.finished:
            self._requests.pop(req, None)
        rows = step_input.requests
        if len(rows) > self.config.max_num_seqs:
            raise RuntimeError(
                f"K3 prefill stage: {len(rows)} rows exceed max_num_seqs "
                f"{self.config.max_num_seqs}"
            )
        plan = self._plan(rows)
        assert plan, "a step without tokens"
        outputs, start = [], 0
        for window, members in plan:
            outputs.append(self._window(step_input, window, members, start))
            start += window // CORES
        if not self.last:
            return StepOutput(
                tokens=None,
                hidden={
                    name: torch.cat([out[name] for out in outputs])
                    if len(outputs) > 1
                    else outputs[0][name]
                    for name in outputs[0]
                },
            )
        logits = torch.cat(
            [
                out[: len(members)]
                for out, (_, members) in zip(outputs, plan, strict=True)
            ]
        )
        members = [member for _, window_members in plan for member in window_members]
        return StepOutput(self._first_tokens(rows, logits, members))

    def _window(self, step_input: StepInput, window: int, members, start: int):
        """One of a step's windows through the stage, its hand-off rows from
        row ``start`` of the step's: its hand-off, or on the last stage its
        sequences' logits."""
        rows = step_input.requests
        count = self.program.window_slots(window)
        states = [self._request(rows[piece.row], piece) for piece, _ in members]
        tokens = np.zeros(window, np.int32)
        offsets, lengths, bases, cached, finals = [], [], [], [], []
        base = 0
        for piece, offset in members:
            row = rows[piece.row]
            tokens[offset : offset + piece.length] = row.token_ids[
                piece.start : piece.end
            ]
            offsets.append(offset)
            lengths.append(piece.length)
            bases.append(base)
            cached.append(piece.start)
            base += piece.start
            finals.append(
                kda_blocks_at_length(row.block_tables, self._kda_groups, piece.end)
            )
        unused = count - len(members)
        descriptor = (
            offsets
            + [window] * unused
            + lengths
            + [0] * unused
            + [0] * count  # no checkpoints
            + bases
            + [base] * unused
            + cached
            + [0] * unused
        )
        finals += [np.zeros(len(self._kda_groups), np.int32)] * unused
        ids, local, table = self._cp_operands(rows, window, members, count)
        initial = self.program.zero_states(window)
        if any(state.state is not None for state in states):
            initial[0], initial[1] = self._stack_states(states, count)
        inputs: dict[str, Any] = {}
        if self.stage == 0:
            order = self._kernels.sp_order(window, CORES)
            local_rows = window // CORES
            mine = tokens[order[self.core * local_rows : (self.core + 1) * local_rows]]
            inputs["tokens"] = torch.from_numpy(mine[None]).to("tpu")
        else:
            assert step_input.hidden is not None
            inputs["prefix"], inputs["blocks"] = self._land(
                step_input.hidden, window, start
            )
        extent = members[-1][1] + members[-1][0].length
        prefix, blocks, out, logits, _, taps = self.program.run(
            window,
            self.decode_page,
            _device_i32([extent]),
            _device_i32(descriptor),
            [self._raw[index] for index in self._mla_raw],
            ids,
            local,
            states=initial,
            **inputs,
        )
        recurrent, conv, _, _, latents = out
        draft = None
        if self.draft is not None:
            assert step_input.hidden is not None
            positions = np.full(window, -1, np.int32)
            for piece, offset in members:
                positions[offset : offset + piece.length] = np.arange(
                    piece.start, piece.end
                )
            draft = self.draft.latents(
                window,
                blocks,
                step_input.hidden[DRAFT_TAP][start : start + window // CORES].unsqueeze(
                    0
                ),
                taps[TAP_LAST],
                positions,
                self._kernels.sp_order(window, CORES),
            )
        self._write(
            window,
            count,
            latents,
            recurrent,
            conv,
            table,
            np.concatenate(finals),
            draft,
        )
        for s, ((piece, _), state) in enumerate(zip(members, states, strict=True)):
            state.start = piece.end
            state.state = (recurrent[:, :, s : s + 1], conv[:, :, s : s + 1])
            if piece.end >= rows[piece.row].num_prompt_tokens:
                self._requests.pop(state.req_id, None)
        if self.core == 0 and self.stage == 0:
            logger.debug(
                "K3 prefill stage: %d-token window: %s",
                window,
                ", ".join(
                    f"{rows[p.row].req_id} {p.start}..{p.end - 1}@{o}"
                    for p, o in members
                ),
            )
        if self.last:
            return logits
        assert prefix is not None and blocks is not None
        hidden = {HIDDEN_STATES: prefix[0]}
        for i in range(HANDOFF_SNAPSHOTS):
            hidden[SNAPSHOT.format(i)] = blocks[0, i]
        if self.tapped and self.stage == 0:
            hidden[DRAFT_TAP] = taps[TAP_FIRST][0]
        elif self.tapped:
            assert step_input.hidden is not None
            hidden[DRAFT_TAP] = step_input.hidden[DRAFT_TAP][
                start : start + window // CORES
            ]
        return hidden

    def _request(self, row: RequestInput, piece: Piece) -> _StageRequest:
        state = self._requests.get(row.req_id)
        if state is None or state.start != piece.start:
            if piece.start % self.page_tokens:
                raise RuntimeError(
                    f"K3 prefill stage: request {row.req_id} continues mid-chunk at "
                    f"token {piece.start}; the stages need chunks of whole blocks"
                )
            state = _StageRequest(row.req_id, piece.start)
            if piece.start and row.req_id not in self._null:
                # A prefix-cache hit: vLLM's align mode keeps the KDA state
                # after column p's tokens in column p of the state groups.
                column = piece.start // self._kda_groups[0][1] - 1
                state.state = self._read_states(
                    kda_blocks_at_column(row.block_tables, self._kda_groups, column)
                )
            self._requests[row.req_id] = state
        return state

    def _read_states(self, blocks: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """The KDA states in state blocks ``blocks`` (``[kda]``, one per layer),
        as the ops take one sequence's: ``[1, kda, 1, 4, ...]`` recurrent and
        conv."""
        if "read" not in self._ops:
            from vllm_torchtpu.kernels import pool_adapters

            num_raw, kda_raw, split = len(self._raw), self._kda_raw, STAGE_SPLIT

            def body(*args):
                pools, blocks = args[:num_raw], args[num_raw]
                raws = [p.reshape(-1, DECODE_PAGE_ROWS, *p.shape[2:]) for p in pools]
                recurrent, conv = [], []
                for k, index in enumerate(kda_raw):
                    ids = split * blocks[k] + jnp.arange(split, dtype=jnp.int32)
                    ssm = pool_adapters.gather_region(
                        raws[index],
                        ids,
                        tok0=0,
                        ntok=SSM_ROWS,
                        out_dtype=jnp.float32,
                        out_lanes=128,
                    )
                    taps = pool_adapters.gather_region(
                        raws[index],
                        ids,
                        tok0=SSM_ROWS,
                        ntok=CONV_ROWS,
                        out_dtype=jnp.bfloat16,
                    )
                    recurrent.append(ssm[:, : 3 * 128].reshape(split, 3, 128, 128))
                    conv.append(
                        taps.reshape(split, -1)[:, : 3 * 3 * 3 * 128].reshape(
                            split, 3, 3, 3, 128
                        )
                    )
                return (
                    jnp.stack(recurrent)[None, :, None],
                    jnp.stack(conv)[None, :, None],
                )

            self._ops["read"] = register_op(
                f"pallas::k3_prefill_stage{self.stage}_read_states_r{num_raw}",
                self.mesh,
                body,
                (P(AXIS),) * num_raw + (P(),),
                (P(AXIS), P(AXIS)),
            )
        recurrent, conv = self._ops["read"](*self._raw, _device_i32(blocks))
        return recurrent, conv

    def _stack_states(self, states: list[_StageRequest], count: int):
        """The window slots' KDA states, ``[1, kda, count, ...]`` (zero for
        new sequences and unused slots)."""
        zero_rec, zero_conv = (
            x[:, :, :1] for x in self.program.zero_states(STAGE_WINDOWS[0])[:2]
        )
        pairs = [state.state or (zero_rec, zero_conv) for state in states]
        pairs += [(zero_rec, zero_conv)] * (count - len(states))
        return (
            torch.cat([rec for rec, _ in pairs], dim=2),
            torch.cat([conv for _, conv in pairs], dim=2),
        )

    def _land(self, hidden: Mapping[str, torch.Tensor], window: int, start: int = 0):
        """The previous stage's hand-off rows ``start`` on as a window's
        ``prefix`` and ``blocks``."""
        rows = window // CORES
        prefix = hidden[HIDDEN_STATES][start : start + rows].unsqueeze(0)
        snapshots = [
            hidden[SNAPSHOT.format(i)][start : start + rows]
            for i in range(HANDOFF_SNAPSHOTS)
        ]
        rest = torch.zeros(
            (8 - HANDOFF_SNAPSHOTS, rows, HIDDEN), dtype=torch.bfloat16
        ).to("tpu")
        return prefix, torch.cat((torch.stack(snapshots), rest)).unsqueeze(0)

    def _cp_operands(self, rows, window: int, members, count: int):
        """This core's context-parallel operands, in decode pages: its own
        prefix pages end to end (``[1, 1 + cp_pages]``: count, then block
        ids), each sequence's base and token count in them (``[1, 2 *
        count]``), and the decode page each window page writes (``[1, W /
        decode page]``; the null block for pages of other cores)."""
        split, own, bases, counts, cursor = STAGE_SPLIT, [], [], [], 0
        table = [0] * (window // self.decode_page)
        for piece, offset in members:
            row = rows[piece.row].block_tables[self._mla_group]
            owned = [
                split * int(row[k // CORES]) + i
                for k in range(self.core, piece.start // self.page_tokens, CORES)
                for i in range(split)
            ]
            bases.append(cursor)
            counts.append(len(owned) * self.decode_page)
            own += owned
            cursor += len(owned) * self.decode_page
            first = piece.start // self.decode_page
            for j in range(piece.rows // self.decode_page):
                chunk, part = divmod(first + j, split)
                if chunk % CORES == self.core:
                    table[offset // self.decode_page + j] = (
                        split * int(row[chunk // CORES]) + part
                    )
        if len(own) > self.cp_pages:
            raise RuntimeError(
                f"K3 prefill stage: this core's prefix share {len(own)} pages "
                f"exceeds {self.cp_pages}"
            )
        unused = count - len(members)
        ids = [len(own)] + own + [0] * (self.cp_pages - len(own))
        local = bases + [cursor] * unused + counts + [0] * unused
        return _device_i32([ids]), _device_i32([local]), _device_i32([table])

    def _write(
        self, window, count, latents, recurrent, conv, table, kda_blocks, draft=None
    ) -> None:
        drafts = () if draft is None else (draft,)
        new = self._write_op(window, count, bool(drafts))(
            latents,
            recurrent,
            conv,
            *drafts,
            *self._raw,
            table,
            _device_i32(kda_blocks),
        )
        self._adopt(new)

    def _adopt(self, new) -> None:
        for old, fresh in zip(self._raw, new, strict=True):
            if fresh is not old:
                old.set_(fresh)  # donated in place: keep vLLM's tensor identity

    def _write_op(self, window: int, count: int, draft: bool = False) -> Any:
        """``(latents, recurrent, conv, [draft,] *pools, table, kda_blocks) ->
        pools``: each window page's latents (and with ``draft`` the MLA DSpark
        draft's, ``[n, layers, W, 576]``) into its decode page (``table``),
        and each slot's final KDA states, head shard ``g`` into decode block
        ``4 b + g`` of its block ``b`` (``kda_blocks``: ``[count * kda]``)."""
        key = ("write", window, count, draft)
        if key in self._ops:
            return self._ops[key]
        from vllm_torchtpu.kernels import pool_adapters

        num_raw, mla_raw, kda_raw = len(self._raw), self._mla_raw, self._kda_raw
        draft_raw = self._draft_raw if draft else ()
        lead = 4 if draft else 3
        pages, split = window // self.decode_page, STAGE_SPLIT

        def body(*args):
            latents, recurrent, conv = args[:3]
            pools, (table, kda_blocks) = (
                args[lead : lead + num_raw],
                args[lead + num_raw :],
            )
            raws = [p.reshape(-1, DECODE_PAGE_ROWS, *p.shape[2:]) for p in pools]
            table, latents, recurrent, conv = (
                table[0],
                latents[0],
                recurrent[0],
                conv[0],
            )
            sources = [(latents[m], index) for m, index in enumerate(mla_raw)]
            if draft:
                sources += [(args[3][0][d], index) for d, index in enumerate(draft_raw)]
            for source, index in sources:
                rows = (
                    jnp.pad(source, ((0, 0), (0, 64)))
                    .astype(raws[index].dtype)
                    .reshape(pages, *raws[index].shape[1:])
                )

                def write_page(p, pool, rows=rows):
                    return jax.lax.dynamic_update_index_in_dim(
                        pool,
                        jax.lax.dynamic_index_in_dim(rows, p, keepdims=False),
                        table[p],
                        axis=0,
                    )

                raws[index] = jax.lax.fori_loop(0, pages, write_page, raws[index])
            shards = split * kda_blocks.reshape(count, -1)[:, :, None] + jnp.arange(
                split
            )
            for k, index in enumerate(kda_raw):
                ids = shards[:, k].reshape(-1)
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
                ssm = recurrent[k].reshape(count * split, -1, 128)
                ssm = jnp.pad(ssm, ((0, 0), (0, ssm_shape[1] - ssm.shape[1]), (0, 0)))
                raws[index] = pool_adapters.scatter_region(
                    raws[index], ssm, ids, tok0=0, ntok=SSM_ROWS
                )
                conv_shape = jax.eval_shape(
                    lambda raw=raws[index], ids=ids: pool_adapters.gather_region(
                        raw, ids, tok0=SSM_ROWS, ntok=CONV_ROWS, out_dtype=jnp.bfloat16
                    )
                ).shape
                taps = conv[k].reshape(count * split, -1).astype(jnp.bfloat16)
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
            return tuple(
                raw.reshape(p.shape) for raw, p in zip(raws, pools, strict=True)
            )

        self._ops[key] = register_op(
            f"pallas::k3_prefill_stage{self.stage}_write{'_d' if draft else ''}"
            f"_c{count}_r{num_raw}_w{window}",
            self.mesh,
            body,
            (P(AXIS),) * (lead + num_raw) + (P(AXIS), P()),
            (P(AXIS),) * num_raw,
            donate_argnums=tuple(range(lead, lead + num_raw)),
        )
        return self._ops[key]

    def _copy_blocks(self, pairs: list[tuple[int, int]]) -> None:
        """vLLM's copy-on-write block copies, ``COPY_PAIRS`` per call."""
        for first in range(0, len(pairs), COPY_PAIRS):
            chunk = list(pairs[first : first + COPY_PAIRS])
            chunk += [(0, 0)] * (COPY_PAIRS - len(chunk))
            self._adopt(
                self._copy_op()(
                    *self._raw,
                    _device_i32([a for a, _ in chunk]),
                    _device_i32([b for _, b in chunk]),
                )
            )

    def _copy_op(self) -> Any:
        if "copy" not in self._ops:
            num_raw = len(self._raw)

            def body(*args):
                raws, source, destination = (
                    args[:num_raw],
                    args[num_raw],
                    args[num_raw + 1],
                )
                return tuple(raw.at[destination].set(raw[source]) for raw in raws)

            self._ops["copy"] = register_op(
                f"pallas::k3_prefill_stage{self.stage}_copy_blocks_r{num_raw}",
                self.mesh,
                body,
                (P(AXIS),) * num_raw + (P(), P()),
                (P(AXIS),) * num_raw,
                donate_argnums=tuple(range(num_raw)),
            )
        return self._ops["copy"]

    def _first_tokens(self, rows, logits, members) -> torch.Tensor:
        """The first token of each row that finished its prompt (``[rows,
        1]``, -1 elsewhere), chosen from the window's logits on the stage's
        cores with the decoder's in-op sampling (one compiled op over
        ``max_num_seqs`` rows)."""
        count = self.config.max_num_seqs
        zero = torch.zeros((1, VOCAB), dtype=torch.float32).to("tpu")
        slot = {piece.row: s for s, (piece, _) in enumerate(members)}
        done = {
            piece.row
            for piece, _ in members
            if piece.end >= rows[piece.row].num_prompt_tokens
        }
        stacked = torch.cat(
            [
                logits[slot[i] : slot[i] + 1].float() if i in done else zero
                for i in range(len(rows))
            ]
            + [zero] * (count - len(rows))
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
        """``(logits [n, VOCAB] replicated, params [n, 5]) -> tokens [n]``,
        as the slice prefill's first-token op, over the stage's cores."""
        key = ("sampler", sampled)
        if key in self._ops:
            return self._ops[key]
        from jax.sharding import Mesh

        from vllm_torchtpu.megakernel.kimi_k3 import sampling

        width = VOCAB // CORES

        def body(logits, params):
            rank = jax.lax.axis_index(sampling.AXIS)
            local = jax.lax.dynamic_slice_in_dim(logits, rank * width, width, axis=1)
            chosen = sampling.greedy(local)
            if sampled:
                temperature, top_p = (
                    jax.lax.bitcast_convert_type(params[:, i], jnp.float32)
                    for i in (0, 1)
                )
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

        self._ops[key] = register_op(
            f"pallas::k3_prefill_stage_first_token{'_sampled' if sampled else ''}",
            Mesh(self.mesh.devices, axis_names=(sampling.AXIS,)),
            body,
            (P(), P()),
            (P(),),
        )
        return self._ops[key]

    def _null_step(self, tokens: int, prefix: int) -> StepInput:
        """A step of one ``tokens``-token request after ``prefix`` cached
        tokens whose block tables name only the null block; zeros as the
        hand-off. Its tokens are spread over the vocabulary: one repeated
        token routes the whole window to the same experts."""
        req_id = f"k3-stage-null-{tokens}-{prefix}"
        self._null.add(req_id)
        spread = np.arange(prefix + tokens, dtype=np.int64) * 7919 % (VOCAB - 1024)
        request = RequestInput(
            req_id=req_id,
            computed=prefix,
            scheduled=tokens,
            token_ids=spread.astype(np.int32),
            num_tokens=prefix + tokens,
            num_prompt_tokens=prefix + tokens,
            block_tables=tuple(np.zeros(w, np.int32) for w in self._table_widths),
        )
        step = StepInput(requests=[request])
        if self.stage > 0:
            rows = self.rows(step)
            step.hidden = {
                name: torch.zeros((rows, *shape), dtype=dtype).to("tpu")
                for name, (shape, dtype) in self.handoff_template().items()
            }
        return step

    def _run_to_end(self, step: StepInput) -> None:
        output = self.step(step)
        if output.tokens is not None:
            materialize(output.tokens)
        else:
            assert output.hidden is not None
            materialize(output.hidden[HIDDEN_STATES])
        for request in step.requests:
            self._requests.pop(request.req_id, None)
            self._null.discard(request.req_id)
