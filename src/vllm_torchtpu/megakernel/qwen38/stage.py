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
"""The Qwen3.8 prefill's pipeline stage: :class:`QwenPrefillStage`.

A step's scheduled tokens are dealt to the stage's cores (``rows``); the
stage embeds them (first stage) or takes the previous stage's hand-off,
runs its layers (``layers``), one op per layer over the stage mesh, and
hands the residual stream on -- or, on the last stage, chooses the first
token of every request that finished its prompt.

GDN states live in vLLM's unified pool, at the block holding a request's
last token of the step (vLLM's ``align`` mode): when a chunk moves a
request to a new state block, the stage first copies the previous block's
state into it, in every GDN layer's pool.
"""

from __future__ import annotations

import math
import re
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
    materialize,
    register_global_op,
    register_op,
)
from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.megakernel.program import (
    KVGroup,
    RequestInput,
    StepInput,
    StepOutput,
)
from vllm_torchtpu.megakernel.qwen38 import layers as qlayers
from vllm_torchtpu.megakernel.qwen38.program import (
    HIDDEN_SIZE,
    MTP_LAYER,
    PCP_SIZE,
    VOCAB,
    check,
    is_full_attention,
)
from vllm_torchtpu.megakernel.qwen38.prologue import CHUNK as RING_CHUNK
from vllm_torchtpu.megakernel.qwen38.reference import BF16, F32, Dims, rms
from vllm_torchtpu.megakernel.qwen38.ring import ring_order
from vllm_torchtpu.megakernel.qwen38.rows import Span, plan_rows, rows_per_core
from vllm_torchtpu.megakernel.qwen38.weights import StageWeights, load, synthetic

logger = init_logger(__name__)

AXIS = qlayers.PCP
# The attention plan's specs (qlayers.attention_plan): per-core rows, runs,
# rotary tables and pooled-key positions, the pooled keys' pages and their
# requests' lengths in them, the path.
PLAN_SPECS = (P(AXIS),) * 8 + (P(),) * 3
Q_BLOCK = 256  # the PCP attention query tile (its VMEM scratch scales with it)
_LAYER = re.compile(r"layers\.(\d+)\.")


class QwenPrefillStage:
    """One pipeline stage of the Qwen3.8 prefill (:class:`PipelineStage`)."""

    def __init__(
        self,
        config: MegakernelEngineConfig,
        checkpoint: Path,
        *,
        layers: range,
        first_stage: bool,
        last_stage: bool,
        mesh: Any,
        core: int,
        synthetic_seed: int | None = None,
        mtp: bool = False,
    ) -> None:
        check(config)
        self.config = config
        self.checkpoint = checkpoint
        self.synthetic_seed = synthetic_seed
        # The last stage writes the MTP layer's KV (the decoder's draft rows).
        self.drafts = mtp and last_stage
        self.dims = Dims()
        self.layers = layers
        self.first_stage, self.last_stage = first_stage, last_stage
        self.mesh, self.core = mesh, core
        self.cores = PCP_SIZE
        self.ring = _ring(mesh)
        self.chunk = int(config.cp_kv_cache_interleave_size or 0)
        self.granule = math.lcm(self.chunk, Q_BLOCK)
        # Rows-per-core sizes a step pads to (:meth:`size_steps`; None: the
        # power-of-two multiples of the granule).
        self._sizes: list[int] | None = None
        self.weights: StageWeights | None = None
        self._pools: dict[int, torch.Tensor] = {}  # layer -> its pool tensor
        self._raw: list[torch.Tensor] = []  # the distinct pool tensors
        self._attention_group = -1
        self._table_widths: tuple[int, ...] = ()
        self._gdn_groups: dict[int, tuple[int, int]] = {}  # layer -> (group, block)
        self._pool_block_tokens = 0
        self._state_column: dict[str, int] = {}
        self._ops: dict[Any, Any] = {}

    # ------------------------------------------------------------ geometry
    def _rows_for(self, tokens: int) -> int:
        """Rows per core of a step of ``tokens`` tokens: the smallest of
        :meth:`buckets` holding them (one compiled set of ops per size)."""
        needed = rows_per_core(tokens, self.cores, self.chunk)
        for rows in self.buckets():
            if rows >= needed:
                return rows
        return -(-needed // self.granule) * self.granule

    def _sized_rows(self, tokens: int) -> int:
        """Rows per core of a scheduler step size: whole rounds of the MoE
        prologue's ring (two streams of ``prologue.CHUNK``-row pushes)."""
        size = math.lcm(self.granule, 2 * RING_CHUNK)
        needed = rows_per_core(tokens, self.cores, self.chunk)
        return -(-needed // size) * size

    def buckets(self) -> list[int]:
        """Every rows-per-core size a step can take: the power-of-two
        multiples of the granule below the first of the scheduler's step
        sizes (:meth:`size_steps`), then each of those sizes; without them,
        power-of-two multiples up to :meth:`max_rows`."""
        top = self.max_rows()
        sizes = sorted(set(self._sizes or ())) or [top]
        small, rows = [], self.granule
        while rows < sizes[0]:
            small.append(rows)
            rows *= 2
        return [*small, *sizes] if self._sizes else [*small, top]

    def size_steps(self, tokens: Sequence[int]) -> None:
        """Pad steps to the token counts the scheduler cuts them at (its step
        cost model's buckets, :meth:`_sized_rows`), rather than to powers of
        two: a step just past a power of two would otherwise cost the next
        one. Before :meth:`warmup`."""
        self._sizes = [min(self._sized_rows(t), self.max_rows()) for t in tokens]

    def _geometry(self, rows: int) -> qlayers.Geometry:
        return qlayers.Geometry(
            dims=self.dims,
            cores=self.cores,
            ring=self.ring,
            chunk=self.chunk,
            rows=rows,
            max_requests=self.config.max_num_seqs,
            pool_block_tokens=self._pool_block_tokens,
            q_block=Q_BLOCK,
            seq_along_lane=_seq_along_lane(),
            history_runs=self._history_runs(),
        )

    def _history_runs(self) -> tuple[int, ...]:
        """The pooled-key sizes (pages per core) a step with cached prefixes
        gathers in: up to 64K and 128K keys, and a request at the model
        length with a granule of others' beside it."""
        span = self.chunk * self.cores
        longest = -(-(self.config.max_model_len + self.granule * self.cores) // span)
        sizes = {runs for runs in (64, 128) if runs < longest} | {-(-longest // 4) * 4}
        return tuple(sorted(sizes))

    # ------------------------------------------------------------ PipelineStage
    def handoff_template(self) -> Mapping[str, tuple[tuple[int, ...], torch.dtype]]:
        return {HIDDEN_STATES: ((HIDDEN_SIZE,), torch.bfloat16)}

    def max_rows(self) -> int:
        needed = rows_per_core(
            self.config.max_num_batched_tokens, self.cores, self.chunk
        )
        return -(-needed // self.granule) * self.granule

    def rows(self, step_input: StepInput) -> int:
        return self._rows_for(sum(max(r.scheduled, 0) for r in step_input.requests))

    def time_steps(self, points: Sequence[tuple[int, int]]) -> list[float]:
        """Each point's step time on this stage (:meth:`_null_step`): the
        median of three runs after a compiling one."""
        times = []
        for tokens, prefix in points:
            step = self._null_step(tokens, prefix)
            runs = []
            for _ in range(4):
                started = time.perf_counter()
                self._run_to_end(step)
                runs.append((time.perf_counter() - started) * 1e3)
            times.append(float(np.median(runs[1:])))
        return times

    def _null_step(self, tokens: int, prefix: int) -> StepInput:
        """A step of one request of ``tokens`` new tokens after ``prefix``
        cached ones, whose block tables name only vLLM's null block (padding's
        block, so no live KV or state is touched). Its tokens and hand-off are
        random: identical rows would all route to the same experts and time
        the MoE at its worst."""
        rng = np.random.default_rng(tokens * 7919 + prefix)
        request = RequestInput(
            req_id=f"qwen38-null-{tokens}-{prefix}",
            computed=prefix,
            scheduled=tokens,
            token_ids=rng.integers(0, VOCAB, prefix + tokens, dtype=np.int32),
            num_tokens=prefix + tokens,
            num_prompt_tokens=prefix + tokens,
            block_tables=tuple(np.zeros(w, np.int32) for w in self._table_widths),
        )
        step = StepInput(requests=[request])
        if not self.first_stage:
            generator = torch.Generator().manual_seed(tokens * 7919 + prefix)
            hidden = torch.randn(
                (self.rows(step), HIDDEN_SIZE), generator=generator
            ).to(torch.bfloat16)
            step.hidden = {HIDDEN_STATES: hidden.to("tpu")}
        return step

    def _run_to_end(self, step: StepInput) -> None:
        output = self.step(step)
        if output.tokens is not None:
            materialize(output.tokens)
        else:
            assert output.hidden is not None
            materialize(output.hidden[HIDDEN_STATES])
        for request in step.requests:
            self._state_column.pop(request.req_id, None)

    # ------------------------------------------------------------ setup
    def load_weights(self) -> None:
        if self.weights is not None:
            return
        if self.synthetic_seed is None:
            self.weights = load(
                self.checkpoint,
                self.dims,
                self.layers,
                self.cores,
                self.core,
                VOCAB,
                first_stage=self.first_stage,
                last_stage=self.last_stage,
                mtp=self.drafts,
            )
        else:
            self.weights = synthetic(
                self.dims,
                self.layers,
                self.cores,
                VOCAB,
                first_stage=self.first_stage,
                last_stage=self.last_stage,
                seed=self.synthetic_seed,
                mtp=self.drafts,
            )

    def bind_kv_caches(
        self, caches: Mapping[str, torch.Tensor], groups: Sequence[KVGroup]
    ) -> None:
        """The pools of this stage's layers, the attention layers' KV cache
        group and each GDN layer's state group."""
        self.load_weights()
        group_of = {
            name: g for g, group in enumerate(groups) for name in group.layer_names
        }
        by_ptr: dict[int, torch.Tensor] = {}
        attention_groups = set()
        for name, cache in caches.items():
            match = _LAYER.search(name)
            if match is None:
                continue
            layer = int(match.group(1))
            if layer not in self.layers and not (self.drafts and layer == MTP_LAYER):
                continue
            # A GDN layer's cache is its one state tensor in a list: the pool.
            tensor = cache[0] if isinstance(cache, (list, tuple)) else cache
            pool = by_ptr.setdefault(int(tensor.data_ptr()), tensor)
            self._pools[layer] = pool
            g = group_of[name]
            if is_full_attention(layer) or layer == MTP_LAYER:
                attention_groups.add(g)
            else:
                self._gdn_groups[layer] = (g, int(groups[g].block_size))
        missing = [
            layer
            for layer in (*self.layers, *((MTP_LAYER,) if self.drafts else ()))
            if layer not in self._pools
        ]
        if missing or len(attention_groups) > 1:
            raise RuntimeError(
                f"Qwen3.8 stage: layers {missing} have no KV cache, or the "
                f"attention layers span groups {sorted(attention_groups)}"
            )
        self._raw = list(by_ptr.values())
        self._table_widths = tuple(int(group.table_width) for group in groups)
        if attention_groups:
            self._attention_group = attention_groups.pop()
        block_sizes = {block for _, block in self._gdn_groups.values()}
        if len(block_sizes) > 1:
            raise RuntimeError(
                f"Qwen3.8 stage: GDN groups of block sizes {block_sizes}"
            )
        # A state slot is one manager block of the pool (vLLM's --block-size).
        self._pool_block_tokens = self.config.block_size

    def warmup(self) -> None:
        """Compile every rows-per-core size's ops: one step that fills it."""
        for rows in self.buckets():
            self._run_to_end(self._null_step(rows * self.cores, 0))

    # ------------------------------------------------------------ steps
    def step(self, step_input: StepInput) -> StepOutput:
        if not self._raw:
            raise RuntimeError(
                "Qwen3.8 stage: a request arrived before the pools were bound"
            )
        for req in step_input.finished:
            self._state_column.pop(req, None)
        requests = [r for r in step_input.requests if r.scheduled > 0]
        if len(step_input.requests) > self.config.max_num_seqs:
            raise RuntimeError("Qwen3.8 stage: more requests than max_num_seqs")
        rows = self.rows(step_input)
        plan = plan_rows(
            [Span(r.computed, r.end) for r in requests], self.cores, self.chunk, rows
        )
        meta = self._metadata(requests, plan)
        self._seed_states(requests)
        if self.first_stage:
            h = self._embed(requests, plan)
        else:
            assert step_input.hidden is not None
            h = self._land(step_input.hidden[HIDDEN_STATES])
        assert self.weights is not None
        for layer in self.layers:
            h = self._run_layer(layer, rows, h, meta)
        if not self.last_stage:
            return StepOutput(tokens=None, hidden={HIDDEN_STATES: h})
        tokens, first = self._first_tokens(step_input.requests, requests, plan, h)
        if self.drafts:
            self._draft_rows(step_input.requests, requests, plan, meta, h, first)
        return StepOutput(tokens)

    def _run_layer(self, layer: int, rows: int, h, meta):
        assert self.weights is not None
        full = is_full_attention(layer)
        op, names = self._layer_op(rows, full_attention=full)
        w = self.weights.layers[layer]
        pool = self._pools[layer]
        state = (
            meta["block_tables"]
            if full
            else meta["state_indices"][self._gdn_slot(layer)]
        )
        h, new_pool = op(
            h,
            pool,
            *(w[name] for name in names),
            meta["positions"],
            meta["query_start_loc"],
            meta["seq_lens"],
            meta["distribution"],
            state,
            *self.layer_plan(layer, rows, meta),
        )
        if new_pool is not pool:
            pool.set_(new_pool)  # donated in place: keep vLLM's tensor identity
        return h

    def layer_plan(self, layer: int, rows: int, meta: dict[str, Any]) -> tuple:
        """The step's inputs every layer like ``layer`` shares, computed by
        the first that needs them and kept in ``meta``."""
        if not is_full_attention(layer):
            slot = self._gdn_slot(layer)
            if f"gdn_plan_{slot}" not in meta:
                meta[f"gdn_plan_{slot}"] = self._gdn_plan_op(rows)(
                    meta["seq_lens"],
                    meta["query_start_loc"],
                    meta["distribution"],
                    meta["state_indices"][slot],
                )
            return meta[f"gdn_plan_{slot}"]
        if "attention_plan" not in meta:
            meta["attention_plan"] = self._attention_plan_op(rows, layer)(
                meta["seq_lens"],
                meta["block_tables"],
                meta["query_start_loc"],
                meta["distribution"],
                meta["positions"],
            )
        return meta["attention_plan"]

    def _attention_plan_op(self, rows: int, layer: int):
        key = ("attention_plan", rows)
        if key not in self._ops:
            g, mesh = self._geometry(rows), self.mesh
            cache_shape = tuple(self._pools[layer].shape)

            def run(seq_lens, block_tables, qsl, distribution, positions):
                meta = {
                    "seq_lens": seq_lens,
                    "block_tables": block_tables,
                    "query_start_loc": qsl,
                    "distribution": distribution,
                    "positions": positions,
                }
                return qlayers.attention_plan(g, mesh, cache_shape, meta)

            self._ops[key] = register_global_op(
                f"pallas::qwen38_prefill_attention_plan_r{rows}",
                mesh,
                run,
                (P(),) * 4 + (P(AXIS),),
                PLAN_SPECS,
            )
        return self._ops[key]

    def _gdn_plan(self, rows: int):
        """``qlayers.gdn_plan`` over the step's metadata, and how many
        arrays it returns."""
        g = self._geometry(rows)

        def run(seq_lens, qsl, distribution, state):
            meta = {
                "seq_lens": seq_lens,
                "query_start_loc": qsl,
                "distribution": distribution,
            }
            return qlayers.gdn_plan(g, meta, state)

        count = self.config.max_num_seqs
        i32 = jax.ShapeDtypeStruct
        abstract = (
            i32((count,), jnp.int32),
            i32((count + 1,), jnp.int32),
            i32((3,), jnp.int32),
            i32((count,), jnp.int32),
        )
        return run, len(jax.eval_shape(run, *abstract))

    def _gdn_plan_op(self, rows: int):
        key = ("gdn_plan", rows)
        if key not in self._ops:
            run, size = self._gdn_plan(rows)
            self._ops[key] = register_global_op(
                f"pallas::qwen38_prefill_gdn_plan_r{rows}",
                self.mesh,
                run,
                (P(),) * 4,
                (P(),) * size,
            )
        return self._ops[key]

    def _gdn_slot(self, layer: int) -> int:
        return sorted({g for g, _ in self._gdn_groups.values()}).index(
            self._gdn_groups[layer][0]
        )

    def _layer_op(self, rows: int, *, full_attention: bool):
        key = ("layer", rows, full_attention)
        if key in self._ops:
            return self._ops[key]
        assert self.weights is not None
        g = self._geometry(rows)
        layer = next(i for i in self.layers if is_full_attention(i) == full_attention)
        names = tuple(sorted(self.weights.layers[layer]))
        sharded = {"conv", "a_log", "dt_bias"} | {
            f"{w}{part}" for w in ("w13", "w2") for part in ("", "_channel", "_ratio")
        }
        mesh = self.mesh

        def run(h, pool, *args):
            w = dict(zip(names, args[: len(names)], strict=True))
            positions, qsl, seq_lens, distribution, state = args[len(names) :][:5]
            plan = args[len(names) + 5 :]
            meta = {
                "positions": positions,
                "query_start_loc": qsl,
                "seq_lens": seq_lens,
                "distribution": distribution,
                "block_tables": state,
                "state_indices": state,
            }
            if full_attention:
                h, pool = qlayers.attention_mixer(g, mesh, h, w, pool, meta, plan)
            else:
                h, pool = qlayers.gdn_mixer(g, mesh, h, w, pool, meta, plan)
            return qlayers.moe_block(g, mesh, h, w), pool

        in_specs = (
            (P(AXIS), P(AXIS))
            + tuple(P(AXIS) if name in sharded else P() for name in names)
            + (P(AXIS), P(), P(), P(), P())
            + (PLAN_SPECS if full_attention else (P(),) * self._gdn_plan(rows)[1])
        )
        kind = "attention" if full_attention else "gdn"
        op = register_global_op(
            f"pallas::qwen38_prefill_{kind}_r{rows}",
            mesh,
            run,
            in_specs,
            (P(AXIS), P(AXIS)),
            donate_argnums=(1,),
        )
        self._ops[key] = (op, names)
        return self._ops[key]

    # ------------------------------------------------------------ host data
    def _metadata(self, requests: Sequence[RequestInput], plan) -> dict[str, Any]:
        """The step's replicated metadata (padded to ``max_num_seqs``
        requests) and each core's row positions."""
        count = self.config.max_num_seqs
        qsl = np.zeros(count + 1, np.int32)
        seq_lens = np.zeros(count, np.int32)
        for i, r in enumerate(requests):
            qsl[i + 1] = qsl[i] + r.scheduled
            seq_lens[i] = r.end
        qsl[len(requests) + 1 :] = qsl[len(requests)]
        # This core's rows only: a sharded operand is each device's own part.
        positions = np.maximum(plan.position[plan.core_rows(self.core)], 0)
        meta: dict[str, Any] = {
            "query_start_loc": qsl,
            "seq_lens": seq_lens,
            "distribution": np.asarray([0, len(requests), len(requests)], np.int32),
            "positions": positions,
        }
        if self._attention_group >= 0:
            width = (
                len(requests[0].block_tables[self._attention_group]) if requests else 1
            )
            tables = np.zeros((count, width), np.int32)
            for i, r in enumerate(requests):
                tables[i] = r.block_tables[self._attention_group]
            meta["block_tables"] = tables
        else:
            meta["block_tables"] = np.zeros((count, 1), np.int32)
        slots = sorted({g for g, _ in self._gdn_groups.values()})
        indices = np.zeros((max(len(slots), 1), count), np.int32)
        for s, g in enumerate(slots):
            block = next(b for gg, b in self._gdn_groups.values() if gg == g)
            for i, r in enumerate(requests):
                table = r.block_tables[g]
                indices[s, i] = table[min((r.end - 1) // block, len(table) - 1)]
        meta["state_indices"] = indices
        out = {k: torch.from_numpy(v).to("tpu") for k, v in meta.items()}
        out["state_indices"] = [
            out["state_indices"][s] for s in range(indices.shape[0])
        ]
        return out

    def _seed_states(self, requests: Sequence[RequestInput]) -> None:
        """Copy each request's GDN state into the block its chunk moves to
        (vLLM's ``align`` mode keeps it at the block of the last token)."""
        if not self._gdn_groups:
            return
        pairs: dict[int, list[tuple[int, int]]] = {}
        for r in requests:
            for g, block in set(self._gdn_groups.values()):
                current = (r.end - 1) // block
                previous = self._state_column.get(r.req_id, (r.computed - 1) // block)
                if 0 <= previous != current:
                    table = r.block_tables[g]
                    pairs.setdefault(g, []).append(
                        (int(table[previous]), int(table[current]))
                    )
            self._state_column[r.req_id] = (r.end - 1) // next(
                iter(self._gdn_groups.values())
            )[1]
        for g, group_pairs in pairs.items():
            layers = [layer for layer, (gg, _) in self._gdn_groups.items() if gg == g]
            for layer in layers:
                pool = self._pools[layer]
                (new,) = self._copy_op(len(group_pairs))(
                    pool, *_pair_tensors(group_pairs)
                )
                if new is not pool:
                    pool.set_(new)

    def _copy_op(self, count: int):
        key = ("copy", count)
        if key not in self._ops:
            block_tokens = self._pool_block_tokens
            self._ops[key] = register_op(
                f"pallas::qwen38_prefill_seed_states_{count}",
                self.mesh,
                lambda pool, src, dst: (
                    copy_state_blocks(pool, src, dst, block_tokens),
                ),
                (P(AXIS), P(), P()),
                (P(AXIS),),
                donate_argnums=(0,),
            )
        return self._ops[key]

    def _embed(self, requests: Sequence[RequestInput], plan):
        tokens = np.zeros(plan.rows, np.int32)  # this core's rows
        for local, f in enumerate(plan.core_rows(self.core)):
            i = plan.request[f]
            if i >= 0:
                tokens[local] = requests[i].token_ids[plan.position[f]]
        key = ("embed", plan.rows)
        if key not in self._ops:
            self._ops[key] = register_op(
                f"pallas::qwen38_prefill_embed_r{plan.rows}",
                self.mesh,
                lambda table, ids: (jnp.take(table, ids, axis=0),),
                (P(), P(AXIS)),
                (P(AXIS),),
            )
        assert self.weights is not None and self.weights.embedding is not None
        (h,) = self._ops[key](
            self.weights.embedding, torch.from_numpy(tokens).to("tpu")
        )
        return h

    def _land(self, hidden: torch.Tensor) -> torch.Tensor:
        """The hand-off as this stage's own array: torch_tpu compiles an op
        for its inputs' placement, and the pipeline's transfer program places
        it differently from the stage's ops, so the layers would otherwise
        compile again at the first step after warmup. Only this copy does."""
        key = ("land", hidden.shape[0])
        if key not in self._ops:
            self._ops[key] = register_op(
                f"pallas::qwen38_prefill_land_r{hidden.shape[0]}",
                self.mesh,
                lambda h: (jnp.copy(h),),
                (P(AXIS),),
                (P(AXIS),),
            )
        (h,) = self._ops[key](hidden)
        return h

    def _first_tokens(
        self, batch, requests, plan, h
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedy first tokens (``[batch, 1]``; -1 for requests that did not
        finish their prompt this step) from the vocabulary-sharded LM head,
        and every batch row's ``[max_num_seqs]`` before that mask."""
        count = self.config.max_num_seqs
        # Row ``b``: the local row of request ``b``'s last token on this core
        # (-1 where another core holds it, or no prompt finished).
        owner = np.full((1, count), -1, np.int32)
        finished = np.zeros(count, bool)
        index = {id(r): i for i, r in enumerate(requests)}
        for b, r in enumerate(batch):
            if r.scheduled <= 0 or r.end < r.num_prompt_tokens:
                continue
            i = index[id(r)]
            f = int(plan.flat_start[i]) + r.scheduled - 1
            if (f // self.chunk) % self.cores == self.core:
                owner[0, b] = (
                    f // (self.chunk * self.cores)
                ) * self.chunk + f % self.chunk
            finished[b] = True
        key = ("first", plan.rows)
        if key not in self._ops:
            width = VOCAB // self.cores
            eps = self.dims.eps

            def body(h, norm, head, owner):
                rows = owner[0]
                picked = jnp.where(
                    (rows >= 0)[:, None],
                    h[jnp.maximum(rows, 0)],
                    jnp.zeros((), h.dtype),
                )
                last = jax.lax.psum(picked.astype(F32), AXIS).astype(BF16)
                logits = jax.lax.dot_general(
                    rms(last, norm, eps),
                    head,
                    (((1,), (1,)), ((), ())),
                    preferred_element_type=F32,
                )
                core = jax.lax.axis_index(AXIS)
                local_id = jnp.argmax(logits, axis=1).astype(jnp.int32)
                local_max = jnp.max(logits, axis=1)
                maxima = jax.lax.all_gather(local_max, AXIS)
                ids = jax.lax.all_gather(local_id + core * width, AXIS)
                best = jnp.argmax(maxima, axis=0)
                return (jnp.take_along_axis(ids, best[None], axis=0)[0],)

            self._ops[key] = register_op(
                f"pallas::qwen38_prefill_first_token_r{plan.rows}",
                self.mesh,
                body,
                (P(AXIS), P(), P(AXIS), P(AXIS)),
                (P(),),
            )
        assert self.weights is not None
        (tokens,) = self._ops[key](
            h,
            self.weights.final_norm,
            self.weights.lm_head,
            torch.from_numpy(owner).to("tpu"),
        )
        keep = torch.from_numpy(finished).to("tpu")
        chosen = torch.where(keep, tokens, torch.full_like(tokens, -1))
        return chosen[: max(len(batch), 1), None], tokens

    def _draft_rows(self, batch, requests, plan, meta, h, first) -> None:
        """Write the MTP layer's KV of the step's rows (``layers.mtp_draft_rows``):
        each row's next token is the prompt's, or for the last row of a prompt
        finished this step its first token (``first``, by batch row)."""
        tokens = np.zeros(plan.rows, np.int32)
        slots = np.full(plan.rows, -1, np.int32)
        row = {id(r): b for b, r in enumerate(batch)}
        for local, f in enumerate(plan.core_rows(self.core)):
            i = plan.request[f]
            if i < 0:
                continue
            r, following = requests[i], plan.position[f] + 1
            if following < r.num_tokens:
                tokens[local] = r.token_ids[following]
            else:
                slots[local] = row[id(r)]
        rows = plan.rows
        key = ("mtp", rows)
        assert self.weights is not None and self.weights.mtp is not None
        names = tuple(sorted(self.weights.mtp))
        if key not in self._ops:
            g, mesh = self._geometry(rows), self.mesh

            def run(h, pool, embedding, final_norm, *args):
                w = dict(zip(names, args[: len(names)], strict=True))
                w.update(embedding=embedding, final_norm=final_norm)
                tokens, slots, first, positions = args[len(names) : len(names) + 4]
                attention_plan = args[len(names) + 4 :]
                meta = {"positions": positions}
                return (
                    qlayers.mtp_draft_rows(
                        g, mesh, h, w, pool, meta, attention_plan, tokens, slots, first
                    ),
                )

            self._ops[key] = register_global_op(
                f"pallas::qwen38_prefill_mtp_r{rows}",
                mesh,
                run,
                (P(AXIS), P(AXIS), P(), P())
                + (P(),) * len(names)
                + (P(AXIS), P(AXIS), P(), P(AXIS))
                + PLAN_SPECS,
                (P(AXIS),),
                donate_argnums=(1,),
            )
        pool = self._pools[MTP_LAYER]
        attention_layer = next(i for i in self.layers if is_full_attention(i))
        (new_pool,) = self._ops[key](
            h,
            pool,
            self.weights.embedding,
            self.weights.final_norm,
            *(self.weights.mtp[n] for n in names),
            torch.from_numpy(tokens).to("tpu"),
            torch.from_numpy(slots).to("tpu"),
            first,
            meta["positions"],
            *self.layer_plan(attention_layer, rows, meta),
        )
        if new_pool is not pool:
            pool.set_(new_pool)


def copy_state_blocks(pool, src, dst, block_tokens: int):
    """``pool`` with each manager block ``src[i]``'s state copied to ``dst[i]``:
    a manager block spans several of the pool's kernel pages, so the copy
    goes through the GDN kernel's own view of the pool by manager block."""
    from vllm_torchtpu.layers.core.gdn_attention import _as_logical_block_pool

    blocks = _as_logical_block_pool(pool, block_tokens)
    return blocks.at[dst].set(blocks[src]).reshape(pool.shape)


def _pair_tensors(pairs):
    src = torch.tensor([a for a, _ in pairs], dtype=torch.int32).to("tpu")
    dst = torch.tensor([b for _, b in pairs], dtype=torch.int32).to("tpu")
    return src, dst


def _ring(mesh) -> tuple[int, ...]:
    """The stage's cores in ring order, from the TPU runtime's attributes of
    the mesh's devices (the mesh's own records carry no chip coordinates)."""
    from torch_tpu._internal.distributed import tpu_distributed

    attributes = {a.id: a for a in tpu_distributed.global_device_attributes()}
    return ring_order([attributes[d.id] for d in mesh.devices.flat])


def _seq_along_lane() -> bool:
    from vllm import envs

    return envs.VLLM_KV_CACHE_LAYOUT == "HND"
