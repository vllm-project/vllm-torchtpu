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
"""The Qwen3.8 decode program (the P/D pair's KV consumer).

TP32 with DCP8 attention over vLLM's unified pools, in the layout the KV
transfer reshards the prefill's into (HND FP8 pages, ``kernels.paged_kv``; GDN
states as ``kernels.pool_state`` describes). Each step is one op for the step's
batch bucket (1, 2, 4, 8 or 16 requests): the decode megakernel reads every
request's pages through its block table, a separate commit writes the new
key/value columns, and the next tokens are chosen in the op (greedy, or
temperature / top-p / top-k / seed on the sharded logits). Every request's GDN
state lives in a private slot, imported from the pool's state blocks on the
request's first step, and its next input token in a device-side bank: under
async scheduling nothing is uploaded per step but the small plan.

With MTP (the checkpoint's draft layer, ``mtp``), every step verifies: each
request's pending token and the three drafts the previous step proposed
(none for a request's first step: its count is forced to zero) run as four
causal rows of the target, the accepted prefix is chosen in the op (greedy,
or exact rejection sampling of the deterministic drafts), only the accepted
rows' KV is committed and the GDN state taken after the last of them, the
accepted drafts' MTP KV rows are recomputed from the target's hidden states
(repair), and three draft calls propose the next step's drafts. A request's
next position, like its next token, is kept on the device: async scheduling
plans a step before the previous one's count is known. The MTP layer's
KV lives in its own pool (``program.MTP_LAYER``), written by the prefill for
the prompt.
"""

from __future__ import annotations

import dataclasses
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import torch
from jax import lax
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.megakernel import bridge
from vllm_torchtpu.megakernel.config import MegakernelEngineConfig
from vllm_torchtpu.megakernel.kimi_k3 import sampling
from vllm_torchtpu.megakernel.kimi_k3.pools import SlotMap
from vllm_torchtpu.megakernel.program import (
    KVGroup,
    RequestInput,
    StepInput,
    StepOutput,
)
from vllm_torchtpu.megakernel.qwen38 import decode_weights
from vllm_torchtpu.megakernel.qwen38.kernels import decode_megakernel, pool_state
from vllm_torchtpu.megakernel.qwen38.program import (
    MAX_DECODE_REQUESTS,
    MTP_DRAFTS,
    MTP_LAYER,
)

logger = init_logger(__name__)

TP_SIZE = 32
BATCHES = (1, 2, 4, 8, MAX_DECODE_REQUESTS)
IMPORT_ROWS = 4  # requests per state import op
_LAYER = re.compile(r"\.layers\.(\d+)\.")


def _layer(name: str) -> int:
    match = _LAYER.search(name)
    if match is None:
        raise ValueError(f"no layer index in KV cache name {name!r}")
    return int(match.group(1))


def _raw(cache: Any) -> torch.Tensor:
    if isinstance(cache, (list, tuple)):
        if len(cache) != 1:
            raise ValueError("expected one raw pool per layer")
        return cache[0]
    return cache


@dataclasses.dataclass
class _Request:
    slot: int
    position: int  # the position of its next input token


class QwenDecoder:
    """See the module docstring."""

    def __init__(
        self,
        config: MegakernelEngineConfig,
        presharded: Path,
        *,
        tp_rank: int,
        mtp_checkpoint: Path | None = None,
    ) -> None:
        self.config = config
        self.presharded = Path(presharded)
        # The checkpoint of the MTP draft layer (None: no speculation).
        self.mtp_checkpoint = mtp_checkpoint
        # The KV transfer lands TP rank q's heads and pages on its device: the
        # kernel mesh follows the TP ranks.
        self.mesh, self.rank = bridge.kernel_mesh("tp", TP_SIZE, rank=tp_rank)
        self.page = int(config.cp_kv_cache_interleave_size or 0)
        self.layout = decode_weights.decode_layout(self.presharded, self.page)
        c = self.layout.config
        self.attention_layers = tuple(c.full_attention)
        self.gdn_layers = tuple(i for i in range(c.layers) if i not in c.full_attention)
        self.batches = tuple(b for b in BATCHES if b <= config.max_num_seqs) or (1,)
        if self.batches[-1] < config.max_num_seqs:
            self.batches += (next(b for b in BATCHES if b >= config.max_num_seqs),)
        self.num_slots = config.max_num_seqs + 1
        self.slots = SlotMap(self.num_slots)
        self._requests: dict[str, _Request] = {}
        self._ops: dict[object, Any] = {}
        self.weights: dict[str, torch.Tensor] = {}
        self.conv = self.rec = self.tokens = None
        self.raws: list[torch.Tensor] = []
        self.mtp_weights: dict[str, torch.Tensor] = {}
        self.mtp_raw: torch.Tensor | None = None
        # By slot: the drafts the next step verifies, and the position of the
        # next input plus one (0: no step yet). Under async scheduling the host
        # position counts the previous step's drafts as accepted.
        self.drafts = self.resume = None
        self.gdn_pools: tuple[int, ...] = ()
        self._attention_group = 0
        self._gdn_groups: list[tuple[int, int]] = []
        self._width = 0

    # ------------------------------------------------------------ setup
    def load_weights(self) -> None:
        self.weights, stats = decode_weights.load(
            self.presharded, self.rank, self.layout
        )
        bridge.report_load_stats("Qwen3.8 decode", self.rank, stats)
        specs = self.layout.state_specs(1024, 1)
        layers = specs["conv"].shape[0]
        self.conv = torch.zeros(
            (1, layers, self.num_slots, *specs["conv"].shape[2:]), dtype=torch.bfloat16
        ).to("tpu")
        self.rec = torch.zeros(
            (1, layers, self.num_slots, *specs["rec"].shape[2:]), dtype=torch.bfloat16
        ).to("tpu")
        self.tokens = torch.zeros((1, self.num_slots), dtype=torch.int32).to("tpu")
        if self.mtp_checkpoint is not None:
            self.mtp_weights, stats = decode_weights.load_mtp(
                self.presharded, self.rank
            )
            bridge.report_load_stats("Qwen3.8 MTP", self.rank, stats)
            self.drafts = torch.zeros(
                (1, self.num_slots, MTP_DRAFTS), dtype=torch.int32
            ).to("tpu")
            self.resume = torch.zeros((1, self.num_slots), dtype=torch.int32).to("tpu")

    def bind_kv_caches(
        self, caches: Mapping[str, torch.Tensor], groups: Sequence[KVGroup]
    ) -> None:
        by_layer = {_layer(name): (name, _raw(cache)) for name, cache in caches.items()}
        if self.mtp_checkpoint is not None:
            if MTP_LAYER not in by_layer:
                raise RuntimeError("Qwen3.8 decode: no KV cache for the MTP layer")
            self.mtp_raw = by_layer[MTP_LAYER][1]
        missing = [
            i for i in (*self.attention_layers, *self.gdn_layers) if i not in by_layer
        ]
        if missing:
            raise RuntimeError(f"Qwen3.8 decode: no KV cache for layers {missing[:4]}")
        self.raws = [by_layer[i][1] for i in self.attention_layers]
        index_of = {raw.data_ptr(): o for o, raw in enumerate(self.raws)}
        if len(index_of) != len(self.raws):
            raise RuntimeError("Qwen3.8 decode: attention layers share a pool")
        gdn_pools = []
        for i in self.gdn_layers:
            pointer = by_layer[i][1].data_ptr()
            if pointer not in index_of:
                raise RuntimeError(
                    f"Qwen3.8 decode: GDN layer {i} is not in a unified pool "
                    "(TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL)"
                )
            gdn_pools.append(index_of[pointer])
        self.gdn_pools = tuple(gdn_pools)
        shape = tuple(self.raws[0].shape)
        head_dim = self.layout.config.head_dim
        if (
            shape[1:] != (2, head_dim // 4, 4, self.page)
            or self.raws[0].dtype != torch.float8_e4m3fn
        ):
            raise RuntimeError(
                f"Qwen3.8 decode: attention pools {shape} {self.raws[0].dtype}, not "
                f"FP8 HND [pages, 2, {head_dim // 4}, 4, {self.page}] "
                "(VLLM_KV_CACHE_LAYOUT=HND, one KV head per rank)"
            )
        of = {name: g for g, group in enumerate(groups) for name in group.layer_names}
        attention = {of[by_layer[i][0]] for i in self.attention_layers}
        if len(attention) != 1:
            raise RuntimeError(
                f"Qwen3.8 decode: attention layers in KV groups {sorted(attention)}"
            )
        self._attention_group = attention.pop()
        self._gdn_groups = [
            (of[by_layer[i][0]], int(groups[of[by_layer[i][0]]].block_size))
            for i in self.gdn_layers
        ]
        self._width = int(groups[self._attention_group].table_width)
        if self.rank == 0:
            logger.info(
                "Qwen3.8 decode: %d unified pools (%d pages), batches %s, table width "
                "%d; HBM %s",
                len(self.raws),
                shape[0],
                self.batches,
                self._width,
                bridge.hbm(self.rank),
            )

    def warmup(self) -> None:
        started = time.perf_counter()
        self._import([(0, np.zeros(len(self.gdn_layers), np.int32))])
        for batch in self.batches:
            if self.mtp_checkpoint is not None:
                self._run_mtp(batch, self._plan(batch, [], {}))
            else:
                self._run(batch, self._plan(batch, [], {}))
        bridge.materialize(self.tokens)
        if self.rank == 0:
            logger.info(
                "Qwen3.8 decode: ops warm in %.1f s", time.perf_counter() - started
            )

    # ------------------------------------------------------------ ops
    def _fields(self, batch: int) -> dict[str, tuple[int, ...]]:
        return dict(
            slots=(batch,),
            positions=(batch,),
            reset=(batch,),
            host_tokens=(batch,),
            scheduled=(batch,),
            row_of=(batch,),
            tables=(batch, self._width),
            temperature=(batch,),
            top_p=(batch,),
            top_k=(batch,),
            seed=(batch,),
        )

    def _step_op(self, batch: int):
        op = self._ops.get(batch)
        if op is not None:
            return op
        layout, config = self.layout, self.layout.config
        names = tuple(self.weights)
        fields = self._fields(batch)
        rows, shard = self.config.max_num_seqs, layout.vocab_shard
        nw, npools = len(names), len(self.raws)

        def body(*args):
            w = {name: a[0] for name, a in zip(names, args[:nw], strict=True)}
            conv, rec, bank = args[nw][0], args[nw + 1][0], args[nw + 2][0]
            raws = tuple(args[nw + 3 : nw + 3 + npools])
            plan = _unpack(fields, args[-1])
            slots, positions = plan["slots"], plan["positions"]
            tokens = jnp.where(plan["reset"] != 0, plan["host_tokens"], bank[slots])
            states = {"conv": conv[:, slots], "rec": rec[:, slots]}
            _, logits, states, raws = decode_megakernel.paged_decode_local(
                config,
                layout,
                w,
                states,
                raws,
                plan["tables"],
                tokens,
                positions,
                pages_per_block=1,
                kv_scales=((1.0, 1.0),) * len(self.attention_layers),
            )
            logits = logits[:, :shard]
            temperature = lax.bitcast_convert_type(plan["temperature"], jnp.float32)
            top_p = lax.bitcast_convert_type(plan["top_p"], jnp.float32)
            greedy = sampling.greedy(logits)
            sampled = sampling.sample_rows(
                logits,
                jnp.where(temperature > 0, temperature, 1.0),
                top_p,
                plan["top_k"],
                plan["seed"],
                positions + 1,
            )
            chosen = jnp.where(temperature > 0, sampled, greedy)
            conv = conv.at[:, slots].set(states["conv"])
            rec = rec.at[:, slots].set(states["rec"])
            bank = bank.at[slots].set(chosen)
            row_of = plan["row_of"]
            out = (
                jnp.full((rows,), -1, jnp.int32)
                .at[jnp.where(row_of >= 0, row_of, rows)]
                .set(chosen, mode="drop")
            )
            return (out, conv[None], rec[None], bank[None], *raws)

        t = P("tp")
        op = bridge.register_op(
            f"pallas::qwen38_decode_b{batch}",
            self.mesh,
            body,
            in_specs=(t,) * (nw + 3 + npools) + (P(),),
            out_specs=(P(),) + (t,) * (3 + npools),
            donate_argnums=tuple(range(nw, nw + 3 + npools)),
        )
        self._ops[batch] = (op, fields)
        return self._ops[batch]

    def _run(self, batch: int, plan: dict[str, np.ndarray]) -> torch.Tensor:
        op, fields = self._step_op(batch)
        out, conv, rec, bank, *raws = op(
            *self.weights.values(),
            self.conv,
            self.rec,
            self.tokens,
            *self.raws,
            torch.from_numpy(_pack(fields, plan)).to("tpu"),
        )
        self.conv, self.rec = _adopt(self.conv, conv), _adopt(self.rec, rec)
        self.tokens = _adopt(self.tokens, bank)
        self.raws = [_adopt(old, new) for old, new in zip(self.raws, raws, strict=True)]
        return out

    def _verify_op(self, batch: int):
        """The MTP step's first op (Torch ops take at most 64 operands): verify
        every request's pending token and drafts, decide the accepted prefix,
        commit its KV and GDN state. Returns the emitted tokens, and each rank's
        copy of the target's hidden states and the decisions for
        :meth:`_draft_op`."""
        key = ("verify", batch)
        op = self._ops.get(key)
        if op is not None:
            return op
        from vllm_torchtpu.megakernel.qwen38.kernels.paged_kv import commit_columns

        layout, config = self.layout, self.layout.config
        names = tuple(self.weights)
        fields = self._fields(batch)
        rows, shard = self.config.max_num_seqs, layout.vocab_shard
        nw, npools = len(names), len(self.raws)
        scales = ((1.0, 1.0),) * len(self.attention_layers)
        width = 1 + MTP_DRAFTS

        def body(*args):
            w = {n: a[0] for n, a in zip(names, args[:nw], strict=True)}
            conv, rec, bank, drafts, resume = (args[nw + i][0] for i in range(5))
            raws = tuple(args[nw + 5 : nw + 5 + npools])
            plan = _unpack(fields, args[-1])
            slots, tables = plan["slots"], plan["tables"]
            live = plan["positions"] >= 0
            fresh = plan["reset"] != 0
            pending = jnp.where(fresh, plan["host_tokens"], bank[slots])
            resumed = ~fresh & (resume[slots] > 0)
            positions = jnp.where(
                live, jnp.where(resumed, resume[slots] - 1, plan["positions"]), -1
            )
            # Drafts are verified only on a request's later steps whose four
            # rows the scheduler scheduled; otherwise the count is forced to zero.
            speculating = live & resumed & (plan["scheduled"] == width)
            proposals = drafts[slots]  # [batch, drafts]
            slot = jnp.arange(width)
            inputs = jnp.concatenate(
                [pending[:, None], jnp.where(speculating[:, None], proposals, 0)],
                axis=1,
            )
            rows_at = jnp.where(live[:, None], positions[:, None] + slot[None], -1)
            active = live[:, None] & ((slot[None] == 0) | speculating[:, None])
            _, logits, states, hidden, cuts = decode_megakernel._stack(
                config,
                layout,
                0,
                width * batch,
                w,
                {"conv": conv[:, slots], "rec": rec[:, slots]},
                inputs.reshape(-1),
                rows_at.reshape(-1),
                False,
                kv_scales=scales,
                return_hidden=True,
                causal_pair=True,
                _mtp_requests=batch,
                _mtp_active=active.reshape(-1),
                _mtp_separate_cuts=True,
                paged={"pools": raws, "tables": tables, "pages_per_block": 1},
            )
            columns = states.pop("columns")
            decision = sampling.verify_acceptance(
                logits[:, :shard],
                inputs,
                jnp.maximum(positions, 0),
                speculating.astype(jnp.int32),
                (
                    lax.bitcast_convert_type(plan["temperature"], jnp.float32),
                    lax.bitcast_convert_type(plan["top_p"], jnp.float32),
                    plan["top_k"],
                    plan["seed"],
                ),
            )
            count, bonus = decision[:, 0], decision[:, 1]
            committed = count + 1  # inputs whose KV and GDN updates stand
            emitted = jnp.where(
                slot[None] < count[:, None],
                jnp.concatenate([proposals, jnp.zeros((batch, 1), jnp.int32)], axis=1),
                jnp.where(slot[None] == count[:, None], bonus[:, None], -1),
            )
            emitted = jnp.where(live[:, None], emitted, -1)
            kept = jnp.where(slot[None] < committed[:, None], rows_at, -1)
            raws = commit_columns(
                raws,
                columns,
                kept.reshape(-1),
                jnp.repeat(tables, width, axis=0),
                pages_per_block=1,
            )
            last = committed - 1

            def after(name):
                # The GDN state after the last committed row: a cut, or the
                # state after all four.
                chosen = states[name]
                for index, cut in enumerate(cuts[name]):
                    shape = (1, batch) + (1,) * (chosen.ndim - 2)
                    chosen = jnp.where((last == index).reshape(shape), cut, chosen)
                return chosen

            conv = conv.at[:, slots].set(after("conv"))
            rec = rec.at[:, slots].set(after("rec"))
            bank = bank.at[slots].set(bonus)
            dropped = jnp.where(live, slots, resume.shape[0])
            resume = resume.at[dropped].set(positions + committed + 1, mode="drop")
            row_of = plan["row_of"]
            out = (
                jnp.full((rows, width), -1, jnp.int32)
                .at[jnp.where(row_of >= 0, row_of, rows)]
                .set(emitted, mode="drop")
            )
            return (
                out,
                conv[None],
                rec[None],
                bank[None],
                resume[None],
                hidden.reshape(batch, width, -1)[None],
                jnp.stack([jnp.where(live, count, -1), positions])[None],
                *raws,
            )

        t = P("tp")
        op = bridge.register_op(
            f"pallas::qwen38_decode_verify_b{batch}",
            self.mesh,
            body,
            in_specs=(t,) * (nw + 5 + npools) + (P(),),
            out_specs=(P(),) + (t,) * (6 + npools),
            donate_argnums=tuple(range(nw, nw + 5 + npools)),
        )
        self._ops[key] = (op, fields)
        return self._ops[key]

    def _draft_op(self, batch: int):
        """The MTP step's second op: per request, repair the accepted drafts'
        MTP KV rows from the target's hidden states; then propose the next
        step's three drafts (one draft call per draft for all requests)."""
        key = ("draft", batch)
        op = self._ops.get(key)
        if op is not None:
            return op
        from vllm_torchtpu.megakernel.qwen38.kernels import (
            mtp_cache_batch,
            mtp_decode,
            mtp_decode_c2_fused,
        )
        from vllm_torchtpu.megakernel.qwen38.kernels.mtp_config import MTPConfig
        from vllm_torchtpu.megakernel.qwen38.kernels.paged_kv import commit_columns

        assert self.mtp_checkpoint is not None
        layout = self.layout
        mtp_config = MTPConfig.from_checkpoint(self.mtp_checkpoint)
        names = tuple(self.mtp_weights)
        fields = self._fields(batch)
        nm = len(names)

        def body(embedding, head, *args):
            embedding, head = embedding[0], head[0]
            mw = {n: a[0] for n, a in zip(names, args[:nm], strict=True)}
            bank, drafts, hidden, decisions = (args[nm + i][0] for i in range(4))
            mtp_pool = args[nm + 4]
            plan = _unpack(fields, args[-1])
            slots, tables = plan["slots"], plan["tables"]
            counts, positions = decisions[0], decisions[1]
            target = {"emb": embedding, "lmw": head}
            repair = {f: mw[f] for f in mtp_cache_batch.FAMILIES}

            def repair_request(b, mtp_pool):
                table = lax.dynamic_slice_in_dim(tables, b, 1)
                states = lax.dynamic_index_in_dim(hidden, b, keepdims=False)
                position = jnp.maximum(
                    lax.dynamic_index_in_dim(positions, b, keepdims=False), 0
                )
                accepted = lax.dynamic_index_in_dim(counts, b, keepdims=False)
                # Accepted draft j (1..count) at row p + j - 1, from the target's
                # hidden after the row before it.
                repaired = jnp.arange(MTP_DRAFTS) < accepted
                repaired_rows = position + jnp.arange(MTP_DRAFTS)
                kv = mtp_cache_batch._cache_batch_stack(
                    mtp_config,
                    layout,
                    0,
                    embedding,
                    repair,
                    None,
                    lax.dynamic_index_in_dim(drafts[slots], b, keepdims=False),
                    repaired_rows,
                    states[:MTP_DRAFTS],
                    repaired,
                    False,
                    paged=True,
                )["columns"]
                (mtp_pool,) = commit_columns(
                    (mtp_pool,),
                    kv,
                    jnp.where(repaired, repaired_rows, -1),
                    jnp.repeat(table, MTP_DRAFTS, axis=0),
                    pages_per_block=1,
                )
                return mtp_pool

            mtp_pool = lax.fori_loop(0, batch, repair_request, mtp_pool)
            # Drafts: from the target's hidden after the last accepted row and
            # the pending token, at the rows after the repaired ones.
            live = positions >= 0
            accepted = jnp.maximum(counts, 0)
            token = bank[slots]
            rows = jnp.where(live, positions + accepted, -1)
            state = jnp.take_along_axis(hidden, accepted[:, None, None], axis=1)[:, 0]
            pool = {"tables": tables, "pages_per_block": 1}
            proposed = []
            for _ in range(MTP_DRAFTS):
                if batch == 1:
                    token, _, written, state = mtp_decode._native_stack(
                        mtp_config,
                        layout,
                        0,
                        target,
                        mw,
                        None,
                        token,
                        jnp.maximum(rows, 0),
                        state,
                        False,
                        paged={"pool": mtp_pool, **pool},
                    )
                else:
                    token, _, written, state = mtp_decode_c2_fused._fused_stack(
                        mtp_config,
                        layout,
                        0,
                        target,
                        mw,
                        (),
                        token,
                        rows,
                        state,
                        False,
                        paged={"pool": mtp_pool, **pool},
                    )
                assert isinstance(written, dict)  # the paged calls' columns
                (mtp_pool,) = commit_columns(
                    (mtp_pool,), written["columns"], rows, tables, pages_per_block=1
                )
                proposed.append(token)
                rows = jnp.where(live, rows + 1, -1)
            new_drafts = jnp.stack(proposed, axis=1)
            drafts = drafts.at[jnp.where(live, slots, drafts.shape[0])].set(
                new_drafts, mode="drop"
            )
            return drafts[None], mtp_pool

        t = P("tp")
        op = bridge.register_op(
            f"pallas::qwen38_decode_draft_b{batch}",
            self.mesh,
            body,
            in_specs=(t,) * (2 + nm + 5) + (P(),),
            out_specs=(t, t),
            donate_argnums=(2 + nm + 1, 2 + nm + 4),
        )
        self._ops[key] = (op, fields)
        return self._ops[key]

    def _run_mtp(self, batch: int, plan: dict[str, np.ndarray]) -> torch.Tensor:
        assert self.mtp_raw is not None
        verify, fields = self._verify_op(batch)
        draft, _ = self._draft_op(batch)
        vector = torch.from_numpy(_pack(fields, plan)).to("tpu")
        out, conv, rec, bank, resume, hidden, decisions, *raws = verify(
            *self.weights.values(),
            self.conv,
            self.rec,
            self.tokens,
            self.drafts,
            self.resume,
            *self.raws,
            vector,
        )
        self.conv, self.rec = _adopt(self.conv, conv), _adopt(self.rec, rec)
        self.tokens = _adopt(self.tokens, bank)
        self.resume = _adopt(self.resume, resume)
        self.raws = [_adopt(old, new) for old, new in zip(self.raws, raws, strict=True)]
        drafts, mtp_raw = draft(
            self.weights["emb"],
            self.weights["lmw"],
            *self.mtp_weights.values(),
            self.tokens,
            self.drafts,
            hidden,
            decisions,
            self.mtp_raw,
            vector,
        )
        self.drafts = _adopt(self.drafts, drafts)
        self.mtp_raw = _adopt(self.mtp_raw, mtp_raw)
        return out

    def _import(self, imports: Sequence[tuple[int, np.ndarray]]) -> None:
        """Slots from each request's GDN state blocks (where the transfer put them)."""
        op = self._ops.get("import")
        npools, gdn_pools = len(self.raws), self.gdn_pools
        c, layout = self.layout.config, self.layout
        pages_per_block = self.config.block_size // self.page
        if op is None:

            def body(conv, rec, *args):
                pools, slots, blocks = args[:npools], args[npools], args[npools + 1]
                read_conv, read_rec = pool_state.read_states(
                    tuple(pool_state.logical_blocks(p, pages_per_block) for p in pools),
                    gdn_pools,
                    blocks,
                    v_heads=layout.local_v_heads,
                    k_dim=c.k_dim,
                    v_dim=c.v_dim,
                    taps=c.conv_size,
                    conv=layout.local_conv,
                )
                # The null block holds no state: a request without one starts clean.
                clean = (jnp.max(blocks, axis=1) == 0)[None, :, None, None]
                conv = (
                    conv[0]
                    .at[:, slots]
                    .set(jnp.where(clean, 0, jnp.swapaxes(read_conv, 0, 1)))
                )
                rec = (
                    rec[0]
                    .at[:, slots]
                    .set(jnp.where(clean[..., None], 0, jnp.swapaxes(read_rec, 0, 1)))
                )
                return conv[None], rec[None]

            t = P("tp")
            op = bridge.register_op(
                "pallas::qwen38_decode_import",
                self.mesh,
                body,
                in_specs=(t, t) + (t,) * npools + (P(), P()),
                out_specs=(t, t),
                donate_argnums=(0, 1),
            )
            self._ops["import"] = op
        for start in range(0, len(imports), IMPORT_ROWS):
            chunk = list(imports[start : start + IMPORT_ROWS])
            chunk += [chunk[-1]] * (IMPORT_ROWS - len(chunk))
            slots = torch.from_numpy(np.asarray([s for s, _ in chunk], np.int32)).to(
                "tpu"
            )
            blocks = torch.from_numpy(
                np.stack([b for _, b in chunk]).astype(np.int32)
            ).to("tpu")
            conv, rec = op(self.conv, self.rec, *self.raws, slots, blocks)
            self.conv, self.rec = _adopt(self.conv, conv), _adopt(self.rec, rec)

    # ------------------------------------------------------------ steps
    def step(self, step_input: StepInput) -> StepOutput:
        if step_input.copies:
            raise NotImplementedError("Qwen3.8 decode: copy-on-write block copies")
        for req_id in step_input.finished:
            self._requests.pop(req_id, None)
            self.slots.release([req_id])
        rows = list(step_input.requests)
        live, imports, reset_tokens = [], [], {}
        for i, row in enumerate(rows):
            if row.scheduled <= 0:
                continue
            if row.scheduled != 1 and not (
                self.mtp_checkpoint is not None and row.scheduled == 1 + MTP_DRAFTS
            ):
                raise NotImplementedError(
                    f"Qwen3.8 decode: {row.scheduled} tokens in one step"
                )
            request = self._requests.get(row.req_id)
            fresh = request is None
            if fresh:
                slot = self.slots.allocate(row.req_id)
                request = self._requests[row.req_id] = _Request(slot, row.computed)
                imports.append((slot, self._state_blocks(row)))
            if fresh or row.computed < row.num_prompt_tokens:
                # The host knows the input (a request's first token, or a prompt
                # decoded token by token); otherwise it is the previous step's
                # output, which under async scheduling the host row may not hold.
                reset_tokens[i] = int(row.token_ids[row.computed])
            if request.position != row.computed and self.mtp_checkpoint is None:
                raise RuntimeError(
                    f"Qwen3.8 decode: request {row.req_id} resumes at "
                    f"{row.computed}, not {request.position}"
                )
            # (With MTP vLLM advances a request by what the op emitted.)
            request.position = row.computed + 1
            live.append((i, row))
        if imports:
            self._import(imports)
        out = None
        for start in range(0, len(live), self.batches[-1]):
            group = live[start : start + self.batches[-1]]
            batch = next(b for b in self.batches if b >= len(group))
            plan = self._plan(batch, group, reset_tokens)
            if self.mtp_checkpoint is not None:
                chosen = self._run_mtp(batch, plan)
            else:
                chosen = self._run(batch, plan).reshape(-1, 1)
            out = chosen if out is None else torch.maximum(out, chosen)
        if out is None:
            width = 1 + (MTP_DRAFTS if self.mtp_checkpoint is not None else 0)
            out = torch.full(
                (self.config.max_num_seqs, width), -1, dtype=torch.int32
            ).to("tpu")
        return StepOutput(out[: max(1, len(rows))])

    def _state_blocks(self, row: RequestInput) -> np.ndarray:
        """Each GDN layer's state block after the request's computed tokens: the
        block holding position ``computed - 1`` (vLLM's align mode); the null
        block for an empty request."""
        if row.computed <= 0:
            return np.zeros(len(self._gdn_groups), np.int32)
        return np.array(
            [
                row.block_tables[g][
                    min((row.computed - 1) // size, len(row.block_tables[g]) - 1)
                ]
                for g, size in self._gdn_groups
            ],
            np.int32,
        )

    def _plan(
        self,
        batch: int,
        group: list[tuple[int, RequestInput]],
        reset_tokens: dict[int, int],
    ) -> dict[str, np.ndarray]:
        plan = {
            name: np.zeros(shape, np.int32)
            for name, shape in self._fields(batch).items()
        }
        plan["positions"][:] = -1
        plan["row_of"][:] = -1
        plan["temperature"] = np.zeros(batch, np.float32)
        plan["top_p"] = np.ones(batch, np.float32)
        for b, (i, row) in enumerate(group):
            request = self._requests[row.req_id]
            plan["slots"][b] = request.slot
            plan["positions"][b] = row.computed
            plan["row_of"][b] = i
            plan["scheduled"][b] = row.scheduled
            if i in reset_tokens:
                plan["reset"][b], plan["host_tokens"][b] = 1, reset_tokens[i]
            table = np.asarray(row.block_tables[self._attention_group], np.int32)[
                : self._width
            ]
            plan["tables"][b, : len(table)] = table
            plan["temperature"][b] = row.sampling.temperature
            plan["top_p"][b] = row.sampling.top_p
            plan["top_k"][b] = row.sampling.top_k
            plan["seed"][b] = row.sampling.seed & 0x7FFFFFFF
        return plan


def _adopt(old: torch.Tensor, fresh: torch.Tensor) -> torch.Tensor:
    old.set_(fresh)
    return old


def _pack(
    fields: Mapping[str, tuple[int, ...]], plan: Mapping[str, np.ndarray]
) -> np.ndarray:
    """A step's plan as one int32 vector (float fields by their bits)."""
    return np.concatenate(
        [
            np.asarray(plan[name]).view(np.int32)
            if np.asarray(plan[name]).dtype == np.float32
            else np.asarray(plan[name], np.int32).reshape(-1)
            for name in fields
        ]
    ).astype(np.int32)


def _unpack(fields: Mapping[str, tuple[int, ...]], vector):
    """:func:`_pack`'s vector as the plan's fields (inside an op)."""
    out, offset = {}, 0
    for name, shape in fields.items():
        size = int(np.prod(shape))
        out[name] = vector[offset : offset + size].reshape(shape)
        offset += size
    return out
