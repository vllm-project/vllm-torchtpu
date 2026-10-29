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

logger = init_logger(__name__)

TP_SIZE = 32
BATCHES = (1, 2, 4, 8, 16)
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
        self, config: MegakernelEngineConfig, presharded: Path, *, tp_rank: int
    ) -> None:
        self.config = config
        self.presharded = Path(presharded)
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

    def bind_kv_caches(
        self, caches: Mapping[str, torch.Tensor], groups: Sequence[KVGroup]
    ) -> None:
        by_layer = {_layer(name): (name, _raw(cache)) for name, cache in caches.items()}
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

        def unpack(vector):
            out, offset = {}, 0
            for name, shape in fields.items():
                size = int(np.prod(shape))
                out[name] = vector[offset : offset + size].reshape(shape)
                offset += size
            return out

        def body(*args):
            w = {name: a[0] for name, a in zip(names, args[:nw], strict=True)}
            conv, rec, bank = args[nw][0], args[nw + 1][0], args[nw + 2][0]
            raws = tuple(args[nw + 3 : nw + 3 + npools])
            plan = unpack(args[-1])
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
        vector = np.concatenate(
            [
                np.asarray(plan[name]).view(np.int32)
                if np.asarray(plan[name]).dtype == np.float32
                else np.asarray(plan[name], np.int32).reshape(-1)
                for name in fields
            ]
        ).astype(np.int32)
        out, conv, rec, bank, *raws = op(
            *self.weights.values(),
            self.conv,
            self.rec,
            self.tokens,
            *self.raws,
            torch.from_numpy(vector).to("tpu"),
        )
        self.conv, self.rec = _adopt(self.conv, conv), _adopt(self.rec, rec)
        self.tokens = _adopt(self.tokens, bank)
        self.raws = [_adopt(old, new) for old, new in zip(self.raws, raws, strict=True)]
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
            if row.scheduled != 1:
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
            if request.position != row.computed:
                raise RuntimeError(
                    f"Qwen3.8 decode: request {row.req_id} resumes at "
                    f"{row.computed}, not {request.position}"
                )
            request.position = row.computed + 1
            live.append((i, row))
        if imports:
            self._import(imports)
        out = None
        for start in range(0, len(live), self.batches[-1]):
            group = live[start : start + self.batches[-1]]
            batch = next(b for b in self.batches if b >= len(group))
            chosen = self._run(batch, self._plan(batch, group, reset_tokens))
            out = chosen if out is None else torch.maximum(out, chosen)
        if out is None:
            out = torch.full((self.config.max_num_seqs,), -1, dtype=torch.int32).to(
                "tpu"
            )
        return StepOutput(out[: max(1, len(rows))].reshape(-1, 1))

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
