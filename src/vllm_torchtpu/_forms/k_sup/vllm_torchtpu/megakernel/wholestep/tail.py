# Copyright 2026 Google LLC
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
"""The decode tail: final hyper-connection mixer, `lm_head`, greedy top token per rank.

Stock (`qwen4_exp/model.py:581, 673`, `runner/greedy_top_tokens.py`): `y = HC mixer(x)` (no
combine, no final RMSNorm, no softcap), local logits `y @ lm_head_r` over this rank's 31 040-entry
vocab shard, per-rank `(max, argmax + rank * shard)`, then an all-gather and an argmax over ranks
(ties to the lower rank). The kernel emits the per-rank pair (`tail_top [M, 128]`, lane 0 = max,
lane 1 = global index); the 8-way argmax over ranks is a tiny XLA op after the call.

The mixer is `HCSublayer._mix` (the same weights format as every HC sublayer; its injection columns
are unused here), so the block is an HC sublayer whose inner block returns no partial.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..blocks.hc import HCSublayer
from ..exec_spec import ExecSpec
from ..models.qwen38 import QWEN38_SEGMENT, Qwen38Geometry, Qwen38ResidualGeometry
from ..ops import BF16, F32, I32
from ..spec import Block, LayerType, ModelSpec, ScratchReq, StateDecl, WeightDecl

TOP = "tail_top"
LM_HEAD = "lm_head"


class GreedyHead(Block):
    """`lm_head` on the mixed hidden + this rank's top token; returns no partial."""

    name = "tail"
    reduction = "none"
    input_norm = None

    def __init__(self, *, vocab_shard: int) -> None:
        self.vocab_shard = vocab_shard

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        return [WeightDecl(LM_HEAD, K=g.H, N=self.vocab_shard, dtype=BF16)]

    def outputs(self, g: Any, ex: ExecSpec) -> list[StateDecl]:
        return [StateDecl(TOP, (ex.M, 128), F32, "w", per_layer=False)]

    def host_update_states(self, g, ex, pos, states, raw_outputs):  # noqa: ANN001
        return {TOP: raw_outputs[TOP]}

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        return [
            ScratchReq("tl_top", pltpu.VMEM((ex.MB, 128), F32)),
            ScratchReq("tl_sem", pltpu.SemaphoreType.DMA(())),
        ]

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex, scr = ctx.ex, ctx.scr
        with ctx.stream.segment(lazy=False):
            logits = ctx.stream.dmatmul(LM_HEAD, scr["xb"], self.vocab_shard)  # [MB, V] f32
        m = jnp.max(logits, axis=-1, keepdims=True)
        col = jax.lax.broadcasted_iota(I32, logits.shape, 1)
        idx = jnp.min(jnp.where(logits == m, col, self.vocab_shard), axis=-1, keepdims=True)
        rank = ctx.me
        gidx = (idx + rank * self.vocab_shard).astype(F32)  # < 2^24: exact in f32
        lane = jax.lax.broadcasted_iota(I32, (ex.MB, 128), 1)
        scr["tl_top"][...] = jnp.where(lane == 0, m, jnp.where(lane == 1, gidx, 0.0))
        cp = pltpu.make_async_copy(scr["tl_top"], ctx.O[TOP].at[pl.ds(0, ex.MB)], scr["tl_sem"])
        cp.start()
        cp.wait()
        return None


def tail_spec(g: Qwen38Geometry, R: int, *, vocab_shard: int) -> ModelSpec:
    """One-layer kernel: the final HC mixer (no combine) feeding `GreedyHead`."""
    gi = g.inner(R, 1)
    head = HCSublayer(
        GreedyHead(vocab_shard=vocab_shard), inner_geometry=gi, nhc=g.NHC, lowrank=g.LOWRANK,
        allreduce=False,
    )
    return ModelSpec(
        name="qwen38_tail",
        config=Qwen38ResidualGeometry(L=1, H=g.NHC * g.H, eps=g.eps),
        layer_type=LayerType(name=QWEN38_SEGMENT, slots=(head,)),
    )


def greedy_tokens(top: jax.Array, R: int, M: int) -> jax.Array:
    """`[R*M, 128]` per-rank pairs -> `[M]` int32 tokens: argmax over ranks, ties to lower rank."""
    t = top.reshape(R, M, 128)
    best = jnp.argmax(t[:, :, 0], axis=0)  # first maximal rank
    return jnp.take_along_axis(t[:, :, 1], best[None], axis=0)[0].astype(I32)


def tail_reference(x: jax.Array, p: dict[str, Any], lm_head: jax.Array, *, nhc: int,
                   eps: float) -> tuple[jax.Array, jax.Array]:
    """`(tokens [B], logits [B, R*V])` of the stock tail; `lm_head [R, H, V]` per rank."""
    from ..blocks.hc.reference import hc_mix

    mixed, _ = hc_mix(x, p, nhc=nhc, eps=eps)
    logits = jnp.concatenate(
        [jnp.dot(mixed.astype(BF16), lm_head[r], preferred_element_type=F32)
         for r in range(lm_head.shape[0])], axis=-1)
    return jnp.argmax(logits, axis=-1).astype(I32), logits
