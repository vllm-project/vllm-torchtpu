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
"""`QSAOutModule`: the output projection of an attention computed outside the kernel.

The attention core of a sparse-attention layer (cache writes, indexer top-k,
sparse attention) runs between megakernel calls. Its per-rank output
`[M, nh * HD]` arrives as the HBM extra input `qsa_attn`; the block projects
it with the row-parallel `o_proj` and returns the `[MB, H]` partial.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...exec_spec import ExecSpec
from ...ops import BF16, F32
from ...spec import ExtraInput, Module, ScratchReq, WeightDecl

ATTN_INPUT = "qsa_attn"


class QSAOutModule(Module):
    """Row-parallel `o_proj` of an externally computed attention output."""

    name = "qsa_out"
    reduction = "allreduce"
    input_norm = None

    def __init__(self, *, prefix: str = "self_attn") -> None:
        self.prefix = prefix

    @staticmethod
    def k_dim(g: Any, ex: ExecSpec) -> int:
        """Per-rank attention output width `nh * HD`."""
        return g.NH // ex.R * g.HD

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        return [WeightDecl("o_proj", K=self.k_dim(g, ex), N=g.H, dtype=BF16)]

    def extra_inputs(self, g: Any, ex: ExecSpec) -> list[ExtraInput]:
        return [ExtraInput(ATTN_INPUT, "hbm", (ex.M, self.k_dim(g, ex)), BF16)]

    def shared_widths(self, g: Any, ex: ExecSpec) -> dict[str, int]:
        return {"xb2": self.k_dim(g, ex)}

    def scratch(self, g: Any, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        return [ScratchReq("qsa_out_sem", pltpu.SemaphoreType.DMA(()))]

    def emit(self, ctx: Any, l: Any, slot: Any) -> Any:  # noqa: E741
        ex, scr = ctx.ex, ctx.scr
        K = self.k_dim(ctx.g, ex)
        cp = pltpu.make_async_copy(
            ctx.extra[ATTN_INPUT].at[pl.ds(0, ex.MB)],
            scr["xb2"].at[:, pl.ds(0, K)],
            scr["qsa_out_sem"],
        )
        cp.start()
        cp.wait()
        with ctx.stream.segment(lazy=False):
            return ctx.stream.dmatmul("o_proj", scr["xb2"], ctx.g.H)

    def reference(
        self, params: Any, x: Any, state: Any = None, **kw: Any
    ) -> tuple[jax.Array, Any]:
        """`sum_r attn_r @ o_proj_r` for per-rank `params['attn'] [R, T, K]`."""
        y = jnp.einsum(
            "rtk,rkn->tn",
            params["attn"].astype(BF16).astype(F32),
            params["o_proj"].astype(BF16).astype(F32),
        )
        return y, None
