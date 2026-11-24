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
"""Per-trace emit state shared by the `GQABlock` emit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jax import lax

from ...pool.config import cache_exec
from .params import GQAParams


@dataclass(frozen=True)
class GQAEmit:
    """References and per-layer values one `GQABlock.emit` trace reads.

    `gi` is this rank's index in its KV-head group, `a` the layer's index into
    the block's stacked states, `n_items` the KV worklist length and `KR` the
    KV ring depth.
    """

    ctx: Any
    ex: Any
    ap: GQAParams
    scr: Any
    ST: Any
    O: Any  # noqa: E741
    pos_ref: Any
    skip: Any
    sc: Any
    stream: Any
    comm: Any
    interpret: bool
    gi: Any
    a: Any
    ksem: Any
    am: Any
    al: Any
    aacc: Any
    n_items: Any
    KR: int

    @classmethod
    def build(cls, ctx: Any, ap: GQAParams, l: Any, name: str) -> GQAEmit:  # noqa: E741
        """Collect `ctx`'s refs and trace the per-layer scalars (in emit order)."""
        ex = ctx.ex
        scr, stream = ctx.scr, ctx.stream
        gi = lax.rem(ctx.me, ap.G) if ap.SPLIT else jnp.int32(0)
        a = stream.sp.block_ord(l, name)
        n_items = scr["wl_cnt"][0]
        return cls(
            ctx=ctx,
            ex=ex,
            ap=ap,
            scr=scr,
            ST=ctx.ST,
            O=ctx.O,
            pos_ref=ctx.pos_ref,
            skip=ctx.skip,
            sc=ctx.sc,
            stream=stream,
            comm=ctx.comm,
            interpret=bool(ex.interpret),
            gi=gi,
            a=a,
            ksem=scr["kv_sem"],
            am=scr["am"],
            al=scr["al"],
            aacc=scr["aacc"],
            n_items=n_items,
            KR=ap.NKVR if cache_exec(ex).paged_kv else ap.NKV,
        )
