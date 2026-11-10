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
"""Redundant routed-expert copies for the Qwen3.8 whole-step megakernel.

`VLLM_TPU_MK_EP_REPLICAS=N`: every rank holds, per layer, `N` copies of experts
homed on other ranks (`qwen35.default_eplb_cores`: copy `k` of rank `r1` is slot
`(layer + k) % ER` of rank `(r1 + 1 + k % (R - 1)) % R`), and the kernel's Po2C
routing sends each hit expert with a copy to the less loaded of its two ranks. The
copies live in their own arrays (`e13x` / `e2x`), so the model's experts stay the
kernel's zero-copy `e13` / `e2` views.

With this placement each rank provides exactly `N` of its experts (copy `k` of
every rank comes from a distinct rank), so one all-gather of an `[N, ...]`
contribution per layer and weight gives every rank its copies.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from vllm_torchtpu import envs
from vllm_torchtpu.megakernel.models import qwen35

from .weight_collector import get_routed_experts


def copies_per_rank() -> int:
    return max(0, int(envs.VLLM_TPU_MK_EP_REPLICAS))


def with_expert_copies(g: Any, R: int) -> Any:
    """`g` with `copies_per_rank()` split expert copies per rank (unchanged if 0)."""
    n = copies_per_rank()
    if n <= 0 or R <= 1:
        return g
    return dataclasses.replace(g, e_slots=g.E // R + n, e_split=True)


def contribution_slots(layer: int, rank: int, R: int, ER: int, n: int) -> list[int]:
    """Own slots this rank provides for copy `k = 0..n-1` (to rank `rank - 1 - k % (R-1)`)."""
    del rank, R
    return [(layer + k) % ER for k in range(n)]


def copy_sources(rank: int, R: int, n: int) -> list[int]:
    """Source rank of this rank's copy `k` (its row in the all-gather is `src * n + k`)."""
    return [(rank + 1 + k % (R - 1)) % R for k in range(n)]


def gather_copies(
    mlps: list[Any], gi: Any, R: int, rank: int, all_gather: Any
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], list[torch.Tensor]]:
    """`(e13x, e2x, es_x)` per layer for this rank: `[n, H, 2I]` / `[n, I, H]` FP8
    copies and their `[n, 1, 2I + H]` f32 scales. `all_gather(t)` concatenates `t`
    over the TP ranks on dim 0 (rank order)."""
    er, n = gi.E // R, gi.e_slots - gi.E // R
    cores = qwen35.default_eplb_cores(gi, R)
    src = copy_sources(rank, R, n)
    e13x, e2x, esx = [], [], []
    for j, mlp in enumerate(mlps):
        ex_mod = get_routed_experts(mlp)
        own = torch.tensor(contribution_slots(j, rank, R, er, n), device=ex_mod.w13_weight.device)
        for k in range(n):  # the kernel's copy table names the same experts
            assert cores[j][rank][er + k] == src[k] * er + (j + k) % er, (j, k)
        rows = torch.tensor([s * n + k for k, s in enumerate(src)], device=own.device)

        def pick(t: torch.Tensor) -> torch.Tensor:
            flat = t.reshape(t.shape[0], -1)
            raw = flat.view(torch.uint8) if flat.dtype.itemsize == 1 else flat
            got = all_gather(raw.index_select(0, own).contiguous())
            got = got.index_select(0, rows)
            if flat.dtype.itemsize == 1:
                got = got.view(flat.dtype)
            return got.reshape(n, *t.shape[1:]).contiguous()

        e13x.append(pick(ex_mod.w13_weight.data))
        e2x.append(pick(ex_mod.w2_weight.data))
        s13 = getattr(ex_mod, "w13_weight_scale_inv", None)
        s13 = ex_mod.w13_weight_scale if s13 is None else s13
        s2 = getattr(ex_mod, "w2_weight_scale_inv", None)
        s2 = ex_mod.w2_weight_scale if s2 is None else s2
        es = torch.cat(
            [s13.data.float().reshape(er, -1), s2.data.float().reshape(er, -1)], dim=-1
        )
        esx.append(pick(es).unsqueeze(1))
    return tuple(e13x), tuple(e2x), esx


def ecopy_rows(gi: Any, R: int) -> torch.Tensor:
    """`[L, E]` f32 copy table of the kernel's `ecopy` smalls field."""
    import numpy as np

    return torch.from_numpy(np.asarray(qwen35.encode_eplb_ecopy(gi, R), np.float32))
