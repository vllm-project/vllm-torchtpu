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
"""`qwen35-baseline`: the default execution profile of the Qwen3.5 model.

Per-bucket tiles bit-exact with the hand-written Qwen3.5 megakernel: its ring
depths, reduce-scatter / hierarchical all-reduce modes and VMEM budget per
batch bucket, with every opt-in performance flag off, split pool I/O and no
in-kernel gather. The Qwen3.5 model declares it as its default profile
(`model_default=True`), so adapters that name no profile run Qwen3.5 with it,
and Qwen3.5 profiles' numerics classes are measured against it.
"""

from __future__ import annotations

import dataclasses

from ..blocks.gqa import GQABlock, GQATunables
from ..blocks.moe import MoEBlock
from ..exec_spec import Tiles
from ..profile import Profile, register_profile

# `ModelSpec.name` of the Qwen3.5 model (`qwen35.qwen35_spec`).
QWEN35_MODEL_NAME = "qwen3_5"
_GQA, _MOE = GQABlock.name, MoEBlock.name

# Decode (`B <= 4`), verify (`8 < B <= 16`) and wide (`B > 16`) tiles.
DECODE_TILES = Tiles(hier=1, mb8=2, blocks={_GQA: GQATunables(pack=1)})
VERIFY_TILES = dataclasses.replace(DECODE_TILES, rsag=True, hier=2).with_blocks(
    {_GQA: {"nkv": 4}, _MOE: {"ecopy_opt": 1}}
)
WIDE_TILES = dataclasses.replace(VERIFY_TILES, nbd=10, vmem_mib=62, hier=0).with_blocks(
    {_GQA: {"nkv": 3}}
)
# Small banks and rings for CPU interpret runs of tiny geometries.
TINY_TILES = Tiles(
    bank=128 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32
).with_blocks({_GQA: {"kv_blk": 256, "nkv": 3}})


def baseline_tiles(B: int, *, moe_tp: bool = False, R: int = 8) -> Tiles:
    """Qwen3.5 baseline tiles for batch row count `B`.

    Selects the per-bucket ring depths, all-reduce mode, and VMEM budget (using
    `nbe=6` for `B > 48` so 64-row MoE scratch fits VMEM). Every opt-in
    performance flag stays at its `Tiles` default (off). `moe_tp` is accepted
    for the tile-rule signature and ignored. Hierarchical all-reduce (`hier`
    1/2) is v7x-8 specific, so `R` outside `{1, 8}` falls back to the flat ring.
    """
    del moe_tp
    if B > 48:
        t = dataclasses.replace(WIDE_TILES, nbe=6)
    elif B > 32:
        t = WIDE_TILES
    elif B > 24:
        t = dataclasses.replace(WIDE_TILES, hier=2)
    elif B > 16:
        t = dataclasses.replace(WIDE_TILES, nbd=12, hier=2)
    elif B > 8:
        t = VERIFY_TILES
    elif B > 4:
        t = dataclasses.replace(DECODE_TILES, rsag=True, hier=2).with_blocks(
            {_MOE: {"ecopy_opt": 1}}
        )
    else:
        t = DECODE_TILES
    if R not in (1, 8) and t.hier != 0:
        t = dataclasses.replace(t, hier=0)
    return t


PROFILE = register_profile(
    Profile(
        name="qwen35-baseline",
        model=QWEN35_MODEL_NAME,
        numerics="exact",
        model_default=True,
        tiles=baseline_tiles,
        description=(
            "Qwen3.5 default: per-bucket tiles bit-exact with the hand-written "
            "Qwen3.5 kernel, split pool I/O, no in-kernel gather, all opt-in "
            "performance flags off."
        ),
    )
)

__all__ = [
    "DECODE_TILES",
    "PROFILE",
    "QWEN35_MODEL_NAME",
    "TINY_TILES",
    "VERIFY_TILES",
    "WIDE_TILES",
    "baseline_tiles",
]
