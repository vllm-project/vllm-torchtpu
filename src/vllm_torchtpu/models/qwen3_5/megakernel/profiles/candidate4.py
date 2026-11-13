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
"""`qwen35-candidate4`: the Qwen3.5 schedule of our PD submission's decode legs.

Select with `additional_config={"megakernel": {"options": {"profile":
"qwen35-candidate4"}}}`. It is `qwen35-candidate30`'s tile rule (the
schedule `candidate30_tiles` reproduces) plus what candidate4's kernel turned
on per bucket:

- the deepest dense ring that fits on fused-pool-I/O verify buckets of 32 to
  64 rows (`FUSED_WIDE_NBD`);
- `hx` on `hier=2` buckets from `HX_MIN_ROWS` rows;
- the `hier=1` cross-chip peers `(3, 5, 7)` on 8-rank meshes;
- the collective / ring schedule options (`arpipe` on fused verify buckets
  of the main kernel except `ARPIPE_OFF_B`, `arpipe_h1`, `ar0s`,
  `dwait_hoist`);
- the GQA deferred waits and RoPE lane swap, the GDN fused-path conv window,
  MXU delta rule (on `MXU_DR_BUCKETS`) and deferred z columns, and the MoE
  routing / shared-expert / refill schedule options.

Each option is requested here on every bucket; the module or the collective
that owns it engages it only where candidate4's own gate holds (tile counts,
EP vs TP, fused pool I/O, ...), so the requests are inert elsewhere.
`qwen35-candidate4-sh-at2` adds the shared expert after the first expert DMAs
(`moe.sh_at=2`) on buckets of at most `SH_AT_MAX_ROWS` rows (point #12).

Numerics: approx (as `qwen35-candidate30`). Verify buckets: 4 to 64 rows.
"""

from __future__ import annotations

import dataclasses
import functools

from vllm_torchtpu.megakernel.core.exec_spec import Tiles
from vllm_torchtpu.megakernel.core.modules.gdn import GDNModule
from vllm_torchtpu.megakernel.core.modules.gqa import GQAModule
from vllm_torchtpu.megakernel.core.modules.moe import MoEModule
from vllm_torchtpu.megakernel.core.profile import TileContext

from .candidate30 import PROFILE as CANDIDATE30_PROFILE
from .candidate30 import candidate30_tiles

_GQA, _GDN, _MOE = GQAModule.name, GDNModule.name, MoEModule.name

# Dense ring depth of fused verify buckets (the deepest ring that fits the
# 62 MiB scoped VMEM limit on v7x-8 at DCP 4); buckets above 64 rows are not
# served by this template.
FUSED_WIDE_NBD: dict[int, int] = {
    32: 12,
    40: 12,
    44: 12,
    48: 12,
    52: 11,
    56: 11,
    60: 10,
    64: 9,
}
# Buckets that keep the plain all-reduce instead of `arpipe`.
ARPIPE_OFF_B: frozenset[int] = frozenset({20, 24, 32})
# Buckets that run the GDN delta rule on the MXU.
MXU_DR_BUCKETS: frozenset[int] = frozenset({4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64})
# `hx` pays from this many rows (smaller payloads lose to the extra round).
HX_MIN_ROWS = 48
# `qwen35-candidate4-sh-at2`: `moe.sh_at` applies up to this many rows.
SH_AT_MAX_ROWS = 8
HIER1_PEERS_R8: tuple[int, int, int] = (3, 5, 7)
# Dense ring depth of an attention-only stack (no GDN layers: the MTP draft).
ATTENTION_ONLY_NBD = 14


def _is_attention_only(tc: TileContext) -> bool:
    g = tc.g
    return g is not None and int(getattr(g, "att_every", 0)) == 1


def candidate4_tiles(tc: TileContext | int, *, sh_at: int = 0) -> Tiles:
    """Per-bucket tiles of the candidate4 schedule (see the module docstring)."""
    if not isinstance(tc, TileContext):
        tc = TileContext(B=int(tc))
    B, T, R = tc.B, tc.T, tc.R
    t = candidate30_tiles(tc)
    if T > 1 and tc.fused_pool_io and B in FUSED_WIDE_NBD:
        t = dataclasses.replace(t, nbd=FUSED_WIDE_NBD[B])
    if R == 8 and t.hier == 2 and B >= HX_MIN_ROWS:
        t = dataclasses.replace(t, hx=True)
    t = dataclasses.replace(
        t,
        arpipe=(
            B not in ARPIPE_OFF_B
            and T > 1
            and tc.fused_pool_io
            and not _is_attention_only(tc)
        ),
        arpipe_h1=True,
        ar0s=True,
        dwait_hoist=True,
        hier1_peers=HIER1_PEERS_R8 if R == 8 else t.hier1_peers,
    )
    if _is_attention_only(tc):
        t = dataclasses.replace(t, nbd=ATTENTION_ONLY_NBD)
    return t.with_modules(
        {
            _GQA: {"gdefer": True, "kvdefer": True, "qswap": True, "wgroup": True},
            _GDN: {"lcv": True, "mxu_dr": B in MXU_DR_BUCKETS, "zdefer": True},
            _MOE: {
                "route_shrink": True,
                "cp_interleave": True,
                "route_early": True,
                "prim_now": True,
                "sh_pre": True,
                "sh2_pre_route": True,
                "flush_in_ar1": True,
                "flush_pre_drain": True,
                "ep_fast_refill": True,
                "sh_at": sh_at if B <= SH_AT_MAX_ROWS else 0,
            },
        }
    )


_DESCRIPTION = (
    "Candidate4 schedule (PD submission decode legs): qwen35-candidate30 tiles "
    "plus fused-bucket dense rings (32..64 rows), hx from 48 rows, hier=1 peers "
    "(3, 5, 7), arpipe / ar0s / dense wait hoist, GQA deferred waits and RoPE "
    "lane swap, GDN conv window / MXU delta rule / deferred z, MoE route and "
    "shared-expert schedule and EP fast refill; fused pool I/O on every bucket. "
    "Numerics: approx."
)

PROFILE = dataclasses.replace(
    CANDIDATE30_PROFILE,
    name="qwen35-candidate4",
    tiles=candidate4_tiles,
    fused_pool_io=True,
    verify_buckets=tuple(range(4, 65, 4)),
    description=_DESCRIPTION,
)

SH_AT2_PROFILE = dataclasses.replace(
    PROFILE,
    name="qwen35-candidate4-sh-at2",
    tiles=functools.partial(candidate4_tiles, sh_at=2),
    description=_DESCRIPTION + f" Shared expert after the first expert DMAs "
    f"(sh_at=2) up to {SH_AT_MAX_ROWS} rows.",
)

__all__ = [
    "ARPIPE_OFF_B",
    "FUSED_WIDE_NBD",
    "HX_MIN_ROWS",
    "MXU_DR_BUCKETS",
    "PROFILE",
    "SH_AT2_PROFILE",
    "SH_AT_MAX_ROWS",
    "candidate4_tiles",
]
