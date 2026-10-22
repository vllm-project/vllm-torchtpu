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
"""Qwen3.8-Flash-Next (`qwen4_exp`) on the megakernel: geometry and segment spec.

The model is 48 layers `[GDN, GDN, GDN, QSA] x 12`, each layer two
hyper-connection (HC) sublayers (mixer, MoE) over a 4-stream residual. The
QSA attention core (indexer top-k on SparseCore, sparse attention) runs
outside the kernel, so one megakernel call is one *segment* between two QSA
cores, four kernel layers:

  j = 0      HC(QSA o_proj)  + HC(MoE)     layer 4s+3, after its attention core
  j = 1..3   HC(GDN)         + HC(MoE)     layers 4s+4 .. 4s+6

The core residual is the full 4-stream residual (`H = NHC * hidden`); the
sublayer blocks see the `hidden`-wide inner geometry. Dense weights are BF16
as in the checkpoint; routed experts are FP8 with per-channel scales.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from ..blocks.gdn import GDNBlock
from ..blocks.hc import HCSublayer
from ..blocks.moe import MoEBlock
from ..blocks.moe.params import MoEParams
from ..blocks.qsa_out import QSAOutBlock
from ..exec_spec import ExecSpec
from ..ops import BF16, rup
from ..spec import LayerType, ModelSpec, SwitchSlot, WeightDecl
from .qwen35 import Qwen35Geometry

QWEN38_SEGMENT = "qwen38_segment"
SEGMENT_LAYERS = 4


def _col_chunk(H: int, cap: int = 1024) -> int:
    """Largest 128-multiple divisor of `H` that is at most `cap`."""
    return max(c for c in range(128, min(cap, H) + 1, 128) if H % c == 0)


@dataclass(frozen=True)
class Qwen38InnerGeometry(Qwen35Geometry):
    """Sublayer geometry: Qwen3.5's field set with Qwen3.8's validity rules.

    `H` need not be a multiple of 1024 (Qwen3.8: 2560); the shared expert is
    padded so each rank's slice is a 128 multiple.
    """

    def check(self, ex: ExecSpec) -> None:
        R, B, T, M = ex.R, ex.B, ex.T, ex.M
        assert self.L % self.att_every == 0 and self.NH % R == 0 and R % self.NKV == 0
        assert self.E % R == 0 and B * self.TOPK <= 640 and 1 <= B <= M <= 64
        assert self.E % 128 == 0 and self.E <= 1024 and (R & (R - 1)) == 0
        assert T >= 1 and B % T == 0, (B, T)
        for n in (self.H, self.HD, self.IR(R), self.ISR(R)):
            assert n % 128 == 0, n
        if self.NGDN > 0:
            assert self.KH % R == 0 and self.VH % R == 0 and self.VH % self.KH == 0
            lay = GDNBlock.lin_in_layout(self, R)
            assert lay.vh <= 64, lay.vh
            for n in (lay.KD, lay.VD, lay.QKV, lay.Z):
                assert n % 128 == 0, n


@dataclass(frozen=True)
class Qwen38Geometry:
    """Qwen3.8-Flash-Next architecture (HF `text_config`)."""

    L: int = 48
    att_every: int = 4
    H: int = 2560
    NH: int = 24
    NKV: int = 2
    HD: int = 256
    ROT: int = 64
    KH: int = 16
    VH: int = 48
    KD: int = 128
    VD: int = 128
    E: int = 512
    TOPK: int = 10
    I: int = 640  # noqa: E741
    IS: int = 640
    NHC: int = 4
    LOWRANK: int = 320
    eps: float = 1e-6
    gdn_l2_eps: float = 1e-6
    theta: float = 10_000_000.0

    @property
    def n_segments(self) -> int:
        """Kernel calls per step: one per QSA layer except the last (the tail)."""
        return self.L // self.att_every - 1

    def shared_inter_padded(self, R: int) -> int:
        """Shared-expert width padded so each rank's slice is a 128 multiple."""
        return rup(self.IS, 128 * R)

    def inner(self, R: int, L: int = SEGMENT_LAYERS) -> Qwen35Geometry:
        """Sublayer geometry seen by the GDN / MoE / o_proj blocks.

        A one-layer (tail) segment has no GDN layer: `att_every=1`.
        """
        return Qwen38InnerGeometry(
            L=L,
            att_every=self.att_every if L > 1 else 1,
            H=self.H,
            NH=self.NH,
            NKV=self.NKV,
            HD=self.HD,
            ROT=self.ROT,
            KH=self.KH,
            VH=self.VH,
            KD=self.KD,
            VD=self.VD,
            E=self.E,
            TOPK=self.TOPK,
            I=self.I,
            IS=self.shared_inter_padded(R),
            moe_tp=False,
            eps=self.eps,
            gdn_l2_eps=self.gdn_l2_eps,
            theta=self.theta,
        )


@dataclass(frozen=True)
class Qwen38ResidualGeometry:
    """Core geometry of a segment kernel: the 4-stream residual."""

    L: int
    H: int
    eps: float = 1e-6
    zero_centered_norm: bool = True


def _bf16(decls: list[WeightDecl], names: tuple[str, ...]) -> list[WeightDecl]:
    return [dataclasses.replace(w, dtype=BF16) if w.name in names else w for w in decls]


class Qwen38GDN(GDNBlock):
    """GDN with BF16 projections (scales `in_s` / `out_s` are ones)."""

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        return _bf16(super().weights(g, ex), ("lin_in", "lin_out"))


class Qwen38MoE(MoEBlock):
    """MoE with a BF16 shared expert (scales `sh13_s` / `sh2_s` are ones).

    The routed-expert output is chunked by columns in `CH`-wide slices; the
    library derives `CH = min(1024, H)`, which must divide `H`. For `H = 2560`
    the chunk is the largest 128-multiple divisor of `H` up to 1024 (640).
    """

    def weights(self, g: Any, ex: ExecSpec) -> list[WeightDecl]:
        return _bf16(super().weights(g, ex), ("sh13", "sh2"))

    def plan_params(self, g: Any, ex: ExecSpec, plan: Any) -> MoEParams:
        ch = _col_chunk(g.H)
        if g.H % min(1024, g.H) == 0:
            return super().plan_params(g, ex, plan)
        # `derive_moe_params` reads `g.H` only for the chunk and the `H` field.
        mp = super().plan_params(dataclasses.replace(g, H=1024), ex, plan)
        return dataclasses.replace(mp, H=g.H, CH=ch)


def segment_spec(g: Qwen38Geometry, R: int, *, L: int = SEGMENT_LAYERS) -> ModelSpec:
    """`ModelSpec` of one segment kernel (`L` kernel layers, QSA o_proj first)."""
    gi = g.inner(R, L)
    hc = dict(inner_geometry=gi, nhc=g.NHC, lowrank=g.LOWRANK)
    qsa = HCSublayer(QSAOutBlock(), mix=False, ar_slot=0, **hc)
    moe = HCSublayer(
        Qwen38MoE(geometry=gi, moe_tp=False), ar_slot=1, emit_residual=True, **hc
    )
    if L == 1:
        mixer: Any = qsa
    else:
        gdn = HCSublayer(Qwen38GDN(geometry=gi), ar_slot=0, **hc)
        mixer = SwitchSlot("mixer", branches=(qsa, gdn), branch_counts=(1, L - 1))
    return ModelSpec(
        name=QWEN38_SEGMENT,
        config=Qwen38ResidualGeometry(L=L, H=g.NHC * g.H, eps=g.eps),
        layer_type=LayerType(name=QWEN38_SEGMENT, slots=(mixer, moe)),
    )


FULL = Qwen38Geometry()
# Tiny geometry with production per-rank GDN shapes at R=2 (2 k-heads, 6 v-heads).
TINY = Qwen38Geometry(
    L=8, H=256, NH=6, KH=4, VH=12, E=128, TOPK=4, I=128, IS=128, LOWRANK=64
)
