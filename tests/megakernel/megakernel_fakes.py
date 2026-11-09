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
"""Shared fake modules, geometries, and tile helpers for megakernel unit tests."""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.ops import BF16, F32, FP8, rmsnorm
from vllm_torchtpu.megakernel.spec import (
    LayerType,
    ModelSpec,
    Module,
    ScratchReq,
    SmallField,
    StateDecl,
    SwitchSlot,
    WeightDecl,
)

# Small banks and rings for CPU interpret unit tests of tiny geometries.
TINY_TILES = Tiles(bank=128 << 10, nbd=10, nbe=6, ebank=256 << 10, vmem_mib=32)


@dataclass(frozen=True)
class FakeGeometry:
    """Compact model geometry for fast CPU unit tests."""

    L: int = 4
    att_every: int = 1
    H: int = 256
    NH: int = 2
    NKV: int = 2
    HD: int = 128
    ROT: int = 32
    KH: int = 2
    VH: int = 2
    KD: int = 128
    VD: int = 128
    E: int = 128
    TOPK: int = 2
    I: int = 256  # noqa: E741
    IS: int = 256
    moe_tp: bool = False
    e_slots: int = 0
    eps: float = 1e-6
    gdn_l2_eps: float = 1e-6
    theta: float = 10_000_000.0
    zero_centered_norm: bool = True


@dataclass(frozen=True)
class FakeMixerTunables:
    """Module-owned knobs of `FakeMixerModule` (no core `Tiles` field needed)."""

    kv_blk: int = 256
    nkv: int = 3

    NUMERICS: ClassVar[dict[str, str]] = {"kv_blk": "approx", "nkv": "exact"}


class FakeMixerModule(Module):
    """Fake attention-style mixer module for testing declarative planning and VMEM."""

    name = "fake_gqa"
    reduction = "allreduce"
    input_norm = "in_norm"
    Tunables = FakeMixerTunables

    def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
        nh = g.NH // ex.R
        nai = (2 * nh + 2) * g.HD
        return [
            WeightDecl("att_in", K=g.H, N=nai, dtype=FP8, stream_ring="static"),
            WeightDecl("att_out", K=nh * g.HD, N=g.H, dtype=FP8, stream_ring="static"),
        ]

    def smalls(self, g: FakeGeometry, ex: ExecSpec) -> list[SmallField]:
        nh = g.NH // ex.R
        nai = (2 * nh + 2) * g.HD
        return [
            SmallField("in_norm", g.H),
            SmallField("in_s", nai),
            SmallField("out_s", g.H),
            SmallField("qn", g.HD),
            SmallField("kn", g.HD),
        ]

    def states(self, g: FakeGeometry, ex: ExecSpec) -> list[StateDecl]:
        return [
            StateDecl("kc", (ex.NS, ex.ctx, g.HD), BF16, access="r"),
            StateDecl("vc", (ex.NS, ex.ctx, g.HD), BF16, access="r"),
        ]

    def outputs(self, g: FakeGeometry, ex: ExecSpec) -> list[StateDecl]:
        return [
            StateDecl("knew", (ex.MR, g.HD), F32, access="w"),
            StateDecl("vnew", (ex.MR, g.HD), F32, access="w"),
        ]

    def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
        nh = g.NH // ex.R
        nai = (2 * nh + 2) * g.HD
        return {
            "acc": max(nai, g.H),
            "ybuf": nai,
            "xb2": nh * g.HD,
            "mix": nh * g.HD,
        }

    def scratch(self, g: FakeGeometry, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        t = self.tunables(ex)
        return [
            ScratchReq(
                "kbuf", pltpu.VMEM((t.nkv, t.kv_blk, g.HD), BF16), phase="persistent"
            ),
            ScratchReq(
                "vbuf", pltpu.VMEM((t.nkv, t.kv_blk, g.HD), BF16), phase="persistent"
            ),
            ScratchReq(
                "branch_tmp", pltpu.VMEM((ex.MR, g.HD), F32), phase="branch:gqa"
            ),
        ]


class FakeAltMixerModule(Module):
    """Fake linear-attention mixer module for testing multi-branch `SwitchSlot`."""

    name = "fake_gdn"
    reduction = "allreduce"
    input_norm = "in_norm"

    def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
        kh = g.KH // ex.R
        vh = g.VH // ex.R
        qkv = 2 * kh * g.KD + vh * g.VD
        z = vh * g.VD
        nli = qkv + z + 128
        return [
            WeightDecl("lin_in", K=g.H, N=nli, dtype=FP8, stream_ring="static"),
            WeightDecl("lin_out", K=z, N=g.H, dtype=FP8, stream_ring="static"),
        ]

    def smalls(self, g: FakeGeometry, ex: ExecSpec) -> list[SmallField]:
        kh = g.KH // ex.R
        vh = g.VH // ex.R
        qkv = 2 * kh * g.KD + vh * g.VD
        z = vh * g.VD
        nli = qkv + z + 128
        return [
            SmallField("in_norm", g.H),
            SmallField("in_s", nli),
            SmallField("out_s", g.H),
            SmallField("conv", 4 * qkv),
            SmallField("gnorm", g.VD),
        ]

    def states(self, g: FakeGeometry, ex: ExecSpec) -> list[StateDecl]:
        vh = g.VH // ex.R
        return [
            StateDecl("rec", (ex.NS * vh, g.KD, g.VD), BF16, access="rw"),
        ]

    def outputs(self, g: FakeGeometry, ex: ExecSpec) -> list[StateDecl]:
        vh = g.VH // ex.R
        return [
            StateDecl("rec", (ex.B * vh, g.KD, g.VD), BF16, access="w"),
        ]

    def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
        kh = g.KH // ex.R
        vh = g.VH // ex.R
        qkv = 2 * kh * g.KD + vh * g.VD
        z = vh * g.VD
        nli = qkv + z + 128
        return {
            "acc": max(nli, g.H),
            "ybuf": nli,
            "xb2": z,
            "mix": z,
        }

    def scratch(self, g: FakeGeometry, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        vh = g.VH // ex.R
        return [
            ScratchReq("sbuf", pltpu.VMEM((vh, g.KD, g.VD), F32), phase="persistent"),
            ScratchReq(
                "branch_tmp", pltpu.VMEM((ex.MR * 2, g.VD), F32), phase="branch:gdn"
            ),
        ]


class FakeFFNModule(Module):
    """Fake MoE FFN module for testing declarative planning and VMEM."""

    name = "fake_moe"
    reduction = "allreduce"
    input_norm = "post_norm"

    def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
        isr = g.IS // ex.R
        ir = g.I // ex.R if g.moe_tp else g.I
        er = g.E // ex.R
        ew = g.e_slots if g.e_slots > 0 else (g.E if g.moe_tp else er)
        lay = "stacked" if ex.stack_e else "per_layer"
        return [
            WeightDecl("router", K=g.H, N=g.E, dtype=BF16, stream_ring="static"),
            WeightDecl("sh13", K=g.H, N=2 * isr, dtype=FP8, stream_ring="static"),
            WeightDecl("sh2", K=isr, N=g.H, dtype=FP8, stream_ring="static"),
            WeightDecl(
                "e13",
                K=g.H,
                N=2 * ir,
                dtype=FP8,
                stream_ring="dynamic",
                layout=lay,
                lead_dims=(ew,),
            ),
            WeightDecl(
                "e2",
                K=ir,
                N=g.H,
                dtype=FP8,
                stream_ring="dynamic",
                layout=lay,
                lead_dims=(ew,),
            ),
            WeightDecl(
                "es", K=1, N=2 * ir + g.H, dtype=F32, stream_ring="hbm", lead_dims=(ew,)
            ),
        ]

    def smalls(self, g: FakeGeometry, ex: ExecSpec) -> list[SmallField]:
        isr = g.IS // ex.R
        er = g.E // ex.R
        fields = [
            SmallField("post_norm", g.H),
            SmallField("sh13_s", 2 * isr),
            SmallField("sh2_s", g.H),
            SmallField("sh_gate", g.H),
        ]
        if not g.moe_tp and g.e_slots > er:
            fields.append(SmallField("ecopy", g.E))
        return fields

    def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
        isr = g.IS // ex.R
        ir = g.I // ex.R if g.moe_tp else g.I
        return {
            "acc": max(g.H, g.E, 2 * isr, 2 * ir),
            "xb2": max(isr, ir),
        }

    def scratch(self, g: FakeGeometry, ex: ExecSpec, plan: Any) -> dict[str, Any]:
        ir = g.I // ex.R if g.moe_tp else g.I
        t = ex.resolved_tiles
        te = plan.nt["e13"] + plan.nt["e2"]
        nss = 3 + (t.nbe - 1) // te
        return {
            "wn": pltpu.VMEM((ex.MR, g.E), F32),
            "es": pltpu.VMEM((nss, 1, 2 * ir + g.H), F32),
        }


def make_fake_model_spec(
    *,
    L: int = 4,
    att_every: int = 1,
    H: int = 256,
    moe_tp: bool = False,
    e_slots: int = 0,
) -> ModelSpec:
    """Create a `ModelSpec` built from fake mixer and FFN modules."""
    g = FakeGeometry(L=L, att_every=att_every, H=H, moe_tp=moe_tp, e_slots=e_slots)
    gqa = FakeMixerModule()
    moe = FakeFFNModule()
    if att_every == 1:
        mixer: Module | SwitchSlot = SwitchSlot(
            "mixer", branches=(gqa,), branch_counts=(1,)
        )
    else:
        gdn = FakeAltMixerModule()
        mixer = SwitchSlot(
            "mixer", branches=(gdn, gqa), branch_counts=(att_every - 1, 1)
        )
    return ModelSpec(
        name="fake_model",
        config=g,
        layer_type=LayerType("fake_layer", slots=(mixer, moe)),
    )


@dataclass
class MiniCtx:
    """Minimal kernel execution context for testing module `emit` in Pallas."""

    g: FakeGeometry
    ex: ExecSpec
    core_plan: Any
    pos_ref: Any
    W: dict[str, Any]
    ST: dict[str, Any]
    O: dict[str, Any]  # noqa: E741
    scr: dict[str, Any]
    me: Any
    stream: Any
    comm: Any
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)
    skip: frozenset[str] = frozenset()
    sc: Any = lambda *_: contextlib.nullcontext()
    norm_x: Any = None

    def module_params(self, name: str) -> Any:
        return self.core_plan.module(name)

    def sm(self, slot: Any, name: str, width: int | None = None) -> Any:
        o, w = self.core_plan.stream.lay[name]
        n = w if width is None else width
        return self.scr["sm"][slot, :, o : o + n]

    def apply_input_norm(self, norm_field: str | None, slot: Any) -> None:
        if norm_field is None:
            self.norm_x = None
            return
        x = rmsnorm(
            self.scr["res"][...],
            self.sm(slot, norm_field),
            self.g.eps,
            zero_centered=bool(self.g.zero_centered_norm),
        )
        self.scr["xb"][...] = x.astype(BF16)
        self.norm_x = x


def with_tile_fields(t: Any, fields: Mapping[str, Any]) -> Any:
    """`t` with knobs replaced: core `field` keys and `<module>.<field>` keys."""
    core = {k: v for k, v in fields.items() if "." not in k}
    modules: dict[str, dict[str, Any]] = {}
    for k, v in fields.items():
        if "." in k:
            module, field = k.split(".", 1)
            modules.setdefault(module, {})[field] = v
    return dataclasses.replace(t, **core).with_modules(modules)


def tile_field(t: Any, key: str) -> Any:
    """Value of knob `key` (`field` or `<module>.<field>`) in tiles `t`."""
    if "." not in key:
        return getattr(t, key)
    module, field = key.split(".", 1)
    return getattr(t.module(module), field)


# Exact-class knobs (core `field` or `<module>.<field>`): (id, knob, value off the
# model default, (T, B), extra knobs).
EXACT_FLIPS: list[tuple[str, str, Any, tuple[int, int], dict[str, Any]]] = [
    ("nbd", "nbd", 8, (2, 4), {}),
    ("nkv", "gqa.nkv", 4, (2, 4), {}),
    ("lazy", "lazy", False, (2, 4), {}),
    ("pair_kv", "gqa.pair_kv", True, (2, 4), {}),
    ("kv_ering", "gqa.kv_ering", True, (2, 4), {}),
    ("gqa_qv", "gqa.gqa_qv", True, (2, 4), {}),
    ("rel_fetch", "rel_fetch", True, (2, 4), {}),
    ("bounds_checks", "bounds_checks", True, (2, 4), {}),
    ("vmem_mib", "vmem_mib", 48, (2, 4), {}),
    ("smem_mib", "smem_mib", 2, (2, 4), {}),
    ("moe_group_rows", "moe.moe_group_rows", 0, (2, 4), {"moe.moe_unroll": True}),
    # PIPE_EP path: I=512 with a 64 KiB expert bank gives T13 = 2 * T2.
    (
        "moe_group_rows-pipe_ep",
        "moe.moe_group_rows",
        0,
        (2, 4),
        {"moe.pipe_ep": True, "ebank": 64 << 10, "nbe": 6, "geom_kw": {"I": 512}},
    ),
    # WIDE verify bucket (chunked GDN states): gdn_bufs only acts here.
    ("gdn_bufs-wide", "gdn.gdn_bufs", 2, (4, 24), {}),
    ("pair_kv-wide", "gqa.pair_kv", True, (4, 24), {}),
    ("kv_ering-wide", "gqa.kv_ering", True, (4, 24), {}),
    ("gqa_qv-wide", "gqa.gqa_qv", True, (4, 24), {}),
]
