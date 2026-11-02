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
    __annotations__["I"] = int
    vars()["I"] = 256
    IS: int = 256
    moe_tp: bool = False
    e_slots: int = 0
    eps: float = 1e-6
    gdn_l2_eps: float = 1e-6
    theta: float = 10_000_000.0
    zero_centered_norm: bool = True
    gdn_gate: str = "silu"

    def vh(self, R: int) -> int:
        return self.VH // R

    def ER(self, R: int) -> int:
        return self.E // R

    def EW(self, R: int) -> int:
        return self.e_slots or (self.E if self.moe_tp else self.ER(R))

    def check(self, ex: ExecSpec) -> None:
        del ex


SynthHybrid = FakeGeometry


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
        nh, hd = g.NH // ex.R, g.HD
        nai = (2 * nh + 2) * hd
        sfs = [("in_norm", g.H), ("in_s", nai), ("out_s", g.H), ("qn", hd), ("kn", hd)]
        return [SmallField(k, w) for k, w in sfs]

    def states(self, g: FakeGeometry, ex: ExecSpec) -> list[StateDecl]:
        sh = (ex.NS, ex.ctx, g.HD)
        return [StateDecl("kc", sh, BF16, "r"), StateDecl("vc", sh, BF16, "r")]

    def outputs(self, g: FakeGeometry, ex: ExecSpec) -> list[StateDecl]:
        sh = (ex.MR, g.HD)
        return [StateDecl("knew", sh, F32, "w"), StateDecl("vnew", sh, F32, "w")]

    def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
        nh = g.NH // ex.R
        nai = (2 * nh + 2) * g.HD
        return {"acc": max(nai, g.H), "ybuf": nai, "xb2": nh * g.HD, "mix": nh * g.HD}

    def scratch(self, g: FakeGeometry, ex: ExecSpec, plan: Any) -> list[ScratchReq]:
        t = self.tunables(ex)
        kv_vmem = pltpu.VMEM((t.nkv, t.kv_blk, g.HD), BF16)
        return [
            ScratchReq("kbuf", kv_vmem, phase="persistent"),
            ScratchReq("vbuf", kv_vmem, phase="persistent"),
            ScratchReq("branch_tmp", pltpu.VMEM((ex.MR, g.HD), F32), "branch:gqa"),
        ]


class FakeAltMixerModule(Module):
    """Fake linear-attention mixer module for testing multi-branch `SwitchSlot`."""

    name = "fake_gdn"
    reduction = "allreduce"
    input_norm = "in_norm"

    def weights(self, g: FakeGeometry, ex: ExecSpec) -> list[WeightDecl]:
        kh, vh = g.KH // ex.R, g.VH // ex.R
        qkv, z = 2 * kh * g.KD + vh * g.VD, vh * g.VD
        nli = qkv + z + 128
        return [
            WeightDecl("lin_in", K=g.H, N=nli, dtype=FP8, stream_ring="static"),
            WeightDecl("lin_out", K=z, N=g.H, dtype=FP8, stream_ring="static"),
        ]

    def smalls(self, g: FakeGeometry, ex: ExecSpec) -> list[SmallField]:
        kh, vh = g.KH // ex.R, g.VH // ex.R
        qkv, z = 2 * kh * g.KD + vh * g.VD, vh * g.VD
        nli = qkv + z + 128
        sfs = [
            ("in_norm", g.H),
            ("in_s", nli),
            ("out_s", g.H),
            ("conv", 4 * qkv),
            ("gnorm", g.VD),
        ]
        return [SmallField(k, w) for k, w in sfs]

    def states(self, g: FakeGeometry, ex: ExecSpec) -> list[StateDecl]:
        return [StateDecl("rec", (ex.NS * (g.VH // ex.R), g.KD, g.VD), BF16, "rw")]

    def outputs(self, g: FakeGeometry, ex: ExecSpec) -> list[StateDecl]:
        return [StateDecl("rec", (ex.B * (g.VH // ex.R), g.KD, g.VD), BF16, "w")]

    def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
        kh, vh = g.KH // ex.R, g.VH // ex.R
        qkv, z = 2 * kh * g.KD + vh * g.VD, vh * g.VD
        nli = qkv + z + 128
        return {"acc": max(nli, g.H), "ybuf": nli, "xb2": z, "mix": z}

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
        isr, ir = g.IS // ex.R, (g.I // ex.R if g.moe_tp else g.I)
        ew, lay = g.EW(ex.R), ("stacked" if ex.stack_e else "per_layer")
        dyn = dict(dtype=FP8, stream_ring="dynamic", layout=lay, lead_dims=(ew,))
        return [
            WeightDecl("router", K=g.H, N=g.E, dtype=BF16, stream_ring="static"),
            WeightDecl("sh13", K=g.H, N=2 * isr, dtype=FP8, stream_ring="static"),
            WeightDecl("sh2", K=isr, N=g.H, dtype=FP8, stream_ring="static"),
            WeightDecl("e13", K=g.H, N=2 * ir, **dyn),
            WeightDecl("e2", K=ir, N=g.H, **dyn),
            WeightDecl(
                "es", K=1, N=2 * ir + g.H, dtype=F32, stream_ring="hbm", lead_dims=(ew,)
            ),
        ]

    def smalls(self, g: FakeGeometry, ex: ExecSpec) -> list[SmallField]:
        isr, er = g.IS // ex.R, g.E // ex.R
        sfs = [
            ("post_norm", g.H),
            ("sh13_s", 2 * isr),
            ("sh2_s", g.H),
            ("sh_gate", g.H),
        ]
        if not g.moe_tp and g.e_slots > er:
            sfs.append(("ecopy", g.E))
        return [SmallField(k, w) for k, w in sfs]

    def shared_widths(self, g: FakeGeometry, ex: ExecSpec) -> dict[str, int]:
        isr, ir = g.IS // ex.R, (g.I // ex.R if g.moe_tp else g.I)
        return {"acc": max(g.H, g.E, 2 * isr, 2 * ir), "xb2": max(isr, ir)}

    def scratch(self, g: FakeGeometry, ex: ExecSpec, plan: Any) -> dict[str, Any]:
        ir = g.I // ex.R if g.moe_tp else g.I
        nss = 3 + (ex.resolved_tiles.nbe - 1) // (plan.nt["e13"] + plan.nt["e2"])
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
    gqa, moe = FakeMixerModule(), FakeFFNModule()
    br, cnt = (
        ((gqa,), (1,))
        if att_every == 1
        else ((FakeAltMixerModule(), gqa), (att_every - 1, 1))
    )
    mixer = SwitchSlot("mixer", branches=br, branch_counts=cnt)
    return ModelSpec(
        "fake_model", config=g, layer_type=LayerType("fake_layer", slots=(mixer, moe))
    )


def synth_hybrid_spec(
    g: FakeGeometry | None = None, *, R: int = 8, **overrides: Any
) -> ModelSpec:
    """Build a synthetic hybrid `ModelSpec` (`GDN` + `GQA` + `MoE` + `PoolBackend`)."""
    from vllm_torchtpu.megakernel import pool
    from vllm_torchtpu.megakernel.modules.gdn import GDNModule
    from vllm_torchtpu.megakernel.modules.gqa import GQAModule
    from vllm_torchtpu.megakernel.modules.moe import MoEModule

    if g is None:
        mtp = bool(overrides.get("moe_tp", False))
        base = dict(
            L=4, att_every=2, H=256, NH=max(2, R), NKV=min(2, R), HD=256, ROT=64
        )
        base.update(
            KH=max(2, R), VH=max(2, R), KD=128, VD=128, E=128, TOPK=2, moe_tp=mtp
        )
        base.update(
            {"I": max(256, 128 * R) if mtp else 256, "IS": max(256, 128 * R)}
            | overrides
        )
        g = FakeGeometry(**base)
    gqa, gdn, moe = GQAModule(), GDNModule(), MoEModule()
    decls = [
        gqa.cache_decl(g) if (i + 1) % g.att_every == 0 else gdn.cache_decl(g)
        for i in range(g.L)
    ]
    play = pool.derive_pool_layout(decls, R=R)
    gqa_p = GQAModule(pool_layout=play)
    br, cnt = (
        ((gqa_p,), (1,)) if g.att_every == 1 else ((gdn, gqa_p), (g.att_every - 1, 1))
    )
    mixer = SwitchSlot("mixer", br, cnt)
    pb = pool.make_pool_backend(play, rec_chunk=GDNModule.chunk_rows)
    return ModelSpec(
        "synth_hybrid", g, LayerType("synth_layer", (mixer, moe)), cache_backend=pb
    )


def synth_full_scale_spec(*, moe_tp: bool = False, R: int = 8) -> ModelSpec:
    """Build a 60-layer full-scale synthetic `ModelSpec` for VMEM fitting tests."""
    kw = dict(L=60, att_every=4, H=4096, NH=32, NKV=2, HD=256, ROT=64, KH=16, VH=64)
    return synth_hybrid_spec(E=512, TOPK=10, I=1024, IS=1024, moe_tp=moe_tp, R=R, **kw)


@dataclass
class MiniCtx:
    """Minimal kernel execution context for testing module `emit` in Pallas."""

    g: FakeGeometry
    ex: ExecSpec
    core_plan: Any
    pos_ref: Any
    W: dict[str, Any]
    ST: dict[str, Any]
    __annotations__["O"] = dict[str, Any]
    scr: dict[str, Any]
    me: Any
    stream: Any
    comm: Any
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)
    skip: frozenset[str] = frozenset()
    sc: Any = lambda *_: contextlib.nullcontext()
    norm_x: Any = None
    raw_refs: Any = None

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
    ("collective_id", "collective_id", 47, (2, 4), {}),
    ("moe_group_rows", "moe.moe_group_rows", 0, (2, 4), {"moe.moe_unroll": True}),
    (
        "moe_group_rows-pipe_ep",
        "moe.moe_group_rows",
        0,
        (2, 4),
        {"moe.pipe_ep": True, "ebank": 64 << 10, "nbe": 6, "geom_kw": {"I": 512}},
    ),
    ("gdn_bufs-wide", "gdn.gdn_bufs", 2, (4, 24), {}),
    ("pair_kv-wide", "gqa.pair_kv", True, (4, 24), {}),
    ("kv_ering-wide", "gqa.kv_ering", True, (4, 24), {}),
    ("gqa_qv-wide", "gqa.gqa_qv", True, (4, 24), {}),
]
