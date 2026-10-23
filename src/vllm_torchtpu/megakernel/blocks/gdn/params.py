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
"""`GDNBlock` build-time parameters, tunables and their derivation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from ...errors import MegakernelConfigError
from ...exec_spec import ExecSpec, NumericsClass
from ...ops import rup
from ...pool.config import cache_exec


@dataclass(frozen=True)
class GDNLinInLayout:
    """Column layout `[qkv | z | b (0:vh) | a (64:64+vh)]` of `lin_in` / `ybuf`."""

    kh: int
    vh: int
    KD: int
    VD: int
    QKV: int
    Z: int
    BA: int
    A_OFF: int
    BA_PAD: int
    NLI: int
    A_DIM: int = 0


@dataclass(frozen=True)
class GDNParams:
    """Block-local parameters owned and computed by `GDNBlock.plan_params`."""

    kh: int
    vh: int
    KD: int
    VD: int
    QKV: int
    Z: int
    BA: int
    A_OFF: int
    BA_PAD: int
    NLI: int
    H: int
    CR: int
    HR: int
    GCH: bool
    CB: int
    NF: int
    TR: int
    NCH: int
    NRB: int
    CBV: int
    NGDN: int
    eps: float
    zero_centered_norm: bool
    l2_eps: float = 1e-6
    NGI: int = 0
    NGO: int = 0
    G_PER_ATT: int = 0
    REC_PAGES: int = 4
    GDN_PAGES: int = 5
    SSM_U32_CAP: int = 512
    ssm_u32_rows: int = 0
    qkv_128: int = 0
    A_conv: int = 16
    conv8_rows: int = 0
    ckpt_is_3d: bool = True
    roff_is_2d: bool = True
    K_ckpt: int = 1
    ckpt_t_off: int = 0
    GPR: bool = False
    GCONV_ROWS: int = 0
    conv_stage: bool = False
    head_major: bool = False
    # Pool layout of the model's pool backend (set only for pool I/O: GPR or
    # fused); the slot geometry above and `page_base` come from it.
    pool_layout: Any = None
    A_DIM: int = 0
    decay_type: str = "head"
    decay_law: str = "softplus"
    gate_type: str = "silu"
    decay_bound: float = 5.0


@dataclass(frozen=True)
class GDNTunables:
    """`GDNBlock` knobs (`Tiles.blocks["gdn"]`); `NUMERICS` classes each field."""

    gdn_rows: int = 8  # State row chunk size of the chunked (`GCH`) path
    gdn_bufs: int = 3  # State ring depth of the chunked path
    # Opt-in performance flags (off by default; named profiles turn them on).
    conv_stage: bool = False  # Stage fused-path conv taps through VMEM `sb`
    # Head-major recurrence (per-head state update; reassociates the sums)
    gdn_head_major: bool = False

    # `exact`: moving the field keeps hidden states and caches bit-identical
    # (scheduling only); `approx`: changes or may change the arithmetic.
    NUMERICS: ClassVar[dict[str, NumericsClass]] = {
        "gdn_rows": "approx",  # chunked state rows (GCH)
        "gdn_bufs": "exact",  # state ring depth (scheduling)
        "conv_stage": "approx",  # fused-path conv staging
        "gdn_head_major": "approx",  # head-major state update (reassociates)
    }


def gdn_lin_in_layout(
    g: Any, ex_or_r: ExecSpec | int, *, decay_type: str | None = None
) -> GDNLinInLayout:
    """Return shared column offsets for `lin_in` (`[qkv | z | b | a]`)."""
    R = ex_or_r.R if isinstance(ex_or_r, ExecSpec) else int(ex_or_r)
    kh = max(1, g.KH // R) if g.KH > 0 else 0
    vh = g.VH // R
    KD, VD = g.KD, g.VD
    qkv = 2 * kh * KD + vh * VD
    z = vh * VD
    ba = qkv + z
    dt = (
        str(decay_type)
        if decay_type is not None
        else str(getattr(g, "decay_type", "head"))
    )
    if dt == "key":
        a_off = ba + 128
        a_dim = vh * KD
        ba_pad = 128 + a_dim
    else:
        a_off = ba + 64
        a_dim = vh
        ba_pad = 128
    return GDNLinInLayout(
        kh=kh,
        vh=vh,
        KD=KD,
        VD=VD,
        QKV=qkv,
        Z=z,
        BA=ba,
        A_OFF=a_off,
        BA_PAD=ba_pad,
        NLI=ba + ba_pad,
        A_DIM=a_dim,
    )


def derive_params(
    g: Any,
    ex: ExecSpec,
    plan: Any,
    tt: GDNTunables,
    name: str,
    *,
    options: dict[str, Any] | None = None,
) -> GDNParams:
    """`GDNParams` for geometry `g` under `ex` and the block's tunables `tt`."""
    tiles = ex.resolve_tiles(g)
    R, B, T = ex.R, ex.B, ex.T
    assert not (ex.VER and tiles.inplace), (
        "verify returns per-token checkpoints: no aliasing"
    )
    assert (g.KH % R == 0 or (g.KH > 0 and R % g.KH == 0)) and g.VH % R == 0, (
        g.KH,
        g.VH,
        R,
    )
    assert g.VH % g.KH == 0, (g.VH, g.KH)
    eff_dt = str((options or {}).get("decay_type", getattr(g, "decay_type", "head")))
    lay = gdn_lin_in_layout(g, ex, decay_type=eff_dt)
    assert lay.vh <= 64, ("b|a gate lanes require vh <= 64", lay.vh)
    for dim_name, dim_val in (
        ("H", g.H),
        ("KD", lay.KD),
        ("VD", lay.VD),
        ("QKV", lay.QKV),
        ("Z", lay.Z),
    ):
        assert dim_val % 128 == 0, (
            f"GDN {dim_name} must be a multiple of 128",
            dim_val,
        )

    CR = rup(3 * B, 8)
    HR = rup(3 + T, 8)

    ngdn = plan.block_layer_counts.get(name, 0)
    GCH = (
        (ex.WIDE or bool(cache_exec(ex).GPR))
        and tt.gdn_rows > 0
        and ngdn > 0
        and not cache_exec(ex).FUSED
    )
    CB = tt.gdn_rows if GCH else B
    assert not GCH or CB % 8 == 0, (
        "gdn_rows must be a multiple of 8",
        tt.gdn_rows,
    )
    assert not (GCH and ex.VER) or CB % T == 0, (
        "gdn_rows must be a multiple of T",
        CB,
        T,
    )
    NF, TR = divmod(B, CB)
    NCH = NF + int(TR > 0)
    NRB = min(tt.gdn_bufs, NCH)
    assert not GCH or NCH == 1 or NRB >= 2, ("gdn_bufs >= 2", tt.gdn_bufs)
    CBV = CB * lay.vh

    GPR = bool(cache_exec(ex).GPR and GCH and ex.VER)
    assert not cache_exec(ex).GPR or GPR, (
        "pool_gather needs the paged chunked verify path",
        B,
        T,
    )
    GCONV_ROWS = (HR * lay.QKV * 2) // (4 * 128)
    assert not GPR or cache_exec(ex).page_size == 128, cache_exec(ex).page_size
    assert not GPR or (HR * lay.QKV * 2) % (4 * 128) == 0, (HR, lay.QKV)
    assert not GPR or GCONV_ROWS <= 64, GCONV_ROWS

    g_per_att = max(0, getattr(g, "att_every", 1) - 1)
    ssm_u32_rows = (lay.vh * lay.KD) // 2
    REC_PAGES = max(4, (ssm_u32_rows + 127) // 128)
    GDN_PAGES = REC_PAGES + 1
    SSM_U32_CAP = REC_PAGES * 128
    pool_layout = None
    qkv_128 = lay.QKV // 128
    A_conv = max(16, rup(3 + T, 8))
    conv8_rows = A_conv * qkv_128
    NGI = min(ex.NS, 8) if cache_exec(ex).FUSED else 0
    NGO = min(ex.NS, 2) if cache_exec(ex).FUSED else 0
    K_ckpt = cache_exec(ex).ckpt_k if cache_exec(ex).ckpt_k > 0 else T
    ckpt_t_off = K_ckpt - T if K_ckpt > T else 0
    if cache_exec(ex).FUSED or GPR:
        pool_layout = checked_pool_layout(plan, ex, g_per_att, ssm_u32_rows)
        REC_PAGES = pool_layout.rec_pages_per_slot
        GDN_PAGES = pool_layout.pages_per_slot
        SSM_U32_CAP = pool_layout.rec_u32_rows
        g_per_att = pool_layout.slots_per_pool

    return GDNParams(
        kh=lay.kh,
        vh=lay.vh,
        KD=lay.KD,
        VD=lay.VD,
        QKV=lay.QKV,
        Z=lay.Z,
        BA=lay.BA,
        A_OFF=lay.A_OFF,
        BA_PAD=lay.BA_PAD,
        NLI=lay.NLI,
        H=g.H,
        CR=CR,
        HR=HR,
        GCH=GCH,
        CB=CB,
        NF=NF,
        TR=TR,
        NCH=NCH,
        NRB=NRB,
        CBV=CBV,
        NGDN=ngdn,
        eps=g.eps,
        zero_centered_norm=bool(g.zero_centered_norm),
        l2_eps=float(g.gdn_l2_eps),
        NGI=NGI,
        NGO=NGO,
        G_PER_ATT=g_per_att,
        REC_PAGES=REC_PAGES,
        GDN_PAGES=GDN_PAGES,
        SSM_U32_CAP=SSM_U32_CAP,
        ssm_u32_rows=ssm_u32_rows,
        qkv_128=qkv_128,
        A_conv=A_conv,
        conv8_rows=conv8_rows,
        ckpt_is_3d=not cache_exec(ex).ckpt_2d,
        roff_is_2d=not cache_exec(ex).roff_1d,
        K_ckpt=K_ckpt,
        ckpt_t_off=ckpt_t_off,
        GPR=GPR,
        GCONV_ROWS=GCONV_ROWS,
        conv_stage=bool(tt.conv_stage),
        head_major=bool(tt.gdn_head_major),
        pool_layout=pool_layout,
        A_DIM=lay.A_DIM,
        decay_type=eff_dt,
        decay_law=str(
            (options or {}).get("decay_law", getattr(g, "decay_law", "softplus"))
        ),
        gate_type=str(
            (options or {}).get("gate_type", getattr(g, "gate_type", "silu"))
        ),
        decay_bound=float(
            (options or {}).get("decay_bound", getattr(g, "decay_bound", 5.0))
        ),
    )


def checked_pool_layout(
    plan: Any, ex: ExecSpec, g_per_att: int, ssm_u32_rows: int
) -> Any:
    """The model's `PoolLayout`, checked against this block's geometry.

    In-kernel pool I/O (GPR / fused) reads checkpoint slots straight from
    the pools, so the slot geometry must come from the pool backend's
    layout; raise `MegakernelConfigError` when it can't serve this block.
    """
    pb = plan.model_spec.cache_backend
    if pb is None or pb.layout is None:
        raise MegakernelConfigError(
            f"GDN in-kernel pool I/O needs a pool backend with a layout on "
            f"ModelSpec {plan.model_spec.name!r}"
        )
    layout = pb.layout
    layout.validate(ex)
    if g_per_att <= 0 or layout.slots_per_pool != g_per_att:
        raise MegakernelConfigError(
            f"GDN in-kernel pool I/O: {g_per_att} recurrent layer(s) per "
            f"attention layer, but the pool layout has "
            f"slots_per_pool={layout.slots_per_pool}"
        )
    if ssm_u32_rows > layout.rec_u32_rows:
        raise MegakernelConfigError(
            f"GDN recurrent state ({ssm_u32_rows} u32 rows per rank) exceeds "
            f"the pool layout's recurrent slot ({layout.rec_u32_rows} rows)"
        )
    return layout
