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
"""Qwen3.5-397B-A17B geometry, weight packer, logical reference, and step builders."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .. import cache, kernel, planner
from ..blocks.gdn import GDNBlock
from ..blocks.gqa import GQABlock
from ..blocks.moe import MoEBlock
from ..blocks.norm import Norm
from ..exec_spec import ExecSpec
from ..ops import (
    BF16,
    F32,
    FP8,
    I32,
    rnd,
    rup,
)
from ..pool.config import POOL_STATE_NAME, cache_exec, resolved_max_pages_per_seq
from ..spec import Block, LayerType, ModelSpec, SwitchSlot
from . import qwen35_pool
from .qwen35_profile import QWEN35_MODEL_NAME


@dataclass(frozen=True)
class Qwen35Geometry:
    """Pure model architecture geometry for Qwen3.5-397B-A17B."""

    L: int = 60
    att_every: int = 4
    H: int = 4096
    NH: int = 32
    NKV: int = 2
    HD: int = 256
    ROT: int = 64
    KH: int = 16
    VH: int = 64
    KD: int = 128
    VD: int = 128
    E: int = 512
    TOPK: int = 10
    I: int = 1024  # noqa: E741
    IS: int = 1024
    moe_tp: bool = True
    e_slots: int = 0
    eps: float = 1e-6
    gdn_l2_eps: float = 1e-6
    theta: float = 10_000_000.0
    zero_centered_norm: bool = True

    def nh(self, R: int) -> int:
        return self.NH // R

    def G(self, R: int) -> int:
        return R // self.NKV

    def kh(self, R: int) -> int:
        return GDNBlock.lin_in_layout(self, R).kh

    def vh(self, R: int) -> int:
        return GDNBlock.lin_in_layout(self, R).vh

    def QKV(self, R: int) -> int:
        return GDNBlock.lin_in_layout(self, R).QKV

    def Z(self, R: int) -> int:
        return GDNBlock.lin_in_layout(self, R).Z

    def BA(self, R: int) -> int:
        return GDNBlock.lin_in_layout(self, R).BA

    def NLI(self, R: int) -> int:
        return GDNBlock.lin_in_layout(self, R).NLI

    def NAI(self, R: int) -> int:
        return (2 * self.nh(R) + 2) * self.HD

    def ER(self, R: int) -> int:
        return self.E // R

    def ISR(self, R: int) -> int:
        return self.IS // R

    def IR(self, R: int) -> int:
        return self.I // R if self.moe_tp else self.I

    def EW(self, R: int) -> int:
        return (
            self.e_slots
            if self.e_slots > 0
            else (self.E if self.moe_tp else self.ER(R))
        )

    @property
    def NGDN(self) -> int:
        return self.L - self.L // self.att_every

    @property
    def NATT(self) -> int:
        return self.L // self.att_every

    def check(self, ex: ExecSpec) -> None:
        R, B, T, M = ex.R, ex.B, ex.T, ex.M
        assert self.L % self.att_every == 0 and self.NH % R == 0 and R % self.NKV == 0
        assert self.E % R == 0 and B * self.TOPK <= 640 and 1 <= B <= M <= 64
        pc = min(1024, self.H)
        assert (
            self.E % 128 == 0
            and self.E <= 1024
            and self.H % pc == 0
            and self.nh(R) <= 8
        )
        assert self.ROT % 2 == 0 and (R & (R - 1) == 0)
        assert T >= 1 and B % T == 0, (B, T)
        assert self.e_slots == 0 or self.e_slots >= (
            self.E if self.moe_tp else self.ER(R)
        )
        if cache_exec(ex).paged_kv:
            assert (
                cache_exec(ex).page_size == 128
                and ex.ctx % cache_exec(ex).page_size == 0
                and cache_exec(ex).num_pages >= 1
                and resolved_max_pages_per_seq(ex) >= ex.ctx // cache_exec(ex).page_size
            ), (
                cache_exec(ex).page_size,
                ex.ctx,
                cache_exec(ex).num_pages,
                resolved_max_pages_per_seq(ex),
            )
        for n in (
            self.H,
            self.HD,
            self.IR(R),
            self.ISR(R),
        ):
            assert n % 128 == 0, n
        if self.NGDN > 0:
            assert self.KH % R == 0 and self.VH % R == 0 and self.VH % self.KH == 0
            lay = GDNBlock.lin_in_layout(self, R)
            assert lay.vh <= 64, lay.vh
            for n in (lay.KD, lay.VD, lay.QKV, lay.Z):
                assert n % 128 == 0, n

    def to_model_spec(self) -> ModelSpec:
        """Convert this geometry into its declarative `ModelSpec`."""
        return qwen35_spec(self)


# Alias for callers using MKConfig name
MKConfig = Qwen35Geometry

FULL = Qwen35Geometry()
FULL_EP = dataclasses.replace(FULL, moe_tp=False)
TINY = Qwen35Geometry(L=4, H=1024, NH=8, KH=8, VH=16, E=128, TOPK=4, I=1024, IS=1024)
TINY_EP = dataclasses.replace(TINY, moe_tp=False)


def qwen35_spec(g: Qwen35Geometry, gdn_block: Block | None = None) -> ModelSpec:
    """Construct the declarative `ModelSpec` for Qwen3.5."""
    gqa = GQABlock(pool_layout=qwen35_pool.qwen35_pool_layout(g))
    moe = MoEBlock()
    if g.att_every == 1:
        mixer: Block | SwitchSlot = SwitchSlot(
            "mixer", branches=(gqa,), branch_counts=(1,)
        )
    else:
        gdn = GDNBlock() if gdn_block is None else gdn_block
        mixer = SwitchSlot(
            "mixer",
            branches=(gdn, gqa),
            branch_counts=(g.att_every - 1, 1),
        )
    lt = LayerType(name="qwen35_layer", slots=(mixer, moe))
    return ModelSpec(
        name=QWEN35_MODEL_NAME,
        config=g,
        layer_type=lt,
        cache_backend=qwen35_pool.qwen35_pool_backend(g),
    )


def build(
    g: Qwen35Geometry,
    ex: ExecSpec,
    *,
    gdn_block: Block | None = None,
):
    """Build Qwen3.5 megakernel `(call, info)` via the generic `kernel.build` engine."""
    spec = qwen35_spec(g, gdn_block=gdn_block)
    return kernel.build(spec, ex)


def make_step(
    mesh,
    g: Qwen35Geometry,
    ex: ExecSpec,
    *,
    gdn_block: Block | None = None,
    donate_states: bool = False,
):
    """Build Qwen3.5 `jax.jit` + `shard_map` step `(step, info)`."""
    spec = qwen35_spec(g, gdn_block=gdn_block)
    return kernel.make_step(mesh, spec, ex, donate_states=donate_states)


def make_pooled_step(
    mesh,
    g: Qwen35Geometry,
    ex: ExecSpec,
    *,
    gdn_block: Block | None = None,
    donate_pools: bool = False,
    fused_pool_io: bool | str | None = None,
    max_b_low: int = 24,
    min_b_hi: int = 56,
):
    """Build Qwen3.5 paged-pool `shard_map` step (`PallasWriter`)."""
    spec = qwen35_spec(g, gdn_block=gdn_block)
    return cache.make_pooled_step(
        mesh,
        spec,
        ex,
        donate_pools=donate_pools,
        fused_pool_io=fused_pool_io,
        max_b_low=max_b_low,
        min_b_hi=min_b_hi,
    )


def accept_states(
    g: Qwen35Geometry,
    ex: ExecSpec,
    st: dict[str, Any],
    acc: Any,
) -> dict[str, Any]:
    """Delegate verify-mode GDN state selection to `GDNBlock.accept`."""
    return GDNBlock.accept(st, st, acc, T=ex.T, vh=g.vh(ex.R))


def default_eplb_cores(g: Qwen35Geometry, R: int) -> list[list[list[int]]]:
    """Build a deterministic per-layer redundant expert placement `[L][R][EW]`."""
    er, ew = g.ER(R), g.EW(R)
    extra = ew - er
    layers_cores: list[list[list[int]]] = []
    for lyr in range(g.L):
        cores_l: list[list[int]] = []
        for r1 in range(R):
            primary = [r1 * er + s0 for s0 in range(er)]
            redundant: list[int] = []
            if not g.moe_tp and extra > 0 and R > 1:
                for k in range(extra):
                    r0 = (r1 + 1 + (k % (R - 1))) % R
                    s0 = (lyr + k) % er
                    redundant.append(r0 * er + s0)
            else:
                redundant = [-1] * max(0, extra)
            cores_l.append(primary + redundant)
        layers_cores.append(cores_l)
    return layers_cores


def encode_eplb_ecopy(
    g: Qwen35Geometry,
    R: int,
    e_cores: list[list[list[int]]] | None = None,
) -> jax.Array:
    """Encode `[L][R][EW]` expert placement table into `[L, E]` `f32` `ecopy` rows."""
    if e_cores is None:
        e_cores = default_eplb_cores(g, R)
    er, ew = g.ER(R), g.EW(R)
    rows = [
        planner.encode_ecopy(g.E, er, ew, g.moe_tp, e_cores[lyr]) for lyr in range(g.L)
    ]
    return jnp.asarray(rows, dtype=F32)


def random_logical_inputs(
    key: jax.Array,
    g: Qwen35Geometry,
    ex: ExecSpec,
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Generate small random logical inputs for `reference_step` and `pack_inputs`."""
    g.check(ex)
    R, B, T, NS, ctx = ex.R, ex.B, ex.T, ex.NS, ex.ctx
    L, H, E, HD = g.L, g.H, g.E, g.HD
    nh = g.nh(R)
    lay_gdn = GDNBlock.lin_in_layout(g, ex)
    vh, KD, VD = lay_gdn.vh, lay_gdn.KD, lay_gdn.VD
    qkv, z, nli = lay_gdn.QKV, lay_gdn.Z, lay_gdn.NLI
    isr, ir, er, ew = g.ISR(R), g.IR(R), g.ER(R), g.EW(R)
    natt, ngdn = g.NATT, g.NGDN

    keys = jax.random.split(key, 36)
    it = iter(keys)

    base_pos = jnp.arange(1, NS + 1, dtype=I32) * min(16, ctx // 4)
    if T == 1:
        pos = base_pos
    else:
        pos = (base_pos[:, None] + jnp.arange(T, dtype=I32)[None, :]).reshape(B)

    hidden = jax.random.normal(next(it), (B, H), dtype=BF16) * 0.1

    def rfp8(shape):
        return (jax.random.normal(next(it), shape, dtype=F32) * 0.25).astype(FP8)

    def rscale(shape):
        return jnp.abs(jax.random.normal(next(it), shape, dtype=F32)) * 0.02 + 0.01

    G = g.G(R)
    nkv = g.NKV
    koff = nh * 2 * HD

    att_q = rfp8((R, natt, H, koff))
    att_kv = jnp.repeat(rfp8((nkv, natt, H, 2 * HD)), G, axis=0)
    att_in = jnp.concatenate([att_q, att_kv], axis=-1)

    att_q_s = rscale((R, natt, koff))
    att_kv_s = jnp.repeat(rscale((nkv, natt, 2 * HD)), G, axis=0)
    att_in_s = jnp.concatenate([att_q_s, att_kv_s], axis=-1)

    kn = jnp.repeat(
        jax.random.normal(next(it), (nkv, natt, HD), dtype=F32) * 0.05, G, axis=0
    )
    sh_gate = jnp.broadcast_to(
        (jax.random.normal(next(it), (L, H), dtype=F32) * 0.1)[None, :, :],
        (R, L, H),
    )

    e13 = rfp8((R, L, ew, H, 2 * ir))
    e13_s = rscale((R, L, ew, 2 * ir))
    e2 = rfp8((R, L, ew, ir, H))
    e2_s = rscale((R, L, ew, H))

    logical_weights: dict[str, Any] = {
        "in_norm": jax.random.normal(next(it), (L, H), dtype=F32) * 0.05,
        "post_norm": jax.random.normal(next(it), (L, H), dtype=F32) * 0.05,
        "final_norm": jax.random.normal(next(it), (H,), dtype=F32) * 0.05,
        "att_in": att_in,
        "att_in_s": att_in_s,
        "att_out": rfp8((R, natt, nh * HD, H)),
        "att_out_s": rscale((R, natt, H)),
        "qn": jax.random.normal(next(it), (R, natt, HD), dtype=F32) * 0.05,
        "kn": kn,
        "router": jax.random.normal(next(it), (L, H, E), dtype=BF16) * 0.1,
        "sh_gate": sh_gate,
        "sh13": rfp8((R, L, H, 2 * isr)),
        "sh13_s": rscale((R, L, 2 * isr)),
        "sh2": rfp8((R, L, isr, H)),
        "sh2_s": rscale((R, L, H)),
        "e13": e13,
        "e13_s": e13_s,
        "e2": e2,
        "e2_s": e2_s,
    }
    if not g.moe_tp and ew > er and R > 1:
        e_cores = default_eplb_cores(g, R)
        for lyr in range(L):
            for r1 in range(R):
                for k_idx, g_id in enumerate(e_cores[lyr][r1][er:ew]):
                    r0, s0 = divmod(int(g_id), er)
                    slot = er + k_idx
                    e13 = e13.at[r1, lyr, slot].set(e13[r0, lyr, s0])
                    e13_s = e13_s.at[r1, lyr, slot].set(e13_s[r0, lyr, s0])
                    e2 = e2.at[r1, lyr, slot].set(e2[r0, lyr, s0])
                    e2_s = e2_s.at[r1, lyr, slot].set(e2_s[r0, lyr, s0])
        logical_weights["e13"] = e13
        logical_weights["e13_s"] = e13_s
        logical_weights["e2"] = e2
        logical_weights["e2_s"] = e2_s
        logical_weights["e_cores"] = e_cores
        logical_weights["ecopy"] = encode_eplb_ecopy(g, R, e_cores)

    if ngdn > 0:
        logical_weights.update(
            {
                "lin_in": rfp8((R, ngdn, H, nli)),
                "lin_in_s": rscale((R, ngdn, nli)),
                "lin_out": rfp8((R, ngdn, z, H)),
                "lin_out_s": rscale((R, ngdn, H)),
                "conv": jax.random.normal(next(it), (R, ngdn, 4 * qkv), dtype=F32)
                * 0.05,
                "alog": jax.random.normal(next(it), (R, ngdn, vh), dtype=F32) * 0.1
                - 1.0,
                "dtb": jax.random.normal(next(it), (R, ngdn, vh), dtype=F32) * 0.1,
                "gnorm": jnp.ones((R, ngdn, VD), dtype=F32)
                + jax.random.normal(next(it), (R, ngdn, VD), dtype=F32) * 0.05,
            }
        )

    kc_nkv = jax.random.normal(next(it), (nkv, natt, NS, ctx, HD), dtype=BF16) * 0.1
    vc_nkv = jax.random.normal(next(it), (nkv, natt, NS, ctx, HD), dtype=BF16) * 0.1
    logical_states: dict[str, Any] = {
        "kc": jnp.repeat(kc_nkv, G, axis=0),
        "vc": jnp.repeat(vc_nkv, G, axis=0),
    }
    if ngdn > 0:
        logical_states["conv"] = (
            jax.random.normal(next(it), (R, ngdn, NS, 3, qkv), dtype=F32) * 0.05
        )
        logical_states["rec"] = (
            jax.random.normal(next(it), (R, ngdn, NS, vh, KD, VD), dtype=BF16) * 0.05
        )
    return pos, hidden, logical_weights, logical_states


def random_hf_weights(
    key: jax.Array,
    g: Qwen35Geometry,
) -> dict[str, jax.Array]:
    """Generate unsharded HuggingFace-style Qwen3.5 weights (`[L, ...]`, full `E`)."""
    L, H, E, inter_dim, IS = g.L, g.H, g.E, g.I, g.IS
    NH, NKV, HD = g.NH, g.NKV, g.HD
    KH, VH, KD, VD = g.KH, g.VH, g.KD, g.VD
    natt, ngdn = g.NATT, g.NGDN

    keys = jax.random.split(key, 56)
    it = iter(keys)

    def rfp8(shape):
        return (jax.random.normal(next(it), shape, dtype=F32) * 0.25).astype(FP8)

    def rscale(shape):
        return jnp.abs(jax.random.normal(next(it), shape, dtype=F32)) * 0.02 + 0.01

    q_proj = rfp8((natt, H, NH * 2 * HD))
    q_proj_s = rscale((natt, NH * 2 * HD))
    k_proj = rfp8((natt, NKV, H, HD))
    k_proj_s = rscale((natt, NKV, HD))
    v_proj = rfp8((natt, NKV, H, HD))
    v_proj_s = rscale((natt, NKV, HD))
    sh_gate_proj = rfp8((L, H, IS))
    sh_gate_proj_s = rscale((L, IS))
    sh_up_proj = rfp8((L, H, IS))
    sh_up_proj_s = rscale((L, IS))
    e_gate_proj = rfp8((L, E, H, inter_dim))
    e_gate_proj_s = rscale((L, E, inter_dim))
    e_up_proj = rfp8((L, E, H, inter_dim))
    e_up_proj_s = rscale((L, E, inter_dim))

    hf: dict[str, jax.Array] = {
        "in_norm": jax.random.normal(next(it), (L, H), dtype=F32) * 0.05,
        "post_norm": jax.random.normal(next(it), (L, H), dtype=F32) * 0.05,
        "final_norm": jax.random.normal(next(it), (H,), dtype=F32) * 0.05,
        "q_proj": q_proj,
        "q_proj_s": q_proj_s,
        "k_proj": k_proj,
        "k_proj_s": k_proj_s,
        "v_proj": v_proj,
        "v_proj_s": v_proj_s,
        "att_out": rfp8((natt, NH * HD, H)),
        "att_out_s": rscale((natt, H)),
        "qn": jax.random.normal(next(it), (natt, NH, HD), dtype=F32) * 0.05,
        "kn": jax.random.normal(next(it), (natt, NKV, HD), dtype=F32) * 0.05,
        "router": jax.random.normal(next(it), (L, H, E), dtype=BF16) * 0.1,
        "sh_gate": jax.random.normal(next(it), (L, H), dtype=F32) * 0.1,
        "sh_gate_proj": sh_gate_proj,
        "sh_gate_proj_s": sh_gate_proj_s,
        "sh_up_proj": sh_up_proj,
        "sh_up_proj_s": sh_up_proj_s,
        "sh13": jnp.concatenate([sh_gate_proj, sh_up_proj], axis=-1),
        "sh13_s": jnp.concatenate([sh_gate_proj_s, sh_up_proj_s], axis=-1),
        "sh2": rfp8((L, IS, H)),
        "sh2_s": rscale((L, H)),
        "e_gate_proj": e_gate_proj,
        "e_gate_proj_s": e_gate_proj_s,
        "e_up_proj": e_up_proj,
        "e_up_proj_s": e_up_proj_s,
        "e13": jnp.concatenate([e_gate_proj, e_up_proj], axis=-1),
        "e13_s": jnp.concatenate([e_gate_proj_s, e_up_proj_s], axis=-1),
        "e2": rfp8((L, E, inter_dim, H)),
        "e2_s": rscale((L, E, H)),
    }
    if ngdn > 0:
        qkv_full = 2 * KH * KD + VH * VD
        hf.update(
            {
                "lin_q": rfp8((ngdn, H, KH * KD)),
                "lin_q_s": rscale((ngdn, KH * KD)),
                "lin_k": rfp8((ngdn, H, KH * KD)),
                "lin_k_s": rscale((ngdn, KH * KD)),
                "lin_v": rfp8((ngdn, H, VH * VD)),
                "lin_v_s": rscale((ngdn, VH * VD)),
                "lin_z": rfp8((ngdn, H, VH * VD)),
                "lin_z_s": rscale((ngdn, VH * VD)),
                "lin_b": rfp8((ngdn, H, VH)),
                "lin_b_s": rscale((ngdn, VH)),
                "lin_a": rfp8((ngdn, H, VH)),
                "lin_a_s": rscale((ngdn, VH)),
                "lin_out": rfp8((ngdn, VH * VD, H)),
                "lin_out_s": rscale((ngdn, H)),
                "conv": jax.random.normal(next(it), (ngdn, 4, qkv_full), dtype=F32)
                * 0.05,
                "alog": jax.random.normal(next(it), (ngdn, VH), dtype=F32) * 0.1 - 1.0,
                "dtb": jax.random.normal(next(it), (ngdn, VH), dtype=F32) * 0.1,
                "gnorm": jnp.ones((ngdn, VD), dtype=F32)
                + jax.random.normal(next(it), (ngdn, VD), dtype=F32) * 0.05,
            }
        )
    return hf


def split_hf_weights(
    hf: dict[str, jax.Array],
    g: Qwen35Geometry,
    ex: ExecSpec,
    *,
    e_cores: list[list[list[int]]] | None = None,
) -> dict[str, Any]:
    """Shard unsharded HuggingFace-style Qwen3.5 weights across `R = ex.R` ranks.

    Produces `logical_weights` ready for `pack_inputs(...)` and `reference_step(...)`,
    including EPLB redundant expert replication (`ew > er`) and `ecopy` metadata.
    """
    g.check(ex)
    R = ex.R
    L, H, E, inter_dim, IS = g.L, g.H, g.E, g.I, g.IS
    HD = g.HD
    nh = g.nh(R)
    isr, ir, er, ew = g.ISR(R), g.IR(R), g.ER(R), g.EW(R)
    natt, ngdn = g.NATT, g.NGDN
    G = g.G(R)
    nkv = g.NKV
    koff = nh * 2 * HD

    # 1. Attention projections: [q_proj (NH*2*HD) | k_proj (NKV,HD) | v_proj (NKV,HD)]
    q_p = jnp.transpose(hf["q_proj"].reshape(natt, H, R, koff), (2, 0, 1, 3))
    q_s = jnp.transpose(hf["q_proj_s"].reshape(natt, R, koff), (1, 0, 2))

    k_in = (
        hf["k_proj"].reshape(natt, nkv, H, HD)
        if hf["k_proj"].ndim == 3
        else hf["k_proj"]
    )
    v_in = (
        hf["v_proj"].reshape(natt, nkv, H, HD)
        if hf["v_proj"].ndim == 3
        else hf["v_proj"]
    )
    ks_in = (
        hf["k_proj_s"].reshape(natt, nkv, HD)
        if hf["k_proj_s"].ndim == 2
        else hf["k_proj_s"]
    )
    vs_in = (
        hf["v_proj_s"].reshape(natt, nkv, HD)
        if hf["v_proj_s"].ndim == 2
        else hf["v_proj_s"]
    )
    kv_p = jnp.repeat(
        jnp.transpose(jnp.concatenate([k_in, v_in], axis=-1), (1, 0, 2, 3)),
        G,
        axis=0,
    )
    kv_s = jnp.repeat(
        jnp.transpose(jnp.concatenate([ks_in, vs_in], axis=-1), (1, 0, 2)),
        G,
        axis=0,
    )
    att_in = jnp.concatenate([q_p, kv_p], axis=-1)
    att_in_s = jnp.concatenate([q_s, kv_s], axis=-1)

    att_out = jnp.transpose(hf["att_out"].reshape(natt, R, nh * HD, H), (1, 0, 2, 3))
    att_out_s = jnp.broadcast_to(hf["att_out_s"][None, ...], (R, natt, H))

    qn_in = hf["qn"]
    if qn_in.ndim == 3:
        qn = jnp.transpose(qn_in.reshape(natt, R, nh, HD), (1, 0, 2, 3)).mean(axis=2)
    else:
        qn = jnp.broadcast_to(qn_in[None, ...], (R, natt, HD))
    kn_in = hf["kn"].reshape(natt, nkv, HD) if hf["kn"].ndim == 2 else hf["kn"]
    kn = jnp.repeat(jnp.transpose(kn_in, (1, 0, 2)), G, axis=0)

    # 2. Shared expert (always TP-sharded along IS across R)
    sh_gate = jnp.broadcast_to(hf["sh_gate"][None, ...], (R, L, H))
    if "sh_gate_proj" in hf and "sh_up_proj" in hf:
        sh_g_in, sh_u_in = hf["sh_gate_proj"], hf["sh_up_proj"]
        sh_gs_in, sh_us_in = hf["sh_gate_proj_s"], hf["sh_up_proj_s"]
    else:
        sh_g_in, sh_u_in = hf["sh13"][:, :, :IS], hf["sh13"][:, :, IS:]
        sh_gs_in, sh_us_in = hf["sh13_s"][:, :IS], hf["sh13_s"][:, IS:]
    sh_g = jnp.transpose(sh_g_in.reshape(L, H, R, isr), (2, 0, 1, 3))
    sh_u = jnp.transpose(sh_u_in.reshape(L, H, R, isr), (2, 0, 1, 3))
    sh13 = jnp.concatenate([sh_g, sh_u], axis=-1)

    sh_gs = jnp.transpose(sh_gs_in.reshape(L, R, isr), (1, 0, 2))
    sh_us = jnp.transpose(sh_us_in.reshape(L, R, isr), (1, 0, 2))
    sh13_s = jnp.concatenate([sh_gs, sh_us], axis=-1)

    sh2 = jnp.transpose(hf["sh2"].reshape(L, R, isr, H), (1, 0, 2, 3))
    sh2_s = jnp.broadcast_to(hf["sh2_s"][None, ...], (R, L, H))

    # 3. Routed experts: TP-sharded (g.moe_tp=True) or EP + EPLB (g.moe_tp=False)
    if "e_gate_proj" in hf and "e_up_proj" in hf:
        e_g_in, e_u_in = hf["e_gate_proj"], hf["e_up_proj"]
        e_gs_in, e_us_in = hf["e_gate_proj_s"], hf["e_up_proj_s"]
        e13_full = jnp.concatenate([e_g_in, e_u_in], axis=-1)
        e13_s_full = jnp.concatenate([e_gs_in, e_us_in], axis=-1)
    else:
        e13_full = hf["e13"]
        e13_s_full = hf["e13_s"]
        e_g_in, e_u_in = e13_full[:, :, :, :inter_dim], e13_full[:, :, :, inter_dim:]
        e_gs_in, e_us_in = (
            e13_s_full[:, :, :inter_dim],
            e13_s_full[:, :, inter_dim:],
        )

    if g.moe_tp:
        e13_g = jnp.transpose(e_g_in.reshape(L, E, H, R, ir), (3, 0, 1, 2, 4))
        e13_u = jnp.transpose(e_u_in.reshape(L, E, H, R, ir), (3, 0, 1, 2, 4))
        e13 = jnp.concatenate([e13_g, e13_u], axis=-1)

        e13_sg = jnp.transpose(e_gs_in.reshape(L, E, R, ir), (2, 0, 1, 3))
        e13_su = jnp.transpose(e_us_in.reshape(L, E, R, ir), (2, 0, 1, 3))
        e13_s = jnp.concatenate([e13_sg, e13_su], axis=-1)

        e2 = jnp.transpose(hf["e2"].reshape(L, E, R, ir, H), (2, 0, 1, 3, 4))
        e2_s = jnp.broadcast_to(hf["e2_s"][None, ...], (R, L, E, H))
    elif ew == er:
        e13 = jnp.transpose(
            e13_full.reshape(L, R, er, H, 2 * inter_dim), (1, 0, 2, 3, 4)
        )
        e13_s = jnp.transpose(e13_s_full.reshape(L, R, er, 2 * inter_dim), (1, 0, 2, 3))
        e2 = jnp.transpose(hf["e2"].reshape(L, R, er, inter_dim, H), (1, 0, 2, 3, 4))
        e2_s = jnp.transpose(hf["e2_s"].reshape(L, R, er, H), (1, 0, 2, 3))
    else:
        if e_cores is None:
            e_cores = default_eplb_cores(g, R)
        idx = jnp.maximum(jnp.asarray(e_cores, dtype=I32), 0)  # [L, R, EW]
        l_idx = jnp.arange(L, dtype=I32)[:, None, None]
        e13 = jnp.transpose(e13_full[l_idx, idx], (1, 0, 2, 3, 4))
        e13_s = jnp.transpose(e13_s_full[l_idx, idx], (1, 0, 2, 3))
        e2 = jnp.transpose(hf["e2"][l_idx, idx], (1, 0, 2, 3, 4))
        e2_s = jnp.transpose(hf["e2_s"][l_idx, idx], (1, 0, 2, 3))

    logical_weights: dict[str, Any] = {
        "in_norm": hf["in_norm"],
        "post_norm": hf["post_norm"],
        "final_norm": hf["final_norm"],
        "att_in": att_in,
        "att_in_s": att_in_s,
        "att_out": att_out,
        "att_out_s": att_out_s,
        "qn": qn,
        "kn": kn,
        "router": hf["router"],
        "sh_gate": sh_gate,
        "sh13": sh13,
        "sh13_s": sh13_s,
        "sh2": sh2,
        "sh2_s": sh2_s,
        "e13": e13,
        "e13_s": e13_s,
        "e2": e2,
        "e2_s": e2_s,
    }
    if not g.moe_tp and ew > er:
        assert e_cores is not None
        logical_weights["e_cores"] = e_cores
        logical_weights["ecopy"] = encode_eplb_ecopy(g, R, e_cores)

    # 4. GDN linear attention: [qkv | z | b(64) | a(64)] (NLI=BA+128, A_OFF=BA+64)
    if ngdn > 0:
        lay_gdn = GDNBlock.lin_in_layout(g, ex)
        kh, vh, KD, VD = lay_gdn.kh, lay_gdn.vh, lay_gdn.KD, lay_gdn.VD
        b_pad = lay_gdn.A_OFF - lay_gdn.BA
        a_pad = lay_gdn.NLI - lay_gdn.A_OFF
        kq_w = kh * KD
        v_w = vh * VD

        lin_q = jnp.transpose(hf["lin_q"].reshape(ngdn, H, R, kq_w), (2, 0, 1, 3))
        lin_k = jnp.transpose(hf["lin_k"].reshape(ngdn, H, R, kq_w), (2, 0, 1, 3))
        lin_v = jnp.transpose(hf["lin_v"].reshape(ngdn, H, R, v_w), (2, 0, 1, 3))
        lin_z = jnp.transpose(hf["lin_z"].reshape(ngdn, H, R, v_w), (2, 0, 1, 3))
        lin_b = jnp.transpose(hf["lin_b"].reshape(ngdn, H, R, vh), (2, 0, 1, 3))
        lin_a = jnp.transpose(hf["lin_a"].reshape(ngdn, H, R, vh), (2, 0, 1, 3))
        if b_pad > vh:
            lin_b = jnp.pad(lin_b, ((0, 0), (0, 0), (0, 0), (0, b_pad - vh)))
        if a_pad > vh:
            lin_a = jnp.pad(lin_a, ((0, 0), (0, 0), (0, 0), (0, a_pad - vh)))
        lin_in = jnp.concatenate([lin_q, lin_k, lin_v, lin_z, lin_b, lin_a], axis=-1)
        assert lin_in.shape[-1] == lay_gdn.NLI, (lin_in.shape[-1], lay_gdn.NLI)

        lin_q_s = jnp.transpose(hf["lin_q_s"].reshape(ngdn, R, kq_w), (1, 0, 2))
        lin_k_s = jnp.transpose(hf["lin_k_s"].reshape(ngdn, R, kq_w), (1, 0, 2))
        lin_v_s = jnp.transpose(hf["lin_v_s"].reshape(ngdn, R, v_w), (1, 0, 2))
        lin_z_s = jnp.transpose(hf["lin_z_s"].reshape(ngdn, R, v_w), (1, 0, 2))
        lin_b_s = jnp.transpose(hf["lin_b_s"].reshape(ngdn, R, vh), (1, 0, 2))
        lin_a_s = jnp.transpose(hf["lin_a_s"].reshape(ngdn, R, vh), (1, 0, 2))
        if b_pad > vh:
            lin_b_s = jnp.pad(lin_b_s, ((0, 0), (0, 0), (0, b_pad - vh)))
        if a_pad > vh:
            lin_a_s = jnp.pad(lin_a_s, ((0, 0), (0, 0), (0, a_pad - vh)))
        lin_in_s = jnp.concatenate(
            [lin_q_s, lin_k_s, lin_v_s, lin_z_s, lin_b_s, lin_a_s], axis=-1
        )
        assert lin_in_s.shape[-1] == lay_gdn.NLI, (lin_in_s.shape[-1], lay_gdn.NLI)

        lin_out = jnp.transpose(hf["lin_out"].reshape(ngdn, R, v_w, H), (1, 0, 2, 3))
        lin_out_s = jnp.broadcast_to(hf["lin_out_s"][None, ...], (R, ngdn, H))

        c_full = hf["conv"]  # [ngdn, 4, 2*KH*KD + VH*VD]
        KH_KD = g.KH * KD
        c_q = jnp.transpose(
            c_full[:, :, :KH_KD].reshape(ngdn, 4, R, kq_w), (2, 0, 1, 3)
        )
        c_k = jnp.transpose(
            c_full[:, :, KH_KD : 2 * KH_KD].reshape(ngdn, 4, R, kq_w), (2, 0, 1, 3)
        )
        c_v = jnp.transpose(
            c_full[:, :, 2 * KH_KD :].reshape(ngdn, 4, R, v_w), (2, 0, 1, 3)
        )
        conv = jnp.concatenate([c_q, c_k, c_v], axis=-1).reshape(
            R, ngdn, 4 * lay_gdn.QKV
        )

        alog = jnp.transpose(hf["alog"].reshape(ngdn, R, vh), (1, 0, 2))
        dtb = jnp.transpose(hf["dtb"].reshape(ngdn, R, vh), (1, 0, 2))
        gnorm = jnp.broadcast_to(hf["gnorm"][None, ...], (R, ngdn, VD))

        logical_weights.update(
            {
                "lin_in": lin_in,
                "lin_in_s": lin_in_s,
                "lin_out": lin_out,
                "lin_out_s": lin_out_s,
                "conv": conv,
                "alog": alog,
                "dtb": dtb,
                "gnorm": gnorm,
            }
        )

    return logical_weights


def pack_inputs(
    pos: jax.Array,
    hidden: jax.Array,
    logical_weights: dict[str, Any],
    logical_states: dict[str, Any],
    g: Qwen35Geometry,
    ex: ExecSpec,
    *,
    mesh: Any = None,
    build_default_pool: bool = False,
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Pack logical model weights and states into the tiled/folded megakernel layout."""
    ms = qwen35_spec(g)
    p = planner.plan(ms, ex)
    R, B, T, NS, M = ex.R, ex.B, ex.T, ex.NS, ex.M
    L, ngdn = g.L, g.NGDN

    hid_pad = jnp.pad(hidden[:B], ((0, M - B), (0, 0)))
    hid_mk = jnp.tile(hid_pad[None, :, :], (R, 1, 1)).reshape(R * M, g.H)

    def tile_w(arr: jax.Array, kind: str) -> jax.Array:
        *lead, k_in, n_in = arr.shape
        nt_, tk_ = p.nt[kind], p.tk[kind]
        exp_n = p.kinds[kind][1]
        assert nt_ * tk_ == k_in, (kind, k_in, nt_, tk_)
        assert n_in == exp_n, (kind, n_in, exp_n)
        return arr.reshape((*lead, nt_, tk_, n_in))

    # Pack smalls [R, L, 1, S]
    lay = p.smalls_layout
    S = p.smalls_width
    sm = jnp.zeros((R, L, 1, S), dtype=F32)

    def set_sm_all(name: str, val: jax.Array):
        nonlocal sm
        if name not in lay:
            return
        o, w = lay[name]
        if val.ndim == 2:
            val = jnp.broadcast_to(val[None, :, :], (R, L, val.shape[-1]))
        assert val.shape[-1] <= w, (name, val.shape[-1], w)
        vw = min(w, val.shape[-1])
        sm = sm.at[:, :, 0, o : o + vw].set(val[:, :, :vw].astype(F32))

    def set_sm_layer(lyr: int, name: str, val_l: jax.Array):
        nonlocal sm
        if name not in lay:
            return
        o, w = lay[name]
        if val_l.ndim == 1:
            val_l = jnp.broadcast_to(val_l[None, :], (R, val_l.shape[-1]))
        assert val_l.shape[-1] <= w, (name, val_l.shape[-1], w)
        vw = min(w, val_l.shape[-1])
        sm = sm.at[:, lyr, 0, o : o + vw].set(val_l[:, :vw].astype(F32))

    def branch_slice(arr: jax.Array, lyr: int, ord_idx: int) -> jax.Array:
        return arr[:, lyr] if arr.shape[1] == L else arr[:, ord_idx]

    set_sm_all("in_norm", logical_weights["in_norm"])
    set_sm_all("post_norm", logical_weights["post_norm"])
    set_sm_all("sh_gate", logical_weights["sh_gate"])
    set_sm_all("sh13_s", logical_weights["sh13_s"])
    set_sm_all("sh2_s", logical_weights["sh2_s"])
    if "ecopy" in logical_weights:
        set_sm_all("ecopy", logical_weights["ecopy"])
    elif "e_cores" in logical_weights:
        set_sm_all("ecopy", encode_eplb_ecopy(g, R, logical_weights["e_cores"]))

    for lyr in range(L):
        is_att = (lyr + 1) % g.att_every == 0
        if is_att:
            a = lyr // g.att_every
            set_sm_layer(lyr, "in_s", branch_slice(logical_weights["att_in_s"], lyr, a))
            set_sm_layer(
                lyr, "out_s", branch_slice(logical_weights["att_out_s"], lyr, a)
            )
            set_sm_layer(lyr, "qn", branch_slice(logical_weights["qn"], lyr, a))
            set_sm_layer(lyr, "kn", branch_slice(logical_weights["kn"], lyr, a))
        else:
            o_gdn = lyr - (lyr + 1) // g.att_every
            set_sm_layer(
                lyr, "in_s", branch_slice(logical_weights["lin_in_s"], lyr, o_gdn)
            )
            set_sm_layer(
                lyr, "out_s", branch_slice(logical_weights["lin_out_s"], lyr, o_gdn)
            )
            set_sm_layer(lyr, "conv", branch_slice(logical_weights["conv"], lyr, o_gdn))
            set_sm_layer(lyr, "alog", branch_slice(logical_weights["alog"], lyr, o_gdn))
            set_sm_layer(lyr, "dtb", branch_slice(logical_weights["dtb"], lyr, o_gdn))
            set_sm_layer(
                lyr, "gnorm", branch_slice(logical_weights["gnorm"], lyr, o_gdn)
            )

    router_rep = jnp.broadcast_to(
        logical_weights["router"][None, ...], (R, L, g.H, g.E)
    )

    def _pack_expert(raw_e: Any, kind: str) -> Any:
        if ex.stack_e:
            if isinstance(raw_e, (tuple, list)):
                raw_e = jnp.stack(raw_e, axis=1)
            return tile_w(raw_e, kind)
        if isinstance(raw_e, (tuple, list)):
            return tuple(tile_w(raw_e[lyr], kind) for lyr in range(L))
        return tuple(tile_w(raw_e[:, lyr], kind) for lyr in range(L))

    e13_mk = _pack_expert(logical_weights["e13"], "e13")
    e2_mk = _pack_expert(logical_weights["e2"], "e2")

    es_mk = jnp.concatenate(
        [logical_weights["e13_s"], logical_weights["e2_s"]], axis=-1
    )[:, :, :, None, :].astype(F32)
    fn_mk = jnp.broadcast_to(
        logical_weights["final_norm"][None, None, :], (R, 1, g.H)
    ).astype(F32)

    weights_mk: dict[str, Any] = {}
    if ngdn > 0:
        weights_mk["lin_in"] = tile_w(logical_weights["lin_in"], "lin_in")
        weights_mk["lin_out"] = tile_w(logical_weights["lin_out"], "lin_out")
    weights_mk.update(
        {
            "att_in": tile_w(logical_weights["att_in"], "att_in"),
            "att_out": tile_w(logical_weights["att_out"], "att_out"),
            "router": tile_w(router_rep, "router"),
            "sh13": tile_w(logical_weights["sh13"], "sh13"),
            "sh2": tile_w(logical_weights["sh2"], "sh2"),
            "e13": e13_mk,
            "e2": e2_mk,
            "es": es_mk,
            "smalls": sm,
            "final_norm": fn_mk,
        }
    )

    states_mk: dict[str, Any] = {}
    if ngdn > 0 and "conv" in logical_states and "rec" in logical_states:
        conv_l = logical_states["conv"]
        rec_l = logical_states["rec"]
        qkv = g.QKV(R)
        vh = g.vh(R)
        if conv_l.ndim == 5 and conv_l.shape[3] == 3:
            if T == 1:
                cr = rup(3 * B, 8)
                conv_3b = jnp.swapaxes(conv_l, 2, 3).reshape(R, ngdn, 3 * B, qkv)
                conv_p = jnp.pad(conv_3b, ((0, 0), (0, 0), (0, cr - 3 * B), (0, 0)))
            else:
                hr = rup(3 + T, 8)
                conv_p = jnp.pad(conv_l, ((0, 0), (0, 0), (0, 0), (0, hr - 3), (0, 0)))
        else:
            conv_p = conv_l
        if rec_l.ndim == 6:
            rec_p = rec_l.reshape(R, ngdn, NS * vh, g.KD, g.VD)
        else:
            rec_p = rec_l
        states_mk["conv"] = kernel.fold(conv_p.astype(F32))
        states_mk["rec"] = kernel.fold(rec_p.astype(BF16))

    if cache_exec(ex).paged_kv:
        if POOL_STATE_NAME in logical_states and "page_table" in logical_states:
            states_mk[POOL_STATE_NAME] = tuple(
                kernel.fold(r) if r.ndim == 6 else r
                for r in logical_states[POOL_STATE_NAME]
            )
            pt_in = logical_states["page_table"]
            max_pt = resolved_max_pages_per_seq(ex)
            if pt_in.ndim == 2:
                if pt_in.shape[0] not in (NS, R * NS) or pt_in.shape[1] < max_pt:
                    raise ValueError(
                        f"page_table has shape {pt_in.shape}, expected "
                        f"({NS}, >={max_pt}) or ({R * NS}, >={max_pt})"
                    )
                pt_in = pt_in[:, :max_pt]
                states_mk["page_table"] = (
                    pt_in if pt_in.shape[0] == R * NS else jnp.tile(pt_in, (R, 1))
                )
            elif pt_in.ndim == 3:
                if pt_in.shape[:2] != (R, NS) or pt_in.shape[2] < max_pt:
                    raise ValueError(
                        f"page_table has shape {pt_in.shape}, expected "
                        f"({R}, {NS}, >={max_pt})"
                    )
                states_mk["page_table"] = kernel.fold(pt_in[:, :, :max_pt])
            else:
                raise ValueError(
                    f"page_table must be 2D or 3D, got shape {pt_in.shape}"
                )
        elif build_default_pool:
            pt, ckpt, roff, _ = qwen35_pool.default_pool_layout(g, ex)
            raw_pools = qwen35_pool.states_to_pool(
                g, ex, logical_states, pt, ckpt, roff
            )
            states_mk[POOL_STATE_NAME] = raw_pools
            states_mk["page_table"] = jnp.tile(
                pt[:, : resolved_max_pages_per_seq(ex)], (R, 1)
            )
        else:
            raise ValueError(
                f"pack_inputs with paged_kv=True requires {POOL_STATE_NAME!r} "
                "and 'page_table' in logical_states (or pass build_default_pool=True)."
            )
    elif "kc" in logical_states and "vc" in logical_states:
        states_mk["kc"] = kernel.fold(logical_states["kc"])
        states_mk["vc"] = kernel.fold(logical_states["vc"])

    if mesh is not None:
        rep = NamedSharding(mesh, P())
        tp = NamedSharding(mesh, P("tp"))
        pos = jax.device_put(pos, rep)
        hid_mk = jax.device_put(hid_mk, tp)
        weights_mk = {
            k: tuple(jax.device_put(x, tp) for x in v)
            if isinstance(v, tuple)
            else jax.device_put(v, tp)
            for k, v in weights_mk.items()
        }
        states_mk = {
            k: tuple(jax.device_put(x, tp) for x in v)
            if isinstance(v, tuple)
            else jax.device_put(v, tp)
            for k, v in states_mk.items()
        }

    return pos, hid_mk, weights_mk, states_mk


def random_inputs(
    key: jax.Array,
    g: Qwen35Geometry,
    ex: ExecSpec,
    *,
    mesh: Any = None,
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Generate packed megakernel inputs `(pos, hidden, weights, states)`."""
    pos, hid_l, w_l, st_l = random_logical_inputs(key, g, ex)
    return pack_inputs(
        pos,
        hid_l,
        w_l,
        st_l,
        g,
        ex,
        mesh=mesh,
        build_default_pool=cache_exec(ex).paged_kv,
    )


def reference_step(
    g: Qwen35Geometry,
    ex: ExecSpec,
    pos: jax.Array,
    hidden: jax.Array,
    logical_weights: dict[str, Any],
    logical_states: dict[str, Any],
    *,
    paged_kv: bool | None = None,
    pool_io: bool = False,
) -> tuple[jax.Array, dict[str, Any]]:
    """Pure-JAX multi-rank reference oracle composing block `reference()` hooks."""
    R, B, T, NS = ex.R, ex.B, ex.T, ex.NS
    L, ngdn = g.L, g.NGDN
    lay_gdn = GDNBlock.lin_in_layout(g, ex)
    vh, KD, VD = lay_gdn.vh, lay_gdn.KD, lay_gdn.VD
    QKV = lay_gdn.QKV
    CR, HR = rup(3 * B, 8), rup(3 + T, 8)
    use_paged_kv = cache_exec(ex).paged_kv if paged_kv is None else bool(paged_kv)

    in_norm_blk = Norm(
        "input_layernorm",
        geometry=g,
        zero_centered=g.zero_centered_norm,
        eps=g.eps,
    )
    post_norm_blk = Norm(
        "post_attention_layernorm",
        geometry=g,
        zero_centered=g.zero_centered_norm,
        eps=g.eps,
    )
    final_norm_blk = Norm(
        "model.norm",
        geometry=g,
        zero_centered=g.zero_centered_norm,
        eps=g.eps,
    )
    gqa_blk = GQABlock(geometry=g)
    gdn_blk = GDNBlock(geometry=g)
    moe_blk = MoEBlock(geometry=g)

    kc_all = logical_states["kc"]  # [R, NATT, NS, ctx, HD]
    vc_all = logical_states["vc"]  # [R, NATT, NS, ctx, HD]
    conv_in = logical_states.get("conv", jnp.zeros((R, ngdn, NS, 3, QKV), dtype=F32))
    if pool_io:
        conv_in = conv_in.astype(BF16).astype(F32)
    rec_in = logical_states.get("rec", jnp.zeros((R, ngdn, NS, vh, KD, VD), dtype=BF16))
    if rec_in.ndim == 5:
        rec_in = rec_in.reshape(R, ngdn, NS, vh, KD, VD)
    conv_out = (
        jnp.zeros((R, ngdn, CR, QKV), dtype=F32)
        if T == 1
        else jnp.zeros((R, ngdn, NS, HR, QKV), dtype=F32)
    )
    rec_out = jnp.zeros((R, ngdn, B * vh, KD, VD), dtype=BF16)
    res = hidden[:B].astype(F32)

    step_state: dict[str, Any] = {
        "kc": kc_all,
        "vc": vc_all,
        "conv": conv_in,
        "rec": rec_in,
        "conv_out": conv_out,
        "rec_out": rec_out,
    }

    for l in range(L):  # noqa: E741
        is_att = (l + 1) % g.att_every == 0
        x_att, _ = in_norm_blk.reference(logical_weights["in_norm"][l], res, g=g)
        if is_att:
            a = l // g.att_every
            att_delta, step_state = gqa_blk.reference(
                logical_weights,
                x_att,
                step_state,
                g=g,
                ex=ex,
                pos=pos,
                att_idx=a,
                layer_idx=l,
                paged_kv=use_paged_kv,
            )
            res = rnd(res + att_delta)
        else:
            o_gdn = l - (l + 1) // g.att_every
            gdn_delta, step_state = gdn_blk.reference(
                logical_weights,
                x_att,
                step_state,
                g=g,
                ex=ex,
                gdn_idx=o_gdn,
                layer_idx=l,
                multi_layer=True,
            )
            res = rnd(res + gdn_delta)

        x_moe, _ = post_norm_blk.reference(logical_weights["post_norm"][l], res, g=g)
        moe_delta, step_state = moe_blk.reference(
            logical_weights,
            x_moe,
            step_state,
            g=g,
            ex=ex,
            layer_idx=l,
        )
        res = rnd(res + moe_delta)

    out_b, _ = final_norm_blk.reference(
        logical_weights["final_norm"], res, g=g, out_dtype=BF16
    )
    conv_out = step_state["conv_out"]
    rec_out = step_state["rec_out"]
    if pool_io:
        conv_out = conv_out.astype(BF16).astype(F32)
    out_states: dict[str, Any] = {"kc": step_state["kc"], "vc": step_state["vc"]}
    if ngdn > 0:
        if ex.VER:
            conv_pad = (
                jnp.pad(
                    conv_in,
                    ((0, 0), (0, 0), (0, 0), (0, HR - conv_in.shape[-2]), (0, 0)),
                )
                if conv_in.ndim == 5 and conv_in.shape[-2] < HR
                else conv_in
            )
            out_states["conv"] = conv_pad
            out_states["rec"] = rec_in.reshape(R, ngdn, NS * vh, KD, VD)
            out_states["conv_ckpt"] = conv_out
            out_states["rec_ckpt"] = rec_out
        else:
            out_states["conv"] = conv_out
            out_states["rec"] = rec_out
    return out_b, out_states
