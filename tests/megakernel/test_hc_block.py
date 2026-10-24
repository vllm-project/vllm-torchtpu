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
"""`HCSublayer`: reference vs the PyTorch definition, and kernel vs reference."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.models.qwen38_pack import pack_weights, shard_step_inputs
from vllm_torchtpu.megakernel.modules.hc import (
    RESIDUAL_OUT,
    HCSublayer,
    hc_combine,
    hc_mix,
)
from vllm_torchtpu.megakernel.modules.moe import MoEModule
from vllm_torchtpu.megakernel.ops import BF16, F32
from vllm_torchtpu.megakernel.spec import LayerType, ModelSpec

NHC, H_SUB, LOWRANK = 4, 256, 64

_TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)


@dataclasses.dataclass(frozen=True)
class _ResidualGeometry:
    """Core geometry of an HC model: the residual is all `NHC` streams."""

    L: int = 1
    H: int = NHC * H_SUB
    eps: float = 1e-6
    zero_centered_norm: bool = True


def _hc_params(key: jax.Array, L: int) -> dict[str, jax.Array]:
    k = jax.random.split(key, 4)
    W = NHC * H_SUB
    return {
        "norm": jax.random.normal(k[0], (L, W), F32) * 0.05,
        "down": (jax.random.normal(k[1], (L, W, LOWRANK), F32) * 0.03).astype(BF16),
        "up": (jax.random.normal(k[2], (L, LOWRANK, W), F32) * 0.1).astype(BF16),
        "inj": (jax.random.normal(k[3], (L, W, NHC), F32) * 0.3).astype(BF16),
    }


def _hc_weight_arrays(name: str, hc: dict[str, jax.Array], R: int) -> dict[str, Any]:
    blk = HCSublayer(
        MoEModule(),
        inner_geometry=dataclasses.replace(qwen35.TINY, H=H_SUB),
        nhc=NHC,
        lowrank=LOWRANK,
    )
    L, W = hc["down"].shape[0], NHC * H_SUB
    hdn = jnp.zeros((L, W, blk.n_down), BF16)
    hdn = (
        hdn.at[:, :, :LOWRANK]
        .set(hc["down"])
        .at[:, :, LOWRANK : LOWRANK + NHC]
        .set(hc["inj"])
    )
    hup = jnp.zeros((L, blk.k_up, W), BF16).at[:, :LOWRANK].set(hc["up"])

    def rep(x):
        return jnp.broadcast_to(x[None], (R, *x.shape))

    out = {f"{name}_hdn": rep(hdn)}
    for b in range(NHC):
        out[f"{name}_hup{b}"] = rep(hup[..., b * H_SUB : (b + 1) * H_SUB])
    return out


def _torch_gated_residual(x, norm_w, down, up, inj, block, nhc, eps):
    """The checkpoint's decode-path GatedResidual mix + combine, verbatim in torch."""
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    h = x.shape[-1] // nhc
    flat = x.float()
    parts = []
    for b in range(nhc):
        v = flat[..., b * h : (b + 1) * h]
        var = v.square().mean(dim=-1, keepdim=True)
        parts.append(v * torch.rsqrt(var + eps))
    norm = (torch.cat(parts, dim=-1) * (1.0 + norm_w.float())).to(x.dtype)
    gate = F.linear(norm, down) / nhc
    gate = F.silu(gate.float()).to(gate.dtype)
    gate = F.linear(gate, up)
    gate = torch.sigmoid(gate.float())
    weighted = gate * norm.float()
    mixed = weighted[..., :h]
    for b in range(1, nhc):
        mixed = mixed + weighted[..., b * h : (b + 1) * h]
    mixed = (mixed / nhc).to(x.dtype)
    injection = 2.0 * torch.sigmoid(F.linear(norm, inj).float() / nhc)
    out = torch.cat(
        [
            x.float()[..., b * h : (b + 1) * h]
            + block.float() * injection[..., b : b + 1]
            for b in range(nhc)
        ],
        dim=-1,
    ).to(x.dtype)
    return mixed, out


def test_hc_reference_matches_torch_definition():
    torch = pytest.importorskip("torch")
    T, W = 8, NHC * H_SUB
    key = jax.random.PRNGKey(0)
    hc = {k: v[0] for k, v in _hc_params(key, 1).items()}
    x = jax.random.normal(jax.random.PRNGKey(1), (T, W), F32).astype(BF16)
    block = jax.random.normal(jax.random.PRNGKey(2), (T, H_SUB), F32).astype(BF16)

    def tt(a, dt=torch.bfloat16):
        return torch.from_numpy(np.asarray(a, np.float32).copy()).to(dt)

    mixed_t, out_t = _torch_gated_residual(
        tt(x),
        tt(hc["norm"], torch.float32),
        tt(hc["down"]).T.contiguous(),
        tt(hc["up"]).T.contiguous(),
        tt(hc["inj"]).T.contiguous(),
        tt(block),
        NHC,
        1e-6,
    )
    mixed, xn = hc_mix(x, hc, nhc=NHC, eps=1e-6)
    out = hc_combine(block, x, xn, hc, nhc=NHC)
    np.testing.assert_allclose(
        np.asarray(mixed), mixed_t.float().numpy(), atol=1e-2, rtol=1e-2
    )
    np.testing.assert_allclose(
        np.asarray(out), out_t.float().numpy(), atol=1e-2, rtol=1e-2
    )
    assert float(np.mean(np.asarray(mixed) == mixed_t.float().numpy())) > 0.98


def _run_hc_moe(T: int, hc_cls: type = HCSublayer) -> tuple[np.ndarray, np.ndarray]:
    R, B = 2, 2 * T
    ex = ExecSpec(R=R, B=B, T=T, ctx=128, interpret=True, tiles=_TILES)
    gi = dataclasses.replace(
        qwen35.TINY, L=1, att_every=1, H=H_SUB, IS=256, moe_tp=False
    )
    g = _ResidualGeometry()
    pos, _, w_l, _ = qwen35.random_logical_inputs(jax.random.PRNGKey(30), gi, ex)
    # Library test scales give a ~5e-5 MoE delta; scale the output projections so
    # the sublayer is visible next to the residual.
    w_l = {**w_l, "e2_s": w_l["e2_s"] * 2000.0, "sh2_s": w_l["sh2_s"] * 2000.0}
    hc = _hc_params(jax.random.PRNGKey(31), 1)
    x = jax.random.normal(jax.random.PRNGKey(32), (B, g.H), F32).astype(BF16)

    moe = MoEModule(geometry=gi, moe_tp=False)
    hc_l0 = {k: v[0] for k, v in hc.items()}
    mixed, xn = hc_mix(x, hc_l0, nhc=NHC, eps=g.eps)
    delta, _ = moe.reference(w_l, mixed, None, g=gi, ex=ex, layer_idx=0)
    ref = hc_combine(delta, x, xn, hc_l0, nhc=NHC)
    # The positive control needs distinct per-stream gates and a visible delta.
    inj = 2.0 * jax.nn.sigmoid(jnp.dot(xn, hc_l0["inj"].astype(F32)) / NHC)
    assert float(jnp.std(inj[:, 0] - inj[:, -1])) > 0.05
    assert float(jnp.std(delta)) > 0.05

    blk = hc_cls(moe, inner_geometry=gi, nhc=NHC, lowrank=LOWRANK, emit_residual=True)
    spec = ModelSpec(
        name="hc_moe", config=g, layer_type=LayerType("hc_moe", slots=(blk,))
    )
    weights = {
        "router": jnp.broadcast_to(w_l["router"][None], (R, *w_l["router"].shape)),
        "sh13": w_l["sh13"],
        "sh2": w_l["sh2"],
        "e13": tuple(w_l["e13"][:, lyr] for lyr in range(gi.L)),
        "e2": tuple(w_l["e2"][:, lyr] for lyr in range(gi.L)),
        "es": jnp.concatenate([w_l["e13_s"], w_l["e2_s"]], axis=-1)[:, :, :, None, :],
        **_hc_weight_arrays(blk.name, hc, R),
    }
    smalls = {
        k: w_l[k]
        for k in ("post_norm", "sh_gate", "sh13_s", "sh2_s", "ecopy")
        if k in w_l
    }
    smalls[f"{blk.name}_hcn"] = hc["norm"]
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    w_mk = pack_weights(spec, ex, weights, smalls)
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(mesh, ex, pos, x, w_mk, {})
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk)
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, g.H)[:, :B], np.float32)
    return res, np.asarray(ref, np.float32)


@pytest.mark.parametrize("T", [1, 2])
def test_hc_moe_kernel_matches_reference(T: int):
    res, ref = _run_hc_moe(T)
    for r in range(res.shape[0]):
        np.testing.assert_allclose(res[r], ref, atol=2e-2, rtol=2e-2)


class _SwappedInjection(HCSublayer):
    """Fault: injection gates applied to the streams in reverse order."""

    def _set_injection(self, ctx, y):
        super()._set_injection(ctx, y[:, : self.nhc][:, ::-1])


def test_hc_moe_fault_is_caught():
    res, ref = _run_hc_moe(1, _SwappedInjection)
    assert float(np.max(np.abs(res[0] - ref))) > 2e-2
