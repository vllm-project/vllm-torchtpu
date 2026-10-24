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
"""Whole-step prototype: two `[GDN, GDN, GDN, QSA]` groups chained in one kernel (CPU).

Every QSA layer is a full in-kernel layer (HC mix, projection, norms, RoPE,
sparse attention, gate, o_proj). The reference chains the library's block
references with `project_reference` + `reference_rows` for the QSA layers.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import kernel
from vllm_torchtpu.megakernel.blocks.gdn import GDNBlock
from vllm_torchtpu.megakernel.blocks.gdn.reference import gdn_reference
from vllm_torchtpu.megakernel.blocks.hc import RESIDUAL_OUT
from vllm_torchtpu.megakernel.blocks.hc.reference import hc_combine, hc_mix
from vllm_torchtpu.megakernel.blocks.moe.reference import moe_reference
from vllm_torchtpu.megakernel.blocks.qsa_attn.block import MICRO, reference_rows
from vllm_torchtpu.megakernel.exec_spec import ExecSpec, Tiles
from vllm_torchtpu.megakernel.models import qwen38
from vllm_torchtpu.megakernel.models.qwen38_inputs import (
    _hc_params,
    _rep,
    _rows,
    hc_weight_arrays,
    random_segment_inputs,
)
from vllm_torchtpu.megakernel.models.qwen38_pack import pack_weights, shard_step_inputs
from vllm_torchtpu.megakernel.ops import BF16, F32, rup
from vllm_torchtpu.megakernel.wholestep.qsa_full import (
    KV_NEW,
    project_reference,
    qkvg_width,
)
from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec

_TILES = Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)
G = qwen38.TINY
BUDGET, PAGES, OFF, L = 64, 8, 1, 8


def _inputs(key: int, ex: ExecSpec, router_gain: float = 4.0) -> dict[str, Any]:
    R, B = ex.R, ex.B
    nh, hd = G.NH // R, G.HD
    # Router gain: eight chained layers drift enough for a near-tied top-k router logit to
    # flip one row's expert set (row 1 diverged by layer 7 at gain 1); the library tests use
    # the same knob for the same reason.
    inp = random_segment_inputs(jax.random.PRNGKey(key), G, ex, L, router_gain=router_gain)
    k = jax.random.split(jax.random.PRNGKey(key + 1), 8)
    n_att, n_gdn = L // G.att_every, L - L // G.att_every
    hc = {
        "gdn": {n: v[:n_gdn] for n, v in inp.hc["gdn"].items()},
        "qsa_full": _hc_params(k[0], G, n_att),
        "moe": inp.hc["moe"],
    }
    qkvg = (jax.random.normal(k[1], (R, n_att, G.H, qkvg_width(nh, hd)), F32) * 0.06).astype(BF16)
    o_proj = (jax.random.normal(k[2], (R, n_att, nh * hd, G.H), F32) * 0.05).astype(BF16)
    qn = jax.random.normal(k[3], (n_att, hd), F32) * 0.05
    kn = jax.random.normal(k[4], (n_att, hd), F32) * 0.05
    rng = np.random.default_rng(key)
    pool = jnp.asarray(rng.standard_normal((R, PAGES, 128, 1, 4, hd)), F32).astype(
        jnp.float8_e4m3fn
    )
    pages = jnp.asarray(rng.integers(0, PAGES * 128 // MICRO, size=(ex.M, BUDGET + 1)), jnp.int32)
    kv_len = jnp.asarray(([0, 1, 9, 100, 255, 257, 130, 64] + [0] * ex.M)[: ex.M], jnp.int32)
    pos = rng.integers(0, 4096, size=(ex.M,))
    inv = 1.0 / (G.theta ** (np.arange(0, G.ROT, 2) / G.ROT))
    ang = pos[:, None] * inv[None, :]
    rope = np.zeros((ex.M, 128), np.float32)
    rope[:, : G.ROT // 2] = np.asarray(jnp.asarray(np.cos(ang), BF16).astype(F32))
    rope[:, G.ROT // 2 : G.ROT] = np.asarray(jnp.asarray(np.sin(ang), BF16).astype(F32))
    return dict(
        inp=inp, hc=hc, qkvg=qkvg, o_proj=o_proj, qn=qn, kn=kn, pool=pool,
        pages=pages, kv_len=kv_len, rope=jnp.asarray(rope),
    )


def _reference(ex: ExecSpec, d: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
    R, B, nhc, eps = ex.R, ex.B, G.NHC, G.eps
    nh, hd = G.NH // R, G.HD
    gi = G.inner(R, L)
    inp, hc = d["inp"], d["hc"]
    x = inp.x[:B]
    o_att = o_gdn = 0
    kv_new = []
    for j in range(L):
        if j % G.att_every == G.att_every - 1:
            p = {n: v[o_att] for n, v in hc["qsa_full"].items()}
            mixed, xn = hc_mix(x, p, nhc=nhc, eps=eps)
            delta, kvs = 0.0, []
            for r in range(R):
                q, gate, kk, vv = project_reference(
                    mixed, d["qkvg"][r, o_att], d["qn"][o_att], d["kn"][o_att], d["rope"][:B],
                    nh=nh, hd=hd, rot=G.ROT, eps=eps,
                )
                kvs.append(jnp.concatenate([kk, vv], axis=-1))
                attn = reference_rows(
                    q, gate, d["pool"][r], d["pages"][:B], d["kv_len"][:B],
                    off=OFF, scale_k=1.0, scale_v=1.0, sm_scale=hd**-0.5,
                ).reshape(B, -1)
                delta = delta + jnp.dot(
                    attn.astype(BF16), d["o_proj"][r, o_att], preferred_element_type=F32
                )
            kv_new.append(jnp.stack(kvs))
            o_att += 1
        else:
            p = {n: v[o_gdn] for n, v in hc["gdn"].items()}
            mixed, xn = hc_mix(x, p, nhc=nhc, eps=eps)
            delta, _ = gdn_reference(
                inp.w_l, mixed, inp.st_l, g=gi, ex=ex, gdn_idx=o_gdn, layer_idx=j
            )
            o_gdn += 1
        x = hc_combine(delta, x, xn, p, nhc=nhc)
        p = {n: v[j] for n, v in hc["moe"].items()}
        mixed, xn = hc_mix(x, p, nhc=nhc, eps=eps)
        delta, _ = moe_reference(inp.w_l, mixed, None, g=gi, ex=ex, layer_idx=j)
        x = hc_combine(delta, x, xn, p, nhc=nhc)
    return x, jnp.stack(kv_new, axis=1)  # [R, n_att, B, 2*hd]


def _pack(spec: Any, ex: ExecSpec, d: dict[str, Any]) -> tuple[dict, dict]:
    R, B, T = ex.R, ex.B, ex.T
    inp, hc, w_l, st_l = d["inp"], d["hc"], d["inp"].w_l, d["inp"].st_l
    att_rows = tuple(j for j in range(L) if j % G.att_every == G.att_every - 1)
    gdn_rows = tuple(j for j in range(L) if j not in att_rows)
    weights = {
        "router": _rep(w_l["router"], R),
        "sh13": w_l["sh13"],
        "sh2": w_l["sh2"],
        "e13": tuple(w_l["e13"][:, j] for j in range(L)),
        "e2": tuple(w_l["e2"][:, j] for j in range(L)),
        "es": jnp.concatenate([w_l["e13_s"], w_l["e2_s"]], axis=-1)[:, :, :, None, :],
        **hc_weight_arrays("moe", hc["moe"], G, R, mix=True),
        "qkvg": d["qkvg"],
        "o_proj": d["o_proj"],
        **hc_weight_arrays("qsa_full", hc["qsa_full"], G, R, mix=True),
        "lin_in": w_l["lin_in"],
        "lin_out": w_l["lin_out"],
        **hc_weight_arrays("gdn", hc["gdn"], G, R, mix=True),
    }
    smalls = {
        "moe_hcn": hc["moe"]["norm"],
        **{k: w_l[k] for k in ("post_norm", "sh_gate", "sh13_s", "sh2_s", "ecopy") if k in w_l},
        "qsa_full_hcn": _rows(_rep(hc["qsa_full"]["norm"], R), att_rows, L),
        "qsa_qn": _rows(_rep(d["qn"], R), att_rows, L),
        "qsa_kn": _rows(_rep(d["kn"], R), att_rows, L),
        "gdn_hcn": _rows(_rep(hc["gdn"]["norm"], R), gdn_rows, L),
        "in_s": _rows(w_l["lin_in_s"], gdn_rows, L),
        "out_s": _rows(w_l["lin_out_s"], gdn_rows, L),
        **{k: _rows(w_l[k], gdn_rows, L) for k in ("conv", "alog", "dtb", "gnorm")},
    }
    n_gdn = len(gdn_rows)
    lay = GDNBlock.lin_in_layout(G.inner(R, L), ex)
    hr = rup(3 + T, 8)
    conv = jnp.pad(st_l["conv"], ((0, 0),) * 3 + ((0, hr - 3), (0, 0)))
    states = {
        "conv": conv.astype(F32),
        "rec": st_l["rec"].reshape(R, n_gdn, ex.NS * lay.vh, lay.KD, lay.VD).astype(BF16),
    }
    return pack_weights(spec, ex, weights, smalls), states


def _run(key: int = 11, T: int = 4):
    R = 2
    ex = ExecSpec(R=R, B=2 * T, T=T, ctx=128, interpret=True, tiles=_TILES)
    d = _inputs(key, ex)
    ref, kv_ref = _reference(ex, d)
    spec = wholestep_spec(G, R, L=L, qsa=dict(budget=BUDGET))
    w_mk, states = _pack(spec, ex, d)
    mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
    pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states)
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    names = [x.name for x in kernel.extra_inputs(spec, ex)]
    assert names == [
        "qsa_pool", "qsa_pages", "qsa_kvlen", "qsa_off", "qsa_scales", "qsa_rope"
    ], names
    extras = (
        jax.device_put(kernel.fold(d["pool"]), tp),
        jax.device_put(d["pages"], rep),
        jax.device_put(d["kv_len"], rep),
        jax.device_put(jnp.asarray([OFF], jnp.int32), rep),
        jax.device_put(jnp.asarray([1.0, 1.0], F32), rep),
        jax.device_put(kernel.fold(_rep(d["rope"], R)), tp),  # HBM extras are per rank
    )
    step, _ = kernel.make_step(mesh, spec, ex)
    _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *extras)
    B = ex.B
    res = np.asarray(outs[RESIDUAL_OUT].reshape(R, ex.M, -1)[:, :B], np.float32)
    kv = np.asarray(outs[KV_NEW], np.float32).reshape(R, L // G.att_every, ex.M, -1)[:, :, :B]
    return res, np.asarray(ref, np.float32), np.asarray(d["inp"].x[:B], np.float32), kv, np.asarray(
        kv_ref, np.float32
    )


def _assert_within_budget(res: np.ndarray, ref: np.ndarray) -> None:
    scale = float(np.abs(ref).mean())
    for r in range(res.shape[0]):
        d = np.abs(res[r] - ref)
        assert d.mean() < 1e-2 * scale, (r, d.mean(), scale)
        assert np.quantile(d, 0.999) < 5e-2 * scale, (r, np.quantile(d, 0.999), scale)
        assert d.max() < 1e-1 * scale, (r, d.max(), scale)


def test_two_groups_match_reference():
    res, ref, x, kv, kv_ref = _run()
    assert float(np.std(ref - x)) > 0.05
    _assert_within_budget(res, ref)
    # The step's K/V rows (cache-write payload) match per rank and per QSA layer.
    np.testing.assert_allclose(kv, kv_ref, rtol=2e-2, atol=2e-2)
