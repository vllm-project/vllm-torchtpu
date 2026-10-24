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
"""Random whole-step inputs, their packing, and the chained pure-JAX reference.

Shared by the CPU kernel-vs-reference test and the on-TPU whole-step benchmark.
GDN / MoE weights and states come from `random_segment_inputs`; the QSA layers
add the fused `q|gate|k|v` projection, `o_proj`, q / k norms, an FP8 paged pool
with per-row micro-page ids and lengths, and per-row RoPE rows.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .. import kernel
from ..blocks.gdn import GDNBlock
from ..blocks.gdn.reference import gdn_reference
from ..blocks.hc.reference import hc_combine, hc_mix
from ..blocks.moe.reference import moe_reference
from ..blocks.qsa_attn.block import MICRO, reference_rows
from ..exec_spec import ExecSpec
from ..models.qwen38 import Qwen38Geometry
from ..models.qwen38_inputs import _hc_params, _rep, _rows, hc_weight_arrays, random_segment_inputs
from ..models.qwen38_pack import pack_weights
from ..ops import BF16, F32, rmsnorm, rnd, rup
from .qsa_full import project_reference, qkvg_width

EXTRA_NAMES = ["qsa_pool", "qsa_pages", "qsa_kvlen", "qsa_off", "qsa_scales", "qsa_rope"]


def att_layers(g: Qwen38Geometry, L: int, first: str = "gdn") -> tuple[int, ...]:
    """Kernel layers that are QSA layers: the last of each cycle, or the first (`first="qsa"`)."""
    at = 0 if first == "qsa" else g.att_every - 1
    return tuple(j for j in range(L) if j % g.att_every == at)


def random_wholestep_inputs(
    key: int,
    g: Qwen38Geometry,
    ex: ExecSpec,
    L: int,
    *,
    budget: int,
    n_pages: int,
    kv_len: list[int] | None = None,
    off: int = 1,
    router_gain: float = 4.0,
    zero: tuple[str, ...] = (),
    identical_experts: bool = False,
    gain: float = 1.0,
    first: str = "gdn",
    vocab_shard: int = 0,
) -> dict[str, Any]:
    """`kv_len` per row (default: `4 * budget + 3`, the full selection plus a partial group)."""
    R = ex.R
    nh, hd = g.NH // R, g.HD
    inp = random_segment_inputs(
        jax.random.PRNGKey(key), g, ex, L, router_gain=router_gain, zero=zero, gain=gain
    )
    if identical_experts:
        # Every routed expert of a layer is the same expert, so a 1-ulp router flip cannot
        # change the output: the strict half of the MoE parity gate.
        for n in ("e13", "e13_s", "e2", "e2_s"):
            w = inp.w_l[n]
            inp.w_l[n] = jnp.broadcast_to(w[0:1, :, 0:1], w.shape)
    k = jax.random.split(jax.random.PRNGKey(key + 1), 8)
    n_att = len(att_layers(g, L, first))
    n_gdn = L - n_att
    hc = {
        "gdn": {n: v[:n_gdn] for n, v in inp.hc["gdn"].items()},
        "qsa_full": _hc_params(k[0], g, n_att),
        "moe": inp.hc["moe"],
    }
    qkvg = (jax.random.normal(k[1], (R, n_att, g.H, qkvg_width(nh, hd)), F32) * 0.06).astype(BF16)
    o_proj = (jax.random.normal(k[2], (R, n_att, nh * hd, g.H), F32) * 0.05 * gain).astype(BF16)
    qn = jax.random.normal(k[3], (n_att, hd), F32) * 0.05
    kn = jax.random.normal(k[4], (n_att, hd), F32) * 0.05
    rng = np.random.default_rng(key)
    pool = jax.random.normal(k[5], (R, n_pages, 128, 1, 4, hd), F32).astype(jnp.float8_e4m3fn)
    pages = jnp.asarray(
        rng.permutation(n_pages * 128 // MICRO)[: ex.M * (budget + 1)].reshape(ex.M, budget + 1)
        if n_pages * 128 // MICRO >= ex.M * (budget + 1)
        else rng.integers(0, n_pages * 128 // MICRO, size=(ex.M, budget + 1)),
        jnp.int32,
    )
    if kv_len is None:
        kv_len = [4 * budget + 3] * ex.B
    kvl = jnp.asarray((list(kv_len) + [0] * ex.M)[: ex.M], jnp.int32)
    pos = rng.integers(0, 1 << 17, size=(ex.M,))
    inv = 1.0 / (g.theta ** (np.arange(0, g.ROT, 2) / g.ROT))
    ang = pos[:, None] * inv[None, :]
    rope = np.zeros((ex.M, 128), np.float32)  # host-only (traceable under jit): ml_dtypes BF16
    rope[:, : g.ROT // 2] = np.cos(ang).astype(ml_dtypes.bfloat16).astype(np.float32)
    rope[:, g.ROT // 2 : g.ROT] = np.sin(ang).astype(ml_dtypes.bfloat16).astype(np.float32)
    out = dict(
        inp=inp, hc=hc, qkvg=qkvg, o_proj=o_proj, qn=qn, kn=kn, pool=pool, pages=pages,
        kv_len=kvl, rope=jnp.asarray(rope), off=off, budget=budget, first=first,
    )
    if vocab_shard:  # the decode tail after the last layer (final HC mixer + lm_head)
        out["hc"]["tail"] = _hc_params(k[6], g, 1)
        out["lm_head"] = (jax.random.normal(k[7], (R, g.H, vocab_shard), F32) * 0.05).astype(BF16)
    return out


def wholestep_reference(
    g: Qwen38Geometry, ex: ExecSpec, d: dict[str, Any], L: int
) -> tuple[jax.Array, jax.Array]:
    """`(residual [B, NHC*H], K/V rows [R, n_att, B, 2*HD])`."""
    R, B, nhc, eps = ex.R, ex.B, g.NHC, g.eps
    nh, hd = g.NH // R, g.HD
    gi = g.inner(R, L)
    inp, hc = d["inp"], d["hc"]
    x = inp.x[:B]
    o_att = o_gdn = 0
    kv_new = []
    step_out: list = []
    d["_step_out"] = step_out
    att = set(att_layers(g, L, d.get("first", "gdn")))
    for j in range(L):
        if j in att:
            p = {n: v[o_att] for n, v in hc["qsa_full"].items()}
            mixed, xn = hc_mix(x, p, nhc=nhc, eps=eps)
            delta, kvs = 0.0, []
            for r in range(R):
                q, gate, kk, vv = project_reference(
                    mixed, d["qkvg"][r, o_att], d["qn"][o_att], d["kn"][o_att], d["rope"][:B],
                    nh=nh, hd=hd, rot=g.ROT, eps=eps,
                )
                kvs.append(jnp.concatenate([kk, vv], axis=-1))
                if "ring" in d:  # stage C: the step's own tokens and group
                    dl = layer_view(d, o_att)
                    if r == 0:
                        ssel = step_selection(g, ex, dl, mixed, o_att, d["budget"])
                        step_out.append(ssel)
                    attn = step_attention(g, ex, dl, ssel["sel"], q, gate, kk, vv, r).reshape(B, -1)
                    delta = delta + jnp.dot(
                        attn.astype(BF16), d["o_proj"][r, o_att], preferred_element_type=F32
                    )
                    continue
                if "idx_w" in d and r == 0:  # replicated indexer: one selection for all ranks
                    pages, kvl = indexer_selection(g, ex, d, mixed, o_att, d["budget"])
                elif "idx_w" not in d:
                    pages, kvl = d["pages"][:B], d["kv_len"][:B]
                attn = reference_rows(
                    q, gate, d["pool"][r], pages, kvl,
                    off=d["off"], scale_k=1.0, scale_v=1.0, sm_scale=hd**-0.5,
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
    return x, jnp.stack(kv_new, axis=1)


def pack_wholestep(
    spec: Any, g: Qwen38Geometry, ex: ExecSpec, d: dict[str, Any], L: int
) -> tuple[dict, dict]:
    """`(weights, states)` per-rank operands (leading `R`, not yet folded)."""
    R, T = ex.R, ex.T
    hc, w_l, st_l = d["hc"], d["inp"].w_l, d["inp"].st_l
    att_rows = att_layers(g, L, d.get("first", "gdn"))
    gdn_rows = tuple(j for j in range(L) if j not in att_rows)
    weights = {
        "router": _rep(w_l["router"], R),
        "sh13": w_l["sh13"],
        "sh2": w_l["sh2"],
        "e13": tuple(w_l["e13"][:, j] for j in range(L)),
        "e2": tuple(w_l["e2"][:, j] for j in range(L)),
        "es": jnp.concatenate([w_l["e13_s"], w_l["e2_s"]], axis=-1)[:, :, :, None, :],
        **hc_weight_arrays("moe", hc["moe"], g, R, mix=True),
        "qkvg": d["qkvg"],
        "o_proj": d["o_proj"],
        **hc_weight_arrays("qsa_full", hc["qsa_full"], g, R, mix=True),
        **({"idx_w": _rep(d["idx_w"], R)} if "idx_w" in d else {}),
        **(
            {"lin_in": w_l["lin_in"], "lin_out": w_l["lin_out"],
             **hc_weight_arrays("gdn", hc["gdn"], g, R, mix=True)}
            if gdn_rows else {}
        ),
        **(
            {"lm_head": d["lm_head"][:, None], **hc_weight_arrays("tail", hc["tail"], g, R, mix=True)}
            if "lm_head" in d else {}
        ),
    }
    smalls = {
        "moe_hcn": hc["moe"]["norm"],
        **{k: w_l[k] for k in ("post_norm", "sh_gate", "sh13_s", "sh2_s", "ecopy") if k in w_l},
        "qsa_full_hcn": _rows(_rep(hc["qsa_full"]["norm"], R), att_rows, L),
        "qsa_qn": _rows(_rep(d["qn"], R), att_rows, L),
        "qsa_kn": _rows(_rep(d["kn"], R), att_rows, L),
        **({"qsa_iqn": _rows(_rep(d["iqn"], R), att_rows, L)} if "iqn" in d else {}),
        **({"qsa_ikn": _rows(_rep(d["ikn"], R), att_rows, L)} if "ikn" in d else {}),
        **(
            {"gdn_hcn": _rows(_rep(hc["gdn"]["norm"], R), gdn_rows, L),
             "in_s": _rows(w_l["lin_in_s"], gdn_rows, L),
             "out_s": _rows(w_l["lin_out_s"], gdn_rows, L),
             **{k: _rows(w_l[k], gdn_rows, L) for k in ("conv", "alog", "dtb", "gnorm")}}
            if gdn_rows else {}
        ),
        **({"tail_hcn": _rows(_rep(hc["tail"]["norm"], R), (L - 1,), L)} if "lm_head" in d else {}),
    }
    if not gdn_rows:
        return pack_weights(spec, ex, weights, smalls), {}
    lay = GDNBlock.lin_in_layout(g.inner(R, L), ex)
    hr = rup(3 + T, 8)
    conv = jnp.pad(st_l["conv"], ((0, 0),) * 3 + ((0, hr - 3), (0, 0)))
    states = {
        "conv": conv.astype(F32),
        "rec": st_l["rec"].reshape(R, len(gdn_rows), ex.NS * lay.vh, lay.KD, lay.VD).astype(BF16),
    }
    return pack_weights(spec, ex, weights, smalls), states


def extras(d: dict[str, Any], mesh: Mesh, R: int) -> tuple:
    """Kernel extra inputs in `EXTRA_NAMES` order: HBM extras folded per rank, SMEM replicated."""
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    return (
        jax.device_put(kernel.fold(d["pool"]), tp),
        jax.device_put(d["pages"], rep),
        jax.device_put(d["kv_len"], rep),
        jax.device_put(jnp.asarray([d["off"]], jnp.int32), rep),
        jax.device_put(jnp.asarray([1.0, 1.0], F32), rep),
        jax.device_put(kernel.fold(_rep(d["rope"], R)), tp),
    )


# ------------------------------------------------------------ stage B (in-kernel indexer)
EXTRA_NAMES_B = [
    "qsa_pool", "qsa_ck", "qsa_nvis", "qsa_tail", "qsa_mtab", "qsa_off", "qsa_scales", "qsa_rope"
]


def add_indexer_inputs(
    key: int, g: Qwen38Geometry, ex: ExecSpec, d: dict[str, Any], L: int, *,
    max_groups: int, ctx: list[int],
) -> dict[str, Any]:
    """Indexer weights, per-request compressed keys, lengths and a shuffled main page table.

    `ctx[q]`: tokens of request `q` before this step; its `T` verify rows see `ctx + t + 1`.
    Replaces `pool` with one large enough for `requests x pages_per_seq` pages.
    """
    from .indexer import IDX_DIM

    R, T, M = ex.R, ex.T, ex.M
    nreq = M // T
    n_att = len(att_layers(g, L))
    k = jax.random.split(jax.random.PRNGKey(key + 7), 4)
    idx_w = (jax.random.normal(k[0], (n_att, g.H, 5 * IDX_DIM), F32) * 0.06).astype(BF16)
    iqn = jax.random.normal(k[1], (n_att, IDX_DIM), F32) * 0.05
    ck = jax.random.normal(k[2], (nreq, max_groups, IDX_DIM), F32).astype(BF16)
    pps = -(-max_groups * 4 // 128)
    rng = np.random.default_rng(key + 7)
    mtab = (rng.permutation(nreq * pps) + 1).reshape(nreq, pps).astype(np.int32)
    lens = np.zeros((M,), np.int32)
    for q in range(nreq):
        for t in range(T):
            if q * T + t < ex.B:
                lens[q * T + t] = ctx[q % len(ctx)] + t + 1
    pool = jax.random.normal(k[3], (R, nreq * pps + 1, 128, 1, 4, g.HD), F32).astype(
        jnp.float8_e4m3fn
    )
    return {
        **d, "idx_w": idx_w, "iqn": iqn, "ck": ck, "mtab": jnp.asarray(mtab), "pool": pool,
        "nvis": jnp.asarray(lens // 4), "tail": jnp.asarray(lens % 4), "max_groups": max_groups,
    }


def indexer_selection(
    g: Qwen38Geometry, ex: ExecSpec, d: dict[str, Any], mixed: jax.Array, o_att: int, K: int
) -> tuple[jax.Array, jax.Array]:
    """Reference `(micro-page ids [B, K+1], kv_len [B])` of one QSA layer from its mixed input."""
    from .indexer import IDX_HEADS, index_query, score_rows, topk_reference

    B, T = ex.B, ex.T
    y = rnd(jnp.dot(mixed.astype(BF16), d["idx_w"][o_att], preferred_element_type=F32))
    rope = d["rope"][:B]
    cos, sin = rope[:, : g.ROT // 2], rope[:, g.ROT // 2 : g.ROT]
    iq = index_query(y, d["iqn"][o_att], cos, sin, g.ROT, g.eps, jax.lax.bitcast_convert_type)
    pages = np.zeros((B, K + 1), np.int32)
    kvl = np.zeros((B,), np.int32)
    mtab = np.asarray(d["mtab"])
    per_page = 128 // 4
    for q in range(-(-B // T)):
        rows = range(q * T, min(B, (q + 1) * T))
        qr = jnp.stack([jnp.stack([iq[h][r] for h in range(IDX_HEADS)]) for r in rows])
        st = np.asarray(score_rows(qr.reshape(-1, qr.shape[-1]).astype(BF16), d["ck"][q]))
        for i, r in enumerate(rows):
            n, tail = int(d["nvis"][r]), int(d["tail"][r])
            sel = sorted(topk_reference(st[i], n, K))  # the kernel emits ascending ids
            gids = sel + ([n] if tail else [])
            for j, gid in enumerate(gids):
                pages[r, j] = mtab[q, gid // per_page] * per_page + gid % per_page
            kvl[r] = 4 * len(sel) + tail
    return jnp.asarray(pages), jnp.asarray(kvl)


def add_carrier(key: int, d: dict[str, Any], width: int = 256, mp: int = 0) -> dict[str, Any]:
    """The served compressed-key layout for `d["ck"]`: kernel pages of 32 groups, BF16 view
    `[pages, 16, 2, width]` (row 2i+j of a page = carrier row i, half j; lanes >= 128 zero),
    pages shuffled per request, `cktab [requests, max_groups // 32]`."""
    ck = np.asarray(d["ck"].astype(F32))
    if ck.ndim == 4:  # per-layer keys: one carrier, pages disjoint across layers
        n_att = ck.shape[0]
        ck = ck.reshape(-1, *ck.shape[2:])
    else:
        n_att = 0
    nreq, G, D = ck.shape
    ppr = G // 32
    rng = np.random.default_rng(key + 23)
    if mp:  # manager pages of mp contiguous kernel pages (manager page 0 unused)
        nm = ppr // mp
        mtab = (rng.permutation(nreq * nm) + 1).reshape(nreq, nm).astype(np.int32)
        tab = (mtab[:, :, None] * mp + np.arange(mp)).reshape(nreq, ppr)
        n_pages = (nreq * nm + 1) * mp
    else:
        mtab = tab = (rng.permutation(nreq * ppr) + 1).reshape(nreq, ppr).astype(np.int32)
        n_pages = nreq * ppr + 1
    car = np.zeros((n_pages, 32, width), np.float32)
    for q in range(nreq):
        car[tab[q], :, :D] = ck[q].reshape(ppr, 32, D)
    carrier = jnp.asarray(car, BF16).reshape(-1, 16, 2, width)
    cktab = jnp.asarray(mtab)
    if n_att:
        cktab = cktab.reshape(n_att, nreq // n_att, -1)
    return {**d, "carrier": carrier, "cktab": cktab}


def _flat(x: Any) -> Any:
    """Per-layer `[n_att, requests, …]` tables -> `[n_att * requests, …]` (row = o * nreq + q)."""
    return x.reshape(-1, *x.shape[2:])


def extras_b(d: dict[str, Any], mesh: Mesh, R: int) -> tuple:
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    pl_ = d.get("per_layer")
    ck = (
        (jax.device_put(kernel.fold(_rep(d["carrier"], R)), tp),
         jax.device_put(_flat(d["cktab"]) if pl_ else d["cktab"], rep))
        if "carrier" in d
        else (jax.device_put(kernel.fold(_rep(d["ck"], R)), tp),)
    )
    assert not pl_ or "carrier" in d, "per-layer metadata needs the served key layout"
    return (
        jax.device_put(kernel.fold(d["pool"]), tp),
        *ck,
        jax.device_put(d["nvis"], rep),
        jax.device_put(d["tail"], rep),
        jax.device_put(_flat(d["mtab"]) if pl_ else d["mtab"], rep),
        jax.device_put(d["off"] if pl_ else jnp.asarray([d["off"]], jnp.int32), rep),
        jax.device_put(jnp.asarray([1.0, 1.0], F32), rep),
        jax.device_put(kernel.fold(_rep(d["rope"], R)), tp),
    )


# ------------------------------------------------------------ stage C (the step's own tokens)
EXTRA_NAMES_C = EXTRA_NAMES_B + ["qsa_ctx", "qsa_ring", "qsa_rope_ck"]


def add_step_inputs(key: int, g: Qwen38Geometry, ex: ExecSpec, d: dict[str, Any], L: int,
                    *, ctx: list[int]) -> dict[str, Any]:
    """Raw index-key ring, request contexts, the completed groups' RoPE rows, k-norm weights."""
    from .indexer import IDX_DIM

    nreq = ex.M // ex.T
    n_att = len(att_layers(g, L))
    k = jax.random.split(jax.random.PRNGKey(key + 11), 2)
    ctxq = np.array([ctx[q % len(ctx)] for q in range(nreq)], np.int32)
    ring = jax.random.normal(k[0], (nreq, 8, IDX_DIM), F32).astype(BF16)
    ikn = jax.random.normal(k[1], (n_att, IDX_DIM), F32) * 0.05
    j0 = ctxq + (3 - ctxq % 4) % 4 - 3  # first position of the group completed this step
    inv = 1.0 / (g.theta ** (np.arange(0, g.ROT, 2) / g.ROT))
    ang = j0[:, None] * inv[None, :]
    rope_ck = np.zeros((nreq, 1, 128), np.float32)
    rope_ck[:, 0, : g.ROT // 2] = np.cos(ang).astype(ml_dtypes.bfloat16).astype(np.float32)
    rope_ck[:, 0, g.ROT // 2 : g.ROT] = np.sin(ang).astype(ml_dtypes.bfloat16).astype(np.float32)
    return {**d, "ctxq": jnp.asarray(ctxq), "ring": ring, "ikn": ikn, "rope_ck": jnp.asarray(rope_ck)}


def step_selection(g: Qwen38Geometry, ex: ExecSpec, d: dict[str, Any], mixed: jax.Array,
                   o_att: int, K: int) -> dict[str, Any]:
    """Stock semantics of one QSA layer's indexer with the step's own group (rank-independent):
    per request the completed group's key and new ring, per row the selected group set."""
    from .indexer import IDX_DIM, IDX_HEADS, index_query, index_rope, score_rows, topk_reference

    B, T = ex.B, ex.T
    bc = jax.lax.bitcast_convert_type
    y = rnd(jnp.dot(mixed.astype(BF16), d["idx_w"][o_att], preferred_element_type=F32))
    rope = d["rope"][:B]
    cos, sin = rope[:, : g.ROT // 2], rope[:, g.ROT // 2 : g.ROT]
    iq = index_query(y, d["iqn"][o_att], cos, sin, g.ROT, g.eps, bc)
    raw = y[:, 4 * IDX_DIM : 5 * IDX_DIM]
    out = {"ck": {}, "ring": {}, "sel": {}}
    for q in range(-(-B // T)):
        c0 = int(d["ctxq"][q])
        p3 = c0 + (3 - c0 % 4) % 4
        ring = d["ring"][q].astype(F32)
        acc = None
        for j in range(4):
            pos = p3 - 3 + j
            m = raw[q * T + pos - c0][None] if pos >= c0 else ring[pos % 8][None]
            acc = m if acc is None else acc + m
        pooled = rnd(acc * 0.25)
        rc = d["rope_ck"][q]
        ck = index_rope(
            rmsnorm(pooled, d["ikn"][o_att], g.eps, zero_centered=True),
            rc[:, : g.ROT // 2], rc[:, g.ROT // 2 : g.ROT], g.ROT, bc,
        ).astype(BF16)
        end = c0 + T
        new_ring = np.array(ring)
        for p in range(end - 8, end):
            if p >= c0:
                new_ring[p % 8] = np.asarray(raw[q * T + p - c0])
        out["ck"][q], out["ring"][q] = ck, jnp.asarray(new_ring).astype(BF16)
        rows = range(q * T, min(B, (q + 1) * T))
        qr = jnp.stack([jnp.stack([iq[h][r] for h in range(IDX_HEADS)]) for r in rows])
        qr = qr.reshape(-1, qr.shape[-1]).astype(BF16)
        st = np.array(score_rows(qr, d["ck"][q]))
        st[:, p3 // 4] = np.asarray(score_rows(qr, jnp.broadcast_to(ck, (8, IDX_DIM))))[:, 0]
        for i, r in enumerate(rows):
            out["sel"][r] = set(topk_reference(st[i], int(d["nvis"][r]), K))
    return out


def step_attention(g: Qwen38Geometry, ex: ExecSpec, d: dict[str, Any], sel: dict[str, Any],
                   q: jax.Array, gate: jax.Array, kk: jax.Array, vv: jax.Array, r_: int) -> jax.Array:
    """Per-rank stock attention of every row: selected groups older than the step from the
    pool, the step's own tokens FP8 write-then-read, the partial group causal."""
    from .qsa_full_c import quant_fp8

    B, T = ex.B, ex.T
    pool = d["pool"][r_]
    off, per_page = d["off"], 128 // MICRO
    mtab = np.asarray(d["mtab"])
    sm = g.HD**-0.5
    outs = []
    for r in range(B):
        qi = r // T
        c0, p_r = int(d["ctxq"][qi]), int(d["ctxq"][qi]) + r % T
        n_r = int(d["nvis"][r])
        keys, vals = [], []

        def token(pos):
            if pos >= c0:  # the step's own token: FP8 write-then-read
                t = qi * T + pos - c0
                return quant_fp8(kk[t].astype(F32), 1.0), quant_fp8(vv[t].astype(F32), 1.0)
            gid = pos // 4
            mic = mtab[qi, gid // per_page] * per_page + gid % per_page
            row = pool[mic // per_page, (mic % per_page) * MICRO + pos % 4, 0]
            return row[2 * off].astype(F32), row[2 * off + 1].astype(F32)

        groups = sorted(sel[r])
        for gid in groups:
            if gid < c0 // 4 or gid < n_r:
                for pos in range(4 * gid, 4 * gid + 4):
                    k_, v_ = token(pos)
                    keys.append(k_), vals.append(v_)
        if (p_r + 1) % 4:  # the row's partial group
            for pos in range(4 * n_r, p_r + 1):
                k_, v_ = token(pos)
                keys.append(k_), vals.append(v_)
        if not keys:
            outs.append(jnp.zeros(q[r].shape, F32))
            continue
        K_, V_ = jnp.stack(keys), jnp.stack(vals)
        s = jnp.einsum("hd,td->ht", q[r].astype(F32), K_) * sm
        attn = jax.nn.softmax(s, axis=-1) @ V_
        gt = jax.nn.sigmoid(gate[r].astype(F32)).astype(BF16).astype(F32)
        outs.append((attn.astype(BF16).astype(F32) * gt).astype(BF16).astype(F32))
    return jnp.stack(outs)


def extras_c(d: dict[str, Any], mesh: Mesh, R: int) -> tuple:
    tp, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    return extras_b(d, mesh, R) + (
        jax.device_put(d["ctxq"], rep),
        jax.device_put(kernel.fold(_rep(_flat(d["ring"]) if d.get("per_layer") else d["ring"], R)), tp),
        jax.device_put(kernel.fold(_rep(d["rope_ck"], R)), tp),
    )


# ------------------------------------------------------------ per-layer metadata (served packing)
PER_LAYER_KEYS = ("ck", "mtab", "ring", "off", "cktab")


def layer_view(d: dict[str, Any], o: int) -> dict[str, Any]:
    """`d` with the per-layer tables of QSA layer `o` (identity without per-layer metadata)."""
    if not d.get("per_layer"):
        return d
    return {**d, **{k: d[k][o] for k in PER_LAYER_KEYS if k in d}}


def add_per_layer(key: int, g: Qwen38Geometry, ex: ExecSpec, d: dict[str, Any], L: int) -> dict:
    """Different keys, main page tables, rings and head-pair offsets per QSA layer, pages disjoint
    across layers in one pool (the served pool packs layers into shared pages)."""
    from .indexer import IDX_DIM

    R, nreq = ex.R, ex.M // ex.T
    n_att = len(att_layers(g, L, d.get("first", "gdn")))
    pps = d["mtab"].shape[1]
    rng = np.random.default_rng(key + 31)
    k = jax.random.split(jax.random.PRNGKey(key + 31), 3)
    ck = jnp.concatenate(
        [d["ck"][None], jax.random.normal(k[0], (n_att - 1,) + d["ck"].shape, F32).astype(BF16)]
    )
    ring = jnp.concatenate(
        [d["ring"][None],
         jax.random.normal(k[1], (n_att - 1,) + d["ring"].shape, F32).astype(BF16)]
    ) if "ring" in d else None
    mtab = (rng.permutation(n_att * nreq * pps) + 1).reshape(n_att, nreq, pps).astype(np.int32)
    pool = jax.random.normal(
        k[2], (R, n_att * nreq * pps + 1, 128, 1, 4, g.HD), F32
    ).astype(jnp.float8_e4m3fn)
    out = {**d, "per_layer": True, "ck": ck, "mtab": jnp.asarray(mtab), "pool": pool,
           "off": jnp.asarray([(o + 1) % 2 for o in range(n_att)], jnp.int32)}
    if ring is not None:
        out["ring"] = ring
    return out
