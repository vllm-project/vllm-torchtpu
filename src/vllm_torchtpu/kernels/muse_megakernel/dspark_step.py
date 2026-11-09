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
"""TP8 Muse Spark DSpark draft step that runs inside the megakernel spec decode op."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from .config import SW_RING, TP_AXIS_NAME, residual_coefficients
from .fastpath_ops import ring_rows_write_local
from .load import Checkpoint

AUX_LAYERS = (13, 29, 45, 57, 61)
DRAFT_LAYERS = 5
DRAFT_HEADS = 128
DRAFT_KV_HEADS = 16
HEAD_DIM = 64
DSPARK_WEIGHT_NAMES = (
    "fc",
    "hn",
    "fn",
    "qgkv",
    "o",
    "gu",
    "dw",
    "sc",
    "lmw",
    "mw1",
    "mw2",
    "cw",
)


class DSparkConfig:
    def __init__(self, ckpt_dir: str | Path) -> None:
        cfg = json.loads((Path(ckpt_dir) / "config.json").read_text())
        self.dim = int(cfg["hidden_size"])
        self.inter = int(cfg["intermediate_size"])
        self.vocab = int(cfg["draft_vocab_size"])
        self.markov_rank = int(cfg["markov_rank"])
        self.mask_id = int(cfg["mask_token_id"])
        self.layers = int(cfg["num_hidden_layers"])
        self.eps = float(cfg["rms_norm_eps"])
        self.post_eps = float(cfg["rms_post_norm_eps"])
        self.rope_theta = float(cfg["rope_theta"])
        self.soft_cap = float(cfg.get("output_soft_cap_temp", 20.0))
        self.target_layer_ids = tuple(int(x) for x in cfg["target_layer_ids"])
        if (
            self.layers != DRAFT_LAYERS
            or self.target_layer_ids != AUX_LAYERS
            or int(cfg["num_attention_heads"]) != DRAFT_HEADS
            or int(cfg["num_key_value_heads"]) != DRAFT_KV_HEADS
            or int(cfg["head_dim"]) != HEAD_DIM
            or int(cfg.get("sliding_window", SW_RING)) != SW_RING
        ):
            raise ValueError(f"Unsupported DSpark geometry in {ckpt_dir}")


def _bf16(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a).astype(ml_dtypes.bfloat16))


def load_dspark_rank(ckpt_dir: str | Path, rank: int, tp: int) -> dict[str, np.ndarray]:
    """This rank's DSpark shard, each array with a leading TP axis of 1.

    q/k projections are permuted from the export's interleaved RoPE pairs to the
    NEOX halves used by the target, matching MuseSparkDSparkModel.
    """
    dc = DSparkConfig(ckpt_dir)
    ck = Checkpoint(ckpt_dir)
    h, hd = dc.dim, HEAD_DIM
    qh, kvh = DRAFT_HEADS // tp, DRAFT_KV_HEADS // tp
    ir = dc.inter // tp
    vs = dc.vocab // tp

    def get(name: str) -> np.ndarray:
        return np.asarray(ck.get(name))

    def neox(w: np.ndarray) -> np.ndarray:
        n = w.shape[0] // hd
        return (
            w.reshape(n, hd // 2, 2, w.shape[-1]).transpose(0, 2, 1, 3).reshape(w.shape)
        )

    fc = ck.memmap("fc.weight")
    qgkv, o, gu, dw, sc = [], [], [], [], []
    for li in range(dc.layers):
        p = f"layers.{li}."
        q = neox(np.asarray(ck.memmap(p + "self_attn.q_proj.weight")))
        k = neox(np.asarray(ck.memmap(p + "self_attn.k_proj.weight")))
        v = ck.memmap(p + "self_attn.v_proj.weight")
        g = ck.memmap(p + "self_attn.o_gate_proj.weight")
        qs, ks = rank * qh * hd, rank * kvh * hd
        qgkv.append(
            np.concatenate(
                (
                    q[qs : qs + qh * hd],
                    np.asarray(g[qs : qs + qh * hd]),
                    k[ks : ks + kvh * hd],
                    np.asarray(v[ks : ks + kvh * hd]),
                ),
                axis=0,
            ).T
        )
        o.append(
            np.asarray(ck.memmap(p + "self_attn.o_proj.weight")[:, qs : qs + qh * hd]).T
        )
        gw = ck.memmap(p + "feed_forward.gate_proj.weight")
        uw = ck.memmap(p + "feed_forward.up_proj.weight")
        gu.append(
            np.concatenate(
                (
                    np.asarray(gw[rank * ir : (rank + 1) * ir]),
                    np.asarray(uw[rank * ir : (rank + 1) * ir]),
                ),
                axis=0,
            ).T
        )
        dw.append(
            np.asarray(
                ck.memmap(p + "feed_forward.down_proj.weight")[
                    :, rank * ir : (rank + 1) * ir
                ]
            ).T
        )
        a_att, b_att = residual_coefficients(
            get(p + "add_attn.gate").astype(np.float32), 0.3
        )
        a_ffn, b_ffn = residual_coefficients(
            get(p + "add_ffn.gate").astype(np.float32), 0.3
        )
        sc.append(
            np.stack(
                (
                    1.0
                    + get(p + "self_attn.pre_qkv_proj_norm.weight").astype(np.float32),
                    1.0 + get(p + "pre_ffn_norm.weight").astype(np.float32),
                    1.0 + get(p + "post_ffn_norm.weight").astype(np.float32),
                    a_att,
                    b_att,
                    a_ffn,
                    b_ffn,
                    np.zeros((h,), np.float32),
                )
            )
        )
    cw = np.zeros((1, h + dc.markov_rank + 128), np.float32)
    cw[0, : h + dc.markov_rank] = (
        get("confidence_head.proj.weight").astype(np.float32).reshape(-1)
    )
    cw[0, h + dc.markov_rank :] = float(
        get("confidence_head.proj.bias").astype(np.float32).reshape(-1)[0]
    )
    return {
        "fc": _bf16(np.asarray(fc[rank * (h // tp) : (rank + 1) * (h // tp)]).T)[None],
        "hn": (1.0 + get("hidden_norm.weight").astype(np.float32))[None],
        "fn": (1.0 + get("norm.weight").astype(np.float32))[None],
        "qgkv": _bf16(np.stack(qgkv))[None],
        "o": _bf16(np.stack(o))[None],
        "gu": _bf16(np.stack(gu))[None],
        "dw": _bf16(np.stack(dw))[None],
        "sc": np.stack(sc)[None],
        "lmw": _bf16(
            np.asarray(ck.memmap("lm_head.weight")[rank * vs : (rank + 1) * vs]).T
        )[None],
        "mw1": _bf16(np.asarray(ck.memmap("markov_head.markov_w1.weight")))[None],
        "mw2": _bf16(
            np.asarray(
                ck.memmap("markov_head.markov_w2.weight")[rank * vs : (rank + 1) * vs]
            ).T
        )[None],
        "cw": cw,
    }


def _rms(x: jax.Array, eps: float) -> jax.Array:
    xf = x.astype(jnp.float32)
    return xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)


def _rope(x: jax.Array, pos: jax.Array, theta: float) -> jax.Array:
    """NEOX RoPE in fp32 on [T, H, 64]."""
    half = HEAD_DIM // 2
    inv = 1.0 / (theta ** (jnp.arange(half, dtype=jnp.float32) / half))
    ang = pos.astype(jnp.float32)[:, None] * inv[None, :]
    c, s = jnp.cos(ang)[:, None, :], jnp.sin(ang)[:, None, :]
    x1, x2 = x.astype(jnp.float32)[..., :half], x.astype(jnp.float32)[..., half:]
    return jnp.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(
        jnp.bfloat16
    )


def dspark_ctx_update_local(
    dc: DSparkConfig,
    w: dict[str, jax.Array],
    kv_draft: jax.Array,
    aux: jax.Array,
    ctx_pos: jax.Array,
    ctx_slot: jax.Array,
    rank: jax.Array | None = None,
) -> jax.Array:
    """Write context K/V of accepted target positions into the draft ring.

    kv_draft: [L, slots, 2048, 256] bf16, per KV head [k(64) | v(64)].
    aux: [M, 5, dim] bf16 target hidden states; ctx_pos < 0 rows are skipped.
    """
    m = aux.shape[0]
    rank = jax.lax.axis_index(TP_AXIS_NAME) if rank is None else rank
    r_order = jnp.argsort(
        jax.lax.all_gather(rank[None], TP_AXIS_NAME, axis=0, tiled=True)
    )
    fc = jnp.dot(
        aux.reshape(m, -1), w["fc"], preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    ctx = jax.lax.all_gather(fc[:, None, :], TP_AXIS_NAME, axis=1, tiled=True)[
        :, r_order, :
    ].reshape(m, -1)
    ctx = (_rms(ctx, dc.eps) * w["hn"]).astype(jnp.bfloat16)
    pos = jnp.maximum(ctx_pos, 0)
    n_slots = kv_draft.shape[1]
    slot = jnp.clip(ctx_slot, 0, n_slots - 1)
    ridx = pos % SW_RING
    rows_kv = []
    qw = w["qgkv"].shape[-1]
    kvh = (qw - 2 * (w["o"].shape[1])) // (2 * HEAD_DIM)
    k_off = 2 * w["o"].shape[1]
    for li in range(dc.layers):
        x = (_rms(ctx, dc.eps) * w["sc"][li, 0]).astype(jnp.bfloat16)
        kv = jnp.dot(
            x, w["qgkv"][li, :, k_off:], preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
        k = _rms(kv[:, : kvh * HEAD_DIM].reshape(m, kvh, HEAD_DIM), dc.eps).astype(
            jnp.bfloat16
        )
        k = _rope(k, pos, dc.rope_theta)
        v = kv[:, kvh * HEAD_DIM :].reshape(m, kvh, HEAD_DIM)
        rows_kv.append(jnp.concatenate([k, v], axis=-1).reshape(m, kvh * 2 * HEAD_DIM))
    packed = jnp.stack(rows_kv, axis=0).astype(jnp.float32)  # [L, M, w]
    kv5 = ring_rows_write_local(
        kv_draft[None], packed[None], slot, ridx, (ctx_pos >= 0).astype(jnp.int32)
    )
    kv_draft = kv5[0]
    return kv_draft


def dspark_draft_local(
    dc: DSparkConfig,
    w: dict[str, jax.Array],
    emb: jax.Array,
    kv_draft: jax.Array,
    bonus: jax.Array,
    base_pos: jax.Array,
    req_slot: jax.Array,
    k_draft: int,
    rank: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Draft k_draft tokens per request from [bonus, MASK...] at base_pos.

    emb: this rank's vocab shard of the target embedding [vshard, dim].
    Returns (draft ids [B, K] int32, confidence [B, K] f32), replicated on all ranks.
    """
    b = bonus.shape[0]
    kq = k_draft
    vs = emb.shape[0]
    # The serving export rejects partition-id, so callers there pass the rank
    # explicitly.
    rank = jax.lax.axis_index(TP_AXIS_NAME) if rank is None else rank
    q_ids = jnp.full((b, kq), dc.mask_id, jnp.int32).at[:, 0].set(bonus)
    q_pos = base_pos[:, None] + jnp.arange(kq, dtype=jnp.int32)[None, :]
    ids_f, pos_f = q_ids.reshape(-1), q_pos.reshape(-1)
    off = ids_f - rank * vs
    rows = emb[jnp.clip(off, 0, vs - 1)]
    rows = jnp.where(((off >= 0) & (off < vs))[:, None], rows, 0)
    h = _rms(jax.lax.psum(rows.astype(jnp.float32), TP_AXIS_NAME), dc.eps)

    qh = w["o"].shape[1] // HEAD_DIM
    kvh = (w["qgkv"].shape[-1] - 2 * qh * HEAD_DIM) // (2 * HEAD_DIM)
    grp = qh // kvh
    qw = qh * HEAD_DIM
    # Context ring index r holds position t(r); valid when 0 <= t < base and
    # inside the strict 2048 window of query base + k.
    r = jnp.arange(SW_RING, dtype=jnp.int32)[None, :]
    t = base_pos[:, None] - 1 - jnp.mod(base_pos[:, None] - 1 - r, SW_RING)
    kk = jnp.arange(kq, dtype=jnp.int32)
    ctx_ok = (t[:, None, :] >= 0) & (
        t[:, None, :] > base_pos[:, None, None] + kk[None, :, None] - SW_RING
    )
    causal = kk[None, :] <= kk[:, None]
    scale = HEAD_DIM**-0.5
    ctx = kv_draft[:, req_slot]  # [L, B, 2048, kvh * 128]

    for li in range(dc.layers):
        sc = w["sc"][li]
        res = h
        x = (_rms(res, dc.eps) * sc[0]).astype(jnp.bfloat16)
        qgkv = jnp.dot(x, w["qgkv"][li], preferred_element_type=jnp.float32).astype(
            jnp.bfloat16
        )
        q = qgkv[:, :qw].reshape(b * kq, qh, HEAD_DIM)
        gate = qgkv[:, qw : 2 * qw]
        k = qgkv[:, 2 * qw : 2 * qw + kvh * HEAD_DIM].reshape(b * kq, kvh, HEAD_DIM)
        v = qgkv[:, 2 * qw + kvh * HEAD_DIM :].reshape(b * kq, kvh, HEAD_DIM)
        q = (_rms(q, dc.eps) * 5.4730064863875).astype(jnp.bfloat16)
        k = _rms(k, dc.eps).astype(jnp.bfloat16)
        q = _rope(q, pos_f, dc.rope_theta).reshape(b, kq, kvh, grp, HEAD_DIM)
        k = _rope(k, pos_f, dc.rope_theta).reshape(b, kq, kvh, HEAD_DIM)
        v = v.reshape(b, kq, kvh, HEAD_DIM)
        c_kv = ctx[li].reshape(b, SW_RING, kvh, 2 * HEAD_DIM)
        k_c, v_c = c_kv[..., :HEAD_DIM], c_kv[..., HEAD_DIM:]
        s_c = (
            jnp.einsum("bqhgd,bshd->bhqgs", q, k_c, preferred_element_type=jnp.float32)
            * scale
        )
        s_c = jnp.where(ctx_ok[:, None, :, None, :], s_c, -1e30)
        s_q = (
            jnp.einsum("bqhgd,bshd->bhqgs", q, k, preferred_element_type=jnp.float32)
            * scale
        )
        s_q = jnp.where(causal[None, None, :, None, :], s_q, -1e30)
        p = jax.nn.softmax(jnp.concatenate([s_c, s_q], axis=-1), axis=-1).astype(
            jnp.bfloat16
        )
        o_c = jnp.einsum(
            "bhqgs,bshd->bqhgd",
            p[..., :SW_RING],
            v_c,
            preferred_element_type=jnp.float32,
        )
        o_q = jnp.einsum(
            "bhqgs,bshd->bqhgd", p[..., SW_RING:], v, preferred_element_type=jnp.float32
        )
        att = (o_c + o_q).astype(jnp.bfloat16).reshape(b * kq, qh, HEAD_DIM)
        att = _rms(att, dc.eps).astype(jnp.bfloat16).reshape(b * kq, qw)
        att = (
            att.astype(jnp.float32) * jax.nn.sigmoid(gate.astype(jnp.float32))
        ).astype(jnp.bfloat16)
        o_full = jax.lax.psum(
            jnp.dot(att, w["o"][li], preferred_element_type=jnp.float32), TP_AXIS_NAME
        )
        h = sc[3] * res + sc[4] * _rms(o_full, dc.post_eps)
        res = h
        x = (_rms(res, dc.eps) * sc[1]).astype(jnp.bfloat16)
        g_u = jnp.dot(x, w["gu"][li], preferred_element_type=jnp.float32)
        half = g_u.shape[-1] // 2
        act = (jax.nn.silu(g_u[:, :half]) * g_u[:, half:]).astype(jnp.bfloat16)
        d_full = jax.lax.psum(
            jnp.dot(act, w["dw"][li], preferred_element_type=jnp.float32), TP_AXIS_NAME
        )
        d = _rms((d_full * sc[2]).astype(jnp.bfloat16), dc.post_eps)
        h = sc[5] * res + sc[6] * d

    out_mult = 0.17677669529663687
    hf = (_rms(h, dc.eps) * w["fn"]).astype(jnp.bfloat16)
    raw = jnp.dot(hf, w["lmw"], preferred_element_type=jnp.float32).reshape(b, kq, vs)
    base = dc.soft_cap * jnp.tanh(raw * out_mult / dc.soft_cap)
    dim = h.shape[-1]
    h3 = h.reshape(b, kq, dim)
    cw = w["cw"]
    conf_h = jnp.dot(h3, cw[:dim], preferred_element_type=jnp.float32)
    cw_m, cb = cw[dim : dim + dc.markov_rank], cw[dim + dc.markov_rank]
    r_order = jnp.argsort(
        jax.lax.all_gather(rank[None], TP_AXIS_NAME, axis=0, tiled=True)
    )
    drafts, confs = [], []
    prev = bonus
    for i in range(kq):
        m_emb = w["mw1"][prev]
        tot = (
            base[:, i]
            + jnp.dot(m_emb, w["mw2"], preferred_element_type=jnp.float32) * out_mult
        )
        lmax = jnp.max(tot, axis=-1)
        lidx = jnp.argmax(tot, axis=-1).astype(jnp.int32) + rank * vs
        amax = jax.lax.all_gather(lmax[:, None], TP_AXIS_NAME, axis=1, tiled=True)[
            :, r_order
        ]
        aidx = jax.lax.all_gather(lidx[:, None], TP_AXIS_NAME, axis=1, tiled=True)[
            :, r_order
        ]
        best = jnp.argmax(amax, axis=-1, keepdims=True)
        tok = jnp.take_along_axis(aidx, best, axis=-1)[:, 0]
        confs.append(
            jax.nn.sigmoid(conf_h[:, i] + jnp.dot(m_emb.astype(jnp.float32), cw_m) + cb)
        )
        drafts.append(tok)
        prev = tok
    return jnp.stack(drafts, axis=1), jnp.stack(confs, axis=1)
