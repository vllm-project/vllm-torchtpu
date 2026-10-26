# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Host / XLA-side glue of the megakernel, kept apart from the kernel body (as upstream does):

* `metadata`: per-step operands from the runner's metadata (query_start_loc, seq_lens, positions, block tables of
  VIRTUAL page ordinals): rmeta / qmeta / rvec / scal and the two local block tables (each cache resolves an
  ordinal `mod` its own page count, attention_interface.py:1229, streamindex_topk.py:1788);
* `rope_table`: the per-row cos / signed-sin lanes (GPT-J pairs in lanes 0..DR-1 and 128-DR..127);
* `smalls_rows` / `dense_operands` / `expert_operands` / `cache_operands`: operand packing from per-rank weights in
  the layout of the instruments' reference (kernel_work/glm52-decode-mk/instruments/inputs.py = production [K, N]
  layouts, TP-sharded experts, rank-local row-parallel scales);
* `step`: one decode step = the megakernel + the barrier-free writer, per rank (call inside shard_map);
* `candidate`: the instruments' candidate contract (glm_ref.step's signature, aux = the kernel's debug exports).
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax

from . import kernel as K
from . import layout as LY
from .config import from_dims
from .writer import build_writer

F32, BF16, I32, U32 = jnp.float32, jnp.bfloat16, jnp.int32, jnp.uint32


# ------------------------------------------------------------------------------------------ metadata


def metadata(
    cfg,
    ids_len,
    positions,
    seq_lens,
    qsl,
    bt_virtual,
    nb,
    nbi,
    max_pages,
    rank,
    delay=(-1, 0, -1),
):
    """-> (scal i32[16], rmeta i32[T, 8], qmeta i32[BM, 4], bt i32[BM, MAXP], bti i32[BM, MAXP], rvec i32[M, 128]).
    qsl [S + 1] (padded entries repeat the total), seq_lens [S] (this step included), positions [>= T],
    bt_virtual [S, W] (DCP virtual page ordinals)."""
    TMAX, BMAX, M = cfg.TMAX, cfg.BMAX, cfg.M

    def fit(x, n, axis=0, val=0):
        k = x.shape[axis]
        if k >= n:
            return lax.slice_in_dim(x, 0, n, axis=axis)
        pad = [(0, 0)] * x.ndim
        pad[axis] = (0, n - k)
        return jnp.pad(x, pad, constant_values=val)

    qsl = qsl.astype(I32)
    T = qsl[-1]
    qsl_b = fit(
        jnp.concatenate([qsl, jnp.broadcast_to(qsl[-1:], (BMAX + 1,))]), BMAX + 1
    )
    seq_b = fit(seq_lens.astype(I32), BMAX)
    btv = fit(fit(bt_virtual.astype(I32), BMAX, 0), max_pages, 1)
    bt = jnp.mod(btv, jnp.int32(nb))
    bti = jnp.mod(btv, jnp.int32(nbi))
    t = jnp.arange(TMAX, dtype=I32)
    valid = (t < T).astype(I32)
    req = jnp.sum((qsl_b[1 : BMAX + 1][None, :] <= t[:, None]).astype(I32), axis=1)
    req = jnp.minimum(req, BMAX - 1)
    p = fit(positions.astype(I32), TMAX) * valid
    seq_t = seq_b[req] * valid
    z = jnp.zeros(TMAX, I32)
    rmeta = jnp.stack([valid, p, req * valid, seq_t, z, z, z, z], 1)
    t0 = qsl_b[:BMAX]
    nq = qsl_b[1 : BMAX + 1] - t0
    qmeta = jnp.stack([t0, nq, seq_b, (nq > 0).astype(I32)], 1)
    B = jnp.sum((nq > 0).astype(I32))
    scal = jnp.zeros(16, I32).at[0].set(T).at[1].set(B).at[2].set(rank)
    scal = scal.at[3].set(delay[0]).at[4].set(delay[1]).at[5].set(delay[2])
    rvec = jnp.zeros((M, 128), I32).at[:TMAX, 0].set(valid).at[:TMAX, 1].set(p)
    return scal, rmeta, qmeta, bt, bti, rvec


def kernel_pages(cfg, max_pages):
    """The megakernel's block-table width: the full DCP-D local table width at VR == 1; this replica's half table
    (ceil(max_pages / VR) tiles) under V16."""
    return -(-max_pages // cfg.VR)


def v16_tables(cfg, bt, bti, rank):
    """V16 (stageB/v16/DESIGN-ADDENDUM.md §1): the full DCP-D local tables [BM, W] -> this rank's half tables
    [BM, ceil(W / VR)]: tile p of request b is local page VR * p + (rank // D + b) % VR (indices past W clip to the last
    column; such tiles are never scanned: the kernel's tile count is (np - hb + VR - 1) // VR)."""
    VR, D = cfg.VR, cfg.D
    W = bt.shape[1]
    Wh = -(-W // VR)
    b = jnp.arange(bt.shape[0], dtype=I32)[:, None]
    p = jnp.arange(Wh, dtype=I32)[None, :]
    idx = jnp.minimum(VR * p + (rank // D + b) % VR, W - 1)
    idx = jnp.broadcast_to(idx, (bt.shape[0], Wh))
    return jnp.take_along_axis(bt, idx, axis=1), jnp.take_along_axis(bti, idx, axis=1)


def rope_table(cfg, cos_sin_rows):
    """cos_sin_rows [T, DR] (bf16 values: cos | sin halves) -> f32 [4, M, 128]: (cos, signed sin) with the rope pairs
    in lanes 0..DR-1 (0, 1) and in lanes 128-DR..127 (2, 3); identity elsewhere (glm5_mk_dcp adapter). Pairs are GPT-J
    interleaved (c0 c0 c1 c1 ..., -s0 s0 -s1 s1 ...), or half-split with cfg.rope_neox (c0..c31 c0..c31,
    -s0..-s31 s0..s31), matching kernel.rope's partner lane."""
    DR, M, TMAX = cfg.DR, cfg.M, cfg.TMAX
    cs = cos_sin_rows.astype(F32)
    if cfg.rope_neox:
        c = jnp.concatenate([cs[:, : DR // 2], cs[:, : DR // 2]], axis=1)
        s = jnp.concatenate([-cs[:, DR // 2 :], cs[:, DR // 2 :]], axis=1)
    else:
        c = jnp.repeat(cs[:, : DR // 2], 2, axis=1)
        s = jnp.stack([-cs[:, DR // 2 :], cs[:, DR // 2 :]], -1).reshape(TMAX, DR)
    ones = jnp.ones((TMAX, 128 - DR), F32)
    zeros = jnp.zeros((TMAX, 128 - DR), F32)
    tab = jnp.stack(
        [
            jnp.concatenate([c, ones], 1),
            jnp.concatenate([s, zeros], 1),
            jnp.concatenate([ones, c], 1),
            jnp.concatenate([zeros, s], 1),
        ]
    )
    if M > TMAX:
        tab = jnp.concatenate(
            [
                tab,
                jnp.broadcast_to(
                    jnp.array([1.0, 0.0, 1.0, 0.0], F32)[:, None, None],
                    (4, M - TMAX, 128),
                ),
            ],
            axis=1,
        )
    return tab


# ------------------------------------------------------------------------------------------ weights


def smalls_rows(cfg, layers, final_norm, k_scales):
    """Per-rank f32 [L + 1, 1, S] from reference-layout layer dicts (instruments/inputs.py `_layer_shard`)."""
    lay, S = LY.smalls_layout(cfg)
    rows = []
    for l in range(cfg.L + 1):
        f = {}
        if l < cfg.L:
            p = layers[l]
            f = {
                "in_ln": p["in_norm"],
                "post_ln": p["post_norm"],
                "q_a_ln": p["q_a_norm"],
                "kv_a_ln": p["kv_a_norm"],
                "qkv_a_s": p["qkv_a_s"],
                "q_b_s": p["q_b_s"],
                "o_proj_s": p["o_s"],
                "k_scale": jnp.full((1,), k_scales[l], F32),
            }
            if cfg.full_indexer(l):
                f.update(
                    idx_wq_s=p["wq_b_s"], idx_kn_w=p["k_norm_w"], idx_kn_b=p["k_norm_b"]
                )
            if cfg.is_moe(l):
                gb = (
                    jnp.full((cfg.EPAD,), -1e30, F32)
                    .at[: cfg.E]
                    .set(p["gate_b"].astype(F32))
                )
                f.update(gate_bias=gb, sw13_s=p["ss13"], sw2_s=p["ss2"])
            else:
                f.update(w13_s=p["gu_s"], w2_s=p["dn_s"])
        else:
            f = {"in_ln": final_norm}
        parts = []
        for name, (_, n) in lay.items():
            v = f.get(name)
            v = jnp.zeros((n,), F32) if v is None else jnp.ravel(v).astype(F32)
            if name == "k_scale":
                v = jnp.broadcast_to(v[:1], (n,))
            v = jnp.pad(v, (0, LY.rup(n) - v.shape[0]))
            parts.append(v)
        rows.append(jnp.concatenate(parts))
    return jnp.stack(rows).reshape(cfg.L + 1, 1, S)


def _pad_last(x, n):
    k = x.shape[-1]
    return x if k == n else jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, n - k)])


def dense_operands(cfg, layers, dkinds):
    """Stacked dense weights [n_layers_of_kind, ...] in layout.DKINDS order (load-time copies in serving)."""
    get = {
        "qkva": lambda p: _pad_last(p["qkv_a_w"], cfg.QKVP),
        "qb": lambda p: p["q_b_w"],
        "wuk": lambda p: p["w_uk"],
        "iwq": lambda p: p["wq_b_w"],
        "iwk": lambda p: _pad_last(p["wk_w"], cfg.IKW),
        "wuv": lambda p: p["w_uv"],
        "wo": lambda p: p["o_w"],
        "gate": lambda p: _pad_last(p["gate_w"], cfg.EPAD),
        "sh13": lambda p: p["sw13"],
        "sh2": lambda p: p["sw2"],
        "dw13": lambda p: p["gu_w"],
        "dw2": lambda p: p["dn_w"],
    }
    return [
        jnp.stack([get[k](layers[l]) for l in LY.layers_of(cfg, k)]) for k in dkinds
    ]


def expert_operands(cfg, layers, stack_e=True):
    """TP16 expert tensors (main's served TP-sharded layout, EP off): stacked [Lm, ...] or per layer."""
    per = []
    for l in LY.layers_of(cfg, "gate"):
        p = layers[l]
        per.append((p["w13"], p["w2"], p["s13"].astype(F32), p["s2"].astype(F32)))
    if stack_e:
        return [jnp.stack([x[i] for x in per]) for i in range(4)]
    return [t for x in per for t in x]


def cache_operands(cfg, caches, G):
    """Per-layer caches [{'nope','rope'[,'idx']}] -> stacked groups in kernel.cache_names order."""
    out = []
    for kind in ("nope", "rope", "idx"):
        ls = LY.layers_of(cfg, "iwq") if kind == "idx" else list(range(cfg.L))
        gs = len(ls) // G[kind]
        for gi in range(G[kind]):
            out.append(
                jnp.stack([caches[l][kind] for l in ls[gi * gs : (gi + 1) * gs]])
            )
    return out


def unstack_caches(cfg, ops, G):
    per = [dict() for _ in range(cfg.L)]
    i = 0
    for kind in ("nope", "rope", "idx"):
        ls = LY.layers_of(cfg, "iwq") if kind == "idx" else list(range(cfg.L))
        gs = len(ls) // G[kind]
        for gi in range(G[kind]):
            for j, l in enumerate(ls[gi * gs : (gi + 1) * gs]):
                per[l][kind] = ops[i][j]
            i += 1
    return per


# ------------------------------------------------------------------------------------------ the step


@functools.lru_cache(maxsize=32)
def _built(cfg, max_pages, nb, nbi, tiles, interpret, debug, stack_e, groups_items):
    groups = dict(groups_items) if groups_items else None
    call, info = K.build(
        cfg,
        max_pages=kernel_pages(cfg, max_pages),
        nb=nb,
        nbi=nbi,
        tiles=tiles,
        interpret=interpret,
        debug=debug,
        stack_e=stack_e,
        groups=groups,
    )
    writer, winfo = build_writer(
        cfg,
        max_pages=max_pages,
        nb=nb,
        nbi=nbi,
        interpret=interpret,
        groups=groups,
        fault=tiles.fault if tiles.fault in ("dcp_owner", "f_replica_nowrite") else "",
    )
    return call, info, writer, winfo


def unpack_qall(cfg, qall):
    """[L, R, T, HR*QA/2] u32 pairs -> q_all [L, T, NH, KL + DR] (head = rank * HR + h, glm_ref A3)."""
    L_, R, T, QH = qall.shape
    QA = cfg.KL + 128
    lo = lax.bitcast_convert_type(lax.shift_left(qall, jnp.uint32(16)), F32)
    hi = lax.bitcast_convert_type(qall & jnp.uint32(0xFFFF0000), F32)
    full = jnp.concatenate([lo, hi], axis=-1).reshape(L_, R, T, cfg.HR, QA)[
        ..., : cfg.KL + cfg.DR
    ]
    return jnp.transpose(full, (0, 2, 1, 3, 4)).reshape(L_, T, cfg.NH, cfg.KL + cfg.DR)


def tables_from_union(cfg, uv, ureq, rmeta, qmeta, T):
    """Per-row rank-local top-k tables [T, KT] (ascending local index, -1 padded) from the debug export of a fresh
    layer's pre-resolution union lists: entries `local idx | row bits << MB_SH` of request b at flat positions
    [uo_b * 128, uo_b * 128 + nu_b)."""
    flat = uv.reshape(-1)
    TMAX, KT = cfg.TMAX, cfg.KT
    t = jnp.arange(TMAX, dtype=I32)
    b = rmeta[:, 2]
    q = t - qmeta[b, 0]
    nu, uo = ureq[b, 0], ureq[b, 1]
    W = cfg.QMAX * KT
    k = jnp.arange(W, dtype=I32)
    idx = jnp.clip(uo[:, None] * 128 + k[None, :], 0, flat.shape[0] - 1)
    e = flat[idx]
    live = (k[None, :] < nu[:, None]) & (
        (lax.shift_right_logical(e, K.MB_SH + q[:, None]) & 1) == 1
    )
    live = live & (t[:, None] < T)
    key = jnp.where(live, k[None, :], W + 1)
    order = jnp.argsort(key, axis=1)[:, :KT]
    sel = jnp.take_along_axis(live, order, axis=1)
    loc = jnp.take_along_axis(e & K.SRC_MASK, order, axis=1)
    return jnp.where(sel, loc, -1).astype(I32)


def router_ids(cfg, wtab):
    """[M, EPAD] routing weights -> [M, TOPK] chosen experts (ascending id; the check compares sets). A row with
    fewer than TOPK nonzero weights (a padded row, or a fault) is filled with the lowest ids, so every id stays a
    valid expert index."""
    n = wtab.shape[-1]
    e = jnp.arange(n, dtype=I32)
    key = jnp.where(wtab > 0, e[None, :], n + e[None, :])
    return (jnp.sort(key, axis=1)[:, : cfg.TOPK] % min(n, cfg.E)).astype(I32)


def candidate_factory(
    tiles=None,
    interpret=True,
    stack_e=True,
    groups=None,
    delay=(-1, 0, -1),
    debug=True,
    embed_from_ref=True,
    dcp=0,
):
    """A per-rank step with glm_ref.step's signature for the instruments (check.check_candidate /
    test_accuracy.run_candidate_suite). `force` / `fault` are ignored; the kernel-side fault is tiles.fault."""
    gi = tuple(sorted(groups.items())) if groups else None
    tiles = K.Tiles(dot="mixed", recip="exact") if tiles is None else tiles

    def step_fn(params, caches, batch, force, fault, *, meta, mode="native", axis="tp"):
        del force, fault, mode
        d, R = meta.dims, meta.R
        ids, positions = batch["input_ids"], batch["positions"]
        seq_lens, qsl, bt_flat = batch["seq_lens"], batch["qsl"], batch["block_tables"]
        B = ids.shape[0]
        Sl = seq_lens.shape[0]
        cfg = from_dims(d, R, B)
        if dcp and dcp != R:  # V16: the caches / block tables are DCP-dcp (the A0 harness's DCP view)
            cfg = dataclasses.replace(cfg, DCP=dcp)
        assert Sl <= cfg.BMAX, (Sl, cfg.BMAX)
        nb = int(caches[0]["nope"].shape[0])
        idx_c = [c["idx"] for c in caches if "idx" in c]
        nbi = int(idx_c[0].shape[0]) if idx_c else nb
        bt2 = bt_flat.reshape(Sl, -1)
        max_pages = int(bt2.shape[1])
        r = lax.axis_index(axis).astype(I32)
        call, info, writer, winfo = _built(
            cfg, max_pages, nb, nbi, tiles, interpret, debug, stack_e, gi
        )
        # vocab-parallel embedding + all-reduce (out of the block; glm_ref E0/E1 op for op)
        vloc = d.vocab // R
        lid = ids - r * vloc
        oob = (lid < 0) | (lid >= vloc)
        e = params["embed"][jnp.where(oob, 0, lid)]
        e = jnp.where(oob[:, None], jnp.zeros_like(e), e)
        allp = lax.all_gather(e, axis)
        acc = allp[0].astype(F32)
        for k_ in range(1, R):
            acc = acc + allp[k_].astype(F32)
        h0 = acc.astype(BF16).astype(F32)
        scal, rmeta, qmeta, bt, bti, rvec = metadata(
            cfg, B, positions, seq_lens, qsl, bt2, nb, nbi, max_pages, r, delay=delay
        )
        wbt, wbti = bt, bti  # the writer's tables: the full local tables
        if cfg.VR > 1:
            bt, bti = v16_tables(cfg, bt, bti, r)
        rpos = positions + (1 if tiles.fault == "rope_off_by_one" else 0)
        rows = params["rope_table"][
            jnp.clip(rpos, 0, params["rope_table"].shape[0] - 1)
        ]
        tab = rope_table(cfg, rows)
        layers = params["layers"]
        sm = smalls_rows(cfg, layers, params["final_norm"], meta.k_scales)
        dw = dense_operands(cfg, layers, info["dkinds"])
        ew = expert_operands(cfg, layers, stack_e)
        cops = cache_operands(cfg, caches, info["groups"])
        ltab = jnp.asarray(info["ltab"])
        dtab = jnp.asarray(info["dtab"])
        with jax.enable_x64(
            False
        ):  # the instruments run with x64 on; the kernel body is 32-bit
            outs = call(
                scal,
                rmeta,
                qmeta,
                bt,
                bti,
                ltab,
                dtab,
                rvec,
                h0,
                tab,
                sm,
                *dw,
                *ew,
                *cops,
            )
            out, nrow, rrow, irec = outs[:4]
            new = writer(scal, rmeta, wbt, wbti, nrow, rrow, irec, *cops)
        new_caches = unstack_caches(cfg, list(new), info["groups"])
        aux = {}
        if debug:
            (
                d_state,
                d_attn,
                d_xin,
                d_xffn,
                d_qc,
                d_kvcn,
                d_qall,
                d_oloc,
                d_wtab,
                d_uv,
                d_ureq,
            ) = outs[4:15]
            aux["state_h"] = d_state[:, 0]
            aux["state_res"] = d_state[:, 1]
            aux["attn_h"] = d_attn
            aux["x_in"] = d_xin
            aux["x_ffn"] = d_xffn
            aux["q_c"] = d_qc
            aux["kv_cn"] = d_kvcn
            aux["q_all"] = unpack_qall(cfg, d_qall)
            aux["o_loc"] = jnp.transpose(d_oloc, (0, 2, 1, 3))
            T = qsl[-1]
            aux["router_ids"] = {
                l: router_ids(cfg, d_wtab[l]) for l in range(cfg.L) if cfg.is_moe(l)
            }
            fl = LY.layers_of(cfg, "iwq")
            aux["topk"] = {
                l: tables_from_union(cfg, d_uv[i], d_ureq[i], rmeta, qmeta, T)
                for i, l in enumerate(fl)
            }
        return out[:B], new_caches, aux

    return step_fn
