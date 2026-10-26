"""K-UDL shared spec: mixed-bucket (T512 / T2048) CSA attention cells with REAL top-k, on top of K1v2's instruments.

Block = `_attention_csa` after `_run_swa` (as K1v2): per token, its top-k compressed rows of a paged fp8 cache attended
with the SWA partial and the sink. Mixed cells: R verify windows (7 tokens, union drawn from the measured served
distribution, K1v2 `_window_lists`) + one prefill chunk whose top-k rows REPLAY a served dump (real weights, topk_tap
probe j-3aa888d2: dp0 = code text, dp1 = random tokens; layers as named). The twin's dummy-weight top-k is random and
unions exceed every cap (W5: the twin cannot judge union-dense), so every decision cell here is real-structure.

Long prefixes past the dumps (127k / 470k tokens) re-target the dump rows with one map per chunk: the recent band
(the chunk's own range + RECENT rows) shifts by the prefix delta, older rows scale onto the longer old range; the map is
injective, so every window's union count is the dump's. 470k > s_cap (65536 compressed rows = 262k tokens): the incumbent falls
back to the sparse path there (served: 14-18% of windows).

Inputs are K1v2-format dicts, so K1v2's device_inputs / oracle / faults / gate apply unchanged.
Data: KUDL_DATA (default: ./data next to this file) holds dp*_kv*.npz with arrays L<layer> = i32[2048, 1024].
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "k1v2"))
import k1v2_common as C  # noqa: E402

DATA = os.environ.get("KUDL_DATA", os.path.join(HERE, "data"))
VQ, PAGE, TOPK, RATIO = C.VQ, C.PAGE, C.TOPK, C.RATIO
RECENT = 1024  # compressed rows before a re-targeted chunk that shift (not scale) with the prefix

# name: (conc, T, R verify windows, dump, layer, chunk rows, prefix tokens (None = the dump's own))
_C16 = dict(conc=16, T=512, R=3, L=491)
_C8 = dict(conc=8, T=2048, R=1, L=2041)
SPECS = {
    "m16-r22k": dict(_C16, dump="dp1_kv6144", layer=2, prefix=None),
    "m16-r51k": dict(_C16, dump="dp0_kv13312", layer=2, prefix=None),
    "m16-r82k": dict(_C16, dump="dp0_kv20992", layer=4, prefix=None),
    "m16-r100k": dict(_C16, dump="dp1_kv25600", layer=2, prefix=None),
    "m16-r100kL1": dict(_C16, dump="dp1_kv25600", layer=1, prefix=None),
    "m16-s127k": dict(_C16, dump="dp1_kv25600", layer=2, prefix=127000),
    "m16-s470k": dict(_C16, dump="dp1_kv25600", layer=2, prefix=470000),
    "m16-s470kL1": dict(_C16, dump="dp1_kv25600", layer=1, prefix=470000),
    # stress (not served-like): uniformly random top-k (the twin's dummy-weight regime; U8 ~ 8k)
    "m16-rand127k": dict(_C16, dump=None, layer=0, prefix=127000),
    # smaller mixed buckets (no union-dense below 512 tokens: the incumbent is the sparse path)
    "m16t256-r100k": dict(_C16, T=256, L=235, dump="dp1_kv25600", layer=2, prefix=None),
    "m8t256-r100k": dict(_C8, T=256, L=249, dump="dp1_kv25600", layer=2, prefix=None),
    "m8t512-r100k": dict(_C8, T=512, L=505, dump="dp1_kv25600", layer=2, prefix=None),
    "m8t1024-r22k": dict(_C8, T=1024, L=1017, dump="dp1_kv6144", layer=2, prefix=None),
    "m8t1024-r100k": dict(_C8, T=1024, L=1017, dump="dp1_kv25600", layer=2, prefix=None),
    # U sweep (critical review): each real row keeps (1 - mix) of its picks and replaces the rest with random visible rows
    "m16-mix25": dict(_C16, dump="dp1_kv25600", layer=1, prefix=None, mix=0.25),
    "m16-mix50": dict(_C16, dump="dp1_kv25600", layer=1, prefix=None, mix=0.5),
    "m16-mix75": dict(_C16, dump="dp1_kv25600", layer=1, prefix=None, mix=0.75),
    "m8-mix50": dict(_C8, dump="dp1_kv25600", layer=1, prefix=None, mix=0.5),
    "m8-rand127k": dict(_C8, dump=None, layer=0, prefix=127000),
    "m8-r22k-p1m": dict(_C8, dump="dp1_kv6144", layer=2, prefix=None, pps=1025),
    "m8-r22k": dict(_C8, dump="dp1_kv6144", layer=2, prefix=None),
    "m8-r51k": dict(_C8, dump="dp0_kv13312", layer=2, prefix=None),
    "m8-r100k": dict(_C8, dump="dp1_kv25600", layer=2, prefix=None),
    "m8-s470k": dict(_C8, dump="dp1_kv25600", layer=2, prefix=470000),
}
# c16 decode replica of a mixed step (K1v2 cell): 4 windows + padding
K1V2_CELLS = ("c16m512-dec",)
BENCH16 = ["m16-r22k", "m16-r51k", "m16-r100k", "m16-r100kL1", "m16-s127k", "m16-s470k", "c16m512-dec"]
BENCH8 = ["m8-r22k", "m8-r100k", "m8-s470k"]


def _load(dump, layer):
    return np.load(os.path.join(DATA, dump + ".npz"))[f"L{layer}"]


def cell_obj(name):
    if name in C.CELLS:
        return C.CELLS[name]
    s = SPECS[name]
    ctx = C.CTX[s["conc"]][1]
    return C.Cell(name, s["conc"], s["T"], s["R"], (ctx,) * s["R"], (C.union_quantile(s["conc"], ctx, 0.5),) * s["R"],
                  prefill=(s["L"], s["prefix"] or 0, 0))


def build_inputs(name, seed=None):
    if name in C.CELLS:
        return C.build_inputs(C.CELLS[name], seed)
    s = SPECS[name]
    cell = cell_obj(name)
    rng = np.random.default_rng(1234 + (seed or 0))
    conc, H, T, R, Lc = s["conc"], cell.heads, s["T"], s["R"], s["L"]
    if s["dump"] is None:
        prefix = dprefix = s["prefix"]
        rows = np.full((Lc, TOPK), -1, np.int64)
        for t in range(Lc):
            nv = (prefix + t + 1) // RATIO
            rows[t, : min(TOPK, nv)] = rng.choice(nv, size=min(TOPK, nv), replace=False)
    else:
        dump_kv = int(s["dump"].split("kv")[1])
        dprefix = dump_kv * RATIO - 2048  # the dump chunk's first token position
        prefix = s["prefix"] if s["prefix"] is not None else dprefix
        rows = _load(s["dump"], s["layer"])[:Lc].astype(np.int64)
    if s.get("mix"):
        out = np.full_like(rows, -1)
        for t in range(Lc):
            r = rows[t][rows[t] >= 0]
            nv = int(r.max()) + 1
            keep = rng.choice(r, size=int(round(r.size * (1 - s["mix"]))), replace=False)
            pool = np.setdiff1d(np.arange(nv), keep)
            add = rng.choice(pool, size=min(r.size - keep.size, pool.size), replace=False)
            r2 = np.concatenate([keep, add])
            out[t, : r2.size] = r2
        rows = out
    if prefix != dprefix:
        # One map for every token (a row picked by several tokens stays one row): rows in the recent band (the
        # chunk's own range + RECENT rows before it) shift by the prefix delta (causality and recency kept exactly);
        # older rows scale onto the longer old range (monotonic, injective for a scale >= 1).
        delta = (prefix - dprefix) // RATIO
        cut = max(0, dprefix // RATIO - RECENT)
        f = (prefix // RATIO - RECENT) / max(cut, 1)
        out = np.full_like(rows, -1)
        for t in range(Lc):
            r = rows[t][rows[t] >= 0]
            r2 = np.where(r >= cut, r + delta, np.floor(r * f).astype(np.int64))
            out[t, : r2.size] = r2
        rows = out
    ms = C.MAX_SEQS[conc]
    ctx = list(cell.ctx)
    U = list(cell.U)
    lens = [VQ] * R + [Lc]
    ctx.append(prefix)
    nseq = len(lens)
    seq_tokens = [c + l for c, l in zip(ctx, lens)]
    kv_lens = [st // RATIO for st in seq_tokens]
    kv_lens[-1] = max(kv_lens[-1], int(rows.max()) + 1)  # rc7 indexer: a token may pick its own open block
    need = [-(-max(k + 1, 1) // PAGE) + 1 for k in kv_lens]
    pps = max(max(need) + 1, s.get("pps", 0))  # "pps": served table width (max-model-len 1M -> 1024 pages)
    total_pages = sum(need) + 8
    perm = rng.permutation(total_pages - 1).astype(np.int32) + 1
    table = np.zeros((ms, pps), np.int32)
    o = 0
    for i in range(nseq):
        table[i, : need[i]] = perm[o:o + need[i]]
        o += need[i]
    cu = np.zeros(ms + 1, np.int32)
    cu[1:nseq + 1] = np.cumsum(lens)
    cu[nseq + 1:] = cu[nseq]
    tk = np.full((T, TOPK), -1, np.int32)
    seq_of = np.full(T, -1, np.int32)
    pos = np.full(T, -1, np.int64)
    t = 0
    for i in range(R):
        nv = [(ctx[i] + j + 1) // RATIO for j in range(VQ)]
        tk[t:t + VQ] = C._window_lists(rng, nv, U[i])
        seq_of[t:t + VQ] = i
        pos[t:t + VQ] = ctx[i] + np.arange(VQ)
        t += VQ
    tk[t:t + Lc] = rows.astype(np.int32)
    seq_of[t:t + Lc] = R
    pos[t:t + Lc] = prefix + np.arange(Lc)
    dist = np.array([0, R, nseq], np.int32)
    union = []
    for i in range(R):
        r_ = tk[i * VQ:(i + 1) * VQ]
        union.append(int(np.unique(r_[r_ >= 0]).size))
    return dict(cell=cell, H=H, T=T, nseq=nseq, ms=ms, lens=lens, ctx=ctx,
                kv_lens=np.array(kv_lens + [0] * (ms - nseq), np.int32), table=table, pps=pps,
                total_pages=total_pages, cu=cu, dist=dist, tk=tk, seq_of=seq_of, pos=pos, union=union,
                U_target=U[:R], rng=rng)


def drivers(inp):
    """Union sizes of the prefill chunk at window sizes 8 / 32 / 64 and the verify windows (work drivers)."""
    tk, R = inp["tk"], inp["cell"].R
    d = C.drivers(inp)
    pf = inp["cell"].prefill
    if pf:
        a = R * VQ
        rows = tk[a:a + pf[0]]
        for bq in (8, 16, 32, 64):
            us = [int(np.unique(b[b >= 0]).size) for b in (rows[i:i + bq] for i in range(0, rows.shape[0], bq))]
            d[f"pf_U{bq}_mean"] = round(float(np.mean(us)), 1)
            d[f"pf_tiles{bq}"] = int(sum(-(-u // TOPK) for u in us))
        d["kv_len_pf"] = int(inp["kv_lens"][R])
    return d


def _region(conc, **extra):
    """`_attention_csa` verbatim (served kwargs from K1v2's kernel_tuning), `_run_swa` stubbed, plus `extra` kwargs."""
    import jax.numpy as jnp
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import deepseek_v4_attention_op as op
    kw = C.kernel_tuning(conc)
    kw.update(extra)

    def f(d, q):
        orig = op._run_swa
        op._run_swa = lambda **k: (d["swa_acc"], k["sw_cache"], d["swa_l"], d["swa_m"])
        try:
            dummy = jnp.zeros((1, 8, 128), jnp.int32)
            out, _ = op._attention_csa(
                q, jnp.zeros((q.shape[0], C.D), jnp.bfloat16), dummy, d["kv_lens"], d["pages"], d["cu"], d["dist"],
                d["nope"], d["kv_lens"], d["topk"], d["pages"], d["cu"], d["dist"], d["sinks"], d["rope"],
                sm_scale=C.SM_SCALE, sliding_window=128, logical_page_size=128, swa_only=False,
                two_caches_same_buffer=False, verify_q_len=VQ, **kw)
        finally:
            op._run_swa = orig
        return out
    return f


def inc(conc):
    return _region(conc)


def win(conc):
    """Candidate: TPU_DSV4_CSA_WINDOW_MIXED=1 (any head count, T > 256)."""
    return _region(conc, window_mixed=True, window_mixed_above=256, window_mixed_min_heads=1)


def win128(conc):
    """Candidate with T > 128 (T256 buckets too)."""
    return _region(conc, window_mixed=True, window_mixed_above=128, window_mixed_min_heads=1)


def c16cfg(conc):
    """Proposed c16 env: TPU_DSV4_CSA_WINDOW_MIXED=1, _ABOVE=128, _GROUPS=4 (MIN_HEADS 32, MAX_TOKENS 2048 defaults)."""
    return _region(conc, window_mixed=True, window_mixed_above=128, window_mixed_min_heads=32, window_mixed_groups=4,
                   window_mixed_max_tokens=2048)


def tp8cfg(conc, groups=1):
    """Proposed c4/c8 env: _MIXED=1, _ABOVE=128, _MIN_HEADS=16, _MAX_TOKENS=1024 (T2048 stays union-dense)."""
    return _region(conc, window_mixed=True, window_mixed_above=128, window_mixed_min_heads=16,
                   window_mixed_groups=groups, window_mixed_max_tokens=1024)


def tp8cfg_g4(conc):
    return tp8cfg(conc, groups=4)


def wq16(conc, groups=1, max_tokens=2048):
    return _region(conc, window_mixed=True, window_mixed_above=128, window_mixed_min_heads=1,
                   window_mixed_groups=groups, window_mixed_max_tokens=max_tokens, window_mixed_wq=16)


def wq16_g2(conc):
    return wq16(conc, groups=2)


def wq16_g4(conc):
    return wq16(conc, groups=4)


def win_g2(conc):
    return _region(conc, window_mixed=True, window_mixed_above=256, window_mixed_min_heads=1, window_mixed_groups=2)


def win_g4(conc):
    return _region(conc, window_mixed=True, window_mixed_above=256, window_mixed_min_heads=1, window_mixed_groups=4)


def sparse(conc):
    """Reference arm: union-dense off (the per-token sparse path)."""
    return _region(conc, union_dense=False)


def win_l1(conc):
    """Component: the window route's SparseCore call alone (dedup + rank + gather), outputs summed into q's dtype."""
    import jax.numpy as jnp
    from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_window_union as cwu

    def f(d, q):
        r = cwu.prepare(d["topk"], d["kv_lens"], d["pages"], d["cu"], d["dist"], d["nope"], d["rope"])
        s = (r["nope_g"][0, 0] + r["rope_g"][0, 0] + r["mask"].reshape(-1)[0] + r["ucnt"][0]).astype(jnp.float32)
        return q + (s * 0).astype(q.dtype)
    return f


def _patched(f, mod, **attrs):
    """Arm wrapper: module attributes set only while the region traces (jit traces inside the call)."""
    def g(d, q):
        old = {k: getattr(mod, k) for k in attrs}
        for k, v in attrs.items():
            setattr(mod, k, v)
        try:
            return f(d, q)
        finally:
            for k, v in old.items():
                setattr(mod, k, v)
    return g


def win_bf16p(conc):
    """Candidate + bf16 P into the PV matmul (L2 pv_bf16)."""
    from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_window_union as cwu
    return _patched(win(conc), cwu, L2_CONFIG=dict(cwu.L2_CONFIG, pv_bf16=True))


def win_l1_nog(conc):
    """Removal arm: the SparseCore call without the row gather (dedup, ranks, masks only)."""
    import jax.numpy as jnp
    from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_window_union as cwu

    def f(d, q):
        T = d["topk"].shape[0]
        ms = d["cu"].shape[0] - 1
        pps = d["pages"].shape[0] // ms
        meta = cwu.window_meta(d["cu"], d["dist"], d["kv_lens"], num_tokens=T, s_cap=pps * PAGE)
        r = cwu.union_gather(meta, d["topk"], d["pages"], d["nope"], d["rope"], num_tokens=T, max_num_seqs=ms,
                             skip_gather=True)
        s = (r[2].reshape(-1)[0] + r[3][0]).astype(jnp.float32)
        return q + (s * 0).astype(q.dtype)
    return f


def win_l2(conc):
    """Component: the TensorCore call alone on precomputed L1 outputs (L1 hoisted: top-k not barriered per call)."""
    from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_window_union as cwu

    def f(d, q):
        r = cwu.prepare(d["topk0"], d["kv_lens"], d["pages"], d["cu"], d["dist"], d["nope"], d["rope"])
        return cwu.attend(q, r, d["sinks"], d["swa_acc"], d["swa_l"], d["swa_m"], sm_scale=C.SM_SCALE)
    return f


ARMS = {"wq16": wq16, "wq16_g2": wq16_g2, "wq16_g4": wq16_g4, "c16cfg": c16cfg, "tp8cfg": tp8cfg, "tp8cfg_g4": tp8cfg_g4, "win128": win128, "win_l2": win_l2, "win_l1_nog": win_l1_nog, "inc": inc, "win": win, "win_g2": win_g2, "win_g4": win_g4, "sparse": sparse, "win_l1": win_l1, "win_bf16p": win_bf16p}
