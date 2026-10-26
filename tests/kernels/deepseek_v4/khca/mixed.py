"""K-HCA instrument (added at Step 4, v2): steps whose requests are NOT all verify windows.

The fused kernel serves every T<=32 step (no runtime cond, STRAT-DSV4-01), so it must be right for decode-only (q_len 1)
requests and short prefill tails mixed with verify windows. hca_common.oracle assumes q_len = 7; this file holds a general
attention-only f64 oracle (SWA window + HCA rows read from the caches + sink) and the TPU check:

  python mixed.py [--cells ...] --candidate module:fn_rope_attn [--out mixed.jsonl]

Per cell: stock ("rope","attn") and candidate vs the f64 oracle, gate = hca_common-style TOL(stock error); SWA cache
bit-exact vs stock; output rows past the last token must be finite (candidate: zeros).
"""
import argparse
import dataclasses
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hca_common as C  # noqa: E402

import numpy as np  # noqa: E402

# name: (heads, [(seq_len incl. new rows, q_len), ...]) ; order = decode (q 1), verify (q 7), mixed (other)
MIXED = {
    "mix-dec-ver": (16, [(90000 + 64, 1), (131072 + 5, 1), (100000 + 40, 7)]),  # T16, 2 decode + 1 verify
    "mix-ver-tail": (16, [(131072 + 64, 7), (5000 + 90, 9)]),  # T16, verify + 9-token prefill tail (2 chunks)
    "mix-tail-long": (32, [(98304 + 64, 7), (20000 + 3, 23)]),  # T32 H32, tail of 23 crossing a 128 boundary
    "mix-dec-only": (16, [(70000 + 10, 1)] * 1 + [(65536 + 127, 1)]),  # T8 two decode rows
    "mix-short": (16, [(40, 1), (300, 7)]),  # tiny contexts: window shorter than 127, 0/2 compressed rows
    "mix-c16": (32, [(90000 + 64, 1), (80000 + 64, 7), (100000 + 77, 7), (110000 + 64, 13)]),  # T32 R4
}


@dataclasses.dataclass
class MInputs:
    heads: int
    reqs: list
    T: int
    q: np.ndarray
    kv: np.ndarray
    sinks: np.ndarray
    positions: np.ndarray
    seq_lens: np.ndarray
    cu: np.ndarray
    dist: np.ndarray
    swa_bt: np.ndarray
    hca_bt: np.ndarray
    swa_words: dict
    hca_words: dict
    pool: int
    max_seqs: int


def bucket(n):
    for b in (8, 16, 32):
        if n <= b:
            return b
    raise ValueError(n)


def build(name, seed=0, max_seqs=16, contig=False):
    H, reqs = MIXED[name]
    rng = np.random.default_rng(seed)
    n_tok = sum(q for _, q in reqs)
    T = bucket(n_tok)
    R = len(reqs)
    P = C.MAX_MODEL_LEN // C.SWA_PAGE
    Ph = C.MAX_MODEL_LEN // C.BLOCK
    need = sum(3 + (L // C.RATIO) // C.HCA_PAGE + 2 for L, _ in reqs)
    pool = need + 8
    perm = iter((np.arange(1, pool) if contig else rng.permutation(np.arange(1, pool))).tolist())
    seq = np.zeros(max_seqs, np.int32)
    cu = np.zeros(max_seqs + 1, np.int32)
    pos = np.zeros(T, np.int32)
    swa_bt = np.zeros((max_seqs, P), np.int32)
    hca_bt = np.zeros((max_seqs, Ph), np.int32)
    swa_words, hca_words = {}, {}
    t = 0
    for r, (L, ql) in enumerate(reqs):
        seq[r] = L
        cu[r + 1] = cu[r] + ql
        pos[t:t + ql] = np.arange(L - ql, L)
        t += ql
        for lp in range(max(0, L - ql - C.WINDOW) // C.SWA_PAGE, (L - 1) // C.SWA_PAGE + 1):
            pg = next(perm)
            swa_bt[r, lp] = pg
            swa_words[pg] = C.tokens_to_words(rng.standard_normal((C.SWA_PAGE, C.D)).astype(np.float32)).reshape(
                2 * C.SWA_PAGE, 128)
        n_entries = (L - 1) // C.RATIO + 1
        for hp in range((n_entries - 1) // C.HCA_PAGE + 1):
            pg = next(perm)
            hca_bt[r, hp] = pg
            hca_words[pg] = C.tokens_to_words(rng.standard_normal((C.HCA_PAGE, C.D)).astype(np.float32)).reshape(
                2 * C.HCA_PAGE, 128)
    cu[R + 1:] = cu[R]
    n_dec = sum(1 for _, q in reqs if q == 1)
    n_ver = sum(1 for _, q in reqs if q == C.VQ)
    order_ok = [q for _, q in reqs] == sorted([q for _, q in reqs], key=lambda q: 0 if q == 1 else (1 if q == C.VQ else 2))
    assert order_ok, "requests must be ordered decode, verify, mixed"
    dist = np.array([n_dec, n_dec + n_ver, R], np.int32)
    bf = C.bf16_round
    return MInputs(H, reqs, T, bf(rng.standard_normal((T, H, C.D))), bf(rng.standard_normal((T, C.D))),
                   rng.uniform(-1, 1, H).astype(np.float32), pos, seq, cu, dist, swa_bt, hca_bt, swa_words,
                   hca_words, pool, max_seqs)


def oracle_attn(inp: MInputs, kv_r_b):
    """f64 attention of every real token (rows of other tokens left zero). kv_r_b: f64 [T,512] new rows (bf16 values)."""
    out = np.zeros((inp.T, inp.heads, C.D))
    t = 0
    for r, (L, ql) in enumerate(inp.reqs):
        pf = L - ql

        def swa_row(p):
            if p >= pf:
                return kv_r_b[t + (p - pf)]
            pg = inp.swa_bt[r, p // C.SWA_PAGE]
            s = p % C.SWA_PAGE
            return C.words_to_tokens(inp.swa_words[pg][2 * s:2 * s + 2]).astype(np.float64)

        def hca_row(k):
            pg = inp.hca_bt[r, k // C.HCA_PAGE]
            s = k % C.HCA_PAGE
            return C.words_to_tokens(inp.hca_words[pg][2 * s:2 * s + 2]).astype(np.float64)

        for i in range(ql):
            p = pf + i
            srows = np.stack([swa_row(x) for x in range(max(0, p - C.WINDOW + 1), p + 1)])
            vis = (p + 1) // C.RATIO
            keys = np.concatenate([srows] + ([np.stack([hca_row(k) for k in range(vis)])] if vis else []), 0)
            s = inp.q[t + i].astype(np.float64) @ keys.T * C.SM_SCALE
            m = s.max(1, keepdims=True)
            pr = np.exp(s - m)
            out[t + i] = (pr @ keys) / (pr.sum(1) + np.exp(inp.sinks - m[:, 0]))[:, None]
        t += ql
    return out


def device(inp: MInputs):
    import jax.numpy as jnp
    from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout as CL
    bf = lambda x: jnp.asarray(x, jnp.float32).astype(jnp.bfloat16)
    return dict(q=bf(inp.q), kv=bf(inp.kv), sinks=jnp.asarray(inp.sinks), positions=jnp.asarray(inp.positions),
                seq_lens=jnp.asarray(inp.seq_lens), cu=jnp.asarray(inp.cu), dist=jnp.asarray(inp.dist),
                swa_bt=jnp.asarray(inp.swa_bt), hca_bt=jnp.asarray(inp.hca_bt),
                swa=jnp.asarray(C.scatter_pages(inp.pool, 2 * C.SWA_PAGE, inp.swa_words)),
                hca=CL.words_to_slabs(jnp.asarray(C.scatter_pages(inp.pool, 2 * C.HCA_PAGE, inp.hca_words))),
                st=jnp.zeros((1,), jnp.int32), hidden=None)


def stock_attn():
    """Stock rope + _attention_hca on a general distribution: returns f(d, cs) -> (out, swa, kv_roped)."""
    C.stock_env()
    from vllm_torchtpu import envs
    from vllm_torchtpu.kernels.deepseek_v4 import rope as rope_kernel
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_attention_op import _attention_hca

    def f(d, cs):
        kv = rope_kernel.rope(d["kv"], d["positions"], cs)
        out, swa = _attention_hca(
            d["q"], kv, d["swa"], d["seq_lens"], d["swa_bt"], d["cu"], d["dist"],
            d["hca"], d["seq_lens"] // C.RATIO, (d["positions"] + 1) // C.RATIO, d["hca_bt"], d["cu"], d["dist"],
            d["sinks"], sm_scale=C.SM_SCALE, sliding_window=C.WINDOW, logical_page_size=C.SWA_PAGE, swa_only=False,
            two_caches_same_buffer=False,
            hca_num_kv_pages_per_block=envs.TPU_DSV4_HCA_NUM_KV_PAGES_PER_BLOCK,
            hca_num_queries_per_block=envs.TPU_DSV4_HCA_NUM_QUERIES_PER_BLOCK, verify_q_len=C.VQ)
        return out, swa, kv
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(MIXED))
    ap.add_argument("--candidate", required=True, help="module:fn, fn(d, cs) -> (out, swa, kv) like stock_attn()")
    ap.add_argument("--out", default=os.path.join(os.environ.get("CDK_OUTPUT_DIR", "."), "mixed.jsonl"))
    a = ap.parse_args()
    C.stock_env()
    import jax
    import jax.numpy as jnp
    m, fn = a.candidate.split(":")
    cand = getattr(importlib.import_module(m), fn)
    f_stock = stock_attn()
    cs_np = C.cos_sin_np()
    cs = jnp.asarray(cs_np)
    from test_accuracy import err, gate
    with open(a.out, "a") as fo:
        for name in a.cells.split(","):
            try:
                inp = build(name)
                d = device(inp)
                d = {k: v for k, v in d.items() if v is not None}
                so, sswa, skv = jax.block_until_ready(jax.jit(f_stock)(d, cs))
                co, cswa, ckv = jax.block_until_ready(jax.jit(cand)(d, cs))
                kv_r_b = np.asarray(skv.astype(jnp.float32)).astype(np.float64)
                ref = oracle_attn(inp, kv_r_b)
                n = int(inp.cu[len(inp.reqs)])
                es = err(np.asarray(so.astype(jnp.float32))[:n], ref[:n])
                ec = err(np.asarray(co.astype(jnp.float32))[:n], ref[:n])
                tol = gate(es)
                pad = np.asarray(co.astype(jnp.float32))[n:]
                r = dict(cell=name, T=inp.T, H=inp.heads, reqs=inp.reqs, dist=inp.dist.tolist(),
                         stock_err_max_rel=es[0], stock_err_mean=es[1], cand_err_max_rel=ec[0], cand_err_mean=ec[1],
                         tol_max_rel=tol[0], out_pass=bool(ec[0] <= tol[0] and ec[1] <= tol[1]),
                         swa_bitexact=bool(np.array_equal(np.asarray(sswa), np.asarray(cswa))),
                         pad_rows_zero=bool(np.all(pad == 0)), pad_rows_finite=bool(np.all(np.isfinite(pad))))
            except Exception as ex:
                import traceback
                traceback.print_exc()
                r = dict(cell=name, error=repr(ex)[:500])
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            print("MIXED " + json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
