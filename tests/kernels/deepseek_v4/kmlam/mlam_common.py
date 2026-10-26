"""K-MLAM instruments: mixed-step (T > 32) HCA-layer attention, one spec shared by acc_mixed.py and bench_mixed.py.

Block under test (DSV4-Pro HCA layer, mixed / prefill-chunk steps, per device):
  _attention_hca at T in {64..2048}: stock = SWA kernel (mixed pass for the prefill chunk) + MLA-p (verify windows)
  + MLA-m (prefill chunk, bq 32, 64 pages/block) + the l/m hand-off glue.            [stock, stack 4a71c4e46]
Inputs: q bf16[T,H,512] (post qnorm_rope), kv bf16[T,512] (post RoPE), caches in the served layouts (block 1024):
SWA int32[N,256,128] (128 tokens/page, 2 word rows/token), HCA u8[N,16,4,128] (8 compressed rows/page).
Requests are ordered decode (q 1), verify (q 7), mixed (any other q_len) as the runner's distribution.
Cells = the served mixed buckets: c16 T512 (TP4, H32, DP4 replica: <=3 verify windows + one prefill chunk),
c4/c8 T2048 (TP8, H16), at prefill prefixes 30k / 120k / 470k (decomposition rows 6/10: the long-prefix T512 step).
"""
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hca_common as C  # noqa: E402

import numpy as np  # noqa: E402

VQ = C.VQ
D = C.D
W = C.WINDOW
RATIO = C.RATIO
P = C.SWA_PAGE
HP = C.HCA_PAGE


@dataclasses.dataclass(frozen=True)
class MCell:
    name: str
    heads: int
    T: int  # step token bucket
    reqs: tuple  # ((seq_len incl. new rows, q_len), ...) ordered decode, verify, mixed
    max_seqs: int
    pool: int


def _pref(prefix, q):
    return (prefix + q, q)


def make_cells():
    c = {}
    ver16 = ((80000 + 64, VQ), (98304 + 64, VQ), (120000 + 64, VQ))
    ver8 = ((90000 + 64, VQ), (131072 + 64, VQ), (165000 + 64, VQ))
    for tag, P_ in (("s", 30000 + 37), ("m", 120000 + 77), ("l", 470000 + 101)):
        c[f"c16T512-{tag}"] = MCell(f"c16T512-{tag}", 32, 512, ver16 + (_pref(P_, 512 - 21),), 16, 4288)
        c[f"c8T2048-{tag}"] = MCell(f"c8T2048-{tag}", 16, 2048, ver8 + (_pref(P_, 2048 - 21),), 8, 4818)
    c["c4T2048-m"] = MCell("c4T2048-m", 16, 2048, ((131072 + 64, VQ), _pref(120000 + 77, 2048 - 7)), 4, 4815)
    # decode-only replica in a T512 program (DP lockstep: the other replicas of a c16 mixed step)
    c["c16T512-ver4"] = MCell("c16T512-ver4", 32, 512, ver16 + ((110000 + 64, VQ),), 16, 4288)
    # edge cells (accuracy; also run by the bench as guardrails where noted)
    c["edge-fresh"] = MCell("edge-fresh", 32, 512, ((80000 + 64, VQ), (491, 491)), 16, 1024)  # first chunk of a prompt
    c["edge-second"] = MCell("edge-second", 16, 512, ((5000 + 64, 1), (98304 + 3, VQ), (1000 + 500, 504)), 8, 1024)
    c["edge-small"] = MCell("edge-small", 16, 64, ((90000 + 64, 1), (70000 + 64, VQ), (40000 + 127, 40)), 8, 1024)
    c["edge-twopf"] = MCell("edge-twopf", 32, 512, ((60000 + 64, VQ), (30000 + 200, 200), (50000 + 3, 305)), 16, 1024)
    c["edge-sink"] = MCell("edge-sink", 16, 256, ((135, VQ), (128 * 900 + 64 + 200, 200)), 8, 1024)
    return c


CELLS = make_cells()
PHASE_CELLS = [k for k in CELLS if not k.startswith("edge")]


@dataclasses.dataclass
class MInputs:
    cell: MCell
    q: np.ndarray  # f32 (bf16 values) [T,H,512]
    kv: np.ndarray  # f32 (bf16 values) [T,512] (post-RoPE new rows)
    sinks: np.ndarray
    positions: np.ndarray
    seq_lens: np.ndarray
    cu: np.ndarray
    dist: np.ndarray
    swa_bt: np.ndarray
    hca_bt: np.ndarray
    swa_words: dict
    hca_words: dict


def build(cell: MCell, seed=0, sink_shift=None):
    rng = np.random.default_rng(seed)
    H, T, ms, N = cell.heads, cell.T, cell.max_seqs, cell.pool
    reqs = cell.reqs
    R = len(reqs)
    n_tok = sum(q for _, q in reqs)
    assert n_tok <= T, (cell.name, n_tok, T)
    Psw = C.MAX_MODEL_LEN // P
    Ph = C.MAX_MODEL_LEN // C.BLOCK
    perm = iter(rng.permutation(np.arange(1, N)).tolist())  # page 0 = null page
    seq = np.zeros(ms, np.int32)
    cu = np.zeros(ms + 1, np.int32)
    pos = np.zeros(T, np.int32)
    swa_bt = np.zeros((ms, Psw), np.int32)
    hca_bt = np.zeros((ms, Ph), np.int32)
    swa_words, hca_words = {}, {}
    t = 0
    for r, (L, ql) in enumerate(reqs):
        seq[r] = L
        cu[r + 1] = cu[r] + ql
        pos[t:t + ql] = np.arange(L - ql, L)
        t += ql
        for lp in range(max(0, L - ql - W) // P, (L - 1) // P + 1):
            pg = next(perm)
            swa_bt[r, lp] = pg
            swa_words[pg] = C.tokens_to_words(rng.standard_normal((P, D)).astype(np.float32)).reshape(2 * P, 128)
        n_entries = (L - 1) // RATIO + 1
        for hpg in range((n_entries - 1) // HP + 1):
            pg = next(perm)
            hca_bt[r, hpg] = pg
            hca_words[pg] = C.tokens_to_words(rng.standard_normal((HP, D)).astype(np.float32)).reshape(2 * HP, 128)
    cu[R + 1:] = cu[R]
    kinds = [0 if q == 1 else (1 if q == VQ else 2) for _, q in reqs]
    assert kinds == sorted(kinds), "requests must be ordered decode, verify, mixed"
    dist = np.array([kinds.count(0), kinds.count(0) + kinds.count(1), R], np.int32)
    bf = C.bf16_round
    shift = (6.0 if cell.name == "edge-sink" else 0.0) if sink_shift is None else sink_shift
    return MInputs(cell, bf(rng.standard_normal((T, H, D))), bf(rng.standard_normal((T, D))),
                   (rng.uniform(-1, 1, H) + shift).astype(np.float32), pos, seq, cu, dist, swa_bt, hca_bt,
                   swa_words, hca_words)


# ----------------------------------------------------------------------------- f64 oracle (vectorized)
FAULTS = ("drop_row", "vis_min_blk", "vis_max_blk", "drop_first512", "swa_window", "causal_fresh", "no_sink",
          "wrong_req")
FAULT_DOC = {
    "drop_row": "last visible compressed row of every token dropped",
    "vis_min_blk": "every token sees only the compressed rows of the first token of its 16-token block",
    "vis_max_blk": "every token sees the compressed rows of the last token of its 16-token block (causal leak)",
    "drop_first512": "compressed rows [0, 512) dropped (tile loop starts late)",
    "swa_window": "oldest window row dropped (window 127)",
    "causal_fresh": "a token also sees the later new rows of its request",
    "no_sink": "attention sink omitted from the denominator",
    "wrong_req": "compressed rows read from the next request's pages",
}


def _memo(inp):
    if not hasattr(inp, "_memo"):
        inp._memo = {}
    return inp._memo


def _hrows(inp, rr, n_h):
    _MEMO = _memo(inp)
    key = ("h", rr, n_h)
    if key not in _MEMO:
        hrows = np.zeros((n_h, D))
        for k in range(n_h):
            pg = inp.hca_bt[rr, k // HP]
            if pg == 0 or pg not in inp.hca_words:
                continue
            s = k % HP
            hrows[k] = C.words_to_tokens(inp.hca_words[pg][2 * s:2 * s + 2])
        _MEMO[key] = hrows
    return _MEMO[key]


def oracle(inp: MInputs, fault=None):
    """f64 attention of every real token -> [T,H,512] (pad rows zero). New rows = inp.kv (bf16 values).
    Vectorized: per request, compressed-row scores of all its tokens in one matmul (memoized per source request),
    masked per token; window scores per token; one softmax over both."""
    c = inp.cell
    H = c.heads
    out = np.zeros((c.T, H, D))
    kv_new = inp.kv.astype(np.float64)
    sinks = inp.sinks.astype(np.float64)
    t0 = 0
    R = len(c.reqs)
    for r, (L, ql) in enumerate(c.reqs):
        pf = L - ql
        rr = (r + 1) % R if fault == "wrong_req" else r
        n_h = L // RATIO + 1
        hrows = _hrows(inp, rr, n_h)
        _MEMO = _memo(inp)
        key = ("s", r, rr)
        if key not in _MEMO:
            qv = inp.q[t0:t0 + ql].astype(np.float64).reshape(ql * H, D)
            _MEMO[key] = (qv @ hrows.T * C.SM_SCALE).reshape(ql, H, n_h)
        sh_all = _MEMO[key]
        s0 = max(pf - (W - 1), 0)
        cached = np.zeros((pf - s0, D))
        for i, p in enumerate(range(s0, pf)):
            pg = inp.swa_bt[r, p // P]
            s = p % P
            cached[i] = C.words_to_tokens(inp.swa_words[pg][2 * s:2 * s + 2])
        allsw = np.concatenate([cached, kv_new[t0:t0 + ql]], 0)  # positions s0 .. L-1
        vis_tok = (pf + np.arange(ql) + 1) // RATIO
        vis_eff = vis_tok.copy()
        if fault in ("vis_min_blk", "vis_max_blk"):
            for b0 in range(0, ql, 16):
                blk = vis_tok[b0:b0 + 16]
                vis_eff[b0:b0 + 16] = blk.min() if fault == "vis_min_blk" else blk.max()
        if fault == "drop_row":
            vis_eff = np.maximum(vis_eff - 1, 0)
        lo_h = 512 if fault == "drop_first512" else 0
        for i in range(ql):
            p = pf + i
            hi = L - 1 if fault == "causal_fresh" else p
            lo = max(0, p - W + 1) + (1 if fault == "swa_window" and p - W + 1 >= 0 else 0)
            sk = allsw[lo - s0:hi - s0 + 1]
            qv = inp.q[t0 + i].astype(np.float64)
            ss = qv @ sk.T * C.SM_SCALE  # [H, nw]
            hs = slice(lo_h, max(lo_h, int(vis_eff[i])))
            sh = sh_all[i][:, hs]
            m = ss.max(1)
            if sh.shape[1]:
                m = np.maximum(m, sh.max(1))
            ps = np.exp(ss - m[:, None])
            num = ps @ sk
            den = ps.sum(1)
            if sh.shape[1]:
                ph = np.exp(sh - m[:, None])
                num = num + ph @ hrows[hs]
                den = den + ph.sum(1)
            if fault != "no_sink":
                den = den + np.exp(sinks - m)
            out[t0 + i] = num / den[:, None]
        t0 += ql
    return out


def device(inp: MInputs):
    import jax.numpy as jnp
    from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout as CL
    c = inp.cell
    bf = lambda x: jnp.asarray(x, jnp.float32).astype(jnp.bfloat16)
    return dict(q=bf(inp.q), kv=bf(inp.kv), sinks=jnp.asarray(inp.sinks), positions=jnp.asarray(inp.positions),
                seq_lens=jnp.asarray(inp.seq_lens), cu=jnp.asarray(inp.cu), dist=jnp.asarray(inp.dist),
                swa_bt=jnp.asarray(inp.swa_bt), hca_bt=jnp.asarray(inp.hca_bt),
                swa=jnp.asarray(C.scatter_pages(c.pool, 2 * P, inp.swa_words)),
                hca=CL.words_to_slabs(jnp.asarray(C.scatter_pages(c.pool, 2 * HP, inp.hca_words))))


def stock_env():
    C.stock_env()
    for k, v in (("TPU_PALLAS_HBM_PIN", "1"), ("TPU_DSV4_FUSED_HCA_DECODE", "1")):
        os.environ.setdefault(k, v)


def stock_attn(**env_override):
    """The incumbent: _attention_hca exactly as the served HCA layer calls it at T > 32 (kv already RoPE'd).
    f(d) -> (out, swa)."""
    stock_env()
    from vllm_torchtpu import envs
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_attention_op import _attention_hca
    kvp = env_override.get("kvp", envs.TPU_DSV4_HCA_NUM_KV_PAGES_PER_BLOCK)
    nq = env_override.get("nq", envs.TPU_DSV4_HCA_NUM_QUERIES_PER_BLOCK)

    def f(d):
        return _attention_hca(
            d["q"], d["kv"], d["swa"], d["seq_lens"], d["swa_bt"], d["cu"], d["dist"],
            d["hca"], d["seq_lens"] // RATIO, (d["positions"] + 1) // RATIO, d["hca_bt"], d["cu"], d["dist"],
            d["sinks"], sm_scale=C.SM_SCALE, sliding_window=W, logical_page_size=P, swa_only=False,
            two_caches_same_buffer=False, hca_num_kv_pages_per_block=tuple(kvp), hca_num_queries_per_block=tuple(nq),
            verify_q_len=VQ)
    return f
