"""K1v2 shared spec: phase cells, served-faithful inputs, float64 oracle, fault list, region builders.

One spec, consumed by test_accuracy.py and bench.py, so test and bench never disagree on shapes.

Block = the post-SWA part of the DSV4 CSA attention op (`_attention_csa` after `_run_swa`): per token, top-k (1024)
compressed rows of a paged fp8 cache (page 256 rows, 640 B/row) attended with the SWA partial (acc, l, m) and the sink
folded into one softmax. Incumbent region = `deepseek_v4_attention_op._attention_csa` VERBATIM with `_run_swa` replaced by
a function returning the precomputed SWA partials (identical in every arm; the SWA kernel is outside the block).

Inputs follow production, not uniform random:
  * verify windows of 7 tokens (6 spec tokens), strict-causal per-token valid rows (pos+1)//4 (TPU_STREAMIDX_STRICT_CAUSAL=1);
  * per-window union size U drawn from the MEASURED served distribution (k1u tap, c16 j-8435dbdc / c8 j-56414133,
    union_hist.npz: [layer, ctx bin of 4096 compressed rows, union bin of 32]) conditional on the window's context bin;
    a window's 7 lists are built to have exactly that union (token 0 = 1024 rows; token j keeps 1024 - t_j of them and adds
    t_j new rows, sum t_j = U - 1024);
  * page tables from a SHUFFLED page pool (page 0 reserved), bucket padding rows with all -1 top-k;
  * mixed T512 (c16): decode-replica (R windows + padding) and prefilling replica (R windows + one prefill chunk at a
    prefix, top-k per 32-token block drawn from a block pool of U32 rows, U32 ~4.5k measured c16csa j-957b8d91).
"""
import dataclasses
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VQ, D, NOPE, ROPE = 7, 512, 448, 64
PAGE = 256
TOPK = 1024
RATIO = 4
SM_SCALE = 512 ** -0.5
MAX_SEQS = {4: 8, 8: 8, 16: 16}  # per-rank metadata width (max-num-seqs padded), served fin
CBIN = 4096  # compressed rows per ctx bin in union_hist.npz
UBIN = 32


@dataclasses.dataclass(frozen=True)
class Cell:
    name: str
    conc: int  # 4 / 8 / 16 -> H, max seqs, env
    T: int  # bucket
    R: int  # verify windows
    ctx: tuple  # tokens before each window (len R)
    U: tuple  # union target per window (len R); 0 = sample from the measured distribution
    prefill: tuple = ()  # (chunk_len, prefix_tokens, u32) for a prefilling-replica mixed cell
    short: bool = False
    seed: int = 0

    @property
    def heads(self):
        return 32 if self.conc == 16 else 16


# ---------------------------------------------------------------------------------------------- measured distribution
_HIST = None


def hist():
    global _HIST
    if _HIST is None:
        _HIST = dict(np.load(os.path.join(HERE, "union_hist.npz"), allow_pickle=True))
    return _HIST


def union_quantile(conc, ctx_tokens, q, layer=None, kind="decode"):
    """Union size at quantile q for windows whose compressed ctx falls in ctx_tokens' bin (c4 uses c8's distribution)."""
    h = hist()[f"{'c16' if conc == 16 else 'c8'}_{kind}"]  # [L, CB, UB]
    h = h[layer] if layer is not None else h.sum(0)
    cb = min(int(ctx_tokens // RATIO // CBIN), h.shape[0] - 1)
    row = h[cb]
    if row.sum() == 0:
        row = h.sum(0)
    c = np.cumsum(row)
    b = int(np.searchsorted(c, c[-1] * q))
    return int(min(VQ * TOPK, max(TOPK, b * UBIN + UBIN // 2)))


def sample_window(conc, rng, layer=None):
    """(ctx_tokens, U) drawn jointly from the measured decode distribution."""
    h = hist()[f"{'c16' if conc == 16 else 'c8'}_decode"]
    h = h[layer] if layer is not None else h.sum(0)
    p = h.reshape(-1).astype(np.float64)
    k = rng.choice(p.size, p=p / p.sum())
    cb, ub = divmod(k, h.shape[1])
    ctx = int((cb + rng.random()) * CBIN * RATIO)
    U = int(ub * UBIN + rng.integers(0, UBIN))
    return max(ctx, 64), U


# ---------------------------------------------------------------------------------------------- cells
def _q(conc, ctx, q):
    return union_quantile(conc, ctx, q)


# Context per phase = measured window ctx distribution (k1u tap, q_len 7 windows): c16 p10/p50/p90 ctx bins 3/7/28,
# c8 2/5/28 (bins of 4096 compressed rows = 16k tokens) -> first/mid/last tokens below (bin centres).
CTX = {16: (57344, 122880, 466944), 8: (40960, 90112, 466944), 4: (40960, 90112, 466944)}
BUCKETS = {4: [(8, 1)], 8: [(16, 2), (32, 4)], 16: [(16, 2), (32, 4)]}


def _cells():
    cells = {}
    for conc, bl in BUCKETS.items():
        for T, R in bl:
            for ph, ctx in zip(("first", "mid", "last"), CTX[conc]):
                n = f"c{conc}T{T}-{ph}"
                cells[n] = Cell(n, conc, T, R, (ctx,) * R, (_q(conc, ctx, 0.5),) * R)
            ctx = CTX[conc][1]
            n = f"c{conc}T{T}-u90"  # p90 union at mid ctx (layers 20/30-like)
            cells[n] = Cell(n, conc, T, R, (ctx,) * R, (_q(conc, ctx, 0.9),) * R)
            n = f"c{conc}T{T}-ragged"  # (ctx, U) sampled jointly per window
            cells[n] = Cell(n, conc, T, R, (0,) * R, (0,) * R, seed=11)
    # c16 T8 exists in served c16 (1 live request in a replica); c4 has only T8 in practice
    for ph, ctx in zip(("first", "mid", "last"), CTX[16]):
        cells[f"c16T8-{ph}"] = Cell(f"c16T8-{ph}", 16, 8, 1, (ctx,), (_q(16, ctx, 0.5),))
    # mixed T512 c16 (TP4, 32 heads): decode replica (4 windows + padding) and prefilling replica (3 windows + chunk)
    cells["c16m512-dec"] = Cell("c16m512-dec", 16, 512, 4, (CTX[16][1],) * 4, (_q(16, CTX[16][1], 0.5),) * 4)
    cells["c16m512-pf32k"] = Cell("c16m512-pf32k", 16, 512, 3, (CTX[16][1],) * 3, (_q(16, CTX[16][1], 0.5),) * 3,
                                  prefill=(491, 31700, 4500))
    cells["c16m512-pf127k"] = Cell("c16m512-pf127k", 16, 512, 3, (CTX[16][1],) * 3, (_q(16, CTX[16][1], 0.5),) * 3,
                                   prefill=(491, 127000, 4500))
    # edge cells (accuracy only)
    cells["edge-short"] = Cell("edge-short", 8, 16, 2, (900, 2400), (0, 0), short=True)  # < top-k valid rows
    cells["edge-tiny"] = Cell("edge-tiny", 4, 8, 1, (5,), (0,), short=True)  # 1-3 valid rows, token 0 has 1
    cells["edge-pageb"] = Cell("edge-pageb", 16, 16, 2, (PAGE * RATIO * 9 - 3, PAGE * RATIO * 40 + 1),
                               (1025, 7168))  # window straddles a page boundary; U = 1025 and 7168 (min / max)
    cells["edge-maxb"] = Cell("edge-maxb", 16, 32, 4, (300000, 1000000, 20000, 140000), (2480, 4096, 1500, 3100))
    cells["edge-sink"] = Cell("edge-sink", 16, 16, 2, (122880,) * 2, (1808,) * 2)  # sinks +6: sink-dominant softmax
    cells["edge-maxb8"] = Cell("edge-maxb8", 8, 64, 8, (90112,) * 8, (1776,) * 8)  # 8 windows (served mns 8 at c8)
    # v1.2: >= 8-token NON-verify requests in decode buckets (short prefills / chunk tails), cut into 8-token windows by the union
    # path; at short context every token takes all valid rows, so most union rows are selected by all 8 tokens (count 8).
    cells["pfw-c8T16-short"] = Cell("pfw-c8T16-short", 8, 16, 0, (), (), prefill=(16, 1500, 4500))
    cells["pfw-c16T32-short"] = Cell("pfw-c16T32-short", 16, 32, 1, (CTX[16][1],), (_q(16, CTX[16][1], 0.5),),
                                     prefill=(24, 3000, 4500))
    cells["pfw-c8T32-mid"] = Cell("pfw-c8T32-mid", 8, 32, 0, (), (), prefill=(32, 31700, 1200))
    return cells


CELLS = _cells()
PHASE_CELLS = [n for n in CELLS if not n.startswith("edge") and not n.endswith("u90") and not n.endswith("ragged")
               and not n.startswith("c16m512")]


# ---------------------------------------------------------------------------------------------- inputs
def _window_lists(rng, n_valid, U):
    """7 top-k lists (seq-local compressed rows) with union exactly U, token j valid rows n_valid[j] (non-decreasing).
    Returns int32[7, TOPK] (-1 padded)."""
    out = np.full((VQ, TOPK), -1, np.int32)
    n0 = n_valid[0]
    if n0 <= 0:
        for j in range(VQ):
            k = min(TOPK, n_valid[j])
            if k > 0:
                out[j, :k] = rng.permutation(n_valid[j])[:k]
        return out
    if n0 <= TOPK:  # short context: every token takes all its valid rows (top-k is the whole set)
        for j in range(VQ):
            k = min(TOPK, n_valid[j])
            out[j, :k] = rng.permutation(k)
        return out
    U = int(np.clip(U, TOPK, min(VQ * TOPK, n0)))
    pool = rng.choice(n0, size=U, replace=False).astype(np.int32)
    base, new = pool[:TOPK], pool[TOPK:]
    t = np.full(VQ - 1, (U - TOPK) // (VQ - 1))
    t[: (U - TOPK) % (VQ - 1)] += 1
    out[0] = rng.permutation(base)
    off = 0
    for j in range(1, VQ):
        keep = rng.choice(TOPK, size=TOPK - t[j - 1], replace=False)
        lst = np.concatenate([base[keep], new[off:off + t[j - 1]]])
        off += t[j - 1]
        out[j] = rng.permutation(lst)
    return out


def build_inputs(cell, seed=None):
    """Host arrays for one cell. Returns dict (numpy)."""
    rng = np.random.default_rng(cell.seed * 1000 + (seed or 0) + 17)
    conc, H = cell.conc, cell.heads
    ms = MAX_SEQS[conc] if cell.R <= MAX_SEQS[conc] else cell.R
    ctx, U = list(cell.ctx), list(cell.U)
    for i in range(cell.R):
        if ctx[i] == 0:  # ragged: joint sample
            ctx[i], U[i] = sample_window(conc, rng)
        elif U[i] == 0:
            U[i] = union_quantile(conc, ctx[i], 0.5)
    lens = [VQ] * cell.R
    pf = cell.prefill
    if pf:
        lens.append(pf[0])
        ctx.append(pf[1])
    nseq = len(lens)
    assert sum(lens) <= cell.T, (cell.name, lens)
    seq_tokens = [c + l for c, l in zip(ctx, lens)]
    kv_lens = [s // RATIO for s in seq_tokens]  # compressed rows of each sequence (incl. this step's tokens)
    need = [-(-max(k, 1) // PAGE) + 1 for k in kv_lens]
    pps = max(need) + 1
    total_pages = sum(need) + 8
    perm = rng.permutation(total_pages - 1).astype(np.int32) + 1  # shuffled pool, page 0 reserved
    table = np.zeros((ms, pps), np.int32)
    o = 0
    for i in range(nseq):
        table[i, : need[i]] = perm[o:o + need[i]]
        o += need[i]
    cu = np.zeros(ms + 1, np.int32)
    cu[1:nseq + 1] = np.cumsum(lens)
    cu[nseq + 1:] = cu[nseq]
    tk = np.full((cell.T, TOPK), -1, np.int32)
    seq_of = np.full(cell.T, -1, np.int32)
    pos = np.full(cell.T, -1, np.int64)
    t = 0
    for i in range(cell.R):
        nv = [(ctx[i] + j + 1) // RATIO for j in range(VQ)]  # strict causal: complete groups only
        tk[t:t + VQ] = _window_lists(rng, nv, U[i])
        seq_of[t:t + VQ] = i
        pos[t:t + VQ] = ctx[i] + np.arange(VQ)
        t += VQ
    if pf:
        L, prefix, u32 = pf
        i = cell.R
        for b0 in range(0, L, 32):
            b1 = min(L, b0 + 32)
            n_first = (prefix + b0 + 1) // RATIO
            if n_first <= TOPK:
                for p in range(b0, b1):
                    nv = (prefix + p + 1) // RATIO
                    k = min(TOPK, nv)
                    tk[t + p, :k] = rng.permutation(nv)[:k]
                continue
            pool = rng.choice(n_first, size=min(u32, n_first), replace=False)
            for p in range(b0, b1):
                tk[t + p] = rng.choice(pool, size=TOPK, replace=False)
        seq_of[t:t + L] = i
        pos[t:t + L] = prefix + np.arange(L)
        t += L
    dist = np.array([0, cell.R, nseq], np.int32)  # verify windows sit in the middle segment; a prefill chunk last
    union = []
    for i in range(cell.R):
        rows = tk[i * VQ:(i + 1) * VQ]
        union.append(int(np.unique(rows[rows >= 0]).size))
    return dict(cell=cell, H=H, T=cell.T, nseq=nseq, ms=ms, lens=lens, ctx=ctx, kv_lens=np.array(kv_lens + [0] * (ms - nseq), np.int32),
                table=table, pps=pps, total_pages=total_pages, cu=cu, dist=dist, tk=tk, seq_of=seq_of, pos=pos,
                union=union, U_target=U[: cell.R], rng=rng)


def drivers(inp):
    """Work drivers (M6): per cell, from the inputs the kernels see."""
    tk, R = inp["tk"], inp["cell"].R
    valid_tok = int((tk >= 0).any(1).sum())
    rows_inc = int((tk >= 0).sum())
    u = inp["union"]
    tiles = sum(-(-x // TOPK) for x in u)
    return dict(valid_tokens=valid_tok, gathered_rows_incumbent=rows_inc, union_rows=int(sum(u)),
                union_tiles=tiles, union=u, inc_tiles=int((tk[: R * VQ] >= 0).any(1).sum()),
                gather_bytes_incumbent=rows_inc * 640, gather_bytes_union=int(sum(u)) * 640)


def quantized_cache(rng, n_rows, poison_rows=None):
    """Real DSV4 FP8 records (e4m3 + e8m0 scales + bf16 rope) as cache words; exact float64 dequant."""
    import jax.numpy as jnp
    import ml_dtypes
    from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout

    x = rng.standard_normal((n_rows, NOPE), dtype=np.float32) * rng.uniform(0.05, 4.0, (n_rows, 1)).astype(np.float32)
    blk = x.reshape(n_rows, NOPE // 64, 64)
    amax = np.clip(np.abs(blk).max(axis=1), 1e-4, None)
    sf = np.power(2.0, np.ceil(np.log2(amax / 448.0))).astype(np.float32)
    fp8 = (blk / sf[:, None, :]).astype(ml_dtypes.float8_e4m3fn).reshape(n_rows, NOPE)
    e8 = sf.astype(ml_dtypes.float8_e8m0fnu)
    rope = rng.standard_normal((n_rows, ROPE), dtype=np.float32).astype(ml_dtypes.bfloat16)
    slabs = np.concatenate([fp8.view(np.uint8), e8.view(np.uint8)], axis=1).reshape(n_rows // PAGE, PAGE, 4, 128)
    nope_w = csa_cache_layout.slabs_to_words(jnp.asarray(slabs))
    rope_w = csa_cache_layout.encode_rope(jnp.asarray(rope)).reshape(n_rows // PAGE, PAGE // 4, 128)
    if poison_rows is not None:  # 0xFF every byte of rows no valid top-k entry references
        pr = jnp.asarray(poison_rows.reshape(n_rows // PAGE, PAGE))
        nope_w = jnp.where(pr[..., None], jnp.int32(-1), nope_w)
        rw = rope_w.reshape(n_rows // PAGE, PAGE // 4, 4, 32)
        rw = jnp.where(pr.reshape(n_rows // PAGE, PAGE // 4, 4)[..., None], jnp.int32(-1), rw)
        rope_w = rw.reshape(n_rows // PAGE, PAGE // 4, 128)
    deq = np.concatenate([fp8.astype(np.float64) * np.tile(e8.astype(np.float64), (1, NOPE // 64)),
                          rope.astype(np.float64)], axis=1)
    return nope_w, rope_w, deq


def random_cache(n_pages, key=3):
    """Fast valid-bit-pattern cache for timing (finite fp8/e8m0/bf16 values)."""
    import jax
    import jax.numpy as jnp
    nope = jax.random.randint(jax.random.PRNGKey(key), (n_pages, PAGE, 128), 0, 0x3F3F3F3F, jnp.int32) & 0x3F3F3F3F
    rope = jax.random.randint(jax.random.PRNGKey(key + 1), (n_pages, PAGE // 4, 128), 0, 0x3F003F00, jnp.int32) & 0x3F003F00
    return nope, rope


def referenced_rows(inp):
    n_rows = inp["total_pages"] * PAGE
    ref = np.zeros(n_rows, bool)
    for t in range(inp["T"]):
        i = inp["seq_of"][t]
        if i < 0:
            continue
        idx = inp["tk"][t][inp["tk"][t] >= 0]
        ref[inp["table"][i, idx // PAGE] * PAGE + idx % PAGE] = True
    return ref


def device_inputs(inp, cache="random", poison=False):
    """jax arrays for the region call; cache 'random' (timing) or 'quant' (accuracy, also returns deq)."""
    import jax
    import jax.numpy as jnp
    rng = np.random.default_rng(99)
    T, H = inp["T"], inp["H"]
    deq = None
    if cache == "quant":
        nope, rope, deq = quantized_cache(rng, inp["total_pages"] * PAGE,
                                          poison_rows=(~referenced_rows(inp)) if poison else None)
    else:
        nope, rope = random_cache(inp["total_pages"])
    q = jax.random.normal(jax.random.PRNGKey(5), (T, H, D), jnp.float32).astype(jnp.bfloat16)
    d = dict(q=q, nope=nope, rope=rope, topk=jnp.asarray(inp["tk"]), kv_lens=jnp.asarray(inp["kv_lens"]),
             pages=jnp.asarray(inp["table"].reshape(-1)), cu=jnp.asarray(inp["cu"]), dist=jnp.asarray(inp["dist"]),
             sinks=jnp.asarray(rng.standard_normal(H).astype(np.float32) + (6.0 if inp["cell"].name == "edge-sink" else 0.0)),
             swa_acc=jnp.asarray(rng.standard_normal((T, H, D)).astype(np.float32) * 0.1).astype(jnp.bfloat16),
             swa_l=jnp.asarray(rng.uniform(4.0, 64.0, (T, H)).astype(np.float32)),
             swa_m=jnp.asarray(rng.uniform(2.0, 8.0, (T, H)).astype(np.float32)))
    return d, deq


# ---------------------------------------------------------------------------------------------- oracle + faults
FAULTS = ("drop_union_row", "mask_leak", "page_shift", "stale_acc", "drop_tail_tile", "no_sink", "causal",
          "cross_request", "no_swa", "pad_token_leak", "drop_window8_row")


def _scores(inp, d, deq, t, idx):
    q = np.asarray(d["q"][t].astype(np.float32), np.float64)
    i = inp["seq_of"][t]
    phys = inp["table"][i, idx // PAGE] * PAGE + idx % PAGE
    return q @ deq[phys].T * SM_SCALE  # [H, k]


_HEAVY = {}


def _heavy_row(inp, d, deq, w):
    key = (id(inp), id(d), w)
    if key not in _HEAVY:
        tk = inp["tk"]
        best, row = -1.0, -1
        for t in range(w * VQ, (w + 1) * VQ):
            idx = tk[t][tk[t] >= 0]
            if not idx.size:
                continue
            s = _scores(inp, d, deq, t, idx)
            p = np.exp(s - s.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            k = np.unravel_index(np.argmax(p), p.shape)[1]
            if p.max() > best:
                best, row = p.max(), int(idx[k])
        _HEAVY[key] = row
    return _HEAVY[key]


def _leak_row(inp, d, deq, t, idx):
    tk = inp["tk"]
    w0 = (t // VQ) * VQ
    other = np.setdiff1d(np.unique(tk[w0:w0 + VQ][tk[w0:w0 + VQ] >= 0]), idx)
    if not other.size:
        return other.astype(idx.dtype)
    s = _scores(inp, d, deq, t, other)
    return other[[int(np.argmax(s.max(0)))]].astype(idx.dtype)


_HEAVY8 = {}


def _heavy_common_row(inp, d, deq, t0, t1):
    """The row every token of [t0, t1) selects that carries the most attention weight (v1.2 fault drop_window8_row)."""
    key = (id(inp), id(d), t0, t1)
    if key not in _HEAVY8:
        tk = inp["tk"]
        common = None
        for t in range(t0, t1):
            v = set(tk[t][tk[t] >= 0].tolist())
            common = v if common is None else common & v
        best, row = -1.0, -1
        if common:
            cand = np.array(sorted(common))
            for t in range(t0, t1):
                idx = tk[t][tk[t] >= 0]
                s = _scores(inp, d, deq, t, idx)
                p = np.exp(s - s.max(1, keepdims=True))
                p /= p.sum(1, keepdims=True)
                pc = p[:, np.isin(idx, cand)]
                if pc.size and pc.max() > best:
                    best, row = pc.max(), int(idx[np.isin(idx, cand)][np.unravel_index(np.argmax(pc), pc.shape)[1]])
        _HEAVY8[key] = row
    return _HEAVY8[key]


def oracle(inp, d, deq, fault=None, frng=None):
    """float64 reference over valid rows -> (out [n, H, D], rows). `fault` builds the output a buggy kernel of that class
    would produce (judged by the same gate as a candidate -> catch rate)."""
    q = np.asarray(d["q"].astype(np.float32), np.float64)
    swa_acc = np.asarray(d["swa_acc"].astype(np.float32), np.float64)
    swa_l = np.asarray(d["swa_l"], np.float64)
    swa_m = np.asarray(d["swa_m"], np.float64)
    sinks = np.asarray(d["sinks"], np.float64)
    table, tk = inp["table"], inp["tk"]
    frng = frng or np.random.default_rng(5)
    rows, outs = [], []
    prev_acc = None
    for t in range(inp["T"]):
        i = inp["seq_of"][t]
        if i < 0:
            continue
        idx = tk[t][tk[t] >= 0]
        if fault == "drop_union_row" and i < inp["cell"].R:
            # a union row missing for every token of the window; the instance is the union row carrying the most
            # attention weight (a dropped row that carries ~0 weight is below bf16 rounding -> undetectable by design)
            idx = idx[idx != _heavy_row(inp, d, deq, t // VQ)]
        if fault == "drop_window8_row" and i == inp["cell"].R and inp["cell"].prefill:
            # an 8-token window of a non-verify request loses the heaviest row all its 8 tokens select (the count-8 class)
            s0 = int(inp["cu"][i])
            w0 = s0 + ((t - s0) // 8) * 8
            w1 = min(w0 + 8, int(inp["cu"][i + 1]))
            if w1 - w0 == 8:
                idx = idx[idx != _heavy_common_row(inp, d, deq, w0, w1)]
        if fault == "mask_leak" and i < inp["cell"].R:
            # this token's mask admits the window-union row (not in its own list) it would score highest
            idx = np.concatenate([idx, _leak_row(inp, d, deq, t, idx)])
        if fault == "causal" and t % VQ == 6:  # attends the still-open compressed group (pre-fix bound)
            nv = (inp["pos"][t] + 1) // RATIO
            extra = np.arange(nv, inp["pos"][t] // RATIO + 2)
            idx = np.concatenate([idx, extra[: 1]]) if extra.size else idx
        if fault == "drop_tail_tile" and idx.size > 0 and i < inp["cell"].R:
            # rows whose union rank >= 1024 * floor(U/1024) dropped (a lost tail tile)
            w0 = (t // VQ) * VQ
            u = np.unique(tk[w0:w0 + VQ][tk[w0:w0 + VQ] >= 0])
            keep_n = (u.size // TOPK) * TOPK if u.size > TOPK else u.size
            keep = set(u[:keep_n].tolist()) if u.size % TOPK else set(u.tolist())
            if u.size % TOPK and u.size > TOPK:
                idx = np.array([x for x in idx if x in keep], np.int64)
        seq = i
        if fault == "cross_request" and inp["cell"].R > 1 and i < inp["cell"].R:
            seq = (i + 1) % inp["cell"].R
        idx = idx[idx < (table.shape[1] * PAGE)]
        pg = table[seq, idx // PAGE]
        if fault == "page_shift":
            pg = table[seq, np.minimum(idx // PAGE + 1, table.shape[1] - 1)]
        phys = pg * PAGE + idx % PAGE
        K = deq[phys] if idx.size else np.zeros((0, D))
        s = q[t] @ K.T * SM_SCALE
        sm, sl, sa = swa_m[t], swa_l[t], swa_acc[t]
        if fault == "no_swa":
            sm, sl, sa = np.full_like(sm, -1e30), np.zeros_like(sl), np.zeros_like(sa)
        m = np.maximum(sm, s.max(axis=1)) if idx.size else sm
        p = np.exp(s - m[:, None])
        sk = np.zeros_like(sinks) if fault == "no_sink" else np.exp(sinks - m)
        l = sl * np.exp(sm - m) + p.sum(axis=1) + sk
        acc = sa * np.exp(sm - m)[:, None] + p @ K
        if fault == "stale_acc" and prev_acc is not None and t % VQ == 0:
            acc = acc + 0.05 * prev_acc
        if fault == "pad_token_leak" and t % VQ == 6:
            acc = acc + 1e-2 * np.abs(acc).max()
        prev_acc = acc
        outs.append(acc / l[:, None])
        rows.append(t)
    return np.stack(outs), np.array(rows)


def gate(out, want, ref_err=None):
    """Candidate gate: max-rel err vs f64 within max(2 x native err, native + 2^-8); default 5e-3 (W2 bound)."""
    scale = np.abs(want).max()
    err = float(np.abs(out - want).max() / scale)
    tol = max(2 * ref_err, ref_err + 2 ** -8) if ref_err is not None else 5e-3
    return err, err <= tol, tol


# ---------------------------------------------------------------------------------------------- regions
def served_env(conc):
    """The CSA-relevant served fin env (dsv4-trial2/e2e3/recipes/*-fin-*.yaml + stack a1dda4fd2 defaults)."""
    env = {"TPU_DSV4_CSA_UNION_DENSE": "1", "TPU_DSV4_SWA_SINGLE_PASS": "1", "TPU_STREAMIDX_VERIFY_BD": "1",
           "TPU_STREAMIDX_STRICT_CAUSAL": "1", "TPU_ROPE_CACHE_ROW_MAJOR": "1", "TPU_DSV4_PAIR_SWAP": "1",
           "TPU_DSV4_ATTN_INTRA_CHIP_TP": "0"}
    if conc == 16:
        env.update(TPU_DSV4_SEQ_PARALLEL="1", TPU_DSV4_SEQ_PARALLEL_MIN_TOKENS="32", TPU_DSV4_INDEXER_TOKEN_SPLIT_ABOVE="256")
    else:
        env.update(TPU_DSV4_INDEXER_KV_SPLIT_MIN_TOKENS="128")
    return env


def kernel_tuning(conc):
    """attention.py's kernel_tuning dict for CSA, computed from envs exactly as the served layer does."""
    for k, v in served_env(conc).items():
        os.environ[k] = v
    from vllm_torchtpu import envs
    return {
        "csa_gather_and_attention_chunk_size": envs.TPU_DSV4_CSA_GATHER_AND_ATTENTION_CHUNK_SIZE,
        "csa_gather_no_pad": envs.TPU_DSV4_CSA_GATHER_NO_PAD,
        "union_dense": envs.TPU_DSV4_CSA_UNION_DENSE,
        "union_dense_min_tokens": envs.TPU_DSV4_CSA_UNION_DENSE_MIN_TOKENS,
        "union_dense_rows_per_token": envs.TPU_DSV4_CSA_UNION_DENSE_ROWS_PER_TOKEN,
        "union_dense_max_heads": envs.TPU_DSV4_CSA_UNION_DENSE_MAX_HEADS,
        "union_dense_max_kv_tp4": envs.TPU_DSV4_CSA_UNION_DENSE_MAX_KV_TP4,
        "union_dense_plan_unconditional": (envs.TPU_DSV4_INDEXER_TOKEN_SPLIT_ABOVE > 0 and bool(envs.TPU_DSV4_SEQ_PARALLEL)),
    }


def inc_region(conc):
    """Arm 0: `_attention_csa` verbatim (served kwargs), `_run_swa` replaced by the precomputed SWA partials."""
    import jax.numpy as jnp
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4 import deepseek_v4_attention_op as op
    kw = kernel_tuning(conc)

    def f(d, q):
        orig = op._run_swa
        op._run_swa = lambda **k: (d["swa_acc"], k["sw_cache"], d["swa_l"], d["swa_m"])
        try:
            dummy = jnp.zeros((1, 8, 128), jnp.int32)
            out, _ = op._attention_csa(
                q, jnp.zeros((q.shape[0], D), jnp.bfloat16), dummy, d["kv_lens"], d["pages"], d["cu"], d["dist"],
                d["nope"], d["kv_lens"], d["topk"], d["pages"], d["cu"], d["dist"], d["sinks"], d["rope"],
                sm_scale=SM_SCALE, sliding_window=128, logical_page_size=128, swa_only=False,
                two_caches_same_buffer=False, verify_q_len=VQ, **kw)
        finally:
            op._run_swa = orig
        return out
    return f


def udx_region(conc, bq=8, groups=4, min_q_len=VQ, count_first=None, split=None, s_cap=65536):
    """Design probe (fact, not a candidate): the EXISTING csa_union_dense code routed for verify windows (min_q_len 7,
    bq 8) with no kernel change -- prices today's plan / gather / dense kernel at decode shapes. Falls back to the sparse
    path when a window's selectable range exceeds s_cap (as the shipped code does)."""
    from vllm_torchtpu.kernels.deepseek_v4.core_attention import csa_union_dense as ud
    kw = kernel_tuning(conc)

    def f(d, q):
        H = q.shape[1]
        T = q.shape[0]
        cf = (H <= 16) if count_first is None else count_first
        sp = (1 if H <= 16 else 2) if split is None else split
        chunk, batch = (32, 8) if T <= 128 else (kw["csa_gather_and_attention_chunk_size"], 16)
        plan = ud.prepare(d["topk"], d["kv_lens"], d["pages"], d["cu"], d["dist"], page_size=PAGE, bq=bq,
                          s_cap=s_cap, min_q_len=min_q_len, rows_per_token=kw["union_dense_rows_per_token"],
                          count_first=cf, split=sp)
        return ud.attention(q, d["nope"], d["rope"], d["topk"], d["kv_lens"], d["pages"], d["cu"], d["dist"], d["sinks"],
                            d["swa_acc"], d["swa_l"], d["swa_m"], sm_scale=SM_SCALE, bq=bq, s_cap=s_cap,
                            rows_per_token=kw["union_dense_rows_per_token"], min_q_len=min_q_len, sparse_chunk=chunk,
                            sparse_batch=batch, gather_no_pad=kw["csa_gather_no_pad"], plan=plan, groups=groups,
                            count_first=cf, split=sp)
    return f


ARMS = {
    "inc": lambda conc: inc_region(conc),
    "udx": lambda conc: udx_region(conc),  # existing UD, bq 8, 4 gather groups (as shipped)
    "udx1": lambda conc: udx_region(conc, groups=1),  # one gather group
    "udx32": lambda conc: udx_region(conc, bq=32),  # mixed T512: windows as blocks of the served bq-32 kernel
}
