"""K-HCA instruments: one shape/dtype spec shared by test_accuracy.py and bench.py.

Block under test (DSV4-Pro HCA layer, decode verify windows, per device):
  kv_rope(kv)  ->  HCA compressor (proj_and_save_state + compress_norm_rope_store)
  ->  _attention_hca (SWA single pass + MLA-p verify + MLA-m)            [stock, 33062efcf]
Inputs: q bf16[T,H,512] (post qnorm_rope), kv bf16[T,512] (post kv_norm, pre-RoPE),
hidden bf16[T,7168], caches in the served layouts (block 1024): SWA int32[N,256,128]
(128 tokens/page, 2 rows/token), HCA u8[N,16,4,128] (8 entries/page, 1 KB bf16 each),
state int32[N,256,128] (f32 kv|score 4 KB/token, 32 tokens/page).
Cells = the served decode buckets (gaps/mk/WTK-BRIEF-attn-block.md).
"""
import dataclasses
import functools
import os

import numpy as np

VQ = 7  # verify window (6 spec tokens + 1)
D = 512
ROPE = 64
WINDOW = 128
RATIO = 128
HIDDEN = 7168
BLOCK = 1024  # served --block-size
SWA_PAGE = 128  # tokens per SWA logical page at block 1024
HCA_PAGE = BLOCK // RATIO  # 8 compressed entries per page
MAX_MODEL_LEN = 1048576
SM_SCALE = D ** -0.5
RMS_EPS = 1e-6


@dataclasses.dataclass(frozen=True)
class Cell:
    name: str
    heads: int
    ctx: tuple  # seq len per request INCLUDING the 7 new rows
    max_seqs: int
    pool: int  # KV pool pages (served TPU_NUM_GPU_BLOCKS_OVERRIDE)
    boundary: tuple = ()  # request indices whose window closes a compressed row
    sink_shift: float = 0.0  # sinks ~ U(-1,1) + shift [assumed distribution]

    @property
    def R(self):
        return len(self.ctx)

    @property
    def T(self):
        n = VQ * self.R
        for b in (8, 16, 32):
            if n <= b:
                return b
        raise ValueError(n)


def _ctx(base, R, boundary=()):
    """Seq lens: base-ish, distinct, window boundary-free unless listed."""
    out = []
    for r in range(R):
        k = (base + 9973 * r) // RATIO
        # (p+1) % 128 == 0 for some p in [L-7, L-1]  <=>  L % 128 in [0, 6]
        out.append(k * RATIO + (3 if r in boundary else 64))
    return tuple(out)


def make_cells():
    c = {}
    # phase points: decode_first (low ctx) / mid (brief cell) / decode_last (high ctx)
    for tag, H, ms, pool, R, ctxs in [
        ("c4T8", 16, 8, 4815, 1, (90000, 131072, 165000)),
        ("c8T16", 16, 8, 4818, 2, (90000, 131072, 165000)),
        ("c8T32", 16, 8, 4818, 4, (90000, 131072, 165000)),
        ("c16T16", 32, 16, 4288, 2, (80000, 98304, 120000)),
        ("c16T32", 32, 16, 4288, 4, (80000, 98304, 120000)),
    ]:
        for ph, base in zip(("first", "mid", "last"), ctxs):
            c[f"{tag}-{ph}"] = Cell(f"{tag}-{ph}", H, _ctx(base, R), ms, pool)
        # one request's window closes a compressed row (production: ~7/128 per request-step)
        c[f"{tag}-mid-b"] = Cell(f"{tag}-mid-b", H, _ctx(ctxs[1], R, (0,)), ms, pool, (0,))
    # ragged: lengths spread across [first, last]
    c["c8T32-ragged"] = Cell("c8T32-ragged", 16, (90003 + 61, 112640 + 64, 140032 + 64, 164992 + 64), 8, 4818)
    c["c16T32-ragged"] = Cell("c16T32-ragged", 32, (80000 + 64 - 80000 % 128, 92224, 106560, 119872), 16, 4288)
    # edge cells (accuracy only): first compressed row, page-boundary HCA counts, tiny ctx
    c["edge-short"] = Cell("edge-short", 16, (135,), 8, 512)  # 1 compressed entry, window crosses nothing
    c["edge-first-close"] = Cell("edge-first-close", 16, (130,), 8, 512, (0,))  # closes entry 0 this step
    c["edge-page"] = Cell("edge-page", 16, (8 * RATIO * 3 + 3, 8 * RATIO * 5 + 64), 8, 512, (0,))  # entry 23 closes = last slot of a page
    c["edge-sink"] = Cell("edge-sink", 16, (135, 1000 * RATIO + 64), 8, 1200, (), 6.0)  # sink dominant
    c["edge-maxbatch"] = Cell("edge-maxbatch", 32, _ctx(20000, 4, (1, 3)), 16, 1024, (1, 3))
    return c


CELLS = make_cells()
PHASE_CELLS = [k for k in CELLS if not k.startswith("edge")]


# ----------------------------------------------------------------------------- layouts
def bf16_bits(x):
    """float array -> uint16 bf16 bits (round-to-nearest-even)."""
    f = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    r = ((f >> 16) & 1) + 0x7FFF
    return ((f + r) >> 16).astype(np.uint16)


def bf16_from_bits(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def bf16_round(x):
    return bf16_from_bits(bf16_bits(x))


def tokens_to_words(x):
    """bf16 values [..., 512] -> int32 words [..., 2, 128]: word (s, w) = bf16(256s+w) | bf16(256s+128+w)<<16."""
    b = bf16_bits(x).astype(np.uint32).reshape(*x.shape[:-1], 2, 2, 128)
    return (b[..., 0, :] | (b[..., 1, :] << 16)).view(np.int32)


def words_to_tokens(w):
    """int32 [..., 2, 128] -> f32 [..., 512] (exact bf16 values)."""
    u = np.asarray(w).view(np.uint32)
    lo = (u & 0xFFFF).astype(np.uint16)
    hi = (u >> 16).astype(np.uint16)
    parts = np.stack([bf16_from_bits(lo), bf16_from_bits(hi)], axis=-2)  # [...,2,2,128]
    return parts.reshape(*w.shape[:-2], 512)


def words_to_slabs_np(w):
    u = np.asarray(w).view(np.uint32)
    return np.stack([((u >> (8 * b)) & 0xFF).astype(np.uint8) for b in range(4)], axis=-2)


def slabs_to_words_np(s):
    s = np.asarray(s).astype(np.uint32)
    return (s[..., 0, :] | (s[..., 1, :] << 8) | (s[..., 2, :] << 16) | (s[..., 3, :] << 24)).view(np.int32)


# ----------------------------------------------------------------------------- inputs
@dataclasses.dataclass
class Inputs:
    cell: Cell
    # host copies (numpy) — the oracle reads these
    q: np.ndarray  # f32 (bf16 values) [T,H,512]
    kv: np.ndarray  # f32 (bf16 values) [T,512]
    hidden: np.ndarray  # f32 (bf16 values) [T,7168]
    w: np.ndarray  # f32 (bf16 values) [7168,1024]
    ape: np.ndarray  # f32 [128,512]
    norm_w: np.ndarray  # f32 [512]
    sinks: np.ndarray  # f32 [H]
    positions: np.ndarray  # i32 [T]
    seq_lens: np.ndarray  # i32 [max_seqs]
    cu: np.ndarray  # i32 [max_seqs+1]
    dist: np.ndarray  # i32 [3]
    swa_bt: np.ndarray  # i32 [max_seqs, P]
    hca_bt: np.ndarray  # i32 [max_seqs, P]
    st_bt: np.ndarray  # i32 [max_seqs, Ps]
    swa_words: np.ndarray  # i32 [N,256,128]
    hca_words: np.ndarray  # i32 [N,16,128]  (u8 slabs built on device)
    st_words: np.ndarray  # i32 [N,256,128]
    sbs: int  # state tokens per page


def cos_sin_np(max_pos=MAX_MODEL_LEN, theta=10000.0):
    inv = 1.0 / (theta ** (np.arange(0, ROPE, 2, dtype=np.float64) / ROPE))
    ang = np.arange(max_pos, dtype=np.float64)[:, None] * inv[None, :]
    return np.concatenate([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)


def build_inputs(cell: Cell, seed=0, sbs=32):
    rng = np.random.default_rng(seed)
    T, H, R, ms, N = cell.T, cell.heads, cell.R, cell.max_seqs, cell.pool
    P = MAX_MODEL_LEN // SWA_PAGE  # SWA group: 128-token pages
    Ph = MAX_MODEL_LEN // BLOCK  # HCA group: 8 entries = 1024 tokens per page
    Ps = MAX_MODEL_LEN // sbs  # state group
    seq = np.zeros(ms, np.int32)
    seq[:R] = cell.ctx
    cu = np.zeros(ms + 1, np.int32)
    cu[1:R + 1] = np.arange(1, R + 1) * VQ
    cu[R + 1:] = cu[R]
    pos = np.zeros(T, np.int32)
    for r in range(R):
        pos[r * VQ:(r + 1) * VQ] = np.arange(cell.ctx[r] - VQ, cell.ctx[r])
    dist = np.array([0, R, R], np.int32)

    perm = iter(rng.permutation(np.arange(1, N)).tolist())  # page 0 = null page
    swa_bt = np.zeros((ms, P), np.int32)
    hca_bt = np.zeros((ms, Ph), np.int32)
    st_bt = np.zeros((ms, Ps), np.int32)
    swa_words = {}
    hca_words = {}
    st_words = {}
    for r in range(R):
        L = cell.ctx[r]
        for lp in range(max(0, L - VQ - WINDOW) // SWA_PAGE, (L - 1) // SWA_PAGE + 1):
            pg = next(perm)
            swa_bt[r, lp] = pg
            vals = rng.standard_normal((SWA_PAGE, D)).astype(np.float32)
            swa_words[pg] = tokens_to_words(vals).reshape(2 * SWA_PAGE, 128)
        n_entries = (L - 1) // RATIO + 1  # entries 0..(L-1)//128 may be written this step
        for hp in range((n_entries - 1) // HCA_PAGE + 1):
            pg = next(perm)
            hca_bt[r, hp] = pg
            vals = rng.standard_normal((HCA_PAGE, D)).astype(np.float32)
            hca_words[pg] = tokens_to_words(vals).reshape(2 * HCA_PAGE, 128)
        for sp in range(max(0, L - VQ - RATIO) // sbs, (L - 1) // sbs + 1):
            pg = next(perm)
            st_bt[r, sp] = pg
            kvs = rng.standard_normal((sbs, D)).astype(np.float32)
            scs = rng.standard_normal((sbs, D)).astype(np.float32)
            st = np.concatenate([kvs, scs], axis=1).view(np.int32)  # [sbs, 1024] words
            st_words[pg] = st.reshape(sbs * 8, 128)

    def bf(x):
        return bf16_round(x)

    return Inputs(
        cell=cell,
        q=bf(rng.standard_normal((T, H, D))),
        kv=bf(rng.standard_normal((T, D))),
        hidden=bf(rng.standard_normal((T, HIDDEN))),
        w=bf(rng.standard_normal((HIDDEN, 2 * D)) / np.sqrt(HIDDEN)),
        ape=(0.1 * rng.standard_normal((RATIO, D))).astype(np.float32),
        norm_w=(1.0 + 0.1 * rng.standard_normal(D)).astype(np.float32),
        sinks=(rng.uniform(-1, 1, H) + cell.sink_shift).astype(np.float32),
        positions=pos, seq_lens=seq, cu=cu, dist=dist,
        swa_bt=swa_bt, hca_bt=hca_bt, st_bt=st_bt,
        swa_words=swa_words, hca_words=hca_words, st_words=st_words, sbs=sbs,
    )


def scatter_pages(N, rows, pages: dict):
    a = np.zeros((N, rows, 128), np.int32)
    for pg, w in pages.items():
        a[pg] = w
    return a


# ----------------------------------------------------------------------------- f64 oracle
def rope_np(x, pos, cs):
    """Interleaved (GPT-J) RoPE on the last 64 dims; x f64 [..., 512], pos int [...]."""
    x = np.array(x, np.float64)
    c = cs[pos, :ROPE // 2].astype(np.float64)
    s = cs[pos, ROPE // 2:].astype(np.float64)
    e = x[..., D - ROPE::2].copy()
    o = x[..., D - ROPE + 1::2].copy()
    x[..., D - ROPE::2] = e * c - o * s
    x[..., D - ROPE + 1::2] = o * c + e * s
    return x


def oracle(inp: Inputs, cs, *, rope_pos_mode="floor", faults=frozenset()):
    """float64 reference of the whole block. Returns (out [T,H,512] f64, new_swa_rows {(r,pos): f64[512]},
    new_hca_rows {(r,entry): f64[512]}, new_states {(r,pos): f64[1024]}).

    rope_pos_mode: position of the closed compressed row's RoPE: 'floor' = (p//128)*128 (MK kv_write), 'pos' = p.
    faults (for the positive control only): see FAULTS.
    """
    c = inp.cell
    H, R = c.heads, c.R
    out = np.zeros((c.T, H, D))
    kv_r = rope_np(inp.kv, inp.positions, cs)  # f64; stock stores bf16
    kv_r_b = bf16_round(kv_r)
    st_new = (inp.hidden.astype(np.float64) @ inp.w.astype(np.float64))
    st_new[:, D:] += inp.ape[inp.positions % RATIO]
    new_swa, new_hca, new_st = {}, {}, {}
    for r in range(R):
        L = c.ctx[r]
        p0 = L - VQ
        toks = range(r * VQ, (r + 1) * VQ)

        def state(p):
            for t in toks:
                if inp.positions[t] == p and "no_state" not in faults:
                    return st_new[t]
            pg = inp.st_bt[r, p // inp.sbs]
            words = inp.st_words.get(pg)
            if words is None:
                return None
            slot = p % inp.sbs
            return words[slot * 8:(slot + 1) * 8].reshape(-1).view(np.float32).astype(np.float64)

        for t in toks:
            new_st[(r, int(inp.positions[t]))] = st_new[t]
            new_swa[(r, int(inp.positions[t]))] = kv_r[t]
        # compressed rows closed in this window
        closed = {}
        for t in toks:
            p = int(inp.positions[t])
            if (p + 1) % RATIO:
                continue
            win = [state(q_) for q_ in range(p - RATIO + 1, p + 1)]
            ok = np.array([w_ is not None and (p - RATIO + 1 + i) >= 0 for i, w_ in enumerate(win)])
            kvw = np.stack([w_[:D] if w_ is not None else np.zeros(D) for w_ in win])
            scw = np.stack([w_[D:] if w_ is not None else np.zeros(D) for w_ in win])
            scw = np.where(ok[:, None], scw, -np.inf)
            m = scw.max(0, keepdims=True)
            e = np.exp(scw - m)
            cvec = (e / e.sum(0, keepdims=True) * kvw).sum(0)
            cvec = cvec / np.sqrt(np.mean(cvec ** 2) + RMS_EPS) * inp.norm_w
            rp = (p // RATIO) * RATIO if rope_pos_mode == "floor" else p
            cvec = rope_np(cvec, np.array(rp), cs)
            entry = p // RATIO + (1 if "wrong_slot" in faults else 0)
            closed[entry] = cvec
            new_hca[(r, entry)] = cvec

        def hca_row(k):
            if k in closed:
                return closed[k]
            pg = inp.hca_bt[r, k // HCA_PAGE + (1 if ("page_shift" in faults and k // HCA_PAGE == 0) else 0)]
            words = inp.hca_words.get(pg)
            if words is None:
                return np.zeros(D)
            s = k % HCA_PAGE
            return words_to_tokens(words[2 * s:2 * s + 2]).astype(np.float64)

        def swa_row(p):
            for t in toks:
                if inp.positions[t] == p:
                    return kv_r_b[t].astype(np.float64)
            pg = inp.swa_bt[r, p // SWA_PAGE]
            s = p % SWA_PAGE
            return words_to_tokens(inp.swa_words[pg][2 * s:2 * s + 2]).astype(np.float64)

        n_hca = L // RATIO + 1  # upper bound of visible entries this step
        hrows = np.stack([hca_row(k) for k in range(n_hca)])
        for t in toks:
            p = int(inp.positions[t])
            lo = max(0, p - WINDOW + 1) + (1 if "swa_window" in faults else 0)
            srows = np.stack([swa_row(x) for x in range(lo, p + 1)])
            vis = (p + 1) // RATIO
            if "causal" in faults:
                vis = L // RATIO
            if "drop_row" in faults:
                vis = max(vis - 1, 0)
            if "drop_tail" in faults:
                vis = (vis // HCA_PAGE) * HCA_PAGE
            keys = np.concatenate([srows, hrows[:vis]], 0)
            qv = inp.q[t].astype(np.float64)  # [H,512]
            s = qv @ keys.T * SM_SCALE  # [H, n]
            if "stale_acc" in faults:
                ns = srows.shape[0]
                m1 = s[:, :ns].max(1, keepdims=True)
                m2 = s[:, ns:].max(1, keepdims=True) if vis else m1
                m = np.maximum(m1, m2)
                p1 = np.exp(s[:, :ns] - m1)  # SWA segment never rescaled
                p2 = np.exp(s[:, ns:] - m)
                num = p1 @ keys[:ns] + p2 @ keys[ns:]
                den = p1.sum(1) + p2.sum(1) + np.exp(inp.sinks - m[:, 0])
            else:
                m = s.max(1, keepdims=True)
                pr = np.exp(s - m)
                num = pr @ keys
                den = pr.sum(1) + (0 if "no_sink" in faults else np.exp(inp.sinks - m[:, 0]))
            out[t] = num / den[:, None]
    return out, new_swa, new_hca, new_st


FAULTS = {
    # name: (cell classes it can discriminate on, description)
    "drop_row": ("all", "last visible compressed row not attended"),
    "page_shift": ("all", "HCA page 0 read from the table entry of page 1 (off-by-one page)"),
    "stale_acc": ("all", "SWA partial not rescaled when the HCA segment raises the max"),
    "drop_tail": ("all", "partial last HCA page dropped"),
    "no_sink": ("all", "attention sink omitted from the denominator"),
    "swa_window": ("all", "oldest SWA row dropped (window 127)"),
    "causal": ("boundary", "rows before the closing token see the row closed later in the window"),
    "wrong_slot": ("boundary", "closed compressed row stored at entry+1"),
    "no_state": ("boundary", "this window's states not saved before pooling"),
}


# ----------------------------------------------------------------------------- stock (incumbent) region
def stock_env():
    """Served fin env for this path (E2E#3 fin recipes, dsv4-trial2/e2e3/recipes/*-fin-*.yaml); set before vllm_torchtpu.envs is read."""
    for k, v in (("TPU_DSV4_SWA_SINGLE_PASS", "1"), ("TPU_DSV4_CSA_UNION_DENSE", "1"),
                 ("TPU_DSV4_HCA_NUM_KV_PAGES_PER_BLOCK", "8,64,64"), ("TPU_ROPE_CACHE_ROW_MAJOR", "1"),
                 ("TPU_KV_DROP_HCA_STATE_CHECKPOINT", "1"), ("TPU_DSV4_PAIR_SWAP", "1"),
                 ("TPU_DSV4_ATTN_INTRA_CHIP_TP", "0"), ("TPU_STREAMIDX_VERIFY_BD", "1")):
        os.environ.setdefault(k, v)


def device_inputs(inp: Inputs):
    import jax.numpy as jnp
    from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout as CL
    c = inp.cell
    N = c.pool
    swa = jnp.asarray(scatter_pages(N, 2 * SWA_PAGE, inp.swa_words))
    st = jnp.asarray(scatter_pages(N, 8 * inp.sbs, inp.st_words))
    hca = CL.words_to_slabs(jnp.asarray(scatter_pages(N, 2 * HCA_PAGE, inp.hca_words)))  # u8 [N,16,4,128]
    bf = lambda x: jnp.asarray(x, jnp.float32).astype(jnp.bfloat16)
    return dict(
        q=bf(inp.q), kv=bf(inp.kv), hidden=bf(inp.hidden), w=bf(inp.w),
        ape=jnp.asarray(inp.ape), norm_w=jnp.asarray(inp.norm_w), sinks=jnp.asarray(inp.sinks),
        positions=jnp.asarray(inp.positions), seq_lens=jnp.asarray(inp.seq_lens), cu=jnp.asarray(inp.cu),
        dist=jnp.asarray(inp.dist), swa_bt=jnp.asarray(inp.swa_bt), hca_bt=jnp.asarray(inp.hca_bt),
        st_bt=jnp.asarray(inp.st_bt.reshape(-1)), swa=swa, hca=hca, st=st,
    )


def stock_region(parts=("rope", "comp", "attn")):
    """Returns f(d, cs) -> (out, swa, hca, st): the incumbent ops exactly as the served HCA layer calls them."""
    stock_env()
    import jax.numpy as jnp
    from vllm_torchtpu import envs
    from vllm_torchtpu.kernels.deepseek_v4 import rope as rope_kernel
    from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import config as ccfg
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_attention_op import _attention_hca
    from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_compressor import _compressor_jax_hca

    sbs = ccfg.state_block_size(ccfg.Mode.HCA, BLOCK, compress_ratio=RATIO, head_dim=D, indexer_seq_along_lane=True)

    def f(d, cs, return_kv=False):
        kv = d["kv"]
        if "rope" in parts:
            kv = rope_kernel.rope(kv, d["positions"], cs)
        hca, st = d["hca"], d["st"]
        if "comp" in parts:
            # wkv_scale (TPU_DSV4_COMPRESSOR_FP8 operand): per-column ones for a bf16
            # weight, never read.
            ws = jnp.ones((d["w"].shape[1],), jnp.float32)
            hca, st = _compressor_jax_hca(
                d["hidden"], d["w"], ws, d["ape"], d["norm_w"], cs, d["positions"], d["st_bt"], d["cu"],
                d["hca_bt"].reshape(-1), d["dist"], hca, st,
                state_block_size=sbs, head_dim=D, compress_ratio=RATIO, overlap=False, rms_eps=RMS_EPS, quant_block=64)
        out, swa = d["q"], d["swa"]
        if "attn" in parts:
            out, swa = _attention_hca(
                d["q"], kv, d["swa"], d["seq_lens"], d["swa_bt"], d["cu"], d["dist"],
                hca, d["seq_lens"] // RATIO, (d["positions"] + 1) // RATIO, d["hca_bt"], d["cu"], d["dist"], d["sinks"],
                sm_scale=SM_SCALE, sliding_window=WINDOW, logical_page_size=SWA_PAGE, swa_only=False,
                two_caches_same_buffer=False,
                hca_num_kv_pages_per_block=envs.TPU_DSV4_HCA_NUM_KV_PAGES_PER_BLOCK,
                hca_num_queries_per_block=envs.TPU_DSV4_HCA_NUM_QUERIES_PER_BLOCK,
                verify_q_len=VQ)
        return (out, swa, hca, st, kv) if return_kv else (out, swa, hca, st)

    return f, sbs
